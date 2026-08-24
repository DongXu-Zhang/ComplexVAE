"""Full-image and tiled reconstruction.

Training sees 256×256 crops. Inference must handle arbitrary H×W.

Geometry facts used here (see encoder/decoder):
- Encoder requires H,W divisible by spatial_compression.
- Bottleneck attention is dense HW×HW with no positional encoding.
- GroupNorm stats are over the whole tensor presented to the layer.

Tiling cannot reproduce full-image attention/GN; it matches the training
crop size. Overlap + blend only average *convolutional* edge mismatch.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _pad_2d(x: torch.Tensor, pad_l: int, pad_r: int, pad_t: int, pad_b: int, mode: str) -> torch.Tensor:
    """Pad like F.pad(..., (L,R,T,B)). Reflect cannot exceed current H/W; step it."""
    if pad_l == pad_r == pad_t == pad_b == 0:
        return x
    if mode != "reflect":
        return F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    while pad_l or pad_r or pad_t or pad_b:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        sl = min(pad_l, max(w - 1, 0))
        sr = min(pad_r, max(w - 1, 0))
        st = min(pad_t, max(h - 1, 0))
        sb = min(pad_b, max(h - 1, 0))
        if sl + sr + st + sb == 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
            break
        x = F.pad(x, (sl, sr, st, sb), mode="reflect")
        pad_l -= sl
        pad_r -= sr
        pad_t -= st
        pad_b -= sb
    return x


def pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad H,W up to multiple; return padded tensor and (pad_h, pad_w) bottom/right pads."""
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, (0, 0)
    return _pad_2d(x, 0, pw, 0, ph, mode), (ph, pw)


def unpad(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    ph, pw = pad_hw
    if ph == 0 and pw == 0:
        return x
    h = x.shape[-2] - ph
    w = x.shape[-1] - pw
    return x[..., :h, :w]


def legacy_snapped_origins(length: int, tile_size: int, overlap: int) -> List[int]:
    """Old range(0, L, step) + snap-last-tile-to-end. Uneven terminal overlap."""
    if overlap >= tile_size:
        raise ValueError("overlap must be < tile_size")
    step = max(tile_size - max(overlap, 0), 1)
    origins: List[int] = []
    for y0 in range(0, length, step):
        y1 = min(y0 + tile_size, length)
        if y1 - y0 < tile_size and y1 == length:
            y0 = max(0, length - tile_size)
        if y0 not in origins:
            origins.append(y0)
        if y0 + tile_size >= length or y1 >= length:
            break
    if not origins:
        origins = [0]
    return origins


def even_tile_origins(
    length: int,
    tile_size: int,
    overlap: int,
    *,
    snap: int = 1,
) -> List[int]:
    """Origins so every tile has size tile_size, first at 0, last ends at length.

    Spacing is uniform (up to rounding). Unlike legacy snap-last, the extra
    leftover is spread across all gaps instead of dumped on the final pair.
    overlap is a *target* minimum; actual gap may be larger so the image is
    covered with fixed-size tiles.
    """
    if tile_size < 1:
        raise ValueError("tile_size must be >= 1")
    if length <= 0:
        raise ValueError("length must be > 0")
    if length <= tile_size:
        return [0]
    ov = max(int(overlap), 0)
    if ov >= tile_size:
        raise ValueError("overlap must be < tile_size")
    span = length - tile_size
    step = max(tile_size - ov, 1)
    n = int(math.ceil(span / step)) + 1
    n = max(n, 2)
    snap = max(int(snap), 1)
    # Rounding to `snap` can widen a gap by up to `snap`. If snap > overlap,
    # that can exceed tile_size and leave a hole. Cap snap to the overlap.
    if ov > 0:
        snap = min(snap, ov)
    origins: List[int] = []
    for i in range(n):
        raw = i * span / (n - 1)
        o = int(round(raw / snap) * snap) if snap > 1 else int(round(raw))
        origins.append(o)
    origins[0] = 0
    origins[-1] = span
    out: List[int] = []
    for o in origins:
        o = max(0, min(int(o), span))
        if not out or o != out[-1]:
            out.append(o)
    if out[-1] != span:
        out.append(span)
    # Guarantee every pixel is covered even if rounding still stretched a gap.
    i = 0
    while i < len(out) - 1:
        if out[i + 1] - out[i] > tile_size:
            out.insert(i + 1, out[i] + tile_size)
        else:
            i += 1
    return out


def clipped_grid_origins(length: int, tile_size: int) -> List[Tuple[int, int]]:
    """Non-overlapping grid (overlap=0): (start, end) with last tile clipped."""
    if length <= tile_size:
        return [(0, length)]
    out: List[Tuple[int, int]] = []
    y = 0
    while y < length:
        y1 = min(y + tile_size, length)
        out.append((y, y1))
        y = y1
    return out


def tile_boxes(
    h: int,
    w: int,
    tile_size: int,
    overlap: int,
    *,
    snap: int = 1,
) -> List[Tuple[int, int, int, int]]:
    """List of (y0, x0, y1, x1) covering the image. y1/x1 exclusive."""
    if overlap <= 0:
        ys = clipped_grid_origins(h, tile_size)
        xs = clipped_grid_origins(w, tile_size)
        return [(y0, x0, y1, x1) for y0, y1 in ys for x0, x1 in xs]
    ys = even_tile_origins(h, tile_size, overlap, snap=snap)
    xs = even_tile_origins(w, tile_size, overlap, snap=snap)
    boxes = []
    for y0 in ys:
        for x0 in xs:
            boxes.append((y0, x0, y0 + tile_size, x0 + tile_size))
    return boxes


def _ramp_1d(
    length: int,
    origin: int,
    img_len: int,
    overlap: int,
    *,
    mode: str,
    device,
    dtype,
) -> torch.Tensor:
    """Per-axis blend weights. Image outer border stays 1 (no darkening)."""
    w = torch.ones(length, device=device, dtype=dtype)
    ov = max(int(overlap), 0)
    if ov <= 0 or length <= 1:
        return w
    if mode == "hann":
        n = torch.arange(length, device=device, dtype=dtype)
        hann = 0.5 - 0.5 * torch.cos(2 * math.pi * (n + 0.5) / length)
        w = hann
        if origin == 0:
            w[: length // 2] = 1.0
        if origin + length >= img_len:
            w[length // 2 :] = 1.0
        return w
    # linear: fade only interior edges, ramp length = overlap (or tile if smaller)
    rlen = min(ov, max(length // 2, 1))
    ramp = torch.linspace(0.0, 1.0, rlen, device=device, dtype=dtype)
    if origin > 0 and length > rlen:
        w[:rlen] = ramp
    if origin + length < img_len and length > rlen:
        w[-rlen:] = ramp.flip(0)
    return w


def tile_blend_window(
    y0: int,
    x0: int,
    th: int,
    tw: int,
    h: int,
    w: int,
    overlap: int,
    *,
    mode: str,
    device,
    dtype,
) -> torch.Tensor:
    wy = _ramp_1d(th, y0, h, overlap, mode=mode, device=device, dtype=dtype)
    wx = _ramp_1d(tw, x0, w, overlap, mode=mode, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


def attention_spatial_tokens(h: int, w: int, spatial_compression: int) -> int:
    return (h // spatial_compression) * (w // spatial_compression)


def attention_matrix_numel(h: int, w: int, spatial_compression: int) -> int:
    n = attention_spatial_tokens(h, w, spatial_compression)
    return n * n


def pad_if_smaller(
    x: torch.Tensor,
    tile_size: int,
    mode: str = "reflect",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad bottom/right so both sides are at least tile_size."""
    h, w = x.shape[-2:]
    ph = max(0, tile_size - h)
    pw = max(0, tile_size - w)
    if ph == 0 and pw == 0:
        return x, (0, 0)
    return _pad_2d(x, 0, pw, 0, ph, mode), (ph, pw)


@torch.no_grad()
def reconstruct_full(
    model,
    x: torch.Tensor,
    *,
    spatial_compression: int,
    padding_mode: str = "reflect",
    return_aux: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"expected [B,1,H,W], got {tuple(x.shape)}")
    x_pad, pads = pad_to_multiple(x, spatial_compression, mode=padding_mode)
    latent_hw = (
        x_pad.shape[-2] // spatial_compression,
        x_pad.shape[-1] // spatial_compression,
    )
    y = model.reconstruct(x_pad)
    y_out = unpad(y, pads)
    if y_out.shape[-2:] != x.shape[-2:]:
        raise RuntimeError(f"full recon shape {tuple(y_out.shape)} != input {tuple(x.shape)}")
    if not return_aux:
        return y_out
    aux = {
        "mode": "full",
        "input_hw": [int(x.shape[-2]), int(x.shape[-1])],
        "padded_hw": [int(x_pad.shape[-2]), int(x_pad.shape[-1])],
        "pad_hw": [int(pads[0]), int(pads[1])],
        "latent_hw": [int(latent_hw[0]), int(latent_hw[1])],
        "attention_tokens": int(latent_hw[0] * latent_hw[1]),
        "attention_matrix_numel": int(attention_matrix_numel(x_pad.shape[-2], x_pad.shape[-1], spatial_compression)),
        "padding_mode": padding_mode,
    }
    return y_out, aux


@torch.no_grad()
def reconstruct_one_tile(
    model,
    tile: torch.Tensor,
    *,
    spatial_compression: int,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """Posterior-mean recon of one tile. Same math as the inner tiled loop."""
    if tile.ndim != 4 or tile.shape[1] != 1:
        raise ValueError(f"expected [B,1,H,W] tile, got {tuple(tile.shape)}")
    want_h, want_w = int(tile.shape[-2]), int(tile.shape[-1])
    tile_p, tpad = pad_to_multiple(tile, spatial_compression, mode=padding_mode)
    recon = unpad(model.reconstruct(tile_p), tpad)
    th, tw = recon.shape[-2:]
    if th != want_h or tw != want_w:
        recon = recon[..., :want_h, :want_w]
    return recon


def fuse_tiled_recons(
    x: torch.Tensor,
    boxes: Sequence[Tuple[int, int, int, int]],
    recons: Sequence[torch.Tensor],
    *,
    tile_size: int,
    overlap: int,
    spatial_compression: int,
    padding_mode: str = "reflect",
    blend_mode: str = "linear",
    snap: Optional[int] = None,
    return_aux: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
    """Accumulate tile reconstructions in *box list order* (order-independent compute).

    ``recons[i]`` must correspond to ``boxes[i]``. Fusion is the same weighted
    sum / weight as ``reconstruct_tiled``.
    """
    if len(boxes) != len(recons):
        raise ValueError(f"boxes ({len(boxes)}) vs recons ({len(recons)})")
    if snap is None:
        snap = spatial_compression
    x_work, small_pad = pad_if_smaller(x, tile_size, mode=padding_mode)
    h, w = x_work.shape[-2:]
    out = torch.zeros_like(x_work)
    weight = torch.zeros((x_work.shape[0], 1, h, w), device=x_work.device, dtype=x_work.dtype)
    tile_meta: List[Dict[str, int]] = []
    for (y0, x0, y1, x1), recon in zip(boxes, recons):
        recon = recon.to(device=x_work.device, dtype=x_work.dtype)
        th, tw = recon.shape[-2:]
        wgt = tile_blend_window(
            y0, x0, th, tw, h, w, overlap, mode=blend_mode, device=x_work.device, dtype=x_work.dtype
        ).view(1, 1, th, tw)
        out[:, :, y0 : y0 + th, x0 : x0 + tw] += recon * wgt
        weight[:, :, y0 : y0 + th, x0 : x0 + tw] += wgt
        tile_meta.append({"y0": int(y0), "x0": int(x0), "y1": int(y0 + th), "x1": int(x0 + tw)})
    if float(weight.min()) <= 0:
        raise RuntimeError("tiled weight map has zeros; coverage is incomplete")
    fused = out / weight
    y_out = unpad(fused, small_pad)
    w_out = unpad(weight, small_pad)
    if y_out.shape[-2:] != x.shape[-2:]:
        raise RuntimeError(f"tiled recon shape {tuple(y_out.shape)} != input {tuple(x.shape)}")
    if not return_aux:
        return y_out
    aux = {
        "mode": "tiled",
        "input_hw": [int(x.shape[-2]), int(x.shape[-1])],
        "work_hw": [int(h), int(w)],
        "small_pad_hw": [int(small_pad[0]), int(small_pad[1])],
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "blend_mode": blend_mode,
        "snap": int(snap),
        "n_tiles": len(tile_meta),
        "tiles": tile_meta,
        "weight": w_out,
        "weight_min": float(w_out.min()),
        "weight_max": float(w_out.max()),
        "padding_mode": padding_mode,
    }
    return y_out, aux


@torch.no_grad()
def reconstruct_tiled(
    model,
    x: torch.Tensor,
    *,
    tile_size: int,
    overlap: int,
    spatial_compression: int,
    padding_mode: str = "reflect",
    blend_mode: str = "linear",
    snap: Optional[int] = None,
    return_aux: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
    """Overlap-tile reconstruction with even origins and linear/Hann blend.

    All tiles use the same global-normalized tensor `x` (caller normalizes once).
    Tile order does not affect the result (accumulate then divide).
    """
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"expected [B,1,H,W], got {tuple(x.shape)}")
    if tile_size % spatial_compression != 0:
        raise ValueError(
            f"tile_size={tile_size} must be divisible by spatial_compression={spatial_compression}"
        )
    if blend_mode not in {"linear", "hann"}:
        raise ValueError(f"unknown blend_mode={blend_mode}")
    if overlap >= tile_size:
        raise ValueError("overlap must be < tile_size")
    if snap is None:
        snap = spatial_compression

    x_work, small_pad = pad_if_smaller(x, tile_size, mode=padding_mode)
    h, w = x_work.shape[-2:]
    boxes = tile_boxes(h, w, tile_size, overlap, snap=snap)
    out = torch.zeros_like(x_work)
    weight = torch.zeros((x_work.shape[0], 1, h, w), device=x_work.device, dtype=x_work.dtype)
    tile_meta: List[Dict[str, int]] = []

    for y0, x0, y1, x1 in boxes:
        tile = x_work[:, :, y0:y1, x0:x1]
        recon = reconstruct_one_tile(
            model, tile, spatial_compression=spatial_compression, padding_mode=padding_mode
        )
        th, tw = recon.shape[-2:]
        wgt = tile_blend_window(
            y0, x0, th, tw, h, w, overlap, mode=blend_mode, device=x_work.device, dtype=x_work.dtype
        ).view(1, 1, th, tw)
        out[:, :, y0 : y0 + th, x0 : x0 + tw] += recon * wgt
        weight[:, :, y0 : y0 + th, x0 : x0 + tw] += wgt
        tile_meta.append({"y0": y0, "x0": x0, "y1": y0 + th, "x1": x0 + tw})

    if float(weight.min()) <= 0:
        raise RuntimeError("tiled weight map has zeros; coverage is incomplete")
    fused = out / weight
    y_out = unpad(fused, small_pad)
    w_out = unpad(weight, small_pad)
    if y_out.shape[-2:] != x.shape[-2:]:
        raise RuntimeError(f"tiled recon shape {tuple(y_out.shape)} != input {tuple(x.shape)}")
    if not return_aux:
        return y_out
    aux = {
        "mode": "tiled",
        "input_hw": [int(x.shape[-2]), int(x.shape[-1])],
        "work_hw": [int(h), int(w)],
        "small_pad_hw": [int(small_pad[0]), int(small_pad[1])],
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "blend_mode": blend_mode,
        "snap": int(snap),
        "n_tiles": len(tile_meta),
        "tiles": tile_meta,
        "weight": w_out,
        "weight_min": float(w_out.min()),
        "weight_max": float(w_out.max()),
        "padding_mode": padding_mode,
    }
    return y_out, aux


def seam_mask_from_tiles(
    h: int,
    w: int,
    tiles: Sequence[Dict[str, int]],
    *,
    width: int = 2,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Binary mask of interior tile edges (not image border), dilated by `width`."""
    m = torch.zeros(h, w, device=device, dtype=dtype or torch.float32)
    for t in tiles:
        y0, x0, y1, x1 = t["y0"], t["x0"], t["y1"], t["x1"]
        if y0 > 0:
            m[max(0, y0 - width) : min(h, y0 + width), x0:x1] = 1
        if y1 < h:
            m[max(0, y1 - width) : min(h, y1 + width), x0:x1] = 1
        if x0 > 0:
            m[y0:y1, max(0, x0 - width) : min(w, x0 + width)] = 1
        if x1 < w:
            m[y0:y1, max(0, x1 - width) : min(w, x1 + width)] = 1
    return m


@torch.no_grad()
def center_vs_edge_tile_delta(
    model,
    *,
    tile_size: int,
    spatial_compression: int,
    padding_mode: str = "reflect",
) -> Dict[str, Any]:
    """Architecture probe (works with random weights): same 64-wide strip as
    tile-center vs tile-left-edge. Reports MAE vs distance from the left edge.

    This does *not* replace a trained-checkpoint measurement. It only shows how
    far a missing left context can move the output for this topology.
    """
    ts = int(tile_size)
    canvas = torch.randn(1, 1, ts, ts * 2)
    # left-edge tile = [:, :, :, 0:ts]; center tile of the canvas = [:, :, :, ts//2:ts//2+ts]
    left = reconstruct_full(
        model, canvas[:, :, :, :ts], spatial_compression=spatial_compression, padding_mode=padding_mode
    )
    # reference: the same pixels reconstructed as the right half of a shifted crop that
    # has context on the left
    shifted = reconstruct_full(
        model,
        canvas[:, :, :, ts - ts // 2 : ts - ts // 2 + ts],
        spatial_compression=spatial_compression,
        padding_mode=padding_mode,
    )
    # compare left tile's right half to shifted tile's right half? Simpler:
    # left-edge recon vs full-canvas tiled? Keep it 1d: MAE of left recon vs
    # the overlapping region of a tile that has extra left context.
    # Overlap region: columns [ts//2:ts] of `left` correspond to columns [0:ts//2]
    # of `shifted` only if shifted starts at ts//2. shifted starts at ts - ts//2 = ts//2.
    # canvas[..., ts//2 : ts//2+ts]  -> shifted
    # left is canvas[..., 0:ts]
    # shared pixels: columns ts//2 : ts of canvas.
    # in left: columns ts//2:ts
    # in shifted: columns 0 : ts//2
    a = left[..., ts // 2 :]
    b = shifted[..., : ts - ts // 2]
    d = (a - b).abs().mean(dim=(0, 1, 2))  # [W_shared]
    return {
        "shared_width": int(d.numel()),
        "mae_by_offset_from_shared_left": [float(v) for v in d.cpu()],
        "mae_mean": float(d.mean()),
        "mae_near_edge": float(d[: max(d.numel() // 8, 1)].mean()),
        "mae_far": float(d[d.numel() // 2 :].mean()) if d.numel() > 1 else float(d.mean()),
    }


def pair_overlaps(origins: Sequence[int], tile_size: int) -> List[int]:
    """Actual overlap (pixels) between consecutive same-size tiles."""
    ov = []
    for a, b in zip(origins, origins[1:]):
        ov.append(max(0, a + tile_size - b))
    return ov
