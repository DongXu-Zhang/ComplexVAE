from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad H,W up to multiple; return padded tensor and (pad_h, pad_w) bottom/right pads."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, (0, 0)
    # F.pad order: left, right, top, bottom
    x_pad = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x_pad, (ph, pw)


def unpad(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    ph, pw = pad_hw
    if ph == 0 and pw == 0:
        return x
    h = x.shape[-2] - ph
    w = x.shape[-1] - pw
    return x[..., :h, :w]


@torch.no_grad()
def reconstruct_full(
    model,
    x: torch.Tensor,
    *,
    spatial_compression: int,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    x_pad, pads = pad_to_multiple(x, spatial_compression, mode=padding_mode)
    y = model.reconstruct(x_pad)
    return unpad(y, pads)


@torch.no_grad()
def reconstruct_tiled(
    model,
    x: torch.Tensor,
    *,
    tile_size: int,
    overlap: int,
    spatial_compression: int,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """Overlap-tile reconstruction with linear blend weights."""
    if overlap >= tile_size:
        raise ValueError("overlap must be < tile_size")
    x_pad, pads = pad_to_multiple(x, spatial_compression, mode=padding_mode)
    b, c, h, w = x_pad.shape
    out = torch.zeros_like(x_pad)
    weight = torch.zeros((b, 1, h, w), device=x_pad.device, dtype=x_pad.dtype)
    step = tile_size - overlap

    def window(th: int, tw: int, *, top: bool, bottom: bool, left: bool, right: bool) -> torch.Tensor:
        """Linear blend only on interior overlaps; outer image border stays weight 1."""
        wy = torch.ones(th, device=x_pad.device, dtype=x_pad.dtype)
        wx = torch.ones(tw, device=x_pad.device, dtype=x_pad.dtype)
        if overlap > 0:
            ramp = torch.linspace(0, 1, overlap, device=x_pad.device, dtype=x_pad.dtype)
            if th > overlap:
                if not top:
                    wy[:overlap] = ramp
                if not bottom:
                    wy[-overlap:] = ramp.flip(0)
            if tw > overlap:
                if not left:
                    wx[:overlap] = ramp
                if not right:
                    wx[-overlap:] = ramp.flip(0)
        return wy[:, None] * wx[None, :]

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            # expand tile to tile_size if possible by shifting back
            if y1 - y0 < tile_size and y1 == h:
                y0 = max(0, h - tile_size)
            if x1 - x0 < tile_size and x1 == w:
                x0 = max(0, w - tile_size)
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile = x_pad[:, :, y0:y1, x0:x1]
            # pad tile to multiple
            tile_p, tpad = pad_to_multiple(tile, spatial_compression, mode=padding_mode)
            recon = model.reconstruct(tile_p)
            recon = unpad(recon, tpad)
            th, tw = recon.shape[-2:]
            wgt = window(
                th,
                tw,
                top=(y0 == 0),
                bottom=(y1 >= h),
                left=(x0 == 0),
                right=(x1 >= w),
            ).view(1, 1, th, tw)
            out[:, :, y0 : y0 + th, x0 : x0 + tw] += recon * wgt
            weight[:, :, y0 : y0 + th, x0 : x0 + tw] += wgt
    out = out / weight.clamp_min(1e-8)
    return unpad(out, pads)
