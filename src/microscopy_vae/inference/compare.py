"""Same-image full vs tiled comparison. Evaluation only; no training changes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from microscopy_vae.inference.tiling import (
    reconstruct_full,
    reconstruct_tiled,
    seam_mask_from_tiles,
)
from microscopy_vae.metrics.extended import (
    background_false_positive_stats,
    fg_bg_error_stats,
    mae_np,
    mse_np,
    nmse,
    psnr_from_mse,
    robust_foreground_mask,
    ssim_mean,
)


def _finite_stats(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    mse = mse_np(pred, target)
    return {
        "mae": mae_np(pred, target),
        "mse": mse,
        "psnr_range1": psnr_from_mse(mse, 1.0),
        "nmse": nmse(pred, target),
        "ssim_range1": ssim_mean(pred, target, data_range=1.0),
        "signed_bias": float(pred.astype(np.float64).mean() - target.astype(np.float64).mean()),
    }


def _speckle_counts(pred: np.ndarray, target: np.ndarray, *, tau: float = 0.02, q: float = 0.20) -> Dict[str, float]:
    t = target.astype(np.float64)
    p = pred.astype(np.float64)
    dark = t <= float(np.quantile(t, q))
    hot = dark & ((p - t) > tau)
    n_comp, areas = _connected_areas(hot)
    return {
        "dark_hot_pixels": float(hot.sum()),
        "n_components": float(n_comp),
        "max_component": float(max(areas) if areas else 0),
        "mean_component": float(np.mean(areas) if areas else 0.0),
    }


def _connected_areas(mask: np.ndarray) -> tuple:
    """4-connected components on a 2D bool array. Returns (n, list of areas)."""
    vis = np.zeros(mask.shape, dtype=np.uint8)
    h, w = mask.shape
    areas = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or vis[y, x]:
                continue
            stack = [(y, x)]
            vis[y, x] = 1
            a = 0
            while stack:
                cy, cx = stack.pop()
                a += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = 1
                        stack.append((ny, nx))
            areas.append(a)
    return len(areas), areas


def pair_metrics(
    full: np.ndarray,
    tiled: np.ndarray,
    target: np.ndarray,
    *,
    seam: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "full": _finite_stats(full, target),
        "tiled": _finite_stats(tiled, target),
        "full_vs_tiled": _finite_stats(full, tiled),
        "full_speckle": _speckle_counts(full, target),
        "tiled_speckle": _speckle_counts(tiled, target),
    }
    out["full"].update(background_false_positive_stats(full, target))
    out["tiled"].update(background_false_positive_stats(tiled, target))
    fg = robust_foreground_mask(target)
    out["full"].update(fg_bg_error_stats(full, target, fg["mask"]))  # type: ignore[arg-type]
    out["tiled"].update(fg_bg_error_stats(tiled, target, fg["mask"]))  # type: ignore[arg-type]
    out["fg_mode"] = fg["mode"]
    out["fg_frac"] = float(fg["fg_frac"])
    if seam is not None and seam.any() and (~seam.astype(bool)).any():
        d_full = np.abs(full.astype(np.float64) - target.astype(np.float64))
        d_til = np.abs(tiled.astype(np.float64) - target.astype(np.float64))
        sm = seam.astype(bool)
        nsm = ~sm
        out["seam"] = {
            "full_mae_seam": float(d_full[sm].mean()),
            "full_mae_nonseam": float(d_full[nsm].mean()),
            "tiled_mae_seam": float(d_til[sm].mean()),
            "tiled_mae_nonseam": float(d_til[nsm].mean()),
        }
        t_s, t_n = out["seam"]["tiled_mae_seam"], out["seam"]["tiled_mae_nonseam"]
        out["seam"]["tiled_seam_ratio"] = float(t_s / t_n) if t_n > 0 else float("nan")
    return out


@torch.no_grad()
def run_full_tiled_compare(
    model,
    x: torch.Tensor,
    *,
    spatial_compression: int,
    tile_size: int,
    overlap: int,
    padding_mode: str = "reflect",
    blend_mode: str = "linear",
    target: Optional[torch.Tensor] = None,
    devices: Optional[list] = None,
    cfg_dump: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """x is already normalized [1,1,H,W] (per-source or global). target defaults to x."""
    tgt = target if target is not None else x
    t0 = time.perf_counter()
    full, full_aux = reconstruct_full(
        model, x, spatial_compression=spatial_compression, padding_mode=padding_mode, return_aux=True
    )
    t_full = time.perf_counter() - t0
    t1 = time.perf_counter()
    use_mp = devices is not None and len(devices) > 1 and cfg_dump is not None
    if use_mp:
        from microscopy_vae.inference.parallel import run_tiled

        tiled, tiled_aux = run_tiled(
            model,
            x,
            cfg_dump=cfg_dump,
            devices=devices,
            tile_size=tile_size,
            overlap=overlap,
            spatial_compression=spatial_compression,
            padding_mode=padding_mode,
            blend_mode=blend_mode,
            return_aux=True,
        )
    else:
        tiled, tiled_aux = reconstruct_tiled(
            model,
            x,
            tile_size=tile_size,
            overlap=overlap,
            spatial_compression=spatial_compression,
            padding_mode=padding_mode,
            blend_mode=blend_mode,
            return_aux=True,
        )
    t_til = time.perf_counter() - t1
    full_np = full.squeeze().detach().cpu().numpy()
    tiled_np = tiled.squeeze().detach().cpu().numpy()
    tgt_np = tgt.squeeze().detach().cpu().numpy()
    h, w = int(x.shape[-2]), int(x.shape[-1])
    seam = seam_mask_from_tiles(h, w, tiled_aux["tiles"], width=2).cpu().numpy()
    metrics = pair_metrics(full_np, tiled_np, tgt_np, seam=seam)
    metrics["timing_s"] = {"full": float(t_full), "tiled": float(t_til)}
    metrics["full_aux"] = {k: v for k, v in full_aux.items()}
    metrics["tiled_aux"] = {k: v for k, v in tiled_aux.items() if k != "weight"}
    weight = tiled_aux["weight"].squeeze().detach().cpu().numpy()
    return {
        "full": full_np,
        "tiled": tiled_np,
        "target": tgt_np,
        "residual_full": full_np - tgt_np,
        "residual_tiled": tiled_np - tgt_np,
        "diff_full_tiled": full_np - tiled_np,
        "weight": weight,
        "seam": seam.astype(np.float32),
        "metrics": metrics,
    }


def save_compare_pack(pack: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in (
        "full",
        "tiled",
        "target",
        "residual_full",
        "residual_tiled",
        "diff_full_tiled",
        "weight",
        "seam",
    ):
        np.save(out_dir / f"{key}.npy", pack[key].astype(np.float32))
    (out_dir / "metrics.json").write_text(
        json.dumps(pack["metrics"], indent=2, default=float) + "\n", encoding="utf-8"
    )


def load_infer_weights(path: Path, model: torch.nn.Module, *, use_ema: bool = True) -> str:
    """Load export or resume_exact checkpoint. Prefer EMA when present (matches val)."""
    from microscopy_vae.engine.checkpoint import _torch_load
    from microscopy_vae.engine.ema import EMA

    payload = _torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Unrecognized weights file")
    if "model" in payload:
        extra = payload.get("extra") or {}
        from microscopy_vae.engine.checkpoint import load_vae_state_dict

        load_vae_state_dict(model, payload["model"], extra=extra if isinstance(extra, dict) else None)
        ema_sd = extra.get("ema") if isinstance(extra, dict) else None
        if use_ema and isinstance(ema_sd, dict) and ema_sd:
            ema = EMA(model, decay=0.999)
            ema.load_state_dict(ema_sd)
            ema.copy_to(model)
            return "ema"
        return "model"
    from microscopy_vae.engine.checkpoint import load_vae_state_dict

    load_vae_state_dict(model, payload, extra=None)
    return "state_dict"
