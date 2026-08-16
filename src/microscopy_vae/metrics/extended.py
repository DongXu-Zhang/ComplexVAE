"""Evaluation-only fidelity helpers. Does not change training loss."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    d = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(d * d))


def mae_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float64) - target.astype(np.float64))))


def rmse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(mse_np(pred, target)))


def psnr_from_mse(mse: float, data_range: float) -> float:
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")
    mse = max(float(mse), 1e-12)
    return float(10.0 * np.log10((data_range**2) / mse))


def psnr_fixed_range(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    return psnr_from_mse(mse_np(pred, target), data_range)


def nmse(pred: np.ndarray, target: np.ndarray) -> float:
    """MSE / variance(target). Scale-invariant; inf if target is constant."""
    t = target.astype(np.float64)
    var = float(t.var())
    if var < 1e-18:
        return float("inf")
    return mse_np(pred, target) / var


def snr_db(pred: np.ndarray, target: np.ndarray) -> float:
    """10 log10(var(target) / MSE). Complementary to NMSE."""
    n = nmse(pred, target)
    if not np.isfinite(n) or n <= 0:
        return float("nan")
    return float(-10.0 * np.log10(n))


def ssim_map_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Single-scale SSIM map, reflect pad. Returns [B,C,H,W]."""
    pred_f = pred.float()
    target_f = target.float()
    channel = pred_f.shape[1]
    coords = torch.arange(window_size, device=pred_f.device, dtype=pred_f.dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5 * 1.5))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(channel, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    pred_f = F.pad(pred_f, (pad, pad, pad, pad), mode="reflect")
    target_f = F.pad(target_f, (pad, pad, pad, pad), mode="reflect")
    mu_x = F.conv2d(pred_f, window, padding=0, groups=channel)
    mu_y = F.conv2d(target_f, window, padding=0, groups=channel)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(pred_f * pred_f, window, padding=0, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(target_f * target_f, window, padding=0, groups=channel) - mu_y2
    sigma_xy = F.conv2d(pred_f * target_f, window, padding=0, groups=channel) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))


def ssim_mean(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pt = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))[None, None]
    tt = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))[None, None]
    return float(ssim_map_torch(pt, tt, data_range=data_range).mean().item())


def robust_foreground_mask(
    image: np.ndarray,
    *,
    k: float = 3.0,
    blur_sigma: float = 1.0,
    min_frac: float = 0.01,
    max_frac: float = 0.85,
) -> Dict[str, np.ndarray | str | float]:
    """Annotation-free foreground mask.

    Bright-on-dark default: pixels above median + k * MAD after light blur.
    If the fraction is outside [min_frac, max_frac], fall back to top-40%
    gradient magnitude (dense ER / inverted contrast).
    """
    x = image.astype(np.float64)
    if blur_sigma > 0:
        t = torch.from_numpy(x.astype(np.float32))[None, None]
        # separable approx via avg pool is not gaussian; use reflect conv
        radius = max(int(round(3 * blur_sigma)), 1)
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g = torch.exp(-(coords**2) / (2 * blur_sigma * blur_sigma))
        g = g / g.sum()
        kx = g.view(1, 1, 1, -1)
        ky = g.view(1, 1, -1, 1)
        t = F.pad(t, (radius, radius, radius, radius), mode="reflect")
        t = F.conv2d(t, kx)
        t = F.conv2d(t, ky)
        xb = t.numpy()[0, 0].astype(np.float64)
    else:
        xb = x
    med = float(np.median(xb))
    mad = float(np.median(np.abs(xb - med))) + 1e-12
    fg = xb > (med + k * 1.4826 * mad)
    frac = float(fg.mean())
    mode = "intensity_mad"
    if frac < min_frac or frac > max_frac:
        gy, gx = np.gradient(xb)
        mag = np.hypot(gx, gy)
        thr = float(np.quantile(mag, 0.60))
        fg = mag >= thr
        frac = float(fg.mean())
        mode = "gradient_top40"
    return {"mask": fg.astype(bool), "mode": mode, "fg_frac": frac, "median": med, "mad": mad}


def fg_bg_error_stats(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    fg = mask.astype(bool)
    bg = ~fg
    out: Dict[str, float] = {}
    for name, m in (("fg", fg), ("bg", bg)):
        if m.any():
            d = pred[m] - target[m]
            out[f"{name}_mae"] = float(np.mean(np.abs(d)))
            out[f"{name}_rmse"] = float(np.sqrt(np.mean(d * d)))
            out[f"{name}_bias"] = float(np.mean(d))
            out[f"{name}_n"] = float(m.sum())
        else:
            out[f"{name}_mae"] = float("nan")
            out[f"{name}_rmse"] = float("nan")
            out[f"{name}_bias"] = float("nan")
            out[f"{name}_n"] = 0.0
    return out


def slice_metric_bundle(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 1.0,
    source_data_range: Optional[float] = None,
) -> Dict[str, float]:
    mse = mse_np(pred, target)
    bundle = {
        "mse": mse,
        "mae": mae_np(pred, target),
        "rmse": float(np.sqrt(mse)),
        "psnr_range1": psnr_from_mse(mse, 1.0),
        "psnr_fixed_range": psnr_from_mse(mse, data_range),
        "nmse": nmse(pred, target),
        "snr_db": snr_db(pred, target),
        "ssim_range1": ssim_mean(pred, target, data_range=1.0),
        "signed_bias": float(pred.astype(np.float64).mean() - target.astype(np.float64).mean()),
        "target_std": float(target.astype(np.float64).std()),
        "target_p01": float(np.quantile(target.astype(np.float64), 0.01)),
        "target_p99": float(np.quantile(target.astype(np.float64), 0.99)),
    }
    if source_data_range is not None:
        bundle["psnr_source_range"] = psnr_from_mse(mse, source_data_range)
    mask_info = robust_foreground_mask(target)
    bundle["fg_mode"] = 0.0 if mask_info["mode"] == "intensity_mad" else 1.0
    bundle["fg_frac"] = float(mask_info["fg_frac"])
    bundle.update(fg_bg_error_stats(pred, target, mask_info["mask"]))  # type: ignore[arg-type]
    return bundle


def volume_pooled_psnr(
    slice_mses: Sequence[float],
    *,
    data_range: float = 1.0,
) -> float:
    """PSNR from mean MSE across slices of one volume (not mean of PSNRs)."""
    if not slice_mses:
        return float("nan")
    return psnr_from_mse(float(np.mean(slice_mses)), data_range)


def macro_mean(rows: Sequence[Mapping[str, float]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(r[key])]
    if not vals:
        return float("nan")
    return float(np.mean(vals))
