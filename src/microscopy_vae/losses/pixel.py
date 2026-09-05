from __future__ import annotations

import torch
import torch.nn.functional as F

from microscopy_vae.losses.structure import scharr_magnitude


def per_sample_robust_range(target: torch.Tensor) -> torch.Tensor:
    """Per-sample p99.5-p0.5, shape [B]. FP32 so quantile is safe under bf16 autocast."""
    flat = target.float().reshape(target.shape[0], -1)
    hi = torch.quantile(flat, 0.995, dim=1)
    lo = torch.quantile(flat, 0.005, dim=1)
    return (hi - lo).clamp_min(0.0)


def per_sample_robust_scale(
    target: torch.Tensor,
    *,
    min_scale: float = 0.05,
    low_structure_range: float | torch.Tensor = 0.0,
    low_structure_scale: float = 1.0,
    smooth: bool = False,
) -> torch.Tensor:
    """Per-sample divisor for amp_norm, shape [B,1,1,1].

    Default (low_structure_range=0) matches v2: s = max(range, min_scale).
    If range < low_structure_range, use low_structure_scale instead of amplifying
    empty / near-black patches (that was producing white speckle on dark bg).
    ``low_structure_range`` may be a per-sample tensor [B].
    """
    rng = per_sample_robust_range(target)
    s = rng.clamp_min(min_scale)
    if torch.is_tensor(low_structure_range):
        thr = low_structure_range.to(device=rng.device, dtype=rng.dtype).reshape(-1)
    else:
        thr = rng.new_full(rng.shape, float(low_structure_range))
    if bool((thr > 0).any()):
        s_amp = rng.clamp_min(min_scale)
        s_idle = s.new_full(s.shape, float(low_structure_scale))
        if smooth:
            lo = 0.5 * thr
            hi = thr.clamp_min(1e-8)
            alpha = ((rng - lo) / (hi - lo)).clamp(0.0, 1.0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            s = s_idle * (1.0 - alpha) + s_amp * alpha
        else:
            idle = (thr > 0) & (rng < thr)
            s = torch.where(idle, s_idle, s_amp)
        s = torch.where(thr > 0, s, s_amp)
    return s.view(-1, 1, 1, 1)


def structure_support_mask(
    target: torch.Tensor,
    *,
    kernel: int = 9,
    floor: float | torch.Tensor = 0.02,
    rel: float = 0.25,
    min_density: float = 0.15,
) -> torch.Tensor:
    """Spatially supported structure, not intensity/color.

    A pixel counts only if its Scharr magnitude is above tau *and* the local
    density of such pixels in a k×k window is high enough. Isolated spikes
    (black or grey) make a small gradient ring and fail the density test;
    filaments and puncta pass. Mask is detached. kernel<=1 disables (all-ones).
    """
    if int(kernel) <= 1:
        return torch.ones_like(target)
    k = int(kernel)
    if k % 2 == 0:
        raise ValueError(f"structure support kernel must be odd, got {k}")
    mag = scharr_magnitude(target)  # FP32
    rel_tau = mag.mean(dim=(1, 2, 3), keepdim=True) * float(rel)
    if torch.is_tensor(floor):
        fl = floor.to(device=mag.device, dtype=mag.dtype)
        if fl.ndim == 1:
            fl = fl.view(-1, 1, 1, 1)
        tau = torch.maximum(rel_tau, fl)
    else:
        tau = torch.clamp(rel_tau, min=float(floor))
    high = (mag > tau).to(dtype=mag.dtype)
    pad = k // 2
    density = F.avg_pool2d(
        F.pad(high, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=k,
        stride=1,
    )
    dense_enough = (density >= float(min_density)).to(dtype=mag.dtype)
    return (high * dense_enough).to(dtype=target.dtype).detach()


def masked_spatial_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample mean of x where mask>0. Shape [B]; 0 if a sample has no mass."""
    m = mask.to(dtype=x.dtype)
    mass = m.flatten(1).sum(dim=1)
    num = (x * m).flatten(1).sum(dim=1)
    out = num / mass.clamp_min(1.0)
    return torch.where(mass > 0, out, torch.zeros_like(out))


def target_grad_weight(
    target: torch.Tensor,
    *,
    edge_weight: float,
    clip: float = 0.0,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    """1 + edge_weight * (|∇t| / mean|∇t|). Extra weight only on `support` if given."""
    if edge_weight <= 0:
        return torch.ones_like(target)
    mag = scharr_magnitude(target).clamp_min(0.0)
    if support is not None:
        mass = support.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        denom = (mag * support).sum(dim=(1, 2, 3), keepdim=True) / mass
        denom = denom.clamp_min(1e-6)
        extra = float(edge_weight) * (mag / denom) * support
    else:
        denom = mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        extra = float(edge_weight) * (mag / denom)
    w = 1.0 + extra
    if clip > 1.0:
        w = w.clamp(max=float(clip))
    return w


def dark_false_positive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dark_quantile: float = 0.20,
) -> torch.Tensor:
    """Mean ReLU(pred-target) on the darkest quantile of each target crop.

    Penalizes white speckle / positive bias in black background without
    requiring a signed global mean (Flux can cancel fg-dark vs bg-bright).
    """
    pred_f = pred.float()
    target_f = target.float()
    b = target_f.shape[0]
    flat = target_f.reshape(b, -1)
    q = min(max(float(dark_quantile), 0.01), 0.49)
    thr = torch.quantile(flat, q, dim=1).view(b, 1, 1, 1)
    dark = (target_f <= thr).to(dtype=pred_f.dtype)
    pos = torch.relu(pred_f - target_f)
    denom = dark.flatten(1).sum(dim=1).clamp_min(1.0)
    per = (pos * dark).flatten(1).sum(dim=1) / denom
    return per.mean()


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-3,
    mask: torch.Tensor | None = None,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = pred - target
    per = torch.sqrt(diff * diff + eps * eps)
    if pixel_weight is not None:
        per = per * pixel_weight
    if mask is not None:
        per = per * mask
        denom = mask.sum().clamp_min(1.0)
        return per.sum() / denom
    return per.mean()


def highpass_residual(x: torch.Tensor) -> torch.Tensor:
    """3×3 box high-pass (x - local mean), reflect pad."""
    x_f = x.float()
    k = x_f.new_full((1, 1, 3, 3), 1.0 / 9.0)
    if x_f.shape[1] != 1:
        k = k.repeat(x_f.shape[1], 1, 1, 1)
    padded = F.pad(x_f, (1, 1, 1, 1), mode="reflect")
    blur = F.conv2d(padded, k, groups=x_f.shape[1])
    return x_f - blur


def flux_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute mean intensity bias (per-sample mean, then batch mean).

    Note: sum/numel is mathematically identical to mean; we keep a single term
    so weight w_flux is not silently doubled.
    """
    return (pred.mean(dim=(1, 2, 3)) - target.mean(dim=(1, 2, 3))).abs().mean()
