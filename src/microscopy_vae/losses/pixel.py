from __future__ import annotations

import torch
import torch.nn.functional as F


def per_sample_robust_range(target: torch.Tensor) -> torch.Tensor:
    """Per-sample p99.5-p0.5, shape [B]."""
    flat = target.reshape(target.shape[0], -1)
    hi = torch.quantile(flat, 0.995, dim=1)
    lo = torch.quantile(flat, 0.005, dim=1)
    return (hi - lo).clamp_min(0.0)


def per_sample_robust_scale(
    target: torch.Tensor,
    *,
    min_scale: float = 0.05,
    low_structure_range: float = 0.0,
    low_structure_scale: float = 1.0,
) -> torch.Tensor:
    """Per-sample divisor for amp_norm, shape [B,1,1,1].

    Default (low_structure_range=0) matches v2: s = max(range, min_scale).
    If range < low_structure_range, use low_structure_scale instead of amplifying
    empty / near-black patches (that was producing white speckle on dark bg).
    """
    rng = per_sample_robust_range(target)
    s = rng.clamp_min(min_scale)
    if low_structure_range > 0:
        idle = rng < float(low_structure_range)
        s = torch.where(idle, s.new_full(s.shape, float(low_structure_scale)), s)
    return s.view(-1, 1, 1, 1)


def target_grad_weight(
    target: torch.Tensor,
    *,
    edge_weight: float,
    clip: float = 0.0,
) -> torch.Tensor:
    """1 + edge_weight * (|∇t| / mean|∇t|). Same Scharr orientation as the grad loss."""
    if edge_weight <= 0:
        return torch.ones_like(target)
    kx = target.new_tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
    ky = target.new_tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
    kx = (kx / kx.abs().sum()).view(1, 1, 3, 3)
    ky = (ky / ky.abs().sum()).view(1, 1, 3, 3)
    t = F.pad(target.float(), (1, 1, 1, 1), mode="reflect")
    mag = (F.conv2d(t, kx).abs() + F.conv2d(t, ky).abs()).clamp_min(0.0)
    denom = mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    w = 1.0 + float(edge_weight) * (mag / denom)
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
