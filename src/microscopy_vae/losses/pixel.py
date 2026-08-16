from __future__ import annotations

import torch
import torch.nn.functional as F


def per_sample_robust_scale(target: torch.Tensor, *, min_scale: float = 0.05) -> torch.Tensor:
    """Per-sample p99.5-p0.5 range, shape [B,1,1,1]."""
    flat = target.reshape(target.shape[0], -1)
    hi = torch.quantile(flat, 0.995, dim=1)
    lo = torch.quantile(flat, 0.005, dim=1)
    return (hi - lo).clamp_min(min_scale).view(-1, 1, 1, 1)


def target_grad_weight(target: torch.Tensor, *, edge_weight: float) -> torch.Tensor:
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
    return 1.0 + float(edge_weight) * (mag / denom)


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
