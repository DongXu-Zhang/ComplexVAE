from __future__ import annotations

import torch


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-3,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = pred - target
    per = torch.sqrt(diff * diff + eps * eps)
    if mask is not None:
        per = per * mask
        denom = mask.sum().clamp_min(1.0)
        return per.sum() / denom
    return per.mean()


def flux_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute mean intensity bias (per-sample mean, then batch mean).

    Note: sum/numel is mathematically identical to mean; we keep a single term
    so weight w_flux is not silently doubled.
    """
    return (pred.mean(dim=(1, 2, 3)) - target.mean(dim=(1, 2, 3))).abs().mean()
