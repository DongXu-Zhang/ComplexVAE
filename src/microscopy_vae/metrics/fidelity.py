from __future__ import annotations

import math

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    m = mse(pred, target).clamp_min(1e-12)
    return 10.0 * torch.log10((data_range**2) / m)


def signed_mean_bias(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.mean() - target.mean())
