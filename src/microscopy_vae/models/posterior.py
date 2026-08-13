"""Diagonal Gaussian posterior utilities (VAE math)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PosteriorStats:
    """Moments of q(z|x) = N(mean, diag(exp(logvar)))."""

    mean: torch.Tensor
    logvar: torch.Tensor

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)

    @property
    def var(self) -> torch.Tensor:
        return torch.exp(self.logvar)


def clamp_logvar(logvar: torch.Tensor, min_val: float = -30.0, max_val: float = 20.0) -> torch.Tensor:
    return torch.clamp(logvar, min=min_val, max=max_val)


def split_moments(moments: torch.Tensor) -> PosteriorStats:
    """Split [B, 2C, H, W] into mean/logvar along channel dim."""
    if moments.ndim != 4:
        raise ValueError(f"moments must be 4D, got {tuple(moments.shape)}")
    if moments.shape[1] % 2 != 0:
        raise ValueError(f"moments channels must be even, got {moments.shape[1]}")
    mean, logvar = torch.chunk(moments, 2, dim=1)
    logvar = clamp_logvar(logvar)
    return PosteriorStats(mean=mean, logvar=logvar)


def sample_latent(
    posterior: PosteriorStats,
    *,
    generator: Optional[torch.Generator] = None,
    sample: bool = True,
) -> torch.Tensor:
    if not sample:
        return posterior.mean
    eps = torch.randn(
        posterior.mean.shape,
        dtype=posterior.mean.dtype,
        device=posterior.mean.device,
        generator=generator,
    )
    return posterior.mean + posterior.std * eps


def kl_to_standard_normal_elements(posterior: PosteriorStats) -> torch.Tensor:
    """Per-element KL to N(0,I): 0.5 * (μ² + σ² - 1 - log σ²), shape [B,C,H,W]."""
    return 0.5 * (posterior.mean.pow(2) + posterior.var - 1.0 - posterior.logvar)


def kl_sum_spatial(posterior: PosteriorStats) -> torch.Tensor:
    """Sum KL over C,H,W → shape [B] (Diffusers DiagonalGaussianDistribution.kl)."""
    return kl_to_standard_normal_elements(posterior).sum(dim=(1, 2, 3))
