from __future__ import annotations

import torch

from microscopy_vae.models.posterior import PosteriorStats, kl_to_standard_normal_elements


def free_bits_kl(
    posterior: PosteriorStats,
    *,
    free_nats: float = 0.5,
) -> torch.Tensor:
    """Mean over all elements of max(KL_element - free_nats, 0)."""
    kl_el = kl_to_standard_normal_elements(posterior)
    return torch.clamp(kl_el - free_nats, min=0.0).mean()


def kl_diagnostics(posterior: PosteriorStats) -> dict[str, torch.Tensor]:
    kl_el = kl_to_standard_normal_elements(posterior)
    # per-channel mean KL
    per_ch = kl_el.mean(dim=(0, 2, 3))
    active = (per_ch > 0.01).float().mean()
    return {
        "kl_mean": kl_el.mean().detach(),
        "kl_per_channel_mean": per_ch.detach(),
        "active_unit_frac": active.detach(),
        "mu_var": posterior.mean.var(unbiased=False).detach(),
        "logvar_mean": posterior.logvar.mean().detach(),
    }


def beta_at_step(
    step: int,
    *,
    t0: int = 2000,
    t1: int = 20000,
    beta_max: float = 1e-2,
) -> float:
    if step < t0:
        return 0.0
    if step >= t1:
        return float(beta_max)
    return float(beta_max) * float(step - t0) / float(max(t1 - t0, 1))
