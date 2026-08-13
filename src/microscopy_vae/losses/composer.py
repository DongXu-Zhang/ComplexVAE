from __future__ import annotations

from typing import Dict, Optional

import torch

from microscopy_vae.losses.kl import beta_at_step, free_bits_kl, kl_diagnostics
from microscopy_vae.losses.pixel import charbonnier_loss, flux_loss
from microscopy_vae.losses.structure import ms_ssim_loss, scharr_grad_loss
from microscopy_vae.losses.types import LossOutput
from microscopy_vae.models.posterior import PosteriorStats
from microscopy_vae.models.vae import VAEOutput


class HQCodecLossComposer:
    """Compose S1 HQ codec losses with explicit weights and schedules."""

    def __init__(
        self,
        *,
        w_char: float = 1.0,
        w_ms_ssim: float = 0.1,
        w_grad: float = 0.05,
        w_flux: float = 0.01,
        free_nats: float = 0.5,
        beta_max: float = 1e-2,
        kl_t0: int = 2000,
        kl_t1: int = 20000,
        ms_ssim_start_step: int = 1000,
        ms_ssim_ramp_steps: int = 500,
        charbonnier_eps: float = 1e-3,
        ssim_data_range: float = 1.0,
    ) -> None:
        self.w_char = w_char
        self.w_ms_ssim = w_ms_ssim
        self.w_grad = w_grad
        self.w_flux = w_flux
        self.free_nats = free_nats
        self.beta_max = beta_max
        self.kl_t0 = kl_t0
        self.kl_t1 = kl_t1
        self.ms_ssim_start_step = ms_ssim_start_step
        self.ms_ssim_ramp_steps = max(int(ms_ssim_ramp_steps), 0)
        self.charbonnier_eps = charbonnier_eps
        self.ssim_data_range = ssim_data_range

    def _ms_weight(self, optimizer_step: int) -> float:
        if optimizer_step < self.ms_ssim_start_step:
            return 0.0
        if self.ms_ssim_ramp_steps <= 0:
            return self.w_ms_ssim
        t = optimizer_step - self.ms_ssim_start_step
        frac = min(1.0, float(t + 1) / float(self.ms_ssim_ramp_steps))
        return self.w_ms_ssim * frac

    def __call__(
        self,
        output: VAEOutput,
        target: torch.Tensor,
        *,
        optimizer_step: int,
        mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        pred = output.reconstruction
        posterior: PosteriorStats = output.posterior

        l_char = charbonnier_loss(pred, target, eps=self.charbonnier_eps, mask=mask)
        w_ms = self._ms_weight(optimizer_step)
        if w_ms > 0:
            l_ms = ms_ssim_loss(pred, target, data_range=self.ssim_data_range)
        else:
            l_ms = pred.new_zeros(())
        l_grad = scharr_grad_loss(pred, target)
        l_flux = flux_loss(pred, target)
        l_kl = free_bits_kl(posterior, free_nats=self.free_nats)
        beta = beta_at_step(optimizer_step, t0=self.kl_t0, t1=self.kl_t1, beta_max=self.beta_max)

        weights: Dict[str, float] = {
            "charbonnier": self.w_char,
            "ms_ssim": w_ms,
            "scharr": self.w_grad,
            "flux": self.w_flux,
            "kl": beta,
        }
        unweighted = {
            "charbonnier": l_char,
            "ms_ssim": l_ms,
            "scharr": l_grad,
            "flux": l_flux,
            "kl": l_kl,
        }
        weighted = {k: weights[k] * unweighted[k] for k in unweighted}
        total = sum(weighted.values())
        diag = kl_diagnostics(posterior)
        diag.update(output.diagnostics)
        diag["beta"] = torch.tensor(beta, device=pred.device)
        diag["w_ms_ssim_effective"] = torch.tensor(w_ms, device=pred.device)
        return LossOutput(
            total=total,
            unweighted=unweighted,
            weights=weights,
            weighted=weighted,
            diagnostics=diag,
        )
