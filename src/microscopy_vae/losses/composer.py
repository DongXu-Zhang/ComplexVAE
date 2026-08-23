from __future__ import annotations

from typing import Dict, Optional

import torch

from microscopy_vae.losses.kl import beta_at_step, free_bits_kl, kl_diagnostics
from microscopy_vae.losses.pixel import (
    charbonnier_loss,
    dark_false_positive_loss,
    flux_loss,
    highpass_residual,
    per_sample_robust_scale,
    target_grad_weight,
)
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
        amp_norm: bool = False,
        amp_norm_min_scale: float = 0.05,
        amp_low_structure_range: float = 0.0,
        amp_low_structure_scale: float = 1.0,
        edge_weight: float = 0.0,
        edge_weight_clip: float = 0.0,
        w_hf: float = 0.0,
        ssim_range_mode: str = "fixed",
        w_dark_fp: float = 0.0,
        dark_fp_quantile: float = 0.20,
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
        self.amp_norm = bool(amp_norm)
        self.amp_norm_min_scale = float(amp_norm_min_scale)
        self.amp_low_structure_range = float(amp_low_structure_range)
        self.amp_low_structure_scale = float(amp_low_structure_scale)
        self.edge_weight = float(edge_weight)
        self.edge_weight_clip = float(edge_weight_clip)
        self.w_hf = float(w_hf)
        self.ssim_range_mode = str(ssim_range_mode)
        self.w_dark_fp = float(w_dark_fp)
        self.dark_fp_quantile = float(dark_fp_quantile)

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

        pred_s, tgt_s = pred, target
        if self.amp_norm:
            scale = per_sample_robust_scale(
                target,
                min_scale=self.amp_norm_min_scale,
                low_structure_range=self.amp_low_structure_range,
                low_structure_scale=self.amp_low_structure_scale,
            )
            pred_s = pred / scale
            tgt_s = target / scale
        pix_w = target_grad_weight(
            tgt_s, edge_weight=self.edge_weight, clip=self.edge_weight_clip
        )
        l_char = charbonnier_loss(
            pred_s, tgt_s, eps=self.charbonnier_eps, mask=mask, pixel_weight=pix_w
        )
        w_ms = self._ms_weight(optimizer_step)
        if w_ms > 0:
            if self.ssim_range_mode == "amp_space":
                l_ms = ms_ssim_loss(pred_s, tgt_s, data_range=1.0)
            else:
                l_ms = ms_ssim_loss(pred, target, data_range=self.ssim_data_range)
        else:
            l_ms = pred.new_zeros(())
        l_grad = scharr_grad_loss(pred_s, tgt_s)
        if self.w_hf > 0:
            l_hf = charbonnier_loss(
                highpass_residual(pred_s),
                highpass_residual(tgt_s),
                eps=self.charbonnier_eps,
            )
        else:
            l_hf = pred.new_zeros(())
        l_flux = flux_loss(pred, target)
        if self.w_dark_fp > 0:
            l_dark = dark_false_positive_loss(
                pred, target, dark_quantile=self.dark_fp_quantile
            )
        else:
            l_dark = pred.new_zeros(())
        l_kl = free_bits_kl(posterior, free_nats=self.free_nats)
        beta = beta_at_step(optimizer_step, t0=self.kl_t0, t1=self.kl_t1, beta_max=self.beta_max)

        weights: Dict[str, float] = {
            "charbonnier": self.w_char,
            "ms_ssim": w_ms,
            "scharr": self.w_grad,
            "hf": self.w_hf,
            "flux": self.w_flux,
            "dark_fp": self.w_dark_fp,
            "kl": beta,
        }
        unweighted = {
            "charbonnier": l_char,
            "ms_ssim": l_ms,
            "scharr": l_grad,
            "hf": l_hf,
            "flux": l_flux,
            "dark_fp": l_dark,
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
