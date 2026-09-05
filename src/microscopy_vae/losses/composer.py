from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from microscopy_vae.losses.kl import beta_at_step, free_bits_kl, kl_diagnostics
from microscopy_vae.losses.schedule import scheduled_weight
from microscopy_vae.losses.pixel import (
    charbonnier_loss,
    dark_false_positive_loss,
    flux_loss,
    highpass_residual,
    masked_spatial_mean,
    per_sample_robust_range,
    per_sample_robust_scale,
    structure_support_mask,
    target_grad_weight,
)
from microscopy_vae.losses.structure import ms_ssim_loss, scharr_grad_map
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
        idle_loss_mult: float = 1.0,
        structure_support_kernel: int = 0,
        structure_support_floor: float = 0.02,
        structure_support_rel: float = 0.25,
        structure_support_min_density: float = 0.15,
        structure_min_frac: float = 0.0,
        perceptual: Optional[nn.Module] = None,
        perc_weight: float = 0.0,
        perc_start_step: int = 0,
        perc_ramp_steps: int = 0,
        perc_layers: Optional[List[str]] = None,
        perc_layer_weights: Optional[Dict[str, float]] = None,
        perc_distance: str = "l1",
        perc_apply_amp_norm: bool = False,
        perc_unstructured_policy: str = "keep",
        perc_normalize_features: bool = False,
        perc_backbone: str = "internal_conv",
        perc_vgg_kwargs: Optional[dict] = None,
        unstructured_bg_weight: float = 0.0,
        amp_smooth: bool = False,
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
        self.idle_loss_mult = float(idle_loss_mult)
        self.structure_support_kernel = int(structure_support_kernel)
        self.structure_support_floor = float(structure_support_floor)
        self.structure_support_rel = float(structure_support_rel)
        self.structure_support_min_density = float(structure_support_min_density)
        self.structure_min_frac = float(structure_min_frac)
        self.perceptual = perceptual
        self.perc_weight = float(perc_weight)
        self.perc_start_step = int(perc_start_step)
        self.perc_ramp_steps = int(perc_ramp_steps)
        self.perc_layers = list(perc_layers) if perc_layers else ["block1", "block2", "block3"]
        self.perc_layer_weights = perc_layer_weights
        self.perc_distance = str(perc_distance)
        self.perc_apply_amp_norm = bool(perc_apply_amp_norm)
        self.perc_unstructured_policy = str(perc_unstructured_policy)
        self.perc_normalize_features = bool(perc_normalize_features)
        self.perc_backbone = str(perc_backbone)
        self.perc_vgg_kwargs = perc_vgg_kwargs
        self.unstructured_bg_weight = float(unstructured_bg_weight)
        self.amp_smooth = bool(amp_smooth)
        self.per_source_thresholds: Dict[str, Dict[str, float]] = {}

    def set_calibrated_thresholds(self, table: Optional[Dict[str, Dict[str, float]]]) -> None:
        yaml_amp = float(self.amp_low_structure_range)
        out: Dict[str, Dict[str, float]] = {}
        for k, v in (table or {}).items():
            rec = dict(v)
            # Never amplify more aggressively than the yaml amp gate.
            if yaml_amp > 0:
                rec["amp_low_structure_range"] = max(
                    float(rec.get("amp_low_structure_range", yaml_amp)), yaml_amp
                )
            out[str(k)] = rec
        self.per_source_thresholds = out

    def _per_source_tensor(
        self,
        sources: Optional[Sequence[str]],
        key: str,
        default: float,
        *,
        bsz: int,
        device,
        dtype,
    ) -> torch.Tensor:
        vals = []
        for i in range(bsz):
            src = None
            if sources is not None and i < len(sources):
                src = str(sources[i])
            rec = self.per_source_thresholds.get(src or "", {})
            vals.append(float(rec.get(key, default)))
        return torch.tensor(vals, device=device, dtype=dtype)

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
        sources: Optional[Sequence[str]] = None,
    ) -> LossOutput:
        # Pixel/structure terms in FP32 even when the trainer autocasts bf16.
        pred = output.reconstruction.float()
        target = target.float()
        posterior: PosteriorStats = output.posterior
        bsz = int(target.shape[0])
        dtype = pred.dtype
        rng = per_sample_robust_range(target)
        floor_b = self._per_source_tensor(
            sources,
            "structure_support_floor",
            self.structure_support_floor,
            bsz=bsz,
            device=target.device,
            dtype=target.dtype,
        )
        amp_thr_b = self._per_source_tensor(
            sources,
            "amp_low_structure_range",
            self.amp_low_structure_range,
            bsz=bsz,
            device=target.device,
            dtype=rng.dtype,
        )
        support = structure_support_mask(
            target,
            kernel=self.structure_support_kernel,
            floor=floor_b.view(bsz, 1, 1, 1),
            rel=self.structure_support_rel,
            min_density=self.structure_support_min_density,
        )
        if mask is not None:
            support = support * mask.to(dtype=support.dtype)
        support_frac = support.mean(dim=(1, 2, 3))
        # Idle = no spatially supported structure. Do NOT use P99.5 range here:
        # a sparse filament occupies << 0.5% of a 256 crop, so robust range stays
        # near zero and would wrongly treat real structure as an empty crop.
        # Range still controls amp_norm (do not amplify empty / low-contrast).
        idle = torch.zeros(bsz, dtype=torch.bool, device=target.device)
        if self.structure_min_frac > 0:
            idle = idle | (support_frac < float(self.structure_min_frac))
        elif float(self.amp_low_structure_range) > 0 or bool(self.per_source_thresholds):
            idle = idle | ((amp_thr_b > 0) & (rng < amp_thr_b))
        active = (~idle).to(dtype=dtype)
        # Pixel gate is off on idle crops (v1 path: no edge/HF/Scharr boost).
        support_on = support * active.view(bsz, 1, 1, 1)

        pred_s, tgt_s = pred, target
        scale = pred.new_ones((bsz, 1, 1, 1))
        if self.amp_norm:
            scale = per_sample_robust_scale(
                target,
                min_scale=self.amp_norm_min_scale,
                low_structure_range=amp_thr_b,
                low_structure_scale=self.amp_low_structure_scale,
                smooth=self.amp_smooth,
            )
            # Support-idle crops must not be amplified even if a handful of
            # isolated spikes inflate P99.5 enough to pass the range floor.
            if self.structure_min_frac > 0:
                scale = torch.where(
                    idle.view(bsz, 1, 1, 1),
                    scale.new_full(scale.shape, float(self.amp_low_structure_scale)),
                    scale,
                )
            pred_s = pred / scale
            tgt_s = target / scale

        pix_w = target_grad_weight(
            tgt_s,
            edge_weight=self.edge_weight,
            clip=self.edge_weight_clip,
            support=support_on if self.structure_support_kernel > 1 else None,
        )
        if bool(idle.any().item()) and self.edge_weight > 0:
            pix_w = pix_w.clone()
            pix_w[idle] = 1.0
        sample_w = torch.where(
            idle,
            pix_w.new_full((bsz,), self.idle_loss_mult),
            pix_w.new_ones(bsz),
        ).view(bsz, 1, 1, 1)
        l_char = charbonnier_loss(
            pred_s,
            tgt_s,
            eps=self.charbonnier_eps,
            mask=mask,
            pixel_weight=pix_w * sample_w,
        )
        w_ms = self._ms_weight(optimizer_step)
        if w_ms > 0:
            if self.ssim_range_mode == "amp_space":
                pred_ms, tgt_ms, ms_range = pred_s, tgt_s, 1.0
            else:
                pred_ms, tgt_ms, ms_range = pred, target, self.ssim_data_range
            # SSIM windows are 11×11: an isolated spike contaminates a whole
            # neighbourhood. On unstructured pixels, copy target into pred so
            # SSIM cannot reward matching (or penalize not matching) noise.
            if self.structure_support_kernel > 1:
                # lambda=0: copy target off-support (old). lambda>0: leak pred error.
                lam = float(self.unstructured_bg_weight)
                mix = (support_on > 0).to(pred_ms.dtype)
                w = mix + lam * (1.0 - mix)
                pred_ms = w * pred_ms + (1.0 - w) * tgt_ms
            l_ms_ps = ms_ssim_loss(pred_ms, tgt_ms, data_range=ms_range, reduce=False)
            if float(active.sum()) > 0:
                l_ms = (l_ms_ps * active).sum() / active.sum()
            else:
                l_ms = pred.new_zeros(())
        else:
            l_ms = pred.new_zeros(())

        if self.w_grad > 0:
            gmap = scharr_grad_map(pred_s, tgt_s)
            g_ps = masked_spatial_mean(gmap, support_on)
            if float(active.sum()) > 0:
                l_grad = (g_ps * active).sum() / active.sum()
            else:
                l_grad = pred.new_zeros(())
        else:
            l_grad = pred.new_zeros(())

        if self.w_hf > 0:
            hp = torch.sqrt(
                (highpass_residual(pred_s) - highpass_residual(tgt_s)) ** 2
                + self.charbonnier_eps**2
            )
            hp_ps = masked_spatial_mean(hp, support_on)
            if float(active.sum()) > 0:
                l_hf = (hp_ps * active).sum() / active.sum()
            else:
                l_hf = pred.new_zeros(())
        else:
            l_hf = pred.new_zeros(())
        if self.w_flux > 0:
            l_flux = flux_loss(pred, target)
        else:
            l_flux = pred.new_zeros(())
        if self.w_dark_fp > 0:
            l_dark = dark_false_positive_loss(
                pred, target, dark_quantile=self.dark_fp_quantile
            )
        else:
            l_dark = pred.new_zeros(())
        l_kl = free_bits_kl(posterior, free_nats=self.free_nats).float()
        beta = beta_at_step(optimizer_step, t0=self.kl_t0, t1=self.kl_t1, beta_max=self.beta_max)
        w_perc = 0.0
        l_perc = pred.new_zeros(())
        if self.perceptual is not None:
            w_perc = scheduled_weight(
                self.perc_weight, optimizer_step, self.perc_start_step, self.perc_ramp_steps
            )
            if w_perc > 0:
                from microscopy_vae.losses.perceptual import perceptual_feature_loss

                perc_pred = pred_s if self.perc_apply_amp_norm else pred
                perc_tgt = tgt_s if self.perc_apply_amp_norm else target
                vgg_kw = self.perc_vgg_kwargs if self.perc_backbone == "vgg16" else None
                l_perc, perc_diag = perceptual_feature_loss(
                    perc_pred,
                    perc_tgt,
                    self.perceptual,
                    selected_layers=self.perc_layers,
                    layer_weights=self.perc_layer_weights,
                    distance=self.perc_distance,
                    normalize_features=self.perc_normalize_features,
                    unstructured_policy=self.perc_unstructured_policy,
                    support=support_on if self.structure_support_kernel > 1 else None,
                    unstructured_bg_weight=self.unstructured_bg_weight,
                    vgg_adapter_kwargs=vgg_kw,
                )

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
        if self.perceptual is not None:
            weights["perceptual"] = w_perc
            unweighted["perceptual"] = l_perc
        weighted = {k: weights[k] * unweighted[k] for k in unweighted}
        total = sum(weighted.values())
        diag = kl_diagnostics(posterior)
        diag.update(output.diagnostics)
        diag["beta"] = torch.tensor(beta, device=pred.device)
        diag["w_ms_ssim_effective"] = torch.tensor(w_ms, device=pred.device)
        diag["idle_frac"] = idle.to(dtype=dtype).mean().detach()
        diag["support_frac"] = support_frac.mean().detach()
        diag["amp_scale_mean"] = scale.detach().float().mean()
        diag["edge_weight_mean"] = pix_w.detach().float().mean()
        diag["edge_weight_max"] = pix_w.detach().float().amax()
        if self.perceptual is not None:
            diag["w_perc_effective"] = torch.tensor(w_perc, device=pred.device)
            if w_perc > 0:
                diag.update(perc_diag)
        return LossOutput(
            total=total,
            unweighted=unweighted,
            weights=weights,
            weighted=weighted,
            diagnostics=diag,
            aux={
                "reconstruction": pred,
                "target": target,
                "support_on": support_on,
            },
        )
