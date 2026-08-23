from __future__ import annotations

from microscopy_vae.config.schema import RootConfig
from microscopy_vae.losses.composer import HQCodecLossComposer
from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.systems.hq_codec import HQCodecSystem
from microscopy_vae.tasks.hq_codec import HQCodecTask


def build_hq_codec_system(cfg: RootConfig) -> HQCodecSystem:
    vae = ModelFactory.create_fresh(
        latent_channels=cfg.model.latent_channels,
        encoder_block_out_channels=tuple(cfg.model.encoder_block_out_channels),
        decoder_block_out_channels=tuple(cfg.model.decoder_block_out_channels),
        layers_per_block=cfg.model.layers_per_block,
        norm_num_groups=cfg.model.norm_num_groups,
        mid_block_add_attention=cfg.model.mid_block_add_attention,
        output_activation=cfg.model.output_activation,
        upsample_mode=getattr(cfg.model, "upsample_mode", "nearest"),
        downsample_pad_mode=getattr(cfg.model, "downsample_pad_mode", "asymmetric"),
        downsample_preblur=bool(getattr(cfg.model, "downsample_preblur", False)),
    )
    if cfg.memory.gradient_checkpointing:
        vae.encoder.gradient_checkpointing = True
        vae.decoder.gradient_checkpointing = True
    loss = HQCodecLossComposer(
        w_char=cfg.loss.w_char,
        w_ms_ssim=cfg.loss.w_ms_ssim,
        w_grad=cfg.loss.w_grad,
        w_flux=cfg.loss.w_flux,
        free_nats=cfg.loss.free_nats,
        beta_max=cfg.kl_schedule.beta_max,
        kl_t0=cfg.kl_schedule.t0,
        kl_t1=cfg.kl_schedule.t1,
        ms_ssim_start_step=cfg.loss.ms_ssim_start_step,
        ms_ssim_ramp_steps=getattr(cfg.loss, "ms_ssim_ramp_steps", 500),
        charbonnier_eps=cfg.loss.charbonnier_eps,
        ssim_data_range=cfg.normalization.ssim_data_range,
        amp_norm=bool(getattr(cfg.loss, "amp_norm", False)),
        amp_norm_min_scale=float(getattr(cfg.loss, "amp_norm_min_scale", 0.05)),
        amp_low_structure_range=float(getattr(cfg.loss, "amp_low_structure_range", 0.0)),
        amp_low_structure_scale=float(getattr(cfg.loss, "amp_low_structure_scale", 1.0)),
        edge_weight=float(getattr(cfg.loss, "edge_weight", 0.0)),
        edge_weight_clip=float(getattr(cfg.loss, "edge_weight_clip", 0.0)),
        w_hf=float(getattr(cfg.loss, "w_hf", 0.0)),
        ssim_range_mode=str(getattr(cfg.loss, "ssim_range_mode", "fixed")),
        w_dark_fp=float(getattr(cfg.loss, "w_dark_fp", 0.0)),
        dark_fp_quantile=float(getattr(cfg.loss, "dark_fp_quantile", 0.20)),
        idle_loss_mult=float(getattr(cfg.loss, "idle_loss_mult", 1.0)),
        structure_support_kernel=int(getattr(cfg.loss, "structure_support_kernel", 0)),
        structure_support_floor=float(getattr(cfg.loss, "structure_support_floor", 0.02)),
        structure_support_rel=float(getattr(cfg.loss, "structure_support_rel", 0.25)),
        structure_support_min_density=float(getattr(cfg.loss, "structure_support_min_density", 0.15)),
        structure_min_frac=float(getattr(cfg.loss, "structure_min_frac", 0.0)),
    )
    task = HQCodecTask(vae, loss, sample_posterior=cfg.task.sample_posterior)
    return HQCodecSystem(vae, task)
