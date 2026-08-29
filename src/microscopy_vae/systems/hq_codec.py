from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn

from microscopy_vae.models.factory import architecture_id
from microscopy_vae.models.latent_spec import LatentSpec, build_latent_spec, fit_latent_center_scale
from microscopy_vae.models.vae import MicroscopyVAE
from microscopy_vae.tasks.base import TaskCapabilities
from microscopy_vae.tasks.hq_codec import HQCodecTask


class HQCodecSystem(nn.Module):
    """Owns VAE parameters for HQ-only training. No LR restore capability."""

    def __init__(
        self,
        vae: MicroscopyVAE,
        task: HQCodecTask,
        perceptual: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.task = task
        # Frozen feature extractor (optional). Excluded from parameters() / G optimizer.
        self.perceptual = perceptual
        self.capabilities = TaskCapabilities(
            hq_reconstruction=True,
            lr_encoding=False,
            paired_restoration=False,
            context_2p5d=False,
            tiled_inference=True,
        )

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:  # type: ignore[override]
        return self.vae.parameters(recurse=recurse)

    def trainability_audit(self) -> Dict[str, Any]:
        return self.vae.trainability_audit()

    def encode_hq(self, hq: torch.Tensor, *, sample_posterior: bool = False, generator=None):
        post = self.vae.encode(hq)
        z = self.vae.sample_latent(post, generator=generator, sample=sample_posterior)
        return post, z

    def encode_page(self, hq: torch.Tensor, *, padding_mode: str = "reflect", sample_posterior: bool = False):
        """Pad to f, encode. Returns (latent, posterior, metadata). Default = posterior mean."""
        from microscopy_vae.inference.tiling import encode_full

        return encode_full(
            self.vae,
            hq,
            spatial_compression=int(self.vae.spatial_compression),
            padding_mode=padding_mode,
            sample_posterior=sample_posterior,
        )

    def decode_page(
        self,
        z: torch.Tensor,
        *,
        pad_hw=(0, 0),
        output_hw=None,
        output_domain: str = "normalized",
    ) -> torch.Tensor:
        from microscopy_vae.inference.tiling import decode_full

        if output_domain != "normalized":
            raise ValueError("Only normalized domain decode in core; inverse norm is caller's job")
        return decode_full(self.vae, z, pad_hw=pad_hw, output_hw=output_hw)

    def decode_hq(self, z: torch.Tensor, *, output_domain: str = "normalized") -> torch.Tensor:
        if output_domain != "normalized":
            raise ValueError("Only normalized domain decode in core; inverse norm is caller's job")
        return self.vae.decode(z)

    def reconstruct_hq(self, hq: torch.Tensor) -> torch.Tensor:
        return self.vae.reconstruct(hq)

    def encode_lr(self, *args, **kwargs):
        raise AttributeError("HQCodecSystem does not support encode_lr (capability lr_encoding=False)")

    def restore_lr(self, *args, **kwargs):
        raise AttributeError("HQCodecSystem does not support restore_lr (capability paired_restoration=False)")

    def export_latent_spec(
        self,
        *,
        weights_sha256: str,
        config_sha256: str,
        normalizer_sha256: str,
        center_scale_from_means,
        transform_id: str,
        stats_sha256: str = "pending",
    ) -> LatentSpec:
        center, scale = fit_latent_center_scale(center_scale_from_means)
        return build_latent_spec(
            architecture_id=architecture_id(self.vae),
            weights_sha256=weights_sha256,
            config_sha256=config_sha256,
            normalizer_sha256=normalizer_sha256,
            spatial_compression=self.vae.spatial_compression,
            latent_channels=self.vae.latent_channels,
            center=center,
            scale=scale,
            stats_sha256=stats_sha256,
            transform_id=transform_id,
        )
