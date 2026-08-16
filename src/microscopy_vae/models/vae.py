"""Independent single-channel MicroscopyVAE (encode / sample / decode)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from microscopy_vae.models.decoder import Decoder
from microscopy_vae.models.encoder import Encoder
from microscopy_vae.models.posterior import PosteriorStats, sample_latent, split_moments


@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    posterior: PosteriorStats
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)


class MicroscopyVAE(nn.Module):
    """HQ codec VAE core. No route logic, no LR, no pretrained hooks."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        encoder_block_out_channels: Sequence[int] = (128, 256, 512, 512),
        decoder_block_out_channels: Sequence[int] = (96, 192, 384, 384),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        output_activation: str = "linear",
        upsample_mode: str = "nearest",
        downsample_pad_mode: str = "asymmetric",
        downsample_preblur: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError("MicroscopyVAE requires single-channel I/O")
        if output_activation != "linear":
            raise ValueError(
                f"output_activation={output_activation!r} rejected; v1 requires linear "
                "(no sigmoid/tanh hard bounds in the training forward)."
            )
        self.latent_channels = latent_channels
        self.output_activation = output_activation
        self.upsample_mode = upsample_mode
        self.downsample_pad_mode = downsample_pad_mode
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            mid_block_add_attention=mid_block_add_attention,
            double_z=True,
            downsample_pad_mode=downsample_pad_mode,
            downsample_preblur=downsample_preblur,
        )
        self.decoder = Decoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            mid_block_add_attention=mid_block_add_attention,
            upsample_mode=upsample_mode,
        )
        if self.encoder.spatial_compression != self.decoder.spatial_compression:
            raise ValueError("Encoder/Decoder spatial compression mismatch")
        self.spatial_compression = self.encoder.spatial_compression
        moments_ch = 2 * latent_channels
        self.quant_conv = nn.Conv2d(moments_ch, moments_ch, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> PosteriorStats:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return split_moments(moments)

    def sample_latent(
        self,
        posterior: PosteriorStats,
        *,
        generator: Optional[torch.Generator] = None,
        sample: bool = True,
    ) -> torch.Tensor:
        return sample_latent(posterior, generator=generator, sample=sample)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> VAEOutput:
        posterior = self.encode(x)
        z = self.sample_latent(posterior, generator=generator, sample=sample_posterior)
        recon = self.decode(z)
        with torch.no_grad():
            oor_hi = (recon > 1.5).float().mean()
            oor_lo = (recon < -0.5).float().mean()
        return VAEOutput(
            reconstruction=recon,
            latent=z,
            posterior=posterior,
            diagnostics={
                "oor_hi_frac": oor_hi.detach(),
                "oor_lo_frac": oor_lo.detach(),
            },
        )

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, sample_posterior=False).reconstruction

    def count_parameters(self) -> Dict[str, int]:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        return {
            "encoder": n(self.encoder),
            "quant_conv": n(self.quant_conv),
            "post_quant_conv": n(self.post_quant_conv),
            "decoder": n(self.decoder),
            "total": n(self),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def trainability_audit(self) -> Dict[str, Any]:
        rows = []
        for name, p in self.named_parameters():
            rows.append(
                {
                    "name": name,
                    "shape": list(p.shape),
                    "requires_grad": bool(p.requires_grad),
                    "numel": int(p.numel()),
                }
            )
        all_train = all(r["requires_grad"] for r in rows)
        return {"all_core_trainable": all_train, "parameters": rows}
