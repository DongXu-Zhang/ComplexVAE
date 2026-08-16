"""Independent VAE encoder with asymmetric channel schedule support."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # explicit: torch.utils has no .checkpoint until imported

from microscopy_vae.models.blocks import DownEncoderBlock2D, UNetMidBlock2D, _group_norm


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        double_z: bool = True,
        downsample_pad_mode: str = "asymmetric",
        downsample_preblur: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("This project requires in_channels=1")
        self.layers_per_block = layers_per_block
        self.block_out_channels = tuple(block_out_channels)
        self.conv_in = nn.Conv2d(in_channels, self.block_out_channels[0], kernel_size=3, padding=1)

        downs = []
        out_c = self.block_out_channels[0]
        for i, cout in enumerate(self.block_out_channels):
            cin = out_c
            out_c = cout
            is_final = i == len(self.block_out_channels) - 1
            downs.append(
                DownEncoderBlock2D(
                    cin,
                    out_c,
                    num_layers=layers_per_block,
                    add_downsample=not is_final,
                    groups=norm_num_groups,
                    downsample_pad_mode=downsample_pad_mode,
                    downsample_preblur=downsample_preblur,
                )
            )
        self.down_blocks = nn.ModuleList(downs)
        self.mid_block = UNetMidBlock2D(
            self.block_out_channels[-1],
            add_attention=mid_block_add_attention,
            groups=norm_num_groups,
        )
        self.conv_norm_out = _group_norm(self.block_out_channels[-1], norm_num_groups)
        self.conv_out = nn.Conv2d(
            self.block_out_channels[-1],
            2 * latent_channels if double_z else latent_channels,
            kernel_size=3,
            padding=1,
        )
        self.spatial_compression = 2 ** (len(self.block_out_channels) - 1)
        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.ndim != 4 or sample.shape[1] != 1:
            raise ValueError(f"Encoder expects [B,1,H,W], got {tuple(sample.shape)}")
        h, w = sample.shape[-2:]
        if h % self.spatial_compression != 0 or w % self.spatial_compression != 0:
            raise ValueError(
                f"H,W=({h},{w}) must be divisible by spatial_compression={self.spatial_compression}"
            )
        sample = self.conv_in(sample)
        if self.training and self.gradient_checkpointing:
            for block in self.down_blocks:
                sample = torch.utils.checkpoint.checkpoint(block, sample, use_reentrant=False)
            sample = torch.utils.checkpoint.checkpoint(self.mid_block, sample, use_reentrant=False)
        else:
            for block in self.down_blocks:
                sample = block(sample)
            sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)
        return sample
