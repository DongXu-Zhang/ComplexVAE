"""Independent VAE decoder with asymmetric channel schedule support."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # explicit: torch.utils has no .checkpoint until imported

from microscopy_vae.models.blocks import UpDecoderBlock2D, UNetMidBlock2D, _group_norm


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 4,
        block_out_channels: Sequence[int] = (96, 192, 384, 384),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError("This project requires out_channels=1")
        self.block_out_channels = tuple(block_out_channels)
        # Diffusers decoder uses layers_per_block + 1 resnets per up stage.
        self.layers_per_up = layers_per_block + 1
        self.conv_in = nn.Conv2d(latent_channels, self.block_out_channels[-1], kernel_size=3, padding=1)
        self.mid_block = UNetMidBlock2D(
            self.block_out_channels[-1],
            add_attention=mid_block_add_attention,
            groups=norm_num_groups,
        )

        reversed_channels = list(reversed(self.block_out_channels))
        ups = []
        # First stage consumes mid channels (= reversed_channels[0]).
        prev = reversed_channels[0]
        for i, cout in enumerate(reversed_channels):
            is_final = i == len(reversed_channels) - 1
            ups.append(
                UpDecoderBlock2D(
                    in_channels=prev,
                    out_channels=cout,
                    num_layers=self.layers_per_up,
                    add_upsample=not is_final,
                    groups=norm_num_groups,
                )
            )
            prev = cout
        self.up_blocks = nn.ModuleList(ups)
        self.conv_norm_out = _group_norm(self.block_out_channels[0], norm_num_groups)
        self.conv_out = nn.Conv2d(self.block_out_channels[0], out_channels, kernel_size=3, padding=1)
        self.spatial_compression = 2 ** (len(self.block_out_channels) - 1)
        self.gradient_checkpointing = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"Decoder expects [B,C,H,W], got {tuple(z.shape)}")
        sample = self.conv_in(z)
        if self.training and self.gradient_checkpointing:
            sample = torch.utils.checkpoint.checkpoint(self.mid_block, sample, use_reentrant=False)
            for block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(block, sample, use_reentrant=False)
        else:
            sample = self.mid_block(sample)
            for block in self.up_blocks:
                sample = block(sample)
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)  # linear head
        return sample
