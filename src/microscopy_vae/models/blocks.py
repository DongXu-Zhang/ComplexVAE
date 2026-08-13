"""Building blocks for the independent microscopy VAE.

Topology intentionally mirrors Diffusers 0.27 Encoder/Decoder ResNet stages
(Hybrid-SD Small asymmetric schedules) but is implemented from scratch with
no dependency on diffusers at runtime.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int, num_groups: int = 32, eps: float = 1e-6) -> nn.GroupNorm:
    if num_channels % num_groups != 0:
        raise ValueError(f"num_channels={num_channels} not divisible by num_groups={num_groups}")
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=True)


class ResnetBlock2D(nn.Module):
    """Pre-activation residual block used in encoder/decoder stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        eps: float = 1e-6,
        groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels, groups, eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_channels, groups, eps)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttentionBlock2D(nn.Module):
    """Single-head spatial self-attention at bottleneck (Diffusers-style)."""

    def __init__(self, channels: int, num_groups: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.norm = _group_norm(channels, num_groups, eps)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).reshape(b, c, h * w).permute(0, 2, 1)  # B, HW, C
        k = self.k(h_).reshape(b, c, h * w)  # B, C, HW
        v = self.v(h_).reshape(b, c, h * w).permute(0, 2, 1)  # B, HW, C
        scale = 1.0 / math.sqrt(c)
        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)  # B, HW, C
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj_out(out)


class Downsample2D(nn.Module):
    """Stride-2 conv downsample (Diffusers Downsample2D with padding=0 + pad)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match Diffusers asymmetric pad for even spatial sizes.
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class Upsample2D(nn.Module):
    """Nearest upsample + conv (avoids transposed-conv checkerboard)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        add_downsample: bool = True,
        groups: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            layers.append(ResnetBlock2D(cin, out_channels, eps=eps, groups=groups))
        self.resnets = nn.ModuleList(layers)
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels)]) if add_downsample else nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resnets:
            x = block(x)
        for down in self.downsamplers:
            x = down(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int = 3,
        add_upsample: bool = True,
        groups: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            layers.append(ResnetBlock2D(cin, out_channels, eps=eps, groups=groups))
        self.resnets = nn.ModuleList(layers)
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resnets:
            x = block(x)
        for up in self.upsamplers:
            x = up(x)
        return x


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        add_attention: bool = True,
        groups: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.resnett_0 = ResnetBlock2D(channels, channels, eps=eps, groups=groups)
        self.attn = AttentionBlock2D(channels, num_groups=groups, eps=eps) if add_attention else None
        self.resnett_1 = ResnetBlock2D(channels, channels, eps=eps, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnett_0(x)
        if self.attn is not None:
            x = self.attn(x)
        x = self.resnett_1(x)
        return x
