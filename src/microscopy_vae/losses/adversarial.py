"""Conditional/unconditional PatchGAN + hinge/LSGAN losses.

S1 HQ codec has input == target. Concat(x, x) vs concat(x, recon) is a
degenerate real/fake cue (identical channels ⇒ real). Default conditioning
is therefore `none`: D(target) vs D(recon). `input` remains available for
later paired routes and is documented as unsafe for S1.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_sn(module: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return module
    return nn.utils.spectral_norm(module)


class PatchDiscriminator(nn.Module):
    """PatchGAN (Isola et al.): C64-C128-... no BatchNorm (SN only, unbounded data)."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        ndf: int = 32,
        n_layers: int = 3,
        kernel_size: int = 4,
        spectral_norm: bool = True,
        conditioning: str = "none",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if kernel_size < 3:
            raise ValueError("kernel_size must be >= 3")
        if conditioning not in ("none", "input"):
            raise ValueError(f"unknown conditioning {conditioning!r}")
        self.conditioning = conditioning
        ch_in = int(in_channels) + (int(in_channels) if conditioning == "input" else 0)
        pad = int(kernel_size) // 2
        layers: list[nn.Module] = []
        # stride-2 stem
        layers.append(_maybe_sn(nn.Conv2d(ch_in, ndf, kernel_size, stride=2, padding=pad), spectral_norm))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, ndf * 8)
            layers.append(_maybe_sn(nn.Conv2d(nf_prev, nf, kernel_size, stride=2, padding=pad), spectral_norm))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        nf_prev = nf
        nf = min(nf * 2, ndf * 8)
        layers.append(_maybe_sn(nn.Conv2d(nf_prev, nf, kernel_size, stride=1, padding=pad), spectral_norm))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        layers.append(_maybe_sn(nn.Conv2d(nf, 1, kernel_size, stride=1, padding=pad), spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError(f"discriminator expects [B,C,H,W], got {tuple(y.shape)}")
        if self.conditioning == "input":
            if cond is None:
                raise ValueError("conditioning=input requires cond tensor")
            x = torch.cat([cond, y], dim=1)
        else:
            x = y
        return self.net(x.float())


def _spatial_mask(mask: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Downsample structure support onto the PatchGAN map.

    Natural-image SR uses the full frame. Here a 1-pixel filament must still
    mark its patch as structured: nearest-resize of a binary mask drops it.
    Adaptive max-pool keeps a patch if ANY support pixel falls inside.
    """
    m = mask.float()
    if m.shape[-2:] == hw:
        return m
    return F.adaptive_max_pool2d(m, output_size=hw)


def _masked_mean(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits.mean()
    m = _spatial_mask(mask, logits.shape[-2:]).to(dtype=logits.dtype)
    mass = m.sum().clamp_min(1.0)
    return (logits * m).sum() / mass


def hinge_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_real = _masked_mean(F.relu(1.0 - real_logits), mask)
    loss_fake = _masked_mean(F.relu(1.0 + fake_logits), mask)
    return loss_real + loss_fake, loss_real, loss_fake


def hinge_generator_loss(fake_logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return -_masked_mean(fake_logits, mask)


def lsgan_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_real = _masked_mean((real_logits - 1.0) ** 2, mask)
    loss_fake = _masked_mean(fake_logits ** 2, mask)
    return 0.5 * (loss_real + loss_fake), loss_real, loss_fake


def lsgan_generator_loss(fake_logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return 0.5 * _masked_mean((fake_logits - 1.0) ** 2, mask)


def r1_penalty(real_logits: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """Zero-centered R1 on real images. Caller must pass real_img with requires_grad."""
    grad = torch.autograd.grad(
        real_logits.sum(), real_img, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad.float().pow(2).reshape(grad.shape[0], -1).sum(dim=1).mean()


def discriminator_scores(
    disc: PatchDiscriminator,
    *,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    gan_loss: str = "hinge",
    r1_gamma: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """D update tensors. `fake` must already be detached by the caller."""
    real_logits = disc(real, cond=cond)
    fake_logits = disc(fake, cond=cond)
    if gan_loss == "hinge":
        total, l_real, l_fake = hinge_discriminator_loss(real_logits, fake_logits, mask)
    elif gan_loss == "lsgan":
        total, l_real, l_fake = lsgan_discriminator_loss(real_logits, fake_logits, mask)
    else:
        raise ValueError(f"unknown gan_loss {gan_loss!r}")
    if r1_gamma > 0:
        real_rg = real.detach().requires_grad_(True)
        real_logits_r1 = disc(real_rg, cond=None if cond is None else cond.detach())
        total = total + float(r1_gamma) * 0.5 * r1_penalty(real_logits_r1, real_rg)
    return {
        "loss_d": total,
        "loss_d_real": l_real,
        "loss_d_fake": l_fake,
        "d_real_mean": _masked_mean(real_logits, mask).detach(),
        "d_fake_mean": _masked_mean(fake_logits, mask).detach(),
    }


def generator_adv_loss(
    disc: PatchDiscriminator,
    fake: torch.Tensor,
    *,
    cond: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    gan_loss: str = "hinge",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """G adversarial term. `fake` is NOT detached. D params should have requires_grad=False."""
    fake_logits = disc(fake, cond=cond)
    if gan_loss == "hinge":
        loss = hinge_generator_loss(fake_logits, mask)
    elif gan_loss == "lsgan":
        loss = lsgan_generator_loss(fake_logits, mask)
    else:
        raise ValueError(f"unknown gan_loss {gan_loss!r}")
    return loss, _masked_mean(fake_logits, mask).detach()
