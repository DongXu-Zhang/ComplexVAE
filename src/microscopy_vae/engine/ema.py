from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class EMA:
    """Exponential moving average of parameters (optional Phase 3)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.resync_from(model)

    def resync_from(self, model: nn.Module) -> None:
        """Replace the shadow with a clone of the live trainable weights.

        Needed after DDP construction: wrap broadcasts rank0 weights, but an
        EMA cloned before wrap can still hold a rank-local init.
        """
        model = _unwrap(model)
        self.shadow = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model = _unwrap(model)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert n in self.shadow
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model = _unwrap(model)
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(device=p.device, dtype=p.dtype))
