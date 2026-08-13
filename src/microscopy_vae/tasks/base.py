from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch

from microscopy_vae.losses.types import LossOutput


@dataclass(frozen=True)
class TaskCapabilities:
    hq_reconstruction: bool = True
    lr_encoding: bool = False
    paired_restoration: bool = False
    context_2p5d: bool = False
    tiled_inference: bool = True


class Task(Protocol):
    name: str
    capabilities: TaskCapabilities

    def forward_loss(self, batch: Any, *, optimizer_step: int) -> LossOutput: ...
