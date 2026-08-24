from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class LossOutput:
    total: torch.Tensor
    unweighted: Dict[str, torch.Tensor]
    weights: Dict[str, float]
    weighted: Dict[str, torch.Tensor]
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)
    # Non-logged tensors (support mask, recon) for GAN / influence. Not in jsonl.
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)
