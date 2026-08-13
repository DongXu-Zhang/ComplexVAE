"""LR schedules with explicit warm-up then cosine decay (optimizer-step based)."""

from __future__ import annotations

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    """Lambda multiplies base_lr. Ensures min_lr via ratio min_lr/base_lr."""
    if base_lr <= 0:
        raise ValueError("base_lr must be > 0")
    min_ratio = min_lr / base_lr
    warmup_steps = max(int(warmup_steps), 0)
    max_steps = max(int(max_steps), 1)

    def lr_lambda(step: int) -> float:
        # LambdaLR calls with last_epoch which starts at -1 then 0 after first step
        # We use step as the number of times scheduler.step() has been called.
        if step < 0:
            step = 0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        # cosine from warmup_steps .. max_steps
        t = step - warmup_steps
        denom = max(max_steps - warmup_steps, 1)
        progress = min(max(t / denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # map cosine in [0,1] to [min_ratio, 1]
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
