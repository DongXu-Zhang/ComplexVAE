"""Weight initialization for fresh_init only."""

from __future__ import annotations

import torch.nn as nn


def init_module_kaiming(module: nn.Module) -> None:
    """Apply Kaiming normal to Conv/Linear; zeros bias; GN ones/zeros."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def assert_no_nan_params(module: nn.Module) -> None:
    for name, p in module.named_parameters():
        if not p.isfinite().all():
            raise RuntimeError(f"Non-finite parameter at init: {name}")
