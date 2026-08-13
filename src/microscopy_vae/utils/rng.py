from __future__ import annotations

import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def derive_seed(base: int, *parts: int) -> int:
    """Stable non-builtin-hash seed derivation."""
    x = (base & 0xFFFFFFFF) ^ 0x9E3779B9
    for p in parts:
        x = (x ^ (p + 0x9E3779B9 + ((x << 6) & 0xFFFFFFFF) + (x >> 2))) & 0xFFFFFFFF
    return int(x)


def torch_generator(device: str, seed: Optional[int] = None):
    import torch

    g = torch.Generator(device=device if device != "cuda" else "cpu")
    if seed is not None:
        g.manual_seed(seed)
    return g
