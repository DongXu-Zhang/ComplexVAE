import torch
from torch import nn

from microscopy_vae.engine.schedulers import build_warmup_cosine_scheduler


def test_warmup_then_decay():
    m = nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = build_warmup_cosine_scheduler(
        opt, warmup_steps=10, max_steps=100, min_lr=1e-5, base_lr=1e-3
    )
    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # first step after warmup start should be near base after warmup
    assert lrs[0] < lrs[9]  # warm-up rising
    assert lrs[9] >= lrs[0]
    assert lrs[-1] < lrs[20]  # later decay
    assert lrs[-1] >= 1e-5 * 0.99
