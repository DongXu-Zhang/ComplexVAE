#!/usr/bin/env python3
"""Fixed-batch loss-scale probe. Does not update weights.

Prints raw L_i, |C_i|, G_i = ||dC_i / d theta_full||, and a *suggested*
weight that would make G_i match the median of the currently active
pixel/structure terms. This is a starting point, not an auto-balancer.

Does not run optimizer.step. Does not clamp inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from microscopy_vae.config.loader import load_config
from microscopy_vae.losses.influence import diagnose_generator_influence, scalar_contrib_ratios
from microscopy_vae.systems.factory import build_hq_codec_system


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=10_000, help="optimizer_step used for schedules")
    args = p.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    system = build_hq_codec_system(cfg)
    system.eval()
    x = torch.randn(args.batch_size, 1, args.size, args.size)
    # Signed, unbounded, not [0,1] — matches linear-head codec.
    x = x * 0.7 - 0.1
    out = system.vae(x, sample_posterior=True)
    loss_out = system.task.loss(out, x, optimizer_step=args.step)
    print(f"step={args.step}  total={float(loss_out.total):.6g}")
    print(f"{'term':<14} {'raw':>12} {'w_eff':>10} {'C_i':>12} {'share':>8}")
    shares = scalar_contrib_ratios(loss_out.weighted)
    for k in loss_out.unweighted:
        raw = float(loss_out.unweighted[k].detach())
        w = float(loss_out.weights[k])
        c = float(loss_out.weighted[k].detach())
        print(f"{k:<14} {raw:12.4g} {w:10.4g} {c:12.4g} {shares[k]:8.3f}")
    infl = diagnose_generator_influence(
        loss_out.weighted, system.vae, param_group_names=("full", "encoder", "decoder", "output")
    )
    print("\ngrad norms (full generator):")
    pixel_g = []
    for k in loss_out.unweighted:
        g = infl.get(f"grad_norm_full_{k}")
        r = infl.get(f"grad_ratio_full_{k}")
        print(f"  {k:<14} G={g:.4g}  R={r:.3f}" if g is not None else f"  {k:<14} (no graph)")
        if k in ("charbonnier", "ms_ssim", "scharr", "hf") and g:
            pixel_g.append(g)
    if pixel_g:
        import statistics

        med = statistics.median(pixel_g)
        print(f"\nmedian G of char/ms/scharr/hf = {med:.4g}")
        print("suggested weight so G_i ≈ that median (clip to [0.1x, 10x] of current):")
        for k in loss_out.unweighted:
            g = infl.get(f"grad_norm_full_{k}")
            w = float(loss_out.weights[k])
            if not g or g == 0 or w == 0:
                continue
            # G_i scales linearly with w_i
            w_new = w * (med / g)
            lo, hi = 0.1 * w, 10.0 * w
            w_clip = min(max(w_new, lo), hi)
            print(f"  {k:<14} current={w:.4g}  unclipped={w_new:.4g}  clipped={w_clip:.4g}")
    print("\nThis is not an automatic training update. Edit yaml by hand after a real batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
