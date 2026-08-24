#!/usr/bin/env python3
"""Plot Mod-3 loss influence from a run's metrics_train.jsonl (+ optional val).

Does not force wildly different magnitudes onto one linear axis: raw losses
are log-scale per panel; ratios are stacked on [0,1]; GAN scores are separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


G_TERMS = [
    "charbonnier",
    "ms_ssim",
    "scharr",
    "hf",
    "flux",
    "dark_fp",
    "kl",
    "perceptual",
    "adv_g",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _col(rows: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for r in rows:
        v = r.get(key)
        out.append(float(v) if isinstance(v, (int, float)) else None)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    run = args.run_dir
    out = args.out_dir or (run / "loss_influence_plots")
    out.mkdir(parents=True, exist_ok=True)
    train_path = run / "metrics_train.jsonl"
    if not train_path.is_file():
        raise SystemExit(f"missing {train_path}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("matplotlib+numpy required for plotting") from exc

    rows = _load_jsonl(train_path)
    steps = np.array([r["step"] for r in rows], dtype=float)

    def _series(prefix: str) -> Dict[str, np.ndarray]:
        data = {}
        for t in G_TERMS:
            col = _col(rows, f"{prefix}{t}")
            if any(v is not None for v in col):
                data[t] = np.array([np.nan if v is None else v for v in col], dtype=float)
        return data

    def _save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / name, dpi=120)
        plt.close(fig)

    raw = _series("loss_raw_")
    if raw:
        fig, axes = plt.subplots(len(raw), 1, figsize=(8, 2.2 * len(raw)), sharex=True)
        if len(raw) == 1:
            axes = [axes]
        for ax, (name, y) in zip(axes, raw.items()):
            ax.plot(steps, y, lw=1.0)
            ax.set_ylabel(name)
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("step")
        fig.suptitle("raw losses (symlog)")
        _save(fig, "01_raw_losses.png")

    w = _series("loss_w_")
    if w:
        fig, axes = plt.subplots(len(w), 1, figsize=(8, 2.2 * len(w)), sharex=True)
        if len(w) == 1:
            axes = [axes]
        for ax, (name, y) in zip(axes, w.items()):
            ax.plot(steps, y, lw=1.0, color="tab:orange")
            ax.set_ylabel(name)
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("step")
        fig.suptitle("weighted contributions C_i = w_i(t) L_i")
        _save(fig, "02_weighted_contrib.png")

    ratios = _series("contrib_ratio_")
    if ratios:
        fig, ax = plt.subplots(figsize=(8, 4))
        stack = []
        labels = []
        for name, y in ratios.items():
            stack.append(np.nan_to_num(y, nan=0.0))
            labels.append(name)
        ax.stackplot(steps, *stack, labels=labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("|C_i| / sum |C_j|")
        ax.set_xlabel("step")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
        ax.set_title("scalar contribution share (abs)")
        _save(fig, "03_contrib_ratio_stack.png")

    gnorm = {k[len("grad_norm_full_") :]: _col(rows, k) for k in rows[0] if k.startswith("grad_norm_full_")}
    gnorm = {k: np.array([np.nan if v is None else v for v in col]) for k, col in gnorm.items() if any(v is not None for v in col)}
    if gnorm:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, y in gnorm.items():
            ax.plot(steps, y, lw=1.0, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("||dC_i / d theta||")
        ax.legend(fontsize=8)
        ax.set_title("grad norms on full generator (log)")
        ax.grid(True, alpha=0.3)
        _save(fig, "04_grad_norms_full.png")

    grat = {k[len("grad_ratio_full_") :]: _col(rows, k) for k in rows[0] if k.startswith("grad_ratio_full_")}
    grat = {k: np.array([np.nan if v is None else v for v in col]) for k, col in grat.items() if any(v is not None for v in col)}
    if grat:
        fig, ax = plt.subplots(figsize=(8, 4))
        stack, labels = [], []
        for name, y in grat.items():
            stack.append(np.nan_to_num(y, nan=0.0))
            labels.append(name)
        if stack:
            ax.stackplot(steps, *stack, labels=labels)
            ax.set_ylim(0, 1)
            ax.legend(loc="upper left", fontsize=8, ncol=3)
            ax.set_title("gradient influence share (full G)")
            ax.set_xlabel("step")
            _save(fig, "05_grad_ratio_stack.png")

    wts = _series("weight_")
    if wts:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, y in wts.items():
            ax.plot(steps, y, lw=1.2, label=name)
        ax.set_xlabel("step")
        ax.set_ylabel("effective weight")
        ax.legend(fontsize=8)
        ax.set_title("scheduled weights (incl. beta, ramps)")
        ax.grid(True, alpha=0.3)
        _save(fig, "06_effective_weights.png")

    if any(r.get("d_real_mean") is not None for r in rows):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, _col(rows, "d_real_mean"), label="D(real)")
        ax.plot(steps, _col(rows, "d_fake_mean"), label="D(fake)")
        ax.plot(steps, _col(rows, "loss_raw_disc"), label="L_D", alpha=0.7)
        ax.legend()
        ax.set_title("GAN D/G scores (not a science metric)")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        _save(fig, "07_gan_balance.png")

    val_path = run / "metrics_val.jsonl"
    if val_path.is_file():
        vrows = _load_jsonl(val_path)
        vstep = [r["step"] for r in vrows]
        mae = [((r.get("group_macro") or {}).get("mae")) for r in vrows]
        psnr = [((r.get("group_macro") or {}).get("psnr")) for r in vrows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(vstep, mae, label="val MAE")
        ax2 = ax.twinx()
        ax2.plot(vstep, psnr, color="tab:red", label="val PSNR")
        ax.set_xlabel("step")
        ax.set_ylabel("MAE")
        ax2.set_ylabel("PSNR")
        ax.set_title("val MAE / PSNR")
        _save(fig, "08_val_mae_psnr.png")
        bg = [((r.get("group_macro") or {}).get("bg_fp_rate")) for r in vrows]
        if any(x is not None for x in bg):
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(vstep, bg)
            ax.set_title("val dark-background false-positive rate")
            ax.set_xlabel("step")
            _save(fig, "09_val_bg_fp.png")

    print(f"wrote plots under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
