"""Post-hoc val report: worst-tail vs predeclared severe failure. Not a training loss."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from microscopy_vae.data.normalization import NormalizationState, Normalizer, apply_raw_floor_np
from microscopy_vae.metrics.extended import (
    background_false_positive_stats,
    bright_edge_stats,
    dark_structure_stats,
    highlight_overshoot_frac,
    mae_np,
    mse_np,
    nmse,
    psnr_from_mse,
    ssim_mean,
)
from microscopy_vae.utils.atomic import atomic_write_text


def to_raw_nonneg(
    norm_arr: np.ndarray,
    normalizer: Optional[Normalizer],
    *,
    is_raw: bool,
    source: Optional[str] = None,
) -> np.ndarray:
    """Map a reconstructed/target array to nonnegative raw intensity.

    If `is_raw`, only apply the artifact floor (or max(0) if no artifact floor).
    If normalized, inverse then floor. Decoder is not clamped during invert.
    """
    x = np.asarray(norm_arr, dtype=np.float32)
    if is_raw:
        if normalizer is not None and normalizer.state.raw_floor_enabled:
            return apply_raw_floor_np(
                x, enabled=True, value=float(normalizer.state.raw_floor_value)
            )
        return apply_raw_floor_np(x, enabled=True, value=0.0)
    if normalizer is None:
        return apply_raw_floor_np(x, enabled=True, value=0.0)
    inv = normalizer.inverse(x, source=source)
    if normalizer.state.raw_floor_enabled:
        return apply_raw_floor_np(inv, enabled=True, value=float(normalizer.state.raw_floor_value))
    return apply_raw_floor_np(inv, enabled=True, value=0.0)


def default_unit_scale(state: NormalizationState) -> float:
    """Common intensity divisor after inverse + floor.

    Per-source V4: use the brightest source's (high-low), otherwise DI3D
    reconstructions sit far above 1 and pred_gt1_frac looks like failure.
    Global V2.2: high-low of the single fitted line.
    """
    if str(getattr(state, "scale_mode", "global")) == "per_source" and state.per_source_scales:
        spans = [float(sc["high"]) - float(sc["low"]) for sc in state.per_source_scales.values()]
        return max(max(spans), 1e-8)
    return max(float(state.high) - float(state.low), 1e-8)


def to_common_unit(raw_nonneg: np.ndarray, scale: float) -> np.ndarray:
    s = max(float(scale), 1e-8)
    return (np.asarray(raw_nonneg, dtype=np.float64) / s).astype(np.float32)


def pair_eval_metrics(
    pred_unit: np.ndarray,
    tgt_unit: np.ndarray,
    *,
    data_range: float = 1.0,
) -> Dict[str, float]:
    mse = mse_np(pred_unit, tgt_unit)
    out: Dict[str, float] = {
        "mae": mae_np(pred_unit, tgt_unit),
        "mse": mse,
        "psnr": psnr_from_mse(mse, data_range),
        "nmse": nmse(pred_unit, tgt_unit),
        "ssim": ssim_mean(pred_unit, tgt_unit, data_range=data_range),
        "signed_bias": float(pred_unit.astype(np.float64).mean() - tgt_unit.astype(np.float64).mean()),
        "pred_gt1_frac": highlight_overshoot_frac(pred_unit, hi=1.0),
    }
    out.update(background_false_positive_stats(pred_unit, tgt_unit))
    out.update(bright_edge_stats(pred_unit, tgt_unit))
    out.update(dark_structure_stats(pred_unit, tgt_unit))
    return out


def classify_severe(m: Dict[str, float], *, cfg: Any) -> Tuple[bool, List[str]]:
    """Predeclared photometric severe rule. Structure metrics are recorded, not OR-ed in."""
    reasons: List[str] = []
    mae = float(m.get("mae", float("nan")))
    bg_fp = float(m.get("bg_fp_rate", float("nan")))
    bg_bias = float(m.get("bg_bias", float("nan")))
    if np.isfinite(mae) and mae >= float(cfg.severe_mae_unit):
        reasons.append("mae_unit")
    if (
        np.isfinite(bg_fp)
        and np.isfinite(bg_bias)
        and bg_fp >= float(cfg.severe_bg_fp_rate)
        and bg_bias >= float(cfg.severe_bg_bias)
    ):
        reasons.append("overbright_bg")
    return bool(reasons), reasons


def structure_flags(m: Dict[str, float], *, cfg: Any) -> List[str]:
    flags: List[str] = []
    br = float(m.get("bright_retention", float("nan")))
    dg = float(m.get("dark_grad_retention", float("nan")))
    if np.isfinite(br) and br < float(cfg.severe_bright_retention):
        flags.append("low_bright_retention")
    if np.isfinite(dg) and dg < float(cfg.severe_dark_grad_retention):
        flags.append("low_dark_grad_retention")
    return flags


@torch.no_grad()
def collect_val_pages(
    system,
    loader,
    *,
    device: torch.device,
    normalizer: Optional[Normalizer],
    unit_scale: float,
) -> List[Dict[str, Any]]:
    """One row per val crop. Input tensors are already normalized by the dataset.

    Caller must load EMA (or raw) weights into `system` before calling.
    """
    system.eval()
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        x = batch.hq.to(device, non_blocking=True)
        recon = system.reconstruct_hq(x)
        for i in range(x.shape[0]):
            xi = x[i, 0].detach().float().cpu().numpy()
            ri = recon[i, 0].detach().float().cpu().numpy()
            meta = batch.metadata[i] if batch.metadata else {}
            src = batch.sources[i]
            tgt_raw = to_raw_nonneg(xi, normalizer, is_raw=False, source=src)
            pred_raw = to_raw_nonneg(ri, normalizer, is_raw=False, source=src)
            tgt_u = to_common_unit(tgt_raw, unit_scale)
            pred_u = to_common_unit(pred_raw, unit_scale)
            m = pair_eval_metrics(pred_u, tgt_u, data_range=1.0)
            m_raw = {
                "mae_raw": mae_np(pred_raw, tgt_raw),
                "signed_bias_raw": float(pred_raw.mean() - tgt_raw.mean()),
            }
            rows.append(
                {
                    "sample_id": batch.sample_ids[i],
                    "group_id": batch.group_ids[i],
                    "source": batch.sources[i],
                    "morphology": str(meta.get("morphology", "unknown")),
                    "page_index": meta.get("page_index"),
                    "transform_id": meta.get("transform_id"),
                    "metrics_unit": m,
                    "metrics_raw": m_raw,
                }
            )
    return rows


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    cfg: Any,
) -> Dict[str, Any]:
    n = len(rows)
    severe_idx: List[int] = []
    for i, r in enumerate(rows):
        is_s, reasons = classify_severe(r["metrics_unit"], cfg=cfg)
        r["severe"] = is_s
        r["severe_reasons"] = reasons
        r["structure_flags"] = structure_flags(r["metrics_unit"], cfg=cfg)
        if is_s:
            severe_idx.append(i)

    def _count(pred) -> Dict[str, Any]:
        sub = [r for r in rows if pred(r)]
        n_s = sum(1 for r in sub if r["severe"])
        vols = {r["group_id"] for r in sub}
        vols_s = {r["group_id"] for r in sub if r["severe"]}
        return {
            "n_slices": len(sub),
            "n_severe_slices": n_s,
            "frac_severe_slices": float(n_s / max(len(sub), 1)),
            "n_volumes": len(vols),
            "n_volumes_with_severe": len(vols_s),
            "frac_volumes_with_severe": float(len(vols_s) / max(len(vols), 1)),
        }

    by_source = {}
    for s in sorted({r["source"] for r in rows}):
        by_source[s] = _count(lambda r, s=s: r["source"] == s)
    by_morph = {}
    for mname in sorted({r["morphology"] for r in rows}):
        by_morph[mname] = _count(lambda r, mname=mname: r["morphology"] == mname)

    order = sorted(range(n), key=lambda i: -float(rows[i]["metrics_unit"].get("mae", 0.0)))
    worst_n = int(getattr(cfg, "worst_n", 20))
    worst = []
    for i in order[:worst_n]:
        r = rows[i]
        worst.append(
            {
                "sample_id": r["sample_id"],
                "group_id": r["group_id"],
                "source": r["source"],
                "morphology": r["morphology"],
                "mae_unit": r["metrics_unit"]["mae"],
                "psnr": r["metrics_unit"]["psnr"],
                "ssim": r["metrics_unit"]["ssim"],
                "signed_bias": r["metrics_unit"]["signed_bias"],
                "bg_fp_rate": r["metrics_unit"].get("bg_fp_rate"),
                "bright_retention": r["metrics_unit"].get("bright_retention"),
                "dark_grad_retention": r["metrics_unit"].get("dark_grad_retention"),
                "severe": r["severe"],
                "severe_reasons": r["severe_reasons"],
                "structure_flags": r["structure_flags"],
            }
        )
    return {
        "n_val_slices": n,
        "overall": _count(lambda r: True),
        "by_source": by_source,
        "by_morphology": by_morph,
        "worst_tail": worst,
        "severe_rule": {
            "mae_unit>=": float(cfg.severe_mae_unit),
            "or_overbright_bg": {
                "bg_fp_rate>=": float(cfg.severe_bg_fp_rate),
                "and_bg_bias>=": float(cfg.severe_bg_bias),
            },
            "structure_flags_are_auxiliary": True,
            "not_a_forced_bottom_percent": True,
        },
        "disclaimer": (
            "Severe is a predeclared photometric rule in a common unit map "
            "(raw-nonneg / scale). It is not 'bottom 1%'. Structure flags are "
            "explanatory only. Calibrate thresholds against a V2.2 val dump "
            "before treating the rate as a scientific failure proportion."
        ),
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def dump_page_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [json.dumps(r, default=str) for r in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def save_case_arrays(
    out_dir: Path,
    stem: str,
    *,
    target: np.ndarray,
    pred: np.ndarray,
    display_lo: float,
    display_hi: float,
) -> None:
    """Save arrays for a case. Display bounds are Target-derived and shared."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.asarray(target, dtype=np.float32)
    p = np.asarray(pred, dtype=np.float32)
    np.save(out_dir / f"{stem}_target.npy", t)
    np.save(out_dir / f"{stem}_recon.npy", p)
    np.save(out_dir / f"{stem}_abs_err.npy", np.abs(p - t))
    np.save(out_dir / f"{stem}_signed_residual.npy", p - t)
    meta = {
        "display_lo": float(display_lo),
        "display_hi": float(display_hi),
        "display_note": "Use the same Target-derived range for target and recon. Do not per-image stretch.",
    }
    atomic_write_text(out_dir / f"{stem}_display.json", json.dumps(meta, indent=2) + "\n")


def compare_row_pairs(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    *,
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    """Match on sample_id. Metrics already in a common unit map."""
    by_a = {r["sample_id"]: r for r in rows_a}
    by_b = {r["sample_id"]: r for r in rows_b}
    common = sorted(set(by_a) & set(by_b))
    deltas = []
    for sid in common:
        a = by_a[sid]["metrics_unit"]
        b = by_b[sid]["metrics_unit"]
        deltas.append(
            {
                "sample_id": sid,
                "source": by_a[sid]["source"],
                "morphology": by_a[sid]["morphology"],
                "d_mae": float(b["mae"] - a["mae"]),
                "d_psnr": float(b["psnr"] - a["psnr"]),
                "d_ssim": float(b["ssim"] - a["ssim"]),
                "d_signed_bias": float(b["signed_bias"] - a["signed_bias"]),
                "d_bg_fp_rate": float(b.get("bg_fp_rate", np.nan) - a.get("bg_fp_rate", np.nan)),
                "d_bright_retention": float(
                    b.get("bright_retention", np.nan) - a.get("bright_retention", np.nan)
                ),
                "d_dark_grad_retention": float(
                    b.get("dark_grad_retention", np.nan) - a.get("dark_grad_retention", np.nan)
                ),
                f"severe_{label_a}": by_a[sid].get("severe"),
                f"severe_{label_b}": by_b[sid].get("severe"),
            }
        )
    def _mean(key: str) -> float:
        vals = [d[key] for d in deltas if np.isfinite(d.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_common_slices": len(common),
        "n_only_a": len(set(by_a) - set(by_b)),
        "n_only_b": len(set(by_b) - set(by_a)),
        "mean_delta_b_minus_a": {
            "mae": _mean("d_mae"),
            "psnr": _mean("d_psnr"),
            "ssim": _mean("d_ssim"),
            "signed_bias": _mean("d_signed_bias"),
            "bg_fp_rate": _mean("d_bg_fp_rate"),
            "bright_retention": _mean("d_bright_retention"),
            "dark_grad_retention": _mean("d_dark_grad_retention"),
        },
        "per_slice": deltas,
        "domain": "common_unit = raw_nonneg / scale; each model used its own inverse",
    }
