"""Train-only per-source crop / support / amp thresholds in normalized space.

Fits after the V4 linear map. Does not look at val/test. Does not min-max per crop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from microscopy_vae.losses.pixel import per_sample_robust_range, structure_support_mask
from microscopy_vae.losses.structure import scharr_magnitude

# v2: noise floor from low-gradient pixels, not low-intensity pixels.
# v1 used intensity-dark pixels as "background"; a dense dim filament mesh
# still has gradient there, so BioTISR/DeepInsight_2D floors snapped to the
# yaml cap 0.02 and dropped real filaments from support.
THRESHOLD_VERSION = "microvae-thresholds-v2"
THRESHOLD_VERSION_V1 = "microvae-thresholds-v1"


def _page_crops(page: np.ndarray, crop_size: int, n: int, rng: np.random.Generator) -> List[np.ndarray]:
    img = np.asarray(page, dtype=np.float32)
    h, w = img.shape[-2], img.shape[-1]
    cs = int(crop_size)
    if h < cs or w < cs:
        return [img]
    out = []
    for _ in range(max(int(n), 1)):
        y0 = int(rng.integers(0, h - cs + 1))
        x0 = int(rng.integers(0, w - cs + 1))
        out.append(img[y0 : y0 + cs, x0 : x0 + cs])
    return out


def _as_bchw(crop: np.ndarray) -> torch.Tensor:
    x = np.asarray(crop, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[None]
    return torch.from_numpy(np.ascontiguousarray(x))


@torch.no_grad()
def _crop_stats(
    crop: np.ndarray,
    *,
    kernel: int,
    rel: float,
    floor: float,
    min_density: float,
    bg_quantile: float = 0.20,
) -> Dict[str, float]:
    t = _as_bchw(crop)
    rng = float(per_sample_robust_range(t)[0])
    mag = scharr_magnitude(t)
    support = structure_support_mask(t, kernel=kernel, floor=floor, rel=rel, min_density=min_density)
    frac = float(support.mean())
    mag_flat = mag.reshape(-1)
    scharr_mean = float(mag_flat.mean())
    q = min(max(float(bg_quantile), 0.01), 0.49)
    # Quiet = lowest-gradient pixels (camera noise / empty), not darkest intensity.
    # Intensity-dark filaments still have Scharr energy and must not set the floor.
    quiet_tau = float(torch.quantile(mag_flat, q))
    quiet = mag_flat[mag_flat <= quiet_tau]
    if quiet.numel() > 8:
        quiet_p90 = float(torch.quantile(quiet, 0.90))
        quiet_mean = float(quiet.mean())
    elif quiet.numel() > 0:
        quiet_p90 = float(quiet.max())
        quiet_mean = float(quiet.mean())
    else:
        quiet_p90 = 0.0
        quiet_mean = 0.0
    flat = t.reshape(-1)
    bg_q = float(torch.quantile(flat, q))
    bg = mag_flat[flat <= bg_q]
    bg_mag = float(bg.mean()) if bg.numel() else 0.0
    bg_p90 = float(torch.quantile(bg, 0.90)) if bg.numel() > 8 else bg_mag
    return {
        "robust_range": rng,
        "support_frac": frac,
        "scharr_mean": scharr_mean,
        "quiet_scharr_mean": quiet_mean,
        "quiet_scharr_p90": quiet_p90,
        "bg_scharr_mean": bg_mag,
        "bg_scharr_p90": bg_p90,
        "mean": float(t.mean()),
    }


def fit_structure_thresholds(
    norm_images: Sequence[np.ndarray],
    sources: Sequence[str],
    *,
    crop_size: int = 256,
    kernel: int = 9,
    rel: float = 0.25,
    min_density: float = 0.15,
    structure_min_frac: float = 0.0003,
    fallback_floor: float = 0.02,
    fallback_range: float = 0.08,
    bg_quantile: float = 0.20,  # kept for contract; bg uses p20 inside _crop_stats
    bg_scharr_q: float = 90.0,
    empty_range_q: float = 90.0,
    struct_range_q: float = 10.0,
    crops_per_page: int = 4,
    seed: int = 0,
    fallback_amp_range: Optional[float] = None,
    floor_min: float = 5e-4,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Return (per_source_thresholds, diagnostics). Train images must already be normalized.

    ``structure_support_floor`` is a per-source noise floor for
    ``max(rel * mean(Scharr), floor)``. yaml ``fallback_floor`` is an upper
    cap and a no-crop fallback, not a magnet that snaps every source to 0.02.
    """
    if len(norm_images) != len(sources):
        raise ValueError("norm_images/sources length mismatch")
    rng = np.random.default_rng(int(seed))
    by_src: Dict[str, List[np.ndarray]] = {}
    for img, s in zip(norm_images, sources):
        by_src.setdefault(str(s), []).append(np.asarray(img, dtype=np.float32))

    thresholds: Dict[str, Dict[str, float]] = {}
    diag: Dict[str, Any] = {}
    for src, pages in sorted(by_src.items()):
        stats: List[Dict[str, float]] = []
        probe_floor = min(1e-4, float(fallback_floor))
        for page in pages:
            for crop in _page_crops(page, crop_size, crops_per_page, rng):
                stats.append(
                    _crop_stats(
                        crop,
                        kernel=int(kernel),
                        rel=float(rel),
                        floor=probe_floor,
                        min_density=float(min_density),
                        bg_quantile=float(bg_quantile),
                    )
                )
        if not stats:
            amp_lock = (
                float(fallback_amp_range)
                if fallback_amp_range is not None
                else float(fallback_range)
            )
            thresholds[src] = {
                "structure_support_floor": float(fallback_floor),
                "amp_low_structure_range": amp_lock if amp_lock > 0 else float(fallback_range),
                "crop_min_robust_range": float(fallback_range),
            }
            diag[src] = {"n_crops": 0, "note": "no crops; yaml fallback"}
            continue
        quiet_p90s = np.array([s["quiet_scharr_p90"] for s in stats], dtype=np.float64)
        bg_p90s = np.array([s["bg_scharr_p90"] for s in stats], dtype=np.float64)
        floor_s = float(np.percentile(quiet_p90s, float(bg_scharr_q)))
        struct_mags = np.array(
            [s["scharr_mean"] for s in stats if s["support_frac"] >= float(structure_min_frac)],
            dtype=np.float64,
        )
        # Keep the additive floor below typical structure so dim meshes
        # (Lifeact / TOMM20) are not gated by max(rel*mean, 0.02).
        if struct_mags.size >= 4:
            cap_by_struct = float(np.percentile(struct_mags, 10.0)) * 0.5
            if cap_by_struct > 0.0:
                floor_s = min(floor_s, cap_by_struct)
        # yaml 0.02 is a ceiling, not the fitted value. Lower bound rejects
        # 1e-5 camera noise without locking BioTISR to 0.02.
        floor_lo = min(float(floor_min), float(fallback_floor))
        floor_s = float(np.clip(floor_s, floor_lo, float(fallback_floor)))

        empty_rr = np.array(
            [s["robust_range"] for s in stats if s["support_frac"] < float(structure_min_frac)],
            dtype=np.float64,
        )
        struct_rr = np.array(
            [s["robust_range"] for s in stats if s["support_frac"] >= float(structure_min_frac)],
            dtype=np.float64,
        )
        if struct_rr.size >= 8 and empty_rr.size >= 8:
            hi_empty = float(np.percentile(empty_rr, float(empty_range_q)))
            lo_struct = float(np.percentile(struct_rr, float(struct_range_q)))
            thr = 0.5 * (hi_empty + lo_struct)
            thr = min(thr, float(np.percentile(struct_rr, 15)))
            thr = max(thr, float(np.percentile(empty_rr, 50)))
        elif struct_rr.size >= 8:
            # Almost no empty crops in the fit sample: only skip the darkest tail.
            thr = float(np.percentile(struct_rr, 5)) * 0.5
        elif empty_rr.size >= 8:
            thr = float(np.percentile(empty_rr, 90))
        else:
            thr = float(fallback_range) * 0.35
        thr = float(np.clip(thr, 1e-4, float(fallback_range)))
        # Crop gate may go below 0.08 so dim filaments are kept.
        # Amp gate is locked to the yaml amp value (not the crop gate).
        amp_lock = (
            float(fallback_amp_range)
            if fallback_amp_range is not None
            else float(fallback_range)
        )
        amp_thr = amp_lock if amp_lock > 0 else thr

        thresholds[src] = {
            "structure_support_floor": floor_s,
            "amp_low_structure_range": amp_thr,
            "crop_min_robust_range": thr,
        }
        diag[src] = {
            "n_crops": int(len(stats)),
            "n_empty": int(empty_rr.size),
            "n_struct": int(struct_rr.size),
            "support_frac_mean": float(np.mean([s["support_frac"] for s in stats])),
            "robust_range_p10": float(np.percentile([s["robust_range"] for s in stats], 10)),
            "robust_range_p50": float(np.percentile([s["robust_range"] for s in stats], 50)),
            "quiet_scharr_p90_mean": float(quiet_p90s.mean()),
            "bg_scharr_p90_mean": float(bg_p90s.mean()),
            "scharr_mean_p10": float(np.percentile([s["scharr_mean"] for s in stats], 10)),
            "floor_min": float(floor_lo),
            "floor_rule": "low_gradient_quiet_pixels",
            "fallback_floor": float(fallback_floor),
            "fallback_range": float(fallback_range),
        }
    return thresholds, diag


def crop_range_accept(robust_range: float, threshold: float) -> bool:
    """Soft band: accept [0.5*thr, inf). Hard-retry only clearly empty crops."""
    if float(threshold) <= 0:
        return True
    return float(robust_range) >= 0.5 * float(threshold)
