"""Deterministic stripe / grid diagnostics on raw float arrays."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _periodogram_1d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    return freqs, spec


def column_and_row_means(image: np.ndarray) -> Dict[str, np.ndarray]:
    arr = image.astype(np.float64)
    return {"col_mean": arr.mean(axis=0), "row_mean": arr.mean(axis=1)}


def directional_power(image: np.ndarray) -> Dict[str, float]:
    """Share of 2D power near the fx=0 axis (vertical stripes) vs fy=0 (horizontal)."""
    x = image.astype(np.float64)
    x = x - x.mean()
    f = np.fft.fftshift(np.fft.fft2(x))
    p = np.abs(f) ** 2
    cy, cx = np.array(p.shape) // 2
    # exclude DC
    p[cy, cx] = 0.0
    total = float(p.sum()) + 1e-18
    # vertical stripes → energy concentrated on horizontal frequency axis (fy≈0)
    band = max(1, p.shape[0] // 32)
    vert_energy = float(p[cy - band : cy + band + 1, :].sum())
    horz_energy = float(p[:, cx - band : cx + band + 1].sum())
    return {
        "vertical_stripe_power_frac": vert_energy / total,
        "horizontal_stripe_power_frac": horz_energy / total,
        "anisotropy": (vert_energy - horz_energy) / (vert_energy + horz_energy + 1e-18),
    }


def period_peaks(profile: np.ndarray, periods: Tuple[int, ...] = (2, 4, 8, 16, 32)) -> Dict[str, float]:
    freqs, spec = _periodogram_1d(profile)
    # skip DC
    if spec.size <= 1:
        return {f"period_{k}": 0.0 for k in periods}
    peak = float(spec[1:].max()) + 1e-18
    out: Dict[str, float] = {}
    n = profile.size
    for k in periods:
        if k <= 0 or k >= n:
            out[f"period_{k}"] = 0.0
            continue
        f0 = 1.0 / k
        j = int(np.argmin(np.abs(freqs - f0)))
        out[f"period_{k}"] = float(spec[j] / peak)
    return out


def stripe_score(image: np.ndarray) -> Dict[str, float]:
    """Scalar scores. Higher vertical_score ⇒ stronger column-periodic structure."""
    profiles = column_and_row_means(image)
    col_peaks = period_peaks(profiles["col_mean"])
    row_peaks = period_peaks(profiles["row_mean"])
    power = directional_power(image)
    vert_from_profile = float(np.std(profiles["col_mean"]) / (np.std(image) + 1e-18))
    horz_from_profile = float(np.std(profiles["row_mean"]) / (np.std(image) + 1e-18))
    out = {
        "vertical_profile_std_ratio": vert_from_profile,
        "horizontal_profile_std_ratio": horz_from_profile,
        "vertical_score": float(power["vertical_stripe_power_frac"]),
        "horizontal_score": float(power["horizontal_stripe_power_frac"]),
        "anisotropy": float(power["anisotropy"]),
    }
    for k, v in col_peaks.items():
        out[f"col_{k}"] = v
    for k, v in row_peaks.items():
        out[f"row_{k}"] = v
    return out


def compare_target_recon_stripes(target: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    t = stripe_score(target)
    r = stripe_score(recon)
    residual = recon.astype(np.float64) - target.astype(np.float64)
    e = stripe_score(residual)
    keys = [
        "vertical_score",
        "horizontal_score",
        "anisotropy",
        "col_period_8",
        "col_period_4",
        "col_period_2",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        out[f"target_{k}"] = t.get(k, float("nan"))
        out[f"recon_{k}"] = r.get(k, float("nan"))
        out[f"residual_{k}"] = e.get(k, float("nan"))
        out[f"delta_{k}"] = float(r.get(k, 0.0) - t.get(k, 0.0))
    return out


def residual_after_shift(target: np.ndarray, recon: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer roll residual; used for axis-locked vs structure-locked tests."""
    return np.roll(recon, shift=(dy, dx), axis=(0, 1)).astype(np.float64) - target.astype(np.float64)
