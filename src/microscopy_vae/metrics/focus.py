"""Deterministic per-slice focus / structure scores. Volume-internal ranking only."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# L1-normalized Scharr, same orientation as training loss.
_SCHARR_X = np.array([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], dtype=np.float64)
_SCHARR_Y = np.array([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], dtype=np.float64)
_SCHARR_X = _SCHARR_X / np.abs(_SCHARR_X).sum()
_SCHARR_Y = _SCHARR_Y / np.abs(_SCHARR_Y).sum()


def _scharr_energy(image: np.ndarray) -> float:
    t = torch.from_numpy(image.astype(np.float32))[None, None]
    t = F.pad(t, (1, 1, 1, 1), mode="reflect")
    kx = torch.from_numpy(_SCHARR_X.astype(np.float32)).view(1, 1, 3, 3)
    ky = torch.from_numpy(_SCHARR_Y.astype(np.float32)).view(1, 1, 3, 3)
    gx = F.conv2d(t, kx)
    gy = F.conv2d(t, ky)
    return float((gx.square() + gy.square()).mean().item())


def _preblur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float64)
    t = torch.from_numpy(image.astype(np.float32))[None, None]
    radius = max(int(round(3 * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    t = F.pad(t, (radius, radius, radius, radius), mode="reflect")
    t = F.conv2d(t, g.view(1, 1, 1, -1))
    t = F.conv2d(t, g.view(1, 1, -1, 1))
    return t.numpy()[0, 0].astype(np.float64)


def slice_structure_signals(image: np.ndarray) -> Dict[str, float]:
    """Raw signals on one slice. Not comparable across sources until volume-zscored."""
    x = image.astype(np.float64)
    xb = _preblur(x, 1.0)
    tenengrad = _scharr_energy(xb)
    contrast = float(np.quantile(x, 0.99) - np.quantile(x, 0.01))
    hf = float(np.mean((x - xb) ** 2))
    # 64-bin entropy after robust scale
    lo, hi = np.quantile(x, [0.01, 0.99])
    if hi <= lo:
        entropy = 0.0
    else:
        y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        hist, _ = np.histogram(y, bins=64, range=(0.0, 1.0), density=False)
        p = hist.astype(np.float64)
        p = p / (p.sum() + 1e-18)
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "tenengrad": tenengrad,
        "robust_contrast": contrast,
        "hf_energy": hf,
        "entropy": entropy,
    }


def _zscore(vals: Sequence[float]) -> np.ndarray:
    a = np.asarray(vals, dtype=np.float64)
    mu = float(a.mean())
    sd = float(a.std())
    if sd < 1e-12:
        return np.zeros_like(a)
    return (a - mu) / sd


def score_volume_slices(
    slices: Sequence[np.ndarray],
    *,
    weights: Dict[str, float] | None = None,
) -> List[Dict[str, float]]:
    """Volume-internal scores. Split must already be assigned at volume level.

    score = 0.50 z(tenengrad) + 0.30 z(hf_energy) + 0.20 z(robust_contrast)
    Entropy is recorded but not in the default score (noise can inflate it).
    """
    w = weights or {"tenengrad": 0.50, "hf_energy": 0.30, "robust_contrast": 0.20}
    raw = [slice_structure_signals(s) for s in slices]
    z_keys = ["tenengrad", "hf_energy", "robust_contrast"]
    zmaps = {k: _zscore([r[k] for r in raw]) for k in z_keys}
    out: List[Dict[str, float]] = []
    n = len(slices)
    for i, r in enumerate(raw):
        score = 0.0
        row = dict(r)
        for k in z_keys:
            row[f"z_{k}"] = float(zmaps[k][i])
            score += float(w[k]) * float(zmaps[k][i])
        row["focus_score"] = float(score)
        row["slice_index"] = float(i)
        row["n_slices_in_volume"] = float(n)
        out.append(row)
    return out


def select_case_slice_index(
    scores: Sequence[Mapping[str, float]],
    *,
    central_fraction: float = 0.5,
) -> int:
    """Highest score among the central `central_fraction` of the volume.

    Avoids first/last slices of a z-stack without cherry-picking globally.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("empty score list")
    if n == 1:
        return 0
    margin = int(np.floor(n * (1.0 - central_fraction) / 2.0))
    lo, hi = margin, n - margin
    if hi <= lo:
        lo, hi = 0, n
    best_i = lo
    best_s = float("-inf")
    for i in range(lo, hi):
        s = float(scores[i]["focus_score"])
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


