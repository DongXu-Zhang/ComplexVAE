from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from microscopy_vae.provenance.hashing import sha256_json
from microscopy_vae.utils.atomic import atomic_write_text


@dataclass
class NormalizationState:
    schema_version: str
    method: str
    fit_split: str
    low: float
    high: float
    clip: bool
    role: str
    n_groups: int
    config_sha256: str
    manifest_sha256: str
    transform_id: str
    # optional diagnostics / balanced-fit metadata
    fit_mode: str = "page_uniform"
    per_source_stats: Dict[str, Any] = field(default_factory=dict)
    n_pages_fit: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NormalizationState":
        from dataclasses import fields

        known = {f.name for f in fields(NormalizationState)}
        filtered = {k: v for k, v in d.items() if k in known}
        return NormalizationState(**filtered)

    def save(self, path: Path) -> str:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        atomic_write_text(path, text)
        return sha256_json(self.to_dict())

    @staticmethod
    def load(path: Path) -> "NormalizationState":
        return NormalizationState.from_dict(json.loads(path.read_text(encoding="utf-8")))


class Normalizer:
    def __init__(self, state: NormalizationState) -> None:
        if state.fit_split != "train":
            raise ValueError("Normalizer state must be train-fitted")
        self.state = state
        self.eps = 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = (x.astype(np.float32) - self.state.low) / (self.state.high - self.state.low + self.eps)
        if self.state.clip:
            y = np.clip(y, 0.0, 1.0)
        return y.astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (y.astype(np.float32) * (self.state.high - self.state.low + self.eps) + self.state.low).astype(
            np.float32
        )

    def transform_torch(self, x):
        import torch

        low = self.state.low
        high = self.state.high
        y = (x - low) / (high - low + self.eps)
        if self.state.clip:
            y = torch.clamp(y, 0.0, 1.0)
        return y


def _percentile_pair(flat: np.ndarray) -> Tuple[float, float]:
    low = float(np.percentile(flat, 0.1))
    high = float(np.percentile(flat, 99.9))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid robust range low={low} high={high}")
    return low, high


def fit_robust_normalizer(
    arrays: Sequence[np.ndarray],
    *,
    method: str = "robust_linear_p0.1_p99.9",
    clip: bool = False,
    role: str = "hq",
    n_groups: int = 0,
    config_sha256: str = "",
    manifest_sha256: str = "",
    sources: Optional[Sequence[str]] = None,
    fit_mode: str = "source_balanced",
    max_pixels_per_page: int = 65536,
) -> NormalizationState:
    """Fit train-only robust linear map.

    fit_mode:
      - page_uniform: concatenate all sampled page pixels (old behavior; 3D-biased)
      - source_balanced: equal weight per source (recommended default for multi-source HQ)
    """
    if method == "identity":
        return NormalizationState(
            schema_version="microvae-normalizer-v1",
            method=method,
            fit_split="train",
            low=0.0,
            high=1.0,
            clip=False,
            role=role,
            n_groups=n_groups,
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
            transform_id="identity_v1",
            fit_mode="identity",
            n_pages_fit=len(arrays),
        )
    if method != "robust_linear_p0.1_p99.9":
        raise ValueError(f"Unknown method {method}")
    if not arrays:
        raise ValueError("No arrays to fit normalizer")

    def _subsample(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        flat = a.astype(np.float64).ravel()
        if flat.size > max_pixels_per_page:
            idx = rng.choice(flat.size, size=max_pixels_per_page, replace=False)
            flat = flat[idx]
        return flat

    rng = np.random.default_rng(0)
    per_source_stats: Dict[str, Any] = {}

    if fit_mode == "page_uniform" or sources is None:
        flats = [_subsample(a, rng) for a in arrays]
        flat = np.concatenate(flats)
        low, high = _percentile_pair(flat)
        used_mode = "page_uniform"
    elif fit_mode == "source_balanced":
        by_src: Dict[str, List[np.ndarray]] = defaultdict(list)
        for a, s in zip(arrays, sources):
            by_src[str(s)].append(_subsample(a, rng))
        # equal source weight: take percentiles per source then median of lows/highs
        lows, highs = [], []
        for s, flist in sorted(by_src.items()):
            f = np.concatenate(flist)
            lo, hi = _percentile_pair(f)
            lows.append(lo)
            highs.append(hi)
            per_source_stats[s] = {
                "low": lo,
                "high": hi,
                "n_pages": len(flist),
                "mean": float(f.mean()),
                "std": float(f.std()),
                "frac_lt_global_will_fill": None,
            }
        low = float(np.median(lows))
        high = float(np.median(highs))
        if high <= low:
            # fallback to union percentiles
            flat = np.concatenate([x for xs in by_src.values() for x in xs])
            low, high = _percentile_pair(flat)
        used_mode = "source_balanced"
        for s, st in per_source_stats.items():
            # after global low/high chosen, report fraction outside [0,1] if mapped
            f = np.concatenate(by_src[s])
            y = (f - low) / (high - low + 1e-8)
            st["frac_lt0"] = float((y < 0).mean())
            st["frac_gt1"] = float((y > 1).mean())
            st["norm_mean"] = float(y.mean())
            st["norm_std"] = float(y.std())
    else:
        raise ValueError(f"Unknown fit_mode={fit_mode}")

    transform_id = (
        f"robust_p0.1_p99.9_hq_{used_mode}_v2_{sha256_json({'low': low, 'high': high, 'mode': used_mode})[:12]}"
    )
    return NormalizationState(
        schema_version="microvae-normalizer-v1",
        method=method,
        fit_split="train",
        low=low,
        high=high,
        clip=clip,
        role=role,
        n_groups=n_groups,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        transform_id=transform_id,
        fit_mode=used_mode,
        per_source_stats=per_source_stats,
        n_pages_fit=len(arrays),
    )
