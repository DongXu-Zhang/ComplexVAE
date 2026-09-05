from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from microscopy_vae.provenance.hashing import sha256_json
from microscopy_vae.utils.atomic import atomic_write_text

LEGACY_METHOD = "robust_linear_p0.1_p99.9"
ROBUST_METHOD = "robust_linear"
IDENTITY_METHOD = "identity"
SCHEMA_V1 = "microvae-normalizer-v1"
SCHEMA_V2 = "microvae-normalizer-v2"


def apply_raw_floor_np(
    x: np.ndarray,
    *,
    enabled: bool,
    value: float = 0.0,
) -> np.ndarray:
    """Raw-intensity floor. Not per-patch min-max. Does not clip the upper tail."""
    y = np.asarray(x, dtype=np.float32)
    if not enabled:
        return y
    return np.maximum(y, np.float32(value))


def apply_raw_floor_torch(x, *, enabled: bool, value: float = 0.0):
    import torch

    if not enabled:
        return x
    return torch.clamp(x, min=float(value))


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
    fit_mode: str = "page_uniform"
    per_source_stats: Dict[str, Any] = field(default_factory=dict)
    n_pages_fit: int = 0
    # V4 contract. Missing keys in old artifacts → these defaults (old behaviour).
    low_percentile: float = 0.1
    high_percentile: float = 99.9
    raw_floor_enabled: bool = False
    raw_floor_value: float = 0.0
    # global: one (low, high) for all sources. per_source: each source has its own.
    scale_mode: str = "global"
    per_source_scales: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Train-fitted crop/support/amp thresholds in normalized space (V5).
    per_source_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    threshold_version: str = ""

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

    def contract_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "fit_split": self.fit_split,
            "fit_mode": self.fit_mode,
            "clip": bool(self.clip),
            "low_percentile": float(self.low_percentile),
            "high_percentile": float(self.high_percentile),
            "raw_floor_enabled": bool(self.raw_floor_enabled),
            "raw_floor_value": float(self.raw_floor_value),
            "scale_mode": str(self.scale_mode),
            "sources": sorted(self.per_source_scales.keys()),
            "threshold_version": str(self.threshold_version or ""),
            "threshold_sources": sorted(self.per_source_thresholds.keys()),
            "floor_before_normalize": bool(self.raw_floor_enabled),
        }


def guess_source_from_path(path: Any, known: Optional[Sequence[str]] = None) -> Optional[str]:
    """Best-effort source from a filesystem path. Prefer longer known names first."""
    text = str(path)
    names = [str(s) for s in (known or ())]
    extra = ("DeepInsight_3D", "DeepInsight_2D", "BioTISR")
    for s in extra:
        if s not in names:
            names.append(s)
    names = sorted(set(names), key=len, reverse=True)
    for s in names:
        if s and s in text:
            return s
    return None


class Normalizer:
    def __init__(self, state: NormalizationState) -> None:
        if state.fit_split != "train":
            raise ValueError("Normalizer state must be train-fitted")
        self.state = state
        self.eps = 1e-8

    def is_per_source(self) -> bool:
        return str(self.state.scale_mode) == "per_source"

    def known_sources(self) -> List[str]:
        return sorted(self.state.per_source_scales.keys())

    def threshold_for(self, source: Optional[str], key: str, default: float) -> float:
        """Per-source calibrated threshold, else yaml/default scalar."""
        table = self.state.per_source_thresholds or {}
        if source is not None and str(source) in table:
            rec = table[str(source)] or {}
            if key in rec:
                return float(rec[key])
        return float(default)

    def scale_for(self, source: Optional[str]) -> Tuple[float, float]:
        if not self.is_per_source():
            return float(self.state.low), float(self.state.high)
        if not self.state.per_source_scales:
            raise ValueError("per_source artifact is missing per_source_scales")
        if source is None or str(source) not in self.state.per_source_scales:
            raise ValueError(
                "per-source normalizer requires a known source "
                f"(got {source!r}; known={self.known_sources()}). "
                "Pass --source or a path containing BioTISR / DeepInsight_2D / DeepInsight_3D."
            )
        sc = self.state.per_source_scales[str(source)]
        return float(sc["low"]), float(sc["high"])

    def _prepare_np(self, x: np.ndarray) -> np.ndarray:
        return apply_raw_floor_np(
            x,
            enabled=bool(self.state.raw_floor_enabled),
            value=float(self.state.raw_floor_value),
        )

    def transform(self, x: np.ndarray, source: Optional[str] = None) -> np.ndarray:
        y0 = self._prepare_np(x)
        low, high = self.scale_for(source)
        y = (y0 - low) / (high - low + self.eps)
        if self.state.clip:
            y = np.clip(y, 0.0, 1.0)
        return y.astype(np.float32)

    def inverse(self, y: np.ndarray, source: Optional[str] = None) -> np.ndarray:
        """Map normalized values back to the fitted intensity axis.

        Does not re-apply or invert the raw floor. Decoder outputs are not clamped.
        """
        low, high = self.scale_for(source)
        return (y.astype(np.float32) * (high - low + self.eps) + low).astype(np.float32)

    def transform_torch(self, x, source: Optional[str] = None):
        import torch

        x_f = apply_raw_floor_torch(
            x,
            enabled=bool(self.state.raw_floor_enabled),
            value=float(self.state.raw_floor_value),
        )
        low, high = self.scale_for(source)
        y = (x_f - low) / (high - low + self.eps)
        if self.state.clip:
            y = torch.clamp(y, 0.0, 1.0)
        return y


def _percentile_pair(flat: np.ndarray, low_p: float, high_p: float) -> Tuple[float, float]:
    low = float(np.percentile(flat, float(low_p)))
    high = float(np.percentile(flat, float(high_p)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid robust range low={low} high={high} (p{low_p}/p{high_p})")
    return low, high


def _maybe_lock_low_to_floor(low: float, high: float, *, floor_on: bool, floor_v: float) -> Tuple[float, float]:
    """After raw floor, keep y=0 at intensity 0. Do not use p0.1 as low (that remaps zeros negative)."""
    if not floor_on:
        return low, high
    locked = float(floor_v)
    if high <= locked:
        raise ValueError(f"high={high} must exceed raw floor {locked}")
    return locked, high


def _resolve_percentiles(method: str, low_percentile: float, high_percentile: float) -> Tuple[float, float]:
    if method == LEGACY_METHOD:
        return 0.1, 99.9
    if method in (ROBUST_METHOD,):
        return float(low_percentile), float(high_percentile)
    raise ValueError(f"Unknown robust method {method}")


def make_transform_id(
    *,
    method: str,
    fit_mode: str,
    scale_mode: str,
    low_p: float,
    high_p: float,
    raw_floor_enabled: bool,
    raw_floor_value: float,
    clip: bool,
    low: float,
    high: float,
    per_source_scales: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    payload = {
        "method": method,
        "mode": fit_mode,
        "scale_mode": str(scale_mode),
        "low_p": float(low_p),
        "high_p": float(high_p),
        "floor_on": bool(raw_floor_enabled),
        "floor": float(raw_floor_value) if raw_floor_enabled else None,
        "clip": bool(clip),
        "low": float(low),
        "high": float(high),
        "per_source": {k: dict(v) for k, v in sorted((per_source_scales or {}).items())},
    }
    return (
        f"robust_p{low_p:g}_p{high_p:g}_floor{int(raw_floor_enabled)}_"
        f"{scale_mode}_{fit_mode}_{sha256_json(payload)[:12]}"
    )


def fit_robust_normalizer(
    arrays: Sequence[np.ndarray],
    *,
    method: str = LEGACY_METHOD,
    clip: bool = False,
    role: str = "hq",
    n_groups: int = 0,
    config_sha256: str = "",
    manifest_sha256: str = "",
    sources: Optional[Sequence[str]] = None,
    fit_mode: str = "source_balanced",
    max_pixels_per_page: int = 65536,
    low_percentile: float = 0.1,
    high_percentile: float = 99.9,
    raw_floor_enabled: bool = False,
    raw_floor_value: float = 0.0,
    scale_mode: str = "global",
) -> NormalizationState:
    """Fit train-only robust linear map.

    fit_mode:
      - page_uniform: concatenate all sampled page pixels (old behavior; 3D-biased)
      - source_balanced: equal weight per source (recommended default for multi-source HQ)

    raw_floor_enabled: max(x, raw_floor_value) *before* percentiles and transform.
    """
    if method == IDENTITY_METHOD:
        return NormalizationState(
            schema_version=SCHEMA_V2,
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
            low_percentile=float(low_percentile),
            high_percentile=float(high_percentile),
            raw_floor_enabled=False,
            raw_floor_value=0.0,
            scale_mode="global",
            per_source_scales={},
        )
    if method not in (LEGACY_METHOD, ROBUST_METHOD):
        raise ValueError(f"Unknown method {method}")
    if not arrays:
        raise ValueError("No arrays to fit normalizer")
    if not (0.0 <= float(low_percentile) < float(high_percentile) <= 100.0):
        raise ValueError(f"Need 0 <= low_p < high_p <= 100, got {low_percentile}/{high_percentile}")

    low_p, high_p = _resolve_percentiles(method, low_percentile, high_percentile)
    floor_on = bool(raw_floor_enabled)
    floor_v = float(raw_floor_value)

    def _subsample(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pre = apply_raw_floor_np(a, enabled=floor_on, value=floor_v)
        flat = pre.astype(np.float64).ravel()
        if flat.size > max_pixels_per_page:
            idx = rng.choice(flat.size, size=max_pixels_per_page, replace=False)
            flat = flat[idx]
        return flat

    rng = np.random.default_rng(0)
    per_source_stats: Dict[str, Any] = {}
    per_source_scales: Dict[str, Dict[str, float]] = {}
    scale_mode_u = str(scale_mode)
    if scale_mode_u not in ("global", "per_source"):
        raise ValueError(f"Unknown scale_mode={scale_mode}")
    if scale_mode_u == "per_source" and (sources is None or len(sources) != len(arrays)):
        raise ValueError("scale_mode=per_source requires a source label for every fit array")

    frac_neg_raw: Dict[str, List[float]] = defaultdict(list)
    if sources is not None:
        for a, s in zip(arrays, sources):
            arr = np.asarray(a, dtype=np.float32)
            frac_neg_raw[str(s)].append(float((arr < 0).mean()))

    if scale_mode_u == "per_source":
        by_src: Dict[str, List[np.ndarray]] = defaultdict(list)
        for a, s in zip(arrays, sources or []):
            by_src[str(s)].append(_subsample(a, rng))
        lows, highs = [], []
        for s, flist in sorted(by_src.items()):
            f = np.concatenate(flist)
            try:
                lo, hi = _percentile_pair(f, low_p, high_p)
                lo, hi = _maybe_lock_low_to_floor(lo, hi, floor_on=floor_on, floor_v=floor_v)
            except ValueError as exc:
                raise ValueError(
                    f"source {s!r} has a degenerate range after floor "
                    f"(n_pages={len(flist)}, min={float(f.min())}, max={float(f.max())}): {exc}"
                ) from exc
            lows.append(lo)
            highs.append(hi)
            per_source_scales[s] = {"low": lo, "high": hi}
            y = (f - lo) / (hi - lo + 1e-8)
            n_above = int((f > hi).sum())
            neg_list = frac_neg_raw.get(s) or []
            per_source_stats[s] = {
                "low": lo,
                "high": hi,
                "n_pages": len(flist),
                "n_pixels_fit": int(f.size),
                "n_pixels_above_high": n_above,
                "frac_above_high": float(n_above / max(f.size, 1)),
                "frac_neg_before_floor": float(np.mean(neg_list)) if neg_list else 0.0,
                "mean": float(f.mean()),
                "std": float(f.std()),
                "min": float(f.min()),
                "frac_lt0": float((y < 0).mean()),
                "frac_gt1": float((y > 1).mean()),
                "norm_mean": float(y.mean()),
                "norm_std": float(y.std()),
                "norm_p50": float(np.percentile(y, 50)),
                "norm_p99": float(np.percentile(y, 99)),
            }
        low = float(np.median(lows))
        high = float(np.median(highs))
        used_mode = str(fit_mode) if fit_mode in ("source_balanced", "page_uniform") else "source_balanced"
    elif fit_mode == "page_uniform" or sources is None:
        flats = [_subsample(a, rng) for a in arrays]
        flat = np.concatenate(flats)
        low, high = _percentile_pair(flat, low_p, high_p)
        low, high = _maybe_lock_low_to_floor(low, high, floor_on=floor_on, floor_v=floor_v)
        used_mode = "page_uniform"
        n_above = int((flat > high).sum())
        per_source_stats["_union"] = {
            "n_pixels_fit": int(flat.size),
            "n_pixels_above_high": n_above,
            "frac_above_high": float(n_above / max(flat.size, 1)),
        }
    elif fit_mode == "source_balanced":
        by_src: Dict[str, List[np.ndarray]] = defaultdict(list)
        for a, s in zip(arrays, sources):
            by_src[str(s)].append(_subsample(a, rng))
        lows, highs = [], []
        for s, flist in sorted(by_src.items()):
            f = np.concatenate(flist)
            lo, hi = _percentile_pair(f, low_p, high_p)
            lo, hi = _maybe_lock_low_to_floor(lo, hi, floor_on=floor_on, floor_v=floor_v)
            lows.append(lo)
            highs.append(hi)
            n_above = int((f > hi).sum())
            per_source_stats[s] = {
                "low": lo,
                "high": hi,
                "n_pages": len(flist),
                "n_pixels_fit": int(f.size),
                "n_pixels_above_high": n_above,
                "frac_above_high": float(n_above / max(f.size, 1)),
                "mean": float(f.mean()),
                "std": float(f.std()),
                "min": float(f.min()),
                "p0": float(np.percentile(f, 0.0)),
            }
        low = float(np.median(lows))
        high = float(np.median(highs))
        if high <= low:
            flat = np.concatenate([x for xs in by_src.values() for x in xs])
            low, high = _percentile_pair(flat, low_p, high_p)
        low, high = _maybe_lock_low_to_floor(low, high, floor_on=floor_on, floor_v=floor_v)
        used_mode = "source_balanced"
        for s, st in per_source_stats.items():
            f = np.concatenate(by_src[s])
            y = (f - low) / (high - low + 1e-8)
            st["frac_lt0"] = float((y < 0).mean())
            st["frac_gt1"] = float((y > 1).mean())
            st["norm_mean"] = float(y.mean())
            st["norm_std"] = float(y.std())
            st["norm_p50"] = float(np.percentile(y, 50))
            st["norm_p99"] = float(np.percentile(y, 99))
    else:
        raise ValueError(f"Unknown fit_mode={fit_mode}")

    transform_id = make_transform_id(
        method=method,
        fit_mode=used_mode,
        scale_mode=scale_mode_u,
        low_p=low_p,
        high_p=high_p,
        raw_floor_enabled=floor_on,
        raw_floor_value=floor_v,
        clip=clip,
        low=low,
        high=high,
        per_source_scales=per_source_scales,
    )
    return NormalizationState(
        schema_version=SCHEMA_V2,
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
        low_percentile=float(low_p),
        high_percentile=float(high_p),
        raw_floor_enabled=floor_on,
        raw_floor_value=floor_v,
        scale_mode=scale_mode_u,
        per_source_scales=per_source_scales,
    )


def summarize_percentile_candidates(
    arrays: Sequence[np.ndarray],
    sources: Sequence[str],
    *,
    candidates: Sequence[float] = (99.9, 99.95, 99.99, 99.995),
    low_percentile: float = 0.1,
    raw_floor_enabled: bool = True,
    raw_floor_value: float = 0.0,
    max_pixels_per_page: int = 65536,
    seed: int = 0,
) -> Dict[str, Any]:
    """Train-only diagnostic. Does not write a normalizer or look at test."""
    if len(arrays) != len(sources):
        raise ValueError("arrays/sources length mismatch")
    rng = np.random.default_rng(int(seed))
    by_src: Dict[str, List[np.ndarray]] = defaultdict(list)
    for a, s in zip(arrays, sources):
        pre = apply_raw_floor_np(a, enabled=raw_floor_enabled, value=raw_floor_value)
        flat = pre.astype(np.float64).ravel()
        if flat.size > max_pixels_per_page:
            idx = rng.choice(flat.size, size=max_pixels_per_page, replace=False)
            flat = flat[idx]
        by_src[str(s)].append(flat)

    src_tables: Dict[str, Dict[str, Any]] = {}
    for s, flist in sorted(by_src.items()):
        f = np.concatenate(flist)
        row: Dict[str, Any] = {
            "n_pages": len(flist),
            "n_pixels": int(f.size),
            "min": float(f.min()),
            "mean": float(f.mean()),
            "p0.1": float(np.percentile(f, 0.1)),
        }
        for p in candidates:
            hi = float(np.percentile(f, float(p)))
            n_above = int((f > hi).sum())
            row[f"p{p:g}"] = hi
            row[f"p{p:g}_n_above"] = n_above
            row[f"p{p:g}_frac_above"] = float(n_above / max(f.size, 1))
        src_tables[s] = row

    globals_out: Dict[str, Any] = {}
    for p in candidates:
        highs = [src_tables[s][f"p{p:g}"] for s in src_tables]
        lows = [src_tables[s]["p0.1"] for s in src_tables]
        g_low = float(np.median(lows))
        g_high = float(np.median(highs))
        per_src_map = {}
        for s, flist in by_src.items():
            f = np.concatenate(flist)
            y = (f - g_low) / (g_high - g_low + 1e-8)
            per_src_map[s] = {
                "source_high": src_tables[s][f"p{p:g}"],
                "global_high": g_high,
                "frac_gt1": float((y > 1).mean()),
                "frac_lt0": float((y < 0).mean()),
                "norm_mean": float(y.mean()),
                "norm_p50": float(np.percentile(y, 50)),
                "norm_p99": float(np.percentile(y, 99)),
            }
        globals_out[f"p{p:g}"] = {
            "source_highs": {s: src_tables[s][f"p{p:g}"] for s in src_tables},
            "global_low_median": g_low,
            "global_high_median": g_high,
            "limiting_source_for_high": min(
                src_tables.keys(),
                key=lambda s: abs(src_tables[s][f"p{p:g}"] - g_high),
            ),
            "per_source_after_global_map": per_src_map,
        }
    per_source_linear: Dict[str, Any] = {}
    for s, flist in sorted(by_src.items()):
        f = np.concatenate(flist)
        hi = float(np.percentile(f, 99.99))
        hi = max(hi, 1e-8)
        y = f / hi
        n_gt1 = int((y > 1).sum())
        per_source_linear[s] = {
            "high_p99.99": hi,
            "norm_mean": float(y.mean()),
            "norm_p50": float(np.percentile(y, 50)),
            "norm_p90": float(np.percentile(y, 90)),
            "norm_p99": float(np.percentile(y, 99)),
            "frac_gt1_clip_false": float(n_gt1 / max(f.size, 1)),
            "frac_clipped_if_clip_true": float(n_gt1 / max(f.size, 1)),
            "frac_lt0": float((y < 0).mean()),
        }

    return {
        "raw_floor_enabled": bool(raw_floor_enabled),
        "raw_floor_value": float(raw_floor_value),
        "low_percentile": float(low_percentile),
        "candidates": [float(c) for c in candidates],
        "per_source": src_tables,
        "per_source_linear_p99_99": per_source_linear,
        "source_balanced_globals": globals_out,
        "note": (
            "With three sources, source-balanced global high is the median of "
            "per-source highs. Raising the percentile only moves the global high "
            "if the middle source's high moves; the brightest source can still "
            "exceed 1 after mapping. V4/V5 transform uses per-source high, not "
            "the global median. clip=false keeps frac_gt1 as linear tail; "
            "clip=true would flatten those pixels to 1 and is not the default."
        ),
    }


def assert_artifact_matches_config(state: NormalizationState, cfg_norm: Any, *, allow_legacy: bool) -> None:
    """Refuse silent mix of a V4 config with a V2 artifact (or the reverse)."""
    want_floor = bool(getattr(cfg_norm, "raw_floor_enabled", False))
    want_low = float(getattr(cfg_norm, "low_percentile", 0.1))
    want_high = float(getattr(cfg_norm, "high_percentile", 99.9))
    method = str(getattr(cfg_norm, "method", LEGACY_METHOD))
    if method == LEGACY_METHOD:
        want_low, want_high = 0.1, 99.9
    got_floor = bool(state.raw_floor_enabled)
    got_low = float(state.low_percentile)
    got_high = float(state.high_percentile)
    want_scale = str(getattr(cfg_norm, "scale_mode", "global"))
    got_scale = str(getattr(state, "scale_mode", "global"))
    want_clip = bool(getattr(cfg_norm, "clip", False))
    got_clip = bool(getattr(state, "clip", False))
    mismatch = (
        got_floor != want_floor
        or abs(got_low - want_low) > 1e-9
        or abs(got_high - want_high) > 1e-9
        or want_scale != got_scale
        or want_clip != got_clip
    )
    cfg_id = method == IDENTITY_METHOD
    art_id = state.method == IDENTITY_METHOD
    if cfg_id != art_id:
        mismatch = True
    if want_scale == "per_source" and not state.per_source_scales:
        mismatch = True
    want_thr = bool(getattr(cfg_norm, "calibrate_thresholds", False))
    got_thr = bool(getattr(state, "per_source_thresholds", None))
    if want_thr and not (got_thr and str(getattr(state, "threshold_version", ""))):
        mismatch = True
    if mismatch and not allow_legacy:
        raise ValueError(
            "Normalizer artifact does not match config contract "
            f"(cfg floor={want_floor} p{want_low:g}/{want_high:g} method={method} "
            f"scale_mode={want_scale} clip={want_clip} calibrate_thresholds={want_thr}; "
            f"artifact floor={got_floor} p{got_low:g}/{got_high:g} method={state.method} "
            f"scale_mode={got_scale} clip={got_clip} "
            f"threshold_version={getattr(state, 'threshold_version', '')!r} "
            f"transform_id={state.transform_id}). "
            "Refusing silent mix. Pass the artifact from the same run, or set "
            "normalization.allow_legacy_artifact=true."
        )
