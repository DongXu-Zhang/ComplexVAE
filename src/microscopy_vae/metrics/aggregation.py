from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

import numpy as np


def page_to_group_macro(
    page_metrics: Sequence[Mapping[str, float]],
    group_ids: Sequence[str],
    metric_keys: Sequence[str],
) -> Dict[str, float]:
    """Equal-weight average across groups after within-group page means."""
    buckets: Dict[str, List[Mapping[str, float]]] = defaultdict(list)
    for m, g in zip(page_metrics, group_ids):
        buckets[g].append(m)
    group_means = []
    for g, rows in buckets.items():
        gm = {k: float(np.mean([r[k] for r in rows])) for k in metric_keys}
        group_means.append(gm)
    if not group_means:
        return {k: float("nan") for k in metric_keys}
    return {k: float(np.mean([gm[k] for gm in group_means])) for k in metric_keys}


def group_bootstrap_ci(
    group_values: Sequence[float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    arr = np.asarray(group_values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    means = []
    n = arr.size
    for _ in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        means.append(sample.mean())
    means_a = np.sort(np.asarray(means))
    lo = float(np.quantile(means_a, alpha / 2))
    hi = float(np.quantile(means_a, 1 - alpha / 2))
    return {"mean": float(arr.mean()), "low": lo, "high": hi}
