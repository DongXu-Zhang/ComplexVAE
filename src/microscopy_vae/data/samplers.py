from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class HierarchicalIndexSampler(Sampler[int]):
    """source→group→page sampler.

    source_weight_mode:
      - sqrt_groups: P(source) ∝ sqrt(n_groups)  [default]
      - n_groups: P(source) ∝ n_groups
      - fixed_prior: use fixed_source_prior dict (renormalized over present sources)
    """

    def __init__(
        self,
        meta: Sequence[Dict],
        *,
        seed: int = 0,
        source_weight_mode: str = "sqrt_groups",
        fixed_source_prior: Optional[Dict[str, float]] = None,
        epoch_length: Optional[int] = None,
    ) -> None:
        if not meta:
            raise ValueError("empty meta for HierarchicalIndexSampler")
        self.seed = seed
        self.source_weight_mode = source_weight_mode
        self.fixed_source_prior = dict(fixed_source_prior or {})
        self.by_source: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for m in meta:
            self.by_source[str(m["source"])][str(m["group_id"])].append(int(m["index"]))
        self.sources = sorted(self.by_source.keys())
        self.exposure: Dict[str, int] = defaultdict(int)
        self.source_draws: Dict[str, int] = defaultdict(int)
        self._step = 0
        self.epoch_length = epoch_length if epoch_length is not None else len(meta)
        # validate prior if requested
        if source_weight_mode == "fixed_prior":
            if not self.fixed_source_prior:
                raise ValueError("source_weight_mode=fixed_prior requires fixed_source_prior")
            missing = [s for s in self.sources if s not in self.fixed_source_prior]
            if missing:
                # allow missing with 0? better fail so config is explicit
                raise ValueError(f"fixed_source_prior missing sources present in data: {missing}")

    def _source_probs(self) -> np.ndarray:
        weights = []
        for s in self.sources:
            n_g = len(self.by_source[s])
            if self.source_weight_mode == "sqrt_groups":
                weights.append(math.sqrt(max(n_g, 1)))
            elif self.source_weight_mode == "n_groups":
                weights.append(float(n_g))
            elif self.source_weight_mode == "fixed_prior":
                weights.append(float(self.fixed_source_prior.get(s, 0.0)))
            else:
                raise ValueError(f"Unknown source_weight_mode={self.source_weight_mode}")
        w = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError(f"invalid source weights: {weights}")
        return w / w.sum()

    def sample_index(self) -> int:
        rng = np.random.default_rng(self.seed + self._step)
        self._step += 1
        probs = self._source_probs()
        source = self.sources[int(rng.choice(len(self.sources), p=probs))]
        self.source_draws[source] += 1
        groups = list(self.by_source[source].keys())
        group = groups[int(rng.integers(0, len(groups)))]
        pages = self.by_source[source][group]
        idx = pages[int(rng.integers(0, len(pages)))]
        self.exposure[group] += 1
        return idx

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.epoch_length):
            yield self.sample_index()

    def __len__(self) -> int:
        return self.epoch_length

    def realized_source_freq(self) -> Dict[str, float]:
        total = sum(self.source_draws.values())
        if total == 0:
            return {s: 0.0 for s in self.sources}
        return {s: self.source_draws[s] / total for s in self.sources}

    def planned_source_probs(self) -> Dict[str, float]:
        p = self._source_probs()
        return {s: float(p[i]) for i, s in enumerate(self.sources)}
