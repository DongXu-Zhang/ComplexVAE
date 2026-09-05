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
        slice_weight_mode: str = "uniform",
        slice_scores: Optional[Dict[int, float]] = None,
        focus_temperature: float = 0.7,
        focus_min_keep: float = 0.15,
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
        self.slice_weight_mode = slice_weight_mode
        self.slice_scores = dict(slice_scores or {})
        self.focus_temperature = float(focus_temperature)
        self.focus_min_keep = float(focus_min_keep)
        self.focus_draws = 0
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
        idx = pages[self._choose_slice(rng, pages)]
        self.exposure[group] += 1
        return idx

    def _choose_slice(self, rng: np.random.Generator, pages: List[int]) -> int:
        if self.slice_weight_mode != "focus_softmax" or len(pages) <= 1:
            return int(rng.integers(0, len(pages)))
        scores = np.array([float(self.slice_scores.get(i, 0.0)) for i in pages], dtype=np.float64)
        if not np.isfinite(scores).all() or float(scores.max() - scores.min()) < 1e-12:
            return int(rng.integers(0, len(pages)))
        temp = max(self.focus_temperature, 1e-6)
        z = (scores - scores.max()) / temp
        sm = np.exp(z)
        sm = sm / sm.sum()
        keep = min(max(self.focus_min_keep, 0.0), 0.95)
        p = (1.0 - keep) * sm + keep / len(pages)
        p = p / p.sum()
        self.focus_draws += 1
        return int(rng.choice(len(pages), p=p))

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

    def state_dict(self) -> Dict[str, object]:
        return {
            "_step": int(self._step),
            "source_draws": dict(self.source_draws),
            "exposure": dict(self.exposure),
            "focus_draws": int(self.focus_draws),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self._step = int(state.get("_step", 0))
        self.source_draws = defaultdict(int, {str(k): int(v) for k, v in dict(state.get("source_draws") or {}).items()})
        self.exposure = defaultdict(int, {str(k): int(v) for k, v in dict(state.get("exposure") or {}).items()})
        self.focus_draws = int(state.get("focus_draws", 0))

    def set_epoch(self, epoch: int) -> None:
        """Kept for DDP DataLoader conventions. The global stream is _step, not rewound."""
        del epoch


class DistributedHierarchicalSampler(Sampler[int]):
    """Shard a *shared* hierarchical stream: every rank sees the same global
    source/group sequence, then keeps only its stride. Merged source proportions
    match the single-process sampler. Each rank gets different indices.
    """

    def __init__(
        self,
        meta: Sequence[Dict],
        *,
        rank: int,
        world_size: int,
        seed: int = 0,
        source_weight_mode: str = "sqrt_groups",
        fixed_source_prior: Optional[Dict[str, float]] = None,
        epoch_length: Optional[int] = None,
        slice_weight_mode: str = "uniform",
        slice_scores: Optional[Dict[int, float]] = None,
        focus_temperature: float = 0.7,
        focus_min_keep: float = 0.15,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank={rank} not in [0, {world_size})")
        self.rank = int(rank)
        self.world_size = int(world_size)
        global_len = epoch_length if epoch_length is not None else len(meta)
        # Per-rank length: enough for drop_last loaders; global stream is world_size times longer.
        self.epoch_length = int(global_len)
        self.inner = HierarchicalIndexSampler(
            meta,
            seed=seed,
            source_weight_mode=source_weight_mode,
            fixed_source_prior=fixed_source_prior,
            epoch_length=max(self.epoch_length * self.world_size, 1),
            slice_weight_mode=slice_weight_mode,
            slice_scores=slice_scores,
            focus_temperature=focus_temperature,
            focus_min_keep=focus_min_keep,
        )

    def sample_index(self) -> int:
        chosen = 0
        for r in range(self.world_size):
            idx = self.inner.sample_index()
            if r == self.rank:
                chosen = idx
        return chosen

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.epoch_length):
            yield self.sample_index()

    def __len__(self) -> int:
        return self.epoch_length

    def realized_source_freq(self) -> Dict[str, float]:
        return self.inner.realized_source_freq()

    def planned_source_probs(self) -> Dict[str, float]:
        return self.inner.planned_source_probs()

    def state_dict(self) -> Dict[str, object]:
        d = self.inner.state_dict()
        d["rank"] = self.rank
        d["world_size"] = self.world_size
        return d

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.inner.load_state_dict(state)

    def set_epoch(self, epoch: int) -> None:
        self.inner.set_epoch(epoch)
