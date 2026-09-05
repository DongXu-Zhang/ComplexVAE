from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from microscopy_vae.data.normalization import Normalizer
from microscopy_vae.data.readers import read_page
from microscopy_vae.data.records import HQBatch, HQPageRecord
from microscopy_vae.data.synthetic import SyntheticPage
from microscopy_vae.data.threshold_calibration import crop_range_accept


def take_crop(
    img: np.ndarray,
    idx: int,
    *,
    crop_size: int,
    fixed: bool,
    seed: int,
    mode: str,
    jitter_frac: float,
    cell_hits: Dict[int, np.ndarray],
    draw_counter: Callable[[], int],
    last_cell: Optional[list] = None,
) -> np.ndarray:
    h, w = img.shape
    cs = crop_size
    if h < cs or w < cs:
        raise ValueError(f"image {h}x{w} smaller than crop {cs} for idx={idx}")
    if fixed:
        if last_cell is not None:
            last_cell.clear()
        y0 = (h - cs) // 2
        x0 = (w - cs) // 2
        return img[y0 : y0 + cs, x0 : x0 + cs]
    if mode != "coverage_jitter":
        if last_cell is not None:
            last_cell.clear()
        rng = np.random.default_rng(seed + idx * 9973)
        y0 = int(rng.integers(0, h - cs + 1))
        x0 = int(rng.integers(0, w - cs + 1))
        return img[y0 : y0 + cs, x0 : x0 + cs]

    ny = max(h // cs, 1)
    nx = max(w // cs, 1)
    hits = cell_hits.get(idx)
    if hits is None or hits.shape != (ny, nx):
        hits = np.zeros((ny, nx), dtype=np.int32)
        cell_hits[idx] = hits
    n_draw = draw_counter()
    rng = np.random.default_rng(seed + idx * 9973 + n_draw * 17)
    min_h = int(hits.min())
    cands = np.argwhere(hits == min_h)
    pick = cands[int(rng.integers(0, len(cands)))]
    cy, cx = int(pick[0]), int(pick[1])
    hits[cy, cx] += 1
    if last_cell is not None:
        last_cell[:] = [idx, cy, cx]
    base_y = int(round(cy * (h - cs) / max(ny - 1, 1))) if ny > 1 else 0
    base_x = int(round(cx * (w - cs) / max(nx - 1, 1))) if nx > 1 else 0
    jitter = int(round(cs * max(jitter_frac, 0.0)))
    y_lo = max(0, base_y - jitter)
    y_hi = min(h - cs, base_y + jitter)
    x_lo = max(0, base_x - jitter)
    x_hi = min(w - cs, base_x + jitter)
    y0 = int(rng.integers(y_lo, y_hi + 1))
    x0 = int(rng.integers(x_lo, x_hi + 1))
    return img[y0 : y0 + cs, x0 : x0 + cs]


def _normalized_robust_range(norm: np.ndarray) -> float:
    lo, hi = np.quantile(norm.astype(np.float64), [0.005, 0.995])
    return float(hi - lo)


def _undo_coverage_hit(
    cell_hits: Optional[Dict[int, np.ndarray]],
    last_cell: Optional[list],
) -> None:
    """Rejected empty retries must not mark a coarse cell as already covered."""
    if not cell_hits or not last_cell or len(last_cell) != 3:
        return
    idx, cy, cx = int(last_cell[0]), int(last_cell[1]), int(last_cell[2])
    hits = cell_hits.get(idx)
    if hits is None:
        return
    if 0 <= cy < hits.shape[0] and 0 <= cx < hits.shape[1]:
        hits[cy, cx] = max(int(hits[cy, cx]) - 1, 0)
    last_cell.clear()


def _select_crop(
    image: np.ndarray,
    idx: int,
    *,
    crop_fn: Callable[[np.ndarray, int], np.ndarray],
    normalizer: Normalizer,
    min_robust_range: float,
    max_retries: int,
    allow_retry: bool,
    source: str | None = None,
    empty_keep_prob: float = 0.0,
    cell_hits: Optional[Dict[int, np.ndarray]] = None,
    last_cell: Optional[list] = None,
) -> tuple:
    crop = crop_fn(image, idx)
    norm = normalizer.transform(crop, source=source)
    thr = float(min_robust_range)
    if hasattr(normalizer, "threshold_for"):
        thr = float(normalizer.threshold_for(source, "crop_min_robust_range", thr))
    if (not allow_retry) or thr <= 0:
        return crop, norm
    keep_p = float(empty_keep_prob)
    rng = np.random.default_rng(int(idx) * 7919 + 13)
    tries = max(int(max_retries), 1)
    for attempt in range(tries):
        if crop_range_accept(_normalized_robust_range(norm), thr):
            return crop, norm
        # Dark/empty crop: keep some so the decoder sees black at train time.
        if keep_p > 0 and float(rng.random()) < keep_p:
            return crop, norm
        if attempt == tries - 1:
            return crop, norm
        _undo_coverage_hit(cell_hits, last_cell)
        crop = crop_fn(image, idx)
        norm = normalizer.transform(crop, source=source)
    return crop, norm


class SyntheticHQDataset(Dataset):
    """In-memory synthetic HQ pages with random or fixed crops."""

    def __init__(
        self,
        pages: List[SyntheticPage],
        *,
        split: str,
        crop_size: int,
        normalizer: Normalizer,
        fixed_crops: bool = False,
        seed: int = 0,
        crop_mode: str = "random",
        coverage_jitter_frac: float = 0.25,
        min_robust_range: float = 0.0,
        max_range_retries: int = 8,
        empty_keep_prob: float = 0.0,
    ) -> None:
        if split == "test":
            raise RuntimeError("Refuse to construct test dataset without freeze credentials")
        self.pages = [p for p in pages if p.split == split]
        if not self.pages:
            raise ValueError(f"No synthetic pages for split={split}")
        self.crop_size = crop_size
        self.normalizer = normalizer
        self.fixed_crops = fixed_crops
        self.seed = seed
        self.crop_mode = crop_mode
        self.coverage_jitter_frac = float(coverage_jitter_frac)
        self.min_robust_range = float(min_robust_range)
        self.max_range_retries = int(max_range_retries)
        self.empty_keep_prob = float(empty_keep_prob)
        self._cell_hits: Dict[int, np.ndarray] = {}
        self._coverage_draws = 0
        self._last_cell: list = []
        # public metadata for hierarchical sampler
        self.meta = [
            {
                "source": p.source,
                "group_id": p.group_id,
                "sample_id": p.sample_id,
                "index": i,
            }
            for i, p in enumerate(self.pages)
        ]

    def __len__(self) -> int:
        return len(self.pages)

    def _crop(self, img: np.ndarray, idx: int) -> np.ndarray:
        return take_crop(
            img,
            idx,
            crop_size=self.crop_size,
            fixed=self.fixed_crops,
            seed=self.seed,
            mode=self.crop_mode,
            jitter_frac=self.coverage_jitter_frac,
            cell_hits=self._cell_hits,
            draw_counter=lambda: self._bump_coverage(),
            last_cell=self._last_cell,
        )

    def _bump_coverage(self) -> int:
        self._coverage_draws += 1
        return self._coverage_draws

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.pages[idx]
        crop, norm = _select_crop(
            p.image,
            idx,
            crop_fn=self._crop,
            normalizer=self.normalizer,
            min_robust_range=self.min_robust_range,
            max_retries=self.max_range_retries,
            allow_retry=(not self.fixed_crops),
            source=p.source,
            empty_keep_prob=self.empty_keep_prob,
            cell_hits=self._cell_hits,
            last_cell=self._last_cell,
        )
        tensor = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0)
        return {
            "hq": tensor,
            "sample_id": p.sample_id,
            "group_id": p.group_id,
            "source": p.source,
            "metadata": {
                "morphology": p.morphology,
                "page_index": p.page_index,
                "page_shape": list(p.image.shape),
                "split": p.split,
                "transform_id": self.normalizer.state.transform_id,
            },
        }


class ManifestHQDataset(Dataset):
    """Lazy-read HQ pages from manifest records (train/val only by construction)."""

    def __init__(
        self,
        records: Sequence[HQPageRecord],
        *,
        split: str,
        crop_size: int,
        normalizer: Normalizer,
        fixed_crops: bool = False,
        seed: int = 0,
        crop_mode: str = "random",
        coverage_jitter_frac: float = 0.25,
        min_robust_range: float = 0.0,
        max_range_retries: int = 8,
        empty_keep_prob: float = 0.0,
    ) -> None:
        if split == "test":
            raise RuntimeError("Refuse to construct test dataset without freeze credentials")
        self.records = [r for r in records if r.split == split]
        if not self.records:
            raise ValueError(f"No HQ records for split={split}")
        # double-check no test slipped in
        if any(r.split == "test" for r in self.records):
            raise RuntimeError("test records present in ManifestHQDataset")
        self.crop_size = crop_size
        self.normalizer = normalizer
        self.fixed_crops = fixed_crops
        self.seed = seed
        self.crop_mode = crop_mode
        self.coverage_jitter_frac = float(coverage_jitter_frac)
        self.min_robust_range = float(min_robust_range)
        self.max_range_retries = int(max_range_retries)
        self.empty_keep_prob = float(empty_keep_prob)
        self._cell_hits: Dict[int, np.ndarray] = {}
        self._coverage_draws = 0
        self._last_cell: list = []
        self.meta = [
            {
                "source": r.source,
                "group_id": r.group_id,
                "sample_id": r.sample_id,
                "index": i,
            }
            for i, r in enumerate(self.records)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _crop(self, img: np.ndarray, idx: int) -> np.ndarray:
        return take_crop(
            img,
            idx,
            crop_size=self.crop_size,
            fixed=self.fixed_crops,
            seed=self.seed,
            mode=self.crop_mode,
            jitter_frac=self.coverage_jitter_frac,
            cell_hits=self._cell_hits,
            draw_counter=lambda: self._bump_coverage(),
            last_cell=self._last_cell,
        )

    def _bump_coverage(self) -> int:
        self._coverage_draws += 1
        return self._coverage_draws

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        page, rmeta = read_page(r.hq_path, r.hq_page, expected_dtype=r.hq_dtype)
        if tuple(page.shape) != r.hq_page_shape:
            # allow if manifest listed file shape incorrectly on H/W only mismatch log
            if page.shape[0] != r.hq_page_shape[0] or page.shape[1] != r.hq_page_shape[1]:
                raise ValueError(
                    f"page shape mismatch for {r.sample_id}: got {page.shape}, "
                    f"manifest {r.hq_page_shape}"
                )
        crop, norm = _select_crop(
            page,
            idx,
            crop_fn=self._crop,
            normalizer=self.normalizer,
            min_robust_range=self.min_robust_range,
            max_retries=self.max_range_retries,
            allow_retry=(not self.fixed_crops),
            source=r.source,
            empty_keep_prob=self.empty_keep_prob,
            cell_hits=self._cell_hits,
            last_cell=self._last_cell,
        )
        tensor = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0)
        return {
            "hq": tensor,
            "sample_id": r.sample_id,
            "group_id": r.group_id,
            "source": r.source,
            "metadata": {
                "morphology": r.morphology,
                "page_index": r.hq_page,
                "page_shape": list(page.shape),
                "split": r.split,
                "target_role": r.target_role,
                "is_derived": r.is_derived,
                "transform_id": self.normalizer.state.transform_id,
                "reader": rmeta,
            },
        }


def collate_hq(batch: List[Dict[str, Any]]) -> HQBatch:
    hq = torch.stack([b["hq"] for b in batch], dim=0)
    return HQBatch(
        hq=hq,
        sample_ids=[b["sample_id"] for b in batch],
        group_ids=[b["group_id"] for b in batch],
        sources=[b["source"] for b in batch],
        metadata=[b["metadata"] for b in batch],
    )
