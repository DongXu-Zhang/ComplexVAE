from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from microscopy_vae.data.normalization import Normalizer
from microscopy_vae.data.readers import read_page
from microscopy_vae.data.records import HQBatch, HQPageRecord
from microscopy_vae.data.synthetic import SyntheticPage


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
        h, w = img.shape
        cs = self.crop_size
        if h < cs or w < cs:
            raise ValueError(f"image {h}x{w} smaller than crop {cs}")
        if self.fixed_crops:
            y0 = (h - cs) // 2
            x0 = (w - cs) // 2
        else:
            rng = np.random.default_rng(self.seed + idx * 9973)
            y0 = int(rng.integers(0, h - cs + 1))
            x0 = int(rng.integers(0, w - cs + 1))
        return img[y0 : y0 + cs, x0 : x0 + cs]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.pages[idx]
        crop = self._crop(p.image, idx)
        norm = self.normalizer.transform(crop)
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
        h, w = img.shape
        cs = self.crop_size
        if h < cs or w < cs:
            raise ValueError(f"image {h}x{w} smaller than crop {cs} for sample idx={idx}")
        if self.fixed_crops:
            y0 = (h - cs) // 2
            x0 = (w - cs) // 2
        else:
            rng = np.random.default_rng(self.seed + idx * 9973)
            y0 = int(rng.integers(0, h - cs + 1))
            x0 = int(rng.integers(0, w - cs + 1))
        return img[y0 : y0 + cs, x0 : x0 + cs]

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
        crop = self._crop(page, idx)
        norm = self.normalizer.transform(crop)
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
