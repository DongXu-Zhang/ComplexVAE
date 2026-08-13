"""Synthetic single-channel HQ-like pages for smoke/overfit without real data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from microscopy_vae.provenance.hashing import stable_sample_id


@dataclass
class SyntheticPage:
    sample_id: str
    split: str
    source: str
    category: str
    condition: str
    morphology: str
    group_id: str
    page_index: int
    image: np.ndarray  # float32 [H,W]


def _make_structure(h: int, w: int, rng: np.random.Generator, morphology: str) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.float32)
    if morphology == "puncta":
        for _ in range(rng.integers(5, 20)):
            cy, cx = rng.integers(0, h), rng.integers(0, w)
            s = rng.uniform(1.0, 3.0)
            amp = rng.uniform(0.5, 2.0)
            img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    elif morphology == "filament":
        for _ in range(rng.integers(3, 8)):
            angle = rng.uniform(0, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            d = (xx - w / 2) * c + (yy - h / 2) * s
            t = -(xx - w / 2) * s + (yy - h / 2) * c
            img += np.exp(-(d**2) / 8.0) * (np.abs(t) < w * 0.4)
    else:
        img += np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (min(h, w) / 6) ** 2))
    img += 0.02 * rng.standard_normal((h, w)).astype(np.float32)
    # allow slight negatives
    img -= 0.01
    return img.astype(np.float32)


def build_synthetic_hq_pool(
    *,
    n_groups: int = 4,
    pages_per_group: int = 2,
    size: int = 64,
    seed: int = 0,
    sources: Tuple[str, ...] = ("SOURCE_A", "SOURCE_B"),
) -> List[SyntheticPage]:
    rng = np.random.default_rng(seed)
    morphs = ["puncta", "filament", "membrane"]
    pages: List[SyntheticPage] = []
    for g in range(n_groups):
        source = sources[g % len(sources)]
        morph = morphs[g % len(morphs)]
        group_id = f"synth_group_{g:04d}"
        split = "train" if g < max(1, int(n_groups * 0.75)) else "val"
        for p in range(pages_per_group):
            img = _make_structure(size, size, rng, morph)
            sid = stable_sample_id(group_id, str(p), source)
            pages.append(
                SyntheticPage(
                    sample_id=sid,
                    split=split,
                    source=source,
                    category="synth",
                    condition="hq",
                    morphology=morph,
                    group_id=group_id,
                    page_index=p,
                    image=img,
                )
            )
    return pages


def pool_summary(pages: List[SyntheticPage]) -> Dict[str, int]:
    return {
        "n_pages": len(pages),
        "n_groups": len({p.group_id for p in pages}),
        "n_train": sum(1 for p in pages if p.split == "train"),
        "n_val": sum(1 for p in pages if p.split == "val"),
    }
