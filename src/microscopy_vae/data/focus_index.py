"""Volume-internal focus scores. Split must already be assigned at volume level."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from microscopy_vae.data.readers import read_page
from microscopy_vae.data.records import HQPageRecord
from microscopy_vae.metrics.focus import score_volume_slices
from microscopy_vae.utils.atomic import atomic_write_text

SCHEMA = "microvae-focus-v1"


def load_focus_sidecar(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("schema") not in (SCHEMA, None):
                raise ValueError(f"Unknown focus schema {rec.get('schema')}")
            rows.append(rec)
    return rows


def scores_for_dataset_indices(
    records: Sequence[HQPageRecord],
    sidecar_rows: Sequence[dict],
) -> Dict[int, float]:
    """Map dataset index (position in `records` after split filter) → focus_score."""
    by_key: Dict[Tuple[str, int], float] = {}
    for r in sidecar_rows:
        by_key[(str(r["volume_id"]), int(r["slice_index"]))] = float(r["focus_score"])
    out: Dict[int, float] = {}
    for i, rec in enumerate(records):
        key = (rec.group_id, int(rec.hq_page))
        if key in by_key:
            out[i] = by_key[key]
    return out


def build_focus_sidecar(
    records: Sequence[HQPageRecord],
    out_path: Path,
    *,
    refuse_test: bool = True,
    logger=None,
) -> Path:
    if refuse_test and any(r.split == "test" for r in records):
        raise RuntimeError("refuse to score test records for focus sidecar")
    groups: Dict[str, List[HQPageRecord]] = defaultdict(list)
    for r in records:
        if r.split == "test":
            continue
        groups[r.group_id].append(r)

    lines: List[str] = []
    n_vol = 0
    for gid, recs in sorted(groups.items()):
        recs = sorted(recs, key=lambda r: (int(r.hq_page), r.sample_id))
        if len(recs) == 1:
            scored = [{"focus_score": 0.0, "tenengrad": 0.0, "robust_contrast": 0.0, "hf_energy": 0.0}]
        else:
            imgs = []
            for r in recs:
                page, _ = read_page(r.hq_path, r.hq_page, expected_dtype=r.hq_dtype)
                imgs.append(page)
            scored = score_volume_slices(imgs)
        ranked = sorted(range(len(recs)), key=lambda i: float(scored[i]["focus_score"]), reverse=True)
        rank_of = {i: rk + 1 for rk, i in enumerate(ranked)}
        for i, r in enumerate(recs):
            row = {
                "schema": SCHEMA,
                "volume_id": r.group_id,
                "slice_index": int(r.hq_page),
                "sample_id": r.sample_id,
                "source": r.source,
                "split": r.split,
                "focus_score": float(scored[i]["focus_score"]),
                "tenengrad": float(scored[i].get("tenengrad", 0.0)),
                "robust_contrast": float(scored[i].get("robust_contrast", 0.0)),
                "hf_energy": float(scored[i].get("hf_energy", 0.0)),
                "rank_in_volume": int(rank_of[i]),
                "n_slices_in_volume": len(recs),
            }
            lines.append(json.dumps(row, sort_keys=True))
        n_vol += 1
        if logger is not None and n_vol % 100 == 0:
            logger.info("focus sidecar volumes scored: %s/%s", n_vol, len(groups))
    atomic_write_text(out_path, "\n".join(lines) + ("\n" if lines else ""))
    return out_path


def resolve_slice_scores(
    records: Sequence[HQPageRecord],
    *,
    sidecar_path: Optional[Path],
    compute_if_missing: bool,
    cache_path: Optional[Path],
    logger=None,
) -> Dict[int, float]:
    path = sidecar_path
    if path is not None and path.is_file():
        rows = load_focus_sidecar(path)
        return scores_for_dataset_indices(records, rows)
    if not compute_if_missing:
        return {}
    if cache_path is None:
        raise ValueError("compute_if_missing requires cache_path")
    if logger:
        logger.info("Computing focus sidecar → %s (train/val only)", cache_path)
    build_focus_sidecar(records, cache_path, refuse_test=True, logger=logger)
    rows = load_focus_sidecar(cache_path)
    return scores_for_dataset_indices(records, rows)
