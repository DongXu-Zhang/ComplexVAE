"""HQ page-level manifest adapter (JSONL). Fail-closed on test / unknown split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set

from microscopy_vae.data.records import HQPageRecord
from microscopy_vae.provenance.hashing import sha256_file, stable_sample_id

ALLOWED_SPLITS = frozenset({"train", "val", "test"})
TRAIN_VAL = frozenset({"train", "val"})


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"manifest missing required field: {key}")
    return d[key]


def parse_hq_record(raw: Dict[str, Any], *, path_key: str = "path") -> HQPageRecord:
    split = str(_require(raw, "split"))
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unknown split token {split!r}; adapter rejects unknown tokens")
    shape = _require(raw, "shape")
    if not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3):
        raise ValueError(f"shape must be [H,W] or [P,H,W], got {shape}")
    if len(shape) == 2:
        page_shape = (int(shape[0]), int(shape[1]))
    else:
        page_shape = (int(shape[1]), int(shape[2]))
    page_index = int(_require(raw, "page_index"))
    group_id = str(_require(raw, "biological_group_id"))
    path = Path(str(_require(raw, path_key)))
    sample_id = str(raw.get("sample_id") or stable_sample_id(group_id, str(page_index), str(path)))
    # Human names vs tokens: target_role may be hq_identity / SIM_gt style
    target_role = str(raw.get("target_role", "hq_identity"))
    provenance = str(raw.get("target_provenance", ""))
    is_derived = "algorithmic" in provenance.lower() or bool(raw.get("is_derived", True))
    return HQPageRecord(
        sample_id=sample_id,
        split=split,  # type: ignore[arg-type]
        source=str(_require(raw, "source_dataset")),
        category=str(raw.get("category_raw", raw.get("category", "unknown"))),
        condition=str(raw.get("condition_raw", raw.get("condition", "unknown"))),
        morphology=str(raw.get("morphology", "unknown")),
        group_id=group_id,
        hq_path=path,
        hq_page=page_index,
        hq_page_shape=page_shape,
        hq_dtype=str(raw.get("dtype", "float32")),
        target_role=target_role,
        is_derived=is_derived,
    )


def load_hq_manifest(
    path: Path,
    *,
    allow_splits: Sequence[str] = ("train", "val"),
    refuse_test: bool = True,
) -> List[HQPageRecord]:
    """Load JSONL HQ page manifest. refuse_test=True is the training default."""
    path = Path(path)
    allow: Set[str] = set(allow_splits)
    if refuse_test and "test" in allow:
        raise ValueError("Training/inspect path refuses allow_splits containing test")
    records: List[HQPageRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"manifest JSON error at line {line_no}: {e}") from e
            rec = parse_hq_record(raw)
            if rec.split == "test" and refuse_test:
                # skip silently for inventory files that include test rows when refuse_test
                # but never return them
                continue
            if rec.split not in allow and not (rec.split == "test" and refuse_test):
                if rec.split not in ALLOWED_SPLITS:
                    raise ValueError(f"unknown split {rec.split}")
                continue
            if rec.split in allow:
                records.append(rec)
    if not records:
        raise ValueError(f"No records loaded from {path} for splits={sorted(allow)}")
    _assert_no_group_split_leak(records)
    return records


def _assert_no_group_split_leak(records: Sequence[HQPageRecord]) -> None:
    group_splits: Dict[str, Set[str]] = {}
    for r in records:
        group_splits.setdefault(r.group_id, set()).add(r.split)
    leaks = {g: s for g, s in group_splits.items() if len(s) > 1}
    if leaks:
        sample = list(leaks.items())[:5]
        raise ValueError(f"biological group split leak detected (examples): {sample}")


def manifest_sha256(path: Path) -> str:
    return sha256_file(Path(path))


def summarize_records(records: Sequence[HQPageRecord]) -> Dict[str, Any]:
    by_split: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    groups: Set[str] = set()
    for r in records:
        by_split[r.split] = by_split.get(r.split, 0) + 1
        by_source[r.source] = by_source.get(r.source, 0) + 1
        groups.add(r.group_id)
    return {
        "n_pages": len(records),
        "n_groups": len(groups),
        "by_split": by_split,
        "by_source": by_source,
    }
