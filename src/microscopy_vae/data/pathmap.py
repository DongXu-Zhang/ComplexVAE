"""Runtime path prefix mapping: Windows inventory paths → Linux mount roots.

Policy (from data_fix STATUS):
- Do NOT rewrite the authoritative JSONL in place.
- Apply an explicit, versioned prefix map at load time.
- Reject fuzzy basename search; each mapped path must resolve uniquely under root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PathPrefixMap:
    """Map inventory path prefixes to a local filesystem root."""

    # e.g. "F:\\Dataset" or "F:/Dataset"
    source_prefixes: Tuple[str, ...]
    # e.g. "/data/microscopy/Dataset"
    target_root: str
    # if True, missing files raise; if False, return mapped path anyway (for dry planning)
    require_exists: bool = False

    def map_one(self, raw: str | Path) -> Path:
        s = str(raw)
        # normalize backslashes for matching while preserving PureWindowsPath semantics
        candidates = [s, s.replace("/", "\\"), s.replace("\\", "/")]
        matched: Optional[str] = None
        remainder: Optional[str] = None
        for pref in self.source_prefixes:
            prefs = {pref, pref.replace("/", "\\"), pref.replace("\\", "/")}
            for c in candidates:
                for p in prefs:
                    if c.lower().startswith(p.lower()):
                        matched = p
                        remainder = c[len(p) :].lstrip("\\/")
                        break
                if matched is not None:
                    break
            if matched is not None:
                break
        if matched is None:
            # already a local path?
            local = Path(s)
            if self.require_exists and not local.is_file():
                raise FileNotFoundError(
                    f"path has no matching prefix and does not exist locally: {s!r}; "
                    f"prefixes={self.source_prefixes}"
                )
            return local

        # join remainder under target_root with POSIX separators; reject path escape
        rel = remainder.replace("\\", "/") if remainder else ""
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise ValueError(f"path traversal rejected in inventory path: {s!r}")
        out = Path(self.target_root).joinpath(*parts) if parts else Path(self.target_root)
        # resolve and ensure still under target_root
        try:
            root_res = Path(self.target_root).resolve()
            out_res = out.resolve(strict=False)
            if root_res != out_res and root_res not in out_res.parents:
                raise ValueError(f"mapped path escapes target_root: {out}")
        except OSError:
            pass
        if self.require_exists and not out.is_file():
            raise FileNotFoundError(f"mapped path does not exist: {out} (from {s})")
        return out


def apply_prefix_map_to_records(
    records: Sequence,  # HQPageRecord
    path_map: PathPrefixMap,
) -> List:
    """Return new HQPageRecord list with hq_path remapped (frozen dataclass → replace)."""
    from microscopy_vae.data.records import HQPageRecord

    out: List[HQPageRecord] = []
    for r in records:
        new_path = path_map.map_one(r.hq_path)
        out.append(
            HQPageRecord(
                sample_id=r.sample_id,
                split=r.split,
                source=r.source,
                category=r.category,
                condition=r.condition,
                morphology=r.morphology,
                group_id=r.group_id,
                hq_path=new_path,
                hq_page=r.hq_page,
                hq_page_shape=r.hq_page_shape,
                hq_dtype=r.hq_dtype,
                target_role=r.target_role,
                is_derived=r.is_derived,
            )
        )
    return out


def default_windows_dataset_map(linux_root: str, *, require_exists: bool = False) -> PathPrefixMap:
    return PathPrefixMap(
        source_prefixes=("F:\\Dataset", "F:/Dataset", "f:\\Dataset", "f:/Dataset"),
        target_root=linux_root,
        require_exists=require_exists,
    )
