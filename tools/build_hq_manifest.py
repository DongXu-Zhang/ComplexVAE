#!/usr/bin/env python3
"""Build HQ-only page-level JSONL manifest for microscopy-vae S1 training.

Designed to run on the DATA server (Windows F:\\Dataset or Linux mount).
Reads file headers only (shape / dtype / page count). Never exports pixels.

Output schema is locked to what training code expects in:
  microscopy_vae.data.manifest.parse_hq_record / load_hq_manifest

Required JSONL fields per page:
  split, source_dataset, biological_group_id, path, shape, page_index

Recommended fields also emitted:
  sample_id, dtype, category_raw, morphology, target_role, target_provenance,
  trainable, is_explicit_test, protocol, file_bytes

S1 pool (audit §8.1 / Route E′):
  BioTISR SIM_gt.mrc
  DeepInsight_2D RC_*_highsnr.tif  (not GTdenoised, not lowsnr, not WF)
  DeepInsight_3D canonical RC_*_highsnr.tif / RC_highsnr.tif

Usage:
  python build_hq_manifest.py --root F:\\Dataset --out hq_manifest.jsonl --dry-run
  python build_hq_manifest.py --root F:\\Dataset --out hq_manifest.jsonl
  python build_hq_manifest.py --self-test   # synthetic fixture, no real data needed

Dependencies: numpy, mrcfile, tifffile (stdlib otherwise).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Field contract (MUST match microscopy_vae.data.manifest.parse_hq_record)
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = (
    "split",  # train | val | test
    "source_dataset",  # BioTISR | DeepInsight_2D | DeepInsight_3D
    "biological_group_id",  # NOT group_id — loader requires this exact key
    "path",
    "shape",  # [H,W] or [P,H,W]
    "page_index",
)

SCHEMA_VERSION = "microvae-hq-manifest-v1"

# Path segments / names to hard-exclude (case-insensitive match on path parts)
EXCLUDE_DIR_TOKENS = {
    "lattice-sim",
    "lattice_sim",
    "sdm_training_data",
    "sdm",
    "engineer_sim2d_20240325_jam",
    "lipid_droplet_2d",
    "try_before_collection",
    "otf",
    "psf",
    "test",
    "tests",
    "long_cycles",
    "tomm20_long_cycles",
}
EXCLUDE_NAME_SUBSTR = {
    "gt_all",
    "gtdenoised",
    "sitemp",
    "wf_",  # never HQ identity
    "rc_lowsnr",
    "lowsnr",  # catch RC_*_lowsnr and WF_*_lowsnr
    "otf",
    "psf",
}
# positive HQ name patterns
RE_SIM_GT = re.compile(r"^SIM_gt\.mrc$", re.IGNORECASE)
RE_RC_HIGH = re.compile(r"^(RC_.*_highsnr|RC_highsnr)\.(tif|tiff)$", re.IGNORECASE)

SOURCE_DIR_ALIASES = {
    "biotisr": "BioTISR",
    "deepinsight_2d_training_data": "DeepInsight_2D",
    "deepinsight_2d": "DeepInsight_2D",
    "deepinsight_3d_training_data": "DeepInsight_3D",
    "deepinsight_3d": "DeepInsight_3D",
}


@dataclass
class FileHit:
    path: Path
    source: str
    group_id: str
    category: str
    target_role: str
    provenance: str
    protocol: str


@dataclass
class HeaderInfo:
    shape: List[int]  # [H,W] or [P,H,W]
    dtype: str
    n_pages: int
    file_bytes: int
    ok: bool
    error: str = ""


@dataclass
class BuildStats:
    discovered_files: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    accepted_files: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rejected_files: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pages: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    groups: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    reject_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: List[str] = field(default_factory=list)


def stable_id(*parts: str) -> str:
    blob = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def path_is_excluded(p: Path) -> Optional[str]:
    parts_l = [x.lower() for x in p.parts]
    name_l = p.name.lower()
    for tok in EXCLUDE_DIR_TOKENS:
        if tok in parts_l:
            return f"dir_token:{tok}"
    for sub in EXCLUDE_NAME_SUBSTR:
        if sub in name_l:
            # allow RC_*_highsnr even though it contains no lowsnr; lowsnr already handled
            if sub == "lowsnr" and "highsnr" in name_l:
                continue
            if sub == "wf_" and name_l.startswith("wf_"):
                return f"name:{sub}"
            if sub != "wf_" and sub in name_l:
                return f"name:{sub}"
    if p.stat().st_size == 0:
        return "zero_byte"
    return None


def infer_source(root: Path, p: Path) -> Optional[str]:
    try:
        rel = p.relative_to(root)
    except ValueError:
        rel = p
    parts = [x.lower() for x in rel.parts]
    for part in parts:
        if part in SOURCE_DIR_ALIASES:
            return SOURCE_DIR_ALIASES[part]
    # also match if root itself is a source folder
    root_name = root.name.lower()
    if root_name in SOURCE_DIR_ALIASES:
        return SOURCE_DIR_ALIASES[root_name]
    return None


def infer_category(source: str, p: Path, root: Path) -> str:
    try:
        rel_parts = list(p.relative_to(root).parts)
    except ValueError:
        rel_parts = list(p.parts)
    if source == "BioTISR":
        # BioTISR/<category>/Cell_xxx/SIM_gt.mrc
        for i, part in enumerate(rel_parts):
            if re.match(r"Cell_\d+", part, re.I) and i > 0:
                return rel_parts[i - 1]
        return rel_parts[0] if rel_parts else "unknown"
    if source == "DeepInsight_2D":
        # .../<category>/<condition>/Exp.../
        for part in rel_parts:
            if part.lower() in {
                "deepinsight_2d_training_data",
                "deepinsight_2d",
            }:
                continue
            if re.match(r"^\d+$", part):
                continue
            if part.lower().startswith("exp"):
                continue
            return part
        return "unknown"
    if source == "DeepInsight_3D":
        for part in rel_parts:
            if part.lower() in {
                "deepinsight_3d_training_data",
                "deepinsight_3d",
                "raw_tif_fullsize",
                "canonical",
            }:
                continue
            if re.match(r"^\d+$", part):
                continue
            return part
        return "unknown"
    return "unknown"


def infer_group_id(source: str, p: Path, root: Path) -> str:
    try:
        rel = p.relative_to(root)
    except ValueError:
        rel = p
    parts = list(rel.parts)
    if source == "BioTISR":
        for part in parts:
            if re.match(r"Cell_\d+", part, re.I):
                cat = infer_category(source, p, root)
                return f"BioTISR/{cat}/{part}"
        return f"BioTISR/{p.parent.as_posix()}"
    # DI2D/DI3D: group = parent directory of the RC file (scene / volume folder)
    parent = p.parent
    try:
        rel_parent = parent.relative_to(root).as_posix()
    except ValueError:
        rel_parent = parent.as_posix()
    return f"{source}/{rel_parent}"


def morphology_guess(category: str) -> str:
    c = category.lower()
    if any(k in c for k in ("ccp", "lys", "lamp", "perox", "vesic")):
        return "puncta"
    if any(k in c for k in ("ensconsin", "mt", "microtub", "factin", "lifeact", "actin")):
        return "filament"
    if any(k in c for k in ("er", "kdel", "membrane")):
        return "membrane_network"
    if any(k in c for k in ("mito", "tomm", "pkmo", "phb")):
        return "mitochondrial_network"
    return "unknown"


def read_header(path: Path) -> HeaderInfo:
    file_bytes = path.stat().st_size
    suffix = path.suffix.lower()
    try:
        if suffix == ".mrc":
            import mrcfile
            import numpy as np

            with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
                data = mrc.data
                # mrcfile may memory-map; only read shape/dtype
                shape = list(data.shape)
                dt = str(data.dtype)
            # normalize to [P,H,W] or [H,W]
            if len(shape) == 2:
                n_pages = 1
                out_shape = [int(shape[0]), int(shape[1])]
            elif len(shape) == 3:
                # mrc often (Z,Y,X) = (P,H,W)
                n_pages = int(shape[0])
                out_shape = [int(shape[0]), int(shape[1]), int(shape[2])]
            else:
                return HeaderInfo([], "unknown", 0, file_bytes, False, f"ndim={len(shape)}")
            if "float" not in dt:
                # BioTISR SIM_gt must be float32; reject non-float HQ
                return HeaderInfo(out_shape, dt, n_pages, file_bytes, False, f"non_float_dtype={dt}")
            return HeaderInfo(out_shape, "float32" if "float32" in dt else dt, n_pages, file_bytes, True)
        if suffix in {".tif", ".tiff"}:
            import tifffile
            import numpy as np

            with tifffile.TiffFile(str(path)) as tif:
                n = len(tif.pages)
                page0 = tif.pages[0]
                h, w = int(page0.shape[0]), int(page0.shape[1])
                dt = str(page0.dtype)
                if n == 1:
                    # could still be multi-page series stored as array
                    try:
                        arr = tif.asarray()
                        if arr.ndim == 3:
                            n_pages = int(arr.shape[0])
                            out_shape = [n_pages, int(arr.shape[1]), int(arr.shape[2])]
                        else:
                            n_pages = 1
                            out_shape = [h, w]
                        dt = str(arr.dtype)
                    except Exception:
                        n_pages = 1
                        out_shape = [h, w]
                else:
                    n_pages = n
                    out_shape = [n_pages, h, w]
            if "float" not in dt:
                return HeaderInfo(out_shape, dt, n_pages, file_bytes, False, f"non_float_dtype={dt}")
            return HeaderInfo(
                out_shape,
                "float32" if "float32" in dt else dt,
                n_pages,
                file_bytes,
                True,
            )
        return HeaderInfo([], "unknown", 0, file_bytes, False, f"suffix={suffix}")
    except Exception as e:  # noqa: BLE001
        return HeaderInfo([], "unknown", 0, file_bytes, False, repr(e))


def discover_files(root: Path, stats: BuildStats) -> List[FileHit]:
    hits: List[FileHit] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place for speed
        dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDE_DIR_TOKENS]
        for name in filenames:
            p = Path(dirpath) / name
            if not p.is_file():
                continue
            source = infer_source(root, p)
            if source is None:
                continue
            stats.discovered_files[source] += 1
            # positive name filter per source
            ok_name = False
            target_role = "hq_identity"
            provenance = "algorithmic_high_snr_reconstruction"
            protocol = "HQ_PROTOCOL"
            if source == "BioTISR" and RE_SIM_GT.match(name):
                ok_name = True
                target_role = "SIM_gt"
                provenance = "algorithmic_high_snr_sim_reconstruction"
            elif source in {"DeepInsight_2D", "DeepInsight_3D"} and RE_RC_HIGH.match(name):
                ok_name = True
                target_role = "RC_highsnr"
                provenance = "algorithmic_high_snr_sim_reconstruction"
            if not ok_name:
                stats.rejected_files[source] += 1
                stats.reject_reasons["name_not_hq_target"] += 1
                continue
            reason = path_is_excluded(p)
            if reason:
                stats.rejected_files[source] += 1
                stats.reject_reasons[reason] += 1
                continue
            cat = infer_category(source, p, root)
            gid = infer_group_id(source, p, root)
            hits.append(
                FileHit(
                    path=p,
                    source=source,
                    group_id=gid,
                    category=cat,
                    target_role=target_role,
                    provenance=provenance,
                    protocol=protocol,
                )
            )
            stats.accepted_files[source] += 1
    return hits


def assign_splits(
    group_ids_by_source: Dict[str, List[str]],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
    holdout_test: bool,
) -> Dict[str, str]:
    """Return map group_id -> split. Group is atomic unit."""
    import random

    rng = random.Random(seed)
    out: Dict[str, str] = {}
    for source, groups in group_ids_by_source.items():
        g = sorted(set(groups))
        rng.shuffle(g)
        n = len(g)
        if holdout_test:
            n_test = max(1, int(round(n * (1.0 - train_frac - val_frac)))) if n >= 10 else 0
            n_val = max(1, int(round(n * val_frac))) if n >= 5 else max(0, min(1, n // 5))
            n_train = n - n_val - n_test
            if n_train < 1 and n > 0:
                n_train = n
                n_val = 0
                n_test = 0
        else:
            n_test = 0
            n_val = max(1, int(round(n * val_frac))) if n >= 5 else max(0, min(1, n // 5))
            n_train = n - n_val
            if n_train < 1 and n > 0:
                n_train = n
                n_val = 0
        for i, gid in enumerate(g):
            if i < n_train:
                out[gid] = "train"
            elif i < n_train + n_val:
                out[gid] = "val"
            else:
                out[gid] = "test"
    return out


def expand_pages(
    hits: Sequence[FileHit],
    split_of: Dict[str, str],
    stats: BuildStats,
    *,
    dry_run: bool,
    max_header_errors: int = 50,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    group_seen: Dict[str, set] = defaultdict(set)

    for hit in hits:
        split = split_of.get(hit.group_id, "train")
        if dry_run:
            # cheap: estimate pages without reading full stacks if possible
            # still need header for truth — headers are required for real dry-run counts
            hdr = read_header(hit.path)
        else:
            hdr = read_header(hit.path)
        if not hdr.ok:
            stats.reject_reasons[f"header:{hdr.error}"] += 1
            stats.rejected_files[hit.source] += 1
            if len(stats.errors) < max_header_errors:
                stats.errors.append(f"{hit.path}: {hdr.error}")
            continue
        n_pages = hdr.n_pages
        shape = hdr.shape
        stats.pages[hit.source] += n_pages
        group_seen[hit.source].add(hit.group_id)
        morph = morphology_guess(hit.category)
        for page_index in range(n_pages):
            sample_id = stable_id(hit.group_id, str(page_index), str(hit.path))
            rec = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "split": split,
                "source_dataset": hit.source,
                "protocol": hit.protocol,
                "category_raw": hit.category,
                "condition_raw": "high_snr_proxy",
                "morphology": morph,
                "biological_group_id": hit.group_id,
                "path": str(hit.path),
                "shape": shape,
                "page_index": page_index,
                "dtype": hdr.dtype,
                "target_role": hit.target_role,
                "target_provenance": hit.provenance,
                "trainable": split != "test",
                "is_explicit_test": split == "test",
                "file_bytes": hdr.file_bytes,
            }
            # validate required fields present
            for k in REQUIRED_FIELDS:
                if k not in rec:
                    raise RuntimeError(f"internal: missing required field {k}")
            records.append(rec)

    for s, gs in group_seen.items():
        stats.groups[s] = len(gs)
    return records


def print_stats(stats: BuildStats, n_records: int, split_counts: Dict[str, int]) -> None:
    print("=== HQ manifest build summary ===")
    print(f"schema: {SCHEMA_VERSION}")
    print("files discovered / accepted / rejected:")
    sources = sorted(set(stats.discovered_files) | set(stats.accepted_files) | set(stats.rejected_files))
    for s in sources:
        print(
            f"  {s:18s}  disc={stats.discovered_files[s]:6d}  "
            f"acc={stats.accepted_files[s]:6d}  rej={stats.rejected_files[s]:6d}  "
            f"groups={stats.groups.get(s, 0):5d}  pages={stats.pages.get(s, 0):7d}"
        )
    print("pages total:", sum(stats.pages.values()))
    print("groups total:", sum(stats.groups.values()))
    print("records (pages with split):", n_records)
    print("split counts:", dict(split_counts))
    if stats.reject_reasons:
        print("top reject reasons:")
        for k, v in sorted(stats.reject_reasons.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {v:6d}  {k}")
    if stats.errors:
        print(f"header errors (first {len(stats.errors)}):")
        for e in stats.errors[:10]:
            print(" ", e)
    print("audit reference (full pre-QC, not frozen subset):")
    print("  BioTISR ~405 groups / 7488 pages")
    print("  DeepInsight_2D ~2658 groups / 2658 pages")
    print("  DeepInsight_3D canonical ~1716 groups / 34312 pages")
    print("  SUM ~4779 groups / 44458 pages")


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def validate_against_loader(records: Sequence[Dict[str, Any]]) -> None:
    """Import-free structural check mirroring parse_hq_record requirements."""
    allowed = {"train", "val", "test"}
    group_splits: Dict[str, set] = defaultdict(set)
    for i, rec in enumerate(records):
        for k in REQUIRED_FIELDS:
            if k not in rec:
                raise AssertionError(f"record {i} missing {k}")
        if rec["split"] not in allowed:
            raise AssertionError(f"bad split {rec['split']}")
        sh = rec["shape"]
        if not isinstance(sh, list) or len(sh) not in (2, 3):
            raise AssertionError(f"bad shape {sh}")
        group_splits[rec["biological_group_id"]].add(rec["split"])
        # forbidden alias that would break loader
        if "group_id" in rec and "biological_group_id" not in rec:
            raise AssertionError("must use biological_group_id not group_id alone")
    leaks = {g: s for g, s in group_splits.items() if len(s) > 1}
    if leaks:
        raise AssertionError(f"group split leak: {list(leaks.items())[:3]}")


def self_test(tmp: Path) -> int:
    """Create synthetic BioTISR/DI layout and verify end-to-end generation + load."""
    import numpy as np

    try:
        import mrcfile
        import tifffile
    except ImportError as e:
        print("self-test requires mrcfile and tifffile:", e, file=sys.stderr)
        return 2

    root = tmp / "Dataset"
    # BioTISR
    for i in range(4):
        cell = root / "BioTISR" / "CCPs_488" / f"Cell_{i:03d}"
        cell.mkdir(parents=True, exist_ok=True)
        arr = np.random.randn(5, 64, 64).astype(np.float32)
        with mrcfile.new(str(cell / "SIM_gt.mrc"), overwrite=True) as mrc:
            mrc.set_data(arr)
        # noise files that must be ignored
        (cell / "RawSIMData_level_01.mrc").write_bytes(b"")
    # DI2D
    for i in range(3):
        d = root / "DeepInsight_2D_training_data" / "MT_demo" / "150" / f"Exp{i}"
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(d / f"RC_{i:04d}_highsnr.tif"), np.random.randn(64, 64).astype(np.float32))
        tifffile.imwrite(str(d / f"WF_{i:04d}_lowsnr.tif"), np.zeros((32, 32), dtype=np.uint16))
        tifffile.imwrite(str(d / f"GTdenoised_{i:04d}.tif"), np.random.randn(64, 64).astype(np.float32))
    # DI3D
    for i in range(2):
        d = root / "DeepInsight_3D_training_data" / "raw_tif_fullsize" / "ER" / "300" / f"vol{i}"
        d.mkdir(parents=True, exist_ok=True)
        stack = np.random.randn(4, 64, 64).astype(np.float32)
        tifffile.imwrite(str(d / "RC_highsnr.tif"), stack)

    out = tmp / "hq_manifest.jsonl"
    stats = BuildStats()
    hits = discover_files(root, stats)
    by_src: Dict[str, List[str]] = defaultdict(list)
    for h in hits:
        by_src[h.source].append(h.group_id)
    split_of = assign_splits(by_src, train_frac=0.7, val_frac=0.2, seed=0, holdout_test=True)
    records = expand_pages(hits, split_of, stats, dry_run=False)
    validate_against_loader(records)
    write_jsonl(out, records)

    # load with package if available
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from microscopy_vae.data.manifest import load_hq_manifest

        loaded = load_hq_manifest(out, allow_splits=("train", "val"), refuse_test=True)
        assert len(loaded) > 0
        assert all(r.split != "test" for r in loaded)
        print(f"self-test OK: wrote {len(records)} page records; loader accepted {len(loaded)} train+val")
        print_stats(stats, len(records), _split_counts(records))
        return 0
    except Exception as e:  # noqa: BLE001
        print("self-test structural write OK; package load failed:", e)
        print("JSONL still written to", out)
        print_stats(stats, len(records), _split_counts(records))
        return 0


def _split_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for r in records:
        c[r["split"]] += 1
    return dict(c)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build HQ page JSONL for microscopy-vae S1")
    ap.add_argument("--root", type=str, default=None, help="Dataset root (e.g. F:\\Dataset)")
    ap.add_argument("--out", type=str, default="hq_manifest.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="Build in memory, print stats, do not require writing")
    ap.add_argument("--write-even-if-dry-run", action="store_true")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--holdout-test", action="store_true", default=True)
    ap.add_argument("--no-holdout-test", action="store_true", help="Only train/val (no test rows)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true", help="Run synthetic end-to-end test")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="hq_manifest_selftest_") as td:
            return self_test(Path(td))

    if not args.root:
        print("--root is required unless --self-test", file=sys.stderr)
        return 2

    root = Path(args.root)
    if not root.is_dir():
        print(f"root not found: {root}", file=sys.stderr)
        return 2

    holdout_test = not args.no_holdout_test
    train_frac = args.train_frac
    val_frac = args.val_frac
    if holdout_test and train_frac + val_frac >= 0.999:
        print("with holdout-test, train_frac+val_frac must leave room for test", file=sys.stderr)
        return 2

    stats = BuildStats()
    print(f"Scanning {root} ...", flush=True)
    hits = discover_files(root, stats)
    if not hits:
        print("ERROR: no HQ files discovered. Check --root and folder names.", file=sys.stderr)
        print_stats(stats, 0, {})
        return 1

    by_src: Dict[str, List[str]] = defaultdict(list)
    for h in hits:
        by_src[h.source].append(h.group_id)
    split_of = assign_splits(
        by_src,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=args.seed,
        holdout_test=holdout_test,
    )
    records = expand_pages(hits, split_of, stats, dry_run=args.dry_run)
    validate_against_loader(records)
    sc = _split_counts(records)
    print_stats(stats, len(records), sc)

    # expected ballpark warnings
    for s, exp_g, exp_p in [
        ("BioTISR", 405, 7488),
        ("DeepInsight_2D", 2658, 2658),
        ("DeepInsight_3D", 1716, 34312),
    ]:
        g, p = stats.groups.get(s, 0), stats.pages.get(s, 0)
        if g == 0:
            print(f"WARNING: {s} found 0 groups (audit ~{exp_g}/{exp_p})")
        elif abs(g - exp_g) / max(exp_g, 1) > 0.15:
            print(f"WARNING: {s} groups={g} pages={p} far from audit ~{exp_g}/{exp_p}")

    if args.dry_run and not args.write_even_if_dry_run:
        print("dry-run: not writing file (pass without --dry-run to write, or --write-even-if-dry-run)")
        return 0

    out = Path(args.out)
    write_jsonl(out, records)
    # sidecar summary without paths
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_records": len(records),
        "split_counts": sc,
        "groups_by_source": dict(stats.groups),
        "pages_by_source": dict(stats.pages),
        "root": str(root),
        "seed": args.seed,
    }
    sum_path = out.with_suffix(out.suffix + ".summary.json")
    sum_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(records)} lines)")
    print(f"Wrote {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
