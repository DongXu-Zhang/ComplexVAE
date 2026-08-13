#!/usr/bin/env python3
"""Validate an HQ JSONL against package contracts without requiring image pixels.

Usage:
  python tools/validate_hq_manifest.py \\
    --manifest /path/to/hq_manifest_v2.jsonl \\
    --expect-sha256 7285a66d... \\
    --prefix-map 'F:\\Dataset'=/data/Dataset   # optional reachability check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument(
        "--prefix-map",
        default=None,
        help="source_prefix=target_root , e.g. F:\\\\Dataset=/mnt/Dataset",
    )
    ap.add_argument("--check-files", action="store_true", help="stat mapped files (sample or all)")
    ap.add_argument("--check-all-files", action="store_true")
    ap.add_argument("--max-check", type=int, default=64)
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.is_file():
        print("missing manifest", mpath, file=sys.stderr)
        return 2

    digest = sha256_file(mpath)
    print("sha256", digest)
    if args.expect_sha256 and digest != args.expect_sha256.lower():
        print("SHA MISMATCH expected", args.expect_sha256, file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from microscopy_vae.data.manifest import load_hq_manifest, summarize_records
    from microscopy_vae.data.pathmap import PathPrefixMap, apply_prefix_map_to_records

    recs_tv = load_hq_manifest(mpath, allow_splits=("train", "val"), refuse_test=True)
    print("train+val", summarize_records(recs_tv))

    # full inventory counts (including test rows in file — loader skips for train)
    splits = Counter()
    sources = Counter()
    groups = defaultdict(set)
    n = 0
    with mpath.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            splits[r["split"]] += 1
            sources[r["source_dataset"]] += 1
            groups[r["source_dataset"]].add(r["biological_group_id"])
    print("file_pages", n, "splits", dict(splits), "groups", {k: len(v) for k, v in groups.items()})

    expected = {
        "BioTISR": (405, 7488),
        "DeepInsight_2D": (2658, 2658),
        "DeepInsight_3D": (1716, 34312),
    }
    ok = True
    for s, (eg, ep) in expected.items():
        g, p = len(groups[s]), sources[s]
        if g != eg or p != ep:
            print(f"COUNT DRIFT {s}: groups={g}/{eg} pages={p}/{ep}")
            ok = False
        else:
            print(f"COUNT OK {s}: groups={g} pages={p}")
    if n != 44458:
        print("total pages", n, "!= 44458")
        ok = False

    if args.prefix_map:
        if "=" not in args.prefix_map:
            print("bad --prefix-map", file=sys.stderr)
            return 2
        src, tgt = args.prefix_map.split("=", 1)
        pmap = PathPrefixMap(source_prefixes=(src,), target_root=tgt, require_exists=False)
        mapped = apply_prefix_map_to_records(recs_tv, pmap)
        print("example map", recs_tv[0].hq_path, "->", mapped[0].hq_path)
        if args.check_files or args.check_all_files:
            to_check = mapped if args.check_all_files else mapped[: args.max_check]
            # unique files
            files = []
            seen = set()
            for r in to_check:
                k = str(r.hq_path)
                if k not in seen:
                    seen.add(k)
                    files.append(r.hq_path)
            hit = sum(1 for p in files if p.is_file())
            print(f"reachable_files {hit}/{len(files)}")
            if hit == 0:
                print("WARNING: no files reachable under prefix map")
                ok = False

    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
