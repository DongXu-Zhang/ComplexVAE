"""Hashable snapshot of the local Python package (not a git substitute)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List


def hash_source_tree(root: Path) -> Dict[str, object]:
    root = root.resolve()
    files: List[Path] = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts or ".venv" in p.parts:
            continue
        files.append(p)
    h = hashlib.sha256()
    rels = []
    for p in files:
        rel = str(p.relative_to(root))
        rels.append(rel)
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\n")
    return {
        "root": str(root),
        "n_files": len(files),
        "sha256": h.hexdigest(),
        "files": rels,
    }
