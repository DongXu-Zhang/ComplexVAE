"""Runtime GPU list for inference. No hardcoded world size or device ids."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch


def parse_devices(spec: str) -> List[torch.device]:
    """Parse --devices.

    auto           → all visible CUDA devices, else [cpu]
    cpu            → [cpu]
    cuda           → [cuda:0]  (first visible GPU only)
    cuda:0,cuda:2  → those logical indices in this process
    0,2,3          → cuda:0,cuda:2,cuda:3

    Indices are *logical* (after CUDA_VISIBLE_DEVICES), not nvidia-smi physical ids.
    """
    raw = (spec or "auto").strip()
    if not raw or raw.lower() == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
        return [torch.device("cpu")]
    if raw.lower() == "cpu":
        return [torch.device("cpu")]
    if raw.lower() == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise ValueError("cuda requested but no CUDA device is visible")
        return [torch.device("cuda", 0)]

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --devices list")
    out: List[torch.device] = []
    seen: set = set()
    n_cuda = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    for p in parts:
        pl = p.lower()
        if pl == "cpu":
            d = torch.device("cpu")
        elif pl.isdigit() or (pl.startswith("-") and pl[1:].isdigit()):
            idx = int(pl)
            d = torch.device("cuda", idx)
        elif pl.startswith("cuda"):
            d = torch.device(p if ":" in p else "cuda:0")
            if d.index is None:
                d = torch.device("cuda", 0)
        else:
            raise ValueError(f"unrecognized device {p!r}")
        if d.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(f"{d} requested but CUDA is not available")
            idx = int(d.index) if d.index is not None else 0
            if idx < 0 or idx >= n_cuda:
                raise ValueError(
                    f"{d} is out of range for this process "
                    f"(torch.cuda.device_count()={n_cuda}; indices are logical, "
                    f"after CUDA_VISIBLE_DEVICES)"
                )
            d = torch.device("cuda", idx)
        key = (d.type, d.index)
        if key in seen:
            raise ValueError(f"duplicate device {d}")
        seen.add(key)
        out.append(d)
    kinds = {d.type for d in out}
    if len(kinds) > 1:
        raise ValueError(f"do not mix device kinds in --devices, got {out}")
    return out


def assign_round_robin(n_tasks: int, n_workers: int) -> List[List[int]]:
    """Static split: tile i → worker i % n_workers. Complete, no overlap."""
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if n_tasks < 0:
        raise ValueError("n_tasks must be >= 0")
    buckets: List[List[int]] = [[] for _ in range(n_workers)]
    for i in range(n_tasks):
        buckets[i % n_workers].append(i)
    return buckets


def describe_devices(devices: Sequence[torch.device]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in devices:
        row: Dict[str, Any] = {"device": str(d), "type": d.type, "index": d.index}
        if d.type == "cuda" and torch.cuda.is_available():
            idx = int(d.index or 0)
            props = torch.cuda.get_device_properties(idx)
            row["name"] = props.name
            row["total_memory_bytes"] = int(props.total_memory)
            row["major_minor"] = [int(props.major), int(props.minor)]
        rows.append(row)
    return rows


def primary_device(devices: Sequence[torch.device]) -> torch.device:
    if not devices:
        raise ValueError("empty device list")
    return devices[0]
