"""Single-node DDP helpers. WORLD_SIZE=1 (or no torchrun) stays single-process."""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import torch


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False
    backend: str = "none"
    device: torch.device = torch.device("cpu")
    # Separate Gloo group for object broadcasts. NCCL DDP must not see
    # all_gather_object between its forward and backward.
    object_group: Any = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _object_group(info: DistInfo):
    return info.object_group


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def probe_world_size() -> int:
    ws = _env_int("WORLD_SIZE")
    return int(ws) if ws is not None and ws >= 1 else 1


def init_distributed() -> DistInfo:
    """Bind this process to one GPU when launched by torchrun; otherwise single device.

    Does not silently fall back to single-GPU if WORLD_SIZE>1 and init fails.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = _env_int("LOCAL_RANK")
        if local is None:
            local = rank
        if torch.cuda.is_available():
            torch.cuda.set_device(int(local))
            device = torch.device("cuda", int(local))
        else:
            device = torch.device("cpu")
        backend = dist.get_backend() if world > 1 else "none"
        return DistInfo(
            rank=rank,
            local_rank=int(local),
            world_size=world,
            enabled=world > 1,
            backend=str(backend),
            device=device,
        )

    world = probe_world_size()
    rank = _env_int("RANK")
    local = _env_int("LOCAL_RANK")
    if world <= 1 or rank is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return DistInfo(device=device)

    if local is None:
        raise RuntimeError(
            f"WORLD_SIZE={world} but LOCAL_RANK is unset. Launch with torchrun "
            "(do not start extra processes by hand)."
        )
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        n_vis = torch.cuda.device_count()
        if n_vis < 1:
            raise RuntimeError("NCCL DDP requested but no CUDA device is visible")
        if int(local) >= n_vis:
            raise RuntimeError(
                f"LOCAL_RANK={local} >= visible CUDA device count {n_vis}. "
                "Set CUDA_VISIBLE_DEVICES and --nproc_per_node to match."
            )
        torch.cuda.set_device(int(local))
        device = torch.device("cuda", int(local))
    else:
        device = torch.device("cpu")
    from datetime import timedelta

    # Rank0-only work (focus sidecar, val) parks other ranks in a collective.
    # Default NCCL timeout (~10 min) would kill a 150k run during the first val.
    timeout_min = int(os.environ.get("MICROVAE_DDP_TIMEOUT_MIN", "360"))
    timeout = timedelta(minutes=max(timeout_min, 1))
    pg_kwargs: Dict[str, Any] = {
        "backend": backend,
        "rank": int(rank),
        "world_size": int(world),
        "timeout": timeout,
    }
    if backend == "nccl":
        try:
            dist.init_process_group(device_id=device, **pg_kwargs)
        except TypeError:
            dist.init_process_group(**pg_kwargs)
    else:
        dist.init_process_group(**pg_kwargs)
    object_group = None
    if backend == "nccl":
        try:
            object_group = dist.new_group(backend="gloo", timeout=timeout)
        except TypeError:
            object_group = dist.new_group(backend="gloo")
    return DistInfo(
        rank=int(rank),
        local_rank=int(local),
        world_size=int(world),
        enabled=True,
        backend=backend,
        device=device,
        object_group=object_group,
    )


def cleanup_distributed(info: Optional[DistInfo] = None) -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        if info is not None and info.object_group is not None:
            try:
                dist.destroy_process_group(info.object_group)
            except Exception:
                pass
            info.object_group = None
    except Exception:
        pass
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def barrier(info: DistInfo) -> None:
    if not info.enabled:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        group = _object_group(info)
        if group is not None:
            dist.barrier(group=group)
        else:
            dist.barrier()


def broadcast_object(obj: Any, info: DistInfo, src: int = 0) -> Any:
    if not info.enabled:
        return obj
    import torch.distributed as dist

    payload = [obj if info.rank == src else None]
    group = _object_group(info)
    if group is not None:
        dist.broadcast_object_list(payload, src=src, group=group)
    else:
        dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_reduce_sum(t: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if not info.enabled:
        return t
    import torch.distributed as dist

    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_mean_scalar(value: float, info: DistInfo, device: torch.device) -> float:
    if not info.enabled:
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    all_reduce_sum(t, info)
    return float(t.item() / max(info.world_size, 1))


def all_reduce_max_flag(flag: bool, info: DistInfo, device: torch.device) -> bool:
    """True if any rank passed True. Single-process returns the local flag."""
    if not info.enabled:
        return bool(flag)
    import torch.distributed as dist

    t = torch.tensor([1.0 if flag else 0.0], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(float(t.item()) > 0.5)


def raise_if_any_rank_failed(ok: bool, message: str, info: DistInfo) -> None:
    """If any rank failed, every rank raises (avoids a NCCL/Gloo hang)."""
    if not info.enabled:
        if not ok:
            raise RuntimeError(message)
        return
    import torch.distributed as dist

    gathered: list = [None] * info.world_size
    payload = None if ok else str(message)
    group = _object_group(info)
    if group is not None:
        dist.all_gather_object(gathered, payload, group=group)
    else:
        dist.all_gather_object(gathered, payload)
    fails = [m for m in gathered if m is not None]
    if fails:
        raise RuntimeError(" | ".join(str(m) for m in fails))


def assert_resume_world_size(ckpt_world_size: Optional[int], live_world_size: int) -> None:
    if ckpt_world_size is None:
        return
    if int(ckpt_world_size) != int(live_world_size):
        raise ValueError(
            f"resume_exact checkpoint world_size={int(ckpt_world_size)} != "
            f"this process world_size={int(live_world_size)}. "
            "The hierarchical sampler stream is sharded by GPU count; "
            "resume with the same nproc_per_node."
        )


def _gathered_key_union(values: Dict[str, float], info: DistInfo) -> list:
    """Same key order on every rank. Missing keys are filled with 0 by the caller.

    One rank skipping GAN logs (empty support) used to allreduce tensors of
    different lengths and hang.
    """
    if not info.enabled:
        return sorted(values)
    import torch.distributed as dist

    gathered: list = [None] * info.world_size
    group = _object_group(info)
    if group is not None:
        dist.all_gather_object(gathered, list(values.keys()), group=group)
    else:
        dist.all_gather_object(gathered, list(values.keys()))
    keys: set = set()
    for ks in gathered:
        keys.update(ks or [])
    return sorted(keys)


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return state
    if not any(str(k).startswith("module.") for k in state):
        return state
    return {str(k)[7:] if str(k).startswith("module.") else k: v for k, v in state.items()}


def wrap_ddp(module: torch.nn.Module, info: DistInfo) -> torch.nn.Module:
    if not info.enabled:
        return module
    from torch.nn.parallel import DistributedDataParallel as DDP

    kwargs: Dict[str, Any] = {
        "find_unused_parameters": False,
        "broadcast_buffers": True,
    }
    if info.device.type == "cuda":
        kwargs["device_ids"] = [info.device.index if info.device.index is not None else info.local_rank]
        kwargs["output_device"] = kwargs["device_ids"][0]
    return DDP(module, **kwargs)


@contextmanager
def maybe_no_sync(module: Optional[torch.nn.Module], enabled: bool) -> Iterator[None]:
    if enabled and module is not None and hasattr(module, "no_sync"):
        with module.no_sync():
            yield
    else:
        with nullcontext():
            yield


def resolve_per_device_batch(
    *,
    yaml_microbatch: int,
    yaml_accum: int,
    world_size: int,
    scale_global_batch: bool,
) -> tuple[int, int, int]:
    """Return (per_device_batch, accum, effective_global_batch).

    Default keeps yaml_microbatch * yaml_accum as the global batch (split across ranks).
    """
    yaml_microbatch = int(yaml_microbatch)
    yaml_accum = int(yaml_accum)
    world_size = max(int(world_size), 1)
    if yaml_microbatch < 1 or yaml_accum < 1:
        raise ValueError("microbatch_size and grad_accum must be >= 1")
    original = yaml_microbatch * yaml_accum
    if world_size == 1:
        return yaml_microbatch, yaml_accum, original
    if scale_global_batch:
        return yaml_microbatch, yaml_accum, original * world_size
    if original % world_size != 0:
        raise ValueError(
            f"Cannot keep effective_global_batch_size={original} across world_size={world_size} "
            f"(not divisible). Use a GPU count that divides {original}, or set "
            "training.ddp_scale_global_batch=true (this changes training dynamics)."
        )
    rank_budget = original // world_size
    if rank_budget < 1:
        raise ValueError(f"effective global batch {original} is smaller than world_size={world_size}")
    if rank_budget % yaml_accum == 0:
        per_device = rank_budget // yaml_accum
        accum = yaml_accum
    else:
        per_device = 1
        accum = rank_budget
    if per_device < 1:
        raise ValueError("resolved per_device_batch_size < 1")
    return int(per_device), int(accum), int(original)


def reduce_mean_map(values: Dict[str, float], info: DistInfo, device: torch.device) -> Dict[str, float]:
    if not info.enabled:
        return dict(values)
    keys = _gathered_key_union(values, info)
    if not keys:
        return {}
    t = torch.tensor([float(values.get(k, 0.0)) for k in keys], device=device, dtype=torch.float64)
    all_reduce_sum(t, info)
    t = t / max(info.world_size, 1)
    return {k: float(t[i].item()) for i, k in enumerate(keys)}


def reduce_sum_map(values: Dict[str, float], info: DistInfo, device: torch.device) -> Dict[str, float]:
    if not info.enabled:
        return dict(values)
    keys = _gathered_key_union(values, info)
    if not keys:
        return {}
    t = torch.tensor([float(values.get(k, 0.0)) for k in keys], device=device, dtype=torch.float64)
    all_reduce_sum(t, info)
    return {k: float(t[i].item()) for i, k in enumerate(keys)}
