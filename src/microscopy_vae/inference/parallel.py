"""Multi-GPU tiled inference: one process per GPU, fuse in original tile order.

Does not change reconstruction math. Full-image inference is not data-parallel
(batch=1 + global GroupNorm/attention); extra GPUs are unused for `full`.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from microscopy_vae.inference.devices import assign_round_robin, describe_devices
from microscopy_vae.inference.tiling import (
    crop_halo_core,
    fuse_halo_cores,
    fuse_tiled_recons,
    halo_jobs,
    pad_if_smaller,
    reconstruct_full,
    reconstruct_halo,
    reconstruct_one_tile,
    reconstruct_tiled,
    tile_boxes,
)


def _tile_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Spawn-safe worker: bind one device, reconstruct assigned tiles."""
    device_s = payload["device"]
    try:
        import torch as _torch

        _torch.set_num_threads(max(int(payload.get("threads", 1)), 1))
        dev = _torch.device(device_s)
        if dev.type == "cuda":
            _torch.cuda.set_device(dev)
            _torch.cuda.reset_peak_memory_stats(dev)
        from microscopy_vae.config.schema import RootConfig
        from microscopy_vae.systems.factory import build_hq_codec_system

        cfg = RootConfig.model_validate(payload["cfg"])
        t_load0 = time.perf_counter()
        system = build_hq_codec_system(cfg)
        system.vae.load_state_dict(payload["state"])
        system.to(dev)
        system.eval()
        if system.perceptual is not None:
            system.perceptual.eval()
        t_load = time.perf_counter() - t_load0
        x = payload["x"].to(dev, non_blocking=False)
        boxes = payload["boxes"]
        indices: List[int] = list(payload["indices"])
        sc = int(payload["spatial_compression"])
        pad_mode = str(payload["padding_mode"])
        tiles: List[Tuple[int, _torch.Tensor]] = []
        if dev.type == "cuda":
            _torch.cuda.synchronize(dev)
        t_fwd0 = time.perf_counter()
        with _torch.no_grad():
            for i in indices:
                y0, x0, y1, x1 = boxes[i]
                tile = x[:, :, y0:y1, x0:x1]
                recon = reconstruct_one_tile(
                    system.vae, tile, spatial_compression=sc, padding_mode=pad_mode
                )
                tiles.append((int(i), recon.detach().cpu()))
        if dev.type == "cuda":
            _torch.cuda.synchronize(dev)
        t_fwd = time.perf_counter() - t_fwd0
        peak = int(_torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
        return {
            "ok": True,
            "device": str(dev),
            "tiles": tiles,
            "n": len(tiles),
            "load_s": float(t_load),
            "forward_s": float(t_fwd),
            "peak_bytes": peak,
        }
    except Exception:  # noqa: BLE001
        return {"ok": False, "device": device_s, "error": traceback.format_exc()}


def run_tiled(
    model,
    x: torch.Tensor,
    *,
    cfg_dump: Dict[str, Any],
    devices: Sequence[torch.device],
    tile_size: int,
    overlap: int,
    spatial_compression: int,
    padding_mode: str = "reflect",
    blend_mode: str = "linear",
    snap: Optional[int] = None,
    return_aux: bool = False,
) -> Any:
    """Tiled recon. 1 device → existing sequential path; N devices → spawn workers."""
    devs = list(devices)
    if len(devs) <= 1:
        return reconstruct_tiled(
            model,
            x,
            tile_size=tile_size,
            overlap=overlap,
            spatial_compression=spatial_compression,
            padding_mode=padding_mode,
            blend_mode=blend_mode,
            snap=snap,
            return_aux=return_aux,
        )

    if snap is None:
        snap = spatial_compression
    # Geometry on CPU; workers only see tiles.
    x_cpu = x.detach().cpu()
    x_work, _small = pad_if_smaller(x_cpu, tile_size, mode=padding_mode)
    h, w = int(x_work.shape[-2]), int(x_work.shape[-1])
    boxes = tile_boxes(h, w, tile_size, overlap, snap=int(snap))
    n_tiles = len(boxes)
    splits = assign_round_robin(n_tiles, len(devs))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    payloads = []
    for d, idxs in zip(devs, splits):
        payloads.append(
            {
                "device": str(d),
                "cfg": cfg_dump,
                "state": state,
                "x": x_work,
                "boxes": boxes,
                "indices": idxs,
                "spatial_compression": int(spatial_compression),
                "padding_mode": padding_mode,
                "threads": 1,
            }
        )

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    t_par0 = time.perf_counter()
    with ctx.Pool(len(devs)) as pool:
        worker_out = pool.map(_tile_worker, payloads)
    t_par = time.perf_counter() - t_par0

    errors = [w for w in worker_out if not w.get("ok")]
    if errors:
        msgs = "\n".join(f"{e.get('device')}: {e.get('error')}" for e in errors)
        raise RuntimeError(f"tiled multi-GPU worker failed:\n{msgs}")

    recons: List[Optional[torch.Tensor]] = [None] * n_tiles
    per_device: List[Dict[str, Any]] = []
    for w, idxs in zip(worker_out, splits):
        for i, t in w["tiles"]:
            if recons[i] is not None:
                raise RuntimeError(f"duplicate tile index {i}")
            recons[i] = t
        per_device.append(
            {
                "device": w["device"],
                "n_tiles": int(w["n"]),
                "indices": list(idxs),
                "load_s": w["load_s"],
                "forward_s": w["forward_s"],
                "peak_bytes": w["peak_bytes"],
            }
        )
    missing = [i for i, r in enumerate(recons) if r is None]
    if missing:
        raise RuntimeError(f"missing tile indices {missing[:20]}")

    t_fuse0 = time.perf_counter()
    fused = fuse_tiled_recons(
        x_cpu,
        boxes,
        recons,  # type: ignore[arg-type]
        tile_size=tile_size,
        overlap=overlap,
        spatial_compression=spatial_compression,
        padding_mode=padding_mode,
        blend_mode=blend_mode,
        snap=snap,
        return_aux=return_aux,
    )
    t_fuse = time.perf_counter() - t_fuse0
    parallel_aux = {
        "parallel": "tile_mp",
        "world_size": len(devs),
        "devices": describe_devices(devs),
        "per_device": per_device,
        "n_tiles": n_tiles,
        "wall_workers_s": float(t_par),
        "fuse_s": float(t_fuse),
    }
    if return_aux:
        y, aux = fused
        aux["parallel"] = parallel_aux
        return y, aux
    return fused


def _halo_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    device_s = payload["device"]
    try:
        import torch as _torch

        _torch.set_num_threads(max(int(payload.get("threads", 1)), 1))
        dev = _torch.device(device_s)
        if dev.type == "cuda":
            _torch.cuda.set_device(dev)
            _torch.cuda.reset_peak_memory_stats(dev)
        from microscopy_vae.config.schema import RootConfig
        from microscopy_vae.systems.factory import build_hq_codec_system

        cfg = RootConfig.model_validate(payload["cfg"])
        t_load0 = time.perf_counter()
        system = build_hq_codec_system(cfg)
        system.vae.load_state_dict(payload["state"])
        system.to(dev)
        system.eval()
        t_load = time.perf_counter() - t_load0
        x = payload["x"].to(dev, non_blocking=False)
        jobs: List[Dict[str, int]] = list(payload["jobs"])
        indices: List[int] = list(payload["indices"])
        sc = int(payload["spatial_compression"])
        pad_mode = str(payload["padding_mode"])
        cores: List[Tuple[int, _torch.Tensor]] = []
        if dev.type == "cuda":
            _torch.cuda.synchronize(dev)
        t_fwd0 = time.perf_counter()
        with _torch.no_grad():
            for i in indices:
                job = jobs[i]
                window = x[:, :, job["hy0"] : job["hy1"], job["hx0"] : job["hx1"]]
                recon_w = reconstruct_one_tile(
                    system.vae, window, spatial_compression=sc, padding_mode=pad_mode
                )
                core = crop_halo_core(recon_w, job)
                cores.append((int(i), core.detach().cpu()))
        if dev.type == "cuda":
            _torch.cuda.synchronize(dev)
        t_fwd = time.perf_counter() - t_fwd0
        peak = int(_torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
        return {
            "ok": True,
            "device": str(dev),
            "cores": cores,
            "n": len(cores),
            "load_s": float(t_load),
            "forward_s": float(t_fwd),
            "peak_bytes": peak,
        }
    except Exception:  # noqa: BLE001
        return {"ok": False, "device": device_s, "error": traceback.format_exc()}


def run_halo(
    model,
    x: torch.Tensor,
    *,
    cfg_dump: Dict[str, Any],
    devices: Sequence[torch.device],
    tile_size: int,
    overlap: int,
    halo: int,
    spatial_compression: int,
    padding_mode: str = "reflect",
    blend_mode: str = "linear",
    return_aux: bool = False,
) -> Any:
    """Halo recon. 1 device → sequential path; N devices shard cores, same windows."""
    devs = list(devices)
    if len(devs) <= 1:
        return reconstruct_halo(
            model,
            x,
            tile_size=tile_size,
            overlap=overlap,
            halo=halo,
            spatial_compression=spatial_compression,
            padding_mode=padding_mode,
            blend_mode=blend_mode,
            return_aux=return_aux,
        )
    halo = max(int(halo), 0)
    if halo <= 0:
        return run_tiled(
            model,
            x,
            cfg_dump=cfg_dump,
            devices=devs,
            tile_size=tile_size,
            overlap=overlap,
            spatial_compression=spatial_compression,
            padding_mode=padding_mode,
            blend_mode=blend_mode,
            return_aux=return_aux,
        )
    x_cpu = x.detach().cpu()
    x_work, _small = pad_if_smaller(x_cpu, tile_size, mode=padding_mode)
    h, w = int(x_work.shape[-2]), int(x_work.shape[-1])
    boxes = tile_boxes(h, w, tile_size, overlap, snap=int(spatial_compression))
    jobs = halo_jobs(h, w, boxes, halo)
    n_tiles = len(jobs)
    splits = assign_round_robin(n_tiles, len(devs))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payloads = []
    for d, idxs in zip(devs, splits):
        payloads.append(
            {
                "device": str(d),
                "cfg": cfg_dump,
                "state": state,
                "x": x_work,
                "jobs": jobs,
                "indices": idxs,
                "spatial_compression": int(spatial_compression),
                "padding_mode": padding_mode,
                "threads": 1,
            }
        )
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    t_par0 = time.perf_counter()
    with ctx.Pool(len(devs)) as pool:
        worker_out = pool.map(_halo_worker, payloads)
    t_par = time.perf_counter() - t_par0
    errors = [w for w in worker_out if not w.get("ok")]
    if errors:
        msgs = "\n".join(f"{e.get('device')}: {e.get('error')}" for e in errors)
        raise RuntimeError(f"halo multi-GPU worker failed:\n{msgs}")
    cores: List[Optional[torch.Tensor]] = [None] * n_tiles
    per_device: List[Dict[str, Any]] = []
    for w, idxs in zip(worker_out, splits):
        for i, t in w["cores"]:
            if cores[i] is not None:
                raise RuntimeError(f"duplicate halo core index {i}")
            cores[i] = t
        per_device.append(
            {
                "device": w["device"],
                "n_tiles": int(w["n"]),
                "indices": list(idxs),
                "load_s": w["load_s"],
                "forward_s": w["forward_s"],
                "peak_bytes": w["peak_bytes"],
            }
        )
    missing = [i for i, r in enumerate(cores) if r is None]
    if missing:
        raise RuntimeError(f"missing halo core indices {missing[:20]}")
    t_fuse0 = time.perf_counter()
    fused = fuse_halo_cores(
        x_cpu,
        jobs,
        cores,  # type: ignore[arg-type]
        tile_size=tile_size,
        overlap=overlap,
        padding_mode=padding_mode,
        blend_mode=blend_mode,
        halo=halo,
        return_aux=return_aux,
    )
    t_fuse = time.perf_counter() - t_fuse0
    parallel_aux = {
        "parallel": "halo_mp",
        "world_size": len(devs),
        "devices": describe_devices(devs),
        "per_device": per_device,
        "n_tiles": n_tiles,
        "wall_workers_s": float(t_par),
        "fuse_s": float(t_fuse),
    }
    if return_aux:
        y, aux = fused
        aux["parallel"] = parallel_aux
        return y, aux
    return fused


def _full_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One worker: full-image recon of assigned pages. Never converts to tiled."""
    device_s = payload["device"]
    try:
        import torch as _torch

        _torch.set_num_threads(max(int(payload.get("threads", 1)), 1))
        dev = _torch.device(device_s)
        if dev.type == "cuda":
            _torch.cuda.set_device(dev)
        from microscopy_vae.config.schema import RootConfig
        from microscopy_vae.systems.factory import build_hq_codec_system

        cfg = RootConfig.model_validate(payload["cfg"])
        system = build_hq_codec_system(cfg)
        system.vae.load_state_dict(payload["state"])
        system.to(dev)
        system.eval()
        sc = int(payload["spatial_compression"])
        pad_mode = str(payload["padding_mode"])
        out_pages: List[Tuple[int, _torch.Tensor]] = []
        with _torch.no_grad():
            for idx, arr in payload["pages"]:
                x = arr.to(dev)
                y = reconstruct_full(
                    system.vae, x, spatial_compression=sc, padding_mode=pad_mode
                )
                out_pages.append((int(idx), y.detach().cpu()))
        return {"ok": True, "device": str(dev), "pages": out_pages}
    except Exception:  # noqa: BLE001
        return {"ok": False, "device": device_s, "error": traceback.format_exc()}


def run_full_images(
    model,
    pages: Sequence[torch.Tensor],
    *,
    cfg_dump: Dict[str, Any],
    devices: Sequence[torch.device],
    spatial_compression: int,
    padding_mode: str = "reflect",
) -> List[torch.Tensor]:
    """Assign whole images to GPUs. Does not tile a single full image."""
    if not pages:
        return []
    for i, p in enumerate(pages):
        if p.ndim != 4 or p.shape[1] != 1:
            raise ValueError(f"page {i} expected [B,1,H,W], got {tuple(p.shape)}")
    devs = list(devices)
    if len(devs) <= 1:
        out: List[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for p in pages:
                y = reconstruct_full(
                    model,
                    p.to(devs[0] if devs else p.device),
                    spatial_compression=spatial_compression,
                    padding_mode=padding_mode,
                )
                out.append(y.detach().cpu())
        return out
    splits = assign_round_robin(len(pages), len(devs))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payloads = []
    cpu_pages = [p.detach().cpu() for p in pages]
    for d, idxs in zip(devs, splits):
        payloads.append(
            {
                "device": str(d),
                "cfg": cfg_dump,
                "state": state,
                "pages": [(i, cpu_pages[i]) for i in idxs],
                "spatial_compression": int(spatial_compression),
                "padding_mode": padding_mode,
                "threads": 1,
            }
        )
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(devs)) as pool:
        worker_out = pool.map(_full_worker, payloads)
    errors = [w for w in worker_out if not w.get("ok")]
    if errors:
        msgs = "\n".join(f"{e.get('device')}: {e.get('error')}" for e in errors)
        raise RuntimeError(f"full multi-image worker failed:\n{msgs}")
    ordered: List[Optional[torch.Tensor]] = [None] * len(pages)
    for w in worker_out:
        for i, t in w["pages"]:
            if ordered[i] is not None:
                raise RuntimeError(f"duplicate full page index {i}")
            ordered[i] = t
    missing = [i for i, t in enumerate(ordered) if t is None]
    if missing:
        raise RuntimeError(f"missing full page indices {missing[:20]}")
    return ordered  # type: ignore[return-value]
