from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from microscopy_vae.config.loader import config_semantic_hash, load_config, resolved_dict
from microscopy_vae.config.validation import validate_for_training


def _parse_override(items: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            val: Any = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--override", action="append", default=[], help="dotted.key=JSON_or_str")
    p.add_argument("--print-resolved-config", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def _cfg(args: argparse.Namespace):
    return load_config(Path(args.config) if args.config else None, _parse_override(args.override or []))


def cmd_validate_config(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    validate_for_training(cfg)
    if args.print_resolved_config:
        print(json.dumps(resolved_dict(cfg), indent=2, sort_keys=True))
    print("OK: config valid")
    return 0


def cmd_smoke_test(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    cfg.training.max_steps = min(cfg.training.max_steps, 4)
    from microscopy_vae.engine.trainer import Trainer

    trainer = Trainer(cfg)
    info = trainer.dry_run()
    print(json.dumps(info, indent=2, default=str))
    if info.get("has_test_loader"):
        raise SystemExit("FAIL: test loader must not exist")
    if args.dry_run:
        print("OK: dry-run only")
        return 0
    result = trainer.train(max_steps=2)
    print(json.dumps({k: result[k] for k in ("final_step", "final_loss", "checkpoint")}, indent=2))
    print("OK: smoke-test")
    return 0


def cmd_overfit_small(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.engine.trainer import Trainer

    trainer = Trainer(cfg)
    result = trainer.overfit_small()
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.engine.trainer import Trainer

    trainer = Trainer(cfg)
    if args.dry_run:
        print(json.dumps(trainer.dry_run(), indent=2, default=str))
        return 0
    result = trainer.train()
    print(
        json.dumps(
            {k: result[k] for k in ("final_step", "final_loss", "checkpoint", "candidate_hits")},
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_fit_normalizer(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.data.normalization import fit_robust_normalizer
    from microscopy_vae.data.synthetic import build_synthetic_hq_pool

    if cfg.data.mode == "synthetic":
        pages = build_synthetic_hq_pool(
            n_groups=cfg.data.synthetic_n_groups,
            pages_per_group=cfg.data.synthetic_pages_per_group,
            size=cfg.data.synthetic_size,
            seed=cfg.experiment.seed,
        )
        train_pages = [p for p in pages if p.split == "train"]
        train = [p.image for p in train_pages]
        sources = [p.source for p in train_pages]
    else:
        raise SystemExit("fit-normalizer for hq_pool: use train path via Trainer (auto-fits)")
    state = fit_robust_normalizer(
        train,
        method=cfg.normalization.method,
        clip=cfg.normalization.clip,
        low_percentile=float(cfg.normalization.low_percentile),
        high_percentile=float(cfg.normalization.high_percentile),
        raw_floor_enabled=bool(cfg.normalization.raw_floor_enabled),
        raw_floor_value=float(cfg.normalization.raw_floor_value),
        scale_mode=str(cfg.normalization.scale_mode),
        fit_mode=str(cfg.normalization.fit_mode),
        max_pixels_per_page=int(cfg.normalization.max_pixels_per_page),
        sources=sources,
    )
    out = Path(args.out or "normalizer.json")
    state.save(out)
    print(
        f"Wrote {out} transform_id={state.transform_id} scale_mode={state.scale_mode} "
        f"low={state.low} high={state.high} sources={sorted(state.per_source_scales.keys())}"
    )
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.systems.factory import build_hq_codec_system

    system = build_hq_codec_system(cfg)
    print(json.dumps(system.vae.count_parameters(), indent=2))
    return 0


def cmd_inspect_data(args: argparse.Namespace) -> int:
    """Bounded audit summary — no pixels, no raw paths in output."""
    cfg = _cfg(args)
    from microscopy_vae.data.synthetic import build_synthetic_hq_pool, pool_summary
    from microscopy_vae.data.manifest import load_hq_manifest, summarize_records
    from microscopy_vae.provenance.hashing import sha256_json

    out: Dict[str, Any] = {
        "schema_version": "microvae-inspect-data-v1",
        "mode": cfg.data.mode,
        "allowed_splits": list(cfg.data.allow_splits),
        "test_refused": True,
    }
    if cfg.data.mode == "synthetic":
        pages = build_synthetic_hq_pool(
            n_groups=cfg.data.synthetic_n_groups,
            pages_per_group=cfg.data.synthetic_pages_per_group,
            size=cfg.data.synthetic_size,
            seed=cfg.experiment.seed,
        )
        # strip anything that could be a path
        out["summary"] = pool_summary(pages)
        out["sources"] = sorted({p.source for p in pages})
        out["morphologies"] = sorted({p.morphology for p in pages})
    elif cfg.data.mode == "hq_pool":
        if not cfg.data.manifest_path:
            raise SystemExit("hq_pool requires data.manifest_path")
        recs = load_hq_manifest(
            Path(cfg.data.manifest_path), allow_splits=("train", "val"), refuse_test=True
        )
        out["summary"] = summarize_records(recs)
        # anonymized: only counts, not paths
        out["n_unique_paths"] = len({str(r.hq_path) for r in recs})
    else:
        raise SystemExit(f"inspect-data does not support mode={cfg.data.mode}")
    out["content_sha256"] = sha256_json(out)
    print(json.dumps(out, indent=2, sort_keys=True))
    # size bound soft check
    if len(json.dumps(out)) > 5_000_000:
        raise SystemExit("inspect-data output exceeded size bound")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.engine.trainer import Trainer
    from microscopy_vae.engine.evaluator import evaluate_hq_loader
    from microscopy_vae.engine.checkpoint import CheckpointManager

    # force short synthetic if needed
    trainer = Trainer(cfg)
    weights = args.weights
    if weights:
        CheckpointManager.load_exported_weights(Path(weights), trainer.system.vae)
    metrics = evaluate_hq_loader(
        trainer.system,
        trainer.val_loader,
        device=trainer.device,
        use_posterior_mean=cfg.evaluation.use_posterior_mean,
        bootstrap_n=cfg.bootstrap.n_resamples,
        bootstrap_seed=cfg.bootstrap.seed,
        extended_metrics=bool(getattr(cfg.evaluation, "extended_metrics", False)),
    )
    # drop heavy page list for stdout unless --full
    if not args.full:
        metrics = {k: v for k, v in metrics.items() if k not in ("page_metrics", "group_ids")}
    print(json.dumps(metrics, indent=2, default=float))
    return 0


def _save_float_image(path: Path, arr) -> None:
    import io

    import numpy as np

    from microscopy_vae.utils.atomic import atomic_write_bytes

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import os
        import tempfile

        import tifffile

        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
        os.close(fd)
        try:
            tifffile.imwrite(tmp, arr)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    else:
        buf = io.BytesIO()
        np.save(buf, arr)
        atomic_write_bytes(path, buf.getvalue())


def cmd_infer(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.systems.factory import build_hq_codec_system
    from microscopy_vae.inference.tiling import (
        attention_matrix_numel,
        reconstruct_full,
    )
    from microscopy_vae.inference.compare import load_infer_weights, run_full_tiled_compare, save_compare_pack
    from microscopy_vae.data.readers import read_page
    from microscopy_vae.data.normalization import NormalizationState, Normalizer, guess_source_from_path
    import numpy as np

    if not args.input or not args.output:
        raise SystemExit("infer requires --input and --output")
    mode = str(getattr(args, "inference_mode", None) or "full")
    # --tiled is a backward-compat alias only when mode was left at default full.
    if bool(getattr(args, "tiled", False)) and mode == "full":
        mode = "tiled"
    if mode not in {"full", "tiled", "compare"}:
        raise SystemExit("inference-mode must be full | tiled | compare")

    from microscopy_vae.inference.devices import describe_devices, parse_devices, primary_device
    from microscopy_vae.inference.parallel import run_tiled
    from microscopy_vae.provenance.hashing import sha256_file
    import time

    spec = str(getattr(args, "devices", None) or "auto")
    try:
        devices = parse_devices(spec)
    except ValueError as exc:
        raise SystemExit(f"--devices: {exc}") from exc
    primary = primary_device(devices)

    system = build_hq_codec_system(cfg)
    weights_kind = "uninitialized"
    if args.weights:
        weights_kind = load_infer_weights(
            Path(args.weights), system.vae, use_ema=not bool(getattr(args, "raw_weights", False))
        )
    system.eval()
    if system.perceptual is not None:
        system.perceptual.eval()
    page, _ = read_page(Path(args.input), int(args.page))
    norm = None
    infer_source = None
    if args.normalizer:
        norm = Normalizer(NormalizationState.load(Path(args.normalizer)))
        infer_source = getattr(args, "source", None) or guess_source_from_path(
            args.input, known=norm.known_sources()
        )
        if norm.is_per_source() and not infer_source:
            raise SystemExit(
                "per-source normalizer: pass --source BioTISR|DeepInsight_2D|DeepInsight_3D "
                "or use an input path that contains one of those names"
            )
        x_np = norm.transform(page, source=infer_source)
    else:
        x_np = page.astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).unsqueeze(0)
    system = system.to(primary)
    x = x.to(primary)
    f = int(system.vae.spatial_compression)
    pad_mode = str(getattr(args, "padding_mode", None) or "reflect")
    tile_size = int(args.tile_size)
    overlap = int(args.overlap)
    blend = str(getattr(args, "blend_mode", None) or "linear")

    if mode in {"tiled", "compare"} and tile_size % f != 0:
        raise SystemExit(
            f"--tile-size={tile_size} must be divisible by spatial_compression=f{f}"
        )
    h, w = int(x.shape[-2]), int(x.shape[-1])
    attn_n = attention_matrix_numel(h if h % f == 0 else h + (f - h % f) % f, w if w % f == 0 else w + (f - w % f) % f, f)
    info: Dict[str, Any] = {
        "mode": mode,
        "devices_requested": spec,
        "devices_actual": [str(d) for d in devices],
        "devices_info": describe_devices(devices),
        "world_size": len(devices),
        "primary_device": str(primary),
        "input_hw": [h, w],
        "spatial_compression": f,
        "weights": weights_kind,
        "posterior": "mean",
        "dtype": str(x.dtype),
        "normalizer": str(args.normalizer) if args.normalizer else None,
        "source": infer_source,
        "attention_matrix_numel_full_padded": int(attn_n),
        "config_sha256": config_semantic_hash(cfg),
    }
    if args.weights:
        info["weights_sha256"] = sha256_file(Path(args.weights))
    if args.normalizer:
        info["normalizer_sha256"] = sha256_file(Path(args.normalizer))
    if mode == "full" and len(devices) > 1:
        info["parallel_note"] = (
            "full inference is batch=1 with global GroupNorm and dense HW×HW attention; "
            "extra GPUs are idle. Use --inference-mode tiled for multi-GPU speedup."
        )
        print(info["parallel_note"], file=sys.stderr)
    if mode == "full" and attn_n > (96 * 96) ** 2:
        info["warning"] = (
            "full-image bottleneck attention is dense HW×HW; "
            f"matrix has {attn_n} elements. tiled mode is the training-size path."
        )

    def _sync() -> None:
        if primary.type == "cuda":
            torch.cuda.synchronize(primary)

    with torch.no_grad():
        if mode == "compare":
            _sync()
            pack = run_full_tiled_compare(
                system.vae,
                x,
                spatial_compression=f,
                tile_size=tile_size,
                overlap=overlap,
                padding_mode=pad_mode,
                blend_mode=blend,
                devices=list(devices),
                cfg_dump=resolved_dict(cfg),
            )
            if norm is not None:
                for k in ("full", "tiled", "target"):
                    pack[k] = norm.inverse(pack[k], source=infer_source)
                pack["residual_full"] = pack["full"] - pack["target"]
                pack["residual_tiled"] = pack["tiled"] - pack["target"]
                pack["diff_full_tiled"] = pack["full"] - pack["tiled"]
            out_dir = Path(args.output)
            save_compare_pack(pack, out_dir)
            tiled_aux = (pack.get("metrics") or {}).get("tiled_aux") or {}
            info.update(
                {
                    "output_dir": str(out_dir),
                    "metrics": pack["metrics"],
                    "parallel": tiled_aux.get("parallel"),
                }
            )
            print(json.dumps(info, indent=2, default=float))
            return 0
        if mode == "tiled":
            _sync()
            t0 = time.perf_counter()
            y, aux = run_tiled(
                system.vae,
                x,
                cfg_dump=resolved_dict(cfg),
                devices=list(devices),
                tile_size=tile_size,
                overlap=overlap,
                spatial_compression=f,
                padding_mode=pad_mode,
                blend_mode=blend,
                return_aux=True,
            )
            _sync()
            info["forward_wall_s"] = float(time.perf_counter() - t0)
            info["tiled_aux"] = {k: v for k, v in aux.items() if k != "weight"}
            info["parallel"] = aux.get("parallel")
        else:
            _sync()
            t0 = time.perf_counter()
            y, aux = reconstruct_full(
                system.vae, x, spatial_compression=f, padding_mode=pad_mode, return_aux=True
            )
            _sync()
            info["forward_wall_s"] = float(time.perf_counter() - t0)
            info["full_aux"] = aux
            info["parallel"] = {"mode": "none", "reason": "full image uses one device"}
    y_np = y.squeeze().detach().cpu().numpy().astype(np.float32)
    if norm is not None:
        y_np = norm.inverse(y_np, source=infer_source)
    out = Path(args.output)
    _save_float_image(out, y_np)
    info["output"] = str(out)
    info["shape"] = list(y_np.shape)
    if out.is_file():
        info["output_sha256"] = sha256_file(out)
    print(json.dumps(info, indent=2, default=float))
    return 0


def _load_infer_system(cfg, args):
    from microscopy_vae.systems.factory import build_hq_codec_system
    from microscopy_vae.inference.compare import load_infer_weights
    from microscopy_vae.inference.devices import parse_devices, primary_device

    spec = str(getattr(args, "devices", None) or "auto")
    devices = parse_devices(spec)
    primary = primary_device(devices)
    system = build_hq_codec_system(cfg)
    weights_kind = "uninitialized"
    if args.weights:
        weights_kind = load_infer_weights(
            Path(args.weights), system.vae, use_ema=not bool(getattr(args, "raw_weights", False))
        )
    system.eval()
    system = system.to(primary)
    return system, primary, weights_kind


def cmd_encode(args: argparse.Namespace) -> int:
    """Write posterior mean (internal unscaled z) plus padding metadata. No SD 0.18215."""
    cfg = _cfg(args)
    from microscopy_vae.data.readers import read_page
    from microscopy_vae.data.normalization import NormalizationState, Normalizer, guess_source_from_path
    from microscopy_vae.inference.tiling import encode_full
    from microscopy_vae.provenance.hashing import sha256_file
    import numpy as np

    if not args.input or not args.output:
        raise SystemExit("encode requires --input and --output")
    system, primary, weights_kind = _load_infer_system(cfg, args)
    page, _ = read_page(Path(args.input), int(args.page))
    infer_source = None
    if args.normalizer:
        norm = Normalizer(NormalizationState.load(Path(args.normalizer)))
        infer_source = getattr(args, "source", None) or guess_source_from_path(
            args.input, known=norm.known_sources()
        )
        if norm.is_per_source() and not infer_source:
            raise SystemExit("per-source normalizer: pass --source or a path containing the source name")
        x_np = norm.transform(page, source=infer_source)
    else:
        x_np = page.astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).unsqueeze(0).to(primary)
    f = int(system.vae.spatial_compression)
    z, post, aux = encode_full(
        system.vae,
        x,
        spatial_compression=f,
        padding_mode=str(getattr(args, "padding_mode", None) or "reflect"),
        sample_posterior=False,
    )
    z_np = z.squeeze(0).detach().cpu().numpy().astype(np.float32)
    out = Path(args.output)
    _save_float_image(out, z_np)
    meta = {
        **aux,
        "weights": weights_kind,
        "source": infer_source,
        "normalizer": str(args.normalizer) if args.normalizer else None,
        "mean_shape": [int(d) for d in post.mean.shape],
        "logvar_shape": [int(d) for d in post.logvar.shape],
        "note": (
            "latent is posterior mean in the model-internal unscaled domain. "
            "Refit per-channel center/scale on the train split of THIS architecture; "
            "do not reuse f4 latent stats on f8, and do not apply SD 0.18215."
        ),
    }
    meta_path = out.with_suffix(".json") if out.suffix.lower() != ".json" else Path(str(out) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    info = {
        "output": str(out),
        "metadata": str(meta_path),
        "latent_shape": list(z_np.shape),
        "spatial_compression": f,
        "output_sha256": sha256_file(out) if out.is_file() else None,
    }
    print(json.dumps(info, indent=2, default=float))
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.data.normalization import NormalizationState, Normalizer, guess_source_from_path
    from microscopy_vae.inference.tiling import decode_full
    from microscopy_vae.provenance.hashing import sha256_file
    import numpy as np

    if not args.input or not args.output:
        raise SystemExit("decode requires --input and --output")
    system, primary, weights_kind = _load_infer_system(cfg, args)
    z_np = np.load(str(args.input))
    if z_np.ndim == 3:
        z_np = z_np[None, ...]
    if z_np.ndim != 4:
        raise SystemExit(f"latent must be [C,H,W] or [B,C,H,W], got {z_np.shape}")
    z = torch.from_numpy(np.ascontiguousarray(z_np.astype(np.float32))).to(primary)
    pad_hw = (0, 0)
    output_hw = None
    meta_path = getattr(args, "meta", None)
    if not meta_path:
        cand = Path(args.input).with_suffix(".json")
        if cand.is_file():
            meta_path = str(cand)
    if meta_path:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        pad_hw = tuple(meta.get("pad_hw") or (0, 0))
        output_hw = tuple(meta["input_hw"]) if meta.get("input_hw") else None
        ckpt_f = meta.get("spatial_compression")
        if ckpt_f is not None and int(ckpt_f) != int(system.vae.spatial_compression):
            raise SystemExit(
                f"latent metadata f{ckpt_f} does not match model f{system.vae.spatial_compression}"
            )
    y = decode_full(system.vae, z, pad_hw=pad_hw, output_hw=output_hw)
    y_np = y.squeeze().detach().cpu().numpy().astype(np.float32)
    infer_source = getattr(args, "source", None) or (
        guess_source_from_path(args.output) if args.output else None
    )
    if args.normalizer:
        norm = Normalizer(NormalizationState.load(Path(args.normalizer)))
        if norm.is_per_source() and not infer_source:
            raise SystemExit("per-source normalizer: pass --source for decode inverse")
        y_np = norm.inverse(y_np, source=infer_source)
    out = Path(args.output)
    _save_float_image(out, y_np)
    print(
        json.dumps(
            {
                "output": str(out),
                "shape": list(y_np.shape),
                "weights": weights_kind,
                "spatial_compression": int(system.vae.spatial_compression),
                "output_sha256": sha256_file(out) if out.is_file() else None,
            },
            indent=2,
        )
    )
    return 0


def cmd_export_latent_spec(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.engine.trainer import Trainer
    from microscopy_vae.engine.checkpoint import CheckpointManager
    from microscopy_vae.models.factory import architecture_id
    from microscopy_vae.models.latent_spec import build_latent_spec, fit_latent_center_scale
    from microscopy_vae.provenance.hashing import sha256_file

    trainer = Trainer(cfg)
    if args.weights:
        CheckpointManager.load_exported_weights(Path(args.weights), trainer.system.vae)
    trainer.system.eval()
    means = []
    with torch.no_grad():
        for bi, batch in enumerate(trainer.train_loader):
            if bi >= int(args.max_batches):
                break
            x = batch.hq.to(trainer.device)
            post, _ = trainer.system.encode_hq(x, sample_posterior=False)
            means.append(post.mean.cpu())
    center, scale = fit_latent_center_scale(means)
    wsha = sha256_file(Path(args.weights)) if args.weights else "unspecified"
    spec = build_latent_spec(
        architecture_id=architecture_id(trainer.system.vae),
        weights_sha256=wsha,
        config_sha256=trainer.config_sha,
        normalizer_sha256=trainer.normalizer_sha,
        spatial_compression=trainer.system.vae.spatial_compression,
        latent_channels=trainer.system.vae.latent_channels,
        center=center,
        scale=scale,
        stats_sha256="fit_on_train_batches",
        transform_id=trainer.normalizer.state.transform_id,
        padding_mode=cfg.latent.padding_mode,
    )
    out = Path(args.out or "latent_spec.json")
    spec.to_json(out)
    print(f"Wrote {out}")
    print(json.dumps(spec.to_dict(), indent=2))
    return 0


def cmd_export_weights(args: argparse.Namespace) -> int:
    """Export inference-only weights (not a resume_exact package)."""
    cfg = _cfg(args)
    from microscopy_vae.engine.trainer import Trainer
    from microscopy_vae.engine.checkpoint import CheckpointManager
    from microscopy_vae.utils.atomic import atomic_write_bytes
    import io

    trainer = Trainer(cfg)
    if args.weights:
        CheckpointManager.load_exported_weights(Path(args.weights), trainer.system.vae)
    out = Path(args.out or "exported_vae.pt")
    buf = io.BytesIO()
    torch.save(
        {
            "format": "microvae-export-v1",
            "mode": "load_exported_weights",
            "model": trainer.system.vae.state_dict(),
            "config_sha256": trainer.config_sha,
            "note": "inference only; do not use as resume_exact",
        },
        buf,
    )
    atomic_write_bytes(out, buf.getvalue())
    print(f"Wrote {out}")
    return 0


def cmd_build_focus_sidecar(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if cfg.data.mode != "hq_pool" or not cfg.data.manifest_path:
        raise SystemExit("build-focus-sidecar requires data.mode=hq_pool and manifest_path")
    from microscopy_vae.data.focus_index import build_focus_sidecar
    from microscopy_vae.data.manifest import load_hq_manifest
    from microscopy_vae.data.pathmap import PathPrefixMap, apply_prefix_map_to_records

    recs = load_hq_manifest(Path(cfg.data.manifest_path), allow_splits=("train", "val"), refuse_test=True)
    if cfg.data.path_prefix_target:
        src_pref = cfg.data.path_prefix_source or "F:\\Dataset"
        pmap = PathPrefixMap(
            source_prefixes=(src_pref, src_pref.replace("\\", "/"), src_pref.replace("/", "\\")),
            target_root=cfg.data.path_prefix_target,
            require_exists=bool(cfg.data.path_require_exists),
        )
        recs = apply_prefix_map_to_records(recs, pmap)
    out = Path(args.out)
    build_focus_sidecar(recs, out, refuse_test=True)
    print(f"Wrote {out}")
    return 0


def _run_val_rows(cfg, weights: str, normalizer_path: Optional[str], unit_scale: Optional[float], max_batches: Optional[int], raw_weights: bool):
    import tempfile

    from microscopy_vae.engine.trainer import Trainer
    from microscopy_vae.engine.val_report import collect_val_pages, default_unit_scale
    from microscopy_vae.inference.compare import load_infer_weights

    if not normalizer_path:
        raise SystemExit(
            "eval-val-report / compare-runs require --normalizer pointing at THAT run's "
            "normalizer.json. Refusing to refit or to write into the training output_dir."
        )
    cfg = cfg.model_copy(deep=True)
    cfg.experiment.output_dir = tempfile.mkdtemp(prefix="microvae_eval_")
    cfg.experiment.allow_existing_output = True
    cfg.normalization.artifact_path = str(normalizer_path)
    trainer = Trainer(cfg)
    load_infer_weights(Path(weights), trainer.system.vae, use_ema=not raw_weights)
    trainer.system.eval()
    if trainer.system.perceptual is not None:
        trainer.system.perceptual.eval()
    norm = trainer.normalizer
    scale = float(unit_scale) if unit_scale is not None else default_unit_scale(norm.state)
    loader = trainer.val_loader
    if max_batches is not None:
        from itertools import islice

        # keep original loader but stop in collect via wrapper
        class _Limited:
            def __init__(self, inner, n):
                self.inner = inner
                self.n = n

            def __iter__(self):
                return islice(iter(self.inner), int(self.n))

        loader = _Limited(loader, max_batches)
    rows = collect_val_pages(
        trainer.system,
        loader,
        device=trainer.device,
        normalizer=norm,
        unit_scale=scale,
    )
    return rows, norm, scale, trainer


def cmd_eval_val_report(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.engine.val_report import (
        dump_page_jsonl,
        save_case_arrays,
        summarize_rows,
        write_report,
    )
    import numpy as np

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, norm, scale, trainer = _run_val_rows(
        cfg,
        args.weights,
        args.normalizer,
        args.unit_scale,
        args.max_batches,
        bool(args.raw_weights),
    )
    summary = summarize_rows(rows, cfg=cfg.evaluation)
    summary["unit_scale"] = scale
    summary["transform_id"] = norm.state.transform_id
    summary["normalizer_contract"] = norm.state.contract_dict()
    dump_page_jsonl(out_dir / "metrics_val_pages.jsonl", rows)
    write_report(out_dir / "val_severe_report.json", summary)
    want = {w["sample_id"]: i for i, w in enumerate(summary["worst_tail"])}
    if want:
        from microscopy_vae.engine.val_report import to_common_unit, to_raw_nonneg

        cases = out_dir / "worst_cases"
        for batch in trainer.val_loader:
            for i in range(batch.hq.shape[0]):
                sid = batch.sample_ids[i]
                if sid not in want:
                    continue
                x = batch.hq[i : i + 1].to(trainer.device)
                recon = trainer.system.reconstruct_hq(x)
                xi = batch.hq[i, 0].numpy()
                ri = recon[0, 0].detach().float().cpu().numpy()
                src = batch.sources[i]
                tgt = to_common_unit(
                    to_raw_nonneg(xi, trainer.normalizer, is_raw=False, source=src), scale
                )
                pred = to_common_unit(
                    to_raw_nonneg(ri, trainer.normalizer, is_raw=False, source=src), scale
                )
                lo = float(np.quantile(tgt, 0.005))
                hi = float(np.quantile(tgt, 0.995))
                save_case_arrays(
                    cases,
                    f"{want[sid]:02d}_{sid[:12]}",
                    target=tgt,
                    pred=pred,
                    display_lo=lo,
                    display_hi=hi,
                )
                del want[sid]
            if not want:
                break
    print(json.dumps({k: summary[k] for k in ("n_val_slices", "overall", "by_source", "by_morphology", "severe_rule")}, indent=2))
    print(f"Wrote {out_dir}")
    return 0


def cmd_compare_runs(args: argparse.Namespace) -> int:
    from microscopy_vae.config.loader import load_config
    from microscopy_vae.data.normalization import NormalizationState
    from microscopy_vae.engine.val_report import (
        compare_row_pairs,
        default_unit_scale,
        summarize_rows,
        write_report,
    )

    cfg_a = load_config(Path(args.config_a), _parse_override(args.override or []))
    cfg_b = load_config(Path(args.config_b), _parse_override(args.override or []))

    st_a = NormalizationState.load(Path(args.normalizer_a))
    st_b = NormalizationState.load(Path(args.normalizer_b))
    scale = args.unit_scale
    if scale is None:
        scale = max(default_unit_scale(st_a), default_unit_scale(st_b))
    rows_a, _, _, _ = _run_val_rows(cfg_a, args.weights_a, args.normalizer_a, scale, args.max_batches, False)
    rows_b, _, _, _ = _run_val_rows(cfg_b, args.weights_b, args.normalizer_b, scale, args.max_batches, False)
    sum_a = summarize_rows(rows_a, cfg=cfg_a.evaluation)
    sum_b = summarize_rows(rows_b, cfg=cfg_b.evaluation)
    cmp_ = compare_row_pairs(rows_a, rows_b, label_a=args.label_a, label_b=args.label_b)
    out = {
        "unit_scale": scale,
        "fairness": (
            "Each model inverted with its own train normalizer, then max(raw, floor). "
            "Metrics use the same scale = max(range_a, range_b) unless --unit-scale is set. "
            "Do not compare MAE in the two normalized spaces."
        ),
        args.label_a: {k: sum_a[k] for k in ("n_val_slices", "overall", "by_source", "by_morphology")},
        args.label_b: {k: sum_b[k] for k in ("n_val_slices", "overall", "by_source", "by_morphology")},
        "delta": cmp_["mean_delta_b_minus_a"],
        "n_common_slices": cmp_["n_common_slices"],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(out_dir / "compare_runs.json", {**out, "per_slice": cmp_["per_slice"]})
    print(json.dumps(out, indent=2, default=float))
    return 0


def cmd_analyze_percentiles(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from microscopy_vae.data.normalization import summarize_percentile_candidates
    from microscopy_vae.engine.trainer import Trainer

    trainer = Trainer(cfg)
    cand = [float(x) for x in str(args.candidates).split(",") if x.strip()]
    # Re-read the same train pages used for the fit if possible.
    imgs = getattr(trainer, "_pages_or_records", None)
    # Trainer does not keep train_imgs; refit sampling:
    if cfg.data.mode != "hq_pool":
        raise SystemExit("analyze-normalizer-percentiles requires data.mode=hq_pool and reachable train files")
    recs = [r for r in trainer._pages_or_records if r.split == "train"]
    reachable = [r for r in recs if Path(r.hq_path).is_file()]
    if not reachable:
        raise SystemExit("no reachable train HQ files (check path_prefix_target); refuse to use test")
    imgs, sources = trainer._sample_train_images_for_norm(reachable)
    payload = summarize_percentile_candidates(
        imgs,
        sources,
        candidates=cand,
        low_percentile=float(cfg.normalization.low_percentile),
        raw_floor_enabled=True,
        raw_floor_value=float(cfg.normalization.raw_floor_value),
        max_pixels_per_page=int(cfg.normalization.max_pixels_per_page),
        seed=int(cfg.experiment.seed),
    )
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("note", "source_balanced_globals")}, indent=2, default=float)[:4000])
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="microscopy-vae", description="Scratch microscopy VAE CLI")
    sub = p.add_subparsers(dest="command", required=True)

    for name in [
        "validate-config",
        "smoke-test",
        "overfit-small",
        "train",
        "profile",
        "fit-normalizer",
        "inspect-data",
        "evaluate",
        "infer",
        "encode",
        "decode",
        "export-latent-spec",
        "export-weights",
        "freeze-candidate",
        "launch-seeds",
        "aggregate-seeds",
        "build-focus-sidecar",
        "eval-val-report",
        "compare-runs",
        "analyze-normalizer-percentiles",
    ]:
        sp = sub.add_parser(name)
        _add_common(sp)
        if name == "fit-normalizer":
            sp.add_argument("--out", type=str, default="normalizer.json")
        if name == "evaluate":
            sp.add_argument("--weights", type=str, default=None)
            sp.add_argument("--full", action="store_true")
        if name == "infer":
            sp.add_argument("--input", type=str, default=None)
            sp.add_argument("--output", type=str, default=None)
            sp.add_argument("--weights", type=str, default=None)
            sp.add_argument("--normalizer", type=str, default=None)
            sp.add_argument("--page", type=int, default=0)
            sp.add_argument(
                "--inference-mode",
                type=str,
                default="full",
                choices=["full", "tiled", "compare"],
                help="full | tiled | compare (same image, same weights, same normalizer)",
            )
            sp.add_argument("--tiled", action="store_true", help="alias for --inference-mode tiled")
            sp.add_argument("--tile-size", type=int, default=256)
            sp.add_argument("--overlap", type=int, default=32)
            sp.add_argument("--blend-mode", type=str, default="linear", choices=["linear", "hann"])
            sp.add_argument("--padding-mode", type=str, default="reflect")
            sp.add_argument(
                "--raw-weights",
                action="store_true",
                help="do not copy EMA even if the checkpoint extra contains it",
            )
            sp.add_argument(
                "--devices",
                type=str,
                default="auto",
                help="auto | cpu | cuda | cuda:0 | cuda:0,cuda:2 (logical ids after CUDA_VISIBLE_DEVICES)",
            )
            sp.add_argument(
                "--source",
                type=str,
                default=None,
                help="BioTISR | DeepInsight_2D | DeepInsight_3D (required for per-source V4 normalizer if path does not contain it)",
            )
        if name in {"encode", "decode"}:
            sp.add_argument("--input", type=str, default=None)
            sp.add_argument("--output", type=str, default=None)
            sp.add_argument("--weights", type=str, default=None)
            sp.add_argument("--normalizer", type=str, default=None)
            sp.add_argument("--page", type=int, default=0)
            sp.add_argument("--padding-mode", type=str, default="reflect")
            sp.add_argument("--raw-weights", action="store_true")
            sp.add_argument("--devices", type=str, default="auto")
            sp.add_argument(
                "--source",
                type=str,
                default=None,
                help="BioTISR | DeepInsight_2D | DeepInsight_3D",
            )
            if name == "decode":
                sp.add_argument(
                    "--meta",
                    type=str,
                    default=None,
                    help="JSON from encode (default: sibling .json of --input)",
                )
        if name == "export-latent-spec":
            sp.add_argument("--weights", type=str, default=None)
            sp.add_argument("--out", type=str, default="latent_spec.json")
            sp.add_argument("--max-batches", type=int, default=8)
        if name == "export-weights":
            sp.add_argument("--weights", type=str, default=None)
            sp.add_argument("--out", type=str, default="exported_vae.pt")
        if name == "build-focus-sidecar":
            sp.add_argument("--out", type=str, default="focus_sidecar.jsonl")
        if name == "eval-val-report":
            sp.add_argument("--weights", type=str, required=True)
            sp.add_argument(
                "--normalizer",
                type=str,
                required=True,
                help="that training run's normalizer.json (required; eval must not refit)",
            )
            sp.add_argument("--out-dir", type=str, required=True)
            sp.add_argument(
                "--unit-scale",
                type=float,
                default=None,
                help="raw_nonneg / scale; default = brightest per-source span (or global high-low)",
            )
            sp.add_argument("--max-batches", type=int, default=None)
            sp.add_argument("--raw-weights", action="store_true")
        if name == "compare-runs":
            sp.add_argument("--config-a", type=str, required=True)
            sp.add_argument("--weights-a", type=str, required=True)
            sp.add_argument("--normalizer-a", type=str, required=True)
            sp.add_argument("--label-a", type=str, default="v2_2")
            sp.add_argument("--config-b", type=str, required=True)
            sp.add_argument("--weights-b", type=str, required=True)
            sp.add_argument("--normalizer-b", type=str, required=True)
            sp.add_argument("--label-b", type=str, default="v4")
            sp.add_argument("--out-dir", type=str, required=True)
            sp.add_argument("--unit-scale", type=float, default=None)
            sp.add_argument("--max-batches", type=int, default=None)
        if name == "analyze-normalizer-percentiles":
            sp.add_argument("--out", type=str, default="percentile_candidates.json")
            sp.add_argument("--candidates", type=str, default="99.9,99.95,99.99,99.995")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "validate-config": cmd_validate_config,
        "smoke-test": cmd_smoke_test,
        "overfit-small": cmd_overfit_small,
        "train": cmd_train,
        "fit-normalizer": cmd_fit_normalizer,
        "profile": cmd_profile,
        "inspect-data": cmd_inspect_data,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
        "encode": cmd_encode,
        "decode": cmd_decode,
        "export-latent-spec": cmd_export_latent_spec,
        "export-weights": cmd_export_weights,
        "build-focus-sidecar": cmd_build_focus_sidecar,
        "eval-val-report": cmd_eval_val_report,
        "compare-runs": cmd_compare_runs,
        "analyze-normalizer-percentiles": cmd_analyze_percentiles,
    }
    if args.command not in dispatch:
        print(
            f"Command {args.command!r} reserved for later orchestration "
            "(freeze-candidate / launch-seeds / aggregate-seeds).",
            file=sys.stderr,
        )
        sys.exit(3)
    code = dispatch[args.command](args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
