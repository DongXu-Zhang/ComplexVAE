#!/usr/bin/env python3
"""Server benchmark: 1 GPU vs 2 vs all visible, tiled inference only.

Does not change reconstruction. Warmup + CUDA sync. Example:

  python tools/bench_infer_devices.py --config configs/experiment/s1_hq_f4z4_v2_2.yaml \\
      --weights CKPT --normalizer NORM --input IMG --tile-size 256 --overlap 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--normalizer", type=Path, default=None)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--page", type=int, default=0)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--source", type=str, default=None)
    args = p.parse_args()

    from microscopy_vae.config.loader import load_config, resolved_dict
    from microscopy_vae.data.normalization import NormalizationState, Normalizer, guess_source_from_path
    from microscopy_vae.data.readers import read_page
    from microscopy_vae.inference.compare import load_infer_weights
    from microscopy_vae.inference.devices import parse_devices
    from microscopy_vae.inference.parallel import run_tiled
    from microscopy_vae.systems.factory import build_hq_codec_system

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark")
    cfg = load_config(args.config)
    all_dev = parse_devices("auto")
    page, _ = read_page(args.input, args.page)
    if args.normalizer:
        nrm = Normalizer(NormalizationState.load(args.normalizer))
        src = args.source or guess_source_from_path(args.input, known=nrm.known_sources())
        x_np = nrm.transform(page, source=src)
    else:
        x_np = page.astype(np.float32)
    x_cpu = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).unsqueeze(0)

    plans = [
        ("1gpu", [all_dev[0]]),
        ("2gpu", all_dev[:2] if len(all_dev) >= 2 else None),
        ("all", all_dev if len(all_dev) >= 1 else None),
    ]
    sys0 = build_hq_codec_system(cfg)
    if args.weights:
        load_infer_weights(args.weights, sys0.vae, use_ema=True)
    sys0.eval()
    dump = resolved_dict(cfg)
    sc = sys0.vae.spatial_compression
    report = []
    ref = None
    for name, devs in plans:
        if not devs:
            continue
        model = sys0.to(devs[0])
        x = x_cpu.to(devs[0])
        # warmup
        _ = run_tiled(
            model.vae if hasattr(model, "vae") else model,
            x,
            cfg_dump=dump,
            devices=devs,
            tile_size=args.tile_size,
            overlap=args.overlap,
            spatial_compression=sc,
        )
        _sync(devs[0])
        times = []
        y = None
        for _i in range(args.repeat):
            _sync(devs[0])
            t0 = time.perf_counter()
            y = run_tiled(
                model.vae,
                x,
                cfg_dump=dump,
                devices=devs,
                tile_size=args.tile_size,
                overlap=args.overlap,
                spatial_compression=sc,
            )
            _sync(devs[0])
            times.append(time.perf_counter() - t0)
        yc = y.detach().cpu()
        row = {
            "plan": name,
            "devices": [str(d) for d in devs],
            "n": len(devs),
            "wall_s_mean": float(sum(times) / len(times)),
            "wall_s_min": float(min(times)),
            "shape": list(yc.shape),
        }
        if ref is None:
            ref = yc
            row["max_abs_vs_1gpu"] = 0.0
        else:
            row["max_abs_vs_1gpu"] = float((ref - yc).abs().max())
            row["mae_vs_1gpu"] = float((ref - yc).abs().mean())
        report.append(row)
        print(json.dumps(row, indent=2))
    print(json.dumps({"summary": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
