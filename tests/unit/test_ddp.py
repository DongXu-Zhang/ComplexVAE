from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from microscopy_vae.config.loader import load_config
from microscopy_vae.data.samplers import DistributedHierarchicalSampler, HierarchicalIndexSampler
from microscopy_vae.engine.distributed import (
    DistInfo,
    all_reduce_max_flag,
    assert_resume_world_size,
    raise_if_any_rank_failed,
    reduce_mean_map,
    resolve_per_device_batch,
    strip_module_prefix,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def test_resolve_per_device_batch_keeps_global_8():
    assert resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=1, scale_global_batch=False
    ) == (4, 2, 8)
    assert resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=2, scale_global_batch=False
    ) == (2, 2, 8)
    assert resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=4, scale_global_batch=False
    ) == (1, 2, 8)
    assert resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=8, scale_global_batch=False
    ) == (1, 1, 8)
    with pytest.raises(ValueError, match="not divisible"):
        resolve_per_device_batch(
            yaml_microbatch=4, yaml_accum=2, world_size=3, scale_global_batch=False
        )
    assert resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=2, scale_global_batch=True
    ) == (4, 2, 16)


def test_ddp_scale_lr_requires_scale_global_batch():
    with pytest.raises(ValueError, match="ddp_scale_lr"):
        load_config(
            _repo() / "configs/experiment/smoke_v5.yaml",
            overrides={"training": {"ddp_scale_lr": True, "ddp_scale_global_batch": False}},
        )


def test_assert_resume_world_size():
    assert_resume_world_size(None, 2)
    assert_resume_world_size(2, 2)
    assert_resume_world_size(1, 1)
    with pytest.raises(ValueError, match="nproc_per_node"):
        assert_resume_world_size(1, 2)


def test_ddp_timeout_env_default_is_six_hours():
    import inspect

    from microscopy_vae.engine import distributed as dmod

    src = inspect.getsource(dmod.init_distributed)
    assert 'MICROVAE_DDP_TIMEOUT_MIN' in src
    assert '"360"' in src


def test_single_process_collectives_are_local():
    info = DistInfo(device=torch.device("cpu"))
    assert all_reduce_max_flag(True, info, info.device) is True
    assert all_reduce_max_flag(False, info, info.device) is False
    raise_if_any_rank_failed(True, "unused", info)
    with pytest.raises(RuntimeError, match="boom"):
        raise_if_any_rank_failed(False, "boom", info)
    with pytest.raises(RuntimeError):
        raise_if_any_rank_failed(False, "", info)
    out = reduce_mean_map({"a": 2.0, "b": 4.0}, info, info.device)
    assert out == {"a": 2.0, "b": 4.0}


def test_cuda_dataloader_uses_spawn_after_init():
    import inspect

    from microscopy_vae.engine.trainer import Trainer

    src = inspect.getsource(Trainer._dataloader_kwargs)
    assert 'multiprocessing_context' in src
    assert "spawn" in src


def test_synthetic_dataset_is_picklable_for_spawn_workers():
    import pickle

    from microscopy_vae.data.hq_dataset import SyntheticHQDataset
    from microscopy_vae.data.normalization import NormalizationState, Normalizer
    from microscopy_vae.data.synthetic import build_synthetic_hq_pool

    pages = build_synthetic_hq_pool(n_groups=2, pages_per_group=2, size=64, seed=0)
    st = NormalizationState(
        schema_version="microvae-normalizer-v2",
        method="identity",
        fit_split="train",
        low=0.0,
        high=1.0,
        clip=False,
        role="hq",
        n_groups=2,
        config_sha256="",
        manifest_sha256="",
        transform_id="identity_v1",
        scale_mode="global",
    )
    ds = SyntheticHQDataset(pages, split="train", crop_size=64, normalizer=Normalizer(st), seed=0)
    blob = pickle.dumps(ds)
    got = pickle.loads(blob)
    assert len(got) == len(ds)
    sample = got[0]
    assert sample["hq"].shape[-2:] == (64, 64)


def test_normalizer_state_roundtrip_for_ddp_broadcast():
    from microscopy_vae.data.normalization import NormalizationState

    st = NormalizationState(
        schema_version="microvae-normalizer-v2",
        method="robust_linear",
        fit_split="train",
        low=0.0,
        high=1.0,
        clip=False,
        role="hq",
        n_groups=1,
        config_sha256="",
        manifest_sha256="",
        transform_id="t",
        scale_mode="per_source",
        raw_floor_enabled=True,
        high_percentile=99.99,
        low_percentile=0.0,
        per_source_scales={"BioTISR": {"low": 0.0, "high": 10.0}},
        per_source_thresholds={"BioTISR": {"amp_low_structure_range": 0.08, "crop_min_robust_range": 0.03}},
        threshold_version="microvae-thresholds-v1",
    )
    got = NormalizationState.from_dict(st.to_dict())
    assert got.scale_mode == "per_source"
    assert got.per_source_scales["BioTISR"]["high"] == 10.0
    assert got.per_source_thresholds["BioTISR"]["amp_low_structure_range"] == 0.08


def test_strip_module_prefix():
    sd = {"module.encoder.weight": torch.ones(1), "decoder.bias": torch.zeros(1)}
    out = strip_module_prefix(sd)
    assert "encoder.weight" in out
    assert "decoder.bias" in out
    assert "module.encoder.weight" not in out


def test_distributed_sampler_shards_shared_stream():
    meta = []
    for i, src in enumerate(["SOURCE_A"] * 20 + ["SOURCE_B"] * 20):
        meta.append({"source": src, "group_id": f"g{i % 8}", "index": i})
    a = DistributedHierarchicalSampler(meta, rank=0, world_size=2, seed=0, epoch_length=40)
    b = DistributedHierarchicalSampler(meta, rank=1, world_size=2, seed=0, epoch_length=40)
    ia = list(iter(a))
    ib = list(iter(b))
    assert len(ia) == len(ib) == 40
    # Same global stream, different stride: not identical lists.
    assert ia != ib
    # Combined source draws equal a full single-process stream of 80.
    single = HierarchicalIndexSampler(meta, seed=0, epoch_length=80)
    list(iter(single))
    comb = {}
    for s in a.inner.sources:
        comb[s] = a.inner.source_draws[s] + b.inner.source_draws[s]
    # Each inner consumed 80 draws (40 * world_size). Combined counts match 2*80 if
    # both ran independently... wait each inner is independent copy with same seed
    # so each consumed 80 global samples (40 steps * 2 ranks). source_draws on each
    # inner equals the FULL 80-stream, not half. realized freq should match single.
    assert a.inner.source_draws == single.source_draws
    assert b.inner.source_draws == single.source_draws


def _ddp_worker(rank: int, world: int, port: int, tmp: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import json as _json
    from microscopy_vae.config.loader import load_config
    from microscopy_vae.engine.trainer import Trainer

    cfg = load_config(
        _repo() / "configs/experiment/smoke_v5.yaml",
        overrides={
            "experiment": {
                "output_dir": tmp,
                "allow_existing_output": True,
                "name": "smoke_ddp",
            },
            "training": {
                "max_steps": 2,
                "microbatch_size": 2,
                "grad_accum": 1,
                "num_workers": 0,
                "val_every_steps": 2,
                "log_every_steps": 1,
                "ema_decay": 0.9,
            },
            "checkpoint": {"save_every_steps": 2, "keep_last": 2},
            "precision": {"amp_dtype": "fp32"},
            "loss": {
                "perceptual": {"enabled": False},
                "adversarial": {"enabled": True, "start_step": 0, "ramp_steps": 0},
                "influence": {"grad_every_steps": 1},
            },
        },
    )
    trainer = Trainer(cfg)
    assert trainer.dist.enabled
    assert trainer.dist.world_size == world
    assert trainer.effective_global_batch == 2
    assert trainer.per_device_batch == 1
    result = trainer.train(max_steps=2)

    def _sig(sd):
        keys = sorted(sd)
        return {k: float(sd[k].detach().float().reshape(-1)[0]) for k in keys[:8]}

    payload = {
        "rank": rank,
        "step": result["final_step"],
        "loss": result["final_loss"],
        "vae": _sig(trainer.system.vae.state_dict()),
        "disc": _sig(trainer.discriminator.state_dict()),
        "wrote_metrics": (Path(tmp) / "metrics_train.jsonl").is_file(),
    }
    (Path(tmp) / f"rank{rank}.json").write_text(_json.dumps(payload) + "\n", encoding="utf-8")
    # Do not destroy the process group here: the peer may still be writing.


def test_gloo_two_process_train_syncs_vae_and_disc(tmp_path):
    import json
    import time
    import torch.multiprocessing as mp

    port = 29527 + int(os.getpid() % 1000)
    run = tmp_path / "ddp"
    run.mkdir()
    ctx = mp.get_context("spawn")
    procs = []
    for r in range(2):
        p = ctx.Process(target=_ddp_worker, args=(r, 2, port, str(run)))
        p.start()
        procs.append(p)
    deadline = time.time() + 300
    while time.time() < deadline:
        if all((run / f"rank{r}.json").is_file() for r in range(2)):
            break
        time.sleep(1)
    for p in procs:
        p.join(timeout=30)
    out = []
    for r in range(2):
        pth = run / f"rank{r}.json"
        assert pth.is_file(), f"missing {pth}; exitcodes={[p.exitcode for p in procs]}"
        out.append(json.loads(pth.read_text(encoding="utf-8")))
    by_rank = {o["rank"]: o for o in out}
    assert by_rank[0]["step"] == by_rank[1]["step"] == 2
    for k, a in by_rank[0]["vae"].items():
        b = by_rank[1]["vae"][k]
        assert a == pytest.approx(b, abs=1e-4, rel=1e-3), k
    for k, a in by_rank[0]["disc"].items():
        b = by_rank[1]["disc"][k]
        assert a == pytest.approx(b, abs=1e-4, rel=1e-3), k
    # File exists (rank0 wrote). We cannot see rank1 skip from this process; rank0 must have written.
    assert by_rank[0]["wrote_metrics"] is True
    lines = (tmp_path / "ddp" / "metrics_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    # log_every_steps=1, 2 optimizer steps → 2 lines if only rank0 writes
    assert len(lines) == 2


def test_single_process_smoke_unchanged(tmp_path):
    from microscopy_vae.engine.trainer import Trainer

    cfg = load_config(
        _repo() / "configs/experiment/smoke_v5.yaml",
        overrides={
            "experiment": {"output_dir": str(tmp_path / "single"), "allow_existing_output": True},
            "training": {"max_steps": 2, "val_every_steps": 100, "num_workers": 0},
        },
    )
    trainer = Trainer(cfg)
    assert trainer.dist.enabled is False
    assert trainer.effective_global_batch == cfg.training.microbatch_size * cfg.training.grad_accum
    result = trainer.train(max_steps=2)
    trainer.close()
    assert int(result["final_step"]) == 2
    assert (tmp_path / "single" / "normalizer.json").is_file()
