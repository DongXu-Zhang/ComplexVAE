"""Distributed val sharding: no padding, same aggregation as sequential."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from microscopy_vae.config.loader import config_semantic_hash, load_config
from microscopy_vae.data.hq_dataset import IndexedView, SyntheticHQDataset, collate_hq
from microscopy_vae.data.normalization import fit_robust_normalizer, Normalizer
from microscopy_vae.data.synthetic import build_synthetic_hq_pool
from microscopy_vae.engine.distributed import strided_indices
from microscopy_vae.engine.evaluator import aggregate_eval_pages, evaluate_hq_loader
from microscopy_vae.losses.composer import HQCodecLossComposer
from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.systems.hq_codec import HQCodecSystem
from microscopy_vae.tasks.hq_codec import HQCodecTask
from torch.utils.data import DataLoader


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def test_strided_indices_no_pad_no_dup():
    n, w = 10, 4
    parts = [strided_indices(n, r, w) for r in range(w)]
    flat = [i for p in parts for i in p]
    assert sorted(flat) == list(range(n))
    assert len(flat) == len(set(flat)) == n
    assert strided_indices(3, 3, 4) == []
    assert strided_indices(0, 0, 2) == []


def test_config_hash_ignores_val_batch_size():
    a = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v6.yaml")
    b = load_config(
        _repo() / "configs/experiment/s1_hq_f8z4_v6.yaml",
        overrides={"evaluation": {"batch_size": 16}},
    )
    assert a.evaluation.batch_size is None
    assert b.evaluation.batch_size == 16
    assert config_semantic_hash(a) == config_semantic_hash(b)


def test_indexed_view_sets_dataset_index():
    pages = build_synthetic_hq_pool(n_groups=4, pages_per_group=2, size=32, seed=0)
    st = fit_robust_normalizer([p.image for p in pages if p.split == "train"], method="identity")
    ds = SyntheticHQDataset(pages, split="val", crop_size=32, normalizer=Normalizer(st), fixed_crops=True, seed=1)
    view = IndexedView(ds, [1, 0])
    a = view[0]
    b = view[1]
    assert a["metadata"]["dataset_index"] == 1
    assert b["metadata"]["dataset_index"] == 0


def _tiny_system():
    vae = ModelFactory.create_fresh(
        latent_channels=4,
        encoder_block_out_channels=(32, 64),
        decoder_block_out_channels=(32, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=False,
    )
    task = HQCodecTask(
        vae, HQCodecLossComposer(w_ms_ssim=0, w_grad=0, w_flux=0, free_nats=0), sample_posterior=False
    )
    return HQCodecSystem(vae, task)


def _val_loader(pages, indices=None, batch_size=2):
    st = fit_robust_normalizer([p.image for p in pages if p.split == "train"], method="identity")
    ds = SyntheticHQDataset(
        pages, split="val", crop_size=32, normalizer=Normalizer(st), fixed_crops=True, seed=1
    )
    if indices is None:
        indices = list(range(len(ds)))
    view = IndexedView(ds, indices)
    return DataLoader(view, batch_size=batch_size, shuffle=False, collate_fn=collate_hq), len(ds)


def test_sharded_eval_matches_sequential_cpu():
    pages = build_synthetic_hq_pool(n_groups=8, pages_per_group=2, size=32, seed=0)
    system = _tiny_system()
    full_loader, n = _val_loader(pages, batch_size=3)
    full = evaluate_hq_loader(
        system, full_loader, device=torch.device("cpu"), bootstrap_n=20, bootstrap_seed=0, expected_n=n
    )
    recs = []
    for r in range(3):
        idx = strided_indices(n, r, 3)
        loader, _ = _val_loader(pages, indices=idx, batch_size=2)
        part = evaluate_hq_loader(
            system, loader, device=torch.device("cpu"), bootstrap_n=20, bootstrap_seed=0
        )
        recs.extend(
            [
                {
                    "dataset_index": i,
                    "sample_id": sid,
                    "group_id": g,
                    "source": s,
                    "metrics": m,
                }
                for i, sid, g, s, m in zip(
                    # dataset_index is inside metadata during compute; page_metrics order is local
                    strided_indices(n, r, 3),
                    part["sample_ids"],
                    part["group_ids"],
                    part["sources"],
                    part["page_metrics"],
                )
            ]
        )
    recs.sort(key=lambda x: x["dataset_index"])
    merged = aggregate_eval_pages(
        [r["metrics"] for r in recs],
        [r["group_id"] for r in recs],
        [r["source"] for r in recs],
        bootstrap_n=20,
        bootstrap_seed=0,
        extended_metrics=False,
        sample_ids=[r["sample_id"] for r in recs],
    )
    assert merged["n_pages"] == full["n_pages"] == n
    assert set(merged["sample_ids"]) == set(full["sample_ids"])
    for k in ("mae", "mse", "psnr", "nmse", "ssim_local"):
        assert merged["group_macro"][k] == pytest.approx(full["group_macro"][k], rel=1e-6, abs=1e-7)
    assert merged["psnr_bootstrap"]["mean"] == pytest.approx(full["psnr_bootstrap"]["mean"], rel=1e-6, abs=1e-7)


def test_train_mode_restored_after_val(tmp_path):
    from microscopy_vae.engine.trainer import Trainer

    cfg = load_config(
        _repo() / "configs/experiment/smoke_v5.yaml",
        overrides={
            "experiment": {"output_dir": str(tmp_path / "tv"), "allow_existing_output": True},
            "training": {
                "max_steps": 2,
                "microbatch_size": 2,
                "grad_accum": 1,
                "num_workers": 0,
                "val_every_steps": 1,
                "log_every_steps": 1,
                "ema_decay": 0.9,
            },
            "precision": {"amp_dtype": "fp32"},
            "loss": {"perceptual": {"enabled": False}, "adversarial": {"enabled": False}},
            "checkpoint": {"save_every_steps": 100},
        },
    )
    tr = Trainer(cfg)
    try:
        tr.train(max_steps=2)
        assert tr.system.vae.training is True
        val_path = tmp_path / "tv" / "metrics_val.jsonl"
        assert val_path.is_file()
        lines = val_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
    finally:
        tr.close()


def test_trainer_val_batch_independent_of_train_microbatch(tmp_path):
    from microscopy_vae.engine.trainer import Trainer

    cfg = load_config(
        _repo() / "configs/experiment/smoke_v5.yaml",
        overrides={
            "experiment": {"output_dir": str(tmp_path / "vb"), "allow_existing_output": True},
            "training": {"microbatch_size": 2, "grad_accum": 1, "num_workers": 0, "max_steps": 1},
            "evaluation": {"batch_size": 8},
        },
    )
    tr = Trainer(cfg)
    try:
        assert tr.per_device_batch == 2
        assert tr.val_batch_size == 8
        assert tr.val_n == len(tr.val_set)
    finally:
        tr.close()


def test_single_forward_eval_finite():
    pages = build_synthetic_hq_pool(n_groups=4, pages_per_group=2, size=32, seed=1)
    system = _tiny_system()
    loader, n = _val_loader(pages, batch_size=4)
    out = evaluate_hq_loader(system, loader, device=torch.device("cpu"), bootstrap_n=8, expected_n=n)
    assert out["n_pages"] == n
    assert np.isfinite(out["group_macro"]["mae"])
