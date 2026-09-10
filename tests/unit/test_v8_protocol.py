"""V8: drop raw floor only. per_source still pins low=0 so y=x/high_s."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from microscopy_vae.config.loader import load_config
from microscopy_vae.data.normalization import (
    Normalizer,
    affine_formula,
    assert_artifact_matches_config,
    fit_robust_normalizer,
)
from microscopy_vae.data.threshold_calibration import THRESHOLD_VERSION


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v8_yaml_is_v6_minus_floor_only():
    v6 = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v6.yaml")
    v8 = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v8.yaml")
    assert v6.normalization.raw_floor_enabled is True
    assert v8.normalization.raw_floor_enabled is False
    assert v8.normalization.clip is False
    assert v8.normalization.scale_mode == "per_source"
    assert v8.normalization.high_percentile == 99.99
    assert v8.normalization.low_percentile == 0.0
    assert v8.normalization.allow_legacy_artifact is False
    assert v8.normalization.calibrate_thresholds is True
    assert v8.model.output_activation == "linear"
    assert v8.experiment.name == "s1_hq_f8z4_v8"
    assert v8.experiment.output_dir == "runs/s1_hq_f8z4_v8"
    assert v8.experiment.output_dir != v6.experiment.output_dir
    assert v8.loss.w_grad == 0.0
    assert v8.loss.w_hf == 0.0
    assert v8.loss.w_flux == 0.0
    assert v8.loss.w_dark_fp == 0.0
    assert v8.loss.adversarial.enabled is False
    dump6 = v6.model_dump()
    dump8 = v8.model_dump()
    dump6["experiment"] = dump8["experiment"]
    dump6["normalization"]["raw_floor_enabled"] = False
    assert dump6 == dump8


def test_per_source_floor_off_is_x_over_high_not_shifted_p0():
    raw = np.array([[-1000.0, 0.0, 2000.0, 23192.0]], dtype=np.float32)
    st = fit_robust_normalizer(
        [raw],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=False,
        sources=["BioTISR"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=8,
    )
    n = Normalizer(st)
    sc = st.per_source_scales["BioTISR"]
    assert sc["low"] == pytest.approx(0.0)
    hi = float(sc["high"])
    assert hi == pytest.approx(23192.0)
    y = n.transform(raw, source="BioTISR")
    assert float(y[0, 0]) == pytest.approx(-1000.0 / hi)
    assert float(y[0, 1]) == pytest.approx(0.0)
    assert float(y[0, 2]) == pytest.approx(2000.0 / hi)
    assert float(y.min()) < 0.0
    rec = n.inverse(y, source="BioTISR")
    assert rec == pytest.approx(raw, abs=1e-3)
    assert affine_formula(st) == "y = x/high"
    assert st.contract_dict()["floor_before_normalize"] is False
    assert st.contract_dict()["affine"] == "y = x/high"
    y_t = n.transform_torch(torch.from_numpy(raw), source="BioTISR").numpy()
    assert np.allclose(y, y_t, atol=1e-5)


def test_global_floor_off_still_uses_percentile_low():
    mixed = np.concatenate(
        [np.linspace(-50.0, -1.0, 128), np.linspace(1.0, 10.0, 128)]
    ).astype(np.float32).reshape(16, 16)
    st = fit_robust_normalizer(
        [mixed],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=False,
        scale_mode="global",
        clip=False,
        max_pixels_per_page=256,
    )
    assert st.low < 0.0
    n = Normalizer(st)
    y0 = float(n.transform(np.array([[0.0]], dtype=np.float32))[0, 0])
    assert y0 > 0.0
    assert affine_formula(st) == "y = (x-low)/(high-low)"


def test_v6_floor_still_collapses_negatives_to_zero():
    raw = np.array([[-1000.0, 0.0, 2000.0]], dtype=np.float32)
    st = fit_robust_normalizer(
        [raw],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=True,
        raw_floor_value=0.0,
        sources=["BioTISR"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=8,
    )
    n = Normalizer(st)
    y = n.transform(raw, source="BioTISR")
    assert float(y[0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(y[0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert affine_formula(st) == "y = max(x,0)/high"


def test_per_source_highs_stay_independent_without_floor():
    bio = np.linspace(-5.0, 10.0, 256, dtype=np.float32).reshape(16, 16)
    di2 = np.linspace(-5.0, 40.0, 256, dtype=np.float32).reshape(16, 16)
    st = fit_robust_normalizer(
        [bio, di2],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=False,
        sources=["BioTISR", "DeepInsight_2D"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=256,
    )
    n = Normalizer(st)
    assert all(abs(float(sc["low"])) < 1e-9 for sc in st.per_source_scales.values())
    y_bio = float(n.transform(np.array([[10.0]], dtype=np.float32), source="BioTISR")[0, 0])
    y_di2 = float(n.transform(np.array([[10.0]], dtype=np.float32), source="DeepInsight_2D")[0, 0])
    assert y_bio > y_di2
    y_neg = n.transform(np.array([[-5.0]], dtype=np.float32), source="BioTISR")
    y_zero = n.transform(np.array([[0.0]], dtype=np.float32), source="BioTISR")
    assert float(y_neg[0, 0]) < 0.0
    assert float(y_zero[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_v8_refuses_floored_v6_artifact():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v8.yaml")
    old = fit_robust_normalizer(
        [np.linspace(-5.0, 20.0, 64, dtype=np.float32).reshape(8, 8)],
        method="robust_linear",
        raw_floor_enabled=True,
        high_percentile=99.99,
        low_percentile=0.0,
        sources=["BioTISR"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=64,
    )
    old.per_source_thresholds = {
        "BioTISR": {
            "structure_support_floor": 0.005,
            "amp_low_structure_range": 0.02,
            "crop_min_robust_range": 0.02,
        }
    }
    old.threshold_version = THRESHOLD_VERSION
    with pytest.raises(ValueError, match="does not match"):
        assert_artifact_matches_config(old, cfg.normalization, allow_legacy=False)


def test_v6_refuses_unfloored_v8_artifact():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v6.yaml")
    new = fit_robust_normalizer(
        [np.linspace(-5.0, 20.0, 64, dtype=np.float32).reshape(8, 8)],
        method="robust_linear",
        raw_floor_enabled=False,
        high_percentile=99.99,
        low_percentile=0.0,
        sources=["BioTISR"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=64,
    )
    new.per_source_thresholds = {
        "BioTISR": {
            "structure_support_floor": 0.005,
            "amp_low_structure_range": 0.02,
            "crop_min_robust_range": 0.02,
        }
    }
    new.threshold_version = THRESHOLD_VERSION
    with pytest.raises(ValueError, match="does not match"):
        assert_artifact_matches_config(new, cfg.normalization, allow_legacy=False)


def test_clip_true_would_hide_negatives_v8_yaml_keeps_clip_false():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v8.yaml")
    assert cfg.normalization.clip is False
    raw = np.array([[-8.0, 0.0, 12.0]], dtype=np.float32)
    st = fit_robust_normalizer(
        [raw],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=False,
        sources=["BioTISR"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=8,
    )
    n = Normalizer(st)
    y = n.transform(raw, source="BioTISR")
    assert float(y.min()) < 0.0
    st.clip = True
    y2 = Normalizer(st).transform(raw, source="BioTISR")
    assert float(y2.min()) >= -1e-6


def test_v8_yaml_trainer_keeps_signed_batch(tmp_path):
    from microscopy_vae.engine.trainer import Trainer

    cfg = load_config(
        _repo() / "configs/experiment/s1_hq_f8z4_v8.yaml",
        overrides={
            "experiment": {
                "output_dir": str(tmp_path / "run"),
                "allow_existing_output": True,
                "seed": 0,
            },
            "data": {
                "mode": "synthetic",
                "synthetic_n_groups": 6,
                "synthetic_pages_per_group": 2,
                "synthetic_size": 64,
            },
            "crop": {"size": 64, "min_robust_range": 0.0},
            "model": {
                "encoder_block_out_channels": [32, 64, 64, 64],
                "decoder_block_out_channels": [32, 64, 64, 64],
                "layers_per_block": 1,
                "norm_num_groups": 8,
                "mid_block_add_attention": False,
            },
            "loss": {
                "w_ms_ssim": 0.0,
                "ms_ssim_start_step": 100000,
                "perceptual": {"enabled": False},
                "adversarial": {"enabled": False},
            },
            "training": {
                "max_steps": 2,
                "microbatch_size": 2,
                "grad_accum": 1,
                "num_workers": 0,
                "val_every_steps": 100,
                "log_every_steps": 1,
                "ema_decay": 0.0,
            },
            "precision": {"amp_dtype": "fp32"},
            "memory": {"gradient_checkpointing": False},
            "sampling": {"slice_weight_mode": "uniform"},
        },
    )
    assert cfg.normalization.raw_floor_enabled is False
    trainer = Trainer(cfg)
    st = trainer.normalizer.state
    assert st.raw_floor_enabled is False
    assert affine_formula(st) == "y = x/high"
    assert all(abs(float(sc["low"])) < 1e-6 for sc in st.per_source_scales.values())
    dry = trainer.dry_run()
    assert dry["normalizer_contract"]["raw_floor_enabled"] is False
    assert dry["normalizer_contract"]["affine"] == "y = x/high"
    assert dry["batch_hq_min"] < 0.0
    assert dry["batch_hq_frac_lt0"] > 0.0
    batch = next(iter(trainer.train_loader))
    assert float(batch.hq.min()) < 0.0
    n = trainer.normalizer
    y_neg = n.transform(np.array([[-5.0]], dtype=np.float32), source="SOURCE_A")
    y_zero = n.transform(np.array([[0.0]], dtype=np.float32), source="SOURCE_A")
    assert float(y_neg[0, 0]) < 0.0
    assert float(y_zero[0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert not np.allclose(y_neg, y_zero)
    result = trainer.train(max_steps=2)
    assert result["final_step"] == 2
    assert np.isfinite(float(result["final_loss"]))
    assert trainer.system.vae.output_activation == "linear"
    batch.hq = batch.hq.to(trainer.device)
    with torch.no_grad():
        recon = trainer.system.vae.reconstruct(batch.hq)
    assert recon.shape == batch.hq.shape


def test_v8_four_gpu_keeps_global_batch_8():
    from microscopy_vae.engine.distributed import resolve_per_device_batch

    assert resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=1, scale_global_batch=False
    ) == (4, 2, 8)
    assert resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=2, scale_global_batch=False
    ) == (4, 1, 8)
    assert resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=4, scale_global_batch=False
    ) == (2, 1, 8)
    per, acc, glob = resolve_per_device_batch(
        yaml_microbatch=4, yaml_accum=2, world_size=4, scale_global_batch=False
    )
    assert per * acc * 4 == glob == 8


def test_v8_production_shapes_params_and_only_four_losses():
    """Production f8/z4: latent, GN, params, and G total is Char+MS+Perc+KL only."""
    from microscopy_vae.data.records import HQBatch
    from microscopy_vae.engine.distributed import resolve_per_device_batch
    from microscopy_vae.losses.influence import GENERATOR_TERM_ORDER, quantify_generator_losses
    from microscopy_vae.models.factory import architecture_id
    from microscopy_vae.systems.factory import build_hq_codec_system

    cfg = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v8.yaml")
    assert cfg.loss.w_char == 1.0
    assert cfg.loss.w_ms_ssim == 0.12
    assert cfg.loss.perceptual.enabled is True
    assert cfg.loss.perceptual.weight == 0.05
    assert cfg.loss.adversarial.enabled is False
    assert cfg.loss.w_grad == cfg.loss.w_hf == cfg.loss.w_flux == cfg.loss.w_dark_fp == 0.0
    assert resolve_per_device_batch(
        yaml_microbatch=cfg.training.microbatch_size,
        yaml_accum=cfg.training.grad_accum,
        world_size=4,
        scale_global_batch=False,
    ) == (2, 1, 8)

    system = build_hq_codec_system(cfg)
    vae = system.vae
    assert architecture_id(vae) == "microvae_f8_z4_enc128-256-512-512_dec96-192-384-384"
    assert vae.spatial_compression == 8
    assert vae.latent_channels == 4
    assert vae.output_activation == "linear"
    x = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        out = vae(x, sample_posterior=False)
    assert tuple(out.latent.shape) == (1, 4, 32, 32)
    assert tuple(out.reconstruction.shape) == (1, 1, 256, 256)
    counts = vae.count_parameters()
    assert counts["trainable"] == counts["total"]
    assert counts["total"] > 1_000_000
    for ch in list(cfg.model.encoder_block_out_channels) + list(cfg.model.decoder_block_out_channels):
        assert ch % cfg.model.norm_num_groups == 0
    assert system.perceptual is not None
    for p in system.perceptual.parameters():
        assert p.requires_grad is False

    x = torch.rand(2, 1, 256, 256)
    batch = HQBatch(
        hq=x,
        sample_ids=["a", "b"],
        group_ids=["g0", "g0"],
        sources=["BioTISR", "BioTISR"],
        metadata=[{}, {}],
    )
    system.train()
    if system.perceptual is not None:
        system.perceptual.eval()
    off_terms = ("scharr", "hf", "flux", "dark_fp", "adv_g")
    on_later = ("ms_ssim", "perceptual")
    out0 = system.task.forward_loss(batch, optimizer_step=0)
    q0 = quantify_generator_losses(out0.unweighted, out0.weights, out0.weighted, total=out0.total)
    assert q0["weight_charbonnier"] == pytest.approx(1.0)
    assert q0["weight_kl"] >= 0.0
    for name in off_terms:
        assert q0[f"weight_{name}"] == 0.0, name
        assert q0[f"loss_w_{name}"] == pytest.approx(0.0), name
    for name in on_later:
        assert q0[f"weight_{name}"] == pytest.approx(0.0), name
    assert torch.isfinite(out0.total).item()
    recon_from_aux = out0.aux["reconstruction"]
    assert recon_from_aux.shape == x.shape
    assert vae.output_activation == "linear"

    out1 = system.task.forward_loss(batch, optimizer_step=25000)
    q1 = quantify_generator_losses(out1.unweighted, out1.weights, out1.weighted, total=out1.total)
    assert q1["weight_ms_ssim"] == pytest.approx(0.12)
    assert q1["weight_perceptual"] == pytest.approx(0.05)
    assert q1["weight_kl"] == pytest.approx(cfg.kl_schedule.beta_max)
    for name in off_terms:
        assert q1[f"weight_{name}"] == 0.0, name
        assert q1[f"loss_w_{name}"] == pytest.approx(0.0), name
    contrib_on = (
        abs(q1["loss_w_charbonnier"])
        + abs(q1["loss_w_ms_ssim"])
        + abs(q1["loss_w_perceptual"])
        + abs(q1["loss_w_kl"])
    )
    assert abs(q1["loss_g_abs_sum"] - contrib_on) < 1e-6
    assert set(GENERATOR_TERM_ORDER) >= {
        "charbonnier",
        "ms_ssim",
        "kl",
        "perceptual",
        "scharr",
        "hf",
        "flux",
        "dark_fp",
        "adv_g",
    }
    assert torch.isfinite(out1.total).item()

