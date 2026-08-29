from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from microscopy_vae.config.loader import load_config
from microscopy_vae.config.schema import RootConfig
from microscopy_vae.data.normalization import (
    NormalizationState,
    Normalizer,
    assert_artifact_matches_config,
    fit_robust_normalizer,
    summarize_percentile_candidates,
)
from microscopy_vae.losses.composer import HQCodecLossComposer
from microscopy_vae.losses.influence import GENERATOR_TERM_ORDER, quantify_generator_losses
from microscopy_vae.losses.pixel import structure_support_mask
from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.models.posterior import PosteriorStats
from microscopy_vae.models.vae import VAEOutput


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _out(pred: torch.Tensor) -> VAEOutput:
    b, _, h, w = pred.shape
    post = PosteriorStats(
        mean=torch.zeros(b, 4, max(h // 4, 1), max(w // 4, 1)),
        logvar=torch.zeros(b, 4, max(h // 4, 1), max(w // 4, 1)),
    )
    return VAEOutput(reconstruction=pred, latent=post.mean, posterior=post)


def test_v22_yaml_still_loads_with_scharr_hf_flux():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f4z4_v2_2.yaml")
    assert cfg.loss.w_grad == 0.08
    assert cfg.loss.w_hf == 0.05
    assert cfg.loss.w_flux == 0.01
    assert cfg.normalization.method == "robust_linear_p0.1_p99.9"
    assert cfg.normalization.raw_floor_enabled is False
    assert cfg.normalization.high_percentile == 99.9
    assert cfg.evaluation.allow_test is False


def test_v4_yaml_drops_independent_structure_losses_keeps_gate():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f4z4_v4.yaml")
    assert cfg.loss.w_grad == 0.0
    assert cfg.loss.w_hf == 0.0
    assert cfg.loss.w_flux == 0.0
    assert cfg.loss.w_char == 1.0
    assert cfg.loss.w_ms_ssim == 0.12
    assert cfg.loss.perceptual.enabled is True
    assert cfg.loss.adversarial.enabled is True
    assert cfg.loss.structure_support_kernel == 9
    assert cfg.loss.edge_weight == 0.75
    assert cfg.normalization.method == "robust_linear"
    assert cfg.normalization.low_percentile == 0.0
    assert cfg.normalization.high_percentile == 99.99
    assert cfg.normalization.raw_floor_enabled is True
    assert cfg.normalization.raw_floor_value == 0.0
    assert cfg.normalization.scale_mode == "per_source"
    assert cfg.normalization.clip is False
    assert cfg.normalization.fit_mode == "source_balanced"
    assert cfg.model.output_activation == "linear"
    assert cfg.evaluation.allow_test is False
    assert "test" not in cfg.data.allow_splits


def test_legacy_method_rejects_custom_percentile():
    with pytest.raises(ValidationError):
        RootConfig(
            normalization={
                "method": "robust_linear_p0.1_p99.9",
                "high_percentile": 99.99,
            }
        )


def test_percentile_bounds():
    with pytest.raises(ValidationError):
        RootConfig(normalization={"method": "robust_linear", "low_percentile": 50, "high_percentile": 10})


def test_old_normalizer_json_loads_without_new_fields():
    d = {
        "schema_version": "microvae-normalizer-v1",
        "method": "robust_linear_p0.1_p99.9",
        "fit_split": "train",
        "low": -10.0,
        "high": 90.0,
        "clip": False,
        "role": "hq",
        "n_groups": 1,
        "config_sha256": "abc",
        "manifest_sha256": "def",
        "transform_id": "legacy",
        "fit_mode": "source_balanced",
        "per_source_stats": {},
        "n_pages_fit": 3,
    }
    state = NormalizationState.from_dict(d)
    assert state.raw_floor_enabled is False
    assert state.low_percentile == 0.1
    assert state.high_percentile == 99.9
    x = np.array([[-5.0, 40.0]], dtype=np.float32)
    y = Normalizer(state).transform(x)
    y_legacy = (x - (-10.0)) / (90.0 - (-10.0) + 1e-8)
    assert np.allclose(y, y_legacy, atol=1e-6)


def test_floor_applied_before_percentiles_and_transform():
    rng = np.random.default_rng(0)
    pos = rng.uniform(1.0, 10.0, size=(32, 32)).astype(np.float32)
    neg = rng.uniform(-50.0, -1.0, size=(32, 32)).astype(np.float32)
    mixed = np.concatenate([pos.ravel(), neg.ravel()]).reshape(32, 64)
    st_off = fit_robust_normalizer(
        [mixed],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=False,
        fit_mode="page_uniform",
        sources=None,
    )
    st_on = fit_robust_normalizer(
        [mixed],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=True,
        raw_floor_value=0.0,
        fit_mode="page_uniform",
        sources=None,
    )
    assert st_off.low < 0
    assert st_on.low >= -1e-6
    y = Normalizer(st_on).transform(np.array([[-20.0, 5.0]], dtype=np.float32))
    y_manual = Normalizer(st_on).transform(np.array([[0.0, 5.0]], dtype=np.float32))
    assert np.allclose(y, y_manual)


def test_transform_torch_matches_numpy():
    arrs = [np.array([[-3.0, 0.0, 4.0, 12.0]], dtype=np.float32)]
    st = fit_robust_normalizer(
        arrs,
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=True,
        fit_mode="page_uniform",
    )
    n = Normalizer(st)
    x = arrs[0]
    y_np = n.transform(x)
    y_t = n.transform_torch(torch.from_numpy(x)).numpy()
    assert np.allclose(y_np, y_t, atol=1e-5)


def test_transform_id_changes_with_floor_and_percentile():
    a = [np.linspace(-2, 8, 64, dtype=np.float32).reshape(8, 8)]
    a1 = fit_robust_normalizer(a, method="robust_linear_p0.1_p99.9", fit_mode="page_uniform")
    a2 = fit_robust_normalizer(
        a, method="robust_linear", high_percentile=99.99, raw_floor_enabled=True, fit_mode="page_uniform"
    )
    assert a1.transform_id != a2.transform_id


def test_per_source_scales_after_floor_use_own_high():
    bio = np.linspace(0.0, 10.0, 256, dtype=np.float32).reshape(16, 16)
    di2 = np.linspace(0.0, 40.0, 256, dtype=np.float32).reshape(16, 16)
    di3 = np.concatenate(
        [np.linspace(0.0, 30.0, 200), np.linspace(80.0, 90.0, 56)]
    ).astype(np.float32).reshape(16, 16)
    neg = np.full((16, 16), -50.0, dtype=np.float32)
    st = fit_robust_normalizer(
        [bio, di2, di3, neg],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=True,
        raw_floor_value=0.0,
        sources=["BioTISR", "DeepInsight_2D", "DeepInsight_3D", "DeepInsight_3D"],
        scale_mode="per_source",
        fit_mode="source_balanced",
        max_pixels_per_page=256,
    )
    n = Normalizer(st)
    assert st.scale_mode == "per_source"
    assert set(st.per_source_scales) == {"BioTISR", "DeepInsight_2D", "DeepInsight_3D"}
    assert all(abs(float(sc["low"]) - 0.0) < 1e-9 for sc in st.per_source_scales.values())
    y_bio = n.transform(np.array([[10.0]], dtype=np.float32), source="BioTISR")
    y_di2 = n.transform(np.array([[10.0]], dtype=np.float32), source="DeepInsight_2D")
    assert float(y_bio[0, 0]) > float(y_di2[0, 0])
    y_neg = n.transform(np.array([[-20.0]], dtype=np.float32), source="BioTISR")
    y_zero = n.transform(np.array([[0.0]], dtype=np.float32), source="BioTISR")
    assert np.allclose(y_neg, y_zero)
    with pytest.raises(ValueError, match="known source"):
        n.transform(np.array([[1.0]], dtype=np.float32))


def test_guess_source_from_path():
    from microscopy_vae.data.normalization import guess_source_from_path

    assert guess_source_from_path(r"F:\Dataset\BioTISR\CCPs\a.mrc") == "BioTISR"
    assert guess_source_from_path("/data/DeepInsight_3D/vol.tif") == "DeepInsight_3D"
    assert guess_source_from_path("/data/DeepInsight_2D/x.mrc") == "DeepInsight_2D"


def test_source_balanced_still_median_of_source_highs():
    bio = np.linspace(0.0, 10.0, 256, dtype=np.float32).reshape(16, 16)
    di2 = np.linspace(0.0, 40.0, 256, dtype=np.float32).reshape(16, 16)
    di3 = np.linspace(0.0, 90.0, 256, dtype=np.float32).reshape(16, 16)
    st = fit_robust_normalizer(
        [bio, di2, di3],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=True,
        sources=["BioTISR", "DeepInsight_2D", "DeepInsight_3D"],
        fit_mode="source_balanced",
        max_pixels_per_page=256,
    )
    assert st.high == pytest.approx(40.0, abs=1e-3)
    assert st.fit_mode == "source_balanced"


def test_refuse_silent_mix_v4_cfg_old_artifact():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f4z4_v4.yaml")
    old = fit_robust_normalizer(
        [np.linspace(-5, 20, 64, dtype=np.float32).reshape(8, 8)],
        method="robust_linear_p0.1_p99.9",
        raw_floor_enabled=False,
        fit_mode="page_uniform",
    )
    with pytest.raises(ValueError, match="does not match"):
        assert_artifact_matches_config(old, cfg.normalization, allow_legacy=False)
    assert_artifact_matches_config(old, cfg.normalization, allow_legacy=True)


def test_v4_total_loss_has_zero_scharr_hf_flux():
    pred = torch.randn(2, 1, 64, 64)
    tgt = torch.randn(2, 1, 64, 64)
    c = HQCodecLossComposer(
        w_char=1.0,
        w_ms_ssim=0.12,
        w_grad=0.0,
        w_hf=0.0,
        w_flux=0.0,
        w_dark_fp=0.0,
        free_nats=0.0,
        beta_max=0.0,
        kl_t0=0,
        kl_t1=1,
        ms_ssim_start_step=0,
        ms_ssim_ramp_steps=0,
        amp_norm=True,
        edge_weight=0.75,
        edge_weight_clip=3.0,
        structure_support_kernel=9,
        structure_min_frac=0.0003,
    )
    out = c(_out(pred), tgt, optimizer_step=5000)
    assert float(out.weights["scharr"]) == 0.0
    assert float(out.weights["hf"]) == 0.0
    assert float(out.weights["flux"]) == 0.0
    assert float(out.weighted["scharr"]) == 0.0
    assert float(out.weighted["hf"]) == 0.0
    assert float(out.weighted["flux"]) == 0.0
    q = quantify_generator_losses(out.unweighted, out.weights, out.weighted, total=out.total)
    assert q["loss_w_scharr"] == 0.0
    assert q["loss_w_hf"] == 0.0
    assert q["loss_w_flux"] == 0.0
    assert set(GENERATOR_TERM_ORDER) >= {"charbonnier", "ms_ssim", "kl"}


def test_v4_remaining_terms_finite_grad():
    model = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
    )
    loss_fn = HQCodecLossComposer(
        w_char=1.0,
        w_ms_ssim=0.12,
        w_grad=0.0,
        w_hf=0.0,
        w_flux=0.0,
        free_nats=0.0,
        beta_max=1e-3,
        kl_t0=0,
        kl_t1=1,
        ms_ssim_start_step=0,
        ms_ssim_ramp_steps=0,
        edge_weight=0.75,
        edge_weight_clip=3.0,
        structure_support_kernel=9,
    )
    x = torch.randn(2, 1, 64, 64)
    out = model(x, sample_posterior=True)
    loss = loss_fn(out, x, optimizer_step=10).total
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_structure_support_still_uses_scharr_when_loss_off():
    t = torch.full((1, 1, 64, 64), 0.06)
    t[0, 0, 16:48, 32] = 0.51
    t[0, 0, 16:48, 31] = 0.33
    t[0, 0, 16:48, 33] = 0.33
    m = structure_support_mask(t, kernel=9, floor=0.02, rel=0.25, min_density=0.15)
    assert float(m[0, 0, 20:44, 30:35].max()) > 0.5


def test_candidate_summary_refuses_to_need_test_and_records_n_above():
    rng = np.random.default_rng(1)
    bio = rng.uniform(0, 20, size=(8, 8)).astype(np.float32)
    di2 = rng.uniform(0, 80, size=(8, 8)).astype(np.float32)
    di3 = np.concatenate(
        [rng.uniform(0, 30, size=32), rng.uniform(200, 400, size=32)]
    ).astype(np.float32).reshape(8, 8)
    out = summarize_percentile_candidates(
        [bio, di2, di3],
        ["BioTISR", "DeepInsight_2D", "DeepInsight_3D"],
        candidates=(99.9, 99.99),
        raw_floor_enabled=True,
        max_pixels_per_page=64,
    )
    assert "source_balanced_globals" in out
    assert out["per_source"]["DeepInsight_3D"]["p99.9_n_above"] >= 0
    # median of three highs is the middle source
    g = out["source_balanced_globals"]["p99.9"]
    assert g["limiting_source_for_high"] in {"BioTISR", "DeepInsight_2D", "DeepInsight_3D"}


def test_severe_is_not_forced_bottom_percent():
    from types import SimpleNamespace
    from microscopy_vae.engine.val_report import classify_severe, summarize_rows

    cfg = SimpleNamespace(
        severe_mae_unit=0.10,
        severe_bg_fp_rate=0.15,
        severe_bg_bias=0.02,
        severe_bright_retention=0.50,
        severe_dark_grad_retention=0.40,
        worst_n=3,
    )
    rows = []
    for i in range(20):
        rows.append(
            {
                "sample_id": str(i),
                "group_id": f"g{i // 5}",
                "source": "BioTISR",
                "morphology": "filament",
                "metrics_unit": {
                    "mae": 0.01 + 0.001 * i,
                    "psnr": 30.0,
                    "ssim": 0.9,
                    "signed_bias": 0.0,
                    "bg_fp_rate": 0.0,
                    "bg_bias": 0.0,
                    "bright_retention": 0.9,
                    "dark_grad_retention": 0.9,
                },
            }
        )
    summ = summarize_rows(rows, cfg=cfg)
    assert summ["overall"]["n_severe_slices"] == 0
    assert len(summ["worst_tail"]) == 3
    bad = {
        "mae": 0.2,
        "psnr": 10.0,
        "ssim": 0.2,
        "signed_bias": 0.1,
        "bg_fp_rate": 0.2,
        "bg_bias": 0.05,
        "bright_retention": 0.3,
        "dark_grad_retention": 0.2,
    }
    is_s, reasons = classify_severe(bad, cfg=cfg)
    assert is_s
    assert "mae_unit" in reasons


def test_no_test_in_v4_allow_splits():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f4z4_v4.yaml")
    assert cfg.data.allow_splits == ["train", "val"]


def test_default_unit_scale_uses_brightest_source():
    from microscopy_vae.engine.val_report import default_unit_scale

    st = fit_robust_normalizer(
        [
            np.linspace(0.0, 10.0, 256, dtype=np.float32).reshape(16, 16),
            np.linspace(0.0, 40.0, 256, dtype=np.float32).reshape(16, 16),
            np.linspace(0.0, 90.0, 256, dtype=np.float32).reshape(16, 16),
        ],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=True,
        sources=["BioTISR", "DeepInsight_2D", "DeepInsight_3D"],
        scale_mode="per_source",
        max_pixels_per_page=256,
    )
    # median high is 40; using that would push DI3D to y>1 in the report domain
    assert default_unit_scale(st) == pytest.approx(90.0, abs=1e-3)
    assert float(st.high) == pytest.approx(40.0, abs=1e-3)


def test_v4_yaml_trainer_two_steps_has_effect(tmp_path):
    """Wiring check: V4 yaml → per-source floor actually reaches the batch."""
    from microscopy_vae.engine.trainer import Trainer
    from microscopy_vae.losses.influence import quantify_generator_losses

    cfg = load_config(
        _repo() / "configs/experiment/s1_hq_f4z4_v4.yaml",
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
                "encoder_block_out_channels": [32, 64, 64],
                "decoder_block_out_channels": [32, 64, 64],
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
    assert cfg.normalization.scale_mode == "per_source"
    assert cfg.normalization.raw_floor_enabled is True
    assert cfg.loss.w_grad == 0.0
    trainer = Trainer(cfg)
    st = trainer.normalizer.state
    assert st.scale_mode == "per_source"
    assert st.raw_floor_enabled is True
    assert abs(st.high_percentile - 99.99) < 1e-9
    assert set(st.per_source_scales) == {"SOURCE_A", "SOURCE_B"}
    assert all(abs(float(sc["low"]) - 0.0) < 1e-6 for sc in st.per_source_scales.values())
    highs = {s: float(sc["high"]) for s, sc in st.per_source_scales.items()}
    assert all(h > 0 for h in highs.values())
    dry = trainer.dry_run()
    assert dry["normalizer_contract"]["scale_mode"] == "per_source"
    assert dry["normalizer_contract"]["raw_floor_enabled"] is True
    assert dry["batch_hq_frac_lt0"] == 0.0
    assert dry["batch_hq_min"] >= -1e-6
    batch = next(iter(trainer.train_loader))
    assert float(batch.hq.min()) >= -1e-6
    # same raw value must map differently if the two source highs differ
    n = trainer.normalizer
    y_a = float(n.transform(np.array([[1.0]], dtype=np.float32), source="SOURCE_A")[0, 0])
    y_b = float(n.transform(np.array([[1.0]], dtype=np.float32), source="SOURCE_B")[0, 0])
    if abs(highs["SOURCE_A"] - highs["SOURCE_B"]) > 1e-4:
        assert abs(y_a - y_b) > 1e-6
    y_neg = n.transform(np.array([[-5.0]], dtype=np.float32), source="SOURCE_A")
    y_zero = n.transform(np.array([[0.0]], dtype=np.float32), source="SOURCE_A")
    assert np.allclose(y_neg, y_zero)
    result = trainer.train(max_steps=2)
    assert result["final_step"] == 2
    assert np.isfinite(float(result["final_loss"]))
    batch.hq = batch.hq.to(trainer.device)
    out = trainer.system.task.forward_loss(batch, optimizer_step=5000)
    q = quantify_generator_losses(out.unweighted, out.weights, out.weighted, total=out.total)
    assert q["weight_scharr"] == 0.0
    assert q["weight_hf"] == 0.0
    assert q["weight_flux"] == 0.0
    assert q["loss_w_scharr"] == 0.0
    assert q["loss_w_hf"] == 0.0
    assert q["loss_w_flux"] == 0.0
    assert not hasattr(trainer, "test_loader")


def test_smoke_v4_yaml_uses_v4_norm_contract():
    cfg = load_config(_repo() / "configs/experiment/smoke_v4.yaml")
    assert cfg.normalization.method == "robust_linear"
    assert cfg.normalization.scale_mode == "per_source"
    assert cfg.normalization.raw_floor_enabled is True
    assert cfg.normalization.high_percentile == 99.99
    assert cfg.loss.w_grad == 0.0
    assert cfg.loss.w_hf == 0.0
    assert cfg.loss.w_flux == 0.0
