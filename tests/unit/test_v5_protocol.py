from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from microscopy_vae.config.loader import load_config
from microscopy_vae.data.hq_dataset import _select_crop
from microscopy_vae.data.normalization import (
    NormalizationState,
    Normalizer,
    assert_artifact_matches_config,
    fit_robust_normalizer,
)
from microscopy_vae.data.threshold_calibration import (
    THRESHOLD_VERSION,
    crop_range_accept,
    fit_structure_thresholds,
)
from microscopy_vae.inference.tiling import reconstruct_full, reconstruct_tiled
from microscopy_vae.losses.pixel import structure_support_mask
from microscopy_vae.models.factory import ModelFactory


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _state(**kw) -> NormalizationState:
    base = dict(
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
    )
    base.update(kw)
    return NormalizationState(**base)


def test_v5_yaml_keeps_v4_losses_and_enables_calibration():
    v4 = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v4.yaml")
    v5 = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v5.yaml")
    assert v5.normalization.calibrate_thresholds is True
    assert v5.normalization.clip is False
    assert v5.normalization.scale_mode == "per_source"
    assert v5.normalization.raw_floor_enabled is True
    assert v5.normalization.high_percentile == 99.99
    assert v5.experiment.name == "s1_hq_f8z4_v5"
    assert v5.experiment.output_dir == "runs/s1_hq_f8z4_v5"
    assert v5.experiment.output_dir != v4.experiment.output_dir
    assert v5.experiment.notes.startswith("V5")
    assert not v5.experiment.notes.startswith("V4")
    for key in (
        "w_char",
        "w_ms_ssim",
        "w_grad",
        "w_hf",
        "w_flux",
        "w_dark_fp",
        "edge_weight",
        "idle_loss_mult",
        "structure_support_kernel",
    ):
        assert getattr(v5.loss, key) == getattr(v4.loss, key), key
    assert v5.loss.perceptual.weight == v4.loss.perceptual.weight
    assert v5.loss.adversarial.weight == v4.loss.adversarial.weight
    assert v5.model.encoder_block_out_channels == v4.model.encoder_block_out_channels
    assert v4.normalization.calibrate_thresholds is False


def test_floor_before_fit_and_transform():
    raw = np.array([[-8.0, 4.0], [0.0, 12.0]], dtype=np.float32)
    st = fit_robust_normalizer(
        [raw],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=True,
        sources=["BioTISR"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=8,
    )
    n = Normalizer(st)
    y = n.transform(raw, source="BioTISR")
    assert float(y.min()) >= -1e-6
    assert st.per_source_scales["BioTISR"]["low"] == 0.0
    with pytest.raises(ValueError, match="known source"):
        n.transform(raw, source="DeepInsight_3D")


def test_clip_false_keeps_tail_clip_true_caps():
    raw = np.linspace(0.0, 100.0, 256, dtype=np.float32).reshape(16, 16)
    st = fit_robust_normalizer(
        [raw],
        method="robust_linear",
        high_percentile=90.0,
        low_percentile=0.0,
        raw_floor_enabled=True,
        sources=["BioTISR"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=256,
    )
    n = Normalizer(st)
    y = n.transform(raw, source="BioTISR")
    assert float((y > 1).mean()) > 0
    st.clip = True
    n2 = Normalizer(st)
    y2 = n2.transform(raw, source="BioTISR")
    assert float(y2.max()) <= 1.0 + 1e-6
    hi = st.per_source_scales["BioTISR"]["high"]
    mid = np.array([[0.5 * hi]], dtype=np.float32)
    rec = n.inverse(n.transform(mid, source="BioTISR"), source="BioTISR")
    assert rec == pytest.approx(0.5 * hi, rel=1e-5, abs=1e-4)


def test_fit_split_train_only_in_state():
    st = fit_robust_normalizer(
        [np.linspace(0.0, 5.0, 64, dtype=np.float32).reshape(8, 8)],
        method="robust_linear",
        sources=["BioTISR"],
        scale_mode="per_source",
        raw_floor_enabled=True,
        high_percentile=99.99,
        low_percentile=0.0,
        max_pixels_per_page=64,
    )
    assert st.fit_split == "train"


def test_v5_refuses_v4_artifact_without_thresholds():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v5.yaml")
    old = fit_robust_normalizer(
        [np.linspace(0, 20, 64, dtype=np.float32).reshape(8, 8)],
        method="robust_linear",
        raw_floor_enabled=True,
        high_percentile=99.99,
        low_percentile=0.0,
        sources=["BioTISR"],
        scale_mode="per_source",
        clip=False,
        max_pixels_per_page=64,
    )
    assert not old.per_source_thresholds
    with pytest.raises(ValueError, match="does not match"):
        assert_artifact_matches_config(old, cfg.normalization, allow_legacy=False)
    old.per_source_thresholds = {
        "BioTISR": {
            "structure_support_floor": 0.005,
            "amp_low_structure_range": 0.02,
            "crop_min_robust_range": 0.02,
        }
    }
    old.threshold_version = THRESHOLD_VERSION
    assert_artifact_matches_config(old, cfg.normalization, allow_legacy=False)


def test_crop_hysteresis_keeps_dim_structure():
    assert crop_range_accept(0.09, 0.08) is True
    assert crop_range_accept(0.05, 0.08) is True  # 0.5*0.08=0.04 band
    assert crop_range_accept(0.03, 0.08) is False
    assert crop_range_accept(0.03, 0.02) is True


def test_calibrated_crop_keeps_dim_filament_that_v4_scalar_retries():
    canvas = np.zeros((64, 64), dtype=np.float32)
    canvas[20:22, 8:56] = 0.04
    n = Normalizer(
        _state(
            per_source_scales={"BioTISR": {"low": 0.0, "high": 1.0}},
            per_source_thresholds={
                "BioTISR": {
                    "structure_support_floor": 0.004,
                    "amp_low_structure_range": 0.02,
                    "crop_min_robust_range": 0.02,
                }
            },
            threshold_version=THRESHOLD_VERSION,
        )
    )

    def crop_fn(image, idx):
        return image

    _, norm_v5 = _select_crop(
        canvas,
        0,
        crop_fn=crop_fn,
        normalizer=n,
        min_robust_range=0.08,
        max_retries=1,
        allow_retry=True,
        source="BioTISR",
    )
    assert float(norm_v5.max()) == pytest.approx(0.04, abs=1e-6)


def test_lower_support_floor_keeps_dim_filament():
    t = torch.zeros(1, 1, 64, 64)
    t[:, :, 30:33, 8:56] = 0.035
    high = structure_support_mask(t, kernel=9, floor=0.02, rel=0.25, min_density=0.15)
    low = structure_support_mask(t, kernel=9, floor=0.004, rel=0.25, min_density=0.15)
    assert float(low.mean()) >= float(high.mean())
    assert float(low.sum()) > 0


def test_per_source_floor_tensor_matches_scalar():
    t = torch.rand(2, 1, 32, 32)
    fl = torch.tensor([0.02, 0.02]).view(2, 1, 1, 1)
    a = structure_support_mask(t, kernel=9, floor=0.02, rel=0.25, min_density=0.15)
    b = structure_support_mask(t, kernel=9, floor=fl, rel=0.25, min_density=0.15)
    assert torch.allclose(a, b)


def test_fit_thresholds_are_source_specific_and_at_most_yaml_floor():
    rng = np.random.default_rng(0)
    bright = rng.random((48, 48)).astype(np.float32)
    bright[10:14, :] = 1.0
    empty = rng.random((48, 48)).astype(np.float32) * 0.01
    thr, diag = fit_structure_thresholds(
        [bright, bright, empty, empty],
        ["A", "A", "B", "B"],
        crop_size=32,
        fallback_floor=0.02,
        fallback_range=0.08,
        crops_per_page=3,
        seed=0,
    )
    assert set(thr) == {"A", "B"}
    assert thr["A"]["structure_support_floor"] <= 0.02
    assert thr["B"]["structure_support_floor"] <= 0.02
    assert thr["A"]["crop_min_robust_range"] <= 0.08
    # Amp gate stays at the yaml 0.08 even if crop gate is lowered.
    assert thr["A"]["amp_low_structure_range"] == pytest.approx(0.08)
    assert thr["B"]["amp_low_structure_range"] == pytest.approx(0.08)
    decoupled, _ = fit_structure_thresholds(
        [bright, bright, empty, empty],
        ["A", "A", "B", "B"],
        crop_size=32,
        fallback_floor=0.02,
        fallback_range=0.03,
        fallback_amp_range=0.08,
        crops_per_page=3,
        seed=0,
    )
    assert decoupled["A"]["amp_low_structure_range"] == pytest.approx(0.08)
    assert decoupled["A"]["crop_min_robust_range"] <= 0.03 + 1e-12
    assert thr["A"]["structure_support_floor"] >= 0.002 - 1e-12
    assert diag["A"]["n_crops"] > 0


def test_rejected_empty_crop_does_not_consume_coverage_cell():
    canvas = np.zeros((64, 64), dtype=np.float32)
    canvas[0:32, 0:32] = 0.5
    n = Normalizer(_state(per_source_scales={"BioTISR": {"low": 0.0, "high": 1.0}}))
    hits = {0: np.zeros((2, 2), dtype=np.int32)}
    last_cell: list = []
    n_calls = {"n": 0}

    def crop_fn(image, idx):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            last_cell[:] = [0, 1, 1]
            hits[0][1, 1] += 1
            return image[32:64, 32:64]
        last_cell[:] = [0, 0, 0]
        hits[0][0, 0] += 1
        return image[0:32, 0:32]

    _select_crop(
        canvas,
        0,
        crop_fn=crop_fn,
        normalizer=n,
        min_robust_range=0.08,
        max_retries=3,
        allow_retry=True,
        source="BioTISR",
        empty_keep_prob=0.0,
        cell_hits=hits,
        last_cell=last_cell,
    )
    assert int(hits[0][1, 1]) == 0
    assert int(hits[0][0, 0]) == 1


def test_tiled_has_no_normalizer_and_full_equals_tiled_on_tiny():
    sig = inspect.signature(reconstruct_tiled)
    assert "normalizer" not in sig.parameters
    model = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64, 64),
        decoder_block_out_channels=(32, 64, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=True,
    )
    model.eval()
    x = torch.zeros(1, 1, 64, 64)
    x[:, :, 8:40, 8:40] = 0.4
    with torch.no_grad():
        full, _ = reconstruct_full(model, x, spatial_compression=8, return_aux=True)
        tiled, aux = reconstruct_tiled(
            model, x, tile_size=32, overlap=8, spatial_compression=8, return_aux=True
        )
    assert full.shape == tiled.shape
    assert aux["tile_size"] == 32


def test_smoke_v5_fits_thresholds_and_trains(tmp_path):
    from microscopy_vae.engine.trainer import Trainer

    cfg = load_config(_repo() / "configs/experiment/smoke_v5.yaml")
    cfg.experiment.output_dir = str(tmp_path / "smoke_v5")
    cfg.training.max_steps = 2
    trainer = Trainer(cfg)
    assert trainer.cfg.experiment.name == "smoke_v5"
    dry = trainer.dry_run()
    assert dry["experiment"] == "smoke_v5"
    assert trainer.normalizer.state.threshold_version == THRESHOLD_VERSION
    assert trainer.normalizer.state.per_source_thresholds
    assert trainer.normalizer.state.clip is False
    result = trainer.train()
    assert int(result["final_step"]) >= 1
    js = tmp_path / "smoke_v5" / "normalizer.json"
    assert js.is_file()
    loaded = NormalizationState.load(js)
    assert loaded.per_source_thresholds
    assert loaded.threshold_version == THRESHOLD_VERSION


def test_empty_keep_can_retain_dark_crop():
    canvas = np.zeros((32, 32), dtype=np.float32)
    n = Normalizer(_state(per_source_scales={"BioTISR": {"low": 0.0, "high": 1.0}}))
    kept = 0
    for i in range(40):
        _, norm = _select_crop(
            canvas,
            i,
            crop_fn=lambda image, idx: image,
            normalizer=n,
            min_robust_range=0.08,
            max_retries=3,
            allow_retry=True,
            source="BioTISR",
            empty_keep_prob=1.0,
        )
        if float(norm.max()) == 0.0:
            kept += 1
    assert kept == 40


def test_soft_bg_weight_leaves_background_error():
    from microscopy_vae.losses.composer import HQCodecLossComposer
    from microscopy_vae.models.posterior import PosteriorStats
    from microscopy_vae.models.vae import VAEOutput

    tgt = torch.zeros(1, 1, 64, 64)
    tgt[:, :, 20:24, 8:56] = 0.5
    pred = tgt.clone()
    pred[:, :, 0:8, 0:8] = 3.0  # off-support blow-up
    post = PosteriorStats(mean=torch.zeros(1, 4, 16, 16), logvar=torch.zeros(1, 4, 16, 16))
    out = VAEOutput(reconstruction=pred, latent=post.mean, posterior=post)
    hard = HQCodecLossComposer(
        w_char=0.0,
        w_ms_ssim=1.0,
        w_grad=0.0,
        w_hf=0.0,
        w_flux=0.0,
        free_nats=0.0,
        beta_max=0.0,
        kl_t0=0,
        kl_t1=1,
        ms_ssim_start_step=0,
        ms_ssim_ramp_steps=0,
        structure_support_kernel=9,
        structure_min_frac=0.0003,
        unstructured_bg_weight=0.0,
    )
    soft = HQCodecLossComposer(
        w_char=0.0,
        w_ms_ssim=1.0,
        w_grad=0.0,
        w_hf=0.0,
        w_flux=0.0,
        free_nats=0.0,
        beta_max=0.0,
        kl_t0=0,
        kl_t1=1,
        ms_ssim_start_step=0,
        ms_ssim_ramp_steps=0,
        structure_support_kernel=9,
        structure_min_frac=0.0003,
        unstructured_bg_weight=0.25,
    )
    lh = float(hard(out, tgt, optimizer_step=0).unweighted["ms_ssim"])
    ls = float(soft(out, tgt, optimizer_step=0).unweighted["ms_ssim"])
    assert ls >= lh - 1e-6


def test_halo_reads_real_context_not_only_core():
    from microscopy_vae.inference.tiling import reconstruct_halo, reconstruct_one_tile

    model = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64, 64),
        decoder_block_out_channels=(32, 64, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=True,
    )
    model.eval()
    x = torch.zeros(1, 1, 64, 64)
    x[:, :, :, 32:] = 0.8
    with torch.no_grad():
        halo, aux = reconstruct_halo(
            model, x, tile_size=32, overlap=0, halo=16, spatial_compression=8, return_aux=True
        )
        tiled = reconstruct_one_tile(model, x[:, :, :, :32], spatial_compression=8)
    assert aux["mode"] == "halo"
    assert aux["context"] == "real_image_crop"
    assert halo.shape == x.shape
    # Left core saw right-side context; should differ from isolated left tile.
    assert not torch.allclose(halo[:, :, :, :32], tiled, atol=1e-5)


def test_frac_neg_before_floor_recorded():
    raw = np.array([[-5.0, 2.0], [0.0, 8.0]], dtype=np.float32)
    st = fit_robust_normalizer(
        [raw],
        method="robust_linear",
        sources=["BioTISR"],
        scale_mode="per_source",
        raw_floor_enabled=True,
        high_percentile=100.0,
        low_percentile=0.0,
        max_pixels_per_page=8,
    )
    assert st.per_source_stats["BioTISR"]["frac_neg_before_floor"] == pytest.approx(0.25)
    assert st.contract_dict()["floor_before_normalize"] is True


def test_source_unit_scale_is_that_source_high():
    from microscopy_vae.engine.val_report import default_unit_scale, source_unit_scale

    st = fit_robust_normalizer(
        [
            np.linspace(0.0, 10.0, 256, dtype=np.float32).reshape(16, 16),
            np.linspace(0.0, 90.0, 256, dtype=np.float32).reshape(16, 16),
        ],
        method="robust_linear",
        low_percentile=0.0,
        high_percentile=100.0,
        raw_floor_enabled=True,
        sources=["BioTISR", "DeepInsight_3D"],
        scale_mode="per_source",
        max_pixels_per_page=256,
    )
    assert default_unit_scale(st) == pytest.approx(90.0, abs=1e-3)
    assert source_unit_scale(st, "BioTISR") == pytest.approx(10.0, abs=1e-3)
    assert source_unit_scale(st, "DeepInsight_3D") == pytest.approx(90.0, abs=1e-3)


def test_per_source_infer_refuses_missing_normalizer():
    from microscopy_vae.cli import cmd_encode

    args = argparse.Namespace(
        config=str(_repo() / "configs/experiment/s1_hq_f8z4_v5.yaml"),
        override=[],
        print_resolved_config=False,
        dry_run=False,
        input="x.tif",
        output="z.npy",
        weights=None,
        normalizer=None,
        page=0,
        padding_mode="reflect",
        raw_weights=True,
        devices="cpu",
        source="BioTISR",
        meta=None,
    )
    with pytest.raises(SystemExit, match="requires --normalizer"):
        cmd_encode(args)


def test_ablation_yamls_load():
    clip = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v5_clip.yaml")
    nogan = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v5_nogan.yaml")
    scharr = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v5_scharr.yaml")
    hf = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v5_hf.yaml")
    rec = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v5.yaml")
    assert clip.normalization.clip is True
    assert rec.normalization.clip is False
    assert nogan.loss.adversarial.enabled is False
    assert rec.loss.adversarial.enabled is True
    assert scharr.loss.w_grad == 0.08
    assert hf.loss.w_hf == 0.05
    assert rec.crop.empty_keep_prob == 0.55
    assert rec.loss.amp_smooth is True
    assert rec.loss.unstructured_bg_weight == 0.25
    # Single-factor ablations must keep the rest of the V5 protocol.
    for cfg in (clip, nogan, scharr, hf):
        assert cfg.loss.amp_smooth is True
        assert cfg.loss.unstructured_bg_weight == 0.25
        assert cfg.crop.empty_keep_prob == 0.55
        assert cfg.normalization.calibrate_thresholds is True
        assert cfg.normalization.scale_mode == "per_source"


def test_zero_input_bias_not_amplified_when_calibrated_amp_gate_on():
    from microscopy_vae.losses.pixel import per_sample_robust_scale

    z = torch.zeros(2, 1, 32, 32)
    s = per_sample_robust_scale(
        z,
        min_scale=0.20,
        low_structure_range=torch.tensor([0.03, 0.03]),
        low_structure_scale=1.0,
        smooth=True,
    )
    assert torch.allclose(s, torch.ones_like(s))
