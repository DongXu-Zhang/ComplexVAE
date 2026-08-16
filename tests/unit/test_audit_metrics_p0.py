import numpy as np
import pytest

from microscopy_vae.metrics.extended import (
    mae_np,
    mse_np,
    nmse,
    psnr_fixed_range,
    psnr_from_mse,
    robust_foreground_mask,
    slice_metric_bundle,
    snr_db,
    volume_pooled_psnr,
)
from microscopy_vae.metrics.focus import score_volume_slices, select_case_slice_index
from microscopy_vae.metrics.stripe import compare_target_recon_stripes, stripe_score


def test_psnr_matches_known_mse():
    tgt = np.zeros((8, 8), dtype=np.float32)
    pred = np.full((8, 8), 0.1, dtype=np.float32)
    assert mse_np(pred, tgt) == pytest.approx(0.01)
    assert psnr_fixed_range(pred, tgt, 1.0) == pytest.approx(20.0)
    assert psnr_from_mse(0.01, 1.0) == pytest.approx(20.0)


def test_volume_pooled_psnr_not_mean_of_psnrs():
    # two slices: mse 0.01 (20 dB) and 0.0001 (40 dB)
    mean_psnr = 30.0
    pooled = volume_pooled_psnr([0.01, 0.0001], data_range=1.0)
    assert pooled < mean_psnr
    assert pooled == pytest.approx(psnr_from_mse(0.00505, 1.0))


def test_low_contrast_inflates_range1_psnr_but_not_nmse():
    rng = np.random.default_rng(0)
    tgt = 0.12 + 0.01 * rng.normal(size=(64, 64)).astype(np.float32)
    pred = tgt - 0.002  # almost mean-matched, filaments not needed
    psnr = psnr_fixed_range(pred, tgt, 1.0)
    # MAE ~0.002 → MSE 4e-6 → PSNR ~54 dB despite tiny structure scale
    assert psnr > 45.0
    assert nmse(pred, tgt) > 0.0
    assert np.isfinite(snr_db(pred, tgt))


def test_foreground_mask_bright_spot():
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:12, 8:12] = 5.0
    info = robust_foreground_mask(img, k=3.0, blur_sigma=0.0)
    assert info["fg_frac"] > 0.01
    assert info["mask"][10, 10]
    assert not info["mask"][0, 0]


def test_negative_d3_like_values_mask_does_not_crash():
    rng = np.random.default_rng(1)
    img = rng.normal(-0.2, 0.6, size=(48, 48)).astype(np.float32)
    img[20:28, 20:28] += 3.0
    info = robust_foreground_mask(img)
    assert 0.0 < float(info["fg_frac"]) < 1.0
    b = slice_metric_bundle(img * 0.9, img, data_range=1.0)
    assert np.isfinite(b["psnr_range1"])
    assert "fg_mae" in b and "bg_mae" in b


def test_vertical_stripes_score_higher_than_horizontal():
    yy, xx = np.mgrid[0:64, 0:64]
    vertical = np.sin(2 * np.pi * xx / 8.0).astype(np.float32)
    horizontal = np.sin(2 * np.pi * yy / 8.0).astype(np.float32)
    sv = stripe_score(vertical)
    sh = stripe_score(horizontal)
    assert sv["vertical_score"] > sh["vertical_score"]
    assert sh["horizontal_score"] > sv["horizontal_score"]
    assert sv["col_period_8"] > 0.3


def test_stripe_delta_detects_recon_only_bars():
    rng = np.random.default_rng(2)
    target = rng.normal(0, 0.05, size=(64, 64)).astype(np.float32)
    _, xx = np.mgrid[0:64, 0:64]
    recon = target + 0.2 * np.sin(2 * np.pi * xx / 8.0).astype(np.float32)
    cmp = compare_target_recon_stripes(target, recon)
    assert cmp["delta_vertical_score"] > 0.0
    assert cmp["recon_col_period_8"] > cmp["target_col_period_8"]


def test_focus_ranks_sharp_slice_above_blur():
    rng = np.random.default_rng(3)
    sharp = rng.normal(0, 0.05, size=(48, 48)).astype(np.float32)
    sharp[10:14, :] = 1.0
    sharp[:, 20:22] = 1.0
    blur = sharp.copy()
    # box blur
    for _ in range(8):
        pad = np.pad(blur, 1, mode="edge")
        blur = (
            pad[:-2, 1:-1] + pad[1:-1, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        ) / 5.0
    rows = score_volume_slices([blur.astype(np.float32), sharp, blur.astype(np.float32)])
    assert rows[1]["focus_score"] > rows[0]["focus_score"]
    assert rows[1]["focus_score"] > rows[2]["focus_score"]
    assert select_case_slice_index(rows, central_fraction=1.0) == 1


def test_case_slice_avoids_first_and_last_when_central():
    # highest score on slice 0, but central 50% of 6 slices is 1..4
    fake = [{"focus_score": 10.0 - i} for i in range(6)]
    assert select_case_slice_index(fake, central_fraction=0.5) == 1


def test_mae_np_matches_definition():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[1.5, 1.0]], dtype=np.float32)
    assert mae_np(a, b) == pytest.approx(0.75)
