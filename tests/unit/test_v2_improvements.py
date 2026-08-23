import numpy as np
import torch

from microscopy_vae.config.loader import load_config
from microscopy_vae.data.hq_dataset import take_crop
from microscopy_vae.data.samplers import HierarchicalIndexSampler
import pytest

from microscopy_vae.losses.pixel import charbonnier_loss, per_sample_robust_scale, target_grad_weight
from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.models.blocks import Downsample2D, Upsample2D


def test_v2_yaml_loads_and_is_f4():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "s1_hq_f4z4_v2.yaml"
    cfg = load_config(path)
    assert cfg.model.upsample_mode == "bilinear"
    assert cfg.model.latent_channels == 4
    assert len(cfg.model.encoder_block_out_channels) == 3
    assert cfg.crop.size % 4 == 0
    assert cfg.loss.amp_norm is True
    assert cfg.sampling.slice_weight_mode == "focus_softmax"
    assert cfg.evaluation.allow_test is False
    assert cfg.training.max_steps == 150000


def test_f4_and_bilinear_shapes():
    model = ModelFactory.create_fresh(
        latent_channels=4,
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=True,
    )
    assert model.spatial_compression == 4
    x = torch.randn(2, 1, 64, 64)
    out = model(x, sample_posterior=True)
    assert out.reconstruction.shape == x.shape
    assert out.latent.shape == (2, 4, 16, 16)


def test_upsample_bilinear_doubles():
    up = Upsample2D(8, mode="bilinear")
    y = up(torch.randn(1, 8, 16, 16))
    assert y.shape == (1, 8, 32, 32)


def test_downsample_symmetric_halves():
    down = Downsample2D(8, pad_mode="symmetric", preblur=True)
    y = down(torch.randn(1, 8, 32, 32))
    assert y.shape == (1, 8, 16, 16)


def test_coverage_crop_hits_all_coarse_cells():
    img = np.zeros((64, 64), dtype=np.float32)
    hits: dict = {}
    draws = {"n": 0}

    def counter():
        draws["n"] += 1
        return draws["n"]

    seen = set()
    for i in range(16):
        take_crop(
            img,
            0,
            crop_size=32,
            fixed=False,
            seed=0,
            mode="coverage_jitter",
            jitter_frac=0.0,
            cell_hits=hits,
            draw_counter=counter,
        )
        seen.add(tuple(int(x) for x in np.argwhere(hits[0] > 0)[-1]))
    assert hits[0].shape == (2, 2)
    assert int(hits[0].sum()) == 16
    assert int((hits[0] > 0).sum()) == 4


def test_focus_softmax_prefers_high_score():
    meta = [
        {"source": "A", "group_id": "g", "sample_id": "0", "index": 0},
        {"source": "A", "group_id": "g", "sample_id": "1", "index": 1},
    ]
    samp = HierarchicalIndexSampler(
        meta,
        seed=0,
        epoch_length=4000,
        slice_weight_mode="focus_softmax",
        slice_scores={0: -2.0, 1: 3.0},
        focus_temperature=0.5,
        focus_min_keep=0.1,
    )
    idxs = list(iter(samp))
    frac1 = idxs.count(1) / len(idxs)
    assert frac1 > 0.6


def test_amp_scale_is_per_sample():
    a = torch.zeros(2, 1, 8, 8)
    a[0, 0, 0, 0] = 0.2
    a[1, 0, 0, 0] = 2.0
    s = per_sample_robust_scale(a, min_scale=0.01)
    assert s.shape == (2, 1, 1, 1)
    assert float(s[1]) > float(s[0])


def test_v21_yaml_loads_artifact_guards():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "s1_hq_f4z4_v2_1.yaml"
    cfg = load_config(path)
    assert cfg.loss.amp_low_structure_range == 0.08
    assert cfg.loss.amp_norm_min_scale == 0.20
    assert cfg.loss.edge_weight_clip == 3.0
    assert cfg.loss.w_dark_fp == 0.15
    assert cfg.crop.min_robust_range == 0.08
    assert cfg.evaluation.allow_test is False


def test_amp_norm_does_not_amplify_empty_patch():
    empty = torch.full((1, 1, 32, 32), 0.06)
    empty[0, 0, 0, 0] = 0.065  # tiny range << 0.08
    s_legacy = per_sample_robust_scale(empty, min_scale=0.05)
    s_guard = per_sample_robust_scale(
        empty, min_scale=0.20, low_structure_range=0.08, low_structure_scale=1.0
    )
    assert float(s_legacy) == pytest.approx(0.05)
    assert float(s_guard) == pytest.approx(1.0)


def test_edge_weight_clip():
    t = torch.zeros(1, 1, 16, 16)
    t[0, 0, 8, 8] = 5.0
    w = target_grad_weight(t, edge_weight=0.75, clip=3.0)
    assert float(w.max()) <= 3.0 + 1e-5


def test_dark_false_positive_penalizes_bg_bright():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 8:12, 8:12] = 1.0
    pred_ok = target.clone()
    pred_spots = target.clone()
    pred_spots[0, 0, 0:3, 0:3] = 0.4  # white speckle on dark bg
    from microscopy_vae.losses.pixel import dark_false_positive_loss

    l_ok = float(dark_false_positive_loss(pred_ok, target))
    l_bad = float(dark_false_positive_loss(pred_spots, target))
    assert l_bad > l_ok


def test_charbonnier_pixel_weight_changes_value():
    pred = torch.zeros(1, 1, 8, 8)
    tgt = torch.zeros(1, 1, 8, 8)
    tgt[..., 3:5, 3:5] = 1.0
    pred[..., 3:5, 3:5] = 0.5
    plain = float(charbonnier_loss(pred, tgt))
    w = torch.ones_like(tgt)
    w[..., 3:5, 3:5] = 4.0
    weighted = float(charbonnier_loss(pred, tgt, pixel_weight=w))
    assert weighted > plain
