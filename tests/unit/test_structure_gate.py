"""Pixel-level structure-support gate: copy filaments, do not copy isolated noise."""

import torch

from microscopy_vae.losses.composer import HQCodecLossComposer
from microscopy_vae.losses.pixel import structure_support_mask, target_grad_weight
from microscopy_vae.models.posterior import PosteriorStats
from microscopy_vae.models.vae import VAEOutput


def _out(pred: torch.Tensor) -> VAEOutput:
    b, _, h, w = pred.shape
    z_h, z_w = max(h // 4, 1), max(w // 4, 1)
    post = PosteriorStats(
        mean=torch.zeros(b, 4, z_h, z_w),
        logvar=torch.zeros(b, 4, z_h, z_w),
    )
    return VAEOutput(reconstruction=pred, latent=post.mean, posterior=post)


def _v21_composer(**kwargs) -> HQCodecLossComposer:
    defaults = dict(
        w_char=1.0,
        w_ms_ssim=0.0,
        w_grad=0.08,
        w_flux=0.0,
        w_hf=0.05,
        w_dark_fp=0.0,
        free_nats=0.0,
        beta_max=0.0,
        kl_t0=0,
        kl_t1=1,
        amp_norm=True,
        amp_norm_min_scale=0.20,
        amp_low_structure_range=0.08,
        amp_low_structure_scale=1.0,
        idle_loss_mult=0.25,
        edge_weight=0.75,
        edge_weight_clip=3.0,
        structure_support_kernel=9,
        structure_support_floor=0.02,
        structure_support_rel=0.25,
        structure_support_min_density=0.15,
        structure_min_frac=0.001,
    )
    defaults.update(kwargs)
    return HQCodecLossComposer(**defaults)


def _filament_target(bg: float = 0.06, height: float = 0.45, n: int = 64) -> torch.Tensor:
    t = torch.full((1, 1, n, n), bg)
    t[0, 0, 16:48, 32] = bg + height
    t[0, 0, 16:48, 31] = bg + 0.6 * height
    t[0, 0, 16:48, 33] = bg + 0.6 * height
    return t


def _mask(t: torch.Tensor) -> torch.Tensor:
    return structure_support_mask(t, kernel=9, floor=0.02, rel=0.25, min_density=0.15)


def test_support_rejects_isolated_spike_on_black_and_grey():
    for bg in (0.0, 0.06, 0.25):
        t = torch.full((1, 1, 64, 64), bg)
        t[0, 0, 32, 32] = bg + 0.4
        m = _mask(t)
        assert float(m[0, 0, 30:35, 30:35].max()) < 0.5, f"spike on bg={bg} should not be structure"


def test_support_keeps_filament_on_grey():
    t = _filament_target(bg=0.06)
    m = _mask(t)
    assert float(m[0, 0, 16:48, 30:35].max()) > 0.5
    assert float(m.mean()) > 0.01


def test_edge_weight_does_not_boost_unsupported_spike():
    t = torch.full((1, 1, 64, 64), 0.06)
    t[0, 0, 32, 32] = 0.5
    w = target_grad_weight(t, edge_weight=0.75, clip=3.0, support=_mask(t))
    assert float(w[0, 0, 30:35, 30:35].max()) <= 1.0 + 1e-5


def test_edge_weight_boosts_filament_not_background():
    t = _filament_target()
    t[0, 0, 8, 8] = 0.35  # isolated bg spike
    w = target_grad_weight(t, edge_weight=0.75, clip=3.0, support=_mask(t))
    assert float(w[0, 0, 16:48, 30:35].max()) > 1.2
    assert float(w[0, 0, 6:11, 6:11].max()) <= 1.0 + 1e-5


def test_scharr_and_hf_ignore_isolated_bg_spike():
    """Mixed crop: filament + isolated spike. Structure terms must not see the spike.

    Crop-level idle would still mark this crop as structured (range is large).
    Pixel Charbonnier still sees the spike (same as v1); edge/HF/Scharr must not.
    """
    target = _filament_target(bg=0.06)
    target[0, 0, 8, 8] = 0.28
    pred_smooth = _filament_target(bg=0.06)
    pred_copy = target.clone()
    loss = _v21_composer()
    a = loss(_out(pred_smooth), target, optimizer_step=0)
    b = loss(_out(pred_copy), target, optimizer_step=0)
    assert abs(float(a.unweighted["scharr"]) - float(b.unweighted["scharr"])) < 1e-6
    assert abs(float(a.unweighted["hf"]) - float(b.unweighted["hf"])) < 1e-6
    assert float(a.diagnostics["idle_frac"]) == 0.0


def test_legacy_edge_boosts_spike_gated_does_not():
    t = _filament_target(bg=0.06)
    t[0, 0, 8, 8] = 0.5
    w_legacy = target_grad_weight(t, edge_weight=0.75, clip=0.0, support=None)
    w_gated = target_grad_weight(t, edge_weight=0.75, clip=3.0, support=_mask(t))
    assert float(w_legacy[0, 0, 6:11, 6:11].max()) > 1.5
    assert float(w_gated[0, 0, 6:11, 6:11].max()) <= 1.0 + 1e-5
    assert float(w_gated[0, 0, 16:48, 30:35].max()) > 1.2


def test_filament_error_still_costs_more_than_bg_spike_error():
    target = _filament_target(bg=0.06)
    pred_good = target.clone()
    pred_miss_fil = target.clone()
    pred_miss_fil[0, 0, 16:48, 31:34] = 0.06
    pred_extra_spike = target.clone()
    pred_extra_spike[0, 0, 10, 10] = 0.4
    loss = _v21_composer()
    l_good = float(loss(_out(pred_good), target, optimizer_step=0).total)
    l_fil = float(loss(_out(pred_miss_fil), target, optimizer_step=0).total)
    l_spk = float(loss(_out(pred_extra_spike), target, optimizer_step=0).total)
    assert l_fil > l_good
    assert l_fil > l_spk


def test_empty_crop_is_not_amplified():
    empty = torch.full((1, 1, 64, 64), 0.055)
    empty[0, 0, 0, 0] = 0.062
    pred = empty + 0.01
    loss = _v21_composer()
    out = loss(_out(pred), empty, optimizer_step=0)
    assert float(out.diagnostics["idle_frac"]) == 1.0
    assert float(out.unweighted["scharr"]) == 0.0
    assert float(out.unweighted["hf"]) == 0.0
    assert float(out.diagnostics["amp_scale_mean"]) == 1.0


def test_no_dark_fp_term_in_v21_path():
    t = _filament_target()
    out = _v21_composer()(_out(t), t, optimizer_step=0)
    assert out.weights["dark_fp"] == 0.0
    assert float(out.weighted["dark_fp"]) == 0.0
