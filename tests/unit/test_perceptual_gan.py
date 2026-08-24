"""Perceptual + GAN + influence: isolation, domain, resume, identity when off."""

from __future__ import annotations

from pathlib import Path

import torch

from microscopy_vae.config.loader import load_config
from microscopy_vae.config.schema import RootConfig
from microscopy_vae.engine.checkpoint import CheckpointManager
from microscopy_vae.engine.state import TrainerState
from microscopy_vae.losses.adversarial import (
    PatchDiscriminator,
    discriminator_scores,
    generator_adv_loss,
)
from microscopy_vae.losses.composer import HQCodecLossComposer
from microscopy_vae.losses.influence import diagnose_generator_influence, scalar_contrib_ratios
from microscopy_vae.losses.perceptual import (
    adapt_vgg_input,
    build_internal_conv_extractor,
    perceptual_feature_loss,
)
from microscopy_vae.losses.schedule import scheduled_weight
from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.models.posterior import PosteriorStats
from microscopy_vae.models.vae import VAEOutput
from microscopy_vae.systems.factory import build_hq_codec_system


def _tiny_vae():
    return ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
    )


def _out(pred: torch.Tensor) -> VAEOutput:
    b, _, h, w = pred.shape
    z_h, z_w = max(h // 4, 1), max(w // 4, 1)
    post = PosteriorStats(
        mean=torch.zeros(b, 4, z_h, z_w),
        logvar=torch.zeros(b, 4, z_h, z_w),
    )
    return VAEOutput(reconstruction=pred, latent=post.mean, posterior=post)


def test_v21_yaml_keeps_new_losses_off():
    path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "s1_hq_f4z4_v2_1.yaml"
    cfg = load_config(path)
    assert cfg.loss.perceptual.enabled is False
    assert cfg.loss.adversarial.enabled is False
    assert cfg.loss.influence.grad_every_steps == 0


def test_default_root_config_new_losses_off():
    cfg = RootConfig()
    assert cfg.loss.perceptual.enabled is False
    assert cfg.loss.adversarial.enabled is False
    assert cfg.training.warmstart_vae_path is None


def test_scheduled_weight_matches_ms_ssim_formula():
    assert scheduled_weight(0.12, 999, 1000, 500) == 0.0
    assert scheduled_weight(0.12, 1000, 1000, 500) == 0.12 * (1.0 / 500.0)
    assert scheduled_weight(0.12, 1499, 1000, 500) == 0.12
    assert scheduled_weight(0.12, 2000, 1000, 0) == 0.12


def test_disabled_composer_total_identical_to_legacy_keys():
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    tgt = torch.randn(2, 1, 32, 32)
    kw = dict(
        w_char=1.0,
        w_ms_ssim=0.0,
        w_grad=0.08,
        w_flux=0.01,
        w_hf=0.05,
        free_nats=0.0,
        beta_max=0.0,
        kl_t0=0,
        kl_t1=1,
        amp_norm=True,
        amp_norm_min_scale=0.2,
        edge_weight=0.75,
        edge_weight_clip=3.0,
        structure_support_kernel=9,
        structure_min_frac=0.0003,
    )
    a = HQCodecLossComposer(**kw)(_out(pred), tgt, optimizer_step=0)
    b = HQCodecLossComposer(**kw, perceptual=None, perc_weight=0.05)(_out(pred), tgt, optimizer_step=0)
    assert "perceptual" not in a.unweighted
    assert "perceptual" not in b.unweighted
    assert torch.allclose(a.total, b.total)
    assert set(a.unweighted) == set(b.unweighted)


def test_internal_conv_accepts_signed_and_gt_one_no_clamp():
    ext = build_internal_conv_extractor(channels=(8, 16), freeze=True, init_seed=0)
    x = torch.tensor([[[[-2.0, 0.5], [1.5, 3.0]]]]).repeat(1, 1, 16, 16)
    x = x.clone().requires_grad_(True)
    feats = ext.forward_features(x)
    assert "block1" in feats
    # values after conv are not a clamp of the input; input still has negatives
    assert float(x.detach().min()) < 0
    assert float(x.detach().max()) > 1
    loss = feats["block1"].abs().mean()
    loss.backward()
    assert x.grad is not None
    for p in ext.parameters():
        assert p.requires_grad is False
        assert p.grad is None


def test_vgg_adapter_no_silent_clamp():
    x = torch.tensor([[[[-0.4, 1.7]]]])
    y = adapt_vgg_input(
        x,
        repeat_channels=True,
        clamp_to_unit=False,
        data_mean=0.0,
        data_std=1.0,
        imagenet_mean=(0.0, 0.0, 0.0),
        imagenet_std=(1.0, 1.0, 1.0),
    )
    assert y.shape[1] == 3
    # without clamp, extrema survive affine (ImageNet mean 0 std 1)
    assert float(y.min()) < 0
    assert float(y.max()) > 1
    z = adapt_vgg_input(
        x,
        repeat_channels=True,
        clamp_to_unit=True,
        data_mean=0.0,
        data_std=1.0,
        imagenet_mean=(0.0, 0.0, 0.0),
        imagenet_std=(1.0, 1.0, 1.0),
    )
    assert float(z.min()) >= 0
    assert float(z.max()) <= 1


def test_perceptual_loss_single_channel_and_frozen():
    ext = build_internal_conv_extractor(channels=(8, 8), freeze=True, init_seed=1)
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    tgt = torch.randn(2, 1, 32, 32)
    loss, diag = perceptual_feature_loss(
        pred, tgt, ext, selected_layers=["block1", "block2"], distance="l1"
    )
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert all(p.grad is None for p in ext.parameters())
    assert float(diag["perc_clamped"]) == 0.0


def test_system_parameters_exclude_perceptual():
    cfg = RootConfig(
        experiment={"output_dir": "runs/unused", "allow_existing_output": True},
        data={"mode": "synthetic", "synthetic_size": 64},
        crop={"size": 64},
        model={
            "encoder_block_out_channels": [32, 64, 64],
            "decoder_block_out_channels": [32, 64, 64],
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "mid_block_add_attention": False,
        },
        loss={
            "perceptual": {
                "enabled": True,
                "weight": 0.05,
                "ramp_steps": 0,
                "channels": [8, 8],
                "selected_layers": ["block1", "block2"],
            }
        },
        precision={"amp_dtype": "fp32"},
        memory={"gradient_checkpointing": False},
    )
    sys = build_hq_codec_system(cfg)
    assert sys.perceptual is not None
    g_ids = {id(p) for p in sys.parameters()}
    perc_ids = {id(p) for p in sys.perceptual.parameters()}
    assert perc_ids.isdisjoint(g_ids)
    assert all(not p.requires_grad for p in sys.perceptual.parameters())


def test_perc_start_and_ramp():
    ext = build_internal_conv_extractor(channels=(8,), freeze=True, init_seed=0)
    pred = torch.randn(1, 1, 32, 32, requires_grad=True)
    tgt = torch.randn(1, 1, 32, 32)
    comp = HQCodecLossComposer(
        w_ms_ssim=0,
        w_grad=0,
        w_flux=0,
        w_hf=0,
        beta_max=0,
        free_nats=0,
        perceptual=ext,
        perc_weight=0.1,
        perc_start_step=10,
        perc_ramp_steps=10,
        perc_layers=["block1"],
    )
    z = comp(_out(pred), tgt, optimizer_step=0)
    assert z.weights["perceptual"] == 0.0
    assert "perceptual" in z.unweighted
    mid = comp(_out(pred), tgt, optimizer_step=10)
    assert 0 < mid.weights["perceptual"] < 0.1
    late = comp(_out(pred), tgt, optimizer_step=30)
    assert abs(late.weights["perceptual"] - 0.1) < 1e-12


def test_gan_detach_does_not_grad_vae():
    vae = _tiny_vae()
    disc = PatchDiscriminator(in_channels=1, ndf=8, n_layers=2, spectral_norm=True)
    x = torch.randn(2, 1, 32, 32)
    out = vae(x, sample_posterior=False)
    scores = discriminator_scores(
        disc, real=x.detach(), fake=out.reconstruction.detach(), gan_loss="hinge"
    )
    scores["loss_d"].backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in vae.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in disc.parameters())


def test_generator_adv_does_not_fill_disc_grad():
    disc = PatchDiscriminator(in_channels=1, ndf=8, n_layers=2, spectral_norm=False)
    fake = torch.randn(2, 1, 32, 32, requires_grad=True)
    disc.requires_grad_(False)
    g_loss, _ = generator_adv_loss(disc, fake, gan_loss="hinge")
    g_loss.backward()
    assert fake.grad is not None
    assert all(p.grad is None for p in disc.parameters())


def test_disc_loss_not_in_generator_total():
    ext = build_internal_conv_extractor(channels=(8,), freeze=True, init_seed=0)
    pred = torch.randn(1, 1, 32, 32, requires_grad=True)
    tgt = torch.randn(1, 1, 32, 32)
    comp = HQCodecLossComposer(
        w_ms_ssim=0, w_grad=0, w_flux=0, beta_max=0, free_nats=0, perceptual=ext, perc_weight=0.05, perc_layers=["block1"]
    )
    out = comp(_out(pred), tgt, optimizer_step=0)
    disc = PatchDiscriminator(ndf=8, n_layers=2)
    scores = discriminator_scores(disc, real=tgt.detach(), fake=pred.detach(), gan_loss="hinge")
    assert "adv_g" not in out.unweighted
    # D loss is a separate tensor; adding it would change G total — we don't.
    assert scores["loss_d"].ndim == 0
    assert out.total is not scores["loss_d"]


def test_gan_start_step_weight_zero():
    assert scheduled_weight(0.02, 100, 5000, 2000) == 0.0
    assert scheduled_weight(0.02, 5000, 5000, 2000) > 0.0


def test_old_checkpoint_loads_vae_only(tmp_path):
    vae = _tiny_vae()
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    ckpt = CheckpointManager(tmp_path)
    path = ckpt.save_exact(
        tag="legacy",
        model=vae,
        optimizer=opt,
        scheduler=None,
        scaler=None,
        state=TrainerState(),
        config_sha256="abc",
        normalizer_sha256="def",
        code_version="0.2.3",
        extra={"ema": None},
    )
    vae2 = _tiny_vae()
    CheckpointManager.load_exported_weights(path, vae2)
    s1 = vae.state_dict()
    s2 = vae2.state_dict()
    assert all(torch.allclose(s1[k], s2[k]) for k in s1)


def test_gan_checkpoint_roundtrip_two_optimizers(tmp_path):
    vae = _tiny_vae()
    disc = PatchDiscriminator(ndf=8, n_layers=2, spectral_norm=False)
    g_opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    d_opt = torch.optim.AdamW(disc.parameters(), lr=2e-4)
    x = torch.randn(2, 1, 32, 32)
    fake = vae(x, sample_posterior=False).reconstruction
    d_loss = discriminator_scores(disc, real=x, fake=fake.detach(), gan_loss="hinge")["loss_d"]
    d_loss.backward()
    d_opt.step()
    g_loss = fake.mean()
    g_loss.backward()
    g_opt.step()
    ckpt = CheckpointManager(tmp_path)
    path = ckpt.save_exact(
        tag="gan",
        model=vae,
        optimizer=g_opt,
        scheduler=None,
        scaler=None,
        state=TrainerState(optimizer_step=7),
        config_sha256="c",
        normalizer_sha256="n",
        code_version="0.2.4",
        extra={
            "discriminator": disc.state_dict(),
            "disc_optimizer": d_opt.state_dict(),
            "gan_enabled": True,
        },
    )
    vae_b = _tiny_vae()
    disc_b = PatchDiscriminator(ndf=8, n_layers=2, spectral_norm=False)
    g_opt_b = torch.optim.AdamW(vae_b.parameters(), lr=1e-3)
    d_opt_b = torch.optim.AdamW(disc_b.parameters(), lr=2e-4)
    state, extra = CheckpointManager.resume_exact(
        path,
        model=vae_b,
        optimizer=g_opt_b,
        scheduler=None,
        scaler=None,
        expected_config_sha256="c",
        expected_normalizer_sha256="n",
        verify_sidecar_hash=True,
    )
    disc_b.load_state_dict(extra["discriminator"])
    d_opt_b.load_state_dict(extra["disc_optimizer"])
    assert state.optimizer_step == 7
    for a, b in zip(disc.parameters(), disc_b.parameters()):
        assert torch.allclose(a, b)


def test_influence_does_not_write_param_grad_or_change_step():
    vae = _tiny_vae()
    x = torch.randn(2, 1, 32, 32)
    out = vae(x, sample_posterior=True)
    comp = HQCodecLossComposer(w_ms_ssim=0, w_grad=0.05, w_flux=0, beta_max=1e-3, kl_t0=0, kl_t1=1, free_nats=0)
    loss_out = comp(out, x, optimizer_step=0)
    before = {n: p.detach().clone() for n, p in vae.named_parameters()}
    assert all(p.grad is None for p in vae.parameters())
    logs = diagnose_generator_influence(loss_out.weighted, vae, param_group_names=("output", "posterior"))
    assert all(p.grad is None for p in vae.parameters()), "diagnostics must not write .grad"
    opt = torch.optim.SGD(vae.parameters(), lr=0.0)
    loss_out.total.backward()
    opt.step()
    after = {n: p.detach().clone() for n, p in vae.named_parameters()}
    # lr=0 → params unchanged even after real backward
    for n in before:
        assert torch.allclose(before[n], after[n])
    assert "grad_norm_output_charbonnier" in logs
    assert "grad_norm_posterior_kl" in logs


def test_influence_two_copies_same_update():
    """Diagnose on / off must not change the actual parameter update (same seed)."""

    def one_step(with_diag: bool) -> dict:
        torch.manual_seed(123)
        vae = _tiny_vae()
        x = torch.randn(2, 1, 32, 32)
        torch.manual_seed(123)
        out = vae(x, sample_posterior=True)
        comp = HQCodecLossComposer(
            w_ms_ssim=0, w_grad=0.05, w_flux=0.01, beta_max=1e-3, kl_t0=0, kl_t1=1, free_nats=0
        )
        loss_out = comp(out, x, optimizer_step=0)
        if with_diag:
            diagnose_generator_influence(loss_out.weighted, vae, param_group_names=("output",))
        opt = torch.optim.SGD(vae.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        loss_out.total.backward()
        opt.step()
        return {n: p.detach().clone() for n, p in vae.named_parameters()}

    a = one_step(False)
    b = one_step(True)
    for n in a:
        assert torch.allclose(a[n], b[n], atol=0.0, rtol=0.0), n


def test_scalar_ratio_uses_abs_for_negative_term():
    t = {
        "charbonnier": torch.tensor(1.0),
        "adv_g": torch.tensor(-1.0),
    }
    r = scalar_contrib_ratios(t)
    assert abs(r["charbonnier"] - 0.5) < 1e-6
    assert abs(r["adv_g"] - 0.5) < 1e-6


def test_quantify_fills_all_terms_and_share_sums_100():
    from microscopy_vae.losses.influence import GENERATOR_TERM_ORDER, quantify_generator_losses

    logs = quantify_generator_losses(
        {"charbonnier": torch.tensor(2.0), "kl": torch.tensor(1.0)},
        {"charbonnier": 1.0, "kl": 0.5},
        {"charbonnier": torch.tensor(2.0), "kl": torch.tensor(0.5)},
        total=torch.tensor(2.5),
    )
    for name in GENERATOR_TERM_ORDER:
        assert f"loss_raw_{name}" in logs
        assert f"weight_{name}" in logs
        assert f"loss_w_{name}" in logs
        assert f"share_pct_{name}" in logs
    assert logs["loss_raw_perceptual"] == 0.0
    assert logs["loss_w_adv_g"] == 0.0
    s = sum(logs[f"share_pct_{n}"] for n in GENERATOR_TERM_ORDER)
    assert abs(s - 100.0) < 1e-4
    assert abs(logs["share_pct_charbonnier"] - 80.0) < 1e-6
    assert abs(logs["share_pct_kl"] - 20.0) < 1e-6


def test_amp_and_finite_with_signed_input():
    vae = _tiny_vae()
    x = torch.randn(2, 1, 32, 32) * 3.0 - 0.5  # negatives and >1
    ext = build_internal_conv_extractor(channels=(8,), freeze=True, init_seed=0)
    disc = PatchDiscriminator(ndf=8, n_layers=2, spectral_norm=True)
    out = vae(x, sample_posterior=True)
    comp = HQCodecLossComposer(
        w_ms_ssim=0,
        w_grad=0.05,
        w_hf=0.05,
        amp_norm=True,
        perceptual=ext,
        perc_weight=0.05,
        perc_layers=["block1"],
        perc_start_step=0,
        perc_ramp_steps=0,
        free_nats=0,
        beta_max=1e-4,
        kl_t0=0,
        kl_t1=1,
    )
    loss_out = comp(out, x, optimizer_step=0)
    disc.requires_grad_(False)
    g_adv, _ = generator_adv_loss(disc, loss_out.aux["reconstruction"], gan_loss="hinge")
    total = loss_out.total + 0.02 * g_adv
    assert torch.isfinite(total)
    total.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in vae.parameters()
    )


def test_perc_and_gan_actually_move_vae():
    """Both losses must be in the graph and change VAE (not the frozen extractor)."""
    torch.manual_seed(0)
    vae = _tiny_vae()
    ext = build_internal_conv_extractor(channels=(8, 8), freeze=True, init_seed=0)
    disc = PatchDiscriminator(in_channels=1, ndf=8, n_layers=2, spectral_norm=True)
    x = torch.randn(2, 1, 32, 32)
    out = vae(x, sample_posterior=True)
    comp = HQCodecLossComposer(
        w_ms_ssim=0,
        w_grad=0,
        w_flux=0,
        w_hf=0,
        beta_max=0,
        free_nats=0,
        perceptual=ext,
        perc_weight=0.05,
        perc_start_step=0,
        perc_ramp_steps=0,
        perc_layers=["block1", "block2"],
    )
    loss_out = comp(out, x, optimizer_step=0)
    assert float(loss_out.unweighted["perceptual"].detach()) > 0
    assert float(loss_out.weights["perceptual"]) == 0.05
    disc.requires_grad_(False)
    g_adv, _ = generator_adv_loss(disc, loss_out.aux["reconstruction"], gan_loss="hinge")
    total = loss_out.total + 0.02 * g_adv
    before_g = {n: p.detach().clone() for n, p in vae.named_parameters()}
    before_p = {n: p.detach().clone() for n, p in ext.named_parameters()}
    before_d = {n: p.detach().clone() for n, p in disc.named_parameters()}
    g_opt = torch.optim.SGD(vae.parameters(), lr=0.1)
    d_opt = torch.optim.SGD(disc.parameters(), lr=0.1)
    disc.requires_grad_(True)
    d_scores = discriminator_scores(
        disc, real=x.detach(), fake=out.reconstruction.detach(), gan_loss="hinge"
    )
    d_opt.zero_grad(set_to_none=True)
    d_scores["loss_d"].backward()
    d_opt.step()
    disc.requires_grad_(False)
    g_opt.zero_grad(set_to_none=True)
    total.backward()
    g_opt.step()
    assert any(not torch.equal(before_g[n], p.detach()) for n, p in vae.named_parameters())
    assert all(torch.equal(before_p[n], p.detach()) for n, p in ext.named_parameters())
    assert any(not torch.equal(before_d[n], p.detach()) for n, p in disc.named_parameters())


def test_disc_mask_keeps_thin_filament():
    from microscopy_vae.losses.adversarial import _spatial_mask

    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, :, 10] = 1.0  # 1-pixel-wide vertical filament
    pooled = _spatial_mask(mask, (8, 8))
    assert pooled.shape[-2:] == (8, 8)
    assert float(pooled.sum()) > 0


def test_v22_yaml_enables_both_losses():
    root = Path(__file__).resolve().parents[2] / "configs" / "experiment"
    v21 = load_config(root / "s1_hq_f4z4_v2_1.yaml")
    v22 = load_config(root / "s1_hq_f4z4_v2_2.yaml")
    assert v21.loss.perceptual.enabled is False
    assert v21.loss.adversarial.enabled is False
    assert v22.loss.perceptual.enabled is True
    assert v22.loss.adversarial.enabled is True
    assert v22.loss.perceptual.backbone == "internal_conv"
    assert v22.loss.perceptual.vgg_clamp_to_unit is False
    assert v22.loss.adversarial.conditioning == "none"
    assert v22.experiment.output_dir != v21.experiment.output_dir
    assert v22.loss.perceptual.start_step == 1000
    assert v22.loss.adversarial.start_step == 5000
    assert v22.loss.influence.cosine_every_steps == 0
    assert v22.loss.influence.param_groups == ["full"]


def test_ablation_yamls_do_not_enable_on_v21():
    root = Path(__file__).resolve().parents[2] / "configs" / "experiment"
    v21 = load_config(root / "s1_hq_f4z4_v2_1.yaml")
    assert v21.loss.perceptual.enabled is False
    assert v21.loss.adversarial.enabled is False
    a = load_config(root / "ablation_mod3" / "A_baseline_short.yaml")
    b = load_config(root / "ablation_mod3" / "B_perceptual_short.yaml")
    c = load_config(root / "ablation_mod3" / "C_gan_short.yaml")
    d = load_config(root / "ablation_mod3" / "D_perc_gan_short.yaml")
    e = load_config(root / "ablation_mod3" / "E_rebalanced_short.yaml")
    assert a.loss.perceptual.enabled is False and a.loss.adversarial.enabled is False
    assert b.loss.perceptual.enabled is True and b.loss.adversarial.enabled is False
    assert c.loss.perceptual.enabled is False and c.loss.adversarial.enabled is True
    assert d.loss.perceptual.enabled is True and d.loss.adversarial.enabled is True
    assert e.loss.perceptual.weight < d.loss.perceptual.weight
    assert e.loss.adversarial.weight < d.loss.adversarial.weight
    assert a.experiment.output_dir != v21.experiment.output_dir
    assert a.training.max_steps == 2000
    assert c.loss.adversarial.conditioning == "none"
    assert b.loss.perceptual.backbone == "internal_conv"
    assert b.loss.perceptual.vgg_clamp_to_unit is False
