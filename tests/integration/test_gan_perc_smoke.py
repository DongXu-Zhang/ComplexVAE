"""Smoke: both new losses off vs on, resume of two optimizers, EMA on VAE only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from microscopy_vae.config.loader import load_config
from microscopy_vae.engine.checkpoint import CheckpointManager
from microscopy_vae.engine.trainer import Trainer


def test_smoke_v4_train_no_test_loader(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_v4.yaml"
    cfg = load_config(cfg_path, overrides={"experiment": {"output_dir": str(tmp_path / "run")}})
    trainer = Trainer(cfg)
    assert not hasattr(trainer, "test_loader")
    assert cfg.loss.w_grad == 0.0
    result = trainer.train(max_steps=2)
    assert result["final_step"] == 2
    assert np.isfinite(float(result["final_loss"]))


def test_smoke_gan_perc_train_and_resume(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_gan_perc.yaml"
    cfg = load_config(cfg_path, overrides={"experiment": {"output_dir": str(tmp_path / "run")}})
    trainer = Trainer(cfg)
    assert trainer.discriminator is not None
    assert trainer.system.perceptual is not None
    assert all(not p.requires_grad for p in trainer.system.perceptual.parameters())
    perc_before = {n: p.detach().clone() for n, p in trainer.system.perceptual.named_parameters()}
    ema_keys = set(trainer.ema.shadow) if trainer.ema else set()
    assert ema_keys  # EMA on VAE
    disc_ids = {id(p) for p in trainer.discriminator.parameters()}
    g_ids = {id(p) for p in trainer.optimizer.param_groups[0]["params"]}
    assert disc_ids.isdisjoint(g_ids)
    result = trainer.train(max_steps=2)
    assert result["final_step"] == 2
    for n, p in trainer.system.perceptual.named_parameters():
        assert torch.allclose(perc_before[n], p), "frozen perc must not move"
    ckpt = Path(result["checkpoint"])
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location="cpu")
    extra = payload["extra"]
    assert extra.get("discriminator") is not None
    assert extra.get("disc_optimizer") is not None
    assert extra.get("perceptual") is not None
    assert extra.get("ema") is not None
    # exact resume on the same objects (config hash matches this run)
    with torch.no_grad():
        for p in trainer.system.vae.parameters():
            p.zero_()
        for p in trainer.discriminator.parameters():
            p.zero_()
    state, extra2 = CheckpointManager.resume_exact(
        ckpt,
        model=trainer.system.vae,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        expected_config_sha256=trainer.config_sha,
        expected_normalizer_sha256=trainer.normalizer_sha,
    )
    trainer.discriminator.load_state_dict(extra2["discriminator"])
    trainer.disc_optimizer.load_state_dict(extra2["disc_optimizer"])
    trainer.system.perceptual.load_state_dict(extra2["perceptual"])
    assert state.optimizer_step == 2
    assert sum(float(p.detach().abs().sum()) for p in trainer.system.vae.parameters()) > 0
    assert sum(float(p.detach().abs().sum()) for p in trainer.discriminator.parameters()) > 0


def test_smoke_baseline_has_no_disc(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_synthetic.yaml"
    cfg = load_config(cfg_path, overrides={"experiment": {"output_dir": str(tmp_path / "base")}})
    trainer = Trainer(cfg)
    assert trainer.discriminator is None
    assert trainer.system.perceptual is None
    trainer.train(max_steps=1)
    # jsonl must not require GAN keys
    recs = (tmp_path / "base" / "metrics_train.jsonl").read_text(encoding="utf-8")
    assert "loss_raw_disc" not in recs
