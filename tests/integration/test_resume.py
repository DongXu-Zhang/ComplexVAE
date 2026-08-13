from pathlib import Path

import torch

from microscopy_vae.config.loader import load_config
from microscopy_vae.engine.checkpoint import CheckpointManager
from microscopy_vae.engine.trainer import Trainer
from microscopy_vae import __version__
from microscopy_vae.engine.state import TrainerState


def test_checkpoint_roundtrip(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_synthetic.yaml"
    cfg = load_config(cfg_path, overrides={"experiment": {"output_dir": str(tmp_path / "run")}})
    trainer = Trainer(cfg)
    trainer.train(max_steps=1)
    # save again via manager
    path = trainer.ckpt.save_exact(
        tag="manual",
        model=trainer.system.vae,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        state=trainer.state,
        config_sha256=trainer.config_sha,
        normalizer_sha256=trainer.normalizer_sha,
        code_version=__version__,
    )
    # mutate weights
    with torch.no_grad():
        for p in trainer.system.vae.parameters():
            p.zero_()
    CheckpointManager.resume_exact(
        path,
        model=trainer.system.vae,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        expected_config_sha256=trainer.config_sha,
        expected_normalizer_sha256=trainer.normalizer_sha,
    )  # returns (state, extra); load is enough for this test
    # weights should be nonzero again somewhere
    total = sum(p.abs().sum().item() for p in trainer.system.vae.parameters())
    assert total > 0


def test_resume_rejects_config_drift(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_synthetic.yaml"
    cfg = load_config(cfg_path, overrides={"experiment": {"output_dir": str(tmp_path / "run")}})
    trainer = Trainer(cfg)
    path = trainer.ckpt.save_exact(
        tag="t",
        model=trainer.system.vae,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        state=TrainerState(),
        config_sha256=trainer.config_sha,
        normalizer_sha256=trainer.normalizer_sha,
        code_version=__version__,
    )
    import pytest

    with pytest.raises(ValueError, match="config_sha256"):
        CheckpointManager.resume_exact(
            path,
            model=trainer.system.vae,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=None,
            expected_config_sha256="deadbeef",
            expected_normalizer_sha256=trainer.normalizer_sha,
            verify_sidecar_hash=False,
        )
