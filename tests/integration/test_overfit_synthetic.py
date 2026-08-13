from pathlib import Path

from microscopy_vae.config.loader import load_config
from microscopy_vae.engine.trainer import Trainer


def test_overfit_loss_decreases(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "overfit_hq.yaml"
    cfg = load_config(
        cfg_path,
        overrides={
            "experiment": {"output_dir": str(tmp_path / "run")},
            "training": {"max_steps": 80, "overfit_n_patches": 4},
        },
    )
    trainer = Trainer(cfg)
    result = trainer.overfit_small()
    assert result["final_loss"] < result["initial_loss"]
