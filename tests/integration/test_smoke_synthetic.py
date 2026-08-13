from pathlib import Path

import pytest

from microscopy_vae.config.loader import load_config
from microscopy_vae.engine.trainer import Trainer


@pytest.mark.slow
def test_smoke_train_two_steps(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_synthetic.yaml"
    cfg = load_config(cfg_path, overrides={"experiment": {"output_dir": str(tmp_path / "run")}})
    trainer = Trainer(cfg)
    info = trainer.dry_run()
    assert info["all_core_trainable"]
    result = trainer.train(max_steps=2)
    assert result["final_step"] == 2
    assert Path(result["checkpoint"]).is_file()
