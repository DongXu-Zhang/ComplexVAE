import pytest
from pydantic import ValidationError

from microscopy_vae.config.loader import load_config
from microscopy_vae.config.schema import RootConfig


def test_default_config_ok():
    cfg = RootConfig()
    assert cfg.model.in_channels == 1


def test_reject_rgb():
    with pytest.raises(ValidationError):
        RootConfig(model={"in_channels": 3})


def test_reject_test_split():
    with pytest.raises(ValidationError):
        RootConfig(data={"allow_splits": ["train", "val", "test"]})


def test_reject_crop_not_divisible():
    with pytest.raises(ValidationError):
        RootConfig(crop={"size": 255}, model={"encoder_block_out_channels": [128, 256, 512, 512]})


def test_reject_allow_test():
    with pytest.raises(ValidationError):
        RootConfig(evaluation={"allow_test": True})


def test_load_smoke_yaml():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_synthetic.yaml"
    cfg = load_config(path)
    assert cfg.experiment.name == "smoke_synthetic"
