import pytest
from pydantic import ValidationError

from microscopy_vae.config.schema import RootConfig
from microscopy_vae.config.loader import load_config
from microscopy_vae.data.hq_dataset import SyntheticHQDataset
from microscopy_vae.data.manifest import load_hq_manifest, parse_hq_record
from microscopy_vae.data.normalization import fit_robust_normalizer, Normalizer
from microscopy_vae.data.synthetic import build_synthetic_hq_pool
from pathlib import Path
import json
import numpy as np


def test_config_rejects_test_split():
    with pytest.raises(ValidationError):
        RootConfig(data={"allow_splits": ["train", "val", "test"]})


def test_dataset_refuses_test_split():
    pages = build_synthetic_hq_pool(n_groups=4, pages_per_group=1, size=64, seed=0)
    state = fit_robust_normalizer([p.image for p in pages if p.split == "train"], method="identity")
    norm = Normalizer(state)
    with pytest.raises(RuntimeError, match="test"):
        SyntheticHQDataset(pages, split="test", crop_size=64, normalizer=norm)


def test_manifest_skips_test_when_refuse(tmp_path):
    rows = []
    for split, gid in [("train", "g0"), ("val", "g1"), ("test", "g2")]:
        rows.append(
            {
                "split": split,
                "source_dataset": "SRC",
                "biological_group_id": gid,
                "path": str(tmp_path / f"{gid}.tif"),
                "shape": [64, 64],
                "page_index": 0,
                "dtype": "float32",
                "morphology": "puncta",
            }
        )
    man = tmp_path / "m.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    recs = load_hq_manifest(man, allow_splits=("train", "val"), refuse_test=True)
    assert all(r.split != "test" for r in recs)
    assert {r.split for r in recs} <= {"train", "val"}


def test_unknown_split_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        parse_hq_record(
            {
                "split": "validation",  # wrong token; must be val
                "source_dataset": "SRC",
                "biological_group_id": "g",
                "path": "/tmp/x.tif",
                "shape": [64, 64],
                "page_index": 0,
            }
        )


def test_trainer_has_no_test_loader(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_synthetic.yaml"
    cfg = load_config(cfg_path, overrides={"experiment": {"output_dir": str(tmp_path / "run")}})
    from microscopy_vae.engine.trainer import Trainer

    tr = Trainer(cfg)
    assert not hasattr(tr, "test_loader")
    info = tr.dry_run()
    assert info["has_test_loader"] is False
    assert info["capabilities"]["paired_restoration"] is False
