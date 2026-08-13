"""Self-test for tools/build_hq_manifest.py field alignment + synthetic discovery."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "build_hq_manifest.py"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    mrcfile = pytest.importorskip("mrcfile")
    tifffile = pytest.importorskip("tifffile")
    root = tmp_path_factory.mktemp("dataset")
    # BioTISR
    for i in range(4):
        cell = root / "BioTISR" / "Lifeact" / f"Cell_{i:03d}"
        cell.mkdir(parents=True)
        arr = (np.random.randn(3, 32, 32)).astype(np.float32)
        with mrcfile.new(str(cell / "SIM_gt.mrc"), overwrite=True) as mrc:
            mrc.set_data(arr)
        (cell / "GT_all.mrc").write_bytes(b"nope")  # should exclude by name if scanned
    # decoy GT_all at category level
    (root / "BioTISR" / "Lifeact" / "GT_all.mrc").write_bytes(b"x" * 10)
    # DI2D
    for i in range(3):
        d = root / "DeepInsight_2D_training_data" / "MT" / "150" / f"Exp{i}"
        d.mkdir(parents=True)
        tifffile.imwrite(str(d / f"RC_{i:04d}_highsnr.tif"), np.random.randn(32, 32).astype(np.float32))
        tifffile.imwrite(str(d / f"WF_{i:04d}_lowsnr.tif"), np.zeros((16, 16), np.uint16))
        tifffile.imwrite(str(d / f"GTdenoised_{i:04d}.tif"), np.random.randn(32, 32).astype(np.float32))
    # DI3D
    for i in range(2):
        d = root / "DeepInsight_3D_training_data" / "raw_tif_fullsize" / "ER" / "300" / f"vol{i}"
        d.mkdir(parents=True)
        tifffile.imwrite(str(d / "RC_highsnr.tif"), np.random.randn(2, 32, 32).astype(np.float32))
    return root


def test_tool_self_test_cli():
    r = subprocess.run(
        [sys.executable, str(TOOL), "--self-test"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "self-test OK" in r.stdout or "structural write OK" in r.stdout


def test_generator_fields_load_in_package(synthetic_root, tmp_path):
    out = tmp_path / "hq.jsonl"
    r = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--root",
            str(synthetic_root),
            "--out",
            str(out),
            "--seed",
            "0",
            "--no-holdout-test",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.is_file()
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert lines
    # required fields
    for rec in lines:
        for k in (
            "split",
            "source_dataset",
            "biological_group_id",
            "path",
            "shape",
            "page_index",
        ):
            assert k in rec
        assert rec["split"] in {"train", "val", "test"}
        assert "group_id" not in rec or "biological_group_id" in rec

    # must not include WF / GTdenoised
    for rec in lines:
        name = Path(rec["path"]).name.lower()
        assert not name.startswith("wf_")
        assert "gtdenoised" not in name
        assert "gt_all" not in name

    # package loader
    sys.path.insert(0, str(SRC))
    from microscopy_vae.data.manifest import load_hq_manifest

    loaded = load_hq_manifest(out, allow_splits=("train", "val"), refuse_test=True)
    assert len(loaded) > 0
    # Bio pages: 4 cells * 3 pages = 12; DI2D 3; DI3D 2*2=4 → total 19 if all train/val
    assert len(loaded) >= 10
