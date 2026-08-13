from pathlib import Path

import pytest

from microscopy_vae.data.manifest import load_hq_manifest, summarize_records
from microscopy_vae.data.pathmap import PathPrefixMap, apply_prefix_map_to_records
from microscopy_vae.data.records import HQPageRecord


# Optional large inventory for offline hosts only (not shipped in git).
# Set MICROVAE_HQ_MANIFEST=/path/to/hq_manifest_v2.jsonl to enable.
_MANIFEST_ENV = "MICROVAE_HQ_MANIFEST"
MANIFEST = Path(__import__("os").environ[_MANIFEST_ENV]) if __import__("os").environ.get(_MANIFEST_ENV) else None
EXPECTED_SHA = "7285a66d9b89b3410b70327d15e656bbb70df926c13fccf82061b8ee3ec50734"


@pytest.mark.skipif(MANIFEST is None or not MANIFEST.is_file(), reason="set MICROVAE_HQ_MANIFEST to run")
def test_load_real_manifest_train_val_no_test():
    import hashlib

    assert MANIFEST is not None
    h = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert h == EXPECTED_SHA
    recs = load_hq_manifest(MANIFEST, allow_splits=("train", "val"), refuse_test=True)
    s = summarize_records(recs)
    assert s["n_pages"] == 35530 + 4494
    assert s["n_groups"] == 325 + 40 + 2126 + 266 + 1372 + 172
    assert all(r.split in ("train", "val") for r in recs)
    # DI3D dominates pages but groups are multi-source
    assert s["by_source"]["DeepInsight_3D"] == 27432 + 3440


def test_path_prefix_map_windows_to_linux():
    rec = HQPageRecord(
        sample_id="x",
        split="train",
        source="BioTISR",
        category="CCPs_488",
        condition="hq",
        morphology="puncta",
        group_id="g",
        hq_path=Path(r"F:\Dataset\BioTISR\CCPs_488\Cell_001\SIM_gt.mrc"),
        hq_page=0,
        hq_page_shape=(1024, 1024),
        hq_dtype="float32",
        target_role="SIM_gt",
        is_derived=True,
    )
    pmap = PathPrefixMap(
        source_prefixes=(r"F:\Dataset", "F:/Dataset"),
        target_root="/mnt/Dataset",
        require_exists=False,
    )
    out = apply_prefix_map_to_records([rec], pmap)[0]
    assert out.hq_path == Path("/mnt/Dataset/BioTISR/CCPs_488/Cell_001/SIM_gt.mrc")
    assert out.sample_id == "x"
