from collections import Counter

from microscopy_vae.data.samplers import HierarchicalIndexSampler
from microscopy_vae.data.synthetic import build_synthetic_hq_pool
from microscopy_vae.data.hq_dataset import SyntheticHQDataset
from microscopy_vae.data.normalization import fit_robust_normalizer, Normalizer


def test_hierarchical_sampler_balances_sources():
    pages = build_synthetic_hq_pool(n_groups=8, pages_per_group=2, size=64, seed=0)
    train = [p for p in pages if p.split == "train"]
    state = fit_robust_normalizer([p.image for p in train], method="identity")
    ds = SyntheticHQDataset(pages, split="train", crop_size=64, normalizer=Normalizer(state), seed=0)
    samp = HierarchicalIndexSampler(ds.meta, seed=0, epoch_length=2000)
    idxs = list(iter(samp))
    assert len(idxs) == 2000
    sources = [ds.meta[i]["source"] for i in idxs]
    # both sources should appear
    c = Counter(sources)
    assert len(c) >= 2
    # neither source should be completely starved
    for s, n in c.items():
        assert n > 200


def test_group_split_leak_detection(tmp_path):
    import json
    from microscopy_vae.data.manifest import load_hq_manifest
    import pytest

    rows = [
        {
            "split": "train",
            "source_dataset": "A",
            "biological_group_id": "same",
            "path": str(tmp_path / "a.tif"),
            "shape": [32, 32],
            "page_index": 0,
        },
        {
            "split": "val",
            "source_dataset": "A",
            "biological_group_id": "same",
            "path": str(tmp_path / "b.tif"),
            "shape": [32, 32],
            "page_index": 0,
        },
    ]
    man = tmp_path / "leak.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="split leak"):
        load_hq_manifest(man, allow_splits=("train", "val"), refuse_test=True)
