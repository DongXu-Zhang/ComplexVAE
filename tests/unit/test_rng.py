from microscopy_vae.utils.rng import derive_seed
from microscopy_vae.provenance.hashing import stable_sample_id


def test_stable_ids_not_builtin_hash():
    a = stable_sample_id("g1", "0")
    b = stable_sample_id("g1", "0")
    c = stable_sample_id("g1", "1")
    assert a == b
    assert a != c
    assert derive_seed(0, 1, 2) == derive_seed(0, 1, 2)
