import numpy as np

from microscopy_vae.data.normalization import Normalizer, fit_robust_normalizer


def test_roundtrip_and_train_only_semantics():
    rng = np.random.default_rng(0)
    arrs = [rng.normal(10, 2, size=(32, 32)).astype(np.float32) for _ in range(5)]
    state = fit_robust_normalizer(arrs, method="robust_linear_p0.1_p99.9", clip=False)
    assert state.fit_split == "train"
    norm = Normalizer(state)
    x = arrs[0]
    y = norm.transform(x)
    x2 = norm.inverse(y)
    assert np.allclose(x, x2, atol=1e-4)


def test_identity():
    state = fit_robust_normalizer([np.ones((4, 4))], method="identity")
    norm = Normalizer(state)
    x = np.array([[0.5]], dtype=np.float32)
    assert norm.transform(x)[0, 0] == 0.5
