import numpy as np
from pathlib import Path

import tifffile

from microscopy_vae.data.readers import read_page


def test_read_tiff_2d_and_stack(tmp_path):
    p2 = tmp_path / "single.tif"
    arr = np.random.randn(48, 48).astype(np.float32)
    tifffile.imwrite(str(p2), arr)
    page, meta = read_page(p2, 0, expected_dtype="float32")
    assert page.shape == (48, 48)
    assert meta["page_shape"] == [48, 48]

    p3 = tmp_path / "stack.tif"
    stack = np.random.randn(5, 32, 40).astype(np.float32)
    tifffile.imwrite(str(p3), stack)
    page2, _ = read_page(p3, 2, expected_dtype="float32")
    assert page2.shape == (32, 40)
    np.testing.assert_allclose(page2, stack[2], rtol=0, atol=0)
