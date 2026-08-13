"""Lazy single-page readers for MRC / TIFF (HQ float32, WF uint16)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _assert_finite(arr: np.ndarray, path: Path, page: int) -> None:
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"Non-finite pixels in {path} page={page}: count={bad}")


def read_page(
    path: Path,
    page_index: int,
    *,
    expected_dtype: str | None = None,
) -> Tuple[np.ndarray, dict]:
    """Read one 2D page as float32 array [H,W] plus metadata.

    Shape rules:
    - file may be [H,W] or [P,H,W]
    - page_index selects P; never treat P as channel
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    meta: dict = {"path_suffix": suffix, "page_index": page_index}

    if suffix in {".mrc", ".map", ".rec"}:
        import mrcfile

        with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
            data = np.asarray(mrc.data)
            meta["mrc_dtype"] = str(data.dtype)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            n = len(tif.pages)
            meta["n_pages_file"] = n
            if page_index < 0 or page_index >= n:
                # some multi-page stored as series array
                arr = tif.asarray()
                if arr.ndim == 2:
                    if page_index != 0:
                        raise IndexError(f"page_index={page_index} but file is 2D")
                    data = arr
                elif arr.ndim == 3:
                    if page_index >= arr.shape[0]:
                        raise IndexError(f"page_index={page_index} out of range {arr.shape[0]}")
                    data = arr[page_index]
                else:
                    raise ValueError(f"Unsupported TIFF ndim={arr.ndim} for {path}")
            else:
                data = tif.pages[page_index].asarray()
    else:
        raise ValueError(f"Unsupported image suffix {suffix} for {path}")

    data = np.asarray(data)
    if data.ndim == 3:
        # [P,H,W]
        if page_index < 0 or page_index >= data.shape[0]:
            raise IndexError(f"page_index={page_index} out of range for shape {data.shape}")
        page = data[page_index]
    elif data.ndim == 2:
        if page_index != 0 and suffix in {".mrc", ".map", ".rec"}:
            raise IndexError(f"2D MRC page_index must be 0, got {page_index}")
        page = data
    else:
        raise ValueError(f"Expected 2D page or 3D stack, got shape {data.shape}")

    if page.ndim != 2:
        raise ValueError(f"Page must be 2D [H,W], got {page.shape}")

    raw_dtype = str(page.dtype)
    meta["raw_dtype"] = raw_dtype
    meta["page_shape"] = [int(page.shape[0]), int(page.shape[1])]

    if expected_dtype is not None:
        # soft check: logical role dtypes
        if expected_dtype == "float32" and page.dtype not in (np.float32, np.float64):
            # allow cast with warning metadata
            meta["dtype_cast"] = f"{page.dtype}->float32"
        if expected_dtype == "uint16" and page.dtype != np.uint16:
            meta["dtype_cast"] = f"{page.dtype}->float32_from_uint_like"

    page_f = page.astype(np.float32, copy=False)
    _assert_finite(page_f, path, page_index)
    return page_f, meta
