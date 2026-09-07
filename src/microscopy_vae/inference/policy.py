"""Choose full / halo / isolated-tile inference without changing reconstruction math."""

from __future__ import annotations

from typing import Optional, Tuple


def resolve_production_infer_mode(
    *,
    requested: str,
    height: int,
    width: int,
    tile_size: int,
    allow_isolated_tiles: bool,
    prefer_halo_over_isolated_tiles: bool = True,
) -> Tuple[str, Optional[str]]:
    """Map a user request onto a production mode.

    Isolated 256 tiles on a larger FOV change GroupNorm and bottleneck
    attention relative to the native page. Prefer halo (real neighbors) or
    full-image inference unless the caller explicitly opts into isolated tiles.
    """
    mode = str(requested)
    note: Optional[str] = None
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid inference size {(height, width)}")
    if mode == "tiled" and (height > int(tile_size) or width > int(tile_size)):
        if bool(prefer_halo_over_isolated_tiles) and not bool(allow_isolated_tiles):
            mode = "halo"
            note = (
                f"image {height}x{width} is larger than tile_size={tile_size}; "
                "using halo (real neighbors) instead of isolated tiles. "
                "Pass --allow-isolated-tiles only to reproduce the old tiled path."
            )
        elif not bool(allow_isolated_tiles):
            note = (
                f"image {height}x{width} is larger than tile_size={tile_size}; "
                "isolated tiles lack real neighbors (GroupNorm/attention)."
            )
    if height == int(tile_size) and width == int(tile_size) and mode in {"full", "tiled", "halo"}:
        extra = (
            "input equals one training tile; if this was cropped from a larger FOV, "
            "pass the native page so full/halo can use real neighbors."
        )
        note = f"{note} {extra}".strip() if note else extra
    return mode, note
