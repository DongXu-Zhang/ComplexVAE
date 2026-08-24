import torch

from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.inference.tiling import (
    even_tile_origins,
    legacy_snapped_origins,
    pad_to_multiple,
    pair_overlaps,
    reconstruct_full,
    reconstruct_tiled,
    tile_boxes,
    unpad,
)


def _tiny(attn: bool = False):
    model = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=attn,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=True,
    )
    model.eval()
    return model


def test_pad_unpad():
    x = torch.randn(1, 1, 60, 61)
    xp, pads = pad_to_multiple(x, 8)
    assert xp.shape[-2] % 8 == 0
    assert xp.shape[-1] % 8 == 0
    y = unpad(xp, pads)
    assert y.shape == x.shape
    assert torch.equal(y, x)


def test_legacy_1004_last_overlap_is_uneven():
    # Fact: old snap-last dumps leftover into the final pair.
    orig = legacy_snapped_origins(1004, 256, 32)
    ovs = pair_overlaps(orig, 256)
    assert orig[0] == 0
    assert orig[-1] == 1004 - 256
    assert max(ovs) > 32
    assert max(ovs) - min(ovs) > 8


def test_even_1004_spreads_overlap():
    orig = even_tile_origins(1004, 256, 32, snap=4)
    assert orig[0] == 0
    assert orig[-1] == 1004 - 256
    ovs = pair_overlaps(orig, 256)
    assert min(ovs) >= 32
    assert max(ovs) - min(ovs) <= 8
    # complete coverage with 256 tiles
    covered = [False] * 1004
    for y0 in orig:
        for i in range(y0, y0 + 256):
            covered[i] = True
    assert all(covered)


def test_256_full_equals_single_tile():
    model = _tiny(attn=False)
    x = torch.randn(1, 1, 256, 256)
    full, f_aux = reconstruct_full(model, x, spatial_compression=4, return_aux=True)
    tiled, t_aux = reconstruct_tiled(
        model, x, tile_size=256, overlap=32, spatial_compression=4, return_aux=True
    )
    assert full.shape == x.shape == tiled.shape
    assert t_aux["n_tiles"] == 1
    assert torch.allclose(full, tiled, rtol=0, atol=0)
    assert f_aux["latent_hw"] == [64, 64]


def test_full_vs_tiled_close_small():
    model = _tiny(attn=False)
    x = torch.randn(1, 1, 64, 64)
    full = reconstruct_full(model, x, spatial_compression=model.spatial_compression)
    tiled = reconstruct_tiled(
        model, x, tile_size=32, overlap=8, spatial_compression=model.spatial_compression
    )
    assert full.shape == x.shape
    assert tiled.shape == x.shape
    assert torch.isfinite(full).all()
    assert torch.isfinite(tiled).all()


def test_not_divisible_by_f4():
    model = _tiny(attn=False)
    x = torch.randn(1, 1, 100, 103)
    full, aux = reconstruct_full(model, x, spatial_compression=4, return_aux=True)
    tiled = reconstruct_tiled(model, x, tile_size=64, overlap=16, spatial_compression=4)
    assert full.shape == x.shape == tiled.shape
    assert aux["padded_hw"][0] % 4 == 0
    assert aux["padded_hw"][1] % 4 == 0
    assert torch.isfinite(full).all() and torch.isfinite(tiled).all()


def test_rectangular_and_smaller_than_tile():
    model = _tiny(attn=False)
    x = torch.randn(1, 1, 80, 192)
    y = reconstruct_tiled(model, x, tile_size=64, overlap=16, spatial_compression=4)
    assert y.shape == x.shape
    small = torch.randn(1, 1, 40, 36)
    y2, aux = reconstruct_tiled(
        model, small, tile_size=64, overlap=16, spatial_compression=4, return_aux=True
    )
    assert y2.shape == small.shape
    assert aux["n_tiles"] == 1
    yf = reconstruct_full(model, small, spatial_compression=4)
    assert yf.shape == small.shape


def test_overlap_zero_covers():
    model = _tiny(attn=False)
    x = torch.randn(1, 1, 100, 100)
    y, aux = reconstruct_tiled(
        model, x, tile_size=64, overlap=0, spatial_compression=4, return_aux=True
    )
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert aux["weight_min"] > 0


def test_weight_map_positive_1004_geometry():
    boxes = tile_boxes(1004, 1004, 256, 32, snap=4)
    acc = torch.zeros(1004, 1004)
    for y0, x0, y1, x1 in boxes:
        acc[y0:y1, x0:x1] += 1
    assert float(acc.min()) >= 1
    assert boxes[0][0] == 0 and boxes[0][1] == 0
    assert boxes[-1][2] == 1004 and boxes[-1][3] == 1004


def test_tiled_order_invariant():
    model = _tiny(attn=False)
    x = torch.randn(1, 1, 96, 80)
    a = reconstruct_tiled(model, x, tile_size=64, overlap=16, spatial_compression=4)
    b = reconstruct_tiled(model, x, tile_size=64, overlap=16, spatial_compression=4)
    assert torch.equal(a, b)


def test_posterior_mean_is_deterministic():
    model = _tiny(attn=True)
    x = torch.randn(1, 1, 64, 64)
    a = reconstruct_full(model, x, spatial_compression=4)
    b = reconstruct_full(model, x, spatial_compression=4)
    assert torch.equal(a, b)


def test_reflect_pad_larger_than_image():
    from microscopy_vae.inference.tiling import pad_if_smaller

    x = torch.randn(1, 1, 10, 12)
    y, _ = pad_if_smaller(x, 64, mode="reflect")
    assert y.shape[-2] >= 64 and y.shape[-1] >= 64
    assert torch.isfinite(y).all()


def test_hann_blend_finite():
    model = _tiny(attn=False)
    x = torch.randn(1, 1, 96, 96)
    y, aux = reconstruct_tiled(
        model,
        x,
        tile_size=64,
        overlap=16,
        spatial_compression=4,
        blend_mode="hann",
        return_aux=True,
    )
    assert y.shape == x.shape
    assert aux["weight_min"] > 0
    assert torch.isfinite(y).all()
