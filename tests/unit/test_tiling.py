import torch

from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.inference.tiling import pad_to_multiple, reconstruct_full, reconstruct_tiled, unpad


def test_pad_unpad():
    x = torch.randn(1, 1, 60, 61)
    xp, pads = pad_to_multiple(x, 8)
    assert xp.shape[-2] % 8 == 0
    assert xp.shape[-1] % 8 == 0
    y = unpad(xp, pads)
    assert y.shape == x.shape


def test_full_vs_tiled_close_small():
    model = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
    )
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    full = reconstruct_full(model, x, spatial_compression=model.spatial_compression)
    tiled = reconstruct_tiled(
        model, x, tile_size=32, overlap=8, spatial_compression=model.spatial_compression
    )
    assert full.shape == x.shape
    assert tiled.shape == x.shape
    # Not required to be equal with random weights, but finite
    assert torch.isfinite(full).all()
    assert torch.isfinite(tiled).all()
