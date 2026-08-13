import torch
import pytest

from microscopy_vae.models.factory import ModelFactory


@pytest.mark.parametrize(
    "enc,dec,factor,size",
    [
        ([32, 64, 64], [32, 64, 64], 4, 64),
        ([32, 64, 128, 128], [32, 64, 128, 128], 8, 64),
    ],
)
def test_encode_decode_shapes(enc, dec, factor, size):
    model = ModelFactory.create_fresh(
        latent_channels=4,
        encoder_block_out_channels=enc,
        decoder_block_out_channels=dec,
        layers_per_block=1,
        norm_num_groups=8 if min(enc) >= 8 and all(c % 8 == 0 for c in enc + dec) else 8,
        mid_block_add_attention=False,
    )
    assert model.spatial_compression == factor
    x = torch.randn(2, 1, size, size)
    out = model(x, sample_posterior=True)
    assert out.reconstruction.shape == x.shape
    assert out.latent.shape == (2, 4, size // factor, size // factor)
    assert out.posterior.mean.shape == out.latent.shape
    assert out.posterior.logvar.shape == out.latent.shape


def test_single_channel_only():
    with pytest.raises(ValueError):
        from microscopy_vae.models.vae import MicroscopyVAE

        MicroscopyVAE(in_channels=3, out_channels=3)
