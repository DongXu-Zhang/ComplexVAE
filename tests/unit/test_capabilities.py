import pytest

from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.losses.composer import HQCodecLossComposer
from microscopy_vae.tasks.hq_codec import HQCodecTask
from microscopy_vae.systems.hq_codec import HQCodecSystem


def test_hq_system_no_restore_lr():
    vae = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
    )
    task = HQCodecTask(vae, HQCodecLossComposer(w_ms_ssim=0, w_grad=0, w_flux=0, free_nats=0))
    sys = HQCodecSystem(vae, task)
    assert sys.capabilities.paired_restoration is False
    assert sys.capabilities.lr_encoding is False
    with pytest.raises(AttributeError):
        sys.restore_lr(None)
    with pytest.raises(AttributeError):
        sys.encode_lr(None)
