import pytest

from microscopy_vae.models.factory import ModelFactory


def test_create_fresh_rejects_pretrained():
    with pytest.raises(ValueError, match="rejects"):
        ModelFactory.create_fresh(pretrained="/some/path")


def test_create_fresh_trainable():
    m = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
    )
    audit = m.trainability_audit()
    assert audit["all_core_trainable"]
    assert m.count_parameters()["total"] > 0
