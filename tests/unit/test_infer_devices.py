"""Device parsing, task split, and tiled fuse-order (CPU; no CUDA required)."""

from __future__ import annotations

import pytest
import torch

from microscopy_vae.inference.devices import assign_round_robin, parse_devices
from microscopy_vae.inference.tiling import (
    fuse_tiled_recons,
    pad_if_smaller,
    reconstruct_one_tile,
    reconstruct_tiled,
    tile_boxes,
)
from microscopy_vae.models.factory import ModelFactory


def test_parse_cpu_and_auto_without_forcing_cuda():
    cpu = parse_devices("cpu")
    assert cpu == [torch.device("cpu")]
    auto = parse_devices("auto")
    assert len(auto) >= 1
    if not torch.cuda.is_available():
        assert auto == [torch.device("cpu")]


def test_parse_rejects_duplicates_and_mix():
    with pytest.raises(ValueError, match="duplicate"):
        parse_devices("cpu,cpu")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="mix"):
            parse_devices("cpu,cuda:0")


def test_parse_cuda_out_of_range():
    if not torch.cuda.is_available():
        with pytest.raises(ValueError):
            parse_devices("cuda:0")
        return
    n = torch.cuda.device_count()
    with pytest.raises(ValueError, match="out of range"):
        parse_devices(f"cuda:{n}")


def test_parse_comma_list_logical_ids():
    if torch.cuda.device_count() < 2:
        pytest.skip("need 2 logical CUDA devices")
    ds = parse_devices("cuda:0,cuda:1")
    assert [d.index for d in ds] == [0, 1]
    ds2 = parse_devices("0,1")
    assert ds2 == ds


def test_round_robin_complete_no_overlap():
    for n_tasks in (0, 1, 7, 16):
        for n_w in (1, 2, 3, 5):
            buckets = assign_round_robin(n_tasks, n_w)
            assert len(buckets) == n_w
            flat = [i for b in buckets for i in b]
            assert sorted(flat) == list(range(n_tasks))
            assert len(flat) == len(set(flat))


def _tiny():
    m = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=True,
    )
    m.eval()
    return m


def test_one_tile_plus_fuse_matches_sequential():
    torch.manual_seed(0)
    model = _tiny()
    x = torch.randn(1, 1, 320, 320)
    y_seq = reconstruct_tiled(
        model, x, tile_size=64, overlap=16, spatial_compression=4, padding_mode="reflect"
    )
    x_work, _ = pad_if_smaller(x, 64, mode="reflect")
    boxes = tile_boxes(x_work.shape[-2], x_work.shape[-1], 64, 16, snap=4)
    recons = [
        reconstruct_one_tile(
            model,
            x_work[:, :, y0:y1, x0:x1],
            spatial_compression=4,
            padding_mode="reflect",
        )
        for y0, x0, y1, x1 in boxes
    ]
    # Shuffle compute order, fuse in original box order.
    order = list(range(len(boxes)))
    order = order[::-1]
    shuffled = [recons[i] for i in order]
    # Must still fuse with original pairing, not shuffled pairing.
    y_fuse = fuse_tiled_recons(
        x,
        boxes,
        recons,
        tile_size=64,
        overlap=16,
        spatial_compression=4,
        padding_mode="reflect",
        blend_mode="linear",
        snap=4,
    )
    assert y_seq.shape == y_fuse.shape
    assert torch.allclose(y_seq, y_fuse, rtol=0, atol=0)
    y_wrong = fuse_tiled_recons(
        x,
        boxes,
        shuffled,
        tile_size=64,
        overlap=16,
        spatial_compression=4,
        padding_mode="reflect",
        blend_mode="linear",
        snap=4,
    )
    # Wrong pairing must not silently equal (unless 1 tile).
    if len(boxes) > 1:
        assert not torch.allclose(y_seq, y_wrong, rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="need 2 CUDA devices")
def test_two_gpu_tiled_close_to_one_gpu():
    from microscopy_vae.config.schema import RootConfig
    from microscopy_vae.inference.parallel import run_tiled
    from microscopy_vae.systems.factory import build_hq_codec_system

    cfg = RootConfig(
        experiment={"output_dir": "runs/unused", "allow_existing_output": True},
        data={"mode": "synthetic", "synthetic_size": 64},
        crop={"size": 64},
        model={
            "encoder_block_out_channels": [32, 64, 64],
            "decoder_block_out_channels": [32, 64, 64],
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "mid_block_add_attention": False,
        },
        precision={"amp_dtype": "fp32"},
        memory={"gradient_checkpointing": False},
        training={"ema_decay": None},
    )
    sys = build_hq_codec_system(cfg).to("cuda:0")
    sys.eval()
    torch.manual_seed(1)
    x = torch.randn(1, 1, 128, 128, device="cuda:0")
    y1 = reconstruct_tiled(
        sys.vae, x, tile_size=64, overlap=16, spatial_compression=4, padding_mode="reflect"
    )
    dump = cfg.model_dump(mode="json")
    y2 = run_tiled(
        sys.vae,
        x,
        cfg_dump=dump,
        devices=[torch.device("cuda", 0), torch.device("cuda", 1)],
        tile_size=64,
        overlap=16,
        spatial_compression=4,
        padding_mode="reflect",
        blend_mode="linear",
    )
    assert y1.shape == y2.shape
    # Same math, same order fusion; allow tiny GPU/CPU accumulate differences.
    mae = float((y1.cpu() - y2.cpu()).abs().mean())
    mx = float((y1.cpu() - y2.cpu()).abs().max())
    assert mx < 1e-4, f"max abs {mx}"
    assert mae < 1e-5, f"mae {mae}"


def test_fuse_rejects_length_mismatch():
    x = torch.zeros(1, 1, 64, 64)
    with pytest.raises(ValueError, match="boxes"):
        fuse_tiled_recons(
            x,
            [(0, 0, 64, 64)],
            [],
            tile_size=64,
            overlap=0,
            spatial_compression=4,
        )
