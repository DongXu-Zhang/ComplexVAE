"""Halo window geometry is independent of GPU count; fuse matches sequential."""

from __future__ import annotations

import torch

from microscopy_vae.inference.devices import assign_round_robin
from microscopy_vae.inference.tiling import (
    crop_halo_core,
    fuse_halo_cores,
    halo_jobs,
    reconstruct_halo,
    reconstruct_one_tile,
    tile_boxes,
)
from microscopy_vae.models.factory import ModelFactory


def test_halo_jobs_independent_of_worker_count():
    h, w, ts, ov, halo = 80, 90, 32, 8, 16
    boxes = tile_boxes(h, w, ts, ov, snap=4)
    jobs = halo_jobs(h, w, boxes, halo)
    assert len(jobs) == len(boxes)
    for n_workers in (1, 2, 3, 7):
        splits = assign_round_robin(len(jobs), n_workers)
        flat = [i for s in splits for i in s]
        assert sorted(flat) == list(range(len(jobs)))
        for s in splits:
            for i in s:
                assert jobs[i]["y0"] == boxes[i][0]


def test_halo_job_fuse_matches_sequential():
    torch.manual_seed(0)
    model = ModelFactory.create_fresh(
        latent_channels=4,
        encoder_block_out_channels=(16, 32),
        decoder_block_out_channels=(16, 32),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=False,
    )
    model.eval()
    x = torch.randn(1, 1, 48, 56)
    kwargs = dict(
        tile_size=32,
        overlap=8,
        halo=12,
        spatial_compression=int(model.spatial_compression),
        padding_mode="reflect",
        blend_mode="linear",
    )
    with torch.no_grad():
        seq = reconstruct_halo(model, x, **kwargs)
        from microscopy_vae.inference.tiling import pad_if_smaller

        x_work, _ = pad_if_smaller(x, 32, mode="reflect")
        h, w = x_work.shape[-2:]
        boxes = tile_boxes(h, w, 32, 8, snap=int(model.spatial_compression))
        jobs = halo_jobs(h, w, boxes, 12)
        cores = []
        for job in jobs:
            window = x_work[:, :, job["hy0"] : job["hy1"], job["hx0"] : job["hx1"]]
            recon_w = reconstruct_one_tile(
                model, window, spatial_compression=int(model.spatial_compression), padding_mode="reflect"
            )
            cores.append(crop_halo_core(recon_w, job))
        fused = fuse_halo_cores(
            x, jobs, cores, tile_size=32, overlap=8, padding_mode="reflect", blend_mode="linear", halo=12
        )
    assert torch.allclose(seq, fused, atol=1e-6, rtol=1e-5)


def test_round_robin_empty_ranks():
    splits = assign_round_robin(2, 5)
    assert splits[0] == [0]
    assert splits[1] == [1]
    assert splits[2] == []
    assert sum(len(s) for s in splits) == 2
