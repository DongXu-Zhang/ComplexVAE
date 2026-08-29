from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from microscopy_vae.config.loader import load_config
from microscopy_vae.config.schema import RootConfig
from microscopy_vae.engine.checkpoint import CheckpointManager, load_vae_state_dict
from microscopy_vae.engine.state import TrainerState
from microscopy_vae.inference.tiling import (
    decode_full,
    encode_full,
    reconstruct_full,
    reconstruct_tiled,
    tile_boxes,
)
from microscopy_vae.models.factory import ModelFactory, architecture_id


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _tiny_f4(**kw):
    return ModelFactory.create_fresh(
        latent_channels=4,
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=True,
        **kw,
    )


def _tiny_f8(**kw):
    return ModelFactory.create_fresh(
        latent_channels=4,
        encoder_block_out_channels=(32, 64, 64, 64),
        decoder_block_out_channels=(32, 64, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
        upsample_mode="bilinear",
        downsample_pad_mode="symmetric",
        downsample_preblur=True,
        **kw,
    )


def test_f4_v4_yaml_unchanged():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f4z4_v4.yaml")
    assert len(cfg.model.encoder_block_out_channels) == 3
    assert len(cfg.model.decoder_block_out_channels) == 3
    assert cfg.model.latent_channels == 4
    assert cfg.crop.size == 256
    assert cfg.experiment.output_dir == "runs/s1_hq_f4z4_v4"


def test_f8_v4_yaml_is_extra_stage_not_resample():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v4.yaml")
    assert len(cfg.model.encoder_block_out_channels) == 4
    assert cfg.model.encoder_block_out_channels == [128, 256, 512, 512]
    assert cfg.model.decoder_block_out_channels == [96, 192, 384, 384]
    assert cfg.model.latent_channels == 4
    assert cfg.model.upsample_mode == "bilinear"
    assert cfg.normalization.scale_mode == "per_source"
    assert cfg.normalization.raw_floor_enabled is True
    assert cfg.loss.w_grad == 0.0
    assert cfg.loss.w_hf == 0.0
    assert cfg.loss.w_flux == 0.0
    assert cfg.experiment.output_dir == "runs/s1_hq_f8z4_v4"
    assert cfg.experiment.output_dir != "runs/s1_hq_f4z4_v4"
    assert cfg.evaluation.allow_test is False
    assert cfg.crop.size % 8 == 0


def test_f8_v4_matches_f4_except_architecture():
    """The f8 experiment must only change spatial compression, not V4 protocol."""
    f4 = load_config(_repo() / "configs/experiment/s1_hq_f4z4_v4.yaml").model_dump()
    f8 = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v4.yaml").model_dump()
    for section in (
        "data",
        "normalization",
        "sampling",
        "crop",
        "loss",
        "kl_schedule",
        "optimizer",
        "scheduler",
        "precision",
        "memory",
        "training",
        "evaluation",
        "checkpoint",
        "task",
        "latent",
    ):
        assert f4[section] == f8[section], section
    for key in (
        "latent_channels",
        "layers_per_block",
        "norm_num_groups",
        "mid_block_add_attention",
        "output_activation",
        "upsample_mode",
        "downsample_pad_mode",
        "downsample_preblur",
    ):
        assert f4["model"][key] == f8["model"][key], key
    assert len(f4["model"]["encoder_block_out_channels"]) == 3
    assert len(f8["model"]["encoder_block_out_channels"]) == 4


def test_resume_overrides_do_not_change_config_hash():
    from microscopy_vae.config.loader import config_semantic_hash, load_config

    base = load_config(_repo() / "configs/experiment/smoke_f8.yaml")
    resumed = load_config(
        _repo() / "configs/experiment/smoke_f8.yaml",
        overrides={
            "experiment": {"allow_existing_output": True},
            "training": {"resume_exact_path": "/tmp/step.pt"},
        },
    )
    assert config_semantic_hash(base) == config_semantic_hash(resumed)


def test_encoder_decoder_stage_counts_must_match():
    with pytest.raises(ValidationError, match="stage counts"):
        RootConfig(
            model={
                "encoder_block_out_channels": [128, 256, 512, 512],
                "decoder_block_out_channels": [96, 192, 384],
            }
        )


def test_f8_256_encode_mean_logvar_shape():
    model = _tiny_f8()
    model.eval()
    x = torch.randn(1, 1, 256, 256)
    post = model.encode(x)
    assert model.spatial_compression == 8
    assert post.mean.shape == (1, 4, 32, 32)
    assert post.logvar.shape == (1, 4, 32, 32)
    z = model.sample_latent(post, sample=False)
    assert z.shape == (1, 4, 32, 32)
    assert torch.equal(z, post.mean)


def test_f8_decode_32_to_256():
    model = _tiny_f8()
    model.eval()
    z = torch.randn(1, 4, 32, 32)
    y = model.decode(z)
    assert y.shape == (1, 1, 256, 256)


def test_f8_train_forward_sample_backward():
    model = _tiny_f8()
    model.train()
    x = torch.randn(2, 1, 64, 64, requires_grad=False)
    out = model(x, sample_posterior=True)
    assert out.reconstruction.shape == x.shape
    assert out.latent.shape == (2, 4, 8, 8)
    loss = (out.reconstruction - x).abs().mean() + out.posterior.mean.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_f8_pad_non_multiple_and_crop_back():
    model = _tiny_f8()
    model.eval()
    x = torch.randn(1, 1, 250, 251)
    y, aux = reconstruct_full(model, x, spatial_compression=8, return_aux=True)
    assert y.shape == x.shape
    assert aux["padded_hw"][0] % 8 == 0
    assert aux["padded_hw"][1] % 8 == 0
    assert aux["latent_hw"] == [aux["padded_hw"][0] // 8, aux["padded_hw"][1] // 8]
    z, post, eaux = encode_full(model, x, spatial_compression=8)
    assert post.mean.shape[-2] == eaux["padded_hw"][0] // 8
    y2 = decode_full(model, z, pad_hw=tuple(eaux["pad_hw"]), output_hw=tuple(eaux["input_hw"]))
    assert y2.shape == x.shape
    assert torch.allclose(y, y2, rtol=0, atol=0)


def test_f8_255x257_pads_to_latent_32x33():
    """Spec: 255×257 → pad to 256×264 → latent [1,4,32,33] → crop back."""
    model = _tiny_f8()
    model.eval()
    x = torch.randn(1, 1, 255, 257)
    z, post, aux = encode_full(model, x, spatial_compression=8)
    assert aux["padded_hw"] == [256, 264]
    assert aux["pad_hw"] == [1, 7]
    assert post.mean.shape == (1, 4, 32, 33)
    assert post.logvar.shape == (1, 4, 32, 33)
    assert z.shape == (1, 4, 32, 33)
    assert torch.isfinite(post.mean).all() and torch.isfinite(post.logvar).all()
    y = decode_full(model, z, pad_hw=tuple(aux["pad_hw"]), output_hw=tuple(aux["input_hw"]))
    assert y.shape == (1, 1, 255, 257)
    yf, faux = reconstruct_full(model, x, spatial_compression=8, return_aux=True)
    assert yf.shape == x.shape
    assert faux["latent_hw"] == [32, 33]
    assert torch.allclose(y, yf, rtol=0, atol=0)


def test_f8_full_equals_single_tile():
    model = _tiny_f8()
    model.eval()
    x = torch.randn(1, 1, 256, 256)
    full, f_aux = reconstruct_full(model, x, spatial_compression=8, return_aux=True)
    tiled, t_aux = reconstruct_tiled(
        model, x, tile_size=256, overlap=32, spatial_compression=8, return_aux=True
    )
    assert full.shape == x.shape == tiled.shape
    assert t_aux["n_tiles"] == 1
    assert torch.allclose(full, tiled, rtol=0, atol=0)
    assert f_aux["latent_hw"] == [32, 32]


def test_f8_multi_tile_covers_without_holes():
    model = _tiny_f8()
    model.eval()
    x = torch.randn(1, 1, 200, 180)
    y, aux = reconstruct_tiled(
        model, x, tile_size=64, overlap=16, spatial_compression=8, return_aux=True
    )
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert float(aux["weight_min"]) > 0
    h, w = x.shape[-2:]
    # coverage on the work canvas (may pad if < tile)
    boxes = tile_boxes(max(h, 64), max(w, 64), 64, 16, snap=8)
    acc = torch.zeros(max(h, 64), max(w, 64))
    for y0, x0, y1, x1 in boxes:
        acc[y0:y1, x0:x1] += 1
    assert float(acc.min()) >= 1


def test_f8_tile_size_must_be_multiple_of_8():
    model = _tiny_f8()
    x = torch.randn(1, 1, 64, 64)
    with pytest.raises(ValueError, match="divisible"):
        reconstruct_tiled(model, x, tile_size=60, overlap=8, spatial_compression=8)


def test_f4_checkpoint_refuses_f8_model():
    f4 = _tiny_f4()
    f8 = _tiny_f8()
    extra = {
        "spatial_compression": int(f4.spatial_compression),
        "latent_channels": 4,
        "architecture_id": architecture_id(f4),
    }
    with pytest.raises(RuntimeError, match="Refusing to load f4"):
        load_vae_state_dict(f8, f4.state_dict(), extra=extra)
    # even without extra tags, strict load must fail with a clear f4/f8 message
    with pytest.raises(RuntimeError, match="strict weight load failed"):
        load_vae_state_dict(f8, f4.state_dict(), extra=None)


def test_f8_checkpoint_save_restore_export_infer(tmp_path):
    model = _tiny_f8()
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    y0 = model.reconstruct(x)
    ckpt = CheckpointManager(tmp_path)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = ckpt.save_exact(
        tag="step_0000001",
        model=model,
        optimizer=opt,
        scheduler=None,
        scaler=None,
        state=TrainerState(),
        config_sha256="cfg",
        normalizer_sha256="norm",
        code_version="0.3.0",
        extra={
            "spatial_compression": 8,
            "latent_channels": 4,
            "architecture_id": architecture_id(model),
        },
    )
    fresh = _tiny_f8()
    CheckpointManager.load_exported_weights(path, fresh)
    y1 = fresh.reconstruct(x)
    assert torch.allclose(y0, y1, rtol=0, atol=1e-5)
    z, _, aux = encode_full(fresh, x, spatial_compression=8)
    assert list(z.shape) == [1, 4, 8, 8]
    y2 = decode_full(fresh, z, pad_hw=tuple(aux["pad_hw"]), output_hw=tuple(aux["input_hw"]))
    assert torch.allclose(y0, y2, rtol=0, atol=1e-5)
    y_full = reconstruct_full(fresh, x, spatial_compression=8)
    y_tile = reconstruct_tiled(fresh, x, tile_size=64, overlap=16, spatial_compression=8)
    assert torch.allclose(y_full, y_tile, rtol=0, atol=0)


def test_f8_not_a_resample_of_f4_latent():
    """f8 latent grid is produced by an extra stride-2 stage, not interpolate(64→32)."""
    f8 = _tiny_f8()
    downs = [b for b in f8.encoder.down_blocks if len(b.downsamplers) > 0]
    ups = [b for b in f8.decoder.up_blocks if len(b.upsamplers) > 0]
    assert len(downs) == 3
    assert len(ups) == 3
    f4 = _tiny_f4()
    assert len([b for b in f4.encoder.down_blocks if len(b.downsamplers) > 0]) == 2
    assert f8.spatial_compression == 8
    assert f4.spatial_compression == 4


def test_historical_f8_yaml_still_loads():
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f8z4.yaml")
    assert len(cfg.model.encoder_block_out_channels) == 4
    assert cfg.model.latent_channels == 4


def test_official_f8_yaml_256_production_shapes():
    """Full channel table from s1_hq_f8z4_v4.yaml, not the tiny test net."""
    cfg = load_config(_repo() / "configs/experiment/s1_hq_f8z4_v4.yaml")
    model = ModelFactory.create_fresh(
        latent_channels=cfg.model.latent_channels,
        encoder_block_out_channels=tuple(cfg.model.encoder_block_out_channels),
        decoder_block_out_channels=tuple(cfg.model.decoder_block_out_channels),
        layers_per_block=cfg.model.layers_per_block,
        norm_num_groups=cfg.model.norm_num_groups,
        mid_block_add_attention=cfg.model.mid_block_add_attention,
        output_activation=cfg.model.output_activation,
        upsample_mode=cfg.model.upsample_mode,
        downsample_pad_mode=cfg.model.downsample_pad_mode,
        downsample_preblur=cfg.model.downsample_preblur,
    )
    model.eval()
    assert model.spatial_compression == 8
    assert architecture_id(model).startswith("microvae_f8_z4_enc128-256-512-512")
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        post = model.encode(x)
        y = model.decode(post.mean)
    assert post.mean.shape == (1, 4, 32, 32)
    assert post.logvar.shape == (1, 4, 32, 32)
    assert y.shape == (1, 1, 256, 256)
    assert torch.isfinite(y).all()


def test_f8_cli_encode_decode_roundtrip(tmp_path, monkeypatch):
    import argparse

    from microscopy_vae.cli import cmd_decode, cmd_encode

    model = _tiny_f8()
    model.eval()
    weights = tmp_path / "f8.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "extra": {
                "spatial_compression": 8,
                "latent_channels": 4,
                "architecture_id": architecture_id(model),
            },
        },
        weights,
    )
    x = torch.randn(1, 1, 64, 64)
    page = x[0, 0].numpy().astype(np.float32)
    with torch.no_grad():
        want = model.reconstruct(x)
    monkeypatch.setattr(
        "microscopy_vae.data.readers.read_page",
        lambda path, page_index, **kw: (page, {}),
    )
    dummy_in = tmp_path / "page.bin"
    dummy_in.write_bytes(b"")
    latent = tmp_path / "latent.npy"
    recon = tmp_path / "recon.npy"
    common = dict(
        override=[],
        print_resolved_config=False,
        dry_run=False,
        weights=str(weights),
        normalizer=None,
        page=0,
        padding_mode="reflect",
        raw_weights=True,
        devices="cpu",
        source=None,
        meta=None,
    )
    enc_args = argparse.Namespace(
        config=str(_repo() / "configs/experiment/smoke_f8.yaml"),
        input=str(dummy_in),
        output=str(latent),
        **common,
    )
    assert cmd_encode(enc_args) == 0
    z = np.load(latent)
    assert z.shape == (4, 8, 8)
    meta = json.loads(latent.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["spatial_compression"] == 8
    assert meta["posterior"] == "mean"
    assert meta["sd_scaling_factor"] is False
    assert meta["domain"] == "internal_unscaled"
    dec_args = argparse.Namespace(
        config=str(_repo() / "configs/experiment/smoke_f8.yaml"),
        input=str(latent),
        output=str(recon),
        **common,
    )
    assert cmd_decode(dec_args) == 0
    got = np.load(recon)
    assert got.shape == (64, 64)
    assert np.allclose(got, want[0, 0].numpy(), atol=1e-5)


def test_f8_trainer_two_steps(tmp_path):
    from microscopy_vae.engine.trainer import Trainer

    cfg = load_config(
        _repo() / "configs/experiment/smoke_f8.yaml",
        overrides={
            "experiment": {"output_dir": str(tmp_path / "run"), "allow_existing_output": True},
            "loss": {"perceptual": {"enabled": False}, "adversarial": {"enabled": False}},
            "training": {"max_steps": 2, "ema_decay": 0.0, "val_every_steps": 100},
        },
    )
    trainer = Trainer(cfg)
    assert trainer.system.vae.spatial_compression == 8
    batch = next(iter(trainer.train_loader))
    assert batch.hq.shape[-2] == 64
    out = trainer.system.vae(batch.hq, sample_posterior=True)
    assert out.latent.shape[-2:] == (8, 8)
    result = trainer.train(max_steps=2)
    assert result["final_step"] == 2
    assert np.isfinite(float(result["final_loss"]))
    extra = trainer._ckpt_extra()
    assert extra["spatial_compression"] == 8
    assert extra["latent_channels"] == 4
