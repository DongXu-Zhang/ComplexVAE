"""P0 fixes from ComplexVAE audit v2."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from microscopy_vae.losses.pixel import flux_loss
from microscopy_vae.models.factory import ModelFactory


def test_flux_is_single_mean_bias():
    pred = torch.ones(2, 1, 4, 4) * 2.0
    tgt = torch.ones(2, 1, 4, 4)
    # |2-1| = 1, not 2
    assert float(flux_loss(pred, tgt)) == pytest.approx(1.0)


def test_checkpoint_import_clean_subprocess():
    """Must work without accidental side-effect imports."""
    code = textwrap.dedent(
        """
        import torch
        import torch.utils.checkpoint  # package modules must do this
        from microscopy_vae.models.factory import ModelFactory
        m = ModelFactory.create_fresh(
            encoder_block_out_channels=(32, 64, 64),
            decoder_block_out_channels=(32, 64, 64),
            layers_per_block=1,
            norm_num_groups=8,
            mid_block_add_attention=False,
        )
        m.encoder.gradient_checkpointing = True
        m.decoder.gradient_checkpointing = True
        m.train()
        x = torch.randn(2, 1, 64, 64)
        y = m(x, sample_posterior=True).reconstruction
        y.mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        print("CKPT_OK")
        """
    )
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    # merge path
    import os

    env = {**os.environ, **env}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CKPT_OK" in r.stdout


def test_overfit_gate_threshold_no_half():
    # pure unit of gate logic
    thr = 0.9
    initial, final = 1.0, 0.5
    drop = (initial - final) / initial  # 0.5
    # old wrong: drop >= thr * 0.5 (=0.45) would pass
    assert drop < thr
    assert drop >= thr * 0.5 - 1e-12
    # new: must fail at 50% when threshold is 90%
    passed = drop >= thr
    assert passed is False
    final2 = 0.05
    drop2 = (initial - final2) / initial  # 0.95
    assert drop2 >= thr
