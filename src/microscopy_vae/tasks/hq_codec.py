from __future__ import annotations

from typing import Any

from microscopy_vae.data.records import HQBatch
from microscopy_vae.losses.composer import HQCodecLossComposer
from microscopy_vae.losses.types import LossOutput
from microscopy_vae.models.vae import MicroscopyVAE
from microscopy_vae.tasks.base import TaskCapabilities


class HQCodecTask:
    name = "hq_codec"
    capabilities = TaskCapabilities(
        hq_reconstruction=True,
        lr_encoding=False,
        paired_restoration=False,
        context_2p5d=False,
        tiled_inference=True,
    )

    def __init__(self, model: MicroscopyVAE, loss: HQCodecLossComposer, *, sample_posterior: bool = True) -> None:
        self.model = model
        self.loss = loss
        self.sample_posterior = sample_posterior

    def forward_loss(self, batch: HQBatch, *, optimizer_step: int) -> LossOutput:
        x = batch.hq
        out = self.model(x, sample_posterior=self.sample_posterior)
        return self.loss(out, x, optimizer_step=optimizer_step, sources=list(batch.sources))
