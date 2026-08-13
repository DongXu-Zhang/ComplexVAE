import torch

from microscopy_vae.models.factory import ModelFactory
from microscopy_vae.losses.composer import HQCodecLossComposer


def test_first_backward_grads():
    model = ModelFactory.create_fresh(
        encoder_block_out_channels=(32, 64, 64),
        decoder_block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=8,
        mid_block_add_attention=False,
    )
    loss_fn = HQCodecLossComposer(w_ms_ssim=0.0, w_grad=0.05, w_flux=0.0, free_nats=0.0, beta_max=1e-3, kl_t0=0, kl_t1=1)
    x = torch.randn(2, 1, 64, 64, requires_grad=False)
    out = model(x, sample_posterior=True)
    loss = loss_fn(out, x, optimizer_step=0).total
    loss.backward()
    for name in ["encoder", "quant_conv", "post_quant_conv", "decoder"]:
        mod = getattr(model, name)
        grads = [p.grad for p in mod.parameters() if p.grad is not None]
        assert grads, f"no grads for {name}"
        assert any(g.abs().sum() > 0 for g in grads), f"zero grads for {name}"
