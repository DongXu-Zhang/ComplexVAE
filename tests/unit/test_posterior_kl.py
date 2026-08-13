import torch

from microscopy_vae.models.posterior import (
    PosteriorStats,
    kl_sum_spatial,
    kl_to_standard_normal_elements,
    sample_latent,
    split_moments,
)
from microscopy_vae.losses.kl import free_bits_kl, beta_at_step


def test_split_and_clamp():
    moments = torch.zeros(1, 8, 4, 4)
    moments[:, 4:] = 100  # huge logvar before clamp
    post = split_moments(moments)
    assert post.logvar.max() <= 20.0


def test_kl_zero_for_standard_normal():
    post = PosteriorStats(mean=torch.zeros(2, 4, 3, 3), logvar=torch.zeros(2, 4, 3, 3))
    kl = kl_to_standard_normal_elements(post)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)
    assert torch.allclose(kl_sum_spatial(post), torch.zeros(2), atol=1e-6)


def test_sample_mean_mode():
    post = PosteriorStats(mean=torch.ones(1, 2, 2, 2), logvar=torch.zeros(1, 2, 2, 2))
    z = sample_latent(post, sample=False)
    assert torch.equal(z, post.mean)


def test_free_bits_and_beta():
    post = PosteriorStats(mean=torch.ones(1, 2, 2, 2), logvar=torch.zeros(1, 2, 2, 2))
    v = free_bits_kl(post, free_nats=10.0)
    assert float(v) == 0.0
    assert beta_at_step(0, t0=10, t1=20, beta_max=1.0) == 0.0
    assert beta_at_step(15, t0=10, t1=20, beta_max=1.0) == 0.5
    assert beta_at_step(25, t0=10, t1=20, beta_max=1.0) == 1.0
