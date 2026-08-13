import torch

from microscopy_vae.models.latent_spec import build_latent_spec, fit_latent_center_scale


def test_export_inverse():
    means = [torch.randn(2, 4, 8, 8) for _ in range(3)]
    c, s = fit_latent_center_scale(means)
    spec = build_latent_spec(
        architecture_id="test",
        weights_sha256="w",
        config_sha256="c",
        normalizer_sha256="n",
        spatial_compression=8,
        latent_channels=4,
        center=c,
        scale=s,
        stats_sha256="s",
        transform_id="t",
    )
    z = means[0]
    z_e = spec.export_from_internal(z)
    z_b = spec.internal_from_export(z_e)
    assert torch.allclose(z, z_b, atol=1e-5)
