from pathlib import Path

from microscopy_vae.config.loader import load_config
from microscopy_vae.engine.trainer import Trainer


def test_evaluate_and_latent_spec(tmp_path):
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "smoke_synthetic.yaml"
    out_dir = tmp_path / "run"
    cfg = load_config(
        cfg_path,
        overrides={
            "experiment": {"output_dir": str(out_dir)},
            "training": {"max_steps": 2, "val_every_steps": 1, "log_every_steps": 1},
            "checkpoint": {"save_every_steps": 100},
        },
    )
    tr = Trainer(cfg)
    tr.train(max_steps=2)
    from microscopy_vae.engine.evaluator import evaluate_hq_loader

    metrics = evaluate_hq_loader(tr.system, tr.val_loader, device=tr.device, bootstrap_n=20)
    assert metrics["n_pages"] > 0
    assert "psnr" in metrics["group_macro"]

    # latent spec export math
    means = []
    import torch

    with torch.no_grad():
        for bi, batch in enumerate(tr.train_loader):
            if bi > 1:
                break
            post, _ = tr.system.encode_hq(batch.hq.to(tr.device), sample_posterior=False)
            means.append(post.mean.cpu())
    from microscopy_vae.models.latent_spec import fit_latent_center_scale, build_latent_spec
    from microscopy_vae.models.factory import architecture_id

    c, s = fit_latent_center_scale(means)
    spec = build_latent_spec(
        architecture_id=architecture_id(tr.system.vae),
        weights_sha256="t",
        config_sha256=tr.config_sha,
        normalizer_sha256=tr.normalizer_sha,
        spatial_compression=tr.system.vae.spatial_compression,
        latent_channels=tr.system.vae.latent_channels,
        center=c,
        scale=s,
        stats_sha256="t",
        transform_id=tr.normalizer.state.transform_id,
    )
    z = means[0]
    assert torch.allclose(spec.internal_from_export(spec.export_from_internal(z)), z, atol=1e-5)
