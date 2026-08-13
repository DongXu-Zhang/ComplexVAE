from __future__ import annotations

from microscopy_vae.config.schema import RootConfig


def validate_for_training(cfg: RootConfig) -> None:
    """Extra runtime checks beyond pydantic (prompt hard boundaries)."""
    if cfg.training.microbatch_size < 1:
        raise ValueError("microbatch_size must be >= 1")
    if cfg.training.grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    if cfg.data.mode == "hq_pool" and not cfg.data.manifest_path:
        raise ValueError("hq_pool mode requires data.manifest_path")
    if cfg.data.mode == "paired_pool":
        raise ValueError(
            "paired_pool is not a valid S1 training mode; HQ-only first (Package A Route E′ S1)"
        )
    if "test" in cfg.data.allow_splits:
        raise ValueError("test split forbidden in training allow_splits")
    if cfg.evaluation.allow_test:
        raise ValueError("evaluation.allow_test must stay false until freeze-candidate")
    if cfg.model.output_activation != "linear":
        raise ValueError("v1 requires linear output_activation")
    if cfg.model.in_channels != 1 or cfg.model.out_channels != 1:
        raise ValueError("single-channel only")
    if cfg.initialization.mode != "fresh_init" and not cfg.training.resume_exact_path:
        raise ValueError("only fresh_init or resume_exact_path allowed")
    # forbid SD scaling as training constant
    if getattr(cfg.latent, "use_sd_scaling_factor", False):
        raise ValueError("SD scaling_factor must not be used")
    # pretrained fields
    if cfg.model.pretrained or cfg.model.from_pretrained or cfg.model.init_from_weights:
        raise ValueError("pretrained init forbidden")
    # route purity for S1
    if cfg.experiment.route != "hq_codec":
        raise ValueError("S1 configs must use experiment.route=hq_codec")
