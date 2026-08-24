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
    perc = cfg.loss.perceptual
    if perc.enabled:
        if not perc.freeze:
            raise ValueError("perceptual.freeze must be true (trainable backbone is not supported)")
        if perc.backbone == "vgg16" and not perc.vgg_repeat_channels:
            raise ValueError("vgg16 requires perceptual.vgg_repeat_channels=true (explicit 1→3 repeat)")
        if perc.weight < 0:
            raise ValueError("perceptual.weight must be >= 0")
        if perc.backbone == "internal_conv":
            allowed = {f"block{i + 1}" for i in range(len(perc.channels))}
            bad = [n for n in perc.selected_layers if n not in allowed]
            if bad:
                raise ValueError(
                    f"perceptual.selected_layers {bad} not in internal_conv blocks {sorted(allowed)}"
                )
    adv = cfg.loss.adversarial
    if adv.enabled:
        if adv.n_critic < 1:
            raise ValueError("adversarial.n_critic must be >= 1")
        if adv.weight < 0:
            raise ValueError("adversarial.weight must be >= 0")
        if adv.conditioning == "input":
            # allowed but degenerate for S1 (input==target); trainer logs a warning
            pass
    infl = cfg.loss.influence
    if infl.grad_every_steps < 0 or infl.cosine_every_steps < 0:
        raise ValueError("influence periods must be >= 0")
