"""Pydantic v2 configuration schema for microscopy-vae."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "s1_hq_f8z4"
    route: Literal["hq_codec"] = "hq_codec"
    output_dir: str = "runs/default"
    seed: int = 0
    notes: str = ""
    # Refuse non-empty output_dir unless resume or explicitly allowed
    allow_existing_output: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["synthetic", "hq_pool", "paired_pool"] = "synthetic"
    manifest_path: Optional[str] = None
    allow_splits: List[Literal["train", "val"]] = Field(default_factory=lambda: ["train", "val"])
    # test is intentionally absent from training configs
    synthetic_n_groups: int = 4
    synthetic_pages_per_group: int = 2
    synthetic_size: int = 64
    raw_data_root: Optional[str] = None
    # Runtime map for inventory paths like F:\Dataset\... → Linux mount root.
    # Do not rewrite the authoritative JSONL; map at load time only.
    path_prefix_source: Optional[str] = "F:\\Dataset"
    path_prefix_target: Optional[str] = None
    path_require_exists: bool = False


class NormalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # robust_linear_p0.1_p99.9 is the locked V1/V2 name (percentiles must stay 0.1/99.9).
    # robust_linear uses low_percentile/high_percentile (V4).
    method: Literal["robust_linear_p0.1_p99.9", "robust_linear", "identity"] = "robust_linear_p0.1_p99.9"
    fit_split: Literal["train"] = "train"
    # Linear-head codec: keep tails (do NOT clip to [0,1]).
    clip: bool = False
    ssim_data_range: float = 1.0
    artifact_path: Optional[str] = None
    # page_uniform biases toward multi-page DI3D; source_balanced is recommended default.
    fit_mode: Literal["source_balanced", "page_uniform"] = "source_balanced"
    max_pages_fit: int = 192
    max_pixels_per_page: int = 65536
    low_percentile: float = 0.1
    high_percentile: float = 99.9
    # Raw intensity floor applied before fit and transform. Off = V2.2 behaviour.
    raw_floor_enabled: bool = False
    raw_floor_value: float = 0.0
    # global: one affine for all sources. per_source: train-only (low, high) per source.
    scale_mode: Literal["global", "per_source"] = "global"
    # If false, refuse to load an artifact whose floor/percentiles differ from this config.
    allow_legacy_artifact: bool = True
    # Train-only per-source crop/support/amp thresholds in *normalized* space.
    # Off = use yaml scalars (V2/V4). On = fit into normalizer.json and refuse
    # artifacts that lack the threshold contract.
    calibrate_thresholds: bool = False
    threshold_bg_quantile: float = 0.20
    threshold_bg_scharr_q: float = 90.0
    threshold_empty_range_q: float = 90.0
    threshold_struct_range_q: float = 10.0
    threshold_crops_per_page: int = 4

    @model_validator(mode="after")
    def percentile_contract(self) -> "NormalizationConfig":
        if not (0.0 <= float(self.low_percentile) < float(self.high_percentile) <= 100.0):
            raise ValueError(
                f"Need 0 <= low_percentile < high_percentile <= 100, "
                f"got {self.low_percentile}/{self.high_percentile}"
            )
        if self.method == "robust_linear_p0.1_p99.9":
            if abs(self.low_percentile - 0.1) > 1e-12 or abs(self.high_percentile - 99.9) > 1e-12:
                raise ValueError(
                    "method robust_linear_p0.1_p99.9 is locked to p0.1/p99.9; "
                    "use method=robust_linear for custom percentiles"
                )
        return self


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["source_group_page"] = "source_group_page"
    source_weight_mode: Literal["sqrt_groups", "n_groups", "fixed_prior"] = "sqrt_groups"
    fixed_source_prior: Optional[Dict[str, float]] = None
    # uniform: original. focus_softmax: upweight in-focus / structured slices inside a volume.
    slice_weight_mode: Literal["uniform", "focus_softmax"] = "uniform"
    focus_sidecar_path: Optional[str] = None
    focus_temperature: float = 0.7
    focus_min_keep: float = 0.15
    focus_compute_if_missing: bool = False


class CropConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = 256
    multi_scale_sizes: List[int] = Field(default_factory=lambda: [256])
    multi_scale_probs: List[float] = Field(default_factory=lambda: [1.0])
    # random: original. coverage_jitter: prefer unseen coarse cells, then jitter.
    mode: Literal["random", "coverage_jitter"] = "random"
    coverage_jitter_frac: float = 0.25
    # Train-only: retry a crop if normalized robust range is below this. 0 disables.
    min_robust_range: float = 0.0
    max_range_retries: int = 8
    # If a crop is below the range gate, keep it with this probability instead of
    # retrying (so dark/empty tiles still appear in training). 0 = V4 always retry.
    empty_keep_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    in_channels: int = 1
    out_channels: int = 1
    latent_channels: int = 4
    encoder_block_out_channels: List[int] = Field(default_factory=lambda: [128, 256, 512, 512])
    decoder_block_out_channels: List[int] = Field(default_factory=lambda: [96, 192, 384, 384])
    layers_per_block: int = 2
    norm_num_groups: int = 32
    mid_block_add_attention: bool = True
    output_activation: Literal["linear"] = "linear"
    upsample_mode: Literal["nearest", "bilinear"] = "nearest"
    downsample_pad_mode: Literal["asymmetric", "symmetric"] = "asymmetric"
    downsample_preblur: bool = False
    input_domain: str = "normalized_hq"
    output_domain: str = "normalized_hq"
    # Forbidden fields must stay absent or false
    pretrained: Optional[Any] = None
    from_pretrained: Optional[Any] = None
    init_from_weights: Optional[Any] = None

    @field_validator("in_channels", "out_channels")
    @classmethod
    def single_channel(cls, v: int) -> int:
        if v != 1:
            raise ValueError("only single-channel models are allowed")
        return v

    @model_validator(mode="after")
    def encoder_decoder_stage_count(self) -> "ModelConfig":
        n_enc = len(self.encoder_block_out_channels)
        n_dec = len(self.decoder_block_out_channels)
        if n_enc != n_dec:
            raise ValueError(
                f"encoder/decoder stage counts must match "
                f"(got enc={n_enc} dec={n_dec}); spatial_compression = 2**(n_stages-1)"
            )
        return self


class LatentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "microvae-latent-spec-v1"
    # Never use SD 0.18215 as a training constant
    use_sd_scaling_factor: Literal[False] = False
    padding_mode: Literal["reflect", "replicate", "constant"] = "reflect"


class InitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["fresh_init"] = "fresh_init"


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["hq_codec"] = "hq_codec"
    sample_posterior: bool = True


class PerceptualLossConfig(BaseModel):
    """Off by default. See losses/perceptual.py for domain-mismatch notes."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    weight: float = 0.05
    start_step: int = 0
    ramp_steps: int = 1000
    backbone: Literal["internal_conv", "vgg16"] = "internal_conv"
    selected_layers: List[str] = Field(default_factory=lambda: ["block1", "block2", "block3"])
    layer_weights: Optional[Dict[str, float]] = None
    distance: Literal["l1", "l2"] = "l1"
    # Default: global normalized domain, NOT amp-scaled (intensity errors stay visible).
    apply_amp_norm: bool = False
    freeze: bool = True
    channels: List[int] = Field(default_factory=lambda: [16, 32, 64])
    kernel_size: int = 3
    init_seed: int = 0
    normalize_features: bool = False
    unstructured_policy: Literal["keep", "match_target"] = "match_target"
    # vgg16 only — all explicit; defaults do not silent-clamp.
    vgg_pretrained: bool = True
    vgg_repeat_channels: bool = True
    vgg_clamp_to_unit: bool = False
    vgg_data_mean: float = 0.0
    vgg_data_std: float = 1.0
    vgg_imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    vgg_imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    fail_if_unavailable: bool = True


class AdversarialLossConfig(BaseModel):
    """Off by default. S1 default conditioning is unconditional (input==target)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    weight: float = 0.02
    start_step: int = 5000
    ramp_steps: int = 2000
    architecture: Literal["patchgan"] = "patchgan"
    conditioning: Literal["none", "input"] = "none"
    gan_loss: Literal["hinge", "lsgan"] = "hinge"
    spectral_norm: bool = True
    ndf: int = 32
    n_layers: int = 3
    kernel_size: int = 4
    disc_lr: float = 1e-4
    disc_betas: Tuple[float, float] = (0.5, 0.9)
    disc_weight_decay: float = 0.0
    disc_scheduler: Literal["none", "cosine"] = "none"
    n_critic: int = 1
    r1_gamma: float = 0.0
    unstructured_policy: Literal["keep", "exclude"] = "exclude"
    grad_clip_norm: float = 1.0


class LossInfluenceConfig(BaseModel):
    """Layer-2 ratios are cheap. Layer-3 autograd.grad is periodic and off by default."""

    model_config = ConfigDict(extra="forbid")
    log_contrib_ratio: bool = True
    grad_every_steps: int = 0
    cosine_every_steps: int = 0
    param_groups: List[str] = Field(default_factory=lambda: ["full", "encoder", "decoder", "output"])
    ema_decay: float = 0.99


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    w_char: float = 1.0
    w_ms_ssim: float = 0.1
    w_grad: float = 0.05
    w_flux: float = 0.01
    free_nats: float = 0.5
    charbonnier_eps: float = 1e-3
    ms_ssim_start_step: int = 1000
    ms_ssim_ramp_steps: int = 500
    # v2: per-sample amplitude normalization of pixel/structure terms (not flux/KL).
    amp_norm: bool = False
    amp_norm_min_scale: float = 0.05
    # If per-crop robust range is below this, do NOT amplify (use amp_low_structure_scale).
    # 0 disables the guard (legacy v2 behaviour that over-weighted empty patches).
    amp_low_structure_range: float = 0.0
    amp_low_structure_scale: float = 1.0
    # Smooth the idle↔amp transition instead of a hard cut at amp_low_structure_range.
    amp_smooth: bool = False
    # MS-SSIM / perceptual background leak: 0 = old match_target (zero error off support).
    unstructured_bg_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    # extra weight on |∇target|-weighted Charbonnier; 0 keeps v1 behaviour.
    edge_weight: float = 0.0
    # 0 = no clip (legacy). v2.1 uses 3.0 so isolated noise cannot get huge weights.
    edge_weight_clip: float = 0.0
    # high-pass residual Charbonnier; 0 keeps v1.
    w_hf: float = 0.0
    ssim_range_mode: Literal["fixed", "amp_space"] = "fixed"
    # Penalize recon > target on the darkest quantile of each crop (background speckle).
    w_dark_fp: float = 0.0
    dark_fp_quantile: float = 0.20
    # Idle (unstructured) crops keep Charbonnier/Flux/KL but this multiplier
    # downweights them in the pixel term. 1.0 = legacy. Not a new loss term.
    idle_loss_mult: float = 1.0
    # Pixel-level structure support. 0/1 = off. Isolated spikes (any intensity)
    # fail the local density test; filaments/puncta pass. Not a color loss.
    structure_support_kernel: int = 0
    structure_support_floor: float = 0.02
    structure_support_rel: float = 0.25
    structure_support_min_density: float = 0.15
    # Additional idle if supported pixel fraction is below this. 0 = off.
    structure_min_frac: float = 0.0
    perceptual: PerceptualLossConfig = Field(default_factory=PerceptualLossConfig)
    adversarial: AdversarialLossConfig = Field(default_factory=AdversarialLossConfig)
    influence: LossInfluenceConfig = Field(default_factory=LossInfluenceConfig)


class KLScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t0: int = 2000
    t1: int = 20000
    beta_max: float = 1e-2


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-4


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["cosine"] = "cosine"
    warmup_steps: int = 500
    min_lr: float = 1e-6


class PrecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amp_dtype: Literal["bf16", "fp32", "fp16"] = "bf16"
    force_fp32_losses: bool = True


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gradient_checkpointing: bool = True


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = 100000
    microbatch_size: int = 4
    grad_accum: int = 2
    grad_clip_norm: float = 1.0
    ema_decay: Optional[float] = 0.999
    val_every_steps: int = 1000
    log_every_steps: int = 50
    candidate_steps: List[int] = Field(default_factory=lambda: [20000, 40000, 60000, 80000, 100000])
    num_workers: int = 0
    overfit_n_patches: int = 16
    # resume uses separate field; never generic checkpoint for init
    resume_exact_path: Optional[str] = None
    # Load VAE weights only (step/optim reset). For ablation warm-start from our
    # own S1 checkpoint. Mutually exclusive with resume_exact_path. Not ImageNet.
    warmstart_vae_path: Optional[str] = None
    # DDP: default splits yaml microbatch*accum across ranks (same global batch).
    # True keeps per-device microbatch and *multiplies* global batch by world_size.
    ddp_scale_global_batch: bool = False
    # Optional LR multiply by world_size. Requires ddp_scale_global_batch.
    ddp_scale_lr: bool = False


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_posterior_mean: bool = True
    stochastic_samples: int = 4  # reserved; not used for checkpoint selection
    allow_test: bool = False
    use_ema_for_val: bool = True
    report_constant_baseline: bool = True
    max_bootstrap: int = 200  # cap for wall-clock; full n_resamples only if smaller
    extended_metrics: bool = False
    # Post-hoc report only (eval-val-report). Not training losses.
    severe_mae_unit: float = 0.10
    severe_bg_fp_rate: float = 0.15
    severe_bg_bias: float = 0.02
    severe_bright_retention: float = 0.50
    severe_dark_grad_retention: float = 0.40
    worst_n: int = 20


class BootstrapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_resamples: int = 1000
    seed: int = 0


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    save_every_steps: int = 2000
    keep_last: int = 3
    keep_best_snr: bool = True
    keep_best_mae: bool = True


class ReproConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deterministic: bool = True
    cudnn_benchmark: bool = False


class SeedOrchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seeds: List[int] = Field(default_factory=lambda: [0, 1, 2])


class ProvenanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_env: bool = True


class GatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Require relative loss drop (initial-final)/initial >= this value (no hidden 0.5).
    overfit_loss_drop_frac: float = 0.9
    min_active_unit_frac: float = 0.5  # logged; hard gate optional in transition scripts


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str = "INFO"


class RootConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "microvae-config-v1"
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    latent: LatentConfig = Field(default_factory=LatentConfig)
    initialization: InitConfig = Field(default_factory=InitConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    kl_schedule: KLScheduleConfig = Field(default_factory=KLScheduleConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    reproducibility: ReproConfig = Field(default_factory=ReproConfig)
    seed_orchestration: SeedOrchConfig = Field(default_factory=SeedOrchConfig)
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def cross_field(self) -> "RootConfig":
        # crop divisible by spatial compression
        n_down = max(len(self.model.encoder_block_out_channels) - 1, 0)
        factor = 2**n_down
        if self.crop.size % factor != 0:
            raise ValueError(f"crop.size={self.crop.size} not divisible by f{factor}")
        for s in self.crop.multi_scale_sizes:
            if s % factor != 0:
                raise ValueError(f"multi_scale size {s} not divisible by f{factor}")
        # channel groups
        for c in list(self.model.encoder_block_out_channels) + list(self.model.decoder_block_out_channels):
            if c % self.model.norm_num_groups != 0:
                raise ValueError(f"channel {c} not divisible by norm groups")
        # forbid pretrained
        if self.model.pretrained or self.model.from_pretrained or self.model.init_from_weights:
            raise ValueError("pretrained / from_pretrained / init_from_weights forbidden")
        if self.initialization.mode != "fresh_init" and not self.training.resume_exact_path:
            raise ValueError("only fresh_init or explicit resume_exact_path allowed")
        if self.training.resume_exact_path and self.training.warmstart_vae_path:
            raise ValueError("resume_exact_path and warmstart_vae_path are mutually exclusive")
        if self.training.ddp_scale_lr and not self.training.ddp_scale_global_batch:
            raise ValueError("training.ddp_scale_lr requires training.ddp_scale_global_batch=true")
        if self.evaluation.allow_test:
            raise ValueError("evaluation.allow_test must be false until freeze-candidate credentials")
        if "test" in self.data.allow_splits:
            raise ValueError("test split cannot appear in training data.allow_splits")
        if self.normalization.fit_split != "train":
            raise ValueError("normalizer may only fit train")
        if self.latent.use_sd_scaling_factor:
            raise ValueError("SD scaling_factor must not be used as training constant")
        # output dir vs raw data
        if self.data.raw_data_root:
            out = self.experiment.output_dir.rstrip("/")
            raw = self.data.raw_data_root.rstrip("/")
            if out == raw or out.startswith(raw + "/"):
                raise ValueError("output_dir must not be inside raw_data_root")
        return self
