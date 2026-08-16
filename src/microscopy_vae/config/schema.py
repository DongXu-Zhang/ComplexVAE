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
    method: Literal["robust_linear_p0.1_p99.9", "identity"] = "robust_linear_p0.1_p99.9"
    fit_split: Literal["train"] = "train"
    # Linear-head codec: keep negatives and the p0.1/p99.9 tails (do NOT clip to [0,1]).
    clip: bool = False
    ssim_data_range: float = 1.0
    artifact_path: Optional[str] = None
    # page_uniform biases toward multi-page DI3D; source_balanced is recommended default.
    fit_mode: Literal["source_balanced", "page_uniform"] = "source_balanced"
    max_pages_fit: int = 192
    max_pixels_per_page: int = 65536


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
    # extra weight on |∇target|-weighted Charbonnier; 0 keeps v1 behaviour.
    edge_weight: float = 0.0
    # high-pass residual Charbonnier; 0 keeps v1.
    w_hf: float = 0.0
    ssim_range_mode: Literal["fixed", "amp_space"] = "fixed"


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


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_posterior_mean: bool = True
    stochastic_samples: int = 4  # reserved; not used for checkpoint selection
    allow_test: bool = False
    use_ema_for_val: bool = True
    report_constant_baseline: bool = True
    max_bootstrap: int = 200  # cap for wall-clock; full n_resamples only if smaller
    extended_metrics: bool = False


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
