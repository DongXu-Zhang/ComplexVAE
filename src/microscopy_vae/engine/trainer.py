from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from microscopy_vae import __version__
from microscopy_vae.config.loader import dump_resolved
from microscopy_vae.config.schema import RootConfig
from microscopy_vae.config.validation import validate_for_training
from microscopy_vae.data.hq_dataset import ManifestHQDataset, SyntheticHQDataset, collate_hq
from microscopy_vae.data.manifest import load_hq_manifest, manifest_sha256, summarize_records
from microscopy_vae.data.normalization import (
    NormalizationState,
    Normalizer,
    assert_artifact_matches_config,
    fit_robust_normalizer,
)
from microscopy_vae.data.threshold_calibration import THRESHOLD_VERSION, fit_structure_thresholds
from microscopy_vae.data.readers import read_page
from microscopy_vae.data.samplers import DistributedHierarchicalSampler, HierarchicalIndexSampler
from microscopy_vae.engine.distributed import (
    DistInfo,
    all_reduce_max_flag,
    assert_resume_world_size,
    barrier,
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    maybe_no_sync,
    raise_if_any_rank_failed,
    reduce_mean_map,
    reduce_sum_map,
    resolve_per_device_batch,
    strip_module_prefix,
    wrap_ddp,
)
from microscopy_vae.data.synthetic import build_synthetic_hq_pool
from microscopy_vae.engine.checkpoint import CheckpointManager
from microscopy_vae.engine.ema import EMA
from microscopy_vae.engine.evaluator import evaluate_hq_loader
from microscopy_vae.engine.schedulers import build_warmup_cosine_scheduler
from microscopy_vae.engine.state import TrainerState
from microscopy_vae.provenance.capture import write_environment
from microscopy_vae.provenance.hashing import sha256_file, sha256_json
from microscopy_vae.provenance.source_tree import hash_source_tree
from microscopy_vae.losses.influence import (
    diagnose_generator_influence,
    format_loss_breakdown,
    quantify_generator_losses,
)
from microscopy_vae.losses.schedule import scheduled_weight
from microscopy_vae.models.factory import architecture_id
from microscopy_vae.systems.factory import build_hq_codec_system
from microscopy_vae.utils.logging import append_jsonl, setup_logging
from microscopy_vae.utils.rng import seed_everything


class Trainer:
    """S1 HQ-codec trainer.

    Load modes:
    - fresh_init: default (ModelFactory)
    - resume_exact: only via training.resume_exact_path (full state)
    Never accepts a generic pretrained checkpoint for init.
    """

    def __init__(self, cfg: RootConfig) -> None:
        validate_for_training(cfg)
        if cfg.task.name != "hq_codec":
            raise ValueError("S1 trainer only supports task.name=hq_codec")
        if cfg.experiment.route != "hq_codec":
            raise ValueError("S1 trainer only supports experiment.route=hq_codec")

        self.cfg = cfg
        self.dist: DistInfo = init_distributed()
        self.device = self.dist.device
        self.logger = setup_logging(cfg.logging.level)
        if self.dist.enabled and not self.dist.is_main:
            import logging as _logging

            self.logger.setLevel(_logging.WARNING)
        self.logger.info(
            "experiment=%s output_dir=%s rank=%s world_size=%s device=%s ddp=%s "
            "clip=%s scale_mode=%s calibrate_thresholds=%s empty_keep_prob=%.3g",
            cfg.experiment.name,
            cfg.experiment.output_dir,
            self.dist.rank,
            self.dist.world_size,
            str(self.device),
            self.dist.enabled,
            bool(cfg.normalization.clip),
            str(cfg.normalization.scale_mode),
            bool(getattr(cfg.normalization, "calibrate_thresholds", False)),
            float(getattr(cfg.crop, "empty_keep_prob", 0.0)),
        )
        self.run_dir = Path(cfg.experiment.output_dir)
        scale_global = bool(getattr(cfg.training, "ddp_scale_global_batch", False))
        self.per_device_batch, self.grad_accum, self.effective_global_batch = resolve_per_device_batch(
            yaml_microbatch=int(cfg.training.microbatch_size),
            yaml_accum=int(cfg.training.grad_accum),
            world_size=int(self.dist.world_size),
            scale_global_batch=scale_global,
        )
        self._base_lr = float(cfg.optimizer.lr)
        if bool(getattr(cfg.training, "ddp_scale_lr", False)) and self.dist.world_size > 1:
            self._base_lr = float(cfg.optimizer.lr) * float(self.dist.world_size)
            self.logger.warning(
                "ddp_scale_lr=true: optimizer lr scaled %g -> %g (training dynamics change)",
                cfg.optimizer.lr,
                self._base_lr,
            )
        self.logger.info(
            "batch per_device=%s accum=%s world_size=%s effective_global=%s scale_global_batch=%s",
            self.per_device_batch,
            self.grad_accum,
            self.dist.world_size,
            self.effective_global_batch,
            scale_global,
        )
        if self.dist.enabled:
            self.logger.info(
                "ddp timeout_min=%s (env MICROVAE_DDP_TIMEOUT_MIN, default 360). "
                "Unset RANK/WORLD_SIZE/LOCAL_RANK before a single-GPU python launch.",
                os.environ.get("MICROVAE_DDP_TIMEOUT_MIN", "360"),
            )
        dir_err = ""
        if self.dist.is_main:
            try:
                if self.run_dir.exists() and any(self.run_dir.iterdir()):
                    allow = bool(getattr(cfg.experiment, "allow_existing_output", False))
                    if cfg.training.resume_exact_path:
                        allow = True
                    if not allow:
                        raise FileExistsError(
                            f"output_dir is non-empty: {self.run_dir}. "
                            "Refuse to mix runs. Set experiment.allow_existing_output=true or use resume_exact_path."
                        )
                self.run_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                dir_err = f"{type(exc).__name__}: {exc}"
        dir_err = broadcast_object(dir_err, self.dist, src=0)
        if dir_err:
            raise FileExistsError(dir_err)
        barrier(self.dist)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        from microscopy_vae.config.loader import config_semantic_hash

        self.config_sha = config_semantic_hash(cfg)
        if self.dist.is_main:
            dump_resolved(cfg, self.run_dir / "resolved_config.yaml")
            write_environment(self.run_dir / "environment.json")
        try:
            pkg_root = Path(__file__).resolve().parents[1]
            snap = hash_source_tree(pkg_root)
            if self.dist.is_main:
                (self.run_dir / "source_snapshot.json").write_text(
                    __import__("json").dumps(
                        {k: snap[k] for k in ("root", "n_files", "sha256")},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            self.source_sha = str(snap["sha256"])
        except Exception as exc:  # noqa: BLE001
            self.source_sha = f"unavailable:{exc}"
        seed_everything(cfg.experiment.seed, cfg.reproducibility.deterministic)
        if self.dist.enabled:
            from microscopy_vae.utils.rng import derive_seed

            torch.manual_seed(derive_seed(int(cfg.experiment.seed), int(self.dist.rank)))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(derive_seed(int(cfg.experiment.seed), int(self.dist.rank), 1))

        self.system = build_hq_codec_system(cfg).to(self.device)
        self.logger.info(
            "model architecture=%s spatial_compression=%s latent_channels=%s",
            architecture_id(self.system.vae),
            self.system.vae.spatial_compression,
            self.system.vae.latent_channels,
        )
        self.state = TrainerState()
        self.manifest_sha = "synthetic"
        self._test_loader_built = False  # invariant

        self._build_data_and_normalizer()
        self._build_optim()
        self._build_disc()
        self.ckpt = CheckpointManager(self.run_dir) if self.dist.is_main else None
        if self.dist.enabled and not self.dist.is_main:
            # Non-zero ranks still need a manager for resume_exact load (read-only).
            self.ckpt = CheckpointManager(self.run_dir)
        self._grad_checked = False
        self._contrib_ema: Dict[str, float] = {}
        self.ema: Optional[EMA] = None
        if cfg.training.ema_decay is not None and cfg.training.ema_decay > 0:
            self.ema = EMA(self.system.vae, decay=float(cfg.training.ema_decay))
        self.best_snr = float("-inf")
        self.best_mae = float("inf")
        self._enter_train_mode()

        # Optional resume_exact after optim built, before DDP wrap (same weights on all ranks).
        if cfg.training.resume_exact_path:
            self._resume_exact(Path(cfg.training.resume_exact_path))
        elif cfg.training.warmstart_vae_path:
            self._warmstart_vae(Path(cfg.training.warmstart_vae_path))
        self._wrap_ddp()

        # Prove no test loader attribute for training API
        assert not hasattr(self, "test_loader")

    def _build_data_and_normalizer(self) -> None:
        cfg = self.cfg
        mode = cfg.data.mode

        self._norm_sources: Optional[List[str]] = None
        if mode == "synthetic":
            pages = build_synthetic_hq_pool(
                n_groups=cfg.data.synthetic_n_groups,
                pages_per_group=cfg.data.synthetic_pages_per_group,
                size=max(cfg.data.synthetic_size, cfg.crop.size),
                seed=cfg.experiment.seed,
            )
            train_pages = [p for p in pages if p.split == "train"]
            train_imgs = [p.image for p in train_pages]
            self._norm_sources = [p.source for p in train_pages]
            self.manifest_sha = "synthetic"
            self._pages_or_records = pages
            train_cls = SyntheticHQDataset
            val_cls = SyntheticHQDataset
            train_arg = pages
            val_arg = pages
        elif mode == "hq_pool":
            if not cfg.data.manifest_path:
                raise ValueError("data.mode=hq_pool requires data.manifest_path")
            mpath = Path(cfg.data.manifest_path)
            # refuse_test=True: never return test rows
            records = load_hq_manifest(mpath, allow_splits=("train", "val"), refuse_test=True)
            # Optional Windows→Linux path prefix map (does not rewrite JSONL on disk)
            if cfg.data.path_prefix_target:
                from microscopy_vae.data.pathmap import PathPrefixMap, apply_prefix_map_to_records

                src_pref = cfg.data.path_prefix_source or "F:\\Dataset"
                pmap = PathPrefixMap(
                    source_prefixes=(src_pref, src_pref.replace("\\", "/"), src_pref.replace("/", "\\")),
                    target_root=cfg.data.path_prefix_target,
                    require_exists=bool(cfg.data.path_require_exists),
                )
                records = apply_prefix_map_to_records(records, pmap)
                self.logger.info(
                    "Applied path prefix map %r -> %r (require_exists=%s)",
                    src_pref,
                    cfg.data.path_prefix_target,
                    cfg.data.path_require_exists,
                )
            self.manifest_sha = manifest_sha256(mpath)
            self.logger.info("Loaded HQ manifest: %s", summarize_records(records))
            artifact_path = cfg.normalization.artifact_path
            if artifact_path and not Path(artifact_path).is_file():
                raise FileNotFoundError(
                    f"normalization.artifact_path does not exist: {artifact_path}. "
                    "Pass this run's normalizer.json; do not omit it and refit."
                )
            existing_norm = self.run_dir / "normalizer.json"
            load_existing = bool(artifact_path) or (
                bool(cfg.training.resume_exact_path) and existing_norm.is_file()
            )
            # Fit normalizer on train pages only (subsample for memory)
            train_recs = [r for r in records if r.split == "train"]
            reachable = [r for r in train_recs if Path(r.hq_path).is_file()]
            if cfg.data.path_require_exists and len(reachable) < min(8, len(train_recs)):
                raise FileNotFoundError(
                    f"path_require_exists=true but only {len(reachable)}/{len(train_recs)} "
                    f"train files are readable. Set data.path_prefix_target to the Linux mount "
                    f"of F:\\Dataset (see data_fix STATUS)."
                )
            if load_existing:
                # Eval / resume: do not reread 192 train pages just to throw them away.
                train_imgs, self._norm_sources = [], []
            elif reachable:
                if self.dist.is_main:
                    train_imgs, self._norm_sources = self._sample_train_images_for_norm(reachable)
                else:
                    train_imgs, self._norm_sources = [], []
            elif cfg.normalization.method == "identity":
                import numpy as np

                self.logger.warning(
                    "No HQ files reachable on this host (%d train records). "
                    "Using identity-range placeholder for normalizer fit. "
                    "Real training requires mount/copy + path_prefix_target.",
                    len(train_recs),
                )
                train_imgs = [np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)]
                self._norm_sources = ["placeholder"]
            else:
                raise FileNotFoundError(
                    "HQ files not reachable and normalization.method is not identity. "
                    "Either mount data and set path_prefix_target, or temporarily use "
                    "normalization.method=identity only for non-training dry checks."
                )
            self._pages_or_records = records
            train_cls = ManifestHQDataset
            val_cls = ManifestHQDataset
            train_arg = records
            val_arg = records
        elif mode == "paired_pool":
            raise ValueError(
                "paired_pool is not enabled in S1 HQ-only trainer. "
                "Use hq_codec route first; paired routes require stage_transition after gates."
            )
        else:
            raise ValueError(f"Unknown data.mode={mode}")

        # Normalizer: load artifact / resume artifact, else fit train-only
        if mode == "synthetic":
            n_train_groups = len(
                {p.group_id for p in self._pages_or_records if p.split == "train"}
            )
        else:
            n_train_groups = len(
                {r.group_id for r in self._pages_or_records if r.split == "train"}
            )

        existing_norm = self.run_dir / "normalizer.json"
        allow_legacy = bool(getattr(cfg.normalization, "allow_legacy_artifact", True))
        fit_err = ""
        norm_payload = None
        norm_sha = ""
        if self.dist.is_main:
          try:
            if cfg.training.resume_exact_path and not (
                (cfg.normalization.artifact_path and Path(cfg.normalization.artifact_path).is_file())
                or existing_norm.is_file()
            ):
                raise FileNotFoundError(
                    "resume_exact requires this run's normalizer.json "
                    f"(missing {existing_norm}). Refusing to refit a new scale on resume."
                )
            if cfg.normalization.artifact_path and Path(cfg.normalization.artifact_path).is_file():
                state = NormalizationState.load(Path(cfg.normalization.artifact_path))
                assert_artifact_matches_config(state, cfg.normalization, allow_legacy=allow_legacy)
                self.normalizer_sha = state.save(self.run_dir / "normalizer.json")
            elif cfg.training.resume_exact_path and existing_norm.is_file():
                state = NormalizationState.load(existing_norm)
                assert_artifact_matches_config(state, cfg.normalization, allow_legacy=allow_legacy)
                self.normalizer_sha = state.save(existing_norm)
            else:
                method = cfg.normalization.method
                state = fit_robust_normalizer(
                    train_imgs,
                    method=method if method != "identity" else "identity",
                    clip=cfg.normalization.clip,
                    n_groups=n_train_groups,
                    config_sha256=self.config_sha,
                    manifest_sha256=self.manifest_sha,
                    sources=self._norm_sources,
                    fit_mode=getattr(cfg.normalization, "fit_mode", "source_balanced"),
                    max_pixels_per_page=getattr(cfg.normalization, "max_pixels_per_page", 65536),
                    low_percentile=float(getattr(cfg.normalization, "low_percentile", 0.1)),
                    high_percentile=float(getattr(cfg.normalization, "high_percentile", 99.9)),
                    raw_floor_enabled=bool(getattr(cfg.normalization, "raw_floor_enabled", False)),
                    raw_floor_value=float(getattr(cfg.normalization, "raw_floor_value", 0.0)),
                    scale_mode=str(getattr(cfg.normalization, "scale_mode", "global")),
                )
                self.normalizer_sha = state.save(self.run_dir / "normalizer.json")
                self.logger.info(
                    "normalizer scale_mode=%s fit_mode=%s floor=%s p_high=%g low=%.6g high=%.6g pages=%s sources=%s",
                    state.scale_mode,
                    state.fit_mode,
                    state.raw_floor_enabled,
                    state.high_percentile,
                    state.low,
                    state.high,
                    state.n_pages_fit,
                    sorted(state.per_source_scales.keys()),
                )
            self.normalizer = Normalizer(state)
            if self.normalizer.state.fit_split != "train":
                raise RuntimeError("Normalizer fit_split must be train")
            self._maybe_calibrate_thresholds(state, train_imgs)
            norm_payload = self.normalizer.state.to_dict()
            norm_sha = str(self.normalizer_sha)
          except Exception as exc:  # noqa: BLE001
            fit_err = f"{type(exc).__name__}: {exc}"
        fit_err = broadcast_object(fit_err, self.dist, src=0)
        if fit_err:
            raise RuntimeError(fit_err)
        # Broadcast the fitted state. Do not make rank1 read a just-written
        # normalizer.json: NFS clients can miss the file after a barrier.
        norm_payload = broadcast_object(norm_payload, self.dist, src=0)
        norm_sha = broadcast_object(norm_sha, self.dist, src=0)
        if not self.dist.is_main:
            if not isinstance(norm_payload, dict):
                raise RuntimeError(f"rank {self.dist.rank} got no normalizer state from rank0")
            state = NormalizationState.from_dict(norm_payload)
            assert_artifact_matches_config(state, cfg.normalization, allow_legacy=allow_legacy)
            self.normalizer = Normalizer(state)
            self.normalizer_sha = str(norm_sha) if norm_sha else sha256_json(norm_payload)
            self.system.task.loss.set_calibrated_thresholds(state.per_source_thresholds)
        if self.normalizer.state.fit_split != "train":
            raise RuntimeError("Normalizer fit_split must be train")
        self._assert_per_source_coverage()
        if self.normalizer.is_per_source():
            for src, sc in sorted(self.normalizer.state.per_source_scales.items()):
                self.logger.info(
                    "per-source scale %s low=%.6g high=%.6g (y = max(x,0)/high)",
                    src,
                    float(sc["low"]),
                    float(sc["high"]),
                )

        crop_mode = str(getattr(cfg.crop, "mode", "random"))
        jitter = float(getattr(cfg.crop, "coverage_jitter_frac", 0.25))
        min_rr = float(getattr(cfg.crop, "min_robust_range", 0.0))
        max_rr = int(getattr(cfg.crop, "max_range_retries", 8))
        self.train_set = train_cls(
            train_arg,
            split="train",
            crop_size=cfg.crop.size,
            normalizer=self.normalizer,
            fixed_crops=False,
            seed=cfg.experiment.seed,
            crop_mode=crop_mode,
            coverage_jitter_frac=jitter,
            min_robust_range=min_rr,
            max_range_retries=max_rr,
            empty_keep_prob=float(getattr(cfg.crop, "empty_keep_prob", 0.0)),
        )
        self.val_set = val_cls(
            val_arg,
            split="val",
            crop_size=cfg.crop.size,
            normalizer=self.normalizer,
            fixed_crops=True,
            seed=cfg.experiment.seed + 1,
            crop_mode="random",
            coverage_jitter_frac=jitter,
            min_robust_range=0.0,
            max_range_retries=1,
        )

        slice_scores: Dict[int, float] = {}
        if str(getattr(cfg.sampling, "slice_weight_mode", "uniform")) == "focus_softmax":
            if mode != "hq_pool":
                self.logger.warning("focus_softmax ignored for mode=%s", mode)
            elif self.dist.is_main:
                from microscopy_vae.data.focus_index import resolve_slice_scores

                side = getattr(cfg.sampling, "focus_sidecar_path", None)
                try:
                    slice_scores = resolve_slice_scores(
                        self.train_set.records,  # type: ignore[attr-defined]
                        sidecar_path=Path(side) if side else None,
                        compute_if_missing=bool(getattr(cfg.sampling, "focus_compute_if_missing", False)),
                        cache_path=self.run_dir / "focus_sidecar_train.jsonl",
                        logger=self.logger,
                    )
                    self.logger.info(
                        "focus scores attached to %s/%s train slices",
                        len(slice_scores),
                        len(self.train_set),
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("focus scoring failed (%s); falling back to uniform slices", exc)
                    slice_scores = {}
        slice_scores = broadcast_object(slice_scores, self.dist, src=0)

        samp_kwargs = dict(
            seed=cfg.experiment.seed,
            source_weight_mode=cfg.sampling.source_weight_mode,
            fixed_source_prior=cfg.sampling.fixed_source_prior,
            epoch_length=max(len(self.train_set), self.per_device_batch * max(self.dist.world_size, 1)),
            slice_weight_mode=str(getattr(cfg.sampling, "slice_weight_mode", "uniform")),
            slice_scores=slice_scores,
            focus_temperature=float(getattr(cfg.sampling, "focus_temperature", 0.7)),
            focus_min_keep=float(getattr(cfg.sampling, "focus_min_keep", 0.15)),
        )
        if self.dist.enabled:
            train_sampler = DistributedHierarchicalSampler(
                self.train_set.meta,
                rank=self.dist.rank,
                world_size=self.dist.world_size,
                **samp_kwargs,
            )
        else:
            train_sampler = HierarchicalIndexSampler(self.train_set.meta, **samp_kwargs)
        self.train_sampler = train_sampler
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.per_device_batch,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
            **self._dataloader_kwargs(int(cfg.training.num_workers), persistent=True),
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.per_device_batch,
            shuffle=False,
            **self._dataloader_kwargs(min(2, int(cfg.training.num_workers)), persistent=False),
        )

    def _dataloader_kwargs(self, num_workers: int, *, persistent: bool) -> Dict[str, Any]:
        """CUDA is already initialized (DDP/device). Fork workers after that can hang."""
        kw: Dict[str, Any] = {
            "num_workers": int(num_workers),
            "collate_fn": collate_hq,
            "pin_memory": self.device.type == "cuda",
        }
        if int(num_workers) > 0:
            kw["persistent_workers"] = bool(persistent)
            if self.device.type == "cuda":
                kw["multiprocessing_context"] = "spawn"
        return kw

    def _log_normalized_batch(self, batch) -> None:
        x = batch.hq.detach().float()
        self.logger.info(
            "first-batch normalized hq min=%.5g max=%.5g mean=%.5g "
            "frac_lt0=%.4g frac_gt1=%.4g sources=%s",
            float(x.min()),
            float(x.max()),
            float(x.mean()),
            float((x < -1e-6).float().mean()),
            float((x > 1).float().mean()),
            sorted(set(str(s) for s in batch.sources)),
        )

    def _maybe_calibrate_thresholds(self, state: NormalizationState, train_imgs) -> None:
        """Fit or load per-source crop/support/amp thresholds. Train split only."""
        cfg = self.cfg
        if not bool(getattr(cfg.normalization, "calibrate_thresholds", False)):
            return
        if not state.per_source_thresholds:
            if not train_imgs or not self._norm_sources:
                raise RuntimeError(
                    "calibrate_thresholds=true but this run has no train images to fit "
                    "and normalizer.json has no per_source_thresholds. "
                    "Do not load a V4 artifact into a V5 config."
                )
            normed = [
                self.normalizer.transform(im, source=s)
                for im, s in zip(train_imgs, self._norm_sources)
            ]
            yaml_amp = float(cfg.loss.amp_low_structure_range or 0.0) or 0.08
            yaml_crop = float(cfg.crop.min_robust_range or 0.0) or yaml_amp
            thr, diag = fit_structure_thresholds(
                normed,
                self._norm_sources,
                crop_size=int(cfg.crop.size),
                kernel=int(cfg.loss.structure_support_kernel),
                rel=float(cfg.loss.structure_support_rel),
                min_density=float(cfg.loss.structure_support_min_density),
                structure_min_frac=float(cfg.loss.structure_min_frac or 0.0003),
                fallback_floor=float(cfg.loss.structure_support_floor),
                fallback_range=yaml_crop,
                fallback_amp_range=yaml_amp,
                bg_quantile=float(cfg.normalization.threshold_bg_quantile),
                bg_scharr_q=float(cfg.normalization.threshold_bg_scharr_q),
                empty_range_q=float(cfg.normalization.threshold_empty_range_q),
                struct_range_q=float(cfg.normalization.threshold_struct_range_q),
                crops_per_page=int(cfg.normalization.threshold_crops_per_page),
                seed=int(cfg.experiment.seed),
            )
            state.per_source_thresholds = thr
            state.threshold_version = THRESHOLD_VERSION
            for src, rec in diag.items():
                bucket = state.per_source_stats.setdefault(str(src), {})
                if not isinstance(bucket, dict):
                    bucket = {}
                    state.per_source_stats[str(src)] = bucket
                bucket["threshold_fit"] = rec
            self.normalizer_sha = state.save(self.run_dir / "normalizer.json")
            self.logger.info("fitted structure thresholds version=%s", THRESHOLD_VERSION)
        self.system.task.loss.set_calibrated_thresholds(state.per_source_thresholds)
        if self.normalizer.is_per_source():
            missing_thr = set(self.normalizer.known_sources()) - set(state.per_source_thresholds)
            if missing_thr:
                raise ValueError(
                    "calibrate_thresholds=true but per_source_thresholds is missing "
                    f"{sorted(missing_thr)}; have={sorted(state.per_source_thresholds)}. "
                    "Refusing yaml 0.08/0.02 fallback for a fitted source."
                )
        for src, rec in sorted(state.per_source_thresholds.items()):
            self.logger.info(
                "threshold %s support_floor=%.6g crop_range=%.6g amp_range=%.6g",
                src,
                float(rec.get("structure_support_floor", 0.0)),
                float(rec.get("crop_min_robust_range", 0.0)),
                float(rec.get("amp_low_structure_range", 0.0)),
            )

    def _assert_per_source_coverage(self) -> None:
        """Fail at init if a train/val source has no fitted scale (would crash mid-epoch)."""
        if not self.normalizer.is_per_source():
            return
        items = self._pages_or_records
        needed = set()
        for it in items:
            split = getattr(it, "split", None)
            if split in ("train", "val"):
                needed.add(str(it.source))
        fitted = set(self.normalizer.known_sources())
        missing = needed - fitted
        if missing:
            raise ValueError(
                "per-source normalizer is missing scales for "
                f"{sorted(missing)}; fitted={sorted(fitted)}. "
                "Those sources probably have no readable train files. "
                "Set data.path_prefix_target to the Linux data root and "
                "data.path_require_exists=true."
            )

    def _sample_train_images_for_norm(
        self, train_recs, max_pages: Optional[int] = None
    ) -> tuple:
        """Source-stratified page sample for normalizer fit. Returns (imgs, sources)."""
        max_pages = max_pages or int(getattr(self.cfg.normalization, "max_pages_fit", 192))
        rng = np.random.default_rng(self.cfg.experiment.seed)
        by_src: Dict[str, List[Any]] = {}
        for r in train_recs:
            by_src.setdefault(str(r.source), []).append(r)
        # equal budget per source
        n_src = max(len(by_src), 1)
        per = max(8, max_pages // n_src)
        chosen = []
        for s, rs in sorted(by_src.items()):
            idxs = np.arange(len(rs))
            if len(idxs) > per:
                idxs = rng.choice(idxs, size=per, replace=False)
            for i in idxs:
                chosen.append(rs[int(i)])
        imgs, sources = [], []
        for r in chosen:
            page, _ = read_page(r.hq_path, r.hq_page, expected_dtype=r.hq_dtype)
            imgs.append(page)
            sources.append(str(r.source))
        if not imgs:
            raise ValueError("No train images for normalizer fit")
        return imgs, sources

    def _build_optim(self) -> None:
        cfg = self.cfg
        self.optimizer = torch.optim.AdamW(
            self.system.parameters(),
            lr=self._base_lr,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=cfg.scheduler.warmup_steps,
            max_steps=cfg.training.max_steps,
            min_lr=cfg.scheduler.min_lr,
            base_lr=self._base_lr,
        )
        self.use_amp = cfg.precision.amp_dtype == "bf16" and self.device.type == "cuda"
        self.scaler = None

    def _build_disc(self) -> None:
        adv = self.cfg.loss.adversarial
        self.discriminator = None
        self.disc_optimizer = None
        self.disc_scheduler = None
        if not adv.enabled:
            return
        from microscopy_vae.losses.adversarial import PatchDiscriminator

        if adv.architecture != "patchgan":
            raise ValueError(f"unsupported discriminator {adv.architecture!r}")
        if adv.conditioning == "input":
            self.logger.warning(
                "adversarial.conditioning=input is degenerate for S1 HQ codec "
                "(input==target: concat(x,x) vs concat(x,recon) is a channel-identity cue). "
                "Prefer conditioning=none unless this is a paired route."
            )
        self.discriminator = PatchDiscriminator(
            in_channels=1,
            ndf=adv.ndf,
            n_layers=adv.n_layers,
            kernel_size=adv.kernel_size,
            spectral_norm=adv.spectral_norm,
            conditioning=adv.conditioning,
        ).to(self.device)
        disc_lr = float(adv.disc_lr)
        if bool(getattr(self.cfg.training, "ddp_scale_lr", False)) and self.dist.world_size > 1:
            disc_lr *= float(self.dist.world_size)
        self.disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=adv.disc_betas,
            eps=self.cfg.optimizer.eps,
            weight_decay=adv.disc_weight_decay,
        )
        if adv.disc_scheduler == "cosine":
            self.disc_scheduler = build_warmup_cosine_scheduler(
                self.disc_optimizer,
                warmup_steps=self.cfg.scheduler.warmup_steps,
                max_steps=self.cfg.training.max_steps,
                min_lr=self.cfg.scheduler.min_lr,
                base_lr=disc_lr,
            )
        self.logger.info(
            "GAN discriminator: patchgan ndf=%s n_layers=%s sn=%s cond=%s loss=%s",
            adv.ndf,
            adv.n_layers,
            adv.spectral_norm,
            adv.conditioning,
            adv.gan_loss,
        )

    def _enter_train_mode(self) -> None:
        """VAE train; frozen perceptual must stay eval (no BN, but keep the contract)."""
        self.system.train()
        if self.system.perceptual is not None:
            self.system.perceptual.eval()
        if getattr(self, "vae_ddp", None) is not None:
            self.vae_ddp.train()
        if getattr(self, "disc_ddp", None) is not None:
            self.disc_ddp.train()

    def _wrap_ddp(self) -> None:
        self.vae_ddp = None
        self.disc_ddp = None
        if not self.dist.enabled:
            return
        self.vae_ddp = wrap_ddp(self.system.vae, self.dist)
        self.system.task.model = self.vae_ddp
        if self.discriminator is not None:
            self.disc_ddp = wrap_ddp(self.discriminator, self.dist)
        self.logger.info("wrapped DDP vae=%s disc=%s", self.vae_ddp is not None, self.disc_ddp is not None)

    def _disc_module(self):
        """DDP discriminator for D updates (grads allreduced)."""
        return self.disc_ddp if getattr(self, "disc_ddp", None) is not None else self.discriminator

    def close(self) -> None:
        cleanup_distributed(self.dist)

    def _adv_weight(self, step: int) -> float:
        adv = self.cfg.loss.adversarial
        if not adv.enabled or self.discriminator is None:
            return 0.0
        return scheduled_weight(adv.weight, step, adv.start_step, adv.ramp_steps)

    def _disc_mask(self, support_on: torch.Tensor) -> Optional[torch.Tensor]:
        adv = self.cfg.loss.adversarial
        if adv.unstructured_policy != "exclude":
            return None
        return support_on.detach()

    def _ckpt_extra(self, **more: Any) -> Dict[str, Any]:
        extra: Dict[str, Any] = {
            "ema": self.ema.state_dict() if self.ema else None,
            "source_sha": getattr(self, "source_sha", ""),
            "gan_enabled": self.discriminator is not None,
            "perc_enabled": self.system.perceptual is not None,
            "spatial_compression": int(self.system.vae.spatial_compression),
            "latent_channels": int(self.system.vae.latent_channels),
            "architecture_id": architecture_id(self.system.vae),
        }
        if self.system.perceptual is not None:
            extra["perceptual"] = self.system.perceptual.state_dict()
        if self.discriminator is not None:
            extra["discriminator"] = strip_module_prefix(self.discriminator.state_dict())
            extra["disc_optimizer"] = self.disc_optimizer.state_dict() if self.disc_optimizer else None
            extra["disc_scheduler"] = (
                self.disc_scheduler.state_dict() if self.disc_scheduler is not None else None
            )
        extra["sampler"] = self.train_sampler.state_dict() if hasattr(self, "train_sampler") else None
        extra["per_device_batch"] = int(self.per_device_batch)
        extra["grad_accum"] = int(self.grad_accum)
        extra["effective_global_batch"] = int(self.effective_global_batch)
        extra["world_size"] = int(self.dist.world_size)
        extra.update(more)
        return extra

    def _save_ckpt(self, tag: str, *, prune: bool = False, **more: Any):
        err = ""
        path = None
        if self.dist.is_main:
            try:
                path = self.ckpt.save_exact(
                    tag=tag,
                    model=self.system.vae,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    state=self.state,
                    config_sha256=self.config_sha,
                    normalizer_sha256=self.normalizer_sha,
                    code_version=__version__,
                    extra=self._ckpt_extra(**more),
                )
                if prune:
                    self.ckpt.prune_periodic(keep_last=int(self.cfg.checkpoint.keep_last))
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
        err = broadcast_object(err, self.dist, src=0)
        if err:
            raise RuntimeError(f"checkpoint save failed: {err}")
        return path

    def _warmstart_vae(self, path: Path) -> None:
        self.logger.info("warmstart VAE weights from %s (optim/step reset, not resume_exact)", path)
        CheckpointManager.load_exported_weights(path, self.system.vae, map_location=str(self.device))
        payload = None
        try:
            payload = torch.load(path, map_location=str(self.device), weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=str(self.device))
        extra = payload.get("extra") if isinstance(payload, dict) else None
        if self.ema is not None and isinstance(extra, dict) and extra.get("ema"):
            self.ema.load_state_dict(extra["ema"])
        # Discriminator / perc stay freshly built for this config. Trainer state is step 0.

    def _resume_exact(self, path: Path) -> None:
        self.logger.info("resume_exact from %s", path)
        err = ""
        extra: Dict[str, Any] = {}
        try:
            self.state, extra = CheckpointManager.resume_exact(
                path,
                model=self.system.vae,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                expected_config_sha256=self.config_sha,
                expected_normalizer_sha256=self.normalizer_sha,
                map_location=str(self.device),
                expected_code_version=None,  # warn-only: set if you need hard pin
            )
            extra = extra or {}
            if self.ema is not None and isinstance(extra, dict) and extra.get("ema"):
                self.ema.load_state_dict(extra["ema"])
            if self.system.perceptual is not None:
                if extra.get("perceptual"):
                    self.system.perceptual.load_state_dict(extra["perceptual"])
                else:
                    self.logger.warning(
                        "perceptual enabled but checkpoint extra has no perceptual weights; "
                        "using freshly initialized frozen extractor (init_seed)"
                    )
            if self.discriminator is not None:
                if not extra.get("discriminator"):
                    raise ValueError(
                        "adversarial.enabled=true but checkpoint has no discriminator state; "
                        "cannot resume_exact a non-GAN run as a GAN run (use warmstart_vae_path)"
                    )
                self.discriminator.load_state_dict(strip_module_prefix(extra["discriminator"]))
                if self.disc_optimizer is not None and extra.get("disc_optimizer"):
                    self.disc_optimizer.load_state_dict(extra["disc_optimizer"])
                if self.disc_scheduler is not None and extra.get("disc_scheduler"):
                    self.disc_scheduler.load_state_dict(extra["disc_scheduler"])
            if extra.get("sampler") and hasattr(self, "train_sampler"):
                assert_resume_world_size(extra.get("world_size"), int(self.dist.world_size))
                self.train_sampler.load_state_dict(extra["sampler"])
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        raise_if_any_rank_failed(not err, err or "resume_exact failed", self.dist)

    def _assert_finite_grads(self) -> None:
        bad = []
        for n, p in self.system.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad.append(n)
        if bad:
            raise RuntimeError(f"Non-finite gradients in: {bad[:20]}")
        critical_prefixes = ("vae.encoder", "vae.quant_conv", "vae.post_quant_conv", "vae.decoder")
        missing = []
        for pref in critical_prefixes:
            has = any(
                n.startswith(pref) and p.grad is not None and float(p.grad.detach().abs().sum()) > 0
                for n, p in self.system.named_parameters()
            )
            if not has:
                missing.append(pref)
        if missing:
            raise RuntimeError(f"Missing nonzero grads for: {missing}")

    def _attach_adv_g(self, loss_out, batch) -> Any:
        """Add generator adversarial term. D params must not receive these grads."""
        if self.discriminator is None:
            return loss_out
        w = self._adv_weight(self.state.optimizer_step)
        loss_out.diagnostics["w_adv_effective"] = torch.tensor(w, device=batch.hq.device)
        if w <= 0:
            zero = loss_out.total.new_zeros(())
            loss_out.unweighted["adv_g"] = zero
            loss_out.weights["adv_g"] = 0.0
            loss_out.weighted["adv_g"] = zero
            return loss_out
        from microscopy_vae.losses.adversarial import generator_adv_loss

        recon = loss_out.aux["reconstruction"]
        cond = batch.hq if self.cfg.loss.adversarial.conditioning == "input" else None
        mask = self._disc_mask(loss_out.aux["support_on"])
        self.discriminator.requires_grad_(False)
        # Unwrapped D: G backward must reach recon, not allreduce unused D params.
        g_adv, fake_score = generator_adv_loss(
            self.discriminator,
            recon,
            cond=cond,
            mask=mask,
            gan_loss=self.cfg.loss.adversarial.gan_loss,
        )
        loss_out.unweighted["adv_g"] = g_adv
        loss_out.weights["adv_g"] = w
        loss_out.weighted["adv_g"] = w * g_adv
        loss_out.total = loss_out.total + w * g_adv
        loss_out.diagnostics["d_fake_mean_g"] = fake_score
        return loss_out

    def _disc_zero_logs(self) -> Dict[str, float]:
        return {
            "loss_raw_disc": 0.0,
            "loss_disc_real": 0.0,
            "loss_disc_fake": 0.0,
            "d_real_mean": 0.0,
            "d_fake_mean": 0.0,
        }

    def _dummy_disc_backward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> None:
        """Keep the DDP reducer happy when this rank has no structured D patches.

        The real D step does two forwards (real + fake) then one backward.
        Dummy must match that count or mixed empty/structured ranks hang.
        """
        if getattr(self, "disc_ddp", None) is None:
            return
        disc = self._disc_module()
        real_logits = disc(real, cond=cond)
        fake_logits = disc(fake, cond=cond)
        ((real_logits.sum() + fake_logits.sum()) * 0.0).backward()

    def _backward_disc(self, loss_out, batch, accum: int) -> Dict[str, float]:
        """Discriminator update backward (accumulated). recon is detached.

        Never raise between a DDP forward and backward: finish the dummy/real
        backward first, then the trainer checks ``_d_loss_ok``.
        """
        self._disc_used_real = False
        self._d_loss_ok = True
        logs: Dict[str, float] = {}
        if self.discriminator is None or self._adv_weight(self.state.optimizer_step) <= 0:
            return logs
        from microscopy_vae.losses.adversarial import discriminator_scores

        adv = self.cfg.loss.adversarial
        recon = loss_out.aux["reconstruction"].detach()
        real = loss_out.aux["target"].detach()
        cond = batch.hq.detach() if adv.conditioning == "input" else None
        mask = self._disc_mask(loss_out.aux["support_on"])
        logs = self._disc_zero_logs()
        empty = mask is not None and float(mask.sum()) <= 0
        self.discriminator.requires_grad_(True)
        if empty:
            self._dummy_disc_backward(real, recon, cond)
            return logs
        scores = discriminator_scores(
            self._disc_module(),
            real=real,
            fake=recon,
            cond=cond,
            mask=mask,
            gan_loss=adv.gan_loss,
            r1_gamma=float(adv.r1_gamma),
        )
        d_loss = scores["loss_d"] / float(accum)
        d_ok = bool(torch.isfinite(d_loss.detach()).all().item())
        d_loss.backward()
        if not d_ok:
            self._d_loss_ok = False
            return logs
        self._disc_used_real = True
        logs["loss_raw_disc"] = float(scores["loss_d"].detach().cpu())
        logs["loss_disc_real"] = float(scores["loss_d_real"].detach().cpu())
        logs["loss_disc_fake"] = float(scores["loss_d_fake"].detach().cpu())
        logs["d_real_mean"] = float(scores["d_real_mean"].detach().cpu())
        logs["d_fake_mean"] = float(scores["d_fake_mean"].detach().cpu())
        return logs

    def _extra_critic_steps(self, loss_out, batch) -> None:
        n_extra = int(self.cfg.loss.adversarial.n_critic) - 1
        if n_extra <= 0 or self.discriminator is None or self.disc_optimizer is None:
            return
        from microscopy_vae.losses.adversarial import discriminator_scores

        adv = self.cfg.loss.adversarial
        recon = loss_out.aux["reconstruction"].detach()
        real = loss_out.aux["target"].detach()
        cond = batch.hq.detach() if adv.conditioning == "input" else None
        mask = self._disc_mask(loss_out.aux["support_on"])
        for _ in range(n_extra):
            self.disc_optimizer.zero_grad(set_to_none=True)
            self.discriminator.requires_grad_(True)
            scores = discriminator_scores(
                self._disc_module(),
                real=real,
                fake=recon,
                cond=cond,
                mask=mask,
                gan_loss=adv.gan_loss,
                r1_gamma=float(adv.r1_gamma),
            )
            scores["loss_d"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                adv.grad_clip_norm if adv.grad_clip_norm > 0 else 1e9,
            )
            self.disc_optimizer.step()

    def _maybe_influence(self, loss_out, last_micro: bool) -> Dict[str, float]:
        infl_cfg = self.cfg.loss.influence
        logs: Dict[str, float] = {}
        if infl_cfg.log_contrib_ratio and last_micro:
            # Shares come from quantify_generator_losses (full 9-term table).
            # Here only keep a running EMA of |C_i|.
            decay = float(infl_cfg.ema_decay)
            for k, t in loss_out.weighted.items():
                c = abs(float(t.detach().cpu()))
                prev = self._contrib_ema.get(k)
                ema = c if prev is None else decay * prev + (1.0 - decay) * c
                self._contrib_ema[k] = ema
                logs[f"contrib_ema_{k}"] = ema
        every = int(infl_cfg.grad_every_steps)
        logged_step = self.state.optimizer_step + 1
        want_grad = last_micro and every > 0 and logged_step % every == 0
        want_cos = (
            last_micro
            and int(infl_cfg.cosine_every_steps) > 0
            and logged_step % int(infl_cfg.cosine_every_steps) == 0
        )
        if want_grad or want_cos:
            if self.dist.enabled:
                # autograd.grad on terms from a DDP forward would fire the reducer
                # before the training backward and crash at find_unused=false.
                logs["influence_grad_skipped_ddp"] = 1.0
            else:
                logs.update(
                    diagnose_generator_influence(
                        loss_out.weighted,
                        self.system.vae,
                        param_group_names=tuple(infl_cfg.param_groups),
                        compute_cosine=want_cos,
                    )
                )
        return logs

    def _reduce_train_diag(self, last_diag: Dict[str, Any], step_loss: float) -> Dict[str, Any]:
        scalars: Dict[str, float] = {"loss": float(step_loss)}
        for k, v in last_diag.items():
            if k in {"sampler_source_freq", "sampler_planned_probs"}:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                scalars[k] = float(v)
        reduced = reduce_mean_map(scalars, self.dist, self.device)
        last_diag.update(reduced)
        inner = getattr(self.train_sampler, "inner", self.train_sampler)
        draws = {str(s): float(inner.source_draws.get(s, 0)) for s in getattr(inner, "sources", [])}
        draws = reduce_sum_map(draws, self.dist, self.device)
        total = sum(draws.values())
        last_diag["sampler_source_freq"] = (
            {k: (v / total if total > 0 else 0.0) for k, v in draws.items()} if draws else {}
        )
        return last_diag

    def _maybe_validate(self) -> Optional[Dict[str, Any]]:
        """Rank0 runs val. Every rank rendezvous so a rank0 exception cannot hang peers."""
        err = ""
        rec = None
        if self.dist.is_main:
            try:
                rec = self._validate_on_main()
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
        err = broadcast_object(err, self.dist, src=0)
        if err:
            raise RuntimeError(f"validation failed: {err}")
        return rec

    def _validate_on_main(self) -> Dict[str, Any]:
        use_ema = bool(getattr(self.cfg.evaluation, "use_ema_for_val", True)) and self.ema is not None
        live_sd = None
        if use_ema:
            live_sd = {k: v.detach().cpu().clone() for k, v in self.system.vae.state_dict().items()}
            self.ema.copy_to(self.system.vae)
        boot_n = min(
            int(getattr(self.cfg.evaluation, "max_bootstrap", 200)),
            int(self.cfg.bootstrap.n_resamples),
        )
        metrics = evaluate_hq_loader(
            self.system,
            self.val_loader,
            device=self.device,
            use_posterior_mean=self.cfg.evaluation.use_posterior_mean,
            bootstrap_n=boot_n,
            bootstrap_seed=self.cfg.bootstrap.seed,
            report_constant_baseline=bool(
                getattr(self.cfg.evaluation, "report_constant_baseline", True)
            ),
            extended_metrics=bool(getattr(self.cfg.evaluation, "extended_metrics", False)),
        )
        if use_ema and live_sd is not None:
            self.system.vae.load_state_dict(live_sd)
        rec = {
            "step": self.state.optimizer_step,
            "weights": "ema" if use_ema else "live",
            "group_macro": metrics["group_macro"],
            "by_source": metrics.get("by_source"),
            "equal_source_macro": metrics.get("equal_source_macro"),
            "constant_baseline": metrics.get("constant_baseline"),
            "n_pages": metrics["n_pages"],
            "n_groups": metrics["n_groups"],
            "psnr_bootstrap": metrics["psnr_bootstrap"],
        }
        append_jsonl(self.run_dir / "metrics_val.jsonl", rec)
        self.logger.info(
            "val step=%s weights=%s group_macro_psnr=%.4f mae=%.6f snr=%.4f pooled=%.4f by_source=%s",
            self.state.optimizer_step,
            rec["weights"],
            metrics["group_macro"].get("psnr", float("nan")),
            metrics["group_macro"].get("mae", float("nan")),
            metrics["group_macro"].get("snr_db", float("nan")),
            metrics["group_macro"].get("psnr_mse_pooled", float("nan")),
            {k: v.get("psnr") for k, v in (metrics.get("by_source") or {}).items()},
        )
        self._maybe_save_best(metrics, rec)
        if self.state.optimizer_step in set(self.cfg.training.candidate_steps):
            tag = f"candidate_step_{self.state.optimizer_step:07d}"
            path = self.ckpt.save_exact(
                tag=tag,
                model=self.system.vae,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                state=self.state,
                config_sha256=self.config_sha,
                normalizer_sha256=self.normalizer_sha,
                code_version=__version__,
                extra=self._ckpt_extra(candidate=True, val=rec),
            )
            self.state.candidate_hits[self.state.optimizer_step] = str(path)
        return rec

    def _maybe_save_best(self, metrics: Dict[str, Any], rec: Dict[str, Any]) -> None:
        gm = metrics.get("group_macro") or {}
        mae_v = gm.get("mae")
        snr_v = gm.get("snr_db")
        if snr_v is None or not np.isfinite(snr_v):
            # fall back to range-1 PSNR if extended metrics off
            snr_v = gm.get("psnr")
        if (
            getattr(self.cfg.checkpoint, "keep_best_snr", True)
            and snr_v is not None
            and np.isfinite(snr_v)
            and float(snr_v) > self.best_snr
        ):
            self.best_snr = float(snr_v)
            self.ckpt.save_exact(
                tag="best_snr",
                model=self.system.vae,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                state=self.state,
                config_sha256=self.config_sha,
                normalizer_sha256=self.normalizer_sha,
                code_version=__version__,
                extra=self._ckpt_extra(kind="best_snr", val=rec),
            )
            self.logger.info("new best SNR/PSNR-proxy=%.4f at step %s", self.best_snr, self.state.optimizer_step)
        if (
            getattr(self.cfg.checkpoint, "keep_best_mae", True)
            and mae_v is not None
            and np.isfinite(mae_v)
            and float(mae_v) < self.best_mae
        ):
            self.best_mae = float(mae_v)
            self.ckpt.save_exact(
                tag="best_mae",
                model=self.system.vae,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                state=self.state,
                config_sha256=self.config_sha,
                normalizer_sha256=self.normalizer_sha,
                code_version=__version__,
                extra=self._ckpt_extra(kind="best_mae", val=rec),
            )
            self.logger.info("new best MAE=%.6f at step %s", self.best_mae, self.state.optimizer_step)

    def train(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        max_steps = max_steps if max_steps is not None else self.cfg.training.max_steps
        self._enter_train_mode()
        loader_err = ""
        it = None
        try:
            it = iter(self.train_loader)
        except Exception as exc:  # noqa: BLE001
            loader_err = f"{type(exc).__name__}: {exc}"
        raise_if_any_rank_failed(
            not loader_err,
            loader_err or "train_loader failed",
            self.dist,
        )
        assert it is not None
        accum = int(self.grad_accum)
        self.optimizer.zero_grad(set_to_none=True)
        history: List[float] = []

        # Audit trainability at start of S1
        audit_ok = True
        audit_msg = "S1 requires all core parameters trainable (fresh_init / unlocked)"
        try:
            audit = self.system.trainability_audit()
            if not audit["all_core_trainable"]:
                audit_ok = False
        except Exception as exc:  # noqa: BLE001
            audit_ok = False
            audit_msg = f"{type(exc).__name__}: {exc}"
        raise_if_any_rank_failed(audit_ok, audit_msg, self.dist)

        while self.state.optimizer_step < max_steps:
            loss_accum = 0.0
            last_diag: Dict[str, Any] = {}
            last_loss_out = None
            last_batch = None
            d_used_any = False
            if self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
            for _micro in range(accum):
                fetch_err = ""
                try:
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(self.train_loader)
                        batch = next(it)
                    batch.hq = batch.hq.to(self.device, non_blocking=True)
                except Exception as exc:  # noqa: BLE001
                    fetch_err = f"{type(exc).__name__}: {exc}"
                raise_if_any_rank_failed(
                    not fetch_err,
                    fetch_err or "batch fetch failed",
                    self.dist,
                )
                if not getattr(self, "_logged_norm_batch", False):
                    self._log_normalized_batch(batch)
                    self._logged_norm_batch = True
                last_micro = _micro == accum - 1
                self._d_loss_ok = True
                sync_ctx = maybe_no_sync(self.vae_ddp, enabled=self.dist.enabled and not last_micro)
                disc_ctx = maybe_no_sync(self.disc_ddp, enabled=self.dist.enabled and not last_micro)
                with sync_ctx, disc_ctx:
                    if self.use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss_out = self.system.task.forward_loss(
                                batch, optimizer_step=self.state.optimizer_step
                            )
                    else:
                        loss_out = self.system.task.forward_loss(
                            batch, optimizer_step=self.state.optimizer_step
                        )
                    loss_out = self._attach_adv_g(loss_out, batch)
                    last_diag.update(
                        quantify_generator_losses(
                            loss_out.unweighted,
                            loss_out.weights,
                            loss_out.weighted,
                            total=loss_out.total,
                        )
                    )
                    last_diag.update(self._maybe_influence(loss_out, last_micro))
                    last_diag.update(self._backward_disc(loss_out, batch, accum))
                    d_used_any = d_used_any or bool(getattr(self, "_disc_used_real", False))
                    if self.discriminator is not None:
                        self.discriminator.requires_grad_(False)
                    loss = loss_out.total / accum
                    loss.backward()
                g_ok = bool(torch.isfinite(loss_out.total.detach()).all().item())
                d_ok = bool(getattr(self, "_d_loss_ok", True))
                raise_if_any_rank_failed(
                    g_ok and d_ok,
                    f"Non-finite loss at step {self.state.optimizer_step} "
                    f"samples={batch.sample_ids} sources={batch.sources} d_ok={d_ok}",
                    self.dist,
                )
                loss_accum += float(loss_out.total.detach().cpu())
                last_loss_out = loss_out
                last_batch = batch
                last_diag.update(
                    {
                        k: float(v.detach().cpu())
                        for k, v in loss_out.diagnostics.items()
                        if torch.is_tensor(v) and v.ndim == 0
                    }
                )
                self.state.microbatch += 1
                self.state.global_samples += int(batch.hq.shape[0]) * int(self.dist.world_size)

            # Standard GAN order: step D first (grads from detached fake), then G.
            disc_pre = 0.0
            if self.disc_optimizer is not None and self._adv_weight(self.state.optimizer_step) > 0:
                d_used_any = all_reduce_max_flag(d_used_any, self.dist, self.device)
                if d_used_any:
                    disc_clip = (
                        self.cfg.loss.adversarial.grad_clip_norm
                        if self.cfg.loss.adversarial.grad_clip_norm > 0
                        else 1e9
                    )
                    disc_pre = float(
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), disc_clip)
                    )
                    last_diag["grad_norm_disc_pre_clip"] = disc_pre
                    self.disc_optimizer.step()
                    if self.disc_scheduler is not None:
                        self.disc_scheduler.step()
                    if last_loss_out is not None and last_batch is not None:
                        self._extra_critic_steps(last_loss_out, last_batch)
                self.disc_optimizer.zero_grad(set_to_none=True)
                self.discriminator.requires_grad_(True)

            pre_clip = float(
                torch.nn.utils.clip_grad_norm_(
                    self.system.parameters(),
                    self.cfg.training.grad_clip_norm
                    if self.cfg.training.grad_clip_norm > 0
                    else 1e9,
                )
            )
            g_grad_ok = True
            bad_name = ""
            if not self._grad_checked:
                try:
                    self._assert_finite_grads()
                except RuntimeError as exc:
                    g_grad_ok = False
                    bad_name = str(exc)
                self._grad_checked = True
            else:
                for n, p in self.system.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        g_grad_ok = False
                        bad_name = n
                        break
            raise_if_any_rank_failed(
                g_grad_ok,
                f"Non-finite grad at step {self.state.optimizer_step}: {bad_name}",
                self.dist,
            )
            last_diag["grad_norm_pre_clip"] = float(pre_clip)
            self.optimizer.step()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.system.vae)
            self.optimizer.zero_grad(set_to_none=True)
            self.state.optimizer_step += 1
            step_loss = loss_accum / accum
            last_diag = self._reduce_train_diag(last_diag, step_loss)
            history.append(float(last_diag.get("loss", step_loss)))

            if self.state.optimizer_step % self.cfg.training.log_every_steps == 0:
                rec = {
                    "step": self.state.optimizer_step,
                    "loss": history[-1],
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm_pre_clip": last_diag.get("grad_norm_pre_clip", pre_clip),
                    "grad_clipped": pre_clip > float(self.cfg.training.grad_clip_norm or 0),
                    "per_device_batch": self.per_device_batch,
                    "world_size": self.dist.world_size,
                    "effective_global_batch": self.effective_global_batch,
                    **last_diag,
                    "sampler_source_freq": last_diag.get("sampler_source_freq")
                    or self.train_sampler.realized_source_freq(),
                    "sampler_planned_probs": self.train_sampler.planned_source_probs(),
                }
                if self.disc_optimizer is not None:
                    rec["lr_disc"] = self.disc_optimizer.param_groups[0]["lr"]
                if self.dist.is_main:
                    append_jsonl(self.run_dir / "metrics_train.jsonl", rec)
                    self.logger.info(
                        "step=%s lr=%.2e idle=%.3f | %s",
                        self.state.optimizer_step,
                        rec["lr"],
                        last_diag.get("idle_frac", float("nan")),
                        format_loss_breakdown(last_diag),
                    )

            if (
                self.cfg.training.val_every_steps > 0
                and self.state.optimizer_step % self.cfg.training.val_every_steps == 0
            ):
                self._maybe_validate()
                self._enter_train_mode()

            if self.state.optimizer_step % self.cfg.checkpoint.save_every_steps == 0:
                if self.state.optimizer_step not in set(self.cfg.training.candidate_steps):
                    self._save_ckpt(f"step_{self.state.optimizer_step:07d}", prune=True)

        path = self._save_ckpt(f"step_{self.state.optimizer_step:07d}_final")
        return {
            "final_step": self.state.optimizer_step,
            "final_loss": history[-1] if history else None,
            "checkpoint": str(path) if path is not None else None,
            "loss_history": history,
            "candidate_hits": dict(self.state.candidate_hits),
            "sampler_source_freq": self.train_sampler.realized_source_freq(),
            "effective_global_batch": self.effective_global_batch,
            "world_size": self.dist.world_size,
        }

    def _select_overfit_indices(self, n_target: int) -> List[int]:
        """Cover all sources with >=2 groups each when possible (fixed crops)."""
        meta = self.train_set.meta
        by_src: Dict[str, Dict[str, List[int]]] = {}
        for m in meta:
            by_src.setdefault(m["source"], {}).setdefault(m["group_id"], []).append(m["index"])
        chosen: List[int] = []
        # at least 2 groups per source, up to 4 pages each
        for src in sorted(by_src.keys()):
            groups = sorted(by_src[src].keys())[: max(2, min(4, len(by_src[src])))]
            for g in groups:
                pages = by_src[src][g][:2]
                chosen.extend(pages)
        # fill remaining
        if len(chosen) < n_target:
            rest = [m["index"] for m in meta if m["index"] not in set(chosen)]
            chosen.extend(rest[: n_target - len(chosen)])
        return chosen[:n_target]

    def overfit_small(self) -> Dict[str, Any]:
        """Multi-source fixed-subset overfit; eval uses full subset + posterior mean."""
        n = min(self.cfg.training.overfit_n_patches, len(self.train_set))
        idxs = self._select_overfit_indices(n)
        if hasattr(self.train_set, "fixed_crops"):
            prev = self.train_set.fixed_crops
            self.train_set.fixed_crops = True
        else:
            prev = None
        subset = torch.utils.data.Subset(self.train_set, idxs)
        loader = DataLoader(
            subset,
            batch_size=min(self.cfg.training.microbatch_size, max(len(idxs), 1)),
            shuffle=True,
            collate_fn=collate_hq,
        )
        eval_loader = DataLoader(
            subset,
            batch_size=min(self.cfg.training.microbatch_size, max(len(idxs), 1)),
            shuffle=False,
            collate_fn=collate_hq,
        )

        def _eval_mean() -> float:
            self.system.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for batch in eval_loader:
                    batch.hq = batch.hq.to(self.device)
                    # posterior mean recon vs target in normalized domain
                    recon = self.system.reconstruct_hq(batch.hq)
                    total += float((recon - batch.hq).abs().mean().cpu()) * batch.hq.shape[0]
                    count += batch.hq.shape[0]
            self._enter_train_mode()
            return total / max(count, 1)

        initial = _eval_mean()
        self._enter_train_mode()
        it = iter(loader)
        losses: List[float] = []
        max_steps = min(self.cfg.training.max_steps, 500)
        # force sample_posterior False during overfit train for stability? keep True but fixed seed noise
        self.optimizer.zero_grad(set_to_none=True)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.cfg.experiment.seed)
        for step in range(max_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch.hq = batch.hq.to(self.device)
            self.state.optimizer_step = step
            loss_out = self.system.task.forward_loss(batch, optimizer_step=step)
            loss_out = self._attach_adv_g(loss_out, batch)
            if not torch.isfinite(loss_out.total.detach()):
                raise RuntimeError(f"Non-finite overfit loss at step {step}")
            if self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
                self._backward_disc(loss_out, batch, 1)
            if self.discriminator is not None:
                self.discriminator.requires_grad_(False)
            loss_out.total.backward()
            if self.disc_optimizer is not None and self._adv_weight(step) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.cfg.loss.adversarial.grad_clip_norm
                    if self.cfg.loss.adversarial.grad_clip_norm > 0
                    else 1e9,
                )
                self.disc_optimizer.step()
                self.disc_optimizer.zero_grad(set_to_none=True)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss_out.total.detach().cpu()))
            self.state.optimizer_step = step + 1
        final = _eval_mean()
        if prev is not None:
            self.train_set.fixed_crops = prev
        drop = (initial - final) / max(abs(initial), 1e-8)
        thr = float(self.cfg.gates.overfit_loss_drop_frac)
        # NO silent 0.5: require configured relative drop on fixed-subset MAE
        passed = drop >= thr
        # coverage report
        src_counts: Dict[str, int] = {}
        for i in idxs:
            s = self.train_set.meta[i]["source"]
            src_counts[s] = src_counts.get(s, 0) + 1
        return {
            "initial_loss": float(initial),
            "final_loss": float(final),
            "drop_frac": float(drop),
            "threshold": thr,
            "passed": bool(passed),
            "steps": len(losses),
            "n_patches": len(idxs),
            "sources_in_subset": src_counts,
            "train_batch_loss_curve_head": losses[:5],
            "train_batch_loss_curve_tail": losses[-5:],
            "metric": "fixed_subset_mae_posterior_mean",
        }

    def dry_run(self) -> Dict[str, Any]:
        params = self.system.vae.count_parameters()
        audit = self.system.trainability_audit()
        batch = next(iter(self.train_loader))
        x = batch.hq[:1].to(self.device)
        hq = batch.hq.detach().float()
        with torch.no_grad():
            out = self.system.vae(x, sample_posterior=False)
        return {
            "experiment": self.cfg.experiment.name,
            "output_dir": str(self.run_dir),
            "params": params,
            "all_core_trainable": audit["all_core_trainable"],
            "input_shape": list(x.shape),
            "recon_shape": list(out.reconstruction.shape),
            "latent_shape": list(out.latent.shape),
            "spatial_compression": self.system.vae.spatial_compression,
            "config_sha256": self.config_sha,
            "normalizer_sha256": self.normalizer_sha,
            "manifest_sha256": self.manifest_sha,
            "device": str(self.device),
            "ddp": self.dist.enabled,
            "rank": self.dist.rank,
            "world_size": self.dist.world_size,
            "per_device_batch": self.per_device_batch,
            "grad_accum": self.grad_accum,
            "effective_global_batch": self.effective_global_batch,
            "has_test_loader": hasattr(self, "test_loader"),
            "normalizer_contract": self.normalizer.state.contract_dict(),
            "batch_sources": list(batch.sources),
            "batch_hq_min": float(hq.min()),
            "batch_hq_max": float(hq.max()),
            "batch_hq_frac_lt0": float((hq < -1e-6).float().mean()),
            "batch_hq_frac_gt1": float((hq > 1).float().mean()),
            "capabilities": {
                "hq_reconstruction": self.system.capabilities.hq_reconstruction,
                "lr_encoding": self.system.capabilities.lr_encoding,
                "paired_restoration": self.system.capabilities.paired_restoration,
            },
        }
