from __future__ import annotations

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
from microscopy_vae.data.normalization import NormalizationState, Normalizer, fit_robust_normalizer
from microscopy_vae.data.readers import read_page
from microscopy_vae.data.samplers import HierarchicalIndexSampler
from microscopy_vae.data.synthetic import build_synthetic_hq_pool
from microscopy_vae.engine.checkpoint import CheckpointManager
from microscopy_vae.engine.ema import EMA
from microscopy_vae.engine.evaluator import evaluate_hq_loader
from microscopy_vae.engine.schedulers import build_warmup_cosine_scheduler
from microscopy_vae.engine.state import TrainerState
from microscopy_vae.provenance.capture import write_environment
from microscopy_vae.provenance.source_tree import hash_source_tree
from microscopy_vae.losses.influence import (
    diagnose_generator_influence,
    format_loss_breakdown,
    quantify_generator_losses,
)
from microscopy_vae.losses.schedule import scheduled_weight
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
        self.logger = setup_logging(cfg.logging.level)
        self.run_dir = Path(cfg.experiment.output_dir)
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
        self.config_sha = dump_resolved(cfg, self.run_dir / "resolved_config.yaml")
        write_environment(self.run_dir / "environment.json")
        try:
            pkg_root = Path(__file__).resolve().parents[1]
            snap = hash_source_tree(pkg_root)
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system = build_hq_codec_system(cfg).to(self.device)
        self.state = TrainerState()
        self.manifest_sha = "synthetic"
        self._test_loader_built = False  # invariant

        self._build_data_and_normalizer()
        self._build_optim()
        self._build_disc()
        self.ckpt = CheckpointManager(self.run_dir)
        self._grad_checked = False
        self._contrib_ema: Dict[str, float] = {}
        self.ema: Optional[EMA] = None
        if cfg.training.ema_decay is not None and cfg.training.ema_decay > 0:
            self.ema = EMA(self.system.vae, decay=float(cfg.training.ema_decay))
        self.best_snr = float("-inf")
        self.best_mae = float("inf")
        self._enter_train_mode()

        # Optional resume_exact after optim built
        if cfg.training.resume_exact_path:
            self._resume_exact(Path(cfg.training.resume_exact_path))
        elif cfg.training.warmstart_vae_path:
            self._warmstart_vae(Path(cfg.training.warmstart_vae_path))

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
            # Fit normalizer on train pages only (subsample for memory)
            train_recs = [r for r in records if r.split == "train"]
            reachable = [r for r in train_recs if Path(r.hq_path).is_file()]
            if cfg.data.path_require_exists and len(reachable) < min(8, len(train_recs)):
                raise FileNotFoundError(
                    f"path_require_exists=true but only {len(reachable)}/{len(train_recs)} "
                    f"train files are readable. Set data.path_prefix_target to the Linux mount "
                    f"of F:\\Dataset (see data_fix STATUS)."
                )
            if reachable:
                train_imgs, self._norm_sources = self._sample_train_images_for_norm(reachable)
            elif cfg.normalization.method == "identity" or cfg.normalization.artifact_path:
                # Structure-only dry path: no pixels yet; identity or pre-fit artifact required
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
        if cfg.normalization.artifact_path and Path(cfg.normalization.artifact_path).is_file():
            state = NormalizationState.load(Path(cfg.normalization.artifact_path))
            self.normalizer_sha = state.save(self.run_dir / "normalizer.json")
        elif cfg.training.resume_exact_path and existing_norm.is_file():
            # resume: reuse the run's train-fitted normalizer (do not refit)
            state = NormalizationState.load(existing_norm)
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
            )
            self.normalizer_sha = state.save(self.run_dir / "normalizer.json")
            self.logger.info(
                "normalizer fit_mode=%s low=%.6g high=%.6g pages=%s",
                state.fit_mode,
                state.low,
                state.high,
                state.n_pages_fit,
            )
        self.normalizer = Normalizer(state)
        if self.normalizer.state.fit_split != "train":
            raise RuntimeError("Normalizer fit_split must be train")

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
            else:
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

        train_sampler = HierarchicalIndexSampler(
            self.train_set.meta,
            seed=cfg.experiment.seed,
            source_weight_mode=cfg.sampling.source_weight_mode,
            fixed_source_prior=cfg.sampling.fixed_source_prior,
            epoch_length=max(len(self.train_set), cfg.training.microbatch_size),
            slice_weight_mode=str(getattr(cfg.sampling, "slice_weight_mode", "uniform")),
            slice_scores=slice_scores,
            focus_temperature=float(getattr(cfg.sampling, "focus_temperature", 0.7)),
            focus_min_keep=float(getattr(cfg.sampling, "focus_min_keep", 0.15)),
        )
        self.train_sampler = train_sampler
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=cfg.training.microbatch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_hq,
            drop_last=True,
            pin_memory=pin,
            persistent_workers=cfg.training.num_workers > 0,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=cfg.training.microbatch_size,
            shuffle=False,
            num_workers=min(2, cfg.training.num_workers),
            collate_fn=collate_hq,
            pin_memory=pin,
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
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=cfg.scheduler.warmup_steps,
            max_steps=cfg.training.max_steps,
            min_lr=cfg.scheduler.min_lr,
            base_lr=cfg.optimizer.lr,
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
        self.disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=adv.disc_lr,
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
                base_lr=adv.disc_lr,
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
        }
        if self.system.perceptual is not None:
            extra["perceptual"] = self.system.perceptual.state_dict()
        if self.discriminator is not None:
            extra["discriminator"] = self.discriminator.state_dict()
            extra["disc_optimizer"] = self.disc_optimizer.state_dict() if self.disc_optimizer else None
            extra["disc_scheduler"] = (
                self.disc_scheduler.state_dict() if self.disc_scheduler is not None else None
            )
        extra.update(more)
        return extra

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
            self.discriminator.load_state_dict(extra["discriminator"])
            if self.disc_optimizer is not None and extra.get("disc_optimizer"):
                self.disc_optimizer.load_state_dict(extra["disc_optimizer"])
            if self.disc_scheduler is not None and extra.get("disc_scheduler"):
                self.disc_scheduler.load_state_dict(extra["disc_scheduler"])

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

    def _backward_disc(self, loss_out, batch, accum: int) -> Dict[str, float]:
        """Discriminator update backward (accumulated). recon is detached."""
        logs: Dict[str, float] = {}
        if self.discriminator is None or self._adv_weight(self.state.optimizer_step) <= 0:
            return logs
        from microscopy_vae.losses.adversarial import discriminator_scores

        adv = self.cfg.loss.adversarial
        recon = loss_out.aux["reconstruction"].detach()
        real = loss_out.aux["target"].detach()
        cond = batch.hq.detach() if adv.conditioning == "input" else None
        mask = self._disc_mask(loss_out.aux["support_on"])
        if mask is not None and float(mask.sum()) <= 0:
            return logs
        self.discriminator.requires_grad_(True)
        scores = discriminator_scores(
            self.discriminator,
            real=real,
            fake=recon,
            cond=cond,
            mask=mask,
            gan_loss=adv.gan_loss,
            r1_gamma=float(adv.r1_gamma),
        )
        d_loss = scores["loss_d"] / float(accum)
        if not torch.isfinite(d_loss.detach()):
            raise RuntimeError(f"Non-finite discriminator loss at step {self.state.optimizer_step}")
        d_loss.backward()
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
                self.discriminator,
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
            logs.update(
                diagnose_generator_influence(
                    loss_out.weighted,
                    self.system.vae,
                    param_group_names=tuple(infl_cfg.param_groups),
                    compute_cosine=want_cos,
                )
            )
        return logs

    def _maybe_validate(self) -> Optional[Dict[str, Any]]:
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
        # candidate step bookkeeping (pre-registered list only)
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
        it = iter(self.train_loader)
        accum = self.cfg.training.grad_accum
        self.optimizer.zero_grad(set_to_none=True)
        history: List[float] = []

        # Audit trainability at start of S1
        audit = self.system.trainability_audit()
        if not audit["all_core_trainable"]:
            raise RuntimeError("S1 requires all core parameters trainable (fresh_init / unlocked)")

        while self.state.optimizer_step < max_steps:
            loss_accum = 0.0
            last_diag: Dict[str, Any] = {}
            last_loss_out = None
            last_batch = None
            if self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
            for _micro in range(accum):
                try:
                    batch = next(it)
                except StopIteration:
                    # new epoch: optional re-log sampler exposure
                    it = iter(self.train_loader)
                    batch = next(it)
                batch.hq = batch.hq.to(self.device, non_blocking=True)
                last_micro = _micro == accum - 1
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss_out = self.system.task.forward_loss(
                            batch, optimizer_step=self.state.optimizer_step
                        )
                    # Composer already forces FP32 pixel/structure terms.
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
                    if self.discriminator is not None:
                        self.discriminator.requires_grad_(False)
                    loss = loss_out.total / accum
                    loss.backward()
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
                    if self.discriminator is not None:
                        self.discriminator.requires_grad_(False)
                    loss = loss_out.total / accum
                    loss.backward()
                if not torch.isfinite(loss_out.total.detach()):
                    raise RuntimeError(
                        f"Non-finite loss at step {self.state.optimizer_step} "
                        f"samples={batch.sample_ids} sources={batch.sources}"
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
                self.state.global_samples += batch.hq.shape[0]

            # Standard GAN order: step D first (grads from detached fake), then G.
            disc_pre = 0.0
            if self.disc_optimizer is not None and self._adv_weight(self.state.optimizer_step) > 0:
                has_d_grad = any(
                    p.grad is not None for p in self.discriminator.parameters()
                )
                if has_d_grad:
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
            if not self._grad_checked:
                self._assert_finite_grads()
                self._grad_checked = True
            else:
                # still check finite every step (cheap relative to forward)
                for n, p in self.system.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise RuntimeError(f"Non-finite grad at step {self.state.optimizer_step}: {n}")
            self.optimizer.step()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.system.vae)
            self.optimizer.zero_grad(set_to_none=True)
            self.state.optimizer_step += 1
            history.append(loss_accum / accum)

            if self.state.optimizer_step % self.cfg.training.log_every_steps == 0:
                rec = {
                    "step": self.state.optimizer_step,
                    "loss": history[-1],
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm_pre_clip": pre_clip,
                    "grad_clipped": pre_clip > float(self.cfg.training.grad_clip_norm or 0),
                    **last_diag,
                    "sampler_source_freq": self.train_sampler.realized_source_freq(),
                    "sampler_planned_probs": self.train_sampler.planned_source_probs(),
                }
                if self.disc_optimizer is not None:
                    rec["lr_disc"] = self.disc_optimizer.param_groups[0]["lr"]
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
                # avoid duplicate file when step is also a candidate step
                if self.state.optimizer_step not in set(self.cfg.training.candidate_steps):
                    self.ckpt.save_exact(
                        tag=f"step_{self.state.optimizer_step:07d}",
                        model=self.system.vae,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        state=self.state,
                        config_sha256=self.config_sha,
                        normalizer_sha256=self.normalizer_sha,
                        code_version=__version__,
                        extra=self._ckpt_extra(),
                    )
                    self.ckpt.prune_periodic(keep_last=int(self.cfg.checkpoint.keep_last))

        path = self.ckpt.save_exact(
            tag=f"step_{self.state.optimizer_step:07d}_final",
            model=self.system.vae,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=self.state,
            config_sha256=self.config_sha,
            normalizer_sha256=self.normalizer_sha,
            code_version=__version__,
            extra=self._ckpt_extra(),
        )
        return {
            "final_step": self.state.optimizer_step,
            "final_loss": history[-1] if history else None,
            "checkpoint": str(path),
            "loss_history": history,
            "candidate_hits": dict(self.state.candidate_hits),
            "sampler_source_freq": self.train_sampler.realized_source_freq(),
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
        with torch.no_grad():
            out = self.system.vae(x, sample_posterior=False)
        return {
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
            "has_test_loader": hasattr(self, "test_loader"),
            "capabilities": {
                "hq_reconstruction": self.system.capabilities.hq_reconstruction,
                "lr_encoding": self.system.capabilities.lr_encoding,
                "paired_restoration": self.system.capabilities.paired_restoration,
            },
        }
