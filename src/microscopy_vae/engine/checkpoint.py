from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from microscopy_vae.engine.state import TrainerState
from microscopy_vae.provenance.hashing import sha256_file, sha256_json
from microscopy_vae.utils.atomic import atomic_write_bytes


CHECKPOINT_FORMAT = "microvae-ckpt-v1"


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    """Load checkpoint with weights_only when supported (PyTorch >= 2.0 safer default)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class CheckpointManager:
    """Separates resume_exact from fresh_init / stage_transition / export load."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_exact(
        self,
        *,
        tag: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: Optional[Any],
        state: TrainerState,
        config_sha256: str,
        normalizer_sha256: str,
        code_version: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "mode": "resume_exact",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "trainer_state": state.to_dict(),
            "config_sha256": config_sha256,
            "normalizer_sha256": normalizer_sha256,
            "code_version": code_version,
            "extra": extra or {},
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        path = self.ckpt_dir / f"{tag}.pt"
        # torch.save to bytes then atomic replace
        import io

        buf = io.BytesIO()
        torch.save(payload, buf)
        atomic_write_bytes(path, buf.getvalue())
        # write sidecar hash
        h = sha256_file(path)
        (self.ckpt_dir / f"{tag}.sha256").write_text(h + "\n", encoding="utf-8")
        return path

    def prune_periodic(self, *, keep_last: int, prefix: str = "step_") -> None:
        """Keep only the newest keep_last periodic checkpoints (not candidate_/final)."""
        if keep_last is None or keep_last < 0:
            return
        files = sorted(
            [
                p
                for p in self.ckpt_dir.glob(f"{prefix}*.pt")
                if "candidate" not in p.name
                and "final" not in p.name
                and not p.name.startswith("best_")
            ],
            key=lambda p: p.stat().st_mtime,
        )
        for p in files[:-keep_last] if keep_last > 0 else files:
            p.unlink(missing_ok=True)
            side = p.with_suffix(".sha256")
            if side.exists():
                side.unlink(missing_ok=True)

    @staticmethod
    def resume_exact(
        path: Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: Optional[Any],
        expected_config_sha256: str,
        expected_normalizer_sha256: str,
        map_location: str = "cpu",
        verify_sidecar_hash: bool = True,
        expected_code_version: Optional[str] = None,
    ) -> tuple:
        """Returns (TrainerState, payload_extra dict)."""
        if verify_sidecar_hash:
            side = Path(str(path) + ".sha256") if not str(path).endswith(".pt") else path.with_suffix(".sha256")
            # also try path.sha256 next to file
            if not side.is_file():
                side = Path(str(path) + ".sha256")
            if side.is_file():
                expected = side.read_text(encoding="utf-8").strip().split()[0]
                actual = sha256_file(path)
                if expected != actual:
                    raise ValueError(f"checkpoint sidecar hash mismatch: {path}")
        payload = _torch_load(path, map_location=map_location)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Unknown checkpoint format: {payload.get('format')}")
        if payload.get("mode") != "resume_exact":
            raise ValueError("This loader only accepts resume_exact checkpoints")
        if payload.get("config_sha256") != expected_config_sha256:
            raise ValueError("config_sha256 mismatch — refuse resume_exact")
        if payload.get("normalizer_sha256") != expected_normalizer_sha256:
            raise ValueError("normalizer_sha256 mismatch — refuse resume_exact")
        if expected_code_version is not None and payload.get("code_version") != expected_code_version:
            raise ValueError(
                f"code_version mismatch: ckpt={payload.get('code_version')} current={expected_code_version}"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if payload.get("rng", {}).get("torch") is not None:
            torch.set_rng_state(payload["rng"]["torch"])
        return TrainerState.from_dict(payload["trainer_state"]), payload.get("extra") or {}

    @staticmethod
    def load_exported_weights(path: Path, model: torch.nn.Module, map_location: str = "cpu") -> None:
        payload = _torch_load(path, map_location=map_location)
        if isinstance(payload, dict) and "model" in payload:
            # Allow full ckpt for inference but mark not for training resume without checks
            model.load_state_dict(payload["model"])
        elif isinstance(payload, dict):
            model.load_state_dict(payload)
        else:
            raise ValueError("Unrecognized weights file")


class StageTransitionLoader:
    """Load parent scratch lineage with explicit module policy (S1→S3 later)."""

    @staticmethod
    def load(
        path: Path,
        *,
        model: torch.nn.Module,
        load_modules: Optional[list[str]] = None,
        freeze_modules: Optional[list[str]] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        payload = _torch_load(path, map_location=map_location)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("stage_transition requires project checkpoint format")
        state = payload["model"]
        # For S1 core we only support full load of VAE weights as parent.
        model.load_state_dict(state, strict=True)
        if freeze_modules:
            for name, p in model.named_parameters():
                if any(name.startswith(m) for m in freeze_modules):
                    p.requires_grad_(False)
        return {
            "parent_path": str(path),
            "load_modules": load_modules or ["*"],
            "freeze_modules": freeze_modules or [],
            "parent_config_sha256": payload.get("config_sha256"),
        }
