"""Versioned LatentSpec: internal z vs export z affine transform."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class LatentSpec:
    schema_version: str
    model_architecture_id: str
    model_weights_sha256: str
    resolved_config_sha256: str
    normalizer_artifact_sha256: str
    spatial_compression_factor: int
    latent_channels: int
    posterior_parameterization: str
    deterministic_policy: str
    channel_center: Tuple[float, ...]
    channel_scale: Tuple[float, ...]
    export_transform: str
    statistics_fit_split: str
    statistics_artifact_sha256: str
    latent_dtype: str
    normalization_transform_id: str
    padding_mode: str
    spatial_alignment: str
    tile_size: Optional[Tuple[int, int]]
    tile_overlap: Optional[Tuple[int, int]]
    compatibility_version: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LatentSpec":
        def tup(x):
            return None if x is None else tuple(x)

        return LatentSpec(
            schema_version=d["schema_version"],
            model_architecture_id=d["model_architecture_id"],
            model_weights_sha256=d["model_weights_sha256"],
            resolved_config_sha256=d["resolved_config_sha256"],
            normalizer_artifact_sha256=d["normalizer_artifact_sha256"],
            spatial_compression_factor=int(d["spatial_compression_factor"]),
            latent_channels=int(d["latent_channels"]),
            posterior_parameterization=d["posterior_parameterization"],
            deterministic_policy=d["deterministic_policy"],
            channel_center=tuple(float(v) for v in d["channel_center"]),
            channel_scale=tuple(float(v) for v in d["channel_scale"]),
            export_transform=d["export_transform"],
            statistics_fit_split=d["statistics_fit_split"],
            statistics_artifact_sha256=d["statistics_artifact_sha256"],
            latent_dtype=d["latent_dtype"],
            normalization_transform_id=d["normalization_transform_id"],
            padding_mode=d["padding_mode"],
            spatial_alignment=d["spatial_alignment"],
            tile_size=tup(d.get("tile_size")),
            tile_overlap=tup(d.get("tile_overlap")),
            compatibility_version=d["compatibility_version"],
        )

    def export_from_internal(self, z_internal: torch.Tensor) -> torch.Tensor:
        if self.export_transform != "affine_per_channel":
            raise ValueError(f"Unknown export_transform {self.export_transform}")
        center = torch.tensor(self.channel_center, device=z_internal.device, dtype=z_internal.dtype).view(
            1, -1, 1, 1
        )
        scale = torch.tensor(self.channel_scale, device=z_internal.device, dtype=z_internal.dtype).view(
            1, -1, 1, 1
        )
        return (z_internal - center) / scale.clamp_min(1e-8)

    def internal_from_export(self, z_export: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.channel_center, device=z_export.device, dtype=z_export.dtype).view(1, -1, 1, 1)
        scale = torch.tensor(self.channel_scale, device=z_export.device, dtype=z_export.dtype).view(1, -1, 1, 1)
        return z_export * scale + center


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_latent_center_scale(
    means: Sequence[torch.Tensor],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Fit per-channel center/scale from a list of μ tensors [B,C,H,W] on train only."""
    if not means:
        raise ValueError("No latent means provided")
    cat = torch.cat([m.detach().float().cpu() for m in means], dim=0)
    # center = mean over B,H,W; scale = std over B,H,W
    center = cat.mean(dim=(0, 2, 3))
    scale = cat.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-4)
    return tuple(float(x) for x in center), tuple(float(x) for x in scale)


def build_latent_spec(
    *,
    architecture_id: str,
    weights_sha256: str,
    config_sha256: str,
    normalizer_sha256: str,
    spatial_compression: int,
    latent_channels: int,
    center: Sequence[float],
    scale: Sequence[float],
    stats_sha256: str,
    transform_id: str,
    padding_mode: str = "reflect",
) -> LatentSpec:
    if len(center) != latent_channels or len(scale) != latent_channels:
        raise ValueError("center/scale length must equal latent_channels")
    return LatentSpec(
        schema_version="microvae-latent-spec-v1",
        model_architecture_id=architecture_id,
        model_weights_sha256=weights_sha256,
        resolved_config_sha256=config_sha256,
        normalizer_artifact_sha256=normalizer_sha256,
        spatial_compression_factor=spatial_compression,
        latent_channels=latent_channels,
        posterior_parameterization="diag_gaussian_mu_logvar",
        deterministic_policy="use_mean_for_eval_export",
        channel_center=tuple(float(c) for c in center),
        channel_scale=tuple(float(s) for s in scale),
        export_transform="affine_per_channel",
        statistics_fit_split="train",
        statistics_artifact_sha256=stats_sha256,
        latent_dtype="float32",
        normalization_transform_id=transform_id,
        padding_mode=padding_mode,
        spatial_alignment="pixel_00_maps_to_latent_00",
        tile_size=None,
        tile_overlap=None,
        compatibility_version="1",
    )
