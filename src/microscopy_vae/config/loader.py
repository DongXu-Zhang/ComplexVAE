from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from microscopy_vae.config.schema import RootConfig
from microscopy_vae.provenance.hashing import sha256_text


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be mapping: {path}")
    return data


def load_config(
    path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RootConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        raw = load_yaml(path)
    if overrides:
        raw = _deep_update(raw, overrides)
    return RootConfig.model_validate(raw)


def resolved_dict(cfg: RootConfig) -> Dict[str, Any]:
    return cfg.model_dump(mode="json")


def canonical_resolved_dict(cfg: RootConfig) -> Dict[str, Any]:
    """Identity of a run for resume_exact.

    `training.resume_exact_path` and `experiment.allow_existing_output` are
    how a later process points at the same run; they must not change the hash
    stored in checkpoints.
    """
    d = resolved_dict(cfg)
    tr = dict(d.get("training") or {})
    tr["resume_exact_path"] = None
    d["training"] = tr
    exp = dict(d.get("experiment") or {})
    exp["allow_existing_output"] = False
    d["experiment"] = exp
    return d


def dump_resolved(cfg: RootConfig, path: Path) -> str:
    """Write the full resolved YAML (including resume path) and return the semantic hash."""
    text = yaml.safe_dump(resolved_dict(cfg), sort_keys=True, default_flow_style=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return config_semantic_hash(cfg)


def config_semantic_hash(cfg: RootConfig) -> str:
    payload = json.dumps(canonical_resolved_dict(cfg), sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)
