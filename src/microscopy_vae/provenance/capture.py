from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

from microscopy_vae import __version__


_SAFE_ENV_PREFIXES = ("CUDA", "NVIDIA", "TORCH", "OMP", "MKL", "PYTHON")
_BLOCKLIST = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def capture_environment() -> Dict[str, Any]:
    env_safe = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if any(b in ku for b in _BLOCKLIST):
            continue
        if any(ku.startswith(p) for p in _SAFE_ENV_PREFIXES) or ku in {
            "HOME",
            "USER",
            "LANG",
            "PATH",
        }:
            # PATH truncated
            env_safe[k] = v if k != "PATH" else (v[:200] + "...")
    info: Dict[str, Any] = {
        "microscopy_vae_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "env_safe": env_safe,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)
    return info


def write_environment(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(capture_environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
