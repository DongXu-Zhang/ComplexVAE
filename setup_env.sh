#!/usr/bin/env bash
# Build an isolated venv for microscopy-vae on a COMPUTE NODE (never login).
# Usage (on compute node, e.g. via srun):
#   bash setup_env.sh
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${MICROVAE_VENV:-$PKG_DIR/.venv310}"
BASE_PY="${MICROVAE_BASE_PY:-/apps/anaconda/2024.02-1/bin/python}"   # py3.11
# CPU torch wheel is enough for smoke/overfit/pytest; swap to cu121 for GPU training.
TORCH_SPEC="${MICROVAE_TORCH_SPEC:-torch --index-url https://download.pytorch.org/whl/cpu}"

echo "[setup] base python: $BASE_PY"; "$BASE_PY" --version
echo "[setup] venv: $VENV"
rm -rf "$VENV"
"$BASE_PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

echo "[setup] installing pinned runtime deps"
pip install "numpy>=1.26,<2" "pydantic>=2.5" "PyYAML>=6.0" \
            "mrcfile>=1.5.0" "tifffile>=2024.1.0" "typing_extensions>=4.8" \
            "pytest>=8.0" "pytest-timeout>=2.2"

echo "[setup] installing torch ($TORCH_SPEC)"
pip install $TORCH_SPEC

echo "[setup] editable install of microscopy-vae"
pip install -e "$PKG_DIR" --no-deps

echo "[setup] versions:"
python - <<'PY'
import numpy, pydantic, torch, mrcfile, tifffile, yaml, pytest
print("numpy", numpy.__version__)
print("pydantic", pydantic.VERSION)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mrcfile", mrcfile.__version__)
print("tifffile", tifffile.__version__)
print("pytest", pytest.__version__)
PY
echo "[setup] DONE"
