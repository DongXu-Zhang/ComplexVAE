#!/usr/bin/env bash
# Run package checks from the repo root (no hard-coded user paths).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f .venv310/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv310/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH=$PWD/src
echo "=================== PYTEST ==================="
python -m pytest -q 2>&1 | tail -60
echo "RC_pytest=${PIPESTATUS[0]}"
echo "=================== SMOKE ==================="
python -m microscopy_vae.cli smoke-test --config configs/experiment/smoke_synthetic.yaml 2>&1 | tail -25
echo "RC_smoke=$?"
echo "=================== OVERFIT ==================="
python -m microscopy_vae.cli overfit-small --config configs/experiment/overfit_hq.yaml 2>&1 | tail -25
echo "RC_overfit=$?"
