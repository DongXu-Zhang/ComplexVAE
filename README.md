# ComplexVAE — Scratch Microscopy HQ Codec

Single-channel **KL-VAE** for high-quality microscopy images:

```text
HQ image → Encoder → latent z → Decoder → reconstruction
```

- **In scope:** encode / decode / train HQ codec / export `LatentSpec`
- **Out of scope:** latent diffusion, WF→RC super-resolution training
- **Weights:** always `fresh_init` (no pretrained SD/Hybrid-SD weights)

Topology is *inspired by* Hybrid-SD Small / Diffusers 0.27 Encoder–Decoder structure (asymmetric channels, f8, z=4) but implemented independently under `src/microscopy_vae/`.

**Next training (recommended):** `configs/experiment/s1_hq_f4z4_v2_1.yaml`  
Same f4 + bilinear as v2. **Pixel-level structure support:** filaments still get amp_norm / clipped edge / HF; isolated spikes on flat background (any intensity) do not. Not a dark-region extra loss, and not a wholesale revert to v1 losses. Frozen test stays closed.

## Install

```bash
git clone git@github.com:DongXu-Zhang/ComplexVAE.git
cd ComplexVAE

# Option A: conda
conda env create -f environment.yml
conda activate complexvae
# install torch matching your GPU (example CUDA 12.1):
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Option B: venv + pip (see pyproject.toml)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch  # choose correct CUDA wheel
```

## Quick check (no real data)

```bash
export PYTHONPATH=$PWD/src   # if not editable-installed
python -m microscopy_vae.cli smoke-test --config configs/experiment/smoke_synthetic.yaml
python -m pytest -q
```

## Real HQ training (data on host)

1. Place `hq_manifest_v2.jsonl` somewhere readable (not in git).
2. Mount or copy HQ files so Windows inventory paths map cleanly, e.g.  
   `F:\Dataset\...` → `/data/Dataset/...`
3. Train:

```bash
python -m microscopy_vae.cli train \
  --config configs/experiment/s1_hq_f8z4.yaml \
  --override data.mode=hq_pool \
  --override data.manifest_path=/data/inventory/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/Dataset \
  --override data.path_require_exists=true \
  --override experiment.output_dir=/data/runs/s1_hq_f8z4 \
  --override experiment.seed=0
```

See host-local deployment notes (not in this repo if kept private): ask the author for `SERVER_SETUP_CN.md`.

## Layout

```text
src/microscopy_vae/   # package
configs/              # YAML experiments
tools/                # build/validate HQ manifest
tests/                # unit + integration
```

## License

Apache-2.0. Third-party topology references: see `THIRD_PARTY_NOTICES.md`.
