# ComplexVAE — Scratch Microscopy HQ Codec

Package version **0.2.5**. Single-channel **KL-VAE** for high-quality microscopy images:

```text
HQ image → Encoder → latent z → Decoder → reconstruction
```

- **In scope:** encode / decode / train HQ codec / export `LatentSpec`
- **Out of scope:** latent diffusion, WF→RC super-resolution training
- **Weights:** always `fresh_init` (no pretrained SD/Hybrid-SD weights)

Topology is *inspired by* Hybrid-SD Small / Diffusers 0.27 Encoder–Decoder structure (asymmetric channels, f8, z=4) but implemented independently under `src/microscopy_vae/`.

**Next training (recommended):** `configs/experiment/s1_hq_f4z4_v2_2.yaml`  
v2.1 structure-support gate plus microscopy-adapted perceptual + GAN (both ON). Keep `s1_hq_f4z4_v2_1.yaml` as the no-perc/GAN control; do not overwrite it. Frozen test split stays closed.

## What 0.2.5 adds over GitHub `73d606b` (v2.1 gate only)

1. **Structure-support pixel gate** (already in `73d606b`): stop fitting isolated speckle on dark/grey background without stacking a dark-only loss and without turning off amp/edge/HF on real filaments.
2. **Inference modes:** `--inference-mode full|tiled|compare`. Tiled uses even tile origins (no last-tile snap overlap bug).
3. **Perceptual + GAN:** frozen 1-channel conv perceptual; unconditional PatchGAN hinge. Schema defaults remain OFF; **v2.2 turns both ON**.
4. **Loss quantification:** every generator term logs raw / weight / weighted contribution / share_pct.
5. **Multi-GPU tiled inference:** `--devices auto|cuda:0|cuda:0,cuda:2` (1 / 2 / 3 / 4 GPUs). Full-image inference stays on one GPU.

## Install

```bash
git clone git@github.com:DongXu-Zhang/ComplexVAE.git
cd ComplexVAE
# HTTPS: git clone https://github.com/DongXu-Zhang/ComplexVAE.git

# Option A: conda
conda env create -f environment.yml
conda activate complexvae
# install torch matching your GPU (example CUDA 12.1):
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Option B: venv + pip (see pyproject.toml)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch  # choose the correct CUDA wheel
```

Confirm you are on **0.2.5**, not the older v2.1-only commit:

```bash
git log -1 --oneline
python -c "import microscopy_vae; print(microscopy_vae.__version__)"
ls configs/experiment/s1_hq_f4z4_v2_2.yaml
```

## Quick check (no real data)

```bash
export PYTHONPATH=$PWD/src   # if not editable-installed
python -m microscopy_vae.cli smoke-test --config configs/experiment/smoke_synthetic.yaml
python -m microscopy_vae.cli smoke-test --config configs/experiment/smoke_gan_perc.yaml
python -m pytest -q
```

## Real HQ training (data on host)

1. Place `hq_manifest_v2.jsonl` somewhere readable (not in git).
2. Map Windows inventory paths to the host, e.g. `F:\Dataset\...` → `/data/Dataset/...`
3. Train **v2.2** (perc + GAN on). `output_dir` must be empty:

```bash
python -m microscopy_vae.cli train \
  --config configs/experiment/s1_hq_f4z4_v2_2.yaml \
  --override data.mode=hq_pool \
  --override data.manifest_path=/data/inventory/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/Dataset \
  --override data.path_require_exists=true \
  --override experiment.output_dir=/data/runs/s1_hq_f4z4_v2_2_seed0 \
  --override experiment.seed=0
```

Do **not** use the frozen test split. Do **not** overwrite a v2.1 run directory.

Perc starts contributing at step 1000; GAN at step 5000. Watch `metrics_train.jsonl` fields `share_pct_*` (generator terms sum to ~100%). Discriminator losses are separate.

## Inference

Use the same config, EMA checkpoint, and `normalizer.json` from that run. Prefer tiled 256 first (matches training crop size); then `compare` vs full.

```bash
CFG=configs/experiment/s1_hq_f4z4_v2_2.yaml
CKPT=/data/runs/s1_hq_f4z4_v2_2_seed0/checkpoints/best_mae.pt
NORM=/data/runs/s1_hq_f4z4_v2_2_seed0/normalizer.json

# One GPU, tiles
python -m microscopy_vae.cli infer --config $CFG --weights $CKPT --normalizer $NORM \
  --input $IMG --output runs/infer_tiled.npy \
  --inference-mode tiled --tile-size 256 --overlap 32 --blend-mode linear \
  --devices cuda:0

# 2 / 3 / 4 GPUs (or auto = all visible devices). Full-image mode still uses one GPU.
python -m microscopy_vae.cli infer --config $CFG --weights $CKPT --normalizer $NORM \
  --input $IMG --output runs/infer_tiled_mgpu.npy \
  --inference-mode tiled --tile-size 256 --overlap 32 --blend-mode linear \
  --devices auto

# Same-image full vs tiled (compare output is a directory)
python -m microscopy_vae.cli infer --config $CFG --weights $CKPT --normalizer $NORM \
  --input $IMG --output runs/infer_compare \
  --inference-mode compare --tile-size 256 --overlap 32 --blend-mode linear \
  --devices auto
```

`--devices` IDs are **logical** (they follow `CUDA_VISIBLE_DEVICES`). One device does not spawn multiprocessing.

Timing 1 vs 2 vs all GPUs on a real checkpoint + large image:

```bash
python tools/bench_infer_devices.py --config $CFG --weights $CKPT --normalizer $NORM --input $IMG
```

Do not assume more GPUs are always faster; measure on the target host.

## Layout

```text
src/microscopy_vae/   # package
configs/              # YAML experiments (v2.1 control, v2.2 train)
tools/                # manifest, loss tables, infer bench
tests/                # unit + integration
```

## License

Apache-2.0. Third-party topology references: see `THIRD_PARTY_NOTICES.md`.
