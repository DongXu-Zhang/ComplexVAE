# ComplexVAE — Scratch Microscopy HQ Codec

Package version **0.3.2**. Single-channel **KL-VAE** for high-quality microscopy images:

```text
HQ image → Encoder → latent z → Decoder → reconstruction
```

- **In scope:** encode / decode / train HQ codec / export `LatentSpec`
- **Out of scope:** latent diffusion, WF→RC super-resolution training
- **Weights:** always `fresh_init` (no pretrained SD/Hybrid-SD weights)

Topology is *inspired by* Hybrid-SD Small / Diffusers 0.27 Encoder–Decoder structure (asymmetric channels, f8, z=4) but implemented independently under `src/microscopy_vae/`.

**Official next train (V5):** `configs/experiment/s1_hq_f8z4_v5.yaml` (per-source p99.99, `clip=false`, f8/z4, train-fitted crop/support/amp gates, `empty_keep_prob=0.55`).  
**Frozen control (V4):** `configs/experiment/s1_hq_f8z4_v4.yaml` — do not overwrite its yaml or run dir. Do not load a V4 `normalizer.json` into V5. Do not load f4 weights into f8. Train V5 from scratch.

V5 keeps V4's linear map (`y = max(x,0)/H_source`) and adds per-source threshold calibration. Independent Scharr/HF/Flux generator weights stay 0; structure-support gate and Charbonnier edge weighting stay on. Infer with a path containing `BioTISR` / `DeepInsight_2D` / `DeepInsight_3D`, or pass `--source`. Frozen test split stays closed. Encode for LDM: `python -m microscopy_vae.cli encode ...` writes posterior mean in the **internal unscaled** domain; refit per-channel center/scale on **this** architecture's train latents (never reuse f4 stats on f8, never apply SD `0.18215`).

## What 0.3.2 adds over 0.3.0 (V4)

1. Official train is **V5** (`s1_hq_f8z4_v5.yaml`): same per-source map, plus train-fitted crop/support/amp gates and `empty_keep_prob=0.55`.
2. Single-node DDP: `python -m` stays 1 GPU; `torchrun --standalone --nproc_per_node=2` splits the yaml global batch of 8 (2×2 per GPU). Do not set `ddp_scale_global_batch`.
3. Infer `--inference-mode halo` uses real image context around each tile (CLI-only).
4. Do not load a V4 `normalizer.json` into a V5 config.

## What 0.3.0 adds over GitHub `78b59c9` (0.2.5)

1. V4 normalizer: floor negatives to 0; per-source p99.99 (`y = max(x,0)/high_source`); `clip=false`.
2. V4 losses: independent Scharr/HF/Flux weights 0; perc + GAN still on (start 1000 / 5000).
3. Configurable f8: extra real stride-2 stage, architecture `microvae_f8_z4_enc128-256-512-512_dec96-192-384-384`.
4. CLI `encode` / `decode` for LDM (unscaled posterior mean + pad metadata).
5. Eval/compare require `--normalizer` and do not write into the training run dir.

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

Confirm you are on **0.3.2**:

```bash
git log -1 --oneline
python -c "import microscopy_vae; print(microscopy_vae.__version__)"
ls configs/experiment/s1_hq_f8z4_v5.yaml configs/experiment/s1_hq_f8z4_v4.yaml
```

## Quick check (no real data)

```bash
export PYTHONPATH=$PWD/src   # if not editable-installed
python -m microscopy_vae.cli smoke-test --config configs/experiment/smoke_v5.yaml
python -m pytest -q tests/unit/test_v5_protocol.py tests/unit/test_v4_protocol.py tests/unit/test_f8_protocol.py tests/unit/test_ddp.py
```

## Real HQ training (data on host)

1. Place `hq_manifest_v2.jsonl` somewhere readable (not in git).
2. Map Windows inventory paths to the host, e.g. `F:\Dataset\...` → `/data/Dataset/...`
3. Train **f8/V5**. `output_dir` must be a **new empty** directory (do not write into V4, f4, or v2.2 runs):

```bash
python -m microscopy_vae.cli train \
  --config configs/experiment/s1_hq_f8z4_v5.yaml \
  --override data.manifest_path=/data/inventory/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/Dataset \
  --override data.path_require_exists=true \
  --override experiment.output_dir=/data/runs/s1_hq_f8z4_v5_seed0 \
  --override experiment.seed=0
```

Two GPUs, **same** global batch 8 (do not set `training.ddp_scale_global_batch`):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m microscopy_vae.cli train \
  --config configs/experiment/s1_hq_f8z4_v5.yaml \
  --override data.manifest_path=/data/inventory/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/Dataset \
  --override data.path_require_exists=true \
  --override experiment.output_dir=/data/runs/s1_hq_f8z4_v5_seed0 \
  --override experiment.seed=0
```

Do **not** use the frozen test split. Do **not** load f4 checkpoints into f8. Unset leftover `RANK`/`WORLD_SIZE`/`LOCAL_RANK` before a single-GPU `python -m` launch.

Logs must show `experiment=s1_hq_f8z4_v5`, `effective_global=8`, `calibrate_thresholds=True`, `spatial_compression=8`, and three per-source scales with `low=0`. Perc starts at step 1000; GAN at 5000; independent Scharr/HF/Flux stay at 0.0%.

## Inference

Use the same config, EMA checkpoint, and `normalizer.json` from that run. Prefer tiled 256 first (matches training crop size); then `compare` vs full.

```bash
CFG=configs/experiment/s1_hq_f8z4_v5.yaml
CKPT=/data/runs/s1_hq_f8z4_v5_seed0/checkpoints/best_mae.pt
NORM=/data/runs/s1_hq_f8z4_v5_seed0/normalizer.json

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

# Halo: real image context around each 256 core (use if tiled dark tiles overshoot)
python -m microscopy_vae.cli infer --config $CFG --weights $CKPT --normalizer $NORM \
  --input $IMG --output runs/infer_halo.npy \
  --inference-mode halo --tile-size 256 --overlap 32 --halo 64 \
  --devices cuda:0
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
configs/              # YAML experiments (V5 recommended, V4/v2.2 controls)
tools/                # manifest, loss tables, infer bench
tests/                # unit + integration
```

## License

Apache-2.0. Third-party topology references: see `THIRD_PARTY_NOTICES.md`.
