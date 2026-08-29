# ComplexVAE — Scratch Microscopy HQ Codec

Package version **0.3.0**. Single-channel **KL-VAE** for high-quality microscopy images:

```text
HQ image → Encoder → latent z → Decoder → reconstruction
```

- **In scope:** encode / decode / train HQ codec / export `LatentSpec`
- **Out of scope:** latent diffusion, WF→RC super-resolution training
- **Weights:** always `fresh_init` (no pretrained SD/Hybrid-SD weights)

Topology is *inspired by* Hybrid-SD Small / Diffusers 0.27 Encoder–Decoder structure (asymmetric channels, f8, z=4) but implemented independently under `src/microscopy_vae/`.

**Official next train:** `configs/experiment/s1_hq_f8z4_v4.yaml` (V4 protocol, spatial compression 8, latent `4×32×32`). Same protocol at f4: `s1_hq_f4z4_v4.yaml`. Complete control (do not overwrite): `s1_hq_f4z4_v2_2.yaml`. Do not load f4 weights into f8 (`strict=False` is refused). Train f8 from scratch.

V4: raw `max(x,0)`, then **per-source** train-only robust scales at p99.99. Independent Scharr/HF/Flux generator weights are 0; structure-support gate and Charbonnier edge weighting stay on. Infer with a path containing `BioTISR` / `DeepInsight_2D` / `DeepInsight_3D`, or pass `--source`. Frozen test split stays closed. Encode for LDM: `python -m microscopy_vae.cli encode ...` writes posterior mean in the **internal unscaled** domain; refit per-channel center/scale on **this** architecture's train latents (never reuse f4 stats on f8, never apply SD `0.18215`).

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

Confirm you are on **0.3.0**:

```bash
git log -1 --oneline
python -c "import microscopy_vae; print(microscopy_vae.__version__)"
ls configs/experiment/s1_hq_f8z4_v4.yaml
```

## Quick check (no real data)

```bash
export PYTHONPATH=$PWD/src   # if not editable-installed
python -m microscopy_vae.cli smoke-test --config configs/experiment/smoke_f8.yaml
python -m pytest -q tests/unit/test_v4_protocol.py tests/unit/test_f8_protocol.py
```

## Real HQ training (data on host)

1. Place `hq_manifest_v2.jsonl` somewhere readable (not in git).
2. Map Windows inventory paths to the host, e.g. `F:\Dataset\...` → `/data/Dataset/...`
3. Train **f8/V4**. `output_dir` must be a **new empty** directory (do not write into f4 or v2.2 runs):

```bash
python -m microscopy_vae.cli train \
  --config configs/experiment/s1_hq_f8z4_v4.yaml \
  --override data.manifest_path=/data/inventory/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/Dataset \
  --override data.path_require_exists=true \
  --override experiment.output_dir=/data/runs/s1_hq_f8z4_v4_seed0 \
  --override experiment.seed=0
```

Do **not** use the frozen test split. Do **not** load f4 checkpoints into f8.

Logs must show `spatial_compression=8` and three per-source scales with `low=0`. Perc starts at step 1000; GAN at 5000; independent Scharr/HF/Flux stay at 0.0%.

## Inference

Use the same config, EMA checkpoint, and `normalizer.json` from that run. Prefer tiled 256 first (matches training crop size); then `compare` vs full.

```bash
CFG=configs/experiment/s1_hq_f8z4_v4.yaml
CKPT=/data/runs/s1_hq_f8z4_v4_seed0/checkpoints/best_mae.pt
NORM=/data/runs/s1_hq_f8z4_v4_seed0/normalizer.json

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
configs/              # YAML experiments (v2.2 control, V4 f4/f8 train)
tools/                # manifest, loss tables, infer bench
tests/                # unit + integration
```

## License

Apache-2.0. Third-party topology references: see `THIRD_PARTY_NOTICES.md`.
