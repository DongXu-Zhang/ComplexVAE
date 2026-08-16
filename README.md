# ComplexVAE — Scratch Microscopy HQ Codec

Single-channel **KL-VAE** for high-quality microscopy images:

```text
HQ image → Encoder → latent z → Decoder → reconstruction
```

- **In scope:** encode / decode / train HQ codec / export `LatentSpec`
- **Out of scope:** latent diffusion, WF→RC super-resolution training
- **Weights:** always `fresh_init` (no pretrained SD / Hybrid-SD weights)

Topology is *inspired by* Hybrid-SD Small / Diffusers 0.27 Encoder–Decoder structure but implemented independently under `src/microscopy_vae/`.

## Which config should I run?

| Config | When |
|---|---|
| **`configs/experiment/s1_hq_f4z4_v2.yaml`** | **Next official training (recommended)** |
| `configs/experiment/s1_hq_f8z8_v2.yaml` | Only if f4 OOMs |
| `configs/experiment/s1_hq_f8z4.yaml` | Reproduce the finished 100k v1 baseline only |

**Full Chinese reproduction guide (clone + sidecar + train + what to look at):**  
[`docs/S1_V2_TRAINING_GUIDE_CN.md`](docs/S1_V2_TRAINING_GUIDE_CN.md)

v2 changes, in one line: **f4 latent, bilinear upsample, focus-weighted slices, coverage-aware crops, amplitude-aware losses, 150k steps, best-SNR/MAE checkpoints.** No GAN / LPIPS. Frozen **test must stay unread**.

## Install

```bash
git clone git@github.com:DongXu-Zhang/ComplexVAE.git
cd ComplexVAE

# Option A: conda
conda env create -f environment.yml
conda activate complexvae
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Option B: venv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch  # pick the correct CUDA wheel
```

## Quick check (no real data)

```bash
export PYTHONPATH=$PWD/src
python -m microscopy_vae.cli smoke-test --config configs/experiment/smoke_synthetic.yaml
python -m pytest -q tests/unit
```

## Next training on the server (v2)

Paths below match the previous 100k host. Change only `--override`s if your mounts differ. **Use a new output_dir.**

```bash
export PYTHONPATH=$PWD/src

python -m microscopy_vae.cli build-focus-sidecar \
  --config configs/experiment/s1_hq_f4z4_v2.yaml \
  --override data.manifest_path=/data/zhangdongxu/VAE/manifests/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/zhangdongxu/VAE/raw/Dataset \
  --override data.path_require_exists=true \
  --out /data/zhangdongxu/VAE/manifests/focus_sidecar_v1.jsonl

python -m microscopy_vae.cli train \
  --config configs/experiment/s1_hq_f4z4_v2.yaml \
  --override data.manifest_path=/data/zhangdongxu/VAE/manifests/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/zhangdongxu/VAE/raw/Dataset \
  --override data.path_require_exists=true \
  --override sampling.focus_sidecar_path=/data/zhangdongxu/VAE/manifests/focus_sidecar_v1.jsonl \
  --override experiment.output_dir=/data/zhangdongxu/VAE/outputs/ComplexVAE/s1_hq_f4z4_v2_seed0 \
  --override experiment.seed=0
```

If GPU memory is insufficient: `training.microbatch_size=2` and `training.grad_accum=4`, or switch to `s1_hq_f8z8_v2.yaml`.

## Layout

```text
src/microscopy_vae/   # package
configs/              # YAML experiments
tools/                # build/validate HQ manifest
tests/                # unit + integration
docs/                 # training guide
```

## License

Apache-2.0. Third-party topology references: see `THIRD_PARTY_NOTICES.md`.
