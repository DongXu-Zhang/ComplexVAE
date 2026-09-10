# Multi-GPU execution (0.3.5)

This is an **execution** change. The scientific recipe is unchanged: architecture, losses, per-source `y=x/H_s` (V8) or `max(x,0)/H_s` (V6), global batch 8, posterior sampling in train / mean+EMA in val, full/tiled/halo meanings.

## What was slow in this tree (code evidence)

1. **Val ran only on rank 0.** Other ranks waited on `broadcast_object`. That is why the NCCL timeout default is 6 hours (focus sidecar / first-fit can still park ranks).
2. **Val DataLoader reused the train microbatch** (1–2 on multi-GPU) and ran that stream on one GPU.
3. **Val encoded twice** per crop: `reconstruct_hq` then `encode_hq`.
4. **Halo infer ignored extra GPUs.** Tiled infer spawned a new process pool **per image** and reloaded weights on every worker.

Training DDP (gradient sync, GAN dummy paths, Gloo object group, sampler shard, `resume_exact` world_size pin) was already in place and is kept.

## What we changed

| Area | Behavior |
|---|---|
| Train | yaml `microbatch × accum` is the **global** batch, split across GPUs. Prefer a larger per-device batch with `accum=1`: 1 GPU **4×2**, 2 GPU **4×1**, 4 GPU **2×1**, 8 GPU **1×1**. 3/5/6/7 GPUs still error unless `ddp_scale_global_batch` (off). |
| EMA | Fresh init clones EMA **after** DDP broadcasts rank0 weights (and after a rank-local Kaiming would have been overwritten). Resume/warmstart that already loaded an EMA shadow is left as-is. |
| Val | Every rank evaluates a **strided, unpadded** shard. Records gather, sort by `dataset_index`, then the **same** group-macro / by-source / bootstrap as before. Rank 0 writes jsonl and best ckpt. |
| Val batch | `evaluation.batch_size` (default 8). Execution-only; **stripped from config hash** so V6 `resume_exact` still matches. |
| Infer tiled | Unchanged math; still shards tiles. |
| Infer halo | Multi-GPU shards **cores**; windows are the same `halo_jobs` as sequential. |
| Infer full | One image still uses one GPU (GN/attention are global). `--input-list` assigns **whole images** to GPUs; never converts full→tiled. |
| Checkpoint | `resume_exact` still requires the same `world_size`. |

## Launch

Train (global batch stays 8):

```bash
# 4 GPU: per_device=2 × accum=1 × 4 = 8
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m microscopy_vae.cli train --config configs/experiment/s1_hq_f8z4_v8.yaml \
  --override data.manifest_path=$MANIFEST \
  --override experiment.output_dir=$RUN

# 2 GPU: per_device=4 × accum=1 × 2 = 8
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m microscopy_vae.cli train --config configs/experiment/s1_hq_f8z4_v8.yaml \
  --override data.manifest_path=$MANIFEST \
  --override experiment.output_dir=$RUN
```

Optional val microbatch (does not change metrics definition):

```bash
--override evaluation.batch_size=8
```

Infer:

```bash
# one native page, halo on all visible GPUs
python -m microscopy_vae.cli infer --config $CFG --weights $CKPT --normalizer $NORM \
  --input $PAGE --output out.npy --inference-mode halo --devices auto --source BioTISR

# many full pages (not tiled)
python -m microscopy_vae.cli infer --config $CFG --weights $CKPT --normalizer $NORM \
  --input-list pages.txt --output-dir outs --inference-mode full --devices auto
```

## Correctness vs speed

- **Results vs original method:** per-page val metrics then group-macro / bootstrap; train global batch 8; infer windows unchanged. Single-process vs 3-way CPU shard of val matches to `1e-6` rel in unit tests. Halo sequential vs job/fuse matches to `1e-6`.
- **Wall time:** this checkout has **0 CUDA devices**. 2-GPU and 4-GPU end-to-end timings are **未实测** here. Expected: val scales with GPU count (minus gather); train already did; halo/tiled pay a spawn+load cost per call, so many small images may not beat 1 GPU.

## Rollback

Use the previous tag/commit. No checkpoint format change except val now always writes from the gathered shard (same jsonl schema). Do not resume_exact a 1-GPU ckpt on 2 GPUs (still refused).
