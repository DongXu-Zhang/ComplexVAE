# 正式训练：V8 四卡（f8/z4，只开四项损失）

配方：`configs/experiment/s1_hq_f8z4_v8.yaml`  
代码：`microscopy_vae.__version__ == 0.3.5`

不要覆盖 V6/V5/V4 run，不要复用它们的 `normalizer.json`。不要开 Scharr / HF / Flux / GAN / dark_fp。

## 这一版是什么

S1 HQ 自重建，Input = Target。相对 V6 **只**关掉 raw floor：

```text
y = x / H_s
H_s = 该源 train 的 p99.99
low = 0          # 灭仍是 0
clip = false
```

负值保留成 `x/H_s`（例如 Bio 上 -1000 → -0.043）。不是把 0 点挪到最暗负值。

## 四个损失函数（只有这些进总损失）

| 项 | yaml | 何时非零 |
|---|---|---|
| Charbonnier | `w_char=1.0` | 全程 |
| MS-SSIM | `w_ms_ssim=0.12` | step ≥ 1000，500 step 爬满 |
| Perceptual | `perceptual.weight=0.05`，`internal_conv` 冻结 | step ≥ 1000，1000 step 爬满 |
| KL | `beta_max=0.008` | step ≥ 2000，到 25000 爬满 |

`w_grad=w_hf=w_flux=w_dark_fp=0`，`adversarial.enabled=false`。  
Scharr 只做 **Char 的像素权重 / support 门**，不是第五项损失。

## 网络 / 中间值（实测）

```text
architecture = microvae_f8_z4_enc128-256-512-512_dec96-192-384-384
spatial_compression = 8
crop 256 → latent [B, 4, 32, 32] → recon [B, 1, 256, 256]
VAE params  = 62,009,061（全部 trainable）
  encoder   = 34,161,288
  decoder   = 27,847,681
perc params = 71,792（冻结，不进 G 优化器）
GroupNorm groups=32，通道 96/128/192/256/384/512 都能整除
output_activation = linear（不 clamp）
train: sample_posterior=true
val/infer: posterior mean + EMA
```

## 四卡 batch（不要改乘积）

yaml `microbatch=4` × `accum=2` → **全局 8**。默认拆成「每卡尽量大、accum=1」（GroupNorm 按样本，不是 BN）：

```text
1 GPU:  per_device=4  accum=2  → 8
2 GPU:  per_device=4  accum=1  → 8
4 GPU:  per_device=2  accum=1  → 8
8 GPU:  per_device=1  accum=1  → 8
```

不要开 `ddp_scale_global_batch`。3/5/6/7 卡会因 8 不能整除而报错。

验证：各卡分片、不补齐重复；`evaluation.batch_size` 默认 8（与训练 microbatch 无关）。EMA 在 DDP wrap 之后按广播后的权重重克隆，各卡 val 用同一份 shadow。

## 启动

```bash
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
CFG=configs/experiment/s1_hq_f8z4_v8.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m microscopy_vae.cli train --config $CFG \
  --override data.manifest_path=$MANIFEST \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=$DATA_ROOT \
  --override data.path_require_exists=true \
  --override experiment.output_dir=$RUN \
  --override experiment.seed=0
```

开训 log 必须有：`experiment=s1_hq_f8z4_v8`，`world_size=4` `per_device=2` `accum=1` `effective_global=8`，`affine=y = x/high`，`floor=False`，三个源 `low=0`，`frac_lt0` 可以 >0，support_floor Bio/DI2D < 0.02，没有 disc 更新。各 rank 应有 `val shard rank=... n_local=...`。
