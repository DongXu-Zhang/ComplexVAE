# ComplexVAE S1 v2 训练说明（给服务器复现 / 给其他 AI 直接执行）

**仓库：** https://github.com/DongXu-Zhang/ComplexVAE  
**本文对应代码版本：** `microscopy_vae 0.2.0`（S1 v2）  
**上一轮基线：** f8/z4，100,000 steps，seed 0，已完成，**不要覆盖那个 run 目录**  
**本文目的：** 在服务器上 clone 本仓库，按这里的命令启动 **下一轮正式训练**

面向用户的术语：

- **slice**：一张二维图，或 3D volume 里的一层（代码字段仍可能叫 `page`）
- **volume**：不可拆分的采集单元（代码字段仍可能叫 `group`）

**冻结 test 禁止读取、禁止调参。**  
不要把重建变平滑说成去噪，也不要把锐利纹理说成真实生物结构恢复。

---

## 0. 30 秒结论

上一轮 100k f8/z4 证明模型能学、不崩、优于常数基线，但存在四类真实问题：

1. 竖条纹（decoder nearest 上采样 + f8 网格）  
2. 细丝 / PKMO 内部纹理被抹平（空间压缩过强 + 损失偏向平均）  
3. DeepInsight-3D 大量离焦 / 空 slice 与第一张 slice 被同等训练和展示  
4. 三个来源归一化后动态范围差一个数量级，用 `PSNR(data_range=1)` 跨源比较会严重误导

v2 **不是**在旧设置上再堆 10 万步，而是一套针对上述问题的新训练配方。主配置：

```text
configs/experiment/s1_hq_f4z4_v2.yaml
```

备用（f4 显存不够时）：

```text
configs/experiment/s1_hq_f8z8_v2.yaml
```

旧配置 `s1_hq_f8z4.yaml` 仍保留，用于复现 100k 基线，**下一轮不要再用它当主训练**。

---

## 1. 上一轮 100k 里已经确认的事实（不要再争论）

| 项目 | 数值 / 设定 |
|---|---|
| 任务 | HQ 自重建，Input = Target，不是去噪、不是超分、不是扩散 |
| 数据 | 44,458 slices / 4,779 volumes；train/val/test 按 volume 隔离 |
| 最终 val volume-macro PSNR | 28.4638 dB；MAE 0.04249 |
| BioTISR / DI2D / DI3D PSNR | 39.80 / 30.94 / 21.99 dB |
| 80k–100k | 已平台，std 约 0.009 dB |
| 结构 | 1×256×256 → 4×32×32 → 1×256×256，约 62M 参数 |
| 上采样 | nearest×2 + 3×3 conv |
| 采样 | source → volume → slice 均匀 → 随机 256 crop |
| 案例图 | 每个 source 按 volume 分位取 20/50/80%，再取该 volume **第一张 slice** |
| Lifeact_nonlinear 案例 | 报告 45.91 dB，细丝几乎消失，竖条纹明显，显示范围仅 [0.112, 0.149] |
| D3 PKMO 案例 | 19.43 dB，显示范围 [-1.97, 5.40]，很像离焦/过曝糊斑 |
| 归一化 | source-balanced 中位数，全局 low/high 落到 DeepInsight-2D；D3 约 30% 像素 < 0 |
| test | 未读 |

因此：继续用同一套 f8/z4 + 均匀 slice + nearest + range=1 PSNR 再训 10 万步，**预期收益很小**。

---

## 2. v2 相对 v1 改了什么、为什么

### 2.1 网络

| 项 | v1 | v2 主配置 | 针对 |
|---|---|---|---|
| 空间压缩 | f8（三次 stride-2） | **f4**（两次 stride-2） | 1–3 px 细丝低于 8×8 latent 格子 |
| latent | 4×32×32 = 4096 个数 | **4×64×64 = 16384** | 元素比 16:1 → 4:1 |
| 上采样 | nearest×2 | **bilinear×2 + conv** | 竖向 2ⁿ 栅格 |
| 下采样 pad | 右侧/下侧 +1（Diffusers 非对称） | **reflect 对称 pad** | 方向性相位/条纹 |
| 下采样前 | 无 | **3×3 二项式预模糊** | 混叠变成条纹 |
| 长跳跃 | 无 | **仍无** | 必须经过独立 latent，不能改成 U-Net 恢复器 |
| GAN / LPIPS | 无 | **仍无** | 显微图上会造假丝 |

备用 `s1_hq_f8z8_v2.yaml`：空间仍 f8，通道 4→8。只在 f4 OOM 时用。

### 2.2 数据怎么进网络

| 项 | v1 | v2 |
|---|---|---|
| volume 内 slice | 均匀随机 | **聚焦 softmax 加权**；离焦层仍保留约 15% 概率，不是删光 |
| 聚焦分数 | 无 | 同一 volume 内 z-score：0.5 Tenengrad + 0.3 高频 + 0.2 稳健对比 |
| 256 crop | 纯随机 | **未覆盖粗格子优先 + 格子内 jitter** |
| 验证 crop | 固定中心 | 仍固定中心（为了和 100k 协议可比） |
| 验证 slices | 全部 val slices | 仍全部报告；聚焦 sidecar 另可用于事后分层 |
| test | 拒绝 | **仍然拒绝** |

聚焦加权的公式：

```text
p = (1 - 0.15) * softmax(score / 0.7) + 0.15 / n_slices_in_volume
```

完全丢掉低分 slice 会让模型以后不会重建真实离焦层，所以必须留最低概率。

### 2.3 损失

训练仍在**同一张全局归一化图**上进行（不改 100k 那套 train-only p0.1–p99.9 映射，以免 latent 尺度和旧实验完全不可比）。改变的是**损失怎么量误差**：

| 项 | v1 | v2 |
|---|---|---|
| Charbonnier | 绝对归一化域 | 先按该 crop 的 p99.5–p0.5 缩放，再算 |
| MS-SSIM | data_range=1 | 在幅度归一化空间里算，range=1 |
| Scharr | 绝对域 | 幅度归一化域 |
| 边缘加权 | 无 | `1 + 0.75 * (|∇target| / mean\|∇target\|)` |
| 高频项 | 无 | 3×3 高通残差 Charbonnier，权重 0.05 |
| Flux / KL | 绝对域 / free-bits | Flux 仍在绝对域（管亮度）；KL β_max 0.01→**0.008**，升满 20k→**25k** |

这样做的直接原因：

- D3 归一化后 std 远大于 Bio，绝对 Charbonnier 会被 D3 主导  
- Lifeact 对比度只有约 0.03 时，range=1 的 SSIM 几乎不罚结构丢失  
- 边缘加权 / 高频项把梯度推到 Target 里已经存在的细丝上，**不会像 GAN 那样发明结构**

### 2.4 优化与训练长度

| 项 | v1 100k | v2 |
|---|---|---|
| 步数 | 100,000（后期平台） | **150,000** |
| 有效 batch | 8 | 仍为 8（microbatch 4 × accum 2） |
| 学习率 | 1e-4 | **8e-5**（减轻长期 grad clip 主导） |
| warmup | 500 | 1000 |
| AMP | BF16 | BF16 |
| gradient checkpointing | 100k 实际配置里是关的 | **开**（f4 激活图更大） |
| EMA | 0.999 | 0.999 |
| 候选步 | 20/40/60/80/100k | **30/60/90/120/150k** |
| 最优点 | keep-last 会删掉 | **`best_snr.pt` / `best_mae.pt` 永久保留** |
| 源码哈希 | 训练时没有 | run 目录写 `source_snapshot.json` |

为什么要更长而不是更短：覆盖感知 crop + 聚焦采样改变了数据效率，f4 也是新结构，必须重新学，不是在旧 100k 上续训。150k × batch 8 ≈ 120 万次 crop。D3 粗格子大约 44 万个，加上另两个来源，这个量级才够把覆盖做上去。

### 2.5 评价（训练过程就会记）

v2 打开 `evaluation.extended_metrics: true`，验证除了旧的 PSNR/MAE，还会记：

- `mse`、`nmse`、`snr_db`（`10 log10(var(target)/MSE)`）  
- `ssim_range1`  
- `psnr_mse_pooled`（先对 volume 内 slices 平均 MSE，再算 PSNR）  
- 分来源宏平均  

**跨源比较请优先看 SNR / NMSE，不要只看 28 dB 那种 range=1 PSNR。**  
旧 PSNR 仍会写日志，方便和 100k 曲线对照，但不要把它当唯一选模标准。

选模：

1. 主看 `best_snr.pt`（有 SNR 用 SNR，否则退回 PSNR）  
2. 同时保留 `best_mae.pt`  
3. 150k `*_final.pt` 只表示跑完，不一定最好  

---

## 3. 仓库里哪些文件是新的

核心新文件：

```text
configs/experiment/s1_hq_f4z4_v2.yaml
configs/experiment/s1_hq_f4z4_v2_real_manifest.yaml
configs/experiment/s1_hq_f8z8_v2.yaml
src/microscopy_vae/models/blocks.py          # bilinear / 对称下采样 / 预模糊
src/microscopy_vae/data/hq_dataset.py        # coverage crop
src/microscopy_vae/data/samplers.py          # focus softmax
src/microscopy_vae/data/focus_index.py       # sidecar
src/microscopy_vae/losses/pixel.py           # 幅度缩放、边缘权、高通
src/microscopy_vae/losses/composer.py
src/microscopy_vae/engine/evaluator.py       # 扩展指标
src/microscopy_vae/engine/trainer.py         # best ckpt、聚焦接入、源码哈希
src/microscopy_vae/metrics/extended.py
src/microscopy_vae/metrics/stripe.py
src/microscopy_vae/metrics/focus.py
docs/S1_V2_TRAINING_GUIDE_CN.md            # 本文
```

旧的 `s1_hq_f8z4.yaml` 默认行为保持 v1（nearest、均匀 slice、随机 crop、无 amp_norm），用来复现 100k。

---

## 4. 在服务器上从零复现

下面路径按你们上一轮实际位置写。若不同，只改 `--override`，不要改代码。

### 4.1 建议路径

```text
代码:     /data/zhangdongxu/VAE/ComplexVAE
清单:     /data/zhangdongxu/VAE/manifests/hq_manifest_v2.jsonl
数据根:   /data/zhangdongxu/VAE/raw/Dataset
         （对应清单里的 F:\Dataset\...）
聚焦旁路: /data/zhangdongxu/VAE/manifests/focus_sidecar_v1.jsonl
输出:     /data/zhangdongxu/VAE/outputs/ComplexVAE/s1_hq_f4z4_v2_seed0
```

**输出目录必须是新的。** 不要写进  
`/data/zhangdongxu/VAE/outputs/ComplexVAE/s1_hq_f8z4_seed0_auditv2_fast80g`

### 4.2 clone 与环境

```bash
cd /data/zhangdongxu/VAE
git clone git@github.com:DongXu-Zhang/ComplexVAE.git
# 或：git clone https://github.com/DongXu-Zhang/ComplexVAE.git
cd ComplexVAE

# 确认是 v2
git log -1 --oneline
grep -n "0.2.0" src/microscopy_vae/__init__.py
ls configs/experiment/s1_hq_f4z4_v2.yaml

# 环境：沿用上一轮可用的 conda/venv 即可，不必重装
# 若新环境：
#   conda env create -f environment.yml
#   conda activate complexvae
#   pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
#   pip install -e ".[dev]"

export PYTHONPATH=$PWD/src
```

### 4.3 先跑测试（可选但推荐）

```bash
cd /data/zhangdongxu/VAE/ComplexVAE
export PYTHONPATH=$PWD/src
python -m pytest -q tests/unit
python -m microscopy_vae.cli smoke-test --config configs/experiment/smoke_synthetic.yaml
```

单元测试应全部通过。合成 smoke 只证明代码能跑，不是业务结论。

### 4.4 确认清单和数据

```bash
# 清单哈希必须仍是
# 7285a66d9b89b3410b70327d15e656bbb70df926c13fccf82061b8ee3ec50734
sha256sum /data/zhangdongxu/VAE/manifests/hq_manifest_v2.jsonl

# 抽查路径映射是否存在真实文件
python - <<'PY'
from pathlib import Path
p = Path("/data/zhangdongxu/VAE/manifests/hq_manifest_v2.jsonl")
line = p.read_text(encoding="utf-8").splitlines()[0]
print(line[:200])
PY
```

清单里是 `F:\Dataset\...`。训练时用

```text
data.path_prefix_source = F:\Dataset
data.path_prefix_target = /data/zhangdongxu/VAE/raw/Dataset
```

**不要改写 jsonl 本体。**

### 4.5 生成聚焦 sidecar（只读 train/val）

这一步会读 3D volume 的所有 train/val slices 并打分，**首次可能要十几分钟到一小时**，但只做一次。

```bash
cd /data/zhangdongxu/VAE/ComplexVAE
export PYTHONPATH=$PWD/src

python -m microscopy_vae.cli build-focus-sidecar \
  --config configs/experiment/s1_hq_f4z4_v2.yaml \
  --override data.manifest_path=/data/zhangdongxu/VAE/manifests/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/zhangdongxu/VAE/raw/Dataset \
  --override data.path_require_exists=true \
  --out /data/zhangdongxu/VAE/manifests/focus_sidecar_v1.jsonl
```

检查：

- 输出文件非空  
- 里面 **没有** `"split": "test"`  
- DeepInsight-3D 的 volume 应有约 20 张 slice 的排名  

如果跳过这一步，配置里 `focus_compute_if_missing: true` 会在 **train 启动时**自动算并写到 run 目录，第一次启动会更慢。

### 4.6 启动正式训练

前台示例（先确认能进第一步再丢到 sbatch/tmux）：

```bash
cd /data/zhangdongxu/VAE/ComplexVAE
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0   # 按实际 GPU 改

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

用 tmux / sbatch 包一层即可。不要改 `evaluation.allow_test`。

### 4.7 显存不够时

按顺序试，一次只改一项：

1. `--override training.microbatch_size=2 --override training.grad_accum=4`（有效 batch 仍是 8）  
2. 整份换成 `configs/experiment/s1_hq_f8z8_v2.yaml`  
3. 仍不够再关 attention：`--override model.mid_block_add_attention=false`（这会改变结构，单独记一笔）

### 4.8 预计时间

上一轮 100k ≈ 24 小时（A100 80GB）。  
v2：150k + f4 更大激活 + 扩展验证，粗估 **36–60 小时**。  
中途看 `metrics_val.jsonl` 即可，不必等跑完才判断有没有学到东西。

---

## 5. 训练中看什么

run 目录里：

```text
resolved_config.yaml
normalizer.json
environment.json
source_snapshot.json
focus_sidecar_train.jsonl     # 仅当启动时现场计算
metrics_train.jsonl
metrics_val.jsonl
checkpoints/step_XXXXXXX.pt
checkpoints/best_snr.pt
checkpoints/best_mae.pt
checkpoints/candidate_step_*.pt
checkpoints/step_0150000_final.pt
```

日志里应出现类似：

```text
val step=1000 weights=ema group_macro_psnr=... mae=... snr=... pooled=...
```

健康信号（和 100k 同类）：

- loss 有限，无 NaN  
- `active_unit_frac` 不要掉到接近 0  
- 三个来源的 sampler 频率应接近 0.18 / 0.46 / 0.37  
- D3 的 **SNR/NMSE** 若相对 Bio 不再像 18 dB PSNR 那样夸张，是尺子修正，不一定是模型突然变强  

选模不要只盯 150k final。先看 `best_snr.pt`。

---

## 6. 什么叫这轮成功（验证集，不是 test）

必须同时看数字和固定案例，不能只看“更干净”。

建议在 60k / 90k / 150k 用 EMA 各出一组案例，规则：

- volume 仍按每个 source 的确定性 20/50/80% 分位  
- volume 内改用 **中央 50% z 里 focus_score 最高的 slice**，不要第一张  
- 至少包括：BioTISR Lifeact_nonlinear、BioTISR PKMO、DeepInsight-3D PKMO、DeepInsight-2D ER  

成功倾向：

1. Lifeact_nonlinear 不再是“45 dB 但没有丝”；丝还在，竖条纹明显减弱  
2. PKMO 内部纹理比 100k 更接近 Target，不是只有一团轮廓  
3. D3 聚焦层案例不再是大块糊斑；全部 slices 的 SNR/NMSE 不崩  
4. BioTISR / DI2D 的 MAE 没有无解释地变差很多  
5. latent 仍然活跃  

失败则停、不要盲目加 GAN：

- 出现 NaN / 持续爆炸  
- 三个来源之一 SNR 崩溃  
- 竖条纹比 100k 更强  
- 只是更平滑、细丝更少  

---

## 7. 明确不要做的事

1. 不要打开 test，不要按 test 选 checkpoint  
2. 不要往 100k 旧 run 目录里 resume 这套 v2（结构都变了，resume 会被拒或无意义）  
3. 不要把 v2 和 v1 的 range=1 PSNR 差 0.02 dB 解释成本质差异  
4. 不要加 ImageNet LPIPS / GAN 当第一补丁  
5. 不要给 encoder–decoder 加长跳跃还声称这是同等 latent codec  
6. 不要改写权威 `hq_manifest_v2.jsonl`  
7. 不要把视觉平滑写成“完成去噪”

---

## 8. 给其他 AI 的最短执行清单

你是在一台已有数据的 Linux GPU 服务器上工作。

1. `git clone https://github.com/DongXu-Zhang/ComplexVAE.git`（或 SSH）  
2. 确认 `src/microscopy_vae/__init__.py` 里版本是 `0.2.0`，且存在 `configs/experiment/s1_hq_f4z4_v2.yaml`  
3. `export PYTHONPATH=$PWD/src`  
4. 用上一轮同样的 manifest 和 `path_prefix_target`  
5. 先 `build-focus-sidecar`（refuse test）  
6. 用第 4.6 节命令启动 `s1_hq_f4z4_v2`，输出目录必须是新的  
7. 若 OOM，按 4.7  
8. 训练中读 `metrics_val.jsonl` 的 `snr_db` / `mae` / 分来源，不要只报 PSNR  
9. 不要读 test，不要改清单，不要加 GAN  

数据与 GPU 路径若与上文不一致，只改 override，不要改默认 yaml 里的科学设定（f4、bilinear、amp_norm、focus_softmax、coverage_jitter、150k）。

---

## 9. 和 100k 基线如何并列报告

以后写结果时请三行一起写：

```text
v1  f8/z4 nearest  100k  seed0   protocol-A PSNR + MAE   （已完成）
v2  f4/z4 bilinear 150k  seed0   protocol-A + SNR/NMSE/SSIM
差值 同时报 MAE、SNR、分来源、聚焦层案例，而不是只报一个 PSNR
```

v2 改了结构、采样和损失，**不是**对 v1 的单变量消融。若以后要归因，再从 v2 往回拆（只改上采样 / 只改 f4 / 只改聚焦）。

---

*文档结束。代码以 GitHub `DongXu-Zhang/ComplexVAE` 当前 main 为准。*
