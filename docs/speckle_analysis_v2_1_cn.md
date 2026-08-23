# 暗背景白点 / 小方块：按 Fact / Inference / Proposal 把因果一次说清，并改对

**日期：** 2026-08-23  
**代码版本：** `microscopy_vae 0.2.3`  
**正式配方：** `microscopy-vae/configs/experiment/s1_hq_f4z4_v2_1.yaml`  
**必须重训。** 旧 v2 权重是在「把孤立噪声当结构」的损失下学出来的，不能靠微调涂掉。

面向用户的用语：slice = 一张二维图（代码字段 `page`）；volume = 不可拆分样本（代码字段 `group`）。冻结 test 不用。

---

## 1. 客观结论（先读这段）

白点不是「VAE 天生在暗处造点」，也不是输入端逐块 min-max。  
v1（f8/z4，100k）没有这类星空；v2 新加的 **amp_norm + 无上限 edge + 高频**，在 **平坦区的孤立噪点** 上把噪声当成必须拟合的结构；f4 每个 latent 格是 **4×4 像素**，拟合出来看起来就是小方块。

前两版改法都是拆东墙补西墙：

| 做法 | 拆了什么 | 补了什么 | 为什么不行 |
|---|---|---|---|
| 再加「暗区假阳性损失」 | 黑色被罚 | 灰色、深色平坦区同样会出事 | 颜色清单加不完 |
| 整项关掉 amp / edge / HF | 噪声没了 | 三源幅度差、细丝/PKMO 又回到 v1 | 把 v2 真正要解决的事一起扔了 |
| 只按 **整张 256 crop** 开关 | 全空块被放过 | **有细胞、但背景很大** 的图仍走 amp/edge/HF | 你看到的白点，正是这种图 |

正确的改法只有 **一条机制**，按空间支撑判断结构，不按黑/灰/颜色：

> 梯度在邻域里连成细丝或颗粒 → 仍用 v2 的幅度对齐、边缘加权、高频。  
> 孤立尖点（黑底或灰底一样）→ 按 v1 对待：不放大、不加边缘/高频/Scharr。  
> 像素重建项仍覆盖全图，所以模型发明的白块照样被 Charbonnier 拉回去。

这不是新的「管某种颜色」的损失。

---

## 2. 事实核验表

### 2.1 已用代码核对（Fact）

| 项 | 事实 | 依据 |
|---|---|---|
| 任务 | HQ 自重建，Input = Target，不是去噪 | `hq_codec`，`sample_id` 同源 crop |
| 输入归一化 | **一套** 全局 `low/high`，`clip=false` | `Normalizer.transform`；**没有**逐 patch min-max |
| 负值 | 读取与训练 **都不清零** | `read_page` 无 clamp |
| v1 结构 | f8，nearest×2+conv，latent `4×32×32` | 100k `resolved_config.yaml` |
| v1 损失 | Charbonnier + 简化 MS-SSIM + Scharr + Flux + free-bits KL | 无 amp_norm / edge / HF |
| v1 白点 | 100k 案例 **没有** 这类暗区星空 | 报告包 PNG；暗区有轻微偏亮，但是平滑偏置 |
| v2 结构 | f4，bilinear×2+conv，latent `4×64×64` | `s1_hq_f4z4_v2.yaml`；全库无 `ConvTranspose2d` |
| v2 新损失 | `s = max(P99.5−P0.5, 0.05)`；`w = 1 + 0.75·\|∇\|/mean\|∇\|` 无上限；`w_hf=0.05` | `losses/pixel.py` 旧行为、`s1_hq_f4z4_v2.yaml` |
| 空块放大 | 动态范围 < 0.05 时误差最多 ×20 | `1/0.05`；你测过约 8.83% crop 撞地板（Bio 更高） |
| 采样 | `coverage_jitter` 不看方差；`focus_softmax` 只加权 slice | 256 crop 仍可能是黑边 |
| Flux | 有符号全图均值，前景略暗与背景略亮可抵消 | `flux_loss` |
| 显示 | 案例图 Target/Recon **共用** Target 的 p0.5–p99.5 | 低对比图上 0.056 vs 0.07 会被拉成「黑 vs 白」 |
| 老师的 min-max 故事 | **不是当前代码** | 作用点在损失加权，不在输入 |

### 2.2 几何（Fact：格子尺寸；Inference：白块间距）

f4：一个 latent 格 = **4×4 像素**。两次 bilinear×2 把单格脉冲摊成小丘，再经 3×3 conv 略糊，肉眼常见约 4–8 px 的块，不是 1 px 砂。  
**本机没有 v2 的 float32 recon**，所以「峰在 4 px」是几何预测，不是测到的 FFT。这是最大证据缺口。

### 2.3 无法在本机文件哈希级核对

- 这次 v2 run 的 `resolved_config.yaml` / `normalizer.json`（你给过 `low=-2517.45, high=42782.38`，与 `0 → 0.055573` 一致，但无文件）
- 问题图的 float32 `target.npy` / `recon.npy`
- 暗区残差 FFT 主峰是 4 还是 8

没有这些，形状只能说「4 px 格子最像」，不能说已经量到。

---

## 3. 因果链（不要拆成两套补丁）

```text
coverage 抽到「大面积平坦 + 可能有一点细胞」的 256 crop
        ↓
amp_norm 用很小的 s 去除          → 这块的梯度被放大（空块最严重，最多 ×20）
        ↓
无上限 edge：平坦里一个尖点，‖∇‖/mean‖∇‖ 可以到十几
        ↓
Scharr + 高频：尖点在高通里还是尖点，继续当「必须对齐的结构」
        ↓
Input = Target，Target 里的算法噪点也被当结构
        ↓
f4 每个格子 4×4，解出来就是稳定的小白块
```

**灰色也会中招**：amp_norm 看的是动态范围，edge/HF 看的是梯度，都不是颜色。

v1 抽到空块时，绝对误差很小，模型懒得拟合那些点；f8 也装不满孤立点。所以没有白点。

一句话：

> **诱因**是 v2 对损失的用法（平坦区噪声被放大、被加权）。  
> **形状**是 latent 4 px 格 + bilinear 展开。  
> 两者叠在一起，才会又密、又方、又出现在暗/灰背景。

---

## 4. 为什么「只按整张 crop 开关」仍然是拆东墙

典型显微图：**一部分是细胞/细丝，其余是大背景**。

- `P99.5 − P0.5` 由亮结构决定，整张 crop 会被判成「有结构」
- 于是 amp/edge/HF **整张图都开**
- 背景里的孤立噪点，相对「整张图的平均梯度」仍然可以很高
- 实测：无上限 edge 在尖点邻域权重可达 **11.9**；细丝才需要大约 2–3

所以你看到的「暗背景白点」主要不是全空块，而是 **有结构的 crop 里的无结构像素**。  
只做 crop 级 idle，等于修好了少见的全黑块，把真正的病例放过去。

---

## 5. 统一机制（v2.1 真正改什么）

对 **每个像素**（mask 从 Target 算，detach，不反传阈值）：

1. Scharr 梯度幅值 `mag`  
2. `tau = max(0.02, 0.25 × mean(mag))`  
3. `high = mag > tau`  
4. 在 **9×9** 窗里算 `high` 的密度；密度 ≥ **0.15** 才算有空间支撑  

| | 孤立尖点（黑或灰） | 细丝 / 4–8 px 颗粒 |
|---|---|---|
| 高梯度像素在 9×9 里 | ~8/81 ≈ 0.10 | 细丝约 0.27；8×8 颗粒能过 |
| 判定 | 无支撑 | 有支撑 |

然后 **同一条规则** 接到原有 v2 项上（不是新损失）：

| 区域 | amp_norm | edge | HF / Scharr | Charbonnier |
|---|---|---|---|---|
| 有空间支撑 | 开，下限 0.20 | 开，封顶 3 | 只在这些像素上平均 | 全图，边缘加权 |
| 平坦（无支撑） | 不放大（除数=1） | 权重=1 | **不算** | 仍算，权重=1（和 v1 一样） |
| 整张几乎无结构 | 同上 | 同上 | 同上 | 再乘 0.25，降低空块在 batch 里的发言权 |

训练裁剪：稳健幅度太小最多再抽 8 次。这是采样，不是损失。

**明确关掉：** `w_dark_fp = 0`。评价里仍报 `bg_fp_rate`（症状指标，按暗分位统计），训练不用这个 mask。

结构仍是 f4 + bilinear。格子还在；不再被逼着在每个空格子里插一个白点。

本机用 torch 核对（64×64 合成图）：

- 黑底 / 0.06 灰底 / 0.25 灰底上的孤立尖点：支撑 = 0  
- 细丝支撑保留；边缘权重 2.30（有封顶）  
- 旧 edge 对尖点邻域 11.9，门控后 = 1.0  
- 抹掉细丝的损失 0.067，多造一个背景尖点 0.0013（细丝仍重要得多）  
- 8×8 颗粒保留（PKMO 那种点不是 1 px 砂）  
- 空块 `idle_frac=1`，`amp_scale=1`，Scharr/HF = 0  

---

## 6. 根因排序（仍用原 prompt 的表）

| 优先级 | 假设 | 角色 | 可信度 | 反证 |
|---|---|---|---|---|
| **主因 A 形状** | 4 px latent 格展开成小方块 | 为什么是一块一块 | 高（几何）；间距待测 | 浮点 FFT 主峰不在 4（或 8） |
| **主因 B 诱因** | 平坦区噪声被 amp/edge/HF 当成结构 | 为什么在暗/灰背景 | 高 | 门控后暗区假阳性几乎不变 |
| 次因 | Flux 管不住背景正偏差 | 允许整体偏亮 | 中高 | 改 Flux 后偏置仍在 |
| 次因 | 三源共用 low/high，Bio 更容易撞 amp 地板 | 加重 Bio | 中 | 分源后 Bio 并不更差 |
| 次因 | GroupNorm + 线性头让格子更像图章 | 块长得像 | 中，未测 | 不作为这一轮改结构的理由 |
| 否定 | 转置卷积棋盘 | — | 极低 | 代码里没有 |
| 否定 | 输入逐块 min-max / 负值清零 | — | 已否定 | — |

GAN / ImageNet LPIPS：**no-go**（假丝，也会放大格子）。

---

## 7. 改了哪些文件（Proposal，已落地）

| 意图 | 位置 |
|---|---|
| 空间支撑 mask | `losses/pixel.py` `structure_support_mask` |
| 边缘权只加在支撑上，且 clip=3 | `target_grad_weight(..., support=)` |
| Scharr / HF 只在支撑像素平均 | `losses/composer.py` |
| 空块不放大、降权 | `per_sample_robust_scale` + `idle_loss_mult` |
| 少抽空块 | `data/hq_dataset.py` `_select_crop`（已有） |
| 配方 | `configs/experiment/s1_hq_f4z4_v2_1.yaml` |
| 评价症状 | `metrics/extended.py` `background_false_positive_stats`（**不是训练损失**） |
| 单测 | `tests/unit/test_structure_gate.py` |

**没改：** 上采样算子、f、全局 normalizer、禁止逐块 min-max、禁止 GAN、禁止读 test。

---

## 8. 副作用与验收（不能再出现噪声）

可能的代价：

- 真的 1 px 孤立亮点不会被当成结构去「对齐」——显微 HQ 里那种更像 SIM/RC 伪影，v1 本来也不会单独拟合。  
- 4–8 px 颗粒、细丝仍走 edge/HF（单测已过）。  
- 关 amp 的空块不再和 D3 抢梯度；有结构的 D3 仍 amp_norm。跨源主指标继续用 **SNR/NMSE**，不要只看一个 PSNR。

重训后必须看（float，不要只看 PNG）：

1. 暗区不要满天星、不要 4 px 小白块。  
2. 细丝 / PKMO 不要明显糊回 v1。  
3. 日志 `idle_frac`、`support_frac`、`amp_scale_mean`。  
4. 分源 `bg_fp_rate` / `bg_bias`（评价）。  

若暗区干净了、只剩很淡的 4 px 网格，那是压缩分辨率，**不要再加损失去涂**。

---

## 9. 服务器

```bash
git pull
python -m microscopy_vae.cli train \
  --config configs/experiment/s1_hq_f4z4_v2_1.yaml \
  --override data.manifest_path=... \
  --override data.path_prefix_target=... \
  --override sampling.focus_sidecar_path=... \
  --override experiment.output_dir=.../s1_hq_f4z4_v2_1_seed0
```

输出目录必须是新的。不要覆盖 v1 100k 或旧 v2。

---

## 10. 仍缺的证据（不用猜测补）

1. 该次 v2 的 `resolved_config.yaml`（确认确实是 f4）。  
2. 3–5 张最明显白点图的 float32 Target/Recon。  
3. 暗区残差自相关/FFT 主峰是 4 还是 8。  
4. 白块是否与 Target 噪点重合（学会噪声 vs 解码器自己盖章）。  

没有 2、3，形状停留在几何 Inference。  
**诱因**已经能在代码和合成测试上证伪「缺一条黑色损失」和「必须关掉全部 v2 项」这两条错路。
