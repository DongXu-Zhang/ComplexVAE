# 下一版训练（V6）：关 GAN + 新 support 门控 + 推理用真邻域

**0.3.4 起官方下一训是 V8**（`configs/experiment/s1_hq_f8z4_v8.yaml`，`y = x/high_s`，不开 floor）。V6 是冻结对照，不要覆盖它的 run。

在本机完成。不要复用 V5 111k 的 `normalizer.json`。不要覆盖 `runs/s1_hq_f8z4_v5`。

## 用哪个配置

```bash
# 推荐：已经关掉 GAN，并带 inference.halo
configs/experiment/s1_hq_f8z4_v6.yaml
```

若你只想在旧 nogan yaml 上开训也可以：`s1_hq_f8z4_v5_nogan.yaml`。  
**代码 0.3.3 仍会按 v2 规则重标定 support floor**（不再锁死 0.02）。新 run 必须用空的 `output_dir`。

关 GAN 不需要改 Python：`loss.adversarial.enabled: false`。判别器不参与普通重建推理；必须 **重训** 才会去掉已经学进生成器的珠状模式。

## 不要做的事

- 不要把 V5 的 `normalizer.json` 拷进 V6。v1 门控用“低强度像素”当背景，密丝图里那些像素仍是细丝，BioTISR / DeepInsight_2D 的 floor 被 yaml 上限钉在 0.02。
- 不要 `allow_legacy_artifact: true` 去混 V4/V5 artifact。
- 不要读 test。
- 不要同时再开 Scharr / HF / 改 amp / 改网络。这一版只验证：无 GAN + 暗丝能进 support。

## 三个源的门控（V5 111k 实测 vs V6 规则）

| 源 | V5 111k floor | 问题 | V6 怎么估 floor |
|---|---:|---|---|
| BioTISR | **0.02**（顶到 yaml 帽） | Lifeact 细丝 Scharr 多半 < 0.02，support 只剩亮结点 0.23% | 用 **低梯度** 像素的 p90，并限制 floor < 0.5×结构 Scharr |
| DeepInsight_2D | **0.02**（同样封顶） | PHB2 / TOMM20 同类弱网、弱斑 | 同一套规则，不再按低强度当背景 |
| DeepInsight_3D | 0.0122 | 更空、负值置零多；相对没那么死 | 仍按低梯度估噪声，允许低于 0.02 |

yaml 里 `structure_support_floor: 0.02` 现在只是 **没有 crop 时的退回值，以及标定的上限**，不是“每个源都用 0.02”。  
`structure_support_floor_min: 0.0005` 防止拟合到 1e-5 把相机噪声当结构。

训起来后看 log：

```text
threshold BioTISR support_floor=... crop_range=... amp_range=0.08
```

BioTISR / DeepInsight_2D 的 `support_floor` 应明显 **小于 0.02**。若仍是 0.020000，停下来查是不是加载了旧 normalizer。

## 推理（不必等重训，111k 也可以这样出图）

原生大图（1536 等）：

```bash
# 默认就是整图，GroupNorm 用全图统计（这张 Lifeact 上比孤立 256 好）
python -m microscopy_vae infer --config ... --weights ... --normalizer ... \
  --input <native_page> --output out.npy --source BioTISR

# 显式 halo：每个 256 核带 64 px 真邻域。不要先裁成 256 再送。
python -m microscopy_vae infer --inference-mode halo --halo 64 ...

# --tiled 在大图上会自动改成 halo。只有对照旧分块时才加：
python -m microscopy_vae infer --inference-mode tiled --allow-isolated-tiles ...
```

**不要**从 1536 上裁 `[640:896, 640:896)` 存成 256 再 infer：那时已经没有真邻域，halo 也救不回来。把整页或至少带邻域的窗口送进去。

训练 val 仍是 256 中心 crop（和训练分布一致）。生产重建看 full / halo。

## 验证日志里会多两项（不进损失）

val 每次会写 `nmse` 和 `ssim_local`（Target 的 p99.5−p0.5 当 SSIM 尺子）。  
报均值的重建 NMSE=1；range=1 的 SSIM 在暗图上会虚高。这两项 **只记账，不反传**，不会和关 GAN / 新 support 抢贡献。看 `metrics_val.jsonl` 和 val 那一行 log。

## 建议的本机步骤

1. `git pull` 到含 0.3.3 的代码。  
2. 新目录，例如 `runs/s1_hq_f8z4_v6_seed0`。  
3. `s1_hq_f8z4_v6.yaml` 里填 `data.manifest_path` / `path_prefix_target`。  
4. 确认 `loss.adversarial.enabled: false`。  
5. 单卡或双卡按现有 V5 启动方式训；全局 batch 仍是 8。  
6. 第一轮 log 核对三源 `support_floor`。  
7. 权重出来后：同一张 Lifeact **原生页** full 与 halo，以及（仅诊断）孤立 256。

## 这一版故意没动的（先不要加）

- amp 0.08 门控、开 Scharr/HF/dark_fp、改 GroupNorm、改 f4、调感知权重。  
  下一版若孤立 256 仍是珠，再单变量做。
