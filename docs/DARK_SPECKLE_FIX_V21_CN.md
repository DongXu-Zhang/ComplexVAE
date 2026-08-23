# 暗背景白点：代码已改（v2.1），以及做不到的事

## 先说清楚「完美解决」的边界

**4×4 小格子不会从几何上消失。** 当前主模型仍是 f4：一个 latent 格对应 4×4 像素。这不是 bug，是压缩本身。  
这次改的是把黑背景里的噪点**过度放大成满屏白块**的那几条路径。格子还在，但不该再被损失和采样当成「必须拟合的亮点」。

必须**重新训练**（或至少从 v2 checkpoint 起用新损失微调）。**旧权重 + 新损失不能叫已经修好显示结果。**

下一轮请用：

```text
configs/experiment/s1_hq_f4z4_v2_1.yaml
```

不要再用会放大空 patch 的 `s1_hq_f4z4_v2.yaml` 当主训练。

---

## 改了什么

| 问题 | v2（出白点） | v2.1 |
|---|---|---|
| 空黑图被 `s=max(range,0.05)` 最多放大 20 倍 | 是 | 动态范围 < 0.08 时 **不再放大**（除数改回 1.0）；有结构时下限改为 0.20 |
| 边缘权重无上限，孤立噪点权可以极大 | 是 | **clip 到 3** |
| 高频项继续奖励暗区砂 | 0.05 | **0.02** |
| Flux 管不住「背景假亮」 | 只有全图均值 | 增加 **暗区假阳性损失**：只罚「原图暗、重建更亮」 |
| coverage 仍可能裁到纯黑 | 是 | 训练 crop 若稳健幅度 < 0.08，**最多再试 8 次** |

没有：逐 patch min-max、负值清零、GAN、转置卷积。

---

## 服务器怎么跑

和 v2 一样，只换配置名和输出目录：

```bash
export PYTHONPATH=$PWD/src
python -m microscopy_vae.cli train \
  --config configs/experiment/s1_hq_f4z4_v2_1.yaml \
  --override data.manifest_path=/data/zhangdongxu/VAE/manifests/hq_manifest_v2.jsonl \
  --override data.path_prefix_source='F:\\Dataset' \
  --override data.path_prefix_target=/data/zhangdongxu/VAE/raw/Dataset \
  --override data.path_require_exists=true \
  --override sampling.focus_sidecar_path=/data/zhangdongxu/VAE/manifests/focus_sidecar_v1.jsonl \
  --override experiment.output_dir=/data/zhangdongxu/VAE/outputs/ComplexVAE/s1_hq_f4z4_v2_1_seed0
```

看结果时除了 PSNR，请看暗区是否还成片冒白点。若只剩很淡的 4 像素网格、不再像满天星，诱因就算压住了。
