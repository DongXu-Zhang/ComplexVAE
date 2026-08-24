# 修改 3 短程消融（不要一上来跑 150k）

正式 baseline 仍是 `../s1_hq_f4z4_v2_1.yaml`，本目录**没有覆盖它**。

| 实验 | 配置 | 新 loss |
|---|---|---|
| A | `A_baseline_short.yaml` | 无（对照：同配方继续/短训） |
| B | `B_perceptual_short.yaml` | perceptual only |
| C | `C_gan_short.yaml` | GAN only |
| D | `D_perc_gan_short.yaml` | both |
| E | `E_rebalanced_short.yaml` | both，权重待校准后改 |

共同约束：

- 相同 `seed: 0`、相同 crop/model/normalizer、相同数据
- `max_steps: 2000`（先看稳不稳定和白点，再考虑完整训练）
- 若从已有 v2.1 checkpoint 微调，设 `training.warmstart_vae_path`（只加载 VAE，step 从 0 计）
- **必须再跑一个 A**：同样 warmstart、原损失再训 2000 step，避免把「继续训练」误当成新 loss 的效果
- `resume_exact_path` 不能用来从旧 config 接 GAN（config hash 会对不上）

```bash
# 例：B，manifest 与路径按你的环境 override
python -m microscopy_vae.cli train \
  --config configs/experiment/ablation_mod3/B_perceptual_short.yaml \
  --override data.manifest_path='"..."' \
  --override data.path_prefix_target='"..."' \
  --override training.warmstart_vae_path='"/path/to/v2_1.pt"'
```

短程通过后再把 `max_steps` 提到 150000，output_dir 换新名字，不要覆盖 `runs/s1_hq_f4z4_v2_1`。
