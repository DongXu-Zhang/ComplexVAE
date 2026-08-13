# Data contract (S1 HQ codec)

## Split tokens

Program values: `train` | `val` | `test` only.  
Human “validation” → `val`. Unknown tokens are **rejected**.

## Biological group integrity

- Split is by `biological_group_id`.
- Pages from the same group never cross splits.
- Metrics: page → group mean → equal-weight group macro.

## HQ page record (JSONL)

Required fields:

```text
split, source_dataset, biological_group_id, path, shape, page_index
```

Recommended: `dtype`, `morphology`, `target_role`, `target_provenance`, `sample_id`.

Shape:

- multi-page: `[P,H,W]`
- single-page: `[H,W]` or `[1,H,W]`

`page_index` selects `P`. **P is never channel.**

## Roles

| Role | S1 use |
|---|---|
| HQ (`SIM_gt` / `RC_highsnr`) | train/val codec target |
| `wf_lowsnr` | not in S1 |
| `wf_highsnr` | not in S1 (control only later) |
| test | locked |

## Normalization

- Fit **train HQ only** (robust p0.1–p99.9 or identity for smoke).
- Val/test apply only.
- No per-image min-max.
- Artifact: `normalizer.json` with transform_id + hashes.

## Sampling

```text
source (sqrt n_groups) → group → page → crop
```

Log realized source frequencies.

## Test fail-closed

- Training configs cannot list `test` in `allow_splits`.
- `SyntheticHQDataset` / `ManifestHQDataset` refuse `split="test"`.
- Trainer never constructs `test_loader`.
