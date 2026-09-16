# Figure 5 MoT attention reproduction

This is a definition-level reproduction of the ST-WAM Figure 5-style
cross-branch diagnostic. The paper does not publicly specify the layer, head
set, denoising step, or action-query aggregation used for that visualization;
the settings below must not be described as a pixel-level replication.

## Fixed protocol

- Official `st-wam-base-step021700` checkpoint, step `21700`.
- LIBERO-PRO, `seed=42`, one complete episode per task.
- Legacy training convention: 256px simulator RGB, 180° image rotation,
  `legacy_rlds` gripper action conversion.
- Replan every 10 environment steps; 10 action denoising steps.
- MoT layer `29`, denoising execution step `9`, all 24 heads, action queries
  `[0,32)`, display percentile normalization `[1,99]`.
- Attention is recomputed from the action branch's projected Q/K and the real
  complete-key mask; DINO/VAE scores are sliced only after softmax.

## Task mapping and result

| Row | Suite/task | Runtime-checked description | Success | Captures |
|---|---|---|---:|---:|
| drawer | `libero_goal/task_id=0` | `open the middle drawer of the cabinet` | 1/1 | 13 calls |
| moka pot | `libero_10/task_id=2` | `turn on the stove and put the moka pot on it` | 1/1 | 24 calls |

The task descriptions were checked against the resolved LIBERO benchmark task
at runtime. Each attention video covers the complete episode at one frame per
environment step. The initial 30 wait steps use an explicit no-attention slate;
each subsequent map is held until the next inference call, and the final map
is held only through the terminal step. The evaluator's rollout MP4 remains
under each task's `results/` directory.

## Auditable artifacts

Run root:

`evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/`

- [protocol.json](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/protocol.json)
- [drawer attention records](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/drawer/attention)
- [drawer attention video](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/drawer/attention_episode.mp4)
- [drawer attention-video manifest](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/drawer/attention_episode.json)
- [drawer rollout video](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/drawer/results/libero_goal/videos)
- [moka-pot attention records](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/moka_pot/attention)
- [moka-pot attention video](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/moka_pot/attention_episode.mp4)
- [moka-pot attention-video manifest](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/moka_pot/attention_episode.json)
- [moka-pot rollout video](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/moka_pot/results/libero_10/videos)
- [final 2×2 panel](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/figure5_2x2.png)
- [RGB-only selection manifest](/home/pai/zxw/ST-WAM/evaluate_results/mot_attention_figure5/base_step021700_seed42_20260916_033752_audited/figure5_2x2.json)

Checkpoint SHA-256:

`b35ef10d6565f14173f7cec155861f8b999185e0c8127678a076a20f5cb3594c`

The matching official `dataset_stats.json` SHA-256 is
`becca8802b928ce8fd1d75cef191940180d3149ddff1ecf9235ad322297fc7f1`.

Representative calls were chosen from RGB-only contact sheets: drawer call 9
(environment step 120), and moka-pot call 23 (environment step 260). Heatmap
appearance was not used to choose either frame. The final panel columns are
`Action Queries → Current DINO` and `Action Queries → Current VAE`, with rows
`drawer` and `moka pot`.

## Verification

```bash
pytest -q tests/test_mot_attention_visualization.py
```

The required suite passes (`10 passed`). The fixed protocol produced stable
branch-specific maps at the RGB-selected interaction stages, so the
pre-registered layer/denoising scan was not triggered. The audited attention
videos contain 152 frames for drawer and 263 frames for moka pot, matching
the full environment-step timelines. Every audited
`attention.npz` includes the exact complete-key mask used for the saved
record; the recorded `mass_all_keys` values are 1.0 up to float32 roundoff.
