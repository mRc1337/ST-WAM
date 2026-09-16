# DINO target correspondence analysis

This is an offline side branch. It uses the same frozen DINOv3 ViT-S/16
checkpoint, preprocessing, 224×448 native input, and `x_norm_patchtokens`
output as the current dual-space ST-WAM task configuration. With
`view_mode: current_concat`, the 14×28 tokens are split back into front/wrist
14×14 maps after DINO; views are never averaged and the policy is untouched.

Run the checked-in pilot:

```bash
source scripts/activate_libero_plus_env.sh
python scripts/analysis/run_dino_target_analysis.py \
  --config configs/analysis/dino_target_similarity.yaml \
  --stage all
```

The stages can be resumed independently:

```bash
python scripts/analysis/run_dino_target_analysis.py --stage extract --run-dir outputs/dino_target_similarity/pilot
python scripts/analysis/run_dino_target_analysis.py --stage analyze --run-dir outputs/dino_target_similarity/pilot
python scripts/analysis/run_dino_target_analysis.py --stage plot --run-dir outputs/dino_target_similarity/pilot
```

Each `heatmaps/<episode>.npz` keeps raw cosine maps under the camera names
(`front`, `wrist`) and also stores fixed-range visualization maps under
`normalized_front` and `normalized_wrist`. The normalization bounds and
strategy are stored alongside the arrays.

For a future evaluation, clean raw RGB plus action/state arrays can be
recorded without changing inference:

```bash
python experiments/libero/eval_libero_single.py \
  task=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 \
  ckpt=/path/to/checked/stwam/checkpoint.pt \
  analysis.record_rollout=true
```

When the LIBERO renderer is available, simulator target masks can be saved
as a frame-specific evaluation-only side channel:

```bash
python experiments/libero/eval_libero_single.py \
  task=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 \
  ckpt=/path/to/checked/stwam/checkpoint.pt \
  analysis.record_gt_masks=true \
  'analysis.target_instances=[bbq_sauce_1]'
```

This uses LIBERO's `SegmentationRenderEnv` and stores `gt_masks[T,V,H,W]` in
the NPZ. `target_instances` is required because LIBERO's BDDL
`obj_of_interest` can include both the manipulated object and its receptacle;
the example selects only the sauce. For a multi-task run, use
`analysis.target_instances_by_task` instead. It does not feed masks to the
policy. If the renderer cannot provide the requested segmentation keys or
instance mapping, evaluation stops instead of producing an incomplete
trajectory.

Use `annotations/dino_target_similarity/` as the minimal annotation format.
`bbox`/`mask_path` is read only at `reference_frame`; future target metrics
require `evaluation_gt_masks`, `gt_masks`, or frame-indexed
`evaluation_gt_bboxes`. A reference mask is never reused as a future GT mask.

For cross-episode correspondence, set `analysis.reference_mode` to
`task_exemplar` and provide `exemplar_episode_id` in the task/camera
annotation (or `analysis.exemplar_episode_by_task`). The exemplar prototype is
fixed and is never updated from the evaluated episode.
