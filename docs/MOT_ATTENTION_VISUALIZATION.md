# MoT cross-branch attention visualization

This repository now has an optional Figure-5-style diagnostic for the
three-branch `FastWAMVAEDinoMoT` model. It records the action branch's real MoT
self-attention to the current VAE (`video`) and DINO (`dino`) tokens from the
same forward call, layer, and action denoising step. It is not DINO encoder
attention, feature cosine similarity, attention rollout, or an active/no-op
difference.

## What the implementation found

The relevant path is:

1. `FastWAMVAEDinoMoT.infer_action` builds the first-frame VAE and DINO
   pre-states and runs the action denoising schedule in execution order.
2. `MoT._build_expert_attention_io` produces each expert's projected,
   RMS-normalized, RoPE-applied Q/K and unnormalized V.
3. `MoT.forward_action_with_static_cache` uses cached VAE+DINO K/V followed by
   freshly computed action K/V. The uncached path uses the same tensors through
   `MoT.forward`.
4. The action rows of `_build_tribranch_attention_mask` can see current/first
   frame VAE tokens, current/first frame DINO tokens, and action tokens. Future
   video tokens remain in the complete key sequence when they exist, with their
   real mask entries preserved.

The live three-branch construction is validated as the observed order
`video` (VAE) → `dino` (DINO) → `action`; the order is recorded in metadata and
an unsupported order raises instead of producing a mislabeled map.

The side path recomputes

```text
softmax((Q_action @ K_all.T) / sqrt(head_dim) + the_real_mask, dim=-1)
```

in float32, over all keys first, and only then slices the VAE/DINO columns.
The original SDPA call, action scheduler, and static KV cache are unchanged.

VAE and DINO spatial layouts come from their actual `pre_dit` metadata. This
includes the VAE/DINO token grid, DINO latent patch merging, and DINO
view-aware ordering. The LIBERO entry point also supplies camera names, the
post-preprocessing camera sizes, and the horizontal/vertical concatenation
layout. The displayed image is the aligned model input; raw camera images are
saved alongside it when available.

The selected layers, head set, action-query ranges, and denoising steps are
implementation choices. They are not claimed to be settings used by the
paper authors.

## LIBERO usage

The existing single-process LIBERO evaluator is the real inference entry
point. With a compatible checkpoint and local LIBERO installation:

```bash
export LIBERO_ROOT=/path/to/LIBERO
python experiments/libero/eval_libero_single.py \
  task=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 \
  ckpt=/path/to/st_wam_checkpoint.pt \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.output_dir=./evaluate_results/mot_attention_smoke \
  EVALUATION.mot_attention_visualization.enabled=true \
  EVALUATION.mot_attention_visualization.output_dir=./evaluate_results/mot_attention_smoke/attention \
  EVALUATION.mot_attention_visualization.layer=last \
  EVALUATION.mot_attention_visualization.denoising_step=last \
  EVALUATION.mot_attention_visualization.max_saves=1
```

The feature is off by default. Explicitly disable it with
`EVALUATION.mot_attention_visualization.enabled=false`. When it is off, the
model does not recompute attention, copy the image to CPU, or write any
visualization files.

The supported settings are configured under
`EVALUATION.mot_attention_visualization` in `configs/sim_libero.yaml`:

- `layer`: legacy single-selection fallback, `last` or a zero-based MoT layer.
- `layers`: optional list such as `[9, 19, 29]`; overrides `layer`.
- `denoising_step`: legacy single-selection fallback, `last` or a zero-based action denoising execution index;
  `last` means the final loop iteration, not the largest timestep value.
- `denoising_steps`: optional list such as `[0, 4, 9]`; overrides `denoising_step`.
- `heads`: `all` or a list such as `[0, 3, 7]`.
- `action_query_range`: legacy single-range fallback, `all`, `[start, end]`, or a string such as `2:8`;
  `end` is exclusive.
- `action_query_ranges`: optional list such as `["0:8", "8:16", "16:24", "24:32"]`.
- `call_indices`: `all` or a list of zero-based `infer_action` call numbers.
- `max_saves`: process-wide maximum number of saved records.
- `display_normalization`: `percentile` (default), `minmax`, or `none`; normalization is
  display-only.
- `display_percentiles`: percentile clipping bounds, default `[1.0, 99.0]`.
- `overlay_alpha`: heatmap overlay opacity.

`camera_layout: null` means the LIBERO entry point uses the actual
`data.train.concat_multi_camera` value (for example `horizontal` or
`vertical`). Set it explicitly only when the supplied camera images and model
input use that same layout.

## Output

Each saved record is under
`<output_dir>/call_<call>/<step_<step>_layer_<layer>>/` and contains:

```text
rgb_input.png                 aligned model-input RGB image
rgb_original_*.png            raw camera images when supplied by the entry point
rgb_original_concatenated.png original camera observation in the configured layout
attention.npz                 raw scores, selected attention, and branch masses
comparison_q*.png             RGB | absolute DINO/VAE | conditional DINO/VAE
metadata.json                 layer/step/timestep, mask, spans, grids, cameras, and settings
```

When only one query range is requested, the legacy `heatmap_*.png`,
`overlay_*.png`, and `comparison.png` aliases are also written.

`attention.npz` contains at least `scores_dino [B,N_dino]`,
`scores_vae [B,N_vae]`, `mass_dino`, `mass_vae`, and `mass_all_keys`. The two
branch masses are `scores_*.sum(-1)` before display normalization. Attention
assigned to action, future, or any other visible keys is intentionally not
renormalized away, so `mass_dino + mass_vae < 1` is expected.

Absolute maps average the original complete-key softmax probabilities. The
conditional maps first normalize every valid head/query inside each visual
branch and then average, preventing a high-mass diffuse head from hiding a
low-mass localized head. Rows with branch mass at or below `1e-8` are excluded.
PNG colors are percentile-clipped independently within each map. Branch mass
is printed in the comparison panels and stored in `metadata.json`/`attention.npz`.

## Reproduction run

The exact two-task run is scripted by
`scripts/reproduce_mot_attention_figure5.sh`. It uses the released
`st-wam-base-step021700` checkpoint, the matching official
`dataset_stats.json`, LIBERO-PRO, legacy RGB/action conventions, seed 42,
10 action denoising steps, and 10 environment steps per replan. The runner
also writes the checkpoint/statistics hashes and runs
`scripts/make_mot_attention_video.py` for each episode. It derives the exact
episode length from the evaluator's rollout video: initial wait steps are
represented by a no-attention slate, and the final attention map is held only
through the terminal environment step.

The fixed-protocol output and RGB-only selection manifest are recorded in
`docs/MOT_ATTENTION_FIGURE5_REPRODUCTION.md`. A scan over
`layers=[9,19,29]` and `denoising_steps=[0,4,9]` is not part of the primary
run; it should be launched only if the fixed-protocol RGB-selected frames do
not show a stable qualitative branch difference, using identical scan
parameters for both tasks.

## Tests and validation

The offline tests cover complete-key softmax, mask zeros, all-key mass,
head/query aggregation, flat and view-aware spatial restoration, explicit
multi-camera geometry, constant-map normalization, save contents, and the
disabled/no-output behavior:

```bash
pytest -q tests/test_mot_attention_visualization.py
```

In addition to the offline tests, the recorded run verifies model loading,
runtime task-description checks, legacy LIBERO rendering/action conventions,
and complete-key mass checks in every saved `attention.npz`.
