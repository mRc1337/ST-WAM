#!/usr/bin/env bash
set -euo pipefail

# Run the official ST-WAM and the user's Dual-Space checkpoint on the same
# first LIBERO-Object episode, then compare raw MoT action-to-VAE/DINO maps.
# The official checkpoint is the full ST-WAM model (including CAIR); the user
# checkpoint defaults to the three-task DINO+VAE model without CAIR.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
VENV_PATH="${STWAM_VENV_PATH:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"

LIBERO_ROOT="${LIBERO_ROOT:-$WORKSPACE_ROOT/LIBERO-PRO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LIBERO_ROOT/.libero}"
EGL_VENDOR_CONFIG="${EGL_VENDOR_CONFIG:-$PROJECT_ROOT/configs/egl/10_nvidia.json}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
TASK_ID="${TASK_ID:-3}"

USER_CKPT="${USER_CKPT:-$PROJECT_ROOT/runs/three_task/dual_space_seed42_1500/checkpoints/weights/step_001500.pt}"
# Keep the official model under the requested /mnt/data hierarchy.
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/mnt/data/embodied_datasets/ST-WAM/checkpoints/modelscope/Semantic_Temporal_World_Action_Model}"
OFFICIAL_CKPT="${OFFICIAL_CKPT:-$OFFICIAL_ROOT/libero/ST-WAM/st-wam-base-step021700/model.pt}"
OFFICIAL_CONFIG="${OFFICIAL_CONFIG:-$OFFICIAL_ROOT/libero/ST-WAM/st-wam-base-step021700/config.yaml}"
OFFICIAL_STATS="${OFFICIAL_STATS:-$OFFICIAL_ROOT/libero/ST-WAM/st-wam-base-step021700/dataset_stats.json}"
# This Qwen directory already exists on the machine and has the matching
# Qwen3-VL-4B architecture.  It is not duplicated into the ST-WAM folder.
QWEN_PATH="${QWEN_PATH:-/mnt/data/zxw/lda/pretrained/vlm/Qwen3-VL-4B-Instruct}"

# Use one normalization file for both models so proprioception does not become
# an uncontrolled difference in the attention comparison.
COMMON_STATS="${COMMON_STATS:-$PROJECT_ROOT/runs/three_task/dino_future_seed42_1500/dataset_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/evaluate_results/official_vs_user_attention/task${TASK_ID}_seed${SEED}_$(date -u +%Y%m%d_%H%M%S)}"
ATTENTION_LAYER="${ATTENTION_LAYER:-29}"
ATTENTION_STEP="${ATTENTION_STEP:-9}"

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_official_vs_user_attention.sh

Environment overrides:
  USER_CKPT OFFICIAL_ROOT OFFICIAL_CKPT OFFICIAL_CONFIG OFFICIAL_STATS
  QWEN_PATH COMMON_STATS OUTPUT_ROOT LIBERO_ROOT GPU_ID SEED TASK_ID
  ATTENTION_LAYER ATTENTION_STEP PYTHON_BIN STWAM_VENV_PATH

The default task is LIBERO-Object task 3 (BBQ sauce), one episode, seed 42,
layer 29, action denoising step 9, all heads and all 32 action queries.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for path in "$PYTHON_BIN" "$USER_CKPT" "$OFFICIAL_CKPT" "$OFFICIAL_CONFIG" \
  "$OFFICIAL_STATS" "$COMMON_STATS" "$QWEN_PATH" "$EGL_VENDOR_CONFIG"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$LIBERO_ROOT/libero/libero/assets" ]] || {
  echo "Invalid LIBERO_ROOT: $LIBERO_ROOT" >&2
  exit 1
}

mkdir -p "$OUTPUT_ROOT"

export LIBERO_ROOT LIBERO_CONFIG_PATH
export PYTHONPATH="$LIBERO_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_CONFIG"
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export EGL_DEVICE_ID="$GPU_ID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="$PROJECT_ROOT/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"

QUERY_RANGES='["0:32"]'

run_one() {
  local name="$1" task_config="$2" checkpoint="$3"
  local attention_dir="$OUTPUT_ROOT/$name/attention"
  local result_dir="$OUTPUT_ROOT/$name/results"
  local -a extra_overrides=()
  if [[ "$name" == "official" ]]; then
    extra_overrides=(
      "model.semantic_history_config.vlm_model_name_or_path=$QWEN_PATH"
      "model.semantic_history_config.allow_vlm_path_relocation=true"
    )
  fi
  mkdir -p "$attention_dir" "$result_dir"
  echo "[$name] checkpoint=$checkpoint"
  "$PYTHON_BIN" experiments/libero/eval_libero_single.py \
    --config-name sim_libero_v3 \
    "task=$task_config" \
    "data=libero_object_tasks_1_4_7_v3" \
    "ckpt=$checkpoint" \
    "seed=$SEED" \
    "EVALUATION.device=cuda:0" \
    "EVALUATION.task_suite_name=libero_object" \
    "EVALUATION.task_id=$TASK_ID" \
    "EVALUATION.num_trials=1" \
    "EVALUATION.num_steps_wait=30" \
    "EVALUATION.replan_steps=10" \
    "EVALUATION.num_inference_steps=10" \
    "EVALUATION.rand_device=cpu" \
    "EVALUATION.output_dir=$result_dir" \
    "EVALUATION.dataset_stats_path=$COMMON_STATS" \
    "EVALUATION.mot_attention_visualization.enabled=true" \
    "EVALUATION.mot_attention_visualization.output_dir=$attention_dir" \
    "EVALUATION.mot_attention_visualization.layers=[$ATTENTION_LAYER]" \
    "EVALUATION.mot_attention_visualization.denoising_steps=[$ATTENTION_STEP]" \
    "EVALUATION.mot_attention_visualization.heads=all" \
    "EVALUATION.mot_attention_visualization.action_query_ranges=$QUERY_RANGES" \
    "EVALUATION.mot_attention_visualization.call_indices=all" \
    "EVALUATION.mot_attention_visualization.max_saves=1" \
    "EVALUATION.mot_attention_visualization.display_normalization=minmax" \
    "EVALUATION.mot_attention_visualization.overlay_alpha=0.45" \
    "model.loss.lambda_dino=0.02" \
    "${extra_overrides[@]}" \
    2>&1 | tee "$OUTPUT_ROOT/$name/eval.log"
}

run_one official \
  libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4 \
  "$OFFICIAL_CKPT"
run_one user \
  libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 \
  "$USER_CKPT"

"$PYTHON_BIN" scripts/compare_mot_attention_records.py \
  --official-dir "$OUTPUT_ROOT/official/attention" \
  --user-dir "$OUTPUT_ROOT/user/attention" \
  --output-dir "$OUTPUT_ROOT/comparison"

echo
echo "Completed official-vs-user attention comparison."
echo "Output: $OUTPUT_ROOT"
