#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${STWAM_VENV_PATH:-${PROJECT_ROOT}/.venv}"
CLEAN_LIBERO_ROOT="${CLEAN_LIBERO_ROOT:-/home/pai/zxw/LIBERO-PRO}"
PLUS_LIBERO_ROOT="${PLUS_LIBERO_ROOT:-/home/pai/zxw/LIBERO-plus}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero}"
STATS_PATH="${STATS_PATH:-${PROJECT_ROOT}/runs/single_task/dino_future_seed42_500/dataset_stats.json}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/evaluate_results/single_task/manifests/bbq_plus_140_seed42}"
CLASSIFICATION_PATH="${CLASSIFICATION_PATH:-${PLUS_LIBERO_ROOT}/libero/libero/benchmark/task_classification.json}"
EGL_DEVICE="${LIBERO_EGL_DEVICE_ID:-0}"

VARIANTS="vae,dual"
CPT_STEP=500
RESULT_TAG=""
CLEAN_TRIALS=50
PLUS_TRIALS=1
RUN_CLEAN=1
RUN_PLUS=1
FORCE_CLEAN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_single_task_ablation.sh [options]

Options:
  --variants LIST       Comma-separated: dino,vae,dual (default: vae,dual)
  --step N              Checkpoint step shared by all variants (default: 500)
  --result-tag TAG      Result-directory tag (default: legacy names for step 500,
                        otherwise stepN_seed42)
  --clean-trials N      Clean rollouts per variant (default: 50)
  --plus-trials N       Rollouts per Plus case (default: 1)
  --clean-only          Run only clean LIBERO evaluation
  --plus-only           Run only LIBERO-Plus evaluation
  --force-clean         Rerun clean even when its result JSON already exists
  -h, --help            Show this help

Environment overrides:
  CLEAN_LIBERO_ROOT, PLUS_LIBERO_ROOT, LIBERO_DATA_ROOT, STATS_PATH,
  MANIFEST_DIR, CLASSIFICATION_PATH, LIBERO_EGL_DEVICE_ID, STWAM_VENV_PATH
EOF
}

while (($#)); do
  case "$1" in
    --variants)
      VARIANTS="${2:?--variants requires a value}"
      shift 2
      ;;
    --step)
      CPT_STEP="${2:?--step requires a value}"
      shift 2
      ;;
    --result-tag)
      RESULT_TAG="${2:?--result-tag requires a value}"
      shift 2
      ;;
    --clean-trials)
      CLEAN_TRIALS="${2:?--clean-trials requires a value}"
      shift 2
      ;;
    --plus-trials)
      PLUS_TRIALS="${2:?--plus-trials requires a value}"
      shift 2
      ;;
    --clean-only)
      RUN_PLUS=0
      shift
      ;;
    --plus-only)
      RUN_CLEAN=0
      shift
      ;;
    --force-clean)
      FORCE_CLEAN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$CLEAN_TRIALS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --clean-trials: $CLEAN_TRIALS" >&2; exit 2; }
[[ "$PLUS_TRIALS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --plus-trials: $PLUS_TRIALS" >&2; exit 2; }
[[ "$CPT_STEP" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --step: $CPT_STEP" >&2; exit 2; }
printf -v CPT_STEP_PADDED '%06d' "$CPT_STEP"
if [[ -z "$RESULT_TAG" && "$CPT_STEP" -ne 500 ]]; then
  RESULT_TAG="step${CPT_STEP}_seed42"
fi
if [[ -n "$RESULT_TAG" && ! "$RESULT_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid --result-tag: $RESULT_TAG" >&2
  exit 2
fi

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing virtual environment: $VENV_PATH" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

cd "$PROJECT_ROOT"
export LIBERO_DATA_ROOT
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="${PROJECT_ROOT}/.runtime/matplotlib"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$MPLCONFIGDIR"

test -s "$STATS_PATH" || { echo "Missing dataset stats: $STATS_PATH" >&2; exit 1; }
test -d "$CLEAN_LIBERO_ROOT/libero/libero/assets" || { echo "Invalid clean LIBERO root: $CLEAN_LIBERO_ROOT" >&2; exit 1; }
test -d "$PLUS_LIBERO_ROOT/libero/libero/assets/new_objects" || { echo "Invalid Plus LIBERO root/assets: $PLUS_LIBERO_ROOT" >&2; exit 1; }
test -s "$CLASSIFICATION_PATH" || { echo "Missing classification: $CLASSIFICATION_PATH" >&2; exit 1; }

if ((RUN_PLUS)); then
  test -s "$MANIFEST_DIR/full.json" || { echo "Missing Plus manifest: $MANIFEST_DIR/full.json" >&2; exit 1; }
  for gpu in 0 1 2 3; do
    test -s "$MANIFEST_DIR/gpu${gpu}.json" || { echo "Missing Plus shard: $MANIFEST_DIR/gpu${gpu}.json" >&2; exit 1; }
  done
  python - "$MANIFEST_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
full = json.loads((root / "full.json").read_text())
shards = [json.loads((root / f"gpu{i}.json").read_text()) for i in range(4)]
assert len(full) == 140, f"expected 140 Plus cases, got {len(full)}"
assert [len(x) for x in shards] == [35, 35, 35, 35]
assert {tuple(sorted(x.items())) for x in full} == {
    tuple(sorted(x.items())) for shard in shards for x in shard
}
print("Manifest verified: 140 cases, four shards of 35")
PY
fi

variant_config() {
  local clean_suffix plus_suffix
  if [[ -z "$RESULT_TAG" ]]; then
    clean_suffix="clean_v3_fixed"
    plus_suffix="plus_140_seed42"
  else
    clean_suffix="clean_${RESULT_TAG}"
    plus_suffix="plus_140_${RESULT_TAG}"
  fi
  case "$1" in
    dino)
      TASK_CONFIG="libero_dino_s_smallvideo_2cam_224_1e-4"
      CHECKPOINT="${PROJECT_ROOT}/runs/single_task/dino_future_seed42_500/checkpoints/weights/step_${CPT_STEP_PADDED}.pt"
      CLEAN_OUTPUT="${PROJECT_ROOT}/evaluate_results/single_task/dino_${clean_suffix}"
      PLUS_OUTPUT="${PROJECT_ROOT}/evaluate_results/single_task/dino_${plus_suffix}"
      ;;
    vae)
      TASK_CONFIG="libero_uncond_2cam224_1e-4"
      CHECKPOINT="${PROJECT_ROOT}/runs/single_task/fastwam_vae_seed42_500/checkpoints/weights/step_${CPT_STEP_PADDED}.pt"
      CLEAN_OUTPUT="${PROJECT_ROOT}/evaluate_results/single_task/vae_${clean_suffix}"
      PLUS_OUTPUT="${PROJECT_ROOT}/evaluate_results/single_task/vae_${plus_suffix}"
      ;;
    dual)
      TASK_CONFIG="libero_wan5b_dino_s_aux_mot_2cam_224_1e-4"
      CHECKPOINT="${PROJECT_ROOT}/runs/single_task/dual_space_seed42_500/checkpoints/weights/step_${CPT_STEP_PADDED}.pt"
      CLEAN_OUTPUT="${PROJECT_ROOT}/evaluate_results/single_task/dual_${clean_suffix}"
      PLUS_OUTPUT="${PROJECT_ROOT}/evaluate_results/single_task/dual_${plus_suffix}"
      ;;
    *)
      echo "Unknown variant: $1 (expected dino, vae, or dual)" >&2
      return 1
      ;;
  esac
  test -s "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT" >&2; return 1; }
}

set_clean_environment() {
  export LIBERO_ROOT="$CLEAN_LIBERO_ROOT"
  export LIBERO_CONFIG_PATH="$CLEAN_LIBERO_ROOT/.libero"
  export PYTHONPATH="$CLEAN_LIBERO_ROOT:${PROJECT_ROOT}/src:${PROJECT_ROOT}"
  export CUDA_VISIBLE_DEVICES=0
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
}

set_plus_environment() {
  export LIBERO_ROOT="$PLUS_LIBERO_ROOT"
  export LIBERO_PLUS_ROOT="$PLUS_LIBERO_ROOT"
  export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.runtime/libero"
  export PYTHONPATH="$PLUS_LIBERO_ROOT:${PROJECT_ROOT}/src:${PROJECT_ROOT}"
  unset CUDA_VISIBLE_DEVICES
}

run_clean() {
  local variant="$1"
  local result_file="${CLEAN_OUTPUT}/libero_object/gpu0_task3_results.json"
  if ((FORCE_CLEAN == 0)) && [[ -s "$result_file" ]]; then
    local existing_trials
    existing_trials="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("total_episodes", 0))' "$result_file" 2>/dev/null || echo 0)"
    if [[ "$existing_trials" -eq "$CLEAN_TRIALS" ]]; then
      echo "[$variant clean] skip existing ${existing_trials}-trial result: $result_file"
      return
    fi
    echo "[$variant clean] existing result has ${existing_trials} trials; rerunning ${CLEAN_TRIALS} trials"
  fi

  set_clean_environment
  echo "[$variant clean] starting ${CLEAN_TRIALS} trials"
  python experiments/libero/eval_libero_single.py \
    --config-name sim_libero_v3 \
    task="$TASK_CONFIG" \
    data=libero_object_task1_v3 \
    ckpt="$CHECKPOINT" \
    EVALUATION.dataset_stats_path="$STATS_PATH" \
    EVALUATION.task_suite_name=libero_object \
    EVALUATION.task_id=3 \
    'EVALUATION.expected_task_description=pick up the bbq sauce and place it in the basket' \
    EVALUATION.num_trials="$CLEAN_TRIALS" \
    EVALUATION.output_dir="$CLEAN_OUTPUT" \
    seed=42
  echo "[$variant clean] complete: $result_file"
}

active_worker_pids=()
cleanup_workers() {
  if ((${#active_worker_pids[@]})); then
    kill "${active_worker_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup_workers INT TERM EXIT

run_plus() {
  local variant="$1"
  local expected_count
  expected_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$MANIFEST_DIR/full.json")"
  local completed_count=0
  if [[ -d "${PLUS_OUTPUT}/libero_object" ]]; then
    completed_count="$(find "${PLUS_OUTPUT}/libero_object" -maxdepth 1 -type f -name 'gpu*_task*_results.json' | wc -l)"
  fi
  if [[ "$completed_count" -eq "$expected_count" && -s "${PLUS_OUTPUT}/plus_summary.json" ]]; then
    echo "[$variant plus] skip complete result: ${completed_count}/${expected_count}"
    return
  fi

  set_plus_environment
  mkdir -p "${PLUS_OUTPUT}/worker_logs"
  active_worker_pids=()
  echo "[$variant plus] starting; existing=${completed_count}/${expected_count}"
  for gpu in 0 1 2 3; do
    MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE" EGL_DEVICE_ID="$EGL_DEVICE" \
    python experiments/libero/eval_libero_worker.py \
      --config-name sim_libero_v3 \
      task="$TASK_CONFIG" \
      data=libero_object_task1_v3 \
      ckpt="$CHECKPOINT" \
      gpu_id="$gpu" \
      EVALUATION.device="cuda:$gpu" \
      EVALUATION.dataset_stats_path="$STATS_PATH" \
      EVALUATION.num_trials="$PLUS_TRIALS" \
      EVALUATION.output_dir="$PLUS_OUTPUT" \
      +EVALUATION.task_manifest="$MANIFEST_DIR/gpu${gpu}.json" \
      +EVALUATION.skip_existing=true \
      +EVALUATION.continue_on_error=false \
      seed=42 \
      >"${PLUS_OUTPUT}/worker_logs/gpu${gpu}.log" 2>&1 &
    active_worker_pids+=("$!")
  done

  local failed=0
  for pid in "${active_worker_pids[@]}"; do
    wait "$pid" || failed=1
  done
  active_worker_pids=()
  if ((failed)); then
    echo "[$variant plus] worker failure; inspect ${PLUS_OUTPUT}/worker_logs" >&2
    return 1
  fi

  python experiments/libero/summarize_results.py --output_dir="$PLUS_OUTPUT"
  python experiments/libero/summarize_libero_plus_results.py \
    --output_dir "$PLUS_OUTPUT" \
    --classification_path "$CLASSIFICATION_PATH"
  echo "[$variant plus] complete: ${PLUS_OUTPUT}/plus_summary.json"
}

IFS=',' read -r -a requested_variants <<<"$VARIANTS"
for variant in "${requested_variants[@]}"; do
  variant="${variant//[[:space:]]/}"
  [[ -n "$variant" ]] || continue
  variant_config "$variant"
  echo "=== Variant: $variant ==="
  echo "checkpoint: $CHECKPOINT"
  ((RUN_CLEAN)) && run_clean "$variant"
  ((RUN_PLUS)) && run_plus "$variant"
done

trap - INT TERM EXIT
echo "All requested evaluations completed."
