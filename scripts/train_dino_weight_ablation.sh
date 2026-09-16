#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${STWAM_VENV_PATH:-${PROJECT_ROOT}/.venv}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/data/embodied_datasets/ST-WAM/checkpoints/dino_weight_ablation}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/dino_weight_ablation}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${PROJECT_ROOT}/runs/three_task/dual_space_seed42_1500}"
MAX_STEPS="${MAX_STEPS:-1500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
EVAL_EVERY="${EVAL_EVERY:-250}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-lambda005,lambda010,condition_only}"
WANDB_MODE="${WANDB_MODE:-online}"
DATA_CONFIG="libero_object_tasks_1_4_7_v3"
TASK_CONFIG="libero_wan5b_dino_s_aux_mot_2cam_224_1e-4"

for value in MAX_STEPS SAVE_EVERY EVAL_EVERY SEED; do
  [[ "${!value}" =~ ^[1-9][0-9]*$|^0$ ]] || { echo "Invalid $value=${!value}" >&2; exit 2; }
done
test -f "$VENV_PATH/bin/activate" || { echo "Missing venv: $VENV_PATH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export LIBERO_DATA_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status" "$CHECKPOINT_ROOT"
exec 9>"$RUN_ROOT/status/train.lock"
flock -n 9 || { echo "Another DINO-weight training pipeline is running." >&2; exit 1; }
printf 'RUNNING pid=%s started_at=%s\n' "$$" "$(date -u +%FT%TZ)" >"$RUN_ROOT/status/train.state"
pipeline_complete=0
finish_state() {
  local status=$?
  if ((pipeline_complete == 0)); then
    printf 'FAILED exit=%s stopped_at=%s\n' "$status" "$(date -u +%FT%TZ)" >"$RUN_ROOT/status/train.state"
  fi
}
trap finish_state EXIT

test -s "configs/data/${DATA_CONFIG}.yaml" || { echo "Missing data config: $DATA_CONFIG" >&2; exit 1; }
test -s "$LIBERO_DATA_ROOT/libero_object/meta/info.json" || { echo "Invalid LIBERO_DATA_ROOT=$LIBERO_DATA_ROOT" >&2; exit 1; }
test -s checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt || { echo "Missing ActionDiT initialization" >&2; exit 1; }
test -s checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt || { echo "Missing DinoVideoDiT initialization" >&2; exit 1; }
test -s checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors || { echo "Missing DINOv3 weights" >&2; exit 1; }

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$GPU_COUNT" -lt 4 ]]; then
  echo "Four visible GPUs are required, but PyTorch sees ${GPU_COUNT}." >&2
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 1
fi

# Reuse exactly the normalization statistics used by the completed baseline.
STATS_PATH="${STATS_PATH:-${PROJECT_ROOT}/runs/three_task/dino_future_seed${SEED}_${MAX_STEPS}/dataset_stats.json}"
if [[ ! -s "$STATS_PATH" && -s "$BASELINE_RUN_DIR/dataset_stats.json" ]]; then
  STATS_PATH="$BASELINE_RUN_DIR/dataset_stats.json"
fi
test -s "$STATS_PATH" || {
  echo "Missing shared dataset statistics: $STATS_PATH" >&2
  echo "Run the original three-task DINO experiment first or set STATS_PATH." >&2
  exit 1
}

variant_fields() {
  case "$1" in
    lambda005)
      RUN_NAME="dual_dino_lambda005_seed${SEED}_${MAX_STEPS}"
      DINO_LAMBDA="0.05"
      DINO_MODE="predict"
      ;;
    lambda010)
      RUN_NAME="dual_dino_lambda010_seed${SEED}_${MAX_STEPS}"
      DINO_LAMBDA="0.10"
      DINO_MODE="predict"
      ;;
    condition_only)
      RUN_NAME="dual_dino_condition_only_seed${SEED}_${MAX_STEPS}"
      DINO_LAMBDA="0.0"
      DINO_MODE="condition_only"
      ;;
    *) echo "Unknown variant: $1 (expected lambda005, lambda010, condition_only)" >&2; return 2 ;;
  esac
}

prepare_checkpoint_link() {
  local run_dir="$1" run_name="$2" link target
  link="$run_dir/checkpoints"
  target="$CHECKPOINT_ROOT/$run_name"
  mkdir -p "$run_dir" "$target"
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]] || {
      echo "Checkpoint symlink points elsewhere: $link -> $(readlink "$link")" >&2
      return 1
    }
  elif [[ -e "$link" ]]; then
    echo "Refusing to replace non-symlink checkpoint path: $link" >&2
    return 1
  else
    ln -s "$target" "$link"
  fi
}

latest_resume() {
  find "$1/checkpoints/state" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null \
    | sort | tail -n 1 || true
}

run_variant() {
  local variant="$1" run_dir final_weight resume_path
  local -a resume_args=()
  variant_fields "$variant"
  run_dir="$RUN_ROOT/$RUN_NAME"
  prepare_checkpoint_link "$run_dir" "$RUN_NAME"
  printf -v final_weight '%s/checkpoints/weights/step_%06d.pt' "$run_dir" "$MAX_STEPS"
  if [[ -s "$final_weight" ]]; then
    echo "[$variant] skip: final checkpoint exists: $final_weight"
    return
  fi
  resume_path="$(latest_resume "$run_dir")"
  if [[ -n "$resume_path" ]]; then
    echo "[$variant] resuming full state: $resume_path"
    resume_args=(resume="$resume_path")
  else
    echo "[$variant] starting from the common pretrained initialization"
  fi

  bash scripts/train_zero1.sh 4 \
    task="$TASK_CONFIG" data="$DATA_CONFIG" output_dir="$run_dir" \
    +data.train.pretrained_norm_stats="$STATS_PATH" \
    batch_size=2 gradient_accumulation_steps=16 \
    learning_rate=1e-4 max_steps="$MAX_STEPS" \
    +model.dino_future_mode="$DINO_MODE" \
    model.loss.lambda_video=1.0 model.loss.lambda_dino="$DINO_LAMBDA" model.loss.lambda_action=1.0 \
    save_every="$SAVE_EVERY" eval_every="$EVAL_EVERY" seed="$SEED" \
    wandb.enabled=true wandb.mode="$WANDB_MODE" \
    wandb.project=st-wam-libero-plus \
    wandb.group="three-task-dino-weight-${MAX_STEPS}step" \
    wandb.name="$RUN_NAME" \
    "${resume_args[@]}" \
    2>&1 | tee -a "$RUN_ROOT/logs/${RUN_NAME}.log"

  test -s "$final_weight" || { echo "Training returned without final checkpoint: $final_weight" >&2; return 1; }
  echo "[$variant] complete: $final_weight"
}

IFS=',' read -r -a requested_variants <<<"$VARIANTS"
for variant in "${requested_variants[@]}"; do
  variant="${variant//[[:space:]]/}"
  [[ -n "$variant" ]] && run_variant "$variant"
done

printf 'COMPLETE finished_at=%s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/status/train.state"
pipeline_complete=1
trap - EXIT
echo "DINO-weight training pipeline complete."
