#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${STWAM_VENV_PATH:-${PROJECT_ROOT}/.venv}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/data/embodied_datasets/ST-WAM/checkpoints/three_task}"
MAX_STEPS="${MAX_STEPS:-1500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
EVAL_EVERY="${EVAL_EVERY:-250}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-dino,vae,dual}"
WANDB_MODE="${WANDB_MODE:-online}"
DATA_CONFIG="libero_object_tasks_1_4_7_v3"
RUN_ROOT="${PROJECT_ROOT}/runs/three_task"

[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MAX_STEPS=$MAX_STEPS" >&2; exit 2; }
[[ "$SAVE_EVERY" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid SAVE_EVERY=$SAVE_EVERY" >&2; exit 2; }
[[ "$EVAL_EVERY" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid EVAL_EVERY=$EVAL_EVERY" >&2; exit 2; }
test -f "$VENV_PATH/bin/activate" || { echo "Missing venv: $VENV_PATH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export LIBERO_DATA_ROOT CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status" "$CHECKPOINT_ROOT"
exec 9>"$RUN_ROOT/status/train.lock"
flock -n 9 || { echo "Another three-task training pipeline is already running." >&2; exit 1; }
printf 'RUNNING pid=%s started_at=%s\n' "$$" "$(date -u +%FT%TZ)" >"$RUN_ROOT/status/train.state"
pipeline_complete=0
finish_state() {
  local status=$?
  if ((pipeline_complete == 0)); then
    printf 'FAILED exit=%s stopped_at=%s\n' "$status" "$(date -u +%FT%TZ)" >"$RUN_ROOT/status/train.state"
  fi
}
trap finish_state EXIT

test -s "configs/data/${DATA_CONFIG}.yaml" || { echo "Missing data config" >&2; exit 1; }
test -s "$LIBERO_DATA_ROOT/libero_object/meta/info.json" || { echo "Invalid LIBERO_DATA_ROOT=$LIBERO_DATA_ROOT" >&2; exit 1; }
test -s checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt || { echo "Missing ActionDiT initialization" >&2; exit 1; }
test -s checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt || { echo "Missing DinoVideoDiT initialization" >&2; exit 1; }
test -s checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors || { echo "Missing DINOv3 weights" >&2; exit 1; }
GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$GPU_COUNT" -lt 4 ]]; then
  echo "Four visible GPUs are required, but PyTorch sees ${GPU_COUNT}." >&2
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  echo "Run: export CUDA_VISIBLE_DEVICES=0,1,2,3" >&2
  exit 1
fi

run_name_for() {
  case "$1" in
    dino) echo "dino_future_seed${SEED}_${MAX_STEPS}" ;;
    vae) echo "fastwam_vae_seed${SEED}_${MAX_STEPS}" ;;
    dual) echo "dual_space_seed${SEED}_${MAX_STEPS}" ;;
    *) echo "Unknown variant: $1" >&2; return 2 ;;
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
    echo "Refusing to replace existing non-symlink checkpoint path: $link" >&2
    return 1
  else
    ln -s "$target" "$link"
  fi
}

latest_resume() {
  local run_dir="$1" latest
  latest="$(find "$run_dir/checkpoints/state" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$latest" ]]; then
    printf '%s\n' "$latest"
  fi
  return 0
}

run_variant() {
  local variant="$1" run_name run_dir final_weight resume_path stats
  local task_config batch accum
  local -a loss_args stats_args resume_args
  run_name="$(run_name_for "$variant")"
  run_dir="$RUN_ROOT/$run_name"
  prepare_checkpoint_link "$run_dir" "$run_name"
  printf -v final_weight '%s/checkpoints/weights/step_%06d.pt' "$run_dir" "$MAX_STEPS"
  if [[ -s "$final_weight" ]]; then
    echo "[$variant] skip: final checkpoint already exists: $final_weight"
    return
  fi

  stats="$RUN_ROOT/dino_future_seed${SEED}_${MAX_STEPS}/dataset_stats.json"
  loss_args=()
  stats_args=()
  case "$variant" in
    dino)
      task_config=libero_dino_s_smallvideo_2cam_224_1e-4
      batch=4; accum=8
      loss_args=(model.loss.lambda_video=1.0 model.loss.lambda_action=1.0)
      ;;
    vae)
      task_config=libero_uncond_2cam224_1e-4
      batch=8; accum=4
      test -s "$stats" || { echo "DINO stats do not exist yet: $stats" >&2; return 1; }
      stats_args=(+data.train.pretrained_norm_stats="$stats")
      loss_args=(model.loss.lambda_action=1.0)
      ;;
    dual)
      task_config=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4
      batch=2; accum=16
      test -s "$stats" || { echo "DINO stats do not exist yet: $stats" >&2; return 1; }
      stats_args=(+data.train.pretrained_norm_stats="$stats")
      loss_args=(model.loss.lambda_video=1.0 model.loss.lambda_dino=0.02 model.loss.lambda_action=1.0)
      ;;
  esac

  resume_args=()
  resume_path="$(latest_resume "$run_dir")"
  if [[ -n "$resume_path" ]]; then
    echo "[$variant] resume full state: $resume_path"
    resume_args=(resume="$resume_path")
  else
    echo "[$variant] start from pretrained initialization"
  fi

  bash scripts/train_zero1.sh 4 \
    task="$task_config" \
    data="$DATA_CONFIG" \
    output_dir="$run_dir" \
    "${stats_args[@]}" \
    batch_size="$batch" gradient_accumulation_steps="$accum" \
    learning_rate=1e-4 max_steps="$MAX_STEPS" \
    "${loss_args[@]}" \
    save_every="$SAVE_EVERY" eval_every="$EVAL_EVERY" seed="$SEED" \
    wandb.enabled=true wandb.mode="$WANDB_MODE" \
    wandb.project=st-wam-libero-plus \
    wandb.group="three-task-${MAX_STEPS}step-ablation" \
    wandb.name="$run_name" \
    "${resume_args[@]}" \
    2>&1 | tee -a "$RUN_ROOT/logs/${run_name}.log"

  test -s "$final_weight" || { echo "[$variant] training returned without final checkpoint: $final_weight" >&2; return 1; }
  [[ "$variant" != dino ]] || test -s "$stats" || { echo "DINO training did not create $stats" >&2; return 1; }
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
echo "All requested three-task training runs completed."
