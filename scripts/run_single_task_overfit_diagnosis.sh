#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${STWAM_VENV_PATH:-${PROJECT_ROOT}/.venv}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/evaluate_results/single_task}"
DIAG_ROOT="${DIAG_ROOT:-${RESULT_ROOT}/overfit_diagnosis_seed42}"
STATS_PATH="${STATS_PATH:-${PROJECT_ROOT}/runs/single_task/dino_future_seed42_500/dataset_stats.json}"
CLASSIFICATION_PATH="${CLASSIFICATION_PATH:-/home/pai/zxw/LIBERO-plus/libero/libero/benchmark/task_classification.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
ACTION_SAMPLES_PER_EPISODE=10
MAX_LOSS_SAMPLES=0
RUN_OPEN_LOOP=1
RUN_CLEAN=1
RUN_PLUS=1
PREFLIGHT_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_single_task_overfit_diagnosis.sh [options]

Runs a restart-safe evaluation pipeline:
  A. deterministic held-out loss on all 1,559 validation windows and action
     L1/L2 on a fixed subset, for 3 variants x steps 300/400/500;
  B. step-400 clean evaluation, 50 rollouts per variant;
  C. step-400 LIBERO-Plus evaluation on the fixed 140-case manifest;
  D. paired VAE/Dual summary with McNemar and bootstrap interval.

Options:
  --action-samples-per-episode N  Action-inference subset per val episode (default: 10)
  --max-loss-samples N           0 means all val windows (default: 0)
  --skip-open-loop               Skip Phase A
  --skip-clean                   Skip Phase B
  --skip-plus                    Skip Phase C and final paired summary
  --preflight-only               Validate inputs without waiting or evaluating
  -h, --help                     Show this help

Environment:
  WAIT_FOR_GPUS=1 waits indefinitely for GPUs 0,1,2,3 to be idle (default).
  Set WAIT_FOR_GPUS=0 only when those GPUs are already reserved for this run.
EOF
}

while (($#)); do
  case "$1" in
    --action-samples-per-episode)
      ACTION_SAMPLES_PER_EPISODE="${2:?missing value}"
      shift 2
      ;;
    --max-loss-samples)
      MAX_LOSS_SAMPLES="${2:?missing value}"
      shift 2
      ;;
    --skip-open-loop)
      RUN_OPEN_LOOP=0
      shift
      ;;
    --skip-clean)
      RUN_CLEAN=0
      shift
      ;;
    --skip-plus)
      RUN_PLUS=0
      shift
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
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

[[ "$ACTION_SAMPLES_PER_EPISODE" =~ ^[0-9]+$ ]] || { echo "Invalid action sample count" >&2; exit 2; }
[[ "$MAX_LOSS_SAMPLES" =~ ^[0-9]+$ ]] || { echo "Invalid loss sample count" >&2; exit 2; }
test -f "$VENV_PATH/bin/activate" || { echo "Missing venv: $VENV_PATH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.runtime/matplotlib}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$DIAG_ROOT/logs" "$DIAG_ROOT/status" "$MPLCONFIGDIR"

STATE_FILE="$DIAG_ROOT/status/pipeline.state"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'RUNNING started_at=%s pid=%s\n' "$STARTED_AT" "$$" >"$STATE_FILE"

active_pids=()
pipeline_complete=0
cleanup() {
  local status=$?
  if ((${#active_pids[@]})); then
    kill "${active_pids[@]}" 2>/dev/null || true
  fi
  if ((pipeline_complete == 0)); then
    printf 'FAILED exit=%s started_at=%s stopped_at=%s\n' \
      "$status" "$STARTED_AT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATE_FILE"
  fi
}
trap cleanup EXIT INT TERM

echo "[$(date -u +%FT%TZ)] preflight"
test -s "$STATS_PATH"
test -s "$CLASSIFICATION_PATH"
test -s "$RESULT_ROOT/manifests/bbq_plus_140_seed42/full.json"
for variant_step in \
  dino_future_seed42_500 \
  fastwam_vae_seed42_500 \
  dual_space_seed42_500; do
  test -s "runs/single_task/${variant_step}/config.yaml"
  for step in 300 400 500; do
    printf -v padded '%06d' "$step"
    test -s "runs/single_task/${variant_step}/checkpoints/weights/step_${padded}.pt"
  done
done

if ((PREFLIGHT_ONLY)); then
  pipeline_complete=1
  printf 'PREFLIGHT_OK checked_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATE_FILE"
  trap - EXIT INT TERM
  echo "[$(date -u +%FT%TZ)] preflight complete"
  exit 0
fi

if [[ "$WAIT_FOR_GPUS" == "1" ]]; then
  echo "[$(date -u +%FT%TZ)] waiting for GPUs ${GPU_IDS}"
  python scripts/wait_for_gpus.py --count 4 --visible "$GPU_IDS"
fi

variant_fields() {
  case "$1" in
    dino)
      RUN_DIR="dino_future_seed42_500"
      ;;
    vae)
      RUN_DIR="fastwam_vae_seed42_500"
      ;;
    dual)
      RUN_DIR="dual_space_seed42_500"
      ;;
    *)
      return 1
      ;;
  esac
}

run_open_loop_step() {
  local step="$1"
  local variants=(dino vae dual)
  local gpus=(0 1 2)
  local failed=0
  local i variant gpu output log
  printf -v padded '%06d' "$step"
  active_pids=()
  echo "[$(date -u +%FT%TZ)] Phase A: step ${step}, three variants in parallel"
  for i in "${!variants[@]}"; do
    variant="${variants[$i]}"
    gpu="${gpus[$i]}"
    variant_fields "$variant"
    output="$DIAG_ROOT/open_loop/${variant}/step_${padded}"
    log="$DIAG_ROOT/logs/open_loop_${variant}_step${step}.log"
    mkdir -p "$output"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_full_validation.py \
      --run-config "runs/single_task/${RUN_DIR}/config.yaml" \
      --checkpoint "runs/single_task/${RUN_DIR}/checkpoints/weights/step_${padded}.pt" \
      --stats "$STATS_PATH" \
      --output-dir "$output" \
      --device cuda:0 \
      --seed 42 \
      --max-loss-samples "$MAX_LOSS_SAMPLES" \
      --action-samples-per-episode "$ACTION_SAMPLES_PER_EPISODE" \
      --inference-steps 10 \
      >"$log" 2>&1 &
    active_pids+=("$!")
    echo "  ${variant}: physical GPU ${gpu}, log=${log}"
  done
  for pid in "${active_pids[@]}"; do
    wait "$pid" || failed=1
  done
  active_pids=()
  if ((failed)); then
    echo "Phase A step ${step} failed; inspect $DIAG_ROOT/logs/open_loop_*_step${step}.log" >&2
    return 1
  fi
  touch "$DIAG_ROOT/status/open_loop_step${step}.complete"
}

summarize_open_loop() {
  DIAG_ROOT="$DIAG_ROOT" python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["DIAG_ROOT"])
rows = []
for step in (300, 400, 500):
    for variant in ("dino", "vae", "dual"):
        path = root / "open_loop" / variant / f"step_{step:06d}" / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            raise RuntimeError(f"Incomplete open-loop result: {path}")
        row = {"variant": variant, "step": step}
        for key, stats in payload["global"].items():
            if isinstance(stats, dict) and "mean" in stats:
                row[key] = stats["mean"]
        rows.append(row)

csv_path = root / "open_loop_comparison.csv"
keys = ["variant", "step", "loss_total", "loss_video", "loss_dino", "loss_action", "action_l1", "action_l2"]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

md = [
    "# Held-out open-loop checkpoint comparison",
    "",
    "| Variant | Step | Total loss | Video/DINO loss | Aux DINO loss | Action loss | Action L1 | Action L2 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
def fmt(value):
    return "—" if value is None else f"{value:.6f}"
for row in rows:
    md.append(
        f"| {row['variant']} | {row['step']} | {fmt(row.get('loss_total'))} | "
        f"{fmt(row.get('loss_video'))} | {fmt(row.get('loss_dino'))} | "
        f"{fmt(row.get('loss_action'))} | {fmt(row.get('action_l1'))} | {fmt(row.get('action_l2'))} |"
    )
(root / "open_loop_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(root / "open_loop_comparison.md")
PY
}

if ((RUN_OPEN_LOOP)); then
  for step in 300 400 500; do
    run_open_loop_step "$step"
  done
  summarize_open_loop
fi

if ((RUN_CLEAN)); then
  echo "[$(date -u +%FT%TZ)] Phase B: unified step-400 clean evaluation"
  bash scripts/eval_single_task_ablation.sh \
    --variants dino,vae,dual \
    --step 400 \
    --result-tag step400_seed42 \
    --clean-trials 50 \
    --clean-only \
    2>&1 | tee "$DIAG_ROOT/logs/phase_b_clean.log"
  touch "$DIAG_ROOT/status/phase_b_clean.complete"
fi

if ((RUN_PLUS)); then
  echo "[$(date -u +%FT%TZ)] Phase C: frozen step-400 Plus evaluation"
  bash scripts/eval_single_task_ablation.sh \
    --variants dino,vae,dual \
    --step 400 \
    --result-tag step400_seed42 \
    --plus-trials 1 \
    --plus-only \
    2>&1 | tee "$DIAG_ROOT/logs/phase_c_plus.log"
  touch "$DIAG_ROOT/status/phase_c_plus.complete"

  python scripts/summarize_single_task_ablation.py \
    --result-root "$RESULT_ROOT" \
    --tag step400_seed42 \
    --classification-path "$CLASSIFICATION_PATH" \
    --output "$DIAG_ROOT/step400_closed_loop_comparison.md"
fi

pipeline_complete=1
printf 'COMPLETE started_at=%s finished_at=%s\n' \
  "$STARTED_AT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATE_FILE"
trap - EXIT INT TERM
echo "[$(date -u +%FT%TZ)] all requested phases complete: $DIAG_ROOT"
