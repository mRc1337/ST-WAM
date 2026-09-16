#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${STWAM_VENV_PATH:-${PROJECT_ROOT}/.venv}"
CLEAN_LIBERO_ROOT="${CLEAN_LIBERO_ROOT:-/home/pai/zxw/LIBERO-PRO}"
PLUS_LIBERO_ROOT="${PLUS_LIBERO_ROOT:-/home/pai/zxw/LIBERO-plus}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero}"
CLASSIFICATION_PATH="${CLASSIFICATION_PATH:-${PLUS_LIBERO_ROOT}/libero/libero/benchmark/task_classification.json}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-${PROJECT_ROOT}/runs/three_task}"
CANDIDATE_RUN_ROOT="${CANDIDATE_RUN_ROOT:-${PROJECT_ROOT}/runs/dino_weight_ablation}"
MAX_STEPS="${MAX_STEPS:-1500}"
SEED="${SEED:-42}"
EXISTING_BASELINE_RESULTS="${EXISTING_BASELINE_RESULTS:-${PROJECT_ROOT}/evaluate_results/three_task/step${MAX_STEPS}_seed${SEED}/dual}"
REUSE_BASELINE_RESULTS="${REUSE_BASELINE_RESULTS:-1}"
VARIANTS="${VARIANTS:-baseline,lambda005,lambda010,condition_only}"
EVAL_SPLITS="${EVAL_SPLITS:-clean,plus}"
CLEAN_TRIALS="${CLEAN_TRIALS:-50}"
PLUS_TRIALS="${PLUS_TRIALS:-1}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/evaluate_results/dino_weight_ablation/step${MAX_STEPS}_seed${SEED}}"
MANIFEST_DIR="$RESULT_ROOT/manifests"
DATA_CONFIG="libero_object_tasks_1_4_7_v3"
TASK_CONFIG="libero_wan5b_dino_s_aux_mot_2cam_224_1e-4"

for value in MAX_STEPS SEED CLEAN_TRIALS PLUS_TRIALS; do
  [[ "${!value}" =~ ^[1-9][0-9]*$|^0$ ]] || { echo "Invalid $value=${!value}" >&2; exit 2; }
done
test -f "$VENV_PATH/bin/activate" || { echo "Missing venv: $VENV_PATH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export LIBERO_DATA_ROOT MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="${PROJECT_ROOT}/.runtime/matplotlib"
export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_CONFIG:-${PROJECT_ROOT}/configs/egl/10_nvidia.json}"

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/status" "$MANIFEST_DIR" "$MPLCONFIGDIR"
exec 9>"$RESULT_ROOT/status/eval.lock"
flock -n 9 || { echo "Another DINO-weight evaluation pipeline is running." >&2; exit 1; }
printf 'RUNNING pid=%s started_at=%s\n' "$$" "$(date -u +%FT%TZ)" >"$RESULT_ROOT/status/eval.state"
active_pids=()
pipeline_complete=0
cleanup() {
  local status=$?
  if ((${#active_pids[@]})); then kill "${active_pids[@]}" 2>/dev/null || true; fi
  if ((pipeline_complete == 0)); then
    printf 'FAILED exit=%s stopped_at=%s\n' "$status" "$(date -u +%FT%TZ)" >"$RESULT_ROOT/status/eval.state"
  fi
}
trap cleanup EXIT

test -s "$CLASSIFICATION_PATH" || { echo "Missing classification: $CLASSIFICATION_PATH" >&2; exit 1; }
test -d "$CLEAN_LIBERO_ROOT/libero/libero/assets" || { echo "Invalid clean LIBERO root: $CLEAN_LIBERO_ROOT" >&2; exit 1; }
test -d "$PLUS_LIBERO_ROOT/libero/libero/assets/new_objects" || { echo "Invalid Plus LIBERO assets: $PLUS_LIBERO_ROOT" >&2; exit 1; }
test -s "$__EGL_VENDOR_LIBRARY_FILENAMES" || { echo "Missing EGL vendor config: $__EGL_VENDOR_LIBRARY_FILENAMES" >&2; exit 1; }
python - <<'PY'
from mujoco.egl import egl_ext as EGL
devices = EGL.eglQueryDevicesEXT()
if len(devices) < 4:
    raise RuntimeError(f"EGL preflight found {len(devices)} device(s), expected at least 4")
print(f"EGL preflight: {len(devices)} rendering devices")
PY

STATS_PATH="${STATS_PATH:-${BASELINE_RUN_ROOT}/dino_future_seed${SEED}_${MAX_STEPS}/dataset_stats.json}"
test -s "$STATS_PATH" || { echo "Missing shared dataset stats: $STATS_PATH" >&2; exit 1; }

python - "$CLASSIFICATION_PATH" "$MANIFEST_DIR" <<'PY'
import json
import sys
from pathlib import Path

classification_path, output_dir = map(Path, sys.argv[1:])
output_dir.mkdir(parents=True, exist_ok=True)
prefixes = [
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
]
entries = json.loads(classification_path.read_text(encoding="utf-8"))["libero_object"]
selected = [x for x in entries if any(x["name"].startswith(p) for p in prefixes)]
counts = {p: sum(x["name"].startswith(p) for x in selected) for p in prefixes}
if any(v == 0 for v in counts.values()):
    raise RuntimeError(f"At least one selected task has no LIBERO-Plus cases: {counts}")
manifest = [{"suite": "libero_object", "task_id": int(x["id"]) - 1}
            for x in sorted(selected, key=lambda x: int(x["id"]))]
(output_dir / "full.json").write_text(json.dumps(manifest, indent=2) + "\n")
for gpu in range(4):
    (output_dir / f"gpu{gpu}.json").write_text(json.dumps(manifest[gpu::4], indent=2) + "\n")
clean = [{"suite": "libero_object", "task_id": x} for x in (3, 1, 9)]
(output_dir / "clean.json").write_text(json.dumps(clean, indent=2) + "\n")
(output_dir / "selection.json").write_text(
    json.dumps({"prefix_counts": counts, "total_plus_cases": len(manifest)}, indent=2) + "\n"
)
print(f"Manifest ready: {len(manifest)} Plus cases; per task={counts}")
PY

variant_fields() {
  MODEL_OVERRIDES=()
  case "$1" in
    baseline)
      RUN_DIR="$BASELINE_RUN_ROOT/dual_space_seed${SEED}_${MAX_STEPS}"
      MODEL_OVERRIDES=(+model.dino_future_mode=predict model.loss.lambda_dino=0.02)
      ;;
    lambda005)
      RUN_DIR="$CANDIDATE_RUN_ROOT/dual_dino_lambda005_seed${SEED}_${MAX_STEPS}"
      MODEL_OVERRIDES=(+model.dino_future_mode=predict model.loss.lambda_dino=0.05)
      ;;
    lambda010)
      RUN_DIR="$CANDIDATE_RUN_ROOT/dual_dino_lambda010_seed${SEED}_${MAX_STEPS}"
      MODEL_OVERRIDES=(+model.dino_future_mode=predict model.loss.lambda_dino=0.10)
      ;;
    condition_only)
      RUN_DIR="$CANDIDATE_RUN_ROOT/dual_dino_condition_only_seed${SEED}_${MAX_STEPS}"
      MODEL_OVERRIDES=(+model.dino_future_mode=condition_only model.loss.lambda_dino=0.0)
      ;;
    *) echo "Unknown variant: $1" >&2; return 2 ;;
  esac
  printf -v CHECKPOINT '%s/checkpoints/weights/step_%06d.pt' "$RUN_DIR" "$MAX_STEPS"
  test -s "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT" >&2; return 1; }
  CLEAN_OUTPUT="$RESULT_ROOT/$1/clean"
  PLUS_OUTPUT="$RESULT_ROOT/$1/plus"
}

wants_split() { [[ ",${EVAL_SPLITS}," == *",$1,"* ]]; }

reuse_baseline_split() {
  local split="$1" source_path="$EXISTING_BASELINE_RESULTS/$1" target_path="$RESULT_ROOT/baseline/$1" marker
  [[ "$REUSE_BASELINE_RESULTS" == "1" ]] || return 1
  if [[ "$split" == "clean" ]]; then
    marker="summary.json"
  else
    marker="plus_summary.json"
  fi
  test -s "$source_path/$marker" || return 1
  if [[ ! -e "$target_path" && ! -L "$target_path" ]]; then
    mkdir -p "$(dirname "$target_path")"
    ln -s "$source_path" "$target_path"
  fi
  test -s "$target_path/$marker" || return 1
  echo "[baseline $split] reusing completed results: $source_path"
  return 0
}

run_clean() {
  local variant="$1"
  export LIBERO_ROOT="$CLEAN_LIBERO_ROOT"
  export LIBERO_CONFIG_PATH="$CLEAN_LIBERO_ROOT/.libero"
  export PYTHONPATH="$CLEAN_LIBERO_ROOT:${PROJECT_ROOT}/src:${PROJECT_ROOT}"
  export CUDA_VISIBLE_DEVICES=0
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  mkdir -p "$CLEAN_OUTPUT"
  echo "[$variant clean] three tasks x $CLEAN_TRIALS rollouts"
  python experiments/libero/eval_libero_worker.py \
    --config-name sim_libero_v3 task="$TASK_CONFIG" data="$DATA_CONFIG" ckpt="$CHECKPOINT" \
    "${MODEL_OVERRIDES[@]}" \
    gpu_id=0 EVALUATION.device=cuda:0 EVALUATION.dataset_stats_path="$STATS_PATH" \
    EVALUATION.num_trials="$CLEAN_TRIALS" EVALUATION.output_dir="$CLEAN_OUTPUT" \
    +EVALUATION.task_manifest="$MANIFEST_DIR/clean.json" \
    +EVALUATION.skip_existing=true +EVALUATION.continue_on_error=false seed="$SEED" \
    2>&1 | tee -a "$RESULT_ROOT/logs/${variant}_clean.log"
  python experiments/libero/summarize_results.py --output_dir="$CLEAN_OUTPUT"
}

run_plus() {
  local variant="$1" failed=0 gpu
  export LIBERO_ROOT="$PLUS_LIBERO_ROOT" LIBERO_PLUS_ROOT="$PLUS_LIBERO_ROOT"
  export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.runtime/libero"
  export PYTHONPATH="$PLUS_LIBERO_ROOT:${PROJECT_ROOT}/src:${PROJECT_ROOT}"
  unset CUDA_VISIBLE_DEVICES
  mkdir -p "$PLUS_OUTPUT/worker_logs"
  active_pids=()
  echo "[$variant plus] four workers; $PLUS_TRIALS rollout(s) per case"
  for gpu in 0 1 2 3; do
    MUJOCO_EGL_DEVICE_ID=0 EGL_DEVICE_ID=0 \
    python experiments/libero/eval_libero_worker.py \
      --config-name sim_libero_v3 task="$TASK_CONFIG" data="$DATA_CONFIG" ckpt="$CHECKPOINT" \
      "${MODEL_OVERRIDES[@]}" \
      gpu_id="$gpu" EVALUATION.device="cuda:$gpu" EVALUATION.dataset_stats_path="$STATS_PATH" \
      EVALUATION.num_trials="$PLUS_TRIALS" EVALUATION.output_dir="$PLUS_OUTPUT" \
      +EVALUATION.task_manifest="$MANIFEST_DIR/gpu${gpu}.json" \
      +EVALUATION.skip_existing=true +EVALUATION.continue_on_error=false seed="$SEED" \
      >"$PLUS_OUTPUT/worker_logs/gpu${gpu}.log" 2>&1 &
    active_pids+=("$!")
  done
  for pid in "${active_pids[@]}"; do wait "$pid" || failed=1; done
  active_pids=()
  ((failed == 0)) || { echo "[$variant plus] failed; inspect $PLUS_OUTPUT/worker_logs" >&2; return 1; }
  python experiments/libero/summarize_results.py --output_dir="$PLUS_OUTPUT"
  python experiments/libero/summarize_libero_plus_results.py \
    --output_dir "$PLUS_OUTPUT" --classification_path "$CLASSIFICATION_PATH"
}

IFS=',' read -r -a requested_variants <<<"$VARIANTS"
for variant in "${requested_variants[@]}"; do
  variant="${variant//[[:space:]]/}"
  [[ -n "$variant" ]] || continue
  variant_fields "$variant"
  if wants_split clean; then
    if [[ "$variant" != "baseline" ]] || ! reuse_baseline_split clean; then
      run_clean "$variant"
    fi
  fi
  if wants_split plus; then
    if [[ "$variant" != "baseline" ]] || ! reuse_baseline_split plus; then
      run_plus "$variant"
    fi
  fi
done

python - "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = {
    "baseline": "predict, lambda=0.02",
    "lambda005": "predict, lambda=0.05",
    "lambda010": "predict, lambda=0.10",
    "condition_only": "condition_only, lambda=0",
}
lines = [
    "# DINO auxiliary-weight ablation", "",
    "| Variant | Clean | Plus |", "| --- | ---: | ---: |",
]
for variant, label in labels.items():
    base = root / variant
    clean_files = sorted((base / "clean" / "libero_object").glob("gpu*_task*_results.json"))
    clean = [json.loads(p.read_text()) for p in clean_files]
    clean_n = sum(int(x["total_episodes"]) for x in clean)
    clean_s = sum(int(x["successes"]) for x in clean)
    clean_text = f"{clean_s}/{clean_n} ({100*clean_s/clean_n:.2f}%)" if clean_n else "N/A"
    plus_path = base / "plus" / "plus_summary.json"
    plus_text = "N/A"
    if plus_path.exists():
        total = json.loads(plus_path.read_text())["leaderboard"]["Total"]
        ps, pn = int(total["Successes"]), int(total["Episodes"])
        plus_text = f"{ps}/{pn} ({100*ps/pn:.2f}%)" if pn else "N/A"
    lines.append(f"| {label} | {clean_text} | {plus_text} |")
(root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(root / "comparison.md")
PY

printf 'COMPLETE finished_at=%s\n' "$(date -u +%FT%TZ)" >"$RESULT_ROOT/status/eval.state"
pipeline_complete=1
trap - EXIT
echo "DINO-weight evaluation complete: $RESULT_ROOT/comparison.md"
