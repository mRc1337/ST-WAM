#!/usr/bin/env bash
set -euo pipefail

# Evaluate the current three-task VAE+DINO MoT checkpoint on:
#   - clean LIBERO-PRO: task ids 3, 1, 9
#   - LIBERO-Plus:      task ids 32, 122, 170
#
# Each task is run for exactly one episode in its own worker process.  This is
# intentional: MotAttentionRecorder's save counter is process-local, so a
# separate worker gives each task an independent heatmap sequence.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
VENV_PATH="${STWAM_VENV_PATH:-$PROJECT_ROOT/.venv}"

CLEAN_LIBERO_ROOT="${CLEAN_LIBERO_ROOT:-$WORKSPACE_ROOT/LIBERO-PRO}"
PLUS_LIBERO_ROOT="${PLUS_LIBERO_ROOT:-$WORKSPACE_ROOT/LIBERO-plus}"
CKPT="${CKPT:-$PROJECT_ROOT/runs/three_task/dual_space_seed42_1500/checkpoints/weights/step_001500.pt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$PROJECT_ROOT/runs/three_task/dual_space_seed42_1500/config.yaml}"
STATS_PATH="${STATS_PATH:-$PROJECT_ROOT/runs/three_task/dino_future_seed42_1500/dataset_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/evaluate_results/three_task_heatmap/dual_space_lambda002_step1500_$(date -u +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2}"
SEED="${SEED:-42}"
CLEAN_TASK_IDS="${CLEAN_TASK_IDS:-3,1,9}"
PLUS_TASK_IDS="${PLUS_TASK_IDS:-32,122,170}"
RUN_SPLITS="${RUN_SPLITS:-clean,plus}"
EXPECTED_DINO_LAMBDA="${EXPECTED_DINO_LAMBDA:-0.02}"
TASK_CONFIG="${TASK_CONFIG:-libero_wan5b_dino_s_aux_mot_2cam_224_1e-4}"
DATA_CONFIG="${DATA_CONFIG:-libero_object_tasks_1_4_7_v3}"
EGL_VENDOR_CONFIG="${EGL_VENDOR_CONFIG:-$PROJECT_ROOT/configs/egl/10_nvidia.json}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"
QWEN_PATH="${QWEN_PATH:-}"
CONFIG_NAME="${CONFIG_NAME:-sim_libero_v3}"
EPISODE_VIDEO="${EPISODE_VIDEO:-0}"
VIDEO_FPS="${VIDEO_FPS:-24}"
VIDEO_HOLD_FRAMES="${VIDEO_HOLD_FRAMES:-10}"
ATTENTION_LAYERS="${ATTENTION_LAYERS:-9,19,29}"
ATTENTION_DENOISING_STEPS="${ATTENTION_DENOISING_STEPS:-0,4,9}"
ACTION_QUERY_RANGES="${ACTION_QUERY_RANGES:-0:8,8:16,16:24,24:32}"
VIDEO_ATTENTION_LAYER="${VIDEO_ATTENTION_LAYER:-29}"
VIDEO_ATTENTION_STEP="${VIDEO_ATTENTION_STEP:-9}"

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_three_task_heatmap.sh [options]

Runs one episode for each of the three clean tasks and three selected
LIBERO-Plus cases, and saves MoT VAE/DINO attention heatmaps.

Options:
  --ckpt PATH             checkpoint (default: current step-1500 Dual-Space ckpt)
  --output-dir PATH       output directory (default: timestamped)
  --gpu-ids LIST          GPUs used in parallel batches (default: 0,1,2)
  --clean-task-ids LIST   clean LIBERO task ids (default: 3,1,9)
  --plus-task-ids LIST    zero-based LIBERO-Plus task ids (default: 32,122,170)
  --splits LIST           comma-separated splits to run: clean,plus (default: clean,plus)
  --episode-video         save the full episode heatmap comparison as MP4
  -h, --help              show this help

Environment overrides:
  CLEAN_LIBERO_ROOT PLUS_LIBERO_ROOT CKPT TRAIN_CONFIG STATS_PATH OUTPUT_ROOT
  GPU_IDS SEED EGL_VENDOR_CONFIG STWAM_VENV_PATH PYTHON_BIN CONFIG_NAME TASK_CONFIG
  DATA_CONFIG QWEN_PATH RUN_SPLITS
  EPISODE_VIDEO VIDEO_FPS VIDEO_HOLD_FRAMES ATTENTION_LAYERS
  ATTENTION_DENOISING_STEPS ACTION_QUERY_RANGES VIDEO_ATTENTION_LAYER
  VIDEO_ATTENTION_STEP

GPU_IDS are physical GPU ids.  Each worker is isolated with
CUDA_VISIBLE_DEVICES=<physical_id>, so its model device is cuda:0 and its
EGL device remains the selected physical id.

Example:
  bash scripts/eval_three_task_heatmap.sh
EOF
}

while (($#)); do
  case "$1" in
    --ckpt) CKPT="${2:?--ckpt requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="${2:?--output-dir requires a value}"; shift 2 ;;
    --gpu-ids) GPU_IDS="${2:?--gpu-ids requires a value}"; shift 2 ;;
    --clean-task-ids) CLEAN_TASK_IDS="${2:?--clean-task-ids requires a value}"; shift 2 ;;
    --plus-task-ids) PLUS_TASK_IDS="${2:?--plus-task-ids requires a value}"; shift 2 ;;
    --splits) RUN_SPLITS="${2:?--splits requires a value}"; shift 2 ;;
    --episode-video) EPISODE_VIDEO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$EPISODE_VIDEO" in
  0|1) ;;
  *) echo "EPISODE_VIDEO must be 0 or 1, got: $EPISODE_VIDEO" >&2; exit 2 ;;
esac
RUN_CLEAN=0
RUN_PLUS=0
IFS=',' read -r -a RUN_SPLIT_ARRAY <<<"$RUN_SPLITS"
for split_name in "${RUN_SPLIT_ARRAY[@]}"; do
  split_name="${split_name//[[:space:]]/}"
  case "$split_name" in
    clean) RUN_CLEAN=1 ;;
    plus) RUN_PLUS=1 ;;
    *) echo "RUN_SPLITS/--splits must contain only clean and/or plus, got: $split_name" >&2; exit 2 ;;
  esac
done
if ((RUN_CLEAN == 0 && RUN_PLUS == 0)); then
  echo "RUN_SPLITS/--splits must select at least one split" >&2
  exit 2
fi

[[ "$VIDEO_FPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "VIDEO_FPS must be a positive integer, got: $VIDEO_FPS" >&2
  exit 2
}
[[ "$VIDEO_HOLD_FRAMES" =~ ^[1-9][0-9]*$ ]] || {
  echo "VIDEO_HOLD_FRAMES must be a positive integer, got: $VIDEO_HOLD_FRAMES" >&2
  exit 2
}

ATTENTION_MAX_SAVES=1
if [[ "$EPISODE_VIDEO" == 1 ]]; then
  # LIBERO-Object has at most 43 replans with the current 10-step cadence;
  # leave headroom for configuration overrides.
  ATTENTION_MAX_SAVES=1000
fi

IFS=',' read -r -a ACTION_QUERY_RANGE_ARRAY <<<"$ACTION_QUERY_RANGES"
QUERY_RANGE_OVERRIDE="[\"${ACTION_QUERY_RANGES//,/\",\"}\"]"
[[ "$ATTENTION_LAYERS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "ATTENTION_LAYERS must be comma-separated non-negative integers, got: $ATTENTION_LAYERS" >&2
  exit 2
}
[[ "$ATTENTION_DENOISING_STEPS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "ATTENTION_DENOISING_STEPS must be comma-separated non-negative integers, got: $ATTENTION_DENOISING_STEPS" >&2
  exit 2
}
for query_range in "${ACTION_QUERY_RANGE_ARRAY[@]}"; do
  [[ "$query_range" =~ ^[0-9]+:[0-9]+$ ]] || {
    echo "ACTION_QUERY_RANGES must contain START:END entries, got: $query_range" >&2
    exit 2
  }
done

test -f "$VENV_PATH/bin/activate" || {
  echo "Missing virtual environment: $VENV_PATH" >&2
  exit 1
}
test -x "$PYTHON_BIN" || {
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
}
test -s "$CKPT" || {
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
}
test -s "$STATS_PATH" || {
  echo "Missing dataset stats: $STATS_PATH" >&2
  exit 1
}
test -s "$TRAIN_CONFIG" || {
  echo "Missing training config used for lambda validation: $TRAIN_CONFIG" >&2
  exit 1
}
test -s "$EGL_VENDOR_CONFIG" || {
  echo "Missing EGL vendor config: $EGL_VENDOR_CONFIG" >&2
  exit 1
}
if [[ -n "$QWEN_PATH" ]]; then
  test -e "$QWEN_PATH" || {
    echo "Missing Qwen path: $QWEN_PATH" >&2
    exit 1
  }
fi
if ((RUN_CLEAN == 1)); then
  test -d "$CLEAN_LIBERO_ROOT/libero/libero/assets" || {
    echo "Invalid clean LIBERO root: $CLEAN_LIBERO_ROOT" >&2
    exit 1
  }
fi
if ((RUN_PLUS == 1)); then
  test -d "$PLUS_LIBERO_ROOT/libero/libero/assets/new_objects" || {
    echo "Invalid LIBERO-Plus root/assets: $PLUS_LIBERO_ROOT" >&2
    exit 1
  }
fi

# The checkpoint itself does not store the training loss coefficient as a
# runtime option.  Check the resolved training config before evaluating it.
"$PYTHON_BIN" - "$TRAIN_CONFIG" "$EXPECTED_DINO_LAMBDA" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
expected = float(sys.argv[2])
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
actual = payload.get("model", {}).get("loss", {}).get("lambda_dino")
if actual is None:
    raise SystemExit(f"model.loss.lambda_dino is missing from {config_path}")
if abs(float(actual) - expected) > 1e-12:
    raise SystemExit(
        f"Expected model.loss.lambda_dino={expected}, "
        f"found {actual} in {config_path}"
    )
print(f"lambda validation: model.loss.lambda_dino={float(actual):g}")
PY

parse_ids() {
  local value="$1" name="$2" expected_count="$3" item
  local -a parsed=()
  IFS=',' read -r -a raw_items <<<"$value"
  for item in "${raw_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ "$item" =~ ^[0-9]+$ ]] || {
      echo "$name must contain non-negative integer ids, got: $item" >&2
      exit 2
    }
    parsed+=("$item")
  done
  if [[ "$expected_count" == "3" ]]; then
    ((${#parsed[@]} == 3)) || {
      echo "$name must contain exactly three ids, got: $value" >&2
      exit 2
    }
  elif [[ "$expected_count" == "at_least_one" ]]; then
    ((${#parsed[@]} >= 1)) || {
      echo "$name must contain at least one id, got: $value" >&2
      exit 2
    }
  else
    echo "Internal error: unsupported id-count policy $expected_count" >&2
    exit 2
  fi
  printf '%s\n' "${parsed[@]}"
}

CLEAN_IDS=()
PLUS_IDS=()
if ((RUN_CLEAN == 1)); then
  mapfile -t CLEAN_IDS < <(parse_ids "$CLEAN_TASK_IDS" CLEAN_TASK_IDS 3)
fi
if ((RUN_PLUS == 1)); then
  mapfile -t PLUS_IDS < <(parse_ids "$PLUS_TASK_IDS" PLUS_TASK_IDS 3)
fi
mapfile -t GPU_ARRAY < <(parse_ids "$GPU_IDS" GPU_IDS at_least_one)

# Validate the task mapping and write one manifest per task.  LIBERO-Plus
# classification ids are one-based; benchmark task ids are zero-based.
CLASSIFICATION_PATH=""
if ((RUN_PLUS == 1)); then
  CLASSIFICATION_PATH="$PLUS_LIBERO_ROOT/libero/libero/benchmark/task_classification.json"
  test -s "$CLASSIFICATION_PATH" || {
    echo "Missing LIBERO-Plus classification: $CLASSIFICATION_PATH" >&2
    exit 1
  }
fi

MANIFEST_ROOT="$OUTPUT_ROOT/manifests"
RESULT_ROOT="$OUTPUT_ROOT/results"
HEATMAP_ROOT="$OUTPUT_ROOT/heatmaps"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$MANIFEST_ROOT" "$RESULT_ROOT" "$HEATMAP_ROOT" "$LOG_ROOT"

"$PYTHON_BIN" - "$CLASSIFICATION_PATH" "$MANIFEST_ROOT" "${CLEAN_IDS[*]}" "${PLUS_IDS[*]}" "$RUN_CLEAN" "$RUN_PLUS" <<'PY'
import json
import sys
from pathlib import Path

classification_path, manifest_root, clean_text, plus_text, run_clean_text, run_plus_text = sys.argv[1:]
manifest_root = Path(manifest_root)
run_clean = run_clean_text == "1"
run_plus = run_plus_text == "1"
clean_ids = [int(x) for x in clean_text.split()] if run_clean else []
plus_ids = [int(x) for x in plus_text.split()] if run_plus else []

clean_names = {
    1: "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    3: "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    9: "pick_up_the_orange_juice_and_place_it_in_the_basket",
}
plus_prefixes = [
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
]

for task_id in clean_ids:
    if task_id not in clean_names:
        raise SystemExit(
            f"Unsupported clean task id {task_id}; expected one of {sorted(clean_names)}"
        )
    payload = [{"suite": "libero_object", "task_id": task_id}]
    (manifest_root / f"clean_task{task_id}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

selection = {"clean": [], "plus": []}
if run_plus:
    classification = json.loads(Path(classification_path).read_text(encoding="utf-8"))
    entries = classification.get("libero_object")
    if not isinstance(entries, list):
        raise SystemExit("task_classification.json has no libero_object list")
    by_id = {int(item["id"]): item for item in entries if isinstance(item, dict) and "id" in item}
    for index, task_id in enumerate(plus_ids):
        entry = by_id.get(task_id + 1)
        expected_prefix = plus_prefixes[index]
        if entry is None:
            raise SystemExit(
                f"Missing Plus task id {task_id} in {classification_path}"
            )
        name = str(entry.get("name", ""))
        if not name.startswith(expected_prefix):
            raise SystemExit(
                f"Plus task id {task_id} maps to {name!r}, expected prefix {expected_prefix!r}"
            )
        payload = [{"suite": "libero_object", "task_id": task_id}]
        (manifest_root / f"plus_task{task_id}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        selection["plus"].append(
            {
                "task_id": task_id,
                "classification_id": task_id + 1,
                "name": name,
                "category": entry.get("category"),
                "difficulty_level": entry.get("difficulty_level"),
            }
        )

for task_id in clean_ids:
    selection["clean"].append(
        {"task_id": task_id, "name": clean_names[task_id]}
    )
(manifest_root / "selection.json").write_text(
    json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(selection, indent=2, ensure_ascii=False))
PY

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_CONFIG"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="$PROJECT_ROOT/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"

run_one() {
  local split="$1" task_id="$2" gpu_id="$3" root="$4" config_path="$5"
  local manifest="$MANIFEST_ROOT/${split}_task${task_id}.json"
  local split_result_root="$RESULT_ROOT/$split"
  local task_heatmap_root="$HEATMAP_ROOT/$split/task${task_id}"
  local log_path="$LOG_ROOT/${split}_task${task_id}_gpu${gpu_id}.log"
  local -a extra_overrides=()
  if [[ -n "$QWEN_PATH" ]]; then
    extra_overrides=(
      "model.semantic_history_config.vlm_model_name_or_path=$QWEN_PATH"
      "+model.semantic_history_config.allow_vlm_path_relocation=true"
    )
  fi
  mkdir -p "$split_result_root" "$task_heatmap_root"

  echo "[$split task=$task_id gpu=$gpu_id] starting one episode"
  (
    export LIBERO_ROOT="$root"
    export LIBERO_CONFIG_PATH="$config_path"
    export PYTHONPATH="$root:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    # GPU_IDS are physical ids.  Restricting each worker to one physical GPU
    # makes CUDA's logical device (cuda:0) and robosuite's EGL selection
    # unambiguous, even when the parent shell already exported a different
    # CUDA_VISIBLE_DEVICES value.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export MUJOCO_EGL_DEVICE_ID="$gpu_id"
    export EGL_DEVICE_ID="$gpu_id"
    "$PYTHON_BIN" experiments/libero/eval_libero_worker.py \
      --config-name "$CONFIG_NAME" \
      "task=$TASK_CONFIG" \
      "data=$DATA_CONFIG" \
      "ckpt=$CKPT" \
      "gpu_id=$gpu_id" \
      "EVALUATION.device=cuda:0" \
      "EVALUATION.dataset_stats_path=$STATS_PATH" \
      "EVALUATION.num_trials=1" \
      "EVALUATION.output_dir=$split_result_root" \
      "+EVALUATION.task_manifest=$manifest" \
      "+EVALUATION.skip_existing=false" \
      "+EVALUATION.continue_on_error=false" \
      "EVALUATION.mot_attention_visualization.enabled=true" \
      "EVALUATION.mot_attention_visualization.output_dir=$task_heatmap_root" \
      "EVALUATION.mot_attention_visualization.layers=[$ATTENTION_LAYERS]" \
      "EVALUATION.mot_attention_visualization.denoising_steps=[$ATTENTION_DENOISING_STEPS]" \
      "EVALUATION.mot_attention_visualization.heads=all" \
      "EVALUATION.mot_attention_visualization.action_query_ranges=$QUERY_RANGE_OVERRIDE" \
      "EVALUATION.mot_attention_visualization.call_indices=all" \
      "EVALUATION.mot_attention_visualization.max_saves=$ATTENTION_MAX_SAVES" \
      "EVALUATION.mot_attention_visualization.display_normalization=percentile" \
      "EVALUATION.mot_attention_visualization.overlay_alpha=0.45" \
      "model.loss.lambda_dino=$EXPECTED_DINO_LAMBDA" \
      "${extra_overrides[@]}" \
      "seed=$SEED" \
      2>&1 | tee "$log_path"
  )
}

run_split() {
  local split="$1" root="$2" config_path="$3"
  shift 3
  local -a task_ids=("$@")
  local -a active_pids=()
  local -a active_labels=()
  local index task_id gpu_id slot
  local failed=0

  # Run at most one model per requested GPU.  With GPU_IDS=0, this naturally
  # falls back to sequential execution without risking an immediate OOM.
  for index in "${!task_ids[@]}"; do
    task_id="${task_ids[index]}"
    gpu_id="${GPU_ARRAY[index % ${#GPU_ARRAY[@]}]}"
    run_one "$split" "$task_id" "$gpu_id" "$root" "$config_path" &
    active_pids+=("$!")
    active_labels+=("$split/task$task_id")

    if ((${#active_pids[@]} == ${#GPU_ARRAY[@]} || index == ${#task_ids[@]} - 1)); then
      for slot in "${!active_pids[@]}"; do
        if ! wait "${active_pids[slot]}"; then
          echo "[${active_labels[slot]}] failed; inspect $LOG_ROOT" >&2
          failed=1
        fi
      done
      active_pids=()
      active_labels=()
      ((failed == 0)) || return 1
    fi
  done
}

echo "checkpoint: $CKPT"
echo "dataset stats: $STATS_PATH"
echo "output root: $OUTPUT_ROOT"
echo "GPU ids: ${GPU_ARRAY[*]}"
echo "splits: $RUN_SPLITS"

if ((RUN_CLEAN == 1)); then
  run_split clean "$CLEAN_LIBERO_ROOT" "$CLEAN_LIBERO_ROOT/.libero" "${CLEAN_IDS[@]}"
fi
if ((RUN_PLUS == 1)); then
  run_split plus "$PLUS_LIBERO_ROOT" "$PROJECT_ROOT/.runtime/libero" "${PLUS_IDS[@]}"
fi

IFS=',' read -r -a ATTENTION_LAYER_ARRAY <<<"$ATTENTION_LAYERS"
IFS=',' read -r -a ATTENTION_STEP_ARRAY <<<"$ATTENTION_DENOISING_STEPS"
expected_task_count=0
if ((RUN_CLEAN == 1)); then
  expected_task_count=$((expected_task_count + ${#CLEAN_IDS[@]}))
fi
if ((RUN_PLUS == 1)); then
  expected_task_count=$((expected_task_count + ${#PLUS_IDS[@]}))
fi
expected_records=$((expected_task_count * ${#ATTENTION_LAYER_ARRAY[@]} * ${#ATTENTION_STEP_ARRAY[@]}))
actual_records="$(find "$HEATMAP_ROOT" -type f -name metadata.json | wc -l)"
if [[ "$EPISODE_VIDEO" == 0 ]]; then
  if [[ "$actual_records" != "$expected_records" ]]; then
    echo "Expected $expected_records heatmap records, found $actual_records." >&2
    exit 1
  fi
else
  if ((RUN_CLEAN == 1)); then
    for task_id in "${CLEAN_IDS[@]}"; do
      count="$(find "$HEATMAP_ROOT/clean/task${task_id}" -type f -name metadata.json | wc -l)"
      [[ "$count" -gt 0 ]] || { echo "Missing clean attention sequence for task $task_id" >&2; exit 1; }
    done
  fi
  if ((RUN_PLUS == 1)); then
    for task_id in "${PLUS_IDS[@]}"; do
      count="$(find "$HEATMAP_ROOT/plus/task${task_id}" -type f -name metadata.json | wc -l)"
      [[ "$count" -gt 0 ]] || { echo "Missing Plus attention sequence for task $task_id" >&2; exit 1; }
    done
  fi
fi

if ((RUN_CLEAN == 1)); then
  for task_id in "${CLEAN_IDS[@]}"; do
    result_files=("$RESULT_ROOT/clean/libero_object"/gpu*_task${task_id}_results.json)
    [[ -s "${result_files[0]}" ]] || {
      echo "Missing clean result for task $task_id under $RESULT_ROOT/clean" >&2
      exit 1
    }
  done
fi
if ((RUN_PLUS == 1)); then
  for task_id in "${PLUS_IDS[@]}"; do
    result_files=("$RESULT_ROOT/plus/libero_object"/gpu*_task${task_id}_results.json)
    [[ -s "${result_files[0]}" ]] || {
      echo "Missing Plus result for task $task_id under $RESULT_ROOT/plus" >&2
      exit 1
    }
  done
fi

if [[ "$EPISODE_VIDEO" == 1 ]]; then
  if ((RUN_CLEAN == 1)); then
    for task_id in "${CLEAN_IDS[@]}"; do
      for query_range in "${ACTION_QUERY_RANGE_ARRAY[@]}"; do
        query_tag="${query_range/:/_}"
        "$PYTHON_BIN" scripts/make_mot_attention_video.py \
          --attention-dir "$HEATMAP_ROOT/clean/task${task_id}" \
          --output "$HEATMAP_ROOT/clean/task${task_id}/heatmap_video_q${query_tag}.mp4" \
          --layer "$VIDEO_ATTENTION_LAYER" \
          --denoising-step "$VIDEO_ATTENTION_STEP" \
          --query-range "$query_range" \
          --fps "$VIDEO_FPS" \
          --hold-frames "$VIDEO_HOLD_FRAMES"
      done
    done
  fi
  if ((RUN_PLUS == 1)); then
    for task_id in "${PLUS_IDS[@]}"; do
      for query_range in "${ACTION_QUERY_RANGE_ARRAY[@]}"; do
        query_tag="${query_range/:/_}"
        "$PYTHON_BIN" scripts/make_mot_attention_video.py \
          --attention-dir "$HEATMAP_ROOT/plus/task${task_id}" \
          --output "$HEATMAP_ROOT/plus/task${task_id}/heatmap_video_q${query_tag}.mp4" \
          --layer "$VIDEO_ATTENTION_LAYER" \
          --denoising-step "$VIDEO_ATTENTION_STEP" \
          --query-range "$query_range" \
          --fps "$VIDEO_FPS" \
          --hold-frames "$VIDEO_HOLD_FRAMES"
      done
    done
  fi
fi

echo
if [[ "$EPISODE_VIDEO" == 1 ]]; then
  echo "Completed $expected_task_count one-episode evaluations. Episode heatmap videos:"
  find "$HEATMAP_ROOT" -type f -name 'heatmap_video_q*.mp4' -print | sort
else
  echo "Completed $expected_task_count one-episode evaluations. Heatmaps:"
  find "$HEATMAP_ROOT" -type f -name metadata.json -print | sort
fi
echo
echo "Selection manifest: $MANIFEST_ROOT/selection.json"
echo "Result JSON root:  $RESULT_ROOT"
