#!/usr/bin/env bash
set -euo pipefail

# One-click evaluation of the three-task LIBERO-Object ablation:
#   dino_only (DINO future), vae_only (VAE future), and dino+vae (Dual-Space).
# The LeRobot task indices [1, 4, 7] map to clean LIBERO ids [3, 1, 9].
# LIBERO-Plus cases are selected by task name from task_classification.json.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
VENV_PATH="${STWAM_VENV_PATH:-$PROJECT_ROOT/.venv}"
CLEAN_LIBERO_ROOT="${CLEAN_LIBERO_ROOT:-$WORKSPACE_ROOT/LIBERO-PRO}"
PRO_LIBERO_ROOT="${PRO_LIBERO_ROOT:-$CLEAN_LIBERO_ROOT}"
PLUS_LIBERO_ROOT="${PLUS_LIBERO_ROOT:-${LIBERO_PLUS_ROOT:-$WORKSPACE_ROOT/LIBERO-plus}}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero}"
CLASSIFICATION_PATH="${CLASSIFICATION_PATH:-$PLUS_LIBERO_ROOT/libero/libero/benchmark/task_classification.json}"
EGL_VENDOR_CONFIG="${EGL_VENDOR_CONFIG:-$PROJECT_ROOT/configs/egl/10_nvidia.json}"
PRO_PREPARE_SCRIPT="${PRO_PREPARE_SCRIPT:-$PROJECT_ROOT/scripts/prepare_libero_pro_runtime.py}"

MAX_STEPS="${MAX_STEPS:-1500}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-dino,vae,dual}"
EVAL_SPLITS="${EVAL_SPLITS:-clean,plus}"
CLEAN_TRIALS="${CLEAN_TRIALS:-50}"
PLUS_TRIALS="${PLUS_TRIALS:-1}"
PRO_TRIALS="${PRO_TRIALS:-50}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
CLEAN_GPU_ID="${CLEAN_GPU_ID:-0}"
PRO_GPU_ID="${PRO_GPU_ID:-0}"
PRO_PERTURBATIONS="${PRO_PERTURBATIONS:-env,swap,object,language,task}"
EXPECTED_PLUS_CASES="${EXPECTED_PLUS_CASES:-745}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/three_task}"
RESULT_ROOT="${RESULT_ROOT:-}"
STATS_PATH="${STATS_PATH:-}"
MANIFEST_DIR="${MANIFEST_DIR:-}"
PRO_RUNTIME_ROOT="${PRO_RUNTIME_ROOT:-}"
DATA_CONFIG=libero_object_tasks_1_4_7_v3
SKIP_EXISTING="${SKIP_EXISTING:-1}"

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_three_task_ablation.sh [options]

Options:
  --variants LIST       dino,vae,dual; aliases dino_only,vae_only,dino+vae
  --step N              checkpoint step (default: 1500)
  --seed N              seed (default: 42)
  --clean-trials N      rollouts per clean task (default: 50)
  --plus-trials N       rollouts per Plus case (default: 1)
  --pro-trials N        rollouts per selected LIBERO-PRO task (default: 50)
  --pro-perturbations L env,swap,object,language,task (default: all five)
  --pro-gpu-id N        logical CUDA/EGL id for the PRO worker (default: 0)
  --splits LIST         clean,plus,pro (default: clean,plus)
  --clean-only          evaluate only clean LIBERO-PRO
  --plus-only           evaluate only LIBERO-Plus
  --pro-only            evaluate only LIBERO-PRO perturbations
  --gpu-ids LIST        logical CUDA/EGL ids for Plus workers (default: 0,1,2,3)
  --force               ignore existing result files
  -h, --help            show this help

Environment overrides:
  STWAM_VENV_PATH CLEAN_LIBERO_ROOT PRO_LIBERO_ROOT PLUS_LIBERO_ROOT LIBERO_DATA_ROOT
  CLASSIFICATION_PATH EGL_VENDOR_CONFIG PRO_PREPARE_SCRIPT RUN_ROOT RESULT_ROOT
  STATS_PATH MANIFEST_DIR PRO_RUNTIME_ROOT GPU_IDS CLEAN_GPU_ID PRO_GPU_ID
  PRO_PERTURBATIONS EXPECTED_PLUS_CASES SKIP_EXISTING

Examples:
  bash scripts/eval_three_task_ablation.sh --plus-only
  bash scripts/eval_three_task_ablation.sh \
    --variants dino_only,vae_only,dino+vae --plus-only
  bash scripts/eval_three_task_ablation.sh \
    --variants dino_only,vae_only,dino+vae --pro-only
EOF
}

while (($#)); do
  case "$1" in
    --variants) VARIANTS="${2:?--variants requires a value}"; shift 2 ;;
    --step) MAX_STEPS="${2:?--step requires a value}"; shift 2 ;;
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --clean-trials) CLEAN_TRIALS="${2:?--clean-trials requires a value}"; shift 2 ;;
    --plus-trials) PLUS_TRIALS="${2:?--plus-trials requires a value}"; shift 2 ;;
    --pro-trials) PRO_TRIALS="${2:?--pro-trials requires a value}"; shift 2 ;;
    --pro-perturbations) PRO_PERTURBATIONS="${2:?--pro-perturbations requires a value}"; shift 2 ;;
    --pro-gpu-id) PRO_GPU_ID="${2:?--pro-gpu-id requires a value}"; shift 2 ;;
    --splits) EVAL_SPLITS="${2:?--splits requires a value}"; shift 2 ;;
    --clean-only) EVAL_SPLITS=clean; shift ;;
    --plus-only) EVAL_SPLITS=plus; shift ;;
    --pro-only) EVAL_SPLITS=pro; shift ;;
    --gpu-ids) GPU_IDS="${2:?--gpu-ids requires a value}"; shift 2 ;;
    --force) SKIP_EXISTING=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Resolve derived defaults after CLI parsing so --step/--seed affect all paths.
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/evaluate_results/three_task/step${MAX_STEPS}_seed${SEED}}"
STATS_PATH="${STATS_PATH:-$RUN_ROOT/dino_future_seed${SEED}_${MAX_STEPS}/dataset_stats.json}"
MANIFEST_DIR="${MANIFEST_DIR:-$RESULT_ROOT/manifests}"
PRO_RUNTIME_ROOT="${PRO_RUNTIME_ROOT:-$RESULT_ROOT/libero_pro_runtime}"

positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
nonnegative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

for name in MAX_STEPS SEED CLEAN_TRIALS PLUS_TRIALS PRO_TRIALS EXPECTED_PLUS_CASES; do
  positive_int "${!name}" || {
    echo "$name must be a positive integer, got ${!name}" >&2
    exit 2
  }
done
nonnegative_int "$CLEAN_GPU_ID" || {
  echo "CLEAN_GPU_ID must be non-negative, got $CLEAN_GPU_ID" >&2
  exit 2
}
nonnegative_int "$PRO_GPU_ID" || {
  echo "PRO_GPU_ID must be non-negative, got $PRO_GPU_ID" >&2
  exit 2
}
[[ "$SKIP_EXISTING" == 0 || "$SKIP_EXISTING" == 1 ]] || {
  echo "SKIP_EXISTING must be 0 or 1, got $SKIP_EXISTING" >&2
  exit 2
}

REQUESTED_SPLITS=()
IFS=',' read -r -a split_tokens <<<"$EVAL_SPLITS"
for split in "${split_tokens[@]}"; do
  split="${split//[[:space:]]/}"
  [[ -z "$split" ]] && continue
  [[ "$split" == clean || "$split" == plus || "$split" == pro ]] || {
    echo "Unknown split $split; expected clean, plus, or pro." >&2
    exit 2
  }
  case " ${REQUESTED_SPLITS[*]} " in
    *" $split "*) ;;
    *) REQUESTED_SPLITS+=("$split") ;;
  esac
done
(( ${#REQUESTED_SPLITS[@]} > 0 )) || {
  echo "At least one evaluation split is required." >&2
  exit 2
}
wants_split() {
  local wanted="$1" split
  for split in "${REQUESTED_SPLITS[@]}"; do
    [[ "$split" == "$wanted" ]] && return 0
  done
  return 1
}

canonical_variant() {
  case "$1" in
    dino|dino_only) echo dino ;;
    vae|vae_only) echo vae ;;
    dual|dino_vae|dino+vae|vae+dino|dino-plus-vae) echo dual ;;
    *) echo "Unknown variant $1" >&2; return 2 ;;
  esac
}
VARIANT_ARRAY=()
IFS=',' read -r -a variant_tokens <<<"$VARIANTS"
for variant in "${variant_tokens[@]}"; do
  variant="${variant//[[:space:]]/}"
  [[ -z "$variant" ]] && continue
  normalized="$(canonical_variant "$variant")"
  case " ${VARIANT_ARRAY[*]} " in
    *" $normalized "*) ;;
    *) VARIANT_ARRAY+=("$normalized") ;;
  esac
done
(( ${#VARIANT_ARRAY[@]} > 0 )) || {
  echo "At least one model variant is required." >&2
  exit 2
}

canonical_pro_perturbation() {
  case "$1" in
    env|environment) echo env ;;
    swap|spatial|spatial_relation) echo swap ;;
    object|objects|object_replacement) echo object ;;
    language|lan|lang) echo language ;;
    task) echo task ;;
    *) echo "Unknown LIBERO-PRO perturbation $1" >&2; return 2 ;;
  esac
}
PRO_PERTURBATION_ARRAY=()
IFS=',' read -r -a pro_perturbation_tokens <<<"$PRO_PERTURBATIONS"
for perturbation in "${pro_perturbation_tokens[@]}"; do
  perturbation="${perturbation//[[:space:]]/}"
  [[ -z "$perturbation" ]] && continue
  normalized="$(canonical_pro_perturbation "$perturbation")"
  case " ${PRO_PERTURBATION_ARRAY[*]} " in
    *" $normalized "*) ;;
    *) PRO_PERTURBATION_ARRAY+=("$normalized") ;;
  esac
done
PRO_PERTURBATION_CSV="$(IFS=,; printf '%s' "${PRO_PERTURBATION_ARRAY[*]}")"
if wants_split pro 2>/dev/null; then
  (( ${#PRO_PERTURBATION_ARRAY[@]} > 0 )) || {
    echo "At least one LIBERO-PRO perturbation is required." >&2
    exit 2
  }
fi

GPU_ID_ARRAY=()
IFS=',' read -r -a gpu_tokens <<<"$GPU_IDS"
for gpu in "${gpu_tokens[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  [[ -z "$gpu" ]] && continue
  nonnegative_int "$gpu" || {
    echo "GPU_IDS must contain non-negative integers, got $gpu" >&2
    exit 2
  }
  case " ${GPU_ID_ARRAY[*]} " in
    *" $gpu "*) ;;
    *) GPU_ID_ARRAY+=("$gpu") ;;
  esac
done
(( ${#GPU_ID_ARRAY[@]} > 0 )) || {
  echo "At least one Plus worker GPU is required." >&2
  exit 2
}

test -f "$VENV_PATH/bin/activate" || {
  echo "Missing virtual environment: $VENV_PATH" >&2
  echo "Run scripts/setup_libero_plus_env.sh or set STWAM_VENV_PATH." >&2
  exit 1
}
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
BASE_PYTHONPATH="${PYTHONPATH:-}"

export LIBERO_DATA_ROOT MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="$PROJECT_ROOT/.runtime/matplotlib"
export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_CONFIG"

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/status" "$MANIFEST_DIR" "$MPLCONFIGDIR"
exec 9>"$RESULT_ROOT/status/eval.lock"
flock -n 9 || {
  echo "Another evaluation is already running for $RESULT_ROOT." >&2
  exit 1
}
printf 'RUNNING pid=%s started_at=%s\n' "$$" "$(date -u +%FT%TZ)" >"$RESULT_ROOT/status/eval.state"
active_pids=()
pipeline_complete=0
cleanup() {
  local status=$?
  if ((${#active_pids[@]})); then
    kill "${active_pids[@]}" 2>/dev/null || true
    wait "${active_pids[@]}" 2>/dev/null || true
  fi
  if ((pipeline_complete == 0)); then
    printf 'FAILED exit=%s stopped_at=%s\n' "$status" "$(date -u +%FT%TZ)" >"$RESULT_ROOT/status/eval.state"
  fi
}
trap cleanup EXIT INT TERM

variant_fields() {
  local variant="$1" run_name
  case "$variant" in
    dino)
      run_name="dino_future_seed${SEED}_${MAX_STEPS}"
      TASK_CONFIG=libero_dino_s_smallvideo_2cam_224_1e-4
      ;;
    vae)
      run_name="fastwam_vae_seed${SEED}_${MAX_STEPS}"
      TASK_CONFIG=libero_uncond_2cam224_1e-4
      ;;
    dual)
      run_name="dual_space_seed${SEED}_${MAX_STEPS}"
      TASK_CONFIG=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4
      ;;
  esac
  CHECKPOINT="$RUN_ROOT/$run_name/checkpoints/weights/step_$(printf '%06d' "$MAX_STEPS").pt"
  CLEAN_OUTPUT="$RESULT_ROOT/$variant/clean"
  PLUS_OUTPUT="$RESULT_ROOT/$variant/plus"
  PRO_OUTPUT="$RESULT_ROOT/$variant/pro"
  test -s "$CHECKPOINT" || {
    echo "Missing checkpoint for $variant: $CHECKPOINT" >&2
    return 1
  }
}

if wants_split clean; then
  test -d "$CLEAN_LIBERO_ROOT/libero/libero/assets" || {
    echo "Invalid clean LIBERO-PRO root: $CLEAN_LIBERO_ROOT" >&2
    exit 1
  }
fi
if wants_split plus; then
  test -d "$PLUS_LIBERO_ROOT/libero/libero/assets/new_objects" || {
    echo "Invalid LIBERO-Plus assets: $PLUS_LIBERO_ROOT" >&2
    exit 1
  }
  test -s "$CLASSIFICATION_PATH" || {
    echo "Missing Plus classification: $CLASSIFICATION_PATH" >&2
    exit 1
  }
fi
if wants_split pro; then
  test -d "$PRO_LIBERO_ROOT/libero/libero/assets" || {
    echo "Invalid LIBERO-PRO root: $PRO_LIBERO_ROOT" >&2
    exit 1
  }
  test -d "$PRO_LIBERO_ROOT/libero/libero/bddl_files/libero_object" || {
    echo "Missing LIBERO-Object BDDL files under: $PRO_LIBERO_ROOT" >&2
    exit 1
  }
  test -s "$PRO_PREPARE_SCRIPT" || {
    echo "Missing PRO runtime preparation script: $PRO_PREPARE_SCRIPT" >&2
    exit 1
  }
fi
test -s "$STATS_PATH" || {
  echo "Missing shared dataset stats: $STATS_PATH" >&2
  exit 1
}
test -s "$EGL_VENDOR_CONFIG" || {
  echo "Missing EGL vendor config: $EGL_VENDOR_CONFIG" >&2
  exit 1
}
for variant in "${VARIANT_ARRAY[@]}"; do
  variant_fields "$variant"
done

if wants_split plus; then
  python - "$CLASSIFICATION_PATH" "$MANIFEST_DIR" "${#GPU_ID_ARRAY[@]}" "$EXPECTED_PLUS_CASES" <<'PY'
import json
import sys
from pathlib import Path

classification_path, output_dir, shard_count, expected_count = sys.argv[1:]
output_dir = Path(output_dir)
shard_count = int(shard_count)
expected_count = int(expected_count)
output_dir.mkdir(parents=True, exist_ok=True)
prefixes = [
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
]
entries = json.loads(Path(classification_path).read_text(encoding="utf-8"))["libero_object"]
selected = [
    entry for entry in entries
    if isinstance(entry, dict)
    and isinstance(entry.get("name"), str)
    and any(entry["name"].startswith(prefix) for prefix in prefixes)
]
counts = {prefix: sum(entry["name"].startswith(prefix) for entry in selected) for prefix in prefixes}
if any(value == 0 for value in counts.values()):
    raise RuntimeError(f"Selected Plus task counts are invalid: {counts}")
if len(selected) != expected_count:
    raise RuntimeError(f"Expected {expected_count} Plus cases, found {len(selected)}.")
ids = [int(entry["id"]) for entry in selected]
if len(ids) != len(set(ids)) or any(value < 1 for value in ids):
    raise RuntimeError("Selected Plus ids must be unique positive integers.")
manifest = [{"suite": "libero_object", "task_id": value - 1} for value in sorted(ids)]
(output_dir / "full.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
for index in range(shard_count):
    (output_dir / f"gpu{index}.json").write_text(
        json.dumps(manifest[index::shard_count], indent=2) + "\n", encoding="utf-8"
    )
clean = [{"suite": "libero_object", "task_id": value} for value in [3, 1, 9]]
(output_dir / "clean.json").write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
(output_dir / "selection.json").write_text(
    json.dumps({
        "prefixes": prefixes,
        "prefix_counts": counts,
        "total_plus_cases": len(manifest),
        "clean_task_ids": [3, 1, 9],
        "shard_count": shard_count,
        "plus_task_ids_are_zero_based": True,
    }, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Manifest ready: {len(manifest)} cases; per task={counts}; shards={shard_count}")
PY
fi
if wants_split clean; then
  python - "$MANIFEST_DIR" <<'PY'
import json
import sys
from pathlib import Path
directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
(directory / "clean.json").write_text(
    json.dumps([{"suite": "libero_object", "task_id": value} for value in [3, 1, 9]], indent=2) + "\n",
    encoding="utf-8",
)
print("Clean manifest ready: task ids [3, 1, 9]")
PY
fi
if wants_split pro; then
  PRO_PERTURBATION_CSV="$(IFS=,; printf '%s' "${PRO_PERTURBATION_ARRAY[*]}")"
  python - "$MANIFEST_DIR" "$PRO_PERTURBATION_CSV" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
perturbations = [item for item in sys.argv[2].split(",") if item]
suite_by_perturbation = {
    "env": "libero_object_env",
    "swap": "libero_object_swap",
    "object": "libero_object_object",
    "language": "libero_object_lan",
    "task": "libero_object_task",
}
task_ids = [3, 1, 9]
directory.mkdir(parents=True, exist_ok=True)
manifest = [
    {"suite": suite_by_perturbation[perturbation], "task_id": task_id}
    for perturbation in perturbations
    for task_id in task_ids
]
(directory / "pro.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
for perturbation in perturbations:
    suite = suite_by_perturbation[perturbation]
    (directory / f"pro_{perturbation}.json").write_text(
        json.dumps(
            [{"suite": suite, "task_id": task_id} for task_id in task_ids],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
(directory / "pro_selection.json").write_text(
    json.dumps(
        {
            "perturbations": perturbations,
            "suite_by_perturbation": {
                key: suite_by_perturbation[key] for key in perturbations
            },
            "task_ids": task_ids,
            "task_ids_are_zero_based": True,
            "tasks_per_perturbation": len(task_ids),
            "total_tasks": len(manifest),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(
    f"PRO manifest ready: {len(manifest)} tasks; "
    f"perturbations={','.join(perturbations)}; task ids={task_ids}"
)
PY
fi

result_state() {
  local manifest_path="$1" output_dir="$2" trials="$3"
  python - "$manifest_path" "$output_dir" "$trials" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
trials = int(sys.argv[3])
tasks = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {(str(item["suite"]), int(item["task_id"])) for item in tasks}
found = defaultdict(list)
invalid = False
for suite in {suite for suite, _ in expected}:
    directory = output_dir / suite
    if not directory.is_dir():
        continue
    for path in directory.glob("gpu*_task*_results.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = (str(payload["task_suite"]), int(payload["task_id"]))
            if key not in expected:
                invalid = True
                continue
            found[key].append(payload)
            invalid |= int(payload.get("total_episodes", -1)) != trials
        except Exception:
            invalid = True
invalid |= any(len(values) != 1 for values in found.values())
if len(found) == len(expected) and not invalid:
    print("complete")
elif invalid:
    print("rerun")
else:
    print("resume")
PY
}

# A report-only rerun should remain usable on a machine without a live GPU.
# Any incomplete result set, or --force, still requires the normal GPU/EGL check.
gpu_preflight_needed=0
if [[ "$SKIP_EXISTING" == 0 ]]; then
  gpu_preflight_needed=1
else
  for variant in "${VARIANT_ARRAY[@]}"; do
    variant_fields "$variant"
    if wants_split clean && [[ "$(result_state "$MANIFEST_DIR/clean.json" "$CLEAN_OUTPUT" "$CLEAN_TRIALS")" != complete ]]; then
      gpu_preflight_needed=1
    fi
    if wants_split plus && [[ "$(result_state "$MANIFEST_DIR/full.json" "$PLUS_OUTPUT" "$PLUS_TRIALS")" != complete ]]; then
      gpu_preflight_needed=1
    fi
    if wants_split pro && [[ "$(result_state "$MANIFEST_DIR/pro.json" "$PRO_OUTPUT" "$PRO_TRIALS")" != complete ]]; then
      gpu_preflight_needed=1
    fi
  done
fi

if ((gpu_preflight_needed)); then
  required_device_index="$CLEAN_GPU_ID"
  if wants_split plus; then
    for gpu in "${GPU_ID_ARRAY[@]}"; do
      ((gpu > required_device_index)) && required_device_index="$gpu"
    done
  fi
  if wants_split pro && ((PRO_GPU_ID > required_device_index)); then
    required_device_index="$PRO_GPU_ID"
  fi
  python - "$required_device_index" <<'PY'
import sys
import torch
from mujoco.egl import egl_ext as EGL
required = int(sys.argv[1])
cuda_count = torch.cuda.device_count()
if cuda_count <= required:
    raise RuntimeError(f"PyTorch sees {cuda_count} CUDA devices; need logical index {required}.")
egl_count = len(EGL.eglQueryDevicesEXT())
if egl_count <= required:
    raise RuntimeError(f"EGL sees {egl_count} devices; need logical index {required}.")
print(f"GPU/EGL preflight: CUDA={cuda_count}, EGL={egl_count}")
PY
else
  echo "GPU/EGL preflight skipped: all requested result sets are complete."
fi

assert_complete() {
  local state
  state="$(result_state "$1" "$2" "$3")"
  [[ "$state" == complete ]] || {
    echo "Incomplete result set: $2 (state=$state)" >&2
    return 1
  }
}

set_clean_environment() {
  export LIBERO_ROOT="$CLEAN_LIBERO_ROOT"
  export LIBERO_CONFIG_PATH="$CLEAN_LIBERO_ROOT/.libero"
  export PYTHONPATH="$CLEAN_LIBERO_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT${BASE_PYTHONPATH:+:$BASE_PYTHONPATH}"
}
set_plus_environment() {
  export LIBERO_ROOT="$PLUS_LIBERO_ROOT"
  export LIBERO_PLUS_ROOT="$PLUS_LIBERO_ROOT"
  export LIBERO_CONFIG_PATH="$PROJECT_ROOT/.runtime/libero"
  export PYTHONPATH="$PLUS_LIBERO_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT${BASE_PYTHONPATH:+:$BASE_PYTHONPATH}"
}
set_pro_environment() {
  export LIBERO_ROOT="$PRO_LIBERO_ROOT"
  export LIBERO_CONFIG_PATH="$PRO_RUNTIME_ROOT/.libero"
  export PYTHONPATH="$PRO_LIBERO_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT${BASE_PYTHONPATH:+:$BASE_PYTHONPATH}"
  unset CUDA_VISIBLE_DEVICES
}

PRO_READY=0
ensure_pro_runtime() {
  set_pro_environment
  ((PRO_READY == 1)) && return 0
  python "$PRO_PREPARE_SCRIPT" \
    --source-root "$PRO_LIBERO_ROOT" \
    --runtime-root "$PRO_RUNTIME_ROOT" \
    --perturbations "$PRO_PERTURBATION_CSV" \
    --task-ids 3,1,9 \
    --init-count "$PRO_TRIALS" \
    --seed "$SEED"

  python - "$MANIFEST_DIR/pro.json" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch
from libero.libero import benchmark, get_libero_path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
benchmark_dict = benchmark.get_benchmark_dict()
expected = {(str(item["suite"]), int(item["task_id"])) for item in manifest}
expected_task_names = {
    3: "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    1: "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    9: "pick_up_the_orange_juice_and_place_it_in_the_basket",
}
if len(expected) != len(manifest):
    raise RuntimeError("PRO manifest contains duplicate suite/task entries")

for suite_name, task_id in sorted(expected):
    if suite_name not in benchmark_dict:
        raise RuntimeError(f"LIBERO-PRO suite is not registered: {suite_name}")
    task_suite = benchmark_dict[suite_name]()
    if task_id < 0 or task_id >= task_suite.get_num_tasks():
        raise RuntimeError(f"Invalid task id {task_id} for {suite_name}")
    if task_id not in expected_task_names:
        raise RuntimeError(f"PRO manifest contains non-training task id: {task_id}")
    task = task_suite.get_task(task_id)
    if task.name != expected_task_names[task_id]:
        raise RuntimeError(
            f"Task order mismatch for {suite_name}:{task_id}: "
            f"expected {expected_task_names[task_id]}, got {task.name}"
        )
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    if not bddl.is_file() or bddl.stat().st_size == 0:
        raise RuntimeError(f"Missing runtime BDDL: {bddl}")
    if not init_states.is_file() or init_states.stat().st_size == 0:
        raise RuntimeError(f"Missing runtime init states: {init_states}")
    states = torch.load(init_states)
    if len(states) == 0:
        raise RuntimeError(f"Runtime init states are empty: {init_states}")
    if suite_name.endswith(("_lan", "_task")):
        text = bddl.read_text(encoding="utf-8")
        if "(:language" not in text:
            raise RuntimeError(f"PRO language block is missing: {bddl}")
print(
    f"PRO runtime verified: {len(expected)} selected tasks across "
    f"{len({suite for suite, _ in expected})} suites; ids=[3, 1, 9]"
)
PY
  PRO_READY=1
}
summarize_clean() {
  local variant="$1"
  local summary_log="$RESULT_ROOT/logs/${variant}_clean_summary.log"
  CKPT="$CHECKPOINT" CONFIG="$TASK_CONFIG" \
    python experiments/libero/summarize_results.py --output_dir="$CLEAN_OUTPUT" \
    >"$summary_log" 2>&1
  tail -n 12 "$summary_log"
}
summarize_plus() {
  local variant="$1"
  local summary_log="$RESULT_ROOT/logs/${variant}_plus_summary.log"
  CKPT="$CHECKPOINT" CONFIG="$TASK_CONFIG" \
    python experiments/libero/summarize_results.py --output_dir="$PLUS_OUTPUT" \
    >"$summary_log" 2>&1
  CKPT="$CHECKPOINT" CONFIG="$TASK_CONFIG" \
    python experiments/libero/summarize_libero_plus_results.py \
      --output_dir "$PLUS_OUTPUT" --classification_path "$CLASSIFICATION_PATH" \
      >>"$summary_log" 2>&1
  tail -n 12 "$summary_log"
}

run_clean() {
  local variant="$1" state skip_existing
  set_clean_environment
  state="$(result_state "$MANIFEST_DIR/clean.json" "$CLEAN_OUTPUT" "$CLEAN_TRIALS")"
  if [[ "$state" == complete && "$SKIP_EXISTING" == 1 ]]; then
    echo "[$variant clean] skip complete result set: $CLEAN_OUTPUT"
    summarize_clean "$variant"
    return
  fi
  skip_existing=false
  [[ "$SKIP_EXISTING" == 1 && "$state" == resume ]] && skip_existing=true
  mkdir -p "$CLEAN_OUTPUT"
  echo "[$variant clean] starting $CLEAN_TRIALS trials (state=$state)"
  local -a command=(
    python experiments/libero/eval_libero_worker.py
    --config-name sim_libero_v3
    "task=$TASK_CONFIG" "data=$DATA_CONFIG" "ckpt=$CHECKPOINT"
    "gpu_id=$CLEAN_GPU_ID" "EVALUATION.device=cuda:$CLEAN_GPU_ID"
    "EVALUATION.dataset_stats_path=$STATS_PATH"
    "EVALUATION.num_trials=$CLEAN_TRIALS" "EVALUATION.output_dir=$CLEAN_OUTPUT"
    "+EVALUATION.task_manifest=$MANIFEST_DIR/clean.json"
    "+EVALUATION.skip_existing=$skip_existing"
    "+EVALUATION.continue_on_error=false" "seed=$SEED"
  )
  env -u MUJOCO_EGL_DEVICE_ID -u EGL_DEVICE_ID \
    "${command[@]}" 2>&1 | tee -a "$RESULT_ROOT/logs/${variant}_clean.log"
  assert_complete "$MANIFEST_DIR/clean.json" "$CLEAN_OUTPUT" "$CLEAN_TRIALS"
  summarize_clean "$variant"
}

run_plus() {
  local variant="$1" state skip_existing failed=0 shard_index gpu
  set_plus_environment
  state="$(result_state "$MANIFEST_DIR/full.json" "$PLUS_OUTPUT" "$PLUS_TRIALS")"
  if [[ "$state" == complete && "$SKIP_EXISTING" == 1 ]]; then
    echo "[$variant plus] skip complete result set: $PLUS_OUTPUT"
    summarize_plus "$variant"
    return
  fi
  skip_existing=false
  [[ "$SKIP_EXISTING" == 1 && "$state" == resume ]] && skip_existing=true
  mkdir -p "$PLUS_OUTPUT/worker_logs"
  active_pids=()
  echo "[$variant plus] starting ${#GPU_ID_ARRAY[@]} workers; $PLUS_TRIALS rollout(s) per case (state=$state)"
  for shard_index in "${!GPU_ID_ARRAY[@]}"; do
    gpu="${GPU_ID_ARRAY[shard_index]}"
    local manifest_path="$MANIFEST_DIR/gpu${shard_index}.json"
    local log_path="$PLUS_OUTPUT/worker_logs/gpu${gpu}.log"
    local -a command=(
      python experiments/libero/eval_libero_worker.py
      --config-name sim_libero_v3
      "task=$TASK_CONFIG" "data=$DATA_CONFIG" "ckpt=$CHECKPOINT"
      "gpu_id=$gpu" "EVALUATION.device=cuda:$gpu"
      "EVALUATION.dataset_stats_path=$STATS_PATH"
      "EVALUATION.num_trials=$PLUS_TRIALS" "EVALUATION.output_dir=$PLUS_OUTPUT"
      "+EVALUATION.task_manifest=$manifest_path"
      "+EVALUATION.skip_existing=$skip_existing"
      "+EVALUATION.continue_on_error=false" "seed=$SEED"
    )
    MUJOCO_EGL_DEVICE_ID="$gpu" EGL_DEVICE_ID="$gpu" \
      "${command[@]}" >"$log_path" 2>&1 &
    active_pids+=("$!")
  done
  for pid in "${active_pids[@]}"; do
    wait "$pid" || failed=1
  done
  active_pids=()
  ((failed == 0)) || {
    echo "[$variant plus] worker failure; inspect $PLUS_OUTPUT/worker_logs" >&2
    return 1
  }
  assert_complete "$MANIFEST_DIR/full.json" "$PLUS_OUTPUT" "$PLUS_TRIALS"
  summarize_plus "$variant"
}

run_pro() {
  local variant="$1" state skip_existing
  state="$(result_state "$MANIFEST_DIR/pro.json" "$PRO_OUTPUT" "$PRO_TRIALS")"
  if [[ "$state" == complete && "$SKIP_EXISTING" == 1 ]]; then
    echo "[$variant pro] skip complete result set: $PRO_OUTPUT"
    return
  fi

  ensure_pro_runtime
  skip_existing=false
  [[ "$SKIP_EXISTING" == 1 && "$state" == resume ]] && skip_existing=true
  mkdir -p "$PRO_OUTPUT"
  echo "[$variant pro] starting ${#PRO_PERTURBATION_ARRAY[@]} perturbations, " \
    "$(( ${#PRO_PERTURBATION_ARRAY[@]} * 3 )) tasks, " \
    "$PRO_TRIALS trials per task (state=$state)"
  local -a command=(
    python experiments/libero/eval_libero_worker.py
    --config-name sim_libero_v3
    "task=$TASK_CONFIG" "data=$DATA_CONFIG" "ckpt=$CHECKPOINT"
    "gpu_id=$PRO_GPU_ID" "EVALUATION.device=cuda:$PRO_GPU_ID"
    "EVALUATION.dataset_stats_path=$STATS_PATH"
    "EVALUATION.num_trials=$PRO_TRIALS" "EVALUATION.output_dir=$PRO_OUTPUT"
    "+EVALUATION.task_manifest=$MANIFEST_DIR/pro.json"
    "+EVALUATION.skip_existing=$skip_existing"
    "+EVALUATION.continue_on_error=false" "seed=$SEED"
  )
  MUJOCO_EGL_DEVICE_ID="$PRO_GPU_ID" EGL_DEVICE_ID="$PRO_GPU_ID" \
    "${command[@]}" 2>&1 | tee -a "$RESULT_ROOT/logs/${variant}_pro.log"
  assert_complete "$MANIFEST_DIR/pro.json" "$PRO_OUTPUT" "$PRO_TRIALS"
}

for variant in "${VARIANT_ARRAY[@]}"; do
  variant_fields "$variant"
  echo "=== $variant ($TASK_CONFIG) ==="
  echo "checkpoint: $CHECKPOINT"
  wants_split clean && run_clean "$variant"
  wants_split plus && run_plus "$variant"
  wants_split pro && run_pro "$variant"
done

python - "$RESULT_ROOT" "$CLASSIFICATION_PATH" "$PRO_PERTURBATION_CSV" "$EVAL_SPLITS" "${VARIANT_ARRAY[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
classification_path = Path(sys.argv[2])
pro_perturbation_csv = sys.argv[3]
requested_splits = {item.strip() for item in sys.argv[4].split(",") if item.strip()}
variants = sys.argv[5:]
labels = {"dino": "DINO-only", "vae": "VAE-only", "dual": "DINO+VAE"}
pro_suite_by_perturbation = {
    "env": "libero_object_env",
    "swap": "libero_object_swap",
    "object": "libero_object_object",
    "language": "libero_object_lan",
    "task": "libero_object_task",
}
prefixes = [
    ("BBQ sauce", "pick_up_the_bbq_sauce_and_place_it_in_the_basket"),
    ("Cream cheese", "pick_up_the_cream_cheese_and_place_it_in_the_basket"),
    ("Orange juice", "pick_up_the_orange_juice_and_place_it_in_the_basket"),
]

def load_results(manifest_path, output_dir):
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = []
    for item in manifest:
        suite = str(item["suite"])
        task_id = int(item["task_id"])
        candidates = sorted((output_dir / suite).glob(f"gpu*_task{task_id}_results.json"))
        if len(candidates) != 1:
            return []
        values.append(json.loads(candidates[0].read_text(encoding="utf-8")))
    return values

def rate(successes, episodes):
    return f"{100.0 * successes / episodes:.2f}%" if episodes else "N/A"

classification = {}
if classification_path.is_file():
    raw = json.loads(classification_path.read_text(encoding="utf-8"))
    classification = {
        (suite, int(entry["id"]) - 1): entry.get("name", "")
        for suite, entries in raw.items()
        for entry in entries
    }

rows = []
plus_results_by_variant = {}
pro_results_by_variant = {}
pro_manifest_path = root / "manifests/pro.json"
pro_selection_path = root / "manifests/pro_selection.json"
pro_perturbations = [item for item in pro_perturbation_csv.split(",") if item]
if "pro" not in requested_splits:
    pro_perturbations = []
if pro_selection_path.is_file():
    pro_perturbations = json.loads(pro_selection_path.read_text(encoding="utf-8"))["perturbations"]
for variant in variants:
    base = root / variant
    clean = (
        load_results(root / "manifests/clean.json", base / "clean")
        if "clean" in requested_splits
        else []
    )
    plus = (
        load_results(root / "manifests/full.json", base / "plus")
        if "plus" in requested_splits
        else []
    )
    clean_s = sum(int(item.get("successes", 0)) for item in clean)
    clean_n = sum(int(item.get("total_episodes", 0)) for item in clean)
    plus_s = sum(int(item.get("successes", 0)) for item in plus)
    plus_n = sum(int(item.get("total_episodes", 0)) for item in plus)
    pro = (
        load_results(pro_manifest_path, base / "pro")
        if "pro" in requested_splits
        else []
    )
    pro_s = sum(int(item.get("successes", 0)) for item in pro)
    pro_n = sum(int(item.get("total_episodes", 0)) for item in pro)
    plus_results_by_variant[variant] = plus
    pro_results_by_variant[variant] = pro
    rows.append({
        "variant": variant,
        "label": labels.get(variant, variant),
        "clean": f"{clean_s}/{clean_n} ({rate(clean_s, clean_n)})" if clean else "N/A",
        "plus": f"{plus_s}/{plus_n} ({rate(plus_s, plus_n)})" if plus else "N/A",
        "pro": f"{pro_s}/{pro_n} ({rate(pro_s, pro_n)})" if pro else "N/A",
    })

lines = [
    "# Three-task LIBERO / LIBERO-Plus / LIBERO-PRO comparison",
    "",
    f"| Variant | Clean LIBERO | LIBERO-Plus | LIBERO-PRO ({len(pro_perturbations)} perturbations) |",
    "| --- | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(f"| {row['label']} | {row['clean']} | {row['plus']} | {row['pro']} |")

if any(pro_results_by_variant.values()):
    lines += [
        "",
        "## LIBERO-PRO breakdown (training task ids 3, 1, 9 only)",
        "",
        "| Perturbation | Suite | Tasks | Episodes | "
        + " | ".join(labels.get(v, v) for v in variants)
        + " |",
        "| --- | --- | ---: | ---: | "
        + " | ".join("---:" for _ in variants)
        + " |",
    ]
    for perturbation in pro_perturbations:
        suite = pro_suite_by_perturbation[perturbation]
        cells = []
        task_count = 0
        episode_count = 0
        for variant in variants:
            selected = [
                item for item in pro_results_by_variant[variant]
                if str(item.get("task_suite")) == suite
            ]
            successes = sum(int(item.get("successes", 0)) for item in selected)
            episodes = sum(int(item.get("total_episodes", 0)) for item in selected)
            task_count = max(task_count, len(selected))
            episode_count = max(episode_count, episodes)
            cells.append(
                f"{successes}/{episodes} ({rate(successes, episodes)})"
                if selected else "N/A"
            )
        lines.append(
            f"| {perturbation} | {suite} | {task_count} | {episode_count} | "
            + " | ".join(cells)
            + " |"
        )
    for variant in variants:
        base = root / variant
        by_perturbation = {}
        for perturbation in pro_perturbations:
            suite = pro_suite_by_perturbation[perturbation]
            selected = [
                item for item in pro_results_by_variant[variant]
                if str(item.get("task_suite")) == suite
            ]
            successes = sum(int(item.get("successes", 0)) for item in selected)
            episodes = sum(int(item.get("total_episodes", 0)) for item in selected)
            by_perturbation[perturbation] = {
                "suite": suite,
                "task_ids": [int(item.get("task_id")) for item in selected],
                "tasks": len(selected),
                "episodes": episodes,
                "successes": successes,
                "success_rate": 100.0 * successes / episodes if episodes else None,
            }
        total_successes = sum(item["successes"] for item in by_perturbation.values())
        total_episodes = sum(item["episodes"] for item in by_perturbation.values())
        (base / "pro_summary.json").write_text(
            json.dumps(
                {
                    "task_ids": [3, 1, 9],
                    "perturbations": pro_perturbations,
                    "by_perturbation": by_perturbation,
                    "total": {
                        "tasks": sum(item["tasks"] for item in by_perturbation.values()),
                        "episodes": total_episodes,
                        "successes": total_successes,
                        "success_rate": (
                            100.0 * total_successes / total_episodes
                            if total_episodes else None
                        ),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

if classification and any(plus_results_by_variant.values()):
    lines += [
        "",
        "## LIBERO-Plus base-task breakdown",
        "",
        "| Base task | Cases | " + " | ".join(labels.get(v, v) for v in variants) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in variants) + " |",
    ]
    for task_label, prefix in prefixes:
        task_ids = {
            key[1] for key, name in classification.items()
            if key[0] == "libero_object" and name.startswith(prefix)
        }
        cells = []
        for variant in variants:
            selected = [
                item for item in plus_results_by_variant[variant]
                if int(item.get("task_id", -1)) in task_ids
            ]
            successes = sum(int(item.get("successes", 0)) for item in selected)
            episodes = sum(int(item.get("total_episodes", 0)) for item in selected)
            cells.append(
                f"{successes}/{episodes} ({rate(successes, episodes)})"
                if selected else "N/A"
            )
        lines.append(f"| {task_label} | {len(task_ids)} | " + " | ".join(cells) + " |")

(root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(root / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(root / "comparison.md")
PY

printf 'COMPLETE finished_at=%s\n' "$(date -u +%FT%TZ)" >"$RESULT_ROOT/status/eval.state"
pipeline_complete=1
trap - EXIT INT TERM
echo "All requested evaluations completed: $RESULT_ROOT/comparison.md"
