#!/usr/bin/env bash
set -euo pipefail

# Reproduce the requested two-row Figure-5-style MoT attention diagnostic
# with the released ST-WAM base checkpoint. Each task gets its own process so
# the process-local attention save counter cannot affect the other task.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
VENV_PATH="${STWAM_VENV_PATH:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"

LIBERO_ROOT="${LIBERO_ROOT:-$WORKSPACE_ROOT/LIBERO-PRO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LIBERO_ROOT/.libero}"
EGL_VENDOR_CONFIG="${EGL_VENDOR_CONFIG:-$PROJECT_ROOT/configs/egl/10_nvidia.json}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"

OFFICIAL_ROOT="${OFFICIAL_ROOT:-/mnt/data/embodied_datasets/ST-WAM/checkpoints/modelscope/Semantic_Temporal_World_Action_Model}"
CHECKPOINT="${CHECKPOINT:-$OFFICIAL_ROOT/libero/ST-WAM/st-wam-base-step021700/model.pt}"
OFFICIAL_CONFIG="${OFFICIAL_CONFIG:-$OFFICIAL_ROOT/libero/ST-WAM/st-wam-base-step021700/config.yaml}"
OFFICIAL_STATS="${OFFICIAL_STATS:-$OFFICIAL_ROOT/libero/ST-WAM/st-wam-base-step021700/dataset_stats.json}"
QWEN_PATH="${QWEN_PATH:-/mnt/data/zxw/lda/pretrained/vlm/Qwen3-VL-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/evaluate_results/mot_attention_figure5/base_step021700_seed${SEED}_$(date -u +%Y%m%d_%H%M%S)}"

QUERY_RANGES='["0:32"]'

for path in "$PYTHON_BIN" "$CHECKPOINT" "$OFFICIAL_CONFIG" "$OFFICIAL_STATS" "$QWEN_PATH" "$EGL_VENDOR_CONFIG"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is required to validate complete episode video length" >&2; exit 1; }
[[ -d "$LIBERO_ROOT/libero/libero/assets" ]] || {
  echo "Invalid LIBERO_ROOT: $LIBERO_ROOT" >&2
  exit 1
}

CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
STATS_SHA256="$(sha256sum "$OFFICIAL_STATS" | awk '{print $1}')"
CONFIG_SHA256="$(sha256sum "$OFFICIAL_CONFIG" | awk '{print $1}')"

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

"$PYTHON_BIN" - "$OUTPUT_ROOT/protocol.json" "$CHECKPOINT_SHA256" "$STATS_SHA256" "$CONFIG_SHA256" "$CHECKPOINT" "$OFFICIAL_CONFIG" "$OFFICIAL_STATS" "$QWEN_PATH" "$LIBERO_ROOT" <<'PY'
import json
import sys
from pathlib import Path

output, checkpoint_sha256, stats_sha256, config_sha256, checkpoint, config, stats, qwen, libero_root = sys.argv[1:]
payload = {
    "format": "st-wam-mot-figure5-reproduction-v1",
    "checkpoint": checkpoint,
    "checkpoint_sha256": checkpoint_sha256,
    "official_config": config,
    "official_config_sha256": config_sha256,
    "official_dataset_stats": stats,
    "official_dataset_stats_sha256": stats_sha256,
    "qwen_vl_path": qwen,
    "libero_root": libero_root,
    "tasks": [
        {
            "row": "drawer",
            "suite": "libero_goal",
            "task_id": 0,
            "expected_description": "open the middle drawer of the cabinet",
        },
        {
            "row": "moka_pot",
            "suite": "libero_10",
            "task_id": 2,
            "expected_description": "turn on the stove and put the moka pot on it",
        },
    ],
    "runtime_protocol": {
        "seed": 42,
        "num_trials": 1,
        "num_inference_steps": 10,
        "replan_steps": 10,
        "num_steps_wait": 30,
        "gripper_action_mode": "legacy_rlds",
        "rotate_observation_images_180": True,
        "env_resolution": 256,
        "layer": 29,
        "denoising_step": 9,
        "heads": "all",
        "action_query_range": [0, 32],
        "display_normalization": "percentile",
        "display_percentiles": [1.0, 99.0],
        "save_all_inference_calls": True,
    },
    "paper_parameter_disclosure": (
        "The paper does not publicly specify the layer, head set, denoising step, "
        "or action-query aggregation used for this visualization. These are "
        "definition-level reproduction choices, not a claim of pixel-level replication."
    ),
}
Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

run_task() {
  local row="$1"
  local suite="$2"
  local task_id="$3"
  local expected="$4"
  local task_root="$OUTPUT_ROOT/$row"
  local attention_dir="$task_root/attention"
  local result_dir="$task_root/results"
  mkdir -p "$attention_dir" "$result_dir"

  echo "[$row] suite=$suite task_id=$task_id expected=$expected"
  "$PYTHON_BIN" experiments/libero/eval_libero_single.py \
    --config-name sim_libero \
    task=libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4 \
    "ckpt=$CHECKPOINT" \
    "seed=$SEED" \
    "EVALUATION.device=cuda" \
    "EVALUATION.task_suite_name=$suite" \
    "EVALUATION.task_id=$task_id" \
    "EVALUATION.expected_task_description=$expected" \
    "EVALUATION.num_trials=1" \
    "EVALUATION.num_steps_wait=30" \
    "EVALUATION.replan_steps=10" \
    "EVALUATION.num_inference_steps=10" \
    "EVALUATION.rand_device=cpu" \
    "EVALUATION.gripper_action_mode=legacy_rlds" \
    "EVALUATION.rotate_observation_images_180=true" \
    "EVALUATION.env_resolution=256" \
    "EVALUATION.output_dir=$result_dir" \
    "EVALUATION.dataset_stats_path=$OFFICIAL_STATS" \
    "EVALUATION.checkpoint_sha256=$CHECKPOINT_SHA256" \
    "EVALUATION.mot_attention_visualization.enabled=true" \
    "EVALUATION.mot_attention_visualization.output_dir=$attention_dir" \
    'EVALUATION.mot_attention_visualization.layers=[29]' \
    'EVALUATION.mot_attention_visualization.denoising_steps=[9]' \
    'EVALUATION.mot_attention_visualization.heads=all' \
    "EVALUATION.mot_attention_visualization.action_query_ranges=$QUERY_RANGES" \
    "EVALUATION.mot_attention_visualization.call_indices=all" \
    'EVALUATION.mot_attention_visualization.max_saves=1000' \
    'EVALUATION.mot_attention_visualization.display_normalization=percentile' \
    'EVALUATION.mot_attention_visualization.display_percentiles=[1.0,99.0]' \
    'EVALUATION.mot_attention_visualization.overlay_alpha=0.45' \
    "model.semantic_history_config.vlm_model_name_or_path=$QWEN_PATH" \
    '+model.semantic_history_config.allow_vlm_path_relocation=true' \
    2>&1 | tee "$task_root/eval.log"

  local rollout_video_path
  rollout_video_path="$(find "$result_dir/$suite/videos" -maxdepth 1 -type f -name '*.mp4' -print | sort | tail -n 1)"
  [[ -n "$rollout_video_path" ]] || {
    echo "Could not locate the completed rollout video under $result_dir/$suite/videos" >&2
    exit 1
  }
  local rollout_action_frames
  rollout_action_frames="$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$rollout_video_path")"
  [[ "$rollout_action_frames" =~ ^[0-9]+$ ]] || {
    echo "Could not determine rollout frame count from $rollout_video_path: $rollout_action_frames" >&2
    exit 1
  }
  local attention_episode_length=$((30 + rollout_action_frames))
  echo "[$row] exact attention video length=$attention_episode_length (initial_wait=30, action_frames=$rollout_action_frames)"

  "$PYTHON_BIN" scripts/make_mot_attention_video.py \
    --attention-dir "$attention_dir" \
    --output "$task_root/attention_episode.mp4" \
    --fps 24 \
    --hold-frames 10 \
    --initial-frames 30 \
    --episode-length "$attention_episode_length" \
    --episode-index 0 \
    --layer 29 \
    --denoising-step 9 \
    --query-range 0:32 \
    2>&1 | tee "$task_root/video.log"

  "$PYTHON_BIN" - "$OUTPUT_ROOT/protocol.json" "$row" "$attention_episode_length" <<'PY'
import json
import sys
from pathlib import Path

protocol_path, row, episode_length = sys.argv[1:]
payload = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
video = payload.setdefault("attention_video_protocol", {})
video.setdefault("fps", 24)
video.setdefault("initial_wait_frames", 30)
video.setdefault("initial_frame", "no_attention_capture_slate")
video.setdefault("coverage", "complete_environment_episode")
video.setdefault(
    "length_source",
    "num_steps_wait plus completed evaluator rollout MP4 frame count",
)
lengths = video.setdefault("episode_length_frames", {})
lengths[str(row)] = int(episode_length)
Path(protocol_path).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
}

run_task drawer libero_goal 0 "open the middle drawer of the cabinet"
run_task moka_pot libero_10 2 "turn on the stove and put the moka pot on it"

echo "Completed Figure-5 attention capture: $OUTPUT_ROOT"
echo "checkpoint_sha256=$CHECKPOINT_SHA256"
