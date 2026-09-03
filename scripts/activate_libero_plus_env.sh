#!/usr/bin/env bash

# Source this file from the ST-WAM repository root:
#   source scripts/activate_libero_plus_env.sh

set -u

STWAM_REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STWAM_WORKSPACE_ROOT="$(dirname "${STWAM_REPRO_ROOT}")"
STWAM_VENV_PATH="${STWAM_VENV_PATH:-${STWAM_REPRO_ROOT}/.venv}"

if [[ ! -f "${STWAM_VENV_PATH}/bin/activate" ]]; then
  echo "Missing ${STWAM_VENV_PATH}; run scripts/setup_libero_plus_env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${STWAM_VENV_PATH}/bin/activate"

export STWAM_ROOT="${STWAM_REPRO_ROOT}"
export LIBERO_ROOT="${STWAM_WORKSPACE_ROOT}/LIBERO-plus"
export LIBERO_PLUS_ROOT="${LIBERO_ROOT}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero}"
export LIBERO_CONFIG_PATH="${STWAM_REPRO_ROOT}/.runtime/libero"
export PYTHONPATH="${LIBERO_ROOT}:${STWAM_REPRO_ROOT}/src:${PYTHONPATH:-}"
export UV_CACHE_DIR="${STWAM_WORKSPACE_ROOT}/.cache/uv"
export HF_HOME="${STWAM_WORKSPACE_ROOT}/.cache/huggingface"
export MODELSCOPE_CACHE="${STWAM_WORKSPACE_ROOT}/.cache/modelscope"
export MPLCONFIGDIR="${STWAM_REPRO_ROOT}/.runtime/matplotlib"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-${STWAM_REPRO_ROOT}/.runtime/egl_vendor.d/10_nvidia.json}"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${LIBERO_CONFIG_PATH}" "${MPLCONFIGDIR}" "${UV_CACHE_DIR}" "${HF_HOME}" "${MODELSCOPE_CACHE}"

echo "Activated ST-WAM environment: ${VIRTUAL_ENV}"
echo "STWAM_ROOT=${STWAM_ROOT}"
echo "LIBERO_ROOT=${LIBERO_ROOT}"
echo "LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT}"
