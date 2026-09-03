#!/usr/bin/env bash

# Build the paper-reference Python environment. This script intentionally does
# not download datasets, model weights, or the 83.6 GB DINO feature cache.

set -euo pipefail

STWAM_REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STWAM_WORKSPACE_ROOT="$(dirname "${STWAM_REPRO_ROOT}")"
LIBERO_PLUS_DIR="${STWAM_WORKSPACE_ROOT}/LIBERO-plus"
UV_CACHE_DIR="${STWAM_WORKSPACE_ROOT}/.cache/uv"
UV_PYTHON_INSTALL_DIR="${STWAM_WORKSPACE_ROOT}/.local/share/uv/python"
STWAM_VENV_PATH="${STWAM_VENV_PATH:-${STWAM_REPRO_ROOT}/.venv}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi
if [[ ! -d "${LIBERO_PLUS_DIR}/.git" ]]; then
  echo "Missing LIBERO-Plus checkout at ${LIBERO_PLUS_DIR}." >&2
  exit 1
fi
if [[ -e "${STWAM_VENV_PATH}" ]]; then
  echo "${STWAM_VENV_PATH} already exists; refusing to overwrite it." >&2
  echo "Move it aside explicitly before rebuilding the reference environment." >&2
  exit 1
fi

cd "${STWAM_REPRO_ROOT}"
uv python install 3.10
uv venv --python 3.10 "${STWAM_VENV_PATH}"

uv pip install --python "${STWAM_VENV_PATH}/bin/python" \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchcodec==0.5 \
  --default-index "${PYPI_INDEX_URL}" \
  --index "${PYTORCH_INDEX_URL}"
uv pip install --python "${STWAM_VENV_PATH}/bin/python" -e . \
  --default-index "${PYPI_INDEX_URL}"

# LIBERO-Plus declares no install_requires. Keep the newer ST-WAM Hydra and
# Transformers pins, and add only the simulator-side packages required here.
uv pip install --python "${STWAM_VENV_PATH}/bin/python" \
  mujoco==3.3.2 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 \
  easydict==1.9 cloudpickle==2.1.0 opencv-python==4.6.0.66 \
  future==1.0.0 matplotlib wand scikit-image \
  --default-index "${PYPI_INDEX_URL}"
uv pip install --python "${STWAM_VENV_PATH}/bin/python" \
  --no-deps --no-build-isolation -e "${LIBERO_PLUS_DIR}"

echo "Environment created at ${STWAM_VENV_PATH}."
echo "Run: source scripts/activate_libero_plus_env.sh"
