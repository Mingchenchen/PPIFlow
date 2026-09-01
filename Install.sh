#!/bin/bash

set -Eeuo pipefail

#######################################
# Constants
#######################################
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found in PATH." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
CONDA_INIT_SCRIPT="${CONDA_BASE}/etc/profile.d/conda.sh"
ENV_YAML_FILE="ppiflow_af3_merged.yaml"
CONDA_ENV_NAME="ppiflow_af3"


PROTEIN_MPNN_REPO_URL="https://github.com/dauparas/ProteinMPNN.git"
PROTEIN_MPNN_REPO_DIR="ProteinMPNN"

AF3SCORE_REPO_URL="https://github.com/Mingchenchen/AF3Score.git"
AF3SCORE_REPO_DIR="AF3Score"

PPI_FLOW_TOOLS_DIR="tools"

TRITON_VERSION="3.1.0"
ZLIB_LIBRARY_NAME="libz.so"

#######################################
# Error handling
#######################################
error_exit() {
    local line_number="$1"
    local failed_command="$2"

    echo "Error: script execution failed." >&2
    echo "Line: ${line_number}" >&2
    echo "Command: ${failed_command}" >&2
    exit 1
}

trap 'error_exit "${LINENO}" "${BASH_COMMAND}"' ERR

#######################################
# Helper functions
#######################################
log_info() {
    echo "[INFO] $1"
}

require_file() {
    local file_path="$1"

    if [[ ! -f "${file_path}" ]]; then
        echo "Error: required file not found: ${file_path}" >&2
        exit 1
    fi
}

require_directory() {
    local dir_path="$1"

    if [[ ! -d "${dir_path}" ]]; then
        echo "Error: required directory not found: ${dir_path}" >&2
        exit 1
    fi
}

change_directory() {
    local target_dir="$1"

    if [[ ! -d "${target_dir}" ]]; then
        echo "Error: cannot change directory because it does not exist: ${target_dir}" >&2
        exit 1
    fi

    cd "${target_dir}"
}

clone_repository() {
    local repo_url="$1"
    local repo_dir="$2"

    log_info "Cloning repository: ${repo_url}"
    git clone "${repo_url}"

    require_directory "${repo_dir}"
}

#######################################
# Pre-checks
#######################################
require_file "${CONDA_INIT_SCRIPT}"
require_file "${ENV_YAML_FILE}"

#######################################
# Initialize conda
#######################################
# shellcheck disable=SC1090
source "${CONDA_INIT_SCRIPT}"

#######################################
# Create and activate conda environment
#######################################
log_info "Creating conda environment from: ${ENV_YAML_FILE}"
conda env create -f "${ENV_YAML_FILE}"

log_info "Activating conda environment: ${CONDA_ENV_NAME}"
set +euo pipefail
conda activate "${CONDA_ENV_NAME}"
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Error: CONDA_PREFIX is not set. The conda environment may not have been activated correctly." >&2
    exit 1
fi

#######################################
# Install required Triton version
#######################################
log_info "Uninstalling existing Triton package"
pip uninstall triton -y

log_info "Installing Triton ${TRITON_VERSION}"
pip install "triton==${TRITON_VERSION}"

#######################################
# Clone repositories
#######################################
change_directory "${PPI_FLOW_TOOLS_DIR}"
clone_repository "${PROTEIN_MPNN_REPO_URL}" "${PROTEIN_MPNN_REPO_DIR}"
clone_repository "${AF3SCORE_REPO_URL}" "${AF3SCORE_REPO_DIR}"

change_directory "${AF3SCORE_REPO_DIR}"

#######################################
# Set build-related environment variables
#######################################
export CPPFLAGS="-I${CONDA_PREFIX}/include"
export LDFLAGS="-L${CONDA_PREFIX}/lib"
export CMAKE_ARGS="-DZLIB_INCLUDE_DIR=${CONDA_PREFIX}/include -DZLIB_LIBRARY=${CONDA_PREFIX}/lib/${ZLIB_LIBRARY_NAME}"

#######################################
# Install project and build data
#######################################
log_info "Installing AF3Score"
pip install --no-deps -e .

log_info "Running build_data"
build_data

log_info "Successfully installed !!!"
