#!/bin/bash
# setup_env.sh - Environment Setup & DATE-LM Data Download
# Task: setup_env
# GPU: 1 (for verification only)
set -e

REMOTE_BASE="/home/jinxulin/sibyl_system"
PROJECT="CRA"
PROJECT_DIR="${REMOTE_BASE}/projects/${PROJECT}"
CONDA_BIN="/home/jinxulin/miniconda3/bin/conda"
ENV_NAME="sibyl_CRA"
SHARED_DIR="${REMOTE_BASE}/shared"

echo "=== Step 1: Create directory structure ==="
mkdir -p "${PROJECT_DIR}/exp/results/pilots"
mkdir -p "${PROJECT_DIR}/exp/results/full"
mkdir -p "${PROJECT_DIR}/exp/results/phase0"
mkdir -p "${PROJECT_DIR}/exp/results/phase1"
mkdir -p "${PROJECT_DIR}/exp/results/phase2"
mkdir -p "${PROJECT_DIR}/exp/results/phase3"
mkdir -p "${PROJECT_DIR}/exp/scripts"
mkdir -p "${PROJECT_DIR}/exp/logs"
mkdir -p "${SHARED_DIR}/datasets/date_lm"
mkdir -p "${SHARED_DIR}/checkpoints"
mkdir -p "${SHARED_DIR}/registry"
echo "Directory structure created."

echo "=== Step 2: Create conda environment ==="
if ${CONDA_BIN} env list | grep -q "${ENV_NAME}"; then
    echo "Environment ${ENV_NAME} already exists, skipping creation."
else
    ${CONDA_BIN} create -n ${ENV_NAME} python=3.11 -y
    echo "Environment ${ENV_NAME} created."
fi

echo "=== Step 3: Install dependencies ==="
${CONDA_BIN} run -n ${ENV_NAME} pip install --upgrade pip
${CONDA_BIN} run -n ${ENV_NAME} pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
${CONDA_BIN} run -n ${ENV_NAME} pip install \
    transformers \
    datasets \
    accelerate \
    traker[fast] \
    rank-bm25 \
    scikit-learn \
    scipy \
    numpy \
    matplotlib \
    tqdm \
    pandas

echo "=== Step 4: Verify installations ==="
${CONDA_BIN} run -n ${ENV_NAME} python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA devices: {torch.cuda.device_count()}')

import transformers
print(f'Transformers: {transformers.__version__}')

import datasets
print(f'Datasets: {datasets.__version__}')

import sklearn
print(f'Scikit-learn: {sklearn.__version__}')

import scipy
print(f'SciPy: {scipy.__version__}')

import numpy
print(f'NumPy: {numpy.__version__}')

from rank_bm25 import BM25Okapi
print('rank-bm25: OK')

print('All dependencies verified.')
"

echo "=== Dependencies installed and verified ==="
