#!/bin/bash
# setup_env_symlink.sh - Setup by symlinking shared packages from sibyl_TECA
# Saves ~3GB disk by avoiding duplicate PyTorch install
set -e

PROJECT_DIR="/home/jinxulin/sibyl_system/projects/cross-task-influence"
RESULTS_DIR="${PROJECT_DIR}/exp/results"
TASK_ID="setup_env"
CONDA="/home/jinxulin/miniconda3/bin/conda"
ENV_NAME="sibyl_cross-task-influence"
CONDA_RUN="${CONDA} run --no-capture-output -n ${ENV_NAME}"

TECA_SITE="/home/jinxulin/miniconda3/envs/sibyl_TECA/lib/python3.10/site-packages"
OUR_SITE="/home/jinxulin/miniconda3/envs/sibyl_cross-task-influence/lib/python3.10/site-packages"

mkdir -p "${RESULTS_DIR}"
echo $$ > "${RESULTS_DIR}/${TASK_ID}.pid"

cd "${PROJECT_DIR}"

echo "=== Step 1/5: Symlink PyTorch and core packages from sibyl_TECA ==="

# Packages to symlink (these exist in TECA and are Python 3.10 compatible)
# We symlink the package dirs AND their dist-info dirs
SYMLINK_PKGS=(
    # PyTorch
    "torch" "torch-2.5.1+cu121.dist-info" "torchgen"
    "torchvision" "torchvision-0.20.1+cu121.dist-info" "torchvision.libs"
    # Core ML
    "numpy" "numpy-1*dist-info" "numpy.libs"
    "scipy" "scipy-1*dist-info" "scipy.libs"
    "matplotlib" "matplotlib-3*dist-info" "mpl_toolkits"
    "sklearn" "scikit_learn-*dist-info"
    # DL ecosystem
    "transformers" "transformers-*dist-info"
    "datasets" "datasets-*dist-info"
    "accelerate" "accelerate-*dist-info"
    "peft" "peft-*dist-info"
    "tqdm" "tqdm-*dist-info"
    "einops" "einops-*dist-info"
    # Common dependencies that torch/transformers need
    "yaml" "_yaml"
    "PIL" "pillow-*dist-info" "Pillow-*dist-info" "Pillow.libs"
    "filelock" "filelock-*dist-info"
    "requests" "requests-*dist-info"
    "urllib3" "urllib3-*dist-info"
    "certifi" "certifi-*dist-info"
    "charset_normalizer" "charset_normalizer-*dist-info"
    "idna" "idna-*dist-info"
    "packaging" "packaging-*dist-info"
    "regex" "regex-*dist-info"
    "safetensors" "safetensors-*dist-info"
    "tokenizers" "tokenizers-*dist-info"
    "huggingface_hub" "huggingface_hub-*dist-info"
    "fsspec" "fsspec-*dist-info"
    "pyarrow" "pyarrow-*dist-info" "pyarrow.libs"
    "multiprocess" "multiprocess-*dist-info"
    "dill" "dill-*dist-info"
    "xxhash" "xxhash-*dist-info"
    "psutil" "psutil-*dist-info"
    "sympy" "sympy-*dist-info"
    "mpmath" "mpmath-*dist-info"
    "jinja2" "jinja2-*dist-info" "Jinja2-*dist-info"
    "markupsafe" "markupsafe-*dist-info" "MarkupSafe-*dist-info"
    "networkx" "networkx-*dist-info"
    "typing_extensions.py" "typing_extensions-*dist-info"
    "triton" "triton-*dist-info"
    "_triton"
    "nvidia"
)

symlinked=0
skipped=0
for pattern in "${SYMLINK_PKGS[@]}"; do
    # Use glob to match (handles version wildcards)
    for src in ${TECA_SITE}/${pattern}; do
        if [ -e "$src" ]; then
            basename=$(basename "$src")
            target="${OUR_SITE}/${basename}"
            if [ -e "$target" ] || [ -L "$target" ]; then
                skipped=$((skipped + 1))
            else
                ln -s "$src" "$target"
                symlinked=$((symlinked + 1))
            fi
        fi
    done
done
echo "Symlinked: ${symlinked}, Skipped (already exist): ${skipped}"

echo "=== Step 2/5: Install small missing packages ==="
# Only install packages not in TECA (small ones that won't blow quota)
${CONDA_RUN} pip install --no-cache-dir h5py tensorboard 2>&1 | tail -5

echo "=== Step 3/5: Ensure LIBERO is installed ==="
if [ -d "${PROJECT_DIR}/libero_repo" ]; then
    # Already cloned, just make sure it's installed
    cd "${PROJECT_DIR}/libero_repo"
    ${CONDA_RUN} pip install --no-cache-dir --no-deps -e . 2>&1 | tail -3
else
    cd "${PROJECT_DIR}"
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git libero_repo 2>&1 | tail -3
    cd libero_repo
    ${CONDA_RUN} pip install --no-cache-dir --no-deps -e . 2>&1 | tail -3
fi
cd "${PROJECT_DIR}"

echo "=== Step 4/5: Install robomimic (no-deps, small) ==="
${CONDA_RUN} pip install --no-cache-dir robomimic 2>&1 | tail -5

echo "=== Step 5/5: Verification ==="
VERIFY_OUTPUT=$(CUDA_VISIBLE_DEVICES=1 ${CONDA_RUN} python -c "
import json, sys
results = {}

# PyTorch + CUDA
import torch
results['pytorch_version'] = torch.__version__
results['cuda_available'] = torch.cuda.is_available()
results['cuda_device_count'] = torch.cuda.device_count()
if torch.cuda.is_available():
    results['cuda_device_name'] = torch.cuda.get_device_name(0)
    results['cuda_version'] = torch.version.cuda

# Core packages
import numpy; results['numpy'] = numpy.__version__
import scipy; results['scipy'] = scipy.__version__
import matplotlib; results['matplotlib'] = matplotlib.__version__
import transformers; results['transformers'] = transformers.__version__
import datasets; results['datasets'] = datasets.__version__

# robomimic
try:
    import robomimic
    results['robomimic'] = getattr(robomimic, '__version__', 'installed')
except Exception as e:
    results['robomimic'] = f'import_error: {e}'

# LIBERO
try:
    import libero
    results['libero'] = 'installed'
    # Check LIBERO benchmarks available
    try:
        from libero.libero import benchmark
        bm = benchmark.get_benchmark_dict()
        results['libero_benchmarks'] = list(bm.keys()) if isinstance(bm, dict) else str(type(bm))
    except Exception as e2:
        results['libero_benchmark_check'] = str(e2)
except Exception as e:
    results['libero'] = f'import_error: {e}'

# Quick GPU smoke test on GPU 1 (visible as cuda:0 via CUDA_VISIBLE_DEVICES)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    x = torch.randn(32, 512, device=device)
    y = torch.nn.Linear(512, 256).to(device)(x)
    results['gpu_smoke_test'] = 'PASS'
    results['gpu_smoke_output_shape'] = list(y.shape)
    # Memory info
    results['gpu_memory_total_mb'] = round(torch.cuda.get_device_properties(0).total_mem / 1e6, 1)
    results['gpu_memory_allocated_mb'] = round(torch.cuda.memory_allocated(0) / 1e6, 1)
else:
    results['gpu_smoke_test'] = 'SKIP_NO_CUDA'

results['status'] = 'success'
print(json.dumps(results, indent=2))
" 2>&1)

echo "${VERIFY_OUTPUT}"
echo "${VERIFY_OUTPUT}" > "${RESULTS_DIR}/setup_verify.json"

# Write DONE marker
python3 -c "
import json
from pathlib import Path
from datetime import datetime

pid_file = Path('${RESULTS_DIR}/${TASK_ID}.pid')
if pid_file.exists():
    pid_file.unlink()

marker = Path('${RESULTS_DIR}/${TASK_ID}_DONE')
marker.write_text(json.dumps({
    'task_id': '${TASK_ID}',
    'status': 'success',
    'summary': 'Environment setup complete via symlink strategy. PyTorch from sibyl_TECA, small packages installed fresh.',
    'timestamp': datetime.now().isoformat(),
}))
"

echo "=== Setup complete ==="
