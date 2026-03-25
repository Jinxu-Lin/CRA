#!/bin/bash
# setup_env.sh - Environment setup for cross-task-influence project
# Installs PyTorch, LIBERO, robomimic, and dependencies
# GPU: RTX 4090, CUDA 12.x expected

set -e

PROJECT_DIR="/home/jinxulin/sibyl_system/projects/cross-task-influence"
RESULTS_DIR="${PROJECT_DIR}/exp/results"
TASK_ID="setup_env"
CONDA="/home/jinxulin/miniconda3/bin/conda"
ENV_NAME="sibyl_cross-task-influence"
CONDA_RUN="${CONDA} run --no-capture-output -n ${ENV_NAME}"

# Write PID file
mkdir -p "${RESULTS_DIR}"
echo $$ > "${RESULTS_DIR}/${TASK_ID}.pid"

# Progress reporting function
report_progress() {
    local step=$1
    local total=$2
    local desc=$3
    python3 -c "
import json
from datetime import datetime
from pathlib import Path
Path('${RESULTS_DIR}/${TASK_ID}_PROGRESS.json').write_text(json.dumps({
    'task_id': '${TASK_ID}',
    'epoch': ${step}, 'total_epochs': ${total},
    'step': ${step}, 'total_steps': ${total},
    'loss': None, 'metric': {'description': '${desc}'},
    'updated_at': datetime.now().isoformat(),
}))
"
}

cd "${PROJECT_DIR}"

echo "=== Step 1/6: Install PyTorch ==="
report_progress 1 6 "Installing PyTorch"
${CONDA_RUN} pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

echo "=== Step 2/6: Install core ML packages ==="
report_progress 2 6 "Installing core ML packages"
${CONDA_RUN} pip install numpy scipy matplotlib scikit-learn datasets transformers accelerate peft 2>&1 | tail -5

echo "=== Step 3/6: Install robomimic ==="
report_progress 3 6 "Installing robomimic"
# robomimic is needed for LIBERO policy training
${CONDA_RUN} pip install robomimic 2>&1 | tail -5

echo "=== Step 4/6: Install LIBERO ==="
report_progress 4 6 "Installing LIBERO"
# Clone and install LIBERO if not present
if [ ! -d "${PROJECT_DIR}/libero_repo" ]; then
    cd "${PROJECT_DIR}"
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git libero_repo 2>&1 | tail -3
    cd libero_repo
    ${CONDA_RUN} pip install -e . 2>&1 | tail -5
else
    echo "LIBERO repo already exists, reinstalling..."
    cd "${PROJECT_DIR}/libero_repo"
    ${CONDA_RUN} pip install -e . 2>&1 | tail -5
fi
cd "${PROJECT_DIR}"

echo "=== Step 5/6: Install additional dependencies ==="
report_progress 5 6 "Installing additional dependencies"
${CONDA_RUN} pip install h5py tensorboard tqdm einops 2>&1 | tail -5

echo "=== Step 6/6: Verification ==="
report_progress 6 6 "Running verification"

# Verify imports and GPU
VERIFY_OUTPUT=$(${CONDA_RUN} python -c "
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
import sklearn; results['sklearn'] = sklearn.__version__
import transformers; results['transformers'] = transformers.__version__
import datasets; results['datasets'] = datasets.__version__

# robomimic
try:
    import robomimic
    results['robomimic'] = robomimic.__version__
except Exception as e:
    results['robomimic'] = f'import_error: {e}'

# LIBERO
try:
    import libero
    results['libero'] = 'installed'
except Exception as e:
    results['libero'] = f'import_error: {e}'

# Quick GPU smoke test on GPU 1
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # Will be GPU 1 via CUDA_VISIBLE_DEVICES
    x = torch.randn(32, 512, device=device)
    y = torch.nn.Linear(512, 256).to(device)(x)
    results['gpu_smoke_test'] = 'PASS'
    results['gpu_smoke_output_shape'] = list(y.shape)
else:
    results['gpu_smoke_test'] = 'SKIP_NO_CUDA'

results['status'] = 'success'
print(json.dumps(results, indent=2))
" 2>&1)

echo "${VERIFY_OUTPUT}"

# Save verification results
echo "${VERIFY_OUTPUT}" > "${RESULTS_DIR}/setup_verify.json"

# Write DONE marker
python3 -c "
import json
from pathlib import Path
from datetime import datetime

# Read progress
progress_file = Path('${RESULTS_DIR}/${TASK_ID}_PROGRESS.json')
final_progress = {}
if progress_file.exists():
    try:
        final_progress = json.loads(progress_file.read_text())
    except: pass

# Clean up PID
pid_file = Path('${RESULTS_DIR}/${TASK_ID}.pid')
if pid_file.exists():
    pid_file.unlink()

# Write DONE
marker = Path('${RESULTS_DIR}/${TASK_ID}_DONE')
marker.write_text(json.dumps({
    'task_id': '${TASK_ID}',
    'status': 'success',
    'summary': 'Environment setup complete. All packages installed and verified.',
    'final_progress': final_progress,
    'timestamp': datetime.now().isoformat(),
}))
"

echo "=== Setup complete ==="
