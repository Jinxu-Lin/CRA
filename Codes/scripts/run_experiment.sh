#!/bin/bash
# Wrapper script: set seed, log GPU info, run command.
# Usage: scripts/run_experiment.sh python run_attribution.py --config configs/base.yaml

set -euo pipefail

echo "=========================================="
echo "CRA Experiment Runner"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Host: $(hostname)"
echo "User: $(whoami)"

# GPU info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Python info
echo ""
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"

# Git info
echo ""
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
echo ""
echo "Command: $@"
echo "=========================================="

# Run the command
"$@"

echo ""
echo "=========================================="
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="
