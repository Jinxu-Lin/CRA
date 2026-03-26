#!/bin/bash
# Run 2x2 ablation (Experiment 2)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== CRA 2x2 Ablation ==="

# Run all conditions
python experiments/ablation_2x2/run_ablation.py --config configs/ablation_2x2.yaml --seeds 3 "$@"

# Analyze
python experiments/ablation_2x2/analyze_ablation.py --config configs/ablation_2x2.yaml "$@"

echo "=== Ablation Complete ==="
