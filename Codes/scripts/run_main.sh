#!/bin/bash
# Run main benchmark (Experiment 1)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== CRA Main Benchmark ==="

# Toxicity filtering
python experiments/benchmark/run_all.py --config configs/benchmark_toxicity.yaml --task toxicity --seeds 3 "$@"

# Data selection
python experiments/benchmark/run_all.py --config configs/benchmark_selection.yaml --task selection --seeds 3 "$@"

# Factual attribution
python experiments/benchmark/run_all.py --config configs/benchmark_factual.yaml --task factual --seeds 3 "$@"

# Layer sweep
python experiments/benchmark/run_layer_sweep.py --config configs/benchmark_layer_sweep.yaml "$@"

# Efficiency profiling
python experiments/benchmark/profile_efficiency.py --config configs/benchmark_toxicity.yaml "$@"

# Statistical analysis
python experiments/benchmark/statistical_analysis.py --config configs/benchmark_toxicity.yaml "$@"

echo "=== Main Benchmark Complete ==="
