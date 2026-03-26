#!/bin/bash
# Run baseline reproduction (Phase 2)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== CRA Baseline Reproduction ==="

# TRAK baseline
python experiments/benchmark/run_baseline.py --method trak --task toxicity "$@"

# BM25 baseline
python experiments/benchmark/run_baseline.py --method bm25 --task toxicity "$@"

# Grad-Sim baseline
python experiments/benchmark/run_baseline.py --method gradsim --task toxicity "$@"

echo "=== Baselines Complete ==="
