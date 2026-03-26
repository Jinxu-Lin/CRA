#!/bin/bash
# Run all CRA experiments in sequence.
# Usage: scripts/run_all.sh [--dry-run]
set -euo pipefail

cd "$(dirname "$0")/.."

ARGS="${@}"

echo "============================================"
echo "CRA Full Experiment Pipeline"
echo "Args: $ARGS"
echo "============================================"

# Phase 0: Sanity checks
echo ""
echo "=== Phase 0: Sanity Checks ==="
python scripts/sanity_check.py --check all

# Phase 1: Probe (Critical Gate)
echo ""
echo "=== Phase 1: Probe ==="
python run_attribution.py --config configs/probe_repsim.yaml $ARGS
python run_attribution.py --config configs/probe_gradsim.yaml $ARGS
python experiments/probe/run_trak.py --config configs/probe_trak.yaml $ARGS
python run_attribution.py --config configs/probe_random_repsim.yaml --random-init $ARGS
python experiments/probe/evaluate.py --config configs/probe_repsim.yaml $ARGS

# Phase 1.5: Pilot 2x2
echo ""
echo "=== Phase 1.5: Pilot 2x2 ==="
python run_attribution.py --config configs/pilot_contrastive_repsim.yaml $ARGS
python run_attribution.py --config configs/pilot_contrastive_trak.yaml $ARGS
python run_attribution.py --config configs/pilot_contrastive_gradsim.yaml $ARGS
python experiments/pilot/evaluate_2x2.py --config configs/pilot_contrastive_repsim.yaml $ARGS

# Phase 2: Baselines
echo ""
echo "=== Phase 2: Baseline Reproduction ==="
bash scripts/run_baseline.sh $ARGS

# Phase 3: Main experiments
echo ""
echo "=== Phase 3: Main Experiments ==="
bash scripts/run_main.sh $ARGS
bash scripts/run_ablation.sh $ARGS

# LoRA vs Full-FT
python experiments/lora_vs_ft/run_comparison.py --config configs/lora_vs_ft.yaml $ARGS
python experiments/lora_vs_ft/analyze_ft.py --config configs/lora_vs_ft.yaml $ARGS

# Phase 4: MAGIC
echo ""
echo "=== Phase 4: MAGIC Feasibility ==="
python experiments/magic/feasibility_check.py --config configs/magic.yaml $ARGS

# Compile results
echo ""
echo "=== Compiling Results ==="
python scripts/compile_results.py --results-dir _Results

echo ""
echo "============================================"
echo "CRA Full Pipeline Complete"
echo "============================================"
