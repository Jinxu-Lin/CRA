# MAGIC Feasibility Report

- Model: Pythia-1B (1.0B params)
- Training steps: 200
- Test samples: 5

## Storage
- Checkpoint storage: **1118 GB** (200 steps x 5.6 GB/step)
- INFEASIBLE (limit: 500 GB)

## Compute
- Time per test sample: ~4.0 hours
- Total for 5 samples: ~20 hours (0.8 GPU-days)
- FEASIBLE (budget: 5 GPU-days = 120 hours)

## Verdict: LIKELY INFEASIBLE