# P4 Gap Decomposition Pilot Summary

**Task**: p4_gap_decomposition
**Model**: EleutherAI/pythia-1b (hidden_dim=2048, num_layers=16)
**Mode**: PILOT (N=100)
**Runtime**: 295s
**Date**: 2026-03-17

## Gap Decomposition Results (Recall@50)

| Configuration | R@50 | Gap to RepSim (pp) | Gap Reduction (pp) | Kendall tau |
|---|---|---|---|---|
| RepSim (reference) | 0.9936 | 0.0 | -- | 0.1704 |
| Standard TRAK-PCA (baseline) | 0.6923 | 30.1 | 0.0 | 0.0578 |
| (a) Last-layer-only | 0.7051 | 28.8 | +1.3 | 0.0797 |
| (b) Cosine-normalized | 0.7051 | 28.8 | +1.3 | 0.0621 |
| (c) Combined (a+b) | 0.7564 | 23.7 | +6.4 | 0.0858 |
| (d) Residual | -- | 23.7 | -- | -- |
| Raw dot product (no PCA) | 0.6923 | 30.1 | +0.0 | 0.0578 |

## Pass Criteria
- Factor (a) or (b) >= 5pp gap reduction: **FAIL** (a=+1.3pp, b=+1.3pp)
- Combined >= 10pp gap reduction: **FAIL** (+6.4pp)
- Overall: **REFINE**

## Gradient Statistics
| Config | Grad Dim | Mean Norm | CV(Norm) |
|---|---|---|---|
| Standard (attn+mlp) | 20,971,520 | 27.477880 | 0.3871 |
| Last-layer (all params) | 50,358,272 | 36.277130 | 0.3514 |

## Pilot Limitations
- N=100 caps PCA rank at 100
- Full-scale (N=5473) will have higher effective rank for PCA