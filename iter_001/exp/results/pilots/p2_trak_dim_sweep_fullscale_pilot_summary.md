# P2: TRAK Dimension Sweep (H5-revised) -- Pilot Summary

## Configuration
- Model: EleutherAI/pythia-1b (d=2048)
- N_train: 100 (pilot), N_ref: 66
- Target layers: layers.15.attention.dense.weight, layers.15.mlp.dense_4h_to_h.weight
- Grad dim D: 20,971,520
- Random projection k: [32, 64, 128, 256, 512, 1024, 2048, 4096]
- PCA projection k: [32, 64, 128, 256, 512, 1024, 2048]

## Random Projection Results
| k | k/d | Recall@50 | MRR | Kendall tau |
|---|-----|-----------|-----|-------------|
| 32 | 0.016 | 0.6603 | 0.1928 | 0.0458 |
| 64 | 0.031 | 0.6859 | 0.2008 | 0.0489 |
| 128 | 0.062 | 0.7051 | 0.2036 | 0.0506 |
| 256 | 0.125 | 0.7853 | 0.2268 | 0.0658 |
| 512 | 0.250 | 0.7500 | 0.2169 | 0.0589 |
| 1024 | 0.500 | 0.6859 | 0.2237 | 0.0582 |
| 2048 | 1.000 | 0.6699 | 0.2399 | 0.0556 |
| 4096 | 2.000 | 0.7147 | 0.2564 | 0.0614 |

## PCA Projection Results
| k | k/d | Recall@50 | MRR | Kendall tau | Explained Var |
|---|-----|-----------|-----|-------------|---------------|
| 32 | 0.016 | 0.6026 | 0.1869 | 0.0359 | 0.6446 |
| 64 | 0.031 | 0.6987 | 0.1839 | 0.0510 | 0.8873 |
| 128 | 0.062 | 0.6859 | 0.2241 | 0.0572 | 1.0000 |
| 256 | 0.125 | 0.6859 | 0.2241 | 0.0572 | 1.0000 |
| 512 | 0.250 | 0.6859 | 0.2241 | 0.0572 | 1.0000 |
| 1024 | 0.500 | 0.6859 | 0.2241 | 0.0572 | 1.0000 |
| 2048 | 1.000 | 0.6859 | 0.2241 | 0.0572 | 1.0000 |

## PCA vs Random Comparison
| k | PCA R@50 | Random R@50 | PCA advantage (pp) |
|---|---------|------------|-------------------|
| 32 | 0.6026 | 0.6603 | -5.77 |
| 64 | 0.6987 | 0.6859 | +1.28 |
| 128 | 0.6859 | 0.7051 | -1.92 |
| 256 | 0.6859 | 0.7853 | -9.94 |
| 512 | 0.6859 | 0.7500 | -6.41 |
| 1024 | 0.6859 | 0.6859 | +0.00 |
| 2048 | 0.6859 | 0.6699 | +1.60 |

## Smoking-Gun Test (TRAK-PCA at k=d vs RepSim)
- TRAK-PCA(k=d=2048) Recall@50: 0.685897
- RepSim Recall@50: 0.99359
- Gap: 30.77pp
- Interpretation: GAP PERSISTS: TRAK-PCA(k=d) differs from RepSim by 30.8pp. FM1 alone does not explain the performance gap.

## H5 Assessment
- Saturation at k=256 (k/d=0.125)
- PCA wins: 2/7
- Overall: PASS

## Pilot Limitations
- N=100 caps PCA rank at 100, making PCA projections at k>100 identical (all project to rank-100 space)
- Random projection Recall@50 may be noisy at pilot scale
- Bootstrap CIs from 66 ref queries may be wide

## Runtime
- Total: 613.2s
- GPU: NVIDIA GeForce RTX 4090