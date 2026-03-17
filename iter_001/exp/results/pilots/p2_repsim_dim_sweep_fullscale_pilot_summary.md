# P2: RepSim PCA Dimension Sweep -- Pilot Summary

## Configuration
- Model: EleutherAI/pythia-1b (d=2048)
- N_train: 100 (pilot)
- PCA dims: [16, 32, 64, 128, 256, 512, 1024, 2048]
- Metrics: R@50/AUPRC, MRR, Kendall tau, Spearman rho

## counterfact (Recall@50, full_dim=0.9936, tau=0.1704)
| k | Recall@50 | Drop(pp) | Retention% | Kendall tau | tau_drop |
|---|---------|----------|-----------|-------------|---------|
| 16 | 1.0000 | -0.64 | 100.7 | 0.1714 | -0.0010 | **knee**
| 32 | 1.0000 | -0.64 | 100.7 | 0.1738 | -0.0034 |
| 64 | 1.0000 | -0.64 | 100.7 | 0.1749 | -0.0045 |
| 128 | 1.0000 | -0.64 | 100.7 | 0.1750 | -0.0046 |
| 256 | 0.9936 | +0.00 | 100.0 | 0.1704 | +0.0000 |
| 512 | 0.9936 | +0.00 | 100.0 | 0.1704 | +0.0000 |
| 1024 | 0.9936 | +0.00 | 100.0 | 0.1704 | +0.0000 |
| 2048 | 0.9936 | +0.00 | 100.0 | 0.1704 | +0.0000 |

## toxicity (AUPRC, full_dim=0.6852, tau=0.3461)
| k | AUPRC | Drop(pp) | Retention% | Kendall tau | tau_drop |
|---|---------|----------|-----------|-------------|---------|
| 16 | 0.7233 | -3.82 | 105.6 | 0.3660 | -0.0199 | **knee**
| 32 | 0.7242 | -3.91 | 105.7 | 0.3681 | -0.0220 |
| 64 | 0.7277 | -4.25 | 106.2 | 0.3759 | -0.0298 |
| 128 | 0.6852 | +0.00 | 100.0 | 0.3461 | +0.0000 |
| 256 | 0.6852 | +0.00 | 100.0 | 0.3461 | +0.0000 |
| 512 | 0.6852 | +0.00 | 100.0 | 0.3461 | +0.0000 |
| 1024 | 0.6852 | +0.00 | 100.0 | 0.3461 | +0.0000 |
| 2048 | 0.6852 | +0.00 | 100.0 | 0.3461 | +0.0000 |

## ftrace (Recall@50, full_dim=0.7563, tau=0.1174)
| k | Recall@50 | Drop(pp) | Retention% | Kendall tau | tau_drop |
|---|---------|----------|-----------|-------------|---------|
| 16 | 0.7681 | -1.18 | 101.5 | 0.1340 | -0.0166 | **knee**
| 32 | 0.7745 | -1.81 | 102.4 | 0.1341 | -0.0168 |
| 64 | 0.8031 | -4.68 | 106.2 | 0.1363 | -0.0189 |
| 128 | 0.7910 | -3.46 | 104.6 | 0.1382 | -0.0209 |
| 256 | 0.7563 | +0.00 | 100.0 | 0.1174 | +0.0000 |
| 512 | 0.7563 | +0.00 | 100.0 | 0.1174 | +0.0000 |
| 1024 | 0.7563 | +0.00 | 100.0 | 0.1174 | +0.0000 |
| 2048 | 0.7563 | +0.00 | 100.0 | 0.1174 | +0.0000 |

## Analysis
- Knee points (97% retention): {'toxicity': 16, 'counterfact': 16, 'ftrace': 16}
- Min k within 3pp: {'toxicity': 128, 'counterfact': 16, 'ftrace': 16}
- H4 cross-validation: {'toxicity': {'knee_k': 16, 'knee_ratio': 0.008, 'in_expected_range': False, 'note': 'knee at k=16 = 0.008*d'}, 'counterfact': {'knee_k': 16, 'knee_ratio': 0.008, 'in_expected_range': False, 'note': 'knee at k=16 = 0.008*d'}, 'ftrace': {'knee_k': 16, 'knee_ratio': 0.008, 'in_expected_range': False, 'note': 'knee at k=16 = 0.008*d'}}
- Pass criteria: {'full_dim_matches_within_1pp': True, 'graceful_degradation': True, 'knee_in_expected_range': False}

## Pilot Limitations
- N=100 caps PCA max_components to ~129 (varies by task pool size); k>max_components falls back to full d=2048
- At pilot scale saturation may appear earlier (fewer samples = lower effective rank)
- Bootstrap CIs may be wide with small ref sets

## Runtime: 409.3s