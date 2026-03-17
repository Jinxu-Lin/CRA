# P5: PCA-Reduced Whitened Attribution -- Pilot Summary (H7-revised)

## Configuration
- Model: EleutherAI/pythia-1b (d=2048)
- N_train: 100 (pilot)
- PCA dims: [16, 32, 64, 128, 256, 512]
- Methods: standard_repsim (M=I), pca_whitened (Ledoit-Wolf), ridge_cv (5-fold CV)

## Decision Gate
- **Condition**: PCA-whitened at any k outperforms standard RepSim by >= 3pp on any task
- **Result**: PASS
- **Best improvement**: {'task': 'toxicity', 'k': 32, 'gain_pp': 30.14, 'method': 'lw'}

## Pass Criteria
- k=64 no >5pp degradation: **PASS**
- Positive SNR-accuracy correlation: **PASS**
- Any method +3pp improvement: **PASS**

## Comparison Table (Standard vs PCA-Whitened vs Ridge-CV)

### counterfact (Recall@50, full_dim_std=0.9936)
| k | N/k | Standard PCA | Whitened (LW) | Ridge CV | Gain(LW) | Gain(Ridge) |
|---|-----|-------------|---------------|----------|----------|-------------|
| 16 | 6.2 | 1.0000 | 1.0000 | 1.0000 | +0.00pp | +0.00pp |
| 32 | 3.1 | 1.0000 | 1.0000 | 1.0000 | +0.00pp | +0.00pp |
| 64 | 1.6 | 1.0000 | 1.0000 | 1.0000 | +0.00pp | +0.00pp |
| 128 | 0.8 | 0.9936 | SKIP | SKIP | N/A | N/A |
| 256 | 0.4 | 0.9936 | SKIP | SKIP | N/A | N/A |
| 512 | 0.2 | 0.9936 | SKIP | SKIP | N/A | N/A |

### toxicity (AUPRC, full_dim_std=0.6852)
| k | N/k | Standard PCA | Whitened (LW) | Ridge CV | Gain(LW) | Gain(Ridge) |
|---|-----|-------------|---------------|----------|----------|-------------|
| 16 | 6.2 | 0.7147 | 0.9385 | 0.9389 | +22.38pp | +22.42pp |
| 32 | 3.1 | 0.7245 | 0.9866 | 0.9839 | +26.21pp | +25.94pp |
| 64 | 1.6 | 0.7270 | 0.9705 | 0.9742 | +24.35pp | +24.72pp |
| 128 | 0.8 | 0.6852 | SKIP | SKIP | N/A | N/A |
| 256 | 0.4 | 0.6852 | SKIP | SKIP | N/A | N/A |
| 512 | 0.2 | 0.6852 | SKIP | SKIP | N/A | N/A |

### ftrace (Recall@50, full_dim_std=0.7563)
| k | N/k | Standard PCA | Whitened (LW) | Ridge CV | Gain(LW) | Gain(Ridge) |
|---|-----|-------------|---------------|----------|----------|-------------|
| 16 | 6.2 | 0.7925 | 0.7888 | 0.7956 | -0.37pp | +0.32pp |
| 32 | 3.1 | 0.7798 | 0.7894 | 0.7862 | +0.96pp | +0.64pp |
| 64 | 1.6 | 0.7888 | 0.7938 | 0.7900 | +0.50pp | +0.12pp |
| 128 | 0.8 | 0.7563 | SKIP | SKIP | N/A | N/A |
| 256 | 0.4 | 0.7563 | SKIP | SKIP | N/A | N/A |
| 512 | 0.2 | 0.7563 | SKIP | SKIP | N/A | N/A |

## SNR Analysis

### counterfact (best_k=16)
- Mean SNR: 3.6906
- Corr(SNR, acc_whitened): 0.0000
- Corr(SNR, acc_standard): 0.0000
- n_queries: 25

### ftrace (best_k=64)
- Mean SNR: 3.7608
- Corr(SNR, acc_whitened): 0.1508
- Corr(SNR, acc_standard): 0.0950
- n_queries: 22

## Pilot Limitations
- N=100 severely limits covariance estimation quality; N/k ranges from 6.2 to 0.2
- Only k<=99 can be estimated at all; larger k fall back to full dim
- Ridge CV with small ref sets may overfit to CV folds
- SNR analysis limited by small number of queries

## Runtime: 32.1s