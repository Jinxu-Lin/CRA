# Phase 3b: Whitened Matched Filter Attribution -- Pilot Summary

## Task: phase3_whitened_attribution
**Status**: GO_WITH_CAVEATS | **H7**: FAIL | **Confidence**: 0.45

## Key Result

Whitened RepSim (phi^T Sigma_noise^{-1} psi) **degraded** performance on all 3 DATE-LM tasks compared to standard RepSim:

| Task | Standard | Whitened | Gain (pp) | Contrastive | Wht+Ctr |
|------|----------|----------|-----------|-------------|---------|
| toxicity (AUPRC) | 0.6852 | 0.5758 | **-10.94** | 0.6852 | 0.5758 |
| counterfact (R@50) | 0.9936 | 0.9135 | **-8.01** | 0.9936 | 0.9135 |
| ftrace (R@50) | 0.7563 | 0.6500 | **-10.63** | 0.7563 | 0.6500 |

**H7 target**: whitened > standard by 3-8pp on factual_attribution. **Actual**: -10.63pp.

## Root Cause Analysis

The failure is attributable to the severely underdetermined covariance estimation (N=100, d=2048):

1. **N/d ratio = 0.049**: The sample covariance has rank at most 99 in a 2048-dim space. Even with Ledoit-Wolf shrinkage, the inverse amplifies estimation error.
2. **Heavy shrinkage**: Coefficients ranged from 0.12 (toxicity) to 0.40 (ftrace), indicating the estimator is heavily regularized toward the identity -- yet the residual estimation error in the inverse still dominates.
3. **Score scale distortion**: Whitened scores are in the thousands (mean ~2000-3500) with very small relative standard deviation (~3-8%), making ranking fragile to estimation noise.

## Positive Signals

Despite the overall failure, several diagnostics are encouraging:

- **No numerical issues**: Zero NaN/Inf values across all tasks
- **Ledoit-Wolf convergence**: Successful on all tasks, shrinkage coefficients reasonable
- **Per-query SNR concept validated**: Positive correlation between SNR and attribution accuracy (r=0.34 on counterfact), confirming the theoretical motivation
- **Large ranking changes**: Whitening substantially reranks samples (deltas up to 87 positions), showing it captures genuinely different information

## Ledoit-Wolf Diagnostics

| Task | Shrinkage | Condition | r_eff(95%) |
|------|-----------|-----------|------------|
| toxicity | 0.117 | 3,910 | 1,171 |
| counterfact | 0.288 | 515 | 1,693 |
| ftrace | 0.395 | 272 | 1,789 |

## Per-Query SNR Analysis

| Task | Mean SNR | Median SNR | Corr(SNR, accuracy) |
|------|----------|------------|---------------------|
| counterfact | 12.59 | 12.30 | 0.336 |
| ftrace | 1.09 | 0.35 | 0.165 |

The lower SNR on ftrace correlates with the larger performance degradation, consistent with the matched filter theory.

## Recommendations for Full Experiment

1. **Increase N**: Use N >= 5000 for stable covariance estimation (N/d > 2.5)
2. **PCA-reduced whitening**: Project to top-k PCA dims first, whiten in lower-dim space
3. **Shrinkage sweep**: Optimize regularization strength rather than using Ledoit-Wolf default
4. **Oracle upper bound**: Use held-out data for Sigma estimation to establish maximum benefit
5. **Paper framing**: Position whitening as a theoretically motivated direction that requires careful regularization, not a plug-and-play improvement. The negative pilot result is itself informative.

## Contrastive Scoring Note

Contrastive scoring (mean-subtraction) produced **identical** results to standard for all tasks. This is consistent with the Phase 1 finding: for cosine-similarity based RepSim with L2-normalized representations, mean-subtraction preserves ranking invariance when the similarity function is symmetric in the bias direction. (Mean-subtraction changes absolute values but not per-query ranking when all training samples shift by the same per-query constant.)
