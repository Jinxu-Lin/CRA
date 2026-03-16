# Phase 3a: Contrastive Scoring Universal Plug-in Matrix -- Pilot Summary

## Task
36-cell experiment: {RepSim, TRAK, LoGra, DDA} x {standard, contrastive, whitened} x {toxicity, counterfact, ftrace}
on Pythia-1B with DATE-LM benchmark. PILOT mode (n=100, seed=42).

## Key Finding: Mean-Subtraction is Rank-Preserving at Pilot Scale

**Contrastive scoring (mean-subtraction) produces exactly zero change in rank-based metrics (AUPRC, Recall@K) for ALL 4 methods on ALL 3 tasks.** This is a mathematical property: subtracting a per-column constant from the similarity matrix preserves the rank ordering of training samples for each reference query. Since AUPRC and Recall@K depend only on rank ordering, the metrics are invariant.

This confirms the Phase 1 finding. Contrastive scoring requires continuous metrics (Kendall-tau, Spearman) or larger-scale experiments where the shift breaks ties.

## Primary Metric Results (Rank-Based)

| Method | Scoring | toxicity (AUPRC) | counterfact (R@50) | ftrace (R@50) |
|--------|---------|------------------|--------------------|---------------|
| RepSim | standard | 0.8092 | 0.9487 | 0.6605 |
| RepSim | contrastive | 0.8092 | 0.9487 | 0.6605 |
| RepSim | whitened | 0.8122 | 0.9263 | 0.6760 |
| TRAK | standard | 0.9256 | 0.6699 | 0.5902 |
| TRAK | contrastive | 0.9256 | 0.6699 | 0.5902 |
| TRAK | whitened | 0.8782 | 0.6026 | 0.5937 |
| LoGra | standard | 0.9222 | 0.7981 | 0.5829 |
| LoGra | contrastive | 0.9222 | 0.7981 | 0.5829 |
| LoGra | whitened | 0.9179 | 0.7756 | 0.6068 |
| DDA | standard | 0.8764 | 0.6923 | 0.6514 |
| DDA | contrastive | 0.8764 | 0.6923 | 0.6514 |
| DDA | whitened | 0.8764 | 0.6923 | 0.7192 |

## Whitened Scoring: Mixed Results

Whitened scoring (Ledoit-Wolf shrinkage + inverse-sqrt covariance) shows a clear pattern:

- **ftrace**: Whitened improves ALL 4 methods (RepSim +1.55pp, TRAK +0.35pp, LoGra +2.39pp, DDA +6.78pp)
- **toxicity**: Whitened slightly improves RepSim (+0.30pp) but hurts TRAK (-4.74pp)
- **counterfact**: Whitened hurts RepSim (-2.24pp) and TRAK (-6.73pp)

This suggests whitened scoring helps when the task has higher noise contamination (ftrace) but can hurt when standard scoring is already effective.

## Continuous Metrics (Kendall-tau)

| Method | Scoring | toxicity (tau) | counterfact (tau) | ftrace (tau) |
|--------|---------|----------------|-------------------|--------------|
| RepSim | standard | 0.4648 | 0.1433 | 0.0301 |
| RepSim | whitened | 0.4072 | 0.1348 | 0.0914 |
| TRAK | standard | 0.5280 | 0.0556 | 0.0114 |
| TRAK | whitened | 0.4932 | 0.0442 | 0.0140 |
| LoGra | standard | 0.5344 | 0.0764 | 0.0139 |
| LoGra | whitened | 0.5366 | 0.0734 | 0.0246 |
| DDA | standard | 0.5046 | 0.0519 | 0.0085 |
| DDA | whitened | 0.5046 | 0.0519 | 0.0379 |

Contrastive scoring produces zero tau change (rank-preserving). Whitened scoring improves tau on ftrace for all methods but degrades it on toxicity for RepSim and TRAK.

## Pass Criteria Assessment

**Criterion**: Contrastive scoring improves >= 3 of 4 methods on >= 2 of 3 tasks; no method degrades by > 3pp.

**Result**: NOT MET for contrastive scoring (zero improvement due to rank-invariance at pilot scale).

**Whitened scoring** partially meets the spirit:
- ftrace: 4/4 methods improve (tau)
- toxicity: 1/4 methods improve (tau), 2/4 degrade
- counterfact: 0/4 methods improve (tau)

**No method degrades by > 3pp** with contrastive scoring (exactly 0pp change). TRAK whitened degrades on toxicity (-4.74pp AUPRC) and counterfact (-6.73pp R@50).

## Observations

1. **Contrastive scoring is a no-op for rank-based evaluation at pilot scale**. This is fundamental, not a bug. The FM2 hypothesis needs testing with: (a) continuous metrics at larger scale, (b) score-calibration-sensitive evaluations, or (c) downstream task performance.

2. **LoGra outperforms TRAK on counterfact** (R@50: 0.798 vs 0.670, +12.8pp). The structured SVD projection captures more signal than random CountSketch.

3. **DDA whitened shows the largest improvement on ftrace** (+6.78pp R@50). DDA's debias step (mean-subtraction in gradient space) is already a partial FM2 fix; whitening adds further noise suppression.

4. **Ledoit-Wolf shrinkage = 1.0 for LoGra and DDA** features (dimension 86-99 << n_train=100). This means the whitened covariance is dominated by the shrinkage target (identity), so whitening acts primarily as centering + rescaling.

5. **TRAK Ledoit-Wolf shrinkage is high (0.70-0.89)** for the projected gradient features (dim=2048, n=100). The random projection preserves some structure but the high shrinkage indicates the covariance estimate is unreliable.

## Recommendation for Full-Scale

1. Add Kendall-tau and Spearman as primary metrics alongside AUPRC/Recall@K
2. Increase to full dataset to break rank ties and expose score calibration effects
3. Test per-query contrastive gain (some queries may benefit more than others)
4. Consider MRR as a more sensitive continuous metric for factual tasks

## Runtime
Total: 543.6s (~9 min) on 1x RTX 4090.
