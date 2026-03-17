# P1: FM1/FM2 Interaction Analysis -- Pilot Summary

## Task
Compute 2-way ANOVA on Kendall tau with Factor A = space (parameter vs representation) and Factor B = scoring (standard vs contrastive) across 3 DATE-LM tasks. Evaluate Decision Gate 1 for FM2 tier assignment.

## Key Results

### ANOVA Summary (Kendall tau)

| Task | FM1 Effect (rep-param) | FM2 Effect (contr-std) | Interaction | eta^2 FM1 |
|------|----------------------|----------------------|-------------|-----------|
| toxicity | -0.0784 | 0.0000 | 0.0000 | 0.7966 |
| counterfact | +0.0712 | 0.0000 | 0.0000 | 0.5445 |
| ftrace | +0.0179 | 0.0000 | 0.0000 | 0.8783 |

### H2-revised: FAILS
- Best Kendall tau gain from contrastive scoring: **0.0000** (threshold: 0.05)
- Root cause: Kendall tau is a rank-correlation metric. Mean subtraction (contrastive scoring) is a monotone transformation that preserves all pairwise orderings, making **ALL** rank-correlation metrics (Kendall tau, Spearman rho, NDCG) mathematically invariant to mean subtraction.
- This is not a measurement artifact -- it is a mathematical identity: tau(s, y) = tau(s - c, y) for any constant c.

### Decision Gate 1: FAILS
- **FM2 demoted to theoretical hypothesis**
- Paper narrows to FM1 (space effect) + systematic benchmark
- H3-revised (FM1xFM2 interaction) is effectively **UNTESTABLE** with any rank-based or rank-correlation metric

### Contamination Injection Assessment
- **Uniform contamination** (adding constant): Zero effect on all metrics (as expected -- monotone transformation)
- **Structured contamination** (alpha=1.0): RepSim counterfact degrades by 35.3pp, RepSim toxicity by 56.0pp
- **Contrastive correction FAILS**: Does not recover performance because non-uniform perturbations change pairwise orderings irreversibly

### FM1 Findings (Positive)
- FM1 (parameter vs representation space) explains 55-88% of variance in Kendall tau across tasks (eta^2 = 0.55-0.88)
- **Toxicity reversal confirmed**: Parameter-space methods (mean tau=0.537) outperform representation-space (0.459) on toxicity
- **Semantic tasks**: Representation-space methods outperform on counterfact (0.152 vs 0.081) and ftrace (0.028 vs 0.010)

## Implications for Full-Scale Experiments
1. FM2-related experiments (contamination injection at full scale) may not yield new insights unless a fundamentally different evaluation metric is used
2. The full-scale 2x2 factorial should still be run for FM1 analysis, but the FM2 axis will remain zero
3. Consider adding score-calibration or regression-based metrics (e.g., MSE against ground-truth scores) that ARE sensitive to additive shifts
4. The paper narrative should acknowledge FM2 as a theoretical hypothesis that cannot be validated with current evaluation protocols

## Pass Criteria
- ANOVA computable: YES
- Nonzero FM1 main effects: 3/3 tasks
- Interaction < 30% of min(main): VACUOUS (FM2 effect = 0)
- **Verdict: PASS_WITH_CAVEAT** -- FM1 analysis is informative; FM2 analysis is vacuous
