# Final Analysis: CRA Pilot Results Summary

## Decision Gates

### Gate 1: FM2 Tier Assignment -- FAILED
- Contrastive scoring (mean subtraction) is a monotone transformation
- ALL rank-correlation metrics (Kendall tau, Spearman rho, NDCG) are mathematically invariant to it
- FM2 **demoted to theoretical hypothesis**; paper narrows to FM1 + benchmark

### Gate 2: Retrieval Boundary -- NOT TRIGGERED (but critical concerns)
- Contriever exceeds RepSim on toxicity by +15.9pp (0.968 vs 0.809 AUPRC)
- GTR-T5 exceeds RepSim on ftrace by +7.6pp (0.832 vs 0.756 R@50)
- BM25 at R@50=1.0 on counterfact (pilot scale ceiling)
- Paper must honestly address attribution-vs-retrieval overlap

### Gate 3: Gap Decomposition -- FAILED
- Combined (last-layer + cosine-norm) closes only 6.4pp of 30.1pp gap (21.3%)
- Residual 23.7pp = nonlinear semantic features
- FM1 is necessary but grossly insufficient

### Gate 4: PCA Whitening -- PASSED (task-specific)
- Toxicity: +30.1pp (standard 0.685 -> whitened 0.987 AUPRC)
- FLAG: >30% improvement needs full-scale validation
- Counterfact/ftrace: negligible gain
- Matched filter theory rescued for behavioral detection tasks only

## Hypothesis Scorecard

| # | Hypothesis | Status | Tier |
|---|-----------|--------|------|
| H1 | RepSim > TRAK on semantic tasks | SUPPORTED | 1 |
| H2-rev | Continuous metrics detect FM2 | FAILED (math impossibility) | Negative |
| H3-rev | FM1 x FM2 interaction | UNTESTABLE | Gap |
| H4-rev | Gradient r_eff << d << B | SUPPORTED (r_eff/B = 5.58e-6) | 1 |
| H5-rev | TRAK saturates at k ~ d/8 | SUPPORTED (peak at k=256) | 1 |
| H7-rev | PCA-whitened > standard RepSim | POSITIVE (toxicity only, +30pp) | 2 |
| H8-rev | Retrieval gap > 5pp vs RepSim | MIXED (task-dependent) | 1 |
| H9-rev | Concentration ratio > 10x | STRONGLY SUPPORTED (65,801x) | 1 |
| H10 | Gap decomposes into factors | PARTIAL (21% explained) | 2 |
| H11 | Contamination injection + recovery | PARTIAL (degradation yes, recovery no) | 2 |

## Paper Positioning

### Tier 1 (Strong evidence)
1. **FM1 Spectral Diagnosis**: 65,801x concentration ratio, r_eff/B = 5.58e-6
2. **Task-Type Boundary**: Semantic (rep space wins) vs behavioral (param space wins)
3. **Systematic Benchmark**: 8+ methods, 3 tasks, retrieval baselines, dual metrics

### Tier 2 (Under investigation)
4. **PCA-Whitened Attribution**: +30pp on toxicity (needs full-scale validation)
5. **Gap Decomposition**: 21% engineering, 79% feature quality
6. **FM2 Evaluation Barrier**: Methodological negative result

### Negative Results
- FM2 undetectable with rank-correlation metrics (mathematical identity)
- Contrastive correction fails on structured contamination
- Retrieval baselines competitive on toxicity (Contriever 0.968)
- H9 original (isotropy) falsified
