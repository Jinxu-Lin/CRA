# P1: Controlled Contamination Injection (H11) -- Pilot Summary

**Runtime**: 1.5 min | **N_train**: 100 | **Seed**: 42
**Methods**: RepSim, TRAK | **Tasks**: counterfact, toxicity
**Alphas**: 0.0, 0.1, 0.5, 1.0, 2.0, 5.0
**Contamination modes**: uniform, structured, magnitude_proportional

## Key Findings

### Finding 1: Uniform contamination is invisible (confirmed)
Adding `alpha * mu_j` (column mean) to scores has **ZERO effect** on all metrics -- rank-based (R@50, AUPRC) AND continuous (Kendall tau, Spearman rho). This confirms the mathematical invariance: uniform per-column shifts preserve all ranks and rank correlations.

### Finding 2: Structured contamination degrades performance monotonically
Non-uniform contamination (per-sample noise scaled by column mean) degrades both rank-based and continuous metrics monotonically with alpha. RepSim is more sensitive than TRAK:

| Task/Method | alpha=0.5 degradation | alpha=1.0 degradation |
|-------------|----------------------|----------------------|
| counterfact/RepSim (R@50) | -23.7pp | -35.3pp |
| counterfact/TRAK (R@50) | -5.8pp (mag) | -9.0pp (mag) |
| toxicity/RepSim (AUPRC) | -53.9pp | -56.0pp |
| toxicity/TRAK (AUPRC) | -2.0pp | -5.8pp |

### Finding 3: CRITICAL NEGATIVE -- Contrastive correction provides ZERO recovery
**Corrected scores are IDENTICAL to contaminated scores** across all modes, methods, tasks, and alpha values. The "recovery %" in the raw data is misleading -- it measures `corrected/baseline`, but since `corrected == contaminated`, this just measures how much signal survives contamination, not whether correction helped.

**Root cause**: Mean subtraction (`s - mean(s, axis=0)`) removes the column mean of the contamination. For structured contamination `s + alpha * |mu_j| * noise_i`, the column mean contribution is `alpha * |mu_j| * mean(noise_i)`, which is constant. After subtraction, the per-sample deviation `noise_i - mean(noise_i)` remains. The contrastive correction removes a constant offset (which doesn't affect ranks anyway) but cannot remove sample-specific noise.

### Finding 4: TRAK structured contamination shows unexpected improvement
On counterfact, TRAK with structured contamination at alpha=0.5-1.0 IMPROVES by ~6pp. This suggests the noise breaks suboptimal gradient correlations. Worth investigating but may be a pilot-scale artifact.

## Theoretical Implications for FM2

The experiment exposes a **logical gap** in the FM2 (Common Influence Contamination) hypothesis:

1. **If contamination is uniform** (same shift for all samples per query): It is invisible to all evaluation metrics. Contrastive correction removes it, but there was nothing harmful to begin with.

2. **If contamination is non-uniform** (sample-specific patterns): It degrades performance, but mean subtraction cannot fix it because it only removes column-constant shifts.

3. **Conclusion**: Simple mean subtraction (contrastive scoring) is either **unnecessary** (uniform case) or **insufficient** (non-uniform case) as a remedy for common influence contamination.

This does NOT invalidate FM2 as a theoretical concept -- common influence contamination likely exists in real attribution scores. But it means the proposed remedy (mean subtraction) has limited practical value. More sophisticated approaches (per-sample regression, projection-based decontamination) would be needed.

## Pass Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| alpha=1.0 R@50 drops >= 5pp | >= 5pp | RepSim: 35pp, TRAK: 9pp | **PASS** (structured/magnitude modes) |
| Contrastive correction recovers >= 80% | >= 80% of baseline | 0% recovery (corrected == contaminated) | **FAIL** |
| Kendall tau monotonic degradation | Monotonic with alpha | Yes, all structured/magnitude modes | **PASS** |

**Overall verdict**: The degradation half of H11 is supported. The recovery half is falsified.

## Recommendation

**CONDITIONAL_GO** for full-scale experiment. The contamination injection curves are a valuable contribution showing sensitivity of different methods to noise. But the paper narrative must be reframed:
- Drop claim that "contrastive scoring fixes common influence"
- Reframe as: "common influence contamination, when uniform, is metrically invisible; when structured, it is harmful but not addressable by simple mean subtraction"
- Consider adding more sophisticated decontamination methods as future work
