# Idea Validation Decision: CRA Pilot Results

## Decision: REFINE

After reviewing all pilot evidence across 4 phases (14 tasks, ~28 min on RTX 4090), cand_a (CRA: Signal Processing Diagnosis + Bilinear Unification) shows genuine promise but has critical methodological gaps that must be addressed before committing full GPU budget.

---

## Candidate Comparison

### cand_a: CRA (Front-runner) -- RETAIN with revisions

**Strengths (reasons to continue):**
1. **FM1 has strong empirical evidence** on attribution tasks: RepSim > TRAK by +32pp (counterfact) and +17pp (ftrace). This is a real, large effect.
2. **H6 confirmed**: K-FAC IF still 17.4pp below RepSim on counterfact. This validates the core diagnostic claim -- FM1/FM2 are not Hessian approximation artifacts.
3. **Spectral evidence is compelling**: Full-model gradient top-5 eigenvalues capture 85.6% variance (vs 34.9% for representations). This is direct, measurable evidence of signal dilution.
4. **TRAK saturation confirmed**: Saturation at k=256 with non-monotonic behavior matches FM1 prediction qualitatively.
5. **Pipeline fully validated**: All 8+ methods work, runtime within budget, 4x RTX 4090 available.
6. **SNR concept directionally validated**: Positive correlation (0.34 counterfact, 0.16 ftrace) despite underdetermined covariance.

**Weaknesses (reasons not to advance yet):**
1. **FM2 has ZERO empirical evidence** (CRITICAL): Contrastive scoring gain is exactly 0.0 across all 12 method-task combinations. Root cause: rank-based metrics (AUPRC, R@K) are invariant to mean-subtraction. This means half the CRA thesis is unvalidated. Full experiments with the same metrics would produce the same zero result -- a waste of GPU budget.
2. **H7 whitened attribution FAILS**: Degrades all tasks by 8-11pp. N/d=0.049 is severely underdetermined. Even at full scale (N=5K, d=2048), N/d=2.5 is marginal.
3. **Toxicity task reversal** (-24pp): TRAK dramatically outperforms RepSim on toxicity, contradicting the universal FM1 claim. Diagnosed as gradient norm artifact (Cohen's d=2.66), but this requires explicit framing as a scope boundary.
4. **H4 quantitative prediction wrong**: r_eff=10 observed vs [256, 1024] predicted. Direction actually strengthens FM1, but the specific numerical prediction was badly calibrated.
5. **H9 falsified**: Condition number direction completely reversed (rep_cond=3.1e10 >> grad_cond=3589). Root cause is rank-deficient covariance at N=100, but the hypothesis as stated is wrong.
6. **30.8pp persistent gap**: TRAK-PCA at k=d still 30.8pp below RepSim, suggesting FM1 alone is necessary but not sufficient. The "smoking gun" test failed.

### cand_b: Hessian Quality Diagnosis -- DROP

H6 was confirmed: K-FAC IF shows a 17.4pp gap with RepSim on counterfact. This directly falsifies the premise of cand_b (that Hessian quality is the primary bottleneck). The gap persists even with high-quality K-FAC eigendecomposition.

**Decision**: Drop. Evidence clearly contradicts cand_b's central hypothesis.

### cand_c: Matched Filter Theory -- DEMOTE to sub-contribution

H7 failed at pilot scale, but SNR concept shows directional validity (positive correlation). The failure is attributable to underdetermined covariance (N/d=0.049), not to a fundamental flaw in the theory. At full scale with PCA-reduced whitening, H7 may recover.

**Decision**: Do not promote to front-runner. Keep as a sub-contribution within cand_a if PCA-reduced whitening works at full scale. If it fails again, frame as an "open direction" in the paper.

---

## Critical Refinements Required Before Full Experiments

### 1. Add Continuous Metrics for FM2 Testing (CRITICAL, blocks full experiment)

**Problem**: Rank-based metrics (AUPRC, R@K) are invariant to mean-subtraction because it's a rank-preserving transformation. This makes FM2 fundamentally untestable with current evaluation.

**Fix**: Add Kendall-tau and Spearman-rho computed on raw attribution scores (not ranks). Also consider score-level NDCG or MSE between predicted and true influence. These continuous metrics will break the rank invariance and allow genuine FM2 testing.

**Impact**: Without this fix, full experiments will produce the same zero FM2 effect, making half the thesis unsubstantiated.

### 2. Revise Hypotheses H4 and H9 (HIGH)

**H4 revision**: Change from "r_eff ~ O(d)" to "r_eff << d << B". The observed r_eff=10 (full model) actually *strengthens* the FM1 argument -- signal is even more concentrated than predicted. Reframe the quantitative prediction to match evidence.

**H9 revision**: Replace condition number comparison with spectral concentration metrics (explained variance ratio, effective dimensionality). The raw condition number is unreliable when N < d.

### 3. Add PCA-Reduced Whitening for H7 (HIGH)

**Problem**: Full whitening with Sigma^{-1} fails when covariance is underdetermined (N/d < 1).

**Fix**: Whiten in the top-k eigenspace only (k ~ r_eff), where covariance estimation is well-conditioned. This is the standard approach in high-dimensional settings.

### 4. Frame Toxicity Reversal as Scope Boundary (MEDIUM)

**Finding**: Toxicity detection is a gradient-norm-sensitive task where parameter-space methods have an inherent advantage (Cohen's d=2.66 between safe/unsafe gradient norms). This is not a CRA failure but a task-type boundary.

**Fix**: Add explicit discussion of when gradient norm is directly informative (binary classification with class-level gradient differences) vs when attribution quality matters (counterfact, ftrace).

### 5. Update Experimental Plan (MEDIUM)

- Increase sample size to N=5K-10K for proper covariance estimation
- Add multi-seed averaging (seeds [42, 123, 456])
- Reframe the "smoking gun" test: FM1 is necessary but not sufficient (curvature, layer selection contribute to the 30.8pp gap)

---

## Risk Assessment

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| FM2 still zero with continuous metrics | Critical | 25% | Would require pivoting to "FM1-only" paper; still publishable but weaker |
| H7 PCA-whitening still fails at full scale | High | 35% | Frame as open direction; drop prescriptive theory contribution |
| Toxicity reversal undermines reviewer confidence | Medium | 40% | Preemptive framing as task-type analysis contribution |
| BM25 competitive at full scale on counterfact | Medium | 30% | Restrict claims; discuss as limitation |

---

## Confidence Analysis

- **Pilot evidence quality**: HIGH -- 14 tasks completed, comprehensive coverage, clean diagnostics
- **FM1 thesis viability**: 0.75 -- strong on attribution tasks, clear spectral evidence
- **FM2 thesis viability**: 0.40 -- completely untested; continuous metrics are necessary but may still show weak effect
- **Overall CRA thesis**: 0.60 -- good diagnostic framework, but needs methodological fixes before full commitment
- **Publication viability (NeurIPS/ICML)**: 0.55 -- if FM2 works with continuous metrics and H7 recovers with PCA-whitening, potential spotlight; if FM2 fails, poster at best with FM1-only story

---

## Next Actions

1. **Revise `hypotheses.md`**: Update H4, H9 to match pilot evidence; add continuous metric requirements for H2/H3
2. **Revise `task_plan.json`**: Add continuous metrics to all Phase 1 tasks; add PCA-reduced whitening variant to Phase 3; increase N to 5K-10K
3. **Revise `proposal.md`**: Add "Evidence-Driven Revisions" section documenting pilot findings and narrative adjustments
4. **Update `methodology.md`**: Add Kendall-tau/Spearman-rho protocol; PCA-whitening procedure; toxicity scope boundary framing
5. **Re-run planner**: Generate updated task plan incorporating all refinements

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.60
DECISION: REFINE
