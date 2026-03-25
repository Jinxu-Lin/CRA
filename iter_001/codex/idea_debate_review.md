# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-17
**模型**: Codex (GPT-5)

## 评审意见

### 1. Overall Assessment (7/10)

Well-conceived as a staged, falsification-first program with clear decision gates. Main issue: the causal claims may outrun what the proposed proxies and budget can support.

### 2. Strengths

- Strong phase-gated design (detect -> measure -> intervene) with pre-registered thresholds.
- Good balance of cheap diagnostics (BCS), stronger proxy (GPTA), and actionable optimizer (LP mixing).
- Explicit attention to negative transfer, significance control, and ID/OOD evaluation.
- Debate surfaces real failure modes (influence fragility, architecture confounds, non-additivity).

### 3. Weaknesses and Blind Spots (underweighted by all six perspectives)

- **Causality gap**: correlations between proxy scores and LOTO outcomes do not establish mechanism.
- **Pairwise bias**: framework is largely pairwise; MISS-style higher-order effects can dominate.
- **Task heterogeneity confounding**: differences in data quality, reward sparsity, horizon, and success metric may masquerade as "interaction."
- **Training dynamics/path dependence**: interaction may depend on task order, optimizer state, and curriculum, not just dataset mix.
- **External validity risk**: LIBERO-10 + small backbone may not transfer to real large-scale VLA regimes despite frozen OpenVLA check.

### 4. Methodological Concerns

- Bonferroni may be too conservative for many pairwise tests, increasing false negatives; power analysis is missing.
- 5 seeds may still be underpowered for small transfer effects in RL-style variance regimes.
- Proxy validation target (rho > 0.6) is useful but incomplete; calibration/error bars and rank stability (Kendall-tau) are also needed.
- LP intervention assumes stable influence estimates; estimation noise can make optimized weights brittle.
- Architecture-vs-data confound is not isolated early enough; should be front-loaded as a blocking factor.

### 5. Novelty Assessment

- **Component novelty**: moderate (BCS/representation geometry, gradient affinity, and LP mixing each have precedents).
- **System novelty**: stronger; the multi-resolution integration plus falsification gates is the genuinely new contribution.
- Overall novelty: **6.5/10**.

### 6. Feasibility Assessment

- **14 GPU-hours**: feasible for pilot and partial phase completion, not for statistically credible end-to-end claims.
- **~34 GPU-hours** estimate is more realistic for full rigor (seeds, rollouts, ablations, and intervention comparisons).
- If budget stays 14h, scope must be reduced to "feasibility + directional evidence," not definitive conclusions.

### 7. Missing Perspectives

- Statistical design perspective: formal power analysis, hierarchical/mixed-effects modeling, and multiplicity strategy beyond Bonferroni.
- Robustness/reproducibility perspective: sensitivity to seeds, hyperparameters, and implementation variance.
- Evaluation-theory perspective: whether task success metrics are commensurate across tasks for interaction inference.
- Data-centric reliability perspective: label/reward noise and demonstrator quality as first-order confounders.

### 8. Specific Recommendations (actionable)

1. Add a **pre-study power analysis** and switch to FDR or hierarchical testing for pairwise scans.
2. Insert an explicit **architecture-vs-data factorial ablation in Phase 1** (shared backbone vs per-task adapters) before proxy benchmarking.
3. Report **rank stability metrics** (Kendall-tau, top-k overlap) with confidence intervals, not only Spearman rho.
4. Stress-test LP mixing with **bootstrap/noise perturbations** of affinity matrix; report robustness of chosen weights.
5. Reserve a small budget for **higher-order checks** (at least triplet interactions on a subset) to test pairwise sufficiency assumptions.

## 评分

**6.8/10**

Promising and more rigorous than typical proposal drafts, but currently over-optimistic on compute and too reliant on pairwise/proxy assumptions for strong mechanistic claims. With stronger statistical design and confound isolation, this could become a high-quality, publishable study.
