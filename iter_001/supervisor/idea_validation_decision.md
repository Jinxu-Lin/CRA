# Idea Validation Decision -- Iteration 1 (Post-Refinement)

## Decision: ADVANCE

After reviewing the refined proposal, revised hypotheses, and updated task plan, all refinement actions from the iteration 0 REFINE decision have been completed. The remaining uncertainties can only be resolved by running full-scale experiments.

---

## Context: Previous REFINE Actions -- All Completed

The iteration 0 decision requested 7 refinements. Status:

1. **Add continuous metrics (Kendall tau, Spearman rho) for FM2 testing** -- DONE. task_plan.json P1 tasks use continuous metrics as primary FM2 evaluation. Controlled contamination injection (H11) added as independent FM2 validation.
2. **Revise hypotheses H4 and H9** -- DONE. H4 reframed to "r_eff << d << B" (strengthens FM1). H9 replaced with spectral concentration ratio (r_eff/d vs r_eff/B).
3. **Add PCA-reduced whitening for H7 recovery** -- DONE. H7-revised targets k=64-128 where N/k >> 40. P5 task designed with k sweep {16..512} and ridge-regularized variant.
4. **Update proposal.md with Evidence-Driven Revisions** -- DONE. Comprehensive section added covering all pilot findings, narrative shifts, negative results to report.
5. **Revise task_plan.json with full-scale design** -- DONE. 12 tasks across 5 priority phases, ~24 GPU-hours, 4-wave parallelization plan, explicit decision gates.
6. **Frame toxicity reversal as task-type boundary** -- DONE. New RQ3 added. Toxicity reversal positioned as genuine contribution (task-type boundary discovery), not failure.
7. **Re-run planner for updated full-experiment task plan** -- DONE. Complete parallelization plan with 4 waves, dependency graph, and wall-clock estimates.

---

## Candidate Comparison

### cand_a: CRA -- FM1 Spectral Diagnosis + Systematic Benchmark + Task-Type Boundary (FRONT RUNNER)

**Tier 1 evidence (strong, ready for full-scale replication):**
- FM1 spectral diagnosis: gradient r_eff=10 (top-5 captures 85.6% variance) vs representation r_eff=63 (34.9%). Direct, measurable signal dilution.
- Task-type boundary: RepSim > TRAK by +32pp (counterfact), +17pp (ftrace); TRAK > RepSim by +24pp (toxicity). Gradient norm artifact (Cohen's d=2.66) explains reversal.
- H6 K-FAC control: RepSim > K-FAC IF by 17.4pp on counterfact. FM1 is independent of Hessian quality.
- TRAK saturation: k=256 (k/d=0.12) with non-monotonic behavior after saturation.

**Tier 2 evidence (redesigned, under investigation at full scale):**
- FM2 via continuous metrics (H2-revised) + contamination injection (H11 new).
- PCA-reduced whitening (H7-revised) at feasible N/k ratios.
- Gap decomposition (H10 new): layer mixing, cosine normalization, semantic features.

**Infrastructure:** Pipeline validated, 28 min pilot runtime, 4x RTX 4090 available.

### cand_b: Hessian Quality Diagnosis -- DROPPED
H6 confirmed: K-FAC IF still 17.4pp below RepSim. No path forward.

### cand_c: Matched Filter Theory -- BACKUP (subsumed into cand_a P5)
Promotion criterion: PCA-whitened attribution outperforms RepSim by >= 5pp on >= 2 tasks.

### cand_d: Attribution vs Retrieval Boundary -- BACKUP (subsumed into cand_a P3)
Promotion criterion: Contriever/GTR matches RepSim (< 3pp gap) on >= 2 tasks.

---

## Arguments for ADVANCE

1. **Refinement is complete.** All 7 REFINE actions executed. Proposal, hypotheses, task plan, and methodology are revised. No remaining pre-experiment actions yield new information.

2. **Strong Tier 1 evidence.** FM1 spectral diagnosis and task-type boundary are well-supported. These alone constitute a publishable contribution (systematic benchmark paper at poster level).

3. **Robust contingency design.** The task plan includes 4 explicit decision gates (after P1, P3, P4, P5) that allow graceful scope narrowing. The paper is viable under multiple failure scenarios:
   - FM2 fails -> narrow to FM1 + benchmark + task-type boundary
   - Retrieval matches RepSim -> pivot to attribution-vs-retrieval analysis
   - Whitening fails -> framework is taxonomic (honest negative)
   - Gap doesn't decompose -> "feature quality, not just dimensionality"

4. **Diminishing returns from further refinement.** The remaining uncertainties (FM2 detectability, whitening effectiveness, gap decomposition) are empirical questions that require experimental data. No additional pilot work or planning would resolve them.

5. **Infrastructure ready.** Pipeline validated, GPU budget feasible (~4h wall-clock with 4-GPU parallelism), all methods functional.

---

## Risk Assessment

| Risk | P(occurs) | Impact | Mitigation in plan |
|------|-----------|--------|--------------------|
| FM2 undetectable with continuous metrics | 0.25 | Narrow to FM1-only (poster) | P1 decision gate; contamination injection backup |
| Retrieval models match RepSim | 0.20 | Reposition paper | P3 decision gate; cand_d promotion |
| PCA whitening still fails | 0.35 | Framework taxonomic only | P5 decision gate; honest negative |
| TRAK-PCA gap doesn't decompose | 0.30 | Weaker mechanistic narrative | P4 still publishable |
| BM25 competitive at full scale | 0.20 | Restrict counterfact claims | Discussed as limitation |

**Worst case** (FM2 + retrieval + whitening all fail, P ~ 0.02): FM1 spectral diagnosis + task-type boundary + systematic benchmark. Weak poster, but publishable.

**Base case** (FM2 partially works OR whitening marginal, P ~ 0.50): Solid NeurIPS poster with FM1 + benchmark + task-type + partial FM2.

**Best case** (FM2 validated + whitening works, P ~ 0.25): NeurIPS spotlight candidate with complete diagnostic framework.

---

## Confidence Analysis

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| FM1 thesis viability | 0.85 | Strong pilot evidence, well-designed full-scale replication |
| FM2 thesis viability | 0.55 | Redesigned with continuous metrics + injection; but untested |
| Task-type boundary contribution | 0.90 | Toxicity reversal is systematic and well-explained |
| Whitened attribution recovery | 0.50 | PCA reduction is principled but pilot signal marginal |
| Publication viability (NeurIPS/ICML) | 0.70 | Multiple viable paper configurations across outcomes |
| **Overall confidence** | **0.75** | Up from 0.60 due to completed refinement and robust contingency |

---

## Post-ADVANCE Confidence Update from 0.60 to 0.75

The 15-point increase reflects:
- (+5) All REFINE actions completed -- no remaining methodology gaps
- (+5) Robust decision gates built into the experimental plan
- (+3) New H10 (gap decomposition) and H11 (contamination injection) add two high-probability contributions
- (+2) Updated task plan has realistic wall-clock estimates and validated infrastructure

The remaining 25-point uncertainty is dominated by:
- (-10) FM2 may be undetectable even with continuous metrics
- (-8) PCA whitening may still fail at full scale
- (-4) Gap decomposition may not yield clean factors
- (-3) Retrieval baselines may match RepSim

These are irreducible until experimental data is collected.

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.75
DECISION: ADVANCE
