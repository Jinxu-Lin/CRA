# CRA: Testable Hypotheses -- Revised After Pilot Evidence

**Revision basis**: Pilot results (N=100, Pythia-1B, DATE-LM, ~28 min). All hypotheses updated to reflect what pilot data confirmed, refuted, or left unresolved.

---

## Hypothesis H1: Representation-Space Superiority on Semantic Attribution Tasks (FM1 Evidence)

**Status after pilot**: SUPPORTED on 2/3 tasks; REVERSED on toxicity.

**Revised Statement**: Representation-space methods (RepSim) systematically outperform parameter-space methods (TRAK) on **semantic attribution tasks** (counterfact, ftrace) but not on **behavioral detection tasks** (toxicity).

**Expected Full-Scale Outcome**: RepSim > TRAK by >= 15pp on counterfact R@50 and >= 10pp on ftrace R@50. TRAK > RepSim by >= 10pp on toxicity AUPRC.

**Falsification Criterion**: RepSim < TRAK - 5pp on counterfact or ftrace at full scale (N=5473).

**Pilot evidence**: RepSim > TRAK by +32pp (counterfact), +17pp (ftrace); TRAK > RepSim by +24pp (toxicity). The task-type boundary is sharp and systematic.

---

## Hypothesis H2-revised: FM2 Detection via Continuous Metrics and Controlled Injection

**Status after pilot**: INCONCLUSIVE (evaluation protocol failure, not FM2 absence).

**Revised Statement**: When measured with continuous metrics (Kendall tau, Spearman rho on raw scores), contrastive scoring (mean subtraction) improves attribution quality for parameter-space methods more than for representation-space methods.

**Expected Full-Scale Outcome**:
- Parameter-space Kendall tau improvement: >= 0.05 (TRAK standard -> TRAK contrastive)
- Representation-space Kendall tau improvement: < 0.02
- Controlled contamination injection: contrastive scoring recovers >= 90% of alpha=0 performance at alpha <= 1.0

**Falsification Criterion**:
- Kendall tau improvement < 0.02 for ALL methods on ALL tasks (FM2 has no detectable effect on continuous metrics)
- AND contamination injection recovery < 50% at alpha=1.0 (FM2 correction mechanism ineffective)
- Both conditions must hold to falsify; either alone is insufficient

**Pilot evidence**: Rank-based metrics showed exactly 0.0pp gain (invariance artifact). Continuous metrics were not computed. This hypothesis is untested, not falsified.

**Design note**: The controlled contamination injection (Empiricist Exp 2.2) provides causal FM2 validation independent of any metric choice.

---

## Hypothesis H3-revised: FM1/FM2 Interaction

**Status after pilot**: TRIVIALLY SATISFIED (no statistical power due to H2 evaluation failure).

**Revised Statement**: Under continuous metrics, the interaction between FM1 correction (representation space) and FM2 correction (contrastive scoring) is small relative to main effects.

**Expected Full-Scale Outcome**: In a 2-way ANOVA on Kendall tau, the interaction term accounts for < 30% of the minimum main effect on at least 2/3 tasks.

**Falsification Criterion**: Interaction term exceeds 30% of minimum main effect on >= 2 tasks. A **negative** interaction (sub-additivity) would support the Innovator's conjugacy hypothesis; a **positive** interaction (super-additivity) would suggest correlated defects.

**Contingency**: This hypothesis is only testable if H2-revised succeeds (FM2 is detectable with continuous metrics). If H2-revised fails, H3 becomes untestable and should be reported as such.

---

## Hypothesis H4-revised: FM1 Is Extreme Rank Deficiency

**Status after pilot**: DIRECTIONALLY SUPPORTED (r_eff=10, far below predicted [256,1024]).

**Revised Statement**: The effective rank of the gradient covariance r_eff(95%) satisfies r_eff << d << B. Specifically, r_eff is O(10-100) rather than O(d) as originally predicted.

**Expected Full-Scale Outcome** (Pythia-70M, N=5000):
- Full-model gradient r_eff(95%) in [5, 100]
- Target-layer gradient r_eff(95%) in [30, 200]
- Representation r_eff(95%) in [50, 300] (at N=5000, reliable estimate possible)
- The key comparison: gradient r_eff / B << representation r_eff / d

**Falsification Criterion**: Full-model gradient r_eff(95%) > 1000 at N=5000 (gradient signal is not low-rank).

**Pilot evidence**: r_eff=10 (full model, N=100), r_eff=53 (target layers, N=100). The signal is even more concentrated than predicted. Original [0.5d, 2d] range was too optimistic.

---

## Hypothesis H5-revised: TRAK Dimension Saturation Below d

**Status after pilot**: SUPPORTED (saturation at k=256, k/d=0.12).

**Revised Statement**: TRAK attribution quality saturates at projection dimension k << d. Specifically, 90% of maximal performance is achieved by k in [d/16, d/4], not k ~ d as originally predicted.

**Expected Full-Scale Outcome**: On Pythia-1B (d=2048, N=5473):
- TRAK-random R@50 peaks or saturates at k in [128, 512]
- TRAK-PCA at same k outperforms TRAK-random (signal subspace vs random)
- The non-monotonic behavior observed at pilot scale (peak at k=256, degradation after) should resolve or be explained

**Falsification Criterion**: TRAK R@50 continues to improve linearly up to k=4096 without saturation.

**Supplementary prediction (SMOKING GUN)**: TRAK-PCA at k=d should substantially close the gap with RepSim (to within 15pp). If the gap remains > 20pp at full scale, FM1 is necessary but another factor dominates.

**Pilot evidence**: Peak at k=256 (R@50=0.785), non-monotonic after. TRAK-PCA at k=d gave R@50=0.686, still 30.8pp below RepSim. The gap is the core challenge.

---

## Hypothesis H6: FM1 Independence from Hessian Quality

**Status after pilot**: CONFIRMED.

**Statement**: FM1 and FM2 are genuine signal processing defects, not downstream symptoms of poor Hessian approximation.

**Pilot evidence**: RepSim > K-FAC IF by 17.4pp on counterfact (R@50+MRR). Even with high-quality Hessian, parameter-space methods fail. cand_b (Hessian Quality Diagnosis) is dropped.

**No further full-scale testing needed.** This control experiment is complete.

---

## Hypothesis H7-revised: PCA-Reduced Whitened Attribution

**Status after pilot**: FAILED at full dimension (N/d=0.049); concept directionally correct (SNR-accuracy r=0.34).

**Revised Statement**: Whitened attribution in a PCA-reduced subspace (phi_PCA^T Sigma_PCA^{-1} psi_PCA at k=64-128 where N/k >> 40) outperforms standard RepSim (M=I).

**Expected Full-Scale Outcome**:
- PCA-whitened at k=64 (N/k ~ 85 for counterfact): +3 to +5pp over standard RepSim on at least 1 task
- Ridge-regularized whitening at optimal lambda: comparable or better than PCA-whitening
- Performance improvement largest on ftrace (pilot showed +6.8pp whitening gain for DDA)

**Falsification Criterion**: PCA-whitened attribution fails to outperform standard RepSim by >= 3pp at ANY k value in {16, 32, 64, 128, 256} on ANY of the 3 tasks.

**Pilot evidence**: Full-dimensional whitening failed (-8 to -11pp), but ftrace showed promising whitening gains across multiple methods. SNR-accuracy positive correlation (r=0.34 counterfact, r=0.16 ftrace) validates the concept.

**If falsified**: Report as honest negative result. The matched filter theory is correct in principle but impractical at DATE-LM scale. Framework demoted from "prescriptive" to "organizational."

---

## Hypothesis H8-revised: Attribution vs Retrieval Boundary

**Status after pilot**: PARTIAL (BM25 perfect on counterfact at N=100).

**Revised Statement**: On semantic attribution tasks, dedicated retrieval models (Contriever, GTR-T5) do NOT match RepSim performance, demonstrating that representation-space TDA captures model-internal attribution signal beyond semantic retrieval.

**Expected Full-Scale Outcome**:
- RepSim > Contriever/GTR by >= 5pp on at least 1 of {counterfact, ftrace}
- BM25 degrades substantially below R@50=1.0 at N=5473 counterfact (more candidates dilute lexical signal)
- RepSim > BM25 by >= 10pp on toxicity and ftrace

**Falsification Criterion**: Contriever/GTR achieves RepSim-level performance (< 3pp gap) on ALL 3 tasks. This would mean "attribution = retrieval" and CRA must reposition.

**Pilot evidence**: BM25 achieves R@50=1.0 on counterfact (N=100), likely benefiting from small candidate pool. The full-scale test is critical.

---

## Hypothesis H9-revised: Spectral Concentration Comparison

**Status after pilot**: FALSIFIED in original form (rep_cond=3.1e10 >> grad_cond=3,589).

**Revised Statement**: Representation space has higher spectral concentration relative to its nominal dimension (r_eff/d is larger for representations than r_eff/B for gradients), even though the absolute condition number may be large. This relative concentration explains why M=I works in representation space.

**Expected Full-Scale Outcome** (Pythia-70M, N=5000 where N >> d=512):
- Representation condition number: uncertain. If it drops to < 1000, pilot value was N < d artifact. If it remains > 10^4, anisotropy is genuine.
- Key comparison: rep_r_eff(95%)/d vs grad_r_eff(95%)/B. Expected: rep ratio >> grad ratio by orders of magnitude.

**Falsification Criterion**: At N=5000, representation r_eff(95%)/d < gradient r_eff(95%)/B (representation space is NOT relatively more concentrated than gradient space).

**If representations remain genuinely anisotropic at N >> d**: Need alternative explanation for why M=I works. Candidate hypotheses: (a) attribution signal aligns with high-variance directions (anisotropy is beneficial); (b) rank-based metrics are insensitive to anisotropy; (c) the relevant information lies in a low-dimensional subspace where effective isotropy holds.

---

## NEW Hypothesis H10: Gap Decomposition

**Statement**: The 30.8pp gap between TRAK-PCA at k=d and RepSim can be substantially decomposed into identifiable mechanistic factors.

**Expected Outcome**:
- Factor (a) Layer mixing: Last-layer-only TRAK-PCA reduces the gap by >= 10pp
- Factor (b) Cosine normalization: Cosine-normalized TRAK-PCA reduces the gap by >= 5pp
- Factor (c) Combined: If (a) + (b) together close to within 10pp of RepSim, the gap is primarily engineering, not fundamental
- Factor (d) If residual gap > 15pp after both corrections, nonlinear semantic features are the dominant factor

**Falsification Criterion**: Neither last-layer-only nor cosine normalization reduces the gap by more than 3pp (the gap is not decomposable into these factors, suggesting a more fundamental mechanism).

---

## NEW Hypothesis H11: Contamination Injection Recovery

**Statement**: Controlled injection of FM2-type contamination into attribution scores degrades performance, and contrastive scoring (mean subtraction) recovers it.

**Expected Outcome**:
- At alpha=0 (no contamination): baseline performance
- At alpha=1.0: performance degrades by >= 10pp on R@50 or equivalent
- Contrastive scoring at alpha=1.0: recovers >= 90% of alpha=0 performance
- Recovery effectiveness decreases as alpha increases (diminishing returns at alpha >= 5.0)

**Falsification Criterion**: Contamination injection at alpha=2.0 has < 5pp effect on rank metrics (FM2-type bias does not affect attribution rankings at all, even when artificially amplified).

---

## Summary: Priority and Status

| Hypothesis | Status | Priority | Phase | Cost (GPU-h) | P(validated) |
|-----------|--------|----------|-------|--------------|-------------|
| H6 (Hessian control) | CONFIRMED | -- | Complete | -- | -- |
| H1 (FM1 space gap) | SUPPORTED 2/3 | Confirmed | Complete (refine at full scale) | 0 | 90% |
| H11 (contamination injection) | NEW | P1 - Critical | FM2 Verification | 2 | 85% |
| H2-rev (FM2 continuous) | UNTESTED | P1 - Critical | FM2 Verification | 1 | 60% |
| H4-rev (extreme rank deficiency) | DIRECTIONAL | P2 - Core | FM1 Diagnostic | 2 | 85% |
| H5-rev (TRAK saturation < d) | SUPPORTED | P2 - Core | FM1 Diagnostic | 8 | 75% |
| H8-rev (attribution vs retrieval) | PARTIAL | P3 - Boundary | Retrieval Baselines | 2.5 | 70% |
| H10 (gap decomposition) | NEW | P4 - Mechanism | Gap Analysis | 5 | 70% |
| H9-rev (spectral concentration) | FALSIFIED (original) | P4 - Mechanism | included in H4-rev | 0 | 55% |
| H7-rev (PCA whitening) | FAILED (original) | P5 - Extension | Framework Validation | 3 | 55% |
| H3-rev (FM1/FM2 interaction) | TRIVIAL | Contingent on H2-rev | FM2 Verification | 0 | 50% |
