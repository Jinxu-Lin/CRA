# CRA: Testable Hypotheses with Falsification Criteria

## Hypothesis H1: Representation-Space Superiority (FM1 Evidence)

**Statement**: Representation-space methods (RepSim, RepT) systematically outperform parameter-space methods (TRAK, LoGra) on DATE-LM tasks when both use standard (non-contrastive) scoring.

**Expected Outcome**: RepSim LDS > TRAK LDS by >= 5pp on data selection task; similar pattern on toxicity filtering (auPRC) and factual attribution (P@K).

**Falsification Criterion**: RepSim (last-layer, standard scoring) < TRAK (k=2048, standard scoring) - 5pp on DATE-LM LDS for data selection task.

**Pre-registered threshold**: 5pp gap on at least 2 of 3 DATE-LM tasks.

---

## Hypothesis H2: Contrastive Scoring Universal Benefit (FM2 Evidence)

**Statement**: Contrastive scoring (mean subtraction) improves attribution performance across both parameter-space and representation-space methods, with larger improvement in parameter space.

**Expected Outcome**:
- Parameter-space improvement: 10-20pp (TRAK standard -> TRAK contrastive)
- Representation-space improvement: 2-5pp (RepSim standard -> RepSim contrastive)
- The asymmetry is predicted by the bias decomposition: ||phi_shared||/||phi_task|| is larger in parameter space

**Falsification Criterion**: Contrastive scoring degrades performance by > 3pp on >= 1 method on >= 1 DATE-LM task.

---

## Hypothesis H3: FM1/FM2 Orthogonality

**Statement**: The improvements from addressing FM1 (representation space) and FM2 (contrastive scoring) are approximately additive -- the 2x2 factorial interaction term is small relative to main effects.

**Expected Outcome**: In a 2-way ANOVA on each DATE-LM task, the interaction term accounts for < 30% of the minimum main effect.

**Falsification Criterion**: Interaction term exceeds 30% of the minimum main effect on >= 2 of 3 DATE-LM tasks.

**Note**: If interaction is synergistic (FM1+FM2 fix together is *better* than sum of parts), this weakens the "orthogonal defects" narrative but strengthens the "representation + contrastive" practical recommendation.

---

## Hypothesis H4: FM1 Is Rank Deficiency (Spectral Evidence)

**Statement**: The effective rank of the gradient covariance matrix Sigma_g scales with the representation dimension d, not the parameter count B. Specifically, r_eff(95%) is in the range [0.5d, 2d].

**Expected Outcome**:
- Pythia-70M (d=512): r_eff(95%) in [256, 1024]
- Pythia-160M (d=768): r_eff(95%) in [384, 1536] (if cross-model validation is run)

**Falsification Criterion**: r_eff(95%) > 10d at any model scale.

---

## Hypothesis H5: TRAK Dimension Saturation (FM1 Mechanistic Test)

**Statement**: TRAK attribution quality saturates at projection dimension k approximately equal to the representation dimension d. Specifically, 90% of maximal LDS is achieved by k = 2d, with < 5% additional improvement from k = 2d to k = 10d.

**Expected Outcome**: On Pythia-1B (d=2048), TRAK LDS forms a characteristic saturation curve with knee at k ~ 2048.

**Falsification Criterion**: TRAK LDS continues to improve linearly (or log-linearly) up to k = 8192 without visible saturation.

**Supplementary prediction**: TRAK with PCA projection (onto top-k eigenvectors of Sigma_g) should saturate at smaller k than TRAK with random projection, and TRAK-PCA at k=d should approach RepSim performance (the "smoking gun" for FM1).

---

## Hypothesis H6: FM1/FM2 Independence from Hessian Quality

**Statement**: FM1 and FM2 are genuine signal processing defects, not downstream symptoms of poor Hessian approximation. When using high-quality Hessian (K-FAC full eigendecomposition), parameter-space IF should still significantly underperform RepSim.

**Expected Outcome**: K-FAC IF on Pythia-70M performs significantly worse than RepSim (>= 10pp gap on DATE-LM LDS).

**Falsification Criterion**: K-FAC IF on Pythia-70M achieves RepSim-level performance (< 5pp gap on LDS). This would mean FM1/FM2 are artifacts of Hessian approximation quality, not independent failure modes.

**Implication of failure**: The entire CRA diagnostic framework pivots from "two signal processing defects" to "Hessian approximation quality is the primary bottleneck, and representation methods succeed because they implicitly bypass the need for Hessian computation."

---

## Hypothesis H7: Whitened Attribution Optimality

**Statement**: Whitened attribution phi^T Sigma_noise^{-1} psi outperforms standard attribution phi^T psi (M=I) when common influence has structured (non-isotropic) covariance, with the improvement largest on high-FM2 tasks.

**Expected Outcome**: Whitened RepSim outperforms standard RepSim by 3-8pp on factual attribution (highest FM2 severity), with smaller gains on toxicity filtering and data selection.

**Falsification Criterion**: Whitened RepSim performs >= 3pp worse than standard RepSim on >= 2 of 3 tasks (indicating whitening amplifies noise rather than removing structured contamination).

---

## Hypothesis H8: BM25 Does Not Dominate Attribution Methods

**Statement**: On data selection and toxicity filtering tasks, representation-space TDA methods significantly outperform BM25 (a lexical matching baseline with no model-internal information).

**Expected Outcome**: RepSim outperforms BM25 by >= 10pp on data selection (LDS) and toxicity filtering (auPRC).

**Note**: BM25 may be competitive on factual attribution (DATE-LM itself hints at this). If BM25 matches RepSim on factual attribution, this limits but does not invalidate the CRA thesis -- it means TDA adds value specifically where lexical overlap is insufficient.

---

## Hypothesis H9: Representation Covariance Is Near-Isotropic

**Statement**: The representation covariance eigenvalue ratio lambda_1/lambda_d is < 100 (near-isotropic), while the gradient covariance eigenvalue ratio lambda_1/lambda_B is > 10^4 (highly anisotropic). This explains why M = I suffices for representation-space methods but M = H^{-1} is needed for parameter-space methods.

**Expected Outcome**: On Pythia-70M, representation covariance condition number < 100; gradient covariance condition number > 10^4.

**Falsification Criterion**: Representation covariance condition number > 1000, which would suggest whitening (M != I) is also necessary in representation space.

---

## Summary Priority

| Hypothesis | Priority | Phase | Cost | P(success) |
|-----------|----------|-------|------|------------|
| H6 (Hessian control) | P1 - Critical | Phase 0 | 2h | 50% |
| H1 (Rep > Param) | P2 - Core | Phase 1 | 2h | 85% |
| H2 (Contrastive universal) | P2 - Core | Phase 1 | 2h | 80% |
| H3 (Orthogonality) | P2 - Core | Phase 1 | included in P2 | 60% |
| H8 (BM25 control) | P2 - Core | Phase 1 | 0.1h | 70% |
| H4 (r_eff ~ d) | P3 - Mechanism | Phase 2 | 1h | 65% |
| H5 (Dim saturation) | P3 - Mechanism | Phase 2 | 2h | 75% |
| H7 (Whitened MF) | P4 - Extension | Phase 3 | 1.5h | 65% |
| H9 (Isotropy) | P4 - Extension | Phase 2 | included in H4 | 70% |
