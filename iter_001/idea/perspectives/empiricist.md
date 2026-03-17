# Empiricist Perspective: CRA Research Proposal

## Executive Summary

The CRA proposal aims to diagnose two signal processing defects (FM1: signal dilution; FM2: common influence contamination) in parameter-space TDA, unify five representation-space methods under a bilinear framework phi^T M psi, and validate this on DATE-LM. After rigorous examination of the pilot evidence (N=100, 14 tasks, ~28 min runtime), I find the proposal has **one strong empirical core (FM1 spectral evidence + TRAK saturation) surrounded by multiple unvalidated or falsified claims (FM2 untested, H7 whitened attribution failed, H9 isotropy reversed)**. Below I redesign the experimental methodology to maximize measurable, reproducible, confound-free evidence -- prioritizing what can actually be measured over what sounds theoretically elegant.

---

## Critical Assessment of Existing Pilot Evidence

Before proposing new experiments, I must honestly evaluate what the pilot data actually shows. An experimentalist who ignores inconvenient data is not an experimentalist.

### What the Pilot Data Supports

1. **FM1 spectral concentration is real.** Full-model gradient top-5 eigenvalues capture 85.6% of variance vs 34.9% for representations (Pythia-70M, N=100). The gradient signal is extremely low-rank (r_eff=10 for full model). This directly measures the dimensional mismatch the CRA thesis predicts.

2. **TRAK dimension saturation is real.** On counterfact (R@50), TRAK peaks at k=256 (k/d=0.12) and degrades non-monotonically after. This pattern is consistent with projecting a low-rank signal through random matrices -- though the saturation point (k=256) is much lower than the predicted k~d=2048.

3. **RepSim > TRAK on attribution tasks.** On counterfact (+32.4pp) and ftrace (+16.6pp), representation-space methods dominate. The K-FAC IF control confirms this is not merely a Hessian approximation artifact (RepSim > K-FAC IF by 17.4pp on counterfact).

### What the Pilot Data Refutes or Leaves Unresolved

1. **FM2 is completely untested.** Contrastive scoring (mean subtraction) produced exactly 0.0pp gain across all 12 method-task cells. Root cause: mean subtraction is a rank-preserving affine transformation, so rank-based metrics (AUPRC, R@K) are invariant to it by construction. This is not "FM2 doesn't exist" -- it is "the experiment was unable to detect FM2 with these metrics." This is a **design flaw in the evaluation protocol**, not evidence for or against FM2.

2. **H7 whitened attribution fails catastrophically.** RepSim-whitened degrades by 8-11pp on all three tasks. The stated cause (N/d=0.049, covariance underdetermined) is plausible but requires verification at full scale. Even at full DATE-LM scale (N=5,473 for counterfact, d=2048), N/d=2.7 is still marginal for Ledoit-Wolf estimation (the estimator assumes N >> d for consistency).

3. **H9 isotropy is reversed.** Representation condition number = 3.1e10 (extremely anisotropic), gradient condition number = 3,589. This is the exact opposite of the theoretical prediction. The representation covariance N=100 < d=512 caveat applies, but the direction of falsification is concerning.

4. **Toxicity task reversal.** TRAK (0.926) > RepSim (0.685) on toxicity AUPRC, a 24pp reversal of the FM1 prediction. k-NN (0.809) also outperforms RepSim. This is not noise -- it is a systematic pattern where gradient-norm correlates directly with toxicity. The contrarian correctly identifies this as evidence that the similarity *function* (not the space) may be the primary driver.

5. **30.8pp TRAK-PCA to RepSim gap.** If FM1 were the full story, TRAK with PCA projection onto the top-d gradient eigenvectors should approach RepSim. The 30.8pp gap (0.686 vs 0.994 on counterfact R@50) shows FM1 is necessary but grossly insufficient. Something else drives RepSim's advantage -- possibly the nonlinearity of the representation map, or the semantic structure of hidden states vs gradient structure.

---

## Proposal: Experiment-Driven Research Plan with Hardened Controls

I propose three experiment packages, each designed to produce falsifiable, confound-controlled evidence. Every experiment specifies: **exact protocol, baselines, ablation plan, falsification criteria, pilot validation estimate, and computational cost.**

---

## Experiment Package 1: The FM1 Diagnostic Suite (Priority: Critical)

### Goal

Establish whether the parameter-to-representation performance gap is causally explained by gradient rank deficiency, or merely correlated with it.

### Experiment 1.1: Full-Scale Eigenspectrum with Controlled N

**Rationale.** Pilot eigenspectrum (N=100) is unreliable because N < d for representations. We need N >> d for both spaces.

**Protocol:**
1. Model: Pythia-70M (d=512, B=70M). Use DATE-LM toxicity task (N=10,187 train).
2. Sample N in {100, 500, 1000, 2000, 5000} training examples.
3. For each N:
   - Compute representation covariance (d x d = 512 x 512) eigenvalues via exact SVD.
   - Compute gradient covariance top-500 eigenvalues via Lanczos (using `torch.lobpcg` or `scipy.sparse.linalg.eigsh` with implicit matrix-vector products via `torch.func.vmap`).
   - Record: r_eff(80%), r_eff(90%), r_eff(95%), condition number, top-5 variance fraction.
4. Cross-validate on Pythia-160M (d=768) at N=2000.

**Key controls:**
- Same training data subset for both spaces (eliminates data-selection confound).
- Same random seed for sampling at each N.
- Report eigenvalue decay curves (log-log plot), not just summary statistics.

**Falsification criteria:**
- H4 (revised): If r_eff(95%) of gradient covariance > 10*d at N=5000, FM1 rank-deficiency narrative is weakened.
- H9 (revised): If representation condition number < 100 at N=5000 (where N >> d=512), the original isotropy claim is rescued. If condition number remains > 10^4, isotropy is definitively falsified and the paper must explain why M=I works despite anisotropy.

**Computational cost:** ~2 GPU-hours on single RTX 4090. Each N sweep takes ~20 min.

**Success probability:** 85% (the measurement itself is routine; the question is whether the numbers fall in a narratively useful range).

### Experiment 1.2: TRAK Dimension Sweep at Full Scale with PCA Comparison

**Rationale.** Pilot sweep (N=100) showed saturation at k=256, far below d=2048. At full scale (N=5K-10K), the saturation point may shift. The TRAK-PCA comparison is the most direct FM1 test.

**Protocol:**
1. Model: Pythia-1B (d=2048). Task: counterfact (N=5,473 full).
2. TRAK with random projection: k in {32, 64, 128, 256, 512, 1024, 2048, 4096}.
3. TRAK with PCA projection: k in {32, 64, 128, 256, 512, 1024, 2048}.
   - PCA eigenvectors from gradient covariance (Lanczos top-2048 on Pythia-1B target layers).
   - If Pythia-1B Lanczos is infeasible, use Pythia-70M eigenvectors as surrogate (cross-model transfer test).
4. Metrics: R@50, MRR, **Kendall tau** (continuous metric to detect FM2 effects).
5. 3 random seeds for TRAK random projection; single deterministic run for PCA.
6. Compute RepSim baseline at same N for direct comparison.

**Key controls:**
- Same train/test split across all k values.
- Report wall-clock time and peak GPU memory per k to quantify compute-quality tradeoff.
- Include k=d (2048) as the critical prediction point.

**Falsification criteria:**
- H5 (original): If TRAK-random shows no saturation up to k=4096 (continues improving log-linearly), FM1 rank deficiency does not explain TRAK's dimension sensitivity.
- FM1 smoking gun: If TRAK-PCA at k=d closes to within 5pp of RepSim, FM1 is confirmed as the primary mechanism. If the gap remains > 15pp (pilot: 30.8pp), additional factors beyond dimension are at play.

**Computational cost:** ~8 GPU-hours (parallelizable to ~2h on 4 GPUs). TRAK at k=4096 may OOM; cap at k=2048 if so.

**Success probability:** 75% for measurement completion; 50% for TRAK-PCA closing the gap with RepSim.

### Experiment 1.3: Representation Layer Sweep (Confound Control)

**Rationale.** RepSim uses last-layer representations. But different layers have different d and different information content. If RepSim performance varies dramatically across layers, the "space" advantage is really a "which features you measure" advantage -- not FM1.

**Protocol:**
1. Model: Pythia-1B (24 layers). Task: counterfact + toxicity (N=full).
2. Extract representations at layers {0, 4, 8, 12, 16, 20, 23} (every 4th + last).
3. Compute RepSim (cosine similarity) at each layer.
4. Also compute RepSim with **Euclidean distance** at each layer (contrarian's falsification test).
5. Metrics: R@50, AUPRC, Kendall tau.

**Key controls:**
- Same train/test data across all layers.
- Report representation dimensionality and condition number at each layer.
- Compare cosine vs Euclidean to test whether similarity function or space matters more.

**Falsification criteria:**
- If RepSim-Euclidean at last layer matches RepSim-cosine within 3pp on 2/3 tasks, the similarity function is secondary and the space matters more (supports FM1).
- If RepSim-Euclidean degrades by > 10pp, the similarity function dominates the space choice (weakens FM1, supports contrarian).
- If middle layers (layer 12) outperform last layer by > 5pp, layer selection is a more important factor than FM1/FM2.

**Computational cost:** ~3 GPU-hours. Representation extraction is the bottleneck (~20 min per layer per task).

**Success probability:** 90% (trivially implementable).

---

## Experiment Package 2: The FM2 Detection Suite (Priority: Critical)

### Goal

Design experiments that can actually detect FM2 (common influence contamination), since the pilot evaluation protocol was unable to do so.

### The Core Problem

Mean subtraction is a rank-preserving affine transformation: if score(z_i) = s_i, then contrastive_score(z_i) = s_i - mean(s) preserves the ordering of all z_i. Therefore rank-based metrics (AUPRC, R@K, MRR) are mathematically guaranteed to be invariant to mean subtraction. **The pilot "found" zero FM2 effect because the evaluation was incapable of detecting it.**

This is an evaluation protocol failure, not an FM2 non-existence result. The CRA team acknowledged this but has not yet proposed a fix. I propose three complementary FM2 detection approaches.

### Experiment 2.1: Continuous Score Metrics (Kendall Tau, Spearman Rho)

**Rationale.** While mean subtraction preserves *ranks*, it changes the *relative magnitudes* between scores. For regression-like evaluation (where the magnitude of attribution matters, not just ranking), continuous metrics can detect FM2 effects.

**Protocol:**
1. Compute raw attribution scores for all methods (RepSim, TRAK, DDA, LoGra) on all 3 DATE-LM tasks at full scale.
2. Compute contrastive scores (global mean subtraction, task-conditional mean subtraction).
3. Evaluate with: Kendall tau, Spearman rho, Pearson r (against DATE-LM ground-truth scores, not just labels).
4. Also compute LDS (Linear Datamodeling Score) which is a continuous metric:
   - LDS = correlation between predicted importance and actual retrain-and-compare effect.
   - DATE-LM provides pre-computed counterfactual importance scores for LDS computation.

**Key controls:**
- Report both rank and continuous metrics side by side.
- If DATE-LM does not provide continuous ground-truth scores, compute surrogate via leave-k-out retraining (k=100, 10 repeats).

**Falsification criteria:**
- H2 (revised): If contrastive scoring improves Kendall tau by >= 0.05 on parameter-space methods but < 0.02 on representation-space methods (on at least 2/3 tasks), FM2 asymmetry is supported.
- If contrastive scoring has zero effect on both rank and continuous metrics, FM2 may not be a real phenomenon for these tasks.

**Computational cost:** ~1 GPU-hour (scores already computed; metrics are cheap). LDS leave-k-out adds ~4 GPU-hours if needed.

**Success probability:** 60%. The outcome depends entirely on whether DATE-LM provides continuous ground-truth scores. If it provides only binary labels, continuous metrics may be uninformative.

### Experiment 2.2: Controlled Contamination Injection

**Rationale.** The cleanest way to test FM2 is to artificially introduce common influence contamination and measure whether contrastive scoring removes it.

**Protocol:**
1. Take Pythia-1B fine-tuned on DATE-LM task.
2. Inject contamination: add a fixed bias vector b to all training representations:
   - phi_contaminated(z) = phi(z) + alpha * b
   - where b = mean(phi(z_train)) and alpha in {0, 0.1, 0.5, 1.0, 2.0, 5.0}.
3. Compute attribution scores with and without contamination, with and without contrastive scoring.
4. Measure: R@50, AUPRC, Kendall tau at each contamination level.

**Key controls:**
- alpha=0 is the uncontaminated baseline.
- alpha=5.0 represents extreme contamination (bias 5x the mean norm).
- The bias vector b is the mean representation (the exact quantity removed by contrastive scoring).

**Falsification criteria:**
- If contrastive scoring perfectly recovers alpha=0 performance at all contamination levels, FM2 correction via mean subtraction is validated mechanistically.
- If contrastive scoring fails to recover performance at alpha >= 1.0, FM2 correction has limited effectiveness and may require more sophisticated debiasing (task-conditional mean, Sigma_noise^{-1} correction).

**Computational cost:** ~2 GPU-hours. 6 alpha values x 2 scoring modes x 3 tasks, but each is a simple matrix operation on pre-computed representations.

**Success probability:** 85%. This is a controlled experiment where we know the ground truth. The only risk is that the injection model does not match real FM2 contamination structure.

### Experiment 2.3: DDA Ablation (Debias vs Denoise)

**Rationale.** DDA (Pang et al., 2410.01285) reports that their "debias" step contributes ~55pp improvement. But DDA's debias is *not* simple mean subtraction -- it subtracts the *base model* influence, not the training mean. This distinction matters: if DDA's debias captures something beyond mean subtraction, FM2 may have a more complex structure than the CRA thesis assumes.

**Protocol:**
1. Implement three versions of DDA:
   - DDA-full: original implementation (debias + denoise).
   - DDA-debias-only: debias without denoise.
   - DDA-mean-sub: replace DDA's debias with simple global mean subtraction.
2. Run on all 3 DATE-LM tasks at full scale.
3. Compare against CRA's contrastive scoring (global mean subtraction on RepSim).

**Key controls:**
- DDA-mean-sub directly tests whether CRA's FM2 correction captures the same phenomenon as DDA's debias.
- If DDA-debias-only >> DDA-mean-sub, DDA's debias captures structure beyond simple FM2.

**Falsification criteria:**
- If DDA-mean-sub matches DDA-debias-only within 3pp on 2/3 tasks, CRA's FM2 = DDA's debias (the theories are equivalent).
- If DDA-debias-only > DDA-mean-sub by > 10pp, FM2 is real but CRA's mean-subtraction fix is insufficient. The paper must acknowledge DDA's more sophisticated correction.

**Computational cost:** ~4 GPU-hours (DDA implementation + evaluation).

**Success probability:** 70%. Depends on DDA code availability and reproducibility.

---

## Experiment Package 3: Framework Validation with Hardened Controls (Priority: High)

### Goal

Test whether the phi^T M psi framework generates predictions beyond notational convenience. After pilot failures on H7 (whitened attribution), H9 (isotropy), and the 30.8pp TRAK-PCA gap, the framework's predictive power is in serious doubt. These experiments aim to rescue or definitively refute the framework.

### Experiment 3.1: Whitened Attribution with PCA-Reduced Covariance

**Rationale.** The pilot H7 failure (whitened RepSim -8 to -11pp) has a clear diagnosis: N/d=0.049 makes the full d-dimensional covariance estimation degenerate. The fix: whiten in a PCA-reduced subspace where N >> k.

**Protocol:**
1. Model: Pythia-1B. Tasks: all 3 DATE-LM at full scale.
2. Compute representation covariance from training data.
3. PCA-reduce representations to k dimensions, where k in {16, 32, 64, 128, 256, 512}.
4. Estimate covariance in PCA space (k x k matrix, well-conditioned when N >> k).
5. Compute whitened attribution: phi_PCA^T Sigma_PCA^{-1} psi_PCA.
6. Also compute Ledoit-Wolf whitened attribution at full d=2048 for comparison.
7. Ridge-regularized whitening: M = (Sigma + lambda*I)^{-1} with lambda in {0.01, 0.1, 1.0, 10.0}.

**Key controls:**
- PCA k=d is equivalent to pilot H7 (should reproduce the failure).
- PCA k=64 with N=5000 gives N/k=78 (well-conditioned).
- Ridge regularization strength lambda provides a continuous tradeoff axis.
- Compare against RepSim (M=I) and k-NN baselines.

**Falsification criteria:**
- H7 (revised): If PCA-whitened attribution at k=64 outperforms standard RepSim by >= 3pp on at least 1 task, the whitened matched filter concept is validated (just poorly conditioned at pilot scale).
- If PCA-whitened attribution fails at *all* k values and regularization strengths, the whitened matched filter theory has no practical value for these task sizes and must be reported as a negative result.

**Computational cost:** ~3 GPU-hours. PCA and covariance estimation are cheap; the cost is in score computation across the k x lambda grid.

**Success probability:** 55%. The theory predicts whitening should help, but the pilot data and the contrarian's H9 falsification are concerning. Even at N/k=78, the noise covariance may lack the structured anisotropy that whitening exploits.

### Experiment 3.2: Multi-Method Tournament (phi^T M psi Taxonomy Validation)

**Rationale.** The framework claims to unify 5 representation-space methods. If they can be shown to vary systematically along the phi/psi/M axes -- and their relative performance is predictable from the framework -- this is genuine predictive power beyond taxonomy.

**Protocol:**
1. Implement all 5 representation-space methods on DATE-LM:
   - RepSim: phi=h_L(z), M=I (off-the-shelf, cosine similarity)
   - RepT: phi=delta_h(z), M=I (from [GitHub: plumprc/RepT](https://github.com/plumprc/RepT))
   - AirRep: phi=f_theta(z), M=I, learned (from [GitHub: sunnweiwei/AirRep](https://github.com/sunnweiwei/AirRep))
   - In-the-Wild: phi=h_post-h_pre, M=I (temporal diff, requires pre/post checkpoints)
   - Concept Influence: phi=encoder(z), M=I (concept-level)
2. Run each on all 3 DATE-LM tasks. Metrics: R@50, AUPRC, Kendall tau.
3. Additionally: apply contrastive scoring and PCA-whitening to each method as "plug-in" corrections.
4. This yields a 5 methods x 3 scoring modes x 3 tasks = 45-cell matrix.

**Key controls:**
- All methods use same model (Pythia-1B), same data, same evaluation script.
- Methods that require special infrastructure (In-the-Wild needs DPO checkpoints; Concept Influence needs encoder) may not be feasible -- report as limitations.
- Include TRAK, DDA as parameter-space baselines for completeness.

**Framework predictions (testable):**
- Methods with similar phi/psi should have correlated performance patterns across tasks.
- Contrastive scoring should help all methods uniformly (if FM2 is real).
- Whitening should help methods with anisotropic noise more than those with isotropic noise.
- AirRep (learned phi) should dominate other M=I methods because it optimizes the feature map.

**Falsification criteria:**
- If the 5 methods show *no* systematic pattern explainable by their phi/psi/M configuration, the framework is taxonomic but not predictive.
- If method performance is better predicted by "which layer" or "which similarity function" than by the phi^T M psi decomposition, the framework's axes are wrong.

**Computational cost:** ~6-10 GPU-hours depending on method availability. AirRep and RepT have public code; In-the-Wild and Concept Influence may require significant implementation effort.

**Success probability:** 60% for completing 3+ methods; 40% for demonstrating predictive framework power.

### Experiment 3.3: The "Smoking Gun" Residual Analysis

**Rationale.** The 30.8pp gap between TRAK-PCA at k=d and RepSim is the single biggest challenge to the CRA thesis. If FM1 fully explained the rep-vs-param gap, this gap should be near zero. Understanding what accounts for this gap is essential.

**Protocol:**
1. Decompose the RepSim score into components:
   - RepSim(z_test, z_train) = phi(z_test)^T psi(z_train) [representation cosine]
   - TRAK-d(z_test, z_train) = g_PCA(z_test)^T M g_PCA(z_train) [PCA-projected gradient]
2. For each test example, compute:
   - Residual = RepSim_rank - TRAK-PCA_rank (rank difference per training example)
   - Characterize training examples where the residual is largest.
3. Hypotheses for the gap:
   - (a) Nonlinearity: RepSim captures nonlinear feature interactions that linear PCA misses.
   - (b) Layer specificity: RepSim uses last-layer features; gradients mix all layers.
   - (c) Normalization: cosine similarity normalizes by magnitude; TRAK does not.
   - (d) Semantic alignment: representations encode semantic similarity; gradients encode loss-landscape similarity.
4. Test (c) by computing TRAK-PCA with cosine-normalized projected gradients.
5. Test (b) by computing TRAK using only last-layer gradients with PCA projection.

**Falsification criteria:**
- If last-layer-only TRAK-PCA closes to within 5pp of RepSim, the gap is a layer-mixing artifact, not fundamental.
- If cosine-normalized TRAK-PCA closes within 5pp, the gap is a normalization artifact.
- If neither closes the gap, the nonlinearity/semantic hypotheses (a/d) are supported, and FM1 is a partial explanation at best.

**Computational cost:** ~2 GPU-hours.

**Success probability:** 70% for identifying the dominant factor; 30% for fully closing the gap.

---

## Summary: Experimental Priorities and Risk Mitigation

### Priority Ranking

| Rank | Experiment | Criticality | GPU-hours | P(success) | What it proves if successful |
|------|-----------|-------------|-----------|------------|------------------------------|
| 1 | 2.1 Continuous metrics | Critical | 1 | 60% | FM2 exists and is detectable |
| 2 | 1.1 Full-scale eigenspectrum | Critical | 2 | 85% | FM1 quantitative prediction validated |
| 3 | 1.2 TRAK dim sweep (full) | Critical | 8 | 75% | TRAK saturation at k~r_eff at scale |
| 4 | 2.2 Contamination injection | Critical | 2 | 85% | FM2 correction mechanism validated |
| 5 | 1.3 Layer sweep + Euclidean | High | 3 | 90% | Space vs similarity function disambiguation |
| 6 | 3.3 Residual analysis | High | 2 | 70% | Identifies what FM1 misses |
| 7 | 3.1 PCA-whitened attribution | High | 3 | 55% | Whitened MF theory rescued or buried |
| 8 | 2.3 DDA ablation | Medium | 4 | 70% | CRA FM2 = DDA debias equivalence |
| 9 | 3.2 Multi-method tournament | Medium | 8 | 40% | Framework predictive power |
| **Total** | | | **33** | | |

### Minimum Viable Experiment Set (if time-constrained)

Execute experiments 1-4 only (~13 GPU-hours, ~4h wall-clock on 4 GPUs). This produces:
- FM1: Full-scale eigenspectrum + dimension sweep (the strongest part of the thesis).
- FM2: Continuous metrics + contamination injection (filling the critical gap).

### Key Confounders to Control Across All Experiments

1. **Sample size N.** Pilot N=100 is insufficient for covariance estimation, statistical testing, and generalization claims. All full-scale experiments must use N >= 2000.

2. **Random seeds.** Report mean +/- std over 3 seeds (42, 123, 456) for any stochastic method (TRAK, bootstrap CI).

3. **Rank vs continuous metrics.** Always report both. The FM2 fiasco (zero gain on rank metrics) demonstrates that metric choice determines what you can detect.

4. **Task heterogeneity.** Report per-task results. Never average across tasks. DATE-LM's three tasks have qualitatively different characteristics (toxicity is a gradient-norm task; counterfact is a lexical-retrieval task; ftrace is intermediate).

5. **BM25 baseline.** Include on every task. If BM25 matches or beats attribution methods, the task may not require model-internal attribution -- which limits the scope of CRA's claims but is essential intellectual honesty.

6. **Compute budget tracking.** Report wall-clock time and peak GPU memory for every experiment. Reviewers will want to know whether CRA's diagnostic methods are more expensive than the methods they diagnose.

---

## Negative Results to Report Honestly

The following negative results from the pilot are real and should be reported in the paper, not explained away:

1. **Toxicity reversal.** TRAK > RepSim by 24pp on toxicity. This defines a clear scope boundary: FM1 diagnosis applies to attribution tasks (where semantic similarity matters), not detection tasks (where gradient norm is directly informative). Framing this as a "task-type boundary" is acceptable if supported by analysis of what gradient norms capture on toxicity data.

2. **BM25 competitiveness.** BM25 achieves R@50=1.0 on counterfact at pilot scale. At full scale, this may degrade (lexical overlap is less reliable with more candidates), but if it persists, CRA must acknowledge that model-internal attribution adds no value over lexical retrieval for factual attribution tasks.

3. **H9 isotropy falsification.** Representation covariance is NOT near-isotropic. If this persists at N >> d, the theoretical justification for M=I in representation space collapses. The paper must either (a) find an alternative explanation for why M=I works despite anisotropy, or (b) show that whitening helps at proper N/d ratios.

4. **30.8pp TRAK-PCA gap.** FM1 is necessary but not sufficient. The paper must explicitly quantify what fraction of the rep-vs-param gap FM1 explains and what fraction remains unexplained.

---

## Specific Risks and What Would Falsify the Entire CRA Thesis

### Risk 1: FM1 and FM2 are not independent defects but aspects of the same phenomenon (P=30%)

If the full-scale 2x2 ANOVA shows strong negative interaction (fixing FM1 substantially reduces the FM2 effect), the "two independent defects" narrative collapses. The paper would need to be restructured as "a single defect with two manifestations."

**Mitigation:** The 2x2 factorial with continuous metrics (Experiment 2.1) directly tests this. Report the interaction term with bootstrap CI.

### Risk 2: The bilinear framework is vacuously universal (P=40%)

If the multi-method tournament (Experiment 3.2) shows no systematic pattern predictable from phi/psi/M configuration, the framework adds no explanatory power beyond taxonomy.

**Mitigation:** Identify at least one non-trivial, *a priori* prediction that validates at full scale. The strongest candidate: TRAK-PCA at k=r_eff should outperform TRAK-random at k=r_eff (PCA uses the signal subspace; random projection wastes capacity on noise). If this prediction fails, present the framework as taxonomy, not theory.

### Risk 3: The entire rep-vs-param gap is explainable by cosine similarity being a better proxy for semantic relatedness (P=25%)

This is the contrarian's core challenge. If RepSim-Euclidean degrades catastrophically while RepSim-cosine succeeds, and if BM25 (lexical similarity) performs comparably to RepSim on all tasks, then the CRA thesis reduces to "cosine similarity of LLM representations is a good semantic similarity measure" -- which is trivially known and not publishable.

**Mitigation:** Experiment 1.3 (layer sweep + Euclidean) directly tests this. Also, the toxicity task (where RepSim *fails*) provides a natural control: on tasks where semantic similarity is not the signal, RepSim underperforms, confirming that the advantage is task-dependent, not space-dependent.

---

## Literature-Informed Methodological Recommendations

### From Hu et al. (2602.10449) -- Unified Random Projection Theory

Their key result: unregularized projection preserves influence iff m >= rank(F). For CRA, this means the TRAK dimension sweep should show a **hard threshold** at k = rank(F_effective), not a gradual saturation. If the pilot's gradual curve (peak at k=256, non-monotonic after) persists at full scale, it contradicts the sharp threshold prediction and suggests regularization effects (TRAK uses ridge regularization internally) dominate the rank structure.

**Recommendation:** Report whether TRAK's internal regularization strength affects the saturation curve shape. Sweep lambda_TRAK in {0.01, 0.1, 1.0, 10.0} at fixed k=256 and k=2048.

### From Tong et al. (2602.01312) -- TRAK Preserves Rankings

Their result that TRAK preserves *rankings* despite large absolute errors supports using rank metrics. But it also means that rank-invariant transformations (like contrastive scoring) cannot be detected by rank metrics -- exactly the FM2 evaluation failure. CRA must use continuous metrics (LDS, Kendall tau) to escape this trap.

### From Quanda (Bareeva et al., 2410.07158) -- TDA Evaluation Toolkit

Quanda provides a unified evaluation framework with multiple metrics. Consider using their toolkit to ensure metric consistency and reproducibility. Their "meta-evaluation" approach (evaluating evaluation metrics themselves) could help validate whether continuous metrics actually detect FM2 effects.

### From DATE-LM (Jiao et al., 2507.09424) -- Benchmark Design

DATE-LM's finding that "no single method dominates" is exactly what CRA explains. But DATE-LM also found that method performance is "sensitive to task-specific evaluation design" -- which means CRA's task-independent FM1/FM2 narrative may be too ambitious. CRA should explicitly characterize which tasks are "FM1-dominated" (attribution tasks) vs "FM1-irrelevant" (detection tasks like toxicity).

### From DDA (Pang et al., 2410.01285) -- Contrastive Scoring

DDA's 55pp improvement from debias uses *base model subtraction*, not simple mean subtraction. The CRA thesis equates these, but they are different operations:
- CRA mean subtraction: score_contrastive(z) = phi(z) - E[phi(z')]
- DDA debias: score_debias(z) = IF_finetuned(z) - IF_base(z)

The DDA ablation (Experiment 2.3) tests whether these are empirically equivalent. If DDA-debias >> CRA-mean-subtraction, the FM2 model needs refinement.

---

## Final Recommendation

**Lead with the empirical findings, not the theory.** The CRA paper's strongest contribution is the first systematic 2x2 factorial on DATE-LM with mechanistic eigenspectrum evidence. The bilinear framework is valuable as taxonomy but risky as theory (3/3 distinctive predictions failed at pilot scale). I recommend:

1. **Core paper:** 2x2 factorial + eigenspectrum + TRAK dimension sweep + FM2 continuous metrics. This is a strong empirical contribution regardless of the theoretical framework's fate.
2. **Framework as organization, not prediction:** Present phi^T M psi as a way to organize existing methods, with the 2x2 factorial as the empirical contribution. Do not overclaim predictive power unless whitened attribution or contrastive scoring show clear gains at full scale.
3. **Honest reporting of failures:** H7 failure, H9 reversal, toxicity reversal, and BM25 competitiveness all belong in the paper. They strengthen credibility and define scope.
4. **Compute budget:** ~33 GPU-hours for all experiments, ~13 GPU-hours for the minimum viable set. Well within the 4x RTX 4090 budget over 2-3 weeks.

The difference between a rejected paper and a strong publication is not whether the theory is beautiful -- it is whether the experiments are airtight. Design experiments that leave no room for ambiguity, report what the data shows (not what you hoped it would show), and let the narrative follow the evidence.
