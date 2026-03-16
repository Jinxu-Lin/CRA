# Empiricist Perspective: CRA Research Proposal

**Agent**: Empiricist (reproducibility, controlled comparisons, ablation-first thinking, statistical rigor)
**Date**: 2026-03-16

---

## Executive Summary

The CRA thesis makes three empirically testable claims: (1) FM1 (signal dilution) and FM2 (common influence contamination) are two independent failure modes of parameter-space TDA, (2) five representation-space methods share a phi^T * psi bilinear structure, and (3) systematic evaluation on DATE-LM will confirm the diagnostic framework. As an experimentalist, I am skeptical of all three until they survive controlled experiments with proper baselines, confound controls, and pre-registered falsification criteria. Below I propose three experiment-driven research angles, each designed around the strongest possible methodology: factorial ablation with orthogonality tests, dimension-controlled causal isolation experiments, and a multi-method benchmark tournament with bootstrap confidence intervals. Every angle specifies exact evaluation protocols, baselines, ablation plans, pilot designs (<1 GPU-hour), computational costs, and what result would falsify the hypothesis.

---

## Angle 1: The 2x2 Factorial Ablation -- Testing FM1/FM2 Independence with Statistical Rigor

### Core Insight (Experiment-Driven: Factorial Design with Interaction Testing)

The CRA paper's load-bearing claim is that FM1 (signal dilution) and FM2 (common influence contamination) are *orthogonal* failure modes -- fixing one should help regardless of whether the other is fixed. This is a textbook factorial experiment: 2 factors (space: parameter vs. representation; scoring: standard vs. contrastive), 2 levels each, measured across 3 DATE-LM tasks. The interaction term in a 2x2 ANOVA directly tests the orthogonality claim.

**What I distrust about the current proposal**: Every other perspective treats the 2x2 ablation as a foregone conclusion ("representation > parameter, contrastive > standard, and they're additive"). But there are at least 4 confounders that could produce a misleading result:

1. **Confounder C1 -- Implementation asymmetry**: TRAK uses random projection + LoRA gradients (lossy), while RepSim uses full-dimensional cosine similarity (lossless). Any performance gap could reflect implementation quality, not a fundamental space difference. *Control*: Use the SAME projection dimension for both TRAK (k=2048) and a PCA-reduced RepSim (d_eff=2048), eliminating dimensionality as a confound.

2. **Confounder C2 -- Normalization asymmetry**: Cosine similarity (RepSim) implicitly normalizes vectors, which partially addresses FM2 by removing magnitude-dependent bias. Standard IF does not normalize. *Control*: Test both L2-normalized and unnormalized variants of ALL methods, creating a 2x2x2 design (space x scoring x normalization).

3. **Confounder C3 -- Layer selection**: RepSim typically uses the last layer; TRAK uses all-layer gradients. The "best layer" for representation methods (likely middle layers, per Vitel & Chhabra 2511.04715 and RepT's phase transition layer) may not be the same as the implicit layer weighting in gradient methods. *Control*: Report representation results at 3 layers (first third, middle, last) and report TRAK results with layer-specific gradient projection as well as all-layer.

4. **Confounder C4 -- Negative sample selection for contrastive scoring**: DDA's contrastive scoring depends critically on the choice of negative samples. Random negatives vs. task-structured negatives (e.g., non-toxic samples for toxicity filtering) could produce very different FM2 correction magnitudes. *Control*: Test 3 negative strategies: (a) random, (b) task-structured, (c) in-batch negatives. Report all three.

### Exact Evaluation Protocol

**Metrics** (from DATE-LM, no reimplementation):
- Data Selection: LDS (Linear Datamodeling Score)
- Toxicity Filtering: auPRC (Area Under Precision-Recall Curve)
- Factual Attribution: P@K (Precision at K, K in {1, 5, 10})

**Statistical Analysis**:
- Bootstrap confidence intervals (B=1000 resamples of test queries) for each cell
- Two-way ANOVA on the 2x2 factorial for each task, reporting:
  - Main effect of space (parameter vs. representation)
  - Main effect of scoring (standard vs. contrastive)
  - Interaction term (the CRA orthogonality test)
- Effect sizes (Cohen's d or eta-squared), not just p-values
- Since there are only 3 DATE-LM tasks, do NOT claim significance across tasks; instead report task-specific results and qualitative consistency

**Pre-registered Falsification Criteria**:
- **FM1 falsified if**: RepSim (last-layer, standard scoring) < TRAK (k=2048, standard scoring) - 5pp on LDS for data selection task. This would mean representation space does NOT systematically outperform parameter space.
- **FM2 falsified if**: Contrastive scoring degrades performance by > 3pp on >= 1 of 3 methods (TRAK, RepSim, RepT) on >= 1 of 3 DATE-LM tasks. This would mean contrastive scoring is not universally beneficial.
- **Orthogonality falsified if**: The 2x2 ANOVA interaction term exceeds 30% of the minimum main effect on >= 2 of 3 tasks. This would mean FM1 and FM2 corrections are not independent.

### Baselines

| Method | Space | Scoring | Implementation |
|--------|-------|---------|---------------|
| TRAK (k=2048) | Parameter | Standard | github.com/MadryLab/trak |
| TRAK + mean-subtraction | Parameter | Contrastive | Custom: g(z) - E[g] |
| LoGra (k=2048) | Parameter | Standard | As second parameter-space baseline |
| RepSim (last layer) | Representation | Standard | Cosine similarity |
| RepSim + mean-subtraction | Representation | Contrastive | h(z) - E[h] |
| RepT (phase transition layer) | Representation | Standard | github.com/plumprc/RepT |
| RepT + mean-subtraction | Representation | Contrastive | Custom extension |
| BM25 | Lexical (control) | N/A | DATE-LM's included baseline |
| k-NN (representation) | Non-linear (control) | N/A | k=10 nearest neighbors in repr space |

BM25 and k-NN serve as *sanity checks*: if BM25 beats all attribution methods on factual attribution, the entire TDA enterprise is questioned (DATE-LM already flagged this risk). k-NN tests whether the bilinear (linear) assumption matters.

### Ablation Plan

| Ablation | What it tests | Expected outcome |
|----------|---------------|-----------------|
| Remove contrastive scoring | FM2 contribution | Degradation of 5-15pp on TRAK, 2-5pp on RepSim |
| Vary projection dim k | FM1 severity | TRAK saturates at k ~ d (repr dim) |
| Vary repr layer | Layer sensitivity | Middle layers >= last layer (per RepT) |
| Use exact Hessian (Pythia-70M only) | Hessian vs. FM1/FM2 disentanglement | If exact-Hessian IF still loses to RepSim, FM1/FM2 are NOT Hessian artifacts |
| L2-normalize all features | Normalization confound | Reduces gap between cosine-based and dot-product-based methods |
| Vary negative sample strategy | FM2 robustness | Task-structured > random > in-batch |

### Pilot Study Design (15 min)

- Model: Pythia-1B, DATE-LM data selection task only
- Training set: 10K samples (subsample from DATE-LM)
- Test set: DATE-LM's provided test queries
- Methods: RepSim (standard) + TRAK (k=1024, LoRA gradients)
- Goal: Verify evaluation pipeline matches DATE-LM leaderboard numbers (within 2pp). If not, debug before proceeding.
- Secondary goal: Measure wall-clock time per method to calibrate full experiment budget.

### What Result Would Falsify the Hypothesis

The strongest falsification: Exact-Hessian IF on Pythia-70M matches RepSim performance. This would mean FM1/FM2 are not fundamental to parameter-space methods but are artifacts of Hessian approximation -- the Contrarian's objection would be validated, and the CRA paper's central diagnosis collapses to "Hessian approximation is hard."

**Computational Cost**: ~4 GPU-hours for full factorial (parallelizable across 4x RTX 4090)
**Success Probability**: 60% for clean, publishable results. The 2x2 main effects are almost certain to show the expected pattern (RepSim > TRAK, contrastive > standard). The risk is in the interaction term: if FM1 and FM2 corrections are strongly synergistic (not additive), the orthogonality narrative weakens. Also, BM25 beating attribution methods on factual attribution is a real possibility that would complicate the story.

---

## Angle 2: Dimension-Controlled Causal Isolation of FM1

### Core Insight (New Diagnostic Experiment: Controlled Dimension Sweep as Causal Evidence)

The FM1 claim -- "parameter gradients suffer signal dilution because the attribution signal occupies a low-rank subspace" -- can be directly tested with a controlled dimension sweep. This is not just "run TRAK with different k" (the Pragmatist's version). It is a carefully designed causal isolation experiment that controls for every confound I can identify.

**The Key Experiment: Representation-Space Gradient Projection**

If FM1 is real, the following should hold:
1. Project parameter gradients onto the top-r eigenspace of the gradient covariance --> performance should recover to near-RepSim levels
2. Project representations onto random subspaces of dimension k --> performance should degrade gracefully with k, not catastrophically
3. The crossover point -- where projected-gradient performance equals projected-representation performance -- should occur at k ~ d (representation dimension)

This is a *causal* test, not a correlation: we are intervening on the dimensionality of the feature space while holding everything else constant.

**Confounders and Controls**:

- **Confounder D1 -- Projection quality**: Random projection (TRAK-style) vs. PCA projection (data-adaptive) could give very different results. *Control*: Test both. If PCA dramatically outperforms random projection, this supports the low-rank hypothesis (PCA aligns with the signal subspace, random projection does not).

- **Confounder D2 -- Hessian interaction**: FM1 manifests differently with different Hessian approximations. *Control*: Test dimension sweep with (a) identity Hessian (gradient similarity only), (b) diagonal Hessian, (c) K-FAC Hessian. If the saturation point k* changes with Hessian quality, FM1 and Hessian error are entangled (bad for CRA).

- **Confounder D3 -- Training set size**: The effective rank of gradient covariance depends on N (training samples). With small N, rank(Sigma_g) <= N regardless of model size. *Control*: Measure r_eff at N in {1K, 5K, 10K, 50K} and verify it saturates, confirming that the low-rank structure is intrinsic to the model, not an artifact of small samples.

### Exact Experimental Design

**Phase 1: Eigenspectrum Measurement (15 min pilot)**

On Pythia-70M (B ~ 70M, d = 512):
1. Compute per-sample gradients g_i for 1K DATE-LM training samples
2. Form gradient covariance Sigma_g = (1/N) sum g_i g_i^T (store as low-rank: N x B matrix, compute top-500 eigenvalues via Lanczos)
3. Measure: r_eff(90%), r_eff(95%), r_eff(99%)
4. **Prediction**: r_eff(95%) in [300, 800], approximately O(d) = O(512)
5. **Falsification**: r_eff(95%) > 10*d would mean the signal is NOT low-rank relative to representation dimension

**Phase 2: Dimension Sweep (30 min)**

On Pythia-1B (B ~ 1B, d = 2048), DATE-LM data selection:
1. TRAK with random projection at k in {64, 128, 256, 512, 1024, 2048, 4096, 8192}
2. TRAK with PCA projection at same k values (using top-k eigenvectors of Sigma_g)
3. RepSim with PCA-reduced representations at d_eff in {64, 128, 256, 512, 1024, 2048}
4. Measure LDS at each k/d_eff

**Expected Result**: Three saturation curves:
- TRAK-random: saturates at k* >> d (random projection wastes dimensions)
- TRAK-PCA: saturates at k* ~ d (eigenspace projection captures signal)
- RepSim-PCA: robust to dimension reduction, saturates at much smaller d_eff

**The Smoking Gun**: If TRAK-PCA at k=d matches RepSim performance, FM1 is proven: the parameter-space failure is entirely explained by projecting onto the wrong subspace, and the "right" subspace has dimension ~ d.

**Phase 3: Cross-Model Scaling (30 min)**

Repeat Phase 1 across Pythia-{70M, 160M, 410M, 1B}:
1. Measure r_eff at each scale
2. Plot r_eff vs. B and r_eff vs. d
3. **Prediction**: r_eff scales with d (sublinearly with B)
4. **Falsification**: r_eff scales linearly with B

### What Result Would Falsify the Hypothesis

1. r_eff(95%) > 10 * d at any model scale --> FM1 is not a rank-deficiency problem
2. TRAK-PCA at k=d performs > 10pp worse than RepSim --> representation space has advantages beyond dimensionality (e.g., nonlinear compression in forward pass)
3. r_eff scales linearly with B --> JL-style dilution is the real story, not rank deficiency; CRA's theoretical framing needs revision

**Computational Cost**: ~3 GPU-hours (Lanczos eigendecomposition + dimension sweeps)
**Success Probability**: 65%. The eigenvalue decay is very likely steep (this is well-established for neural networks), but the quantitative prediction r_eff ~ d is the critical test. If r_eff ~ sqrt(B) or r_eff ~ B/log(B), the story changes.

---

## Angle 3: Multi-Method Tournament on DATE-LM -- The phi^T * psi Unification Test

### Core Insight (Systematic Benchmark: Controlled Multi-Method Comparison with Framework Validation)

The phi^T * psi unification claim is that all 5 representation-space methods (RepSim, RepT, AirRep, Concept Influence, In-the-Wild) are instances of a common bilinear form. The Contrarian rightly points out this could be vacuous ("everything linear is bilinear"). The empirical test is whether the framework has *predictive power*: can phi^T * psi decomposition predict which method will perform best on which task, and can we derive a new method from the framework that outperforms existing ones?

**What Can Actually Be Measured**:

1. **Phi/Psi Decomposition Accuracy**: For each method, extract the implicit phi and psi, compute the attribution matrix A = phi * psi^T, and measure how well this bilinear approximation matches the method's actual scores. If the bilinear form captures > 95% of variance, the framework is descriptively accurate.

2. **SVD-Optimal Baseline**: Compute the LOO retraining oracle on a small subset (500 training samples, feasible on Pythia-70M), take SVD of the oracle attribution matrix, and use the top-r singular vectors as phi* and psi*. This is the *best possible* phi^T * psi method. Measure how close each existing method comes to this optimum.

3. **Framework-Derived Method**: From the phi^T * psi analysis, identify if any combination of existing phi and psi choices has not been tried. For example: RepT's phi (representation + gradient) with In-the-Wild's psi (activation difference). If this novel combination outperforms both parent methods, the framework demonstrates genuine predictive power.

4. **Non-Linear Control**: Include k-NN attribution (k=10, representation space) and Daunce-style covariance attribution as methods that fall outside phi^T * psi. If these outperform ALL bilinear methods, the framework's coverage is incomplete.

### Exact Evaluation Protocol

**Methods to Evaluate** (all on Pythia-1B, DATE-LM):

| Method | phi(z) | psi(z_i) | Category |
|--------|--------|----------|----------|
| RepSim | h_L(z) | h_L(z_i) | Bilinear (framework) |
| RepT | h_L(z) + alpha * grad_h L(z) | h_L(z_i) | Bilinear (framework) |
| AirRep-lite | Linear projection of h(z) | Linear projection of h(z_i) | Bilinear (framework) |
| Concept Influence proxy | v_c^T h(z) | v_c^T h(z_i) | Bilinear (framework, rank-1) |
| In-the-Wild proxy | h_post(z) - h_pre(z) | h_post(z_i) - h_pre(z_i) | Bilinear (framework) |
| TRAK (k=2048) | P*g(z) | P*g(z_i) | Bilinear (parameter-space) |
| SVD-optimal (r=50) | U_r^T h(z) | Sigma_r V_r^T h(z_i) | Bilinear (oracle-derived) |
| Framework-derived hybrid | RepT-phi + InTheWild-psi | Cross-method | Bilinear (novel) |
| k-NN (k=10) | N/A | N/A | Non-linear control |
| BM25 | N/A | N/A | Lexical control |

**For each method**, report:
- Performance on all 3 DATE-LM tasks (LDS, auPRC, P@K)
- Wall-clock time (scoring + evaluation)
- Cosine alignment of phi/psi with SVD-optimal phi*/psi*

**Statistical Protocol**:
- Bootstrap CI (B=1000) for all metrics
- Pairwise method comparison via bootstrap hypothesis testing (is method A significantly better than method B?)
- Rank correlation (Kendall's tau) between phi/psi alignment with SVD-optimal and actual performance ranking

### Pre-registered Falsification Criteria

- **Unification falsified if**: The bilinear approximation explains < 80% of variance in any method's actual attribution scores. This would mean the phi^T * psi decomposition loses critical information.
- **Predictive power falsified if**: Rank correlation between phi/psi SVD-alignment and performance ranking has tau < 0.3 (p > 0.1). This would mean the framework cannot predict which method works best.
- **Completeness falsified if**: k-NN outperforms the best bilinear method by > 5pp on >= 2 of 3 tasks. This would mean nonlinear attribution is fundamentally superior.
- **Framework novelty falsified if**: The framework-derived hybrid method performs worse than both parent methods on all 3 tasks. This would mean the framework does not generate useful new methods.

### Pilot Study Design (15 min)

- Model: Pythia-70M, DATE-LM data selection task
- Training set: 500 samples (for LOO oracle feasibility)
- Methods: RepSim, TRAK (k=512), SVD-optimal (r=20), k-NN
- Goal: (a) Verify LOO oracle is computable in reasonable time, (b) measure SVD effective rank of oracle attribution matrix, (c) quick sanity check on bilinear approximation accuracy.

### What Result Would Falsify the Hypothesis

The strongest falsification: k-NN (a trivially simple non-linear method) outperforms ALL bilinear methods including the SVD-optimal one. This would mean:
1. The attribution relationship is fundamentally nonlinear
2. The phi^T * psi framework, while mathematically valid, misses the critical structure
3. The paper should pivot from "bilinear unification" to "why simple non-parametric methods work"

The second-strongest falsification: SVD-optimal has no meaningful advantage over RepSim. This would mean the attribution matrix is already well-approximated by raw representation similarity, and the entire framework adds mathematical sophistication without empirical payoff.

**Computational Cost**: ~5 GPU-hours (LOO oracle on 500 samples is the bottleneck)
**Success Probability**: 55%. This is the most ambitious angle. The LOO oracle computation is feasible but slow. The SVD analysis is novel -- nobody has done this for TDA on LLMs. The risk is that the oracle attribution matrix is NOT low-rank, making the bilinear optimality story weak.

---

## Critical Confounders Across All Angles

### Confounder Matrix

| Confounder | Affects | Detection | Mitigation |
|-----------|---------|-----------|------------|
| Implementation asymmetry (TRAK=lossy, RepSim=lossless) | Angle 1, 3 | Compare at matched dimensionality | Use same k for both; also include LoGra as second parameter baseline |
| Cosine normalization implicitly debiases | Angle 1 | Compare cosine vs. dot product for RepSim | Test both normalized and unnormalized |
| Layer selection | Angle 1, 3 | Report multiple layers | Use RepT's phase transition layer as principled selection |
| DATE-LM task heterogeneity | All | Tasks test different attribution properties | Report per-task and do NOT average across tasks |
| Small N for oracle computation | Angle 3 | LOO oracle only feasible with small N | Use N=500 for oracle, validate with subsampling stability test |
| Hessian quality vs. FM1/FM2 | Angle 1, 2 | Include exact-Hessian control | Exact Hessian on Pythia-70M isolates this confound |
| Pre-training vs. fine-tuning checkpoint availability | Angle 1 (contrastive) | In-the-Wild requires both checkpoints | Use DATE-LM's provided checkpoints (both pre and post fine-tuning) |
| Random seed sensitivity | All | Methods with stochastic components (TRAK, LoGra) may vary | Report mean +/- std over 3 random seeds for all stochastic methods |

### The One Confounder Nobody Has Addressed: BM25

DATE-LM's own finding that BM25 (a pure lexical method) sometimes matches attribution methods is deeply troubling. If BM25 performs comparably to RepSim on factual attribution, it suggests that "representation similarity" may simply be detecting lexical overlap, not genuine model-internal attribution. The CRA paper MUST include BM25 as a control and explicitly discuss this possibility. If BM25 beats all representation methods on any task, that section of the paper needs a candid "limitations" discussion rather than sweeping it under the rug.

---

## Synthesis: Recommended Experiment Priority

| Priority | Experiment | Time | GPU-hours | P(clean result) | What it proves |
|----------|-----------|------|-----------|-----------------|---------------|
| **P0** | Pilot: RepSim + TRAK on DATE-LM data selection | 15 min | 0.25 | 90% | Pipeline works; baseline numbers match leaderboard |
| **P1** | 2x2 Factorial (Angle 1) with L2-norm control | 2h | 2 | 75% | FM1/FM2 main effects + interaction term |
| **P2** | Exact-Hessian control on Pythia-70M | 2h | 2 | 50% | FM1/FM2 are NOT Hessian artifacts (critical control) |
| **P3** | Dimension sweep (Angle 2, Phases 1-2) | 1.5h | 1.5 | 65% | FM1 is rank-deficient signal (mechanistic evidence) |
| **P4** | Multi-method tournament (Angle 3, without oracle) | 3h | 3 | 70% | phi^T * psi descriptive accuracy |
| **P5** | SVD-optimal oracle (Angle 3, full) | 5h | 5 | 55% | phi^T * psi predictive/generative power |
| **P6** | Cross-model scaling (Angle 2, Phase 3) | 1h | 1 | 65% | FM1 scales with d, not B |

**Critical path**: P0 --> P1 --> P2 --> P3. If P2 shows exact-Hessian IF matches RepSim, the entire project pivots. If P1 interaction terms are too large, the orthogonality narrative needs revision. P3 provides the mechanistic evidence for FM1. P4-P6 are high-value extensions that strengthen the paper but are not essential.

**Total budget for core experiments (P0-P3)**: ~6 GPU-hours, well within 1-day budget on 4x RTX 4090.

---

## Negative Results I Expect (and the Paper Should Report)

1. **BM25 will beat attribution methods on factual attribution.** DATE-LM already hints at this. Facts are encoded lexically; representation similarity captures lexical overlap. The paper should acknowledge this and argue that factual attribution is not where representation-space TDA adds value -- data selection and toxicity filtering are the strong suits.

2. **Contrastive scoring will help parameter-space methods more than representation-space methods.** RepSim's cosine normalization already partially removes the mean component (FM2). Explicitly mean-subtracting representations may produce only marginal improvement (2-3pp) compared to the dramatic improvement for TRAK (10-20pp). This is actually *predicted* by the theory (representation space has smaller ||phi_shared||/||phi_task|| ratio) and should be framed as a confirmatory result.

3. **AirRep-lite (linear projection) will underperform full AirRep.** The learned nonlinear encoder captures structure that a simple linear projection misses. This limits the phi^T * psi framework's explanatory power for AirRep specifically, and should be reported honestly.

4. **The SVD-optimal method will NOT dramatically outperform RepSim.** If representations are already well-aligned with the oracle attribution matrix's singular structure (as the Theoretical perspective predicts), there is little room for improvement. This is actually good news for the framework -- it means RepSim is near-optimal within the bilinear class -- but limits the "generate new methods" angle.

---

## References

### Core Papers (Directly Evaluated)
- DATE-LM (2507.09424) -- Primary benchmark; 3-task evaluation protocol
- Li et al. (2409.19998) -- IF vs RepSim systematic comparison
- DDA (2410.01285) -- Contrastive scoring; debias 55pp vs. denoise 9pp
- TRAK (2303.14186) -- Parameter-space random projection baseline
- RepT (2510.02334) -- Representation gradient tracing; phase transition layer
- AirRep (2505.18513) -- Learned representation attribution
- Better Hessians Matter (2509.23437) -- Hessian decomposition; competing explanation
- LoRIF (2601.21929) -- Low-rank IF; quality-scalability tradeoff with projection dim D
- Hu et al. (2602.10449) -- Unified random projection theory; sketch dim >= rank(F)

### Methodology and Evaluation References
- Schioppa (2402.03994) -- Efficient gradient/HVP sketching; intrinsic dimension measurement
- Scalable Forward-Only TDA (2511.19803) -- Forward-only attribution; matches TRAK on LDS/LOO
- Vitel & Chhabra (2511.04715) -- Layer selection evidence: middle attention layers outperform first/last
- Pan et al. (2502.11411) -- Denoised Representation Attribution; token-level denoising
- Daunce (2505.23223) -- Non-gradient attribution via ensemble perturbation covariance
- CKA pathologies: Davari et al. (2210.16156), Cloos et al. (2407.07059), Okatan et al. (2511.01023)

### Statistical Methodology
- Bootstrap CI for attribution metrics: standard in DATE-LM evaluation protocol
- Factorial ANOVA for 2x2 designs: foundational experimental design
- Pre-registration of falsification criteria: following Open Science Framework best practices
