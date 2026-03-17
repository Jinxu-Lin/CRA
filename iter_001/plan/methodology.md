# CRA Full-Scale Methodology -- Post-Pilot Revision

## Overview

This methodology reflects the revised CRA thesis after comprehensive pilot experiments (N=100, Pythia-1B, 14 tasks, ~28 min). The plan is restructured into 5 priority tiers with explicit decision gates. Total budget: ~24 GPU-hours, ~7.5h wall-clock on 4x RTX 4090.

**Key revisions from pilot:**
- FM2 evaluation redesigned with continuous metrics (Kendall tau) and controlled contamination injection
- H7 whitened attribution redesigned with PCA-reduced whitening at feasible N/k ratios
- New RQ3 (task-type boundary) and RQ4 (gap decomposition) added
- Retrieval baselines (Contriever/GTR-T5) mandated to test "attribution vs retrieval" boundary
- H9 reframed from condition number comparison to spectral concentration ratio

## Models and Data

- **Primary model**: Pythia-1B (d=2048, ~1B params) for all main experiments
- **Secondary model**: Pythia-70M (d=512, ~70M params) for eigenspectrum analysis at large N
- **Benchmark**: DATE-LM (NeurIPS 2025) with 3 task types:
  - data_selection (counterfact): N_train=5,473, semantic attribution
  - toxicity_filtering: N_train=10,187, behavioral detection
  - factual_attribution (ftrace): semantic attribution
- **Seeds**: [42] for pilots; [42, 123, 456] for full-scale
- **Environment**: 4x RTX 4090 (24GB each), conda env `sibyl_CRA`, PyTorch 2.5.1+cu121

## Methods Under Evaluation

### Parameter-Space Methods
1. **TRAK** (k=2048 default, sweep k in {32..4096}): Random projection + gradient inner products
2. **LoGra**: Structured low-rank gradient projection
3. **DDA**: Debias + denoise attribution (parameter-space with contrastive correction)
4. **K-FAC IF**: Kronecker-factored inverse Fisher (Pythia-70M only, completed in pilot)
5. **Raw Dot IF**: Gradient dot product without Hessian (gradient norm baseline)
6. **Diag IF**: Diagonal Fisher inverse

### Representation-Space Methods
7. **RepSim**: Last-layer cosine similarity (phi^T psi / ||phi|| ||psi||)
8. **k-NN**: k=50 nearest neighbors in representation space

### Retrieval Baselines (NEW -- Priority 3)
9. **BM25**: Lexical retrieval baseline
10. **Contriever**: Dense retrieval model (facebook/contriever)
11. **GTR-T5**: Generalist Text Retrieval model (sentence-transformers/gtr-t5-base)

## Priority 1: FM2 Verification Suite (~3 GPU-hours, ~1h wall-clock)

### Rationale
Pilot showed exactly 0.0pp contrastive gain because rank-based metrics (AUPRC, R@K) are mathematically invariant to mean-subtraction. This is a measurement artifact, not evidence against FM2. We need continuous metrics AND controlled injection to test FM2 causally.

### Experiment 1.1: Continuous Metrics on Full Factorial
- Compute **Kendall tau** and **Spearman rho** on raw attribution scores (before rank transformation) for the full 2x2 factorial {TRAK, RepSim} x {standard, contrastive} on all 3 DATE-LM tasks
- Use full training pool (N=5,473 counterfact, N=10,187 toxicity)
- Also compute score-level NDCG as supplementary continuous metric
- **Pass criteria**: Kendall tau improvement >= 0.05 for TRAK standard -> TRAK contrastive on >= 1 task

### Experiment 1.2: Controlled Contamination Injection (H11)
- For each method M in {RepSim, TRAK}:
  1. Compute raw attribution scores s_i = M(z_test, z_train_i)
  2. Compute mean attribution: mu = mean(s_i over all z_train)
  3. Inject contamination: s_contaminated_i = s_i + alpha * mu, alpha in {0, 0.1, 0.5, 1.0, 2.0, 5.0}
  4. Apply contrastive correction: s_corrected_i = s_contaminated_i - mean(s_contaminated)
  5. Evaluate both contaminated and corrected scores on ALL metrics (rank-based AND continuous)
- **Pass criteria**: At alpha=1.0, contamination degrades rank metric by >= 10pp AND contrastive correction recovers >= 90% of alpha=0 performance

### Experiment 1.3: FM1/FM2 Interaction (H3-revised, contingent on H2)
- If H2-revised passes (FM2 detectable with continuous metrics):
  - 2-way ANOVA on Kendall tau scores: Factor A = space, Factor B = scoring
  - Interaction term vs main effects comparison
- If H2-revised fails: report H3 as untestable

### Decision Gate 1
- **If** Kendall tau gain >= 0.05 AND injection recovery succeeds: FM2 is Tier 1 contribution
- **If not**: FM2 demoted to "theoretical hypothesis"; paper narrows to FM1 + benchmark

## Priority 2: FM1 Diagnostic Suite (~10 GPU-hours, ~3h wall-clock)

### Experiment 2.1: Full-Scale Eigenspectrum (H4-revised, H9-revised)
- **Pythia-70M** at N in {500, 1000, 2000, 5000}:
  - Gradient covariance eigenspectrum via Lanczos (top-500 eigenvalues)
  - Full representation covariance eigenspectrum (d=512, exact at N >= 512)
  - At each N: r_eff(95%), r_eff(99%), condition number, top-5/10/50 explained variance
- Compare r_eff/B (gradient) vs r_eff/d (representation) -- the spectral concentration ratio
- At N=5000 (N >> d=512): representation condition number is reliable; test H9-revised
- **Pass criteria**: Full-model gradient r_eff(95%) in [5, 100]; rep_r_eff/d >> grad_r_eff/B by >= 10x

### Experiment 2.2: TRAK Dimension Sweep at Full Scale (H5-revised)
- **Pythia-1B**, N=5,473 (full counterfact training set):
  - TRAK-random at k in {32, 64, 128, 256, 512, 1024, 2048, 4096}
  - TRAK-PCA at k in {32, 64, 128, 256, 512, 1024, 2048} (project onto top-k gradient eigenvectors)
  - Evaluate with both rank-based (R@50, MRR) AND continuous (Kendall tau) metrics
- **Smoking gun**: If TRAK-PCA at k=d closes to within 15pp of RepSim, FM1 is the primary mechanism

### Experiment 2.3: RepSim Dimension Sweep at Full Scale
- **Pythia-1B**, N=5,473: RepSim with PCA reduction at k in {16, 32, 64, 128, 256, 512, 1024, 2048}
- Identify minimum k for RepSim to maintain near-full performance (within 3pp of k=d)

## Priority 3: Retrieval Baselines (~2.5 GPU-hours, ~1h wall-clock)

### Experiment 3.1: Dense Retrieval Baselines (H8-revised)
- Run Contriever (facebook/contriever) and GTR-T5 (sentence-transformers/gtr-t5-base) on all 3 DATE-LM tasks at full scale
- Encode training + test samples; compute cosine similarity scores
- Evaluate with same metric suite as TDA methods (R@50, MRR, AUPRC, Kendall tau)

### Experiment 3.2: BM25 at Full Scale
- BM25 on all 3 tasks with full training pool (N=5,473 counterfact, N=10,187 toxicity)
- Check if BM25's R@50=1.0 on counterfact (at N=100) persists at full scale

### Decision Gate 3
- **If** Contriever/GTR matches RepSim (< 3pp gap) on 2+ tasks: reposition as "attribution vs retrieval boundary"
- **If** BM25 competitive at full scale: restrict semantic attribution claims

## Priority 4: Mechanism Suite (~5 GPU-hours, ~1.5h wall-clock)

### Experiment 4.1: Gap Decomposition (H10)
- Factor (a): Last-layer-only TRAK-PCA (restrict gradients to final transformer layer)
- Factor (b): Cosine-normalized TRAK-PCA (normalize gradient vectors before projection)
- Factor (c): Combined: last-layer-only + cosine-normalized TRAK-PCA
- All on Pythia-1B, full counterfact, compared to standard TRAK-PCA and RepSim

### Experiment 4.2: Layer Sweep
- RepSim at layers {0, 4, 8, 12, 16, 20, 23} on Pythia-1B, all 3 tasks
- Identify where attribution signal concentrates per task type

### Experiment 4.3: Cosine vs Euclidean RepSim
- RepSim with Euclidean distance on all 3 tasks (contrarian falsification)
- If Euclidean matches cosine, normalization is not a gap factor

## Priority 5: Framework Validation (~3.5 GPU-hours, ~1h wall-clock)

### Experiment 5.1: PCA-Reduced Whitened Attribution (H7-revised)
- Whitened RepSim in PCA subspace: phi_PCA^T Sigma_PCA^{-1} psi_PCA
- k in {16, 32, 64, 128, 256, 512} where N/k ranges from 342 to 11 (at N=5,473)
- Ridge-regularized whitening at optimal lambda (5-fold cross-validated)
- **Pass criteria**: PCA-whitened at k=64 outperforms standard RepSim by >= 3pp on >= 1 task

### Experiment 5.2: OVB Sensitivity Analysis (optional, time permitting)
- Cinelli-Hazlett Robustness Value for FM2 severity quantification

## Metrics

### Rank-Based (established, used in pilot)
- **R@50**: Recall at 50 (counterfact, ftrace)
- **MRR**: Mean Reciprocal Rank
- **AUPRC**: Area Under Precision-Recall Curve (toxicity)

### Continuous (NEW -- critical for FM2 testing)
- **Kendall tau**: Rank correlation on raw attribution scores vs ground-truth relevance
- **Spearman rho**: Monotone rank correlation
- **NDCG**: Normalized Discounted Cumulative Gain
- **Bootstrap CI** (B=1000) for all metrics

## Expected Visualizations

- **Figure 1**: phi^T M psi bilinear framework diagram showing all representation-space methods as special cases
- **Table 1**: Main 2x2 factorial results (rank + continuous metrics, bootstrap CI)
- **Table 2**: Full method tournament (10+ methods, 3 tasks, dual metric suite)
- **Figure 2**: Eigenspectrum comparison (log-log, gradient vs representation, N in {100, 500, 1000, 2000, 5000})
- **Figure 3**: TRAK dimension sweep (R@50 vs k, random vs PCA, with RepSim reference line)
- **Figure 4**: Contamination injection/recovery curves (performance vs alpha, with/without correction)
- **Figure 5**: Layer sweep heatmap (layer x task)
- **Table 3**: Gap decomposition (factor contributions to TRAK-PCA vs RepSim gap)
- **Figure 6**: PCA-whitened performance vs subspace dimension k
- **Table 4**: Task-type boundary characterization (semantic vs behavioral)

## Shared Resources

### Reusable from Pilot (iter_001)
- Conda env `sibyl_CRA` fully configured
- DATE-LM datasets downloaded
- Pythia-70M and Pythia-1B checkpoints cached
- Cached last-layer representations (Pythia-1B, N=100)

### Additional Downloads Needed
- Contriever model: facebook/contriever (~440MB)
- GTR-T5 model: sentence-transformers/gtr-t5-base (~220MB)
- No additional datasets needed (DATE-LM already downloaded)

## Risks and Contingencies

| Risk | Severity | Mitigation |
|------|----------|------------|
| Retrieval models match RepSim (attribution = retrieval) | Critical | Toxicity reversal as genuine attribution; reposition paper |
| FM2 undetectable with continuous metrics | High | Report honest negative; narrow to FM1-only paper |
| PCA whitening fails at all k | Medium | Report negative; framework taxonomic not prescriptive |
| BM25 competitive at full scale on counterfact | Medium | Restrict semantic attribution claims |
| TRAK-PCA gap doesn't decompose cleanly | Medium | Report multi-factorial; "feature quality" narrative |
| Full-scale H9 still reversed | Medium | Alternative explanation for M=I efficacy |
| Full-scale computation exceeds budget | Low | Prioritize P1-P3; P4-P5 can be deferred to next iteration |
