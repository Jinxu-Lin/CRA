# CRA: Experiment Methodology

## Overview

This experiment plan validates the CRA thesis through a phased approach: (1) a critical Hessian control experiment to confirm FM1/FM2 are genuine signal processing defects, (2) a hardened 2x2 factorial on DATE-LM to measure FM1 and FM2 main effects and their interaction, (3) mechanistic spectral evidence, and (4) framework extensions including whitened attribution.

All experiments target the **DATE-LM benchmark** (Jiao et al., NeurIPS 2025) with three tasks:
- **Data Selection**: LDS (Linear Datamodeling Score)
- **Toxicity Filtering**: auPRC (Area Under Precision-Recall Curve)
- **Factual Attribution**: P@K (Precision at K)

Primary models: **Pythia-70M** (spectral analysis, K-FAC control) and **Pythia-1B** (main factorial, dimension sweep).

## Experimental Setup

### Models
- **Pythia-70M** (d=512, B~70M): Used for computationally expensive spectral analysis (eigendecomposition, K-FAC IF) where full-parameter operations are needed.
- **Pythia-1B** (d=2048, B~1B): Primary model for the factorial experiment, dimension sweep, and whitened attribution -- the scale where parameter-space methods demonstrably fail.

### DATE-LM Benchmark Configuration
- Three evaluation tasks: data_selection (LDS), toxicity_filtering (auPRC), factual_attribution (P@K)
- Training pool: DATE-LM provided training sets per task
- Evaluation: DATE-LM standard evaluation protocol with bootstrap CI (B=1000)
- Pilot: 100 samples, seed=42, timeout=900s
- Full: complete DATE-LM evaluation sets, seeds=[42, 123, 456]

### Attribution Methods
| Method | Space | M in phi^T M psi | Contrastive variant |
|--------|-------|-------------------|---------------------|
| RepSim | Representation | I | RepSim-C (mean-subtracted) |
| TRAK | Parameter | Random projection | TRAK-C (mean-subtracted) |
| LoGra | Parameter | Structured projection | LoGra-C (mean-subtracted) |
| DDA | Parameter | Debias+Denoise | Already contrastive (DDA debias ~ FM2 fix) |
| BM25 | Lexical | N/A | N/A |
| k-NN | Representation | N/A (nonlinear) | N/A |
| K-FAC IF | Parameter | H^{-1}_{K-FAC} | K-FAC IF-C (mean-subtracted) |
| Whitened RepSim | Representation | Sigma_noise^{-1} | Whitened-C |

### Contrastive Scoring Protocol
Mean-subtraction deconfounding: for method f, the contrastive score is:
```
s_C(z_test, z_train) = f(z_test, z_train) - E_{z' in D_train}[f(z_test, z')]
```
The expectation is estimated over the full training pool (or a 10K subsample for computational efficiency).

### Baselines
- **BM25**: Lexical matching baseline (no model-internal information). Implemented via rank-bm25 library.
- **k-NN**: Cosine k-NN in representation space (nonlinear control; tests whether linear attribution structure matters).
- **DDA**: Strong parameter-space baseline with built-in debias (tests whether DDA's debias is equivalent to our FM2 fix).

## Phase 0: Foundation & Critical Control (H6)

### Task 0a: Pipeline Validation Pilot
- Run RepSim (last-layer cosine) + TRAK (k=2048) on Pythia-1B with DATE-LM data_selection task
- Purpose: validate pipeline correctness, establish baseline numbers, confirm DATE-LM integration works
- Success: both methods produce valid LDS scores; RepSim > TRAK (sanity check)

### Task 0b: K-FAC Hessian Control (H6 -- CRITICAL)
- Run K-FAC full-eigendecomp IF on Pythia-70M for DATE-LM data_selection
- Compare K-FAC IF vs RepSim (last-layer) on same data
- **Decision gate**: If K-FAC IF achieves <5pp gap with RepSim, CRA thesis requires fundamental revision -- pivot to cand_b (Hessian Quality Diagnosis)
- Expected: K-FAC IF still underperforms RepSim by >=10pp, confirming FM1/FM2 are independent of Hessian quality

## Phase 1: Core 2x2 Factorial (H1, H2, H3, H8)

### Factorial Design
2x2: {parameter-space (TRAK), representation-space (RepSim)} x {standard, contrastive scoring}

| Cell | Space | Scoring | Method |
|------|-------|---------|--------|
| A | Parameter | Standard | TRAK (k=2048) |
| B | Parameter | Contrastive | TRAK-C (k=2048, mean-subtracted) |
| C | Representation | Standard | RepSim (last-layer, cosine) |
| D | Representation | Contrastive | RepSim-C (last-layer, mean-subtracted cosine) |

Run on Pythia-1B, all three DATE-LM tasks. Bootstrap CI (B=1000).

### Additional Controls (run in parallel)
- BM25 on all 3 tasks
- k-NN (representation space, cosine, k=50) on all 3 tasks
- DDA on data_selection task (strongest parameter-space baseline)

### Analysis
- H1: Compare C vs A (RepSim standard vs TRAK standard). Pass: gap >= 5pp on >= 2/3 tasks.
- H2: Compare B vs A (TRAK-C vs TRAK) and D vs C (RepSim-C vs RepSim). Pass: parameter-space gain > representation-space gain.
- H3: 2-way ANOVA interaction term. Pass: interaction < 30% of min(main_effect_FM1, main_effect_FM2) on >= 2/3 tasks.
- H8: Compare C vs BM25. Pass: RepSim > BM25 by >= 10pp on data_selection and toxicity_filtering.

## Phase 2: Mechanistic Evidence (H4, H5, H9)

### Task 2a: Gradient Covariance Eigenspectrum (H4, H9)
- Compute gradient covariance matrix eigenspectrum on Pythia-70M using Lanczos iteration (top-500 eigenvalues)
- Compute representation covariance eigenspectrum (exact, d=512)
- Measure r_eff(95%) for both spaces
- H4 pass: r_eff(Sigma_g) in [0.5d, 2d] = [256, 1024]
- H9 pass: representation condition number < 100, gradient condition number > 10^4

### Task 2b: TRAK Dimension Sweep (H5)
- Run TRAK on Pythia-1B with k in {64, 128, 256, 512, 1024, 2048, 4096} for DATE-LM data_selection
- Plot LDS vs k to identify saturation knee
- H5 pass: 90% of max LDS achieved by k = 2d = 4096, with < 5% additional improvement from k=2d to k=10d
- Supplementary: TRAK-PCA (project onto top-k eigenvectors of Sigma_g) at k=d=2048 -- "smoking gun" test

### Task 2c: RepSim Dimension Reduction Sweep
- Run RepSim with PCA dimension reduction: k in {64, 128, 256, 512, 1024, 2048} (full d=2048)
- Cross-validates H4 from representation side: performance should degrade gracefully until k << d

## Phase 3: Framework Extensions (H7)

### Task 3a: Contrastive Scoring Universal Plug-in
- 36-cell matrix: {RepSim, TRAK, LoGra, DDA} x {standard, contrastive, whitened} x {data_selection, toxicity_filtering, factual_attribution}
- Purpose: validate that contrastive scoring is a universal FM2 fix across methods

### Task 3b: Whitened Attribution (H7)
- Compute Sigma_noise (noise covariance) from training representations using Ledoit-Wolf shrinkage
- Implement whitened RepSim: phi^T Sigma_noise^{-1} psi
- Compare whitened vs standard vs contrastive RepSim on all 3 DATE-LM tasks
- H7 pass: whitened RepSim > standard RepSim by 3-8pp on factual_attribution (highest FM2 severity)

## Expected Visualizations

- **Architecture diagram**: CRA bilinear framework phi^T M psi with method taxonomy
- **Table 1**: 2x2 factorial results (4 cells x 3 tasks, with bootstrap CI)
- **Table 2**: Full method comparison including baselines (BM25, k-NN, DDA)
- **Figure 1**: Eigenspectrum plot (gradient vs representation covariance, log-log scale)
- **Figure 2**: TRAK dimension sweep saturation curve (LDS vs k, with d marker)
- **Figure 3**: Contrastive scoring gain heatmap (method x task)
- **Figure 4**: Whitened vs standard attribution comparison (bar chart per task)
- **Table 3**: Method taxonomy under phi^T M psi (method, phi, M, psi, FM1 fix, FM2 fix)

## Shared Resources

### Datasets
- DATE-LM benchmark data (all 3 tasks): download once, share across all experiments
- Pre-tokenized Pythia training data subsets (if DATE-LM requires)

### Checkpoints
- Pythia-70M: EleutherAI/pythia-70m (HuggingFace)
- Pythia-1B: EleutherAI/pythia-1b (HuggingFace)

### Libraries
- `trak` (TRAK implementation)
- `transformers` + `torch`
- `rank-bm25` (BM25 baseline)
- `scikit-learn` (Ledoit-Wolf, PCA, k-NN)
- `scipy` (Lanczos eigendecomp, bootstrap)
- DATE-LM evaluation toolkit (if available as package, otherwise reproduce from paper)

## Risk Mitigations

| Risk | Detection | Mitigation |
|------|-----------|------------|
| K-FAC IF matches RepSim (H6 falsified) | Phase 0 pilot | Pivot to cand_b; stop further CRA experiments |
| DATE-LM not publicly available | Setup phase | Reproduce evaluation protocol from paper; contact authors |
| Pythia-1B OOM on full gradient operations | Phase 1 pilot | Use gradient checkpointing; reduce batch size; fall back to Pythia-410M |
| TRAK library incompatible with Pythia | Phase 0 pilot | Use custom TRAK implementation (random projection + JL lemma) |
| Eigendecomp too slow on Pythia-70M | Phase 2 pilot | Reduce Lanczos iterations; use stochastic trace estimator |
| Whitened attribution numerically unstable | Phase 3 pilot | Increase Ledoit-Wolf shrinkage; use pseudo-inverse with eigenvalue floor |
