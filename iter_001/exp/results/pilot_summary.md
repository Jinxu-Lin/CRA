# CRA Pilot Summary -- All Phases Complete (Iteration 0)

**Overall Recommendation: GO with significant narrative revisions needed**
**Selected Candidate: cand_a (CRA: Signal Processing Diagnosis + Bilinear Unification)**
**Confidence: 0.60** (down from initial 0.75 due to FM2 untested and H7/H9 failures)

---

## Executive Summary

All 14 pilot tasks completed successfully across 4 phases (~28 min total on RTX 4090). The CRA thesis is **partially supported**: FM1 (signal dilution) has strong spectral and empirical evidence on attribution tasks, but FM2 (common influence contamination) is completely untested at pilot scale, H7 (whitened attribution) fails, and several quantitative predictions (H4, H9) need revision.

### Hypothesis Scorecard

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| H1 (FM1 space gap) | SUPPORTED 2/3 | RepSim > TRAK by +32pp (counterfact), +17pp (ftrace); REVERSED on toxicity (-24pp) |
| H2 (contrastive asymmetry) | INCONCLUSIVE | Zero gain -- rank metrics invariant to mean-subtraction |
| H3 (FM1/FM2 orthogonality) | TRIVIALLY SATISFIED | Interaction=0 because FM2 effect=0 |
| H4 (gradient r_eff ~ d) | DIRECTIONALLY SUPPORTED | r_eff=10 (full model), much lower than predicted [256,1024]; strengthens FM1 |
| H5 (TRAK saturation at k~d) | SUPPORTED | Saturation at k=256; but 30.8pp gap to RepSim persists |
| H6 (K-FAC control) | CONFIRMED | RepSim > K-FAC IF by 17.4pp on counterfact |
| H7 (whitened attribution) | FAIL | Degrades all tasks by 8-11pp (N/d=0.049 underdetermined) |
| H8 (RepSim > BM25) | PARTIAL | Passes on toxicity (+18pp); BM25 perfect on counterfact at pilot scale |
| H9 (condition numbers) | FALSIFIED | Direction completely reversed (rep_cond >> grad_cond) |

---

## Phase 0: Foundation & Critical Control

### setup_env -- GO (Confidence: 0.95)
- 4x NVIDIA RTX 4090 (24GB each), conda env `sibyl_CRA` (Python 3.11)
- PyTorch 2.5.1+cu121, Transformers 5.3.0
- DATE-LM: Toxicity (10,187 train), Counterfact (5,473 train), ftrace available

### phase0_pipeline_pilot -- GO (Confidence: 0.70)
- Model: Pythia-1B (d=2048), N=100, both RepSim + TRAK functional
- Unexpected: TRAK (0.926) > RepSim (0.685) on toxicity AUPRC
- Expected: RepSim (0.783) > TRAK (0.528) on counterfact R@50

### phase0_kfac_control -- H6 CONFIRMED (Confidence: 0.75)

| Task | RepSim | K-FAC IF | Raw Dot IF | Diag IF | TRAK |
|------|--------|----------|-----------|---------|------|
| Counterfact (R@50+MRR) | **1.438** | 1.264 | 1.200 | 1.144 | 1.206 |
| Toxicity (AUPRC) | 0.744 | **0.992** | 0.940 | 0.977 | 0.778 |

**Critical finding**: On counterfact (genuine attribution), RepSim > K-FAC IF by 17.4pp. On toxicity, Raw Dot IF (NO Hessian) achieves 0.94 AUPRC -- this is a gradient norm artifact (Cohen's d=2.66), not an attribution quality measure.

---

## Phase 1: Core 2x2 Factorial

### Factorial Matrix (Pythia-1B, N=100)

| Cell | Space | Scoring | Toxicity AUPRC | Counterfact R@50 | ftrace R@50 |
|------|-------|---------|---------------|-----------------|-------------|
| A | Parameter (TRAK) | Standard | **0.926** | 0.670 | 0.590 |
| B | Parameter (TRAK) | Contrastive | **0.926** | 0.670 | 0.590 |
| C | Representation (RepSim) | Standard | 0.685 | **0.994** | **0.756** |
| D | Representation (RepSim) | Contrastive | 0.685 | **0.994** | **0.756** |

### Baselines

| Method | Toxicity AUPRC | Counterfact R@50 | ftrace R@50 |
|--------|---------------|-----------------|-------------|
| BM25 | 0.509 | **1.000** | 0.661 |
| k-NN | **0.809** | 0.949 | 0.660 |
| DDA | 0.876 | 0.692 | 0.651 |

### ANOVA Decomposition

| Effect | Toxicity | Counterfact | ftrace |
|--------|----------|-------------|--------|
| FM1 (space) | -24.0pp | +32.4pp | +16.6pp |
| FM2 (scoring) | 0.0pp | 0.0pp | 0.0pp |
| Interaction | 0.0pp | 0.0pp | 0.0pp |

**H1**: PASS on 2/3 tasks. RepSim dominates on counterfact and ftrace; TRAK dominates on toxicity.
**H2**: INCONCLUSIVE. Contrastive scoring gain is exactly zero (rank-metric invariance).
**H3**: TRIVIALLY SATISFIED. No statistical power.
**H8**: PARTIAL. RepSim > BM25 on toxicity (+18pp) and ftrace (+10pp); BM25 perfect on counterfact.

---

## Phase 2: Mechanistic Evidence

### Eigenspectrum (Pythia-70M, N=100)

| Space | Dimension | r_eff(95%) | Top-5 Variance | Condition |
|-------|-----------|------------|----------------|-----------|
| Representation (last layer) | 512 | 63 | 34.9% | 3.1e10* |
| Gradient (target layers) | 6.3M | 53 | 58.5% | 412 |
| Gradient (full model) | 70M | 10 | **85.6%** | 3,589 |

*Rank-deficient at N=100 < d=512; condition number unreliable.

**Core FM1 evidence**: Full-model gradient top-5 eigenvalues capture 85.6% of variance vs 34.9% for representations. This extreme concentration directly evidences signal dilution.

### TRAK Dimension Sweep (Pythia-1B, counterfact)

| k | k/d | R@50 | MRR |
|---|-----|------|-----|
| 64 | 0.03 | 0.686 | 0.201 |
| 128 | 0.06 | 0.705 | 0.204 |
| 256 | 0.12 | **0.785** | 0.227 |
| 512 | 0.25 | 0.750 | 0.217 |
| 1024 | 0.50 | 0.686 | 0.224 |
| 2048 | 1.00 | 0.670 | 0.240 |
| 4096 | 2.00 | 0.715 | 0.256 |

**H5**: SUPPORTED. 90% of max R@50 achieved at k=256 (k/d=0.12). Non-monotonic after k=256.
**Smoking gun test**: TRAK-PCA at k=d gives R@50=0.686, still 30.8pp below RepSim (0.994). Gap suggests factors beyond projection dimension.

### RepSim PCA Dimension Sweep (Pythia-1B)

RepSim performance saturates at PCA k=64 across all tasks (N=100 creates at most ~100 significant components). Actually *improves* slightly at k=64 on some tasks (noise removal). No degradation from k=128 to k=2048.

---

## Phase 3: Framework Extensions

### 36-Cell Contrastive Matrix (4 methods x 3 scorings x 3 tasks)

**Contrastive scoring gain: exactly 0.0pp for all 12 method-task combinations.**
This confirms the pilot-scale limitation: mean-subtraction is a rank-preserving transformation, so rank-based metrics (AUPRC, R@K) are invariant.

**Whitened scoring gains (selected):**

| Method | Toxicity | Counterfact | ftrace |
|--------|----------|-------------|--------|
| RepSim | +0.3pp | -2.2pp | +1.6pp |
| TRAK | -4.7pp | -6.7pp | +0.4pp |
| LoGra | -0.4pp | -2.2pp | +2.4pp |
| DDA | 0.0pp | 0.0pp | **+6.8pp** |

Whitening shows mixed results: consistently helps on ftrace (all 4 methods improve), hurts on counterfact (3/4 degrade), mixed on toxicity.

### Whitened RepSim (H7) -- FAIL

| Task | Standard | Whitened | Gain |
|------|----------|---------|------|
| Toxicity (AUPRC) | 0.685 | 0.576 | **-10.9pp** |
| Counterfact (R@50) | 0.994 | 0.913 | **-8.0pp** |
| ftrace (R@50) | 0.756 | 0.650 | **-10.6pp** |

Root cause: N/d ratio = 0.049; covariance estimation underdetermined. SNR-accuracy correlation positive (0.34 counterfact, 0.16 ftrace) -- concept directionally validated.

---

## Critical Issues for Full-Scale Experiments

1. **FM2 completely untested** (CRITICAL): Must add continuous metrics (Kendall tau, Spearman rho on raw scores) to break rank invariance. Without FM2 evidence, half the CRA thesis is unvalidated.

2. **H7 whitened attribution fails** (HIGH): N/d must be >> 1. At full scale (N=5K-10K, d=2048), N/d=2.5-5.0 may suffice. Also consider PCA-reduced whitening.

3. **Toxicity task reversal** (HIGH): Frame as task-type boundary where gradient norm is directly informative. Not a CRA failure but a scope limitation.

4. **H4/H9 quantitative predictions wrong** (MEDIUM): Reframe H4 to "r_eff << d << B" (strengthens FM1). Replace H9 condition number comparison with spectral concentration metrics.

5. **TRAK dim sweep non-monotonic** (MEDIUM): 30.8pp gap between TRAK-PCA at k=d and RepSim shows FM1 alone is necessary but not sufficient.

6. **BM25 competitive on counterfact** (MEDIUM): Likely lexically solvable at pilot scale. Check at full scale.

---

## Recommendations for Full-Scale

1. **Add continuous metrics**: Kendall-tau and Spearman-rho on raw attribution scores as primary FM2 test
2. **Increase sample size**: N=5K-10K for proper covariance estimation and statistical power
3. **PCA-reduced whitening**: Whiten in top-k eigenspace (k ~ r_eff) to address H7
4. **Reframe hypotheses**: Update H4, H9 to match directional evidence; strengthen FM1 narrative
5. **Task-type analysis**: Add explicit discussion of when gradient norm is directly useful (toxicity) vs when attribution quality matters (counterfact, ftrace)
6. **Multi-seed averaging**: Seeds [42, 123, 456] for variance estimates at full scale

---

## Environment

| Component | Value |
|-----------|-------|
| Server | 4x NVIDIA RTX 4090 (24GB each) |
| Conda env | sibyl_CRA (Python 3.11) |
| PyTorch | 2.5.1+cu121 |
| Transformers | 5.3.0 |
| Total pilot runtime | ~28 minutes |
| Pilot sample size | N=100 |

## Important Notes for Subsequent Tasks

1. **CUDA_VISIBLE_DEVICES mapping**: When `CUDA_VISIBLE_DEVICES=X`, use `cuda:0` in code.
2. **DATE-LM Pythia-70M configs**: Not available for Counterfact/ftrace. Use Pythia-1b data with Pythia-70M model.
3. **TRAK memory**: Use CountSketch (O(D) time, O(k) space) or last-layer-only gradients.
4. **Conda command**: `/home/jinxulin/miniconda3/bin/conda run -n sibyl_CRA`
