# CRA: Why Parameter-Space Attribution Fails on LLMs -- A Signal Processing Diagnosis and Empirical Unification of Representation-Space Methods

## Title

**Signal Dilution, Not Hessian Error: Diagnosing Parameter-Space TDA Failure on LLMs via Gradient Spectral Analysis and a Systematic Representation-Space Benchmark**

## Abstract

Training Data Attribution (TDA) methods systematically underperform on large language models when operating in parameter space. We diagnose **FM1 (Signal Dilution)** as the primary failure mode: attribution signal occupies a low-rank subspace (r_eff ~ 10) within the B-dimensional gradient space, causing signal-to-noise collapse. A controlled K-FAC experiment confirms this is independent of Hessian approximation quality. We unify five independently proposed representation-space methods under a bilinear framework phi(z_test)^T M psi(z_train), where operating in R^d implicitly addresses FM1. Through a systematic 2x2 factorial experiment on the DATE-LM benchmark with both rank-based and continuous metrics, we provide the first controlled comparison across three task types, revealing a sharp task-type boundary: representation-space methods dominate on semantic attribution tasks (counterfact +32pp, ftrace +17pp) while parameter-space methods dominate on behavioral detection tasks (toxicity +24pp). A TRAK dimension sweep demonstrates saturation at k ~ d/8, directly measuring the signal subspace. We investigate FM2 (Common Influence Contamination) through controlled contamination injection experiments with continuous metrics, and explore PCA-reduced whitened attribution as a principled method improvement. Our systematic residual analysis decomposes the 30.8pp gap between TRAK-PCA and RepSim into specific mechanistic factors.

## Motivation

### The Problem

TDA for LLMs is fragmented. Parameter-space methods (Influence Functions, TRAK, LoGra) consistently underperform on LLM-scale models despite strong theoretical foundations. Meanwhile, five representation-space methods have independently emerged (RepSim, RepT, In-the-Wild, Concept Influence, AirRep), each demonstrating superior performance in specific settings -- but they have never been recognized as a coherent method family, never diagnosed through a common lens, and never benchmarked together.

### What the Pilot Data Revealed

A comprehensive pilot study (N=100, Pythia-1B, 14 tasks, ~28 min) produced five critical findings that reshape the original CRA thesis:

1. **FM1 spectral concentration is real and severe.** Full-model gradient top-5 eigenvalues capture 85.6% of variance (r_eff=10), far more concentrated than representation space (34.9%, r_eff=63). This directly evidences the dimensional mismatch.

2. **Toxicity task reversal defines a task-type boundary.** TRAK (0.926) > RepSim (0.685) on toxicity AUPRC -- a 24pp reversal. Raw Dot IF without Hessian achieves 0.94, confirming gradient norm as the primary signal (Cohen's d=2.66). FM1 diagnosis applies to semantic attribution tasks, not behavioral detection.

3. **FM2 evaluation protocol failed.** Contrastive scoring (mean subtraction) produced exactly 0.0pp gain across all 12 method-task combinations because rank-based metrics (AUPRC, R@K) are mathematically invariant to mean-subtraction. This is a measurement artifact, not evidence against FM2.

4. **Whitened attribution fails at pilot scale.** M = Sigma_noise^{-1} degrades all tasks by 8-11pp (N/d=0.049, covariance underdetermined). However, SNR-accuracy correlation is positive (r=0.34 counterfact), suggesting the concept is directionally correct.

5. **30.8pp TRAK-PCA gap.** Even with optimal PCA projection to k=d dimensions, TRAK achieves only R@50=0.686 vs RepSim's 0.994. FM1 is necessary but grossly insufficient as explanation.

### Revised Thesis

The CRA thesis is restructured from "two independent signal processing defects with predictive theory" to an **evidence-driven empirical framework** with three tiers of claims:

- **Tier 1 (Strong evidence):** FM1 spectral diagnosis, task-type boundary, systematic 2x2 benchmark
- **Tier 2 (Under investigation):** FM2 via continuous metrics and controlled injection, whitened attribution via PCA reduction
- **Tier 3 (Organizational):** phi^T M psi as a taxonomic unification, not a predictive theory (unless Tier 2 experiments rescue predictive claims)

## Research Questions

**RQ1 (FM1 -- Supported)**: Do representation-space methods systematically outperform parameter-space methods on semantic attribution tasks, and does this advantage correlate with gradient rank deficiency?

**RQ2 (FM2 -- Under Investigation)**: When measured with continuous metrics (Kendall tau, Spearman rho) and controlled contamination injection, does contrastive scoring improve attribution quality, particularly in parameter space?

**RQ3 (Task-Type Boundary -- New)**: What mechanistic factors explain why parameter-space methods dominate on behavioral detection (toxicity) while representation-space methods dominate on semantic attribution (counterfact, ftrace)?

**RQ4 (Gap Decomposition -- New)**: What fraction of the 30.8pp TRAK-PCA to RepSim gap is attributable to (a) layer mixing, (b) cosine normalization, (c) nonlinear semantic features, vs (d) projection dimensionality?

**RQ5 (Framework Predictive Power)**: Does PCA-reduced whitened attribution (at N/k >> 1) outperform standard RepSim, validating the matched filter theory at feasible sample sizes?

## Hypotheses

See `hypotheses.md` for detailed testable hypotheses with revised falsification criteria incorporating pilot evidence.

## Expected Contributions

1. **FM1 Spectral Diagnosis**: First direct measurement of gradient covariance effective rank at LLM scale, showing r_eff << d << B, with controlled K-FAC experiment confirming independence from Hessian quality.

2. **Systematic DATE-LM Benchmark**: First 2x2 factorial {parameter-space, representation-space} x {standard, contrastive scoring} on all 3 DATE-LM tasks with both rank and continuous metrics. Includes multi-method tournament with 5+ representation methods.

3. **Task-Type Boundary Discovery**: Formal characterization of when gradient-norm (parameter-space) vs semantic similarity (representation-space) drives attribution quality. Toxicity reversal as the empirical anchor.

4. **FM2 Mechanistic Validation**: Controlled contamination injection providing the first causal evidence for FM2, independent of metric choice. Continuous metrics breaking the rank-invariance barrier.

5. **phi^T M psi Taxonomic Unification**: Systematic organization of 5+ representation-space methods under a common framework, with empirical characterization of each method's phi/psi/M configuration.

6. **Gap Decomposition Analysis**: Quantitative attribution of the TRAK-PCA to RepSim performance gap to specific factors (layer selection, normalization, nonlinearity, semantic features), answering what FM1 does and does not explain.

## Methodology Overview

### Priority 1: FM2 Verification Suite (~3 GPU-hours, ~1h wall-clock)
- **Continuous metrics**: Kendall tau / Spearman rho on raw attribution scores for all methods and tasks
- **Controlled contamination injection**: phi_contaminated = phi + alpha * mean(phi_train), alpha in {0, 0.1, 0.5, 1.0, 2.0, 5.0}, measuring contrastive scoring recovery
- **Decision gate**: If Kendall tau gain >= 0.05 for parameter-space methods AND injection recovery succeeds, FM2 narrative is validated

### Priority 2: FM1 Diagnostic Suite (~10 GPU-hours, ~3h wall-clock)
- **Full-scale eigenspectrum**: Pythia-70M at N in {500, 1000, 2000, 5000}, confirming H9 direction at N >> d
- **TRAK dimension sweep**: Pythia-1B, N=5473 full counterfact, k in {32..4096}, TRAK-random vs TRAK-PCA
- **Decision gate**: If TRAK-PCA at k=d closes to within 5pp of RepSim, FM1 is the primary mechanism

### Priority 3: Retrieval Baselines (~2.5 GPU-hours, ~1h wall-clock)
- Contriever or GTR-T5 vs RepSim on all 3 tasks
- BM25 at full scale (N=5473)
- **Decision gate**: If retrieval models match RepSim, "attribution vs retrieval" boundary must be addressed

### Priority 4: Mechanism Suite (~5 GPU-hours, ~1.5h wall-clock)
- **Gap decomposition**: Last-layer-only TRAK-PCA + cosine-normalized TRAK-PCA
- **Layer sweep**: RepSim at layers {0, 4, 8, 12, 16, 20, 23} on Pythia-1B
- **Cosine vs Euclidean**: RepSim with Euclidean distance (contrarian falsification test)

### Priority 5: Framework Validation (~3.5 GPU-hours, ~1h wall-clock)
- **PCA-reduced whitening**: k in {16, 32, 64, 128, 256, 512}, whitened in PCA subspace where N/k >> 1
- **OVB sensitivity analysis**: Cinelli-Hazlett Robustness Value for FM2 severity quantification

### Total: ~24 GPU-hours, ~7.5h wall-clock on 4x RTX 4090

## Key Decision Points

1. **After FM2 Verification (Priority 1)**: If continuous metrics show >= 0.05 Kendall tau gain AND contamination injection succeeds, FM2 is validated as Tier 1 contribution. If not, FM2 is demoted to "theoretical hypothesis" and paper narrows to FM1 + benchmark.

2. **After Retrieval Baselines (Priority 3)**: If Contriever/GTR match RepSim on 2+ tasks, the paper must reposition from "attribution quality" to "attribution vs retrieval boundary analysis."

3. **After PCA Whitening (Priority 5)**: If PCA-whitened attribution at k=64 outperforms RepSim by >= 3pp on any task, the matched filter theory is rescued. If all k values fail, whitened attribution is reported as a negative result.

## Evidence-Driven Revisions (from Pilot Round)

### What Changed from Iteration 0

1. **Hypothesis H2 redesigned**: Original rank-based evaluation cannot detect FM2 effects. Replaced with continuous metrics (Kendall tau) and controlled contamination injection.

2. **Hypothesis H7 narrowed**: Full-dimensional whitening abandoned (structural N/d problem at DATE-LM scale). Replaced with PCA-reduced whitening at feasible N/k ratios.

3. **Hypothesis H9 reversed**: Original claim (representation space near-isotropic) is falsified. Replaced with spectral concentration comparison (r_eff/d ratio) rather than condition number.

4. **New RQ3 added**: Toxicity task reversal is not a failure mode but a genuine contribution -- task-type boundary characterization.

5. **New RQ4 added**: 30.8pp TRAK-PCA gap demands systematic decomposition. This is the core challenge to the FM1-only narrative.

6. **Paper positioning shifted**: From "diagnostic theory + prescriptive framework" to "empirical diagnosis + systematic benchmark." phi^T M psi is organizational, not predictive, unless Priority 5 experiments rescue it.

7. **Retrieval baselines mandated**: Contrarian's "RepSim is just retrieval" hypothesis must be tested before any claims about attribution quality.

### Negative Results to Report Honestly

- H7 (whitened attribution) fails at pilot scale (-8 to -11pp)
- H9 (isotropy) is falsified (direction completely reversed)
- H2 (contrastive scoring) is inconclusive due to rank-metric invariance
- BM25 achieves R@50=1.0 on counterfact at pilot scale
- 30.8pp gap between TRAK-PCA and RepSim persists

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Retrieval models match RepSim (attribution = retrieval) | Critical | Reposition as boundary analysis; toxicity reversal provides genuine attribution contribution |
| FM2 undetectable even with continuous metrics | High | Report as honest negative; narrow to FM1-only paper |
| PCA whitening fails at all k | Medium | Report negative result; framework is taxonomic, not prescriptive |
| BM25 competitive at full scale | Medium | Restrict semantic attribution claims; emphasize toxicity/mechanism contributions |
| TRAK-PCA gap doesn't decompose cleanly | Medium | Report as multi-factorial; strengthens "feature quality not just dimensionality" narrative |
| Full-scale H9 still reversed | Medium | Develop new theoretical explanation for why M=I works despite anisotropy |

## Target Venue

NeurIPS 2026 / ICML 2027. Contribution ceiling: poster to spotlight. Core strength is the systematic benchmark + FM1 spectral diagnosis + task-type boundary discovery. If FM2 validation and PCA whitening succeed, theoretical depth increases toward spotlight.
