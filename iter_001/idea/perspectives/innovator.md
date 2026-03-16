# Innovator Perspective: CRA Research Proposal

**Agent**: Innovator (bold, cross-domain, counter-intuitive angles)
**Date**: 2026-03-16

---

## Executive Summary

The CRA project's core thesis -- diagnosing FM1/FM2 as signal processing defects and unifying representation-space methods via phi^T * psi -- is sound but risks becoming a well-executed incremental contribution (the 2x2 ablation matrix is clean but predictable). Below I propose three unconventional angles that could elevate this from a solid diagnostic paper to a genuinely surprising result. Each is designed so that pilot experiments complete in 10-15 minutes on Pythia-1B, and full experiments fit within 1 GPU-hour on 4x RTX 4090.

---

## Angle 1: The Spectral Scalpel -- Eigenspectrum Surgery as a Mechanistic Explanation of FM1

### Core Insight (Cross-Domain Transfer: Compressed Sensing --> TDA)

The FM1 "signal dilution" story currently relies on the Johnson-Lindenstrauss (JL) intuition: gradients in R^B are near-orthogonal, so attribution signal drowns in noise. But this is hand-wavy. A much sharper diagnosis is available from **compressed sensing theory**: the attribution signal lives in a low-rank subspace of the gradient space, and what matters is not the ambient dimension B but the **effective rank** of the gradient covariance matrix Sigma_g = E[g g^T].

**Hypothesis**: For LLMs, the effective rank r_eff(Sigma_g) grows sublinearly with model size (roughly O(sqrt(B))), meaning the "useful" attribution subspace is tiny relative to B. Representation-space methods succeed because they implicitly project onto this subspace -- the representation h in R^d already concentrates signal along the top-r_eff eigenvectors.

**What's New**: Hu et al. (2602.10449) just proved that random projection preserves influence iff the sketch dimension m >= rank(F), and that ridge regularization changes the barrier to the effective dimension. Their theory is *abstract* -- nobody has measured the actual effective rank of Sigma_g for LLMs and connected it to the parameter-space vs. representation-space performance gap. We would be the first to:
1. Measure r_eff(Sigma_g) across model sizes (Pythia-70M/160M/410M/1B)
2. Show r_eff << B and r_eff ~ d (representation dimension)
3. Demonstrate that projecting parameter gradients onto the top-r_eff eigenspace recovers representation-space performance
4. This would prove FM1 is not just "high dimensionality" but specifically "rank-deficient signal in full-rank noise"

**Experimental Plan**:
- **Pilot (15 min)**: On Pythia-70M with 1K training samples from DATE-LM data selection task, compute full gradient covariance (feasible at 70M params with gradient checkpointing), plot eigenvalue decay, measure r_eff at 90%/95%/99% energy thresholds
- **Core experiment (45 min)**: For Pythia-{70M, 160M, 410M, 1B}, subsample 5K training points, compute gradient covariance via Lanczos (top-500 eigenvalues), compare r_eff scaling with model size B, representation dimension d, and attribution performance
- **Validation (15 min)**: On the best model, project TRAK gradients onto top-r_eff eigenspace, compare attribution quality with RepSim -- if they converge, FM1 is explained

**Computational Cost**: ~2 GPU-hours total (parallelizable across 4 GPUs)
**Success Probability**: 70% -- the eigenvalue decay is almost certainly steep (this is well-known for NTK spectra), but the quantitative match r_eff ~ d needs to hold
**Failure Mode**: r_eff might not match d cleanly, or the eigenspace projection might not recover full RepSim performance (suggesting FM1 is more than just rank)

**Key References**:
- Hu et al. 2026 (2602.10449) -- Unified theory of random projection for influence functions; proves sketch dimension must exceed rank(F)
- Park et al. 2023 (TRAK) -- Random projection baseline
- Ghorbani et al. 2020 -- Eigenspectrum of NTK for neural networks

---

## Angle 2: Difference-in-Differences Attribution -- Importing Causal Inference to Fix FM2

### Core Insight (Cross-Domain Transfer: Econometrics --> TDA)

FM2 (common influence contamination) is exactly the **confounding variable** problem from causal inference. Standard IF scores I(z_train, z_test) are contaminated by the "common cause" -- pre-training knowledge that influences all training samples similarly. The current fix (DDA's contrastive scoring) is ad-hoc debiasing.

But econometrics has a 30-year-old principled solution: **Difference-in-Differences (DiD)**. The classic DiD estimator removes time-invariant confounders by taking double differences. We can construct a precise analogy:

| Econometrics | TDA |
|---|---|
| Treatment group | Test-relevant training samples |
| Control group | Test-irrelevant training samples |
| Pre-treatment outcome | Representation before fine-tuning |
| Post-treatment outcome | Representation after fine-tuning |
| Time-invariant confounder | Pre-training knowledge (FM2) |

**The DiD Attribution Score**:
```
A_DiD(z_train, z_test) = [sim(h_post(z_test), h_post(z_train)) - sim(h_pre(z_test), h_pre(z_train))]
                        - E_z'~neg [sim(h_post(z_test), h_post(z')) - sim(h_pre(z_test), h_pre(z'))]
```

This double-differencing removes both (a) pre-existing similarity (pre-training confound = FM2) and (b) general representation shift (fine-tuning drift).

**What's New vs. DDA**: DDA uses a single contrastive difference (post vs. negative). DiD takes a *double* difference that also accounts for pre-training similarity. This is strictly more principled -- if the parallel trends assumption holds (pre-training similarity is stable across relevant/irrelevant samples), DiD is unbiased for the causal effect of fine-tuning.

**Connection to phi^T * psi Framework**: DiD attribution naturally decomposes as:
```
A_DiD = phi_DiD(z_test)^T * psi_DiD(z_train)
```
where phi_DiD = h_post(z_test) - h_pre(z_test) - E[h_post - h_pre], psi_DiD = h_post(z_train) - h_pre(z_train) - E[h_post - h_pre]. This is a new phi/psi instantiation in the bilinear framework, with causal justification.

**Experimental Plan**:
- **Pilot (10 min)**: On Pythia-1B + DATE-LM toxicity filtering task, compute DiD scores using pre-fine-tuning and post-fine-tuning representations. Compare with RepSim and DDA.
- **Core experiment (45 min)**: Implement DiD on all 3 DATE-LM tasks x {Pythia-1B, Llama-2-7B}. Compare with: RepSim (no FM2 fix), DDA (single difference), DiD (double difference).
- **Ablation (15 min)**: Test the "parallel trends" assumption by measuring pre-training similarity stability across relevant/irrelevant groups.

**Computational Cost**: ~1.5 GPU-hours (need both pre and post checkpoints, but representation extraction is cheap)
**Success Probability**: 55% -- the parallel trends assumption is strong and may not hold in practice; also, the marginal gain over DDA's simpler approach might be small
**Failure Mode**: (a) Parallel trends violated badly -- pre-training similarity is already highly predictive of relevance, making the pre-treatment differencing harmful. (b) DDA already captures most of the FM2 correction, leaving little room for DiD improvement.

**Key References**:
- Angrist & Pischke 2009 -- Mostly Harmless Econometrics (DiD theory)
- Pan et al. 2025 (2502.11411) -- Denoised Representation Attribution (related but different: they denoise at token level, we difference at checkpoint level)
- Wu et al. 2024 (DDA, 2410.01285) -- Current best FM2 fix via contrastive scoring

---

## Angle 3: The Information Bottleneck Lens -- Why Representation Space is *Necessarily* Better

### Core Insight (New Theoretical Method: Information Theory --> TDA)

The current narrative for why representation-space TDA works is empirical ("it just does better") plus our signal processing intuition (FM1 + FM2). But there's a deeper, more provocative claim available from **Information Bottleneck (IB) theory**:

A well-trained network's hidden representation h is an approximate solution to the IB objective: maximize I(h; Y) while minimizing I(h; X). This means h is already an *optimal compression* of X for predicting Y. Consequently:

**Theorem sketch**: If h is an IB-optimal representation, then any attribution method operating on h has higher signal-to-noise ratio than the same method operating on theta (parameters), because h discards task-irrelevant information that theta retains.

More formally: Let s(z_train, z_test) = the true attribution signal we want to detect. In representation space, Var(noise) is bounded by I(h; X|Y) (task-irrelevant information in h). In parameter space, Var(noise) is bounded by I(theta; X) (all information about X in theta, including irrelevant). Since I(h; X|Y) << I(theta; X) for a well-trained model, representation space has fundamentally higher SNR.

**What's New**: This argument has never been made in the TDA literature. It provides a *necessary* theoretical guarantee that representation-space methods are better, not just an empirical observation. It also predicts:
1. **Layer selection matters**: Middle layers (best IB compression point) should give best attribution -- explaining RepT's "phase transition layer" finding
2. **Compression quality predicts attribution quality**: Models with better IB tradeoff should show larger representation-over-parameter gaps
3. **Overtrained models may lose the advantage**: If fine-tuning proceeds too long, h overfits and I(h; X|Y) grows, reducing the representation-space advantage

**Experimental Plan**:
- **Pilot (15 min)**: On Pythia-1B, compute RepSim attribution at each layer (layers 1-16). Simultaneously compute a proxy for IB compression quality at each layer (using MINE or binning-based MI estimator). Check if best attribution layer ~ best IB compression layer.
- **Core experiment (40 min)**: Vary fine-tuning epochs {1, 2, 5, 10, 20} on DATE-LM data selection task. At each epoch, measure: (a) layer-wise attribution performance, (b) layer-wise IB proxy, (c) parameter-space vs. representation-space gap. Predict that gap peaks when IB compression is tightest.
- **Cross-model validation (20 min)**: Compare Pythia-{70M, 160M, 410M, 1B}: larger models should have better IB compression (more capacity to discard irrelevant info) and thus larger representation-over-parameter gap.

**Computational Cost**: ~2 GPU-hours (MI estimation is the bottleneck, but MINE is fast on small batches)
**Success Probability**: 45% -- the IB theory for LLMs is contested (Shwartz-Ziv & Tishby debate), and the connection to attribution SNR is a theoretical leap that may not manifest cleanly in experiments
**Failure Mode**: (a) MI estimation is unreliable in high dimensions -- proxy measures might not track actual IB compression. (b) The "best layer for attribution = best IB compression" prediction might fail if attribution depends on task-specific features that don't align with the IB objective. (c) The connection between IB and attribution SNR may be too loose to produce quantitative predictions.

**Key References**:
- Tishby et al. 2000 -- Information Bottleneck method
- Shwartz-Ziv & Tishby 2017 -- Opening the black box of DNNs via information (controversial)
- Saxe et al. 2019 -- IB theory in deep learning (more rigorous treatment)
- RepT (2510.02334) -- Phase transition layer finding that IB theory could explain

---

## Synthesis: How These Angles Strengthen the Core Paper

The three angles serve distinct roles in the CRA paper:

| Angle | Role | If Succeeds | If Fails |
|---|---|---|---|
| 1. Spectral Scalpel | **Sharpens FM1 diagnosis** from hand-wavy JL to precise rank-deficiency | Upgrades "signal dilution" from metaphor to theorem | Still useful as negative result: FM1 is more than rank |
| 2. DiD Attribution | **Elevates FM2 fix** from ad-hoc to causally principled | New phi/psi instantiation with causal guarantee | DDA sufficient; appendix-worthy comparison |
| 3. IB Lens | **Provides necessity proof** for representation-space superiority | Transforms paper from "what works" to "why it must work" | IB connection too loose; revert to empirical story |

**Recommended Priority**: Angle 1 > Angle 3 > Angle 2

Rationale: Angle 1 has highest success probability and directly strengthens the paper's central FM1 claim with hard evidence (eigenspectra are measurable). Angle 3, if it works, provides the most impactful theoretical contribution but is riskier. Angle 2 is elegant but may not offer enough marginal improvement over DDA to justify the added complexity.

---

## Risk Assessment

### What Could Kill Each Angle

1. **Spectral Scalpel**: If effective rank scales linearly with B (not sublinearly), the "signal dilution" story weakens -- it would mean parameter-space *should* work given enough projection dimensions, contradicting empirical evidence. Mitigation: check whether the eigenvalue decay follows power law or exponential.

2. **DiD Attribution**: The pre/post checkpoint comparison requires access to both checkpoints. DATE-LM may not provide pre-fine-tuning checkpoints for all models. Mitigation: use HuggingFace base models as "pre" and DATE-LM fine-tuned models as "post".

3. **IB Lens**: Mutual information estimation in R^4096 is notoriously unreliable. MINE can overfit to small batches. Mitigation: use multiple MI estimators (MINE, binning, KSG) and report only claims that are consistent across all three.

### What if ALL Three Fail?

The core 2x2 ablation paper remains viable and publishable. These angles are upside bets, not load-bearing structure. The bilinear unification framework phi^T * psi stands independently of whether we can explain *why* it works at a deeper level.

---

## Supplementary Literature Found During Search

The following papers were discovered during targeted literature search and are directly relevant to the proposed angles:

1. **Hu et al. 2026 (2602.10449)** -- "A Unified Theory of Random Projection for Influence Functions". Proves projection preserves influence iff sketch dim >= rank(F); ridge regularization lowers the barrier to effective dimension. *Critical for Angle 1*: their theory is abstract, we would be the first to measure effective rank empirically on LLMs and connect it to representation-space performance.

2. **Pan et al. 2025 (2502.11411)** -- "Detecting and Filtering Unsafe Training Data via Data Attribution with Denoised Representation (DRA)". Token-level denoising of representations for safety-focused attribution. *Related to Angle 2*: they denoise individual representations, we difference across checkpoints. Complementary, not competing.

3. **Shan & Bordelon 2021 (2105.14301)** -- "A Theory of NTK Alignment". NTK alignment during training reflects feature learning and drives kernel specialization. *Relevant to Angle 1*: the alignment process could explain why gradient covariance concentrates on a low-rank subspace aligned with task structure.

4. **Davari et al. 2022 (2210.16156)** -- "Reliability of CKA as a Similarity Measure". Warns that CKA (and by extension RepSim-like metrics) can be manipulated without changing functional behavior. *Risk for core paper*: if RepSim's success is partially an artifact of representation geometry rather than true attribution, our narrative needs qualification.

---

## Summary Table

| Angle | Type | Hypothesis | Pilot Time | Full Time | P(success) | Impact if Succeeds |
|---|---|---|---|---|---|---|
| 1. Spectral Scalpel | Cross-domain (compressed sensing) | r_eff(Sigma_g) ~ d << B explains FM1 | 15 min | 1h | 70% | High -- turns FM1 from metaphor to measurement |
| 2. DiD Attribution | Cross-domain (econometrics) | Double-differencing removes FM2 more principally than DDA | 10 min | 1h | 55% | Medium -- new phi/psi with causal backing |
| 3. IB Lens | New theory (information theory) | IB-optimal h has provably higher attribution SNR | 15 min | 1h | 45% | Very High -- necessity proof for representation superiority |
