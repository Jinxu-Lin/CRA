# CRA: Contrastive Representation Attribution -- A Signal Processing Diagnosis and Bilinear Unification of Representation-Space TDA for LLMs

## Title

**Diagnosing Why Parameter-Space Attribution Fails: Signal Dilution, Common Influence Contamination, and the Bilinear Unification of Representation-Space TDA**

## Abstract

Training Data Attribution (TDA) methods systematically underperform on large language models when operating in parameter space. We diagnose two independent signal processing defects responsible for this failure: **FM1 (Signal Dilution)** -- attribution signal occupies a low-rank subspace of effective dimension ~d within the B-dimensional gradient space, causing signal-to-noise collapse under standard random projection; and **FM2 (Common Influence Contamination)** -- pre-training knowledge creates a bias term that dominates task-specific attribution scores. We show that five independently proposed representation-space TDA methods (RepSim, RepT, In-the-Wild, Concept Influence, AirRep) are all instances of a single bilinear framework phi(z_test)^T M psi(z_train), where representation space naturally addresses FM1 (operating in the signal-rich R^d subspace) and contrastive scoring addresses FM2 (removing a formally characterizable bias term). Through a hardened 2x2 factorial experiment on the DATE-LM benchmark, we provide the first systematic evidence that FM1 and FM2 corrections are approximately orthogonal. A controlled gradient dimension sweep demonstrates that TRAK attribution saturates at projection dimension k ~ d, directly measuring the rank-deficient signal structure. Whitened attribution (phi^T Sigma_noise^{-1} psi) derived from matched filter optimality theory provides the first prescriptive method selection criterion within the framework.

## Motivation

### The Problem

TDA for LLMs is in crisis. Parameter-space methods (Influence Functions, TRAK, LoGra) consistently underperform on LLM-scale models despite strong theoretical foundations. Meanwhile, five representation-space methods have independently emerged, each demonstrating superior performance in specific settings -- but they have never been recognized as a coherent method family, never diagnosed through a common lens, and never benchmarked together on a standard evaluation.

### Why This Matters

Without understanding *why* parameter-space methods fail and *why* representation-space methods succeed, practitioners have no principled basis for method selection. The current state is: "try RepSim, it usually works" -- which is unsatisfying both scientifically and practically.

### Our Diagnosis

We identify two independent signal processing defects in parameter-space TDA:

1. **FM1 (Signal Dilution)**: Parameter gradients in R^B contain attribution signal in a low-rank subspace of effective dimension r_eff ~ O(d), where d << B is the representation dimension. Standard random projection (TRAK-style) wastes dimensions on noise, while representation-space methods operate directly in this signal-rich subspace.

2. **FM2 (Common Influence Contamination)**: Standard attribution scores are dominated by pre-training knowledge shared across all training samples. This manifests as a bias term in the bilinear attribution decomposition. DDA's debias step (contributing 55pp of their improvement) is a special case of mean-subtraction deconfounding, which we formalize as the removal of a characterizable shared component phi_shared.

### The Unification

All five representation-space methods can be expressed as phi(z_test)^T M psi(z_train) with specific choices of feature maps and metric tensor. This is not merely notational convenience -- the framework:
- Reveals that representation methods succeed because they implicitly address FM1 (dimension reduction) while operating with M = I (no curvature correction needed, because representation covariance is near-isotropic)
- Predicts that contrastive scoring (FM2 correction) should improve methods in both spaces, with larger gains in parameter space
- Derives an optimal M = Sigma_noise^{-1} from matched filter theory, providing the first prescriptive recommendation

## Research Questions

**RQ1 (FM1)**: Do representation-space methods systematically outperform parameter-space methods on DATE-LM, and does this advantage correlate with the rank deficiency of the gradient covariance?

**RQ2 (FM2)**: Does contrastive scoring universally improve both parameter-space and representation-space methods, with larger gains in parameter space?

**RQ3 (Orthogonality)**: Are FM1 and FM2 corrections approximately additive (interaction term < 30% of minimum main effect)?

**RQ4 (Framework)**: Does the phi^T M psi framework have predictive power -- can whitened attribution (M = Sigma_noise^{-1}) outperform M = I?

## Hypotheses

See `hypotheses.md` for detailed testable hypotheses with falsification criteria.

## Expected Contributions

1. **Diagnostic Framework**: First formal identification and separation of FM1 and FM2 as independent failure modes of parameter-space TDA on LLMs, supported by controlled factorial experiments
2. **Bilinear Unification**: Systematic taxonomy of 5+ representation-space TDA methods under a common phi^T M psi framework with formal bias decomposition (Theorems 3-4) and method-specific instantiation table (Theorem 7)
3. **Mechanistic Evidence**: Direct measurement of gradient covariance effective rank and TRAK dimension sweep demonstrating signal saturation at k ~ d
4. **Prescriptive Theory**: Whitened matched filter attribution as the optimal linear detector, with per-query reliability estimates via output SNR

## Methodology Overview

### Phase 0: Foundation (Day 1)
- Pipeline validation pilot: RepSim + TRAK on Pythia-1B + DATE-LM data selection
- **Critical control**: K-FAC full-eigendecomp IF on Pythia-70M to disentangle Hessian error from FM1/FM2. If K-FAC IF matches RepSim (<5pp gap), the entire diagnostic framework requires revision.

### Phase 1: Core Factorial (Day 2-3)
- Hardened 2x2 ablation: {parameter-space, representation-space} x {standard, contrastive scoring}
- Controls: BM25 (lexical baseline), k-NN (nonlinear control), DDA (strong parameter-space baseline)
- Three DATE-LM tasks, bootstrap CI (B=1000), pre-registered falsification criteria

### Phase 2: Mechanistic Evidence (Day 3-4)
- Gradient covariance eigenspectrum on Pythia-70M (Lanczos top-500)
- TRAK dimension sweep k in {64, 128, 256, 512, 1024, 2048, 4096} on Pythia-1B
- RepSim-PCA dimension reduction sweep for cross-validation

### Phase 3: Framework Extensions (Day 4-5)
- Contrastive scoring as universal plug-in (36-cell matrix: 4 methods x 3 variants x 3 tasks)
- Whitened matched filter attribution (phi^T Sigma_noise^{-1} psi with Ledoit-Wolf regularization)
- Multi-method tournament for phi^T M psi taxonomy validation

### Phase 4: Theoretical Consolidation
- Bias decomposition formalization (Theorems 3-4)
- Method taxonomy table (Theorem 7)
- Whitened attribution optimality argument

## Key Decision Points

1. **After K-FAC control (Day 1)**: If K-FAC IF matches RepSim, pivot from "two independent failure modes" to "Hessian approximation quality is the primary bottleneck"
2. **After 2x2 factorial (Day 3)**: If interaction term > 30% on >= 2 tasks, revise orthogonality claim; report FM1/FM2 as correlated rather than independent
3. **After BM25 comparison (Day 3)**: If BM25 beats all attribution methods on factual attribution, restrict claims to data selection and toxicity filtering tasks

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| K-FAC IF matches RepSim | Critical | Pivot to Hessian-quality diagnosis |
| FM1/FM2 interaction too large | High | Report as correlated defects; still valuable empirically |
| BM25 competitive on factual attribution | Medium | Restrict scope; discuss as limitation |
| phi^T psi framework is vacuously universal | High | Derive non-trivial predictions (whitened MF, dimension sweep saturation) |
| LoRA gradients partially fix FM1 | Medium | Report both LoRA-TRAK and LoGra; position LoRA as partial FM1 fix |
| DATE-LM N=3 tasks insufficient for generality | Medium | Report per-task results; do not claim cross-task universality |

## Target Venue

NeurIPS 2026 / ICML 2027. Contribution ceiling: poster to spotlight (with whitened MF and strong mechanistic evidence, potential oral).

## Evidence-Driven Revisions

*This is the first iteration; no prior pilot evidence exists. This section will be populated after pilot experiments.*
