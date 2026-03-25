---
version: "1.1"
created: "2026-03-16"
last_modified: "2026-03-25"
entry_mode: "first"
iteration_major: 1
iteration_minor: 1
---

# Problem Statement

## 1. Gap Definition

### 1.1 Current Methods Overview

Training Data Attribution (TDA) for LLMs computes "which training samples most influenced this model output?" Two families of methods have emerged, operating in fundamentally different spaces:

**Parameter-space methods** compute influence via gradients w.r.t. model weights theta in R^B (B ~ billions). The landscape spans three tiers of sophistication:

| Tier | Representative Methods | Key Mechanism | Best Known LDS |
|------|----------------------|---------------|----------------|
| Approximate IF | TRAK, LoGra, EK-FAC, LESS | Random projection / Kronecker-factored iHVP | 0.2 - 0.5 |
| Improved Hessian | ASTRA, Better Hessians Matter | Iterative iHVP refinement | ~0.6 |
| Exact IF | MAGIC | Metagradient exact computation (deterministic training) | 0.95 - 0.99 |
| Contrastive IF | DDA | IS_{theta'} - IS_{theta_0} (subtract base-model IF) | AUC 93.5% (hallucination) |

**Representation-space methods** use internal activations h^(l) and/or their gradients at specific hidden layers. Five independently proposed methods share a rough bilinear structure phi(z_test)^T * psi(z_train), but differ substantially in what phi and psi encode:

| Method | phi (test) | psi (train) | Dim | Contrastive? |
|--------|-----------|-------------|-----|-------------|
| RepSim | h^(l) | h^(l) | d ~ 4096 | No |
| RepT | concat[h^(l*), nabla_h L] | concat[h^(l*), nabla_h L] | 2d ~ 8192 | Implicit (gradient = "how to change") |
| In-the-Wild | v_behavior (activation diff) | v_data (activation diff) | d ~ 4096 | Yes (chosen - rejected) |
| Concept IF | J_l^T v (concept gradient) | nabla_theta f(z_train) | p (params) | Via concept direction |
| AirRep | Enc(z_test) | Agg(Enc(z_i)) | d_enc ~ 384 | No (learned space) |

**Important caveat on "unified family"**: These methods share the representation-space operating principle but differ in non-trivial ways. Concept IF projects back to parameter space for the train-side encoding. AirRep learns an entirely new encoder rather than using model internals. In-the-Wild uses activation *differences* (inherently contrastive). The bilinear form phi^T psi is a structural observation, not a deep theoretical unification -- it is the starting point for analysis, not the conclusion.

### 1.2 Gap Statement

**One-sentence Gap**: Parameter-space TDA methods face three distinct bottlenecks at LLM scale -- Hessian approximation error, signal dilution (FM1), and common influence contamination (FM2) -- but existing work addresses these in isolation, no systematic evaluation disentangles their relative contributions, and five independently proposed representation-space methods that implicitly bypass FM1 have never been evaluated on a common benchmark or explained through a unified diagnostic lens.

**Three concrete consequences of this fragmentation**:

1. **No comparative evaluation of representation-space methods on DATE-LM exists.** RepT reports P@10 on controlled experiments; AirRep reports LDS on data selection only; DDA reports AUC on hallucination tracing. No representation-space method has been evaluated on the full DATE-LM benchmark across all three tasks (G-RepT4, G-AR2). Practitioners cannot make informed method selections.

2. **The relative importance of three TDA bottlenecks at LLM scale is unknown.** Better Hessians Matter shows Hessian quality consistently improves attribution (LDS: H >= GGN >> EK-FAC), but at small scale (<1M params). MAGIC achieves LDS ~0.95-0.99 with exact IF, but only on fine-tuning with deterministic training. DDA shows debias contributes +55pp at 7B scale. No study has quantified: at LLM fine-tuning scale, how much does each bottleneck (Hessian error, FM1, FM2) contribute to total attribution failure?

3. **The "why" behind representation-space success is task-dependent and poorly understood.** Li et al. diagnosed iHVP degeneracy under LoRA, but this is LoRA-specific (H-RepT4). RepSim achieves 96-100% on harmful data identification but its performance on counterfactual metrics (LDS) is unknown (H-IF-LLM4). The question "does representation space capture correlation or causation?" is unanswered.

### 1.3 Root Cause Analysis

**Root Cause Type**: Three distinct bottlenecks compound at LLM scale, but the field treats them as a single problem ("IF doesn't work on LLMs").

**Layer 1 (symptom)**: Parameter-space IF performs poorly on LLM tasks. RepSim 96-100% vs IF 0-7% on harmful data identification (Li et al.); RepT P@10 = 0.97-1.00 vs LESS 0.59-0.73 (RepT paper).

**Layer 2 (Bottleneck 1 -- Hessian Approximation Error)**: Standard iHVP approximations (EK-FAC, K-FAC) introduce substantial error. Better Hessians Matter demonstrates consistent LDS improvement with better Hessians: H >= GGN >> Block-GGN >> EK-FAC >> K-FAC. MAGIC shows that exact IF (metagradient, no iHVP) achieves LDS ~0.95-0.99, proving that within parameter space, Hessian quality is a major bottleneck. **This bottleneck is real and significant -- the CRA thesis does NOT claim Hessians don't matter.**

**Layer 3 (Bottleneck 2 -- FM1: Signal Dilution)**: Even with a perfect Hessian, parameter-space gradients in R^B have per-sample gradients that are approximately orthogonal (JL phenomenon in high dimensions). Task-relevant signal occupies a tiny subspace with extremely low SNR. **Critical limitation**: The strongest evidence for FM1 (Li et al.'s iHVP degeneracy analysis) is LoRA-specific. Under full fine-tuning, the Hessian is not low-rank, and FM1's severity is an open empirical question.

**Layer 4 (Bottleneck 3 -- FM2: Common Influence Contamination)**: Standard IF scoring I(z_test, z_train) measures total influence dominated by shared pre-training knowledge. DDA ablation: removing debias causes -55.2pp AUC drop, vs. removing denoise only -8.71pp. This means the dominant attribution error is not random noise but systematic bias from common-mode pre-training signals.

**The three bottlenecks interact but are conceptually distinct**:
- Hessian error: computable approximation error (fixable with MAGIC-level exact IF, but at O(N*n) cost)
- FM1: inherent to operating in R^B regardless of Hessian quality (fixable by moving to representation space, d << B)
- FM2: inherent to standard scoring regardless of space or Hessian (fixable by contrastive scoring)

**Key unresolved tension**: MAGIC achieves LDS ~0.95-0.99 with exact IF in parameter space. If FM1 is truly a dominant bottleneck, why does MAGIC work so well? Possible resolutions: (a) MAGIC's deterministic training setting implicitly reduces FM1 (no stochastic gradient noise); (b) MAGIC experiments use Gemma-2B with LoRA where FM1 is artificially mild; (c) LDS itself may not capture the failure modes FM1 causes (H-RF1, H-DVEmb3). **This tension MUST be addressed experimentally, not argued away.**

### 1.4 Gap Assessment

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| **Importance** | **High** | 50+ papers in KB; multiple NeurIPS/ICML 2025-2026 papers; 5 independent representation-space methods in 12 months; no practitioner guidance |
| **Novelty** | **Medium** | The "three bottlenecks" framing is new, but the individual bottlenecks are recognized (Hessian: extensively studied; FM1: Li et al. partially identified; FM2: DDA identified). Risk of concurrent unification work is real given field velocity. Downgraded from "Medium-High" because the bilinear unification is more organizational than theoretical. |
| **Solvability** | **Medium** | DATE-LM is open-source; RepSim/TRAK implementations exist. Key uncertainties: (1) RepSim may fail on LDS (counterfactual metric vs correlational metric); (2) 2x2 ablation may not produce clean results; (3) MAGIC's near-perfect LDS complicates the "parameter space fails" narrative. The probe has NOT been executed -- solvability assessment is provisional. |

### 1.5 Research Questions

**RQ1 (Bottleneck Decomposition)**: At LLM fine-tuning scale (Pythia-1B to Llama-7B), what is the relative contribution of the three TDA bottlenecks (Hessian error, FM1, FM2) to total attribution failure on DATE-LM?

- *Operationalization*: Compare LDS across: (a) approximate IF (TRAK/EK-FAC), (b) improved IF (ASTRA-level), (c) MAGIC (exact IF, if feasible at scale), (d) RepSim/RepT (bypasses Hessian + FM1), (e) DDA/contrastive variants (addresses FM2). The gaps between (a)-(c) quantify Hessian contribution; gaps between (c)-(d) quantify FM1; gaps from adding contrastive scoring quantify FM2.
- *Falsification*: If MAGIC (exact parameter-space IF) achieves LDS within 3pp of RepSim/RepT on all DATE-LM tasks, FM1 is negligible at this scale -- the "signal dilution" thesis fails.
- *Prediction*: MAGIC will approach RepSim on tasks where FM2 is low (data selection), but RepSim + contrastive will outperform MAGIC on tasks where FM2 is high (toxicity filtering). The ordering of bottleneck contributions is: Hessian > FM1 > FM2 on small models, FM1 > FM2 > Hessian on large models.

**RQ2 (Representation-Space Benchmark)**: How do representation-space TDA methods (RepSim, RepT) compare to parameter-space methods (TRAK, DDA, MAGIC) on the full DATE-LM benchmark across all three tasks (data selection, toxicity filtering, factual attribution)?

- *Falsification*: If RepSim LDS < TRAK LDS - 5pp on all three DATE-LM tasks, the "representation space systematically addresses FM1" narrative is refuted. The "family" observation remains valid (organizational contribution) but loses its diagnostic teeth.
- *Prediction*: Representation-space methods excel on toxicity filtering (high FM2 bias, analogous to Li et al.'s harmful data ID) and are competitive on data selection, but may struggle on factual attribution (where lexical overlap matters more than semantic representation, per TrackStar's BM25 finding).
- *Key distinction*: RepSim captures representational similarity (correlational), while LDS measures counterfactual influence. A gap between RepSim's P@K and LDS scores would itself be informative -- it would quantify the "correlation vs causation" gap in representation-space TDA (H-IF-LLM4).

**RQ3 (FM1-FM2 Independence)**: Are the performance gains from representation-space operation (addressing FM1) and contrastive scoring (addressing FM2) approximately additive?

- *Operationalization*: 2x2 ablation {parameter-space, representation-space} x {standard scoring, contrastive scoring} on each DATE-LM task. Two-way ANOVA: main effects + interaction term.
- *Falsification*: If the interaction term exceeds 30% of the minimum main effect on >= 2/3 tasks, the "two independent signal-processing defects" framework is an oversimplification.
- *Prediction*: Interaction should be small if FM1 and FM2 address different signal components (dimensionality vs bias). But representation-space methods may implicitly address FM2 (by operating in a semantically structured space), which would show up as a smaller FM2 main effect in representation-space conditions.

### 1.6 Core Assumptions and Risk Assessment

| # | Assumption | Type | Strength | If False | Mitigation |
|---|-----------|------|----------|----------|------------|
| H1 | FM1 (signal dilution) is a dominant bottleneck at LLM scale, not just a LoRA artifact | Theoretical + Empirical | **Weak-Medium** (Li et al. evidence is LoRA-only; no full-FT evidence) | Theoretical framework loses FM1 leg; paper becomes FM2-focused benchmark study | Test both LoRA and full-FT on DATE-LM; compare RepSim advantage magnitude |
| H2 | Contrastive scoring effectiveness generalizes beyond DDA's hallucination tracing | Empirical | Weak-Medium (2 task types: hallucination + DPO alignment) | "Universal enhancement" claim scoped to high-FM2 tasks only | Test on all 3 DATE-LM tasks; accept task-dependent narrative |
| H3 | FM1 and FM2 repair gains are approximately additive | Theoretical | Weak (no verification; signal processing analogy is suggestive but not proof) | 2x2 narrative weakened; "synergistic improvement" framing needed | 2x2 ANOVA will directly test; accept either clean or messy result |
| H4 | RepSim is competitive with TRAK on LDS (counterfactual metric) | Empirical | **None** (probe NOT executed) | "Representation space superiority" narrative requires fundamental revision; pivot to nuanced "different metrics capture different aspects" | **This is the gate**: probe must run before full investment |
| H5 | MAGIC's LDS ~0.99 does not invalidate the FM1 thesis | Theoretical | Weak (MAGIC's setting may not expose FM1) | FM1 is not the bottleneck it's claimed to be; paper becomes "Hessian + FM2" story | Investigate MAGIC's training setup (deterministic, LoRA, scale) for confounds |

## 2. Attack Angle

### 2.1 Selected Angle

**Angle A (Revised)**: Three-bottleneck diagnostic framework + systematic benchmark evaluation.

**Core idea**: Propose a diagnostic framework decomposing LLM TDA failure into three complementary bottlenecks: (1) Hessian approximation error, (2) FM1 (signal dilution in parameter space), (3) FM2 (common influence contamination from pre-training). Verify through systematic evaluation on DATE-LM + Li et al. benchmarks, with the 2x2 ablation {parameter-space, representation-space} x {standard, contrastive} as the core experiment.

**Why it may work**:
(1) The "three bottlenecks" framing is strictly more informative than any existing single-bottleneck explanation. Even if one bottleneck turns out negligible, the decomposition itself clarifies the field's conflicting results.
(2) DATE-LM gap for representation-space methods is domain-acknowledged (G-RepT4, G-AR2). Filling this gap has standalone citation value regardless of diagnostic narrative.
(3) The 2x2 design is information-dense: any outcome pattern (additive, dominated by one factor, interaction-heavy) yields a publishable and field-clarifying result.

**Why it might fail**:
(1) **MAGIC challenge**: If MAGIC achieves LDS ~0.95+ on all DATE-LM tasks at Pythia-1B scale, the "parameter space has residual bottlenecks beyond Hessian" claim is weak. RepSim would need to beat MAGIC, not just TRAK.
(2) **RepSim LDS failure**: If RepSim fails on LDS despite succeeding on P@K, the result is informative but the paper becomes "representation methods capture different signals than influence" -- a weaker and more nuanced message.
(3) **Concurrent competition**: The field is moving fast (5 methods in 12 months). Someone may publish a similar unification during our research period.
(4) **Ceiling risk**: Pure diagnostic + benchmark is poster-level at best. Without a novel method contribution, the impact ceiling is bounded.

### 2.2 Attack Angle Limitations (Honest Assessment)

1. **FM1 may be a LoRA artifact.** Li et al.'s strongest evidence involves LoRA's extreme low-rank constraint. Under full fine-tuning, FM1 severity is unknown. If FM1 vanishes under full FT, the "three bottlenecks" reduces to "two bottlenecks" (Hessian + FM2), which is less novel (Hessian is well-studied, FM2 is DDA's contribution).

2. **The bilinear unification is shallow.** Calling phi^T psi a "unified framework" risks being seen as reformulation rather than insight. Concept IF and AirRep don't cleanly fit the phi^T psi mold. The unification must be positioned as taxonomic convenience, not theoretical depth.

3. **Contrastive scoring reference construction is non-trivial.** DDA's debias uses natural pairs (correct vs. incorrect entity). For DATE-LM's data selection task, there is no natural contrastive reference. Constructing z_cf ad hoc for each task weakens the "universal enhancement" claim.

4. **MAGIC invalidation risk is non-negligible.** MAGIC's LDS ~0.99 in parameter space could mean FM1 is negligible when Hessian error is eliminated. The CRA thesis then collapses to: "Hessian approximation is the main problem, and representation space is just a cheap alternative to exact IF." This is true but not novel.

5. **LDS metric reliability.** H-RF1 (Revisiting Fragility) and H-DVEmb3 raise concerns about LDS as an evaluation metric. If LDS doesn't capture what we're measuring (real attribution quality), all quantitative comparisons are compromised.

## 3. Probe Experiment (Critical Gate)

### 3.1 Status

**PROBE NOT YET EXECUTED.** The CRA_old probe design (RepSim vs TRAK on DATE-LM toxicity filtering) has not been run. The CRA (Sibyl) LIBERO-10 pilot is in a completely different domain (VLA policy learning) and does NOT validate any FM1/FM2 hypothesis. All claims about representation-space superiority on DATE-LM are unverified predictions.

### 3.2 Minimal Probe Design

**Step 1 (0.5 day)**: Clone DATE-LM, select Pythia-1B + toxicity filtering.
**Step 2 (1 day)**: Implement RepSim on DATE-LM (extract h^(l) at middle + last layer, compute cosine similarity, submit to LDS evaluation).
**Step 3 (0.5 day)**: Run TRAK baseline using DATE-LM's existing implementation.
**Step 4 (0.5 day)**: Compare + interpret. If possible, also run data selection task.

**Total compute**: < 1 GPU-day on a single A100.

### 3.3 Pass Criteria

| Signal | Criterion | Interpretation |
|--------|-----------|----------------|
| Strong pass | RepSim LDS >= TRAK LDS on toxicity filtering | FM1 thesis strongly supported; full evaluation justified |
| Pass | RepSim LDS >= TRAK LDS - 5pp | Competitive; proceed with caution |
| Weak pass | RepSim < TRAK - 5pp on toxicity, but >= TRAK - 5pp on data selection | Task-dependent; CRA thesis needs scoping |
| Fail | RepSim < TRAK - 5pp on both tasks | FM1 thesis does not generalize to LDS; direction needs fundamental revision |

### 3.4 Failure Diagnosis

If RepSim fails:
- **Check absolute LDS**: If both RepSim AND TRAK score low, the issue may be DATE-LM evaluation design, not method quality.
- **Check layer sensitivity**: If middle and last layer both fail, it's not a layer selection issue. If one succeeds, RepT's auto-layer detection is warranted.
- **Check Spearman correlation**: High RepSim-TRAK rank correlation + low RepSim LDS = calibration issue, not signal issue.
- **Check P@K alongside LDS**: If RepSim has high P@K but low LDS, the "correlation vs causation" gap is real (H-IF-LLM4), and the paper reframes as a diagnostic contribution.

## 4. Metadata

- **Gap source**: Combinatorial derivation (5 independent representation-space methods + DDA contrastive + MAGIC exact IF + Better Hessians Matter tension)
- **Root cause type**: Overlooked decomposition (three bottlenecks treated as one problem)
- **Attack angle source**: Signal processing analogy (matched filtering <-> representation space; differential detection <-> contrastive scoring) + systematic benchmark gap
- **Key uncertainties (rank-ordered)**:
  1. RepSim performance on LDS (probe not run -- highest uncertainty)
  2. FM1 severity under full fine-tuning (LoRA artifact risk)
  3. MAGIC invalidation (exact IF in parameter space achieves near-perfect LDS)
  4. Contrastive scoring generality beyond 2 task types
  5. Concurrent competition risk
- **Episteme references**: G-RepT4, G-AR2, H-IF-LLM1, H-IF-LLM4, H-RepT4, H-RF1, H-DVEmb3
- **Strategic review conditions (from CRA_old Round 1)**:
  1. [ADDRESSED] Correct "Hessian doesn't matter" -> "three complementary bottlenecks" (Section 1.3)
  2. [ADDRESSED] Downgrade orthogonality from assumption to testable hypothesis (RQ3 falsification criterion)
  3. [ADDRESSED] Narrow experimental scope to RepSim + RepT + TRAK + DDA + MAGIC (Section 2.1)
  4. [ADDRESSED] Include MAGIC + DDA as mandatory baselines (Section 1.1 table, RQ1)
