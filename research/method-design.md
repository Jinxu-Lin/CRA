---
version: "1.0"
created: "2026-03-26"
last_modified: "2026-03-26"
entry_mode: "first"
iteration_major: 1
iteration_minor: 0
---

# Method Design

## 1. Probe Signal Summary

**Status**: Primary probe (RepSim vs TRAK on DATE-LM) NOT YET EXECUTED. All design decisions below are based on theoretical analysis and indirect evidence. The probe is the FIRST priority before full experimental execution.

**Available indirect signals**:
- **Li et al. (2409.19998)**: RepSim achieves 96-100% vs IF 0-7% on harmful data identification (binary classification, NOT counterfactual LDS). This validates representation-space superiority on detection tasks but says nothing about LDS performance.
- **DATE-LM results (2507.09424)**: RepSim achieves AUPRC 0.989 (homogeneous toxicity), 0.585 (heterogeneous toxicity). Fine-tuning data selection: RepSim > Grad-Sim. Cosine-similarity methods consistently outperform inner-product methods across all tasks.
- **RepT (2510.02334)**: P@10 = 0.97-1.00 on controlled experiments with automatic phase-transition layer detection. However, P@K and LDS measure fundamentally different things (ranking precision vs counterfactual prediction).
- **MAGIC (2504.16430)**: LDS ~0.95-0.99 on Gemma-2B LoRA and GPT-2 fine-tuning. Exact IF in parameter space. TRAK achieves LDS 0.06-0.24 on the same settings. This means the gap between approximate and exact IF in parameter space is enormous (~0.8 LDS points).
- **DDA (2410.01285)**: Removing debias (FM2 fix) causes -55.2pp AUC drop on hallucination tracing, vs removing denoise only -8.71pp.

**Design constraints derived from signals**:
1. **RepSim LDS is completely unknown.** Design must handle both success and failure scenarios.
2. **MAGIC's near-perfect LDS is the primary challenge.** If MAGIC works at Pythia-1B scale, "parameter space fails" is nuanced -- parameter space fails only with approximate IF, not exact IF.
3. **Cosine similarity > inner product** is robust across DATE-LM tasks. All representation-space methods should use cosine normalization.
4. **Heterogeneous toxicity filtering degrades all methods** (RepSim 0.989 → 0.585). This is a natural FM2 signal -- stylistically similar safety-aligned data contaminates attribution.
5. **MAGIC cost is O(N*n)** -- scales linearly with test samples. At Pythia-1B with DATE-LM's test set size, feasibility is uncertain on 48GB A6000s.
6. **LoRA vs full-FT dimension is essential.** Li et al.'s FM1 evidence is LoRA-only. MAGIC experiments use LoRA. FM1 may be a LoRA-specific artifact.

## 2. Compute Resource Budget

**Hardware**: 4x NVIDIA RTX A6000 48GB (shared server via SSH MCP)

**Hard constraints**:
- Per-GPU memory: 48GB. Pythia-1B (~2GB fp16) fits easily; Llama-7B (~14GB fp16) fits but gradient storage is tight.
- Shared server: intermittent availability. Budget at ~75% utilization = 3 effective GPUs.
- Timeline: NeurIPS 2026 submission ~May 2026, ~2 months remaining.

**Model scale decisions**:
- **Primary scale**: Pythia-1B (all conditions). Fits on single A6000 with full gradients.
- **Scale-up**: Llama-7B for selected conditions only (best methods, 1-2 tasks). Full-FT at 7B is infeasible on 48GB (gradient checkpointing needed); LoRA fine-tuning at 7B is feasible.
- **Full-FT at Pythia-1B**: Feasible. ~4B parameters, per-sample gradient ~8GB fp16. Requires gradient accumulation but manageable.

**Budget allocation** (total available: ~60 GPU-days in 2 months at 75% utilization):

| Phase | GPU-days | Notes |
|-------|----------|-------|
| Probe (Experiment 0) | 2 | RepSim vs TRAK on DATE-LM toxicity, Pythia-1B |
| Pilot (Experiment 0.5) | 3 | 2x2 on toxicity, single seed, reduced data |
| Main benchmark (Experiment 1) | 15 | 6 methods x 3 tasks x 3 seeds, Pythia-1B |
| 2x2 ablation (Experiment 2) | 10 | 4 conditions x 3 tasks x 3 seeds |
| LoRA vs Full-FT (Experiment 3) | 12 | 2 FT modes x key methods x toxicity+selection |
| MAGIC feasibility (Experiment 4) | 5 | Attempt MAGIC on Pythia-1B, single task |
| Scale-up (Experiment 5) | 8 | Selected methods on Llama-7B, 1-2 tasks |
| Buffer | 5 | Reruns, debugging, unexpected needs |
| **Total** | **60** | Within budget |

**Critical constraint**: The 2x2x2 design (adding LoRA vs Full-FT) doubles the core ablation from 4 to 8 conditions. To stay within budget, Llama-7B scale-up is selective (not full matrix).

## 3. Attack Angle → Component Mapping

| Attack Angle Sub-goal | Method Component | Root Cause Addressed | Validation Experiment |
|----------------------|------------------|---------------------|----------------------|
| Quantify FM1 contribution | RepSim/RepT (representation-space operation) | FM1: signal dilution in R^B | Exp 2: Row comparison in 2x2 (repr vs param) |
| Quantify FM2 contribution | Contrastive scoring enhancement | FM2: common influence contamination | Exp 2: Column comparison in 2x2 (contrastive vs standard) |
| Test FM1-FM2 independence | 2x2 ablation interaction analysis | FM1 x FM2 interaction | Exp 2: Interaction term in 2-way ANOVA |
| Bound Hessian contribution | MAGIC exact IF (if feasible) | Hessian approximation error | Exp 4: MAGIC vs TRAK gap |
| Test FM1 LoRA-specificity | LoRA vs Full-FT comparison | FM1 generality | Exp 3: RepSim advantage under LoRA vs Full-FT |
| Benchmark representation family | RepSim + RepT on DATE-LM | Evaluation gap (G-RepT4, G-AR2) | Exp 1: Full benchmark |
| Diagnose metric gap | P@K vs LDS comparison | Correlation vs causation (H-IF-LLM4) | Exp 1: Dual metric reporting |

## 4. Method Framework Overview

### 4.1 Architecture: Diagnostic Framework, Not a Novel Method

CRA is NOT proposing a new TDA method. CRA is a **diagnostic framework** that:
1. Decomposes LLM TDA failure into three independent bottlenecks (Hessian error, FM1, FM2)
2. Maps existing methods to which bottlenecks they address
3. Validates the decomposition through systematic controlled experiments
4. Provides the first benchmark of representation-space methods on DATE-LM

The "method components" below are existing methods configured and combined to test specific hypotheses, not novel algorithmic contributions.

### 4.2 Component Overview

```
                    ┌─────────────────────────────────────┐
                    │     CRA Diagnostic Framework        │
                    │                                     │
                    │  Three-Bottleneck Decomposition:     │
                    │  1. Hessian Error                    │
                    │  2. FM1 (Signal Dilution)            │
                    │  3. FM2 (Common Contamination)       │
                    └──────────┬──────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
      ┌───────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
      │  Component A  │ │ Component B  │ │ Component C  │
      │ Repr-Space    │ │ Contrastive  │ │ MAGIC Exact  │
      │ Attribution   │ │ Scoring      │ │ IF (control) │
      │ (fixes FM1)   │ │ (fixes FM2)  │ │ (fixes Hess) │
      └───────┬──────┘ └──────┬───────┘ └──────┬───────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  2x2(x2) Ablation   │
                    │  {param, repr}       │
                    │  x {std, contrastive}│
                    │  x {LoRA, Full-FT}   │
                    └─────────────────────┘
```

**Information flow**: For each (test sample, training sample) pair:
1. Extract features: either gradients (parameter-space) or hidden representations (representation-space)
2. Score: either standard (direct similarity) or contrastive (subtract base-model similarity)
3. Evaluate: submit scores to DATE-LM evaluation pipeline for LDS/AUPRC/Recall

## 5. Core Mechanism Details

### Component A: Representation-Space Attribution (addresses FM1)

**Function**: Compute training data attribution scores using internal model representations instead of parameter gradients. This bypasses FM1 by operating in R^d (d ~ 4096) rather than R^B (B ~ 10^9), concentrating task-relevant signal.

**Input**: Fine-tuned model M, training set D_train, test sample z_test
**Output**: Attribution scores I(z_test, z_i) for each z_i in D_train

**Two variants**:

**A1: RepSim** (simplest baseline)
- Extract h^(l)(z) = hidden representation at layer l for all samples
- I_RepSim(z_test, z_train) = cos(h^(l)(z_test), h^(l)(z_train))
- Layer selection: middle layer (l = L/2) and last layer (l = L), report both. RepT's phase-transition finding suggests a single "critical layer" exists, but since we don't implement RepT's auto-detection for RepSim, we use the two most informative positions.
- Complexity: O(N * d) per test sample. For Pythia-1B (d=2048, N=10K): ~80MB, negligible.

**A2: RepT** (state-of-the-art representation method)
- Extract phi(z) = concat[h^(l*)(z), nabla_h L(z)] where l* is the phase-transition layer
- l* detection: layer where gradient norm exhibits phase transition (sharp change)
- I_RepT(z_test, z_train) = cos(phi(z_test), phi(z_train))
- Complexity: O(N * 2d) per test sample + one backward pass per sample for nabla_h L. For Pythia-1B: ~160MB representations + ~5 min gradient extraction per 10K samples on A6000.

**Causal argument (Level 3)**:
- Root cause (FM1): In R^B, per-sample gradients are approximately orthogonal (JL phenomenon: inner product of random B-dimensional vectors concentrates around 0 as B grows). Task-relevant signal occupies a subspace of dimension << B, but standard IF operates in full R^B.
- Evidence: Li et al. show iHVP degeneracy under LoRA (eigenvalues collapse). MAGIC achieves LDS ~0.95+ with exact IF, confirming that when Hessian error is removed, parameter space CAN work -- but at O(N*n) cost. DATE-LM shows RepSim > Grad-Sim on fine-tuning data selection and homogeneous toxicity.
- Mechanism: Representation space h^(l) in R^d is a learned compression that preserves task-relevant features. Cosine similarity in R^d effectively performs matched filtering -- projecting high-dimensional signals onto the task-relevant subspace before comparison.
- Why not simpler: The simplest alternative is random projection of gradients (TRAK does this). TRAK's LDS is 0.06-0.24 on MAGIC's benchmarks, showing random projection is insufficient. Representation space is not random projection -- it uses the model's learned features, which are structured by the training task.
- **Critical caveat**: This argument assumes that h^(l) captures influence-relevant features, not just task-relevant features. RepSim captures representational similarity (correlational), not counterfactual influence (causal). The gap between P@K and LDS will directly test this assumption.

**Validation**: → experiment-design.md §3.1 (main benchmark) + §3.2 (2x2 ablation row comparison)
**Expected ablation result**: RepSim LDS >= TRAK LDS - 5pp on toxicity filtering; RepT LDS >= RepSim LDS (gradient component adds causal signal). If RepSim LDS < TRAK LDS - 5pp on all tasks, FM1 thesis is weakened at LDS level.
**If ablation not significant**: RepSim's LDS deficit means representation similarity captures correlation not causation. The paper reframes: "representation space captures different attribution signals (correlation vs causal influence), and practitioners should choose based on their application type."

### Component B: Contrastive Scoring Enhancement (addresses FM2)

**Function**: Remove common-mode pre-training influence from attribution scores by subtracting base-model (pre-fine-tuning) scores from fine-tuned model scores.

**Input**: Fine-tuned model M_ft, base model M_base, training set, test sample
**Output**: Contrastive attribution scores I_contrastive(z_test, z_i)

**Mechanism**:
```
I_contrastive(z_test, z_train) = I_M(z_test, z_train; M_ft) - I_M(z_test, z_train; M_base)
```

This applies to BOTH parameter-space and representation-space methods:
- **Parameter-space contrastive** (DDA-style): IS_{theta_ft} - IS_{theta_base}. Subtracts gradient similarity computed at base model weights.
- **Representation-space contrastive** (RepSim-C): cos(h_ft^(l)(z_test), h_ft^(l)(z_train)) - cos(h_base^(l)(z_test), h_base^(l)(z_train)). Subtracts representation similarity at base model.

**Task-specific reference construction**:
- **Toxicity filtering**: Natural reference. Base model has no toxicity-specific knowledge; fine-tuned model has seen toxic examples. Contrastive score isolates toxicity-related influence.
- **Data selection**: Less natural. Base model already has general language knowledge. Contrastive score isolates fine-tuning-specific value. Risk: if fine-tuning shifts are small, contrastive scores may be noisy.
- **Factual attribution**: Moderate. Base model has pre-training world knowledge; fine-tuned model has additional entity-fact associations. Contrastive score isolates fine-tuning-introduced facts.

**Causal argument (Level 2)**:
- Root cause (FM2): Standard IF scoring measures TOTAL influence including pre-training knowledge shared across all samples. DDA ablation shows debias (FM2 fix) contributes -55.2pp, confirming common-mode contamination is the dominant error term in at least one task.
- Mechanism: Subtraction removes the shared component, isolating task-specific influence. Signal processing analogy: differential detection subtracts the reference channel to extract the differential signal.
- Why not simpler: The simplest debiasing is mean-centering attribution scores. This removes the global mean but not sample-specific common-mode influence. Contrastive scoring is the minimal additional complexity that addresses sample-specific FM2.
- **Critical caveat**: Contrastive reference (M_base) must be the same architecture at the same checkpoint before fine-tuning. For pre-training data selection, there is no natural "before" model. This limits FM2 correction to fine-tuning scenarios.

**Complexity**: 2x the computation of the base method (run attribution twice, once on M_ft, once on M_base). Memory: requires loading both models (or sequentially processing). For Pythia-1B: 2x ~2GB = ~4GB, trivial.

**Validation**: → experiment-design.md §3.2 (2x2 ablation column comparison)
**Expected ablation result**: Contrastive scoring improves LDS by 3-10pp on toxicity filtering (high FM2 task). Smaller improvement on data selection (low FM2). If contrastive scoring improves parameter-space more than representation-space, it suggests representation space partially addresses FM2 (the 2x2 interaction term).
**If ablation not significant**: FM2 is not a major bottleneck on DATE-LM tasks, or the subtraction reference (M_base) is too different from M_ft for effective debiasing. Check: does DDA's specific debiasing technique outperform simple subtraction?

### Component C: MAGIC Exact IF (Hessian error control, addresses Hessian bottleneck)

**Function**: Compute exact influence function via metagradient, eliminating Hessian approximation error. Serves as the upper bound for parameter-space TDA quality.

**Input**: Deterministic training run (fixed seeds, data order), test sample
**Output**: Exact attribution scores

**Feasibility assessment for CRA**:
- MAGIC cost: O(N * n * T) where N = training samples, n = test samples, T = training steps.
- DATE-LM toxicity: N ~10K, T ~200 steps (WSD decay), n ~100 test samples.
- At Pythia-1B: single test sample requires ~T backward passes through the full model. Estimated: ~3-5 hours per test sample on A6000.
- For 100 test samples: ~300-500 GPU-hours = 12-20 GPU-days on single GPU.
- **Verdict**: Likely infeasible for full evaluation (100 test samples). Feasible for a SUBSET (5-10 test samples) as proof-of-concept. Budget 5 GPU-days.

**If MAGIC is infeasible at Pythia-1B scale**: FM1 thesis stands by default (exact IF can't be computed to test against representation space). Paper acknowledges this limitation.
**If MAGIC is feasible and achieves LDS >= 0.90**: FM1 thesis weakened. Paper pivots to "representation space as efficient approximation of exact IF."
**If MAGIC achieves LDS 0.70-0.90**: Mixed. FM1 is real but secondary to Hessian error.

**Causal argument**: MAGIC eliminates Hessian approximation by computing exact influence. If MAGIC LDS >> TRAK LDS but MAGIC LDS ~ RepSim LDS, then Hessian error is the dominant bottleneck and FM1 is negligible. If MAGIC LDS >> TRAK LDS but MAGIC LDS < RepSim LDS, then both Hessian and FM1 matter.

**Validation**: → experiment-design.md §3.4 (MAGIC feasibility experiment)

### Component D: LoRA vs Full-FT Dimension (tests FM1 generality)

**Function**: Compare all methods under both LoRA fine-tuning and full fine-tuning to test whether FM1 is a LoRA-specific artifact.

**Rationale**: Li et al.'s strongest FM1 evidence (iHVP degeneracy) is under LoRA, where the effective parameter space is extremely low-rank (rank 16-64). Under full fine-tuning, the Hessian is not artificially low-rank, and FM1's severity is unknown.

**Signal processing prediction (from formalize review, Interdisciplinary)**: FM1 should be MORE severe under full-FT (higher effective dimensionality B) and LESS under LoRA (lower effective dimensionality due to low-rank constraint). If the opposite is observed (FM1 more severe under LoRA), it suggests FM1 is a conditioning problem (LoRA's rank constraint creates ill-conditioning) rather than a dimensionality problem.

**Design**:
- LoRA: rank 16, standard DATE-LM configuration
- Full-FT: all parameters, same learning rate schedule (may need adjustment)
- Compare: RepSim advantage (RepSim LDS - TRAK LDS) under LoRA vs Full-FT

**Complexity**: Full-FT at Pythia-1B requires ~4GB for model + ~8GB for gradients per sample. Feasible on 48GB A6000 with gradient checkpointing.

**Validation**: → experiment-design.md §3.3 (LoRA vs Full-FT experiment)
**Expected result**: RepSim advantage is larger under full-FT than LoRA (FM1 scales with dimensionality). If RepSim advantage is ONLY present under LoRA, FM1 is reframed as LoRA-specific pathology.

## 6. Causal Argument Chain

### Complete chain: Gap → Root Causes → Method → Resolution

**Gap**: Parameter-space TDA fails at LLM scale (RepSim 96-100% vs IF 0-7% on Li et al.). Five representation-space methods independently proposed but never compared or explained.

**Root Causes** (three independent bottlenecks):

1. **Hessian Error** (approximate IF ≠ true IF):
   - Evidence: MAGIC LDS ~0.95-0.99 vs TRAK LDS 0.06-0.24 on same benchmarks.
   - Fixed by: exact IF computation (MAGIC), but at O(N*n) cost.

2. **FM1 (Signal Dilution)** (R^B too high-dimensional for meaningful similarity):
   - Evidence: Li et al. iHVP degeneracy under LoRA. DATE-LM RepSim > Grad-Sim on fine-tuning tasks.
   - Fixed by: operating in representation space R^d where d << B.
   - **Open question**: Is FM1 significant under full-FT? Is it a LoRA artifact?

3. **FM2 (Common Contamination)** (pre-training influence dominates):
   - Evidence: DDA debias ablation -55.2pp. DATE-LM heterogeneous toxicity drops all methods.
   - Fixed by: contrastive scoring (subtract base-model influence).

**Method** (CRA diagnostic framework):
- Component A (repr-space) addresses FM1 → validated by 2x2 row comparison
- Component B (contrastive) addresses FM2 → validated by 2x2 column comparison
- Component C (MAGIC) addresses Hessian → validated by MAGIC vs TRAK gap
- Component D (LoRA vs Full-FT) tests FM1 generality → validated by cross-condition comparison
- 2x2 interaction term tests FM1-FM2 independence

**Why this resolves the gap**: The diagnostic framework provides a DECOMPOSITION that explains conflicting results in the literature. Each existing paper addresses one bottleneck; CRA shows how they fit together. The benchmark fills the acknowledged evaluation gap for representation-space methods on DATE-LM.

### Chain validity conditions (what would break it):

1. If RepSim fails on LDS (< TRAK - 5pp on all tasks): FM1 thesis has no LDS-level support. Framework loses predictive power for the primary metric.
2. If MAGIC achieves LDS >= 0.95 at Pythia-1B scale: Hessian error is the dominant (possibly only significant) bottleneck. FM1/FM2 are secondary.
3. If 2x2 interaction > 30% of min main effect: FM1 and FM2 are not independent. "Three independent bottlenecks" oversimplifies to "tangled failure modes."
4. If FM1 is absent under Full-FT: Three-bottleneck framework reduces to two (Hessian + FM2) for the most common training regime.

## 7. Theoretical Analysis

### 7.1 Signal-Processing Analogy (guiding intuition, not formal proof)

**Matched filtering**: In signal processing, matched filtering maximizes SNR by projecting onto the signal subspace. Representation-space attribution is analogous: h^(l) projects from R^B (noisy, high-dimensional) to R^d (structured, task-relevant). The SNR improvement is approximately d/B ~ 4096/10^9 ~ 10^{-6} → 1 (normalized), meaning the relevant signal that was drowned in R^B becomes dominant in R^d.

**Differential detection**: Subtracting a reference channel removes correlated noise. Contrastive scoring subtracts base-model attribution, removing the "shared pre-training influence" noise floor.

**Caveat**: This analogy is suggestive, not rigorous. h^(l) is not a linear projection (it's a highly nonlinear learned transformation). The "matched filter" analogy breaks if h^(l) discards influence-relevant information that happened to be in the "noise" subspace in R^B.

### 7.2 Dimensionality and FM1 Severity

JL lemma: For N random vectors in R^B, inner products concentrate around 0 when B >> log(N)^2. At B ~ 10^9, N ~ 10^4: log(N)^2 ~ 170. The ratio B/log(N)^2 ~ 6*10^6, meaning per-sample gradient inner products are deeply in the "approximately orthogonal" regime.

Under LoRA (rank r = 16): effective dimension is r * (d_model + d_ffn) ~ 16 * 10K ~ 160K. Still >> log(N)^2, but the signal may concentrate in this low-rank subspace, making FM1 a conditioning rather than dimensionality problem.

Under full-FT: effective dimension is the full B ~ 10^9. FM1 should be at maximum severity.

**Prediction**: RepSim advantage (over TRAK) should be LARGER under full-FT than LoRA.

## 8. Method Positioning

### 8.1 What CRA inherits

- **From DDA**: Contrastive scoring concept (subtract base-model influence). CRA extends this to representation-space methods and tests generality across tasks.
- **From RepT**: Representation-space attribution with gradient augmentation. CRA includes RepT as a method in the benchmark.
- **From MAGIC**: Exact IF as the Hessian-error-free reference. CRA uses MAGIC as a diagnostic control, not a competitor.
- **From DATE-LM**: Evaluation framework and benchmark infrastructure. CRA builds on top of DATE-LM.
- **From Li et al.**: FM1 diagnosis (iHVP degeneracy). CRA extends this to a general three-bottleneck framework.

### 8.2 What CRA changes

- **From individual bottleneck studies → systematic decomposition**: No prior work decomposes TDA failure into three independent bottlenecks and tests their interactions.
- **From single-method evaluation → family benchmark**: First comparative evaluation of representation-space methods on DATE-LM.
- **From LoRA-only FM1 evidence → LoRA vs Full-FT controlled comparison**: Tests FM1 generality for the first time.

### 8.3 Differentiation from nearest competitors

- **vs "Towards Unified Attribution" (2501.18887)**: They provide conceptual taxonomy across attribution types. CRA provides empirical diagnostics within TDA specifically. Complementary, not competing.
- **vs Better Hessians Matter (2509.23437)**: They show Hessian quality matters at small scale. CRA tests whether Hessian quality is sufficient at LLM scale (where FM1/FM2 may dominate).
- **vs individual representation-space methods**: Each proposes a single method for a specific task. CRA evaluates them as a family on a common benchmark.

## 9. Probe Code Reuse

**`Codes/probe/` status**: Empty directory. No existing probe code to reuse.

**Reusable assets from VITA project** (`~/Research/VITA/Codes/`):
- EK-FAC / cosine scoring code: may provide gradient extraction utilities. Needs evaluation for DATE-LM compatibility.

**DATE-LM codebase** (to be cloned):
- TRAK implementation: provided by DATE-LM benchmark. Direct use.
- RepSim implementation: NOT provided by DATE-LM (they use it as a baseline but implementation details may be in their code). Check DATE-LM GitHub.
- Evaluation pipeline: LDS computation, AUPRC computation. Direct use.

**Implementation plan for probe**:
1. Clone DATE-LM repository
2. Identify RepSim implementation in DATE-LM codebase (or implement: extract h^(l), cosine similarity, format scores for evaluation)
3. Run TRAK using DATE-LM's provided implementation
4. Submit both to DATE-LM evaluation pipeline
5. Compare LDS scores
