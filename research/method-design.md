---
version: "1.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "assimilated"
iteration_major: 1
iteration_minor: 0
---

# Method Design

> [ASSIMILATED: Synthesized from CRA_old's attack angle A and strategic review conditions. Pending formal design phase for full detail.]

## 1. Framework Overview

### 1.1 Signal-Processing Diagnostic Framework

Two independent signal-processing defects in parameter-space TDA for LLMs:

- **FM1 (Signal Dilution)**: In R^B (B ~ 10^9), per-sample gradients are nearly orthogonal (JL phenomenon). Task-relevant signal occupies a tiny subspace -> SNR collapse. **Fix**: Operate in representation space (d ~ 4096 << B), concentrating signal.

- **FM2 (Common Influence Contamination)**: Standard IF measures total influence dominated by shared pre-training knowledge. **Fix**: Contrastive scoring (subtract base-model influence), isolating task-specific signal.

These are complementary to Hessian approximation error (the third bottleneck). At LLM scale, FM1/FM2 become dominant.

### 1.2 Unified Bilinear Taxonomy

All 5 representation-space methods share: I(z_test, z_train) = phi(z_test)^T * psi(z_train)

Each method differs in how phi and psi are constructed (see problem-statement.md Section 1.1 table).

### 1.3 2x2 Ablation Design

Core experimental matrix:

|  | Standard Scoring | Contrastive Scoring |
|--|-----------------|-------------------|
| **Parameter-space** (TRAK, LoGra/IF) | Baseline | FM2 fix only |
| **Representation-space** (RepSim, RepT) | FM1 fix only | FM1 + FM2 fix |

This design allows:
- Main effect of representation space (FM1 fix): row comparison
- Main effect of contrastive scoring (FM2 fix): column comparison
- Interaction term: tests independence of FM1 and FM2

## 2. Method Components

### 2.1 Representation-Space Methods

**RepSim**: Cosine similarity of hidden representations h^(l) at selected layer. Simplest baseline.

**RepT**: Concatenated [h^(l*), nabla_h L] signatures with automatic phase-transition layer detection. State-of-the-art representation method.

### 2.2 Parameter-Space Baselines

**TRAK**: Existing DATE-LM implementation. Standard gradient-based TDA.
**DDA**: Contrastive scoring on parameter-space gradients. IS_DDA = IS_{theta'} - IS_{theta_0}.
**MAGIC**: Key competitor (discovered in strategic review, arXiv 2504.16430). Must be included.

### 2.3 Contrastive Scoring Enhancement

For each base method M, construct contrastive variant:
- I_contrastive(z_test, z_train) = I_M(z_test, z_train; theta_ft) - I_M(z_test, z_train; theta_base)

Challenge: z_cf (counterfactual reference) construction is task-dependent. Toxicity filtering has natural contrasts; data selection does not.

## 3. Evaluation

### 3.1 Benchmarks

- **DATE-LM** (primary): 3 tasks (data selection, toxicity filtering, factual attribution). LDS metric. Pythia-1B to Llama-7B.
- **Li et al. benchmark** (secondary): Harmful data identification, class attribution, backdoor detection.

### 3.2 Metrics

- **Primary**: LDS (Linear Datamodeling Score) on DATE-LM
- **Secondary**: P@K, AUC, Spearman rank correlation
- **Analysis**: 2x2 ANOVA (main effects + interaction term)

### 3.3 Methods to Evaluate

| Method | Space | Scoring | Category |
|--------|-------|---------|----------|
| TRAK | Parameter | Standard | Baseline |
| LoGra/IF | Parameter | Standard | Baseline |
| DDA | Parameter | Contrastive | FM2 fix |
| MAGIC | Parameter | Enhanced | Competitor |
| RepSim | Representation | Standard | FM1 fix |
| RepT | Representation | Standard | FM1 fix |
| RepSim + Contrastive | Representation | Contrastive | FM1+FM2 |
| RepT + Contrastive | Representation | Contrastive | FM1+FM2 |

## 4. Open Design Questions (for formal design phase)

1. **Layer selection strategy** for RepSim across DATE-LM tasks
2. **Contrastive reference construction** for data selection task (no natural pairs)
3. **Per-sample gradient extraction** memory management for 7B models
4. **Hessian scaling argument** formalization (scaling experiment on intermediate-size models?)
5. **Fixed-IF** (optional): Projected IF + contrastive gradient in parameter space. Defer decision based on core ablation results.
