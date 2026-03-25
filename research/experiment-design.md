---
version: "1.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "assimilated"
iteration_major: 1
iteration_minor: 0
---

# Experiment Design

> [ASSIMILATED: Synthesized from CRA_old's probe design + CRA's pilot methodology. Pending formal design phase for full detail.]

## 1. Probe Experiment (Priority)

### 1.1 Setup

- **Benchmark**: DATE-LM (GitHub: DataAttributionEval/DATE-LM)
- **Model**: Pythia-1B (smallest DATE-LM option)
- **Task**: Toxicity filtering (DATE-LM Task 2) -- most analogous to Li et al.'s harmful data identification
- **Compute**: < 1 GPU-day on a single A100

### 1.2 Protocol

1. Clone DATE-LM codebase
2. Extract hidden representations h^(l)(z) for all training and test samples at middle and last layers
3. Compute RepSim: I(z_test, z_train) = cos(h^(l)(z_test), h^(l)(z_train))
4. Submit scores to DATE-LM evaluation pipeline (LDS computation)
5. Run TRAK baseline using DATE-LM's existing implementation
6. Compare RepSim LDS vs TRAK LDS

### 1.3 Pass Criteria

| Signal | Criterion | Interpretation |
|--------|-----------|----------------|
| Strong pass | RepSim LDS >= TRAK LDS | Representation space wins with simplest method |
| Pass | RepSim LDS >= TRAK LDS - 5pp | Competitive; full evaluation justified |
| Weak pass | RepSim < TRAK - 5pp on toxicity, >= TRAK - 5pp on data selection | Task-dependent |
| Fail | RepSim < TRAK - 5pp on both tasks | Direction needs re-evaluation |

## 2. Core Experiments (Post-Probe)

### 2.1 Experiment 1: Systematic Benchmark

All methods evaluated on DATE-LM across 3 tasks x 2-3 model sizes.

| Method | Tasks | Models | Runs |
|--------|-------|--------|------|
| RepSim, RepT | All 3 DATE-LM tasks | Pythia-1B, Llama-7B | 3 seeds each |
| TRAK, LoGra | All 3 DATE-LM tasks | Same | 3 seeds each |
| DDA, MAGIC | All 3 DATE-LM tasks | Same | 3 seeds each |

### 2.2 Experiment 2: 2x2 Ablation

For each of the 3 DATE-LM tasks:
- 4 conditions: {parameter, representation} x {standard, contrastive}
- 3 seeds per condition
- ANOVA analysis for main effects and interaction

### 2.3 Experiment 3: Li et al. Benchmark Extension

Replicate Li et al.'s harmful data identification + class attribution + backdoor detection with:
- Added methods: RepT, DDA-enhanced RepSim
- Added analysis: 2x2 ablation on these tasks too

### 2.4 Experiment 4 (Optional): Fixed-IF

If core ablation is clean (small interaction term), implement projected IF + contrastive gradient in parameter space to verify diagnostic framework's predictive power. Decision contingent on Experiments 1-3 results.

## 3. Baselines

| Baseline | Type | Source |
|----------|------|--------|
| TRAK | Parameter-space gradient TDA | DATE-LM included |
| LoGra/IF | Parameter-space influence function | Standard |
| DDA | Parameter-space + contrastive | EMNLP 2024 |
| MAGIC | Enhanced parameter-space | arXiv 2504.16430 |
| Random | Random ranking | Lower bound |
| BM25 | Lexical similarity | Non-neural baseline |

## 4. Compute Budget

| Phase | Estimate | Hardware |
|-------|----------|----------|
| Probe (RepSim vs TRAK on toxicity) | 1 GPU-day | 1x A100 |
| DATE-LM full benchmark (8 methods x 3 tasks x 3 seeds) | 10-15 GPU-days | 4x A6000 |
| Li et al. extension | 3-5 GPU-days | 4x A6000 |
| 2x2 ANOVA analysis | Negligible | CPU |
| Total | ~20 GPU-days | |

## 5. Statistical Analysis Plan

- **2x2 ANOVA**: Main effects (space, scoring) + interaction term
- **Bonferroni correction** for multiple comparisons across tasks
- **Effect size**: Cohen's d for pairwise comparisons
- **Interaction threshold**: If interaction > 30% of min(main effect), FM1/FM2 independence is questionable
