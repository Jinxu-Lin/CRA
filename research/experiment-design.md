---
version: "1.0"
created: "2026-03-26"
last_modified: "2026-03-26"
entry_mode: "first"
iteration_major: 1
iteration_minor: 0
---

# Experiment Design

## 1. Probe → Full Experiment Transition

### 1.1 What the probe tests

The probe (Experiment 0) is the CRITICAL GATE. It tests H4: whether RepSim (simplest representation-space method) achieves competitive LDS on DATE-LM. Without this signal, the full experimental plan is speculative.

### 1.2 Dimensions expanded from probe to full experiments

| Dimension | Probe | Full Experiments | Justification |
|-----------|-------|-----------------|---------------|
| Tasks | Toxicity filtering only | All 3 DATE-LM tasks | Probe validates on easiest task (analogous to Li et al.); full tests generality |
| Methods | RepSim vs TRAK | 6+ methods (RepSim, RepT, TRAK, DDA, MAGIC, contrastive variants) | Probe gates the direction; full characterizes the landscape |
| Seeds | 1 seed | 3 seeds | Statistical significance |
| Scoring | Standard only | Standard + contrastive | 2x2 ablation requires contrastive variants |
| Fine-tuning | LoRA only | LoRA + Full-FT | LoRA vs Full-FT tests FM1 generality |
| Scale | Pythia-1B | Pythia-1B + selective Llama-7B | Scale-up for publication strength |

### 1.3 Why probe signals should hold at full scale

- Toxicity filtering is the most favorable task for representation-space methods (analogous to Li et al.'s harmful data ID where RepSim achieved 96-100%). If RepSim fails HERE, it will likely fail on harder tasks (data selection, factual attribution).
- DATE-LM's evaluation protocol is standardized. Method rankings at single-seed should be indicative of multi-seed rankings (confirmed by DATE-LM's own analysis showing consistent orderings).
- Expanding to contrastive scoring and Full-FT should not reverse RepSim's advantage -- contrastive scoring is additive, and Full-FT should INCREASE FM1 (making representation-space more advantageous).

### 1.4 Probe failure contingencies

If probe fails (RepSim LDS < TRAK LDS - 5pp on toxicity):
1. Check absolute LDS values (both low → DATE-LM evaluation issue, not method issue)
2. Check layer sensitivity (middle vs last layer)
3. Check P@K alongside LDS (high P@K + low LDS → correlation vs causation gap, itself a publishable finding)
4. Try RepT (gradient augmentation may recover causal signal)
5. If all fail: pivot to "correlation vs causation in representation-space TDA" paper

## 2. Pilot Quick Validation (Experiment 0 + 0.5)

### 2.1 Experiment 0: Probe (Critical Gate)

**Verification target**: Does RepSim achieve competitive LDS on DATE-LM?

**Setup**:
- Model: Pythia-1B, LoRA fine-tuning (DATE-LM default)
- Task: Toxicity filtering (homogeneous setting)
- Methods: RepSim (middle layer + last layer) vs TRAK (DATE-LM implementation)
- Seeds: 1
- Metric: LDS (primary), AUPRC (secondary)

**Protocol**:
1. Clone DATE-LM codebase, verify environment setup (0.5 day)
2. Extract hidden representations h^(l) at layers L/2 and L for all train + test samples (0.5 day)
3. Compute RepSim scores: cos(h^(l)(z_test), h^(l)(z_train)) for all pairs (trivial compute)
4. Run TRAK using DATE-LM's provided implementation (0.5 day)
5. Submit both to DATE-LM LDS evaluation pipeline (0.5 day)
6. Analyze: also compute P@K and Spearman correlation between RepSim and TRAK rankings

**Pass/Adjust/Fail criteria**:

| Signal | Criterion | Action |
|--------|-----------|--------|
| Strong pass | RepSim LDS >= TRAK LDS on toxicity | Proceed to full experiments with high confidence |
| Pass | RepSim LDS >= TRAK LDS - 5pp | Proceed with caution; flag RepSim's limitation |
| Weak pass | RepSim LDS < TRAK - 5pp on toxicity BUT P@K competitive | Pivot emphasis: "correlation vs causation" diagnostic paper |
| Fail | RepSim LDS < TRAK - 5pp AND P@K < TRAK P@K | Direction needs re-evaluation; try RepT before abandoning |

**Time budget**: 2 GPU-days (< 1/30 of total budget)

### 2.2 Experiment 0.5: Mini Pilot (2x2 sanity check)

**Verification target**: Does the 2x2 ablation produce interpretable patterns?

**Setup** (only if Experiment 0 passes):
- Model: Pythia-1B, LoRA
- Task: Toxicity filtering only
- 4 conditions: {RepSim, TRAK} x {standard, contrastive}
- Seeds: 1
- Reduced data: use DATE-LM's smallest evaluation subset if available

**Protocol**:
1. Implement contrastive scoring for RepSim: cos(h_ft(z_test), h_ft(z_train)) - cos(h_base(z_test), h_base(z_train)) (0.5 day)
2. Implement contrastive scoring for TRAK: TRAK_ft - TRAK_base (0.5 day)
3. Run 4 conditions on toxicity filtering (1 day)
4. Compute 2x2 table, check if main effects and interaction have interpretable signs

**Pass criteria**: Both main effects (repr-space, contrastive) are positive (improve LDS). Interaction term is bounded.
**Adjust**: If one main effect is zero or negative, that dimension is dropped from the full experiment.
**Fail**: Both main effects negative → diagnostic framework is wrong.

**Time budget**: 3 GPU-days

## 3. Core Validation (Full Experiments)

### 3.1 Experiment 1: Systematic Benchmark (addresses RQ2)

**Goal**: First comprehensive evaluation of representation-space TDA methods on DATE-LM.

**Methods**:

| Method | Space | Scoring | Implementation Source |
|--------|-------|---------|---------------------|
| TRAK | Parameter | Standard | DATE-LM codebase |
| Grad-Sim | Parameter | Standard | DATE-LM codebase |
| DDA | Parameter | Contrastive | Reimplemented (TRAK_ft - TRAK_base) |
| RepSim | Representation | Standard | Custom (h^(l) cosine similarity) |
| RepT | Representation | Standard | Reimplemented (concat[h^(l*), nabla_h L] cosine) |
| BM25 | Lexical | Standard | DATE-LM codebase (non-neural baseline) |

**Evaluation matrix**:

| | Data Selection | Toxicity Filtering | Factual Attribution |
|--|---------------|-------------------|-------------------|
| Pythia-1B LoRA | All 6 methods x 3 seeds | All 6 methods x 3 seeds | All 6 methods x 3 seeds |

**Metrics**: LDS (primary), AUPRC (toxicity), Recall@50 + MRR (factual), P@K (all tasks as secondary)

**Compute**: 6 methods x 3 tasks x 3 seeds x ~0.3 GPU-days/run = ~16 GPU-days

**Cross-references**: RepSim and RepT results validate Component A (method-design.md §5 Component A). DDA results validate Component B effectiveness in parameter space.

### 3.2 Experiment 2: 2x2 Ablation (addresses RQ3 + core of RQ1)

**Goal**: Quantify FM1 and FM2 contributions and test their independence.

**Design**: {parameter-space, representation-space} x {standard scoring, contrastive scoring}

| Cell | Method Instance | FM1 Status | FM2 Status |
|------|----------------|------------|------------|
| (param, standard) | TRAK | Present | Present |
| (param, contrastive) | TRAK_ft - TRAK_base | Present | Fixed |
| (repr, standard) | RepSim | Fixed | Present |
| (repr, contrastive) | RepSim_ft - RepSim_base | Fixed | Fixed |

**For each DATE-LM task** (3 tasks x 4 conditions x 3 seeds = 36 runs):

**Statistical analysis**:
- Per-sample permutation test (NOT task-level ANOVA): For each test sample, compute LDS contribution across the 4 conditions. Permutation test on per-sample differences.
- Report: FM1 main effect = mean(repr conditions) - mean(param conditions)
- Report: FM2 main effect = mean(contrastive conditions) - mean(standard conditions)
- Report: Interaction = (repr+contrastive) - (repr+standard) - (param+contrastive) + (param+standard)
- Bootstrap 95% CI for all effects

**Interaction interpretation guide** (from formalize review):
- |Interaction| < 10% of min(|FM1 effect|, |FM2 effect|): Clean independence. Three-bottleneck framework validated.
- |Interaction| 10-30% of min(main effects): Moderate interaction. Framework holds with "partial overlap" caveat.
- |Interaction| > 30% of min(main effects): Strong interaction. "Independent bottlenecks" narrative requires revision. Reframe as "tangled failure modes with partial separability."

**CMRR as secondary FM2 metric** (from formalize review, Interdisciplinary): Common-Mode Rejection Ratio = |standard score - contrastive score| / |standard score|. Quantifies how much of the standard attribution is common-mode contamination. Report CMRR per-task.

**Compute**: 4 conditions x 3 tasks x 3 seeds x ~0.3 GPU-days = ~11 GPU-days. Some overlap with Experiment 1 (TRAK and RepSim runs shared).

**Cross-references**: FM1 main effect validates Component A (method-design.md §5). FM2 main effect validates Component B. Interaction validates the independence assumption (method-design.md §6).

### 3.3 Experiment 3: LoRA vs Full-FT (addresses RQ1 LoRA-specificity)

**Goal**: Test whether FM1 is a LoRA-specific artifact or a general LLM-scale phenomenon.

**Design**: {LoRA, Full-FT} x {RepSim, TRAK} on 2 tasks (toxicity filtering + data selection)

| Cell | FT Mode | Method | Expected FM1 Severity |
|------|---------|--------|----------------------|
| (LoRA, TRAK) | LoRA r=16 | TRAK | High (Li et al. evidence) |
| (LoRA, RepSim) | LoRA r=16 | RepSim | Low (FM1 bypassed) |
| (Full-FT, TRAK) | Full parameters | TRAK | Very high (prediction) |
| (Full-FT, RepSim) | Full parameters | RepSim | Low (FM1 bypassed) |

**Key metric**: RepSim advantage = RepSim LDS - TRAK LDS

- Under LoRA: RepSim advantage = X
- Under Full-FT: RepSim advantage = Y
- If Y > X: FM1 scales with dimensionality (general phenomenon)
- If Y < X: FM1 is LoRA-specific (conditioning artifact)
- If Y ~ X: FM1 severity is independent of FT mode (unexpected; suggests different mechanism)

**Full-FT protocol for Pythia-1B**:
- Learning rate: sweep {1e-5, 5e-5, 1e-4} on dev set (3 quick runs)
- DATE-LM's WSD scheduler with 200-step decay
- Gradient checkpointing to fit in 48GB
- 3 seeds per condition

**Compute**: 2 FT modes x 2 methods x 2 tasks x 3 seeds x ~0.5 GPU-days = ~12 GPU-days

**Cross-references**: Validates Component D (method-design.md §5). Tests the dimensionality prediction in method-design.md §7.2.

### 3.4 Experiment 4: MAGIC Feasibility (addresses RQ1 Hessian contribution)

**Goal**: Assess whether exact IF achieves high LDS on DATE-LM, bounding the Hessian error contribution.

**Protocol**:
1. Implement MAGIC for Pythia-1B on DATE-LM toxicity filtering
2. Use deterministic training (fixed seed, fixed data order, fixed batch composition)
3. Store all training checkpoints (200 steps x Pythia-1B checkpoint ~ 200 x 2GB = 400GB)
4. Compute MAGIC attribution for 5-10 test samples (feasibility limit)
5. Compare MAGIC LDS (on subset) vs TRAK LDS vs RepSim LDS

**Feasibility assessment**:
- Memory: Need to store optimizer state + gradients at each step. For Pythia-1B with Adam: ~8GB per checkpoint. 200 checkpoints = 1.6TB. **This may exceed available disk.** Mitigation: gradient checkpointing at every N steps, re-compute intermediate.
- Time: ~3-5 hours per test sample. 10 samples = 30-50 hours = 1.5-2 GPU-days per seed.
- **Most likely outcome** (per Pragmatist): MAGIC is infeasible at Pythia-1B due to memory/disk constraints. In this case, report infeasibility and note that FM1 thesis is not tested against exact IF at this scale.

**Decision rule** (from problem-statement.md §1.3):
- MAGIC LDS >= 0.90 on DATE-LM toxicity: FM1 thesis weakened. Paper pivots to "efficiency" argument.
- MAGIC infeasible: FM1 thesis stands by default with acknowledged limitation.
- MAGIC LDS 0.70-0.90: FM1 is secondary to Hessian error.

**Compute**: 5 GPU-days (capped; if infeasible, reallocate to other experiments)

### 3.5 Experiment 5: Scale-Up (publication strength)

**Goal**: Demonstrate findings generalize beyond Pythia-1B.

**Design**: Run best-performing methods from Experiments 1-3 on Llama-7B, toxicity filtering + data selection.

**Methods**: Top 2-3 methods + TRAK baseline. LoRA only (Full-FT at 7B exceeds 48GB).

**Compute**: 3 methods x 2 tasks x 3 seeds x ~1.5 GPU-days = ~8 GPU-days (if budget allows)

**Contingency**: If budget is tight, reduce to 1 task (toxicity) and 2 methods (RepSim + TRAK). Minimum viable: 2 x 1 x 3 x 1.5 = 9 GPU-days.

## 4. Baseline Selection & Justification

| Baseline | Type | Why Included | Source |
|----------|------|-------------|--------|
| **TRAK** | Parameter, standard | Most widely used gradient-based TDA. DATE-LM reference implementation. Strongest approximate IF baseline. | DATE-LM codebase |
| **Grad-Sim** | Parameter, standard | Simple gradient cosine similarity. Separates "projection" (TRAK) from "similarity" contributions. DATE-LM shows competitive on factual attribution. | DATE-LM codebase |
| **DDA (TRAK contrastive)** | Parameter, contrastive | Tests FM2 fix in parameter space. DDA's core contribution. | Reimplemented |
| **RepSim** | Representation, standard | Simplest representation-space baseline. DATE-LM includes it. Tests FM1 fix. | DATE-LM / custom |
| **RepT** | Representation, standard | SOTA representation method. Gradient augmentation adds causal signal. | Reimplemented |
| **BM25** | Lexical | Non-neural baseline. DATE-LM shows BM25 competitive on factual attribution (lexical overlap). Essential to separate "retrieval" from "influence". | DATE-LM codebase |
| **Random** | Lower bound | Lower bound. Calibrates effect sizes. | Trivial |
| **MAGIC** | Parameter, exact | Upper bound for parameter-space IF (Hessian error = 0). If feasible. | Reimplemented (best-effort) |

**Fair comparison protocol**:
- All methods use DATE-LM's evaluation pipeline (identical test sets, identical metrics)
- All methods use the same fine-tuned model checkpoint (same theta_ft)
- For contrastive variants: same base model checkpoint (same theta_base)
- Hyperparameter tuning: each method gets equivalent tuning budget (layer selection for RepSim, projection dimension for TRAK, etc.)
- Cosine similarity normalization applied universally (DATE-LM finding: cosine > inner product)

**Not included** (with justification):
- EK-FAC: DATE-LM already benchmarks it. Adding another approximate IF method is redundant given TRAK + Grad-Sim + MAGIC span the parameter-space quality range.
- LESS: Task-specific variant of gradient similarity. May include if time permits but not core to the diagnostic framework.
- AirRep: Learns a new encoder space (not model-internal representations). Conceptually different from RepSim/RepT. Exclude for scope.
- Concept IF: Projects back to parameter space. Doesn't cleanly test FM1 (representation-space operation). Exclude for scope.

## 5. Metric Definition (RQ-Aligned)

### 5.1 Primary Metrics

| Metric | RQ Alignment | Definition | Task |
|--------|-------------|-----------|------|
| **LDS** | RQ1, RQ2, RQ3 | Spearman correlation between predicted and actual model output changes when training subsets are removed | All tasks (primary) |
| **AUPRC** | RQ2 | Area under precision-recall curve for detecting unsafe training samples | Toxicity filtering |
| **Recall@50 + MRR** | RQ2 | Retrieval quality for factual attribution | Factual attribution |

### 5.2 Diagnostic Metrics

| Metric | Purpose | Definition |
|--------|---------|-----------|
| **FM1 main effect** | Quantify signal dilution contribution | mean(LDS_repr) - mean(LDS_param) from 2x2 |
| **FM2 main effect** | Quantify common contamination contribution | mean(LDS_contrastive) - mean(LDS_standard) from 2x2 |
| **Interaction term** | Test FM1-FM2 independence | (repr+contr) - (repr+std) - (param+contr) + (param+std) |
| **CMRR** | Secondary FM2 quantifier | |standard_score - contrastive_score| / |standard_score| averaged over samples |
| **RepSim advantage** | FM1 severity proxy | RepSim_LDS - TRAK_LDS, reported per FT mode |
| **P@K vs LDS gap** | Correlation vs causation diagnostic | P@10(method) - LDS(method), reported per method |

### 5.3 Efficiency Metrics

| Metric | Definition |
|--------|-----------|
| **GPU-hours per 1K test samples** | Total wall-clock time for scoring 1K test samples against full training set |
| **Peak memory (GB)** | Maximum GPU memory during attribution computation |
| **Scoring throughput** | Test samples scored per GPU-hour |

### 5.4 Statistical Significance Plan

- **Per-sample permutation test**: For each pair of methods, compute per-sample LDS contribution differences. Permutation test (10K permutations) for significance.
- **Bootstrap CI**: 95% bootstrap confidence intervals for all reported metrics (1K bootstrap samples over 3 seeds).
- **Effect size**: Cohen's d for pairwise method comparisons.
- **Multiple comparison correction**: Benjamini-Hochberg FDR control (q = 0.05) across all pairwise comparisons within each task.
- **Minimum detectable effect**: With 3 seeds and ~100 test samples per task, we can detect LDS differences of ~3-5pp at alpha=0.05, power=0.80 (estimated from DATE-LM variance reports).

## 6. Application Value (Downstream Tasks)

### 6.1 Practitioner Guidance Table

The primary application contribution is a decision table for practitioners:

| Your Task | Best Method | Compute Budget | Recommendation |
|-----------|------------|----------------|----------------|
| Data selection (pre-training) | TBD from Exp 1 | Low | Use [method] |
| Data selection (fine-tuning) | TBD from Exp 1 | Low | Use [method] |
| Toxicity filtering | TBD from Exp 1 | Low-Medium | Use [method] |
| Factual attribution | TBD from Exp 1 | Low | Use [method] |
| Any task (high accuracy needed) | MAGIC (if feasible) | Very high | Use MAGIC |

### 6.2 Cost-Benefit Analysis

For each method, report: LDS per GPU-hour. This is the primary practitioner-relevant metric -- how much attribution quality can you buy with a given compute budget?

## 7. Efficiency Validation

All methods will be profiled for:
1. **Scoring time**: Wall-clock seconds to compute attribution scores for all test-train pairs
2. **Memory footprint**: Peak GPU memory during scoring
3. **Preprocessing time**: Time to extract features (gradients or representations) from the model

Expected ordering (fastest to slowest):
- RepSim (forward pass only, no gradients) < RepT (forward + backward for nabla_h) < Grad-Sim (full backward) < TRAK (full backward + projection) < DDA (2x TRAK) < MAGIC (T backward passes)

This ordering is itself informative: if representation-space methods are BOTH more accurate (higher LDS) AND more efficient (lower GPU-hours), the practical case for them is overwhelming regardless of the theoretical framework.

## 8. Scientific Discovery (Conditional)

If core validation succeeds (Experiments 1-3 produce clean results), secondary analyses:

### 8.1 Layer Selection Analysis

For RepSim, sweep all layers and plot LDS vs layer index. Compare with RepT's phase-transition detection. This characterizes WHERE in the model the attribution-relevant information resides.

### 8.2 Task-Dependent Bottleneck Profiles

Plot FM1 main effect and FM2 main effect per task. Expected pattern:
- Toxicity filtering: high FM2 (shared language patterns), moderate FM1
- Data selection: low FM2 (no natural "common mode" in pre-training selection), moderate FM1
- Factual attribution: low FM2, potentially low FM1 (lexical features matter, per BM25 finding)

This task-dependent profile is a novel characterization of WHERE each bottleneck matters.

### 8.3 Model-Size Scaling (if Experiment 5 completes)

Plot FM1 main effect vs model size (Pythia-1B vs Llama-7B). If FM1 increases with model size, it validates the dimensionality argument. If FM1 is constant, it suggests FM1 is architecture-dependent rather than scale-dependent.

## 9. Dataset & Compute Plan

### 9.1 Dataset Details

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| DATE-LM Toxicity | UltraChat 10K + <100 unsafe | ~10K train, ~100 test | Toxicity filtering |
| DATE-LM Data Selection | Fineweb corpus | Variable (DATE-LM provides subsets) | Data selection |
| DATE-LM Factual | ROME entity-fact pairs | ~5K train, ~100 test | Factual attribution |

### 9.2 Compute Budget Summary

| Experiment | GPU-days | Priority | Depends On |
|------------|----------|----------|------------|
| Exp 0: Probe | 2 | P0 (MUST DO FIRST) | Nothing |
| Exp 0.5: Mini pilot | 3 | P0 (gate for full exp) | Exp 0 pass |
| Exp 1: Benchmark | 15 | P1 | Exp 0 pass |
| Exp 2: 2x2 ablation | 10 | P1 | Exp 0.5 pass |
| Exp 3: LoRA vs Full-FT | 12 | P1 | Exp 0 pass |
| Exp 4: MAGIC feasibility | 5 | P2 | Independent |
| Exp 5: Scale-up | 8 | P3 | Exp 1 results |
| Buffer | 5 | -- | -- |
| **Total** | **60** | | |

**Available**: 4x A6000 x ~2 months x 75% utilization = ~180 GPU-days. Budget of 60 GPU-days is well within capacity (33% utilization accounts for shared server, debugging, failed runs).

### 9.3 Execution Timeline

| Week | Experiments | Deliverable |
|------|------------|-------------|
| 1 | Exp 0 (probe) + Exp 4 (MAGIC feasibility, parallel) | Probe result → gate decision |
| 2 | Exp 0.5 (mini pilot) + start Exp 1 | 2x2 sanity check |
| 3-4 | Exp 1 (benchmark) + Exp 2 (2x2 ablation) + Exp 3 (LoRA vs FT) | Core results |
| 5 | Exp 5 (scale-up) + analysis | Complete results |
| 6 | Paper writing | Draft |

## 10. Expected Results & Failure Plans

### 10.1 Expected Results (Concrete Predictions)

**Experiment 1 (Benchmark)**:
- RepSim LDS on toxicity: 0.15-0.35 (above TRAK's expected 0.05-0.20, based on DATE-LM's Grad-Sim ~0.10-0.25)
- RepT LDS on toxicity: 0.20-0.40 (gradient augmentation adds ~5pp over RepSim)
- RepSim AUPRC on toxicity (homogeneous): 0.95+ (matching DATE-LM's reported 0.989)
- BM25 competitive on factual attribution (DATE-LM finding: lexical overlap matters)
- No single method best across all tasks (DATE-LM finding)

**Experiment 2 (2x2 Ablation)**:
- FM1 main effect (repr vs param): +5 to +15pp LDS on toxicity filtering
- FM2 main effect (contrastive vs standard): +3 to +10pp LDS on toxicity filtering
- Interaction term: < 30% of min(main effects) on at least 2/3 tasks
- CMRR on toxicity: 0.3-0.6 (30-60% of standard score is common-mode)

**Experiment 3 (LoRA vs Full-FT)**:
- RepSim advantage under LoRA: +5 to +15pp LDS
- RepSim advantage under Full-FT: +10 to +25pp LDS (larger, confirming FM1 scales with dimensionality)
- TRAK LDS under Full-FT < TRAK LDS under LoRA (Full-FT makes parameter-space harder)

**Experiment 4 (MAGIC)**:
- Most likely: MAGIC infeasible at Pythia-1B scale on A6000 (memory/disk constraints)
- If feasible: MAGIC LDS 0.70-0.95 (lower than MAGIC's reported 0.95-0.99 because DATE-LM evaluation may be harder than MAGIC's original benchmarks)

### 10.2 Failure Modes & Responses

| Failure Mode | Probability | Signal | Response |
|-------------|------------|--------|----------|
| RepSim fails on LDS (< TRAK - 5pp, all tasks) | 25% | Probe (Exp 0) | Pivot to "correlation vs causation" diagnostic. Still publish benchmark + FM2 analysis |
| 2x2 interaction dominates (> 30% of main effects, all tasks) | 15% | Exp 2 | Reframe "independent bottlenecks" → "partially separable failure modes." Result is still informative |
| FM1 absent under Full-FT | 20% | Exp 3 | Reframe FM1 as "LoRA-specific pathology." Paper becomes: Hessian + FM2 + LoRA conditioning |
| MAGIC achieves LDS > 0.90 at Pythia-1B | 10% | Exp 4 | Pivot to "representation space as efficient approximation." Cost-benefit analysis becomes central |
| All methods have similar LDS on DATE-LM | 10% | Exp 1 | DATE-LM may not discriminate well between methods. Add Li et al. benchmark as primary evaluation |
| Contrastive scoring hurts on data selection | 30% | Exp 2 | Expected for low-FM2 tasks. Report task-dependent FM2 profile; do not claim universal enhancement |

### 10.3 Abandon Criteria

The project should be abandoned (redirected) if:
1. RepSim fails on LDS AND RepT fails on LDS AND P@K is also not competitive → representation-space methods have no diagnostic value for TDA
2. MAGIC achieves LDS > 0.95 at feasible cost AND contrastive scoring provides < 2pp improvement → all three bottlenecks are either solved (Hessian by MAGIC) or negligible (FM1, FM2) → no diagnostic contribution

These are unlikely (estimated combined probability < 5%) but must be stated.

### 10.4 Tier-3 Risk: Concurrent Competition

If a directly competing paper appears during the research period (someone publishes representation-space benchmark on DATE-LM):
- Check their method coverage. If they don't do the 2x2 ablation + FM1/FM2 decomposition, CRA still has unique contribution.
- If they do full decomposition: accelerate timeline, focus on LoRA vs Full-FT dimension (most likely to be unique).
- Worst case: pivot to emphasizing the diagnostic framework and novel findings rather than the benchmark.
