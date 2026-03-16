# Pragmatist Perspective: CRA Research Proposal

**Agent**: Pragmatist (computational feasibility, engineering simplicity, reliable baselines)
**Date**: 2026-03-16

---

## Executive Summary

The CRA thesis is strong: diagnosing FM1 (signal dilution) and FM2 (common influence contamination) as two independent failure modes, unifying 5 representation-space methods under phi^T * psi, and validating on DATE-LM. My focus is on what is *actually buildable* on 4x RTX 4090 within realistic time budgets, what engineering pitfalls will kill the project, and which experimental design choices maximize the probability of clean, publishable results. I propose three angles: (1) the minimal viable 2x2 ablation with hardened engineering, (2) a diagnostic gradient-dimension sweep that provides direct evidence for FM1, and (3) a contrastive-scoring plug-in that stress-tests FM2 across methods. Each angle is designed so that a pilot runs in 10-15 minutes and the full experiment completes within 1 GPU-hour.

---

## Angle 1: Hardened 2x2 Ablation Matrix -- Get the Core Result Right

### Core Insight (Improve Existing: Engineering-First Execution of the Core Claim)

The 2x2 ablation {parameter-space, representation-space} x {standard scoring, contrastive scoring} is the paper's load-bearing experiment. If this fails or produces ambiguous results, nothing else matters. The pragmatist's job is to identify and pre-empt the 5 engineering risks that could ruin it.

**What Exactly Needs to Be Built**:

| Cell | Method | phi | psi | Implementation Source |
|------|--------|-----|-----|---------------------|
| Param + Standard | TRAK | grad(theta, z_test) | grad(theta, z_train) | github.com/MadryLab/trak |
| Param + Contrastive | TRAK + DDA-style debias | grad(theta, z_test) - E[grad] | grad(theta, z_train) - E[grad] | Custom on top of TRAK |
| Rep + Standard | RepSim | h(z_test) | h(z_train) | Cosine similarity, trivial |
| Rep + Contrastive | RepSim + contrastive | h(z_test) - h(z_neg) | h(z_train) | Custom, ~50 lines |

Plus 3 additional representation methods (RepT, AirRep-lite, Concept Influence proxy) to populate the unification table.

**5 Engineering Risks and Mitigations**:

1. **TRAK on Pythia-1B memory blow-up**: TRAK's random projection requires storing projected gradients for all training samples. At 1B params with projection dim k=4096 and N=50K training samples, this is ~800MB -- manageable. But the gradient computation itself for 1B params requires gradient checkpointing. *Mitigation*: Use LoRA-only gradients (rank=16, ~2M params) as LESS does. This reduces gradient dim from 1B to 2M, making TRAK tractable. If TRAK with full gradients is needed, use LoGra's backpropagation-aware projection.

2. **DATE-LM data loading and evaluation protocol**: DATE-LM has specific evaluation protocols (LDS, LOO correlation, auPRC) with potential pitfalls in data preprocessing. *Mitigation*: Fork DATE-LM's official codebase (github.com/DataAttributionEval/DATE-LM) directly. Do NOT reimplement metrics. Use their model checkpoints verbatim.

3. **Negative sample selection for contrastive scoring**: DDA's contrastive scoring requires a set of negative (irrelevant) training samples. The choice of negatives critically affects performance -- random negatives may not sufficiently debias. *Mitigation*: Use DATE-LM's task structure to define negatives naturally (e.g., for toxicity filtering: non-toxic samples are negatives; for data selection: out-of-domain samples). Report results with 3 different negative sampling strategies to show robustness.

4. **Layer selection for representation methods**: Vitel & Chhabra (2511.04715) just showed that middle attention layers outperform first/last layers for influence estimation, contradicting prior wisdom. RepT's "phase transition layer" finding aligns with this. *Mitigation*: Always report results at the best layer (selected by validation) AND at a fixed layer (layer L/2) for reproducibility. This also generates a clean figure for the paper.

5. **Statistical significance with 3 DATE-LM tasks**: With only 3 tasks, you cannot run a meaningful ANOVA. The "interaction term < 30% of min main effect" pre-registration condition is appropriate, but report confidence intervals via bootstrap (1000 resamples of test queries) rather than p-values.

**Experimental Plan**:
- **Pilot (15 min)**: On Pythia-1B + DATE-LM data selection task only. Implement RepSim (standard) and TRAK (standard, LoRA gradients, k=1024). Verify evaluation pipeline produces reasonable numbers by comparing with DATE-LM leaderboard.
- **Core experiment (45 min)**: Run full 2x2 matrix on all 3 DATE-LM tasks. Use Pythia-1B checkpoint from DATE-LM. Compute attribution scores for N=10K training samples (subsample if DATE-LM training set is larger).
- **Extension (30 min)**: Add RepT, simplified AirRep (learned linear projection of representations, without full encoder training), and Concept Influence proxy (linear probe direction) to the representation-space column.

**Computational Cost**: ~2 GPU-hours total on single RTX 4090
**Success Probability**: 85% -- the core 2x2 result is almost certain to show representation > parameter and contrastive > standard; the risk is that the effects are not cleanly additive (interaction term too large)
**Failure Modes**:
- (a) TRAK with LoRA gradients performs so poorly that the "parameter-space" baseline is unfairly weak -- mitigation: also run LoGra with full gradient projection as a second parameter-space baseline
- (b) Contrastive scoring helps parameter-space but not representation-space methods (because RepSim already implicitly debiases via cosine normalization) -- this would actually be an interesting finding, suggesting FM2 is partially addressed by the choice of similarity metric
- (c) DATE-LM checkpoint incompatibility or API changes -- mitigation: pin exact commit hash and model versions

**Key References**:
- DATE-LM (2507.09424) -- benchmark and evaluation protocol
- TRAK (2303.14186) -- parameter-space baseline
- DDA (2410.01285) -- contrastive scoring reference
- Vitel & Chhabra (2511.04715) -- layer selection evidence (middle layers > first/last)
- Li et al. (2512.09103) -- Natural geometry of robust attribution; warns that TRAK scores are "geometrically fragile" under Euclidean perturbation, motivating representation-space alternatives

---

## Angle 2: Gradient Dimension Sweep -- Direct Empirical Evidence for FM1

### Core Insight (New Diagnostic Method: Controlled Dimension Reduction Experiment)

FM1 claims that parameter-space methods suffer from signal dilution because gradients live in R^B where B is huge. The current narrative relies on the JL lemma intuition. But we can *directly measure* this: systematically vary the projection dimension k in TRAK and plot attribution quality vs. k. If FM1 is real, there should be a characteristic curve: quality increases with k but saturates far below B, and the saturation point should approximately match the representation dimension d.

This is not the Innovator's eigenspectrum analysis (which requires computing gradient covariance). This is a dead-simple sweep that anyone can reproduce: run TRAK with k in {64, 128, 256, 512, 1024, 2048, 4096, 8192} and measure LDS/auPRC at each k.

**Hypothesis**: Attribution quality saturates at k ~ O(d) where d is the representation dimension (~2048 for Pythia-1B). This provides direct evidence that the "useful signal" in parameter gradients is low-dimensional and approximately matches the representation space dimensionality.

**What Makes This Practical**:
- No new algorithm needed -- just run TRAK multiple times with different k
- Each run with k=4096 takes ~5 min on Pythia-1B with 10K samples
- Total sweep (8 values of k) takes ~40 min
- The resulting plot (k vs. attribution quality) is a single, compelling figure

**Complementary Sweep**: Also vary the representation dimension by applying PCA to representations before computing RepSim. Sweep d_eff in {64, 128, ..., 2048}. If representation-space methods are robust to dimensionality reduction (quality saturates at much smaller d_eff), this further supports the "signal is low-rank" narrative.

**Connection to phi^T * psi Framework**: The dimension sweep directly probes the rank of the effective phi^T * psi matrix. If phi = projected_gradient and psi = projected_gradient, then the sweep measures how many dimensions of the bilinear form carry attribution signal.

**Experimental Plan**:
- **Pilot (10 min)**: Run TRAK with k in {128, 512, 2048, 8192} on Pythia-1B + DATE-LM data selection. Plot the 4-point curve. If quality is flat (no saturation), FM1 narrative needs revision.
- **Core experiment (40 min)**: Full sweep k in {64, 128, 256, 512, 1024, 2048, 4096, 8192} on all 3 DATE-LM tasks. Parallel sweep: RepSim with PCA-reduced representations at d_eff in {64, 128, 256, 512, 1024, 2048}.
- **Cross-model validation (20 min)**: Repeat the k-sweep on Pythia-160M (d=768) and Pythia-410M (d=1024). Check if saturation point scales with d.

**Computational Cost**: ~2 GPU-hours (parallelizable: 4 sweeps on 4 GPUs simultaneously)
**Success Probability**: 75% -- the saturation behavior is very likely (TRAK's own paper shows diminishing returns with k), but the precise match k_sat ~ d is the key prediction
**Failure Modes**:
- (a) Quality keeps improving up to k=8192 without saturation -- this would suggest the useful subspace is larger than d, weakening the FM1-as-dimension-reduction story. Mitigation: check if the improvement rate matches logarithmic (JL prediction) vs. sublinear (our FM1 prediction)
- (b) Saturation happens at k << d (e.g., k ~ 256 for d ~ 2048) -- this would actually strengthen FM1 even further: the useful signal is even lower-dimensional than representation space
- (c) Results vary wildly across DATE-LM tasks -- this would complicate the narrative but is informative: FM1 severity may be task-dependent

**Key References**:
- Park et al. (2303.14186) -- TRAK projection dimension analysis (Figure 3 in their paper shows k-dependence on CIFAR/ImageNet, but not on LLMs)
- Hu et al. (2602.10449) -- Theoretical prediction that sketch dimension must exceed rank(F); our sweep would empirically validate this on LLMs
- LoGra (2405.13954) -- Also reports projection dimension sensitivity; can use LoGra as a second parameter-space method in the sweep

---

## Angle 3: Contrastive Scoring as a Plug-in -- Stress-Testing FM2 Across Methods

### Core Insight (Cross-Method Transfer: Generalize DDA's Fix)

DDA showed that contrastive scoring (debias + denoise) dramatically improves IF-based attribution (AUC from ~60% to 91.64%). But DDA only applied this to their specific IF variant. The key pragmatic question is: **does contrastive scoring generalize as a plug-in improvement across ALL TDA methods?**

If yes, this provides direct evidence that FM2 is a universal failure mode (not specific to DDA's IF formulation) and that contrastive scoring is a universal fix. If no (some methods don't benefit), this reveals which methods already implicitly handle FM2 -- informing the phi^T * psi analysis.

**Concrete Experimental Design**:

Apply contrastive scoring to each method in the phi^T * psi framework:

| Method | Standard Score s(z_test, z_train) | Contrastive Score c(z_test, z_train) |
|--------|----------------------------------|--------------------------------------|
| TRAK | g(z_test)^T * g(z_train) | [g(z_test) - g_bar]^T * [g(z_train) - g_bar] |
| RepSim | h(z_test)^T * h(z_train) | [h(z_test) - h_bar]^T * [h(z_train) - h_bar] |
| RepT | (h + alpha*grad_h)(z_test)^T * ... | Same with mean subtraction |
| LoGra | g_logra(z_test)^T * g_logra(z_train) | Same with mean subtraction |

Where g_bar = E_{z' ~ D_neg}[g(z')] and h_bar = E_{z' ~ D_neg}[h(z')] are the mean gradient/representation over negative samples.

**Three Variants of Contrastive Scoring** (to understand what works and why):
1. **Mean subtraction**: phi = x - E[x] (removes DC component, simplest)
2. **Reference subtraction**: phi = x - x_ref (DDA-style, using a specific reference point)
3. **Whitening**: phi = Sigma^{-1/2} * (x - mu) (full decorrelation, most aggressive)

**What This Reveals for the Paper**:
- If mean subtraction alone recovers most of DDA's gains, FM2 is primarily about the DC component (pre-training bias)
- If whitening is needed, FM2 involves correlated noise beyond just a mean shift
- If contrastive scoring helps parameter-space methods more than representation-space methods, this quantifies the FM2 severity difference between the two spaces

**Experimental Plan**:
- **Pilot (10 min)**: On Pythia-1B + DATE-LM toxicity filtering (where FM2 should be strongest -- pre-training knowledge about toxicity is a strong confounder). Compare TRAK and RepSim with/without mean subtraction.
- **Core experiment (40 min)**: All 4 methods x 3 contrastive variants x 3 DATE-LM tasks = 36 cells. Each cell takes ~1 min (representation extraction is cached, only scoring changes). Use cached gradients/representations from Angle 1.
- **Analysis (15 min)**: Compute the "FM2 severity index" = (contrastive_score - standard_score) / standard_score for each method x task. If this index is consistently larger for parameter-space methods, FM2 hits them harder.

**Computational Cost**: ~1 GPU-hour (dominated by initial gradient/representation extraction; the scoring variants are cheap)
**Success Probability**: 80% -- DDA already proved contrastive scoring works for IF; the question is whether the improvement transfers, which is very likely for mean subtraction at minimum
**Failure Modes**:
- (a) Contrastive scoring hurts RepSim -- possible if cosine similarity already normalizes away the DC component, and mean subtraction distorts the similarity structure. Mitigation: always report cosine similarity (which has implicit L2 normalization) separately from dot product
- (b) The improvement is tiny (< 3pp) across all representation methods -- this would weaken the FM2 narrative for representation space. But it would still be a positive finding: representation methods are inherently robust to FM2, which is exactly the claim
- (c) Results are inconsistent across DATE-LM tasks -- FM2 severity genuinely varies by task (factual attribution may have less pre-training contamination than toxicity filtering). This would be informative, not a failure

**Key References**:
- DDA (2410.01285) -- Original contrastive scoring; ablation showing debias contributes 55pp vs. denoise 9pp
- IF-GUIDE (2506.01790) -- Token-level IF attribution for detoxification; their finding that "standard influence functions are ineffective" aligns with FM2 diagnosis
- Quanda toolkit (2410.07158) -- Provides unified TDA evaluation interface; could streamline the 36-cell comparison

---

## Synthesis: Engineering-Driven Research Strategy

### Execution Order (Critical Path)

```
Week 0, Day 1-2: Setup (4 hours)
  - Fork DATE-LM repo, verify evaluation pipeline
  - Download Pythia-1B checkpoint, verify inference
  - Implement RepSim baseline, verify against Li et al. numbers

Week 0, Day 3-4: Angle 1 Core (4 hours)
  - Run 2x2 ablation on all 3 tasks
  - Generate main result table
  - DECISION POINT: If interaction term > 30%, revisit FM1/FM2 orthogonality claim

Week 1, Day 1-2: Angle 2 Sweep (3 hours)
  - Dimension sweep for TRAK and RepSim
  - Cross-model validation on Pythia-{160M, 410M}
  - Generate dimension-quality curve figure

Week 1, Day 3-4: Angle 3 Contrastive (3 hours)
  - 36-cell contrastive scoring matrix
  - FM2 severity index computation
  - Generate heatmap figure

Week 1, Day 5: Unification (2 hours)
  - Populate phi^T * psi table for all 5 methods
  - Verify framework mathematically (each method as special case)
  - Write framework section draft
```

Total compute: ~8 GPU-hours across ~2 weeks. Well within 4x RTX 4090 budget.

### How Each Angle Serves the Paper

| Angle | Paper Section | If Succeeds | If Fails |
|-------|--------------|-------------|----------|
| 1. 2x2 Ablation | **Table 1** (main result) | Clean evidence for FM1 + FM2 orthogonality | Paper pivots to "representation methods are better, here's why" without orthogonality claim |
| 2. Dimension Sweep | **Figure 2** (FM1 evidence) | Direct measurement of signal dilution; k_sat ~ d is a crisp result | Saturation curve still informative; just can't claim FM1 = "exactly d dimensions" |
| 3. Contrastive Plug-in | **Table 2** (FM2 evidence) + **Figure 3** (FM2 severity heatmap) | Universal FM2 diagnosis with quantitative severity index | Some methods immune to FM2; still informative for method selection |

### Key Differences from Innovator's Proposal

| Aspect | Innovator | Pragmatist |
|--------|-----------|------------|
| FM1 diagnosis | Eigenspectrum analysis (gradient covariance) | Dimension sweep (run TRAK at different k) |
| FM2 diagnosis | Difference-in-Differences (econometrics import) | Contrastive scoring as plug-in (generalize DDA) |
| Theory depth | IB lens (information theory) | phi^T * psi instantiation table (engineering catalog) |
| Compute requirement | ~6 GPU-hours, needs eigenvalue decomposition | ~8 GPU-hours, uses only standard TRAK/RepSim |
| Implementation complexity | High (Lanczos eigendecomposition, MINE MI estimation) | Low (parameter sweeps, mean subtraction) |
| Risk profile | High-ceiling, high-risk (45-70% success) | Lower-ceiling, low-risk (75-85% success) |

**My honest assessment**: The Innovator's eigenspectrum analysis (Angle 1) is more intellectually exciting but also more likely to produce noisy results (eigenvalue estimation in high dimensions is fraught). My dimension sweep achieves 70% of the same insight with 20% of the implementation effort. If time permits, the eigenspectrum analysis can be attempted as a follow-up.

### Defensive Engineering Checklist

Before starting any experiment:

- [ ] Verify Pythia-1B inference produces expected perplexity on DATE-LM validation set
- [ ] Confirm TRAK gradient computation fits in single RTX 4090 memory (24GB) with gradient checkpointing
- [ ] Verify DATE-LM evaluation metrics match their published leaderboard numbers when using their provided method outputs
- [ ] Pin all dependency versions (transformers, trak, torch) in requirements.txt
- [ ] Set up automated result logging to `exp/results/` with timestamp and git hash
- [ ] Pre-compute and cache all representations/gradients once, reuse across all angles

---

## Risk Assessment

### What Could Kill Each Angle

1. **2x2 Ablation**: DATE-LM's training set size may make TRAK infeasible at full scale on Pythia-1B. The paper reports using multiple model sizes; if the provided checkpoints are only for 7B+ models, we need to fine-tune our own Pythia-1B. *Mitigation*: DATE-LM's HuggingFace page (huggingface.co/DataAttributionEval) should have checkpoints -- verify on Day 1.

2. **Dimension Sweep**: If TRAK's random projection uses a fixed seed and the quality-vs-k curve is noisy (high variance across seeds), the saturation point may not be identifiable. *Mitigation*: Run 5 random seeds at each k and report mean +/- std. This multiplies compute by 5x but k=128 takes only ~1 min, so total is still manageable.

3. **Contrastive Plug-in**: Computing E[g(z')] over negative samples requires choosing negative set size. Too few negatives = noisy estimate; too many = expensive. *Mitigation*: Use 1000 random negatives as default (following DDA), but sweep {100, 500, 1000, 5000} to check sensitivity.

### What if ALL Three Produce Ambiguous Results?

The phi^T * psi unification framework is a theoretical contribution that stands independently of any specific empirical result. Even with ambiguous 2x2 results, the framework paper is publishable if the mathematical unification is clean. Minimum viable paper: "We show that 5 seemingly different representation-space TDA methods are all instances of a single bilinear framework, and benchmark them on DATE-LM." This is already a useful contribution -- a survey + unification + benchmark paper.

### The One Thing I'm Most Worried About

**DATE-LM benchmark operationalization**. This is a NeurIPS 2025 benchmark, meaning the codebase is relatively new (~6 months old). New benchmarks often have undocumented quirks, missing edge cases, and version incompatibilities. I would allocate a full day (Day 1-2 in the plan) to just getting the benchmark pipeline running end-to-end with a toy method before attempting anything novel. If this takes longer than 2 days, consider falling back to Li et al.'s simpler evaluation protocol as a Plan B.

---

## Supplementary Literature Found During Search

1. **Vitel & Chhabra 2025 (2511.04715)** -- "First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies". Demonstrates middle attention layers are optimal for influence estimation, proposes vote-based aggregation across layers. *Directly relevant*: our RepSim and RepT implementations need to evaluate multiple layers, not just use the last layer by default.

2. **Li et al. 2025 (2512.09103)** -- "Natural Geometry of Robust Data Attribution". Shows TRAK scores are "geometrically fragile" under Euclidean perturbation (0% certification), proposes Natural Wasserstein metric. *Relevant for framing*: parameter-space attribution is not just inaccurate but also unstable, further motivating representation-space alternatives.

3. **IF-GUIDE (2506.01790)** -- Coalson et al. demonstrate token-level influence for detoxification, finding standard IF ineffective for identifying harmful training data. *Supports FM2 narrative*: standard IF fails specifically because pre-training knowledge dominates the influence signal.

4. **Quanda (2410.07158)** -- Python toolkit for TDA evaluation with unified interface. *Practical resource*: could streamline our multi-method comparison if it supports LLM-scale experiments.

5. **DATE-LM GitHub**: github.com/DataAttributionEval/DATE-LM with HuggingFace checkpoints at huggingface.co/DataAttributionEval. *Critical resource*: must be the starting point for all experiments.

---

## Summary Table

| Angle | Type | Hypothesis | Pilot Time | Full Time | P(success) | Impact if Succeeds |
|-------|------|-----------|------------|-----------|------------|-------------------|
| 1. Hardened 2x2 | Improve existing (engineering) | FM1 and FM2 are orthogonal, additive defects | 15 min | 1.5h | 85% | High -- main result table for the paper |
| 2. Dimension Sweep | New diagnostic method | k_sat ~ d proves signal dilution = dimension mismatch | 10 min | 1h | 75% | Medium-High -- crisp visual evidence for FM1 |
| 3. Contrastive Plug-in | Cross-method transfer | Contrastive scoring universally fixes FM2 | 10 min | 1h | 80% | Medium -- quantitative FM2 severity index |

**Recommended Priority**: Angle 1 >> Angle 3 > Angle 2

Rationale: Angle 1 is the paper's main result and must be bulletproof. Angle 3 directly validates the FM2 claim with minimal extra compute (reuses cached data from Angle 1). Angle 2 provides the strongest visual evidence for FM1 but can be deferred if time is tight. All three angles together require ~8 GPU-hours, well within the 4x RTX 4090 x 2-week budget.
