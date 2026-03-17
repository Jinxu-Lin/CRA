# Pragmatist Perspective: CRA Research Proposal

## Executive Summary

The CRA proposal is scientifically compelling but carries serious engineering risks that could derail timelines. Below I propose three practical angles that maximize the probability of producing publishable results within the compute budget (4x RTX 4090, ~2-3 weeks), each designed so that even partial success yields a clear contribution. My overarching recommendation: **strip the project to its empirical core first, build the theory around what the data actually shows, and keep every experiment under 1 GPU-hour.**

---

## Angle 1: Minimal Viable Factorial -- The 2x2 Ablation on DATE-LM with Off-the-Shelf Code

### The Pragmatic Insight

The core novelty of CRA is the 2x2 factorial {parameter-space, representation-space} x {standard, contrastive scoring}. This experiment has *never been done* on a standard LLM benchmark. If executed cleanly on DATE-LM, it is publishable regardless of the theoretical framework. The bilinear unification and signal processing theory are valuable but secondary -- they can be retrofitted to whatever the data shows.

### Why This Should Be Priority #1

- **DATE-LM** provides pre-trained checkpoints, standardized evaluation protocols, and a public leaderboard ([OpenReview](https://openreview.net/forum?id=e2cD5xuHix), [GitHub](https://github.com/DataAttributionEval/DATE-LM)). This eliminates the most dangerous engineering risk: custom training pipelines.
- **TRAK** has a mature PyTorch implementation ([GitHub: MadryLab/trak](https://github.com/MadryLab/trak)) with CUDA-optimized JL projection kernels. Single-GPU attribution for ~10K training examples on a 1B model is tractable.
- **RepSim** is trivially implementable: extract last-layer hidden states, compute cosine similarity. No external library needed.
- **Contrastive scoring** (mean subtraction) is a 3-line operation on pre-computed attribution scores.

### Concrete Implementation Plan

**Step 1: Environment setup (2 hours)**
```
- Clone DATE-LM repo, verify checkpoint loading on RTX 4090
- Install TRAK (pip install traker[fast]) with CUDA kernels
- Verify RepSim baseline: extract hidden states for 500 test + 10K train examples
```

**Step 2: Representation extraction (30 min per task x 3 tasks = 1.5 hours)**
```
- For each DATE-LM task (data selection, toxicity filtering, factual attribution):
  - Extract last-layer representations for all train + test examples
  - Save as .npy files (~80 MB per task at d=2048)
  - Compute RepSim scores (cosine similarity matrix)
```

**Step 3: TRAK attribution (45 min per task x 3 tasks = 2.25 hours)**
```
- Use TRAK with k=2048 (matching representation dimension d)
  - 1 model checkpoint, standard random projection
  - DATE-LM provides the fine-tuned checkpoints
- Compute TRAK scores for same train/test pairs
```

**Step 4: Contrastive scoring (15 min total)**
```
- For both RepSim and TRAK scores:
  - Compute mean attribution vector across all training examples
  - Subtract mean to get contrastive scores
  - This is DDA's "debias" step, stripped to its essence
```

**Step 5: Evaluation (30 min)**
```
- Compute DATE-LM metrics: LDS (data selection), auPRC (toxicity), P@K (factual)
- Run 2x2 ANOVA with bootstrap CI (B=1000) for each task
- Include BM25 baseline (DATE-LM provides this) and k-NN control
```

**Total wall-clock time: ~6 hours on 1x RTX 4090.** Can be parallelized to ~2 hours on 4 GPUs.

### Engineering Risks and Mitigations

| Risk | P(occur) | Impact | Mitigation |
|------|----------|--------|------------|
| DATE-LM checkpoint won't load on 4090 (VRAM) | 20% | High | Use Pythia-1B (2.6 GB in fp16, fits in 24 GB with room for activations) |
| TRAK CUDA kernel compilation fails | 15% | Medium | Fall back to `traker` CPU/basic GPU mode; slower but functional |
| TRAK OOM on 10K training examples | 25% | Medium | Subsample to 5K; or compute in batches (TRAK supports chunked computation) |
| RepSim scores saturate (all near 1.0) | 10% | High | Use CKA or centered cosine similarity; try layer -2 instead of last layer |
| Contrastive scoring hurts performance | 15% | Low | This is a valid finding (falsifies H2); report honestly |

### Computational Cost

- GPU memory: ~16 GB peak (Pythia-1B in fp16 + batch of representations)
- Storage: ~500 MB for all representations and score matrices
- Wall-clock: 6 hours total, each sub-experiment under 1 hour

### Success Probability: 90%

This angle has the highest success probability because it uses only off-the-shelf code on a standard benchmark. Even if the results are "boring" (e.g., RepSim always wins, contrastive scoring helps uniformly), it is the first systematic 2x2 comparison on DATE-LM and fills a clear gap identified in the DATE-LM paper itself ("no single method dominates").

### Failure Mode

The only true failure: DATE-LM evaluation code is broken or undocumented, forcing us to re-implement the evaluation protocol from scratch. Mitigation: the paper describes the protocol in detail, and it uses standard metrics (LDS, auPRC, P@K) available in scikit-learn.

---

## Angle 2: Gradient Eigenspectrum as a Diagnostic Tool -- Measuring FM1 Directly

### The Pragmatic Insight

H4 (r_eff ~ d) is the "smoking gun" for FM1. If we can directly show that the gradient covariance has effective rank ~d (representation dimension) rather than ~B (parameter count), the entire FM1 narrative becomes data-driven rather than theoretical. The key engineering insight: **this measurement is tractable on Pythia-70M (d=512, B=70M) using Lanczos iteration, and does NOT require computing full gradients.**

### Why This Is Feasible

- **Pythia-70M** fits entirely in 4 GB GPU memory. Full gradient computation for a single example costs ~280 MB (70M params x 4 bytes).
- **Lanczos iteration** for top-k eigenvalues of Sigma_g requires only matrix-vector products (gradient x vector), not explicit covariance matrix construction. The `scipy.sparse.linalg.eigsh` or PyTorch's `torch.lobpcg` can compute top-500 eigenvalues with ~50 Lanczos iterations.
- **Effective rank** from eigenspectrum: r_eff(95%) = min k such that sum(lambda_1..k) / sum(lambda_all) >= 0.95. If r_eff ~ 512 (= d for Pythia-70M), FM1 is confirmed.

### Concrete Implementation Plan

**Step 1: Gradient extraction (30 min)**
```
- Fine-tune Pythia-70M on DATE-LM data selection task (or use pre-existing checkpoint)
- For 1000 training examples, compute per-example gradients using backprop
- Store as [1000 x 70M] matrix (or use implicit matrix-vector products)
```

**Step 2: Lanczos eigendecomposition (20 min)**
```
- Compute top-500 eigenvalues of gradient covariance using Lanczos
- Plot eigenvalue decay curve (log scale)
- Compute r_eff(90%), r_eff(95%), r_eff(99%)
```

**Step 3: Representation covariance comparison (10 min)**
```
- Compute eigenvalues of representation covariance (d=512, cheap)
- Compare condition numbers: gradient (expected >10^4) vs representation (expected <100)
- This simultaneously tests H9 (isotropy)
```

**Step 4: Cross-validation on Pythia-160M (30 min, optional)**
```
- Repeat eigenspectrum on Pythia-160M (d=768)
- Check if r_eff scales with d, not B
```

### Engineering Risks

| Risk | P(occur) | Impact | Mitigation |
|------|----------|--------|------------|
| Lanczos convergence too slow | 15% | Medium | Use randomized SVD (sklearn.utils.extmath.randomized_svd) instead; less precise but faster |
| Per-example gradients OOM on 70M model | 10% | Low | Use gradient accumulation; compute matrix-vector products without storing full gradient matrix |
| Eigenspectrum shows r_eff >> 2d | 20% | High | FM1 narrative weakens; but report as evidence against signal dilution hypothesis |
| r_eff(95%) is ambiguous (depends heavily on threshold) | 25% | Medium | Report multiple thresholds (80%, 90%, 95%, 99%); plot full spectrum for visual inspection |

### Computational Cost

- GPU memory: ~8 GB (Pythia-70M + gradients for 1 batch)
- Storage: ~20 MB (eigenvalues only, not eigenvectors)
- Wall-clock: 1 hour total on single RTX 4090

### Success Probability: 65%

The measurement itself will succeed (eigenspectrum computation is well-understood). The question is whether r_eff falls in the predicted [0.5d, 2d] range. Prior work (Park et al., TRAK paper) implicitly observes that random projection dimension matters, but nobody has directly measured the effective rank. Even if r_eff is outside the predicted range, the measurement itself is novel and informative.

### Key References for Implementation

- `torch.lobpcg` for large-scale eigenvalue computation in PyTorch
- `functorch.vmap` + `functorch.grad` for efficient per-example gradient computation
- Pythia checkpoints: [HuggingFace EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m) with 154 intermediate checkpoints

---

## Angle 3: TRAK Dimension Sweep as Empirical FM1 Test

### The Pragmatic Insight

H5 (TRAK dimension saturation) is the most practically impactful experiment. If TRAK performance saturates at k ~ d (representation dimension), this directly tells practitioners: "don't waste compute on k > 2d projections." It also provides the most accessible FM1 evidence -- no eigenspectrum computation needed, just run TRAK seven times with different k values.

### Why This Is Both Novel and Easy

- **TRAK's `traker` library** accepts projection dimension as a hyperparameter. Sweeping k is literally changing one integer.
- **Nobody has published a systematic dimension sweep** on LLMs. The TRAK paper shows sweeps on CIFAR-10 and ImageNet, but not on language models where d << B is most extreme.
- **The saturation curve shape is diagnostic**: if it saturates at k ~ d, FM1 is confirmed; if it keeps improving log-linearly, FM1 is refuted; if it saturates at k << d, something more interesting is happening.

### Concrete Implementation Plan

**Step 1: TRAK sweep (45 min per k x 7 values = 5.25 hours; parallelizable to 1.5 hours on 4 GPUs)**
```
- Model: Pythia-1B (d=2048) on DATE-LM data selection task
- k values: {64, 128, 256, 512, 1024, 2048, 4096}
- For each k:
  - Run TRAK with random projection to R^k
  - Compute LDS on DATE-LM test set
  - Record wall-clock time and peak GPU memory
```

**Step 2: PCA-TRAK comparison (45 min, optional but high-value)**
```
- Compute top-2048 eigenvectors of gradient covariance (from Angle 2)
- Run TRAK with PCA projection (deterministic, onto top-k eigenvectors)
  for k in {64, 128, 256, 512, 1024, 2048}
- If PCA-TRAK saturates at smaller k than random-TRAK, this is the "smoking gun":
  directed projection captures signal more efficiently
```

**Step 3: Visualization and analysis (30 min)**
```
- Plot LDS vs log(k) for random-TRAK and PCA-TRAK
- Mark d=2048 on x-axis
- Compute "knee point" using Kneedle algorithm
- Estimate r_eff indirectly from saturation point
```

### Engineering Risks

| Risk | P(occur) | Impact | Mitigation |
|------|----------|--------|------------|
| TRAK at k=4096 OOM on RTX 4090 | 30% | Medium | Use gradient checkpointing; or cap at k=2048 and extrapolate |
| LDS computation is noisy (large variance) | 20% | Medium | Run each k with 3 random seeds; report mean +/- std |
| No clear saturation (keeps improving) | 20% | High | This falsifies H5 but is informative; report as negative result |
| PCA eigenvector computation infeasible at Pythia-1B scale | 40% | Medium | Skip PCA-TRAK; random-TRAK sweep alone is still novel |

### Computational Cost

- GPU memory: ~20 GB peak at k=4096 (Pythia-1B + projection matrices)
- Storage: ~100 MB per k (score matrices)
- Wall-clock: 5.25 hours sequential, ~1.5 hours with 4-GPU parallelism

### Success Probability: 75%

High confidence the experiment runs successfully (TRAK is mature software). Moderate confidence the saturation curve shows a clear knee at k ~ d. The Wang et al. (2505.24261) hyperparameter sensitivity study confirms that TRAK is sensitive to projection dimension -- our sweep systematically characterizes this sensitivity for the first time on LLMs.

---

## Synthesis: Practical Execution Roadmap

### Phase 0: Infrastructure Validation (Day 0, 2 hours)
```
1. Verify DATE-LM checkpoint loading on RTX 4090 (Pythia-1B)
2. Verify TRAK installation with CUDA kernels
3. Run RepSim on 100 examples as smoke test
4. Confirm DATE-LM evaluation script works end-to-end
```
**Gate**: If any step fails, debug for max 2 hours, then fall back to Pythia-410M or smaller model.

### Phase 1: Core 2x2 Factorial (Day 1, 6 hours) -- Angle 1
```
- Execute the full 2x2 ablation on all 3 DATE-LM tasks
- Compute ANOVA + bootstrap CI
- This alone is sufficient for a workshop paper or short paper
```
**Gate**: If RepSim < TRAK by >5pp, stop and investigate before proceeding.

### Phase 2: Mechanistic Evidence (Day 2, 2 hours) -- Angle 2
```
- Eigenspectrum on Pythia-70M (1 hour)
- Representation covariance comparison (30 min)
- Analysis and visualization (30 min)
```
**Gate**: If r_eff > 10d, reconsider FM1 narrative.

### Phase 3: Dimension Sweep (Day 2-3, 5 hours) -- Angle 3
```
- TRAK sweep on Pythia-1B (parallelized across 4 GPUs)
- Optional PCA-TRAK comparison
```

### Phase 4: Framework Validation (Day 3-4, 4 hours)
```
- Whitened attribution (Ledoit-Wolf regularized) on RepSim
  - scikit-learn's LedoitWolf estimator is battle-tested and O(n*d^2)
  - For d=2048 and n=10K, covariance estimation takes <1 min
  - Apply whitening: scores = phi^T @ Sigma_inv @ psi
- Multi-method comparison: add RepT, AirRep if code is available
  - RepT: https://github.com/plumprc/RepT (public)
  - AirRep: https://github.com/sunnweiwei/AirRep (public, NeurIPS 2025)
```

### Total Budget

| Phase | GPU-hours | Wall-clock (4 GPUs) | Critical? |
|-------|-----------|---------------------|-----------|
| Phase 0 | 2 | 2h | Yes -- gate for everything |
| Phase 1 | 6 | 2h | Yes -- core contribution |
| Phase 2 | 1 | 1h | Yes -- mechanistic evidence |
| Phase 3 | 5.25 | 1.5h | High priority |
| Phase 4 | 4 | 2h | Nice to have |
| **Total** | **18.25** | **8.5h** | |

This is well within the 2-3 week budget. Remaining time is for:
- Debugging and re-runs (~2x buffer)
- Additional methods (DDA, LoGra, Concept Influence)
- Theoretical write-up
- Paper drafting

---

## Critical Engineering Warnings

### Warning 1: DATE-LM Task Heterogeneity

DATE-LM's three tasks have *very different characteristics*:
- **Data selection** (LDS): Counterfactual metric, requires retraining. This is expensive -- verify DATE-LM provides pre-computed counterfactual scores, or we need to retrain ~100 models.
- **Toxicity filtering** (auPRC): Binary classification of training data. Cheapest to evaluate.
- **Factual attribution** (P@K): Retrieval task. BM25 may be surprisingly competitive here (DATE-LM found this).

**Recommendation**: Start with toxicity filtering (cheapest, cleanest signal), then data selection, then factual attribution. If time is tight, two tasks are sufficient for a NeurIPS submission.

### Warning 2: Contrastive Scoring Is Not Trivial

DDA's "debias" step is more nuanced than simple mean subtraction:
- They subtract the *task-conditional* mean, not the global mean
- They also apply a "denoise" step (matrix regularization)
- Their 55pp improvement from debias may not replicate with naive mean subtraction

**Recommendation**: Implement both (a) global mean subtraction and (b) task-conditional mean subtraction. If (a) fails, (b) is the fallback. If both fail, this is actually interesting (FM2 is more complex than a simple bias term).

### Warning 3: Whitened Attribution May Fail Silently

Sigma_noise^{-1} estimation is fragile when d > n (more features than samples). For d=2048 and typical DATE-LM training set sizes:
- If n < 2048: raw inverse is singular; MUST use Ledoit-Wolf or ridge regularization
- Even with regularization, the condition number of Sigma_noise may be too high for meaningful whitening

**Recommendation**: Always compare whitened attribution against ridge-regularized attribution (M = (Sigma + lambda*I)^{-1}) with lambda tuned on held-out data. Report regularization strength alongside performance.

### Warning 4: Pythia-1B Is Not Typical of Production LLMs

Pythia-1B has d=2048, which is relatively small. Production LLMs (Llama-2-7B: d=4096, Llama-2-70B: d=8192) have larger d but also much larger B. The d/B ratio changes, which could affect FM1 severity.

**Recommendation**: Run the core 2x2 on Pythia-1B as the main experiment. If time permits, replicate one task on a 7B model (Llama-2-7B or Qwen-2-7B) to check scaling. DATE-LM supports multiple model architectures.

---

## Risk Assessment Summary

| Angle | Novelty | Engineering Risk | P(success) | Time (GPU-h) | Fallback Value |
|-------|---------|-----------------|------------|--------------|---------------|
| 1. 2x2 Factorial | Medium | Low | 90% | 6 | First systematic comparison on DATE-LM -- publishable as-is |
| 2. Eigenspectrum | Medium-High | Medium | 65% | 1 | Direct measurement of gradient rank -- informative even if r_eff != d |
| 3. Dimension Sweep | Medium | Low | 75% | 5.25 | Practical guide for TRAK hyperparameter selection on LLMs |

**Expected value**: At least two angles succeed with P = 1 - (0.10)(0.35)(0.25) = 99.1%. The paper has strong empirical foundations with very high probability.

**My strong recommendation**: Execute Angles 1-3 in order before touching any theory. The bilinear framework and matched filter theory are only as valuable as the empirical evidence supporting them. If the 2x2 factorial shows clear FM1/FM2 separation, the theory writes itself. If it doesn't, no amount of elegant math will save the paper.

---

## Key Resources and Implementation References

### Off-the-Shelf Code
- **DATE-LM**: [GitHub](https://github.com/DataAttributionEval/DATE-LM) -- benchmark, checkpoints, evaluation scripts
- **TRAK**: [GitHub (MadryLab/trak)](https://github.com/MadryLab/trak) -- CUDA-optimized random projection attribution
- **AirRep**: [GitHub (sunnweiwei/AirRep)](https://github.com/sunnweiwei/AirRep) -- learned representation attribution (NeurIPS 2025)
- **RepT**: [GitHub (plumprc/RepT)](https://github.com/plumprc/RepT) -- representation gradient tracking
- **Pythia**: [HuggingFace (EleutherAI)](https://huggingface.co/EleutherAI/pythia-1b) -- 154 checkpoints per model size

### Libraries
- **Ledoit-Wolf**: `sklearn.covariance.LedoitWolf` -- battle-tested shrinkage estimator, O(n*d^2)
- **Per-example gradients**: `torch.func.vmap` + `torch.func.grad` (PyTorch 2.0+)
- **Eigendecomposition**: `torch.lobpcg` for large-scale; `scipy.sparse.linalg.eigsh` for Lanczos
- **Bootstrap CI**: `scipy.stats.bootstrap` or manual implementation

### Key Papers Informing This Perspective
- Park et al. (2303.14186): TRAK -- establishes random projection methodology and dimension sensitivity
- Ma & Nyarko (2511.19803): Forward-only attribution -- complementary efficiency approach, validates scalability concerns
- Wang et al. (2505.24261): Hyperparameter sensitivity in TDA -- confirms projection dimension matters
- Daunce / Pan et al. (2505.23223): Uncertainty-based attribution -- alternative scalable approach if gradient methods fail
- DATE-LM / Jiao et al. (2507.09424): Benchmark design -- "no single method dominates" motivates systematic comparison
- Li et al. (2409.19998): RepSim >> IF on LLMs -- the core observation that CRA explains
