# Pragmatist Perspective: Cross-Task Influence in Multi-Task VLA

## Engineering Reality Check

The innovator's three angles (Temporal Influence Tomography, Coalition Influence Probing, RepFinger) are intellectually exciting, but as a practitioner I need to ask: **what can we actually build and validate on a single A6000 GPU within 1-hour experiment budgets, using small models, and produce reliable enough signals to write a paper?**

Below I propose three practical research directions that prioritize computational feasibility, reproducibility, and clear experimental protocols. Each is designed to produce publishable results even if partial hypotheses fail.

---

## Angle 1: LOO-Proxy Influence Matrix via Cheap Gradient Features (Improve Existing)

### Core Insight
Full influence functions (Hessian-based) are too expensive for iterative experimentation. LESS (arXiv 2402.04333) showed that random-projected gradients + cosine similarity is a surprisingly effective proxy for influence in LLMs. The question is: **does this cheap proxy work for multi-task policy learning?** Nobody has validated it in the VLA/robot setting.

### Method: Gradient-Projected Task Affinity (GPTA)
1. **Train a small base policy** on all tasks jointly (ResNet-18 encoder + 2-layer MLP action head, or Qwen-0.5B with LoRA rank=8).
2. **Collect per-sample gradient features**: For each training sample $z$, compute $\nabla_\theta \ell(z; \theta^*)$ and project via a fixed random matrix $R \in \mathbb{R}^{d \times k}$ (k=256-512). This is the LESS/LoGra trick — O(k) storage per sample.
3. **Task-level influence matrix**: For each task pair $(i, j)$, compute $M_{ij} = \frac{1}{|\mathcal{D}_j|} \sum_{z' \in \mathcal{D}_j} \max_{z \in \mathcal{D}_i} \cos(Rg_z, Rg_{z'})$. This measures how well task $i$'s training data "covers" task $j$'s gradient directions.
4. **Validate against ground truth**: Run actual LOO retraining (drop task $i$, measure task $j$ performance) for all $T(T-1)$ pairs. Compute Spearman correlation between $M_{ij}$ and actual LOO effect.
5. **Data mixing application**: Use $M_{ij}$ to define a simple mixing rule: $w_i^{(j)} \propto \max(0, M_{ij})$ for task $j$'s training. Compare vs. uniform mixing and Re-Mix (DRO).

### Why This Is Practical
- **Gradient collection**: Single backward pass per sample. For 500 samples × 10 tasks = 5000 samples on a small model, this takes ~10 min.
- **Random projection**: O(nk) total, fits in memory even for k=512.
- **LOO validation**: With a small model (training ~5 min per run), 10-task LOO costs 10 × 5 min = 50 min. Parallelizable with careful GPU scheduling.
- **No Hessian computation**: Avoids the main computational bottleneck of classical influence functions.

### Experimental Plan (<=1hr per experiment task)
- **Dataset**: LIBERO-10 (10 tasks, ~50 demos each, readily available)
- **Model**: ResNet-18 + MLP action head (~5M params) — trains in ~5 min on single A6000
- **Pilot** (15 min): Train base model, collect gradient features for 2 tasks, sanity-check cosine similarity distribution
- **Main experiment** (45 min): Full 10×10 influence matrix + 3 LOO validation runs
- **Samples**: Use all 50 demos per task (500 total) — config says pilot_samples=100, so use 100 for pilot, full set for main

### Computational Cost
| Step | Time | GPU Memory |
|------|------|-----------|
| Base model training (10 tasks) | 5 min | ~4 GB |
| Gradient feature collection (5000 samples) | 10 min | ~6 GB |
| Influence matrix computation | 2 min | CPU only |
| LOO validation (3 pairs) | 15 min | ~4 GB |
| **Total pilot** | **~30 min** | **~6 GB peak** |

### Success Probability: 70%
High because the LESS proxy is well-validated for LLMs and the method is simple. Main risk: policy loss landscapes may be qualitatively different from language modeling loss.

### Failure Modes
- Gradient features may be too noisy for small robot datasets (50 demos per task)
- Cosine similarity may not capture directional conflicts in action space
- LOO effect may be too small to detect with limited data

### Fallback
If gradient proxy fails, switch to representation-based proxy (see Angle 3) — the experimental infrastructure (LOO validation) is reusable.

### Key References
- LESS (arXiv 2402.04333, ICML 2024): gradient projection for efficient influence — [code available](https://github.com/princeton-nlp/LESS)
- LoGra/LogIX (ICLR 2025): efficient gradient logging for TDA — [code available](https://github.com/logix-project/logix)
- ETAP (arXiv 2602.18591, 2026): ensemble task affinity prediction combining gradient-based + learned estimators
- DUET (arXiv 2502.00270, 2025): influence function + Bayesian optimization for data mixture
- DataMIL (arXiv 2505.09603): datamodels for robot data selection — closest prior work in robotics

---

## Angle 2: Task Affinity via Leave-One-Task-Out with Efficient Proxies (Cross-Domain Transfer)

### Core Insight
The MTL task grouping literature (ETAP, DMTG, RMB-CLE) has developed cheap affinity estimation methods, but **none have been applied to robot policy learning**. The key engineering question: which cheap proxy best predicts actual multi-task performance in the VLA setting? This is an empirical horse-race that the field needs.

### Method: Affinity Proxy Benchmark for Robot MTL
Compare 4 cheap affinity estimation methods on the same benchmark:

1. **Gradient cosine** (GradCos): Average cosine similarity of per-task gradient vectors $\cos(\bar{g}_i, \bar{g}_j)$. Cost: 1 forward + backward pass per task.
2. **Loss-based LOO proxy** (LossLOO): Train on all tasks, evaluate task $j$ loss, then remove task $i$ data and retrain. Approximate with early stopping at 20% of full training budget. Cost: $T$ partial retraining runs.
3. **Representation CKA** (RepCKA): Compute CKA similarity between task-conditioned activations at the penultimate layer. Cost: 1 forward pass per task. (This is a simplified version of the innovator's RepFinger.)
4. **GPTA** (from Angle 1): Gradient-projected task affinity.

### Ground Truth
Full LOO retraining: for each of the 45 task pairs in LIBERO-10, train without task $i$ and measure task $j$'s success rate. This is expensive (~50 min × 10 = 8+ hours total) but can be distributed across multiple experiment tasks.

### Why This Is Practical
- Each proxy computation takes <15 min
- Ground truth collection is embarrassingly parallel (10 independent training runs)
- The benchmark structure makes partial results publishable — even comparing 2 proxies is informative
- Directly useful for any future work on robot data mixing

### Experimental Plan
- **Phase 1 — Ground truth** (10 experiment tasks × 5 min each = 50 min total, can run as batch):
  - 10 LOO training runs (drop task $i$, evaluate all other tasks)
  - 1 baseline training (all tasks)
  - Record per-task success rates → build ground-truth influence matrix
- **Phase 2 — Proxy computation** (1 experiment task, ~30 min):
  - Compute all 4 proxies from the all-task model
  - Measure Spearman/Kendall-tau correlation with ground truth
- **Phase 3 — Mixing application** (3 experiment tasks × 15 min):
  - Take the best proxy, derive mixing weights
  - Compare: uniform / proxy-guided / Re-Mix (DRO) / oracle (LOO-based)

### Computational Cost
| Step | Time | Notes |
|------|------|-------|
| Ground truth LOO (10 runs) | 50 min total | Parallelizable to ~5 min with 10 GPUs, but single-GPU = sequential |
| 4 proxy computations | 30 min | Sequential on single GPU |
| 3 mixing experiments | 45 min | Sequential |
| **Total** | **~2 hours** | Split across 3-4 experiment tasks |

### Success Probability: 75%
This is primarily an empirical benchmark — something useful comes out regardless of which proxy wins. Even a negative result ("no cheap proxy works for robot MTL") is publishable.

### Failure Modes
- All proxies may correlate poorly with ground truth (→ finding itself is the contribution)
- LOO effects may be uniformly small in LIBERO-10 (tasks may be similar enough that removing one has negligible impact)
- Small model results may not transfer to larger VLAs

### Key References
- ETAP (arXiv 2602.18591): ensemble task affinity predictor — gradient + learned estimators
- Efficient Task Grouping (arXiv 2412.04413): samplewise optimization landscape for task similarity
- DMTG (arXiv 2407.05082): differentiable multi-task grouping — [code](https://github.com/ethanygao/DMTG)
- RMB-CLE (arXiv 2602.14231): cross-task error clustering for robust MTL
- Principled Task Grouping (arXiv 2402.15328): theoretical framework for transfer gain

---

## Angle 3: Representation Geometry Diagnostic (New, Low-Cost Method)

### Core Insight
The innovator proposed RepFinger (CKA + linear probe direction). I agree this is the highest-value practical tool, but I'd simplify and harden it for reliability:

**Don't compute CKA across all layers — focus on the action-prediction bottleneck layer.** In a VLA, the representation conflict that matters is at the layer where visual features get mapped to action predictions. Earlier layers (vision encoder) are likely shared and non-conflicting; the conflict lives in the policy head.

### Method: Bottleneck Conflict Score (BCS)
1. **Identify the bottleneck**: The last shared layer before task-specific heads (or the LoRA-adapted layers in a VLA).
2. **Extract task-conditioned features**: For each task, run validation data through the model, collect the bottleneck activations $H_i \in \mathbb{R}^{n_i \times d}$.
3. **Compute subspace overlap**: For each task, compute the top-$k$ principal components $U_i \in \mathbb{R}^{d \times k}$. The subspace overlap is $O_{ij} = \|U_i^T U_j\|_F^2 / k$ (normalized Grassmann distance).
4. **Compute readout conflict**: Fit a linear probe $W_i$ from bottleneck features to actions for each task. Conflict score: $C_{ij} = O_{ij} \cdot (1 - |\cos(W_i, W_j)|)$. High overlap + low readout alignment = conflict.
5. **BCS matrix**: $BCS_{ij} = O_{ij} \cdot \text{sign}(\cos(W_i, W_j))$. Positive = positive transfer potential, negative = conflict.

### Simplifications vs. RepFinger
- **Single layer** instead of all layers → 10x faster, easier to interpret
- **PCA subspace overlap** instead of CKA → more interpretable, no kernel choice
- **Focus on action prediction** instead of full representation → directly relevant to policy quality

### Experimental Plan (<=1hr)
- **Step 1** (5 min): Load a pre-trained multi-task model (from Angle 1/2 base training)
- **Step 2** (10 min): Forward pass all validation data, cache bottleneck activations
- **Step 3** (5 min): PCA per task, compute subspace overlap matrix
- **Step 4** (10 min): Fit linear probes per task (least squares, ~seconds each), compute readout cosines
- **Step 5** (5 min): Compute BCS matrix, visualize as heatmap
- **Step 6** (15 min): Validate against LOO ground truth (reuse from Angle 2)

### Computational Cost
| Step | Time | GPU Memory |
|------|------|-----------|
| Feature extraction (forward pass) | 5 min | ~4 GB |
| PCA + overlap | 1 min | CPU only |
| Linear probes (10 tasks) | 2 min | CPU only |
| BCS computation | <1 min | CPU only |
| **Total** | **~10 min** | **~4 GB peak** |

This is **the cheapest diagnostic** in the entire proposal. If it works, it's a plug-and-play tool for any multi-task robot learning pipeline.

### Success Probability: 60%
The theoretical grounding is solid (Hiratani 2405.20236 proved high feature overlap + low readout alignment is catastrophic). Risk is that the single-layer approximation loses important information.

### Failure Modes
- Single-layer focus may miss conflicts that emerge from cross-layer interactions
- PCA may not capture the relevant subspace structure (non-linear manifolds)
- Linear probes may be too simple for multi-modal action spaces (7-DOF + gripper)
- The bottleneck layer may not be well-defined in transformer architectures

### Fallback
If single-layer BCS fails, progressively add layers (2-3 key layers) until signal emerges. If PCA fails, switch to CKA (more robust to non-linearity). The infrastructure is identical.

### Key References
- Hiratani (arXiv 2405.20236): analytical proof that feature similarity + readout divergence = catastrophic transfer
- AirRep (arXiv 2505.18513, NeurIPS 2025): representation-based TDA for single-task data valuation
- CKA (Kornblith et al., ICML 2019): centered kernel alignment — gold standard for representation comparison
- MINT (arXiv 2506.02308): multimodal interaction grouping — uses task interaction type for MTL grouping

---

## Practical Synthesis: Recommended Execution Order

The three angles are **not independent** — they share infrastructure and ground truth. Here's the optimal execution plan for a single-GPU setup:

### Phase 1: Foundation (2 experiment tasks, ~1hr total)
1. **Train base multi-task model** on LIBERO-10 (ResNet-18 + MLP, ~5 min)
2. **Collect gradient features** for all samples (LESS-style projection, ~10 min)
3. **Collect bottleneck activations** for all validation data (~5 min)
4. **Compute all proxies**: GPTA, GradCos, RepCKA, BCS (~15 min)

### Phase 2: Ground Truth (10 experiment tasks, ~50 min total)
5. **LOO retraining**: 10 independent runs, each ~5 min
6. **Build ground-truth influence matrix**: $M^{GT}_{ij}$ = success rate change

### Phase 3: Validation & Application (3 experiment tasks, ~45 min total)
7. **Proxy benchmark**: Correlate all proxies with ground truth
8. **Best-proxy mixing**: Apply top proxy to data mixing optimization
9. **Compare**: uniform / proxy-guided / Re-Mix / oracle

### Phase 4: Mechanism Analysis (2 experiment tasks, ~30 min total)
10. **Identify top-3 negative transfer pairs** from ground truth
11. **Diagnose mechanism** via BCS decomposition: is it subspace overlap + readout conflict?
12. **Targeted intervention**: Remove conflicting data for top negative pair, measure improvement

**Total: ~15 experiment tasks, ~3 hours wall-clock on single A6000.**

### What We Get
- **Contribution 1**: First systematic proxy benchmark for task affinity in robot manipulation
- **Contribution 2**: BCS — a 10-minute, gradient-free diagnostic for multi-task conflicts
- **Contribution 3**: Evidence-based data mixing strategy with measured improvement over uniform/DRO baselines
- **Contribution 4**: Mechanism-level understanding of negative transfer in specific task pairs

---

## Risk Assessment

### Critical Risk: Signal Strength
The biggest practical risk is that LIBERO-10 tasks are too similar to produce strong negative transfer signals. If all LOO effects are within noise margin:

**Plan B — Switch to heterogeneous task set:**
- Create a custom 5-task subset mixing LIBERO tasks with deliberately conflicting demands (e.g., "push left" + "push right" on same object type)
- Or use Meta-World MT10 where tasks are more diverse (reach, push, pick-place, drawer, window, etc.)
- 5 tasks reduce LOO cost to 5 × 5 min = 25 min

### Critical Risk: Small Model ≠ Large VLA
Results on ResNet-18 + MLP may not transfer to 7B VLA models.

**Mitigation:**
- Run one validation experiment on a frozen OpenVLA backbone with LoRA (rank=8): forward pass + BCS computation only (~20 min, no retraining needed)
- If BCS rankings are consistent between small and large models, the diagnostic generalizes
- This is a cheap "transferability check" that doesn't require full LOO on the large model

### What I Would NOT Do
- **Coalition influence probing** (innovator's Angle 2): With only 10 tasks, $\binom{10}{3}=120$ triplets are too many to validate and the signal is likely dominated by pairwise effects. Defer to a later stage when the pairwise story is solid.
- **Full Hessian-based influence functions**: Too expensive for iterative experimentation on single GPU. The gradient proxy is sufficient for the diagnostic and mixing applications.
- **Training large VLAs from scratch**: We need fast iteration cycles. Small models trained in 5 min allow 10+ experiments per hour.

---

## Implementation Checklist

### Required Software
- [ ] LIBERO benchmark: `pip install libero` ([GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO))
- [ ] PyTorch (already available)
- [ ] LogIX/LESS for gradient projection: `pip install logix` ([GitHub](https://github.com/logix-project/logix))
- [ ] scikit-learn for PCA/CKA computation

### Required Data
- [ ] LIBERO-10 dataset (~2 GB download)
- [ ] No external pre-trained weights needed for small model experiments

### Key Metrics
| Metric | Purpose |
|--------|---------|
| Spearman $\rho$ (proxy vs. LOO) | Proxy quality |
| Success rate (task $j$ | mixing strategy) | Mixing effectiveness |
| Subspace overlap $O_{ij}$ | Representation sharing |
| Readout cosine $\cos(W_i, W_j)$ | Action alignment |
| BCS vs. gradient proxy correlation | Method agreement |

### Time Budget Summary
| Phase | Experiment Tasks | Wall-Clock |
|-------|-----------------|------------|
| Foundation | 2 | ~1 hr |
| Ground Truth | 10 | ~50 min |
| Validation | 3 | ~45 min |
| Mechanism | 2 | ~30 min |
| **Total** | **17** | **~3 hr** |

This is achievable within 1-2 iteration cycles of the Sibyl pipeline.
