# Empiricist Perspective: Cross-Task Influence in Multi-Task VLA

## Evaluation Through an Experimentalist's Lens

The preceding perspectives offer creative methods (innovator), practical engineering (pragmatist), rigorous theory (theorist), and healthy skepticism (contrarian). What none of them provides is an **airtight experimental protocol** -- the kind where every comparison is controlled, every confounder is identified, every success criterion is pre-registered, and every claimed effect has a falsification test. This perspective fills that gap.

I approach the topic with three questions:
1. **What can actually be measured?** -- Not every theoretically appealing quantity is empirically accessible at the scale and budget we have.
2. **What confounders are lurking?** -- Multi-task policy learning has many hidden variables (task ordering, demo quality, simulation stochasticity) that can masquerade as cross-task influence.
3. **What experiment would convince a hostile reviewer?** -- Design around the weakest link in the argument chain.

---

## Critical Assessment of Measurability

### What the Proposal Needs to Measure

The core claim is a Task-to-Task Influence Matrix $M_{ij}$ that quantifies how task $i$'s training data affects task $j$'s performance. This requires:

1. A **ground truth** definition of influence (what $M_{ij}$ "really is")
2. A **cheap proxy** that approximates the ground truth (gradient cosine, BCS, RepFinger, etc.)
3. A **downstream application** (data mixing) where the proxy demonstrably outperforms baselines

Each of these has serious measurement challenges that the other perspectives gloss over.

### Ground Truth Problem: LOO Retraining Is Not Ground Truth

Everyone treats Leave-One-Task-Out (LOTO) retraining as "ground truth," but this is deeply problematic:

**Confounder 1: Training stochasticity.** A single LOTO run has variance from random initialization, data shuffling, and optimization noise. The pragmatist estimates each training run at ~5 min on ResNet-18 + MLP. But the *signal* we're trying to detect -- the performance change from removing one task out of ten -- may be 1-3% in success rate. With LIBERO's stochastic evaluation (50 rollouts typical), 95% CI on success rate is roughly $\pm \sqrt{p(1-p)/50} \approx \pm 7\%$ at $p=0.5$. **The noise floor exceeds the expected signal.**

**Confounder 2: Optimization path dependence.** With and without task $i$, the optimizer follows a different trajectory. The measured $\Delta_j$ conflates (a) the data content effect (what we want) with (b) the optimization dynamics effect (different effective learning rate, different loss landscape curvature). These are inseparable in a single LOTO experiment.

**Confounder 3: Sample size interaction.** Removing task $i$ also changes the total training set size. If task $i$ has 50 demos, removing it drops the dataset from 500 to 450 samples. The 10% reduction in data volume alone may affect task $j$ -- this is not a cross-task influence effect, it's a data scaling effect.

### Implication: We Need a Far More Rigorous Ground Truth Protocol

I propose **Controlled LOTO (C-LOTO)** as the proper ground truth:

1. **Multiple seeds**: Run each LOTO configuration with $k \geq 5$ random seeds. Report the mean and 95% CI of $\Delta_j^{(i)} = \text{SR}_j(\text{all tasks}) - \text{SR}_j(\text{without task } i)$.
2. **Data-volume control**: When removing task $i$ (50 demos), replace with 50 additional demos *from the remaining tasks* (proportional upsampling). This isolates cross-task content effects from data volume effects.
3. **Evaluation budget**: Use 200+ rollouts per condition (not 50). At 200 rollouts, 95% CI narrows to $\pm 3.5\%$ at $p=0.5$, which is barely sufficient for detecting 5% effects.
4. **Statistical significance test**: Apply a paired permutation test (not just point comparison) to determine which $\Delta_j^{(i)}$ are significantly non-zero at $\alpha = 0.05$ after Bonferroni correction for $T(T-1) = 90$ comparisons.

**Cost**: This is expensive. With 5 seeds $\times$ 11 configurations (10 LOTO + 1 baseline) $\times$ 5 min/run = ~275 min = 4.6 hours of training. Plus 11 $\times$ 10 $\times$ 200 = 22,000 evaluation rollouts. This is the real cost of a proper ground truth. The pragmatist's "50 min for LOO" is an underestimate by ~5x once you account for adequate statistical power.

---

## Proposal: Three Experiment-Driven Research Ideas

### Idea 1: Pre-Registered Proxy Benchmark with Blinded Evaluation (Improve Existing)

#### Hypothesis (Pre-Registered)
**H1**: At least one cheap proxy (among GradCos, GPTA, BCS, RepFinger, Kernel Surrogate) achieves Spearman $\rho > 0.6$ with C-LOTO ground truth on LIBERO-10. If no proxy exceeds $\rho = 0.4$, we declare the task-level influence estimation problem **practically unsolvable** at this scale.

#### Why This Specific Threshold
- $\rho = 0.6$ is the minimum for a useful predictive tool (explains ~36% of variance)
- $\rho = 0.4$ is the noise floor observed in the influence function fragility literature (Basu et al., arXiv 2006.14651)
- Pre-registering thresholds prevents post-hoc narrative adjustment

#### Experimental Design

**Phase 1 -- Ground Truth Construction** (~5 hours, distributed across experiment tasks)

| Step | Detail | Time |
|------|--------|------|
| Train all-tasks baseline | ResNet-18 + MLP, LIBERO-10, 5 seeds | 25 min |
| 10 LOTO trainings | Each drops one task, 5 seeds each | 250 min |
| Evaluation rollouts | 11 configs $\times$ 10 tasks $\times$ 200 rollouts, 5 seeds | ~3 hours |
| Volume-controlled LOTOs | 10 configs with proportional upsampling, 5 seeds | 250 min |

Total ground truth effort: ~10 hours, distributed across ~20 experiment tasks. This is substantial but **necessary** -- without it, we're building on sand.

**Phase 2 -- Proxy Computation** (1 experiment task, ~30 min)
Compute all 5 proxies from the all-tasks model:
1. **GradCos**: Average cosine of per-task mean gradients. Cost: 1 backward pass per task.
2. **GPTA** (LESS-style): Random-projected gradient features, cosine scoring. Cost: 1 backward pass per sample + projection.
3. **BCS** (pragmatist's Bottleneck Conflict Score): PCA subspace overlap $\times$ readout direction cosine at bottleneck layer. Cost: 1 forward pass + linear probes.
4. **RepFinger** (innovator's CKA + readout): Layer-wise CKA + linear probe direction cosine. Cost: 1 forward pass per layer.
5. **Kernel Surrogate** (Zhang et al., arXiv 2602.03783): Second-order task attribution via kernel ridge regression on gradient features. Cost: gradient features + kernel computation.

**Phase 3 -- Blinded Evaluation** (1 experiment task, ~15 min)
- Proxy rankings computed without access to ground truth
- Evaluation: Spearman $\rho$, Kendall $\tau$, and top-3/bottom-3 hit rate (does the proxy correctly identify the 3 most-positive and 3 most-negative task pairs?)
- **Critical diagnostic**: Plot proxy scores vs. C-LOTO $\Delta$ values with error bars. If the error bars on $\Delta$ are so large that *any* monotone function fits, the ground truth is too noisy to discriminate proxies.

#### Confounders Controlled
- **Training stochasticity**: 5 seeds per configuration
- **Data volume**: Volume-controlled LOTO isolates content from size effects
- **Evaluation stochasticity**: 200 rollouts with CI reporting
- **Post-hoc bias**: Pre-registered $\rho$ thresholds
- **Multiple comparisons**: Bonferroni correction on 90 pairwise tests

#### Falsification Criterion
If **all** proxies score $\rho < 0.4$, we conclude: "Task-level influence estimation is unreliable for multi-task robot policy learning at the scale of LIBERO-10 (50 demos/task, ~5M param model)." This would redirect the field toward either (a) larger-scale estimation where signals are stronger, or (b) instance-level curation (CUPID/SCIZOR path).

#### Success Probability: 55%
Lower than the pragmatist's 70% because we're holding ourselves to a stricter standard. The pragmatist's success criterion was "proxy correlates with LOO" -- but with single-seed LOO and 50 evaluation rollouts, even noise correlates. Under C-LOTO with 5 seeds and 200 rollouts, the bar is much higher.

#### Key References
- Basu et al. (arXiv 2006.14651, ICLR 2021): Influence function fragility -- baselines for expected correlation floors
- Zhang et al. (arXiv 2602.03783): Kernel surrogate models for task attribution -- 25% higher correlation than linear surrogates
- ETAP (arXiv 2602.18591): Ensemble task affinity predictor -- gradient + learned estimators
- Grad-TAG (arXiv 2409.06091): Gradient-based task affinity at 3% FLOPs -- our GradCos/GPTA proxies derive from this
- Li et al. (arXiv 2512.09103): 0% Euclidean certification for TRAK -- motivates our seed stability analysis

---

### Idea 2: Confounder-Controlled Negative Transfer Detection via Factorial Design (New Method)

#### The Problem Nobody Addresses
Every perspective assumes we *know* negative transfer exists in LIBERO-10 and just need to *measure* it. But has anyone actually demonstrated statistically significant negative transfer in this benchmark? CORAL (arXiv 2603.09298) claims "negative transfer" in joint VLA training but measures it as a gap between per-task LoRA and joint training -- which conflates capacity constraints with actual data interference.

#### Hypothesis (Pre-Registered)
**H2**: In LIBERO-10 with a ResNet-18+MLP policy, at least 3 task pairs exhibit statistically significant negative transfer ($\Delta_j^{(i)} > 0$ at $p < 0.05$ after Bonferroni correction with C-LOTO protocol). If fewer than 3 pairs are significant, negative transfer is **too weak to be a research target** at this model/data scale.

**H2-corollary**: Negative transfer is concentrated in task pairs that share visual scenes but require different manipulation strategies (same environment, different goals). This is the "Representation conflict" mechanism from the spec -- same state, different optimal action.

#### Experimental Design: 2-Level Factorial

Instead of the standard LOTO design (remove one task at a time), I propose a **fractional factorial design** that is statistically more powerful for detecting interactions:

**Factor structure**: 10 binary factors $X_1, \ldots, X_{10}$ (task included vs. excluded). A full factorial has $2^{10} = 1024$ configurations -- infeasible. Instead, use a **Resolution IV fractional factorial** design with 64 runs (generated via Plackett-Burman or D-optimal design). This estimates:
- All 10 main effects (each task's average contribution to all other tasks)
- All 45 pairwise interactions (task $i$ $\times$ task $j$ synergy/conflict)
- With confounding only between 3-way and higher interactions (which we assume negligible per Occam's razor)

**Cost**: 64 training runs $\times$ 3 seeds $\times$ 5 min = 960 min = 16 hours. Distributed across ~32 experiment tasks.

**Advantages over LOTO**:
- LOTO estimates each $\Delta_j^{(i)}$ from 2 measurements (with vs. without). Factorial estimates from ~32 measurements each, giving ~4x higher statistical power.
- Factorial directly estimates *interaction effects* (the innovator's "coalition influence") without additional computation.
- The design is orthogonal, so main effects and interactions are estimated independently.

**Evaluation**: For each configuration, run 200 evaluation rollouts per task (with 3 seeds). Total evaluation: 64 $\times$ 3 $\times$ 10 $\times$ 200 = 384,000 rollouts. This is large but parallelizable (each seed is independent).

**Analysis**:
1. ANOVA with main effects + pairwise interactions. Test each effect at $\alpha = 0.05 / 55 = 0.0009$ (Bonferroni for 10 main + 45 interactions).
2. Effect size: Cohen's $d$ for each significant effect. Practically significant only if $|d| > 0.5$.
3. **Mechanism attribution**: For each significant negative-transfer pair, examine whether the pair shares the same LIBERO scene (visual overlap). Compute the BCS diagnostic for the pair vs. a control pair with similar visual overlap but no negative transfer. This tests the "representation conflict" mechanism.

#### Confounders Controlled
- **Multi-collinearity**: Orthogonal factorial design eliminates correlation between factors
- **Data volume**: Each configuration trains on a different total number of demos. Include total dataset size as a covariate in the ANOVA.
- **Task ordering**: Within each run, shuffle task data. Across runs, use the same shuffling seed.
- **Evaluation variance**: 200 rollouts $\times$ 3 seeds per cell = 600 effective evaluations

#### Pilot Study (Must Complete in <15 min)
Before committing to the full factorial:
1. Train all-10-tasks model and a 5-task subset (tasks 1-5 only), 1 seed each, 5 min each
2. Evaluate tasks 1-5 in both conditions, 50 rollouts each
3. If performance difference for any task is $> 10\%$, there is enough signal to justify the full study
4. If all differences are $< 3\%$, LIBERO-10 tasks are too homogeneous -- switch to a heterogeneous task set (see Risk Assessment)

#### Falsification Criterion
If the factorial ANOVA finds no significant pairwise interactions ($p > 0.05$ after correction for all 45 pairs), negative transfer is not a meaningful phenomenon in LIBERO-10. The paper pivots to: "Negative Transfer in Multi-Task Robot Learning Is Weaker Than Assumed" -- a valuable empirical finding that challenges CORAL's premise.

#### Success Probability: 50%
This is deliberately a coin-flip because we genuinely do not know whether LIBERO-10 has meaningful negative transfer. Either outcome is publishable.

#### Key References
- CORAL (arXiv 2603.09298): Claims negative transfer in multi-task VLA -- our factorial design provides the rigorous test
- Ortho-LoRA (arXiv 2601.09684): Gradient conflict in multi-task LoRA -- "negative cosine similarity between task gradients" but measured only as gradient statistics, never as actual task performance interaction
- CMTA (arXiv 2311.01075): Addresses "negative transfer within the task" using contrastive modules with temporal attention on Meta-World
- PiKE (arXiv 2502.06244): Found little gradient conflict at large scale -- our factorial tests whether this holds at small scale
- "It's a Match!" (arXiv 2301.02873): Pairwise affinity scores predict MTL performance poorly -- our factorial estimates interactions directly from performance, not proxies

---

### Idea 3: Intervention Study -- Does Influence-Guided Mixing Actually Improve Deployment Performance? (Cross-Domain Transfer)

#### The Gap in All Prior Perspectives
Every perspective (innovator, pragmatist, theorist, contrarian) evaluates mixing strategies by **training-time loss or validation success rate**. Nobody evaluates the downstream consequence that actually matters: **does the influence-optimized policy perform better when deployed on the target task in the simulation environment, across diverse initial conditions, with proper distribution shift?**

The distinction matters because:
- A mixing strategy that reduces training loss may overfit to the validation distribution
- Success rate on LIBERO's standard evaluation protocol (50 fixed initial configs) may not reflect robustness to novel configurations
- The mixing optimization loop introduces its own overfitting risk: if mixing weights are tuned to maximize validation SR, they exploit validation-specific statistics

#### Hypothesis (Pre-Registered)
**H3**: Influence-guided data mixing improves standard LIBERO evaluation success rate by $\geq 5\%$ absolute over uniform mixing, BUT the gap shrinks to $< 2\%$ on out-of-distribution evaluation configs (perturbed object positions, novel distractor objects). The apparent benefit is partially an artifact of evaluation protocol, not genuine policy robustness.

#### Experimental Design: Mixing Strategy Comparison with OOD Generalization Test

**Training phase**: Use the best proxy from Idea 1 to derive mixing weights $w_1, \ldots, w_{10}$.

**4 mixing strategies** (each with 5 seeds):
1. **Uniform**: $w_i = 1/10$ for all tasks
2. **Influence-guided**: $w_i^{(j)} \propto \max(0, M_{ij})$ for task $j$'s data in joint training
3. **Re-Mix (DRO)**: Distributionally robust optimization over task weights (arXiv 2408.14037)
4. **Oracle**: LOO-derived optimal weights (upper bound, not practical)

**Evaluation on two protocols**:
- **Standard (In-Distribution)**: LIBERO's default 50 initial configurations per task, 200 rollouts
- **Perturbed (Out-of-Distribution)**: 50 *novel* initial configurations with:
  - Object positions shifted by 2-5 cm from training distribution
  - Camera viewpoint rotated by 5-10 degrees
  - (If LIBERO supports it) adding 1-2 distractor objects not seen in training

**Analysis**:
1. Compare success rates across 4 strategies $\times$ 2 evaluation protocols $\times$ 10 tasks
2. Two-way ANOVA: strategy $\times$ evaluation protocol, with seed as random effect
3. **Key test**: Is the strategy $\times$ protocol interaction significant? If yes, influence-guided mixing is exploiting in-distribution statistics. If no, the improvement is genuine.
4. Report per-task breakdown: influence-guided mixing likely helps tasks with detected negative transfer but may hurt tasks that benefited from diverse co-training data (the "positive transfer tax").

#### Confounders Controlled
- **Hyperparameter tuning**: All strategies use the same model architecture, learning rate, and total training steps. Only the data mixture changes.
- **Data volume**: All strategies see the same total number of training samples per epoch (upsampled/downsampled to match)
- **Evaluation**: Same rollout seeds across strategies for paired comparison
- **Mixing overfitting**: Mixing weights are derived from the proxy (fixed, not tuned on validation) to avoid validation overfitting

#### Computational Cost
| Step | Time | Notes |
|------|------|-------|
| Training (4 strategies $\times$ 5 seeds) | 100 min | 20 runs $\times$ 5 min |
| Standard eval (20 $\times$ 10 $\times$ 200) | ~3 hours | Depends on rollout speed |
| OOD eval (20 $\times$ 10 $\times$ 200) | ~3 hours | Same |
| **Total** | **~7 hours** | Split across ~14 experiment tasks |

#### Falsification Criterion
If influence-guided mixing does NOT outperform uniform mixing by $\geq 5\%$ on standard eval, AND does not outperform on OOD eval either, then influence-guided mixing is **not practically useful** at this scale. The contrarian's prediction (architecture > data mixing) is supported.

If influence-guided mixing wins on standard eval but NOT on OOD eval, then the benefit is **evaluation-protocol-specific** -- a cautionary finding for the entire data mixing literature.

#### Success Probability: 45%
This is the hardest test. The contrarian's CORAL baseline (per-task LoRA, no mixing needed) will likely be competitive. The influence-guided mixing needs to beat not just uniform mixing but also show robustness to distribution shift. I estimate the approach has a genuine ~45% chance of demonstrating a practically significant and robust improvement.

#### Key References
- Re-Mix (arXiv 2408.14037): DRO-based data mixing baseline -- +38% on Open X-Embodiment, but evaluated only on in-distribution
- DUET (arXiv 2502.00270): Influence function + Bayesian optimization for data mixture -- but no OOD evaluation
- AC-ODM (arXiv 2505.23878): Actor-critic online data mixing for LLMs -- captures intra-domain interactions
- LIBERO (arXiv 2306.03310, NeurIPS 2023): Standard evaluation protocol uses fixed initial configs -- no OOD test
- MTBench (RLJ 2025): Massively parallelized multi-task robot benchmark -- provides broader evaluation infrastructure

---

## Synthesis: An Experiment-First Research Plan

The three ideas form a logical chain:

```
Idea 2 (Detection)     → "Does negative transfer exist at all?"
         ↓ (If yes)
Idea 1 (Measurement)   → "Can we measure it cheaply and reliably?"
         ↓ (If yes)
Idea 3 (Application)   → "Does the measurement actually help?"
```

This chain is the **only honest way** to build this paper. Starting with Idea 3 (as most perspectives implicitly recommend) puts the cart before the horse -- you can't optimize data mixing based on influence if you haven't first verified that (a) influence exists and (b) you can measure it.

### Recommended Execution Order

**Phase 0: Pilot Feasibility (1 experiment task, 15 min)**
- Train 10-task model and 5-task subset
- Quick SR comparison: is there any measurable performance difference when removing half the tasks?
- **Gate**: If $|\Delta| < 3\%$ for all tasks, switch benchmark to Meta-World MT10 or a deliberately heterogeneous LIBERO subset

**Phase 1: Does Negative Transfer Exist? (Idea 2, 32 experiment tasks, ~16 hours)**
- Fractional factorial design, 64 configs $\times$ 3 seeds
- ANOVA analysis with Bonferroni correction
- **Gate**: Proceed to Phase 2 only if $\geq 3$ task pairs show significant negative transfer

**Phase 2: Can We Measure It? (Idea 1, 22 experiment tasks, ~11 hours)**
- C-LOTO ground truth (5 seeds, 200 rollouts, volume-controlled)
- 5 proxy computations + blinded evaluation
- **Gate**: Proceed to Phase 3 only if best proxy achieves $\rho > 0.6$

**Phase 3: Does It Help? (Idea 3, 14 experiment tasks, ~7 hours)**
- 4 mixing strategies $\times$ 5 seeds
- Standard + OOD evaluation
- Statistical analysis of strategy $\times$ protocol interaction

### Total Budget
| Phase | Experiment Tasks | Wall-Clock |
|-------|-----------------|------------|
| Pilot | 1 | 15 min |
| Phase 1 (Detection) | 32 | ~16 hrs |
| Phase 2 (Measurement) | 22 | ~11 hrs |
| Phase 3 (Application) | 14 | ~7 hrs |
| **Total** | **69** | **~34 hrs** |

This is ~4x larger than the pragmatist's estimate. The difference is entirely due to **statistical rigor**: multiple seeds, adequate evaluation rollouts, volume controls, and Bonferroni correction. Cutting corners on any of these produces results that a hostile reviewer can dismiss.

Note: The spec allocates "Pilot ~1-2 GPU-days, mid-scale ~10-15 GPU-days." Our full plan uses ~34 GPU-hours $\approx$ 1.4 GPU-days for the pilot phase and ~34 hours total, well within the mid-scale budget on a single A6000. With 4 GPUs available, wall-clock time can be reduced to ~9 hours.

---

## Confounder Registry

Every experiment must control for these variables. I flag this because none of the other perspectives explicitly lists confounders:

| Confounder | Mechanism | Control |
|-----------|-----------|---------|
| **Random seed** | Different initialization → different loss landscape basin → different task interactions | 5 seeds per config, report CI |
| **Data volume** | Removing a task reduces total data → performance drops from less data, not from removing influence | Volume-controlled LOTO (proportional upsampling) |
| **Data ordering** | Multi-task training is sequential within batches → order effects | Consistent shuffling seed across configs |
| **Evaluation stochasticity** | LIBERO rollouts have random initial configs | 200 rollouts, paired evaluation seeds |
| **Task difficulty** | Hard tasks are more sensitive to any perturbation → false positives for "influence" | Normalize $\Delta_j^{(i)}$ by single-task variance |
| **Optimization convergence** | Different configs converge at different rates → early stopping bias | Fixed epoch count (not early stopping) |
| **Gradient accumulation** | Batch composition changes when tasks are removed → different effective gradient signal | Fixed batch size; oversample remaining tasks to maintain batch composition |

---

## What I Would NOT Do

1. **Compute influence on a 7B VLA** (too expensive for proper statistical controls; defer to after the small-model story is complete and validated)
2. **Use custom toy datasets** (no external validity; LIBERO-10 and Meta-World MT10 are the established benchmarks)
3. **Report single-seed results** (this is the #1 sin in robot learning papers; see Henderson et al. (arXiv 1709.06560) "Deep Reinforcement Learning that Matters")
4. **Skip the detection phase** (jumping straight to mixing optimization assumes the conclusion)
5. **Tune mixing weights on the validation set** (this introduces mixing overfitting and inflates reported gains)
6. **Trust gradient cosine as ground truth** (it's a proxy, not a measurement of actual performance change)

---

## Risk Assessment

### Risk 1: LIBERO-10 Is Too Homogeneous (No Detectable Negative Transfer)
**Probability**: 35%
**Mitigation**: Pre-screen in Pilot Phase 0. If no signal, switch to:
- **Meta-World MT10**: 10 tasks spanning reach, push, pick-place, drawer, window operations. Known to have significant task interference in MTL (CMTA, arXiv 2311.01075 showed 30%+ gaps).
- **Heterogeneous LIBERO subset**: Hand-pick 5 tasks with maximal visual/action diversity (e.g., tasks from different LIBERO suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal).

### Risk 2: Ground Truth Is Too Noisy to Discriminate Proxies
**Probability**: 25%
**Mitigation**: The factorial design (Idea 2) provides ~4x more statistical power than LOTO. If even the factorial cannot detect effects, we conclude negative transfer is below the detection threshold -- itself a publishable finding.

### Risk 3: All Proxies Fail ($\rho < 0.4$)
**Probability**: 30%
**Mitigation**: This validates the contrarian's thesis and produces a strong negative result paper: "Cheap Proxies for Task Affinity Are Unreliable in Robot Policy Learning." Redirect to instance-level curation (CUPID path) or architecture-based isolation (CORAL path).

### Risk 4: Budget Overrun
**Probability**: 20%
**Mitigation**: Phases are gated. If Phase 1 fails the detection gate, we save ~18 hours by skipping Phases 2-3 and writing a detection-focused paper. If Phase 2 fails the proxy quality gate, we save ~7 hours and write a "proxy benchmark + negative result" paper.

---

## Falsification Summary

| Idea | Hypothesis | What Would Falsify It | Consequence If Falsified |
|------|-----------|----------------------|-------------------------|
| 1 | At least one proxy achieves $\rho > 0.6$ | All proxies $\rho < 0.4$ | Task-level influence estimation is impractical for robot MTL |
| 2 | $\geq 3$ task pairs show significant negative transfer | $< 3$ significant pairs after Bonferroni | Negative transfer is too weak to be a research target at this scale |
| 3 | Influence-guided mixing improves SR by $\geq 5\%$ | Improvement $< 2\%$ on both ID and OOD | Influence-guided mixing is not practically useful |

**Every falsification outcome is itself a publishable finding.** This is the hallmark of a well-designed research program -- heads we learn something useful, tails we learn something useful.

---

## Key References (Consolidated)

### Ground Truth / Evaluation Methodology
- Henderson et al. (arXiv 1709.06560): "Deep RL That Matters" -- seed sensitivity, evaluation protocol rigor
- Basu et al. (arXiv 2006.14651, ICLR 2021): Influence function fragility in deep networks
- Li et al. (arXiv 2512.09103): 0% Euclidean certification for TRAK attribution scores
- LIBERO (arXiv 2306.03310, NeurIPS 2023): Benchmark definition and standard evaluation protocol

### Task Affinity Estimation
- Grad-TAG (arXiv 2409.06091): Gradient-based task affinity at 3% FLOPs
- ETAP (arXiv 2602.18591): Ensemble task affinity predictor with gradient + learned estimators
- Zhang et al. (arXiv 2602.03783): Kernel surrogate models -- 25% higher correlation than linear surrogates, second-order task interactions
- LESS (arXiv 2402.04333, ICML 2024): Gradient projection for efficient influence
- "It's a Match!" (arXiv 2301.02873): Pairwise affinity scores predict MTL performance poorly

### Negative Transfer in Robot/MTL
- CORAL (arXiv 2603.09298, March 2026): Per-task LoRA isolates interference -- our head-to-head baseline
- Ortho-LoRA (arXiv 2601.09684): Orthogonal gradient projection for multi-task LoRA
- CMTA (arXiv 2311.01075): Contrastive modules with temporal attention on Meta-World -- prior evidence of significant task interference in robot MTL
- PiKE (arXiv 2502.06244): Large-scale pretraining shows low gradient conflict
- Kang (arXiv 2512.22740): Empirical study of negative transfer -- gradient misalignment from mismatched functional forms

### Data Mixing
- Re-Mix (arXiv 2408.14037): DRO data mixing for imitation learning
- AC-ODM (arXiv 2505.23878): Actor-critic online data mixing for LLM pretraining
- CUPID (arXiv 2506.19121): Per-demonstration influence for robot data curation
- MISS (arXiv 2409.18153): Set influence is non-additive
