# Contrarian Perspective: Cross-Task Influence in Multi-Task VLA

## Core Thesis: The Entire Premise May Be Wrong

The proposal's central assumption -- that we can build a reliable task-to-task influence matrix, diagnose negative transfer mechanisms, and use this to optimize data mixing -- rests on a chain of beliefs, each of which has serious counter-evidence. I challenge three foundational assumptions and propose research directions that exploit the gaps.

---

## Challenged Assumption 1: Influence Functions Can Reliably Quantify Cross-Task Effects in Deep Policy Networks

### The Mainstream Belief
The proposal assumes that influence functions (or their gradient-projected proxies like LESS) can produce a meaningful task-to-task influence matrix $M_{ij}$ for VLA models. The innovator, pragmatist, and theorist all build on this assumption -- they disagree on *which* proxy to use but agree that *some* gradient-based signal will work.

### The Counter-Evidence

**Influence functions are provably fragile in deep networks.** Basu et al. (ICLR 2021, arXiv 2006.14651) demonstrated that influence function estimates in deep learning are "fragile" -- they fail catastrophically for deep architectures like ResNet-50 due to:
- Non-convexity of the loss function, which violates the core theoretical assumption
- Initialization sensitivity: different random seeds during leave-one-out retraining produce wildly different parameter trajectories
- Inverse-Hessian approximation error grows super-linearly with network depth

Epifano et al. (2023, arXiv 2303.12922) confirmed that the observed fragility is not just a validation artifact -- the underlying approximation genuinely breaks down for non-convex models. Class-based influence functions (Nguyen-Duc et al., 2023, arXiv 2305.01384) showed that cross-class influence estimates are *systematically unreliable*, which is exactly the regime we operate in: cross-task influence is inherently cross-distribution.

Li et al. (2025, arXiv 2512.09103) proved that standard TRAK scores, while accurate as point estimates, are "geometrically fragile" -- naive Euclidean robustness analysis yields **0% certification** for neural network attribution on CIFAR-10/ResNet-18. Their Natural Wasserstein metric reduces worst-case sensitivity by 76x, but this fix has never been validated for policy networks.

Vitel & Chhabra (2025, arXiv 2511.04715) showed that even the *choice of which layer to compute influence from* produces contradictory results -- first layers vs. middle layers vs. last layers give different rankings, contradicting the seminal Yeh et al. (2022) recommendation.

**The policy learning setting makes things worse.** VLA models have:
- Sequential, non-i.i.d. data (trajectories, not independent images)
- Multi-modal action distributions (diffusion heads, GMMs) where the loss landscape is highly non-convex
- Closed-loop deployment where small parameter perturbations can cascade into dramatically different trajectories

Matelsky et al. (2024, arXiv 2406.00509) tested empirical influence functions on both CNNs and LLMs and found that basic desiderata (transitivity, noise invariance, logical consistency) are *violated* -- neural networks "cannot generalize or perform logic in the way they appear to" when probed via fine-tuning influence.

### The Contrarian Research Direction

**Direction 1: Falsification-First Influence Validation for VLA**

Instead of assuming influence-based analysis works and building a whole methodology on it, start with a rigorous falsification study:

1. Compute influence matrix $M_{ij}$ using LESS/GPTA on LIBERO-10 with a small policy
2. Compute the *same* matrix using 5 different random seeds for the base model training
3. Measure the **seed-to-seed Kendall-$\tau$ correlation** of influence rankings
4. If $\tau < 0.5$ (rankings are unstable across seeds), the entire influence-guided mixing approach is built on sand

**Prediction**: Based on the fragility literature, I predict $\tau \approx 0.3$-$0.4$ for deep policy networks -- marginally better than random. The VLA setting (sequential data, multi-modal actions) will amplify instability beyond what was observed for image classification.

**If confirmed**, this is itself a strong negative result paper: "Influence Functions Fail for Multi-Task Robot Policy Learning" -- directly challenging CUPID (arXiv 2506.19121), DataMIL (arXiv 2505.09603), and every influence-based robot data curation method.

**Computational cost**: ~45 min (5 training runs x 5 min + gradient features x 5 + comparison)

**Success probability**: 60% (that influence *does* fail; 40% chance it works better than expected, in which case we pivot to using it)

### Key References
- Basu et al. (arXiv 2006.14651, ICLR 2021): "Influence Functions in Deep Learning Are Fragile"
- Epifano et al. (arXiv 2303.12922): "Revisiting the Fragility of Influence Functions"
- Li et al. (arXiv 2512.09103): Natural Geometry of Robust Data Attribution -- 0% Euclidean certification for TRAK
- Vitel & Chhabra (arXiv 2511.04715): Layer choice produces contradictory influence rankings
- Matelsky et al. (arXiv 2406.00509): Empirical influence functions violate basic logical desiderata
- MISS (arXiv 2409.18153): Set influence is non-additive, undermining pairwise matrices

---

## Challenged Assumption 2: The Right Intervention Level is Data Mixing, Not Architecture

### The Mainstream Belief
The proposal treats architecture as fixed and optimizes the data mixture. All three prior perspectives (innovator, pragmatist, theorist) share this assumption -- they debate *how* to compute influence and *how* to derive mixing weights, but never question whether data mixing is the right lever to pull.

### The Counter-Evidence

**Parameter isolation trivially eliminates the problem.** CORAL (arXiv 2603.09298, March 2026) freezes a single VLA backbone and attaches one LoRA expert per task. This "strict parameter isolation avoids complex gating networks and prevents parameter-level cross-task interference by construction." CORAL **substantially outperforms** joint training on LIBERO, WidowX, and Google Robot -- without any influence analysis, without any data mixing optimization, without any mechanism diagnosis.

MergeVLA (arXiv 2511.18810) found that directly merging VLA experts trained on different tasks results in **near-zero success rates** and identified the root cause: "finetuning drives LoRA adapters toward divergent, task-specific directions beyond the capacity of existing merging methods to unify." Their solution -- sparsely activated task masks -- again bypasses the data mixing problem entirely.

CDSP-MoE (arXiv 2512.20291) uses gradient conflict not to *analyze* interference but as a *structural supervisory signal* to prune conflicting pathways in a shared parameter manifold. This shifts the problem from "which data to mix" to "which parameters to share."

**The implication is devastating for the proposal**: if a $0.1\%$-parameter LoRA adapter per task eliminates negative transfer more effectively than any data mixing strategy, then the entire influence matrix apparatus is solving a problem that architecture already solves more cheaply.

**PiKE's inconvenient finding.** PiKE (arXiv 2502.06244) found that large-scale pretraining often exhibits *little to no gradient conflict*. If VLA pretraining is similarly low-conflict, then the "negative transfer" we're trying to diagnose may be an artifact of (a) insufficient model capacity, (b) poor optimization, or (c) distribution shift -- none of which are addressable by data mixing.

### The Contrarian Research Direction

**Direction 2: Architecture vs. Data -- A Controlled Ablation Study**

Run a head-to-head comparison that the field has never done:

| Configuration | Architecture | Data Strategy | Expected Outcome |
|---|---|---|---|
| Baseline | Shared trunk + shared head | Uniform mixing | Worst (negative transfer) |
| Influence-guided | Shared trunk + shared head | Optimized mixing (our method) | Moderate improvement? |
| Per-task LoRA | Shared trunk + LoRA per task | Uniform mixing | Strong (CORAL-style) |
| Per-task LoRA + Influence | Shared trunk + LoRA per task | Optimized mixing | Marginal gain over LoRA alone? |

**The critical test**: Does influence-guided mixing on a shared architecture *ever* match per-task LoRA with uniform mixing? If not, the entire data mixing research direction is dominated by a simpler architectural solution.

**My prediction**: Per-task LoRA with uniform mixing will match or beat influence-guided shared-architecture training. The marginal gain of influence-guided mixing *on top of* LoRA will be <2% -- within noise. This suggests the field should invest in better routing/isolation architectures, not better data mixing.

**However**, there is a scenario where data mixing matters: when the task is *unknown* at test time (truly zero-shot generalization). Per-task LoRA requires a task router, which fails on novel tasks. In this regime, the shared model must generalize, and data mixing optimization may help. This is the only setting where the proposal's approach is not dominated.

**Computational cost**: ~50 min (4 training runs x 10 min + evaluation)

**Success probability**: 55% (that architecture dominates data mixing; 45% that data mixing shows non-trivial gains)

### Key References
- CORAL (arXiv 2603.09298): Per-task LoRA substantially outperforms joint training
- MergeVLA (arXiv 2511.18810): Near-zero success rate from naive VLA expert merging
- CDSP-MoE (arXiv 2512.20291): Gradient conflict as structural signal, not analytical signal
- PiKE (arXiv 2502.06244): Large-scale pretraining shows low gradient conflict
- STRAP (arXiv 2412.15182): Test-time retrieval outperforms multi-task generalist policies

---

## Challenged Assumption 3: Task-Level Analysis is the Right Granularity

### The Mainstream Belief
The proposal frames everything at the task level: task-to-task influence, task-level mixing weights, task-pair mechanism diagnosis. The innovator partially challenges this with "Temporal Influence Tomography" (sub-skill phases), but still treats tasks as the fundamental unit for data mixing.

### The Counter-Evidence

**"It's a Match!" (arXiv 2301.02873) proved that simple pairwise task affinity scores correlate poorly with actual MTL performance.** This isn't just a "use a better proxy" problem -- it's a fundamental issue with task-level analysis. Tasks are *not* atomic units of knowledge transfer.

**MISS (arXiv 2409.18153)** formalized that set influence is non-additive -- the influence of task set $\{B, C\}$ on task $A$ is NOT the sum of $B$'s influence and $C$'s influence. This means any pairwise matrix $M_{ij}$ is *inherently incomplete*. The theorist's superadditivity bound (Proposition 3) attempts to quantify this gap but relies on the same fragile Hessian approximation.

**The real unit of transfer is the data point, not the task.** CUPID (arXiv 2506.19121) demonstrated that within a single task, individual demonstrations have wildly different influence on policy performance -- some demonstrations *within the target task's own dataset* are harmful. If intra-task variation exceeds inter-task variation (which is plausible for heterogeneous demonstration datasets), then task-level analysis smooths away the actual signal.

**STRAP (arXiv 2412.15182)** showed that sub-trajectory retrieval at test time outperforms both full-trajectory retrieval and multi-task generalist policies. The granularity that matters is sub-trajectory, not task. This aligns with the observation that robot tasks share low-level behaviors (approach, grasp) while diverging on high-level strategies.

### The Contrarian Research Direction

**Direction 3: Instance-Level Data Valuation Beats Task-Level Influence Matrices**

Instead of building a $T \times T$ task influence matrix, build an $N$-dimensional per-sample value score and show it captures everything the task matrix captures -- plus more:

1. Train a base multi-task model on LIBERO-10
2. Use CUPID-style per-demonstration influence to rank ALL demonstrations across ALL tasks
3. Cluster demonstrations by influence pattern (not by task label) using k-means on gradient features
4. Show that the emergent clusters do NOT align with task boundaries -- demonstrations from different tasks cluster together when they share low-level manipulation primitives
5. Use cluster-aware data selection (keep high-value demonstrations, remove harmful ones regardless of task) and compare against task-level mixing optimization

**Hypothesis**: Removing the bottom 20% of demonstrations (ranked by per-sample influence) will improve multi-task performance more than any task-level mixing strategy. The harmful demonstrations are not uniformly distributed across tasks -- they concentrate in specific, identifiable patterns (e.g., demonstrations with unusual grasp strategies, demonstrations near workspace boundaries where different tasks conflict).

**Why this is contrarian**: The proposal's entire conceptual framework is "tasks interact." I'm arguing: "No -- *demonstrations* interact, and task labels are a poor proxy for the actual structure of data interactions."

**Computational cost**: ~40 min (base training + gradient features + clustering + validation)

**Success probability**: 50% (ambitious but grounded -- CUPID already showed 33% data removal improves performance)

### Key References
- CUPID (arXiv 2506.19121): Per-demonstration influence for robot data curation -- training with <33% curated data matches full-data performance
- MISS (arXiv 2409.18153): Set influence is non-additive, pairwise matrices are incomplete
- "It's a Match!" (arXiv 2301.02873): Task affinity scores predict MTL performance poorly
- STRAP (arXiv 2412.15182): Sub-trajectory granularity outperforms task-level

---

## Meta-Critique: The Proposal is a Solution in Search of a Problem

Let me be blunt about the overall research narrative:

1. **If negative transfer is weak** (PiKE's finding): There's no problem to solve. The "influence matrix" will be mostly zeros with noise.

2. **If negative transfer is strong** (CORAL's motivation): Architecture already solves it more cheaply. Per-task LoRA eliminates the need for influence analysis.

3. **If negative transfer is moderate and nuanced**: Then task-level analysis is too coarse (MISS, "It's a Match!"), influence functions are too fragile (Basu et al., Li et al.), and instance-level curation (CUPID, SCIZOR) is the better path.

In every scenario, the proposed "task-to-task influence matrix + mechanism diagnosis + data mixing optimization" pipeline is either unnecessary or insufficient.

### The One Scenario Where the Proposal Wins

There IS a narrow but important scenario: **open-vocabulary generalization with a shared model**. When you cannot enumerate tasks at training time (truly open-ended manipulation), per-task LoRA is impossible and instance-level curation doesn't have test-task labels. In this setting:
- You need a shared model
- You need to understand cross-task data interactions to maximize the shared model's generalization
- Task-level influence analysis (or sub-task influence, per the innovator) becomes the only viable approach

If the paper focuses on THIS scenario -- "How should we compose training data for an open-world VLA that must generalize to novel tasks?" -- rather than the generic "diagnose negative transfer" framing, it becomes much more compelling and harder to dismiss with the counter-arguments above.

---

## Recommended Contrarian Contribution: "When Does Influence-Guided Mixing Actually Help?"

Rather than assuming the proposed approach works, I recommend the following paper structure:

### Study 1: Fragility Audit (Contrarian Direction 1)
- Measure seed-to-seed stability of influence matrices for VLA
- Identify the regime (model size, dataset size, loss curvature) where influence becomes reliable
- **Expected finding**: Influence is only stable for small, near-convex models -- exactly the models nobody deploys

### Study 2: Architecture vs. Data Head-to-Head (Contrarian Direction 2)
- Controlled comparison: influence-guided mixing vs. per-task LoRA vs. both
- **Expected finding**: Architecture dominates for known tasks; data mixing matters only for open-vocabulary generalization

### Study 3: Task-Level vs. Instance-Level (Contrarian Direction 3)
- Compare task-to-task influence matrix against per-sample data valuation
- **Expected finding**: Instance-level captures strictly more signal; task labels are a lossy aggregation

### Study 4: The Positive Transfer Opportunity (Contrarian Reframing)
- Instead of "mitigating negative transfer," focus on "amplifying positive transfer"
- Identify demonstrations that help the *most* tasks simultaneously (universal positives)
- Show that curating a small set of universal-positive demonstrations improves all tasks without any mixing optimization
- **Connection**: This reframes data curation as finding "foundation demonstrations" for robotics -- analogous to foundation model pretraining data curation

### Computational Budget
| Study | Experiment Tasks | Wall-Clock |
|---|---|---|
| Fragility Audit | 5 | ~50 min |
| Architecture vs. Data | 4 | ~50 min |
| Task vs. Instance Level | 3 | ~40 min |
| Positive Transfer | 3 | ~30 min |
| **Total** | **15** | **~3 hrs** |

---

## Risk Assessment: What If I'm Wrong?

### If influence functions ARE stable for VLA (my Direction 1 fails):
- The proposal's approach is validated, and we proceed with it. My fragility audit becomes a positive validation result rather than a negative finding.
- **Fallback**: Reframe as "Conditions for Reliable Influence Estimation in Policy Learning" -- still a useful contribution since nobody has verified this.

### If data mixing DOES beat architecture (my Direction 2 fails):
- This would be genuinely surprising and very publishable: "When Data Mixing Outperforms Parameter Isolation in Multi-Task Robot Learning."
- **Fallback**: Identify the specific conditions (dataset heterogeneity? task similarity?) where mixing wins.

### If task-level analysis IS sufficient (my Direction 3 fails):
- Then the proposal is essentially correct and my contrarian perspective strengthened it by stress-testing.
- **Fallback**: The instance-level analysis still provides additional signal even if task-level is sufficient.

**In every case, the contrarian experiments produce publishable findings** -- either validating or falsifying the core assumptions. This is the mark of a well-designed research program.

---

## Summary: Three Assumptions, Three Challenges, Three Directions

| # | Assumption Challenged | Counter-Evidence | Proposed Direction | Success Prob. |
|---|---|---|---|---|
| 1 | Influence functions work for VLA | Fragility in deep nets (Basu 2021), 0% certification (Li 2025), layer choice contradictions (Vitel 2025) | Falsification-first fragility audit | 60% |
| 2 | Data mixing is the right lever | CORAL's LoRA isolation, PiKE's low-conflict finding, MergeVLA's parameter divergence | Architecture vs. data head-to-head | 55% |
| 3 | Task-level is the right granularity | "It's a Match!" poor correlation, MISS non-additivity, CUPID instance-level success | Instance-level data valuation | 50% |

**My strongest recommendation**: Lead with Direction 2 (architecture vs. data). This is the most devastating challenge to the proposal AND produces the most actionable insight. If architecture wins (likely), pivot the paper to: "Understanding When Shared-Model Multi-Task Learning is Worth the Complexity" -- a paper that characterizes the narrow conditions under which influence-guided data mixing outperforms simpler architectural solutions. This is a contribution the field genuinely needs.
