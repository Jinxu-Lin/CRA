# Innovator Perspective: Cross-Task Influence in Multi-Task VLA

## Meta-Observation

The current proposal (influence matrix + mechanism diagnosis + guided mixing) is solid but methodologically conservative -- it applies existing TDA tools (influence functions, gradient similarity) to a new domain (VLA). The core risk is that it becomes an "application paper" rather than a methods contribution. Below I propose three unconventional angles that could elevate the work from "influence functions applied to robots" to "a new framework for understanding multi-task data interactions."

---

## Angle 1: Sub-Skill Decomposition as the Natural Unit of Cross-Task Influence (Improve Existing)

### Counter-Intuitive Claim
**Task-level influence matrices are the wrong abstraction.** Robot manipulation tasks share sub-skills (approach, grasp, lift, place, retreat) at a much finer granularity than task labels suggest. Two tasks labeled "pick-and-place" vs. "drawer-open" may share 70% of their trajectory (approach + grasp) while conflicting only in the last 30% (lift-place vs. pull). A task-level influence score averages over these complementary dynamics and produces a misleading "mildly positive" signal that hides both strong positive transfer in shared sub-skills and strong negative transfer in divergent ones.

### Hypothesis
**H1**: Decomposing trajectories into sub-skill segments and computing influence at the sub-skill level reveals a *block-diagonal* structure in the influence matrix -- sub-skills cluster into "transferable" and "conflicting" blocks that are invisible at the task level.

### Method: Temporal Influence Tomography (TIT)
1. **Sub-skill segmentation**: Use a simple change-point detection on action velocity/gripper state to segment each trajectory into 3-5 phases (e.g., approach, contact, manipulate, retreat). Alternatively, use the VLA's attention map over time steps as a learned segmentation signal.
2. **Phase-conditioned influence**: For each phase $p$, compute a separate influence matrix $M^{(p)}_{ij}$ using LESS-style gradient projections restricted to time steps within phase $p$. This gives us a *tensor* $\mathcal{M} \in \mathbb{R}^{T \times T \times P}$ instead of a flat matrix.
3. **Conflict localization**: Identify which phases drive negative transfer. Prediction: "approach" phases will show near-universal positive transfer, while "manipulation" phases will show task-specific conflicts.
4. **Phase-aware mixing**: Weight training data not uniformly per task, but per (task, phase) pair. Up-weight shared approach data from all tasks; down-weight conflicting manipulation data from interfering tasks.

### Why This Is Novel
- STRAP (2024) does sub-trajectory *retrieval* but never computes influence at this granularity.
- MISS (2024) proves set influence is non-additive -- our temporal decomposition is one concrete way to capture the non-additivity by showing it arises from phase-level interactions.
- No existing work computes a *temporal influence tensor* for policy learning.

### Experimental Plan (<=1hr per experiment)
- **Dataset**: LIBERO-10 (10 tasks, ~50 demos each) or AgiBot World 5-task subset
- **Model**: Small policy (ResNet-18 + MLP action head, or Qwen-0.5B LoRA adapter on a tiny VLA)
- **Step 1** (15 min): Segment trajectories using gripper-state change points
- **Step 2** (30 min): Compute phase-conditioned gradient features (LESS-style random projection, d=256)
- **Step 3** (15 min): Compare task-level vs. phase-level influence matrices; measure whether phase-level reveals block structure invisible at task level
- **Validation**: LOO at phase level -- remove all "manipulation phase" data from task B, check if task A performance on its manipulation phase improves (confirming phase-specific negative transfer)

### Computational Cost
- Gradient computation: ~20 min for 500 trajectories x 5 tasks on single A6000
- Influence scoring: ~10 min (LESS random projection is O(nd) per sample)
- Total pilot: ~45 min

### Success Probability: 65%
### Failure Modes
- Change-point segmentation may not align with meaningful sub-skill boundaries
- Phase-level influence may be too noisy with small data
- The block-diagonal structure may not emerge if tasks are too similar

### Key References
- STRAP (arXiv 2412.15182): sub-trajectory retrieval for cross-task sharing
- MISS (arXiv 2409.18153): set influence non-additivity
- LESS (arXiv 2402.04333): gradient projection for efficient influence
- Influence Dynamics (arXiv 2510.12071): stagewise data attribution with sign flips

---

## Angle 2: Higher-Order Task Interactions via Combinatorial Influence (Cross-Domain Transfer)

### Counter-Intuitive Claim
**Pairwise task affinity is provably insufficient.** Li et al. (2023, arXiv 2306.14009) showed that higher-order task affinities predict negative transfer more accurately than pairwise measures in graph tasks. "It's a Match!" (2023) demonstrated that simple pairwise affinity scores correlate poorly with actual MTL performance. The reason: negative transfer often requires a *coalition* -- task C alone doesn't hurt task A, but tasks B+C together create a conflicting gradient subspace that harms A. This coalition effect is exactly what MISS (2024) formalizes as non-additive set influence.

### Hypothesis
**H2**: In VLA multi-task training, the dominant negative transfer events are driven by *coalitions* of 2-3 interfering tasks, not by single pairwise conflicts. A higher-order influence tensor $\mathcal{I}_{ij|S}$ (influence of task $i$ on task $j$ in the presence of task set $S$) will reveal "toxic coalitions" invisible to pairwise analysis.

### Method: Coalition Influence Probing (CIP)
1. **Efficient higher-order estimation**: Instead of training $2^T$ models, use the Grad-TAG linearization trick (arXiv 2409.06091) -- train one base model on all tasks, then estimate the loss for any task subset via gradient-based linear approximation. Cost: one training run + O(T^2) gradient inner products.
2. **Coalition discovery**: For each target task $j$, compute $\Delta_S(j) = L_j(\theta_{-S}) - L_j(\theta_{all})$ for all subsets $S$ of size 1, 2, 3. Identify "toxic coalitions" where $\Delta_S(j)$ is large but $\sum_{i \in S} \Delta_{\{i\}}(j)$ is small (super-additive harm).
3. **Shapley-like decomposition**: Use the one-sample-fits-all framework (arXiv 2410.23808) to efficiently compute Shapley values for each task's contribution to every other task's performance.
4. **Coalition-aware mixing**: Design a data mixing strategy that breaks toxic coalitions by never co-training conflicting task groups at full weight simultaneously. This is a constrained optimization: maximize total performance subject to coalition toxicity constraints.

### Why This Is Novel
- Grad-TAG (2024) computes pairwise and higher-order affinity but only for classification tasks on graphs, never for sequential policy learning.
- Li et al. (2023) proved higher-order affinity outperforms pairwise for task grouping, but this has never been applied to robotics.
- The "toxic coalition" concept and coalition-aware mixing strategy are entirely new.

### Experimental Plan (<=1hr per experiment)
- **Dataset**: LIBERO-10 or Meta-World MT10
- **Model**: Small MLP policy or tiny LoRA VLA
- **Step 1** (20 min): Train base model on all 10 tasks
- **Step 2** (20 min): Compute Grad-TAG linearized loss estimates for all $\binom{10}{1} + \binom{10}{2} + \binom{10}{3} = 175$ subsets
- **Step 3** (20 min): Identify super-additive harm coalitions; validate top-3 by actual LOO retraining
- **Metric**: Correlation between predicted and actual LOO performance for subsets of size 2-3

### Computational Cost
- Base training: ~15 min on A6000
- Gradient features: ~5 min (Grad-TAG uses d=128 random projection)
- Subset evaluation: ~20 min (175 linear regressions)
- Validation retraining (top-3): ~15 min
- Total: ~55 min

### Success Probability: 50%
### Failure Modes
- Linearization may be inaccurate for policy networks (non-convex loss landscape)
- With only 10 tasks, higher-order effects may be weak
- Computational cost scales as $O(T^3)$ for triplets, limiting scalability beyond ~20 tasks

### Key References
- Li et al. (arXiv 2306.14009): higher-order task affinities via spectral clustering
- Grad-TAG (arXiv 2409.06091): gradient-based task affinity estimation at 3% FLOPs
- MISS (arXiv 2409.18153): non-additive set influence
- "It's a Match!" (arXiv 2301.02873): pairwise affinity scores are unreliable
- One-for-all Shapley (arXiv 2410.23808): efficient probabilistic value estimation

---

## Angle 3: Representation Fingerprinting -- Influence Without Gradients (New Method)

### Counter-Intuitive Claim
**You don't need gradients or influence functions to build a task interaction map.** Gradient-based methods (IF, LESS, Grad-TAG) are computationally expensive and fragile for large models. The VITA project already proved that frozen-backbone gradients carry no task-discriminative signal. But there's a simpler signal hiding in plain sight: **the representation geometry itself encodes task compatibility.**

If two tasks induce similar feature subspaces (measured by CKA, linear probes, or attention patterns), their data is likely to transfer positively. If they induce *orthogonal* but non-conflicting subspaces, they're independent. If they compete for the *same* feature dimensions with *different* optimal directions, they conflict. This geometric view is gradient-free, works on frozen or fine-tuned models, and scales trivially.

### Hypothesis
**H3**: A representation-based task interaction score, computed from CKA similarity of layer-wise activations conditioned on task identity, predicts actual multi-task performance changes with higher fidelity than gradient-based affinity scores (Grad-TAG, ETAP), at 10x lower computational cost.

### Method: Representation Fingerprinting (RepFinger)
1. **Task-conditioned feature extraction**: For each task $i$, collect activations $\{h^{(l)}_i\}$ at every layer $l$ of the VLA on task $i$'s validation data.
2. **Layer-wise CKA matrix**: Compute CKA$(h^{(l)}_i, h^{(l)}_j)$ for all task pairs $(i,j)$ at each layer $l$. This gives a 3D tensor $\mathcal{C} \in \mathbb{R}^{T \times T \times L}$.
3. **Conflict detection via directional analysis**: For task pairs with high CKA (shared subspace), check if the *linear readout directions* agree. Fit a linear probe $W_i$ for task $i$'s action prediction from layer $l$ features. Compute $\cos(W_i, W_j)$. High CKA + low cosine(W_i, W_j) = representation conflict (same features, different readouts). This is the precise mechanism behind negative transfer.
4. **RepFinger score**: $RF_{ij} = \sum_l \alpha_l \cdot \text{CKA}^{(l)}_{ij} \cdot \text{sign}(\cos(W^{(l)}_i, W^{(l)}_j))$ where $\alpha_l$ is a layer importance weight (learned or uniform).
5. **Downstream application**: Use RF scores as drop-in replacement for influence scores in data mixing optimization.

### Why This Is Novel
- AirRep (arXiv 2505.18513, NeurIPS 2025) uses representation-based TDA but only for single-task data valuation, not cross-task interaction.
- Rep-MTL (arXiv 2507.21049) analyzes representation-level task saliency but for dense prediction, not policy learning, and doesn't combine CKA with directional conflict detection.
- SGW-based MTL (arXiv 2410.03778) uses information bottleneck to reduce inter-task interference but doesn't provide a diagnostic tool.
- The CKA + linear probe direction combination is a new diagnostic that directly operationalizes "representation conflict" -- the mechanism most cited but never precisely measured.
- Hiratani (arXiv 2405.20236) analytically showed that high input feature similarity + low readout similarity is catastrophic for transfer. RepFinger is the *empirical operationalization* of this theoretical insight.

### Experimental Plan (<=1hr per experiment)
- **Dataset**: LIBERO-10 or AgiBot World 5-task subset
- **Model**: Pre-trained OpenVLA-7B (frozen, just extract features) or a trained small policy
- **Step 1** (15 min): Forward pass all validation data through the model, cache activations at 4-6 layers
- **Step 2** (10 min): Compute CKA matrices (mini-batch CKA, O(batch^2) per layer)
- **Step 3** (10 min): Fit linear probes per task per layer; compute readout direction cosines
- **Step 4** (10 min): Compute RepFinger scores
- **Step 5** (15 min): Validate against actual LOO performance (use cached results from Angle 1/2 or quick 5-task LOO)
- **Metric**: Spearman correlation between RepFinger scores and actual task-pair influence

### Computational Cost
- Feature extraction: ~10 min (single forward pass, no backprop)
- CKA computation: ~5 min
- Linear probes: ~10 min (simple least squares)
- Total: ~30 min, **no gradient computation needed**

### Success Probability: 55%
### Failure Modes
- CKA may not capture fine-grained differences relevant to policy performance
- Linear probes may be too simple to capture the action space conflict (robot actions are high-dimensional and multimodal)
- The method may only work well for frozen backbones (where representations are stable) but poorly for LoRA-tuned models (where representations shift)

### Key References
- AirRep (arXiv 2505.18513): representation-based TDA
- Hiratani (arXiv 2405.20236): theoretical analysis of feature similarity vs. readout similarity in transfer
- Rep-MTL (arXiv 2507.21049): representation-level task saliency
- CKA (Kornblith et al., ICML 2019): centered kernel alignment for representation comparison

---

## Recommended Synthesis: The Three Angles Are Complementary

The three angles attack different limitations of the baseline proposal:

| Angle | Limitation Addressed | Key Innovation | Risk Level |
|-------|---------------------|---------------|------------|
| 1. Temporal Influence Tomography | Task-level averaging hides phase-specific dynamics | Influence *tensor* over (task, task, phase) | Medium |
| 2. Coalition Influence Probing | Pairwise affinity misses higher-order interactions | Toxic coalition discovery + coalition-aware mixing | High |
| 3. Representation Fingerprinting | Gradient-based methods are expensive and fragile | Gradient-free geometric diagnostic | Medium |

**My strongest recommendation**: Lead with **Angle 3 (RepFinger)** as the primary diagnostic tool -- it's cheap, novel, and directly operationalizes the "representation conflict" mechanism that every paper cites but nobody measures. Use **Angle 1 (TIT)** as the temporal refinement that makes influence analysis actionable for robot manipulation specifically. Keep **Angle 2 (CIP)** as a theoretical contribution/analysis tool rather than the main method, since its computational scaling is limited.

**Proposed narrative**: "We first diagnose cross-task interactions using RepFinger (fast, gradient-free), then localize conflicts to specific trajectory phases using TIT (temporal precision), and validate that the discovered interactions are non-additive using CIP (theoretical grounding). Together, these three tools form a *multi-resolution diagnostic framework* for understanding data interactions in multi-task VLA training."

This framing elevates the paper from "influence functions for robots" to "a new diagnostic framework for multi-task policy learning" -- a much stronger contribution.

---

## Risk Assessment and Plan B

**If all three angles produce weak signals** (no clear negative transfer, no phase structure, no coalition effects):
- This itself is a publishable finding: "Negative transfer in VLA training is weaker than assumed; the real bottleneck is capacity, not interference." This would directly challenge CORAL's premise and reframe the field.
- Pivot to: "Positive Transfer Cartography" -- map which tasks actively help each other and design *amplification* strategies rather than *mitigation* strategies.

**If gradient-based methods fail (VITA redux)**:
- RepFinger (Angle 3) is the safety net -- it requires no gradients at all.
- If even representation geometry shows no task-discriminative signal, the model likely suffers from "information collapse" (LangForce, 2026), and the right intervention is architectural, not data-level.

## Additional Literature Found During Search

- **AC-ODM** (arXiv 2505.23878): Actor-critic based online data mixing for LLM pre-training, captures intra-domain interactions with reward function. Could inspire an online/adaptive version of our influence-guided mixing.
- **DGA** (arXiv 2410.02498): Dynamic Gradient Alignment for online data mixing, scalable gradient alignment with minimal overhead. Relevant as a baseline for our mixing optimization.
- **Datamodel Matching for Unlearning** (arXiv 2410.23232): Uses datamodels to predict counterfactual model outputs. The "predict output without subset" idea maps directly to our task LOO setting.
- **D3M** (arXiv 2406.16846): Data debiasing via datamodels -- isolates specific training examples driving failures. Analogous to our goal of isolating task data driving negative transfer.
