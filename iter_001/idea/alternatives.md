# Backup Ideas for Potential Pivot

## Alternative A: "Negative Transfer Is Weaker Than Assumed" -- A Falsification Paper

### When to Pivot
Trigger: Phase 1 (Detection) finds fewer than 3 statistically significant negative transfer pairs in LIBERO-10, even after switching to Meta-World MT10 or a heterogeneous LIBERO subset.

### Framing
"Negative transfer in multi-task robot policy learning is weaker than assumed: a rigorous empirical study with pre-registered statistical tests." This directly challenges the premise of CORAL (2603.09298), Re-Mix (2408.14037), and every influence-guided data mixing method.

### Core Contribution
1. **Rigorous non-existence evidence**: C-LOTO with Bonferroni correction across multiple benchmarks (LIBERO-10, Meta-World MT10) and model scales (5M, 1B)
2. **Alternative explanation**: Performance gaps attributed to "negative transfer" in prior work are actually caused by capacity constraints (insufficient parameters for all tasks) or optimization dynamics (different effective learning rates), not data-level interference
3. **PiKE validation for robotics**: Extend PiKE's finding (little gradient conflict in large-scale pretraining) to the small-data, fine-tuning regime of robot manipulation
4. **Reframing**: The field should invest in better architectures and more data, not smarter mixing

### Estimated Effort
~10 GPU-hours (detection experiments + analysis). Much of the work is already done during the main proposal's Phase 1.

### Target Venue
CoRL 2026 (empirical focus), or a workshop paper at RSS 2026.

---

## Alternative B: Positive Transfer Cartography -- Amplifying What Works

### When to Pivot
Trigger: Negative transfer exists but is uniformly mild ($\Delta < 3\%$ for all pairs), while several task pairs show strong positive transfer ($\Delta > 10\%$).

### Framing
Instead of "mitigating negative transfer," focus on "amplifying positive transfer." Map the positive transfer landscape and design strategies to maximize it.

### Core Contribution
1. **Positive transfer map**: Which tasks actively help each other, and by how much? Which sub-skills drive the positive transfer?
2. **Universal-positive demonstrations**: Identify a small set of demonstrations that improve ALL tasks simultaneously. Show that training on this curated subset (e.g., 30% of data) matches or exceeds full-data training.
3. **Foundation demonstrations for robotics**: Analogous to foundation model pretraining data curation -- identify the minimal diverse demonstration set that provides a strong initialization for any downstream manipulation task
4. **Connection to Diversity paper** (2507.06219): Explain AgiBot World's finding that "single-embodiment can outperform multi-embodiment" through the lens of positive transfer -- the right single-embodiment data provides more positive transfer than diverse but conflicting multi-embodiment data

### Method
- Use GPTA/BCS to identify strong positive-transfer pairs
- Rank individual demonstrations by their "universal positive influence" score: $U(z) = \sum_j I(z \to \mathcal{D}_j)$
- Select top-K universal demonstrations, train on curated subset, compare to full-data and random-subset baselines

### Estimated Effort
~15 GPU-hours. Reuses the influence computation infrastructure from the main proposal.

### Target Venue
CoRL 2026 / NeurIPS 2026 (if the "foundation demonstrations" narrative is compelling).

---

## Alternative C: Architecture vs. Data -- When Does Sharing Pay Off?

### When to Pivot
Trigger: Phase 3 (Architecture ablation) shows per-task LoRA consistently beats influence-guided shared models. The contrarian's prediction is confirmed.

### Framing
"Understanding when shared-model multi-task learning justifies its complexity over simple per-task adaptation." A systematic empirical study delineating the regime boundaries.

### Core Contribution
1. **Regime map**: Along axes of (task similarity, model capacity, number of tasks, data volume), characterize when shared models beat per-task models and vice versa
2. **The crossover point**: Identify the critical task-similarity threshold below which per-task LoRA dominates, and above which shared training dominates (due to positive transfer outweighing capacity cost)
3. **Open-vocabulary argument**: The ONE scenario where shared models win is when task identity is unknown at test time. Provide the first empirical evidence for this claim.
4. **Practical guideline**: A decision tree for practitioners: "Given your task set characteristics, should you use shared training, per-task LoRA, or mixture-of-experts?"

### Method
- Systematic grid of experiments: 3 levels of task similarity (high/medium/low, selected from LIBERO subsets) x 3 model capacities (1M/5M/20M) x 2 strategies (shared + mixing vs. per-task LoRA)
- 18 cells x 5 seeds x 5 min = ~7.5 hours total
- Measure success rate, zero-shot transfer to held-out task, and computational cost

### Estimated Effort
~8 GPU-hours plus analysis.

### Target Venue
ICRA 2027 (practical robotics audience) or CoRL 2026 (if framed as a foundational empirical study).

---

## Alternative D: Instance-Level Data Valuation Supersedes Task-Level Analysis

### When to Pivot
Trigger: Phase 2 shows all task-level proxies fail ($\rho < 0.4$), but per-demonstration gradient features show clear clustering structure that crosses task boundaries.

### Framing
"Task labels are a lossy abstraction for data interactions in multi-task robot learning: instance-level valuation reveals the true structure." Builds on CUPID (2506.19121) and SCIZOR (2505.22626).

### Core Contribution
1. **Demonstration that task-level analysis fails**: Rigorous evidence (from our proxy benchmark) that task-level influence estimation is unreliable
2. **Instance-level clustering**: Gradient features cluster demonstrations by manipulation primitive (approach, grasp, lift, place), not by task label
3. **Primitive-aware data curation**: Remove harmful demonstrations identified at the instance level; show this outperforms any task-level mixing strategy
4. **Connection to STRAP**: Sub-trajectory retrieval (STRAP, 2412.15182) works because the relevant granularity is sub-trajectory, not task. Our finding provides the formal justification.

### Method
- Compute per-demonstration gradient features (LESS-style) for all 500 demonstrations across 10 tasks
- Cluster with k-means (k=5, 10, 20), measure silhouette scores against task labels vs. learned clusters
- For each cluster, compute intra-cluster influence and inter-cluster influence
- Remove bottom 20% demonstrations (by universal influence score), retrain, compare

### Estimated Effort
~6 GPU-hours. Largely reuses Phase 2 infrastructure.

### Target Venue
CoRL 2026 (robot learning focus on data curation).

---

## Pivot Decision Matrix

| Signal from Main Experiments | Primary Pivot | Secondary Pivot |
|------------------------------|--------------|-----------------|
| No significant negative transfer | Alt A (Falsification) | Alt B (Positive Transfer) |
| Negative transfer exists but proxies fail | Alt D (Instance-Level) | Alt A + C combined |
| Proxies work but mixing doesn't beat LoRA | Alt C (Architecture vs. Data) | Alt B (Positive Transfer) |
| Everything works but only on standard eval | Main paper + OOD analysis as key finding | -- |
| Everything works on both ID and OOD | Main paper as proposed | -- |

Every path produces a publishable contribution. This is the mark of a robust research program.
