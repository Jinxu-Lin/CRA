# Testable Hypotheses

## Primary Hypotheses

### H1: Existence of Negative Transfer
**Statement**: In LIBERO-10 with a ResNet-18+MLP policy (~5M params), at least 3 task pairs out of 45 exhibit statistically significant negative transfer, measured as $\Delta_j^{(i)} > 0$ at $p < 0.05$ after Bonferroni correction under the Controlled LOTO (C-LOTO) protocol (5 seeds, 200 rollouts, volume-controlled).

**Expected outcome**: Negative transfer concentrates in task pairs that share visual scenes but require different manipulation strategies (same environment, different goals). Tasks from the same LIBERO suite show weaker negative transfer than cross-suite pairs.

**Falsification**: Fewer than 3 significant pairs after correction. This implies negative transfer is too weak to be a research target at this model/data scale.

**Pilot focus**: Train 10-task model vs. 5-task subset, check if any task shows $>$5% success rate change.

### H2: Proxy Reliability
**Statement**: At least one cheap proxy among {GradCos, GPTA, BCS, RepFinger, Kernel Surrogate} achieves Spearman $\rho > 0.6$ with C-LOTO ground truth on LIBERO-10.

**Expected outcome**: BCS and GPTA will outperform GradCos, with BCS winning at smallest sample sizes (gradient-free advantage) and GPTA winning with adequate data. Theory predicts the ranking: Fisher-metric influence > GPTA > GradCos > RepCKA (depending on spectral gap and sample size).

**Falsification**: All proxies score $\rho < 0.4$. This implies task-level influence estimation is unreliable for multi-task robot policy learning at this scale.

**Pilot focus**: Compute BCS and GPTA on a 5-task subset, correlate with 5 quick LOO runs.

### H3: BCS Mechanism Detection
**Statement**: The Bottleneck Conflict Score (BCS) correctly identifies the representation conflict mechanism: task pairs with high subspace overlap ($O_{ij} > 0.5$) AND low readout alignment ($|\cos(W_i, W_j)| < 0.3$) exhibit significantly stronger negative transfer than pairs with low subspace overlap, controlling for other factors.

**Expected outcome**: The subspace overlap $\times$ readout conflict decomposition explains $>$50% of the variance in C-LOTO $\Delta$ values, directly operationalizing Hiratani's (2405.20236) theoretical prediction.

**Falsification**: The BCS components (overlap, readout alignment) are not individually predictive of transfer direction. This implies the linear readout model assumption is too strong for VLA action spaces.

**Pilot focus**: Compute BCS components for all task pairs, visualize scatter plot of overlap vs. readout alignment colored by LOO effect sign.

### H4: Mixing Effectiveness
**Statement**: Influence-guided data mixing (using the best proxy from H2) improves standard LIBERO evaluation success rate by $\geq$5% absolute over uniform mixing.

**Expected outcome**: The improvement is largest for tasks that suffered the most negative transfer under uniform mixing. Per-task breakdown shows influence-guided mixing helps "victim" tasks while marginally hurting tasks that benefited from diverse co-training.

**Falsification**: Improvement $<$2% on both in-distribution and out-of-distribution evaluation. This implies influence-guided mixing is not practically useful at this scale.

**Pilot focus**: Quick 2-task mixing ablation on the strongest negative transfer pair from H1.

## Secondary Hypotheses

### H5: Architecture vs. Data
**Statement**: Per-task LoRA with uniform mixing matches or beats influence-guided mixing on a shared architecture for in-distribution evaluation. However, the shared model with influence-guided mixing outperforms per-task LoRA on a zero-shot novel task evaluation (where the task router has no target task data).

**Expected outcome**: Architecture dominates for known tasks; data mixing matters for open-vocabulary generalization. The marginal gain of influence-guided mixing on top of per-task LoRA is $<$2%.

**Falsification**: Influence-guided shared model beats per-task LoRA even for known tasks. This would be surprising and highly publishable.

**Pilot focus**: Train per-task LoRA baseline alongside shared model; compare on 2 tasks.

### H6: Seed Stability of Influence Estimates
**Statement**: Influence matrix rankings are moderately stable across base model training seeds, with Kendall-$\tau > 0.5$ for the top-5 positive and top-5 negative task pairs across 5 seeds.

**Expected outcome**: Overall Kendall-$\tau$ is 0.4-0.6 (moderate stability). Extreme positive and extreme negative pairs are more stable than near-zero pairs. BCS rankings are more stable than gradient-based rankings (because BCS operates on representation geometry, which is more consistent across seeds than gradient directions).

**Falsification**: $\tau < 0.3$ for all methods. This validates the contrarian's fragility concern and implies influence-based approaches need seed-averaging protocols.

**Pilot focus**: Train 3 base models with different seeds, compute BCS for each, measure rank correlation.

### H7: Instance-Level vs. Task-Level
**Statement**: Per-demonstration influence clustering (k-means on gradient features across all tasks) reveals clusters that do NOT align with task boundaries. Demonstrations from different tasks cluster together when they share manipulation primitives (approach, grasp, etc.).

**Expected outcome**: Removing the bottom 20% of demonstrations (ranked by per-sample influence) improves multi-task performance by more than any task-level mixing strategy, consistent with CUPID's finding that 33% data removal can maintain performance.

**Falsification**: Gradient feature clusters align tightly with task labels (silhouette score > 0.7 for task-label clustering). This implies task-level is the correct granularity.

**Pilot focus**: Compute gradient features for 200 demonstrations across 5 tasks, cluster with k=5 and k=10, measure alignment with task labels.

### H8: Ecological Stability Threshold
**Statement**: The multi-task influence matrix exhibits a May-type stability threshold: when $\sigma(M) \cdot \sqrt{T \cdot C} > 1$ (where $\sigma(M)$ is the standard deviation of influence scores, $T$ is task count, $C$ is the fraction of significantly non-zero interactions), multi-task performance degrades sharply.

**Expected outcome**: The eigenvalue spectrum of the influence matrix has a clear transition -- most eigenvalues are negative (stable) with a few positive ones corresponding to conflicting task clusters. The dominant positive eigenvector identifies the "axis of conflict."

**Falsification**: No stability transition is observed when varying $T$ from 3 to 10 tasks. This implies the ecological analogy is too simplistic for the non-equilibrium dynamics of neural network training.

**Pilot focus**: Compute eigenvalue spectrum of the 10x10 influence matrix; check if positive eigenvalues correspond to known conflicting pairs.

## Hypothesis Priority for Pilot Experiments

| Priority | Hypothesis | Pilot Time | Pilot Task |
|----------|-----------|------------|------------|
| 1 | H1 (Existence) | 15 min | Quick LOO on 5 tasks |
| 2 | H3 (BCS Mechanism) | 15 min | BCS computation + visualization |
| 3 | H2 (Proxy Reliability) | 30 min | BCS + GPTA vs. 5 LOO runs |
| 4 | H6 (Seed Stability) | 20 min | 3-seed BCS comparison |
| 5 | H7 (Instance vs. Task) | 20 min | Gradient clustering on 200 demos |

Total pilot budget: ~100 min on single A6000. Fits within 1-2 experiment tasks.
