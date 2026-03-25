# Methodology: Multi-Resolution Diagnostic Framework for Cross-Task Data Interactions

## Overview

We design a falsification-first experimental program to diagnose, quantify, and exploit cross-task data interactions in multi-task Vision-Language-Action (VLA) training. The program is structured in four gated phases, each with pre-registered success/failure criteria that determine the next step.

## Platform and Model

- **Benchmark**: LIBERO-10 (10 tabletop manipulation tasks with diverse goals)
- **Policy architecture**: ResNet-18 encoder + 2-layer MLP action head (~5M parameters)
- **Training**: LoRA fine-tuning (rank=8) on all shared layers; avoids frozen-backbone pitfalls identified in VITA
- **Framework**: PyTorch + HuggingFace Transformers + robomimic/libero codebase
- **Hardware**: 4x NVIDIA A6000 (48GB each)

## Phase 0: Pilot Feasibility

**Goal**: Verify that measurable performance differences exist when varying task composition.

**Setup**:
- Train a 10-task joint model (all LIBERO-10 tasks, uniform mixing)
- Train a 5-task subset model (tasks 0-4 only, same total steps via upsampling)
- 1 seed (42), 100 evaluation rollouts per task
- ResNet-18 + MLP, ~5 min training per model

**Gate**: If all per-task success rate differences between 10-task and 5-task models are < 3%, switch benchmark to Meta-World MT10 or deliberately heterogeneous LIBERO subset (cross-suite mixing).

**Metrics**: Per-task success rate, aggregate success rate, training loss curves.

## Phase 1: Detection -- Controlled Leave-One-Task-Out (C-LOTO)

**Goal**: Establish ground truth for cross-task influence via rigorous controlled experiments (RQ1).

**Protocol**:
1. Train the full 10-task model (baseline)
2. For each task $i \in \{1, ..., 10\}$, train a Leave-One-Out model excluding task $i$ (volume-controlled: proportionally upsample remaining tasks to match total data volume)
3. 5 random seeds per configuration (seeds: 42, 123, 456, 789, 1024)
4. 200 evaluation rollouts per task per seed
5. Compute transfer effect: $\Delta_j^{(i)} = \text{SR}_j^{\text{full}} - \text{SR}_j^{\text{LOO}_i}$ for all $j \neq i$
6. Statistical test: paired t-test with Bonferroni correction for 90 pairwise comparisons

**Pre-registered criterion**: At least 3 task pairs show $\Delta_j^{(i)} > 0$ at $p < 0.05$ after Bonferroni correction.

**Computational note**: 11 configurations x 5 seeds = 55 training runs. Each run ~5 min = ~4.6 hours total. With 4 GPUs: ~1.2 hours wall-clock.

## Phase 2: Proxy Benchmark -- Cheap Influence Estimation (RQ2)

**Goal**: Evaluate 5 candidate proxies against C-LOTO ground truth.

**Proxies**:
1. **GradCos**: Cosine similarity of per-task mean gradients at convergence
2. **GPTA**: LESS-style gradient projection ($k$=512 random projection dimensions, per-sample gradient features, max-cosine aggregation)
3. **BCS**: Bottleneck Conflict Score (PCA subspace overlap x readout alignment at the last shared layer, top-$k$=20 principal components)
4. **RepFinger**: Multi-layer CKA (layers 1, 3, 5, final) + readout direction cosine
5. **Kernel Surrogate**: Second-order task attribution via EK-FAC approximation

**Validation**: Spearman rank correlation $\rho$ between each proxy's 90-element influence vector and C-LOTO ground truth $\Delta$ vector.

**Pre-registered threshold**: Best proxy $\rho > 0.6$.

**Seed stability audit**: Compute each proxy with 5 different base model seeds, report Kendall-$\tau$ of pairwise rankings across seeds. Flag if $\tau < 0.5$.

## Phase 3: Mechanism Diagnosis (RQ3)

**Goal**: Understand *why* negative transfer occurs for the top-3 detected pairs.

**Analyses**:
1. **BCS decomposition**: Separate subspace overlap ($O_{ij}$) from readout alignment ($\cos(W_i, W_j)$). Plot scatter of overlap vs. alignment colored by LOO effect sign.
2. **Gradient conflict**: Per-task mean gradient cosine at bottleneck layer. Check if gradient conflict correlates with representation conflict.
3. **Architecture ablation**: Compare 4 configurations:
   - Shared model + uniform mixing
   - Shared model + influence-guided mixing (best proxy from Phase 2)
   - Per-task LoRA + uniform mixing
   - Per-task LoRA + influence-guided mixing
4. **Instance-level clustering**: k-means (k=5, 10, 20) on gradient features across all tasks. Measure silhouette score for task labels vs. emergent clusters.
5. **Ecological stability**: Eigenvalue spectrum of 10x10 influence matrix. Check May stability criterion $\sigma(M) \cdot \sqrt{T \cdot C} > 1$.

## Phase 4: Intervention -- Mixing Optimization (RQ4, RQ5)

**Goal**: Test whether influence-guided mixing improves deployment performance.

**4 mixing strategies** (5 seeds each):
1. Uniform: $w_i = 1/10$
2. Influence-guided: minimax LP using best proxy's influence matrix
3. Re-Mix (DRO baseline): distributionally robust optimization over task weights
4. Per-task LoRA (architectural baseline): separate LoRA adapters per task

**Evaluation protocols**:
- **In-distribution**: Standard LIBERO-10 evaluation, 200 rollouts per task
- **Out-of-distribution**: Perturbed object positions (+/- 2cm), rotated camera (+/- 15 degrees)

**Pre-registered criterion**: Influence-guided mixing improves success rate by $\geq$ 5% absolute over uniform on in-distribution. If gap < 2% on OOD, the benefit is protocol-specific.

**Transferability check**: Compute BCS on frozen OpenVLA-7B (no retraining) for the same LIBERO-10 tasks. Compare BCS rankings between 5M model and 7B model (Spearman $\rho$). This validates whether the diagnostic generalizes to large VLAs.

## Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Single-task | Train each task independently | Upper bound for no interference |
| Uniform multi-task | All 10 tasks, $w_i = 1/10$ | Default practice baseline |
| Re-Mix (DRO) | Distributionally robust mixing | State-of-art mixing baseline |
| Per-task LoRA | Separate LoRA per task, shared backbone | Architecture isolation baseline |

## Metrics

- **Primary**: Per-task success rate (binary: task completed within time limit)
- **Aggregate**: Mean success rate across 10 tasks, worst-case task success rate
- **Transfer**: $\Delta_j^{(i)}$ = success rate change when adding/removing task $i$ for task $j$
- **Proxy quality**: Spearman $\rho$ vs. C-LOTO, Kendall $\tau$ across seeds
- **Mechanism**: Subspace overlap $O_{ij}$, readout alignment $\cos(W_i, W_j)$, gradient cosine

## Computational Budget

| Phase | Tasks | GPU-Hours | Wall-Clock (4x A6000) |
|-------|-------|-----------|----------------------|
| Phase 0: Pilot | 2 | 0.25 | 15 min |
| Phase 1: C-LOTO | 55 runs | 4.6 | 1.2 hrs |
| Phase 2: Proxies | 5 proxies | 1.5 | 30 min |
| Phase 3: Mechanism | 4 configs + analysis | 3 | 1 hr |
| Phase 4: Mixing | 4 strategies x 5 seeds | 4 | 1 hr |
| **Total** | | **~13.4** | **~4 hrs** |

## Expected Visualizations

- **Architecture diagram**: Overall multi-resolution framework pipeline (BCS -> GPTA -> LP mixing)
- **Table 1**: C-LOTO ground truth matrix (10x10 heatmap of $\Delta_j^{(i)}$ values)
- **Figure 1**: BCS scatter plot (subspace overlap vs. readout alignment, colored by transfer sign)
- **Table 2**: Proxy benchmark results (5 proxies x correlation metrics)
- **Figure 2**: Proxy calibration plot (predicted influence vs. observed C-LOTO effect)
- **Figure 3**: Architecture ablation (grouped bar chart: 4 configs x 10 tasks)
- **Table 3**: Main mixing results (4 strategies x in-dist + OOD success rates)
- **Figure 4**: Per-task improvement breakdown (influence-guided vs. uniform, sorted by baseline performance)
- **Figure 5**: Eigenvalue spectrum of influence matrix with May stability threshold annotation
- **Figure 6**: Instance-level gradient clusters (t-SNE visualization colored by task vs. emergent cluster)

## Reproducibility

- All seeds: [42, 123, 456, 789, 1024]
- LIBERO-10 standard task suite (10 tasks from LIBERO benchmark)
- Training hyperparameters: lr=3e-4, batch_size=auto-detect, LoRA rank=8, optimizer=AdamW
- Evaluation: deterministic action selection (no sampling noise)
- Code, configs, and checkpoints will be released
