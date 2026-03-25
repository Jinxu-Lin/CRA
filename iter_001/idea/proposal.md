# Multi-Resolution Diagnostic Framework for Cross-Task Data Interactions in Multi-Task VLA Training

## Title

**Diagnosing and Exploiting Cross-Task Data Interactions in Multi-Task Vision-Language-Action Models: A Multi-Resolution Framework from Representation Geometry to Influence-Guided Mixing**

## Abstract

Multi-task training of Vision-Language-Action (VLA) models combines heterogeneous manipulation task data under shared parameters, but the field lacks principled tools to understand cross-task data interactions. We propose a multi-resolution diagnostic framework that operates at three complementary granularities: (1) **Bottleneck Conflict Score (BCS)** -- a 10-minute, gradient-free representation geometry diagnostic that detects task conflicts by combining subspace overlap with readout direction alignment at the policy bottleneck layer; (2) **Gradient-Projected Task Affinity (GPTA)** -- an efficient gradient-based influence proxy validated against rigorous Leave-One-Task-Out ground truth; and (3) **Influence-Linearized Minimax Mixing** -- an LP formulation that converts the diagnosed interaction structure into provably near-optimal data mixing weights. Before deploying these tools, we conduct a falsification-first experimental program: we first verify that statistically significant negative transfer exists (factorial design with Bonferroni correction), then benchmark whether cheap proxies reliably predict it (pre-registered correlation threshold $\rho > 0.6$), and only then apply influence-guided mixing with both in-distribution and out-of-distribution evaluation. Critically, we include an architecture-vs-data ablation comparing our approach against per-task LoRA isolation to delineate the precise regime where influence-guided mixing outperforms simpler architectural solutions. Experiments use LIBERO-10 with small policies (ResNet-18 + MLP, ~5M params, 5-min training) for rapid iteration, with transferability validation on frozen OpenVLA-7B.

## Motivation

### The Problem

The 2025-2026 VLA landscape is defined by data scaling: Physical Intelligence, Google RT-2, NVIDIA GR00T, and open-source projects (Octo, OpenVLA, RDT) all combine data from N tasks, M robots, and K environments. Data mixing ratios are set by heuristics or uniform sampling, with no principled understanding of cross-task data interactions. Adding task B's data may help task A (positive transfer), hurt it (negative transfer), or have no effect -- but nobody can systematically diagnose which case applies or why.

### Why Existing Work Is Insufficient

- **Intra-task TDA** (QoQ, CUPID, DataMIL, SCIZOR): Answer "which demonstrations within task X are useful?" but never address cross-task interactions.
- **Black-box mixing** (Re-Mix, DRO): Optimize task weights without explaining the interaction structure. You get weights but no understanding.
- **Set influence theory** (MISS): Proves pairwise influence is non-additive but provides no practical diagnostic tool.
- **Architecture-based isolation** (CORAL, MergeVLA): Eliminates negative transfer by construction via per-task LoRA, but requires task identity at inference and cannot generalize to novel tasks.

### The Gap We Fill

No existing work provides a **task-to-task interaction diagnostic** for VLA that is simultaneously: (a) cheap enough for iterative experimentation, (b) grounded in formal theory, (c) validated against rigorous ground truth, and (d) actionable for data mixing optimization. Our multi-resolution framework fills all four requirements.

### Lessons from VITA

Our prior VITA project demonstrated that frozen-backbone VLA gradients carry no task-discriminative signal (SC-1 ~ 0 on RDT-1B). This project uses LoRA fine-tuning throughout, avoiding that failure mode. We also inherit reusable infrastructure: AgiBot World data processing pipeline, EK-FAC / cosine scoring code, and evaluation metric systems.

## Research Questions

1. **RQ1 (Detection):** Does statistically significant negative transfer exist in standard multi-task robot learning benchmarks (LIBERO-10), or is the phenomenon too weak to be a research target at this model/data scale?

2. **RQ2 (Measurement):** Can cheap computational proxies (gradient-based or representation-based) reliably predict the sign and magnitude of cross-task influence, as measured by rigorous Controlled Leave-One-Task-Out (C-LOTO) experiments?

3. **RQ3 (Mechanism):** When negative transfer is detected, what is the primary mechanism -- representation conflict (shared features, divergent readouts), optimization conflict (gradient misalignment), or capacity competition?

4. **RQ4 (Intervention):** Does influence-guided data mixing improve deployment performance compared to uniform mixing, DRO-based mixing, and per-task LoRA isolation? Is the improvement robust to out-of-distribution evaluation?

5. **RQ5 (Granularity):** Does sub-task or instance-level analysis capture cross-task dynamics that task-level analysis misses?

## Core Framework

### Level 1: Representation Geometry Diagnostic (BCS) -- Fastest, Gradient-Free

**Rationale.** The pragmatist's simplification of the innovator's RepFinger is the most cost-effective entry point, theoretically justified by the theoretical perspective's Spectral Interaction Theorem and grounded in Hiratani (2405.20236)'s proof that high feature overlap + low readout alignment is catastrophic.

**Method.** At the bottleneck layer (last shared layer before action prediction):
1. Extract task-conditioned activations $H_i$ for each task's validation data
2. Compute top-$k$ PCA subspace overlap: $O_{ij} = \|U_i^T U_j\|_F^2 / k$
3. Fit linear probes $W_i$ from bottleneck features to actions per task
4. **BCS score**: $BCS_{ij} = O_{ij} \cdot \text{sign}(\cos(W_i, W_j))$

**Cost**: ~10 minutes. No gradients, no backpropagation. Works on frozen or fine-tuned models.

**Theoretical backing**: The spectral interaction score $\mathcal{S}_{ij}$ predicts the actual transfer gap with error $O(d/n + 1/\text{gap}_j)$, where $\text{gap}_j$ is the spectral gap of task $j$'s feature matrix. Larger spectral gaps (clearer subspace structure) yield more accurate predictions.

### Level 2: Gradient-Projected Task Affinity (GPTA) -- Precise, Moderate Cost

**Rationale.** LESS-style gradient projection is well-validated for LLMs but unvalidated for policy learning. This is the gap we fill.

**Method.**
1. Train a small base policy on all tasks jointly
2. Collect per-sample gradient features via random projection ($R \in \mathbb{R}^{d \times k}$, $k$=256-512)
3. Task influence: $M_{ij} = \frac{1}{|\mathcal{D}_j|} \sum_{z' \in \mathcal{D}_j} \max_{z \in \mathcal{D}_i} \cos(Rg_z, Rg_{z'})$
4. Validate against C-LOTO ground truth

**Theoretical backing**: The information-theoretic decomposition shows influence decomposes into variance inflation vs. gradient alignment benefit, with estimation consistency at rate $O(\sqrt{d \log n / n} + 1/\sqrt{d})$ (Proposition 1 from theoretical perspective).

**Cost**: ~30 minutes for 10 tasks on a small model.

### Level 3: Influence-Linearized Minimax Mixing -- Actionable, Theory-Grounded

**Rationale.** Given the influence matrix from Level 1 or 2, the data mixing problem becomes a tractable LP.

**Method.**
$$\min_{w \in \Delta^{T-1}} \max_{j \in [T]} \left[c_j + \sum_k w_k M_{kj}\right]$$
where $c_j$ is the single-task baseline loss. The LP converges in $O(T \log(1/\epsilon))$ iterations and is robust to $O(1/\sqrt{n})$ estimation error in $M$.

**Cost**: < 1 second to solve. The cost is entirely in computing $M$.

### Pipeline

**Use Level 1 (BCS) for fast screening** $\to$ **Level 2 (GPTA) for precise quantification of flagged pairs** $\to$ **Level 3 (LP mixing) for optimal data composition.**

## Experimental Program: Falsification-First Design

### Phase 0: Pilot Feasibility (1 experiment task, 15 min)

- Train all-10-tasks model and a 5-task subset, 1 seed each
- Quick success-rate comparison: is there any measurable performance difference?
- **Gate**: If all differences < 3%, switch to Meta-World MT10 or deliberately heterogeneous LIBERO subset

### Phase 1: Detection -- Does Negative Transfer Exist? (RQ1)

**Protocol**: Controlled Leave-One-Task-Out (C-LOTO) with:
- 5 random seeds per configuration
- 200+ evaluation rollouts per condition
- Volume-controlled LOTO (proportional upsampling when removing a task)
- Bonferroni correction for 90 pairwise tests

**Pre-registered criterion**: At least 3 task pairs exhibit statistically significant negative transfer ($\Delta_j^{(i)} > 0$ at $p < 0.05$ after correction).

**If falsified**: Pivot to "Negative Transfer in Multi-Task Robot Learning Is Weaker Than Assumed" -- a valuable finding challenging CORAL's premise.

### Phase 2: Measurement -- Can We Estimate Influence Cheaply? (RQ2)

**Proxy benchmark** (5 candidates):
1. GradCos (gradient cosine similarity)
2. GPTA (LESS-style gradient projection)
3. BCS (bottleneck conflict score)
4. RepFinger (multi-layer CKA + readout direction)
5. Kernel surrogate (second-order task attribution)

**Pre-registered threshold**: Best proxy must achieve Spearman $\rho > 0.6$ against C-LOTO ground truth.

**Seed stability audit** (contrarian's Direction 1): Compute influence matrix with 5 different base model seeds, report Kendall-$\tau$ of rankings across seeds. If $\tau < 0.5$, flag influence estimation as unreliable.

### Phase 3: Mechanism Diagnosis (RQ3)

For the top-3 detected negative transfer pairs:
- **BCS decomposition**: Is it subspace overlap + readout conflict?
- **Gradient conflict check**: Cosine of per-task mean gradients
- **Architecture ablation** (contrarian's Direction 2): Compare shared model with influence-guided mixing vs. per-task LoRA with uniform mixing vs. per-task LoRA with influence-guided mixing
- **Instance-level check** (contrarian's Direction 3): Cluster demonstrations by gradient features across task boundaries, assess whether the emergent clusters align with task labels

### Phase 4: Intervention -- Does Mixing Actually Help? (RQ4, RQ5)

**4 mixing strategies** (5 seeds each):
1. Uniform: $w_i = 1/10$
2. Influence-guided (best proxy from Phase 2)
3. Re-Mix (DRO baseline)
4. Per-task LoRA (architectural baseline)

**Two evaluation protocols**:
- Standard (in-distribution): LIBERO default, 200 rollouts
- Perturbed (out-of-distribution): shifted object positions, rotated camera viewpoints

**Pre-registered criterion**: Influence-guided mixing must improve success rate by $\geq 5\%$ absolute over uniform mixing on standard eval. If the gap shrinks to < 2% on OOD eval, the benefit is evaluation-protocol-specific.

## Contributions (Expected)

1. **C1: Multi-Resolution Diagnostic Framework** -- BCS (10 min, gradient-free) + GPTA (30 min, gradient-based) + LP mixing (instant). First systematic toolkit for diagnosing cross-task data interactions in robot policy learning.

2. **C2: Rigorous Proxy Benchmark** -- First apples-to-apples comparison of 5 task affinity proxies against C-LOTO ground truth in robot manipulation, with seed stability analysis and pre-registered quality thresholds.

3. **C3: Influence-Guided Mixing with Architecture Ablation** -- First controlled comparison of data mixing vs. architectural isolation (per-task LoRA), delineating precisely when each approach is preferred.

4. **C4: Mechanism-Level Understanding** -- Formal decomposition of negative transfer into variance inflation (gradient noise from other tasks) and alignment benefit (implicit regularization), operationalized via BCS's subspace overlap + readout conflict decomposition.

5. **C5: Falsification-First Methodology** -- Pre-registered experimental protocol with gated phases, Bonferroni-corrected statistical tests, volume-controlled baselines, and OOD generalization evaluation. A methodological template for rigorous evaluation in multi-task robot learning.

## Computational Budget

| Phase | Experiment Tasks | Wall-Clock (single A6000) |
|-------|-----------------|--------------------------|
| Phase 0: Pilot | 1 | 15 min |
| Phase 1: Detection (C-LOTO) | 12 | ~5 hrs |
| Phase 2: Proxy Benchmark | 3 | ~1.5 hrs |
| Phase 3: Mechanism + Ablation | 6 | ~3 hrs |
| Phase 4: Mixing Comparison | 8 | ~4 hrs |
| **Total** | **30** | **~14 hrs** |

With 4 A6000 GPUs available, effective wall-clock: ~4 hours for parallelizable phases.

**Design choice**: We use ResNet-18 + MLP (~5M params, 5-min training) for the main experimental program. This enables 10+ experiments per hour and adequate statistical power (5 seeds). We validate key findings on frozen OpenVLA-7B with BCS-only (no retraining needed) as a transferability check.

## Relationship to Other Perspectives

### What We Took from Each Perspective

- **Innovator**: The multi-resolution framing (representation geometry + gradient-based + mixing optimization). We adopt RepFinger's core insight but simplify it (BCS, single layer) per the pragmatist's recommendation. The Temporal Influence Tomography (sub-skill decomposition) is reserved as a RQ5 extension.

- **Pragmatist**: The execution order, GPTA method, BCS simplification, and practical focus on LIBERO-10 with small models. The proxy benchmark structure is directly from the pragmatist's Angle 2.

- **Theorist**: The formal decomposition (variance inflation vs. alignment benefit), spectral interaction theorem justifying BCS, and minimax LP formulation for mixing optimization. The superadditivity bound (Proposition 3) guides whether higher-order coalition analysis is needed.

- **Contrarian**: The falsification-first experimental structure. The three challenges (influence fragility, architecture vs. data, task vs. instance granularity) are incorporated as explicit experimental phases rather than being assumed away. The architecture ablation (per-task LoRA comparison) and seed stability audit are directly from the contrarian.

- **Interdisciplinary**: The ecological stability analysis (eigenvalue analysis of the influence matrix) is incorporated as a diagnostic tool. The frustration index from spin glass physics is used to identify higher-order task incompatibilities. The immunological "capacity conservation" diagnostic tracks total multi-task performance to detect capacity-limited regimes.

- **Empiricist**: The entire experimental methodology -- C-LOTO, volume controls, Bonferroni correction, 200 rollouts, pre-registered thresholds, OOD evaluation protocol. The gated phase structure (Detection $\to$ Measurement $\to$ Application) is the empiricist's central contribution.

### What We Deprioritized and Why

- **Coalition Influence Probing (innovator Angle 2)**: With only 10 tasks, higher-order effects are hard to detect and validate. The theorist's superadditivity bound (Proposition 3) lets us screen for cases where coalition analysis is needed without exhaustive testing. Deferred to a follow-up if pairwise analysis proves insufficient.

- **Full factorial design (empiricist Idea 2)**: 64 configurations x 3 seeds is ~16 hours -- too expensive for the pilot phase. We use C-LOTO (11 configs x 5 seeds) as a more cost-effective ground truth, with the understanding that it estimates main effects more precisely than interactions.

- **Ecological dynamical models (interdisciplinary Analogy 4)**: Fitting Wilson-Cowan dynamics to training trajectories is novel but speculative. We incorporate the static eigenvalue analysis (stability diagnostic) but defer the full dynamical model.

- **Immunological iterative repertoire selection (interdisciplinary Analogy 3)**: Online re-estimation of mixing weights during training is computationally expensive. We use offline mixing optimization as the primary approach, with online adaptation as a stretch goal.

## Risk Assessment

### Risk 1: No Detectable Negative Transfer in LIBERO-10

**Probability**: 35%. LIBERO tasks may be too homogeneous.

**Mitigation**: Phase 0 pilot screens for signal strength. Fallback: Meta-World MT10 (known to have 30%+ interference per CMTA) or deliberately heterogeneous LIBERO subset mixing tasks from LIBERO-Spatial, LIBERO-Object, and LIBERO-Goal.

**If confirmed**: "Negative Transfer in Multi-Task Robot Learning Is Weaker Than Assumed" is itself a valuable contribution challenging CORAL's premise. Pivot to positive transfer amplification.

### Risk 2: All Proxies Fail ($\rho < 0.4$)

**Probability**: 30%. Influence estimation may be fundamentally unreliable at this scale.

**Mitigation**: The seed stability audit will distinguish "proxies are noisy but correlated" from "proxies are uncorrelated noise." If the latter, redirect to instance-level curation (CUPID/SCIZOR path) or architecture-based isolation.

**If confirmed**: "Cheap Proxies for Task Affinity Are Unreliable in Robot Policy Learning" is a publishable negative result that redirects the field.

### Risk 3: Architecture Dominates Data Mixing

**Probability**: 45%. Per-task LoRA may eliminate negative transfer more cheaply than any mixing strategy.

**Mitigation**: This is explicitly tested in Phase 3. If confirmed, the paper pivots to: "When Does Shared-Model Multi-Task Learning Justify Its Complexity?" -- delineating the open-vocabulary generalization regime where a shared model is necessary and influence-guided mixing provides value.

### Risk 4: Small-Model Results Don't Transfer to Large VLAs

**Probability**: 25%.

**Mitigation**: Validate BCS rankings (gradient-free, cheap) on frozen OpenVLA-7B. If BCS rankings are consistent between 5M and 7B model, the diagnostic generalizes. Full LOO on large models is deferred.

## Target Venue

CoRL 2026 / RSS 2026 (primary), ICRA 2027 (fallback). The falsification-first methodology and architecture-vs-data ablation make the paper defensible regardless of which hypotheses are confirmed.
