# Experiments

We empirically validate the three-bottleneck framework through five experiments of increasing depth, progressing from landscape characterization (benchmark) through bottleneck decomposition (2x2 ablation) to generality tests (LoRA vs. Full-FT, MAGIC feasibility, scale-up).

## 4.1 Experimental Setup

**Models.** Our primary evaluation uses Pythia-1B, which fits on a single 48GB GPU with full gradient storage. For scale-up experiments, we additionally evaluate on Llama-7B under LoRA fine-tuning.

**Benchmark.** We adopt DATE-LM (NeurIPS 2025, codebase version pinned to commit hash {{PENDING: datelm_commit | DATE-LM git commit hash used | e.g., abc1234}}), a standardized benchmark for LLM training data attribution comprising three tasks:
- *Data selection*: identifying training samples most valuable for downstream task performance on Fineweb corpus subsets.
- *Toxicity filtering*: detecting unsafe training samples in a mixture of UltraChat ($\sim$10K safe) and a small number ($<$100) of unsafe samples, in both homogeneous and heterogeneous toxicity settings.
- *Factual attribution*: tracing entity-fact associations in ROME-style knowledge editing data ($\sim$5K training, $\sim$100 test).

**Methods.** We evaluate six TDA methods spanning parameter-space, representation-space, and lexical approaches:

| Method | Space | Scoring | Source |
|--------|-------|---------|--------|
| TRAK | Parameter | Standard | DATE-LM codebase |
| Grad-Sim | Parameter | Standard | DATE-LM codebase |
| DDA (contrastive TRAK) | Parameter | Contrastive | Reimplemented |
| RepSim | Representation | Standard | Custom implementation |
| RepT | Representation | Standard | Reimplemented |
| BM25 | Lexical | Standard | DATE-LM codebase |

Additionally, we include Random as a lower bound and MAGIC (exact IF) as a diagnostic upper bound for parameter-space methods, subject to computational feasibility. For the 2x2 ablation, we construct contrastive variants of RepSim (RepSim-C, Eq.~\ref{eq:repsim_c}) and TRAK (TRAK-C).

**Metrics.** The Linear Datamodeling Score (LDS) serves as our primary metric across all tasks, measuring Spearman correlation between predicted and actual model output changes under training subset removal. Task-specific secondary metrics include AUPRC for toxicity filtering, Recall@50 and MRR for factual attribution, and P@K as a secondary ranking metric for all tasks. We report means and standard deviations over 3 random seeds.

**Fair comparison protocol.** All methods use the same fine-tuned model checkpoint ($\theta_\text{ft}$), the same base model checkpoint ($\theta_\text{base}$) for contrastive variants, and the same DATE-LM evaluation pipeline. Following DATE-LM's finding that cosine similarity consistently outperforms inner product, we apply cosine normalization to all methods. Each method receives an equivalent hyperparameter tuning budget (layer selection for RepSim, projection dimension for TRAK, etc.). To avoid hidden researcher degrees of freedom, we report results for all evaluated hyperparameter settings (e.g., both layers for RepSim) rather than only the best-performing configuration.

**Implementation details.** For RepSim, we extract hidden representations at layer $l = L/2$ (middle) and $l = L$ (last), reporting results for both layers per task to avoid hidden degrees of freedom in layer selection. RepT uses automatic phase-transition layer detection via the gradient norm discontinuity criterion described in Section 3.3. TRAK uses DATE-LM's default projection dimension ($k = 4096$). For contrastive variants (RepSim-C, TRAK-C), both $\theta_\text{ft}$ and $\theta_\text{base}$ are loaded sequentially (not simultaneously) to stay within GPU memory; contrastive scores are computed by subtracting base-model scores from fine-tuned model scores element-wise after independent computation. RepSim-C uses the same layer index for both models; TRAK-C shares the random projection matrix across both models to ensure score comparability. All cosine similarities are computed after L2-normalization of the representation vectors.

**Fine-tuning details.** For LoRA experiments, we use rank $r = 16$, $\alpha = 32$, applied to all attention projection matrices (Q, K, V, O), with learning rate $2 \times 10^{-4}$ and AdamW optimizer ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $0.01$). For full fine-tuning experiments, we use learning rate $2 \times 10^{-5}$ with the same optimizer settings. Both use cosine learning rate schedule with 10\% warmup. Training runs for 3 epochs with batch size 8. All experiments are conducted on NVIDIA RTX A6000 GPUs (48GB).

## 4.2 Systematic Benchmark (Experiment 1)

This experiment provides the first comparative evaluation of model-internal representation-space methods (RepSim, RepT) alongside parameter-space methods on the DATE-LM benchmark, directly addressing RQ2. We note that two additional representation-space approaches---Concept IF and In-the-Wild---are not included due to differences in problem formulation (Concept IF requires concept-level supervision; In-the-Wild targets DPO alignment); extending the benchmark to these methods is left to future work.

**Table 1: Main benchmark results.** Methods (rows) $\times$ tasks (columns), reporting LDS (primary) and task-specific secondary metrics. Mean $\pm$ std over 3 seeds. Bold: best; underline: second-best.

| Method | Data Selection (LDS) | Toxicity Filtering (LDS) | Toxicity (AUPRC) | Factual Attribution (LDS) | Factual (Recall@50) |
|--------|---------------------|-------------------------|------------------|--------------------------|-------------------|
| TRAK | {{PENDING: trak_ds_lds \| TRAK LDS on data selection \| 0.10-0.25}} | {{PENDING: trak_tox_lds \| TRAK LDS on toxicity \| 0.05-0.20}} | {{PENDING: trak_tox_auprc \| TRAK AUPRC \| 0.60-0.85}} | {{PENDING: trak_fact_lds \| TRAK LDS on factual \| 0.05-0.20}} | {{PENDING: trak_fact_recall \| TRAK Recall@50 \| 0.30-0.60}} |
| Grad-Sim | {{PENDING: gradsim_ds_lds \| Grad-Sim LDS on data selection \| 0.10-0.25}} | {{PENDING: gradsim_tox_lds \| Grad-Sim LDS on toxicity \| 0.05-0.15}} | {{PENDING: gradsim_tox_auprc \| Grad-Sim AUPRC \| 0.55-0.80}} | {{PENDING: gradsim_fact_lds \| Grad-Sim LDS on factual \| 0.10-0.25}} | {{PENDING: gradsim_fact_recall \| Grad-Sim Recall@50 \| 0.30-0.60}} |
| DDA | {{PENDING: dda_ds_lds \| DDA LDS on data selection \| 0.10-0.30}} | {{PENDING: dda_tox_lds \| DDA LDS on toxicity \| 0.15-0.35}} | {{PENDING: dda_tox_auprc \| DDA AUPRC \| 0.70-0.93}} | {{PENDING: dda_fact_lds \| DDA LDS on factual \| 0.10-0.25}} | {{PENDING: dda_fact_recall \| DDA Recall@50 \| 0.35-0.65}} |
| RepSim | {{PENDING: repsim_ds_lds \| RepSim LDS on data selection \| 0.15-0.35}} | {{PENDING: repsim_tox_lds \| RepSim LDS on toxicity \| 0.15-0.35}} | {{PENDING: repsim_tox_auprc \| RepSim AUPRC \| 0.80-0.99}} | {{PENDING: repsim_fact_lds \| RepSim LDS on factual \| 0.05-0.25}} | {{PENDING: repsim_fact_recall \| RepSim Recall@50 \| 0.40-0.70}} |
| RepT | {{PENDING: rept_ds_lds \| RepT LDS on data selection \| 0.20-0.40}} | {{PENDING: rept_tox_lds \| RepT LDS on toxicity \| 0.20-0.40}} | {{PENDING: rept_tox_auprc \| RepT AUPRC \| 0.85-0.99}} | {{PENDING: rept_fact_lds \| RepT LDS on factual \| 0.10-0.30}} | {{PENDING: rept_fact_recall \| RepT Recall@50 \| 0.45-0.75}} |
| BM25 | {{PENDING: bm25_ds_lds \| BM25 LDS on data selection \| 0.05-0.15}} | {{PENDING: bm25_tox_lds \| BM25 LDS on toxicity \| 0.05-0.15}} | {{PENDING: bm25_tox_auprc \| BM25 AUPRC \| 0.50-0.70}} | {{PENDING: bm25_fact_lds \| BM25 LDS on factual \| 0.10-0.30}} | {{PENDING: bm25_fact_recall \| BM25 Recall@50 \| 0.40-0.70}} |
| Random | {{PENDING: random_ds_lds \| Random LDS \| ~0.00}} | {{PENDING: random_tox_lds \| Random LDS \| ~0.00}} | {{PENDING: random_tox_auprc \| Random AUPRC \| ~0.01}} | {{PENDING: random_fact_lds \| Random LDS \| ~0.00}} | {{PENDING: random_fact_recall \| Random Recall@50 \| ~0.02}} |

**Figure 3 description.** Grouped bar chart comparing LDS across all methods on the three DATE-LM tasks. Methods grouped by space (parameter vs. representation vs. lexical). The visualization is designed to reveal whether a single method dominates across all tasks or whether the optimal method is task-dependent.

**Analysis.** {{PENDING: benchmark_analysis | Key patterns from Table 1: (1) whether representation-space methods are competitive on LDS, (2) whether method rankings vary across tasks, (3) whether P@K and LDS rankings agree or diverge, (4) whether BM25 is competitive on factual attribution | Expected: no single method dominates; RepSim competitive on toxicity but may struggle on factual; P@K vs LDS divergence on representation methods would indicate correlation-vs-causation gap}}

If results match expectations, the key finding is that TDA method effectiveness is task-dependent, with no single method dominating across all tasks. This motivates the 2x2 ablation to understand *why* methods differ across tasks. A divergence between P@K and LDS rankings for representation-space methods would quantify the "correlation vs. causation" gap discussed in Section 3.3.

**BM25 diagnostic.** BM25 serves as a non-model-based baseline that captures lexical overlap. Strong BM25 performance on a task (e.g., factual attribution, where entity names provide strong lexical signal) indicates that surface-level features suffice, reducing the need for model-internal attribution. Weak BM25 performance (e.g., toxicity filtering, where toxic patterns are semantic rather than lexical) indicates that model-internal representations capture genuinely different information.

**Failure case analysis.** Beyond aggregate metrics, we qualitatively examine cases where each method's top-5 attributed samples are clearly incorrect (e.g., attributed training samples share no semantic or topical relationship with the test sample). We report the failure rate (fraction of test samples with $\geq$3 incorrect top-5 attributions) per method and characterize common failure modes: for parameter-space methods, we expect failures to cluster around test samples with atypical gradient directions; for representation-space methods, we expect failures on samples where surface similarity diverges from causal influence.

## 4.3 2x2 Ablation: Decomposing FM1 and FM2 (Experiment 2)

This experiment directly tests the core thesis: FM1 and FM2 are independent bottlenecks that can be separately addressed. This addresses RQ3 and provides the quantitative foundation for the three-bottleneck framework (C0, C2).

**Table 2: 2x2 ablation results.** Four conditions $\times$ three tasks, with main effects and interaction. Mean $\pm$ std over 3 seeds.

| Condition | FM1 Status | FM2 Status | Data Selection (LDS) | Toxicity (LDS) | Factual (LDS) |
|-----------|-----------|-----------|---------------------|----------------|---------------|
| TRAK (param, std) | Present | Present | {{PENDING: trak_ds_2x2 \| TRAK LDS in 2x2 \| 0.10-0.25}} | {{PENDING: trak_tox_2x2 \| TRAK LDS in 2x2 \| 0.05-0.20}} | {{PENDING: trak_fact_2x2 \| TRAK LDS in 2x2 \| 0.05-0.20}} |
| TRAK-C (param, contr) | Present | Fixed | {{PENDING: trakc_ds_2x2 \| TRAK-C LDS in 2x2 \| 0.15-0.30}} | {{PENDING: trakc_tox_2x2 \| TRAK-C LDS in 2x2 \| 0.15-0.35}} | {{PENDING: trakc_fact_2x2 \| TRAK-C LDS in 2x2 \| 0.10-0.25}} |
| RepSim (repr, std) | Fixed | Present | {{PENDING: repsim_ds_2x2 \| RepSim LDS in 2x2 \| 0.15-0.35}} | {{PENDING: repsim_tox_2x2 \| RepSim LDS in 2x2 \| 0.15-0.35}} | {{PENDING: repsim_fact_2x2 \| RepSim LDS in 2x2 \| 0.05-0.25}} |
| RepSim-C (repr, contr) | Fixed | Fixed | {{PENDING: repsimc_ds_2x2 \| RepSim-C LDS in 2x2 \| 0.20-0.40}} | {{PENDING: repsimc_tox_2x2 \| RepSim-C LDS in 2x2 \| 0.25-0.45}} | {{PENDING: repsimc_fact_2x2 \| RepSim-C LDS in 2x2 \| 0.10-0.30}} |
| **$\Delta_\text{FM1}$** | | | {{PENDING: fm1_ds \| FM1 main effect on data selection \| +5 to +15pp}} | {{PENDING: fm1_tox \| FM1 main effect on toxicity \| +5 to +15pp}} | {{PENDING: fm1_fact \| FM1 main effect on factual \| +0 to +10pp}} |
| **$\Delta_\text{FM2}$** | | | {{PENDING: fm2_ds \| FM2 main effect on data selection \| +0 to +5pp}} | {{PENDING: fm2_tox \| FM2 main effect on toxicity \| +3 to +10pp}} | {{PENDING: fm2_fact \| FM2 main effect on factual \| +0 to +5pp}} |
| **$\Xi$ (interaction)** | | | {{PENDING: xi_ds \| Interaction on data selection \| small}} | {{PENDING: xi_tox \| Interaction on toxicity \| small}} | {{PENDING: xi_fact \| Interaction on factual \| small}} |
| **CMF** | | | {{PENDING: cmrr_ds \| CMF on data selection \| 0.1-0.3}} | {{PENDING: cmrr_tox \| CMF on toxicity \| 0.3-0.6}} | {{PENDING: cmrr_fact \| CMF on factual \| 0.1-0.3}} |

**Figure 4 description.** Grouped bar chart (or heatmap) showing FM1 and FM2 main effect sizes per task, with 95\% bootstrap confidence intervals. The visualization reveals whether bottleneck severity is task-dependent. The expected pattern is that FM2 dominates toxicity filtering (high common-mode contamination from pre-training language patterns) while FM1 dominates data selection (high dimensionality, low common-mode bias).

**Analysis.** {{PENDING: ablation_analysis | Detailed analysis of: (1) FM1 and FM2 main effect magnitudes and significance per task, (2) interaction term relative to main effects, (3) task-dependent bottleneck profiles, (4) CMF interpretation | Expected: FM1 and FM2 are both positive and significant on at least 2/3 tasks; interaction < 30% of min main effect; FM2 dominant on toxicity, FM1 dominant on data selection}}

Statistical significance is assessed via per-sample permutation tests (10K permutations) with bootstrap 95\% confidence intervals over 3 seeds. We apply Benjamini-Hochberg FDR correction ($q = 0.05$) across all pairwise comparisons within each task. We acknowledge that 3 seeds may be underpowered for detecting small interaction terms; if the interaction confidence intervals are wide, we increase to 5 seeds for the $2 \times 2$ conditions specifically (adding $\sim$7 GPU-days).

If the interaction term is small ($|\Xi| < 30\%$ of the minimum main effect on at least 2 of 3 tasks), the three-bottleneck framework is validated: FM1 and FM2 are approximately independent, and their repairs are additive. A large interaction would indicate that representation-space methods partially address FM2 (by operating in a semantically structured space), requiring a nuanced "partially separable failure modes" framing.

## 4.4 LoRA vs. Full Fine-Tuning (Experiment 3)

This experiment tests whether FM1 is a general LLM-scale phenomenon or a LoRA-specific artifact, directly addressing a core uncertainty in the three-bottleneck framework (C3).

**Table 3: LoRA vs. Full-FT comparison.**

| FT Mode | RepSim LDS | TRAK LDS | $\text{Adv}_\text{RepSim}$ |
|---------|-----------|---------|--------------------------|
| **Data Selection** | | | |
| LoRA ($r = 16$) | {{PENDING: repsim_lora_ds \| RepSim LDS under LoRA, data selection \| 0.15-0.35}} | {{PENDING: trak_lora_ds \| TRAK LDS under LoRA, data selection \| 0.10-0.25}} | {{PENDING: adv_lora_ds \| RepSim advantage under LoRA, data selection \| +5 to +15pp}} |
| Full-FT | {{PENDING: repsim_fullft_ds \| RepSim LDS under Full-FT, data selection \| 0.15-0.35}} | {{PENDING: trak_fullft_ds \| TRAK LDS under Full-FT, data selection \| 0.05-0.20}} | {{PENDING: adv_fullft_ds \| RepSim advantage under Full-FT, data selection \| +10 to +25pp}} |
| **Toxicity Filtering** | | | |
| LoRA ($r = 16$) | {{PENDING: repsim_lora_tox \| RepSim LDS under LoRA, toxicity \| 0.15-0.35}} | {{PENDING: trak_lora_tox \| TRAK LDS under LoRA, toxicity \| 0.05-0.20}} | {{PENDING: adv_lora_tox \| RepSim advantage under LoRA, toxicity \| +5 to +15pp}} |
| Full-FT | {{PENDING: repsim_fullft_tox \| RepSim LDS under Full-FT, toxicity \| 0.15-0.35}} | {{PENDING: trak_fullft_tox \| TRAK LDS under Full-FT, toxicity \| 0.03-0.15}} | {{PENDING: adv_fullft_tox \| RepSim advantage under Full-FT, toxicity \| +10 to +25pp}} |

**Figure 5 description.** Bar chart comparing RepSim advantage ($\text{Adv}_\text{RepSim}$) under LoRA vs. Full-FT for each task. Error bars show 95\% bootstrap CI over 3 seeds. The visualization directly tests the dimensionality prediction: if FM1 scales with effective parameter count, the bars should be taller under Full-FT.

**Analysis.** {{PENDING: lora_ft_analysis | Analysis of: (1) whether RepSim advantage is larger under Full-FT (confirming FM1 scales with dimensionality), (2) whether TRAK degrades more under Full-FT (confirming FM1 severity increases), (3) interpretation for the three-bottleneck framework | Expected: Adv_RepSim larger under Full-FT; TRAK LDS lower under Full-FT; FM1 confirmed as general phenomenon}}

If $\text{Adv}_\text{RepSim}$ is larger under full fine-tuning, this confirms that FM1 is a general dimensionality phenomenon that worsens as effective parameter count increases. Conversely, if $\text{Adv}_\text{RepSim}$ is present only under LoRA, FM1 is better understood as a conditioning artifact of LoRA's low-rank constraint, and the paper reframes it as a "LoRA-specific pathology" rather than a general LLM bottleneck.

## 4.5 MAGIC Feasibility and Hessian Error Bound (Experiment 4)

We attempt to bound the Hessian error contribution by computing exact influence functions via MAGIC at Pythia-1B scale on DATE-LM's toxicity filtering task. MAGIC requires deterministic training and $O(N \cdot n \cdot T)$ metagradient computation, estimated at $\sim$3--5 GPU-hours per test sample. We evaluate on a subset of 5--10 test samples within a budget of 5 GPU-days.

{{PENDING: magic_feasibility | Whether MAGIC is computationally feasible at Pythia-1B scale on 48GB GPUs: requires storing $\sim$200 checkpoints ($\sim$400GB disk) and $T$ backward passes per test sample | Expected: likely infeasible at full evaluation scale; subset evaluation may be possible}}

{{PENDING: magic_results | MAGIC LDS vs TRAK LDS vs RepSim LDS on toxicity subset | Expected: MAGIC LDS 0.70-0.95 on DATE-LM (possibly lower than MAGIC's reported 0.95-0.99 due to harder evaluation)}}

If MAGIC is infeasible at Pythia-1B scale, this is itself informative: it demonstrates that exact IF remains impractical for LLM-scale TDA, and we acknowledge that FM1's contribution is established relative to approximate IF only. If feasible and MAGIC achieves LDS $\geq 0.90$, the paper reframes representation space as an efficient approximation of exact IF.

## 4.6 Scale-Up to Llama-7B (Experiment 5)

To assess the generality of our findings beyond Pythia-1B, we evaluate the top-performing methods on Llama-7B with LoRA fine-tuning.

**Table 4: Scale-up results.** Selected methods on Llama-7B vs. Pythia-1B, toxicity filtering and data selection tasks.

| Method | Pythia-1B (LDS) | Llama-7B (LDS) |
|--------|----------------|----------------|
| **Data Selection** | | |
| TRAK | {{PENDING: trak_1b_ds_scale \| TRAK LDS Pythia-1B data selection \| 0.10-0.25}} | {{PENDING: trak_7b_ds \| TRAK LDS Llama-7B data selection \| 0.05-0.20}} |
| RepSim | {{PENDING: repsim_1b_ds_scale \| RepSim LDS Pythia-1B data selection \| 0.15-0.35}} | {{PENDING: repsim_7b_ds \| RepSim LDS Llama-7B data selection \| 0.15-0.35}} |
| **Toxicity Filtering** | | |
| TRAK | {{PENDING: trak_1b_tox_scale \| TRAK LDS Pythia-1B toxicity \| 0.05-0.20}} | {{PENDING: trak_7b_tox \| TRAK LDS Llama-7B toxicity \| 0.03-0.15}} |
| RepSim | {{PENDING: repsim_1b_tox_scale \| RepSim LDS Pythia-1B toxicity \| 0.15-0.35}} | {{PENDING: repsim_7b_tox \| RepSim LDS Llama-7B toxicity \| 0.15-0.35}} |

**Analysis.** {{PENDING: scaleup_analysis | Analysis of: (1) whether FM1 main effect increases with model size (higher B -> more severe signal dilution), (2) whether representation-space methods maintain their advantage at larger scale, (3) whether method rankings are consistent across scales | Expected: FM1 effect increases with model size; RepSim advantage maintained or enlarged}}

If the RepSim advantage increases from Pythia-1B to Llama-7B, this supports the dimensionality argument: larger models have more parameters, exacerbating signal dilution in $\mathbb{R}^B$ while representation dimensionality $d$ grows more slowly.

## 4.7 Efficiency Analysis

We profile all methods for computational cost to complement the accuracy comparison.

**Table 5: Efficiency comparison.** GPU-hours per 1K test samples and peak memory on Pythia-1B, DATE-LM toxicity task.

| Method | GPU-hours / 1K test | Peak Memory (GB) | LDS / GPU-hour |
|--------|--------------------|--------------------|----------------|
| RepSim | {{PENDING: repsim_time \| RepSim GPU-hours \| 0.1-0.5}} | {{PENDING: repsim_mem \| RepSim peak memory \| 3-5}} | {{PENDING: repsim_efficiency \| RepSim LDS per GPU-hour \| high}} |
| RepT | {{PENDING: rept_time \| RepT GPU-hours \| 0.5-2.0}} | {{PENDING: rept_mem \| RepT peak memory \| 5-8}} | {{PENDING: rept_efficiency \| RepT LDS per GPU-hour \| medium-high}} |
| Grad-Sim | {{PENDING: gradsim_time \| Grad-Sim GPU-hours \| 1.0-3.0}} | {{PENDING: gradsim_mem \| Grad-Sim peak memory \| 8-15}} | {{PENDING: gradsim_efficiency \| Grad-Sim LDS per GPU-hour \| low-medium}} |
| TRAK | {{PENDING: trak_time \| TRAK GPU-hours \| 2.0-5.0}} | {{PENDING: trak_mem \| TRAK peak memory \| 10-20}} | {{PENDING: trak_efficiency \| TRAK LDS per GPU-hour \| low}} |
| DDA | {{PENDING: dda_time \| DDA GPU-hours \| 4.0-10.0}} | {{PENDING: dda_mem \| DDA peak memory \| 10-20}} | {{PENDING: dda_efficiency \| DDA LDS per GPU-hour \| low-medium}} |

The expected ordering from fastest to slowest is: RepSim (forward pass only) $<$ RepT (forward + backward for $\nabla_h$) $<$ Grad-Sim (full backward) $<$ TRAK (full backward + projection) $<$ DDA ($2\times$ TRAK). If representation-space methods are both more accurate and more efficient, the practical case for adopting them is compelling regardless of the theoretical framework.
