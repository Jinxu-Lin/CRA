# Paper Outline: CRA

## Metadata

### Candidate Titles
1. **Diagnosing Training Data Attribution at Scale: Signal Dilution, Common Contamination, and When Representations Suffice**
2. **Three Bottlenecks of LLM Data Attribution: A Diagnostic Framework with Systematic Evaluation**
3. **Why Parameter-Space Attribution Fails on LLMs: Decomposing Signal Dilution, Common Contamination, and Hessian Error**

### Venue & Format
- **Target venue**: NeurIPS 2026
- **Paper type**: Analysis Paper (deep understanding of existing methods/phenomena)
  - Method: ~15-20% (diagnostic framework, not a novel algorithm)
  - Experiments: ~50-55% (core contribution is systematic empirical decomposition)
  - Related Work + Intro + Conclusion: ~25-30%
- **Page limit**: 9 pages + unlimited references (NeurIPS format)
- **Mode**: PLACEHOLDER -- experiments not yet run. All experimental data marked `{{PENDING:...}}`

### Narrative Strategy
- **Strategy**: **Problem-driven**
- **Rationale**: CRA's strength is diagnosing *why* parameter-space TDA fails at LLM scale. The paper shows failure cases (RepSim 96-100% vs IF 0-7%), analyzes root causes (three bottlenecks), and validates the decomposition through controlled ablation. This is stronger than contrastive (no single dramatic baseline improvement) or insight-driven (the "insight" is a decomposition, not a single surprising observation).

### Narrative Spine

| Node | Content |
|------|---------|
| **Gap** | Parameter-space TDA systematically fails at LLM scale, but the field treats this as a single problem ("IF doesn't work on LLMs"). Five representation-space alternatives were independently proposed in 12 months, yet none are evaluated on a common benchmark or explained through a unifying lens. |
| **Insight** | The failure decomposes into three structurally distinct bottlenecks -- Hessian approximation error, FM1 (signal dilution in R^B), and FM2 (common influence contamination from pre-training) -- each addressable by a different mechanism. Existing methods each fix one bottleneck without recognizing the others. |
| **Method** | A diagnostic framework that maps each bottleneck to a repair mechanism (representation-space operation for FM1, contrastive scoring for FM2, exact IF for Hessian) and validates through a 2x2 ablation {param, repr} x {standard, contrastive} on the DATE-LM benchmark. |
| **Validation** | The 2x2 ablation decomposes FM1 and FM2 contributions per task; LoRA vs Full-FT comparison tests FM1 generality; MAGIC feasibility bounds Hessian error contribution. Any outcome pattern yields a field-clarifying result. |

**If the reviewer remembers only one thing**: Parameter-space TDA failure at LLM scale is not a single problem but three independent bottlenecks (Hessian error, signal dilution, common contamination), and the 2x2 ablation cleanly decomposes their contributions.

### Writing Order
Method (S3) -> Experiments (S4) -> Introduction (S1) -> Related Work (S2) -> Conclusion (S5) -> Abstract

### Reference Model Paper
DATE-LM (NeurIPS 2025) -- similar structure: benchmark paper with diagnostic insights, Analysis Paper type, ~55% experiments.

---

## Material Inventory

| Source Document | Key Content | Maps To |
|----------------|------------|---------|
| `problem-statement.md` S1.1 | Method landscape (parameter-space vs representation-space tables) | S2 Related Work, S3.1 Preliminaries |
| `problem-statement.md` S1.2 | One-sentence gap + three consequences | S1 Introduction P1-P2 |
| `problem-statement.md` S1.3 | Root cause analysis (three bottlenecks) | S1 P3, S3.2 Framework |
| `problem-statement.md` S1.5 | RQ1-RQ3 | S1 P4 (contribution list) |
| `problem-statement.md` S1.6 | Core assumptions + risk | S5 Limitations |
| `problem-statement.md` S2.1 | Attack angle | S1 P3-P4 |
| `method-design.md` S4 | Architecture: diagnostic framework overview | S3.2 Framework, Fig.1 |
| `method-design.md` S5 | Components A-D details | S3.3-S3.5 |
| `method-design.md` S6 | Causal argument chain | S3.2 Framework justification |
| `method-design.md` S7 | Signal-processing analogy | S3.2 or Appendix |
| `experiment-design.md` S3.1 | Experiment 1: Systematic benchmark | S4.1 |
| `experiment-design.md` S3.2 | Experiment 2: 2x2 ablation | S4.2 |
| `experiment-design.md` S3.3 | Experiment 3: LoRA vs Full-FT | S4.3 |
| `experiment-design.md` S3.4 | Experiment 4: MAGIC feasibility | S4.4 |
| `experiment-design.md` S3.5 | Experiment 5: Scale-up | S4.5 |
| `experiment-design.md` S4 | Baseline selection + justification | S4.1 setup |
| `experiment-design.md` S5 | Metric definitions | S4 setup |
| `experiment-design.md` S10 | Expected results + failure plans | S4 analysis, S5 discussion |
| `contribution.md` | C0-C3 contribution list | S1 P5 (contribution bullets) |
| `probe_result.md` | Probe NOT YET EXECUTED; Sibyl pilot (irrelevant domain) | Not directly referenced in paper |

---

## Chapter Outline

### Abstract (~5% of paper, ~0.4 page)

**Core argument**: Training data attribution for LLMs suffers from three independent bottlenecks that existing work addresses in isolation. We propose a diagnostic framework, provide the first systematic benchmark of representation-space methods on DATE-LM, and validate the decomposition through controlled ablation.

**Material mapping**: Written last. Key numbers from S4 (Tab.1 benchmark, Tab.2 ablation main effects, Fig.4 task-dependent profiles).

**Key numbers to include** (all pending):
- {{PENDING: RepSim vs TRAK LDS gap on toxicity filtering}}
- {{PENDING: FM1 main effect size (pp)}}
- {{PENDING: FM2 main effect size (pp)}}
- {{PENDING: Interaction term magnitude relative to main effects}}

---

### S1. Introduction (~15%, ~1.3 pages)

**Core argument**: Parameter-space TDA fails at LLM scale due to three distinct bottlenecks that the field has not decomposed. Five representation-space methods independently emerged but lack common evaluation. We provide the diagnostic framework and systematic benchmark.

**Transition**: Opens the paper; sets up the entire narrative arc.

**Paragraph structure**:

| P# | Function | Content | Source |
|----|----------|---------|--------|
| P1 | Context + importance | TDA is critical for LLM safety/interpretability; parameter-space methods dominate but fail at scale | `problem-statement.md` S1.1 |
| P2 | Gap + evidence | Dramatic failure: RepSim 96-100% vs IF 0-7% (Li et al.); five independent representation-space methods in 12 months; no common benchmark | `problem-statement.md` S1.2 |
| P3 | Root cause (the insight) | Three independent bottlenecks: Hessian error, FM1 (signal dilution), FM2 (common contamination). Signal-processing analogy: dimensionality reduction = matched filtering, contrastive = differential detection. Each existing method fixes one bottleneck without recognizing the others. | `problem-statement.md` S1.3, `method-design.md` S7 |
| P4 | Our approach | This paper: (1) diagnostic framework decomposing TDA failure; (2) first benchmark of representation-space methods on DATE-LM; (3) 2x2 ablation quantifying FM1/FM2 contributions; (4) LoRA vs Full-FT test of FM1 generality | `contribution.md` C0-C3, `problem-statement.md` RQ1-RQ3 |
| P5 | Contribution bullets | Enumerate C0-C3 explicitly. Optionally reference Figure 1 here. | `contribution.md` |

**Figure 1 placement**: Yes, in Introduction (after P3 or P4). Visual abstract of the three-bottleneck framework + 2x2 ablation design.

**Space estimate**: 1.3 pages (including Figure 1 if placed here)

---

### S2. Related Work (~10%, ~0.9 pages)

**Core argument**: Organize the fragmented TDA landscape by the three bottlenecks, showing each prior work addresses at most one.

**Transition from S1**: "We now review existing TDA methods through the lens of the three bottlenecks identified above."

**Subsection structure**:

**2.1 Parameter-Space TDA and Hessian Approximation**
- Classical IF (Koh & Liang), TRAK, EK-FAC, LESS
- Better Hessians Matter (H >= GGN >> EK-FAC) -- addresses Hessian bottleneck only
- MAGIC (exact IF, LDS ~0.95-0.99) -- eliminates Hessian error but at O(N*n) cost
- Differentiation: these works improve the Hessian; we show Hessian quality is necessary but not sufficient at LLM scale

**2.2 Representation-Space TDA**
- RepSim, RepT, In-the-Wild, Concept IF, AirRep
- Bilinear form phi^T psi as organizing taxonomy (not deep theoretical unification)
- Each proposed for different tasks, never compared on common benchmark
- Differentiation: we provide the first systematic comparison on DATE-LM

**2.3 Contrastive/Debiased Attribution**
- DDA (debias +55pp on hallucination tracing)
- Contrastive directions in representation space (In-the-Wild)
- Differentiation: we test contrastive scoring generality across tasks and in both parameter and representation space

**2.4 TDA Benchmarks**
- DATE-LM (NeurIPS 2025): 3 tasks, LDS metric
- TrackStar, D-TRAK
- Differentiation: we extend DATE-LM coverage to representation-space methods

**Each group ends with differentiation sentence.** Related Work follows Method (S3) because the three-bottleneck framework is needed to organize the literature meaningfully.

**Alternative placement**: If reviewers find the framework hard to grasp, move Related Work before Method. Decision deferred to P2.

**Space estimate**: 0.9 pages

---

### S3. Method: Three-Bottleneck Diagnostic Framework (~20%, ~1.8 pages)

**Core argument**: Decompose LLM TDA failure into three structurally independent bottlenecks, each with a distinct repair mechanism, and formalize the 2x2 ablation as the core diagnostic tool.

**Transition from S1**: "We now formalize the three-bottleneck framework and describe the diagnostic methodology."

**Subsection structure**:

**3.1 Preliminaries and Notation** (~0.3 pages)
- Standard IF formulation: $I(z_\text{test}, z_\text{train}) = \nabla_\theta \mathcal{L}(z_\text{test})^\top H_\theta^{-1} \nabla_\theta \mathcal{L}(z_\text{train})$
- LLM fine-tuning setting: base model $\theta_\text{base}$, fine-tuned $\theta_\text{ft}$, training set $\mathcal{D}$
- Parameter space $\mathbb{R}^B$ ($B \sim 10^9$) vs representation space $\mathbb{R}^d$ ($d \sim 10^3$)
- Source: `method-design.md` S5, `problem-statement.md` S1.1

**3.2 Three-Bottleneck Framework** (~0.6 pages)
- **Bottleneck 1: Hessian Approximation Error** -- gap between approximate and exact IF. Evidence: MAGIC LDS ~0.95 vs TRAK LDS ~0.1.
- **Bottleneck 2: FM1 (Signal Dilution)** -- per-sample gradients approximately orthogonal in $\mathbb{R}^B$ (JL concentration). Evidence: Li et al. iHVP degeneracy; RepSim >> IF on detection tasks.
- **Bottleneck 3: FM2 (Common Influence Contamination)** -- standard scoring dominated by shared pre-training signals. Evidence: DDA debias ablation -55pp.
- Independence argument: three different mechanisms (computation error, dimensionality, bias), each fixable independently.
- Source: `method-design.md` S4, S6; `problem-statement.md` S1.3

**3.3 Repair Mechanisms** (~0.5 pages)
- FM1 repair: Representation-space attribution. $I_\text{repr}(z_\text{test}, z_\text{train}) = \cos(h^{(l)}(z_\text{test}), h^{(l)}(z_\text{train}))$
- FM2 repair: Contrastive scoring. $I_\text{contr}} = I(z; \theta_\text{ft}) - I(z; \theta_\text{base})$
- Hessian repair: Exact IF (MAGIC). Used as diagnostic control, not proposed method.
- Source: `method-design.md` S5 Components A-C

**3.4 Diagnostic Design: 2x2 Ablation** (~0.4 pages)
- 2x2 matrix: {parameter, representation} x {standard, contrastive}
- Cell mapping: TRAK, TRAK-C (contrastive), RepSim, RepSim-C
- Statistical analysis: FM1 main effect, FM2 main effect, interaction term
- LoRA vs Full-FT as third dimension testing FM1 generality
- Source: `experiment-design.md` S3.2

**Space estimate**: 1.8 pages (including Fig.2 method framework diagram)

---

### S4. Experiments (~45%, ~4.0 pages)

**Core argument**: Systematic evaluation validates the three-bottleneck decomposition, reveals task-dependent bottleneck profiles, and provides first representation-space benchmark on DATE-LM.

**Transition from S3**: "We now empirically validate the three-bottleneck framework through five experiments of increasing depth."

**Presentation order** (argument order, not chronological):
1. Main benchmark (establish landscape) -- RQ2
2. 2x2 ablation (decompose bottlenecks) -- RQ3 + RQ1 core
3. LoRA vs Full-FT (test FM1 generality) -- RQ1 dimension
4. MAGIC feasibility (bound Hessian contribution) -- RQ1 control
5. Scale-up (generalization) -- robustness

**Subsection structure**:

**4.1 Experimental Setup** (~0.5 pages)
- Model: Pythia-1B (primary), Llama-7B (scale-up)
- Benchmark: DATE-LM (3 tasks: data selection, toxicity filtering, factual attribution)
- Methods: RepSim, RepT, TRAK, Grad-Sim, DDA (contrastive TRAK), BM25, MAGIC (if feasible), Random
- Metrics: LDS (primary), AUPRC (toxicity), Recall@50 + MRR (factual), P@K (secondary)
- Fair comparison protocol: same checkpoint, same evaluation pipeline, cosine normalization
- Source: `experiment-design.md` S4, S5

**4.2 Systematic Benchmark (Experiment 1)** (~0.8 pages, Tab.1 + Fig.3)
- Main comparison table: 6+ methods x 3 tasks x primary metrics
- Result: {{PENDING: method rankings per task}}
- Key finding: {{PENDING: whether representation-space methods are competitive on LDS}}
- P@K vs LDS comparison: {{PENDING: correlation vs causation gap}}
- Source: `experiment-design.md` S3.1

**4.3 2x2 Ablation: Decomposing FM1 and FM2 (Experiment 2)** (~1.0 pages, Tab.2 + Fig.4)
- 2x2 results table: {TRAK, TRAK-C, RepSim, RepSim-C} x 3 tasks
- FM1 main effect: {{PENDING: magnitude per task}}
- FM2 main effect: {{PENDING: magnitude per task}}
- Interaction term: {{PENDING: magnitude and significance}}
- CMRR per task: {{PENDING: common-mode rejection ratio}}
- Task-dependent bottleneck profile visualization (Fig.4): FM1 vs FM2 effect size per task
- Source: `experiment-design.md` S3.2

**4.4 LoRA vs Full Fine-Tuning (Experiment 3)** (~0.6 pages, Tab.3)
- RepSim advantage under LoRA vs Full-FT
- Result: {{PENDING: whether FM1 is LoRA-specific or general}}
- TRAK performance degradation under Full-FT: {{PENDING}}
- Source: `experiment-design.md` S3.3

**4.5 MAGIC Feasibility and Hessian Error Bound (Experiment 4)** (~0.4 pages)
- MAGIC feasibility at Pythia-1B: {{PENDING: feasible or infeasible}}
- If feasible: MAGIC LDS vs TRAK LDS vs RepSim LDS on toxicity subset
- If infeasible: report infeasibility analysis + acknowledge limitation
- Source: `experiment-design.md` S3.4

**4.6 Scale-Up to Llama-7B (Experiment 5)** (~0.4 pages, Tab.4)
- Selected methods on Llama-7B, toxicity + data selection
- FM1 main effect scaling with model size: {{PENDING}}
- Source: `experiment-design.md` S3.5

**4.7 Efficiency Analysis** (~0.3 pages, Tab.5)
- GPU-hours per 1K test samples, peak memory
- Cost-benefit: LDS per GPU-hour
- Source: `experiment-design.md` S7

**Space estimate**: 4.0 pages (including 3-4 tables + 2-3 figures)

---

### S5. Discussion and Conclusion (~10%, ~0.9 pages)

**Core argument**: Synthesize findings into practitioner guidance, acknowledge limitations honestly, outline future work.

**Transition from S4**: "Our experiments reveal that..."

**Subsection structure**:

**5.1 Key Findings Summary** (~0.2 pages)
- Three-bottleneck decomposition: {{PENDING: which bottleneck dominates, task-dependent profiles}}
- Representation-space benchmark: {{PENDING: practitioner recommendation table}}
- FM1 generality: {{PENDING: LoRA-specific or general}}

**5.2 Practitioner Guidance** (~0.2 pages)
- Decision table: Your Task x Compute Budget -> Recommended Method
- Source: `experiment-design.md` S6

**5.3 Limitations** (~0.3 pages)
1. Probe not executed prior to full experiment design -- all predictions may not hold (self-aware)
2. FM1 evidence from Li et al. is LoRA-only; Full-FT results may overturn FM1 thesis
3. MAGIC may be infeasible, leaving Hessian error contribution unbounded from above
4. Bilinear taxonomy is organizational, not theoretically deep
5. DATE-LM scope: 3 tasks on 2 models; generalization to other LLM settings uncertain
6. LDS metric reliability concerns (H-RF1, H-DVEmb3)
- Source: `problem-statement.md` S1.6, S2.3

**5.4 Future Work** (~0.2 pages)
- Extending to pre-training data attribution (FM2 correction requires contrastive reference)
- Formal theoretical analysis of FM1 severity bounds
- Hybrid methods combining representation-space + contrastive + improved Hessian

**Space estimate**: 0.9 pages

---

## Figure and Table Plan

| ID | Type | Content | Takeaway | Data Source | Section |
|----|------|---------|----------|-------------|---------|
| Fig.1 | Concept / Visual abstract | Three-bottleneck framework: left shows three bottlenecks stacked with evidence arrows; right shows 2x2 ablation grid mapping bottlenecks to repair mechanisms | "TDA failure is three independent problems, not one" | `method-design.md` S4 diagram | S1 (Introduction) |
| Fig.2 | Framework diagram | CRA diagnostic methodology: Component A (repr-space) / B (contrastive) / C (MAGIC) feeding into 2x2(x2) ablation | "Each component isolates one bottleneck" | `method-design.md` S4.2 | S3 (Method) |
| Fig.3 | Bar chart | Method comparison across 3 DATE-LM tasks (LDS) | "No single method dominates; space-task interaction exists" | `{{PENDING: Experiment 1 results}}` | S4.2 |
| Fig.4 | Grouped bar / heatmap | FM1 and FM2 main effect sizes per task | "Bottleneck severity is task-dependent: FM2 dominates toxicity, FM1 dominates data selection (predicted)" | `{{PENDING: Experiment 2 results}}` | S4.3 |
| Fig.5 | Bar chart or scatter | RepSim advantage (RepSim LDS - TRAK LDS) under LoRA vs Full-FT | "FM1 is / is not a LoRA artifact" | `{{PENDING: Experiment 3 results}}` | S4.4 |
| Tab.1 | Main results | Methods (rows) x Tasks (columns) x Metrics; 3-seed mean +/- std | "First representation-space benchmark on DATE-LM" | `{{PENDING: Experiment 1 results}}` | S4.2 |
| Tab.2 | 2x2 ablation | 4 conditions x 3 tasks; main effects + interaction at bottom | "FM1 and FM2 are approximately additive (predicted)" | `{{PENDING: Experiment 2 results}}` | S4.3 |
| Tab.3 | LoRA vs Full-FT | {LoRA, Full-FT} x {RepSim, TRAK} x 2 tasks; RepSim advantage row | "FM1 scales with effective dimensionality (predicted)" | `{{PENDING: Experiment 3 results}}` | S4.4 |
| Tab.4 | Scale-up | Selected methods on Llama-7B vs Pythia-1B | "Findings generalize to larger scale" | `{{PENDING: Experiment 5 results}}` | S4.6 |
| Tab.5 | Efficiency | Methods x {GPU-hours, Peak Memory, LDS/GPU-hour} | "Representation-space methods are both accurate and efficient (predicted)" | `{{PENDING: Experiment profiling}}` | S4.7 |

**Count**: 5 figures + 5 tables = 10 visual elements. Appropriate for 9-page NeurIPS paper (dense analysis paper). If space is tight, Fig.5 and Tab.4 can be moved to appendix.

---

## Narrative Consistency Check: Contribution-Evidence Alignment

| Contribution | Claim | Method Location | Experiment Location | Evidence Strength |
|-------------|-------|----------------|--------------------|--------------------|
| C0: Three-bottleneck diagnostic framework (FM1 + FM2 + Hessian error) | "TDA failure decomposes into three independent bottlenecks" | S3.2 (framework formalization) | Tab.2 (2x2 main effects), Fig.4 (task profiles), Exp 4 (MAGIC Hessian bound) | Quantitative (2x2 ANOVA) + Qualitative (task profiles). **Pending.** |
| C1: First representation-space evaluation on DATE-LM | "No prior common benchmark evaluation" | S3.3 (repair mechanisms) | Tab.1 (full benchmark), Fig.3 (method comparison) | Quantitative (multi-method, multi-task, multi-seed). **Pending.** |
| C2: 2x2 ablation verifying FM1/FM2 independence | "FM1 and FM2 repairs are approximately additive" | S3.4 (2x2 design) | Tab.2 (interaction term), CMRR | Quantitative (permutation test, bootstrap CI). **Pending.** |
| C3: LoRA vs Full-FT testing FM1 generality | "FM1 is / is not a LoRA artifact" | S3.4 (LoRA vs Full-FT dimension) | Tab.3 (RepSim advantage comparison) | Quantitative (cross-condition). **Pending.** |

**Dangling contributions**: None. Every C maps to both Method and Experiment sections.
**Dangling experiments**: Exp 5 (scale-up) supports C1 generalization but is not tied to a unique contribution -- this is acceptable (robustness check, not independent claim).

---

## Section-to-Section Transition Logic

| From | To | Transition |
|------|----|------------|
| Abstract | S1 Introduction | (Standard; abstract summarizes, intro develops) |
| S1 Introduction | S3 Method | "We now formalize the three-bottleneck framework..." |
| S3 Method | S2 Related Work | "Having established the diagnostic framework, we review existing TDA methods through this lens." (Related Work after Method -- framework needed to organize literature) |
| S2 Related Work | S4 Experiments | "We now empirically validate the three-bottleneck decomposition..." |
| S4 Experiments | S5 Discussion & Conclusion | "Our experiments reveal that..." |

**Note**: Related Work placement after Method is deliberate -- the three-bottleneck framework is needed to organize the literature meaningfully. If reviewers prefer standard ordering (Related Work before Method), adjust in P2.

---

## Space Budget Summary

| Section | Pages | % |
|---------|-------|---|
| Abstract | 0.4 | 4% |
| S1 Introduction | 1.3 | 14% |
| S2 Related Work | 0.9 | 10% |
| S3 Method | 1.8 | 20% |
| S4 Experiments | 4.0 | 45% |
| S5 Discussion & Conclusion | 0.9 | 10% |
| **Total (excl. references)** | **9.3** | **~103%** |

**Overshoot**: ~0.3 pages. Mitigation: compress S4 efficiency analysis to a single table row within Tab.1 (merge Tab.5 into appendix), or tighten Related Work. Decision deferred to P2 writing.
