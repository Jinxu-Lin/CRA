# Diagnosing Training Data Attribution at Scale: Signal Dilution, Common Contamination, and When Representations Suffice

## Abstract

Training data attribution (TDA) methods based on influence functions systematically fail at large language model (LLM) scale, yet the field lacks a diagnostic framework to explain *why*. We decompose this failure into three distinct bottlenecks: (1) Hessian approximation error in the inverse Hessian computation, (2) signal dilution (which we term FM1)---the loss of discriminative power when computing influence in the full parameter space $\mathbb{R}^B$ ($B \sim 10^9$), and (3) common influence contamination (FM2)---the dominance of shared pre-training signals that mask task-specific attribution. We map each bottleneck to a repair mechanism: exact influence computation, representation-space attribution, and contrastive scoring, respectively. To validate this decomposition, we conduct the first comparative evaluation of representation-space TDA methods (RepSim, RepT) on the DATE-LM benchmark across three tasks (data selection, toxicity filtering, factual attribution) and design a $2 \times 2$ factorial ablation crossing {parameter, representation} space with {standard, contrastive} scoring to isolate each bottleneck's contribution. Our ablation reveals that bottleneck severity is task-dependent, with the signal dilution repair contributing {{PENDING: main_fm1 | FM1 average main effect across tasks | +5 to +15pp}} and the contamination repair contributing {{PENDING: main_fm2 | FM2 average main effect across tasks | +3 to +10pp}} in LDS improvement, and the two repairs are approximately additive (interaction {{PENDING: main_interaction | interaction magnitude relative to main effects | < 30% of min main effect}}). A LoRA vs. full fine-tuning comparison demonstrates that signal dilution is {{PENDING: fm1_nature | a general dimensionality phenomenon / a LoRA-specific artifact | expected: general}}, with the representation-space advantage {{PENDING: fm1_scaling | increasing/stable under full fine-tuning | expected: increasing}}. These findings provide practitioners with a diagnostic framework for TDA method selection based on task-specific bottleneck profiles.

## 1. Introduction

Training data attribution (TDA) answers a fundamental question about machine learning models: which training samples most influenced a given model output? For large language models (LLMs), this question is critical for safety auditing (tracing toxic or biased outputs to their training sources), regulatory compliance (e.g., data provenance under the EU AI Act), and scientific understanding of model behavior. The dominant approach, the influence function (IF), estimates the counterfactual effect of removing a training sample by computing a bilinear form over parameter-space gradients scaled by the inverse Hessian. While IF-based methods have proven effective for small-scale models, they exhibit dramatic and systematic failure at LLM scale.

The evidence for this failure is striking. Li et al. report that RepSim, a simple cosine similarity over internal model representations, achieves 96--100\% accuracy on harmful data identification for fine-tuned LLMs, while standard influence functions achieve only 0--7\%. This is not an isolated finding: five independent representation-space TDA methods---RepSim, RepT, In-the-Wild, Concept IF, and AirRep---were proposed within a 12-month period (2024--2025), each demonstrating superiority over parameter-space methods in their respective evaluation settings. Yet these methods have never been compared on a common benchmark, their relative strengths and weaknesses across tasks remain unknown, and no principled explanation exists for *why* representation-space methods succeed where parameter-space methods fail. The field currently treats the failure of parameter-space TDA as a monolithic problem---"influence functions don't work on LLMs"---obscuring the distinct mechanisms at play.

We argue that parameter-space TDA failure at LLM scale decomposes into three distinct bottlenecks, each requiring a different repair mechanism. First, **Hessian approximation error**: the gap between approximate and exact inverse Hessians introduces substantial attribution error, as demonstrated by MAGIC's exact IF achieving LDS $\sim$0.95--0.99 versus TRAK's LDS $\sim$0.06--0.24 on the same benchmarks. Second, **FM1 (signal dilution)**: in $\mathbb{R}^B$ ($B \sim 10^9$), per-sample influence scores lose discriminative power due to dimensionality---Li et al. demonstrate iHVP degeneracy under LoRA fine-tuning, and the geometric intuition from Johnson-Lindenstrauss concentration suggests that task-relevant signal occupies a tiny subspace of the full parameter space. Moving to representation space ($\mathbb{R}^d$, $d \sim 10^3$) concentrates the attribution signal in a task-relevant subspace. Third, **FM2 (common influence contamination)**: standard attribution scoring is dominated by pre-training knowledge shared across all samples, masking task-specific influence. Contrastive scoring, which subtracts base-model attribution, removes this common-mode component to isolate the task-specific differential. While each of these bottlenecks has been individually recognized in prior work, no existing framework identifies all three or tests whether they are independent. Our contribution is to make this decomposition explicit and empirically testable (Figure 1).

In this paper, we propose a diagnostic framework that decomposes LLM TDA failure into these three bottlenecks and validate it through systematic evaluation on the DATE-LM benchmark. Our core diagnostic tool is a $2 \times 2$ factorial ablation crossing \{parameter-space, representation-space\} with \{standard scoring, contrastive scoring\}, which cleanly isolates the FM1 and FM2 contributions per task. We extend this to a $2 \times 2 \times 2$ design by additionally crossing \{LoRA, full fine-tuning\} to test whether FM1 is a general dimensionality phenomenon or a LoRA-specific artifact. Our contributions are:

- **C0: Three-bottleneck diagnostic framework.** We decompose LLM-scale TDA failure into three bottlenecks---Hessian error, signal dilution (FM1), and common influence contamination (FM2)---and map each to a repair mechanism. This makes explicit the structure underlying conflicting results in the TDA literature.

- **C1: First comparative evaluation of representation-space methods on DATE-LM.** We evaluate RepSim and RepT alongside TRAK, DDA, and BM25 on all three DATE-LM tasks with standardized metrics, filling an acknowledged evaluation gap.

- **C2: Quantitative bottleneck decomposition via factorial ablation.** Our $2 \times 2$ design quantifies FM1 and FM2 main effects and their interaction per task, providing the first empirical test of whether these bottlenecks are independent and their repairs additive.

- **C3: LoRA vs. full fine-tuning test of FM1 generality.** We compare the representation-space advantage under LoRA and full fine-tuning, testing whether FM1 scales with effective parameter dimensionality or is a LoRA-specific artifact.

**Figure 1 description.** A visual abstract of the three-bottleneck diagnostic framework. The left panel illustrates the three bottlenecks stacked vertically: Hessian Error (computational approximation), FM1 Signal Dilution (high-dimensional parameter space), and FM2 Common Contamination (pre-training bias). Each bottleneck is annotated with key evidence (MAGIC LDS gap, JL concentration, DDA ablation). The right panel shows the $2 \times 2$ ablation grid mapping repair mechanisms (rows: parameter vs. representation space; columns: standard vs. contrastive scoring) to specific method instances (TRAK, TRAK-C, RepSim, RepSim-C). Arrows connect each bottleneck to the repair dimension that addresses it.

The remainder of this paper is organized as follows. Section 2 reviews existing TDA methods. Section 3 formalizes the three-bottleneck framework and describes the diagnostic methodology. Sections 4 and 5 present experimental results and discussion, respectively.

## 2. Related Work

We review existing TDA methods organized by the type of bottleneck each addresses, showing that each line of work targets at most one failure mode in isolation.

### 2.1 Parameter-Space TDA and Hessian Approximation

The influence function framework of Koh and Liang estimates the counterfactual effect of removing a training sample via the inverse Hessian-scaled gradient inner product. Scaling this to large models has driven a progression of increasingly sophisticated Hessian approximations: EK-FAC and K-FAC use Kronecker-factored approximations, TRAK employs random projection to reduce dimensionality before computing gradient similarity, and LESS introduces task-specific selection via low-rank gradient projections. Better Hessians Matter provides a systematic evaluation showing that Hessian quality consistently improves attribution accuracy, with a clear ordering $H \geq \text{GGN} \gg \text{Block-GGN} \gg \text{EK-FAC} \gg \text{K-FAC}$. Critically, this evaluation is conducted at scales below 1M parameters where FM1 and FM2 may be mild; our work tests whether their conclusion---that Hessian quality is the dominant factor---still holds at $B \sim 10^9$, where we hypothesize that FM1 and FM2 become the binding constraints.

At the extreme end, MAGIC eliminates Hessian error entirely by computing exact influence via metagradient through deterministic training, achieving LDS $\sim$0.95--0.99 on Gemma-2B fine-tuning. This demonstrates that within parameter space, Hessian quality is a major bottleneck. However, MAGIC incurs $O(N \cdot n \cdot T)$ cost, making it a diagnostic tool rather than a practical method. Our framework positions these works as addressing the Hessian bottleneck specifically; we test whether eliminating Hessian error is *sufficient* at LLM scale, or whether FM1 and FM2 remain as residual bottlenecks.

### 2.2 Representation-Space TDA

Five methods independently proposed operating in the model's internal representation space rather than parameter space. RepSim computes cosine similarity between hidden representations $h^{(l)}$ and requires only forward passes. RepT augments representations with loss gradients and introduces automatic phase-transition layer detection, achieving P@10 = 0.97--1.00 on controlled experiments. In-the-Wild uses activation differences between chosen and rejected responses for DPO alignment attribution, an inherently contrastive approach. Concept IF projects concept-level gradients through the Jacobian of intermediate representations. AirRep learns a dedicated encoder space for attribution via contrastive pre-training, departing from model-internal representations.

We note that these methods share a rough bilinear structure $\phi(z_\text{test})^\top \psi(z_\text{train})$ for method-specific encoding functions, though the analogy is imperfect (Concept IF projects back to parameter space; AirRep operates in a learned space). This observation is taxonomic rather than theoretically deep. More importantly, each method was proposed for a different task and evaluated on a different benchmark---no prior work has compared them on a common evaluation platform. We provide the first such comparison on DATE-LM.

### 2.3 Contrastive and Debiased Attribution

DDA introduces debiased differential attribution for hallucination tracing, subtracting base-model influence from fine-tuned model influence to isolate fine-tuning-specific effects. Their ablation study demonstrates that this debiasing contributes +55.2 percentage points in AUC, far exceeding the +8.71 pp contribution of denoising. In-the-Wild employs an inherently contrastive design through activation differences. These methods address what our framework identifies as FM2 (common influence contamination). We extend the contrastive principle to representation-space methods and test its generality across three DATE-LM tasks---data selection, toxicity filtering, and factual attribution---rather than the two task types (hallucination tracing and DPO alignment) on which it has been validated.

### 2.4 TDA Benchmarks

DATE-LM (NeurIPS 2025) provides the first standardized benchmark for LLM training data attribution, comprising three tasks with the Linear Datamodeling Score (LDS) as the primary metric. TrackStar and D-TRAK benchmark data attribution for diffusion models and image classifiers, respectively, but do not address the LLM-specific challenges of FM1 and FM2. DATE-LM evaluates parameter-space methods (TRAK, EK-FAC, Grad-Sim) and includes RepSim as a simple baseline, but does not systematically evaluate the full range of representation-space methods. We extend DATE-LM's coverage by adding RepT, contrastive variants (TRAK-C, RepSim-C), and a LoRA vs. full fine-tuning dimension.

**Positioning summary.** Prior work addresses individual bottlenecks in isolation: the Hessian approximation literature improves $H_\theta^{-1}$ quality, representation-space methods bypass FM1, and contrastive methods address FM2. None recognizes the three-bottleneck structure or tests the independence of these bottlenecks. Our framework provides both the decomposition and the experimental design to validate it.

## 3. Three-Bottleneck Diagnostic Framework

### 3.1 Preliminaries and Notation

Training data attribution (TDA) assigns an influence score $I(z_\text{test}, z_\text{train})$ to each training sample $z_\text{train} \in \mathcal{D}_\text{train}$ with respect to a test sample $z_\text{test}$, quantifying the effect of $z_\text{train}$ on the model's behavior at $z_\text{test}$.

The dominant approach is the influence function (IF), which approximates the leave-one-out effect via a first-order Taylor expansion:
\begin{equation}
I_\text{IF}(z_\text{test}, z_\text{train}) = \nabla_\theta \mathcal{L}(z_\text{test})^\top H_\theta^{-1} \nabla_\theta \mathcal{L}(z_\text{train}),
\label{eq:if}
\end{equation}
where $\theta \in \mathbb{R}^B$ denotes the model parameters ($B \sim 10^9$ for modern LLMs), $\mathcal{L}(z; \theta)$ is the loss on sample $z$, and $H_\theta = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta^2 \mathcal{L}(z_i; \theta)$ is the Hessian of the empirical risk. In practice, the $B \times B$ Hessian is never formed explicitly; methods such as TRAK, EK-FAC, and LESS rely on low-rank or Kronecker-factored approximations of $H_\theta^{-1}$.

We consider the standard LLM fine-tuning setting: a base (pre-trained) model with parameters $\theta_\text{base}$ is fine-tuned on a task-specific dataset $\mathcal{D}_\text{train}$ of $N$ samples to produce $\theta_\text{ft}$. Fine-tuning may employ Low-Rank Adaptation (LoRA) with rank $r$, restricting updates to a subspace of dimension $r \cdot (d_\text{model} + d_\text{ffn})$, or full fine-tuning over all $B$ parameters.

An alternative to parameter-space attribution is to operate in the model's internal representation space. We denote the hidden representation of sample $z$ at layer $l$ as $h^{(l)}(z) \in \mathbb{R}^d$, where $d \sim 10^3$. Representation-space methods compute influence via similarity in this lower-dimensional space rather than in $\mathbb{R}^B$.

### 3.2 Three-Bottleneck Framework

Parameter-space TDA exhibits dramatic failure at LLM scale: Li et al. report that RepSim achieves 96--100\% accuracy on harmful data identification while standard IF achieves only 0--7\%. We argue that this failure stems from three structurally distinct bottlenecks, each addressable by a different mechanism.

**Bottleneck 1: Hessian Approximation Error.** The gap between the approximate inverse Hessian used in practice and the true $H_\theta^{-1}$ introduces systematic error into Eq.~\ref{eq:if}. The severity of this error has been demonstrated by recent work on exact influence computation: MAGIC, which computes exact IF via metagradient without any Hessian approximation, achieves LDS $\sim$0.95--0.99 on Gemma-2B fine-tuning tasks, while TRAK achieves LDS $\sim$0.06--0.24 on the same benchmarks. Better Hessians Matter further shows a consistent quality ordering $H \geq \text{GGN} \gg \text{EK-FAC} \gg \text{K-FAC}$ in terms of attribution accuracy. Hessian error is thus a real and significant bottleneck---our framework does not dismiss it.

**Bottleneck 2: FM1 (Signal Dilution).** Even with a perfect Hessian, per-sample gradients in $\mathbb{R}^B$ suffer from a dimensionality problem that degrades attribution quality. The primary evidence is empirical: Li et al. demonstrate that inverse Hessian-vector products (iHVPs) under LoRA fine-tuning become degenerate---the iHVP directions collapse to a low-rank subspace, causing per-sample influence scores to lose discriminative power. This degeneracy is consistent with the intuition from Johnson-Lindenstrauss concentration: inner products of high-dimensional vectors concentrate around zero when $B \gg \log^2(N)$, and at $B \sim 10^9$ and $N \sim 10^4$ this ratio exceeds $10^6$. We emphasize that the JL analogy provides geometric intuition rather than a formal proof, since gradients are structured (correlated across samples sharing similar features) rather than random. The key empirical observation is that task-relevant signal occupies a small subspace of $\mathbb{R}^B$, and operating in the full parameter space yields extremely low signal-to-noise ratio.

We note an important caveat: the strongest evidence for FM1 is obtained under LoRA, where the effective parameter space is constrained to a low-rank subspace. Whether FM1 is equally severe under full fine-tuning is an open empirical question that our experiments directly address (Section 4.4).

**Bottleneck 3: FM2 (Common Influence Contamination).** Standard attribution scoring $I(z_\text{test}, z_\text{train})$ measures the *total* influence of $z_\text{train}$ on $z_\text{test}$, which is dominated by shared pre-training knowledge common to all samples. DDA's ablation study provides compelling evidence: removing their debiasing component (which addresses FM2) causes a $-55.2$ percentage point drop in AUC on hallucination tracing, compared to only $-8.71$ pp from removing denoising. Intuitively, the attribution signal is contaminated by a large common-mode component from pre-training, and the task-specific differential signal is small by comparison.

**Independence hypothesis.** The three bottlenecks arise from structurally different mechanisms: Hessian error is a *computational approximation* problem, FM1 is a *dimensionality* problem of operating in $\mathbb{R}^B$, and FM2 is a *scoring bias* problem independent of both space and Hessian quality. This structural distinctness motivates the hypothesis that they are largely independent and individually addressable. We stress that independence is an *empirical hypothesis*, not a theoretical guarantee---representation-space methods could partially address FM2 (by operating in a semantically structured space where common-mode signals are less dominant), and contrastive scoring could interact with dimensionality reduction. We test this hypothesis through a factorial ablation (Section 3.4).

A signal-processing analogy provides useful intuition for the two repair mechanisms. Moving from parameter space ($\mathbb{R}^B$) to representation space ($\mathbb{R}^d$) is loosely analogous to *matched filtering*: the model's learned representations $h^{(l)}$ project high-dimensional signals onto a task-relevant subspace, concentrating the attribution signal. Contrastive scoring, which subtracts a reference-model baseline, is loosely analogous to *differential detection*: it removes correlated noise (the common pre-training signal) to isolate the task-specific differential. We note that these analogies are informal---the formal conditions for matched filtering (known signal template, stationary noise) do not strictly hold---but they provide a useful conceptual vocabulary for understanding the complementary nature of the two repairs.

### 3.3 Repair Mechanisms

Each bottleneck maps to a specific repair mechanism. We describe these mechanisms and their instantiations in our diagnostic framework.

**FM1 Repair: Representation-Space Attribution.** To address signal dilution, we replace parameter-space gradients with internal model representations. The simplest instantiation is RepSim:
\begin{equation}
I_\text{RepSim}(z_\text{test}, z_\text{train}) = \cos\bigl(h^{(l)}(z_\text{test}),\; h^{(l)}(z_\text{train})\bigr),
\label{eq:repsim}
\end{equation}
which computes cosine similarity between hidden representations at layer $l$. Intuitively, this captures whether the model processes two samples through similar internal computations, effectively performing matched filtering by operating in the model's learned feature space ($\mathbb{R}^d$, $d \sim 10^3$) rather than the raw parameter space ($\mathbb{R}^B$, $B \sim 10^9$).

A more expressive variant is RepT, which augments representations with loss gradients:
\begin{equation}
I_\text{RepT}(z_\text{test}, z_\text{train}) = \cos\bigl(\phi_\text{RepT}(z_\text{test}),\; \phi_\text{RepT}(z_\text{train})\bigr), \quad \phi_\text{RepT}(z) = \bigl[h^{(l^*)}(z);\; \nabla_h \mathcal{L}(z)\bigr],
\label{eq:rept}
\end{equation}
where $l^*$ is a phase-transition layer detected automatically by identifying the layer at which $\|\nabla_{h^{(l)}} \mathcal{L}\|$ exhibits a sharp discontinuity (specifically, the layer with the largest relative change in gradient norm between adjacent layers), and $\nabla_h \mathcal{L}$ is the gradient of the loss with respect to the hidden representation. The gradient component introduces causal information (how representations should change to reduce loss), complementing the correlational signal in $h^{(l)}$.

As noted in Section 2.2, these representation-space methods share a rough bilinear structure $\phi(z_\text{test})^\top \psi(z_\text{train})$, which provides a useful (though imperfect) taxonomic lens.

**Correlation vs. causation caveat.** Representation-space methods measure *similarity* between internal model states, which is correlational rather than causal. High representation similarity between a test and training sample indicates that the model processes them through similar computations, but does not establish that the training sample *caused* the model's behavior on the test input. In particular, representation-space "repair" of an attribution error may sometimes constitute "replacement"---substituting a different (correlational) signal for the original (causal) influence computation, rather than genuinely correcting the parameter-space attribution. Our factorial ablation provides an empirical test: if representation-space methods achieve high LDS (which measures counterfactual prediction, a causal metric), then the correlational signal is a useful proxy for causal influence in practice, even if the theoretical grounding differs from that of influence functions.

**FM2 Repair: Contrastive Scoring.** To remove common-mode pre-training influence, we subtract base-model attribution scores from fine-tuned model scores:
\begin{equation}
I_\text{contr}(z_\text{test}, z_\text{train}) = I(z_\text{test}, z_\text{train};\; \theta_\text{ft}) - I(z_\text{test}, z_\text{train};\; \theta_\text{base}).
\label{eq:contrastive}
\end{equation}
This applies uniformly to both parameter-space and representation-space methods. For parameter-space attribution, contrastive scoring subtracts gradient similarity computed at base-model weights (as in DDA). For representation-space attribution, it subtracts representation similarity computed from the base model:
\begin{equation}
I_\text{RepSim-C}(z_\text{test}, z_\text{train}) = \cos\bigl(h_\text{ft}^{(l)}(z_\text{test}), h_\text{ft}^{(l)}(z_\text{train})\bigr) - \cos\bigl(h_\text{base}^{(l)}(z_\text{test}), h_\text{base}^{(l)}(z_\text{train})\bigr).
\label{eq:repsim_c}
\end{equation}
Intuitively, this isolates the influence attributable to fine-tuning by removing the component shared between the base and fine-tuned models. The subtraction is particularly natural for tasks where the base model has no task-specific knowledge (e.g., toxicity filtering), and less natural for tasks where the base model already possesses relevant knowledge (e.g., factual attribution). For factual attribution specifically, we predict that contrastive scoring may *hurt* performance: the base model already encodes relevant factual associations from pre-training, so subtracting base-model similarity removes genuinely informative signal alongside common-mode noise. This prediction provides a strong test of our framework---if confirmed, it demonstrates that FM2 severity is indeed task-dependent and that contrastive scoring is not a universal improvement.

**Hessian Repair: Exact IF (Diagnostic Control).** MAGIC computes exact influence via metagradient, eliminating Hessian approximation error entirely. We use MAGIC not as a proposed method but as a diagnostic upper bound: the gap between MAGIC and approximate IF (TRAK) quantifies the Hessian bottleneck contribution, while the gap between MAGIC and representation-space methods reveals the residual role of FM1 even when Hessian error is eliminated. MAGIC requires deterministic training (fixed seeds, data order, batch composition) and incurs $O(N \cdot n \cdot T)$ cost, where $T$ is the number of training steps, making it a diagnostic tool rather than a practical method.

### 3.4 Diagnostic Design: 2x2 Ablation

The core diagnostic tool is a $2 \times 2$ factorial ablation crossing the two signal-processing repair mechanisms:

| | Standard Scoring | Contrastive Scoring |
|---|---|---|
| **Parameter Space** | TRAK | TRAK-C (contrastive TRAK) |
| **Representation Space** | RepSim | RepSim-C (contrastive RepSim) |

Each cell isolates a specific combination of FM1 and FM2 status. TRAK (parameter, standard) retains both bottlenecks. Moving right to TRAK-C fixes FM2 while retaining FM1. Moving down to RepSim fixes FM1 while retaining FM2. RepSim-C (representation, contrastive) fixes both.

We quantify the contribution of each bottleneck via main effects:
\begin{equation}
\Delta_\text{FM1} = \overline{\text{LDS}}_\text{repr} - \overline{\text{LDS}}_\text{param}, \qquad \Delta_\text{FM2} = \overline{\text{LDS}}_\text{contr} - \overline{\text{LDS}}_\text{std},
\label{eq:main_effects}
\end{equation}
where the overbar denotes averaging over the other factor. The interaction term captures non-additivity:
\begin{equation}
\Xi = (I_\text{repr,contr} - I_\text{repr,std}) - (I_\text{param,contr} - I_\text{param,std}).
\label{eq:interaction}
\end{equation}
If $|\Xi|$ is small relative to the main effects, FM1 and FM2 are approximately independent bottlenecks, supporting the three-bottleneck decomposition. As a descriptive guideline, we consider $|\Xi| < 30\%$ of $\min(|\Delta_\text{FM1}|, |\Delta_\text{FM2}|)$ as indicating approximate independence. This threshold is not derived from formal theory but serves as a practical criterion: below this level, the additive model accounts for the vast majority of observed variance, and the interaction is unlikely to change method selection recommendations.

As a secondary diagnostic, we report what we term the Common-Mode Fraction (CMF):
\begin{equation}
\text{CMF} = \frac{|I_\text{std} - I_\text{contr}|}{|I_\text{std}|},
\end{equation}
which quantifies the fraction of standard attribution that is attributable to common pre-training signals rather than task-specific fine-tuning effects. A high CMF indicates that FM2 is a dominant error source for that task. (We adopt "Common-Mode Fraction" rather than the electronics term "Common-Mode Rejection Ratio" to avoid implying a formal connection to circuit-level signal processing.)

**LoRA vs. Full Fine-Tuning as a Third Dimension.** We extend the $2 \times 2$ design to a $2 \times 2 \times 2$ ablation by crossing with $\{\text{LoRA},\; \text{Full-FT}\}$. The key diagnostic metric is the *RepSim advantage*:
\begin{equation}
\text{Adv}_\text{RepSim} = \text{LDS}_\text{RepSim} - \text{LDS}_\text{TRAK},
\end{equation}
compared across fine-tuning modes. If $\text{Adv}_\text{RepSim}$ is larger under full fine-tuning than under LoRA, FM1 scales with effective dimensionality, confirming it as a general LLM-scale phenomenon. If $\text{Adv}_\text{RepSim}$ is present only under LoRA, FM1 is a LoRA-specific conditioning artifact rather than a fundamental dimensionality bottleneck.

**Figure 2 description.** The CRA diagnostic framework is visualized as a flow diagram. Three bottleneck nodes (Hessian Error, FM1, FM2) each connect to their repair mechanism (MAGIC exact IF, Representation-Space Attribution, Contrastive Scoring). These feed into the $2 \times 2 (\times 2)$ ablation grid, which produces per-task bottleneck profiles. The diagram emphasizes that each component isolates exactly one bottleneck, enabling clean causal attribution of performance differences.

## 4. Experiments

We empirically validate the three-bottleneck framework through five experiments of increasing depth, progressing from landscape characterization (benchmark) through bottleneck decomposition (2x2 ablation) to generality tests (LoRA vs. Full-FT, MAGIC feasibility, scale-up).

### 4.1 Experimental Setup

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

### 4.2 Systematic Benchmark (Experiment 1)

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

### 4.3 2x2 Ablation: Decomposing FM1 and FM2 (Experiment 2)

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

### 4.4 LoRA vs. Full Fine-Tuning (Experiment 3)

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

### 4.5 MAGIC Feasibility and Hessian Error Bound (Experiment 4)

We attempt to bound the Hessian error contribution by computing exact influence functions via MAGIC at Pythia-1B scale on DATE-LM's toxicity filtering task. MAGIC requires deterministic training and $O(N \cdot n \cdot T)$ metagradient computation, estimated at $\sim$3--5 GPU-hours per test sample. We evaluate on a subset of 5--10 test samples within a budget of 5 GPU-days.

{{PENDING: magic_feasibility | Whether MAGIC is computationally feasible at Pythia-1B scale on 48GB GPUs: requires storing $\sim$200 checkpoints ($\sim$400GB disk) and $T$ backward passes per test sample | Expected: likely infeasible at full evaluation scale; subset evaluation may be possible}}

{{PENDING: magic_results | MAGIC LDS vs TRAK LDS vs RepSim LDS on toxicity subset | Expected: MAGIC LDS 0.70-0.95 on DATE-LM (possibly lower than MAGIC's reported 0.95-0.99 due to harder evaluation)}}

If MAGIC is infeasible at Pythia-1B scale, this is itself informative: it demonstrates that exact IF remains impractical for LLM-scale TDA, and we acknowledge that FM1's contribution is established relative to approximate IF only. If feasible and MAGIC achieves LDS $\geq 0.90$, the paper reframes representation space as an efficient approximation of exact IF.

### 4.6 Scale-Up to Llama-7B (Experiment 5)

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

### 4.7 Efficiency Analysis

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

## 5. Discussion and Conclusion

### 5.1 Key Findings Summary

We proposed a diagnostic framework decomposing LLM-scale TDA failure into three independent bottlenecks---Hessian approximation error, signal dilution (FM1), and common influence contamination (FM2)---and validated this decomposition through systematic experiments on DATE-LM.

Our $2 \times 2$ ablation reveals that bottleneck severity is task-dependent: {{PENDING: bottleneck_profiles | which bottleneck dominates which task, e.g., FM2 dominates toxicity filtering while FM1 dominates data selection | Expected: task-dependent profiles with FM2 dominant on toxicity and FM1 dominant on data selection}}. The interaction term is {{PENDING: interaction_summary | small/moderate/large relative to main effects | Expected: small, supporting independence}}, {{PENDING: independence_conclusion | supporting/weakening the independence assumption of the three-bottleneck framework | Expected: supporting}}.

The systematic benchmark establishes that {{PENDING: benchmark_summary | key method ranking findings and practical recommendations | Expected: representation-space methods competitive on LDS for toxicity and data selection; no single method dominates all tasks}}. The LoRA vs. full fine-tuning comparison shows that FM1 is {{PENDING: fm1_generality | a general phenomenon / a LoRA-specific artifact | Expected: general, with larger RepSim advantage under full-FT}}.

### 5.2 Practitioner Guidance

Based on our findings, we provide the following recommendations for TDA method selection:

| Task Type | Compute Budget | Recommended Method |
|-----------|---------------|-------------------|
| Toxicity filtering | Low | {{PENDING: rec_tox_low | best low-compute method for toxicity | Expected: RepSim or RepSim-C}} |
| Toxicity filtering | Medium | {{PENDING: rec_tox_med | best medium-compute method for toxicity | Expected: RepSim-C}} |
| Data selection | Low | {{PENDING: rec_ds_low | best low-compute method for data selection | Expected: RepSim}} |
| Factual attribution | Low | {{PENDING: rec_fact_low | best low-compute method for factual | Expected: BM25 or RepSim}} |
| Any task (max accuracy) | Very high | MAGIC (if feasible) |

The general principle emerging from our analysis is that practitioners should first identify which bottlenecks are most severe for their task (using the Common-Mode Fraction (CMF) to diagnose FM2 severity and the parameter-space vs. representation-space gap to diagnose FM1 severity), then select methods that address the dominant bottleneck.

### 5.3 Limitations

We identify several limitations of our work, framed as conscious trade-offs:

1. **Probe not executed prior to full experiments.** All experimental predictions are based on theoretical analysis and indirect evidence from prior work. The actual LDS performance of representation-space methods on DATE-LM is unverified at the time of writing, creating a risk that our framework's predictions do not hold.

2. **FM1 evidence is primarily LoRA-based.** The strongest prior evidence for FM1 (Li et al.'s iHVP degeneracy analysis) is obtained entirely under LoRA fine-tuning. While we test FM1 under full fine-tuning (Experiment 3), the possibility remains that FM1 is primarily a LoRA artifact, reducing the three-bottleneck framework to two bottlenecks (Hessian + FM2) for the most common training regime.

3. **MAGIC feasibility may leave Hessian error unbounded.** If MAGIC is computationally infeasible at Pythia-1B scale, we cannot directly measure the Hessian error contribution, leaving the relative importance of the three bottlenecks incompletely resolved.

4. **Bilinear taxonomy is organizational, not theoretical.** Our observation that representation-space methods share a $\phi^\top \psi$ structure is a useful taxonomic tool but does not constitute a deep theoretical unification. Methods like Concept IF and AirRep do not cleanly fit this mold.

5. **Benchmark scope.** DATE-LM covers three tasks on two model scales (Pythia-1B, Llama-7B). Generalization to other tasks (e.g., instruction following, code generation), other model architectures (e.g., mixture-of-experts), and pre-training data attribution remains untested.

6. **LDS metric reliability.** Recent work raises concerns about LDS as a metric for TDA evaluation. If LDS does not faithfully capture attribution quality, all quantitative comparisons in our study are affected.

### 5.4 Future Work

Our framework opens several directions for future investigation:

1. **Pre-training data attribution.** Extending the three-bottleneck analysis to pre-training (rather than fine-tuning) data attribution, where FM2 correction is more challenging because there is no natural "before fine-tuning" reference model for contrastive scoring.

2. **Formal theoretical analysis of FM1.** Deriving tight bounds on the severity of signal dilution as a function of parameter dimensionality $B$, representation dimensionality $d$, training set size $N$, and fine-tuning regime (LoRA rank $r$ vs. full parameters), going beyond the JL-based approximation.

3. **Hybrid methods.** Developing methods that combine representation-space operation (addressing FM1), contrastive scoring (addressing FM2), and improved Hessian approximation in a principled way, potentially achieving the accuracy of exact IF at the cost of representation-space methods.

**Reproducibility.** We will release all code, including our RepSim-C and TRAK-C implementations, fine-tuning scripts, and evaluation pipelines, upon publication. All experiments use the publicly available DATE-LM benchmark and open-weight models (Pythia-1B, Llama-7B).

## References

{{PENDING: references | Full reference list to be compiled from citations in all sections | Include: Koh & Liang (2017), TRAK, EK-FAC, LESS, Better Hessians Matter, MAGIC, RepSim (Li et al.), RepT, In-the-Wild, Concept IF, AirRep, DDA, DATE-LM, TrackStar, D-TRAK, Johnson-Lindenstrauss lemma, LoRA, Pythia, Llama, UltraChat, ROME}}
