# Three-Bottleneck Diagnostic Framework

## 3.1 Preliminaries and Notation

Training data attribution (TDA) assigns an influence score $I(z_\text{test}, z_\text{train})$ to each training sample $z_\text{train} \in \mathcal{D}_\text{train}$ with respect to a test sample $z_\text{test}$, quantifying the effect of $z_\text{train}$ on the model's behavior at $z_\text{test}$.

The dominant approach is the influence function (IF), which approximates the leave-one-out effect via a first-order Taylor expansion:
\begin{equation}
I_\text{IF}(z_\text{test}, z_\text{train}) = \nabla_\theta \mathcal{L}(z_\text{test})^\top H_\theta^{-1} \nabla_\theta \mathcal{L}(z_\text{train}),
\label{eq:if}
\end{equation}
where $\theta \in \mathbb{R}^B$ denotes the model parameters ($B \sim 10^9$ for modern LLMs), $\mathcal{L}(z; \theta)$ is the loss on sample $z$, and $H_\theta = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta^2 \mathcal{L}(z_i; \theta)$ is the Hessian of the empirical risk. In practice, the $B \times B$ Hessian is never formed explicitly; methods such as TRAK, EK-FAC, and LESS rely on low-rank or Kronecker-factored approximations of $H_\theta^{-1}$.

We consider the standard LLM fine-tuning setting: a base (pre-trained) model with parameters $\theta_\text{base}$ is fine-tuned on a task-specific dataset $\mathcal{D}_\text{train}$ of $N$ samples to produce $\theta_\text{ft}$. Fine-tuning may employ Low-Rank Adaptation (LoRA) with rank $r$, restricting updates to a subspace of dimension $r \cdot (d_\text{model} + d_\text{ffn})$, or full fine-tuning over all $B$ parameters.

An alternative to parameter-space attribution is to operate in the model's internal representation space. We denote the hidden representation of sample $z$ at layer $l$ as $h^{(l)}(z) \in \mathbb{R}^d$, where $d \sim 10^3$. Representation-space methods compute influence via similarity in this lower-dimensional space rather than in $\mathbb{R}^B$.

## 3.2 Three-Bottleneck Framework

Parameter-space TDA exhibits dramatic failure at LLM scale: Li et al. report that RepSim achieves 96--100\% accuracy on harmful data identification while standard IF achieves only 0--7\%. We argue that this failure stems from three structurally distinct bottlenecks, each addressable by a different mechanism.

**Bottleneck 1: Hessian Approximation Error.** The gap between the approximate inverse Hessian used in practice and the true $H_\theta^{-1}$ introduces systematic error into Eq.~\ref{eq:if}. The severity of this error has been demonstrated by recent work on exact influence computation: MAGIC, which computes exact IF via metagradient without any Hessian approximation, achieves LDS $\sim$0.95--0.99 on Gemma-2B fine-tuning tasks, while TRAK achieves LDS $\sim$0.06--0.24 on the same benchmarks. Better Hessians Matter further shows a consistent quality ordering $H \geq \text{GGN} \gg \text{EK-FAC} \gg \text{K-FAC}$ in terms of attribution accuracy. Hessian error is thus a real and significant bottleneck---our framework does not dismiss it.

**Bottleneck 2: FM1 (Signal Dilution).** Even with a perfect Hessian, per-sample gradients in $\mathbb{R}^B$ suffer from a dimensionality problem that degrades attribution quality. The primary evidence is empirical: Li et al. demonstrate that inverse Hessian-vector products (iHVPs) under LoRA fine-tuning become degenerate---the iHVP directions collapse to a low-rank subspace, causing per-sample influence scores to lose discriminative power. This degeneracy is consistent with the intuition from Johnson-Lindenstrauss concentration: inner products of high-dimensional vectors concentrate around zero when $B \gg \log^2(N)$, and at $B \sim 10^9$ and $N \sim 10^4$ this ratio exceeds $10^6$. We emphasize that the JL analogy provides geometric intuition rather than a formal proof, since gradients are structured (correlated across samples sharing similar features) rather than random. The key empirical observation is that task-relevant signal occupies a small subspace of $\mathbb{R}^B$, and operating in the full parameter space yields extremely low signal-to-noise ratio.

We note an important caveat: the strongest evidence for FM1 is obtained under LoRA, where the effective parameter space is constrained to a low-rank subspace. Whether FM1 is equally severe under full fine-tuning is an open empirical question that our experiments directly address (Section 4.4).

**Bottleneck 3: FM2 (Common Influence Contamination).** Standard attribution scoring $I(z_\text{test}, z_\text{train})$ measures the *total* influence of $z_\text{train}$ on $z_\text{test}$, which is dominated by shared pre-training knowledge common to all samples. DDA's ablation study provides compelling evidence: removing their debiasing component (which addresses FM2) causes a $-55.2$ percentage point drop in AUC on hallucination tracing, compared to only $-8.71$ pp from removing denoising. Intuitively, the attribution signal is contaminated by a large common-mode component from pre-training, and the task-specific differential signal is small by comparison.

**Independence hypothesis.** The three bottlenecks arise from structurally different mechanisms: Hessian error is a *computational approximation* problem, FM1 is a *dimensionality* problem of operating in $\mathbb{R}^B$, and FM2 is a *scoring bias* problem independent of both space and Hessian quality. This structural distinctness motivates the hypothesis that they are largely independent and individually addressable. We stress that independence is an *empirical hypothesis*, not a theoretical guarantee---representation-space methods could partially address FM2 (by operating in a semantically structured space where common-mode signals are less dominant), and contrastive scoring could interact with dimensionality reduction. We test this hypothesis through a factorial ablation (Section 3.4).

A signal-processing analogy provides useful intuition for the two repair mechanisms. Moving from parameter space ($\mathbb{R}^B$) to representation space ($\mathbb{R}^d$) is loosely analogous to *matched filtering*: the model's learned representations $h^{(l)}$ project high-dimensional signals onto a task-relevant subspace, concentrating the attribution signal. Contrastive scoring, which subtracts a reference-model baseline, is loosely analogous to *differential detection*: it removes correlated noise (the common pre-training signal) to isolate the task-specific differential. We note that these analogies are informal---the formal conditions for matched filtering (known signal template, stationary noise) do not strictly hold---but they provide a useful conceptual vocabulary for understanding the complementary nature of the two repairs.

## 3.3 Repair Mechanisms

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

## 3.4 Diagnostic Design: 2x2 Ablation

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
