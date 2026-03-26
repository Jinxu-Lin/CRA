# Notation Table: CRA

> Unified notation for all mathematical symbols and abbreviations used in the paper.
> Domain conventions: $\theta$ for parameters, $\mathcal{L}$ for loss, $\mathcal{D}$ for dataset.

## Mathematical Symbols

| Symbol | Meaning | First Appearance |
|--------|---------|-----------------|
| $\theta$ | Model parameters (generic) | S3.1 Preliminaries |
| $\theta_\text{base}$ | Base (pre-trained) model parameters | S3.1 |
| $\theta_\text{ft}$ | Fine-tuned model parameters | S3.1 |
| $B$ | Total number of model parameters ($\theta \in \mathbb{R}^B$, $B \sim 10^9$) | S3.1 |
| $d$ | Hidden representation dimensionality ($h^{(l)} \in \mathbb{R}^d$, $d \sim 10^3$) | S3.1 |
| $L$ | Total number of layers in the model | S3.1 |
| $l$ | Layer index ($1 \leq l \leq L$) | S3.1 |
| $l^*$ | Phase-transition layer (RepT automatic detection) | S3.3 |
| $\mathcal{D}$ | Training dataset | S3.1 |
| $\mathcal{D}_\text{train}$ | Training set | S3.1 |
| $z$ | A data sample (input-output pair) | S3.1 |
| $z_\text{test}$ | Test sample for which attribution is computed | S3.1 |
| $z_\text{train}$ | Training sample being attributed | S3.1 |
| $N$ | Number of training samples ($|\mathcal{D}_\text{train}|$) | S3.1 |
| $n$ | Number of test samples | S3.1 |
| $\mathcal{L}(z; \theta)$ | Loss on sample $z$ at parameters $\theta$ | S3.1 |
| $\nabla_\theta \mathcal{L}$ | Gradient of loss w.r.t. model parameters | S3.1 |
| $H_\theta$ | Hessian matrix of the loss at $\theta$ ($H_\theta \in \mathbb{R}^{B \times B}$) | S3.1 |
| $H_\theta^{-1}$ | Inverse Hessian (or approximation thereof) | S3.1 |
| $I(z_\text{test}, z_\text{train})$ | Influence score of $z_\text{train}$ on $z_\text{test}$ (generic) | S3.1 |
| $h^{(l)}(z)$ | Hidden representation of sample $z$ at layer $l$ | S3.3 |
| $\nabla_h \mathcal{L}$ | Gradient of loss w.r.t. hidden representation $h$ | S3.3 |
| $\phi(z)$ | Test-side encoding function (bilinear form) | S3.3 |
| $\psi(z)$ | Train-side encoding function (bilinear form) | S3.3 |
| $\cos(\cdot, \cdot)$ | Cosine similarity | S3.3 |
| $r$ | LoRA rank | S3.4 |
| $T$ | Number of training steps | S3.2 (MAGIC) |

## Method-Specific Notation

| Symbol | Meaning | First Appearance |
|--------|---------|-----------------|
| $I_\text{IF}$ | Standard influence function score: $\nabla_\theta \mathcal{L}(z_\text{test})^\top H_\theta^{-1} \nabla_\theta \mathcal{L}(z_\text{train})$ | S3.1 |
| $I_\text{RepSim}$ | RepSim score: $\cos(h^{(l)}(z_\text{test}), h^{(l)}(z_\text{train}))$ | S3.3 |
| $I_\text{RepT}$ | RepT score: $\cos(\phi_\text{RepT}(z_\text{test}), \phi_\text{RepT}(z_\text{train}))$ where $\phi_\text{RepT}(z) = [h^{(l^*)}(z); \nabla_h \mathcal{L}(z)]$ | S3.3 |
| $I_\text{TRAK}$ | TRAK score (random-projected gradient similarity) | S3.1 / S4.1 |
| $I_\text{contr}$ | Contrastive score: $I(\cdot; \theta_\text{ft}) - I(\cdot; \theta_\text{base})$ | S3.3 |
| $I_\text{MAGIC}$ | Exact IF via metagradient computation | S3.2 |

## Diagnostic Metrics

| Symbol | Meaning | First Appearance |
|--------|---------|-----------------|
| $\Delta_\text{FM1}$ | FM1 main effect: $\overline{\text{LDS}}_\text{repr} - \overline{\text{LDS}}_\text{param}$ | S3.4 / S4.3 |
| $\Delta_\text{FM2}$ | FM2 main effect: $\overline{\text{LDS}}_\text{contr} - \overline{\text{LDS}}_\text{std}$ | S3.4 / S4.3 |
| $\Xi$ | Interaction term: $(I_\text{repr,contr} - I_\text{repr,std}) - (I_\text{param,contr} - I_\text{param,std})$ | S3.4 / S4.3 |
| CMF | Common-Mode Fraction: $|I_\text{std} - I_\text{contr}| / |I_\text{std}|$ | S3.4 / S4.3 |
| $\text{Adv}_\text{RepSim}$ | RepSim advantage: $\text{LDS}_\text{RepSim} - \text{LDS}_\text{TRAK}$ | S4.4 |

## Evaluation Metrics

| Abbreviation | Full Name | Definition Context |
|-------------|-----------|-------------------|
| LDS | Linear Datamodeling Score | Spearman correlation between predicted and actual model output changes under training subset removal. Primary metric. |
| AUPRC | Area Under Precision-Recall Curve | Detection metric for toxicity filtering task. |
| P@K | Precision at K | Fraction of top-K attributed samples that are truly influential. |
| MRR | Mean Reciprocal Rank | Retrieval metric for factual attribution task. |
| Recall@50 | Recall at 50 | Fraction of truly influential samples in top-50 attributions. |

## Abbreviations

| Abbreviation | Full Form | First Appearance |
|-------------|-----------|-----------------|
| TDA | Training Data Attribution | S1 P1 |
| IF | Influence Function | S1 P1 |
| FM1 | Failure Mode 1: Signal Dilution | S1 P3 |
| FM2 | Failure Mode 2: Common Influence Contamination | S1 P3 |
| iHVP | inverse Hessian-Vector Product | S2.1 |
| LLM | Large Language Model | S1 P1 |
| SNR | Signal-to-Noise Ratio | S3.2 |
| JL | Johnson-Lindenstrauss (lemma) | S3.2 |
| LoRA | Low-Rank Adaptation | S3.4 |
| FT | Fine-Tuning | S3.4 |
| CMF | Common-Mode Fraction | S3.4 |
| ANOVA | Analysis of Variance | S4.3 |
| CI | Confidence Interval | S4 |
| FDR | False Discovery Rate | S4 |
