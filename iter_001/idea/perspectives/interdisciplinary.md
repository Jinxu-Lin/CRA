# Interdisciplinary Perspective: CRA Research Proposal

## Executive Summary

The CRA framework -- diagnosing two signal processing defects (FM1: signal dilution, FM2: common influence contamination) and unifying representation-space TDA methods under a bilinear phi^T M psi framework -- has deep structural analogs in at least four fields outside ML. These are not surface-level metaphors; each analogy carries precise mathematical correspondences that yield novel predictions, experimental designs, and theoretical tools the CRA team has not yet exploited. Below I develop three interdisciplinary angles, grounded in specific literature, with concrete experimental plans.

---

## Angle 1: Radar Signal Processing -- CFAR Detection as the Formal Ancestor of Contrastive Scoring (FM2 Fix)

### The Cross-Disciplinary Connection

The CRA proposal's contrastive scoring -- subtracting the mean attribution score to remove common influence contamination -- is a rediscovery of **Constant False Alarm Rate (CFAR) detection**, a 60-year-old technique from radar engineering (Finn & Johnson, 1968; Rohling, 1983). The structural isomorphism is exact:

| Radar CFAR | CRA Contrastive Scoring |
|---|---|
| Target return signal s(t) | Task-specific attribution signal phi_task |
| Background clutter c(t) | Common pre-training influence phi_shared |
| Noise floor estimation (averaging over reference cells) | Mean attribution score across training set |
| Adaptive threshold = k * estimated_clutter | Contrastive score = raw_score - mean_score |
| Detection: s(t) > threshold | Attribution: contrastive_score > 0 |

This is not a vague metaphor. In both systems, the core problem is identical: **detect a weak task-specific signal embedded in a strong, structured background that is shared across all observations**. The CFAR literature provides a rich taxonomy of solutions that directly map to TDA:

1. **Cell-Averaging CFAR (CA-CFAR)**: Estimates clutter by averaging over neighboring range cells. Direct analog: CRA's mean subtraction over training examples. Known limitation: performance degrades in heterogeneous clutter (multiple target environments). Prediction for CRA: mean subtraction will fail when the training set contains multiple distinct "clusters" of influence (e.g., overlapping topics in pre-training data).

2. **Order-Statistic CFAR (OS-CFAR)** (Rohling, 1983): Uses the k-th largest sample instead of the mean for robust clutter estimation in multi-target environments. **Novel CRA prediction**: replacing mean subtraction with median or trimmed-mean subtraction should improve contrastive scoring when training data has heterogeneous influence distributions.

3. **Clutter Map CFAR (CM-CFAR)**: Maintains a spatial map of background clutter level. Analog: computing per-cluster or per-topic mean attribution scores and subtracting locally rather than globally. **Novel CRA prediction**: topic-conditioned contrastive scoring (subtracting the mean within each training data cluster) should outperform global mean subtraction, especially on heterogeneous datasets.

### Why the Transplant Works

The mathematical structure is identical. In radar:

```
Detection statistic: T = s(t) / clutter_estimate
```

In CRA (with contrastive scoring):

```
Attribution statistic: A = phi(z_test)^T psi(z_train) - E_z'[phi(z_test)^T psi(z')]
                      = phi(z_test)^T [psi(z_train) - E[psi]]
```

Both are instances of the general problem: **linear detection of a signal in structured noise, where the noise covariance must be estimated from the data itself**. The 70+ years of radar detection theory provide optimality proofs, performance bounds, and failure mode analyses that CRA can directly inherit.

### Grounding in Existing Cross-Disciplinary Work

The connection between signal detection theory and ML is not new but has never been applied to TDA:

- Van Trees (2001, *Detection, Estimation, and Modulation Theory*) formalized the matched filter as the optimal linear detector -- exactly the phi^T Sigma^{-1} psi whitened attribution that CRA proposes.
- The radar literature's distinction between **coherent detection** (phase-sensitive, amplitude-and-phase) and **non-coherent detection** (amplitude-only) maps onto CRA's distinction between attribution methods that preserve sign information (influence functions) vs. those that use absolute similarity (k-NN).
- Recent work on learned CFAR detectors using neural networks (arxiv:2412.12620, Multi-Domain Features Guided Supervised Contrastive Learning for Radar Target Detection) demonstrates that the radar community is converging toward the same contrastive learning paradigm from the opposite direction.

### Concrete Experimental Plan

**Experiment R1: OS-CFAR Contrastive Scoring**
- Replace global mean subtraction with order-statistic (median, trimmed 10%-mean) estimation of the common influence baseline
- Test on DATE-LM across all 3 tasks
- Prediction: OS-CFAR variants outperform CA-CFAR (mean subtraction) by 2-5pp on tasks with heterogeneous training data distributions
- Model: Pythia-1B; Compute: < 1 GPU-hour (post-hoc score adjustment, no retraining)

**Experiment R2: Clustered Contrastive Scoring (CM-CFAR Analog)**
- K-means cluster training examples by representation similarity (k=10, 20, 50)
- Subtract within-cluster mean instead of global mean
- Test on DATE-LM data selection (LDS) and factual attribution (P@K)
- Prediction: clustered contrastive outperforms global contrastive by 3-8pp on factual attribution (highest heterogeneity)
- Compute: < 0.5 GPU-hours

**Success Probability: 60%** -- The analog is mathematically sound but whether training data distributions are sufficiently heterogeneous to differentiate CA-CFAR from OS/CM-CFAR variants is an empirical question.

---

## Angle 2: Neuroscience RSA -- Representational Similarity Analysis as the Brain Science Twin of phi^T M psi

### The Cross-Disciplinary Connection

The CRA bilinear framework phi(z_test)^T M psi(z_train) is structurally identical to **Representational Similarity Analysis (RSA)** from computational neuroscience (Kriegskorte et al., 2008). This is not a loose analogy -- both frameworks solve the same mathematical problem: **quantifying the functional relationship between two systems by comparing their internal representations**.

| Neuroscience RSA | CRA Bilinear Framework |
|---|---|
| Neural population vector r(stimulus) in R^d | Representation phi(z) in R^d |
| Representational Dissimilarity Matrix (RDM) | Attribution score matrix S[i,j] = phi(z_i)^T M psi(z_j) |
| Brain region comparison via RDM correlation | Method comparison via phi^T M psi unification |
| Model RDM from computational model | Metric tensor M (I, Sigma^{-1}, or learned) |
| Noise ceiling estimation | Upper bound from whitened matched filter |

The RSA literature (Kriegskorte, 2008; Nili et al., 2014; Diedrichsen & Kriegeskorte, 2017) has spent 15+ years developing:

1. **Noise ceiling estimation**: An upper bound on how well any model RDM can predict the neural RDM, given measurement noise. **Direct CRA analog**: The whitened matched filter phi^T Sigma_noise^{-1} psi provides an analogous noise ceiling for attribution quality. The RSA noise ceiling methodology could provide a principled way to estimate this bound without computing the full whitened filter.

2. **Crossvalidated RSA (cvRSA)**: Splits neural data into independent halves to avoid overfitting the similarity structure. **CRA prediction**: The current phi^T M psi framework does not crossvalidate -- the same representations used to compute similarity are used to evaluate it. Borrowing cvRSA methodology could reveal how much of CRA's attribution quality is genuine signal vs. shared noise structure.

3. **Deconfounded RSA** (Cui et al., 2022, arxiv:2202.00095): Explicitly removes population structure confounds from similarity measures. This is formally identical to CRA's FM2 correction -- removing the shared component that inflates similarity scores. The deconfounded RSA paper shows that standard CKA and RSA metrics are "confounded by the population structure of data items in the input space, leading to spuriously high similarity for even completely random neural networks." This is exactly the FM2 problem restated in neuroscience language.

4. **Feature-Reweighted RSA (FR-RSA)** (Kaniuth & Hebart, 2022): Learns an optimal weighting of representation dimensions for predicting neural similarity. **CRA analog**: This is equivalent to learning a diagonal M matrix in phi^T M psi -- an intermediate complexity between M=I (standard RepSim) and M=Sigma^{-1} (whitened matched filter).

### Why the Transplant Works

Both CRA and RSA face identical methodological challenges:

1. **The metric tensor problem**: What is the "right" way to compare representations? RSA uses correlation distance, CKA uses kernel alignment, CRA uses the phi^T M psi family. The RSA literature has proven (Harvey et al., 2024, arxiv:2411.08197) that CKA and CCA "quantify the average alignment between optimal linear readouts across a distribution of decoding tasks." This result directly applies to CRA: the choice of M in phi^T M psi determines what distribution of downstream tasks the attribution is optimized for.

2. **The spectral sampling problem**: Kang et al. (2025, arxiv:2502.19648) showed that representational similarity measures are "systematically underestimated under finite neuron sampling, mainly due to eigenvector delocalization," and that "the number of localized eigenvectors scales as the square root of the number of recorded neurons." **CRA translation**: Attribution quality degrades with limited representation dimensions, and the effective number of "attribution-informative" dimensions scales as sqrt(d), not d. If confirmed, this would revise the FM1 prediction from r_eff ~ d to r_eff ~ sqrt(d).

### Concrete Experimental Plan

**Experiment N1: Noise Ceiling Estimation via cvRSA Methodology**
- Split training data into two independent halves
- Compute phi^T psi attribution scores separately on each half
- Correlation between halves = reliability ceiling; compare to each method's actual performance
- This provides, for the first time, an empirical upper bound on TDA quality without access to ground truth
- Model: Pythia-1B; Compute: < 1 GPU-hour (double the representation extraction)

**Experiment N2: Feature-Reweighted Attribution (FR-RSA Analog)**
- Learn a diagonal weight vector w such that A(z_test, z_train) = phi(z_test)^T diag(w) psi(z_train)
- Optimize w on a held-out validation set (10% of DATE-LM test queries)
- Compare to M=I (RepSim), M=Sigma^{-1} (whitened), and learned diagonal M
- Prediction: diagonal M achieves 80-90% of whitened M's improvement over M=I, at much lower computational cost
- Compute: < 0.5 GPU-hours

**Experiment N3: Spectral Dimension Scaling Test**
- If RSA spectral analysis applies to CRA, then the number of attribution-informative dimensions scales as sqrt(d), not d
- Test by sweeping PCA dimensionality of representations: d' in {16, 32, 64, 128, 256, 512, 1024}
- If attribution quality saturates at d' ~ sqrt(d) ~ sqrt(2048) ~ 45, this fundamentally revises the FM1 theory
- Model: Pythia-1B (d=2048); Compute: < 1 GPU-hour

**Success Probability: 70%** -- The RSA-CRA correspondence is mathematically tight. The main risk is that TDA and neural decoding differ in ways that invalidate the spectral scaling prediction.

---

## Angle 3: Econometrics -- Omitted Variable Bias as FM2 and Instrumental Variables as a Novel FM2 Fix

### The Cross-Disciplinary Connection

CRA's FM2 (Common Influence Contamination) is a precise instance of **Omitted Variable Bias (OVB)** from econometrics (Angrist & Pischke, 2009). The structural correspondence is:

| Econometrics OVB | CRA FM2 |
|---|---|
| Outcome variable Y | Model prediction f(z_test) |
| Treatment variable X | Training example z_train |
| Omitted confounding variable Z | Pre-training knowledge (common influence) |
| Regression coefficient beta_OLS (biased) | Standard attribution score phi^T psi (contaminated) |
| Bias = Cov(X,Z)*gamma/Var(X) | FM2 bias = phi_shared^T psi_shared |
| Fixed-effect regression (within-group variation) | Contrastive scoring (mean subtraction) |

This mapping is not approximate -- it is formally exact under the linear model. In econometrics:

```
Y = beta*X + gamma*Z + epsilon
E[beta_OLS] = beta + gamma * Cov(X,Z)/Var(X)    [OVB formula]
```

In CRA:

```
f(z_test) = sum_i alpha_i * K(z_test, z_i)
alpha_i = phi(z_test)^T psi(z_i) = phi_task^T psi_task + phi_shared^T psi_shared
                                                           [FM2 bias term]
```

The econometrics of OVB is a mature field with 70+ years of theory. Three specific tools from econometrics are directly transplantable:

### Tool 1: Sensitivity Analysis (Cinelli & Hazlett, 2020)

The **omitted variable bias sensitivity analysis** provides bounds on how large an unobserved confounder must be to explain away an observed effect. In CRA terms: **how large must phi_shared be to explain all the attribution signal?** This provides a formal falsification criterion for FM2 -- if the sensitivity analysis shows that an implausibly large phi_shared is needed to explain the RepSim advantage, then FM2 is confirmed as a real phenomenon rather than a statistical artifact.

The Cinelli & Hazlett (2020) framework introduces the **Robustness Value (RV)**: the minimum strength of confounding needed to reduce the estimated effect to zero. Computing RV for CRA's attribution scores would provide the first formal quantification of FM2 severity.

### Tool 2: Instrumental Variables (IV) Estimation

In econometrics, when OVB cannot be eliminated by controlling for confounders, **instrumental variables** provide an alternative: find a variable that affects the treatment but not the outcome except through the treatment. In CRA terms: find a feature of z_train that affects attribution scores but is unrelated to pre-training knowledge.

**Novel CRA proposal**: Use **data augmentation** as an instrument. If z_train' is a semantically-preserving augmentation of z_train (e.g., paraphrase, synonym substitution), then phi(z_train') shares the same task-specific component phi_task but has a different common influence component (different surface forms activate different pre-training patterns). The IV estimator:

```
alpha_IV = Cov(f(z_test), phi(z_train')) / Cov(phi(z_train), phi(z_train'))
```

This should remove FM2 contamination without requiring mean subtraction, providing an independent validation of the FM2 diagnosis.

### Tool 3: Difference-in-Differences (DiD) Design

The CRA 2x2 factorial {parameter-space, representation-space} x {standard, contrastive} is formally a **Difference-in-Differences** design from causal econometrics. The DiD estimator:

```
DiD = (Rep_contrastive - Rep_standard) - (Param_contrastive - Param_standard)
```

measures the interaction between FM1 and FM2 corrections. Econometrics provides well-established tests for the **parallel trends assumption** underlying DiD validity: the improvement from contrastive scoring should follow the same trajectory in parameter and representation space (parallel trends), with any deviation indicating FM1-FM2 interaction. This is more rigorous than the current "interaction term < 30% of minimum main effect" criterion.

### Grounding in Existing Cross-Disciplinary Work

- Hammoudeh & Lowd (2024, "Training Data Attribution via Approximate Unrolled Differentiation") implicitly treat TDA as a causal inference problem but do not connect to the OVB literature.
- Park et al. (2023, TRAK) use random projection without acknowledging the econometric parallel to dimensionality reduction in high-dimensional IV estimation (Belloni et al., 2012).
- The "debiasing" step in DDA (Choe et al., 2024) is exactly the Frisch-Waugh-Lovell theorem from econometrics: partialing out confounding variables before regression yields the same coefficient as full multivariate regression. DDA's 55pp improvement from debiasing is thus theoretically grounded in a 90-year-old econometric result.

### Concrete Experimental Plan

**Experiment E1: OVB Sensitivity Analysis for FM2**
- Compute Cinelli-Hazlett Robustness Value for CRA attribution scores
- Formalize: how large must phi_shared be (as fraction of total representation norm) to explain RepSim's superiority over TRAK?
- If RV > 0.5 (confounder must explain >50% of variance): FM2 is a robust finding
- If RV < 0.2: FM2 may be a statistical artifact; attribution differences are better explained by other factors
- Model: Pythia-1B; Compute: < 0.5 GPU-hours (post-hoc analysis of pre-computed scores)

**Experiment E2: Instrumental Variable Attribution via Data Augmentation**
- Generate paraphrases of training examples using a separate LM (e.g., GPT-2 or Flan-T5-small)
- Compute attribution scores using original and paraphrased training examples
- IV estimator should isolate task-specific attribution by using augmentation as an instrument for common influence
- Compare IV-attributed scores to contrastive (mean-subtracted) scores
- Prediction: IV attribution achieves similar or better performance to contrastive scoring, validating the FM2-as-OVB interpretation
- Compute: ~2 GPU-hours (paraphrase generation + representation extraction)

**Experiment E3: Parallel Trends Test for DiD Validity**
- Compute contrastive scoring improvement across multiple levels of "FM1 severity" (varying projection dimensions in TRAK: k=64 to k=4096)
- If contrastive improvement is constant across k: parallel trends hold, FM1 and FM2 are independent
- If contrastive improvement varies with k: FM1-FM2 interaction exists, quantifiable via econometric DiD analysis
- Model: Pythia-1B; Compute: included in main TRAK dimension sweep

**Success Probability: 65%** -- The econometric framework is formally sound. The main risk is that the IV estimator (augmentation-based) may not satisfy the exclusion restriction if augmentation changes both task-specific and common influence components.

---

## Angle 4 (Supplementary): Statistical Physics -- Phase Transitions and the Emergence of Attribution Signal

### The Cross-Disciplinary Connection

The statistical physics of phase transitions provides a theoretical lens for understanding **when and why** representation-space attribution suddenly outperforms parameter-space attribution. The key insight: FM1 (signal dilution) is an instance of a **signal-noise phase transition**, analogous to the BBP (Baik-Ben Arous-Peche) transition in random matrix theory.

The BBP transition (Baik et al., 2005) shows that for a rank-1 signal embedded in a d-dimensional Gaussian noise matrix, the signal is detectable from the top eigenvalue **if and only if** the signal-to-noise ratio exceeds a critical threshold proportional to sqrt(d). Below this threshold, the top eigenvalue merges with the Marchenko-Pastur bulk -- the signal is invisible.

**CRA translation**: In parameter space (d=B ~ 10^9), the BBP threshold for detecting attribution signal is proportional to sqrt(B) ~ 30,000 -- extremely high. In representation space (d ~ 2048), the threshold is sqrt(d) ~ 45 -- much lower. This provides a formal, physics-derived explanation for why representation-space methods succeed: they operate below the BBP transition while parameter-space methods operate above it.

This connects to the Machine Learning Renormalization Group (MLRG) framework (Hou & You, 2023, arxiv:2306.11054; Mehta & Schwab, 2014, arxiv:1410.3831), where deep learning is understood as a form of coarse-graining that extracts relevant features by discarding irrelevant degrees of freedom -- exactly what the parameter-to-representation projection does in CRA.

The recent work by Goring et al. (2025, arxiv:2510.15174) on mean-field theory of feature learning shows that neural networks undergo a **symmetry-breaking phase transition** where they abruptly align with target functions. In CRA terms, this phase transition is the point where task-specific attribution signal (phi_task) separates from common influence (phi_shared) in the representation space. If the network has not undergone this transition (e.g., early layers, insufficient fine-tuning), representation-space attribution should fail.

The work by Bordelon & Pehlevan (2022, arxiv:2210.02157) on how learning rules affect representation dynamics provides additional structure: the effective Neural Tangent Kernel (eNTK) evolution in the rich mean-field regime corresponds to the evolution of the attribution metric tensor M during training.

### Concrete Experimental Plan

**Experiment P1: BBP Transition Test**
- Compute the eigenspectrum of gradient covariance (Pythia-70M)
- Identify whether the top eigenvalues separate from the Marchenko-Pastur bulk
- If yes: the number of separated eigenvalues = effective rank r_eff, directly measuring FM1
- If no: the attribution signal is below the BBP threshold even in representation space
- Compute: ~1 GPU-hour (Lanczos top-500 eigenvalues + MP bulk fit)

**Experiment P2: Layer-wise Phase Transition**
- Compute RepSim attribution quality using representations from each layer (1, 6, 12, 18, 24, 32) of Pythia-1B
- Prediction: attribution quality undergoes a sharp transition at some critical layer l*, corresponding to the point where task-specific features emerge
- This l* should correlate with the layer at which representation covariance becomes near-isotropic (H9)
- Compute: ~2 GPU-hours

**Success Probability: 55%** -- The BBP transition is well-characterized for simple spiked covariance models. Whether LLM gradient covariance obeys a clean BBP transition is uncertain; the spectrum may be more complex (multiple phase transitions, power-law tails).

---

## Unified Interdisciplinary Framework: The Signal Detection Pipeline

The four angles above converge on a unified view of TDA as a **signal detection pipeline**:

```
Raw Signal Space         Projection           Detection           Decision
(Parameter Space)  -->  (Representation)  -->  (Attribution)  -->  (Ranking)
   R^B                     R^d                 phi^T M psi         Top-K

Radar:     Full spectrum    Beamforming        Matched filter      CFAR threshold
Neuro:     Neural pop.     Dimensionality      RSA/CKA             Noise ceiling
Econ:      Raw regression  Feature selection   IV estimation        Significance test
Physics:   Microscopic     Coarse-graining     Order parameter      Phase transition
CRA:       Gradient space  Rep. extraction     Bilinear form        Contrastive score
```

Each stage corresponds to a well-studied problem in multiple disciplines:
1. **Projection** (FM1 fix): Radar beamforming, neural dimensionality reduction, econometric feature selection, RG coarse-graining
2. **Detection** (metric choice): Matched filter, CKA/RSA, IV estimation, order parameter
3. **Decision** (FM2 fix): CFAR thresholding, noise ceiling, OVB correction, phase transition detection

This unification suggests that the phi^T M psi framework is not just a notational convenience but an instance of a universal detection-theoretic structure.

---

## Summary of Predictions and Experiments

| Experiment | Source Field | Key Prediction | Compute | P(success) |
|---|---|---|---|---|
| R1: OS-CFAR scoring | Radar | Median subtraction > mean subtraction on heterogeneous data | 1h | 60% |
| R2: Clustered contrastive | Radar | Topic-conditioned subtraction improves factual attribution 3-8pp | 0.5h | 55% |
| N1: cvRSA noise ceiling | Neuroscience | Split-half reliability bounds TDA quality from above | 1h | 70% |
| N2: FR-RSA diagonal M | Neuroscience | Learned diagonal M achieves 80-90% of whitened M | 0.5h | 65% |
| N3: Spectral dimension | Neuroscience | Attribution saturates at sqrt(d) not d | 1h | 40% |
| E1: OVB sensitivity | Econometrics | Robustness Value quantifies FM2 severity formally | 0.5h | 75% |
| E2: IV attribution | Econometrics | Augmentation-based IV removes FM2 without mean subtraction | 2h | 50% |
| E3: Parallel trends | Econometrics | DiD test validates FM1-FM2 independence | 0h (included) | 65% |
| P1: BBP transition | Physics | Gradient spectrum shows clean separation from MP bulk | 1h | 55% |
| P2: Layer-wise transition | Physics | Attribution quality shows sharp transition at critical layer | 2h | 60% |

**Total additional compute: ~9.5 GPU-hours. All experiments use small models (Pythia-70M or Pythia-1B) and can complete within the project timeline.**

---

## Key Risks and Limitations

1. **Surface analogy risk**: The radar-CFAR and RSA connections are mathematically rigorous, but the econometric IV approach requires the augmentation instrument to satisfy exclusion restrictions that may not hold in practice.

2. **The sqrt(d) prediction (N3) is the highest-risk, highest-reward experiment**: If confirmed, it fundamentally revises the FM1 theory from r_eff ~ d to r_eff ~ sqrt(d). But existing RSA spectral theory assumes Gaussian neural populations, which LLM representations are not.

3. **Physics angle is least actionable**: The BBP transition provides beautiful theoretical framing but the experimental test (P1) is already planned in the CRA proposal as part of H4. The interdisciplinary contribution here is primarily interpretive, not methodological.

4. **Negative results are informative**: If OS-CFAR variants do not outperform simple mean subtraction, this tells us that training data distributions are sufficiently homogeneous that CA-CFAR (mean subtraction) is already near-optimal -- itself a useful empirical finding.

---

## References

### Radar / Signal Detection
- Finn, H. M., & Johnson, R. S. (1968). Adaptive detection mode with threshold control as a function of spatially sampled clutter-level estimates. *RCA Review*, 29, 414-464.
- Rohling, H. (1983). Radar CFAR thresholding in clutter and multiple target situations. *IEEE Trans. Aerospace & Electronic Systems*, AES-19(4), 608-621.
- Van Trees, H. L. (2001). *Detection, Estimation, and Modulation Theory, Part I*. Wiley.
- [Multi-Domain Features Guided Supervised Contrastive Learning for Radar Target Detection](https://arxiv.org/abs/2412.12620)

### Neuroscience / RSA
- Kriegskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis -- connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.
- Nili, H., et al. (2014). A toolbox for representational similarity analysis. *PLoS Computational Biology*, 10(4), e1003553.
- Diedrichsen, J., & Kriegeskorte, N. (2017). Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis. *PLoS Computational Biology*, 13(4), e1005508.
- Cui, T., et al. (2022). [Deconfounded Representation Similarity for Comparison of Neural Networks](https://arxiv.org/abs/2202.00095).
- Harvey, S. E., et al. (2024). [What Representational Similarity Measures Imply about Decodable Information](https://arxiv.org/abs/2411.08197).
- Kang, H., et al. (2025). [Spectral Analysis of Representational Similarity with Limited Neurons](https://arxiv.org/abs/2502.19648).
- Bo, Y., et al. (2024). [Evaluating Representational Similarity Measures from the Lens of Functional Correspondence](https://arxiv.org/abs/2411.14633).

### Econometrics / Causal Inference
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society Series B*, 82(1), 39-67.
- Frisch, R., & Waugh, F. V. (1933). Partial time regressions as compared with individual trends. *Econometrica*, 1(4), 387-401.
- Belloni, A., et al. (2012). Sparse models and methods for optimal instruments with an application to eminent domain. *Econometrica*, 80(6), 2369-2429.

### Statistical Physics / Random Matrix Theory
- Baik, J., Ben Arous, G., & Peche, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. *Annals of Probability*, 33(5), 1643-1697.
- Mehta, P., & Schwab, D. J. (2014). [An exact mapping between the Variational Renormalization Group and Deep Learning](https://arxiv.org/abs/1410.3831).
- Hou, W., & You, Y.-Z. (2023). [Machine Learning Renormalization Group for Statistical Physics](https://arxiv.org/abs/2306.11054).
- Goring, N., et al. (2025). [A simple mean field model of feature learning](https://arxiv.org/abs/2510.15174).
- Bordelon, B., & Pehlevan, C. (2022). [The Influence of Learning Rule on Representation Dynamics in Wide Neural Networks](https://arxiv.org/abs/2210.02157).
- Howard, J. N., et al. (2024). [Bayesian RG Flow in Neural Network Field Theories](https://arxiv.org/abs/2405.17538).
