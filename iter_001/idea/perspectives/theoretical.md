# Theoretical Perspective: CRA Research Proposal

**Agent**: Theoretical (mathematical foundations, provable guarantees, information-theoretic analysis)
**Date**: 2026-03-16

---

## Executive Summary

The CRA project's core claims -- FM1 (signal dilution) and FM2 (common influence contamination) as two orthogonal failure modes, plus the phi^T * psi bilinear unification -- are currently supported by intuition and analogy. This perspective develops the mathematical scaffolding needed to upgrade these claims from heuristic narratives to provable (or at least formally bounded) statements. I propose three theoretically grounded angles: (1) a rank-deficiency theorem that formalizes FM1 as a spectral gap phenomenon with quantitative SNR bounds, (2) a confounding decomposition that casts FM2 as a bias term in a formal attribution estimator and proves contrastive scoring eliminates it under stated conditions, and (3) a completeness result for the phi^T * psi framework showing it characterizes *all* linear attribution methods and deriving optimality conditions. Each angle produces theorems with testable corollaries that can be validated on Pythia-1B within 1 GPU-hour.

---

## Angle 1: Spectral Attribution Theory -- FM1 as Rank-Deficient Signal in Full-Rank Noise

### Core Insight (New Theoretical Method: Random Matrix Theory + Sketching Theory --> TDA Diagnosis)

The current FM1 narrative says: "parameter gradients in R^B are near-orthogonal by JL, so attribution signal drowns." This is imprecise. The JL lemma says inner products are *preserved* under random projection when the projection dimension m = O(log N / epsilon^2). If JL were the whole story, parameter-space methods should *work* with enough projection dimensions. The real problem is more subtle and requires a spectral analysis.

**Formal Setup**: Let g_i = nabla_theta L(z_i, theta*) in R^B be the gradient of the loss at the trained parameters for training sample z_i. The attribution score between test point z and training point z_i is:

```
A(z, z_i) = g_z^T F^{-1} g_i
```

where F is the Fisher Information Matrix (or Hessian approximation). Define the *attribution signal subspace* S = span({g_i}_{i=1}^N) intersect range(F). Let r = dim(S).

**Theorem 1 (Signal Dilution Bound)**: Let the eigenvalues of the gradient covariance Sigma_g = (1/N) sum_i g_i g_i^T be lambda_1 >= lambda_2 >= ... >= lambda_B. Define the effective rank r_eff(alpha) = sum_j lambda_j / lambda_1 (at threshold alpha, the number of eigenvalues accounting for alpha-fraction of total variance). For a random projection P in R^{m x B} with m < r_eff(0.95), the expected relative error of the projected attribution score satisfies:

```
E[|P g_z^T (PFP^T)^{-1} P g_i - g_z^T F^{-1} g_i|] / |g_z^T F^{-1} g_i| >= Omega(sqrt(r_eff / m - 1))
```

This bound is non-vanishing when m < r_eff, meaning projection-based parameter-space methods *necessarily* lose signal unless the projection dimension exceeds the effective rank.

**Theorem 2 (Representation Space as Natural Projection)**: Let h(z) = f_L(z) in R^d be the representation at layer L. Under the neural tangent kernel (NTK) approximation, g_i approximately equals J^T delta_i where J in R^{B x d} is the Jacobian of the last layer and delta_i in R^d is the output-layer gradient. Then:

```
g_z^T g_i approximately equals delta_z^T J J^T delta_i = delta_z^T K delta_i
```

where K = J J^T in R^{d x d} is the representation-space kernel. The effective rank of K is at most d << B, and RepSim(z, z_i) = h(z)^T h(z_i) approximates a specific K-weighted inner product.

**Key Prediction**: r_eff(Sigma_g) ~ O(d) for well-trained LLMs, where d is the representation dimension. This is because the gradient covariance is dominated by the NTK structure, which has rank at most d.

**Connection to Hu et al. (2602.10449)**: Their unified projection theory proves that sketching preserves influence iff sketch dimension m >= rank(F). Our Theorem 1 complements this by showing that for LLMs, rank(F) is governed by the representation dimension d, not the parameter count B. Their theory is abstract; ours predicts the *specific* numerical threshold where parameter-space methods break down.

**Connection to Deb et al. (2509.21847)**: Their work on "Beyond Johnson-Lindenstrauss: Uniform Bounds for Sketched Bilinear Forms" provides the mathematical machinery for analyzing sketched bilinear forms g^T F^{-1} g' with uniform guarantees over pairs of sets. Their framework via generic chaining could tighten our Theorem 1 bounds from expected-case to uniform-over-all-query-pairs guarantees.

### What This Proves About CRA

If Theorems 1-2 hold empirically, FM1 is not merely "high dimensionality" but specifically "the attribution signal occupies an O(d)-dimensional subspace in an O(B)-dimensional space." Representation-space methods succeed because they operate directly in this signal-rich subspace, bypassing the noise dimensions. This is a *structural* explanation, not just a scaling argument.

### Testable Corollary

**Corollary 1.1**: For Pythia-{70M, 160M, 410M, 1B} with representation dimensions d in {512, 768, 1024, 2048}, the TRAK attribution quality (LDS on DATE-LM) should saturate at projection dimension k ~ O(d). Specifically, 90% of maximal LDS should be achieved at k = 2d, and further increases to k = 10d should yield < 5% additional improvement.

**Experimental Validation (15 min pilot)**: Compute the eigenvalue spectrum of gradient covariance for Pythia-70M using 1K DATE-LM training samples. Verify exponential/power-law decay with knee at approximately d = 512.

### Failure Mode

If r_eff scales linearly with B (not sublinearly), the spectral story collapses. This would mean parameter-space methods should work with sufficiently large projection dimension, contradicting Li et al.'s empirical findings. In this case, FM1 would need a different explanation (e.g., optimization landscape issues rather than intrinsic dimensionality).

**Computational Cost**: ~2 GPU-hours (Lanczos top-500 eigenvalues at each model scale)
**Success Probability**: 65% -- eigenvalue decay is almost certainly steep, but the quantitative match r_eff ~ d is the critical prediction
**Impact if Succeeds**: High -- transforms FM1 from metaphor to theorem with measurable constants

### Key References

- Hu et al. 2026 (2602.10449) -- Unified theory of random projection for influence functions; proves sketch dimension must exceed rank(F) but does not measure rank(F) for LLMs
- Deb et al. 2025 (2509.21847) -- Uniform bounds for sketched bilinear forms via generic chaining; provides mathematical tools for tightening our bounds
- Park et al. 2023 (2303.14186) -- TRAK: establishes the random projection approach; their Figure 3 shows k-dependence on CIFAR but not LLMs
- Li et al. 2025 (2512.09103) -- Natural geometry of robust attribution; identifies spectral amplification as source of fragility in TRAK scores, complementary to our rank-deficiency analysis
- Wei et al. 2024 (2412.03906) -- Final-model-only TDA; unifies gradient-based methods as approximations to "further training," providing an alternative gold standard against which our spectral predictions can be validated

---

## Angle 2: Attribution Bias Decomposition -- FM2 as Formally Characterizable Confounding

### Core Insight (Cross-Domain Transfer: Causal Inference Theory --> TDA Debiasing)

FM2 (common influence contamination) claims that standard influence scores are dominated by "pre-training knowledge" shared across all training samples. This is a precise analog of the *confounding variable* problem in causal inference. We can formalize this and prove that contrastive scoring is a *consistent estimator* of the causal attribution effect under stated assumptions.

**Formal Setup**: Define the attribution score as an inner product in some feature space:

```
A(z, z_i) = phi(z)^T psi(z_i)
```

Decompose the feature map into task-specific and shared components:

```
phi(z) = phi_task(z) + phi_shared(z)
psi(z_i) = psi_task(z_i) + psi_shared(z_i)
```

where phi_shared(z) = E_{z'}[phi(z')] is the mean feature (capturing pre-training knowledge common to all inputs) and phi_task(z) = phi(z) - phi_shared(z) is the residual.

**Theorem 3 (Attribution Bias Decomposition)**: The standard attribution score decomposes as:

```
A(z, z_i) = phi_task(z)^T psi_task(z_i)     [true causal attribution]
           + phi_shared(z)^T psi_task(z_i)    [test-side confounding]
           + phi_task(z)^T psi_shared(z_i)    [train-side confounding]
           + phi_shared(z)^T psi_shared(z_i)  [pure bias term]
```

The last three terms constitute the FM2 contamination. The pure bias term phi_shared^T psi_shared is constant across all training samples and only shifts the ranking by an additive constant (harmless for ranking). However, the cross terms (test-side and train-side confounding) are *data-dependent* and can corrupt rankings.

**Theorem 4 (Mean-Subtraction Eliminates FM2 Under Orthogonality)**: Define the contrastive score as:

```
A_c(z, z_i) = [phi(z) - E_z'[phi(z')]]^T [psi(z_i) - E_z'[psi(z')]]
            = phi_task(z)^T psi_task(z_i)
```

This equals the true causal attribution *exactly* when the decomposition phi = phi_task + phi_shared holds with E[phi_task] = 0. No distributional assumptions (e.g., Gaussianity) are needed -- only that the shared component equals the population mean.

**Theorem 5 (Parallel Trends and DiD Attribution)**: When pre-training and post-fine-tuning checkpoints are available, define the DiD score:

```
A_DiD(z, z_i) = [phi_post(z) - phi_pre(z)]^T [psi_post(z_i) - psi_pre(z_i)]
              - E_z'{ [phi_post(z) - phi_pre(z)]^T [psi_post(z') - psi_pre(z')] }
```

Under the parallel trends assumption (pre-training feature similarity is independent of fine-tuning attribution relevance), A_DiD is an unbiased estimator of the fine-tuning-induced attribution effect. The double differencing removes both time-invariant confounders (pre-training knowledge) and general representation drift.

**Why DDA's Approach is a Special Case**: DDA (2410.01285) applies debias (mean subtraction of gradients) and denoise (removal of irrelevant gradient components). Their debias step is exactly our mean-subtraction (Theorem 4), but applied only to gradients, not representations. Their key finding -- debias contributes 55pp while denoise contributes only 9pp -- is predicted by our theory: FM2 (addressed by debias/mean-subtraction) is the dominant contamination, while denoising addresses residual variance.

### What This Proves About CRA

The bias decomposition provides a clean mathematical account of *why* contrastive scoring works: it removes a formally characterizable bias term. More importantly, it predicts *when* contrastive scoring will help most: when phi_shared has large norm relative to phi_task (i.e., when pre-training knowledge dominates task-specific signal). This predicts that FM2 severity should be higher for:
- Larger pre-trained models (more compressed shared knowledge)
- Tasks more aligned with pre-training distribution (e.g., factual attribution > toxicity filtering)
- Parameter-space features (gradients encode all learned knowledge) vs. representation-space features (representations already compress)

### Testable Corollaries

**Corollary 2.1**: The FM2 severity index, defined as ||phi_shared||^2 / ||phi_task||^2, should be systematically larger for parameter-space features than for representation-space features. Predicted ratio: > 10x for parameter space, < 2x for representation space (last-layer activations).

**Corollary 2.2**: Mean subtraction alone should recover >= 80% of DDA's contrastive scoring gains, since the cross terms (Theorem 3) are the dominant FM2 mechanism.

**Corollary 2.3**: DiD attribution (Theorem 5) should outperform single-difference contrastive scoring on tasks where pre-training similarity is confounded with fine-tuning relevance (likely factual attribution on DATE-LM, where pre-training already encodes facts).

### Failure Mode

The orthogonality assumption E[phi_task] = 0 may not hold exactly. If the task-specific component has a non-zero mean (e.g., task-relevant training samples are systematically different from the population), mean subtraction over-corrects. This is analogous to the "parallel trends" violation in DiD. Mitigation: use task-stratified negative samples rather than population-wide negatives.

**Computational Cost**: ~1.5 GPU-hours (representation extraction + scoring with cached features)
**Success Probability**: 70% -- the bias decomposition is mathematically clean, but the quantitative predictions about FM2 severity ratios need empirical validation
**Impact if Succeeds**: Medium-High -- provides the first formal justification for contrastive scoring in TDA, connecting to the well-established causal inference literature

### Key References

- DDA (2410.01285) -- Empirical evidence: debias 55pp vs. denoise 9pp; our theory explains this ratio
- Angrist & Pischke 2009 -- "Mostly Harmless Econometrics"; DiD theory
- Zhang et al. 2025 (2501.18887) -- Unified attribution framework across XAI/Data-Centric/Mechanistic; shows perturbation, gradient, and linear approximation as shared techniques, but does not formalize the confounding structure we identify
- Wei et al. 2024 (2412.03906) -- Unifying gradient-based TDA methods via "further training" gold standard; our bias decomposition applies to their framework
- Pan et al. 2025 (2502.11411) -- Denoised Representation Attribution; token-level denoising is complementary to our checkpoint-level differencing

---

## Angle 3: The phi^T * psi Bilinear Framework -- Completeness, Optimality, and Taxonomy

### Core Insight (New Theoretical Framework: Functional Analysis --> TDA Unification)

The claim that "all 5 representation-space methods are instances of phi^T * psi" needs to be made mathematically precise. More importantly, we should prove a *completeness* result: every reasonable linear attribution method is a phi^T * psi bilinear form, and then derive the *optimal* phi and psi for each attribution task.

**Formal Setup**: An attribution method assigns a score A(z, z_i) in R to each (test, train) pair. We restrict attention to *linear* attribution methods, where A(z, z_i) is linear in the features of z_i for fixed z (and vice versa). This excludes methods based on k-NN or nonlinear kernels, but captures all gradient-based and representation-based methods in the literature.

**Theorem 6 (Representation Theorem for Linear Attribution)**: Any attribution method A(z, z_i) that satisfies:
1. *Linearity*: A(z, alpha z_i + beta z_j) = alpha A(z, z_i) + beta A(z, z_j) in feature space
2. *Continuity*: A is continuous in both arguments
3. *Finite-dimensionality*: A has finite rank (i.e., the operator T_A: z -> A(z, .) has finite-dimensional range)

can be written as A(z, z_i) = phi(z)^T M psi(z_i) for some feature maps phi: Z -> R^p, psi: Z -> R^q, and matrix M in R^{p x q}.

When p = q and M = I, this reduces to phi^T psi. The general case phi^T M psi captures methods with a *metric tensor* M (e.g., influence functions where M = F^{-1}).

**Theorem 7 (Taxonomy of Existing Methods)**: Under the phi^T M psi framework:

| Method | phi(z) | psi(z_i) | M | Space |
|--------|--------|----------|---|-------|
| IF (Influence Functions) | nabla_theta L(z) | nabla_theta L(z_i) | H^{-1} | Parameter R^B |
| TRAK | P nabla_theta L(z) | P nabla_theta L(z_i) | (PHP^T)^{-1} | Projected param R^m |
| RepSim | h_L(z) | h_L(z_i) | I_d | Representation R^d |
| RepT | h_L(z) + alpha nabla_h L(z) | h_L(z_i) | I_d | Augmented repr R^d |
| AirRep | E(h(z); theta_E) | E(h(z_i); theta_E) | I_p | Learned repr R^p |
| Concept Influence | v_c^T h(z) | v_c^T h(z_i) | I_1 | Concept direction R^1 |
| In-the-Wild | h_post(z) - h_pre(z) | h_post(z_i) - h_pre(z_i) | I_d | Activation diff R^d |
| DDA (contrastive) | g(z) - g_bar | g(z_i) - g_bar | F_tilde^{-1} | Debiased param R^B |

This table reveals structural patterns:
- **Representation methods** all use M = I (no curvature correction needed), while **parameter methods** require M = H^{-1} or approximations
- **Contrastive variants** apply mean subtraction to phi and/or psi (addressing FM2)
- **The critical distinction** is not phi vs. psi but the choice of *feature space* (R^B vs. R^d) and *metric* (curvature-corrected vs. identity)

**Theorem 8 (Optimal phi^T psi for Attribution)**: Given a ground-truth attribution oracle A*(z, z_i) (e.g., leave-one-out retraining), the optimal bilinear approximation minimizing E[|phi(z)^T psi(z_i) - A*(z, z_i)|^2] over all linear phi, psi satisfies:

```
phi_opt(z) = U_r^T f(z)
psi_opt(z_i) = Sigma_r V_r^T f(z_i)
```

where f(z) is any sufficient feature representation, and U_r Sigma_r V_r^T is the rank-r truncated SVD of the true attribution matrix A* = [A*(z_j, z_i)]_{j,i}. The *optimal* feature maps are determined by the SVD of the attribution matrix itself.

**Key Insight**: This theorem shows that the "best" phi and psi are the left and right singular vectors of the attribution matrix. If the attribution matrix has low effective rank (as suggested by Angle 1), then a low-dimensional bilinear form suffices. Furthermore, if A* approximately equals h(z)^T h(z_i) (i.e., RepSim approximates the oracle), this implies the representation space is already aligned with the top singular vectors of the attribution matrix.

### Theoretical Predictions

**Prediction 3.1 (Curvature Correction is Unnecessary in Representation Space)**: Parameter-space methods need M = H^{-1} because the gradient covariance is highly anisotropic (eigenvalue spread > 10^6 for LLMs). In contrast, representation-space methods can use M = I because the representation covariance is far more isotropic (predicted eigenvalue spread < 100 for last-layer activations). This is because backpropagation through many layers amplifies directional anisotropy, while forward propagation through normalization layers (LayerNorm) actively isotropizes representations.

**Prediction 3.2 (Layer Selection Determines Optimality)**: The optimal layer L* for representation-based attribution is the one where h_L(z) best approximates the top-r singular vectors of the attribution oracle A*. This should coincide with RepT's "phase transition layer" and with the layer achieving the best information bottleneck tradeoff (compressing input while preserving task-relevant information). Predicted: L* is in the range [0.5L_total, 0.75L_total] for decoder-only LLMs.

**Prediction 3.3 (FM1 and FM2 Correction Are Orthogonal)**: In the phi^T M psi framework, FM1 correction (dimension reduction: R^B -> R^d) changes the feature space, while FM2 correction (mean subtraction) changes the feature map within a fixed space. These two operations commute: (mean-subtraction after projection) = (projection after mean-subtraction). Therefore, their effects should be *approximately additive* in a 2x2 factorial experiment, with interaction term < 30% of the minimum main effect. This is the pre-registered falsification condition for the CRA paper.

### Testable Corollary

**Corollary 3.1**: Compute the SVD of the empirical attribution matrix on DATE-LM data selection task (using LOO retraining on a 1K subsample as oracle). Compare the top-r right singular vectors with: (a) representation vectors h(z_i), (b) projected gradients Pg(z_i). If cosine alignment is > 0.8 for representations and < 0.3 for gradients, this confirms that representations are naturally aligned with the optimal psi while gradients are not.

**Corollary 3.2**: The representation covariance eigenvalue ratio lambda_1/lambda_d should be < 100 (isotropic), while the gradient covariance ratio lambda_1/lambda_B should be > 10^4 (anisotropic). This explains why M = I suffices for representations but M = H^{-1} is needed for gradients.

### Failure Mode

The representation theorem (Theorem 6) only covers *linear* attribution. If the true attribution relationship is fundamentally nonlinear (e.g., interactions between training samples matter), the phi^T psi framework is incomplete. AirRep's learned encoder and Concept Influence's probe layer both introduce nonlinearity, which our framework captures only as approximations. If nonlinear methods significantly outperform the best linear phi^T psi, the framework's explanatory power is limited.

**Computational Cost**: ~1 GPU-hour (SVD of small attribution matrix + eigenvalue computations)
**Success Probability**: 75% -- the taxonomy is definitionally correct (it's a reformulation), but the *optimality* predictions depend on the attribution matrix having low effective rank and representations being aligned with its singular structure
**Impact if Succeeds**: Very High -- provides the first complete theoretical taxonomy of TDA methods with optimality conditions, enabling principled method selection

### Key References

- Park et al. 2023 (2303.14186) -- TRAK as phi^T M psi with random projection
- Hu et al. 2026 (2602.10449) -- Proves projection preserves influence iff m >= rank(F); our framework shows *why* rank(F) ~ d
- Zhang et al. 2025 (2501.18887) -- Unified attribution across XAI/Data/Mechanistic; our phi^T M psi is a more specific mathematical instantiation of their "shared techniques" observation
- RepT (2510.02334) -- Phase transition layer = optimal L* in our framework
- AirRep (2505.18513) -- Learned E(h; theta_E) approximates phi_opt from our Theorem 8
- Concept Influence (2602.14869) -- v_c^T h is a rank-1 specialization of our framework
- Vitel & Chhabra 2025 (2511.04715) -- Layer selection evidence supporting Prediction 3.2

---

## Synthesis: How These Angles Form a Complete Theoretical Contribution

The three angles are not independent proposals; they form a coherent theoretical narrative:

```
Angle 3 (Framework)     --> "All linear TDA methods are phi^T M psi"
    |                         |
    v                         v
Angle 1 (FM1 Theory)    --> "Parameter space has rank-deficient signal"
    |                    --> "Representation space IS the signal subspace"
    |                    --> "Therefore M = I suffices in repr space"
    |
Angle 2 (FM2 Theory)    --> "Shared features contaminate attribution"
                         --> "Mean subtraction = causal deconfounding"
                         --> "Corrections are orthogonal (commuting)"
```

Together, they answer three questions that the current CRA proposal leaves as intuitions:

1. **Why do representation-space methods work?** (Angle 1: they operate in the signal-rich subspace of effective rank ~ d)
2. **Why does contrastive scoring help?** (Angle 2: it removes a formally characterizable bias term)
3. **What is the optimal attribution method?** (Angle 3: it's the SVD-aligned bilinear form in representation space with mean subtraction)

### Theoretical Contribution Hierarchy

| Theorem | Novelty | Proof Difficulty | Experimental Testability |
|---------|---------|-----------------|------------------------|
| T1 (Signal Dilution Bound) | High -- first quantitative bound | Medium -- relies on RMT | High -- eigenvalue computation |
| T2 (Repr as Natural Projection) | Medium -- NTK connection is known, application is new | Low -- follows from NTK theory | High -- direct comparison |
| T3 (Bias Decomposition) | Medium -- standard linear algebra, but novel application | Low -- algebraic | High -- measure component norms |
| T4 (Mean-Subtraction Correctness) | High -- first formal justification for contrastive TDA | Low -- follows from T3 | High -- ablation |
| T5 (DiD Attribution) | High -- novel cross-domain import | Medium -- needs parallel trends | Medium -- requires both checkpoints |
| T6 (Representation Theorem) | Medium -- standard functional analysis | Low -- Riesz representation | N/A (definitional) |
| T7 (Taxonomy) | High -- novel systematic categorization | Low -- instantiation | High -- table is directly verifiable |
| T8 (Optimal phi psi) | Very High -- first optimality result for TDA | Medium -- SVD analysis | Medium -- needs oracle attribution |

### Recommended Priority

**T7 (Taxonomy) + T3-T4 (FM2 Deconfounding) > T1-T2 (FM1 Spectral) > T8 (Optimality) > T5 (DiD)**

Rationale: The taxonomy (T7) is the paper's unique theoretical contribution -- no one has categorized all representation-space methods in a common framework. The FM2 deconfounding (T3-T4) is the most surprising result for the community because it connects TDA to causal inference. The FM1 spectral theory (T1-T2) is the strongest technical result but requires more experimental validation. The optimality result (T8) is the most ambitious but also the most fragile. DiD (T5) is elegant but incremental over DDA.

---

## Risk Assessment

### Theoretical Risks

1. **NTK approximation breaks for LLMs**: Theorems 1-2 rely on the NTK approximation (g ~ J^T delta), which is known to be imprecise for large models with significant feature learning. *Mitigation*: State the theorem under the NTK assumption and separately validate empirically whether the predictions hold beyond this regime. The spectral prediction r_eff ~ d can hold even if the NTK mechanism is not the explanation.

2. **Attribution matrix is not low-rank**: Theorem 8 assumes the attribution oracle has low effective rank. If the true attribution is high-rank (many independent attribution modes), the bilinear framework is complete but not *efficient* -- you need high-dimensional phi, psi. *Mitigation*: Measure the empirical rank of LOO attribution matrices on small subsets.

3. **Parallel trends assumption fails for DiD**: The pre-training similarity between test and training samples may already be informative about fine-tuning attribution, violating the key DiD assumption. *Mitigation*: Test the assumption directly by measuring correlation between pre-training similarity and post-fine-tuning attribution relevance.

4. **Community reception of "theoretical" contributions**: The ML community sometimes values empirical results over theoretical analysis. *Mitigation*: Ensure every theorem has a testable corollary with a concrete experimental prediction. The theory serves the experiments, not the other way around.

### What if ALL Theorems Are Empirically Invalidated?

The phi^T M psi taxonomy (Theorem 7) is definitionally correct and cannot be "falsified." Even if Theorems 1-2 (spectral) and 3-5 (deconfounding) fail quantitatively, the taxonomy and the 2x2 empirical ablation remain publishable. The theoretical angles are upside bets that could elevate the paper from a solid empirical contribution to a theoretical milestone.

---

## Supplementary Literature Discovered During Search

1. **Hu et al. 2026 (2602.10449)** -- "A Unified Theory of Random Projection for Influence Functions." The most directly relevant theoretical paper. Proves projection preserves influence iff sketch dim >= rank(F); ridge regularization changes the barrier to effective dimension. *Critical*: their theory predicts our Theorem 1 as a corollary -- the rank(F) they analyze is exactly the r_eff we propose to measure. Their Kronecker-factored result (F = A tensor E) also connects to the NTK structure in our Theorem 2.

2. **Deb et al. 2025 (2509.21847)** -- "Beyond Johnson-Lindenstrauss: Uniform Bounds for Sketched Bilinear Forms." Develops uniform bounds for sketched bilinear forms using generic chaining. *Directly applicable*: our phi^T M psi framework involves exactly such sketched bilinear forms when M involves a projection. Their framework could yield tighter, uniform versions of our Theorem 1 bounds.

3. **Li et al. 2025 (2512.09103)** -- "Natural Geometry of Robust Data Attribution." Shows TRAK scores have 0% Euclidean certification due to "spectral amplification" -- the ill-conditioning of deep representations inflates Lipschitz bounds by > 10000x. Their Natural Wasserstein metric uses the model's own feature covariance to stabilize attribution. *Complementary*: their spectral amplification diagnosis supports our Theorem 1 (the gradient covariance is highly anisotropic), and their Natural Wasserstein metric is a specific choice of M in our phi^T M psi framework.

4. **Wei et al. 2024 (2412.03906)** -- "Final-Model-Only Data Attribution with a Unifying View." Shows all gradient-based TDA methods approximate "further training" in different ways. *Our framework subsumes theirs*: "further training" provides the oracle A* in our Theorem 8, and their unification of gradient-based methods corresponds to different (phi, psi, M) choices in parameter space.

5. **Wang et al. 2023 (2312.17174)** -- "Multi-Modal Information Bottleneck Attribution (M2IB)." Applies information bottleneck to vision-language attribution, learning latent representations that compress irrelevant information. *Tangential but relevant*: their IB-based attribution outperforms gradient/perturbation methods, supporting the general claim that representations optimized for information compression are better for attribution.

6. **Zhang et al. 2025 (2501.18887)** -- "Towards Unified Attribution in XAI, Data-Centric AI, and Mechanistic Interpretability." Position paper arguing feature/data/component attribution share fundamental techniques (perturbation, gradient, linear approximation). *Our phi^T M psi is a concrete mathematical realization of their conceptual unification*, specifically for data attribution.

---

## Summary Table

| Angle | Type | Core Theorem | Pilot Time | Full Time | P(success) | Impact if Succeeds |
|-------|------|-------------|------------|-----------|------------|-------------------|
| 1. Spectral FM1 Theory | New theory (RMT + sketching) | r_eff ~ d explains signal dilution | 15 min | 1h | 65% | High -- first quantitative FM1 bound |
| 2. FM2 Bias Decomposition | Cross-domain (causal inference) | Mean subtraction = deconfounding | 10 min | 45 min | 70% | Medium-High -- formal justification for contrastive scoring |
| 3. phi^T M psi Completeness | New framework (functional analysis) | All linear TDA is bilinear; optimality via SVD | 15 min | 1h | 75% | Very High -- complete taxonomy + optimality conditions |
