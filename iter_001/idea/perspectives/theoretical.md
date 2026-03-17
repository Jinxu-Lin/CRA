# Theoretical Perspective: CRA Research Proposal

## Summary of Theoretical Angles

This perspective grounds the CRA diagnostic framework in rigorous mathematical formalism drawn from signal detection theory, random projection theory, and information-theoretic estimation bounds. I propose three theoretically motivated angles -- each with provable guarantees or falsifiable predictions -- that transform the FM1/FM2 narrative from an empirical observation into a formal theory with quantifiable bounds.

---

## Angle 1: Signal Subspace Detection Theory -- Formalizing FM1 via the Effective Dimensionality Barrier

### Theoretical Foundation

The FM1 (Signal Dilution) diagnosis rests on an implicit claim: attribution signal lives in a low-rank subspace of the gradient space. This can be formalized rigorously using the **subspace signal detection framework** from array signal processing (Van Trees, 2002; Scharf & Friedlander, 1994).

**Definition (Attribution Detection Problem).** Given test point z_test and candidate training point z_train, the attribution problem is equivalent to a binary hypothesis test:

```
H_0: z_train has no influence on f(z_test)  [null: score ~ N(0, sigma_noise^2)]
H_1: z_train influences f(z_test)           [signal: score ~ N(mu_signal, sigma_noise^2)]
```

The test statistic in parameter space is:

```
T_param(z_test, z_train) = g(z_test)^T H^{-1} g(z_train)
```

where g(.) are per-example gradients in R^B and H is the Hessian. In representation space:

```
T_rep(z_test, z_train) = phi(z_test)^T psi(z_train)
```

where phi, psi are feature maps in R^d with d << B.

**Theorem 1 (FM1 as SNR Collapse).** Let Sigma_g = E[g(z)g(z)^T] be the gradient covariance with eigendecomposition Sigma_g = U Lambda U^T, and let r_eff(alpha) = min{k : sum_{i=1}^k lambda_i / sum_j lambda_j >= alpha} be the effective rank at threshold alpha. For a random projection P in R^{k x B} (as in TRAK), the output SNR of the projected attribution score is:

```
SNR_out(k) = SNR_in * (r_eff / k)     when k >> r_eff
SNR_out(k) ~ SNR_in                    when k <= r_eff
```

This follows directly from the Johnson-Lindenstrauss preservation guarantees applied to the signal subspace. When B ~ 10^9 (LLM scale) and r_eff ~ d ~ 10^3, random projection into k ~ 10^3 dimensions wastes ~(B - r_eff)/B fraction of projection capacity on pure noise directions.

**Connection to Hu et al. (2602.10449).** The recent unified theory of random projection for influence functions by Hu, Hu, Ma & Zhao provides the formal machinery we need. They prove that unregularized projection preserves influence iff the projection is injective on range(F), necessitating m >= rank(F). Their "effective dimension" at regularization scale directly corresponds to our r_eff. Crucially, they show that ridge regularization *fundamentally alters the sketching barrier* -- the approximation quality is governed by the effective dimension of F at the regularization scale, not the full rank. This provides the theoretical basis for our claim that representation-space methods operate at the "correct" dimensionality.

**Key Prediction (Testable).** The TRAK dimension sweep should exhibit a phase transition at k ~ r_eff:
- For k < r_eff: LDS improves approximately as sqrt(k/r_eff)
- For k > r_eff: LDS saturates (< 5% improvement per doubling of k)
- The transition point directly measures r_eff from data

This prediction goes beyond the empirical observation in Park et al. (2303.14186) by providing a *rate* for the improvement curve, not just the existence of saturation.

### Mathematical Motivation

The signal subspace interpretation yields a formal **dimension-performance tradeoff**:

```
MSE_projection(k) = sigma_noise^2 / k + sigma_signal^2 * max(0, 1 - k/r_eff)^2
```

The first term (variance) decreases with more projections; the second term (bias from missing signal dimensions) disappears once k >= r_eff. The optimal k* = r_eff minimizes total error, and this k* ~ d for representations extracted from a d-dimensional bottleneck layer.

**Why representation space naturally addresses FM1**: The feature map phi: R^B -> R^d is an *optimal* projection (in the information-theoretic sense) because the network architecture forces all task-relevant information through the d-dimensional bottleneck. This is the trained equivalent of PCA-projection onto the top-d eigenvectors of Sigma_g -- except that the network learns a *nonlinear* projection that may capture signal structure better than linear PCA.

### Guarantees

**Proposition 1.1.** If the gradient covariance Sigma_g has effective rank r_eff(0.95) in [0.5d, 2d], then RepSim (operating in R^d) achieves at least 95% of the attribution signal captured by the full-parameter influence function, while TRAK at k = d captures at most (d/B) * 100% of the noise subspace -- yielding an SNR advantage of approximately B/d.

**Proposition 1.2 (From Hu et al. 2602.10449).** For Kronecker-factored curvatures F = A x E (as in EK-FAC), decoupled sketches P = P_A x P_E preserve influence guarantees despite row correlations violating i.i.d. assumptions. This provides theoretical justification for LoGra-style factored approximations, while simultaneously explaining why they only partially fix FM1: the factored sketch dimension is constrained by the factor dimensions, not r_eff directly.

### Computational Cost

- Eigenspectrum computation on Pythia-70M: ~1 GPU-hour (Lanczos top-500)
- TRAK dimension sweep on Pythia-1B: ~5 GPU-hours (parallelizable)
- Representation covariance analysis: ~0.5 GPU-hours

### Success Probability: 70%

The eigenspectrum computation will succeed (well-established methodology). The r_eff ~ d prediction has moderate confidence -- it depends on the gradient signal structure being dominated by the representation bottleneck, which is theoretically motivated but empirically unverified at LLM scale. The Tong et al. (2602.01312) analysis showing that TRAK's estimated influence preserves rankings despite approximation errors provides indirect support for a low-rank signal structure.

### Failure Modes

- r_eff >> 2d: Signal is distributed across many more dimensions than the representation bottleneck, weakening the FM1 narrative. Would suggest parameter-space methods need *more* dimensions, not fewer.
- r_eff << 0.5d: Signal is even more compressed than expected. While strengthening FM1 claims, it would suggest that representation space itself is partially redundant and PCA-RepSim should work with d' << d dimensions.
- Eigenspectrum lacks a clear "knee": Gradual decay without a sharp rank cutoff would make the "effective rank" concept less crisp and the TRAK phase transition prediction less testable.

---

## Angle 2: Bias Decomposition and the Neyman-Pearson Optimality of Contrastive Scoring

### Theoretical Foundation

FM2 (Common Influence Contamination) can be formalized as a **composite hypothesis testing problem** where the null distribution is contaminated by a structured bias term. The connection to classical detection theory provides both a formal justification for contrastive scoring and a derivation of the optimal test statistic.

**Definition (Bias Decomposition).** For any training example z_train, the representation phi(z_train) admits a decomposition:

```
phi(z_train) = phi_shared + phi_task(z_train) + epsilon(z_train)
```

where:
- phi_shared = E[phi(z)] is the common component (pre-training knowledge)
- phi_task(z_train) is the task-specific component (what attribution should capture)
- epsilon(z_train) is noise

The standard attribution score is:

```
score_standard = phi(z_test)^T psi(z_train)
     = phi(z_test)^T phi_shared  +  phi(z_test)^T phi_task(z_train)  +  phi(z_test)^T epsilon(z_train)
       [FM2 bias term]              [true signal]                       [noise]
```

**Theorem 2 (FM2 Severity Bound).** The FM2 contamination ratio, defined as R_FM2 = ||phi_shared||^2 / E[||phi_task||^2], satisfies:

```
R_FM2^{param} >= R_FM2^{rep} * (B/d)
```

In parameter space, the shared gradient component is amplified by the ambient dimensionality B, while in representation space the shared component is bounded by d. This formally explains why contrastive scoring yields larger gains in parameter space (Hypothesis H2).

**Proof sketch.** The gradient of the loss decomposes as g(z) = J^T * phi(z) where J is the Jacobian of the representation map. The shared gradient g_shared = J^T * phi_shared has ||g_shared||^2 = phi_shared^T J J^T phi_shared ~ ||phi_shared||^2 * ||J||_F^2 / d. Since ||J||_F^2 scales with B (total parameters), the amplification factor B/d emerges.

**Theorem 3 (Optimality of Contrastive Scoring).** Under the bias decomposition, the Neyman-Pearson optimal test for "z_train influences z_test" at significance level alpha is:

```
T_NP(z_test, z_train) = [phi(z_test) - phi_shared]^T Sigma_noise^{-1} [psi(z_train) - psi_shared]
```

This is exactly contrastive scoring (mean subtraction) composed with whitening (Sigma_noise^{-1}). The optimality follows from the Generalized Likelihood Ratio Test (GLRT) for detecting a signal in structured (colored) noise:

- Mean subtraction removes the deterministic bias (FM2 fix)
- Whitening decorrelates the residual noise (optimal weighting)

**Connection to DDA.** Pang et al.'s DDA (2410.01285) empirically discovered that mean subtraction contributes ~55pp of their improvement. Our Theorem 3 provides the formal justification: DDA's debias step is an *approximation* to the Neyman-Pearson optimal detector, with the 55pp gain corresponding to the reduction in FM2 bias. The remaining gap between DDA and the optimal detector is attributable to the absence of whitening (M = I instead of M = Sigma_noise^{-1}).

### The phi^T M psi Unification as a Generalized Likelihood Ratio

Every method in the phi^T M psi family corresponds to a specific choice of test statistic for the attribution detection problem:

| Method | phi/psi | M | FM1 Fix? | FM2 Fix? | Detection Theory Analog |
|--------|---------|---|----------|----------|------------------------|
| IF (full) | gradient | H^{-1} | No (R^B) | No | Optimal in R^B if Sigma_noise = H |
| TRAK | projected gradient | (Phi Phi^T)^{-1} | Partial (R^k) | No | Matched subspace detector |
| RepSim | representation | I | Yes (R^d) | No | Energy detector in signal subspace |
| DDA | gradient - mean | Regularized | No | Yes | GLRT with bias removal |
| RepSim+contrastive | representation - mean | I | Yes | Yes | Subspace GLRT |
| **Whitened CRA** | representation - mean | Sigma_noise^{-1} | Yes | Yes | **Neyman-Pearson optimal** |

This table provides the first systematic mapping between TDA methods and signal detection theory. The key theoretical prediction: *Whitened CRA achieves the highest attribution quality among all linear methods in the phi^T M psi family*, with the improvement most pronounced when Sigma_noise is anisotropic (structured FM2 residuals).

### Guarantees

**Proposition 2.1 (Contrastive Improvement Bound).** The LDS improvement from contrastive scoring is lower-bounded by:

```
Delta_LDS >= C * R_FM2 / (1 + R_FM2)
```

where C is a task-dependent constant related to the label-feature correlation. For high-FM2 tasks (factual attribution, where pre-training knowledge dominates), R_FM2 >> 1 and Delta_LDS approaches C -- predicting large gains. For low-FM2 tasks (toxicity filtering, where the signal is behavioral), R_FM2 may be moderate.

**Proposition 2.2 (Whitening Improvement Bound).** The additional improvement from whitening (Sigma_noise^{-1}) over identity (M=I) is bounded by:

```
Delta_whitening <= (kappa(Sigma_noise) - 1) / kappa(Sigma_noise)
```

where kappa is the condition number. If Sigma_noise is near-isotropic (kappa ~ 1), whitening provides negligible benefit. If Sigma_noise is highly anisotropic (kappa >> 1), whitening can substantially improve performance.

### Computational Cost

- Bias decomposition measurement: ~0.5 GPU-hours (mean computation over training set)
- Whitened attribution with Ledoit-Wolf: ~0.5 GPU-hours
- Contrastive strength sweep (alpha in {0, 0.1, ..., 1.0}): ~1 GPU-hour
- Total: ~2 GPU-hours

### Success Probability: 60%

The bias decomposition formalism is mathematically clean and will hold by construction. The empirical question is whether the Neyman-Pearson optimal detector (whitened contrastive) actually outperforms simpler alternatives. If Sigma_noise is near-isotropic (H9 holds), whitening adds nothing and the theory reduces to "mean subtraction is sufficient" -- a weaker but still valid theoretical contribution.

### Failure Modes

- Sigma_noise estimation degenerates: If d > n_samples, the covariance inverse is unreliable even with shrinkage. The Ledoit-Wolf estimator mitigates this but cannot overcome fundamental sample-size limitations.
- Non-Gaussian noise: The Neyman-Pearson optimality assumes Gaussian noise. If the attribution noise is heavy-tailed (likely for rare training examples), robust alternatives (e.g., Huber-weighted whitening) may be needed.
- Contrastive scoring hurts some methods: If phi_shared contains *useful* information for certain attribution tasks (e.g., factual knowledge attribution where pre-training is relevant), subtracting the mean removes signal along with bias.

---

## Angle 3: Information-Theoretic Lower Bound on Attribution Error -- The FM1/FM2 Tradeoff Surface

### Theoretical Foundation

The most ambitious theoretical contribution: derive a *fundamental lower bound* on attribution error that holds for any method in the phi^T M psi family. This bound reveals whether FM1 and FM2 are truly independent (as the proposal claims) or conjugate (as the innovator perspective suggests).

**Definition (Attribution Error).** For a linear attribution method T(z_test, z_train) = phi(z_test)^T M psi(z_train), the mean squared attribution error is:

```
MSE(T) = E[(T - T_true)^2] = Bias^2(T) + Var(T)
```

where T_true is the counterfactual ground truth (effect of removing z_train on f(z_test)).

**Theorem 4 (FM1/FM2 Decomposition of MSE).** Under the bias decomposition and Gaussian noise model:

```
MSE(T) = ||M phi_shared||^2 * ||psi_shared||^2    [FM2: bias from common influence]
        + trace(M Sigma_task M^T Sigma_noise)      [FM1: variance from dimension]
        + cross_term(M, phi_shared, Sigma_noise)    [FM1-FM2 interaction]
```

The interaction term determines whether FM1 and FM2 are independent:
- If cross_term = 0: FM1 and FM2 are orthogonal (original CRA claim, H3 holds)
- If cross_term > 0: They are positively coupled (fixing one exacerbates the other)
- If cross_term < 0: They are synergistic (fixing one partially fixes the other)

**Theorem 5 (Cramer-Rao-type Lower Bound).** For any unbiased linear attribution estimator in the phi^T M psi family, the attribution variance satisfies:

```
Var(T) >= phi_signal^T I_F^{-1} phi_signal
```

where I_F is the Fisher information matrix of the attribution model. This bound is the analog of the Cramer-Rao bound for the attribution problem.

For the specific case of Gaussian noise with covariance Sigma_noise:

```
I_F = Sigma_noise^{-1}
```

and the minimum-variance unbiased estimator is exactly the whitened matched filter T* = phi^T Sigma_noise^{-1} psi, achieving the bound with equality.

**Connection to classical estimation theory.** Diskin, Eldar & Wiesel (2110.12403) established that deep learning estimators can approach the Cramer-Rao bound via bias-constrained optimization. Our contribution is mapping this framework to the attribution problem: the "estimand" is the counterfactual influence, the "observations" are the representations, and the "noise" is the residual after bias removal.

**Theorem 6 (Tradeoff Surface).** For the class of regularized linear attributors T_lambda = phi^T (Sigma_noise + lambda I)^{-1} psi, the Pareto frontier of the (FM1_severity, FM2_severity) tradeoff is:

```
FM1(lambda) * FM2(lambda) >= C(phi_signal, Sigma_noise, phi_shared)
```

where C depends on the alignment between the signal subspace and the bias subspace. If these subspaces are orthogonal, C = 0 and FM1/FM2 are truly independent (supporting H3). If they overlap, C > 0 and a nontrivial tradeoff exists.

**Testable Prediction.** The sign and magnitude of the interaction term in the 2x2 factorial directly tests this theory:
- Near-zero interaction -> Orthogonal subspaces -> FM1/FM2 independent -> Original CRA narrative
- Negative interaction -> Overlapping subspaces -> Fixing FM1 partially addresses FM2 -> Conjugacy narrative (innovator Angle 3)
- Positive interaction -> Anti-aligned subspaces -> Fixing both simultaneously is harder -> New theoretical insight

### The SNR Output Framework

Beyond error bounds, we derive a practical **per-query reliability metric**:

```
SNR_out(z_test) = ||phi_task(z_test)||^2 / (phi_task(z_test)^T Sigma_noise phi_task(z_test))
```

This output SNR predicts, for each test query, how reliable its attribution scores will be. Queries with high SNR_out will have accurate attributions; queries with low SNR_out should be flagged as unreliable. This provides the first **per-query confidence estimate** for TDA, analogous to CFAR (Constant False Alarm Rate) normalization in radar.

**Testable Prediction.** If SNR_out is a valid reliability predictor, then:
1. Restricting evaluation to top-50% SNR_out queries should improve LDS by >= 5pp
2. The correlation between SNR_out and per-query attribution accuracy (measured by leave-one-out ground truth) should be r >= 0.3

### Computational Cost

- Fisher information computation: ~1 GPU-hour (covariance estimation + inversion)
- SNR_out computation per query: negligible (dot products)
- Tradeoff surface visualization: ~0.5 GPU-hours
- Total: ~1.5 GPU-hours

### Success Probability: 45%

This is the highest-risk, highest-reward angle. The FM1/FM2 decomposition of MSE is mathematically sound, but the Gaussian assumptions are strong and the empirical validation (interaction term sign) depends on the 2x2 factorial results. The per-query SNR_out reliability metric is novel and practical -- even if the lower bound is loose, SNR_out may correlate with accuracy.

### Failure Modes

- The Cramer-Rao bound is too loose: If the gap between the bound and achievable performance is large, the bound provides no practical guidance.
- Gaussian noise assumption fails: Real attribution noise may be heavy-tailed or multimodal, invalidating the Fisher information framework.
- SNR_out does not predict accuracy: If per-query reliability is dominated by factors outside the linear model (e.g., nonlinear interactions, out-of-distribution effects), the SNR metric is uninformative.
- The interaction term is inconclusive: If it is small but non-zero, neither the "independent" nor "conjugate" narrative is strongly supported.

---

## Synthesis: Unified Theoretical Framework for CRA

### Hierarchy of Results

The three angles form a **theorem hierarchy** of increasing generality:

1. **Angle 1 (FM1 Formalism)**: Signal subspace detection theory -> Effective rank bound -> TRAK dimension sweep prediction. *Self-contained and testable with minimal assumptions.*
   - Strengthened by Hu et al. (2602.10449): Their unified random projection theory for influence functions provides the exact machinery for formalizing when and why projection dimension matters.

2. **Angle 2 (FM2 Formalism + Optimality)**: Bias decomposition -> Neyman-Pearson optimal detector -> Whitened contrastive attribution. *Requires the bias decomposition to hold empirically.*
   - Strengthened by Tong et al. (2602.01312): Their result that TRAK preserves rankings despite approximation errors suggests the signal structure is robust, supporting the bias-signal decomposition.

3. **Angle 3 (Lower Bound + Tradeoff)**: MSE decomposition -> FM1/FM2 interaction characterization -> Cramer-Rao bound -> Per-query SNR reliability. *Requires Gaussian assumptions and measurable covariance structure.*

### Integration with the phi^T M psi Framework

The bilinear framework phi(z_test)^T M psi(z_train) is not merely notational -- it is the **generalized detection statistic** for a composite hypothesis test with two interference sources (FM1 and FM2). The theoretical contribution organizes the framework as:

```
phi, psi: Feature maps that determine the detection subspace (addresses FM1)
M: Metric tensor that determines the noise weighting (addresses FM2 residuals)
Contrastive: Mean subtraction that removes deterministic bias (addresses FM2 directly)
```

The optimal configuration is:
- phi, psi: representations from the signal-rich R^d subspace (FM1-optimal)
- Mean subtraction applied to both phi and psi (FM2-optimal)
- M = Sigma_noise^{-1}: Neyman-Pearson optimal weighting (residual-optimal)

This is the **Whitened Contrastive Representation Attribution** -- the unique member of the phi^T M psi family that simultaneously achieves:
- Minimum FM1 severity (by operating in R^d)
- Minimum FM2 bias (by contrastive scoring)
- Minimum residual variance (by optimal whitening)

### Formal Instantiation Table (Theorem 7)

| Method | phi(z) | psi(z) | M | Space | Contrastive | Theoretical Status |
|--------|--------|--------|---|-------|-------------|-------------------|
| Classical IF | nabla_theta L(z) | nabla_theta L(z) | H^{-1} | R^B | No | Optimal if noise ~ N(0, H) |
| TRAK | P * nabla L(z) | P * nabla L(z) | (Phi Phi^T)^{-1} | R^k | No | JL-approximate IF |
| RepSim | h_L(z) | h_L(z) | I | R^d | No | Suboptimal (ignores FM2, M=I) |
| RepT | delta_h(z) | delta_h(z) | I | R^d | Implicit | Gradient tracking approximation |
| In-the-Wild | h_post(z) - h_pre(z) | h_post(z) - h_pre(z) | I | R^d | Temporal | DiD estimator (DPO only) |
| Concept Influence | encoder(z) | encoder(z) | I | R^d' | Implicit | Learned subspace |
| AirRep | f_theta(z) | f_theta(z) | I | R^d' | Trained | Learned optimal phi |
| DDA | nabla L(z) - mean | nabla L(z) - mean | Regularized | R^B | Yes | GLRT in R^B |
| **Whitened CRA** | h_L(z) - mean | h_L(z) - mean | Sigma_noise^{-1} | R^d | Yes | **NP-optimal (Thm 3)** |

### Relationship to Recent Theoretical Work

- **Hu et al. (2602.10449)**: Their unified theory of random projection for influence functions provides the formal foundation for Angle 1. Key result: exact preservation requires m >= rank(F), and regularization fundamentally changes this barrier. Our CRA framework interprets representation extraction as the "ideal" projection that achieves rank(F)-dimensional projection without explicit eigendecomposition.

- **Tong et al. (2602.01312)**: Their analysis of TRAK shows that despite significant approximation errors, TRAK preserves influence *rankings*. This is consistent with our FM1 theory: random projection loses magnitude information (SNR collapse) but approximately preserves relative ordering because the signal subspace structure is maintained. Their characterization of when TRAK breaks down complements our prediction of the dimension sweep phase transition.

- **Bae et al. (2405.12186v2)**: Source's unrolling-based approach addresses the "multi-stage training" limitation that our FM1/FM2 framework does not directly treat. In the phi^T M psi taxonomy, Source uses phi and psi that account for optimization trajectory, not just final-checkpoint representations. This suggests a natural extension: temporal phi^T M psi where the feature maps incorporate training dynamics (connecting to the innovator's CATCL angle).

- **Stankovic & Mandic (2108.11663v3, 2108.10751)**: Their matched filtering interpretation of CNNs provides a precedent for our matched filter interpretation of attribution. Key insight: convolution layers perform matched filtering to detect features in input data. We extend this to the *output* side: attribution scores perform matched filtering to detect training data influence in model predictions.

---

## Risk Assessment Summary

| Angle | Novelty | Mathematical Risk | P(success) | Fallback Value |
|-------|---------|-------------------|------------|---------------|
| 1. FM1 Subspace Detection | Medium-High | Low | 70% | Eigenspectrum measurement is novel regardless of r_eff ~ d |
| 2. NP-Optimal Attribution | High | Medium | 60% | Bias decomposition formalism is rigorous even if whitening underperforms |
| 3. CR Lower Bound + SNR | Very High | High | 45% | Per-query SNR metric is practical even if bound is loose |

**Expected value**: At least one angle produces a strong theorem with P = 1 - (0.30)(0.40)(0.55) = 93.4%.

**Recommended presentation order**: Lead with Angle 2 (bias decomposition + optimality) as the main theoretical framework, support with Angle 1 (FM1 mechanism) as constructive evidence, and present Angle 3 (lower bound) as "theoretical depth" for the ambitious version of the paper.

### Key Warning: The Vacuous Framework Risk

The phi^T M psi framework, as currently stated, is *too general* -- almost any bilinear scoring function can be written in this form. The theoretical angles above address this by deriving *non-trivial predictions*:
1. The TRAK dimension sweep phase transition at k ~ r_eff (Angle 1)
2. The asymmetric contrastive improvement bound (Angle 2)
3. The FM1/FM2 interaction sign prediction (Angle 3)
4. The per-query SNR reliability correlation (Angle 3)

If none of these predictions validate, the framework is indeed vacuously universal and should be presented as notation rather than theory. This is a critical intellectual honesty checkpoint.

---

## Key References Integrated

### Directly Relevant to CRA Theory
- Hu, Hu, Ma & Zhao (2602.10449): **A Unified Theory of Random Projection for Influence Functions** -- provides formal JL-based guarantees for when projection preserves influence; establishes effective dimension as the key quantity. *Core reference for Angle 1.*
- Tong, Ghosh, Zou & Maleki (2602.01312): **Imperfect Influence, Preserved Rankings: A Theory of TRAK** -- shows TRAK preserves rankings despite approximation errors; characterizes error regimes. *Supports FM1 low-rank signal hypothesis.*
- Park, Georgiev, Ilyas, Leclerc & Madry (2303.14186): **TRAK** -- establishes random projection methodology for attribution; provides empirical dimension sensitivity evidence.
- Sun, Liu, Kandpal, Raffel & Yang (2505.18513): **AirRep** -- learned representations optimized for attribution; validates representation-space TDA at scale.
- Bae, Lin, Lorraine & Grosse (2405.12186v2): **Source** -- approximate unrolling connecting implicit differentiation and unrolling approaches; extends the phi^T M psi framework to training trajectories.

### Signal Processing Foundations
- Stankovic & Mandic (2108.11663v3): **CNNs as Matched Filters** -- establishes the matched filtering interpretation of neural network operations; precedent for our attribution-as-detection framework.
- Van Trees (2002): *Optimum Array Processing* -- foundational reference for subspace signal detection and Neyman-Pearson optimal detectors.
- Scharf & Friedlander (1994): *Matched subspace detectors* -- formal framework for detection in low-rank signal subspaces with colored noise.

### Estimation Theory
- Diskin, Eldar & Wiesel (2110.12403): **Learning to Estimate Without Bias** -- connects deep learning estimators to Cramer-Rao bounds; provides the template for our attribution CRB. *Foundational for Angle 3.*
- Ghojogh et al. (2108.04172): **JL Lemma Tutorial** -- comprehensive treatment of random projection theory including sparse projections and kernel extensions.

### TDA Benchmarks and Methods
- Jiao et al. (2507.09424): **DATE-LM** -- standard benchmark; "no single method dominates" finding motivates theoretical unification.
- Pang et al. (2410.01285): **DDA** -- empirical discovery of contrastive scoring's 55pp improvement; our Theorem 3 provides the theoretical justification.
- Li et al. (2409.19998): **RepSim >> IF on LLMs** -- core observation that CRA explains through FM1/FM2 formalism.
- Hong et al. (2509.23437): **Better Hessians Matter** -- Hessian quality as competing explanation; our H6 control experiment addresses this.
