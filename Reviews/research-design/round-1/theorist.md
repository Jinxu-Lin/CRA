## [Theorist] 理论家视角

### 理论基础审查

**核心操作的理论动机**：

CRA proposes a three-bottleneck decomposition of TDA failure: Hessian error, FM1 (signal dilution), FM2 (common contamination). The theoretical backbone is a signal-processing analogy: matched filtering (representation space concentrates signal) and differential detection (contrastive scoring removes common-mode noise).

**评价**：有直觉未形式化 — The signal-processing analogy is suggestive but not rigorous, and the authors correctly acknowledge this (method-design.md §7.1). The core theoretical argument for FM1 rests on the Johnson-Lindenstrauss (JL) concentration phenomenon: in R^B with B ~ 10^9, per-sample gradient inner products concentrate around zero. This is mathematically sound as a statement about random vectors, but the critical gap is that **gradients are NOT random vectors** — they are structured by the loss landscape, training data, and model architecture. The JL argument establishes an upper bound on what could happen in the worst case, not what does happen with real gradients.

**Specific theoretical assessments**:

1. **Component A (RepSim)**: The claim that h^(l) in R^d "concentrates task-relevant signal" is an empirical hypothesis, not a theoretical result. Representation space is a nonlinear learned transformation, and there is no guarantee it preserves influence-relevant features (the authors note this caveat in §5 Component A). The matched filtering analogy breaks because matched filtering requires knowing the signal template — representation space is a learned, not designed, compression. **Rating: Intuition without formal support, but testable.**

2. **Component B (Contrastive scoring)**: The differential detection analogy is more solid. Subtracting base-model scores to isolate fine-tuning-specific influence has a clear information-theoretic interpretation: if I_total = I_pretrain + I_finetune + I_interaction, then I_ft - I_base approximates I_finetune under the assumption that I_interaction is small. **This linearity assumption (that influence decomposes additively) is the key hidden assumption.** For linear models, influence IS additive. For deep networks, the interaction term could be substantial due to nonlinear feature interactions during fine-tuning. **Rating: Reasonable with explicit linearity caveat.**

3. **2x2 ANOVA framework**: The statistical design is sound in principle. The interpretation of interaction terms (§3.2 of experiment-design.md) has proper thresholds. However, **the ANOVA assumes that LDS is a well-behaved continuous metric with approximately normal per-sample distributions**. If LDS contributions are heavy-tailed (which is plausible for attribution scores), the permutation test is appropriate (as designed), but bootstrap CIs may need adjustment. **Rating: Statistically sound with minor distributional caveats.**

### 数学欠账

1. **JL-to-gradient leap**: The JL concentration argument (§7.2) correctly states that random vectors in R^B are approximately orthogonal. But then applies this to gradients, which have low-rank structure imposed by the training objective. The actual SNR in parameter space depends on the alignment of per-sample gradient subspaces, not just dimensionality. Specifically, if two training samples contribute to the same output behavior, their gradients share a common component in the loss-relevant subspace even in R^B. **The dimensionality argument is necessary but not sufficient for FM1.** FM1 also requires that the task-relevant gradient subspace has rank << B, which is plausible but unverified.

2. **Influence linearity assumption**: Component B's contrastive scoring assumes I(z_test, z_train; M_ft) - I(z_test, z_train; M_base) isolates fine-tuning influence. For exact influence functions, this decomposition is NOT valid because IF depends on the Hessian at M_ft, not a linear combination of influences at different parameter points. The contrastive subtraction is a heuristic that works well empirically (DDA's -55pp evidence) but lacks theoretical justification beyond the linear regime. **This is the most significant mathematical gap.**

3. **RepSim captures correlation, not causation**: The authors correctly identify this (§5 Component A critical caveat), but the design does not include a theoretical analysis of WHEN correlation-based attribution approximates causal attribution. A sufficient condition would be: if the representation mapping is smooth and the fine-tuning perturbation is small, then representational similarity approximates influence up to first order. This Taylor expansion argument could strengthen the theoretical case substantially.

### 已有理论支撑或反例

**支撑**:
- **JL Lemma**: Directly supports the claim that inner products in R^B concentrate. The ratio B/log(N)^2 ~ 6*10^6 means we are deeply in the concentration regime. This is rigorous.
- **Influence function theory (Cook & Weisberg, 1982)**: For convex losses, IF exactly measures leave-one-out impact. MAGIC's near-perfect LDS confirms this for deterministic training. The theory is solid for the Hessian bottleneck.
- **Representer point theorem (Yeh et al., 2018)**: For the last layer of a neural network with L2 regularization, the representer point value equals the activation inner product weighted by training coefficient. This provides partial theoretical justification for RepSim at the last layer — but ONLY for the last layer and ONLY with L2 regularization. CRA uses middle layers and does not require L2 regularization, so this support is partial.

**反例/挑战**:
- **MAGIC's LDS ~0.95-0.99**: This is the strongest theoretical challenge. If exact parameter-space IF achieves near-perfect attribution, then FM1 (signal dilution in parameter space) is NOT the bottleneck when Hessian error is eliminated. The "signal dilution" narrative requires that even with a perfect Hessian, parameter-space similarity should fail due to dimensionality — but MAGIC shows it doesn't. **The authors handle this correctly via the MAGIC decision rule in problem-statement.md §1.3, but the theoretical tension remains unresolved.**
- **TRAK's random projection theory**: TRAK uses random projections of gradients, justified by JL. If JL works for TRAK's projection, why does the same concentration argument imply FM1? The answer is that TRAK projects to a fixed low-dimensional space (k ~ 4096) that may not align with the task-relevant subspace. But representation space h^(l) is learned to be task-relevant. **This distinction is crucial and under-articulated in the method-design.md.**

**理论-实践 gap**: The theoretical analysis is "guiding" for the broad direction (representation space should have higher SNR) but "decorative" for the specific predictions (exact LDS values, interaction magnitudes). The 2x2 experimental design is what actually tests the framework, not the theory.

### 理论强化建议

The single most impactful theoretical addition would be a **first-order Taylor expansion argument** connecting representational similarity to influence: For a loss function L, fine-tuning from theta_base to theta_ft with small perturbation delta, the influence of z_train on z_test's loss can be approximated as:

I(z_test, z_train) ~ nabla_h L(z_test) . (h_ft(z_train) - h_base(z_train))

where h is the representation at the critical layer. This directly connects RepSim (cosine similarity of representations) to influence (change in loss), and the approximation quality depends on the magnitude of delta (fine-tuning perturbation). Under LoRA (small delta), this approximation should be tighter; under full-FT (larger delta), it degrades — which would PREDICT the LoRA vs Full-FT result in Experiment 3. This argument would transform the signal-processing analogy from decorative to predictive.
