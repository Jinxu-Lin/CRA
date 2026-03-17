# Innovator Perspective: CRA Research Proposal

## Summary of Creative Angles

I propose three unconventional angles that push the CRA framework beyond its current "diagnose + unify + benchmark" formulation. The core proposal maintains the FM1/FM2 diagnostic thesis but adds a **prescriptive, adaptive** dimension that transforms it from a retrospective analysis into a forward-looking method design principle.

---

## Angle 1: Multi-Scale Attribution Telescope -- Layer-Adaptive Bilinear Attribution (LABA)

### The Counter-Intuitive Insight

The current CRA proposal treats representation-space attribution as a monolithic operation at a fixed layer (typically last-layer or a single "best" layer). But Vitel & Chhabra (2511.04715) demonstrated that **middle attention layers** outperform both first and last layers, contradicting the prior consensus. This finding is devastating for the fixed-layer phi^T M psi framework -- it suggests that **the optimal attribution signal lives at different depths for different queries**.

### Hypothesis

**H-Innov-1**: The FM1 signal dilution severity varies across layers in a predictable, query-dependent pattern. Specifically, the effective rank r_eff(l) of layer-l gradient covariance follows a U-shaped curve (high at embedding, low at middle layers, rising again at output), and the optimal attribution layer for each query correlates with the layer where its task-specific signal-to-noise ratio peaks.

### Proposed Method: LABA (Layer-Adaptive Bilinear Attribution)

Instead of phi_L^T M psi_L at a fixed layer L, compute:

```
score(z_test, z_train) = sum_l w_l(z_test) * phi_l(z_test)^T M_l psi_l(z_train)
```

where w_l(z_test) are **query-adaptive layer weights** derived from each layer's local SNR estimate:

```
w_l(z_test) = softmax(SNR_l(z_test))
SNR_l(z_test) = ||phi_l(z_test) - mean(phi_l)||^2 / var(phi_l)
```

This turns the phi^T M psi framework from a descriptive taxonomy into a **prescriptive multi-scale detector** -- a "telescope" that automatically focuses on the layer where the attribution signal is strongest.

### Why This Is Novel

- Vitel & Chhabra showed middle layers are better *on average* but did not propose query-adaptive selection
- AirRep (2505.18513) learns a fixed encoder; LABA adapts *per-query* without training
- The signal processing analogy: LABA is a **multi-band matched filter** where each layer is a frequency band, and the query determines which band to attend to

### Experimental Plan

1. **Layer SNR profile** (30 min): On Pythia-1B, compute r_eff(l) and per-query SNR_l for l in {0, 4, 8, 12, 16, 20, 24, 28, 32} across 500 DATE-LM test queries. Verify U-shaped r_eff curve.
2. **LABA vs. fixed-layer RepSim** (30 min): Compare LABA-weighted attribution vs. last-layer, middle-layer, and oracle-layer RepSim on DATE-LM data selection (LDS).
3. **Ablation of weighting schemes** (30 min): uniform, SNR-adaptive, attention-norm-adaptive, learned (logistic regression on layer features)

### Computational Cost

- Total: ~1.5 hours on 1x RTX 4090
- Storage: 9 layers x 2048-dim representations for ~10K training points = ~150 MB
- No model retraining required

### Success Probability: 55%

**Risk**: If representation covariance is approximately isotropic at ALL layers (H9 holds uniformly), then w_l will be nearly uniform and LABA degenerates to ensemble averaging with no query-adaptive benefit. Mitigation: even uniform multi-layer averaging may outperform single-layer by ~2-3pp through variance reduction (an acceptable fallback finding).

### Failure Modes

- Layer representations may be highly correlated (consecutive layers differ by small residuals), making multi-layer combination redundant
- SNR estimation from finite samples may be noisy, especially for rare query types
- If the U-shaped r_eff curve does not hold, the theoretical motivation collapses

---

## Angle 2: Causal Attribution via Temporal Contrastive Learning (CATCL) -- Exploiting Checkpoint Trajectories

### The Cross-Domain Transfer

The current CRA proposal treats FM2 (common influence contamination) as a static bias to be subtracted. But in causal inference, the gold standard for removing confounders is **Difference-in-Differences (DiD)**, which exploits temporal variation. Modern LLM training produces a trajectory of checkpoints -- this temporal structure is an untapped goldmine for attribution.

The cross-domain insight comes from **econometrics**: just as DiD uses pre/post treatment variation to remove time-invariant confounders, we can use pre/post fine-tuning checkpoint differences to remove pre-training knowledge contamination (FM2) mechanistically rather than statistically.

### Hypothesis

**H-Innov-2**: Temporal contrastive attribution, defined as:

```
score_DiD(z_test, z_train) = [phi_post(z_test) - phi_pre(z_test)]^T [psi_post(z_train) - psi_pre(z_train)]
```

where phi_pre, phi_post are representations from pre-trained and fine-tuned checkpoints respectively, will:
(a) Eliminate FM2 contamination without requiring explicit mean-subtraction heuristics
(b) Outperform standard contrastive scoring (mean-subtraction) by >=3pp on DATE-LM factual attribution
(c) Satisfy the "parallel trends" assumption: pre-training representation drift is approximately uniform across training examples

### Why This Is Novel

- DDA (2410.01285) uses debias scoring but as an ad-hoc mean subtraction, not grounded in causal econometrics
- In-the-Wild (2602.11079) uses activation differences but only for DPO preference pairs, not general attribution
- RepT (2510.02334) uses gradient-based tracking through training but not the checkpoint-difference formulation
- No existing work frames TDA deconfounding as a proper DiD estimator with testable parallel trends

### The phi^T M psi Integration

CATCL is a natural instantiation of the bilinear framework:

```
phi(z) = phi_post(z) - phi_pre(z)     (temporal difference feature map)
psi(z) = psi_post(z) - psi_pre(z)     (temporal difference feature map)
M = I                                   (identity -- FM2 already removed by differencing)
```

This provides the bilinear framework with **causal interpretation**: M=I works because temporal differencing has already removed the structured confounding that M=Sigma_noise^{-1} was designed to address. This is a testable prediction: whitened CATCL (M != I) should NOT outperform standard CATCL (M=I), whereas whitened RepSim SHOULD outperform standard RepSim.

### Experimental Plan

1. **Checkpoint extraction** (15 min): Save Pythia-1B representations at pre-training checkpoint and after fine-tuning on DATE-LM training set. Both representations needed anyway for baseline experiments.
2. **Parallel trends test** (15 min): For 1000 training examples, compute ||phi_post - phi_pre|| variance across examples. If CV (coefficient of variation) < 0.5, parallel trends approximately holds.
3. **CATCL vs. RepSim vs. DDA** (30 min): Head-to-head comparison on DATE-LM factual attribution (P@K) and data selection (LDS).
4. **Whitening ablation** (30 min): Verify prediction that whitening improves RepSim but not CATCL.

### Computational Cost

- Total: ~1.5 hours on 1x RTX 4090
- Requires access to pre-trained Pythia-1B checkpoint (publicly available on HuggingFace)
- Storage: 2x representation matrices (pre + post) = ~160 MB

### Success Probability: 45%

**Risk**: The parallel trends assumption may fail badly -- pre-training representations drift non-uniformly, especially for in-distribution vs. out-of-distribution training examples. Mitigation: report DiD with reweighting (IPW-DiD) as a robustness check. Even if CATCL doesn't beat mean-subtraction, the causal framing provides theoretical grounding for WHY mean-subtraction works.

### Failure Modes

- Fine-tuning may not change representations enough for the difference signal to dominate noise (especially for Pythia-1B with limited fine-tuning)
- Parallel trends violation could introduce new biases worse than FM2
- The pre-trained checkpoint may encode domain-specific structure that gets removed by differencing, accidentally discarding useful attribution signal

---

## Angle 3: The Attribution Uncertainty Principle -- FM1 and FM2 Are Conjugate (New Theoretical Framework)

### The Boldest Claim

The current CRA proposal treats FM1 (signal dilution) and FM2 (common influence contamination) as **independent, orthogonal** defects with H3 testing their additivity. I propose a more radical theoretical claim: FM1 and FM2 are **conjugate** in the sense of the uncertainty principle -- they represent complementary views of the same underlying information bottleneck.

### The Theoretical Argument

Consider the attribution score as a detection problem. The test statistic is:

```
T(z_test, z_train) = phi(z_test)^T M psi(z_train)
```

There are two noise sources:
- **FM1 noise**: high-dimensional projection noise (random directions in R^B dominate signal)
- **FM2 noise**: structured bias (pre-training knowledge creates correlated background)

The key insight from signal processing: **dimension reduction (fixing FM1) amplifies structured bias (worsening FM2), while bias removal (fixing FM2) increases variance (worsening FM1)**. This is exactly the bias-variance tradeoff recast in signal processing language.

Formally, for any linear attribution method in the phi^T M psi family:

```
MSE(T) = Bias^2(T) + Var(T)
       = [FM2 severity]^2 + [FM1 severity]
```

where:
- FM1 severity = trace(M * Sigma_noise) / ||phi_signal||^2  (variance term, grows with dimension)
- FM2 severity = phi_shared^T M phi_shared / ||phi_task||^2  (bias term, structured)

The "uncertainty principle" states:

```
FM1_severity * FM2_severity >= C * ||phi_signal||^2 / ||phi_total||^2
```

This lower bound means you cannot simultaneously minimize both FM1 and FM2 without improving the underlying signal strength (phi_signal). The optimal tradeoff is achieved by M = Sigma_total^{-1} (the Wiener filter), which is exactly the whitened matched filter proposed in the original CRA paper -- but now with a **fundamental optimality guarantee** rather than a heuristic motivation.

### Why This Is Novel and Counter-Intuitive

- The original CRA proposal predicts FM1 and FM2 are **approximately additive** (H3)
- The conjugate hypothesis predicts **sub-additivity** -- fixing both simultaneously should yield less than the sum of individual fixes
- This reframes the 2x2 factorial result: a negative interaction term is not a failure of orthogonality, but evidence of conjugacy
- It provides a **fundamental lower bound** on attribution error for any linear method, similar to the Cramer-Rao bound in estimation theory

### Testable Predictions

**H-Innov-3a**: The 2x2 factorial interaction term is **negative** (sub-additive) on at least 2/3 DATE-LM tasks. This is the opposite of what a "synergistic" interaction would predict.

**H-Innov-3b**: The whitened matched filter (M = Sigma_noise^{-1}) achieves the lowest MSE among all M choices in the phi^T M psi family, and its improvement over M=I is larger when FM1 and FM2 severities are more balanced (neither dominates).

**H-Innov-3c**: For fixed representation dimension d, there exists an optimal contrastive strength alpha* (interpolating between standard and fully contrastive scoring) that minimizes MSE. Neither alpha=0 (no contrastive) nor alpha=1 (full contrastive) is optimal -- the optimum lies at 0 < alpha* < 1.

### Experimental Plan

1. **Interaction sign test** (0 min, reuses 2x2 factorial data): Check sign of interaction term in 2x2 ANOVA. If negative on >=2/3 tasks, supports conjugacy over independence.
2. **FM1/FM2 severity measurement** (30 min): For each DATE-LM task, directly compute FM1 severity (trace ratio) and FM2 severity (bias ratio) at each layer. Plot FM1 vs FM2 tradeoff curve.
3. **Contrastive strength sweep** (45 min): Sweep alpha in {0, 0.1, 0.2, ..., 1.0} for RepSim on all 3 DATE-LM tasks. Check whether optimal alpha is interior (supports conjugacy) or boundary (supports independence).
4. **Wiener filter attribution** (30 min): Compute M = Sigma_total^{-1} (Ledoit-Wolf regularized). Compare with M=I and M=Sigma_noise^{-1}. If Wiener filter is optimal, supports the MSE decomposition.

### Computational Cost

- Total: ~1.75 hours on 1x RTX 4090 (much of it reuses factorial experiment data)
- Additional storage: covariance matrices at multiple regularization strengths

### Success Probability: 35%

**Risk**: The FM1/FM2 interaction may genuinely be near-zero (truly independent), in which case the conjugacy framework is wrong but the original orthogonality claim is strengthened. This is a "win either way" situation for the paper.

### Failure Modes

- The bias-variance decomposition may not cleanly separate into FM1/FM2 components in practice
- The uncertainty bound may be too loose to be informative (C may be very small)
- Contrastive strength sweep may show monotonic improvement (alpha*=1 is optimal), refuting the interior optimum prediction
- Sigma_total estimation may be too noisy for Wiener filter to outperform ridge-regularized alternatives

---

## Synthesis: How These Angles Strengthen the Core CRA Paper

### Integration Strategy

The three angles are ordered by risk and novelty:
1. **LABA (Angle 1)**: Low risk, immediate practical payoff. Extends phi^T M psi to multi-layer, adds query-adaptive capability. Easy to integrate as "Section 5.1: Multi-Scale Extensions."
2. **CATCL (Angle 2)**: Medium risk, strong theoretical novelty. Provides causal grounding for FM2 correction. Integrates as "Section 4.3: Causal Interpretation of Contrastive Attribution."
3. **Conjugacy (Angle 3)**: High risk, potentially transformative. Replaces "orthogonal defects" with "conjugate tradeoff." If validated, upgrades the theoretical contribution from taxonomy to fundamental bound.

### Recommended Priority

**If the 2x2 factorial shows near-zero interaction (H3 holds)**: Lead with Angle 1 (LABA) as the practical extension, mention Angle 2 as theoretical grounding for FM2, treat conjugacy as "discussion" material.

**If the 2x2 factorial shows negative interaction (H3 partially fails)**: This is actually the more exciting outcome. Lead with Angle 3 (Conjugacy) as the theoretical framework, use Angle 2 (CATCL) to demonstrate the optimal tradeoff, and Angle 1 (LABA) as practical mitigation.

### Total Additional Experimental Budget

- Angle 1: 1.5 GPU-hours
- Angle 2: 1.5 GPU-hours
- Angle 3: 1.75 GPU-hours (partially overlapping with core experiments)
- **Total**: ~4 GPU-hours incremental beyond core experiments
- All experiments fit within the 1-hour-per-task constraint when parallelized across 4x RTX 4090

---

## Key References Integrated

- Vitel & Chhabra (2511.04715): Layer selection for influence estimation -- middle layers better than first/last, contradicts prior consensus. Directly motivates Angle 1 (LABA).
- AirRep / Sun et al. (2505.18513): Learned representations for attribution -- shows representation optimization is viable. LABA extends this with per-query adaptation without training.
- IF-GUIDE / Coalson et al. (2506.01790): Token-level attribution for toxicity -- shows fine-grained attribution beyond sample-level. Tangential but validates representation-space methods for safety applications.
- In-the-Wild / Xiao & Aranguri (2602.11079): Activation-difference vectors for DPO attribution -- partial precedent for temporal contrastive (Angle 2), but limited to DPO setting without DiD formalization.
- DATE-LM / Jiao et al. (2507.09424): Standard benchmark confirming "no single method dominates" -- supports the need for adaptive (Angle 1) and theoretically grounded (Angle 3) approaches.
- Scalable Forward-Only Attribution / Ma & Nyarko (2511.19803): Forward-only inference for attribution -- orthogonal efficiency improvement compatible with all three angles.
- Hyperparameter Sensitivity / Wang et al. (2505.24261): Shows attribution methods are sensitive to hyperparameters, especially regularization -- supports Angle 3's claim that optimal M (regularization) depends on FM1/FM2 balance.
- Data Kernel Perspective Space (2602.05106): Kernel-theoretic framework for training data -- provides complementary mathematical language for the phi^T M psi bilinear structure.

---

## Risk Assessment Summary

| Angle | Novelty | Risk | P(success) | Fallback Value |
|-------|---------|------|------------|---------------|
| 1. LABA | Medium-High | Low | 55% | Even uniform multi-layer averages likely improve by ~2pp |
| 2. CATCL | High | Medium | 45% | Causal framing of mean-subtraction is valuable even if DiD doesn't beat it |
| 3. Conjugacy | Very High | High | 35% | Negative result (true independence) strengthens original H3 claim |

**Expected value calculation**: At least one angle succeeds with P = 1 - (0.45)(0.55)(0.65) = 83.9%. The paper gains at least one novel extension with high probability.
