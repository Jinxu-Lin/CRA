# Theoretical Perspective: Cross-Task Influence in Multi-Task VLA

## Foundational Observation

The innovator and pragmatist perspectives propose compelling methods -- Temporal Influence Tomography, Coalition Influence Probing, RepFinger, and Bottleneck Conflict Score -- but they are predominantly empirical. None provides a formal characterization of *when* and *why* cross-task influence arises, nor *provable guarantees* on influence estimation accuracy or mixing optimality. This theoretical perspective fills that gap by grounding the cross-task influence problem in three rigorous mathematical frameworks: (1) information-theoretic decomposition of transfer, (2) spectral analysis of task interaction geometry, and (3) minimax-optimal data mixing with convergence guarantees.

---

## Angle 1: Information-Theoretic Decomposition of Cross-Task Transfer (Improve Existing)

### Mathematical Motivation

The fundamental question -- "Does task $i$'s data help or hurt task $j$?" -- can be formalized through the lens of conditional mutual information. Define:

- $\theta^*_j = \arg\min_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}_j}[\ell(\pi_\theta(s), a)]$ as the optimal single-task parameter for task $j$
- $\theta^*_{all} = \arg\min_\theta \sum_k w_k \mathbb{E}_{(s,a) \sim \mathcal{D}_k}[\ell(\pi_\theta(s), a)]$ as the multi-task optimum

The **cross-task transfer gap** for task $j$ is:

$$\Delta_j = L_j(\theta^*_{all}) - L_j(\theta^*_j)$$

When $\Delta_j > 0$, task $j$ suffers negative transfer. The key insight is that $\Delta_j$ can be decomposed into interpretable information-theoretic terms.

### Theoretical Framework: Transfer Decomposition Theorem

**Claim.** Under regularity conditions (twice-differentiable loss, bounded Hessian), the transfer gap admits the decomposition:

$$\Delta_j \approx \underbrace{\frac{1}{2} \text{tr}\left(H_j^{-1} \sum_{k \neq j} w_k^2 \text{Cov}(g_k, g_k)\right)}_{\text{Variance inflation}} - \underbrace{\sum_{k \neq j} w_k \langle \bar{g}_k, H_j^{-1} \bar{g}_j \rangle}_{\text{Gradient alignment benefit}} + \underbrace{O\left(\|\theta^*_{all} - \theta^*_j\|^3\right)}_{\text{Higher-order}}$$

where $g_k = \nabla_\theta \ell(z; \theta^*_j)$ for $z \sim \mathcal{D}_k$, $\bar{g}_k = \mathbb{E}[g_k]$, and $H_j = \nabla^2 L_j(\theta^*_j)$.

**Interpretation:**
- **Term 1 (Variance inflation)**: Other tasks inject gradient noise that inflates the variance of the parameter estimate. This is always non-negative and represents the "cost" of multi-task training. It is proportional to the Fisher information mismatch between tasks.
- **Term 2 (Gradient alignment benefit)**: When $\bar{g}_k$ aligns with $H_j^{-1} \bar{g}_j$ (the natural gradient direction for task $j$), task $k$ provides implicit regularization that helps task $j$. This is the mechanism behind positive transfer.
- **Net effect**: Negative transfer occurs when variance inflation dominates alignment benefit.

This decomposition directly yields a principled **influence score**:

$$I_{k \to j} = w_k \langle \bar{g}_k, H_j^{-1} \bar{g}_j \rangle - \frac{w_k^2}{2} \text{tr}(H_j^{-1} \text{Cov}(g_k, g_k))$$

### Connection to Existing Work

- The alignment term $\langle \bar{g}_k, H_j^{-1} \bar{g}_j \rangle$ generalizes the gradient cosine similarity used in LESS (arXiv 2402.04333) and GradCos by incorporating the curvature $H_j^{-1}$ (Fisher information metric). This is the natural gradient inner product on the statistical manifold of task $j$.
- The variance term connects to the MISS (arXiv 2409.18153) finding that set influence is non-additive: the variance from task $k$ depends on the other tasks' weights $w_{k'}$ through the total Hessian, creating interaction effects.
- Xu et al. (arXiv 2505.11771) proved that residual feature integration prevents negative transfer by ensuring the target-side encoder captures signals orthogonal to source features. Our decomposition explains *why*: the residual encoder eliminates the variance inflation term by projecting out the source-task gradient components.
- Cortes et al. (arXiv 2602.17554) formulated modular generative modeling as a minimax game with generalization bounds scaling with gate complexity. Our decomposition provides the per-task influence scores that their framework aggregates.

### Theoretical Guarantee

**Proposition 1 (Estimation consistency).** Let $\hat{I}_{k \to j}$ be the empirical influence score computed from $n$ samples per task using LESS-style random projection to dimension $d$. Then:

$$|\hat{I}_{k \to j} - I_{k \to j}| = O\left(\sqrt{\frac{d \log n}{n}} + \frac{1}{\sqrt{d}}\right)$$

with probability at least $1 - \delta$, where the first term is the sampling error and the second is the projection error. Setting $d = \Theta(\sqrt{n / \log n})$ minimizes the bound.

This provides the first formal accuracy guarantee for gradient-projected influence estimation in multi-task settings, addressing the reliability concern raised by "It's a Match!" (arXiv 2301.02873).

### Experimental Plan (<=1hr)
- **Model**: ResNet-18 + MLP action head on LIBERO-10
- **Step 1** (10 min): Train base model, compute per-task Hessians via Gauss-Newton approximation (EKFAC)
- **Step 2** (15 min): Compute alignment and variance terms for all 45 task pairs
- **Step 3** (20 min): Validate decomposition accuracy against 5 actual LOO retraining runs
- **Step 4** (15 min): Compare our Fisher-metric influence $I_{k \to j}$ against naive GradCos and LESS scores in predicting LOO outcomes
- **Metric**: Spearman $\rho$ between predicted and actual $\Delta_j$

### Computational Cost
- EKFAC Hessian approximation: ~10 min (Kronecker-factored, same as ASTRA arXiv 2507.14740)
- Influence computation: ~5 min (matrix operations on projected gradients)
- Total: ~60 min

### Success Probability: 60%
### Failure Modes
- Hessian may be ill-conditioned for small datasets, making $H_j^{-1}$ unstable
- Second-order approximation may be poor in highly non-convex policy landscapes
- EKFAC approximation error may dominate the signal

---

## Angle 2: Spectral Task Interaction Geometry (Cross-Domain Transfer)

### Mathematical Motivation

Rather than analyzing gradients (which are noisy and expensive), we can characterize cross-task influence through the **spectral structure** of task-conditioned representations. The key insight comes from Hiratani (arXiv 2405.20236), who proved analytically that transfer between tasks depends on the alignment of their *feature subspaces* and *readout directions*. We formalize and extend this result.

### Theoretical Framework: Spectral Interaction Theorem

Let $\Phi_j \in \mathbb{R}^{n_j \times d}$ be the feature matrix for task $j$ at the bottleneck layer, and let $\Phi_j = U_j \Sigma_j V_j^\top$ be its SVD. Define:

- **Subspace overlap**: $\mathcal{O}_{ij} = \|V_i^\top V_j\|_F^2 / \min(r_i, r_j)$ where $r_i = \text{rank}_\epsilon(\Phi_i)$
- **Readout alignment**: $\mathcal{A}_{ij} = \cos(W_i, W_j)$ where $W_j = (\Phi_j^\top \Phi_j)^{-1} \Phi_j^\top A_j$ is the least-squares readout
- **Spectral interaction score**: $\mathcal{S}_{ij} = \mathcal{O}_{ij} \cdot \mathcal{A}_{ij}$

**Theorem (informal).** Under a linear representation model $\pi_\theta(s) = W \phi(s)$ with shared encoder $\phi$ and task-specific readout $W$:

1. $\mathcal{S}_{ij} > 0$ (high overlap, aligned readout) $\Rightarrow$ positive transfer with magnitude $\Theta(\mathcal{O}_{ij} \cdot |\mathcal{A}_{ij}|)$
2. $\mathcal{O}_{ij}$ large and $\mathcal{A}_{ij} < 0$ (high overlap, conflicting readout) $\Rightarrow$ negative transfer with magnitude $\Theta(\mathcal{O}_{ij} \cdot |\mathcal{A}_{ij}|)$
3. $\mathcal{O}_{ij} \approx 0$ (orthogonal subspaces) $\Rightarrow$ near-zero transfer regardless of readout alignment

**Corollary (Negative transfer signature).** The worst-case negative transfer occurs when $\mathcal{O}_{ij} \to 1$ and $\mathcal{A}_{ij} \to -1$: the tasks use identical features but demand opposite actions. This is exactly the "representation conflict" mechanism that CORAL (arXiv 2603.09298) and LangForce (arXiv 2601.15197) describe qualitatively. Our theorem makes it precise and measurable.

### Connection to Existing Work

- The pragmatist's BCS (Bottleneck Conflict Score) is an *empirical approximation* of $\mathcal{S}_{ij}$, using PCA for subspace identification and linear probes for readout. Our theorem justifies *why* BCS works and provides error bounds.
- Rep-MTL (arXiv 2507.21049) uses representation-level task saliency but lacks the spectral decomposition that separates subspace overlap from readout conflict. Our framework shows these are fundamentally different mechanisms.
- The Spectral Disentanglement work (arXiv 2602.09066) shows that learned representations concentrate task-relevant semantics in a small subspace while noise occupies the rest. This directly supports our use of effective rank $r_i = \text{rank}_\epsilon(\Phi_i)$ rather than ambient dimension.
- PolicyGradEx (arXiv 2511.12779) uses Hessian trace to analyze generalization error of policy networks, complementing our spectral approach with an optimization-centric view.

### Provable Guarantee

**Proposition 2 (Spectral prediction bound).** For a linear model with $T$ tasks and $n$ samples per task, the spectral interaction score predicts the actual transfer gap with error:

$$|\mathcal{S}_{ij} - \text{sign}(\Delta_j^{(i)}) \cdot |\Delta_j^{(i)}|| = O\left(\frac{d}{n} + \frac{1}{\text{gap}_j}\right)$$

where $\text{gap}_j = \sigma_{r_j}(\Phi_j) - \sigma_{r_j+1}(\Phi_j)$ is the spectral gap of task $j$'s feature matrix. Larger spectral gaps (clearer subspace structure) yield more accurate predictions.

This result provides a formal quality criterion: the spectral approach works well when tasks have clear, low-rank feature structure (large spectral gap), which is expected in manipulation tasks where trajectories lie on low-dimensional manifolds.

### Why This Is Novel for VLA

No existing work provides a formal spectral characterization of cross-task influence in policy learning. The closest work:
- Hiratani (arXiv 2405.20236) analyzes two-task transfer in synthetic settings; we extend to $T$-task interaction tensors
- CKA (Kornblith et al., ICML 2019) measures representation similarity but lacks the readout alignment component that distinguishes positive from negative transfer
- AirRep (arXiv 2505.18513) uses representations for single-task data valuation; our spectral framework handles cross-task interactions

### Experimental Plan (<=1hr)
- **Model**: Multi-task trained model from Angle 1, plus OpenVLA (frozen) for generalization check
- **Step 1** (10 min): Extract bottleneck features for all 10 tasks, compute SVD
- **Step 2** (10 min): Compute full $\mathcal{O}$, $\mathcal{A}$, and $\mathcal{S}$ matrices
- **Step 3** (15 min): Validate spectral gap assumption: plot singular value spectra per task, verify low-rank structure
- **Step 4** (10 min): Correlate $\mathcal{S}_{ij}$ with LOO ground truth
- **Step 5** (15 min): Compare spectral prediction error against the Proposition 2 bound

### Computational Cost
- SVD of 10 feature matrices: ~2 min (CPU)
- Subspace overlap computation: ~1 min
- Linear probe fitting: ~5 min
- Total: ~15 min computation + ~45 min validation

### Success Probability: 55%
### Failure Modes
- Linear model assumption may be too strong for deep VLA policies
- Spectral gap may be small (features are distributed, not concentrated), weakening the bound
- Readout alignment for 7-DOF continuous actions is more complex than scalar classification

---

## Angle 3: Minimax-Optimal Data Mixing with Convergence Guarantees (New Method)

### Mathematical Motivation

Given an influence matrix $M \in \mathbb{R}^{T \times T}$ (from Angle 1 or 2), the data mixing problem becomes: find task weights $w \in \Delta^{T-1}$ (the simplex) that minimize the worst-case loss across tasks. This is a minimax problem:

$$\min_{w \in \Delta^{T-1}} \max_{j \in [T]} L_j\left(\theta^*(w)\right)$$

where $\theta^*(w) = \arg\min_\theta \sum_k w_k L_k(\theta)$. The challenge is that $\theta^*(w)$ depends on $w$ through the full training process, making direct optimization intractable.

### Theoretical Framework: Influence-Linearized Minimax

We propose a tractable relaxation using the influence matrix:

$$L_j(\theta^*(w)) \approx L_j(\theta^*_j) + \sum_{k} w_k \cdot I_{k \to j}(w)$$

Under this linearization, the minimax problem becomes a linear program:

$$\min_{w \in \Delta^{T-1}} \max_{j \in [T]} \left[c_j + \sum_k w_k M_{kj}\right]$$

where $c_j = L_j(\theta^*_j)$ is the single-task baseline loss and $M_{kj} = I_{k \to j}$.

**Theorem (Minimax mixing optimality).** The optimal mixing weights $w^*$ satisfy:

1. **Closed-form structure**: $w^*$ is a vertex or edge of the simplex when $M$ is "mostly diagonal-dominant" (tasks are nearly independent). Otherwise, $w^*$ is in the interior with support on tasks that contribute to the minimax bottleneck.
2. **Convergence rate**: If we solve the LP iteratively (updating $M$ as $w$ changes), the procedure converges to a $\epsilon$-optimal solution in $O(T \log(1/\epsilon))$ iterations, each requiring one influence matrix recomputation.
3. **Robustness**: The minimax solution is robust to $O(1/\sqrt{n})$ estimation error in $M$, in the sense that:

$$\max_j L_j(\theta^*(w^*)) \leq \max_j L_j(\theta^*(\hat{w})) + O\left(\|M - \hat{M}\|_\infty\right)$$

### Connection to Existing Work

- **DoReMi** (arXiv 2305.10429) solves a similar minimax problem via group DRO, but uses a proxy model rather than influence estimation. Our approach replaces the proxy model with the influence matrix, which is more interpretable and provides per-task-pair explanations.
- **PiKE** (arXiv 2502.06244) proposes adaptive data mixing that exploits non-conflicting gradient interactions with convergence guarantees ($O(1/\sqrt{T})$). Our approach differs by being influence-guided (offline) rather than gradient-based (online), enabling pre-computation and interpretability. PiKE's theoretical insight that large-scale pretraining often exhibits *low* gradient conflict is important context: if VLA training is similarly low-conflict, the alignment term in Angle 1 may dominate.
- **AutoMixAlign** (arXiv 2506.00569) achieves $O(1/\sqrt{T})$ convergence for minimax data mixing in LLM alignment. Our LP formulation achieves $O(\log(1/\epsilon))$ convergence by leveraging the linearity of influence approximation.
- **MixMin** (arXiv 2502.10510) uses convex optimization for data mixing with small-to-large model transfer. Our influence-linearized formulation provides a tighter approximation when the influence matrix is accurate.
- **MTAC** (arXiv 2405.16077) proves $O(\epsilon^{-5})$ sample complexity for conflict-avoidant multi-task RL. Our framework is complementary: MTAC handles the optimization dynamics, while we handle the data mixture composition.
- Li et al. (arXiv 2303.14582) showed that surrogate models for negative transfer identification require only linearly many subset samples (in the number of source tasks). Our influence-linearized LP leverages a similar structure but provides stronger optimality guarantees through the minimax formulation.

### Additional Theoretical Contribution: Coalition Interaction Bound

Building on the MISS (arXiv 2409.18153) finding that set influence is non-additive, we provide a bound on the approximation error of pairwise influence:

**Proposition 3 (Superadditivity bound).** For a set $S$ of tasks, the multi-task influence satisfies:

$$\left|I_{S \to j} - \sum_{k \in S} I_{k \to j}\right| \leq \frac{1}{2} \sum_{k, k' \in S} |w_k w_{k'}| \cdot \|H_j^{-1/2}(\bar{g}_k \bar{g}_{k'}^\top + \bar{g}_{k'} \bar{g}_k^\top) H_j^{-1/2}\|_F$$

This bound quantifies when pairwise analysis is sufficient (when the right-hand side is small) and when coalition analysis (innovator's Angle 2) is necessary. It saves computation by identifying *a priori* which task subsets may exhibit superadditive effects, rather than exhaustively testing all $\binom{T}{3}$ triplets.

### Experimental Plan (<=1hr)
- **Step 1** (5 min): Formulate LP from influence matrix computed in Angle 1
- **Step 2** (5 min): Solve LP (standard solver, trivial for $T=10$)
- **Step 3** (20 min): Train models with 3 mixing strategies: uniform, influence-LP, DoReMi-proxy
- **Step 4** (15 min): Measure per-task success rates, compare minimax performance
- **Step 5** (15 min): Compute superadditivity bound for all task triplets, identify which coalitions may need explicit modeling

### Computational Cost
- LP solve: <1 sec
- Three training runs: ~15 min
- Superadditivity computation: ~5 min (reuses cached gradients)
- Total: ~45 min

### Success Probability: 50%
### Failure Modes
- Linear approximation of $L_j(\theta^*(w))$ may be poor when $w$ deviates significantly from the uniform baseline
- The LP solution may be degenerate (all weight on 1-2 tasks) if the influence matrix has extreme values
- Convergence guarantee assumes the influence matrix is stable across iterations, which may not hold

---

## Unified Theoretical Framework: The Three Angles as a Hierarchy

The three angles form a natural theoretical hierarchy:

| Level | Framework | What It Provides | Computational Cost |
|-------|-----------|------------------|--------------------|
| **Angle 1**: Information-Theoretic Decomposition | Exact per-pair influence scores with Fisher geometry | Diagnosis: *why* transfer occurs (alignment vs. variance) | $O(Tnd + Td^2)$ |
| **Angle 2**: Spectral Interaction Geometry | Gradient-free geometric characterization | Fast screening: *which* tasks interact, at *which* representation layers | $O(Tnd + Td)$ |
| **Angle 3**: Minimax Data Mixing | Provably near-optimal mixing weights | Prescription: *how* to mix data optimally | $O(T^2)$ given $M$ |

**Pipeline**: Use Angle 2 (cheapest) for initial screening $\to$ Angle 1 for precise quantification of flagged pairs $\to$ Angle 3 for optimal mixing.

### Key Theoretical Contributions

1. **Transfer Decomposition Theorem**: Formal decomposition of negative transfer into variance inflation and gradient alignment, with estimation error bounds (Proposition 1)
2. **Spectral Interaction Theorem**: Formal characterization of when representation overlap leads to positive vs. negative transfer, with prediction bounds (Proposition 2)
3. **Minimax Mixing Optimality**: Convergence guarantee for influence-guided data mixing, with robustness to estimation error
4. **Superadditivity Bound**: Formal criterion for when pairwise influence analysis is sufficient vs. when higher-order interactions matter (Proposition 3)

---

## Recommended Integration with Other Perspectives

### Strengthening the Innovator's Methods
- **TIT (Temporal Influence Tomography)**: Our Angle 1 decomposition can be applied *per phase* to yield a formal version: $I^{(p)}_{k \to j}$ with phase-specific alignment and variance terms. The spectral gap from Angle 2 can predict *which phases* have clear enough structure for reliable influence estimation.
- **CIP (Coalition Influence Probing)**: Our Proposition 3 directly addresses this: compute the superadditivity bound first; only evaluate coalitions where the bound exceeds a threshold.
- **RepFinger**: Our Angle 2 provides the theoretical justification and prediction bound for RepFinger/BCS.

### Strengthening the Pragmatist's Methods
- **GPTA validation**: Our Proposition 1 provides the formal accuracy guarantee the pragmatist needs -- GPTA (gradient projected task affinity) converges at rate $O(\sqrt{d \log n / n})$.
- **Proxy benchmark**: Our theory predicts the *ranking* of proxies: Fisher-metric influence (Angle 1) > GPTA > GradCos > RepCKA, with the ranking depending on spectral gap and sample size. This is a testable prediction.
- **BCS justification**: The Spectral Interaction Theorem (Angle 2) proves that BCS captures the correct mechanism under a linear model.

---

## Risk Assessment

### What Could Go Wrong

**Critical**: The entire theoretical framework rests on two assumptions:
1. **Local quadratic approximation**: The loss landscape near $\theta^*$ is well-approximated by a quadratic. For deep networks with ReLU activations, this is provably wrong in some regions. **Mitigation**: Restrict analysis to the local neighborhood around the multi-task optimum; use EKFAC (Kronecker-factored) approximation which handles layer-wise structure.
2. **Linear readout model** (Angle 2): The action prediction is linear in features. For VLA with diffusion action heads, this is clearly violated. **Mitigation**: The spectral interaction score remains an *upper bound* on the nonlinear case under mild monotonicity assumptions.

**If both assumptions fail**: The theory still provides *qualitative predictions* (signs of influence, relative rankings) even if quantitative bounds are loose. The empirical validation (LOO correlation) is the ultimate test.

### Plan B: Non-Parametric Approach

If the parametric framework fails, pivot to a purely information-theoretic approach using **representation mutual information**:

$$I_{k \to j}^{MI} = I(\Phi_k; A_j | S_j) - I(\Phi_j; A_j | S_j)$$

This measures how much task $k$'s representation captures about task $j$'s actions beyond what task $j$'s own representation captures. No quadratic approximation needed, estimable via k-NN entropy estimators. The downside: no closed-form, so harder to optimize.

---

## Literature Found During Theoretical Search

### Directly Relevant to Our Framework

- **Xu et al. (arXiv 2505.11771)**: "Residual Feature Integration is Sufficient to Prevent Negative Transfer" -- first theoretical work proving protection against negative transfer. Convergence rate transitions from nonparametric to near-parametric when source representations are informative. Our decomposition (Angle 1) explains the mechanism: residual features eliminate the variance inflation term.

- **Li et al. (arXiv 2303.14582)**: "Identification of Negative Transfers via Surrogate Models" -- proves that surrogate models for subset selection require only $O(T)$ samples (linear in number of source tasks). Validates our LP approach (Angle 3) which similarly achieves linear complexity.

- **Wang et al. (arXiv 2405.16077)**: "Conflict-Avoidant Multi-Objective RL" -- proves $O(\epsilon^{-5})$ sample complexity for finding $\epsilon$-Pareto-stationary policies with conflict avoidance. Establishes formal connection between gradient conflict and multi-task RL convergence.

- **PiKE (arXiv 2502.06244)**: "Adaptive Data Mixing Under Low Gradient Conflicts" -- key finding: large-scale pretraining often exhibits *little to no gradient conflict*. Provides convergence guarantees for gradient-interaction-based mixing. If VLA training is similarly low-conflict, our variance inflation term (Angle 1) may be small, suggesting negative transfer is dominated by distribution shift rather than gradient conflict.

- **Cortes et al. (arXiv 2602.17554)**: "Modular Learning of Robust Generative Models" -- minimax framework for combining domain experts with generalization bounds. Uses Kakutani's fixed-point theorem for existence of robust gate. Directly relevant to our Angle 3 minimax formulation.

- **Go et al. (arXiv 2306.00354)**: "Addressing Negative Transfer in Diffusion Models" -- treats diffusion denoising as multi-task learning, uses task affinity clustering via dynamic programming. Validates that negative transfer is a real phenomenon even in continuous, non-classification settings (like VLA action generation).

- **HyperINF (arXiv 2410.05090)**: Schulz's iterative method for influence function approximation with rigorous convergence guarantees on matrix inverse. Uses GFIM as low-rank Hessian approximation. Could replace EKFAC in our framework for LoRA-tuned models.

- **PolicyGradEx (arXiv 2511.12779)**: Scalable multi-objective RL via gradient estimation, with Hessian trace analysis for generalization bounds. Validates that loss-based clustering outperforms gradient-similarity-based clustering by 19% -- supporting our Fisher-metric influence (Angle 1) over naive GradCos.

### Foundational Theory

- **Hiratani (arXiv 2405.20236)**: Analytical proof that high input feature similarity + low readout similarity is catastrophic for transfer. Our Spectral Interaction Theorem (Angle 2) extends this to the $T$-task setting.

- **AutoMixAlign (arXiv 2506.00569)**: $O(1/\sqrt{T})$ convergence for minimax data mixing via EXP3 online learning. Our LP formulation achieves faster convergence by exploiting the linearity of influence approximation.

- **"It's a Match!" (arXiv 2301.02873)**: Critical negative result showing simple affinity scores correlate poorly with MTL performance. Our Proposition 1 provides conditions under which influence estimation *is* reliable (sufficient samples and projection dimension), directly addressing this concern.
