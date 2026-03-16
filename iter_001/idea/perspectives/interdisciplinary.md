# Interdisciplinary Perspective: CRA Research Proposal

**Agent**: Interdisciplinary (cross-domain structural analogies from neuroscience, physics, immunology, econometrics)
**Date**: 2026-03-16

---

## Executive Summary

The CRA project diagnoses two signal processing defects in parameter-space TDA (FM1: signal dilution, FM2: common influence contamination), unifies 5 representation-space methods under a phi^T * psi bilinear framework, and validates on DATE-LM. The existing perspectives (Theoretical, Innovator, Pragmatist, Contrarian) have thoroughly explored the core ML-internal angles. This interdisciplinary perspective goes further by drawing deep structural correspondences from four fields outside ML -- neuroscience (credit assignment), statistical physics (renormalization group), immunology (self/non-self discrimination), and econometrics (difference-in-differences) -- to generate novel mechanistic insights, testable predictions, and potentially transformative methodological imports.

The key insight unifying all four analogies: **the CRA problem is fundamentally about separating task-relevant signal from shared background in high-dimensional biological/physical systems -- a problem that nature and the social sciences have solved repeatedly using structurally similar mechanisms.** Each field offers a different lens on the same challenge, and the structural isomorphisms are deep enough to yield concrete algorithmic improvements, not just metaphors.

---

## Angle 1: Neuroscience -- Credit Assignment as Training Data Attribution

### The Structural Correspondence

The brain faces a problem strikingly parallel to TDA: given a behavioral outcome (prediction), which synaptic experiences (training data) were causally responsible? This is the **credit assignment problem** -- arguably the central open question in computational neuroscience. The structural mapping is precise:

| Neuroscience | Training Data Attribution |
|---|---|
| Behavioral outcome (reward/error) | Model prediction on test input |
| Synaptic experience (stimulus-response pair) | Training sample z_i |
| Synapse weight change due to experience | Influence score I(z_test, z_i) |
| Eligibility trace (decaying memory at synapse) | Gradient at training point g(z_i) |
| Neuromodulatory signal (dopamine/reward signal) | Test-time gradient g(z_test) |
| Credit assignment score | phi(z_test)^T * psi(z_i) |

The brain's solution to credit assignment has evolved over hundreds of millions of years into a **three-factor learning rule**: synaptic plasticity = (presynaptic activity) * (postsynaptic activity) * (neuromodulatory signal). This trilinear form is structurally richer than the bilinear phi^T * psi -- the third factor (neuromodulation) acts as a task-dependent gating signal that selectively amplifies relevant attributions.

### Why This Analogy Is Not Superficial

Recent neuroscience has formalized credit assignment through **eligibility traces** -- biochemical tags at synapses that mark "which experiences happened" and decay over time, waiting for a neuromodulatory signal to consolidate them into lasting weight changes (Mazurek et al. 2025, arXiv:2504.05341; Meulemans et al. 2022, arXiv:2204.07249). The key insight from cascading eligibility traces (arXiv:2506.14598) is that standard exponentially-decaying traces suffer from a **temporal blurring problem**: they mix together any events occurring during the delay window.

This is structurally identical to FM2 (common influence contamination): standard influence functions "blur" together all training samples that contributed to a parameter region, because the gradient g(z_i) encodes ALL learned knowledge at the synapse/parameter, not just the task-specific contribution of z_i.

**The neuroscience solution**: Cascading eligibility traces use a state-space model inspired by biochemical reaction cascades to maintain temporally precise memory. The cascade structure provides **event-specific** tagging rather than aggregate tagging.

**The TDA import**: This suggests that FM2 can be addressed not just by mean subtraction (the current approach) but by maintaining **sample-specific attribution markers** during training -- akin to eligibility traces that tag each training sample's contribution separately. Concretely:

1. **Eligibility-Trace Attribution (ETA)**: During fine-tuning, record per-sample "eligibility snapshots" -- the representation h_t(z_i) at the moment z_i is processed at training step t. The attribution score becomes:
   ```
   A_ETA(z_test, z_i) = phi(z_test)^T * [h_T(z_i) - h_{t_i}(z_i)]
   ```
   where t_i is the step when z_i was last trained on and T is the final step. This "trace" captures the sample-specific representation change, analogous to the synapse-specific eligibility tag.

2. **Neuromodulated Attribution**: Introduce a third factor m(z_test, task) that gates the bilinear score:
   ```
   A_neuro(z_test, z_i) = m(z_test, task) * phi(z_test)^T * psi(z_i)
   ```
   where m is a learned or task-derived modulation signal. This corresponds to the neuromodulatory gating in three-factor rules and could be implemented as a simple task-specific scalar weight on different representation dimensions.

### Testable Predictions

**Prediction N1**: ETA (tracking per-sample representation changes during training) should outperform static contrastive scoring (mean subtraction at the end of training) on DATE-LM factual attribution, because factual knowledge is acquired at specific training steps, not uniformly.

**Prediction N2**: The per-sample representation change ||h_T(z_i) - h_{t_i}(z_i)|| should correlate strongly (r > 0.6) with the true attribution score (LOO), while the static representation ||h_T(z_i)|| should correlate weakly (r < 0.3) -- paralleling the finding that eligibility traces are more informative than steady-state activity for credit assignment.

**Prediction N3**: The neuroscience-inspired three-factor score should improve attribution on tasks where different test inputs require different attribution "channels" (e.g., toxicity vs. factual recall), because the modulation factor m selectively amplifies task-relevant representation dimensions.

### Failure Mode

The eligibility trace analogy requires access to intermediate training checkpoints or per-sample training dynamics, which may not be available in the standard DATE-LM setup. If only final-model representations are available, ETA degenerates to In-the-Wild's checkpoint differencing (h_post - h_pre), losing the per-sample temporal precision.

**Computational Cost**: ~1.5 GPU-hours (need to save representations at training steps; adds ~30% overhead to standard fine-tuning)
**Success Probability**: 50% -- the per-sample tracking is theoretically grounded but adds significant implementation complexity; the marginal gain over simpler differencing is the key question
**Impact if Succeeds**: High -- introduces a genuinely novel attribution mechanism inspired by neuroscience with no precedent in the TDA literature

### Key References

- Mazurek et al. 2025 (arXiv:2504.05341) -- Three-factor learning rules in spiking neural networks; comprehensive overview of eligibility trace mechanisms
- Meulemans et al. 2022 (arXiv:2204.07249) -- Learning as control minimization; shows strong feedback + local learning rules match backpropagation
- Greedy et al. 2022 (arXiv:2206.11769) -- BurstCCN: single-phase credit assignment via burst multiplexing in cortical networks
- Ororbia 2023 (arXiv:2312.09257) -- Survey of neurobiologically-plausible credit assignment; taxonomy of six learning rule families
- Cascading Eligibility Traces (arXiv:2506.14598) -- State-space model for temporally precise eligibility traces across behavioral timescales

---

## Angle 2: Statistical Physics -- Renormalization Group as Principled Coarse-Graining

### The Structural Correspondence

The renormalization group (RG) in statistical physics systematically eliminates irrelevant degrees of freedom to reveal the essential physics at macroscopic scales. The correspondence with the parameter-to-representation transition in TDA is striking:

| Statistical Physics (RG) | Training Data Attribution |
|---|---|
| Microscopic degrees of freedom (individual spins) | Parameter-space gradients g in R^B |
| Macroscopic order parameters | Representation-space features h in R^d |
| RG transformation (block spin → coarse-grained) | Neural network forward pass (R^B → R^d) |
| Irrelevant operators (high-energy modes) | FM1 noise: gradient dimensions orthogonal to task signal |
| Relevant operators (low-energy, universal) | Task-relevant attribution signal in h |
| Universality class | Method family in phi^T * psi framework |
| Phase transition / critical point | RepT's "phase transition layer" |
| Information Bottleneck = optimal RG | Optimal layer selection for attribution |

### Why This Is More Than Metaphor

Gordon et al. (2020, arXiv:2012.01447) proved a fundamental theorem: **the information-theoretic notion of "relevance" in the Information Bottleneck formalism is equivalent to the field-theoretic notion of "relevance" in the Renormalization Group.** Specifically, for systems described by a field theory, the degrees of freedom identified as "relevant" by IB compression correspond exactly to the operators with the lowest scaling dimensions in the RG sense.

This result directly connects to the CRA thesis:

1. **FM1 as "irrelevant operators"**: In RG language, parameter-space gradients in R^B contain a huge number of "irrelevant" high-energy modes that do not affect macroscopic (attribution-relevant) behavior. The representation h at an intermediate layer is the RG-transformed "effective theory" that retains only the relevant operators. FM1 (signal dilution) is precisely the statement that working in the full microscopic space (parameter space) includes enormous irrelevant noise that overwhelms the relevant signal.

2. **Optimal layer selection as critical RG flow**: De Mello Koch & Ghosh (2025, arXiv:2504.12700) showed that deep learning exhibits two phases: a rapid fitting phase followed by a slower **compression/coarse-graining** phase. They argue this second phase is analogous to RG flow, where the network progressively discards irrelevant information. RepT's "phase transition layer" -- the optimal layer for attribution -- corresponds to the point in the RG flow where the effective theory transitions from retaining microscopic details to capturing only macroscopic (task-relevant) structure.

3. **Universality and the phi^T * psi framework**: In statistical physics, systems in the same universality class have identical macroscopic behavior despite different microscopic details. Similarly, the phi^T * psi framework shows that all 5 representation-space methods produce equivalent macroscopic (attribution) behavior despite different microscopic implementations -- they are in the same "universality class" of bilinear attribution methods.

### The Novel Prediction: RG-Inspired Optimal Attribution

The RG framework makes a concrete prediction that goes beyond existing perspectives:

**Theorem sketch (RG-optimal attribution)**: Define the "attribution effective dimension" d_eff(L) at layer L as the number of IB-relevant dimensions (those carrying mutual information I(h_L; Y) above a threshold). The optimal attribution layer L* satisfies:
```
L* = argmin_L { d_eff(L) such that I(h_L; A*) > (1 - epsilon) * max_L' I(h_L'; A*) }
```
where A* is the ground-truth attribution oracle. In words: the optimal layer is the one that achieves maximum compression (minimum d_eff) while retaining nearly all attribution-relevant information.

**RG scaling prediction**: d_eff(L) should follow a power-law decay with layer depth (analogous to RG flow towards the fixed point), not an exponential or linear decay:
```
d_eff(L) ~ d_0 * (L/L_total)^{-alpha}
```
with alpha determined by the "critical exponents" of the attribution task. Different DATE-LM tasks (data selection, toxicity, factual) should have different alpha values, reflecting different universality classes.

**Practical RG-Attribution algorithm**: Rather than using fixed-layer representations, apply a learned RG transformation (block averaging + nonlinearity) to the representations:
```
h_RG(z) = sigma(W_coarse * BlockAvg(h_L(z)))
```
where BlockAvg groups representation dimensions into blocks (guided by the covariance structure) and W_coarse is a learned projection. This "RG-filtered" representation should achieve better attribution with fewer dimensions than raw layer representations.

### Connection to Existing Work

- **Kline & Palmer (2021, arXiv:2107.13700)**: Proved formal equivalence between Gaussian Information Bottleneck and non-perturbative RG. Their semigroup structure (successive IB transformations remain IB-optimal) suggests that the phi^T * psi bilinear form should be invariant under the representation RG flow -- i.e., the attribution score should be approximately constant across layers near the critical point L*.

- **Lee et al. (2024, arXiv:2410.00396)**: Introduced "dynamic neurons" to reveal translational symmetry in deep networks, enabling RG analysis. Their approach could be used to identify which representation dimensions are "relevant" vs "irrelevant" for attribution, providing a physics-based dimension reduction that complements the empirical eigenspectrum analysis proposed by the Innovator.

### Testable Predictions

**Prediction P1**: d_eff(L) measured via IB proxy (MINE or binning-based MI) should follow a power-law decay with layer depth. The exponent alpha should differ across DATE-LM tasks: factual attribution (more compressed, higher alpha) > toxicity (medium alpha) > data selection (lower alpha, more distributed signal).

**Prediction P2**: Attribution performance (LDS on DATE-LM) as a function of layer L should exhibit a "critical plateau" -- a range of layers where performance is nearly constant, analogous to the RG fixed-point regime. RepT's phase transition layer should fall at the onset of this plateau.

**Prediction P3**: RG-filtered representations (block-averaged + projected) should achieve comparable attribution quality to full representations with ~4x fewer dimensions, analogous to how RG transformations preserve essential physics with far fewer degrees of freedom.

### Failure Mode

The RG analogy assumes that neural network layers perform a hierarchical coarse-graining similar to block-spin RG. If the actual information flow is non-hierarchical (e.g., skip connections, attention patterns that mix all layers), the RG picture breaks down. Furthermore, the IB-RG equivalence is only proven for Gaussian statistics; extension to the highly non-Gaussian representations in LLMs is an open theoretical question.

**Computational Cost**: ~2 GPU-hours (layer-wise MI estimation + block averaging experiments)
**Success Probability**: 45% -- the conceptual connection is strong but quantitative predictions (power-law exponent, critical plateau) may not manifest cleanly
**Impact if Succeeds**: Very High -- would establish the first rigorous physics-based theory for why intermediate layers are optimal for attribution, with predictions that no existing TDA paper has made

### Key References

- Gordon et al. 2020 (arXiv:2012.01447) -- Equivalence between IB relevance and RG relevance (foundational result)
- De Mello Koch & Ghosh 2025 (arXiv:2504.12700) -- Two-phase learning as RG flow; compression phase = coarse-graining
- Kline & Palmer 2021 (arXiv:2107.13700) -- Gaussian IB = non-perturbative RG; semigroup structure of optimal compression
- Lee et al. 2024 (arXiv:2410.00396) -- Dynamic neuron approach revealing translational symmetry for RG analysis of DNNs
- Hou & You 2023 (arXiv:2306.11054) -- Machine Learning Renormalization Group: automatic discovery of optimal RG transformations

---

## Angle 3: Immunology -- Self/Non-Self Discrimination as Attribution Debiasing

### The Structural Correspondence

The adaptive immune system solves a problem with a remarkable structural parallel to FM2 (common influence contamination): it must distinguish between "self" antigens (ubiquitous proteins produced by the host's own cells) and "non-self" antigens (foreign molecules from pathogens). This self/non-self discrimination is the immune system's version of separating common influence (pre-training knowledge = "self") from task-specific attribution (fine-tuning-specific = "non-self").

| Immunology | Training Data Attribution |
|---|---|
| Self antigens (host proteins) | Pre-training knowledge (common influence) |
| Non-self antigens (pathogen-derived) | Fine-tuning-specific attribution signal |
| T-cell receptor (TCR) binding affinity | Attribution score A(z_test, z_i) |
| Negative selection in thymus | Mean subtraction (contrastive scoring) |
| Clonal deletion (remove self-reactive T-cells) | Removing common-influence component from scores |
| Mature T-cell repertoire | Debiased attribution scores |
| Autoimmune disease | FM2 contamination (scoring "self" as "non-self") |
| Immune tolerance | Correct attribution that ignores pre-training influence |

### Why This Goes Beyond Metaphor

The immune system's solution to self/non-self discrimination is implemented through a two-stage process that offers a more nuanced approach than simple mean subtraction:

1. **Positive selection** (thymic cortex): T-cells must demonstrate the ability to bind MHC molecules (basic functional competence). This selects for T-cells that can "see" the relevant molecular space at all.

   **TDA analog**: Representation-space methods already implement positive selection by operating in a feature space where samples are "visible" (non-orthogonal representations in R^d, unlike near-orthogonal gradients in R^B). This is the FM1 fix.

2. **Negative selection** (thymic medulla): T-cells that bind too strongly to self antigens are eliminated (clonal deletion). This removes the "common influence" that would cause autoimmune responses.

   **TDA analog**: This is the FM2 fix -- but the immune system's negative selection is more sophisticated than mean subtraction. It eliminates specific high-affinity self-reactive clones rather than subtracting a global mean. This suggests a **threshold-based debiasing** approach:

```
A_immune(z_test, z_i) = phi(z_test)^T * psi(z_i) * I[phi(z_test)^T * psi(z_i) < tau_self]
```

where tau_self is a "self-reactivity threshold" above which scores are considered contaminated by common influence. Samples with very high similarity to the test point (above tau_self) are flagged as potentially spurious attributions driven by pre-training knowledge.

3. **Peripheral tolerance** (regulatory T-cells): Even after thymic selection, the immune system maintains peripheral tolerance through regulatory T-cells (Tregs) that actively suppress self-reactive responses that escaped negative selection.

   **TDA analog**: This suggests a **two-stage debiasing** approach:
   - Stage 1 (negative selection): Mean subtraction to remove the dominant common influence
   - Stage 2 (peripheral tolerance): A learned suppression mechanism that identifies and downweights residual spurious attributions that survived mean subtraction

### The Novel Algorithmic Proposal: Immune-Inspired Two-Stage Debiasing

```python
# Stage 1: Negative Selection (thymic -- remove common influence)
phi_debiased = phi(z_test) - E_z'[phi(z')]  # standard mean subtraction
psi_debiased = psi(z_i) - E_z'[psi(z')]

# Stage 2: Peripheral Tolerance (regulatory -- suppress residual self-reactivity)
A_raw = phi_debiased^T * psi_debiased
similarity_to_self = phi(z_test)^T * E_z'[psi(z')]  # how "self-like" is z_test?
suppression_factor = sigmoid(-beta * (similarity_to_self - tau))
A_immune = A_raw * suppression_factor
```

The suppression factor is high (close to 1) when the test point is dissimilar from the population mean ("non-self"), and low (close to 0) when the test point is very similar to the mean ("self-like"). This addresses a blind spot identified by the Contrarian: for test inputs that are highly typical of the pre-training distribution, mean subtraction alone may not be sufficient because the residual common influence is still large.

### Connection to Artificial Immune Systems Literature

The negative selection algorithm (NSA), originally proposed by Forrest et al. (1994) and recently surveyed comprehensively (e.g., Nature Scientific Reports 2025), generates detectors that recognize "non-self" patterns by eliminating those that match "self." Recent work (NSA-AE, ScienceDirect 2025) enhances NSA with autoencoders to handle high-dimensional spaces -- directly relevant to our problem of debiasing in high-dimensional representation space.

RAILS (Wang et al. 2020, arXiv:2012.10485) demonstrated that immune-inspired defense mechanisms (B-cell flocking, clonal expansion, affinity maturation) can harden deep k-NN architectures, achieving 5-13% robustness improvement. The deep k-NN structure is relevant because RepSim-based attribution is essentially a k-NN-like operation in representation space.

### Testable Predictions

**Prediction I1**: Two-stage immune debiasing should outperform single-stage mean subtraction by > 3pp on DATE-LM factual attribution, where pre-training knowledge is the dominant confounder (test inputs asking about facts that the model "knows" from pre-training are highly "self-like").

**Prediction I2**: The benefit of the second stage (peripheral tolerance) should be largest for test inputs with high "self-similarity" (similarity_to_self > median), and negligible for test inputs with low self-similarity. This predicts an interaction between debiasing strategy and test-input typicality.

**Prediction I3**: The optimal threshold tau_self should correlate with the model's pre-training perplexity on the test input: low perplexity (model "knows" this well from pre-training) should require stronger suppression.

### Failure Mode

The immune analogy assumes a clean separation between "self" (pre-training knowledge) and "non-self" (fine-tuning attribution). In practice, some training samples contribute both common knowledge and task-specific information simultaneously -- the boundary is not sharp. The threshold-based suppression may discard genuinely relevant attributions for samples that happen to be highly typical.

**Computational Cost**: ~1 GPU-hour (representation extraction is cached; the two-stage scoring adds minimal overhead)
**Success Probability**: 40% -- the conceptual mapping is appealing but the quantitative improvement over simple mean subtraction may be marginal
**Impact if Succeeds**: Medium -- introduces a novel debiasing paradigm with biological grounding; the two-stage framework could become a standard component in TDA pipelines

### Key References

- Forrest et al. 1994 -- Original negative selection algorithm for self/non-self discrimination
- Wang et al. 2020 (arXiv:2012.10485) -- RAILS: immune-inspired defense for deep k-NN; B-cell flocking and affinity maturation
- NSA-AE (ScienceDirect 2025) -- Autoencoder-augmented negative selection for high-dimensional anomaly detection
- Nature Scientific Reports 2025 -- Generating detectors from anomaly samples via negative selection for network intrusion detection

---

## Angle 4: Econometrics -- Matched Filtering as Optimal Linear Detection

### The Structural Correspondence

The CRA project's signal processing analogy (matched filtering for FM1, differential detection for FM2) is correct but underexploited. The matched filter is not just a metaphor -- it is the **provably optimal linear detector** for a known signal in additive noise (Neyman-Pearson lemma). The key insight from signal processing theory that the existing perspectives miss is the **orthogonality principle**: the matched filter's output is uncorrelated with the noise, which is precisely the condition for optimal signal extraction.

Recent work establishes this connection rigorously:

| Signal Processing | Training Data Attribution |
|---|---|
| Known signal template s(t) | Test-point representation phi(z_test) |
| Received signal r(t) = s(t) + n(t) | Training-point feature psi(z_i) = signal + noise |
| Matched filter output s^T * r | Attribution score phi^T * psi |
| Noise covariance Sigma_n | Common influence covariance Sigma_shared |
| Whitened matched filter s^T * Sigma_n^{-1} * r | Mahalanobis-distance attribution |
| SNR of matched filter = s^T * Sigma_n^{-1} * s | Attribution signal quality |

### The Deep Connection: CNNs ARE Matched Filters

Stankovic & Mandic (2021, arXiv:2108.10751) proved that convolution layers in CNNs **exactly implement matched filtering** -- the convolution of input data with learned filters is mathematically equivalent to searching for the presence of template patterns. This was extended to complex-valued CNNs via widely-linear matched filters (Wang et al. 2024, arXiv:2401.16729), showing that the matched filter interpretation holds generally with enhanced SNR guarantees.

Yan et al. (2021, arXiv:2104.03961) went further for gravitational wave detection, proving that **matched filtering is formally equivalent to a particular neural network** that can be constructed analytically and then further trained for improved performance. They introduced "MNet" architectures that initialize as matched filters and then adapt.

The import for CRA: **representation-space attribution (phi^T * psi) is not merely analogous to matched filtering -- it IS matched filtering, with phi as the template and psi as the received signal.** This yields immediate algorithmic consequences:

1. **Whitened attribution (optimal matched filter)**: The standard matched filter is optimal only under white noise. When the noise (common influence) is colored (has non-trivial covariance structure), the optimal detector is the **whitened matched filter**:
   ```
   A_opt(z_test, z_i) = phi(z_test)^T * Sigma_noise^{-1} * psi(z_i)
   ```
   where Sigma_noise is the covariance of the common-influence component. This is exactly the phi^T * M * psi form from the Theoretical perspective's Theorem 7, with M = Sigma_noise^{-1}. The signal processing theory tells us that M = I (identity, i.e., RepSim) is suboptimal when common influence has non-trivial structure -- which it always does in LLMs.

2. **Detection theory predicts attribution quality**: The output SNR of the matched filter is:
   ```
   SNR_out = phi^T * Sigma_noise^{-1} * phi
   ```
   This is computable from the data and predicts, for each test input, how well the attribution will work. High SNR_out (test input has features well-separated from common influence) predicts reliable attribution; low SNR_out predicts unreliable attribution. No existing TDA paper provides per-query reliability estimates.

3. **Adaptive matched filter (CFAR detection)**: In radar/sonar, the Constant False Alarm Rate (CFAR) detector adapts the detection threshold based on local noise estimates. The TDA analog: for each test input, estimate the local "noise floor" (expected attribution score under the null hypothesis of no true attribution) and normalize the attribution score by this floor:
   ```
   A_CFAR(z_test, z_i) = phi(z_test)^T * psi(z_i) / sqrt(Var_z'[phi(z_test)^T * psi(z')])
   ```
   This produces a z-score-like normalized attribution that is comparable across different test inputs -- addressing the known issue that raw attribution scores are not calibrated across queries.

### Experimental Plan

**Pilot (15 min)**: On Pythia-1B + DATE-LM data selection, compute:
(a) Standard RepSim (phi^T * psi with M=I)
(b) Whitened RepSim (phi^T * Sigma_noise^{-1} * psi, where Sigma_noise is estimated from population covariance)
(c) CFAR-normalized RepSim
Compare all three on LDS metric.

**Core experiment (45 min)**:
- Compute Sigma_noise for each DATE-LM task (covariance of representations over a random negative sample set)
- Evaluate whitened and CFAR-normalized versions of RepSim, RepT on all 3 tasks
- Compute per-query SNR_out and correlate with per-query attribution accuracy

**Validation (15 min)**: Test the SNR_out prediction: bin test queries by SNR_out quintiles and verify that high-SNR queries have better attribution than low-SNR queries. If the correlation exceeds r=0.5, the matched filter theory has predictive power.

### Testable Predictions

**Prediction S1**: Whitened RepSim (M = Sigma_noise^{-1}) should outperform standard RepSim (M = I) by 3-8pp on DATE-LM tasks where common influence is structured (factual attribution > toxicity > data selection).

**Prediction S2**: Per-query SNR_out should predict per-query attribution accuracy with r > 0.5, providing the first reliability estimate for TDA scores.

**Prediction S3**: CFAR-normalized attribution should improve cross-query consistency (lower variance in attribution quality across test inputs) without sacrificing average performance.

**Prediction S4**: The performance ranking of phi^T * M * psi methods should follow the matched filter optimality hierarchy: Sigma_noise^{-1} > I > random M, with the gap largest on high-FM2 tasks.

### Failure Mode

Sigma_noise estimation requires a sufficient number of "negative" samples to estimate the covariance reliably. In high dimensions (d > 2048), the sample covariance may be poorly conditioned, requiring regularization (shrinkage estimation). The optimal M may also vary across layers, complicating the implementation.

**Computational Cost**: ~1.5 GPU-hours (covariance estimation + matrix inverse are the bottlenecks; can use Woodbury identity for efficiency)
**Success Probability**: 65% -- the mathematical framework is rigorous and the implementation is straightforward; the question is whether the common-influence covariance is sufficiently structured to benefit from whitening
**Impact if Succeeds**: High -- transforms the phi^T * psi framework from a descriptive taxonomy to a prescriptive theory with an optimality criterion (whitened matched filter is provably optimal)

### Key References

- Stankovic & Mandic 2021 (arXiv:2108.10751) -- CNNs as matched filters; provides the mathematical foundation for interpreting convolution as template matching
- Wang et al. 2024 (arXiv:2401.16729) -- Widely-linear matched filter for complex-valued CNNs; proves enhanced SNR for noncircular data
- Yan et al. 2021 (arXiv:2104.03961) -- Matched filtering is formally equivalent to a neural network; "complexity standard candle" for comparing detection methods
- Li et al. 2021 (arXiv:2210.08521) -- Demystifying CNNs via matched filters; convolution-activation-pooling chain as matched filtering pipeline
- Anders et al. 2025 (arXiv:2512.13542) -- Deep learning for signal detection: DNN approaches outperform matched filters for unknown signal parameters

---

## Synthesis: How the Four Analogies Form a Coherent Theoretical Framework

The four interdisciplinary angles are not independent proposals -- they form a unified narrative about **signal extraction from noisy, structured backgrounds**, viewed through four complementary lenses:

```
PROBLEM: Separate task-relevant attribution signal from common background

Neuroscience (Angle 1)     --> "Credit assignment via eligibility traces"
  Mechanism: Per-sample temporal tagging separates specific from shared influence
  Import:    Eligibility-Trace Attribution (ETA) for per-sample debiasing

Statistical Physics (Angle 2) --> "Renormalization group as principled compression"
  Mechanism: Hierarchical coarse-graining eliminates irrelevant degrees of freedom
  Import:    Layer selection = optimal RG flow; d_eff follows power-law decay

Immunology (Angle 3)       --> "Self/non-self discrimination"
  Mechanism: Two-stage selection (negative selection + peripheral tolerance)
  Import:    Two-stage debiasing: mean subtraction + adaptive suppression

Signal Processing (Angle 4) --> "Matched filtering in colored noise"
  Mechanism: Optimal linear detection via noise-whitened template matching
  Import:    Whitened attribution phi^T * Sigma^{-1} * psi is provably optimal
```

### The Unified Formula

All four angles converge on an enhanced attribution score:

```
A_unified(z_test, z_i) = [phi(z_test) - phi_shared]^T * Sigma_noise^{-1} * [psi(z_i) - psi_shared] * gate(z_test)
```

where:
- `phi - phi_shared` = mean subtraction (FM2 fix, immunology's negative selection)
- `Sigma_noise^{-1}` = whitening (signal processing's optimal matched filter)
- `gate(z_test)` = task/query-dependent modulation (neuroscience's neuromodulatory third factor, immunology's peripheral tolerance)
- The feature space R^d at layer L* = the RG-optimal compression point (statistical physics)

This unified formula subsumes all existing representation-space methods and adds three provably beneficial modifications: whitening, gating, and optimal layer selection. Each modification has independent theoretical justification from a different field.

### Contribution Hierarchy

| Angle | Novelty | Implementability | P(success) | Impact |
|-------|---------|-----------------|------------|--------|
| 4. Matched Filter (Signal Processing) | Medium -- connection noted in CRA but not formalized | High -- straightforward covariance estimation | 65% | High -- whitened MF is provably optimal |
| 2. RG (Statistical Physics) | High -- first RG analysis of TDA layer selection | Medium -- MI estimation in high-d is noisy | 45% | Very High -- physics-based theory for optimal layer |
| 1. Credit Assignment (Neuroscience) | High -- novel ETA mechanism | Medium -- requires training-time instrumentation | 50% | High -- new attribution paradigm |
| 3. Self/Non-Self (Immunology) | Medium -- extends existing debiasing | High -- simple two-stage scoring | 40% | Medium -- marginal improvement over mean subtraction |

### Recommended Priority

**Angle 4 (Matched Filter) > Angle 2 (RG) > Angle 1 (Neuroscience) > Angle 3 (Immunology)**

Rationale: Angle 4 has the highest probability of success and immediately strengthens the paper's core phi^T * psi framework by adding the optimal M = Sigma^{-1} prediction. Angle 2 provides the deepest theoretical contribution but is riskier. Angle 1 is the most novel but requires training-time instrumentation. Angle 3 is elegant but likely produces only marginal improvements.

---

## Risk Assessment

### Cross-Cutting Risks

1. **Analogy depth vs. formal rigor**: The danger with interdisciplinary analogies is that they remain suggestive rather than rigorous. Mitigation: for each angle, we specify a concrete formula, a testable quantitative prediction, and a falsification condition. The analogies serve as motivation; the predictions stand or fall on their own.

2. **Implementation complexity**: Adding whitening, gating, and layer selection on top of the basic phi^T * psi framework creates a multi-parameter method. Mitigation: the Pragmatist's hardened 2x2 ablation remains the core experiment. The interdisciplinary enhancements are tested as independent ablation factors.

3. **Overfit to DATE-LM**: All predictions are tested on 3 DATE-LM tasks. With only 3 data points, any pattern could be noise. Mitigation: report per-task results with bootstrap confidence intervals; supplement with Li et al.'s benchmark for cross-benchmark validation.

### What if ALL Four Analogies Fail Quantitatively?

The qualitative framework still adds value: the paper can discuss the matched filter interpretation, RG layer selection, and immune debiasing as conceptual tools that motivate the phi^T * psi framework, even if the quantitative predictions (power-law exponents, SNR correlations, improvement percentages) do not pan out. The core 2x2 ablation and bilinear taxonomy are independent of these analogies.

---

## Summary Table

| Angle | Field | Core Mechanism | TDA Import | Pilot | Full | P(success) | Impact |
|-------|-------|---------------|------------|-------|------|-----------|--------|
| 1. Credit Assignment | Neuroscience | Three-factor rule + eligibility traces | Per-sample temporal tagging (ETA) | 15 min | 1.5h | 50% | High |
| 2. Renormalization Group | Statistical Physics | Hierarchical coarse-graining | Optimal layer = RG fixed point; d_eff power law | 15 min | 2h | 45% | Very High |
| 3. Self/Non-Self | Immunology | Negative selection + peripheral tolerance | Two-stage debiasing with adaptive suppression | 10 min | 1h | 40% | Medium |
| 4. Matched Filter | Signal Processing | Optimal detection in colored noise | Whitened attribution phi^T Sigma^{-1} psi | 15 min | 1.5h | 65% | High |
