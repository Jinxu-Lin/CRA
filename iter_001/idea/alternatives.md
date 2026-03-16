# CRA: Backup Ideas for Potential Pivot

## Alternative 1: Hessian Quality Diagnosis Paper

**Trigger**: K-FAC IF on Pythia-70M matches RepSim performance (H6 falsified), proving FM1/FM2 are downstream symptoms of Hessian approximation error.

**Pivot Thesis**: "The primary bottleneck of parameter-space TDA on LLMs is Hessian approximation quality, not inherent signal processing defects. Representation-space methods succeed because they implicitly bypass the Hessian entirely, computing a near-optimal attribution score without curvature correction."

**Key Experiments**:
1. Hessian quality ladder: Diagonal -> K-FAC diagonal -> K-FAC full eigendecomp -> exact (Pythia-70M), measuring attribution quality at each level
2. Correlation between Hessian eigenvalue approximation error (Hong et al. 2509.23437) and attribution quality degradation
3. Representation-space methods as "implicit Hessian-free attribution": show that RepSim ~ IF with M=I works because representation covariance is near-isotropic (no curvature correction needed)

**Strengths**: Novel and surprising finding; directly builds on Hong et al. (2509.23437); strong mechanistic story
**Weaknesses**: Less novel framework contribution (no phi^T psi unification); may be seen as "just an engineering insight"
**Venue fit**: NeurIPS 2026 poster (engineering contribution)

---

## Alternative 2: Representation-Space TDA Benchmark and Survey

**Trigger**: 2x2 factorial interaction term too large (H3 falsified), or phi^T psi framework fails to generate predictive power.

**Pivot Thesis**: "We systematically benchmark five representation-space TDA methods on DATE-LM, revealing that all succeed via similar mechanisms (semantic similarity in representation space) but with task-dependent performance profiles. We provide the first comprehensive comparison and practitioner guide for method selection."

**Key Experiments**:
1. Multi-method tournament on DATE-LM: all 5 representation methods + TRAK + BM25 + k-NN
2. Per-task analysis revealing which methods excel where (data selection vs. toxicity vs. factual attribution)
3. Layer selection analysis across methods (middle layers optimal per Vitel & Chhabra)
4. Computational cost comparison (wall-clock time, memory, ease of implementation)

**Strengths**: Immediately useful to practitioners; lower risk (benchmark papers rarely fail); comprehensive coverage
**Weaknesses**: Perceived as "just a benchmark" without theoretical depth; phi^T psi framework demoted to notation
**Venue fit**: EMNLP 2026 or COLM 2026 (empirical contribution)

---

## Alternative 3: Matched Filter Theory for Data Attribution

**Trigger**: Whitened attribution (H7) succeeds dramatically, suggesting the signal processing theory is the main contribution.

**Pivot Thesis**: "Representation-space TDA is mathematically equivalent to matched filtering in colored noise. We derive the optimal attribution score phi^T Sigma_noise^{-1} psi from detection theory, provide per-query reliability estimates via output SNR, and demonstrate 3-8pp improvement over standard RepSim on DATE-LM."

**Key Experiments**:
1. Whitened RepSim vs. standard RepSim across all DATE-LM tasks
2. Per-query SNR_out prediction of attribution accuracy (r > 0.5 validation)
3. CFAR-normalized attribution for cross-query calibration
4. Noise covariance structure analysis (is FM2 really "colored noise" or white noise?)

**Strengths**: Strong theoretical grounding (Neyman-Pearson optimality); novel connection between signal processing and TDA; practical improvement with reliability estimates
**Weaknesses**: Narrower scope (single contribution rather than diagnostic framework); may not work if common influence covariance is near-isotropic
**Venue fit**: NeurIPS 2026 spotlight (if SNR prediction validates)

---

## Alternative 4: Causal Deconfounding Framework for TDA

**Trigger**: FM2 analysis (Theorems 3-4) proves especially compelling; DiD parallel trends assumption validated (H7 pilot passes).

**Pivot Thesis**: "Training data attribution is a causal inference problem: standard scores are confounded by pre-training knowledge. We provide the first formal bias decomposition, prove that mean subtraction is a consistent deconfounding estimator, and introduce Difference-in-Differences attribution as a principled alternative when pre/post checkpoints are available."

**Key Experiments**:
1. Bias decomposition validation: measure ||phi_shared||/||phi_task|| in parameter vs. representation space
2. Mean subtraction vs. DiD vs. DDA comparison on DATE-LM
3. Parallel trends validation for DiD
4. Task-dependent FM2 severity index

**Strengths**: Novel theoretical framing connecting TDA to causal inference; formal guarantees under stated assumptions; builds on well-established econometrics methodology
**Weaknesses**: DiD may fail if parallel trends violated (55% success probability); may be seen as incremental over DDA
**Venue fit**: ICML 2027 poster to spotlight
