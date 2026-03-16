# Contrarian Perspective: CRA Research Proposal

**Agent**: Contrarian (devil's advocate, stress-testing assumptions, blind spots)
**Date**: 2026-03-16

---

## Executive Summary

The CRA project rests on three pillars: (1) FM1 and FM2 are the root causes of parameter-space TDA failure, (2) the phi^T * psi bilinear framework meaningfully unifies representation-space methods, and (3) systematic evaluation on DATE-LM will produce clean, publishable results. I challenge all three. My goal is not to kill the project but to identify where the narrative is weakest, where the evidence is thinnest, and where the experiments could produce results that embarrass rather than support the thesis. The strongest version of this paper must survive these attacks.

---

## Challenged Assumption 1: "FM1 and FM2 Are the Primary Failure Modes of Parameter-Space TDA"

### The Prevailing Assumption

The CRA narrative claims parameter-space methods fail on LLMs due to two signal processing defects: signal dilution (FM1) in high-dimensional gradient space and common influence contamination (FM2) from pre-training knowledge. The implicit claim is that if you fix these two defects, parameter-space methods should recover.

### The Contrarian Challenge: The Real Problem May Be Optimization Landscape Misspecification, Not Signal Processing

**Counter-evidence 1: Better Hessians DO matter, and the improvement is not orthogonal to FM1/FM2.** Hong et al. (2509.23437) demonstrate that improving Hessian approximation quality *consistently* improves attribution quality. Critically, they show K-FAC eigenvalue mismatch is the dominant error source. This is NOT a signal processing issue -- it is a curvature estimation issue. If the Hessian is the bottleneck, then FM1/FM2 may be downstream symptoms, not root causes. The CRA paper claims FM1/FM2 are "orthogonal to Hessian error" -- but this is an assertion, not a proven fact. The 2x2 ablation cannot distinguish between:
- (a) FM1/FM2 are independent failure modes (CRA claim)
- (b) FM1/FM2 are consequences of poor Hessian approximation that happen to manifest as signal dilution and contamination

**Counter-evidence 2: DDA's "debias" may not be fixing FM2 at all.** DDA (2410.01285) shows debias contributes 55pp of improvement. CRA interprets this as "removing common influence contamination." But an equally valid interpretation is that debias is correcting the *mean-shift bias* inherent in first-order Taylor approximations of the influence function. This is a linearization error, not a "common influence" problem. The debias step subtracts the population mean gradient -- which is exactly what you would do to correct the constant term in a Taylor expansion. The CRA narrative imputes a causal mechanism (pre-training knowledge contamination) to what may be a simple mathematical artifact (first-order approximation bias).

**Counter-evidence 3: Daunce (2505.23223) achieves competitive attribution WITHOUT gradients at all.** Daunce uses ensemble perturbation covariance -- no gradients, no Hessians, no representations. If parameter-space methods fail because of FM1/FM2, Daunce should also fail (it does not operate in representation space either). Yet it achieves SOTA or competitive performance. This suggests the failure of gradient-based parameter-space methods may be specific to the gradient/Hessian computation pipeline, not to "parameter space" as a concept.

**Counter-evidence 4: Infra (2505.19949) successfully uses influence functions for fine-grained attribution.** Kou et al. use influence functions to attribute LLM reasoning abilities to individual training tokens and discover non-trivial cross-domain effects. If parameter-space IF were truly crippled by FM1/FM2, this fine-grained token-level attribution should fail. Its success suggests the FM1/FM2 story is oversimplified -- parameter-space methods may work fine when the task structure is right.

### The Inconvenient Research Direction

**Proposal: Disentangle Hessian error from FM1/FM2 before claiming orthogonality.** Design an experiment where the Hessian is exactly computed (feasible on Pythia-70M with small training sets), then test whether FM1/FM2 still manifest. If exact-Hessian parameter-space IF performs well, FM1/FM2 are NOT independent failure modes but symptoms of approximation error. If it still fails, FM1/FM2 are genuine. This experiment is absent from all three other perspectives and is the single most important control for the CRA thesis.

**Estimated cost**: 2 GPU-hours (exact Hessian on Pythia-70M, 1K training samples)
**Success probability for CRA thesis**: 50% -- genuinely uncertain which way this goes
**Risk if CRA thesis fails**: The paper degenerates from "two fundamental failure modes" to "Hessian approximation is hard, and by the way representations work better"

---

## Challenged Assumption 2: "The phi^T * psi Bilinear Framework Is a Meaningful Unification"

### The Prevailing Assumption

All 5 representation-space methods (RepSim, RepT, AirRep, Concept Influence, In-the-Wild) can be expressed as phi(z_test)^T * psi(z_train), and this common structure reveals deep relationships between them.

### The Contrarian Challenge: The Framework May Be Vacuous -- Everything Linear Is Bilinear

**Counter-evidence 1: The phi^T * psi form is trivially universal for any linear similarity.** As the Theoretical perspective's own Theorem 6 acknowledges, ANY linear attribution method can be written as phi^T * M * psi. This is not a discovery -- it is the Riesz Representation Theorem applied to bilinear forms. Claiming that 5 methods all fit this form is like claiming that 5 different functions can all be written as f(x) -- it tells you nothing about their relationships, just about the expressiveness of the notation.

The real question is: what are the constraints on phi, psi, and M that distinguish one method from another? Without such constraints, the "unification" is a taxonomy, not a theory. A taxonomy of the form "Method A uses phi_A, Method B uses phi_B" is a table, not a theorem.

**Counter-evidence 2: The methods differ in ways the bilinear framework cannot capture.** RepT has a *layer selection* mechanism (the "phase transition layer") that is not encoded in phi or psi but in which layer's representations are used. AirRep involves *learning* the representation encoder -- the phi is itself optimized, making it fundamentally different from methods with fixed phi. Concept Influence uses a *concept-specific* direction -- the phi is conditioned on external labels. These structural differences are more important than the shared bilinear form, and the framework sweeps them under the rug.

**Counter-evidence 3: CKA-like similarity measures are known to be manipulable and unreliable.** Davari et al. (2210.16156) demonstrate that CKA -- a representation similarity metric closely related to RepSim -- can be manipulated without changing model behavior. Cloos et al. (2407.07059) show that high CKA similarity scores do not guarantee encoding task-relevant information consistently with ground truth, and that CKA over-emphasizes high-variance principal components. If RepSim inherits these pathologies, its strong empirical performance may be partially artifactual: it detects surface-level geometric similarity rather than genuine causal influence. Okatan et al. (2511.01023) further demonstrate that global CKA > 0.9 can coexist with radically different subspace-level behavior -- transfer strength depends on alignment within a *trait-discriminative subspace*, not global similarity. This directly undermines the assumption that representation similarity = attribution quality.

**Counter-evidence 4: The framework ignores non-linear attribution methods that may outperform all linear ones.** k-NN based attribution (which is inherently non-linear) is a strong baseline that the phi^T * psi framework cannot express. Daunce's covariance-based approach is quadratic in the features. If these non-linear methods outperform all phi^T * psi methods on DATE-LM, the "unification of the best methods" claim collapses.

### The Inconvenient Research Direction

**Proposal: Test whether the phi^T * psi decomposition adds predictive power beyond "use representations."** Design a meta-experiment: for each method, compute (a) its performance on DATE-LM, and (b) the similarity of its phi/psi to the optimal SVD-derived phi*/psi* (per the Theoretical perspective's Theorem 8). If the phi/psi similarity does NOT predict performance ranking, the framework is decorative rather than explanatory. Also include k-NN attribution and Daunce as controls that fall outside the framework.

**Estimated cost**: 3 GPU-hours (need LOO oracle on small subset + SVD)
**Success probability for CRA thesis**: 40% -- bilinear frameworks rarely have tight enough bounds to predict rankings
**Risk if CRA thesis fails**: "Unification" section gets demoted to "notation" section; paper loses one of three claimed contributions

---

## Challenged Assumption 3: "DATE-LM Will Produce Clean, Generalizable Evidence"

### The Prevailing Assumption

DATE-LM is the community-standard benchmark for LLM TDA, and strong results on its three tasks (data selection, toxicity filtering, factual attribution) will provide convincing evidence for the CRA thesis.

### The Contrarian Challenge: DATE-LM's Own Results Undermine the Representation-Space Narrative

**Counter-evidence 1: DATE-LM explicitly finds that "no single method dominates across all tasks."** The benchmark paper itself (2507.09424) reports that simple baselines like BM25 sometimes match or exceed attribution methods. If BM25 -- a lexical matching method with zero understanding of model internals -- can compete with RepSim on factual attribution, then RepSim's success may reflect lexical overlap rather than genuine representation-level attribution. The CRA paper risks building an elaborate theoretical edifice on foundations that DATE-LM itself questions.

**Counter-evidence 2: DATE-LM's three tasks test fundamentally different things.**
- *Data selection*: Which training data would most improve the model? (prospective, causal)
- *Toxicity filtering*: Which training data caused toxic behavior? (retrospective, causal)
- *Factual attribution*: Where did the model learn this fact? (provenance, correlational)

The FM1/FM2 diagnosis may apply to some tasks but not others. Factual attribution is especially problematic: a model may "know" a fact from pre-training, not fine-tuning, making ANY fine-tuning attribution method fundamentally wrong for that fact. FM2 (common influence) is not just a contamination to be removed in this case -- it IS the correct attribution.

**Counter-evidence 3: The 2x2 ablation has only 3 data points per cell.** With 3 DATE-LM tasks, statistical power is extremely low. The pre-registered condition "interaction term < 30% of min main effect" cannot be tested with meaningful statistical significance. Bootstrap over test queries (as the Pragmatist suggests) helps with within-task variance but does not address the N=3 problem across tasks. A reviewer will rightfully ask: "3 tasks is not enough to claim generality."

**Counter-evidence 4: The representation-space advantage may be task-dependent, not method-dependent.** Li et al. (2409.19998) showed RepSim dominates on identification tasks (which training data was used?). But DDA (2410.01285) showed that properly debiased IF achieves AUC 91.64% -- competitive or superior to representation methods on *different* tasks. The parameter vs. representation gap may collapse on certain task types, making the CRA narrative task-specific rather than universal.

**Counter-evidence 5: In-the-Wild (2602.11079) outperforms gradient-based attribution by being 10x cheaper, but on a DPO-specific task.** Its success may not transfer to DATE-LM's tasks. If In-the-Wild fails on DATE-LM while succeeding on DPO attribution, the "unified family" claim is weakened -- these methods may be successful for different, incompatible reasons.

### The Inconvenient Research Direction

**Proposal: Include a "break the narrative" experiment.** Specifically:
1. Run properly-tuned DDA (with both debias and denoise) as a parameter-space method and show it as a fourth row in the 2x2 matrix. If DDA's debiased IF matches or exceeds representation methods, the "parameter space is fundamentally broken" claim needs heavy qualification.
2. Run BM25 and k-NN baselines alongside all methods. If these trivial methods match representation-space TDA on >= 1/3 of tasks, the sophisticated framework is solving a problem that may not need sophisticated solutions.
3. Test on Li et al.'s benchmark in ADDITION to DATE-LM. If the 2x2 pattern only appears on one benchmark, the results are benchmark-specific, not general.

**Estimated cost**: 2 GPU-hours (DDA + baselines, cached representations)
**Success probability for CRA thesis**: 60% -- representation methods probably do win on average, but the margins may be embarrassingly small
**Risk if CRA thesis fails**: Paper becomes "representation methods are slightly better in some settings" instead of "we diagnose two fundamental failure modes"

---

## Additional Blind Spots

### Blind Spot 1: The Signal Processing Analogy May Not Withstand Scrutiny

The matched filter / differential detection analogy is elegant but may not survive peer review by signal processing experts. Classical matched filtering assumes:
- Known signal waveform (in TDA: we don't know which training samples are relevant)
- Additive white Gaussian noise (in TDA: gradient noise is highly structured, not white)
- Linear observation model (in TDA: neural networks are highly nonlinear)

If a reviewer from the signal processing community points out that none of the classical assumptions hold, the analogy becomes misleading rather than illuminating. The paper would be better served by presenting the dimension reduction / mean subtraction as empirical techniques rather than dressing them in borrowed theory that does not strictly apply.

### Blind Spot 2: RepSim's Success May Be a Shortcut, Not Deep Attribution

RepSim computes cosine similarity between test and training representations. In a fine-tuned LLM, representations of semantically similar inputs are close by construction (that is what fine-tuning optimizes for). RepSim may simply be detecting semantic similarity -- a "shortcut" that correlates with attribution but is not causally correct.

Evidence: Consider a model fine-tuned on toxic text. A toxic test query has high RepSim with ALL toxic training samples, not just the ones that caused the model's toxic behavior. RepSim would attribute to all of them equally, which is technically correct as a class-level attribution but uninformative at the instance level. This is exactly the FM2 problem that CRA claims to diagnose -- but it may apply to representation-space methods just as much as parameter-space methods. The other perspectives gloss over this possibility.

### Blind Spot 3: The "5 Methods" May Not Be Independent

RepSim and Concept Influence are highly correlated (Concept Influence's probe is a linear readout of the same representations RepSim uses). RepT augments RepSim with a gradient term. In-the-Wild is RepSim applied to activation differences. AirRep learns a transformation of the same representations. These are not 5 independent methods -- they are variations on a single theme: "use representations." Claiming to "unify 5 methods" when they are all minor variants of the same idea inflates the contribution.

A more honest framing: "We observe that all successful LLM TDA methods rely on representation similarity, and we provide a notation (phi^T * psi) to catalog the variations."

---

## Synthesis: What the Paper Must Do to Survive These Attacks

| Attack | Severity | Mitigation |
|--------|----------|------------|
| FM1/FM2 may be symptoms, not causes | **Critical** | Include exact-Hessian control on small model |
| phi^T * psi is vacuously universal | **High** | Derive non-trivial predictions from the framework that differentiate methods |
| DATE-LM N=3 tasks | **High** | Add Li et al. benchmark; report per-task results prominently |
| DDA-debiased IF may match RepSim | **High** | Include DDA as a strong parameter-space baseline |
| RepSim may be a shortcut | **Medium** | Include instance-level attribution evaluation, not just ranking metrics |
| Signal processing analogy is loose | **Medium** | Present as "intuition" not "theory"; do not over-claim |
| BM25 baseline competitive | **Medium** | Include BM25; if competitive, reframe as "attribution methods must beat retrieval" |
| 5 methods not independent | **Medium** | Acknowledge explicitly; reframe as "variations on representation similarity" |

### The Strongest Version of This Paper

If the paper survives all these attacks, it would look like:

1. **FM1/FM2 as genuine, separable defects** -- verified by exact-Hessian control showing the defects persist even with perfect curvature
2. **phi^T * psi as a generative framework** -- not just a notation, but producing testable predictions about which phi/psi choices are optimal for which tasks
3. **Broad empirical validation** -- DATE-LM + Li et al. + DDA as strong parameter baseline + BM25/k-NN as retrieval baselines
4. **Honest limitations** -- acknowledging that RepSim may partly succeed via semantic shortcut, and that the signal processing analogy is suggestive rather than rigorous

This paper would be genuinely strong. The current proposal, without these controls, is vulnerable to multiple lines of attack that could reduce it from a "diagnostic framework" paper to a "methods comparison" paper.

---

## Risk Assessment

### What Could Validate the Contrarian View

1. **Exact-Hessian IF on Pythia-70M achieves RepSim-competitive performance** -- would prove FM1/FM2 are approximation artifacts, not fundamental. P(this happens) = 30%
2. **DDA-debiased IF matches RepSim on >= 2/3 DATE-LM tasks** -- would prove parameter space is not fundamentally broken, just needs proper debiasing. P(this happens) = 35%
3. **BM25 matches RepSim on factual attribution** -- would prove RepSim detects lexical overlap, not causal influence. P(this happens) = 45%

### What Would Disprove the Contrarian View

1. **Exact-Hessian IF still fails dramatically** -- would confirm FM1/FM2 are real, beyond-approximation defects. P(this happens) = 50%
2. **Representation methods dominate ALL tasks with large margins (> 10pp)** -- would make the parameter-space failure story compelling. P(this happens) = 40%
3. **The 2x2 interaction term is consistently < 15% across both benchmarks** -- would confirm FM1/FM2 orthogonality. P(this happens) = 35%

### Honest Assessment

The CRA thesis is *plausible* but *not yet proven*. The danger is that the paper assumes its conclusion and then designs experiments to confirm rather than falsify it. The contrarian's job is to demand: run the experiments that could *kill* your thesis first. If it survives, it's real science.

---

## Summary Table

| Challenged Assumption | Counter-Evidence | Proposed Falsification Experiment | P(CRA survives) | Impact if CRA Fails |
|---|---|---|---|---|
| FM1/FM2 are root causes | Better Hessians paper; DDA debias as linearization fix; Daunce works without gradients | Exact-Hessian IF on Pythia-70M | 50% | Paper collapses from "diagnosis" to "comparison" |
| phi^T * psi is meaningful | Trivially universal (Riesz); methods differ in un-captured ways; CKA unreliability | Meta-experiment: does phi/psi similarity predict performance? | 40% | "Unification" demoted to notation |
| DATE-LM gives clean evidence | No method dominates; BM25 competitive; N=3 tasks; task-dependent gaps | Add benchmarks, baselines, per-task analysis | 60% | Results are noisy, benchmark-specific |

**Bottom line**: The CRA project is worth pursuing, but only if it front-loads the experiments that could kill it. The current plan is confirmation-biased. Add the three falsification experiments above (exact Hessian, DDA baseline, BM25/k-NN controls) and the paper becomes genuinely rigorous. Without them, a skeptical reviewer will find these holes.

---

## Literature References

- Hong et al. 2025 (2509.23437) -- Better Hessians consistently improve attribution; K-FAC eigenvalue mismatch is dominant error
- Wu et al. 2024 (DDA, 2410.01285) -- Debias contributes 55pp; may be linearization correction, not FM2 fix
- Pan et al. 2025 (Daunce, 2505.23223) -- Ensemble covariance attribution without gradients; competitive at scale
- Kou et al. 2025 (Infra, 2505.19949) -- IF successfully attributes reasoning to tokens; contradicts "IF always fails" narrative
- Davari et al. 2022 (2210.16156) -- CKA can be manipulated without changing functional behavior
- Cloos et al. 2024 (2407.07059) -- High CKA scores do not guarantee task-relevant encoding; CKA over-emphasizes high-variance PCs
- Okatan et al. 2025 (2511.01023) -- Global CKA > 0.9 with radically different subspace behavior; transfer tracks subspace, not global similarity
- DATE-LM (2507.09424) -- "No single method dominates"; BM25 competitive on some tasks
- Li et al. 2025 (2409.19998) -- RepSim dominates on identification tasks specifically
- Xiao & Aranguri 2026 (In-the-Wild, 2602.11079) -- Activation-based TDA 10x cheaper but DPO-specific
- Hu et al. 2026 (2602.10449) -- Projection preserves IF iff sketch dim >= rank(F); abstract theory not measured on LLMs
