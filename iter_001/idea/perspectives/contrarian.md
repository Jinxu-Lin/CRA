# Contrarian Perspective: CRA Research Proposal

## Executive Summary

The CRA proposal is intellectually elegant but built on foundations that are shakier than the team acknowledges. After systematic stress-testing against the pilot evidence, literature, and logical consistency, I identify **three widely-held assumptions** that the CRA thesis depends on -- and challenge each with concrete evidence. My conclusion: the CRA framework risks being a beautiful theory destroyed by ugly facts. The pilot data already contains warning signs that the team is selectively interpreting.

---

## Assumption 1: "Parameter-Space Methods Fail Because of Signal Processing Defects (FM1/FM2)"

### The CRA Claim

Parameter-space TDA methods (TRAK, IF) fail on LLMs due to two independent signal processing defects: FM1 (signal dilution via rank deficiency in gradient space) and FM2 (common influence contamination from pre-training knowledge). Representation-space methods succeed because they implicitly fix FM1.

### The Contrarian Challenge: It Is Simpler Than Signal Processing -- It Is Just Cosine Similarity Being a Better Proxy for Semantic Relatedness

**Evidence 1: The pilot data itself refutes the FM1 narrative on toxicity.**
TRAK (parameter-space) achieves 0.926 AUPRC vs. RepSim's 0.685 on the toxicity task -- a **24pp reversal** of the predicted direction. The CRA team dismisses this as a "task-type boundary" or "gradient norm artifact," but this is post-hoc rationalization. If FM1 were a genuine signal processing defect, it would degrade parameter-space methods uniformly across tasks, not selectively. The fact that parameter-space methods *dominate* on toxicity suggests that the performance difference is task-specific, not space-specific.

**Evidence 2: BM25 achieves perfect R@50=1.0 on counterfact -- the task where RepSim "wins."**
On the very task where RepSim supposedly demonstrates FM1-fixing superiority (counterfact, R@50=0.994), a trivial lexical baseline (BM25) achieves R@50=1.0. This means the "FM1 advantage" of representation space may simply be that cosine similarity of last-layer hidden states captures lexical/semantic overlap -- exactly what BM25 does, but with more parameters. RepSim's advantage may be *retrieval quality*, not *attribution quality*.

**Evidence 3: k-NN (0.809) outperforms RepSim (0.685) on toxicity.**
Both k-NN and RepSim operate in representation space, so FM1 is equally "fixed" for both. Yet k-NN uses a nonlinear similarity measure and substantially outperforms RepSim. This directly contradicts the CRA thesis that the *space* (parameter vs. representation) is the critical factor. The similarity *function* matters more than the space.

**Evidence 4: Cosine similarity has known pathologies in high-dimensional LLM representations.**
Steck, Ekanadham & Kallus (2403.05440) proved that cosine similarity of learned embeddings can yield *arbitrary* similarities depending on implicit regularization. Draganov et al. (2406.16468) showed gradients of cosine similarity go to zero for high-magnitude embeddings -- a known issue for LLM hidden states. The CRA framework assumes representation-space cosine similarity is a meaningful attribution signal, but this assumption is undermined by fundamental mathematical properties of cosine similarity in the spaces where LLMs operate. Bouhsine (2602.19393) recently showed the "problem with cosine similarity is not cosine similarity -- it is the failure to normalize," but CRA's RepSim uses unnormalized representations.

**Counter-proposal:** The performance gap between RepSim and TRAK is better explained by *task-specific information content* of different similarity measures than by "signal processing defects." On tasks where semantic similarity is the primary signal (counterfact, ftrace), cosine similarity of representations naturally captures this. On tasks where behavioral properties matter (toxicity), gradient norms capture this better. No FM1/FM2 framework is needed.

### Risk to CRA

If the "FM1/FM2 defect" narrative is reducible to "different similarity measures capture different information," the entire diagnostic framework collapses into a tautology: "methods that measure the right thing perform better."

### Falsification Test

Run the 2x2 factorial with **Euclidean distance** instead of cosine similarity in representation space. If RepSim-Euclidean matches RepSim-cosine, the similarity function is irrelevant and FM1 may hold. If RepSim-Euclidean degrades significantly, it is the similarity function, not the space, that matters.

---

## Assumption 2: "The phi^T M psi Bilinear Framework Has Predictive Power Beyond Notational Convenience"

### The CRA Claim

All five representation-space TDA methods can be unified under phi(z_test)^T M psi(z_train), and this framework generates non-trivial predictions: (1) whitened attribution (M = Sigma_noise^{-1}) should outperform M = I, (2) contrastive scoring should help universally, (3) the framework predicts a TRAK dimension sweep saturation curve.

### The Contrarian Challenge: The Framework Is Vacuously Universal and Its Predictions Have Already Failed

**Evidence 1: The framework is too general to be falsifiable.**
Any bilinear scoring function can be written as phi^T M psi. This includes random baselines, nonsensical methods, and everything in between. The theoretical perspective already flagged this: "The phi^T M psi framework, as currently stated, is *too general* -- almost any bilinear scoring function can be written in this form." The CRA team claims non-trivial predictions rescue the framework from vacuity. But how have those predictions fared?

**Evidence 2: H7 (whitened attribution) fails catastrophically.**
The framework's most distinctive prediction -- that M = Sigma_noise^{-1} should improve over M = I -- is falsified across all three tasks:
- Toxicity: -10.9pp
- Counterfact: -8.0pp
- ftrace: -10.6pp

The team attributes this to N/d ratio (0.049, underdetermined covariance). But this is a fundamental problem, not a pilot-scale artifact. For Pythia-1B (d=2048), achieving N/d >> 1 requires N >> 2048 training examples per task. DATE-LM's counterfact task has only 5,473 examples total. Even at full scale, N/d ~ 2.7 -- still marginal for covariance estimation. The Ledoit-Wolf shrinkage estimator cannot rescue this: it was designed for N ~ d, not N/d ~ 3.

**Evidence 3: H2 (contrastive scoring) shows zero effect.**
Contrastive scoring (the framework's FM2 correction) produced exactly 0.0pp gain across all 12 method-task combinations. The team blames rank-based metrics. But this reveals a deeper problem: if the FM2 "correction" preserves rank ordering exactly, then FM2 contamination does not affect the *relative* ordering of training examples -- which is the only thing that matters for attribution. A "defect" that does not change rankings is not a defect for ranking-based tasks.

**Evidence 4: H9 (representation isotropy) is falsified -- direction completely reversed.**
The framework predicts representation covariance condition number < 100 (near-isotropic, hence M = I suffices). Actual measurement: condition number = 4.45 x 10^7. Representations are *extremely* anisotropic. This undermines the theoretical justification for why M = I works in representation space -- the claimed near-isotropy that supposedly makes curvature correction unnecessary simply does not exist.

**Evidence 5: The 30.8pp gap between TRAK-PCA at k=d and RepSim.**
The framework predicts that TRAK with PCA projection onto the top-d gradient eigenvectors should approach RepSim performance (the "smoking gun" for FM1). Instead, TRAK-PCA at k=d achieves R@50=0.686, still 30.8pp below RepSim (0.994). This gap is enormous and suggests factors far beyond projection dimension drive the performance difference. The framework cannot explain this gap.

**Counter-proposal:** The phi^T M psi framework is notational, not explanatory. It organizes existing methods into a taxonomy (valuable) but generates no predictions that survive empirical testing (not a theory). The CRA paper should be restructured as a **systematic benchmark with post-hoc analysis**, not a "diagnostic framework with predictive theory."

### Risk to CRA

A framework whose three major predictions (H7, H2/H3, H9) are all either falsified or trivially satisfied has no predictive power. Reviewers will ask: "What does this framework predict that we did not already know?"

### Falsification Test

Identify one prediction of the phi^T M psi framework that is:
(a) non-trivial (not already known from prior work),
(b) testable at full scale, and
(c) not already falsified by pilot data.

If no such prediction exists, demote the framework from "theoretical contribution" to "notation."

---

## Assumption 3: "Representation-Space Methods Succeed Because They Operate in the 'Signal-Rich R^d Subspace'"

### The CRA Claim

Representations live in R^d where d << B, and this subspace concentrates the attribution signal. Gradient-space methods fail because they operate in R^B where most dimensions are noise.

### The Contrarian Challenge: The "Signal-Rich Subspace" Story Is Backwards -- Representations Succeed Because They Are High-Level Features, Not Because They Are Low-Dimensional

**Evidence 1: RepSim PCA dimension sweep shows saturation at k=64, not k=d=2048.**
The pilot shows RepSim performance saturates at PCA k=64 across all tasks with N=100. If the "signal-rich R^d subspace" story were correct, you would need all d=2048 dimensions. Instead, only ~64 dimensions carry attribution-relevant information. This means representation space itself has massive redundancy -- the "signal" lives in a ~64-dimensional subspace of the 2048-dimensional representation space.

**Evidence 2: The effective rank of representations (r_eff_95=63) is almost identical to gradients (r_eff_95=53 for target layers).**
The eigenspectrum data shows that both spaces have similar effective dimensionality (~50-60 at N=100). If FM1 were about dimensionality mismatch, representation-space and gradient-space methods should perform similarly when both are projected to comparable effective dimensions. The 30+pp gap between them suggests dimensionality is not the explanation.

**Evidence 3: What representations actually capture is semantic similarity, not "attribution signal."**
RepSim computes cosine similarity of last-layer hidden states. These hidden states encode semantic content of the input text -- they are the features the model uses for next-token prediction. High RepSim between training example and test example means "these texts are semantically similar." This is fundamentally a *retrieval* operation, not an *attribution* operation. The distinction matters: attribution asks "did this training example *cause* the model's behavior on the test example?", while retrieval asks "is this training example *similar to* the test example?" These are different questions, and RepSim answers the retrieval question.

**Evidence 4: AirRep (Sun et al., 2505.18513) explicitly learns task-specific representations for attribution -- if raw representations were "signal-rich," this would be unnecessary.**
AirRep trains a specialized encoder optimized for attribution quality and matches gradient-based methods while being 100x cheaper at inference. The key insight: *generic* representations are not optimized for attribution. AirRep's success actually undermines the CRA claim that representation space is *inherently* superior -- it suggests that representation space is superior only when the representations are *designed* for attribution, which raw last-layer hidden states are not.

**Evidence 5: Denoised Representation Attribution (Pan et al., 2502.11411) shows that raw representations contain "noise" for attribution purposes.**
Pan et al. identify that unsafe target texts contain neutral tokens whose representations dilute the attribution signal -- exactly the kind of contamination CRA attributes to parameter space (FM2). This means FM2-type contamination exists in representation space too, contradicting the CRA narrative that representation space inherently avoids FM2.

**Evidence 6: Influence dynamics are non-static (Lee et al., 2510.12071).**
Training data attribution treats influence as static, but Lee et al. show influence changes non-monotonically during training, including sign flips at developmental transitions. The phi^T M psi framework operates at a single checkpoint and cannot capture these dynamics. This is a fundamental limitation, not a minor gap.

**Counter-proposal:** RepSim succeeds not because of low-dimensional signal concentration, but because last-layer representations encode high-level semantic features that happen to correlate with training data relevance for certain tasks (counterfact, ftrace). For tasks where relevance is behavioral rather than semantic (toxicity), this correlation breaks down. The correct explanation is **feature quality**, not **dimensionality**.

### Risk to CRA

If the advantage of representation space is feature quality rather than dimensionality, the entire FM1 formalism (signal dilution, rank deficiency, dimension sweep predictions) becomes explanatorily irrelevant. The paper loses its signal processing framing.

### Falsification Test

Compare RepSim using *random* representations (randomly initialized, untrained model, same d=2048) against RepSim using *trained* representations. If FM1 (dimensionality) is the explanation, random representations should perform comparably (same d). If feature quality is the explanation, random representations should fail.

---

## Additional Contrarian Observations from the Pilot Data

### Observation 1: The Pilot Sample Size (N=100) Makes Most Results Unreliable

With N=100, at most 99 nonzero eigenvalues exist. The eigenspectrum analysis, condition number estimates, and covariance-based methods are all fundamentally limited. Yet the CRA team draws strong conclusions about spectral properties from this data. The pilot summary's "GO" recommendation at confidence 0.60 is generous -- the evidence is directional at best, and several key findings (H9 falsification, H7 failure) might reverse at full scale. Or they might not -- we simply cannot tell.

### Observation 2: The "TRAK Dimension Sweep Non-Monotonicity" Is Suspicious

TRAK R@50 peaks at k=256 (0.785) then *decreases* to 0.670 at k=2048. Non-monotonic performance as a function of projection dimension is unusual and may indicate implementation bugs, numerical instability, or overfitting to the small test set (N=100). The CRA team interprets this as supporting H5, but non-monotonicity contradicts the smooth saturation curve predicted by the signal subspace theory.

### Observation 3: DDA (Parameter-Space + Contrastive) Does Not Outperform TRAK (Parameter-Space, Standard)

DDA achieves 0.876 AUPRC on toxicity vs. TRAK's 0.926. DDA's "debias" step (the supposed 55pp improvement) does not appear here. If FM2 correction is so powerful, why does DDA not dominate? The CRA team does not discuss this discrepancy.

### Observation 4: The 36-Cell Contrastive Matrix Shows Universal Zero Gain

Not a single method-task combination benefits from contrastive scoring. This is not a metric limitation -- it is evidence that mean-subtraction does not change the relative ordering of attribution scores. If FM2 contamination were a real "defect," it should introduce *systematic* errors in ordering (biasing toward high-common-influence examples). The fact that rank ordering is invariant to mean subtraction suggests FM2 is a constant offset, not a source of ordering errors.

---

## Contrarian Research Directions

### Direction 1: The "RepSim Is Just Retrieval" Hypothesis

**Claim:** RepSim's success on attribution benchmarks is a measurement artifact -- current attribution benchmarks (including DATE-LM) inadvertently measure retrieval quality rather than genuine causal influence.

**Evidence:**
- BM25 (pure retrieval) is competitive or superior on factual attribution
- RepSim (semantic retrieval) dominates on tasks where relevant training data is semantically similar to test data
- TRAK (gradient-based) dominates on toxicity where the relevant signal is behavioral, not semantic

**Experiment:** Compare RepSim against a strong retrieval baseline (e.g., Contriever, GTR) on DATE-LM. If dedicated retrieval models match or beat RepSim, the "attribution vs. retrieval" distinction is confirmed. Estimated cost: 2 GPU-hours on Pythia-1B.

**P(success): 60%.** BM25's strong performance on counterfact strongly suggests retrieval confounds.

### Direction 2: The Nonlinear Attribution Hypothesis

**Claim:** The phi^T M psi bilinear framework is fundamentally limited because attribution involves nonlinear interactions. k-NN's superiority over RepSim on toxicity (0.809 vs. 0.685) demonstrates this.

**Evidence:**
- k-NN outperforms all bilinear methods on toxicity
- The 30.8pp TRAK-PCA gap cannot be explained within the bilinear framework
- Kernel methods (RBF, polynomial) in representation space may capture nonlinear attribution patterns

**Experiment:** Run RBF-kernel attribution (phi_i^T phi_j -> exp(-||phi_i - phi_j||^2 / sigma^2)) on all three DATE-LM tasks and compare against RepSim. If RBF-kernel consistently outperforms, the bilinear assumption is a bottleneck. Estimated cost: 1 GPU-hour.

**P(success): 50%.** k-NN's toxicity advantage is suggestive but may be task-specific.

### Direction 3: The "Attribution Benchmarks Measure the Wrong Thing" Hypothesis

**Claim:** DATE-LM's metrics (LDS, AUPRC, R@K) are rank-based and cannot detect the phenomena CRA claims to diagnose. A benchmark that measures *calibrated influence scores* (not just rankings) would reveal different method dynamics.

**Evidence:**
- Contrastive scoring has zero effect on all rank-based metrics but changes raw scores
- FM2 contamination, if real, would manifest as a constant offset that is invisible to rank metrics
- The entire H2/H3 analysis is uninformative because the metrics cannot detect the claimed effects

**Experiment:** Evaluate methods using Kendall-tau correlation between predicted influence scores and leave-one-out retrained ground truth, or use the Linear Datamodeling Score at score level (not rank level). Estimated cost: 4 GPU-hours for LDS ground truth computation.

**P(success): 70%.** This is a legitimate gap in the current evaluation methodology.

---

## Risk Assessment Summary

| Challenged Assumption | Severity | P(CRA narrative survives) | Evidence Strength |
|----------------------|----------|--------------------------|-------------------|
| FM1/FM2 as signal processing defects | Critical | 40% | Strong (toxicity reversal, BM25 competitive, k-NN > RepSim) |
| phi^T M psi predictive power | Critical | 30% | Very strong (H7 fails, H2 trivial, H9 falsified, 30.8pp gap) |
| R^d "signal-rich subspace" | High | 50% | Moderate (PCA saturation at k=64, r_eff similar across spaces) |

**Overall assessment:** The CRA proposal has a compelling narrative but the pilot evidence does not support it as strongly as the team believes. I estimate a **35% probability** that the core thesis survives full-scale experiments intact. The most likely outcome is that the paper needs significant narrative revision -- from "signal processing diagnosis of two independent defects" to "systematic empirical comparison with post-hoc signal processing interpretation."

**Strongest recommendation:** Before investing in full-scale experiments, run the three falsification tests proposed above (RepSim-Euclidean, random-representation RepSim, retrieval model comparison). Total cost: ~5 GPU-hours. If all three support CRA, proceed with high confidence. If any fail, the narrative must be revised before scaling up.

---

## Key References Supporting Contrarian Arguments

- Steck, Ekanadham & Kallus (2403.05440): *Is Cosine-Similarity of Embeddings Really About Similarity?* -- cosine similarity yields arbitrary similarities depending on regularization
- Draganov, Vadgama & Bekkers (2406.16468): *The Hidden Pitfalls of the Cosine Similarity Loss* -- gradients of cosine similarity go to zero for high-magnitude embeddings
- Bouhsine (2602.19393): *In Defense of Cosine Similarity* -- normalization eliminates gauge freedom; failure to normalize is the real problem
- Sun et al. / AirRep (2505.18513): *Enhancing Training Data Attribution with Representational Optimization* -- learned representations outperform raw representations for TDA, undermining "inherent superiority" of representation space
- Pan et al. / DRA (2502.11411): *Detecting and Filtering Unsafe Training Data via Data Attribution with Denoised Representation* -- raw representations contain attribution-irrelevant noise, contradicting FM2-free narrative
- Lee et al. (2510.12071): *Influence Dynamics and Stagewise Data Attribution* -- influence is non-static, changes non-monotonically, undermining single-checkpoint phi^T M psi framework
- Mlodozeniec et al. / d-TDA (2506.12965): *Distributional Training Data Attribution* -- stochasticity in training fundamentally limits deterministic attribution approaches
- Yang et al. / Integrated Influence (2508.05089): *Data Attribution with Baseline* -- LOO-based methods overlook collective influence, suggesting bilinear pairwise frameworks are fundamentally incomplete
- Rubinstein & Hopkins / RIF (2506.06656): *Rescaled Influence Functions* -- influence functions systematically underestimate effect of sample removals in high dimensions
- DATE-LM / Jiao et al. (2507.09424): *Benchmarking Data Attribution Evaluation for LLMs* -- "no single method dominates across all tasks" is the benchmark's own conclusion
- Wang et al. (2409.05657): *Adversarial Attacks on Data Attribution* -- attribution methods exploitable via outlier bias, questioning fundamental reliability

---

## Constructive Recommendations

Despite the contrarian critique, the CRA proposal has genuine value if repositioned correctly:

1. **Lead with the benchmark, not the theory.** The 2x2 factorial on DATE-LM with multiple methods has never been done. This alone is publishable. Let the data tell the story rather than imposing a signal processing narrative.

2. **Reframe FM1/FM2 as post-hoc interpretive lenses, not causal diagnoses.** "Parameter-space and representation-space methods capture different information" is defensible. "Two independent signal processing defects explain performance gaps" is not supported by the evidence.

3. **Add retrieval baselines.** Contriever, GTR-T5, or any dedicated dense retriever should be included. If RepSim performs comparably to these, it confirms the "attribution as retrieval" interpretation.

4. **Abandon whitened attribution as a contribution.** H7 has failed. The N/d ratio problem is structural, not fixable at DATE-LM scale. Whitened attribution requires either much larger datasets or much lower-dimensional representations to work.

5. **Investigate the toxicity reversal seriously.** This is the most interesting finding in the pilot data and currently the most under-analyzed. Why does TRAK dominate on toxicity? Understanding this could lead to a genuine contribution: task-dependent method selection criteria grounded in the information content of different scoring mechanisms.
