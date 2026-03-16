# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-17
**模型**: Codex (GPT-5)

## 评审意见

**Overall Verdict: 6/10 (promising, but currently over-claimed and under-identified)**

The proposal is strong in ambition and experimental effort, but its central narrative is vulnerable: it tries to explain a broad empirical phenomenon with a two-factor causal story (FM1/FM2) using benchmarks and proxies that are not yet sufficient to uniquely identify that story.

### 1. Overlooked Risks
1. **Identifiability risk**: FM1 and FM2 may be *descriptive decompositions*, not causal mechanisms. Many alternative decompositions can fit the same attribution patterns.
2. **Interaction-order risk**: Pairwise/bilinear attribution can miss higher-order training-example interactions; this is a direct threat to "orthogonal independent failure modes."
3. **Estimator-variance confound**: Representation-space gains may mostly come from variance reduction/regularization, not from "true signal subspace recovery."
4. **Metric-target mismatch**: "Influence," "factual support," and "retrieval relevance" are different objectives; optimizing one can degrade another.
5. **Covariance inversion fragility**: Whitened scoring can amplify noisy small-eigenvalue directions unless shrinkage/regularization is very carefully controlled.
6. **Stage dependence**: FM1/FM2 may differ across pretraining vs instruction tuning vs alignment phases; one global diagnosis may be false.
7. **Artifact dependence**: DATE-LM uses counterfactual corruption/paraphrasing choices; method ranking can depend on those design knobs.
8. **Multiple-comparison risk**: With many sweeps/cells, false-positive "effects" are likely without strict pre-registration and correction.

### 2. Methodological Weaknesses
1. **No hard ground truth for causality**: Most evaluation remains proxy-based; removal/retrain or intervention-based validation is still sparse.
2. **Orthogonality test may be underpowered**: A 2x2 factorial is not enough unless effect-size targets and power are pre-specified.
3. **Scale extrapolation weakness**: Eigenspectrum and Hessian analyses at 70M may not transfer to 1B+ behavior.
4. **Unequal method tuning**: If hyperparameter effort differs across families, conclusions about "space" (param vs repr) are confounded.
5. **Runtime/compute fairness unclear**: Without compute-normalized Pareto comparisons, practical superiority claims are weak.
6. **Insufficient robustness checks**: Needs adversarial lexical controls, template controls, and OOD queries as default, not appendix.
7. **Potential leakage via reference design**: If reference/query construction encodes lexical/format biases, contrastive methods may be advantaged.
8. **No calibration analysis**: Top-k retrieval wins do not show calibrated influence scores.

### 3. Assumption Gaps
1. Assumes attribution signal is low-rank in a stable way across tasks/models.
2. Assumes FM1 and FM2 are independent, not coupled manifestations of anisotropic gradients.
3. Assumes linear bilinear form is expressive enough for practical TDA behaviors.
4. Assumes matched-filter "optimality" objective matches the paper's actual downstream utility objective.
5. Assumes noise covariance can be estimated robustly in realistic high-dimensional, nonstationary settings.
6. Assumes benchmark "factual attribution" approximates real causal provenance.
7. Assumes single-example attribution is the right granularity despite known group/curriculum effects.

### 4. Novelty Assessment
1. **Diagnosis framing** is interesting but not yet clearly novel in mechanism.
2. **Bilinear unification** risks being viewed as mostly notational unless you add strict constraints + nontrivial falsifiable predictions.
3. **Whitened attribution** is a known signal-processing idea; novelty is in empirical adaptation, not core theory.
4. Net: likely **incremental-to-moderate novelty** unless causal identification and non-vacuous theory are strengthened.

### 5. Competitive Landscape / Scoop Risk
High risk. Relevant work is already very close:
1. **Scalable Influence and Fact Tracing** (arXiv Oct 2024, ICLR 2025) already reports attribution-vs-influence misalignment and BM25 tension.
2. **DATE-LM** (arXiv July 2025, NeurIPS 2025) shows no single method dominates and results are evaluation-sensitive.
3. **AirRep** (arXiv May 2025, NeurIPS 2025) pushes representation-based scalable attribution strongly.
4. **Do Influence Functions Work on LLMs?** (EMNLP Findings Nov 2025) already documents IF failure modes.
5. **Scalable Multi-Stage IF** (IJCAI 2025; revised Feb 2026) continues to improve parameter-space IF.
6. **Concept Influence** (arXiv Feb 2026) shifts toward behavior-level semantic attribution and may outflank your representation narrative.

### 6. Framing Risks (Alternative Explanations)
1. "Representation-space wins" could be mostly from better conditioning and normalization.
2. "Contrastive fixes FM2" could just be subtracting frequency/common-template effects (an IR-style reweighting effect).
3. "Whitening works" could be regularization benefit, not matched-filter optimality.
4. FM1/FM2 could be one shared geometry problem, not two orthogonal defects.

### 7. Specific Improvement Suggestions
1. **Pre-register** 3 primary hypotheses and analysis plans (effect sizes, alpha correction, stopping rules).
2. Add a **synthetic known-ground-truth testbed** where true influence is controllable.
3. Include **group-interaction attribution tests** (pair/triple removal) to stress non-additivity.
4. Report **compute-normalized Pareto frontiers** (quality vs GPU-hours vs memory).
5. For whitening, compare **diagonal / low-rank / shrinkage / full** covariance with condition-number diagnostics.
6. Add strict **negative controls**: shuffled labels, query-template swaps, lexical-overlap-matched decoys.
7. Do **cross-phase analysis** (pretrain, SFT, aligned checkpoints) before claiming universal FM1/FM2.
8. Downgrade claim language from "optimal" to **"optimal under explicit assumptions"**, then test assumption violations.
9. Add **calibration metrics** (not just ranking): score stability, uncertainty, and intervention consistency.
10. Convert bilinear unification into a **non-vacuous theorem** with constraints that exclude at least one plausible method family.

### Score Justification
**6/10** because the project is coherent and potentially publishable, but current claims outpace causal evidence and novelty insulation. With stronger identification, pre-registered confirmatory design, and tighter claim scope, this could move to ~7.5+.

### References
- https://arxiv.org/abs/2410.17413
- https://arxiv.org/abs/2507.09424
- https://openreview.net/pdf/c10cde391483c2539edf72a397e3ea5ebe7d4123.pdf
- https://arxiv.org/abs/2505.18513
- https://aclanthology.org/2025.findings-emnlp.775/
- https://arxiv.org/abs/2505.05017
- https://arxiv.org/abs/2602.14869
- https://arxiv.org/abs/2303.08114
- https://arxiv.org/abs/2303.14186
- https://aclanthology.org/2022.findings-emnlp.180/
- https://aclanthology.org/2025.acl-demo.18/

## 评分

6/10
