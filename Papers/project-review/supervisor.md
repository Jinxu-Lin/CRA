# Supervisor Review Report

## Overall
- **Quality Score**: 6.5 / 10
- **Core Assessment**: This is a well-conceived diagnostic framework paper with strong experimental design and unusually honest epistemic framing. The three-bottleneck decomposition addresses a real gap in the TDA literature. However, the paper is entirely in PLACEHOLDER mode (43 pending items, zero experiments executed, probe not run), which means its actual contribution is currently an experimental plan, not a scientific finding. The quality assessment is thus a ceiling estimate conditional on results confirming predictions.
- **Submission Readiness**: Do Not Submit (in current PLACEHOLDER state)
- **AC Decision Simulation**: Reject (desk reject likely for 43 placeholders). If results were filled in and confirmed predictions: Borderline Accept -- the framework is clean but novelty concern is real (organizing known bottlenecks, no new method).

## Dimension Scores

| Dimension | Score | Brief |
|-----------|-------|-------|
| Problem Quality | 8/10 | The TDA-at-LLM-scale problem is real, important, and well-defined. Three bottlenecks are clearly articulated with evidence. The gap (no comparative benchmark, no independence test) is genuine and acknowledged by the field. |
| Method-Problem Fit | 7/10 | The 2x2 factorial design directly targets the three-bottleneck decomposition. Each cell isolates one bottleneck combination. The LoRA vs Full-FT extension is well-motivated. Deduction: "diagnostic framework" is not a method -- there is no algorithmic contribution beyond combining existing methods. |
| Method Rigor | 7/10 | Mathematical formalization is adequate (main effects, interaction term, CMF metric). Causal arguments are properly hedged. The correlation-vs-causation caveat is explicit. Deduction: 30% interaction threshold is arbitrary; JL analogy still occupies more text than warranted for an informal analogy. |
| Experiment Sufficiency | 3/10 | Design is excellent (5 experiments, statistical methodology, falsification criteria). But zero experiments have been executed. The probe has not been run. Score reflects execution, not design. If experiments were complete with results confirming predictions: 7.5/10. |
| Presentation | 7/10 | Well-structured, clear writing, honest hedging. Figure descriptions are informative. Abstract is dense but improved from P3. Bilinear taxonomy is still unresolved. Paper will exceed page limit when results fill in. |
| Overall Contribution | 5/10 | Currently zero -- no results exist. Ceiling if results confirm predictions: 7/10 (useful diagnostic framework + first representation-space benchmark on DATE-LM). Without a novel method, impact is bounded to the TDA subfield. |

## Contribution-Evidence Audit

| Claim | Argumentation | Validation | Evidence Strength | Assessment |
|-------|---------------|------------|-------------------|------------|
| C0: Three-bottleneck framework | S3.2 (framework formalization) | Table 2 (2x2 ablation) | **None** (all PENDING) | Framework is logically sound but empirically unvalidated. If interaction term is large, C0 is weakened. |
| C1: First comparative evaluation on DATE-LM | S2.2, S4.2 | Table 1 (benchmark results) | **None** (all PENDING) | Claim is valid by construction (no prior work does this). But actual results needed to demonstrate the comparison yields insights. |
| C2: Quantitative bottleneck decomposition | S3.4 (2x2 design) | Table 2 (main effects, interaction) | **None** (all PENDING) | Design is the strongest element. ANOVA methodology is appropriate. Statistical testing plan is adequate. But no data exists. |
| C3: LoRA vs Full-FT FM1 test | S3.4 (LoRA dimension), S4.4 | Table 3 (RepSim advantage) | **None** (all PENDING) | Well-motivated by the LoRA artifact concern. But H1 is rated "Weak-Medium" in problem statement. |

## Risk Assessment: Most Likely Reviewer Challenges

1. **Challenge**: "There are no results in this paper."
   **Current Defense**: Insufficient -- the paper cannot be submitted in PLACEHOLDER mode.
   **Suggested Supplement**: Execute at minimum Experiments 0 (probe) and 2 (2x2 ablation) before submission. These require ~13 GPU-days total and directly validate C0 and C2.

2. **Challenge**: "The three bottlenecks are individually known. Where is the new insight?"
   **Current Defense**: Adequate in writing (diagnostic methodology framing) but not demonstrated empirically.
   **Suggested Supplement**: If the 2x2 ablation reveals task-dependent bottleneck profiles (e.g., FM2 dominates toxicity, FM1 dominates data selection), present a "bottleneck map" figure that is the paper's signature visualization. Also frame the 2x2 methodology as reusable for evaluating future TDA methods.

3. **Challenge**: "RepSim is not fixing IF -- it is replacing IF with a different signal. Your 'repair' framing is misleading."
   **Current Defense**: Partially adequate (correlation-vs-causation paragraph in S3.3).
   **Suggested Supplement**: If RepSim achieves high LDS, argue that the practical distinction between "repair" and "replacement" is moot -- the bottleneck decomposition still correctly predicts which method works where. If RepSim achieves low LDS, acknowledge this openly and reframe FM1 repair as "alternative signal" rather than "correction."

4. **Challenge**: "MAGIC already solves the problem with exact IF. FM1 does not exist as an independent bottleneck."
   **Current Defense**: Partially adequate (MAGIC as diagnostic upper bound, acknowledged in Section 3.3).
   **Suggested Supplement**: Must run or scope MAGIC feasibility. If infeasible, state clearly. If feasible with high LDS, pivot paper framing. The current paper does not adequately distinguish "FM1 is real" from "we cannot test FM1 because MAGIC is too expensive."

5. **Challenge**: "43 placeholder values with 'expected' ranges that all confirm your predictions suggest confirmation bias."
   **Current Defense**: None. This is a structural consequence of writing the paper before running experiments.
   **Suggested Supplement**: Remove "expected" ranges from final paper. Present results neutrally and discuss how they confirm or refute each prediction.

## Improvement Suggestions (prioritized)

1. **[BLOCKING] Execute experiments before submission.** At absolute minimum: Experiment 0 (probe, 2 GPU-days) and Experiment 2 (2x2 ablation, 11 GPU-days). Without these, the paper has no scientific content.

2. **[High] Prepare pivot framings.** The paper is written for the "predictions confirmed" scenario. Prepare alternative framings for:
   - RepSim fails on LDS → "correlation vs causation" diagnostic paper
   - MAGIC achieves LDS > 0.90 → "representation space as efficient approximation" paper
   - FM1 absent under Full-FT → "LoRA-specific pathology" paper
   - Interaction term large → "tangled failure modes" paper
   Each of these is still publishable, but the narrative shifts substantially.

3. **[High] Add a random projection baseline.** A random d-dimensional projection of parameter-space gradients would cleanly separate the "dimensionality reduction" hypothesis from the "learned representation" hypothesis. This is cheap to implement and closes a major interpretive gap.

4. **[Medium] Resolve bilinear taxonomy.** Either formalize it with a comparison table of phi/psi for each method (spending 0.5 pages) or cut it to a single sentence. Its current presence dilutes the contribution.

5. **[Medium] Plan appendix content.** MAGIC feasibility (S4.5), efficiency analysis (S4.7), and detailed scale-up results (S4.6) should move to appendix to keep the main paper within 9 pages.

6. **[Low] Diversify beyond LDS.** Report the 2x2 ANOVA for both LDS and P@K. If they disagree, this is itself a finding (the "correlation vs causation" gap). If they agree, it strengthens the conclusions.

## Best Practices Checklist

- [ ] Reproducible (details complete) -- **Partial**: hyperparameters specified, commit hash placeholder present, but no actual code exists for RepSim-C, TRAK-C, or RepT implementations
- [ ] Statistical significance reported -- **Planned**: permutation tests, bootstrap CIs, FDR correction described, but no results
- [ ] Ablations cover key components -- **Designed**: 2x2 ablation is the core experiment, but not executed
- [ ] Compared with recent SOTA (1-2 years) -- **Planned**: MAGIC (2024), DDA (2024), RepT (2025) included, but no results
- [ ] Limitations explicitly stated -- **Yes**: Section 5.3 lists 6 limitations including probe not run, FM1 LoRA specificity, MAGIC feasibility, LDS reliability
- [ ] Code/data release planned -- **Yes**: reproducibility statement in Section 5.4
- [ ] Ethics considered (if applicable) -- **Not applicable**: no human subjects or sensitive data
