# Final Review Report

## Overall
- **Composite Score**: 7.15 / 10.0
- **Decision**: Pass
- **One-line assessment**: A well-structured diagnostic framework paper with strong experimental design that decomposes a real problem into testable hypotheses, currently limited by placeholder data but architecturally sound.
- **Predicted outcome at NeurIPS 2026**: Weak Accept (conditional on experimental results confirming predictions)
- **Confidence**: Medium (PLACEHOLDER mode introduces irreducible uncertainty about experimental outcomes)

## Six-Dimension Scores

| Dimension | Score | Brief |
|-----------|-------|-------|
| Novelty | 7/10 | The three-bottleneck decomposition reorganizes known observations into a testable framework; the factorial ablation methodology for TDA is genuinely new. Not a breakthrough insight, but a useful structuring contribution. |
| Soundness | 7/10 | The logical structure is rigorous: each bottleneck is defined, mapped to a repair, and tested via a clean factorial design. The FM1 argument has been appropriately softened from P3 (JL is now framed as analogy, iHVP degeneracy is primary evidence). The correlation-vs-causation caveat is explicitly addressed. The 30% interaction threshold is now properly framed as descriptive rather than theoretically derived. |
| Significance | 7/10 | Fills a real gap: no prior work compares representation-space TDA methods on a common benchmark, and the bottleneck decomposition provides actionable practitioner guidance. Impact is bounded to the TDA subfield but high within it. |
| Experiments | 7.5/10 | The 2x2 factorial design is the paper's strongest element -- clean, interpretable, and directly tied to the framework. Five experiments of increasing depth (benchmark, ablation, LoRA vs Full-FT, MAGIC feasibility, scale-up) provide thorough coverage. Statistical methodology is sound (permutation tests, FDR correction, bootstrap CIs). The acknowledgment that 3 seeds may be underpowered with a contingency plan for 5 seeds is good practice. |
| Presentation | 7/10 | Significantly improved from P3: abstract now expands FM1/FM2 on first use, signal-processing analogies are properly hedged, section structure follows logical reading order. The paper is dense but readable. Figure descriptions are clear. The paper may still be long when results fill in (experiments section is already substantial in skeleton form). |
| Reproducibility | 7/10 | Implementation details are now adequately specified: contrastive variant construction (layer matching, projection sharing, normalization), fine-tuning hyperparameters (LoRA rank/alpha, learning rates, optimizer settings), DATE-LM commit hash placeholder included. Code release statement present. RepT phase-transition detection algorithm described. |

## Detailed Review

### Strengths
(What I would argue at an AC meeting in defense of this paper)

1. **Clean diagnostic methodology.** The 2x2 factorial crossing {parameter, representation} x {standard, contrastive} is an elegant experimental design that directly tests the independence of FM1 and FM2. This is the kind of systematic ablation that the TDA field needs -- most prior work proposes a method and evaluates it; this paper asks *why* methods succeed or fail. The extension to 2x2x2 with LoRA vs Full-FT is a natural and well-motivated addition.

2. **Honest epistemic framing.** The paper is unusually careful about what it claims vs. what it hypothesizes. The JL analogy is explicitly called an analogy. The independence of bottlenecks is framed as an empirical hypothesis. The correlation-vs-causation gap in representation-space attribution is directly acknowledged rather than swept under the rug. The contrastive scoring prediction for factual attribution (may hurt performance) is a strong falsifiable prediction that demonstrates intellectual honesty.

3. **Practical value.** The practitioner guidance table (Section 5.2) and the CMF diagnostic metric provide actionable tools for method selection. Even if the framework is eventually refined, the benchmark results and method comparison fill a real gap.

4. **Thorough experimental coverage.** Five experiments systematically build from landscape characterization to generality testing. The MAGIC experiment is properly scoped as a diagnostic rather than a proposed method, and the paper acknowledges the feasibility risk. The efficiency analysis adds practical value.

5. **Well-addressed P3 critique issues.** The critical issue (#8, FM1 JL argument) has been addressed by reframing JL as analogy and foregrounding iHVP degeneracy evidence. Major issues (#2/#3/#10 novelty reframing, #9 correlation-vs-causation, #17/#18 implementation details) are substantially resolved.

### Weaknesses
(Each tagged: severity + rewrite-fixable / needs-additional-experiments)

1. **[Major] [rewrite-fixable] Novelty ceiling concern.** The three bottlenecks (Hessian error, dimensionality, pre-training contamination) are individually well-known in the TDA literature. The paper's novelty rests on (a) making the decomposition explicit and (b) testing independence via factorial ablation. A skeptical reviewer could argue this is "well-organized empirical study" rather than "new scientific insight." The paper would benefit from more explicitly framing the contribution as a diagnostic *methodology* that can be applied to future TDA method development, not just a one-time decomposition result.

2. **[Major] [rewrite-fixable] Bilinear taxonomy underdeveloped.** The observation that representation-space methods share a $\phi^\top \psi$ structure is mentioned in Sections 2.2 and 3.3 but never developed. It is correctly hedged as "taxonomic rather than theoretically deep" and "imperfect," but it occupies text without earning its keep. Either formalize it (e.g., show what the encoding functions are for each method, discuss when the analogy breaks) or cut it to a single sentence. Currently it sits in an uncanny valley between contribution and aside.

3. **[Major] [rewrite-fixable] Paper length risk.** The experiments section is already 4+ pages in skeleton form with placeholder tables. When results, analysis paragraphs, and error bars fill in, the paper will likely exceed the NeurIPS 9-page limit. The MAGIC feasibility discussion (Section 4.5) and efficiency analysis (Section 4.7) are candidates for moving to an appendix. The scale-up experiment (Section 4.6) could also be compressed if space is tight.

4. **[Minor] [rewrite-fixable] Incomplete method coverage acknowledged but not fully mitigated.** The paper includes only 2 of 5 representation-space methods (RepSim, RepT). Concept IF and In-the-Wild are excluded with justification, but this limits the "first comparative evaluation" claim. The paper has already narrowed this to "model-internal representation methods" per P3 feedback, which is appropriate, but could go further by discussing what the excluded methods would test if included.

5. **[Minor] [rewrite-fixable] Factual attribution prediction needs stronger setup.** The prediction that contrastive scoring may hurt factual attribution (because the base model already encodes relevant knowledge) is a strong test of the framework. However, the argument could be tightened: what specific mechanism causes harm? Is it that subtracting base-model similarity removes signal that correlates with the correct attribution? A 2-3 sentence elaboration would strengthen this prediction.

6. **[Minor] [rewrite-fixable] CMF metric is ad hoc.** The Common-Mode Fraction is defined but its relationship to FM2 severity is asserted rather than derived. Stating that "high CMF indicates FM2 is a dominant error source" requires the assumption that the common-mode component is indeed noise rather than signal -- which may not hold for factual attribution. This is partially addressed by the factual attribution prediction but could be more explicit.

### Questions for Authors
(What I would ask during rebuttal)

1. If the interaction term $|\Xi|$ turns out to be large (>30% of min main effect) on 2+ tasks, does the framework survive? What would you conclude -- that FM1 and FM2 are not independent, or that representation space partially addresses FM2?

2. The paper uses LDS as the primary metric throughout. Given recent concerns about LDS reliability (acknowledged in Limitations), have you considered a secondary counterfactual metric? What happens if LDS rankings disagree with P@K rankings for key comparisons?

3. For the LoRA vs Full-FT experiment: if Full-FT TRAK actually *improves* (because full gradients are more informative than LoRA gradients), this would complicate the FM1 interpretation. How would you distinguish "LoRA degrades TRAK" from "high dimensionality degrades TRAK"?

4. The paper positions MAGIC as a diagnostic upper bound for parameter-space methods. But if MAGIC is infeasible even at Pythia-1B, the Hessian error contribution remains unbounded. Does this weaken the three-bottleneck framework to effectively a two-bottleneck framework in practice?

## P3 Issue Fix Status

| P3 # | Severity | Issue | Status | Notes |
|-------|----------|-------|--------|-------|
| 1 | Major | Abstract FM1/FM2 undefined | **Fixed** | FM1/FM2 now expanded inline as "signal dilution (which we term FM1)" and "common influence contamination (FM2)" |
| 2 | Major | Three bottlenecks individually known | **Partially fixed** | Framework novelty reframed around diagnostic methodology; could be stronger |
| 3 | Major | Signal-processing analogies overstated | **Fixed** | "70 years of grounding" removed; analogies explicitly called "informal" and "loose" |
| 7 | Major | Bilinear taxonomy underdeveloped | **Partially fixed** | Properly hedged but still occupies text without earning its place |
| 8 | Critical | FM1 JL argument assumes random gradients | **Fixed** | JL now framed as "geometric intuition rather than a formal proof"; iHVP degeneracy is primary evidence; explicit caveat about structured gradients |
| 9 | Major | Correlation-vs-causation gap | **Fixed** | Dedicated paragraph in Section 3.3 distinguishing "repair" from "replacement" |
| 10 | Major | Independence argument informal | **Fixed** | Explicitly framed as "empirical hypothesis, not a theoretical guarantee" |
| 11 | Minor | 30% threshold arbitrary | **Fixed** | Reframed as "descriptive guideline" with practical justification |
| 12 | Minor | CMRR terminology | **Fixed** | Renamed to "Common-Mode Fraction (CMF)" with explicit note avoiding electronics terminology |
| 13 | Minor | Contrastive scoring may hurt factual | **Fixed** | Prediction explicitly made with rationale |
| 14 | Major | Benchmark claim overstated | **Fixed** | Narrowed to "model-internal representation methods"; Concept IF and In-the-Wild exclusion justified |
| 17 | Major | Contrastive variant implementation underspecified | **Fixed** | Layer matching, projection sharing, normalization all specified in Section 4.1 |
| 18 | Major | Fine-tuning details incomplete | **Fixed** | LoRA rank/alpha, learning rates, optimizer, schedule, batch size all specified |
| 19 | Minor | No failure case analysis | **Fixed** | Failure case analysis paragraph added in Section 4.2 |
| 20 | Minor | RepSim layer selection degree of freedom | **Fixed** | Both layers reported per task; explicitly avoids hidden degrees of freedom |
| 21 | Minor | BM25 diagnostic interpretation | **Fixed** | BM25 diagnostic paragraph added in Section 4.2 |
| 22 | Minor | No code release statement | **Fixed** | Reproducibility statement in Section 5.4 |
| 23 | Major | Section numbering inconsistent | **Fixed** | Sections now follow logical reading order |
| 26 | Minor | DATE-LM version not pinned | **Fixed** | Commit hash placeholder included |
| 27 | Minor | RepT phase-transition detection not described | **Fixed** | Algorithm described in Section 3.3 |
| 5 | Minor | "Supplementary reading" note | **Fixed** | Removed |
| 15 | Major | 3 seeds underpowered | **Addressed** | Contingency plan for 5 seeds described; actual execution pending |
| 16 | Major | MAGIC experiment large footprint | **Partially fixed** | Properly scoped as diagnostic; still occupies a full subsection |
| 24 | Major | Experiments overweight | **Not fixed** | Still substantial; will likely need appendix material when results fill in |
| 25 | Minor | Terminology mismatch C0-C4 vs C0-C3 | **Fixed** | Paper uses C0-C3 consistently |
| 6 | Minor | Differentiation from Better Hessians Matter | **Partially fixed** | Mentioned but could be sharper |

**Summary**: 17 of 22 tracked issues fully fixed. 4 partially fixed. 1 not fixed (experiments section length). The P4 editing round was effective -- all critical and most major issues are resolved. No new critical issues introduced.

## Edit Suggestions (for future revision if needed)

| Priority | Suggestion | Dimension | Estimated Score Gain |
|----------|-----------|-----------|---------------------|
| P1 | Strengthen novelty framing: add 2-3 sentences explicitly positioning factorial ablation as a reusable diagnostic methodology for TDA, not just a one-time result | Novelty | +0.3 |
| P1 | Resolve bilinear taxonomy: either formalize into a comparison table or cut to one sentence | Presentation | +0.2 |
| P2 | Plan appendix structure for experiments overflow: move MAGIC feasibility and efficiency analysis to appendix | Presentation | +0.3 |
| P2 | Tighten factual attribution contrastive prediction with mechanistic detail | Soundness | +0.1 |
| P3 | Sharpen differentiation from Better Hessians Matter in Related Work | Novelty | +0.1 |
