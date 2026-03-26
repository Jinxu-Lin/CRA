# External AI Review — RS (Formalize Review R2) — CRA

**Reviewer**: Independent External AI (Codex)
**Date**: 2026-03-26
**Artifact reviewed**: `research/problem-statement.md` v1.2 (post-R1 revision)
**Prior review**: R1 codex review scored 6/10 on v1.1

---

## Overall Impression

The problem statement v1.2 is a mature diagnostic framing of LLM-scale TDA failure, now with three well-targeted revisions from round 1. The MAGIC invalidation decision rule, the Unified Attribution competitive positioning, and the LoRA vs full-FT elevation are all competently executed. The document has moved from "intellectually interesting but ungrounded" to "intellectually interesting, honestly self-aware, and experimentally pre-committed." The biggest remaining concern is unchanged from R1: the entire three-bottleneck edifice rests on an unexecuted probe, and the theoretical scaffolding's value depends entirely on whether empirical results cooperate.

---

## Revision Assessment (R1 -> R2)

### Revision 1: MAGIC Invalidation Decision Rule (Section 1.3)

**Adequacy: Well-addressed.** The three-tier rule (LDS >= 0.90 / infeasible / 0.70-0.90) with corresponding narrative pivots is exactly what was needed. Each tier has a concrete implication for the paper's framing. This transforms a vague "we'll deal with MAGIC if it works" into a pre-committed decision protocol. The rule is specific enough to be actionable and flexible enough to accommodate mixed evidence.

**Remaining gap**: The rule assumes MAGIC and other methods are evaluated under comparable conditions. MAGIC requires deterministic training (no SGD noise), which changes the optimization landscape. If MAGIC is run under deterministic training and RepSim under stochastic training, the LDS comparison conflates training regime with attribution method. The document should explicitly state whether all methods will be evaluated on the same fine-tuned checkpoint or whether MAGIC requires a separate training run -- and if the latter, how the comparison remains fair.

### Revision 2: Competitive Landscape (Section 2.2)

**Adequacy: Adequately addressed.** The distinction (CRA = TDA-specific empirical diagnostic; Unified Attribution = conceptual taxonomy across attribution types) is correctly drawn. The "complementary rather than competitive" framing is honest and defensible.

**Minor note**: The competitive landscape could also mention LoGra+LogIX (ICLR 2026) and MDA (Jan 2026) as indirect competitors on the scalable-TDA front, as the Comparativist flagged in R1. These are not direct threats to CRA's diagnostic angle but they do narrow the "no systematic evaluation exists" claim.

### Revision 3: LoRA vs Full-FT as Core Dimension (Section 1.5 RQ1)

**Adequacy: Well-addressed with a remaining concern.** The addition is substantive -- FM1 as a potential LoRA artifact is now a first-class experimental question rather than a footnote. The explicit statement "if FM1 is severe under LoRA but negligible under full-FT, the paper reframes FM1 as a LoRA-specific pathology" shows commitment to following the evidence. Assumption H1 has been updated to reflect the weakened evidence base.

**Remaining concern**: The 2x2 ablation has now expanded to 2x2x2 (adding LoRA/full-FT). The iteration log notes "需在 design 阶段评估计算成本," but at Pythia-1B with full fine-tuning, each LDS evaluation requires leave-one-out retraining. The total compute for 2x2x2 across three DATE-LM tasks may exceed available GPU budget. This compute feasibility question should be acknowledged in the problem statement itself, not deferred entirely to design.

---

## Blind Spot Report

- **Blind spot 1: Training dynamics non-stationarity is an unacknowledged fourth bottleneck.** The three-bottleneck framework (Hessian, FM1, FM2) assumes a static influence function model. But LLM fine-tuning involves learning rate schedules, warmup phases, and often early stopping. TracIn-style methods accumulate per-checkpoint influence and implicitly address this non-stationarity. TracIn is entirely absent from the method table and bottleneck decomposition. If training dynamics account for a meaningful fraction of attribution error, the 2x2 ablation will produce unexplained variance that the three-bottleneck framework cannot capture, and the authors will likely misattribute it to "interaction effects" or measurement noise.

- **Blind spot 2: The RQ3 ANOVA is statistically meaningless at N=3 tasks.** The two-way ANOVA with a 30% interaction threshold sounds rigorous but is a descriptive exercise with three data points (one per DATE-LM task) and four cells. There is no statistical power to detect or reject an interaction effect. The 30% threshold is arbitrary -- no power analysis or effect size justification is provided. Presenting this as a formal statistical test at a venue like NeurIPS would invite methodological criticism.

- **Blind spot 3: The document has no concrete "H4 fails" contingency beyond vague reframing.** Section 3.4 lists diagnostic checks (absolute LDS, layer sensitivity, Spearman correlation, P@K), but the actual contingency is "the paper reframes as a diagnostic contribution" -- which is unspecified. What is the title? What is the core claim? What is the target venue? The H4-failure scenario is the most likely failure mode (H4 strength = "None"), yet it receives the least concrete planning.

- **Blind spot 4: The concurrent competition risk is still underweighted.** Five independent representation-space methods in 12 months means the field is converging rapidly. Running RepSim/RepT on DATE-LM is the obvious next experiment for any lab in this space. The document lists competition as risk #5 (last), but given that the core benchmark contribution (RQ2) requires no novel insight -- only GPU access and implementation effort -- this risk should be elevated. A 6-month timeline to NeurIPS 2026 submission is tight if a competing benchmark paper appears on arXiv in the interim.

---

## Strengths

- **The MAGIC decision rule (new in v1.2) is the document's strongest addition.** Pre-committing to narrative pivots based on specific LDS thresholds eliminates the most dangerous failure mode in research: post-hoc rationalization. This is genuine methodological discipline.
- **Falsification criteria remain excellent.** Every RQ has a numerical threshold that kills the associated hypothesis. RQ1's "MAGIC within 3pp of RepSim" and RQ3's interaction term criterion are concrete and pre-registered.
- **The assumption table (H1-H5) with explicit evidence strength ratings is unusually honest.** Labeling H4 as "None" and H1 as "Weak-Medium" sets appropriate expectations and builds reviewer trust.
- **Section 2.3 (Attack Angle Limitations) is the best section in the document.** Five specific failure modes, honestly stated, including the uncomfortable "MAGIC invalidation risk" and "ceiling risk." This is how research risk should be communicated.

---

## Weaknesses / Concerns

- **The probe remains unexecuted after two rounds of review.** This is now a process concern, not just a technical one. The probe costs < 1 GPU-day and would resolve the highest-uncertainty assumption (H4). Two review rounds have been spent refining theoretical framing that may be invalidated by a 2-day experiment. The opportunity cost of continued theoretical refinement without empirical grounding is mounting.

- **The three-bottleneck decomposition still lacks a controlled experimental design to support causal attribution.** Comparing TRAK vs MAGIC vs RepSim vs DDA and attributing LDS gaps to specific bottlenecks requires ceteris paribus conditions that the current plan does not enforce. Each method differs in multiple dimensions simultaneously. The gap between TRAK and MAGIC reflects not just Hessian quality but also deterministic vs stochastic training, metagradient vs standard gradient computation, and potentially different checkpoint strategies.

- **The "ceiling risk" (Section 2.1, point 4) remains unaddressed.** The document acknowledges "pure diagnostic + benchmark is poster-level at best" but proposes no escalation path. No novel method contribution is planned. At NeurIPS 2026, a paper titled "we benchmarked existing methods and found X" needs an extraordinary X. The diagnostic framework must either predict method performance on new tasks (demonstrating predictive power) or produce a concrete method improvement (demonstrating engineering value).

- **Novelty self-assessment at "Medium" may be generous given the document's own caveats.** The bilinear unification is "more organizational than theoretical" (Section 1.4). FM1 is partially identified by Li et al., FM2 by DDA, and Hessian error by Better Hessians Matter. The novel contribution is the three-bottleneck decomposition itself, but if it does not produce surprising empirical predictions, it may be seen as relabeling known problems.

---

## Simpler Alternative Challenge

**Simpler problem formulation**: Strip the three-bottleneck diagnostic framework and submit as "Systematic Evaluation of Representation-Space TDA Methods on DATE-LM." Run RepSim, RepT, TRAK, DDA on all three DATE-LM tasks at Pythia-1B. Report LDS, P@K, and rank correlations. No FM1/FM2 theory, no 2x2 ablation, no MAGIC comparison. This achieves ~70% of the citation impact (practitioners want benchmark numbers) at ~30% of the risk and effort. The three-bottleneck analysis can appear as a discussion section if results support it, rather than being the paper's structural skeleton that collapses if FM1 is negligible. The reason to keep the full framework is that it elevates the paper from "benchmark contribution" to "conceptual contribution" -- but only if empirical results cooperate.

---

## Specific Recommendations

1. **Add TracIn to the method comparison table** (Section 1.1): TracIn addresses training dynamics non-stationarity, a plausible bottleneck absent from the framework. Even a brief discussion of why TracIn-style methods are excluded from the decomposition would strengthen the argument's completeness.

2. **Estimate total GPU-days for the 2x2x2 design in the problem statement**: The LoRA vs full-FT expansion is now a core dimension, but compute implications are deferred to design. A back-of-envelope calculation (number of LDS evaluations x cost per evaluation) should appear here so the design phase starts with realistic constraints.

3. **Downgrade RQ3 from ANOVA to descriptive analysis**: With N=3 tasks, present effect sizes and interaction magnitudes descriptively. Remove the 30% threshold or reframe it as a qualitative guideline rather than a statistical criterion.

4. **Develop a concrete H4-failure paper outline**: Specify the alternative paper title, core claim, and key figures for the scenario where RepSim fails on LDS. "The paper reframes" is not a plan -- "the paper becomes 'Correlation vs Causation in Data Attribution: Why Representation Similarity Fails the Counterfactual Test' targeting ICML 2026" is a plan.

5. **Ensure MAGIC comparison uses identical training conditions**: State explicitly in Section 1.3 whether MAGIC and RepSim will be evaluated on the same fine-tuned model. If MAGIC requires deterministic training, acknowledge this as a confound in the decision rule.

6. **Elevate concurrent competition risk from #5 to #2-3**: The benchmark contribution (RQ2) is low-barrier-to-entry. Any funded lab could publish RepSim-on-DATE-LM results within weeks. Time-to-submission is a critical factor that the problem statement should explicitly acknowledge.

---

## Score

**6.5 / 10** -- The three R1 revisions are competently executed: the MAGIC decision rule is the standout improvement, transforming a hand-wavy acknowledgment into a concrete protocol. The competitive landscape positioning and LoRA vs full-FT elevation are adequate. The document has improved from v1.1 (previously scored 6/10) by addressing the most actionable review feedback. However, the fundamental structural risks remain unchanged: (a) zero empirical evidence after two review rounds, (b) the three-bottleneck causal attribution lacks controlled experiments, (c) no novel method contribution caps the venue ceiling, and (d) the H4-failure contingency is still vague. The half-point improvement reflects genuine progress on pre-commitment and self-awareness, but the document cannot score higher until probe data exists. At NeurIPS 2026, this formulation is borderline -- the diagnostic framework is intellectually clear but the paper's fate is entirely empirical, and no empirical work has begun.
