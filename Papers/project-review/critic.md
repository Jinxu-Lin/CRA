# Critic Review Report

## Overall
- **Attack Strength**: 7 / 10
- **Core Weakness**: The paper proposes a diagnostic decomposition of three individually well-known bottlenecks but contains zero empirical results -- 43 PENDING placeholders with speculative ranges. The entire contribution hinges on experimental outcomes that have not been obtained, and the primary probe (RepSim vs TRAK on DATE-LM) has never been executed. The three-bottleneck framing could collapse to a trivial observation if the interaction term is large or if MAGIC invalidates FM1.
- **Most Likely Reject Reason**: "This is a well-organized experimental plan, not a research paper. The claimed contributions are entirely speculative -- every table is a placeholder. Submit when you have results."

## Issues (by severity)

### [Critical] No experimental results exist -- entire paper is speculative
- **Location**: All of Section 4 (Tables 1-5), Abstract (7 PENDING items), Section 5
- **Problem**: The paper contains 43 `{{PENDING}}` placeholders. Not a single experimental number has been obtained. The probe experiment (the "critical gate" defined in problem-statement.md Section 3) has NOT been executed. The paper is currently an experimental design document dressed in paper format.
- **Evidence**: Every table entry in Sections 4.2-4.7 is a PENDING placeholder. The Abstract contains 7 PENDING items. Section 5 discussion and practitioner guidance tables are entirely PENDING.
- **Simulated Reviewer Attack**: "I cannot review a paper with zero results. The authors present predicted ranges for all experiments, which suggests they are retrofitting a framework to anticipated outcomes. How would the framing change if RepSim underperforms TRAK on LDS?"
- **Suggested Fix**: Execute at minimum the probe experiment (Experiment 0) and the 2x2 ablation (Experiment 2) before submission. These are the two experiments that directly validate the core thesis. Without them, the paper should not be submitted.

### [Critical] Primary hypothesis (H4: RepSim LDS competitiveness) is untested
- **Location**: problem-statement.md Section 1.6, probe_result.md
- **Problem**: The entire framework rests on the premise that representation-space methods are competitive on LDS (the counterfactual metric). H4 is rated "None" strength in the problem statement itself. The only probe result available is from a completely different domain (VLA policy learning on LIBERO-10, ~1M param CNN), which has zero relevance to LLM TDA.
- **Evidence**: probe_result.md explicitly states "PROBE NOT YET EXECUTED" and "The pilot results do NOT directly validate the CRA_old FM1/FM2 thesis."
- **Simulated Reviewer Attack**: "The authors admit their core hypothesis has strength 'None' (Table in Section 1.6). RepSim's success on binary classification (Li et al., 96-100%) says nothing about LDS performance. The entire paper could be invalidated by a single 2-GPU-day experiment that the authors chose not to run."
- **Suggested Fix**: Run the probe. This is a 2-GPU-day investment that gates a multi-month project. It is inexcusable to write a full paper without this signal.

### [Major] MAGIC invalidation risk is inadequately addressed in the paper
- **Location**: Sections 3.2, 4.5
- **Problem**: MAGIC achieves LDS ~0.95-0.99 with exact IF in parameter space. If MAGIC works at Pythia-1B scale (feasibility uncertain), it demonstrates that parameter-space IF works excellently when Hessian error is removed -- making FM1 irrelevant. The paper acknowledges this in the research documents but the written paper hedges insufficiently. The "decision rule" from problem-statement.md (MAGIC LDS >= 0.90 means "FM1 thesis weakened") would undermine C0 (the framework) and C2 (the 2x2 ablation), leaving only C1 (benchmark) as a standalone contribution.
- **Evidence**: problem-statement.md Section 1.3: "CRA thesis then collapses to: 'Hessian approximation is the main problem, and representation space is just a cheap alternative to exact IF.' This is true but not novel."
- **Simulated Reviewer Attack**: "MAGIC already achieves LDS 0.95+ in parameter space by fixing Hessian error alone. Your FM1 bottleneck may not exist. Your paper's conceptual contribution evaporates if the Hessian is the only real bottleneck."
- **Suggested Fix**: (1) Run MAGIC or at minimum scope the feasibility experiment before submission. (2) In the paper, present the "representation space as efficient approximation" framing as an equally valid interpretation, not just a fallback. (3) If MAGIC is infeasible, state clearly that the three-bottleneck decomposition is validated only relative to *approximate* IF methods.

### [Major] FM1 evidence is LoRA-specific -- generality unproven
- **Location**: Sections 3.2 (Bottleneck 2), 4.4
- **Problem**: The primary evidence for FM1 is Li et al.'s iHVP degeneracy analysis, which is conducted entirely under LoRA fine-tuning. The paper correctly identifies this (Section 3.2 caveat paragraph), but the three-bottleneck framework NAMES FM1 as a general bottleneck. If Experiment 3 shows FM1 is absent under full fine-tuning, the framework reduces to two bottlenecks (Hessian + FM2) for the dominant training regime.
- **Evidence**: problem-statement.md Section 1.6, H1 rated "Weak-Medium." method-design.md Section 5 Component D: "RepSim advantage is ONLY present under LoRA, FM1 is reframed as LoRA-specific pathology."
- **Simulated Reviewer Attack**: "Your FM1 bottleneck may be an artifact of LoRA's artificial rank constraint, not a fundamental property of high-dimensional parameter spaces. Most serious LLM fine-tuning uses full parameters. A two-bottleneck framework (Hessian + FM2) is less novel and is essentially DDA's contribution (FM2) plus Better Hessians Matter (Hessian)."
- **Suggested Fix**: Frame FM1 as "dimensionality-dependent signal dilution, hypothesized to scale with effective parameter count" and present the LoRA vs Full-FT experiment as a central contribution rather than a side experiment. If FM1 turns out LoRA-specific, this is itself a valuable finding.

### [Major] Novelty ceiling -- diagnostic framework without novel method
- **Location**: Introduction (C0-C3), Section 3
- **Problem**: All three bottlenecks are individually recognized in prior work. Hessian error: extensively studied (Better Hessians Matter, MAGIC). FM1: identified by Li et al. FM2: identified and addressed by DDA. The paper's novelty is (a) naming all three and (b) testing their independence via 2x2 ablation. This is "well-organized empirical study" rather than "new scientific insight." Without a novel method (the original C4 "Fixed-IF" was dropped), the impact ceiling is poster-level at a top venue.
- **Evidence**: contribution.md shows C4 (Fixed-IF method) marked as "Optional, contingent on C2/C3" and was never pursued. The paper explicitly states "CRA is NOT proposing a new TDA method" (method-design.md Section 4.1).
- **Simulated Reviewer Attack**: "This paper names three known problems and runs a factorial experiment. The 2x2 design is clean but the individual components are all from prior work. Where is the new insight that goes beyond organizing existing knowledge?"
- **Suggested Fix**: (1) Emphasize the 2x2 factorial methodology as a *reusable diagnostic tool* for future TDA method development. (2) If RepSim-C (the combined FM1+FM2 fix) substantially outperforms all existing methods, position it as a practical "combination recipe" contribution. (3) Consider adding a lightweight "method recommendation algorithm" based on CMF and bottleneck profiling.

### [Major] Correlation-vs-causation gap may undermine FM1 "repair" framing
- **Location**: Section 3.3 (correlation-vs-causation caveat)
- **Problem**: The paper frames RepSim as "repairing" FM1, but RepSim measures representational similarity (correlational), not counterfactual influence (causal). If RepSim achieves high P@K but low LDS, it does not "fix" signal dilution -- it measures something different. The paper acknowledges this but still uses "repair" language throughout.
- **Evidence**: Section 3.3: "representation-space 'repair' of an attribution error may sometimes constitute 'replacement' -- substituting a different (correlational) signal."
- **Simulated Reviewer Attack**: "You call moving to representation space a 'repair' for FM1, but it may simply be measuring a different quantity. If RepSim's LDS is low, the 'repair' framing is misleading. Even if LDS is high, you are not 'fixing' influence functions -- you are abandoning them for a simpler metric that happens to work."
- **Suggested Fix**: Use "alternative" or "bypass" instead of "repair" consistently. Explicitly distinguish "repairing parameter-space IF" from "replacing IF with a different attribution signal." The 2x2 ablation can still quantify the effect, but the causal language should be more careful.

### [Minor] Bilinear taxonomy ($\phi^\top \psi$) contributes nothing
- **Location**: Sections 2.2, 3.3
- **Problem**: The bilinear observation occupies text in two sections but is explicitly called "taxonomic rather than theoretically deep" and "imperfect." It is neither developed into a formal contribution nor dropped. Concept IF and AirRep do not fit the mold.
- **Evidence**: P5 review.md: "either formalize it or cut it to a single sentence. Currently it sits in an uncanny valley between contribution and aside."
- **Suggested Fix**: Reduce to a single sentence in Related Work. Drop from contribution list (already dropped from C0-C3 vs original C0-C4).

### [Minor] 30% interaction threshold is arbitrary
- **Location**: Section 3.4
- **Problem**: The 30% threshold for "approximate independence" has no theoretical basis. The paper acknowledges this ("not derived from formal theory") but still uses it as a decision criterion.
- **Suggested Fix**: Report the actual interaction magnitude with confidence intervals and let readers judge. Provide context by comparing to typical interaction magnitudes in factorial designs in related fields.

### [Minor] Paper will exceed page limit
- **Location**: Section 4 (experiments)
- **Problem**: The experiments section is already 4+ pages in skeleton form. When 43 PENDING items fill in with actual numbers, analysis paragraphs, and figures, the paper will substantially exceed NeurIPS's 9-page limit.
- **Suggested Fix**: Move MAGIC feasibility (Section 4.5), efficiency analysis (Section 4.7), and scale-up details (Section 4.6) to appendix. Keep main paper focused on benchmark + 2x2 ablation + LoRA vs Full-FT.

## Proxy Metric Gaming Check
- **Result**: At Risk (but assessment is provisional due to PLACEHOLDER mode)
- **Analysis**: The paper relies on LDS as the sole primary metric. LDS measures Spearman correlation between predicted and actual model output changes under training subset removal. Several concerns:
  1. **LDS metric reliability**: The paper itself acknowledges (Section 5.3, Limitation 6) that "recent work raises concerns about LDS as a metric for TDA evaluation." H-RF1 and H-DVEmb3 from the Episteme knowledge base question LDS's validity. If LDS does not capture real attribution quality, all conclusions are compromised.
  2. **Single-metric dependence**: While secondary metrics are listed (AUPRC, Recall@50, P@K), all main comparisons and the 2x2 ANOVA are conducted on LDS. If LDS rankings disagree with secondary metrics, the framework's recommendations become ambiguous.
  3. **Predicted ranges are suspiciously well-behaved**: All PENDING placeholders include "expected" ranges that confirm the framework's predictions. This suggests the paper was written to fit expected outcomes rather than discovered from actual data.
  4. **No actual output inspection**: Cannot assess whether any method produces degenerate outputs because no experiments have been run.
- **Suggested Cross-Validation**: (1) Report all 2x2 main effects for BOTH LDS and P@K. If they disagree, discuss. (2) On toxicity filtering, manually inspect top-10 attributed samples for a random subset of test samples. (3) Report the RepSim-TRAK rank correlation (Spearman) to check whether they identify similar or different samples.

## Missing Baselines / Ablations
1. **Random projection baseline**: TRAK already uses random projection, but the paper does not include a "random projection of representations" baseline. This would test whether the benefit of representation space is the *learned structure* of h^(l) or simply the *dimensionality reduction* from R^B to R^d. A random d-dimensional projection of gradients would isolate the dimensionality factor.
2. **Oracle upper bound**: No oracle/upper bound is provided beyond MAGIC (which may be infeasible). For each task, what is the maximum achievable LDS? DATE-LM may provide this.
3. **Mean-centering baseline for FM2**: The paper compares standard vs contrastive scoring but does not include simple mean-centering as an intermediate FM2 fix. This would test whether contrastive scoring's benefit comes from removing the per-sample common mode (contrastive) vs. the global mean (centering).
4. **Layer sweep for RepSim**: Only middle and last layers are evaluated. A full layer sweep (or at least 4-5 positions) would strengthen the layer selection analysis.
5. **Ensemble of RepSim layers**: Averaging RepSim scores across multiple layers could outperform single-layer selection -- this is a simple baseline that is missing.

## "Kill Shot" Test

**Single most fatal question**: "You have written a 10-page paper with 43 placeholder values and zero executed experiments. Your own problem statement rates the core hypothesis (H4) at strength 'None.' Why should a reviewer spend time evaluating a speculative framework when the 2-GPU-day probe experiment that would validate or kill the entire direction has not been run?"

**Does the paper have a defense?** No. The paper explicitly and repeatedly acknowledges that the probe has not been executed. The paper's own internal documents (probe_result.md, problem-statement.md) frame the probe as a "CRITICAL GATE." There is no defense for submitting a paper whose critical gate experiment has not been run. The paper's architecture is sound, the experimental design is strong, and the writing is careful -- but none of this matters without data.
