# Experiment Critique Report

## Overall Assessment
- **Score**: 7.5 / 10
- **Core Assessment**: The experimental design is comprehensive and well-structured, with five experiments addressing distinct research questions in a logical progression. The 2x2 factorial ablation is the standout element -- a clean, interpretable design that directly tests the core thesis. The fair comparison protocol (same checkpoint, same evaluation pipeline, cosine normalization) addresses common pitfalls. Main concerns are: the absence of Concept IF and AirRep from the benchmark weakens the "systematic evaluation" claim; the MAGIC experiment is likely to be inconclusive; and 3 seeds may be underpowered for detecting small interaction effects. Note: all experimental *results* are PENDING and are not penalized.
- **Would this dimension cause reject at a top venue?**: No, if results fill in as expected. The experimental *design* is above average for a NeurIPS analysis paper.

## Issues (by severity)

### [Major] "First systematic benchmark" claim is overstated given method exclusions
- **Location**: Section 4.1 (Methods table), Introduction C1
- **Problem**: The paper claims "first systematic evaluation of representation-space methods on DATE-LM" but evaluates only RepSim and RepT out of five representation-space methods (RepSim, RepT, In-the-Wild, Concept IF, AirRep). The exclusion of Concept IF, AirRep, and In-the-Wild is justified in experiment-design.md but not in the paper text. A reviewer encountering the "first systematic benchmark" claim will immediately check which methods are included and find 60% missing.
- **Simulated reviewer phrasing**: "The authors claim a 'systematic' evaluation of representation-space methods but include only 2 of 5 methods. Concept IF and AirRep are excluded without justification in the paper. Including these methods -- or at minimum explaining their exclusion and why the remaining methods are representative -- is necessary to support the 'systematic benchmark' claim."
- **Suggested fix**: Either (a) add Concept IF and In-the-Wild to the benchmark (AirRep can reasonably be excluded as it operates in a learned space, not model internals), or (b) rename the contribution to "first comparative evaluation of model-internal representation-space methods on DATE-LM" and explicitly discuss exclusion criteria in S4.1. The justification from experiment-design.md (Concept IF projects back to parameter space; AirRep uses learned space) is valid but must appear in the paper.

### [Major] Statistical power for interaction term detection may be insufficient
- **Location**: Section 4.3 (2x2 Ablation)
- **Problem**: The paper reports "means and standard deviations over 3 random seeds" and uses "per-sample permutation tests (10K permutations) with bootstrap 95% CIs." With 3 seeds, the between-run variance estimate has only 2 degrees of freedom, making confidence intervals wide. The interaction term is expected to be small (< 30% of main effects), and detecting a small effect with high confidence requires more statistical power. The experiment-design.md acknowledges "we can detect LDS differences of ~3-5pp at alpha=0.05, power=0.80" -- but this is for main effects, not interactions, which have even lower power.
- **Simulated reviewer phrasing**: "The interaction term is the key validation of the independence hypothesis, but with only 3 seeds, the confidence interval on Xi will be wide. The paper may fail to distinguish 'Xi is small' from 'Xi is too noisy to measure,' undermining the core contribution (C2)."
- **Suggested fix**: (1) Increase to 5 seeds for the 2x2 ablation (the core experiment) even if other experiments stay at 3 seeds. (2) Report a formal power analysis for the interaction term specifically. (3) If 3 seeds is a hard constraint, use the per-sample permutation test (which leverages per-test-sample variation, providing more statistical units than 3 seeds alone) and make this clear.

### [Major] MAGIC experiment has unclear value given likely infeasibility
- **Location**: Section 4.5 (Experiment 4)
- **Problem**: The paper repeatedly flags that MAGIC is "likely infeasible" at Pythia-1B scale (method-design.md, experiment-design.md, paper text). The experiment allocates 5 GPU-days but acknowledges 400GB-1.6TB disk requirements and uncertain compute per test sample. If MAGIC is infeasible (the most likely outcome per the paper's own assessment), the Hessian bottleneck contribution remains unbounded, and Experiment 4 contributes nothing except a feasibility report. This is a lot of paper space (0.4 pages) for a likely negative result.
- **Simulated reviewer phrasing**: "The MAGIC experiment is hedged so heavily that it seems unlikely to produce useful results. If the authors expect it to be infeasible, why allocate a full experiment to it? Either commit the resources to make it feasible (e.g., smaller model, fewer training steps) or move it to supplementary material."
- **Suggested fix**: (1) Attempt MAGIC on a smaller model (e.g., Pythia-160M) as a proof-of-concept, then discuss scaling implications. (2) If MAGIC remains infeasible even at smaller scale, use it only as a brief discussion point in S5.3 (Limitations) rather than a full experiment section. (3) Alternatively, use TRAK vs. Grad-Sim performance gap as a proxy for Hessian error severity (TRAK uses random projection while Grad-Sim uses raw gradients; the difference reflects TRAK's dimensionality reduction, not Hessian quality, so this may not work cleanly).

### [Minor] Missing failure case analysis in experiment plan
- **Location**: Section 4 (all experiments)
- **Problem**: The experiments plan to report means and standard deviations, but do not describe any qualitative analysis. For a diagnostic paper, understanding *where* methods fail is as important as aggregate metrics. For example: on which test samples does RepSim rank badly? Do failure cases cluster by sample type?
- **Suggested fix**: Add a qualitative analysis subsection (possibly in S4.3 after the ablation): "For each task, we examine the 10 test samples with the largest discrepancy between RepSim and TRAK rankings to understand whether failures are systematic."

### [Minor] RepSim layer selection heuristic (L/2 and L) is not well-motivated
- **Location**: Section 4.1 (Implementation details)
- **Problem**: "For RepSim, we extract hidden representations at layer l = L/2 (middle) and l = L (last), reporting the better-performing layer per task." This is a reasonable heuristic but reporting the "better-performing layer" introduces a degree of freedom that inflates RepSim's performance. RepT's automatic phase-transition detection is more principled.
- **Suggested fix**: (1) Report results for both layers, not just the better one. (2) Add a layer sweep analysis (all layers) as a secondary experiment to characterize where attribution information resides. (3) Alternatively, adopt RepT's phase-transition detection for RepSim's layer selection.

### [Minor] BM25 inclusion is good but the paper doesn't fully exploit it
- **Location**: Section 4.1, 4.2
- **Problem**: BM25 (lexical baseline) is included, which is commendable. However, the paper does not discuss what it means if BM25 outperforms neural methods on a task. DATE-LM suggests BM25 is competitive on factual attribution. If BM25 beats RepSim on factual attribution, this undermines the representation-space narrative and suggests the task has strong lexical signal that doesn't require neural attribution.
- **Suggested fix**: Explicitly discuss the BM25 comparison as a diagnostic: "If BM25 outperforms neural TDA methods on a task, it indicates that task-relevant influence is primarily lexical, reducing the relevance of both FM1 and FM2 bottlenecks."

## Strengths
- The 2x2 factorial design is elegant and provides clear, interpretable main effects. It is the right tool for the research question.
- The five-experiment progression (landscape -> decomposition -> generality -> upper bound -> scale) is logical and builds incrementally.
- The fair comparison protocol (same checkpoint, same pipeline, cosine normalization) addresses a common criticism of TDA benchmarks.
- Experiment 3 (LoRA vs. Full-FT) directly addresses the most important open question about FM1 generality.

## Summary Recommendations
The experimental design is strong overall. The three main improvements needed are: (1) expand the method coverage to support the "systematic" claim (or narrow the claim); (2) address statistical power for the interaction term (more seeds or better justification of per-sample tests); (3) rethink the MAGIC experiment's role -- either make it feasible or shrink its paper footprint. The secondary recommendations (failure case analysis, layer selection, BM25 interpretation) would elevate the paper from good to excellent.
