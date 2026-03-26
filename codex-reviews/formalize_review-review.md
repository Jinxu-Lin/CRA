# External AI Review — RS (Strategic Review) — CRA (Contrastive Representation Attribution)

**Reviewer**: Independent External AI (Codex)
**Date**: 2026-03-25
**Artifact reviewed**: `research/problem-statement.md` v1.1 + `project.md` v1.0

---

## Overall Impression

This is a well-structured diagnostic framing of a real and timely problem -- why parameter-space TDA fails at LLM scale -- with an honest enumeration of risks that is rare in research proposals. The three-bottleneck decomposition (Hessian error, FM1, FM2) is the core intellectual contribution and is genuinely clarifying. However, the project's biggest vulnerability is that its strongest claims rest on experiments that have not been run, and the theoretical scaffolding (signal-processing analogy, bilinear unification) is more suggestive than it is rigorous, leaving the paper dangerously dependent on empirical outcomes it cannot yet predict.

---

## Blind Spot Report

- **Blind spot 1: The "three bottleneck" decomposition assumes clean separability that may not exist.** The entire RQ1 operationalization assumes you can read off bottleneck contributions from gaps between method tiers (approximate IF -> exact IF -> RepSim -> contrastive). But each method differs in *multiple* dimensions simultaneously. MAGIC differs from TRAK not only in Hessian quality but also in requiring deterministic training, different checkpointing, and different gradient computation. RepSim differs from MAGIC not only in "bypassing FM1" but also in discarding all Hessian information entirely. The attribution of LDS gaps to specific bottlenecks requires strong ceteris paribus assumptions that no experimental design in the document actually enforces. You are not running controlled ablations on a single method -- you are comparing fundamentally different methods and attributing the delta to one factor. This is the kind of reasoning error that immersed researchers make because the decomposition feels clean on paper.

- **Blind spot 2: The signal-processing analogy may be actively misleading, not just "suggestive."** Matched filtering assumes a known signal template and additive Gaussian noise. Neither condition holds for neural network gradients. The "70+ years of orthogonality theory" is impressive as rhetoric but the mapping from signal processing to TDA has no formal justification in the document. There is a risk that reviewers at NeurIPS will see this as false rigor -- borrowing the prestige of a mature field without doing the mathematical work to justify the mapping. If a reviewer asks "prove that representation-space operation is equivalent to matched filtering in any formal sense," the current document has no answer.

- **Blind spot 3: The paper has no compelling story if MAGIC works well.** Section 1.3 acknowledges the MAGIC tension but treats it as a complication to "address experimentally." In reality, if MAGIC achieves LDS > 0.90 on Pythia-1B DATE-LM tasks, the entire FM1 narrative collapses because you cannot claim "parameter space has inherent signal dilution" when a parameter-space method with exact computation succeeds. The fallback ("MAGIC's deterministic training implicitly reduces FM1") is an unfalsifiable post-hoc rationalization. A rival lab reading this document would immediately run MAGIC on DATE-LM and, if it works, publish a rebuttal that invalidates CRA's premise before CRA is even submitted.

- **Blind spot 4: The "five methods as a family" framing overstates coherence.** The document itself notes that Concept IF projects back to parameter space, AirRep learns a new encoder, and the bilinear form is "structural observation, not deep theoretical unification." Yet the project still frames this as discovering an unrecognized family. A skeptical reviewer will ask: if three of five methods don't cleanly fit the bilinear template, what exactly is being unified? The risk is that the "family" claim reads as taxonomy-by-assertion rather than taxonomy-by-insight.

---

## Strengths

- **Honest risk enumeration.** The assumptions table (H1-H5) with explicit "If False" columns is unusually rigorous for a problem statement. The acknowledgment that H4 has "None" evidence strength and that the probe has NOT been executed is exactly the kind of intellectual honesty that builds reviewer trust.

- **Well-designed probe experiment.** The pass/fail criteria with graduated thresholds (strong pass / pass / weak pass / fail) and the failure diagnosis plan (checking absolute LDS, layer sensitivity, Spearman correlation) show methodological maturity. This is not a "run it and see" probe.

- **The 2x2 ablation is genuinely information-dense.** Any outcome of {parameter-space, representation-space} x {standard, contrastive} produces an interpretable and publishable result. The ANOVA framework for testing FM1-FM2 independence (RQ3) is clean.

- **Explicit falsification criteria for each RQ.** RQ1's "if MAGIC within 3pp of RepSim, FM1 thesis fails" is a concrete, pre-registered prediction. This is how research should be designed.

---

## Weaknesses / Concerns

- **RQ1's operationalization conflates method differences with bottleneck isolation.** The gap between TRAK and MAGIC is attributed to "Hessian error," but MAGIC also uses deterministic training, metagradient computation, and operates on fine-tuning only. Multiple confounds are present. A controlled experiment would fix the method and vary only the Hessian quality (as Better Hessians Matter actually does), but the document proposes comparing across fundamentally different methods.

- **The "three bottleneck" count is fragile.** If FM1 turns out to be a LoRA artifact (acknowledged as possible), and if contrastive scoring doesn't generalize beyond hallucination tracing (H2 is weak-medium), you are left with "Hessian error is the main bottleneck" -- which is exactly what Better Hessians Matter already said. The document does not have a compelling fallback narrative for this scenario.

- **Concurrent competition risk is underweighted.** Five methods in 12 months, DATE-LM just published at NeurIPS 2025, and the obvious next step (benchmark representation-space methods on DATE-LM) requires no deep insight. Any lab with GPU access could do this in 2-3 weeks. The document lists this as risk #5 (lowest priority), but it should arguably be #2, right after the probe.

- **The "ceiling risk" (Section 2.1) is dismissed too quickly.** "Pure diagnostic + benchmark is poster-level at best" is stated but then not addressed. The document mentions no novel method contribution beyond taxonomy and evaluation. At NeurIPS 2026, a paper that says "we benchmarked five existing methods and found X" needs an extremely compelling X. The diagnostic framework itself must be demonstrably useful (e.g., it predicts which method works best for a new task), not just descriptive.

- **LDS metric concerns (H-RF1, H-DVEmb3) are mentioned but not integrated.** If LDS is unreliable, every quantitative claim in the paper is compromised. The document should specify what alternative metrics will be used alongside LDS (P@K, AUC, leave-one-out retraining) and how disagreements between metrics will be interpreted.

---

## Simpler Alternative Challenge

**A simpler problem formulation**: Drop the three-bottleneck decomposition entirely. Instead, frame the paper as: "First systematic evaluation of representation-space TDA methods on DATE-LM" -- a pure benchmark contribution. Run RepSim, RepT, AirRep, TRAK, DDA on all three DATE-LM tasks at Pythia-1B and Llama-7B scale. Report results. No theoretical framework needed.

This achieves ~80% of the citation value (practitioners want the benchmark numbers, not the signal-processing analogy) with ~30% of the intellectual risk (no FM1/FM2 thesis to defend, no MAGIC tension to resolve, no 2x2 ANOVA to interpret). The reason NOT to do this is that it caps the paper at a "datasets and benchmarks" track acceptance rather than a main track oral. But the current proposal may also cap at poster level if the theoretical claims don't hold up empirically. The expected value calculation is closer than the document implies.

---

## Specific Recommendations

1. **Run the probe IMMEDIATELY.** Every day without probe results is a day of planning that may be invalidated. The 2.5-day estimate is reasonable. Nothing else should happen until RepSim LDS numbers on DATE-LM exist.

2. **Add MAGIC to the probe, not just the full evaluation.** If MAGIC on Pythia-1B DATE-LM achieves LDS > 0.90, you need to know NOW, not after 4 weeks of framework development. This changes the entire project direction.

3. **Develop a concrete fallback narrative for "FM1 is negligible."** If MAGIC works and RepSim is competitive but not superior, the paper becomes "Representation-space methods are a computationally cheaper alternative to exact IF, and contrastive scoring (FM2) is the real differentiator." Write this alternative framing now so you're not scrambling later.

4. **Strengthen or drop the signal-processing analogy.** Either prove a formal equivalence (representation-space TDA as matched filtering with specific noise model and signal template) or demote it to a one-paragraph intuition in the introduction. Currently it occupies a middle ground that invites criticism.

5. **Add at least one novel method contribution.** Even a simple one: RepSim + contrastive scoring (subtract base-model RepSim scores, analogous to DDA's debias but in representation space). This would be the "CRA method" and would give the paper a concrete artifact beyond taxonomy and benchmarking.

6. **Pre-register predictions for each DATE-LM task.** Before running experiments, write down: "We predict RepSim LDS = X +/- Y on toxicity filtering, Z +/- W on data selection, etc." This converts any outcome into evidence about the framework's predictive power, even negative results.

7. **Address LDS metric concerns explicitly in the experiment design.** Report LDS, P@K, and Spearman rank correlation for every method-task pair. If LDS and P@K disagree, this is itself a finding (the "correlation vs causation gap" from H-IF-LLM4).

---

## Score

**6 / 10** -- The problem is real, the framing is intellectually honest, and the experimental design is sound in principle. But the paper's fate hinges entirely on unverified empirical predictions (probe not run), the three-bottleneck decomposition lacks controlled experiments to support causal claims, and the absence of a novel method contribution caps the impact ceiling. At NeurIPS 2026, this is borderline: a strong benchmark result could push it to 7+, but a messy result (MAGIC works, FM1 is negligible, contrastive scoring is task-dependent) would leave it as a workshop paper. The project urgently needs probe data before further theoretical investment.
