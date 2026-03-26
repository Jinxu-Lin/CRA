# External AI Review — RT (Technical Review) — CRA

## Overall Impression

This is a well-structured diagnostic framework with a clearly articulated 2x2 ablation as its experimental core. The strongest aspect is the intellectual honesty: the authors explicitly enumerate conditions under which each hypothesis would be falsified, and pre-commit to pivot narratives. The biggest concern is that the entire edifice rests on an unexecuted probe -- Hypothesis H4 (RepSim competitive on LDS) has zero empirical support, and the 60 GPU-day plan is speculative until that 2 GPU-day gate passes.

## Blind Spot Report

- **Blind spot 1: The "representation space bypasses FM1" argument conflates dimensionality with information content.** The method design frames FM1 as a dimensionality problem (R^B vs R^d, d << B), then claims representation space "fixes" FM1 by operating in R^d. But TRAK already projects gradients to low dimension (random projection to ~4096). If pure dimensionality were the issue, TRAK should work. The real difference is that h^(l) encodes *learned* structure while TRAK uses *random* projections -- but the authors never directly test this distinction. A missing control: TRAK with PCA-projected gradients (structured projection, same target dimension d). If PCA-TRAK matches RepSim, the story is about projection quality, not "parameter vs representation space." This is a simpler explanation the authors cannot see because they have anchored on the space dichotomy.

- **Blind spot 2: Contrastive scoring subtraction assumes the base model and fine-tuned model share a representation geometry close enough for subtraction to be meaningful.** For RepSim-C, the formula cos(h_ft, h_ft) - cos(h_base, h_base) assumes that the cosine similarities in the two spaces are on comparable scales and that their difference isolates fine-tuning-specific influence. But fine-tuning can rotate the representation space substantially (especially full-FT at scale). If the representation geometry shifts enough, the subtraction is not "removing common mode" -- it is subtracting two incommensurable quantities. The authors acknowledge this risk for data selection ("less natural") but not the more fundamental geometric issue. No experiment checks whether base and fine-tuned representations are geometrically aligned (e.g., CKA or Procrustes similarity between h_base and h_ft).

- **Blind spot 3: The 2x2 ablation uses TRAK as "parameter-space" representative, but TRAK is already a lossy approximation.** The 2x2 design claims to test {parameter-space, representation-space} x {standard, contrastive}. But the parameter-space arm is TRAK (random projection + diagonal Hessian approximation), not raw gradient similarity. This means "parameter-space" cell performance includes both FM1 AND Hessian approximation error. The 2x2 cannot cleanly isolate FM1 because the parameter-space arm conflates two of the three bottlenecks. The FM1 main effect = mean(repr) - mean(param) actually measures FM1 + partial Hessian error, not FM1 alone. The authors are aware of the Hessian confound in principle (Experiment 4 with MAGIC), but MAGIC is acknowledged as "likely infeasible." Without MAGIC, the 2x2 does not decompose what it claims to decompose.

- **Blind spot 4: LDS as a metric may systematically favor representation-space methods for reasons unrelated to attribution quality.** LDS measures Spearman correlation between predicted and actual output changes when training subsets are removed. But representation similarity is inherently correlated with output similarity (similar representations produce similar outputs). This means RepSim scores are partially measuring "how similar are outputs" rather than "what caused this output" -- and LDS, which also measures output changes, may reward this correlation. The metric and the method share a confound (output similarity), creating a circularity risk that the experiment design does not address.

## Strengths

- **Explicit falsification criteria for every hypothesis.** H4's probe gate, MAGIC invalidation decision rule with concrete LDS thresholds, 2x2 interaction interpretation guide with 10%/30% boundaries -- these are unusually rigorous for a design document. The authors have clearly thought about what "failure" means at each stage.

- **Phased compute allocation with hard gates.** The probe (2 GPU-days) -> pilot (3 GPU-days) -> full experiments (55 GPU-days) structure with explicit pass/adjust/fail criteria at each gate is excellent resource management. The total 60 GPU-day budget within 180 GPU-days available is conservative and appropriate.

- **Component D (LoRA vs Full-FT) addresses the most important open question.** Li et al.'s FM1 evidence is entirely LoRA-based. Testing full-FT is the single most informative experiment in this plan, and the authors correctly elevate it from "secondary ablation" to core experiment.

- **Honest treatment of MAGIC tension.** Rather than dismissing MAGIC's LDS ~0.95-0.99, the authors explicitly identify it as the primary challenge to the FM1 thesis and pre-commit to pivot narratives for each outcome range.

- **Comprehensive failure mode table with probabilities.** Section 10.2 of experiment-design.md lists 6 failure modes with estimated probabilities and specific responses. This is the mark of a mature research design.

## Weaknesses / Concerns

- **The 2x2 ablation's "FM1 main effect" is confounded with Hessian approximation error (as noted in Blind Spot 3).** Method-design.md Section 6 claims the 2x2 validates FM1 via "row comparison (repr vs param)," but the param arm uses TRAK, which suffers from both FM1 AND Hessian error. To cleanly measure FM1, the param arm should use Grad-Sim (no Hessian approximation, just raw gradient cosine similarity). Alternatively, include Grad-Sim as a third level: {Grad-Sim (param, no Hessian), TRAK (param, with Hessian), RepSim (repr)}. This 3-level design separates Hessian from FM1 within parameter space.

- **RepT reimplementation risk is underestimated.** The experiment design budgets time for "reimplementation" of RepT, but RepT's phase-transition layer detection is non-trivial (requires computing gradient norms across all layers and identifying a "sharp change"). The authors plan to implement this from the paper description. In practice, phase-transition detection can be sensitive to smoothing, threshold selection, and model architecture. If RepT's auto-detection fails on Pythia-1B, the fallback (manually selecting a layer) undermines RepT's claimed advantage over RepSim.

- **Statistical power concern: 3 seeds with ~100 test samples.** The authors estimate detecting LDS differences of ~3-5pp at alpha=0.05, power=0.80. But the 2x2 interaction term -- the key test of FM1-FM2 independence -- requires detecting a difference-of-differences, which has roughly sqrt(2) higher variance than a simple difference. The minimum detectable interaction is thus ~4-7pp, which may be too coarse to distinguish "clean independence" (< 10% of main effect) from "moderate interaction" (10-30%). Consider whether 5 seeds is feasible for the 2x2 ablation specifically.

- **MAGIC feasibility analysis underestimates disk requirements.** The experiment design estimates 200 checkpoints x 2GB = 400GB for model checkpoints, then separately notes that with optimizer state this becomes 200 x 8GB = 1.6TB. But MAGIC requires the full training trajectory including all parameter states, not just periodic checkpoints. If gradient checkpointing is used (recomputing intermediates), the time cost per test sample increases further. The "5 GPU-days" budget appears optimistic; "likely infeasible" assessment in Section 10.1 is more realistic.

- **DDA reimplementation as "TRAK_ft - TRAK_base" may not faithfully reproduce DDA.** DDA's debiasing involves specific techniques beyond simple subtraction (normalization, denoising). The experiment design acknowledges DDA's full approach but implements a simplified version. If this simplified version underperforms, it is unclear whether FM2 correction is weak or the implementation is incomplete. Consider contacting DDA authors for code or implementing the full pipeline.

- **No validation that DATE-LM's toxicity filtering actually exhibits high FM2.** The design assumes toxicity filtering is a "high FM2 task" based on DDA's hallucination tracing results and the logic that pre-training language patterns contaminate toxicity attribution. But this is untested. If toxicity filtering on DATE-LM happens to have low FM2 (e.g., because the toxic examples are stylistically distinctive), the 2x2's FM2 column will show null effects, and the interpretation becomes ambiguous.

## Simpler Alternative Challenge

A substantially simpler study could achieve ~80% of CRA's contribution:

**Simpler method**: Skip the diagnostic framework entirely. Run RepSim, RepT, TRAK, and Grad-Sim on all three DATE-LM tasks with 3 seeds. Report results. This alone fills the acknowledged benchmark gap (G-RepT4, G-AR2) and would be the first comparative evaluation of representation-space methods on DATE-LM.

**Why this captures most of the value**: The benchmark results are the concrete deliverable that practitioners will actually use. The "three bottleneck" framework is intellectually satisfying but the 2x2 ablation is confounded (Blind Spot 3), and the LoRA-vs-Full-FT experiment is the only part that cleanly tests a novel hypothesis. A benchmark paper with a LoRA-vs-Full-FT ablation as the "insight" section would be simpler, less risky, and still novel.

**What you lose**: The diagnostic decomposition narrative, which is the paper's claim to being more than a benchmark. But if the 2x2 is confounded and MAGIC is infeasible, the decomposition is incomplete anyway. Better to have a clean benchmark than an ambitious-but-incomplete diagnostic.

**Counter-argument (why the full design is still worthwhile)**: The 2x2, even if imperfect, provides more structured evidence than uncontrolled comparisons. And the "three bottleneck" framing organizes thinking about TDA failure even if the quantitative decomposition is approximate.

## Specific Recommendations

1. **Add Grad-Sim as a second parameter-space arm in the 2x2 ablation.** This separates Hessian approximation error from FM1 within parameter space. Compute cost is negligible (Grad-Sim is already in Experiment 1). The 2x2 becomes a 3x2: {Grad-Sim, TRAK, RepSim} x {standard, contrastive}. The FM1 effect is then RepSim vs Grad-Sim (both without Hessian approximation machinery), and the Hessian effect is Grad-Sim vs TRAK (both in parameter space, with and without Hessian-informed scoring).

2. **Add a representation alignment check between base and fine-tuned models.** Before running contrastive scoring, compute CKA similarity between h_base^(l) and h_ft^(l) at the chosen layers. If CKA < 0.7 (substantial geometric shift), flag that contrastive subtraction is operating across misaligned spaces. This is a 1-hour computation that validates a core assumption.

3. **Increase seeds to 5 for the 2x2 ablation (Experiment 2) only.** The interaction term is the hardest-to-detect effect and the most theoretically important. 36 runs become 60 runs (additional 8 GPU-days), which is within the 5-day buffer.

4. **Pre-register the probe outcome before running full experiments.** Write down the specific LDS values for RepSim and TRAK before running any experiment beyond Exp 0. This prevents post-hoc threshold adjustment and strengthens the paper's credibility.

5. **For the DDA baseline, implement the full DDA pipeline (not just TRAK_ft - TRAK_base).** If DDA code is unavailable, at minimum include a "DDA-simplified" vs "DDA-full" comparison on one task to bound the implementation gap.

6. **Report CKA/Procrustes between RepSim and TRAK score vectors as a sanity check.** If RepSim and TRAK produce highly correlated rankings (Spearman > 0.8) despite different spaces, the "fundamentally different attribution signals" narrative is weakened. If correlation is low, it supports the "complementary information" claim.

7. **Consider adding PCA-projected gradient similarity (PCA-Grad) as a diagnostic baseline in Experiment 1.** Project parameter gradients to d=2048 using PCA (top principal components of the gradient matrix). If PCA-Grad matches RepSim, the advantage is structured dimensionality reduction, not representation space per se. This directly tests Blind Spot 1.

## Score

**6.5 / 10** -- The experimental design is thorough and the intellectual honesty is commendable, but two structural issues prevent a higher score: (1) the core 2x2 ablation confounds FM1 with Hessian error in the parameter-space arm, undermining the primary claimed contribution (bottleneck decomposition); (2) the entire plan rests on an unexecuted probe with the authors themselves rating H4 as having "None" empirical strength. At a top venue, reviewers will note that the "three independent bottlenecks" claim requires MAGIC to disentangle Hessian from FM1, and MAGIC is acknowledged as likely infeasible. The design is borderline: strong enough to produce a useful benchmark paper, but the diagnostic framework's quantitative decomposition is not fully supported by the experimental structure. Addressing Recommendation 1 (Grad-Sim in the 2x2) would substantially strengthen the decomposition claim and could raise this to a 7.
