# External AI Review — P7 (Project Review) — CRA

## Overall Impression

CRA is an ambitious diagnostic paper that decomposes LLM-scale training data attribution (TDA) failure into three bottlenecks (Hessian error, signal dilution FM1, common contamination FM2) and proposes a 2x2 factorial ablation to validate their independence. The conceptual framing is clear and the experimental design is well-structured for a NeurIPS submission. However, the paper's central vulnerability is severe: **every single experimental result is a PENDING placeholder**. This is not a paper; it is a detailed experimental plan with a polished narrative wrapper. No reviewer at any venue would accept a manuscript where 100% of quantitative claims are unverified predictions.

## Blind Spot Report

- **Blind spot 1: The paper is written backward — narrative precedes evidence.** The authors have constructed an elaborate three-bottleneck framework and then designed experiments to confirm it, rather than letting empirical findings shape the framework. This creates a strong confirmation bias risk: the text is structured so that almost any experimental outcome can be rationalized within the framework (see the extensive "failure contingencies" in the research docs). A NeurIPS reviewer will notice that the paper reads as if results were cherry-picked to match the narrative, even though no results exist yet. The honest approach is to present the framework as a hypothesis, run all experiments, and then let the data determine which bottlenecks actually matter — potentially revising the framework post-hoc.

- **Blind spot 2: The "three independent bottlenecks" claim is doing far more theoretical work than the evidence supports.** The independence claim rests on a single 2x2 ablation with a 30% interaction threshold that is acknowledged as arbitrary. In a 2x2 design with only 4 cells and 3 seeds, the power to detect interactions is extremely low. The authors may convince themselves that a non-significant interaction supports independence, when in reality the test simply cannot detect moderate interactions. This is a classic Type II error trap. A statistician reviewer will flag this immediately.

- **Blind spot 3: The signal-processing analogies (matched filtering, differential detection) may mislead more than they illuminate.** These analogies are presented with extensive caveats ("informal," "not rigorous," "conditions don't strictly hold"), yet the entire conceptual vocabulary of the paper is built on them. A hostile reviewer could argue that the analogies are decorative rather than substantive — they do not generate testable predictions that would not already follow from simpler reasoning (e.g., "lower-dimensional space = less noise" does not require the matched filtering framing).

- **Blind spot 4: The paper implicitly assumes representation-space methods are "fixing" FM1, but they may be measuring something entirely different.** RepSim computes cosine similarity of hidden states — this is a measure of functional similarity, not causal influence. High LDS for RepSim would not prove it "fixes signal dilution"; it would prove that functional similarity happens to correlate with counterfactual influence for certain tasks. The paper acknowledges this ("correlation vs. causation caveat" in Section 3.3) but then proceeds to frame the entire 2x2 ablation as if RepSim occupying the "FM1 Fixed" cell is established fact. The cell label is the conclusion, not the premise.

- **Blind spot 5: MAGIC invalidation risk is underweighted in the paper narrative.** The paper devotes extensive discussion to what happens if MAGIC is infeasible, but relatively little to the scenario where MAGIC achieves LDS > 0.90. If exact parameter-space IF works well, the entire FM1 story collapses: the "signal dilution" was just Hessian approximation error all along, and representation-space methods are simply cheap proxies. The problem-statement.md acknowledges this (Section 1.3), but the paper.md buries it. A NeurIPS reviewer who knows the MAGIC paper will immediately ask this question.

## Strengths

- **Well-defined experimental design with falsification criteria.** The 2x2 factorial ablation is a clean, interpretable experimental structure. The explicit falsification criteria (e.g., RepSim LDS < TRAK LDS - 5pp = FM1 thesis fails) demonstrate intellectual honesty rarely seen in ML papers. This is genuinely commendable.

- **Comprehensive related work positioning.** The paper correctly identifies the gap: five representation-space methods proposed independently, never compared on a common benchmark. The DATE-LM benchmark gap for representation-space methods is real and acknowledged by the community.

- **Honest limitations section.** Section 5.3 is unusually candid. Stating "probe not executed prior to full experiments" as limitation #1 is admirable, though it also means the paper should not have been written yet.

- **Practitioner-oriented contribution.** The guidance table (Section 5.2) mapping task type and compute budget to recommended method is a genuinely useful deliverable, assuming the experiments produce clean results.

- **Fair comparison protocol.** Using the same checkpoint, same evaluation pipeline, cosine normalization for all methods, and reporting all hyperparameter settings (not just best) is methodologically rigorous.

## Weaknesses / Concerns

- **No experimental results whatsoever.** Every table contains PENDING placeholders. The paper cannot be evaluated as a research contribution in its current state. The expected value ranges in the PENDING tags (e.g., "0.10-0.25") span factors of 2.5x, meaning the authors themselves have high uncertainty about outcomes.

- **The three-bottleneck decomposition is not as novel as claimed.** The individual bottlenecks are well-known: Hessian error (Better Hessians Matter), signal dilution (Li et al.), common contamination (DDA). The novelty claim is the decomposition itself — putting these three into one framework. But this is primarily organizational/taxonomic, not theoretical. The paper acknowledges the bilinear unification is "organizational, not theoretical" (Limitation 4) but does not extend this honesty to the three-bottleneck framework itself. A reviewer could argue: "You identified three known problems and put them in a 2x2 table. Where is the insight?"

- **Contribution C0 (three-bottleneck framework) is unfalsifiable as stated.** The framework is designed so that any pattern of results can be accommodated: if bottlenecks are independent, the framework is "validated"; if they interact, the framework is "refined to partial separability"; if one bottleneck is absent, the framework is "reduced to two bottlenecks." This is not a scientific framework — it is a classification scheme that cannot be wrong.

- **Statistical power concerns for the 2x2 interaction test.** With 3 seeds and ~100 test samples, the confidence intervals on interaction terms will be wide. The 30% threshold for "approximate independence" is arbitrary and not justified by power analysis. The paper mentions increasing to 5 seeds "if interaction confidence intervals are wide" (Section 4.3), but does not commit to this.

- **The LoRA vs. Full-FT experiment (Experiment 3) has a confound.** LoRA and Full-FT produce different models with different loss surfaces, different generalization properties, and different hidden representations. Comparing RepSim advantage across these two settings conflates FM1 severity with fundamental differences in what the models learned. A cleaner test would vary the LoRA rank (r=4, 16, 64, 256) to trace FM1 as a function of effective dimensionality within the same family.

- **Section 3.2 presents the JL concentration argument as "geometric intuition" but leans on it heavily.** The JL lemma applies to random vectors; gradients are highly structured and correlated. Acknowledging this caveat does not neutralize the problem — the argument simply does not apply. A reviewer trained in high-dimensional geometry will find this unconvincing.

- **The paper does not compare against LESS**, which is a strong task-specific parameter-space baseline that may already address FM1 through task-specific gradient projection. Excluding it weakens the claim that parameter-space methods cannot compete.

- **DATE-LM toxicity filtering has only ~100 unsafe samples in ~10K training samples.** This extreme class imbalance means that attribution methods are evaluated on their ability to find needles in haystacks. LDS may behave very differently in this regime compared to balanced settings, and the authors do not discuss this.

## Simpler Alternative Challenge

A simpler paper with ~80% of the diagnostic value: "RepSim and RepT on DATE-LM: A Comprehensive Benchmark" — run the six methods on all three DATE-LM tasks, report LDS/AUPRC/Recall, include BM25 and Random baselines, add one contrastive variant (RepSim-C), and discuss when representation-space methods win. Drop the three-bottleneck framework, the signal-processing analogies, the MAGIC feasibility study, and the LoRA vs. Full-FT dimension. This removes the most speculative claims and delivers the acknowledged community need (representation-space methods evaluated on DATE-LM) with lower risk and higher chance of clean results. The conceptual framework can be added in a later paper once the empirical foundation exists.

## Specific Recommendations

1. **Run the probe experiment (Exp 0) before writing the paper.** The problem-statement.md correctly identifies this as the critical gate. Writing a full paper before running the gate experiment is premature and will lead to sunk-cost bias in interpreting results.

2. **Replace the 30% interaction threshold with a formal power analysis.** Given 3 seeds and ~100 test samples, compute the minimum detectable interaction effect. If this exceeds a scientifically meaningful threshold, the 2x2 independence claim is untestable at the proposed sample size.

3. **Add LoRA rank sweep (r=4, 16, 64, 256) as a cleaner FM1 test.** This holds the fine-tuning paradigm constant while varying effective dimensionality, avoiding the LoRA-vs-Full-FT confound.

4. **Include LESS as a baseline.** LESS performs task-specific gradient projection, which is an alternative FM1 mitigation within parameter space. Its absence weakens the claim that moving to representation space is necessary.

5. **Weaken the "independence" language.** Instead of testing whether FM1 and FM2 are "independent bottlenecks," frame the ablation as measuring their "relative contributions and interaction magnitude." The independence framing sets up a binary pass/fail that the experimental design cannot reliably adjudicate.

6. **Address the MAGIC invalidation scenario more prominently.** If MAGIC achieves high LDS on DATE-LM at feasible cost, the paper's central premise collapses. This scenario should be discussed in the Introduction alongside the problem statement, not buried in Section 4.5.

7. **Drop the signal-processing analogies or formalize them.** Either derive actual SNR improvement bounds under the matched filtering interpretation (with stated assumptions) or remove the analogies entirely. The current presentation invites attack from reviewers who will point out the analogies are vacuous.

8. **Discuss LDS metric limitations earlier and more substantively.** The paper notes LDS concerns in Limitation 6 (Section 5.3) but builds the entire evaluation on LDS without hedging. If LDS is unreliable, all conclusions are compromised. Add P@K as a co-primary metric and discuss when LDS and P@K diverge.

9. **Fill PENDING placeholders with actual experiments before submission.** This is obvious but must be stated: a NeurIPS submission with PENDING placeholders will be desk-rejected.

10. **Add a "Negative Results" subsection.** If contrastive scoring hurts on factual attribution (as predicted in Section 3.3), report this prominently as evidence that FM2 is task-dependent. Negative results that confirm framework predictions are more convincing than uniformly positive results.

## Score

**4 / 10** — The experimental design is well-conceived and the research questions are relevant, but the paper cannot be scored as a research contribution because it contains zero experimental results. The three-bottleneck framework is primarily organizational rather than theoretically deep, the independence claim is likely untestable at the proposed sample size, and the signal-processing analogies add rhetorical flourish without substance. If the experiments produce clean results matching predictions, this could become a 6-7/10 paper. If results are messy (which the authors' own failure probability estimates suggest is likely), it may settle at 5/10 as a useful-but-incremental benchmark paper. The path from 4 to 7 requires: (a) running all experiments, (b) letting results reshape the narrative rather than fitting results to the pre-written story, and (c) substantially toning down the theoretical framing in favor of empirical honesty.
