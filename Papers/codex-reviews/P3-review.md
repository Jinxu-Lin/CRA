# External AI Review --- P3 --- CRA (Three-Bottleneck Diagnostic Framework for LLM TDA)

## Overall Impression

This paper proposes a clean diagnostic decomposition of LLM-scale training data attribution (TDA) failure into three bottlenecks (Hessian error, signal dilution FM1, common influence contamination FM2) and validates the decomposition via a 2x2 factorial ablation on DATE-LM. The framework is intellectually appealing: it imposes structure on a fragmented literature and maps each bottleneck to a repair mechanism. The biggest concern is that the entire empirical story is currently PENDING -- every table is placeholder -- meaning the framework rests entirely on theoretical plausibility and indirect evidence from prior work. If the ablation interaction terms turn out large, the paper's central thesis collapses.

## Blind Spot Report

- **Blind spot 1: The "independence" argument is unfalsifiable as stated.** The 30% threshold for interaction magnitude ($|\Xi| < 30\%$ of min main effect) is arbitrary and conveniently chosen. There is no principled justification for why 30% constitutes "approximate independence" vs. 40% or 20%. A hostile reviewer will note that the authors can always adjust the threshold post-hoc to match results. The paper needs either a pre-registered threshold with theoretical grounding or an honest acknowledgment that the threshold is a judgment call.

- **Blind spot 2: Representation space is not a single thing -- layer choice is a hidden confound.** The paper treats "representation space" as a unified concept, but RepSim at layer L/2 vs. layer L can yield dramatically different attribution scores. The choice to report "the better-performing layer per task" (Section 4.1) is a form of oracle selection that inflates representation-space results. In a real deployment, the practitioner does not know the best layer in advance. This selection bias is never discussed and could account for a significant portion of the RepSim advantage.

- **Blind spot 3: The signal-processing analogy (matched filtering / differential detection) is suggestive but misleading.** In classical signal processing, matched filtering is optimal for known signal in Gaussian noise -- conditions that do not hold for neural network representations. The analogy provides intuitive appeal but may give the reader false confidence that the decomposition has stronger theoretical backing than it actually does. The paper never states the conditions under which the analogy breaks.

- **Blind spot 4: Confounding between "FM1 repair" and "different similarity notion."** Moving from parameter-space gradients to representation-space cosine similarity changes two things simultaneously: (1) the dimensionality (FM1 repair), and (2) the semantic content being compared (gradient directions vs. learned features). The paper attributes the entire improvement to dimensionality reduction (FM1), but some (possibly most) of the gain may come from representations being a semantically richer similarity signal. The 2x2 ablation cannot distinguish these two explanations because they are bundled in the same factor.

- **Blind spot 5: LDS as primary metric is a single point of failure.** The paper acknowledges LDS reliability concerns in Limitations (5.3.6) but treats LDS as the sole primary metric across all experiments. If LDS is unreliable, every quantitative claim in the paper is invalidated. The secondary metrics (AUPRC, Recall@50) should be given equal weight, not relegated to supplementary status.

## Strengths

- **Clean experimental design.** The 2x2 factorial ablation is the right diagnostic tool for this question. It is simple, interpretable, and directly tests the independence claim. The extension to 2x2x2 with LoRA vs. Full-FT is well-motivated.

- **Honest limitations section.** The paper is unusually forthcoming about its weaknesses (Limitation 1: probe not executed; Limitation 2: FM1 evidence is LoRA-based; Limitation 4: bilinear taxonomy is organizational). This builds credibility.

- **Strong literature integration.** The related work section genuinely organizes a fragmented field. The observation that five representation-space methods were proposed independently within 12 months is a compelling motivation for a unifying framework.

- **Clear contribution scoping.** C0-C3 are well-delineated and testable. The paper does not overclaim.

## Weaknesses / Concerns

- **No empirical results exist yet.** Every table contains PENDING placeholders. This is acknowledged as PLACEHOLDER mode, but the structural consequence is severe: the paper is currently a well-argued hypothesis, not a validated contribution. The "expected" ranges in placeholders reveal strong prior beliefs -- if results fall outside these ranges, significant rewriting will be needed, and the paper's narrative may not survive.

- **Section 3.2 FM1 argument has a circularity risk.** The FM1 argument uses JL concentration to claim gradients are approximately orthogonal. But JL applies to random vectors; fine-tuning gradients are highly structured (they live on the loss landscape). The paper cites Li et al.'s iHVP degeneracy as empirical evidence, but that evidence is specific to LoRA. The theoretical argument (JL) does not apply, and the empirical argument (Li et al.) is narrow. A reviewer will attack this gap.

- **MAGIC comparison (Experiment 4) is likely infeasible, and the paper admits it.** If MAGIC cannot run, the Hessian error bottleneck remains unquantified. This means only 2 of the 3 bottlenecks are empirically validated, yet the paper's title and framing center on a *three*-bottleneck framework. The framing should be more cautious: "We empirically validate two of three bottlenecks and bound the third indirectly."

- **Contrastive scoring definition (Eq. 5) assumes a clean base model reference.** For tasks where the base model already has relevant knowledge (e.g., factual attribution, as the paper itself notes), the contrastive subtraction may remove task-relevant signal, not just common-mode contamination. This is acknowledged in one sentence but not addressed experimentally. An ablation showing contrastive scoring *hurting* performance on factual attribution would be very informative.

- **Missing baseline: random projection of gradients to $\mathbb{R}^d$.** If the FM1 story is about dimensionality, then randomly projecting parameter-space gradients from $\mathbb{R}^B$ to $\mathbb{R}^d$ (matching representation dimensionality) should also fix FM1. This is essentially what TRAK does with random projection. If TRAK's random projection to dimension $d \sim 10^3$ does not match RepSim, then the gain is not purely dimensionality but rather the *semantic structure* of representations -- which undermines the FM1-as-dimensionality narrative. This control is conspicuously absent.

- **RepT's automatic layer detection is a free variable.** RepT selects layers via "gradient norm discontinuity," which is a heuristic. The paper should report which layers are selected and whether they differ across tasks. If RepT selects the same layer as manually-tuned RepSim, the automatic detection adds no value. If different, the comparison between RepSim and RepT is confounded by layer choice.

- **Scale-up experiment (4.6) only uses LoRA on Llama-7B.** This means the LoRA vs. Full-FT comparison (Experiment 3) is only at 1B scale, while the scale-up experiment only uses LoRA. The interaction between model scale and fine-tuning mode is never tested, which is the most interesting regime (does FM1 get worse at 7B full-FT?).

## Simpler Alternative Challenge

A simpler framing that captures ~80% of this paper's value: "Representation-space TDA methods outperform parameter-space methods on LLM fine-tuning tasks. We benchmark them on DATE-LM and show when contrastive scoring helps." This is a straightforward empirical contribution (C1) that does not require the three-bottleneck theoretical apparatus. The value-add of the framework (C0) is that it *explains* why -- but since the explanation rests on a questionable JL argument (FM1) and an unverifiable Hessian error bound (Bottleneck 1), the framework may be more scaffolding than substance. The paper should ensure the empirical contribution (C1, C2) stands on its own even if the theoretical framing is disputed.

## Specific Recommendations

1. **Add the random-projection control.** Project parameter-space gradients to $\mathbb{R}^d$ via random projection (or PCA) and include this as a 2x2 ablation cell. If it matches RepSim, FM1 is pure dimensionality. If not, the paper must disentangle dimensionality from semantic structure.

2. **Pre-register or justify the 30% interaction threshold.** Either derive it from statistical power analysis (given 3 seeds, what interaction magnitude is detectable?) or reframe: report the interaction magnitude and let readers judge, rather than imposing a binary pass/fail criterion.

3. **Promote secondary metrics.** Report AUPRC, Recall@50, and P@K alongside LDS in Table 2 (the ablation), not just Table 1. If the bottleneck decomposition holds under LDS but not under AUPRC, that is a critical finding.

4. **Discuss the layer-selection confound explicitly.** Report results for multiple layers (not just the best) in at least one task. Show a sensitivity analysis: how much does RepSim LDS vary with layer choice?

5. **Strengthen the FM1 theoretical argument.** The JL argument applies to random vectors; gradients are structured. Either provide a tighter bound that accounts for gradient structure (e.g., citing effective dimensionality of the gradient subspace) or weaken the theoretical claim and let the empirical result speak.

6. **Add a "contrastive scoring hurts" analysis.** On factual attribution, where the base model has relevant knowledge, show whether RepSim-C < RepSim. If so, this is a strong validation of the framework (FM2 correction is task-dependent). If not, the framework's prediction is wrong and should be discussed.

7. **Address the "Section 2 before Section 3" ordering.** The introduction says Section 3 presents the framework and Section 2 reviews related work, but the related work opens with "Having established the three-bottleneck diagnostic framework." This presupposes Section 3 content. Either reorder (3 before 2) or rewrite the related work opening.

## Score

**5 / 10** --- Borderline. The framework is intellectually clean and the experimental design is sound, but no experiments have been executed. The theoretical grounding for FM1 has a gap (JL on structured gradients), and the missing random-projection control leaves the dimensionality-vs-semantics question unresolved. If experiments validate the predictions (small interaction, task-dependent profiles, FM1 scales with model size), this could rise to 7+. In its current PLACEHOLDER state, no top venue would accept it.
