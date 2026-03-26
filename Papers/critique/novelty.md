# Novelty Critique Report

## Overall Assessment
- **Score**: 6.5 / 10
- **Core Assessment**: The paper provides a useful organizational lens (three-bottleneck decomposition) for understanding TDA failure at LLM scale, and the 2x2 factorial ablation design is a clean diagnostic tool. However, the novelty is primarily conceptual organization of known phenomena rather than fundamentally new insight; the three bottlenecks (Hessian error, signal dilution, common contamination) are individually well-documented in prior work. The contribution hinges on whether the *independence* of these bottlenecks is empirically validated, which remains pending.
- **Would this dimension cause reject at a top venue?**: Borderline. A skeptical reviewer could argue that the framework is a repackaging of known problems. The paper needs the empirical validation (2x2 interaction results) to elevate beyond "useful taxonomy" to "validated diagnostic framework." The experimental design itself (2x2 ablation) is the strongest novel element.

## Issues (by severity)

### [Major] Three bottlenecks are individually well-known; the "decomposition" is the only novel conceptual element
- **Location**: Section 3.2 (Three-Bottleneck Framework), Introduction P3
- **Problem**: Each bottleneck is directly sourced from a single prior paper: Hessian error from MAGIC/Better Hessians Matter, FM1 from Li et al., FM2 from DDA. The paper explicitly acknowledges this: "Each existing method implicitly addresses at most one of these bottlenecks without recognizing the others." But recognizing that three known problems are distinct is a lower bar of novelty than discovering a new problem or mechanism.
- **Simulated reviewer phrasing**: "The three bottlenecks identified are not new -- Hessian error, high-dimensional gradient dilution, and pre-training bias are well-understood limitations. The claim of novelty rests on their 'independence,' but the paper provides no theoretical proof of independence; it only tests this empirically through a 2x2 ablation. If the interaction term turns out to be large, the entire framework collapses."
- **Suggested fix**: Strengthen the novelty claim by (1) providing at least a semi-formal argument for why the three bottlenecks should be independent (beyond the current paragraph in S3.2), (2) emphasizing that the diagnostic *methodology* (factorial ablation for TDA) is novel, not just the decomposition, and (3) explicitly comparing with the closest organizational framework -- "Towards Unified Attribution" (2501.18887) -- showing what CRA's lens enables that theirs does not.

### [Major] The "unified bilinear taxonomy" (C1 in contribution.md) is not developed in the paper
- **Location**: Section 2.2, Section 3.3
- **Problem**: The contribution tracker lists "Unified bilinear taxonomy: 5 representation-space TDA methods as phi^T psi instances" as C1, and the Related Work section mentions that methods "share a rough bilinear structure." However, this taxonomy is not developed into a formal contribution in the paper -- it is a brief observation in two sentences. The paper even hedges: "the analogy is imperfect: Concept IF projects back to parameter space... and AirRep operates in a learned space." If two of five methods don't fit, this is not a unification.
- **Simulated reviewer phrasing**: "The bilinear taxonomy is mentioned in passing but never leveraged analytically. Either develop it into a formal framework with clear inclusion/exclusion criteria, or remove it as a claimed contribution."
- **Suggested fix**: Either (a) develop the bilinear taxonomy into a proper formal subsection with a table mapping each method to its phi/psi functions, clearly noting which methods are exact fits and which are approximations, or (b) demote it from a contribution to a brief organizational remark in Related Work.

### [Major] Signal-processing analogies are suggestive but may not withstand scrutiny
- **Location**: Section 3.2 (paragraphs on "matched filtering" and "differential detection")
- **Problem**: The paper uses matched filtering and differential detection as analogies for representation-space and contrastive attribution. While pedagogically useful, these analogies may invite criticism: matched filtering assumes a known signal template and linear projection, while h^(l) is a highly nonlinear learned transformation. The paper acknowledges this caveat in method-design.md S7.2 ("This analogy is suggestive, not rigorous") but the paper text presents it more assertively: "Moving from parameter space to representation space is analogous to matched filtering" and "These are complementary signal-processing operations with over 70 years of theoretical grounding for their approximate orthogonality."
- **Simulated reviewer phrasing**: "The claim that FM1/FM2 repairs are 'approximately orthogonal' based on signal-processing analogies is hand-wavy. The theoretical grounding for orthogonality of matched filtering and differential detection applies to linear systems; neural network representations are neither linear nor guaranteed to preserve the signal subspace relevant to influence."
- **Suggested fix**: Tone down the signal-processing language. Present it explicitly as "a guiding analogy (not a formal result)" and remove the "70 years of theoretical grounding" claim, which implies a formal connection that doesn't exist. The 2x2 ablation is the actual test of independence; the analogy should motivate the design, not substitute for validation.

### [Minor] Differentiation from "Better Hessians Matter" could be sharper
- **Location**: Section 2.1, Section 3.2
- **Problem**: Better Hessians Matter shows a clear quality ordering of Hessian approximations but "at scales below 1M parameters where FM1 and FM2 may be mild." CRA claims to extend this to LLM scale. However, the paper does not explicitly test what happens when you combine the best Hessian (from Better Hessians Matter) with representation-space methods. The positioning would be cleaner if CRA addressed: "Even with the best possible Hessian approximation, FM1 and FM2 remain -- that is our added insight."
- **Suggested fix**: Add a sentence in S3.2 or S2.1 explicitly stating: "Our framework predicts that even using the highest-quality Hessian approximation (per [Better Hessians Matter]), FM1 and FM2 would persist as independent bottlenecks at LLM scale."

### [Minor] Missing discussion of concurrent/recent representation-space TDA work
- **Location**: Section 2.2, end paragraph
- **Problem**: The paper itself flags this: "Directions needing supplementary reading: Recent concurrent work on representation-space TDA submitted to ICML/NeurIPS 2026 (post-arXiv, if any)." This self-aware note should not appear in the final paper. More importantly, the field is moving fast -- any representation-space TDA work appearing between now and submission could directly overlap with C2.
- **Suggested fix**: Before submission, conduct a thorough literature sweep of arXiv for representation-space TDA papers from 2025-2026. Remove the "supplementary reading" note, which signals incomplete scholarship.

## Strengths
- The 2x2 factorial ablation design is genuinely novel in the TDA literature and provides a clean, interpretable diagnostic methodology that could become standard practice.
- The paper is refreshingly honest about what it does NOT contribute -- "CRA is NOT proposing a new TDA method" -- and positions itself clearly as a diagnostic framework.
- The identification of task-dependent bottleneck profiles (FM2 dominant on toxicity, FM1 on data selection) would be a valuable practical contribution if validated.

## Summary Recommendations
The novelty is adequate for a diagnostic/analysis paper at NeurIPS, but it sits at the borderline. The paper's strength is the experimental *design* (2x2 ablation), not the conceptual framework (which reorganizes known problems). To strengthen novelty: (1) sharpen the independence argument beyond analogy, (2) either properly develop or drop the bilinear taxonomy, (3) ensure the literature review catches all concurrent work. The paper should lean harder into the "first systematic benchmark" angle (C1/C2) since this is the contribution most immune to novelty criticism -- nobody has done this comparison, and the field needs it.
