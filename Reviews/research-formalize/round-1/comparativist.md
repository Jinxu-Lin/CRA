## [Comparativist] 文献对标者视角

### SOTA 定位

**绝对 SOTA (TDA for LLMs)**:
- **MAGIC** (arXiv:2504.16430, April 2025): LDS ~0.95-0.99 on fine-tuning attribution via metagradient exact IF. Currently the highest reported LDS for any single-model TDA method, though limited to deterministic training and small data-drop fractions.
- **DDA** (arXiv:2410.01285, EMNLP 2024): AUC 91.64% on hallucination tracing, best contrastive-scoring method for LLM TDA. Averaged across LLaMA2, QWEN2, Mistral.
- **RepT** (arXiv:2510.02334): P@10 = 0.97-1.00 on controlled experiments, but P@K metric, not LDS.

**最相近 approach**:
- **"Towards Unified Attribution in Explainable AI, Data-Centric AI, and Mechanistic Interpretability"** (arXiv:2501.18887, January 2026): This position paper explicitly argues for unifying feature attribution, data attribution, and mechanistic interpretability under a common framework. While broader in scope (not TDA-specific), it directly competes with CRA's "unified taxonomy" contribution. CRA must differentiate by being TDA-specific and empirically grounded, not just taxonomic.
- **"Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods"** (arXiv:2412.03906, December 2024): Provides a unified view of gradient-based TDA methods. If it covers the phi^T psi bilinear structure for representation-space methods, CRA's taxonomic novelty is compromised.
- **"A Survey of Data Attribution: Methods, Applications, and Evaluation in the Era of Generative AI"** (September 2025 comprehensive survey): This survey covers TDA methods, applications, and evaluation protocols. CRA's diagnostic framework must provide insight beyond what this survey already offers.

**最强简单 baseline**:
- **BM25 / lexical retrieval**: TrackStar's finding that BM25 can be competitive on factual attribution tasks suggests that for some DATE-LM tasks, no sophisticated TDA method is needed. CRA must demonstrate that representation-space methods beat BM25 on tasks where BM25 is competitive.

**其他关键竞争方法**:
- **AirRep** (arXiv:2505.18513, NeurIPS 2025): Learned representation-space TDA. Reports being "nearly two orders of magnitude more efficient" than gradient-based methods on instruction-tuned LLMs. AirRep is the strongest representation-space competitor and already positions itself as a scalable alternative.
- **Mechanistic Data Attribution (MDA)** (arXiv:2601.21996, January 2026): Uses influence functions to trace training origins of interpretable LLM units (attention heads). A different use case but demonstrates active development of IF-based attribution for LLMs, potentially with insights on FM1/FM2.
- **LoGra + LogIX** (ICLR 2026): Efficient gradient projection for scalable influence functions. Directly addresses the Hessian approximation bottleneck through engineering rather than representation-space switching.

### 文献覆盖漏洞

**缺失关键工作**:

1. **"Towards Unified Attribution" (arXiv:2501.18887)**: This January 2026 position paper is a DIRECT conceptual competitor to CRA's unification claim. It must be cited and CRA must differentiate. The problem-statement does not mention it.

2. **"A Survey of Data Attribution" (2025 comprehensive survey)**: A recent survey covering TDA landscape. CRA must position relative to this survey's coverage. Not mentioned in problem-statement.

3. **MDA / Mechanistic Data Attribution (arXiv:2601.21996, January 2026)**: Demonstrates IF-based attribution working on specific LLM components (interpretable heads) using Pythia family — the same model family CRA plans to use. May provide evidence for or against FM1 at specific layers.

4. **LoGra + LogIX (ICLR 2026)**: Engineering approach to scalable IF that may reduce the Hessian bottleneck enough to undermine the "parameter-space IF doesn't work" narrative.

**覆盖充分方向**: The problem-statement's coverage of Li et al., DDA, RepT, MAGIC, Better Hessians Matter, In-the-Wild, Concept IF, AirRep, and DATE-LM is strong. The baseline papers table (project.md §1.3) is well-curated.

### 贡献边际

**实际 delta**: CRA proposes three things: (1) three-bottleneck diagnostic framework (Hessian + FM1 + FM2), (2) unified phi^T psi taxonomy of representation-space methods, (3) first systematic evaluation of representation-space methods on DATE-LM.

- Contribution (1) is novel IF the three bottlenecks are empirically separable. The decomposition itself is new — no prior work has proposed this specific tripartite structure. However, the individual bottlenecks are all known (Hessian: extensively studied; FM1: Li et al.; FM2: DDA).
- Contribution (2) is organizational, not theoretical. The bilinear form phi^T psi is a structural observation. §2.2 honestly labels this as "shallow." Risk: reviewers see this as reformulation.
- Contribution (3) fills a genuine benchmark gap (G-RepT4, G-AR2). This is the most defensible contribution — DATE-LM is NeurIPS 2025, and no representation-space method has been evaluated on it.

**是否足够**: **边缘** — Without a novel method contribution (e.g., Fixed-IF, representation-contrastive hybrid), the paper is diagnostic + benchmark. This is poster-level at NeurIPS if executed cleanly, but unlikely to reach oral/spotlight without a method that outperforms existing approaches.

**创新类型**: **有意义增量** — The three-bottleneck decomposition provides genuinely new diagnostic clarity. The benchmark fills a real gap. But the phi^T psi taxonomy risks being seen as cosmetic, and the lack of a new method limits ceiling.

**核心差异点**: CRA is the first to propose separating LLM TDA failure into three orthogonal bottlenecks and testing their independence via 2x2 ablation — this specific experimental design is unique.

### 并发工作风险

**风险等级**: **中-高**

**依据**:
- 5 independent representation-space TDA methods appeared in ~12 months (2024-2025). This signals high community activity.
- The "Towards Unified Attribution" (2501.18887) paper shows the unification idea is in the air.
- DATE-LM was published at NeurIPS 2025, creating a natural "evaluate existing methods on new benchmark" opportunity that multiple groups will pursue.
- AirRep (NeurIPS 2025) already positions representation-space TDA as a scalable alternative.
- The comprehensive TDA survey (2025) covers the landscape, making the "survey/taxonomy" contribution harder to differentiate.
- LoGra/LogIX (ICLR 2026) may close the Hessian gap, weakening the "parameter-space IF fails" narrative.

**如果风险高**: CRA's unique angle is the 2x2 ablation testing FM1-FM2 independence. No competitor is doing this specific decomposition experiment. CRA should lean into the diagnostic framework + ablation rather than the taxonomy contribution. Speed matters — the probe should be run immediately to validate direction before investing in full evaluation.

### 继续的最强理由

The systematic benchmark evaluation of representation-space methods on DATE-LM fills a recognized gap that no current or upcoming work addresses specifically. The 2x2 ablation design is information-dense and unique.

### 最危险的失败点

A comprehensive TDA survey or unified framework paper appearing on arXiv in the next 3-4 months that subsumes the diagnostic framework and includes DATE-LM evaluations.

### 被施压的假设

H4 (RepSim competitive on LDS) — no empirical support. If RepSim fails on LDS, the "representation space systematically addresses FM1" narrative is weakened.

### 探针一致性检查

The primary probe has NOT been executed. The Sibyl pilot (LIBERO-10) is in an entirely different domain and provides no evidence for FM1/FM2 in LLM TDA. The problem-statement is honest about this (§3.1 "PROBE NOT YET EXECUTED") but the risk assessment should be elevated — proceeding to design without any empirical validation of the core thesis is premature.

### 推荐判定

**Revise** — The direction is sound and fills a real gap, but two issues require action before proceeding:

1. **Execute the probe before design phase**. H4 has zero evidence. The probe is <1 GPU-day and determines whether the entire direction is viable. Proceeding to design without it is strategically unwise.
2. **Address the "Towards Unified Attribution" (2501.18887) competition**. CRA must explicitly differentiate from this recent unification work. Add it to the literature review and clarify CRA's empirical + diagnostic contribution vs. that paper's conceptual framework.
