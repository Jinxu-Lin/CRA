## [Comparativist] 文献对标者视角

### Round 2 Revision Assessment

**Issue 2 (Towards Unified Attribution)**: ADDRESSED. The new §2.2 correctly positions arXiv:2501.18887 as complementary (conceptual umbrella across attribution types) rather than competitive (same-level TDA-specific empirical claims). The three differentiation axes are sound: (a) TDA-specific vs cross-type, (b) diagnostic + empirical vs conceptual taxonomy, (c) practitioner-usable benchmark results.

### SOTA 定位

**绝对 SOTA**: MAGIC (arXiv:2504.16430) -- LDS 0.95-0.99 on fine-tuning TDA with exact metagradient IF. This is the performance ceiling in parameter space.

**最相近 approach**: DDA (arXiv:2410.01285) -- contrastive scoring for LLM TDA, directly addresses FM2. CRA's differentiation: DDA addresses FM2 only; CRA proposes the three-bottleneck decomposition including FM1.

**最强简单 baseline**: TRAK with proper random projection on DATE-LM. TRAK is well-established, has open-source implementation in DATE-LM, and represents the "good-enough approximate IF" baseline. If TRAK + better Hessian (ASTRA-level) closes most of the gap, the FM1 thesis weakens.

**其他关键竞争方法**:
- RepT (arXiv:2510.02334): Representation gradients with auto-layer detection, P@10 = 0.97-1.00. Not yet evaluated on DATE-LM LDS.
- AirRep (arXiv:2505.18513, NeurIPS 2025): Learned encoder-based TDA. Different from CRA's model-internal representation approach.
- Better Hessians Matter (arXiv:2509.23437): Demonstrates Hessian quality ordering matters -- challenges the "Hessian doesn't matter" narrative that CRA v1.2 has correctly moved away from.

### 文献覆盖漏洞

**缺失关键工作**: None identified in this round. The v1.2 addition of "Towards Unified Attribution" (2501.18887) closes the gap flagged in Round 1. The paper landscape in §1.1 is comprehensive: TRAK, LoGra, EK-FAC, LESS, ASTRA, Better Hessians Matter, MAGIC, DDA for parameter-space; RepSim, RepT, In-the-Wild, Concept IF, AirRep for representation-space.

**覆盖充分方向**:
- Parameter-space TDA methods: Comprehensive (approximate → improved → exact → contrastive)
- Representation-space methods: All 5 independently proposed methods included with phi/psi decomposition
- Evaluation benchmarks: DATE-LM correctly identified as the primary benchmark

**Potential minor gap**: LoGra + LogIX (ICLR 2026) was flagged in Round 1 as indirect competition but is not discussed in §2.2. This is a minor point -- LoGra/LogIX address efficient computation of gradient-based attribution, not the representation-space vs parameter-space distinction. Does not warrant a Revise.

### 贡献边际

**实际 delta**: First systematic evaluation of representation-space TDA methods on DATE-LM + three-bottleneck diagnostic framework explaining why they work + 2x2 ablation quantifying FM1/FM2 independence.

**是否足够**: 足够 -- The benchmark contribution alone (5 methods, 0 cross-comparisons currently) fills an acknowledged gap. The diagnostic framework adds conceptual depth beyond pure benchmarking.

**创新类型**: 有意义增量 -- new diagnostic lens (three-bottleneck decomposition) + systematic benchmark (representation-space on DATE-LM). Not a fundamental paradigm shift, but a much-needed field-clarifying contribution. The bilinear unification is correctly positioned as "taxonomic convenience" (v1.2 §2.3 point 2), avoiding overclaiming.

**核心差异点**: CRA provides the first empirical decomposition of TDA failure into three quantified bottlenecks, whereas existing work addresses each bottleneck in isolation without measuring relative contributions.

### 并发工作风险

**风险等级**: 中 (unchanged from Round 1)

**依据**: The TDA-for-LLMs field is highly active (5 representation-space methods in 12 months, MAGIC/DDA/Better Hessians all 2025-2026). However, a systematic *diagnostic* framework + benchmark evaluation requires both the conceptual framing AND the engineering effort to run all methods on DATE-LM. No current work does both. The risk is that individual method papers (e.g., RepT on DATE-LM, DDA on more tasks) progressively fill the benchmark gap piecemeal.

**独特 angle**: The three-bottleneck decomposition + 2x2 ablation is CRA's differentiator. Even if individual methods publish DATE-LM results, the systematic decomposition study would remain unique for 6-12 months.

### 继续的最强理由
Five independently proposed representation-space TDA methods exist with zero cross-comparison on any common benchmark. The diagnostic gap is real, acknowledged by the community (G-RepT4, G-AR2), and has standalone citation value.

### 最危险的失败点
Concurrent work from RepT or AirRep teams publishing DATE-LM results before CRA, reducing the benchmark novelty to the diagnostic framework alone.

### 被施压的假设
H2 (contrastive scoring generality) -- only validated on 2 task types (hallucination + DPO alignment). DATE-LM has 3 tasks; data selection may lack natural contrastive references (§2.3 point 3).

### 探针一致性检查
No probe executed. The competitive landscape analysis is based entirely on published results, not CRA's own experiments. The problem statement is honest about this throughout.

### 推荐判定：**Pass**

The Round 1 literature gap (Towards Unified Attribution) is resolved. The competitive landscape is comprehensive. The contribution positioning (diagnostic framework + benchmark, not pure method novelty) is realistic. Concurrent competition risk is medium but the three-bottleneck decomposition provides 6-12 months of differentiation. No new blocking issues found.
