## [Contrarian] 反对者视角

### 假设挑战

- **H1 (FM1 is a dominant bottleneck at LLM scale)**: This is the weakest load-bearing assumption. The strongest evidence (Li et al.'s RepSim 96-100% vs IF 0-7%) comes exclusively from LoRA fine-tuning, where the Hessian is trivially low-rank. The problem-statement §1.3 explicitly acknowledges this: "The strongest evidence for FM1 (Li et al.'s iHVP degeneracy analysis) is LoRA-specific. Under full fine-tuning, the Hessian is not low-rank, and FM1's severity is an open empirical question." Yet the entire three-bottleneck framework hinges on FM1 being real and significant. If FM1 vanishes under full fine-tuning, the framework collapses to Hessian error + FM2 — which is essentially what DDA + Better Hessians Matter already cover. **The "three bottleneck" novelty claim rests on the least supported leg.**

- **H4 (RepSim is competitive with TRAK on LDS)**: This is explicitly rated "None" evidence strength in §1.6. The entire experimental strategy assumes RepSim will perform well on LDS — a counterfactual metric — when RepSim was only validated on correlational metrics (P@K on harmful data identification). The problem-statement §1.5 RQ2 even predicts RepSim "may struggle on factual attribution." If RepSim fails on LDS across multiple DATE-LM tasks, the paper's empirical backbone breaks. The probe has NOT been executed — this is a direction bet with zero empirical support.

- **H5 (MAGIC's LDS ~0.99 does not invalidate FM1)**: MAGIC achieves near-perfect attribution in parameter space with exact IF. If MAGIC works at Pythia-1B scale on DATE-LM, the FM1 thesis is directly contradicted: parameter space works fine when you compute the Hessian exactly. The problem-statement lists possible resolutions (deterministic training, LoRA, metric limitations) but these are speculative escape hatches, not evidence. §1.3 states "This tension MUST be addressed experimentally" — agreed, but if MAGIC wins, CRA loses its raison d'être.

### 反事实场景

**如果核心洞察是错的**: FM1 is an artifact of poor Hessian approximation under LoRA constraints, not an independent bottleneck. MAGIC proves this by achieving LDS ~0.95+ with exact parameter-space IF at Pythia-1B. RepSim achieves lower LDS than TRAK because cosine similarity in representation space captures semantic correlation but not counterfactual influence. The "three bottleneck" framework is a misdiagnosis — there is really one bottleneck (Hessian approximation quality) with one confound (FM2/common influence bias). DDA's contrastive scoring + better Hessians is sufficient.

**最可能的实验失败场景**:

- **Scenario 1 (RepSim LDS collapse)**: RepSim scores high on P@K (identifying relevant training samples) but low on LDS (predicting counterfactual model behavior under data removal). This is because cosine similarity in hidden space captures "semantic relatedness" but not "causal influence on training dynamics." The LDS metric specifically measures whether attribution scores predict actual model change upon data removal — a causal quantity that correlation-based methods fundamentally cannot capture. Result: the core "FM1 → representation space" thesis survives as a correlational tool but loses its diagnostic teeth.

- **Scenario 2 (MAGIC domination)**: MAGIC achieves LDS 0.90+ on DATE-LM toxicity filtering at Pythia-1B, outperforming both RepSim and TRAK. This proves Hessian quality is the dominant bottleneck and parameter space is fine when computed exactly. CRA's narrative that "parameter space is fundamentally limited" is refuted. The paper becomes "MAGIC is expensive, RepSim is a cheap approximation" — a systems paper, not a diagnostic contribution.

- **Scenario 3 (2x2 interaction dominance)**: The FM1 x FM2 interaction term exceeds 30% of minimum main effect. This means the "two independent signal-processing defects" framework is an oversimplification. FM1 and FM2 are coupled — representation space implicitly addresses FM2, making the clean decomposition narrative invalid. The paper needs a fundamentally different theoretical framing.

### 被低估的竞争方法

**有** — Two specific competitors are underestimated:

1. **MAGIC** (arXiv:2504.16430): The problem-statement treats MAGIC as a "key tension" but not as a direct competitor. If MAGIC's compute cost can be reduced (e.g., via amortization across test points, or efficient meta-gradient computation), it becomes the dominant solution: exact IF in parameter space without any need for the FM1/FM2 decomposition. The problem-statement's §2.2 point 4 acknowledges this but does not quantify the computational gap or assess tractability of reducing MAGIC's cost.

2. **"Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods"** (arXiv:2412.03906): This paper already provides a unified view of gradient-based attribution methods. If it covers representation-space methods or the bilinear phi^T psi structure, CRA's "unified taxonomy" contribution is pre-empted.

3. **"Towards Unified Attribution" (arXiv:2501.18887)**: This 2026 position paper explicitly argues for unifying feature, data, and component attribution. If it covers the representation-space TDA family, CRA's organizational contribution overlaps.

### 生死线评估

**如果结果上限是**: RepSim LDS within 5pp of TRAK (but below MAGIC) + 2x2 interaction term > 20% + no novel method contribution → **不值得发表** at a top venue as an oral/spotlight. This would be a benchmark paper showing "representation methods are competitive but not superior, FM1 and FM2 are correlated not independent." Poster-level at best, and only if the benchmark evaluation is comprehensive enough (5+ methods, 3+ tasks, 3+ model scales).

**如果结果上限是**: RepSim LDS > TRAK + contrastive scoring adds independent gain + clean 2x2 decomposition + MAGIC is infeasible at scale → **值得发表** as a solid poster at NeurIPS. The diagnostic framework provides genuine clarity to the field. But without a novel method (e.g., Fixed-IF or representation-contrastive hybrid), the ceiling is poster, not oral.

### 继续的最强理由

The field genuinely lacks a diagnostic decomposition of why parameter-space TDA fails on LLMs. Five independent representation-space methods appearing in 12 months IS a real signal of a paradigm shift that no one has named. Even if FM1 is weaker than claimed, the systematic benchmark evaluation on DATE-LM fills a recognized gap (G-RepT4, G-AR2).

### 最危险的失败点

MAGIC invalidation: if exact parameter-space IF achieves LDS ~0.95+ at Pythia-1B on DATE-LM, the FM1 thesis is dead, and the "three bottleneck" framework reduces to well-known territory.

### 被施压的假设

H1 (FM1 dominance) is the most fragile. Li et al.'s evidence is LoRA-only. No full-FT evidence exists. The probe has not been run.

### 探针一致性检查

**Critical problem**: The primary probe has NOT been executed. The problem-statement §3.1 explicitly states "PROBE NOT YET EXECUTED." The secondary evidence (Sibyl/LIBERO-10 pilot) is from a completely different domain (VLA policy learning) and provides zero information about FM1/FM2 in LLM TDA. The problem-statement correctly identifies this gap but proceeds to build an elaborate framework on unverified assumptions. The honest assessment in §2.2 and §1.6 is commendable but does not change the fact that zero empirical support exists for the core thesis.

### 推荐判定

**Revise** — The three-bottleneck framework is intellectually coherent and the field gap is real, but three critical issues must be addressed before proceeding to design:

1. The probe MUST be executed before committing to the full experimental program. H4 (RepSim LDS competitiveness) has zero evidence.
2. The MAGIC invalidation scenario needs an explicit decision rule: if MAGIC LDS > X at Pythia-1B on DATE-LM, what happens to the paper narrative? Currently this is hand-waved in §1.3.
3. FM1's LoRA-specificity risk must be elevated from a known limitation to a core experimental question — the 2x2 ablation should include a LoRA vs full-FT dimension.
