## [Contrarian] 反对者视角

### 假设挑战

**Assumption H1 (FM1 is a dominant bottleneck at LLM scale, not just a LoRA artifact)**:

This is the weakest assumption and the entire framework hinges on it. The strongest evidence for FM1 is Li et al.'s iHVP degeneracy analysis, which is ENTIRELY under LoRA. The method-design.md acknowledges this (§5 Component D) but still treats FM1 as a "general" bottleneck. **The JL orthogonality argument is theoretically valid for random vectors but empirically unverified for actual gradients.** Here's the critical counter-evidence the design doesn't adequately address:

- **MAGIC achieves LDS 0.95-0.99 in parameter space.** If FM1 were a dominant bottleneck, even exact IF should fail because the signal is "diluted" in high dimensions. But MAGIC succeeds brilliantly. The resolution offered (MAGIC uses deterministic training, LoRA, small models) is speculative. The more parsimonious explanation: **FM1 is not real. The "approximately orthogonal gradients" argument doesn't apply because gradients are structured, not random.** TRAK fails because of poor Hessian approximation (which MAGIC fixes), not because of dimensionality.

- **Better Hessians Matter (2509.23437)** shows that simply improving Hessian quality (H > GGN > EK-FAC > K-FAC) consistently improves attribution. This is a MONOTONIC improvement chain within parameter space. If FM1 were dominant, better Hessians should hit a ceiling (you can't fix dimensionality by improving the Hessian). The absence of such a ceiling in their experiments suggests **Hessian quality, not FM1, is the bottleneck.**

**Assumption H4 (RepSim is competitive with TRAK on LDS)**:

This has ZERO evidence. The probe hasn't been run. RepSim's known successes are all on binary classification tasks (harmful data identification: 96-100%) or retrieval metrics (P@K). LDS is a counterfactual metric that measures "what happens when you remove training data." Representational similarity measures "how similar are these samples in feature space." These are fundamentally different questions. **High RepSim does not imply high causal influence.** Two samples can have similar representations without one influencing the other (they could both be influenced by a third common cause in the training data).

The authors note this tension (problem-statement.md §1.5 RQ2) but still design the entire experimental framework assuming RepSim will be at least competitive. **If the probe fails, 80% of the experimental design (Experiments 1-3) needs restructuring.** The pivot plan (§1.4 of experiment-design.md) exists but is vague: "pivot to correlation vs causation diagnostic paper" — this sounds like a consolation prize, not a strong paper.

**Assumption H3 (FM1 and FM2 repair gains are approximately additive)**:

The 2x2 design assumes FM1 and FM2 are approximately independent. But there's a strong theoretical reason they MIGHT NOT be: representation space (which "fixes" FM1) also partially addresses FM2, because learned representations compress out the shared pre-training features that cause FM2. If moving to representation space already reduces FM2, then:
- The FM2 main effect (contrastive vs standard) will be SMALLER in representation space than parameter space
- This shows up as a negative interaction term
- The "independent bottlenecks" narrative collapses

The experiment design identifies this possibility (§3.2 of experiment-design.md: "representation-space methods may implicitly address FM2") but treats it as a secondary observation. **It should be a primary hypothesis.**

### 反事实场景

**如果核心洞察是错的**: FM1 doesn't exist as a meaningful bottleneck. RepSim works on some tasks because representational similarity happens to correlate with task-relevant influence for those specific tasks (toxicity filtering: toxic samples cluster in representation space), but this is task-specific, not a general principle. The "three bottleneck" framework is an overcomplication of a simpler truth: **TDA quality = Hessian quality, period. Everything else is method-specific noise.**

**最可能的实验失败场景**:

1. **RepSim achieves good AUPRC but poor LDS on toxicity filtering** (probability: 35%). RepSim correctly identifies toxic training samples (high AUPRC) because toxic samples cluster in representation space, but RepSim scores don't predict the MAGNITUDE of influence change upon removal (low LDS). This is the "correlation vs causation" gap. The paper pivots to this finding, but it's a weaker story than the original "three bottlenecks" framework.

2. **MAGIC is feasible and achieves LDS > 0.90 at Pythia-1B** (probability: 15%). This destroys the FM1 thesis. Even if RepSim also achieves high LDS, the narrative becomes "representation space is a cheap approximation of exact IF" — a engineering contribution, not a scientific insight. The diagnostic framework loses its explanatory power.

3. **2x2 interaction is large and negative** (probability: 25%). Representation space implicitly addresses FM2, making contrastive scoring redundant in representation space. The "three independent bottlenecks" reduces to "parameter space has two problems (Hessian + dimensionality), representation space has one (neither Hessian nor dimensionality matter)." This is informative but undermines the clean decomposition story.

### 被低估的竞争方法

**Yes — TRAK with better projection**. TRAK uses random projections to reduce gradient dimensionality from B to k. The CRA thesis argues this random projection is inferior to the "learned projection" of representation space. But TRAK's projection dimension k is typically set to 4096 — the SAME as representation dimension d. If TRAK used a LEARNED projection (e.g., PCA of gradients to dimension k, or projecting onto the top-k gradient singular vectors), it might match or exceed RepSim while staying in parameter space. **This would refute FM1 entirely** (the problem isn't parameter space per se, but the RANDOM projection in TRAK). The experiment design does not include this control.

**Also worth considering: gradient cosine similarity (Grad-Sim) with gradient clipping/normalization.** DATE-LM shows Grad-Sim is competitive on factual attribution. If Grad-Sim with careful normalization (L2 normalize per-sample gradients before computing similarity) approaches RepSim on other tasks, the "dimensionality" argument weakens — L2 normalization already addresses the "approximately orthogonal" problem by projecting onto the unit sphere.

### 生死线评估

**If RepSim LDS advantage over TRAK is < 3pp averaged across all three DATE-LM tasks**: The paper has no strong empirical claim. The "three bottleneck" framework remains a theoretical contribution, but one that can't be validated because the differences are within noise. A 3pp LDS difference across 3 seeds is at the boundary of statistical significance (given ~3-5pp std). **Verdict: If the average RepSim advantage is < 3pp, the paper is borderline — likely poster at best, possibly desk reject if reviewers challenge FM1.**

**If the 2x2 shows both main effects < 5pp on all tasks**: The diagnostic decomposition has no teeth. Even if the directions are correct (FM1 positive, FM2 positive), 5pp effects in TDA are practically negligible. The paper becomes "we decomposed the problem, and each piece is small." **Verdict: Not publishable at a top venue unless the interaction analysis or LoRA vs Full-FT dimension provides a surprising insight.**

**Minimum viable paper (my estimate)**: RepSim LDS >= TRAK LDS on at least 1/3 tasks AND FM1 main effect >= 5pp on at least 1/3 tasks AND the LoRA vs Full-FT comparison shows a statistically significant difference in RepSim advantage. This gives: (1) first benchmark result, (2) quantified FM1 contribution, (3) novel LoRA-specificity finding. Enough for a poster; the diagnostic framework elevates it to spotlight/oral if the narrative is tight.
