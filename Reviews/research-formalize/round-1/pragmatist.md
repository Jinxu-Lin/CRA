## [Pragmatist] 务实者视角

### 工程组件拆解

✓ **DATE-LM benchmark infrastructure** — Open-source (GitHub: DataAttributionEval/DATE-LM), NeurIPS 2025 Datasets track. Includes evaluation scripts, model checkpoints, pre-computed baselines (TRAK, LESS, MATES). Clone and run.

✓ **TRAK baseline** — DATE-LM includes TRAK implementation. Direct execution.

✓ **RepSim implementation** — Conceptually trivial: extract hidden states h^(l) at target layer, compute cosine similarity. 50-100 lines of PyTorch code on top of existing model loading infrastructure.

△ **RepT implementation** — arXiv:2510.02334 describes the method but no official open-source implementation is confirmed. Need to reimplement: extract h^(l*) + compute nabla_h L, concatenate, find phase transition layer. Modification effort: ~3-5 days including debugging layer selection and gradient computation.

△ **DDA contrastive scoring** — DDA paper (EMNLP 2024) has code for hallucination tracing. Adapting to DATE-LM tasks requires constructing contrastive references (z_cf) for each task. Toxicity filtering: natural pairs (toxic vs. clean). Data selection: no natural contrastive reference — this is a non-trivial design decision. Modification effort: ~3-5 days.

△ **MAGIC (metagradient exact IF)** — arXiv:2504.16430. Requires deterministic training (fixed random seeds, no data augmentation randomness) + Replay algorithm for meta-gradient computation. Adapting to DATE-LM's models/tasks: ~5-7 days. Major concern: MAGIC's compute cost scales as O(N * n) where N = training set size, n = test set size. At Pythia-1B scale with DATE-LM training sets, this may be prohibitively expensive (see compute estimate below).

✗ **2x2 ablation framework** — Need to implement 4 conditions: {parameter-space, representation-space} × {standard, contrastive}. The "representation-space + contrastive" condition (RepSim with contrastive scoring) has never been implemented. Need to design: what is contrastive scoring in representation space? Use DDA's debias principle but with hidden states instead of gradients? ~5-7 days to design + implement + validate.

✗ **LDS evaluation pipeline customization** — DATE-LM provides LDS computation, but integrating new methods (RepSim, RepT, DDA, 2x2 variants) requires adapter code for each method to produce the right attribution score format. ~2-3 days.

### 最小 Pilot 设计

**实验内容**: RepSim vs TRAK on DATE-LM toxicity filtering task with Pythia-1B. This is exactly the probe described in problem-statement §3.2. The probe is the gate — nothing else should proceed until it runs.

**缩放策略**: Pythia-1B is already the smallest practical LLM for TDA evaluation. No further scaling down — FM1/FM2 effects may vanish at smaller scales. Toxicity filtering is the best first task (most analogous to Li et al.'s harmful data ID where RepSim dominates).

**所需已就位组件**: DATE-LM clone, Pythia-1B checkpoint, TRAK baseline (all in DATE-LM repo), RepSim implementation (50-100 lines).

**预计算力**: < 1 A100 GPU-day. Breakdown: model loading + hidden state extraction (~2-4 hours), TRAK run (~4-8 hours using DATE-LM's existing infrastructure), LDS evaluation (~1-2 hours).

### 工程陷阱

- **MAGIC compute feasibility at Pythia-1B**: MAGIC requires differentiating through the entire training process (metagradient). The Replay algorithm stores all intermediate training states. For Pythia-1B with a non-trivial training set, this could require: (a) 100+ GB storage for training checkpoints, (b) O(N * n) forward passes where N, n are training/test set sizes. MAGIC's original experiments use Gemma-2B but with small fine-tuning datasets and LoRA. On DATE-LM's full training sets, MAGIC may be infeasible within the project's GPU budget. **This is critical**: if MAGIC cannot be run, RQ1's bottleneck decomposition loses its most important reference point.

- **Contrastive reference construction for data selection task**: DDA's debias uses natural contrastive pairs (correct vs. incorrect entity for hallucination). DATE-LM's data selection task has no natural contrastive structure — you select "useful" vs "not useful" training data, but "not useful" is the entire complement set, not a clean contrastive reference. The z_cf design decision could dominate the 2x2 ablation results and become a confound rather than a clean signal.

- **Layer selection sensitivity for RepSim**: Li et al.'s RepSim results depend on layer choice. The problem-statement mentions "middle + last layer" but different DATE-LM tasks may require different layers. RepT's phase transition detection addresses this, but RepT is harder to implement. If RepSim fails because of wrong layer choice, the failure is attributable to implementation, not to the FM1 thesis.

- **LDS metric reliability**: H-RF1 (Revisiting Fragility) and H-DVEmb3 raise concerns about LDS. If LDS is unreliable at the scale tested, all comparisons are compromised. There is no fallback metric in the problem-statement.

### 综合预估

- **日历时间（到第一个有意义结果 — probe）**: 1 week (includes DATE-LM setup, RepSim implementation, TRAK run, analysis). This should be done BEFORE proceeding to design.
- **日历时间（到full experimental results — all methods, all tasks, 2x2 ablation）**: 6-10 weeks (×3-5 calendar multiplier applied). Major risks: MAGIC feasibility, contrastive reference design, multi-task evaluation.
- **算力（probe）**: < 1 A100 GPU-day
- **算力（full evaluation）**: 20-50 A100 GPU-days (estimate: 5 methods × 3 tasks × 2-3 model scales × multiple layers/variants). MAGIC could dominate if attempted at scale.
- **Available resources**: 4× RTX A6000 48GB (shared server). A6000 is ~80% of A100 throughput for inference. Adequate for probe and most evaluation, but MAGIC's memory requirements may exceed 48GB for Pythia-1B metagradients.
- **主要工程风险**: MAGIC infeasibility at target scale. If MAGIC cannot be run, the bottleneck decomposition loses its "exact IF" anchor, and the Hessian bottleneck contribution cannot be cleanly isolated.

### ROI 评估

**当前 ROI 判断**: Medium-favorable. The probe costs < 1 GPU-day and determines viability. If pass, the full evaluation is tractable within 6-10 weeks on available hardware. The main risk-reward concern is the ceiling: without a novel method, the contribution is diagnostic + benchmark = poster-level.

**ROI 优化建议**: Run the probe first. If RepSim LDS >= TRAK - 5pp, proceed. If not, pivot early rather than investing 6-10 weeks in a direction with broken empirical foundation.

### 继续的最强理由

The probe is cheap (< 1 GPU-day) and provides a decisive signal. If it passes, the full evaluation is tractable with available resources. DATE-LM infrastructure is mature and open-source, reducing engineering risk.

### 最危险的失败点

MAGIC proving infeasible at Pythia-1B scale on A6000 hardware, removing the exact-IF anchor from the bottleneck decomposition.

### 被施压的假设

H4 (RepSim competitive on LDS) — zero evidence, but the probe is cheap and should be run immediately. The assumption that MAGIC is feasible at target scale is also untested and could be a critical blocker.

### 探针一致性检查

The probe has NOT been executed. The problem-statement is transparent about this but has not run the < 1 GPU-day experiment that would validate or invalidate the entire direction. This is the single most important action item. Every day spent on formalization without probe results is wasted if RepSim fails on LDS.

### 推荐判定

**Revise** — The formalization is intellectually sound but proceeding to design without the probe is premature. Specific action:

1. **Execute the probe before any design work** (< 1 GPU-day). RepSim vs TRAK on DATE-LM toxicity filtering with Pythia-1B.
2. Assess MAGIC compute feasibility: estimate metagradient memory/time requirements for Pythia-1B on DATE-LM training set sizes. If infeasible, RQ1's operationalization needs revision.
3. Design a fallback for contrastive reference construction on DATE-LM data selection task. Current proposal is underspecified.
