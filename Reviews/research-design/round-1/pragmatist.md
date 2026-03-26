## [Pragmatist] 务实者视角

### 工程组件拆解

**✓ DATE-LM benchmark** — Open-source, includes TRAK + Grad-Sim + evaluation pipeline. Clone and use.

**✓ Pythia-1B model** — HuggingFace, well-tested. Fits on single A6000 with room to spare.

**✓ TRAK implementation** — DATE-LM codebase provides reference implementation. Direct use.

**✓ BM25 baseline** — DATE-LM codebase provides it.

**△ RepSim implementation** — Conceptually trivial (forward pass + cosine similarity). Needs: (1) determine token aggregation strategy (last token vs mean pooling — NOT specified in method-design.md); (2) handle batch processing for 10K training samples; (3) format scores for DATE-LM evaluation pipeline. Estimated: 1-2 days.

**△ RepT implementation** — More complex. Needs: (1) phase-transition layer detection algorithm; (2) per-sample gradient w.r.t. hidden representation extraction (backward hook); (3) concatenation and cosine similarity. The phase-transition detection is the trickiest part — RepT's original paper describes it but implementation details may require experimentation. Estimated: 3-5 days.

**△ Contrastive scoring variants** — Straightforward subtraction. Needs: (1) run base model forward pass for all samples; (2) subtract scores. Main engineering concern: memory management when loading both M_ft and M_base. At Pythia-1B (2GB each), trivial on 48GB. Estimated: 1-2 days.

**△ Full-FT pipeline** — DATE-LM likely defaults to LoRA. Modifying for full-FT requires: (1) learning rate sweep on validation set; (2) gradient checkpointing setup; (3) verify training stability at full-FT. Per-sample gradient extraction at full-FT uses ~8GB per sample (Pythia-1B) — need sequential processing or gradient accumulation. Estimated: 3-5 days to set up + debug.

**✗ MAGIC implementation** — Substantial engineering effort. Requires: (1) deterministic training (fixed seeds, fixed data order, fixed batch composition — non-trivial with DataLoader workers); (2) checkpoint storage at every training step (~400GB for 200 steps at Pythia-1B); (3) metagradient computation (backward through the training loop). The MAGIC paper's code may be available but adaptation to DATE-LM + Pythia-1B is non-trivial. **Estimated: 1-2 weeks.** High risk of infeasibility.

### 最小 Pilot 设计

**实验内容**: Experiment 0 (probe) is well-designed. RepSim vs TRAK, Pythia-1B, LoRA, toxicity filtering, single seed.

**缩放策略**: Already at minimum viable scale. Pythia-1B is the smallest useful LLM for TDA. Toxicity filtering with ~10K training samples is DATE-LM's standard scale.

**所需已就位组件**: DATE-LM codebase cloned + environment set up + RepSim implementation (△, 1-2 days)

**预计算力**: 2 GPU-days on A6000 (generous estimate; actual computation for RepSim is < 1 GPU-hour, but DATE-LM's TRAK + LDS evaluation is the bottleneck)

### 工程陷阱

1. **DATE-LM environment setup (MEDIUM risk)**: DATE-LM is a NeurIPS 2025 benchmark — the codebase may have specific dependency requirements (particular torch version, specific transformers version). Environment conflicts with existing server setup could cost 1-3 days. **Mitigation**: Use conda/docker environment isolation.

2. **LDS evaluation wall-clock time (HIGH risk)**: LDS requires retraining models with data subsets removed. For each condition, this means N retrained models (where N depends on DATE-LM's leave-K-out protocol). At Pythia-1B with LoRA, each retrain takes ~30 min. If DATE-LM requires 100 retrains per condition: 100 * 30 min = 50 GPU-hours per condition. With 6 methods x 3 tasks x 3 seeds = 54 conditions: 2700 GPU-hours = 112.5 GPU-days. **THIS BLOWS THE ENTIRE BUDGET.** Must verify DATE-LM's LDS computation protocol. If they use leave-K-out with shared retraining (train once per data subset, evaluate all methods), the cost is much lower. **This is the single biggest risk to the project timeline.**

3. **MAGIC disk space (HIGH risk for Experiment 4)**: Storing 200 checkpoints x ~2-4GB each = 400-800GB. Shared server may not have this much scratch space available. **Mitigation**: Store every 10th checkpoint and recompute intermediate (at 10x compute cost per test sample). Or use gradient checkpointing to reduce storage at increased compute.

4. **Full-FT gradient extraction memory (MEDIUM risk)**: Per-sample gradients for Pythia-1B full-FT: ~4GB (fp16). For TRAK, need to store projections for all training samples. With 10K samples: 10K x 4GB = **40TB** if stored naively. Obviously need sequential processing + immediate projection. TRAK's standard implementation handles this, but integration with DATE-LM's pipeline needs verification. RepSim at full-FT: only needs forward pass representations (~80MB for all samples). **RepSim has a massive engineering advantage over TRAK at full-FT.**

5. **Shared server availability (MEDIUM risk)**: 4x A6000 shared server with ~75% availability. Experiments 1-3 need parallel GPU access for 3-seed runs. Server queue/sharing could stretch timelines significantly. **Mitigation**: Design experiments for serial execution on 1 GPU where possible; parallelize only when all 4 GPUs are available.

### 综合预估

- **日历时间（到第一个有意义结果 = Experiment 0 probe）**: 1.5-2 weeks (including DATE-LM setup + RepSim implementation + probe execution + analysis)
- **日历时间（到complete core results = Experiments 0-3）**: 6-8 weeks (including engineering setup, debugging, execution, analysis)
- **算力（probe）**: 2-3 GPU-days
- **算力（core experiments）**: 40-55 GPU-days (within 60 GPU-day budget, but tight)
- **算力（MAGIC Experiment 4）**: 5-15 GPU-days (highly uncertain; may be infeasible)
- **主要工程风险**: **LDS evaluation cost uncertainty.** If DATE-LM's LDS evaluation requires per-condition model retraining, the 60 GPU-day budget is insufficient. This must be verified BEFORE committing to the full experimental plan. The probe (Experiment 0) should include LDS timing as a critical measurement.

### Budget Feasibility Assessment

The 60 GPU-day budget in method-design.md §2 appears optimistic. Critical risk: LDS computation cost. The budget allocates 15 GPU-days for Experiment 1 (6 methods x 3 tasks x 3 seeds = 54 conditions). That's 0.28 GPU-days per condition, which must cover both attribution scoring AND LDS evaluation (including retraining). If a single LDS evaluation per condition takes 10 GPU-hours (conservative for leave-K-out at Pythia-1B), that's 0.42 GPU-days per condition, already exceeding the budget per condition.

**Recommendation**: During the probe (Experiment 0), measure actual wall-clock time for ONE full LDS evaluation cycle. Use this to recalibrate the budget for Experiments 1-5. If LDS evaluation is expensive, options are: (a) reduce from 3 seeds to 2 seeds for non-core experiments; (b) use AUPRC instead of LDS as primary metric for toxicity filtering (no retraining needed); (c) compute LDS on a subset of test samples.
