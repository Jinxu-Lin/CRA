## [Empiricist] 实验主义者视角

### 否证条件

- **主要**: If RepSim LDS < TRAK LDS - 5pp on ALL three DATE-LM tasks (toxicity, data selection, factual attribution) at Pythia-1B with LoRA, then the FM1 thesis (representation space addresses signal dilution) is refuted at the LDS level. Threshold basis: 5pp is approximately 1 standard deviation of LDS across seeds based on DATE-LM variance reports (~3-5pp std for TRAK across seeds). This means we require at least a 1-sigma effect.

- **Secondary (2x2)**: If the FM1 main effect (mean LDS_repr - mean LDS_param) is negative on >= 2/3 tasks, the diagnostic framework's central claim is wrong. Threshold: negative means the sign is wrong, not just magnitude.

- **Secondary (interaction)**: If |interaction| > 50% of min(|FM1 effect|, |FM2 effect|) on >= 2/3 tasks, the "independent bottlenecks" framework is a misleading oversimplification (stricter than the 30% threshold in experiment-design.md — I argue 30% is too lenient for a "independence" claim).

- **早期信号**: In the probe (Experiment 0), if RepSim LDS on toxicity filtering is below 0.05 (essentially random-level), stop immediately. Do not proceed to adjust layers, try RepT, etc. This signals a fundamental failure of the representation-similarity approach for counterfactual attribution. P@K should be checked as a secondary signal, but the probe's primary gate is LDS.

### 最小 Pilot 设计

**实验内容**: Already well-designed as Experiment 0 (RepSim vs TRAK, Pythia-1B, LoRA, toxicity filtering, single seed). Budget: 2 GPU-days. This is appropriate.

**核心测量量**: LDS difference (RepSim - TRAK). Secondary: rank-correlation between RepSim and TRAK attribution scores (high correlation + similar LDS → methods capture similar information; low correlation + different LDS → methods capture different signals).

**自我欺骗风险**:
1. "The layer wasn't optimal" — the probe uses L/2 and L. If neither works, adding more layer search is scope creep that delays the critical gate decision.
2. "Toxicity filtering isn't representative" — the probe deliberately chose the MOST favorable task. If RepSim fails here, it won't suddenly work on harder tasks.
3. "Single seed is noisy" — true, but the probe only needs to see the SIGN of the difference. If RepSim is substantially worse (> 10pp), one seed is sufficient to detect this.
4. "DATE-LM evaluation has issues" — this is the hardest to rebut. If BOTH methods score poorly (LDS < 0.10), it may indeed be a DATE-LM issue. But if TRAK scores normally (~0.15-0.25) and RepSim doesn't, it's a RepSim issue.

### Confounders 审查

1. **Representation extraction protocol (CRITICAL)**: RepSim results depend heavily on HOW representations are extracted — which token position (last token? mean over sequence?), which layer, whether to normalize. The method-design.md specifies cosine similarity and two layer choices, but does NOT specify token aggregation strategy. **For autoregressive LLMs like Pythia, the last token representation contains the most task-relevant information (it predicts the next token). Mean pooling dilutes this signal.** This implementation detail could make or break RepSim, and it's unspecified. **Must be controlled and reported.**

2. **Contrastive scoring asymmetry**: When computing TRAK_ft - TRAK_base, the TRAK projection matrices are typically fitted to the fine-tuned model's gradients. The base model's gradients may require different projection parameters. Using the same projection for both could artificially degrade the contrastive baseline. **Should use independently fitted projections for M_ft and M_base.**

3. **DATE-LM LDS computation variance**: LDS requires retraining models with data subsets removed. The number and composition of removed subsets affects LDS precision. DATE-LM's protocol should be followed exactly, but any deviation (different subset sizes, different number of retrained models) would change absolute LDS values. **Cross-study LDS comparisons are unreliable; only within-study rankings matter.**

4. **Full-FT confound (Experiment 3)**: When comparing LoRA vs Full-FT, the fine-tuned models are DIFFERENT (different parameters updated, different optimization trajectory). Any LDS difference between LoRA and Full-FT could be due to the models themselves being different quality, not FM1 severity. **Control**: Report each model's downstream task performance (validation loss, accuracy) alongside TDA metrics. If Full-FT model is significantly better/worse than LoRA model, the TDA comparison is confounded.

5. **MAGIC feasibility confound (Experiment 4)**: If MAGIC is run on only 5-10 test samples (feasibility limit), the resulting LDS estimate has enormous variance. With 5 samples, the 95% CI on Spearman correlation is roughly +/- 0.40. **Any conclusion drawn from 5-sample MAGIC is essentially anecdotal**, not statistical. The paper should present this as a "proof of concept" not as evidence for or against FM1.

### 评估协议完整性

**Benchmark/Metric**: LDS is the appropriate gold-standard metric for TDA. DATE-LM is the most comprehensive LLM TDA benchmark available. Both are well-chosen.

**Gaming risk**: Low for LDS (Spearman correlation is hard to game without actually improving attribution). Moderate for AUPRC on toxicity (AUPRC is sensitive to the score threshold; methods that produce well-calibrated scores may have an advantage independent of ranking quality).

**统计严谨性**:
- 3 seeds: Minimum acceptable. 5 would be better but budget-constrained. **Acceptable given the 60 GPU-day budget.**
- Permutation test with 10K permutations: Appropriate for non-parametric comparison.
- Bootstrap 95% CI: Appropriate.
- Benjamini-Hochberg FDR correction: Appropriate for multiple comparisons.
- **Missing**: Minimum detectable effect calculation is provided (~3-5pp at alpha=0.05, power=0.80) but based on "estimated from DATE-LM variance reports." **Should verify this estimate with Experiment 0 data before committing to 3-seed design.** If variance is higher than expected, 3 seeds may be insufficient.

**Ablation 结构**: The 2x2 design is well-structured for decomposing FM1/FM2. The interaction analysis is correctly specified. **One concern**: the 2x2 uses TRAK and RepSim as representatives of parameter-space and representation-space. But TRAK is an APPROXIMATE method (with its own projection error). A cleaner 2x2 would use Grad-Sim (no projection) for the parameter-space cell. TRAK's random projection introduces a confound: is the FM1 main effect measuring "representation space vs parameter space" or "learned compression vs random projection"? **Recommend**: Run the 2x2 with Grad-Sim as an additional parameter-space representative.

**Cross-dataset**: DATE-LM has three tasks (toxicity, data selection, factual attribution), which provides adequate cross-task validation. Cross-model validation (Llama-7B in Experiment 5) is included but budget-constrained. **Acceptable**.
