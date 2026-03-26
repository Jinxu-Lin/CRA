## [Skeptic] 统计怀疑论者视角

### 统计有效性

**样本量**: 3 seeds is the absolute minimum for reporting mean +/- std. With ~100 test samples per task and 3 seeds, the effective sample size for per-sample analysis is ~300. This is adequate for detecting moderate effects (Cohen's d >= 0.3) but may miss small effects. **Verdict: Marginal but acceptable given compute constraints.**

**多次运行计划**: The design correctly uses 3 seeds across all conditions. However, it does NOT distinguish between training randomness and evaluation randomness. Specifically:
- The fine-tuned model is trained with a specific seed (training randomness)
- Attribution scores are computed deterministically given the model (no additional randomness for RepSim; TRAK has projection randomness)
- LDS evaluation requires retraining models with data removed (retraining randomness)

**The 3 seeds control training randomness but NOT retraining randomness in LDS computation.** If DATE-LM's LDS evaluation uses a single retraining seed per data-removal condition, the LDS variance estimate conflates attribution method variance with retraining variance. **Should confirm DATE-LM's protocol and, if possible, also vary retraining seeds.**

**多次比较**: The design includes 6 methods x 3 tasks = 18 primary comparisons for Experiment 1. With Benjamini-Hochberg FDR correction at q = 0.05, this is manageable. The 2x2 ANOVA in Experiment 2 has only 4 statistical tests per task (FM1 main, FM2 main, interaction, overall F). **Multiple comparison burden is well-controlled.**

**稳定性报告**: 95% bootstrap CIs planned. Permutation tests planned. Effect sizes (Cohen's d) planned. **This is a strong statistical plan — among the best I've seen in TDA papers.**

### 混淆因素

1. **TRAK projection dimension confound**: TRAK's random projection to dimension k introduces its own quality/dimensionality tradeoff. If k is set too low, TRAK performance degrades for reasons unrelated to FM1. If k is set optimally, TRAK already performs dimensionality reduction (just random rather than learned). **The FM1 main effect in the 2x2 is partially confounded with "learned vs random dimensionality reduction."** A cleaner test: include Grad-Sim (no projection) as a parameter-space baseline, and compare RepSim advantage over Grad-Sim (pure FM1, no projection confound) vs. RepSim advantage over TRAK (FM1 + projection quality).

2. **Contrastive scoring computational confound**: Contrastive scoring doubles computation (run on M_ft AND M_base). This means contrastive methods use 2x the information (two model checkpoints vs one). A skeptical interpretation: contrastive scoring works not because it "removes FM2" but because it uses more model information (ensembling effect). **Control**: Test a "double RepSim" baseline that averages RepSim scores from two independent runs of the fine-tuned model (same model, different random augmentation or data ordering). If this matches contrastive RepSim, the improvement is from ensembling, not debiasing.

3. **Pre-training model quality confound**: Different base models (Pythia-1B vs Llama-7B) differ in pre-training data, architecture, and quality. Scale-up experiment (Experiment 5) conflates model scale with model quality. **Not a major issue if the primary claims are made on Pythia-1B, with Llama-7B as "generalization" evidence.**

### 最简替代解释

**替代假说**: "RepSim performs well not because it addresses FM1 (signal dilution), but because representation similarity directly measures functional similarity between training and test samples — a trivially useful feature for data attribution that happens to work better than gradient inner products simply because gradients are noisy approximations of the Hessian-weighted functional relationship."

In other words: RepSim works because it's a **better feature** for similarity measurement, not because of the FM1 signal-processing argument. The FM1 narrative (dimensionality reduction, SNR concentration) is a post-hoc rationalization for a simpler truth: cosine similarity of learned features is a good proxy for data relevance.

**区分实验**: If this alternative is correct, then:
1. RepSim advantage should be **independent** of parameter dimensionality B (since it's about feature quality, not dimensionality reduction). But the CRA prediction is that RepSim advantage scales with B. Experiment 3 (LoRA vs Full-FT) directly tests this.
2. RepSim should work equally well with representations from an UNTRAINED model (since the "better feature" argument doesn't require training-specific representations). **Missing experiment**: RepSim with random model representations. If random-model RepSim is near zero, trained representations specifically capture task-relevant information, supporting (but not proving) the FM1 narrative.

### 缺失证据

1. **Random-model RepSim control**: Compute RepSim using a randomly initialized Pythia-1B (no pre-training, no fine-tuning). If random RepSim achieves non-trivial LDS, representation similarity is capturing something about data structure independent of model learning, which would undermine the FM1 narrative. If random RepSim LDS ~ 0 (as expected), it confirms that trained representations are necessary. **This takes < 1 GPU-hour and is highly informative.** Should be included in the probe.

2. **Gradient subspace analysis**: Compute the effective rank of the per-sample gradient matrix (N x B, where N = training samples). If the gradient matrix has effective rank k << B, this directly validates FM1 (most gradient dimensions carry no signal). If effective rank is close to N (which is already << B), it suggests gradients are NOT approximately orthogonal in the task-relevant subspace, weakening FM1. **This analysis can be done on stored gradients at minimal additional cost.**

3. **TRAK with optimal projection dimension**: TRAK's default projection dimension may not be optimal for DATE-LM. If TRAK with tuned projection dimension matches RepSim, the "representation space" advantage is really just a "better dimensionality reduction" advantage that can be achieved in parameter space with effort. **Should include TRAK with k = d (matching representation dimension) as a controlled comparison.**
