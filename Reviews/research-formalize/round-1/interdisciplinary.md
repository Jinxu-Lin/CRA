## [Interdisciplinary] 跨学科者视角

### 跨域对应物

#### 类比 A — 信号处理（Signal Processing）

**对应关系**: FM1 (signal dilution in high-dimensional parameter space) ↔ matched filtering in communications; FM2 (common influence contamination) ↔ differential detection / common-mode rejection in instrumentation.

**类比深度**: **深层** — The mathematical correspondence is precise:
- **Matched filter**: Project received signal onto subspace that maximizes SNR. In TDA: project from R^B (parameter space) to R^d (representation space, d << B) where task-relevant signal concentrates. The Johnson-Lindenstrauss lemma provides the theoretical guarantee that inner products are approximately preserved under random projection from R^B to R^d — this is exactly what representation-space methods do, except the projection is learned (not random), and thus can be BETTER than JL bounds.
- **Differential detection**: Subtract reference channel to remove common-mode noise. In TDA: contrastive scoring I(z_test, z_train) = IS_{theta'} - IS_{theta_0} subtracts base-model influence to isolate fine-tuning-specific signal. DDA's debias is literally differential detection.
- **Two-stage receiver**: First matched filter (dimensionality reduction), then differential detection (bias removal). CRA's 2x2 ablation tests whether these two stages are independent — a question signal processing theory predicts they should be, since matched filtering addresses noise bandwidth while differential detection addresses DC bias.

**已有解法**: In signal processing, the optimal receiver is well-understood: matched filter → differential detector → decision. The 70+ years of theory provides:
1. Conditions under which matched filtering is optimal (additive white Gaussian noise, known signal shape)
2. Conditions under which differential detection is independent of matched filtering (stationary common-mode, non-correlated noise)
3. Performance bounds (SNR gain = d_signal / d_noise for matched filter; rejection ratio for differential detection)

**可借鉴洞察**: Signal processing theory predicts that FM1 and FM2 corrections SHOULD be approximately additive IF: (a) the common-mode signal (FM2) is approximately stationary across training samples, and (b) the task-relevant signal subspace is approximately orthogonal to the common-mode subspace. These are testable conditions. If violated, the 2x2 interaction will be large — and signal processing theory tells you WHY.

#### 类比 B — causal inference (Instrumental Variables)

**对应关系**: The distinction between "correlational" attribution (RepSim, P@K) and "causal" attribution (LDS, counterfactual data removal) ↔ observational correlation vs. causal effect estimation in econometrics.

**类比深度**: **中层** — The analogy is structurally informative but not mathematically isomorphic. LDS measures the effect of data removal (an intervention), analogous to average treatment effect (ATE). RepSim measures representational similarity (an observational quantity), analogous to observational correlation. The gap between RepSim P@K and RepSim LDS would correspond to confounding bias in causal inference.

**已有解法**: Instrumental variable (IV) methods in econometrics address the gap between observational and causal quantities by finding variables that affect treatment (data inclusion) but not outcome (model prediction) except through treatment. In TDA context: data ordering during training could serve as a quasi-instrument (affects which data point is included in gradient update at time t, but is otherwise random).

**可借鉴洞察**: The correlation-causation gap (H-IF-LLM4) may be addressable through quasi-experimental designs rather than brute-force retraining (LDS). If RepSim high + LDS low, the "confounding" is from shared representation structure — representation similarity picks up shared features (confounder) alongside true influence (causal effect).

### 未被利用的工具

- **Fisher Information metric in representation space**: Instead of cosine similarity (RepSim) or gradient norms, use the Fisher Information Matrix in representation space as a natural metric for measuring "distance" between data points in the model's learned representation. FIM accounts for the local geometry of the loss landscape in representation space, potentially capturing causal influence better than cosine similarity. **引入障碍**: Computing FIM in representation space is O(d^2) where d ~ 4096, manageable but requires second-order derivatives through the model.

- **Common-mode rejection ratio (CMRR) as a diagnostic metric**: Signal processing uses CMRR to quantify how well a differential detector removes common-mode signals. Define CMRR for TDA: ratio of task-specific attribution signal power to common-mode (pre-training) attribution signal power. This gives a single number quantifying FM2 severity per task, enabling principled comparison across DATE-LM tasks. **引入障碍**: Requires operationalizing "common-mode signal" — could use base model (before fine-tuning) attributions as reference, per DDA's approach.

### 跨域盲点与教训

- **Matched filter optimality requires known signal shape**: In signal processing, the matched filter is optimal only when the signal shape is known. In TDA, the "signal" (task-relevant influence) has unknown shape. Representation-space methods work IF the learned representations align with the task-relevant signal — but this alignment is not guaranteed. For example, factual attribution may depend on lexical features poorly captured by deep representations (TrackStar's BM25 finding). **Lesson**: Don't assume representation space is universally the right projection — it may be task-dependent, and for some tasks, shallow features (lexical, syntactic) may carry more causal influence than deep representations.

- **Differential detection fails when common-mode is non-stationary**: If the common-mode signal (pre-training knowledge contribution) varies across training samples, subtraction doesn't cleanly remove it. In LLMs, pre-training knowledge IS non-stationary — different training samples interact with different pre-training knowledge. This means DDA's debias may systematically over-correct for some samples and under-correct for others. **Lesson**: The FM2 correction may need to be sample-adaptive, not a global subtraction.

- **Independent decomposition assumption in signal processing**: The assumption that matched filtering and differential detection are independent rests on noise being uncorrelated with common-mode signal. In neural networks, this is unlikely — the representation space (which addresses FM1) is shaped by pre-training (which causes FM2). They are structurally coupled through the shared model. **Lesson**: The 2x2 interaction term may be non-negligible, and this doesn't mean the framework is wrong — it means the framework needs a coupling term, which is a RICHER theoretical contribution than clean independence.

### 建议引入路径

1. **Introduce CMRR as a diagnostic metric**: For each DATE-LM task, compute CMRR = Var(task-specific attribution) / Var(common-mode attribution) using base model vs. fine-tuned model attributions. This quantifies FM2 severity per task and predicts where contrastive scoring will help most. This is a 1-day implementation on top of existing DDA code.

2. **Test the "stationarity" assumption of FM2**: Compare DDA's global debias vs. a sample-adaptive version where the reference is the K nearest neighbors in representation space rather than a global base model. If sample-adaptive debias outperforms global, FM2 is non-stationary — a finding with theoretical significance.

3. **Reframe the 2x2 interaction term as a theoretical contribution, not a failure mode**: If FM1 and FM2 corrections interact strongly, this reveals structural coupling between representation geometry and pre-training bias — a finding that deepens the diagnostic framework rather than invalidating it.

### 继续的最强理由

The signal processing analogy is genuinely deep (not just metaphorical) and provides testable predictions about when the three-bottleneck decomposition holds vs. breaks. This theoretical grounding elevates CRA above a pure benchmark paper.

### 最危险的失败点

The "matched filter" analogy breaks if representation space does not align with task-relevant signal (e.g., factual attribution depending on lexical features). Task-dependent failure of RepSim would expose this.

### 被施压的假设

H3 (FM1 and FM2 approximately additive) — Signal processing theory provides conditions under which this holds (uncorrelated noise, stationary common-mode). These conditions may not hold in neural networks due to structural coupling between representation geometry and pre-training.

### 探针一致性检查

Probe not executed. The signal processing analogy predicts RepSim should work best on tasks with high FM2 (toxicity filtering) and worst on tasks where deep representations don't align with causal signal (factual attribution). This prediction is testable but untested.

### 推荐判定

**Pass** (conditional) — The formalization is intellectually strong. The three-bottleneck framework has genuine theoretical grounding via signal processing, the gap is real (5 independent methods, no common benchmark), and the 2x2 ablation is an elegant experimental design. However, the unexecuted probe is a strategic risk — I recommend proceeding to design WITH the probe as the first design-phase experiment, not a blocker for formalization approval. The formalization is sound; it's the empirical validation that's missing, and that's design's job.

Conditional on:
1. Acknowledging the MAGIC invalidation risk explicitly in the problem-statement with a concrete decision rule.
2. Reframing the 2x2 interaction as potentially informative (coupling between representation geometry and pre-training) rather than purely a failure mode.
3. Adding the "Towards Unified Attribution" (2501.18887) to the competitive landscape discussion.
