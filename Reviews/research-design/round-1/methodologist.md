## [Methodologist] 方法论审查者视角

### Baseline 公平性

**评估**：基本公平，有若干需注意的细节。

**公平标准审查**：

| 公平标准 | 状态 | 说明 |
|---------|------|------|
| 计算预算匹配 | OK | All methods use same fine-tuned model checkpoint; scoring computation varies by method but this IS the comparison variable |
| 超参搜索对等 | **需注意** | RepSim has layer selection (2 choices: L/2, L); TRAK has projection dimension; RepT has phase-transition detection. These are NOT equivalent search spaces. Need to document search budgets per method. |
| 数据增强统一 | OK | All methods use same DATE-LM data pipeline |
| 模型大小匹配 | OK | All methods use same base model (Pythia-1B). No additional parameters. |
| Training schedule 匹配 | OK | All methods evaluate the same fine-tuned checkpoint |
| 代码版本 | **需注意** | TRAK uses DATE-LM's implementation (current). RepT is reimplemented — need to verify against RepT's original code/results on their benchmarks before using in CRA. DDA is also reimplemented (TRAK_ft - TRAK_base). |

**缺失关键 baseline**：

1. **EK-FAC**: Excluded with justification ("DATE-LM already benchmarks it; TRAK + Grad-Sim + MAGIC span the quality range"). This is acceptable IF DATE-LM's EK-FAC numbers are cited. However, **EK-FAC represents a different Hessian approximation quality tier** (between TRAK and MAGIC). Without it, the Hessian error bottleneck is characterized by only two points (TRAK and MAGIC), which is sparse. **Recommend**: At minimum cite DATE-LM's EK-FAC results; ideally run EK-FAC for completeness.

2. **LESS**: Excluded for scope. Acceptable — it's a task-specific variant.

3. **A "simple but strong" baseline**: The experiment design includes BM25 (lexical) and Random (lower bound), which is good. However, **missing a "mean representation similarity" baseline** — i.e., averaging training set representations and computing distance to test representation. This would test whether per-sample attribution is necessary vs. just measuring proximity to training distribution. If mean-rep-sim is competitive, per-sample RepSim is overengineered.

### 指标合适性

**主 Metric (LDS)**：Appropriate for the core claims about counterfactual attribution quality. LDS measures Spearman correlation between predicted and actual model output changes when training subsets are removed. This is the gold standard for TDA.

**Gaming risk**: Moderate. LDS can be gamed if a method specializes in correctly ranking the most/least influential samples but fails on the middle. The Spearman correlation is somewhat robust to this, but **AUPRC on toxicity filtering measures a complementary aspect** (top-K precision). The dual metric design is well-thought-out.

**指标-Claim 对应**：

| Claim | Metric | Coverage |
|-------|--------|----------|
| FM1 diagnosis (RQ1) | FM1 main effect in 2x2 | Complete |
| FM2 diagnosis (RQ1) | FM2 main effect in 2x2 | Complete |
| Rep-space benchmark (RQ2) | LDS, AUPRC, Recall@50, MRR | Complete |
| FM1-FM2 independence (RQ3) | Interaction term in 2x2 | Complete |
| LoRA specificity | RepSim advantage under LoRA vs Full-FT | Complete |
| Correlation vs causation | P@K vs LDS gap | **Partial** — no formal metric for this gap |
| Efficiency | GPU-hours, peak memory | Complete |

**建议补充指标**：
1. **Kendall's tau** alongside Spearman for LDS — Kendall's tau is more robust to outliers and ties, which are common in attribution scores.
2. **Per-quartile analysis**: Break LDS into quartile contributions (most influential, least influential, middle) to understand WHERE methods differ.

### 消融设计

**覆盖率**: Good overall. The 2x2 design {param, repr} x {standard, contrastive} covers the core FM1/FM2 decomposition. The LoRA vs Full-FT dimension adds FM1 generality testing.

**缺失消融**:

1. **Layer ablation for RepSim**: method-design.md §5 mentions middle layer (L/2) and last layer (L). But the experiment design does not include a systematic layer sweep as a required ablation — it's listed under "Scientific Discovery (Conditional)" in §8.1. **This should be promoted to a required ablation** because layer selection is RepSim's only hyperparameter, and its sensitivity directly affects reproducibility. If someone uses a different layer, will they get different conclusions?

2. **Contrastive scoring reference ablation**: The design uses M_base (pre-fine-tuning checkpoint) as the contrastive reference. But there could be alternative references: (a) a randomly initialized model, (b) a model fine-tuned on unrelated data. Testing at least one alternative reference on one task would strengthen the claim that base-model subtraction specifically addresses FM2 vs. just adding noise/regularization.

3. **Parameter-matched ablation for RepT**: RepT uses concat[h^(l*), nabla_h L], effectively doubling the feature dimension from d to 2d. While RepT doesn't add model parameters, the doubled scoring dimension could improve results simply through richer features, not through the gradient component specifically. **Need**: RepSim at layer l* with dimension-matched random features (concat[h^(l*), random(d)]) to isolate the gradient contribution.

**超参敏感性分析**:

Partially addressed. TRAK projection dimension and RepSim layer are noted. However:
- **Missing**: CMRR sensitivity to contrastive reference quality (what if base model checkpoint is from earlier in pre-training?)
- **Missing**: RepT phase-transition detection sensitivity (what if the detection algorithm picks a suboptimal layer?)
- **Missing**: Full-FT learning rate sensitivity (§3.3 mentions sweeping {1e-5, 5e-5, 1e-4}, which is good, but need to verify that method rankings are robust across the sweep)

### 可复现性

**评分**: 中-高

**Strengths**:
- DATE-LM benchmark is open-source with standardized evaluation
- 3 seeds per condition
- Compute budget fully specified
- Model (Pythia-1B) is public and deterministic given seeds

**缺失信息**:
- Exact DATE-LM commit hash / version to use
- RepSim and RepT implementation details (which token position to extract representations from — CLS token? last token? mean pooling?)
- Contrastive scoring normalization details (subtract raw scores? normalize before subtraction?)
- MAGIC implementation: which checkpointing strategy (every step? every N steps?)

### 数据污染风险

**风险**: 低

- Pythia-1B is trained on The Pile, which predates DATE-LM's construction
- DATE-LM uses UltraChat (generated data) for toxicity and FineWeb for data selection — no direct overlap with Pile test sets
- Factual attribution uses ROME entity-fact pairs — these facts ARE in Pythia's pre-training data by design (the task is to attribute which training data taught the model a fact)
- **No temporal leakage risk**: all methods evaluate the same checkpoint, no train/test temporal ordering

**One caveat**: DATE-LM's evaluation computes LDS by actually retraining models with subsets removed. This retraining must use the same random seed protocol across all conditions. If different methods induce different data removal patterns that interact with the retraining seed, there could be a subtle confound. DATE-LM's protocol should handle this, but worth verifying.
