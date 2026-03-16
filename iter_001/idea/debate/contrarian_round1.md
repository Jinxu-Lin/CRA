# Contrarian 交叉批评 — Round 1

**角色**: Contrarian（反对者，魔鬼代言人）
**日期**: 2026-03-16
**任务**: 对 innovator / pragmatist / theoretical / interdisciplinary / empiricist 五份 perspective 进行尖锐批评

---

## 总体声明

本轮批评的立场：**所有 perspective 都在同一个核心病症上犯了不同程度的错误——它们将 FM1/FM2 这对诊断框架预设为正确，然后各自精心地为其服务，却没有人真正挑战这个框架是否是最有力的解释。** 这正是 Contrarian 的工作：不是锦上添花，而是摸清楚什么地方会垮掉。

---

## 1. Innovator Perspective

### 评分：**6 / 10**

### 亮点
三个角度（Spectral Scalpel、DiD Attribution、IB Lens）都有真正的跨域迁移思维，不是简单的"做更多实验"。Angle 1 的 r_eff ~ d 预测是本次五份文档中最具体、最可证伪的定量假设之一。

### 尖锐批评

**（1）创新性高估，执行风险严重低估**

Innovator 给 Angle 1 打了 70% 成功概率，但这个数字过于乐观。关键问题：在 Pythia-70M（70M 参数）上计算梯度协方差矩阵的精确全谱是一回事；在 Pythia-1B 上通过 Lanczos 做 top-500 近似又是另一回事。Lanczos 方法对数值精度极度敏感，LLM 梯度协方差矩阵的条件数往往超过 10^6（Li et al. 2512.09103 已指出"谱放大"问题），这意味着 Lanczos 迭代极易提前收敛到伪特征向量。**Innovator 对此没有任何缓解措施。**

**（2）Angle 2（DiD Attribution）可能是新瓶装旧酒**

DiD 估计量的关键假设是 "平行趋势"（parallel trends）——pre-training 阶段的表示相似度与 fine-tuning 阶段的因果归因无关。但 LLM 中这个假设几乎肯定违反：一个 training sample 越接近 test sample 的语义（pre-training 相似度高），它在 fine-tuning 阶段被模型改变的方式也越接近（post-training 相似度也高）。换言之，pre-training 相似度**本身就是因果归因的预测因子**，平行趋势假设直接崩溃。Innovator 在成功概率一栏诚实地写了 55%，却没有追问：如果平行趋势违反，DiD 会产生系统性的**反向偏差**（比单差更差）——这种失败模式才是真正危险的，不只是"边际增益很小"。

**（3）Angle 3（IB Lens）理论上最动人，但 LLM 的 IB 争议根本没被认真对待**

Innovator 提到了 Shwartz-Ziv & Tishby 争议，但只是一带而过。核心问题是：Tishby 的 IB 理论（2017 版）在 ReLU 网络上基本被 Saxe et al.（2019）证伪——信息压缩相变只在特定激活函数（tanh）下出现。Transformer 的 LayerNorm + Softmax + Attention 是远比 ReLU 更复杂的非线性，没有任何证据说明 IB 理论在 decoder-only LLM 上适用。用 MINE 在 4096 维空间里估计互信息本来就是已知的高方差问题（MINE overfit 小批量的现象在文献中有充分记录）。Angle 3 成功概率 45% 的估计实际上可能还过于乐观，**更可能的结果是"实验产生了大量噪声但无法得出结论"**，而不是干净的正负结果。

**（4）优先级建议本末倒置**

Innovator 建议优先级 Angle 1 > Angle 3 > Angle 2，这是合理的。但所有三个角度都预设了 FM1/FM2 诊断框架是正确的。如果 FM1 的真正来源是 Hessian 近似误差（Contrarian perspective 的核心挑战），那么 Spectral Scalpel 测量的有效秩无论结果如何都无法区分"FM1 是信号稀释"和"FM1 是 Hessian 偏差的谱效应"——因为 Innovator 的实验里根本没有控制 Hessian 质量的对照组。

**潜在盲点**: 全部三个角度都依赖 DATE-LM 的 3 个任务作为验证集。3 个任务完全不足以区分"FM1/FM2 是普遍规律"和"FM1/FM2 在特定任务结构下成立"。Innovator 对此没有任何讨论。

---

## 2. Pragmatist Perspective

### 评分：**7 / 10**

### 亮点
这是五份文档里工程细节最扎实的。5 个工程风险的逐一识别（TRAK 内存、DATE-LM 数据加载、负样本选择、层选择、统计检验）是对实际执行的重要贡献。关键在于 Pragmatist 真正考虑了"什么会让实验不成功"，而不仅仅是"实验成功了会证明什么"。

### 尖锐批评

**（1）评分 85% 的乐观是一个红旗**

Pragmatist 给 Angle 1（Hardened 2x2 Ablation）打了 85% 的成功概率，并说"FM1 和 FM2 的主效应几乎肯定会显示预期模式"。这种乐观背后的逻辑是循环的：正是因为我们从文献中已知 RepSim 比 TRAK 好、对比打分比标准打分好，所以才做这个实验——但这意味着实验不是在**检验**假设，而是在**重复已知事实**。2x2 的新贡献在于正交性（interaction term），而 Pragmatist 自己也承认这才是风险所在。**85% 的整体成功概率遮蔽了"正交性测试通过"的真实概率可能只有 50-60%**。

**（2）最深的工程风险没有被列入 5 个风险之中**

Pragmatist 列出的 5 个风险都是技术性的（内存、数据格式、层选择等）。但最危险的风险是**概念性的**：用 LoRA 梯度代替全参数梯度作为 TRAK 的 phi 输入，本质上已经在做"维度约化到低秩子空间"——这正是 FM1 的理论解法！如果 LoRA-TRAK 已经部分缓解了 FM1，那么 2x2 矩阵中"参数空间 + 标准打分"这个 cell 就不是一个干净的"未修复 FM1"基准，而是一个已经部分修复了 FM1 的混合方法。这会严重混淆 FM1/FM2 主效应的解读。**Pragmatist 提到了这个问题（Risk 1a），但把它当成一个可接受的 mitigation，而不是一个破坏性的实验设计缺陷。**

**（3）Angle 3（Contrastive Plug-in）的"FM2 Severity Index"定义是循环的**

FM2 Severity Index = (contrastive_score - standard_score) / standard_score，然后用它来"量化 FM2 严重程度"。但这个指标本质上只是量化了"对比打分相对于标准打分的改进幅度"，无法区分三种可能的解释：(a) FM2 更严重所以改进更大，(b) 标准打分对这类方法本身更弱（与 FM2 无关），(c) 对比打分的负样本选择策略在这个任务上碰巧更有效。**用改进幅度作为"失败模式严重程度"的代理指标，在逻辑上是不完整的。**

**（4）它的主要价值是执行，而不是创新**

Pragmatist 的三个角度与 Innovator、Theoretical 的想法高度重叠（2x2 在每个 perspective 里都出现了），唯一独特贡献是工程清单。这是有价值的，但如果要问"哪个 perspective 推动了研究前沿"，Pragmatist 基本上是在优化执行质量，而不是提出新问题。

**潜在盲点**: 在 36-cell 的对比打分矩阵里（4 methods × 3 contrastive variants × 3 tasks），没有讨论多重比较校正（multiple comparisons correction）。36 个 cell、每个有置信区间，随机涌现出几个"显著"结果的概率相当高，这可能产生误导性的发现。

---

## 3. Theoretical Perspective

### 评分：**7.5 / 10**

### 亮点
这是五份文档中理论深度最高的，特别是 Theorem 7（分类学，phi^T M psi 的系统化）和 Theorem 4（均值减法等价于因果去混淆）是本轮最接近"真正可以写进论文"的理论贡献。Theoretical 明确区分了"定义上正确"的 T6/T7 和"需要实验验证"的 T1-T5，这种诚实是值得肯定的。

### 尖锐批评

**（1）Theorem 1 的推导依赖 NTK 近似，而 NTK 近似在 LLM 上基本无效**

Theorem 1 和 Theorem 2 的核心逻辑链：g_i ≈ J^T δ_i（NTK 近似）→ 梯度协方差的秩约为 d（表示维度）→ FM1 的谱解释。问题在于，NTK 近似在"大量特征学习"的条件下失效，而 LLM 的 fine-tuning 恰恰是典型的特征学习场景（而非内核机制）。Theoretical 在"理论风险"部分自己也承认了这一点，但随后说"可以把定理放在 NTK 假设下，再单独做实验验证"。这是一个学术上的"两手空空"策略：**如果理论的核心假设在实际模型上不成立，那么定理在论文中的地位就变成了"在一个不现实的特例下的正式结果"——这对 ML 会议来说很难过 reviewer 关。**

**（2）Theorem 8（最优 phi/psi 通过 SVD）在实践中几乎无法验证**

Theorem 8 给出了最优 phi_opt 和 psi_opt——它们是 LOO 重训练归因矩阵 A* 的奇异向量。问题：(a) 在 Pythia-1B 上计算真正的 LOO 归因矩阵计算量是不现实的（需要为每个 training sample 重训练），(b) 即使退化到 500 sample 的近似，Empiricist 也只能在 Pythia-70M 上完成，(c) 在 70M 模型上的 SVD 结论能否迁移到 1B 模型是开放问题。**Theorem 8 是理论上最精彩的贡献，却也是实验上最脆弱的一个——它的可测试推论（Corollary 3.1）的成功概率被 Theoretical 自己评估为 medium，但这个评估没有考虑到 LOO 计算的根本可行性边界。**

**（3）Theorem 3 的"共同影响污染"分解是正确的，但不新颖**

Theorem 3 的偏差分解（A = task-task + shared-task + task-shared + shared-shared）在代数上是正确的，也有解释价值。但是，这个分解在因果推断文献中是标准的混淆分解，在机器学习文献中也不陌生（batch normalization 的理论分析就用了类似的均值/方差分解）。在 TDA 领域的应用是新的，但称其为"第一个形式化 FM2 的论证"需要更仔细的文献搜索——Pan et al.（2502.11411）的"去噪表示归因"和 DDA 本身的分析中是否有类似的偏差分解，值得核查。

**（4）理论优先级排序（T7+T3-T4 > T1-T2 > T8 > T5）与论文定位冲突**

如果最高优先级是 Taxonomy（T7），那么 CRA 的主要贡献就是一个"统一框架 + 分类学"——这基本上是一篇综述/立场文章，而不是原创研究。ML 顶会（NeurIPS/ICML）通常对纯分类学贡献持怀疑态度，除非伴随着显著的经验发现或理论突破。**Theoretical 自己最精彩的理论（T1 谱上界）被放在了第二优先级，但没有充分讨论为什么它在实验中可能失败——是因为 NTK 假设违反，还是因为谱测量本身有噪声？**

**潜在盲点**: Prediction 3.1 预测"表示向量与 LOO 归因矩阵奇异向量的余弦对齐度 > 0.8"——这个预测太强了。如果表示空间的"归因能力"来自于捕捉了任务相关的语义维度，那么对齐度可能只有 0.4-0.6，仍然比梯度空间（< 0.3）好得多，足以支持 CRA 的论点，但 < 0.8 的结果在 Prediction 3.1 框架下将被算作"预测失败"。

---

## 4. Interdisciplinary Perspective

### 评分：**4.5 / 10**

### 亮点
Angle 4（Matched Filter）是五份文档中第二强的即时可执行想法——Whitened RepSim（M = Σ_noise^{-1}）有理论保证（Neyman-Pearson 最优性），实现直接（协方差矩阵估计 + 矩阵求逆），是对 phi^T × psi 框架真正有算法层面贡献的想法。如果其他角度全部失败，Whitened MF 作为一个独立的增量贡献是站得住脚的。

### 尖锐批评

**（1）四个类比中三个（神经科学、统计物理、免疫学）在 1 小时实验约束下根本无法产生可信证据**

- **Angle 1（神经科学/Eligibility Traces）**：需要在 fine-tuning 过程中逐步记录每个 training sample 的中间表示快照 h_t(z_i)。这不只是"在计算量上增加 30%"——它需要修改训练循环、存储中间激活（对 1B 参数模型每步都存储会产生 TB 级数据），在 1 小时内完成是不可能的。**声称"计算成本约 1.5 GPU-hours"是严重低估。** 更根本的是，"eligibility trace"类比成立的前提是 training sample 有明确的"输入时间步"——但在 fine-tuning 中，每个 sample 在每个 epoch 都会被访问，不存在单一的"触发时刻"t_i，这使得类比的基础直接崩溃。

- **Angle 2（统计物理/RG）**：预测 d_eff(L) 遵循幂律衰减，具体指数 α 因任务而异。但用 MINE 估计互信息本身在高维空间（d > 1000）中方差极高，LLM 表示的高度各向异性会进一步加剧这个问题。更重要的是，**"幂律"是一个极其宽泛的预测**——几乎任何单调递减曲线都能被拟合成"近似幂律"，这个预测实际上无法被证伪。成功概率 45% 是过高估计，更可能的结果是"测量到了某种递减，但无法区分幂律与指数衰减，也无法测量到有意义的 critical plateau"。

- **Angle 3（免疫学/自我/非自我区分）**：两阶段去偏的 Stage 2（peripheral tolerance，sigmoid 抑制）引入了两个超参数 β 和 τ，选择这些超参数需要验证集，但 DATE-LM 的 3 个任务中很难有足够的"验证集"来可靠地选择超参数而不产生过拟合。更根本的是，Stage 2 的改进能否超过 Stage 1（简单均值减法）的边际，本身就是实验问题——但免疫学类比提供的"理论理由"（peripheral tolerance 必然优于 negative selection alone）在 TDA 的上下文中并不成立，因为免疫系统的两阶段设计是进化出来针对一个非常特定的问题（自身免疫风险与感染风险的权衡），而不是针对"pre-training 知识污染"的最优设计。

**（2）"统一公式"A_unified 是一个过度工程化的特洛伊木马**

文档末尾的"统一公式"：
```
A_unified(z_test, z_i) = [phi - phi_shared]^T Σ^{-1} [psi - psi_shared] * gate(z_test)
```
将均值减法（FM2 修复）、白化（matched filter 最优性）、门控（神经调制/免疫 tolerance）三者组合。这个公式在概念上是自洽的，但在实验中引入了太多未分解的变量。如果这个方法比 RepSim 好，无法知道是哪个成分起了作用；如果它不好，也无法知道是哪个成分有害。**CRA 的整个论证依赖于 FM1 和 FM2 修复的正交性——但这个"统一公式"恰好引入了多个相互纠缠的修复，直接破坏了因果推断的可能性。**

**（3）跨域类比的深度被系统性地高估**

Interdisciplinary 在每个角度里都声称该类比是"不只是隐喻"而是"形式上同构"。但仔细审视：
- 神经科学类比要成立，需要 LLM 中存在"synapse-specific eligibility trace"，而 LLM 的 parameter space 是完全 entangled 的（every parameter potentially encodes information about all training samples）——这与 neuroscience 的 synapse-specific 假设是根本对立的。
- RG 类比要成立，需要 LLM 的层级结构满足"block-spin coarse-graining"的马尔可夫性——但 Transformer 的 attention 机制是全局的，每层可以直接"看到"所有位置的信息，这与 RG 的局部性原理根本矛盾。

这两个类比的"不只是隐喻"声明需要更仔细的数学论证，而不是用"最近文献表明..."来一带而过。

**潜在盲点**: Interdisciplinary 完全没有讨论 Contrarian 的核心挑战（Hessian 误差 vs. FM1/FM2）。这不是一个小疏漏——它意味着四个角度都建立在一个可能是错误的基础上。

---

## 5. Empiricist Perspective

### 评分：**8 / 10**

### 亮点
这是五份文档中最接近"可以直接用于驱动实验设计"的一份。四个混淆因素的系统识别（实现不对称、归一化不对称、层选择、负样本选择）以及"预先注册证伪准则"是本轮五份文档中唯一真正遵循严格实验设计原则的内容。关键贡献是 BM25 作为词汇控制的强烈建议——这是所有其他 perspective 都忽视的真正威胁。

### 尖锐批评

**（1）Angle 3（Multi-Method Tournament）低估了 LOO Oracle 的可行性边界**

Empiricist 计划在 500 个训练样本上计算 LOO Oracle，然后将结论迁移到 Pythia-1B（d=2048）的 attribution matrix。问题：500 × 500 的归因矩阵的 SVD 是统计上不可靠的——LOO retraining 500 次在 Pythia-70M 上每次需要约 2-5 分钟，总计超过 16 小时，根本不在 1 GPU-hour 约束范围内。**Empiricist 将这个实验的计算成本估计为 5 GPU-hours，这显然是基于"LOO 重训练在 70M 模型上很快"的假设，但没有说明具体训练多少步/epoch，也没有说明用什么标准确认"模型已经重训练到位"。** 如果用近似的 IF 来代替真正的 LOO，那 oracle 本身就是有偏的，SVD 分析的结论也会受污染。

**（2）"BM25 会打败归因方法"的预测没有被当成实验设计的起点**

Empiricist 在"Negative Results I Expect"中预测 BM25 会在事实归因任务上击败所有归因方法，然后说"论文应该坦诚承认这一点"。但如果这是预期结果，**实验设计本身需要解释为什么 CRA 的贡献不依赖于 TDA 在事实归因任务上优于 BM25**——否则这个"诚实的限制讨论"会变成 reviewer 的核弹。Empiricist 没有提出如何在论文定位层面回应这个挑战：是聚焦于数据选择和毒性过滤任务？是论证 BM25 在 OOD 设置下会失败？还是重新定义 TDA 的价值主张？

**（3）双向 ANOVA 在 3 个任务上的统计功效分析缺失**

Empiricist 提议用两因素 ANOVA 检验 FM1/FM2 正交性（interaction term < 30% of min main effect），但对 3 个任务来说，这个设计的统计功效几乎为零。用 bootstrap CI（B=1000 resamples of test queries）处理的是**查询层面**的变异，而不是**任务层面**的变异。FM1/FM2 正交性是**跨任务的**声明，需要任务作为统计单元——但 3 个任务意味着 n=3，任何统计检验都缺乏功效。**Empiricist 在统计方法上最严格，却在最关键的地方——跨任务推广的统计功效——留了一个盲点。**

**（4）"Exact-Hessian IF 控制"被列为 P2（第二优先级），但它应该是 P1 的一部分**

Empiricist（受 Contrarian 启发）提议了精确 Hessian 控制实验，并正确地说"如果精确 Hessian IF 在 Pythia-70M 上等于 RepSim 性能，整个项目就要 pivot"。但这个实验被放在了 P2，排在 2x2 Factorial（P1）之后。逻辑上，这个控制实验应该优先于 2x2 Factorial——否则是在没有确认基础假设成立的情况下就开始收集证据。**这个排序错误可能导致花费大量计算和时间在 2x2 矩阵上，最后发现整个框架需要 pivot。**

**潜在盲点**: Empiricist 对 CKA 病理学（Davari et al., Cloos et al., Okatan et al.）的提及只在参考文献中，没有被整合进实验设计。如果 RepSim 继承了 CKA 的病理（高全局相似度但子空间行为不同），那么 RepSim 的"好性能"可能部分来自于它检测到了高方差主成分而不是因果归因信号——这需要通过一个控制实验（在归因最相关的子空间上计算相似度 vs. 全空间）来检验。

---

## 跨 Perspective 系统性盲点

以下问题在所有五份文档中均未得到充分处理：

1. **FM1/FM2 框架的可证伪性**：没有任何 perspective 设计了一个"如果 FM1/FM2 是错误的，实验结果会是什么样"的完整思想实验。Contrarian 的精确 Hessian 控制是唯一真正的证伪尝试（Empiricist 提到了，但排在 P2）。

2. **DATE-LM 的 3 个任务是否覆盖了重要的 TDA 应用场景**：所有 perspective 都在 DATE-LM 上做文章，但 DATE-LM 的任务设计本身可能对表示空间方法有内在的偏向（DATE-LM 数据集的选取和设计者可能已经知道表示方法更好，并针对性地选择了任务）。这是一个潜在的 benchmark 选择偏差，没有人讨论。

3. **"phi^T × psi 双线性框架的意义"**：Contrarian 挑战了这个框架是否是平凡的（一切线性方法都是双线性）。没有任何 perspective 正面回应了这个挑战——它们都在假设框架有意义的前提下推进，而不是先论证框架为什么有意义。

4. **计算成本的现实性**：多个 perspective 给出的 GPU-hour 估计（特别是 Innovator 的 Lanczos 特征分解、Interdisciplinary 的 ETA 和 RG 实验）在 1 GPU-hour 约束下根本不可行，但这个约束在五份文档里都没有被严肃对待。

---

## 总分汇总

| Perspective | 总分 | 核心优点 | 核心弱点 |
|---|---|---|---|
| Innovator | 6/10 | 跨域迁移有真实内容，假设可证伪 | 计算成本低估，NTK/IB 理论适用性存疑，无 Hessian 对照 |
| Pragmatist | 7/10 | 工程细节扎实，风险识别清晰 | LoRA 梯度已部分修复 FM1（实验设计污染），FM2 指标定义循环 |
| Theoretical | 7.5/10 | 理论深度最高，Taxonomy 贡献实质 | NTK 假设在 LLM 上失效，T8 可验证性差，部分"理论"是已知结果 |
| Interdisciplinary | 4.5/10 | Matched Filter（Angle 4）有真实价值 | 三个类比在 1 小时约束下不可执行，统一公式破坏因果可分性，类比深度被高估 |
| Empiricist | 8/10 | 实验设计最严格，BM25 警告关键 | LOO Oracle 可行性边界低估，Exact-Hessian 控制优先级排序错误，跨任务统计功效为零 |

---

## Contrarian 的最终裁定

如果只能选一件事让所有 perspective 重做，那就是：**先运行 Exact-Hessian IF 控制实验（在 Pythia-70M 上，1K 训练样本，精确 Hessian），然后根据结果决定整个实验计划。** 如果精确 Hessian IF 接近 RepSim 性能，整个 FM1/FM2 框架需要修订，所有五份文档建立的上层建筑都需要重新搭建。如果精确 Hessian IF 仍然远低于 RepSim，FM1/FM2 的基础故事才真正成立。

**在这个关键控制实验运行之前，Innovator 的 Spectral Scalpel、Theoretical 的 T1-T2、Empiricist 的 Dimension Sweep——所有依赖 FM1 诊断的实验——都是建立在沙滩上的城堡。**
