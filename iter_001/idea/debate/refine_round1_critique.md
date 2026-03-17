# REFINE 轮第 1 轮交叉批评报告

**辩论类型**: 证据驱动精炼轮（第 2 次 idea_debate，pilot 实验已完成）
**评审基准**: Pilot 结果（N=100，Pythia-1B，DATE-LM 三任务）
**评审日期**: 2026-03-17

---

## Pilot 实验关键事实（共识基础）

在对各 perspective 打分前，先明确所有评审必须面对的实验事实：

| 假设 | 状态 | 核心证据 |
|------|------|----------|
| H1 (FM1 空间差距) | 通过 2/3 | RepSim > TRAK +32pp (counterfact), +17pp (ftrace)；**毒性任务逆转 -24pp** |
| H2 (对比评分不对称) | 无法判定 | 所有 12 个 method-task 组合增益恰好 0.0pp（rank 指标对均值平移不变） |
| H3 (FM1/FM2 正交) | 平凡满足 | FM2 效果 =0 → 无统计功效，交互项无意义 |
| H4 (梯度 r_eff ~ d) | 方向支持 | r_eff=10（全模型），远低于预测 [256,1024]；**FM1 比预期更严重** |
| H5 (TRAK 饱和于 k~d) | 支持 | k=256 饱和，但 TRAK-PCA@k=d 距 RepSim 仍差 **30.8pp** |
| H6 (K-FAC 控制) | 确认 | RepSim > K-FAC IF +17.4pp (counterfact)；毒性任务梯度范数伪信号 |
| H7 (白化归因) | **彻底失败** | 所有任务下降 8-11pp（N/d=0.049，协方差欠定） |
| H8 (RepSim > BM25) | 部分 | 毒性/ftrace 通过；BM25 在 counterfact 完美（pilot 规模） |
| H9 (各向同性) | **被否证** | 方向完全反转：rep_cond=3.1e10 >> grad_cond=3,589 |

---

## 各 Perspective 评分与分析

---

### 1. Innovator（创新者）——评分：**6.5/10**

#### 优点

**对 pilot 证据的回应有亮点**。Innovator 在 perspective 撰写时已预见了"固定层可能非最优"的问题（LABA Angle 1），这与 pilot 中毒性任务的逆转隐含地一致：不同任务或许需要不同层的表示。CATCL（Angle 2）作为时序差分估计器，提供了一种**无需均值减法的 FM2 去偏方式**，这在 H2 评测协议失效的情况下尤为有价值——即便 rank 指标无法检测 FM2，CATCL 从机制上绕开了这一评测盲区。共轭性框架（Angle 3）对 H3 的重新诠释则与 pilot 的"FM2 效果 =0→交互项无意义"发现相吻合：如果 FM1/FM2 是共轭的，那么 pilot 中的平凡满足恰好说明系统已在某种权衡点上运行。

#### 不足

**对 H7 失败和 H9 被否证的回应不充分**。Innovator 的 Angle 3 提出"Wiener 滤波器 M = Sigma_total^{-1} 是最优的"，但 pilot 数据显示白化 RepSim 在所有任务下降 8-11pp，而且 H9 否证表明表示空间极度各向异性（条件数 3.1e10）。Innovator 没有解释：在如此极端各向异性下，LABA 的层权重 SNR 估计是否仍有意义？表示空间的各向异性可能使任何基于 M=I 的 SNR 计算完全失真。

**TRAK-PCA 30.8pp 缺口被忽视**。Innovator 假设 LABA 通过多层加权能弥补表示与梯度空间的差距，但若单层 TRAK-PCA@k=d 与 RepSim 之间仍存在 30pp 缺口，多层加权未必有帮助——问题可能不在于层选择，而在于梯度空间本身缺乏表示空间具备的语义结构。

**期望计算成本低估了 H9 的影响**。CATCL 的成功依赖于"平行趋势假设"成立，但在表示空间极度各向异性时，微调前后的表示漂移将高度非均匀，直接威胁平行趋势。Innovator 的缓解方案（IPW-DiD 重加权）思路正确，但未给出具体如何在 Pythia-1B 规模下验证。

#### 对 full-scale 实验的指导价值

**中等**。CATCL 提供了一种独立于 rank 指标的 FM2 检验机制，值得在全规模实验中与连续指标（Kendall tau）并行验证。LABA 的实验计划清晰，风险可控，但需要首先确认 H9 在 N>>d 时是否部分恢复，否则 SNR 权重的理论基础存疑。

---

### 2. Pragmatist（实用主义者）——评分：**8.5/10**

#### 优点

**对工程风险的预见性最强，且与 pilot 结果高度吻合**。Pragmatist 在 pilot 之前就警告了：(1) 对比评分的"任务条件均值减法"而非全局均值减法的必要性；(2) 白化归因在 d > n 时的奇异性风险；(3) 毒性任务与数据选择/事实归因在信息结构上的根本差异。这三个预警全部被 pilot 数据证实。

**对 H7 失败的预判最准确**。Pragmatist 明确说明：当 n < 2048 时协方差逆矩阵奇异，必须使用 Ledoit-Wolf 正则化，且即便如此，当 N/d~3 时仍边缘可用。这与 pilot 的 N/d=0.049 失败完全对应。

**工程路线图切实可行**。Phase 0→1→2→3 的渐进式验证方案在 pilot 中被证明是对的——Phase 0 提前发现了毒性逆转，避免了在错误方向上投入全部计算资源。

#### 不足

**对 H9 被否证的后续处理不够深入**。Pragmatist 提到了"始终与脊正则化归因对比"，但在 H9 方向完全反转（表示极度各向异性）的情况下，这一建议仅是补丁。Pragmatist 没有解释：为什么在如此各向异性的表示空间中，M=I 的 RepSim 仍能优于 TRAK？这是 CRA 叙事中最需要解释的矛盾之一。

**对 TRAK-PCA 30.8pp 缺口的分析停留在现象描述**。Pragmatist 建议进行 PCA-TRAK 比较（其 Angle 3 中），但没有提出机制假说——是非线性因素、层混合、还是余弦相似度的特性？这一缺口是全规模实验中必须回答的核心问题。

**对 FM2 评测协议失效（H2/H3 平凡满足）的方案不完整**。Pragmatist 只是在 Phase 4 列了"白化归因"的改进，没有明确提出 Kendall tau 等连续指标作为 FM2 检测的主要手段。这是一个严重的方法论漏洞，在 pilot 结果公布后应已成为最高优先级。

#### 对 full-scale 实验的指导价值

**最高**。Pragmatist 提供的工程路线图是最直接可执行的。其核心建议——先跑核心 2x2，再做机制实验，最后做框架验证——已被 pilot 证明是正确的顺序。在全规模实验中应将 Pragmatist 的"Warning 3（白化可能静默失败）"提升为强制检查项，并补充连续指标作为 FM2 的主要评测手段。

---

### 3. Theoretical（理论派）——评分：**7.0/10**

#### 优点

**Theorem 3（对比评分的 Neyman-Pearson 最优性）提供了对 H2/H3 平凡满足的理论出口**。即使 rank 指标无法检测 FM2 效果，N-P 最优检测器框架仍然正确地预测了：**mean subtraction 去除确定性偏置，白化处理残余噪声的各向异性**——这一框架的正确性不依赖于 pilot 的 rank 指标能否检测 FM2。Theorem 3 是 CRA 框架在 H2 平凡满足下的重要理论支撑。

**对 Hu et al. (2602.10449) 的整合最为深入**。Theoretical 正确地将"有效维度"与"正则化尺度"联系起来，解释了为什么 TRAK 在 k<r_eff 时性能下降，并将此与 Proposition 1.1 的 B/d SNR 优势连接。这为 H4/H5 提供了比单纯实验更强的理论支撑。

**Per-query SNR 可靠性度量是全规模实验的新工具**。即使 Cramer-Rao 下界过于宽松，SNR_out(z_test) 作为每查询置信度估计，可与 pilot 中发现的正相关（r=0.34 于 counterfact）直接对接，值得在全规模验证。

#### 不足

**对 H9 被否证的理论回应存在根本矛盾**。Theoretical 的 Proposition 1.1 的核心假设是"表示协方差近各向同性（kappa ~ 1），因此 M=I 足够"；但 pilot 测量表明 rep_cond=3.1e10（极度各向异性）。Theoretical 需要正面回答这一矛盾：若 Sigma_noise 极度各向异性，为何 M=I 的 RepSim 仍然有效？一种可能的出路是将 Proposition 2.2 的"各向异性时白化增益更大"与"H7 在 N/d<<1 时失败"分离——前者是理论预测，后者是工程限制——但 perspective 文档中没有做这一分离。

**Theorem 5（Cramer-Rao 类下界）的形式化依赖高斯噪声假设，而 pilot 的极高 rep_cond 暗示这一假设可能不成立**。若表示空间的噪声是重尾的（高条件数协方差矩阵通常由少数极大特征值主导），Fisher 信息矩阵的逆并不存在良定义的下界。这是需要在论文中明确讨论的假设违反风险。

**对 TRAK 非单调性的理论解释不完整**。Pilot 显示 TRAK R@50 在 k=256 达到峰值后非单调下降（k=512: 0.750，k=1024: 0.686，k=2048: 0.670，k=4096: 0.715）。Theoretical 的 MSE_projection(k) 公式预测单调递减后饱和，无法解释非单调性。这一非单调性是否来自 TRAK 内部正则化与投影维度的相互作用，是全规模实验需要回答的问题。

#### 对 full-scale 实验的指导价值

**中等偏高**。理论框架为全规模实验提供了可证伪的定量预测（TRAK 维度扫描的相变点、对比评分改进下界、每查询 SNR 相关性）。但 H9 的根本性否证要求理论层在全规模实验前对 M=I 有效性给出新的解释，否则 Proposition 1.1 的应用前提不成立。

---

### 4. Contrarian（反对派）——评分：**8.0/10**

#### 优点

**对 pilot 证据的解读最为犀利，且全部有据可查**。Contrarian 提出的三个核心挑战都直接被 pilot 数据支持：
- 毒性任务 TRAK > RepSim +24pp（挑战 1："FM1/FM2 是信号处理缺陷"）
- H7 失败 + H9 被否证（挑战 2："phi^T M psi 有预测能力"）
- RepSim PCA 饱和于 k=64 而非 k=d（挑战 3："R^d 是信号富集子空间"）

**"RepSim 只是检索"的假说具有高度可证伪性**。BM25 在 counterfact pilot 规模下完美（R@50=1.0），而 RepSim 在同任务达到 0.994——两者几乎一致，这是对 Contrarian 核心假说的强烈佐证。将专用检索模型（Contriever、GTR）纳入 baseline 的建议是 full-scale 实验的强制项。

**三个伪证测试设计精准、成本低**。Euclidean vs. cosine RepSim、随机初始化表示 RepSim、专用检索模型对比——这三个测试总计约 5 GPU 小时，每一个都能单独排除或确认一个重要混淆变量。这比其他 perspective 的大规模实验计划更具优先级。

#### 不足

**"抛弃白化归因贡献"的建议过于极端，忽略了 pilot 的积极信号**。Contrarian 正确指出 H7 在 pilot 规模下失败（N/d=0.049），但没有提到：(1) ftrace 任务上白化对 DDA 的增益 +6.8pp；(2) SNR-准确性正相关（r=0.34 counterfact, r=0.16 ftrace），概念方向正确。在 N/d 充足时，白化可能仍有价值——这是需要全规模实验验证的，而非现在就放弃。

**DDA > TRAK 的反驳（DDA 0.876 vs TRAK 0.926 on toxicity）是最薄弱的论据**。Contrarian 将 DDA 毒性任务不如 TRAK 作为"FM2 correction 无效"的证据，但这混淆了两件事：DDA 的 debias 是基于"减去基础模型影响函数"而非简单均值减法，且毒性任务本身就是梯度范数驱动的（Phase 0 确认 Raw Dot IF 无 Hessian 已达 0.94 AUPRC）。这一任务上的"失败"不是 FM2 修正的失败，而是 FM2 修正对梯度范数驱动任务无效——这恰恰支持了 CRA 的任务类型边界主张。

**建议"将论文重构为系统 benchmark + 事后分析"可能低估了 CRA 框架的理论价值**。即便 H7、H9、H2 在 pilot 规模失败，CRA 的 phi^T M psi 分类学仍然是首次对 5 个表示空间方法的系统统一，加上 2x2 factorial 也是 DATE-LM 上的首次，这在没有理论框架的情况下也是有贡献的。但 Contrarian 的警告——"如果没有任何非平凡预测在全规模存活，则降格为'符号'而非'理论'"——是合理的知识诚信要求。

#### 对 full-scale 实验的指导价值

**高**。Contrarian 提出的三个伪证测试和"检索 baseline 必须纳入"的建议是 full-scale 实验设计中不可忽视的约束。尤其是：若专用检索模型在全规模实验中匹敌或超过 RepSim，则 CRA 的"空间优势"叙事必须大幅修正。

---

### 5. Interdisciplinary（跨领域派）——评分：**6.0/10**

#### 优点

**CFAR 类比在 pilot H2=0 的背景下获得了新的相关性**。H2 零增益的核心原因是均值减法不改变 rank——这在 CFAR 文献中对应"CA-CFAR 在均匀杂波中是最优的，但在非均匀杂波中退化"。OS-CFAR（中位数/截尾均值）提供了一种可能绕过 rank 不变性问题的思路：如果用中位数而非均值作为基准，去偏后的分数分布不同，**rank 不再严格保持不变**，因此可以用 rank 指标检测 FM2。这一洞察具有实际操作价值。

**经济学工具的形式化精度高**。OVB 敏感性分析（Cinelli-Hazlett Robustness Value）将 FM2 严重程度形式化为可数量化的置信度指标，0.5 GPU 小时的成本极低，是全规模实验的强力候选项。DiD 平行趋势检验也为 2x2 factorial 提供了比 30% 阈值更严格的统计标准。

#### 不足

**对 H9 被否证和 H7 失败的回应基本缺席**。Interdisciplinary 在撰写时似乎没有充分整合 pilot 的否证结果。BBP 相变（Angle 4）的实验计划（P1）预测"梯度谱从 Marchenko-Pastur bulk 中分离"，但 pilot 已经测量了梯度协方差特征谱（r_eff=10 for full model, Top-5 捕获 85.6% 方差）——这与 BBP 分离的预测相容，但 Interdisciplinary 没有将 pilot 数据明确连接到 BBP 框架，错失了一次强有力的理论整合机会。

**IV 估计器（工具变量归因，Experiment E2）的排他性限制未充分讨论**。数据增强（同义词替换、改写）作为"不影响任务特定信号但改变共同影响"的工具变量，这一假设在 LLM 表示中极难验证——改写可能改变语义内容（影响 phi_task），而非仅改变表面形式（phi_shared）。对此 Interdisciplinary 只在"Key Risks"中一笔带过，缺乏实验设计层面的处理。

**sqrt(d) 预测（N3，类比 RSA 谱分析）与 pilot 数据存在直接矛盾，但未正视**。Interdisciplinary 预测"归因质量在 d' ~ sqrt(d) ~ 45 时饱和"，而 pilot 的 RepSim PCA 扫描显示饱和于 k=64（N=100 的自然截止，并非真实 sqrt(d) 预测），无法区分是 sqrt(d) 效应还是 N=100 限制。在全规模实验中验证 sqrt(d) vs d 的正确标度预测成本约 1 GPU 小时，应明确列入计划。

#### 对 full-scale 实验的指导价值

**中等**。CFAR 类比对 FM2 评测协议的改进有直接操作价值，OVB 敏感性分析成本低信息量高。但整体上，Interdisciplinary 对 pilot 负面结果的整合不足，部分实验计划（E2 IV 估计）在执行可行性上存疑。

---

### 6. Empiricist（实证派）——评分：**9.0/10**

#### 优点

**对 pilot 证据的解读最为诚实、完整，且方法论诊断最准确**。Empiricist 是唯一明确将"H2/H3 平凡满足"诊断为**评测协议缺陷**而非实验结果的 perspective：

> "This is not 'FM2 doesn't exist' -- it is 'the experiment was unable to detect FM2 with these metrics.' This is a **design flaw in the evaluation protocol**, not evidence for or against FM2."

这一区分对于全规模实验的设计方向至关重要，且是其他 perspective 普遍忽视的。

**受控污染注入（Experiment 2.2）是验证 FM2 机制的最干净设计**。通过人工注入已知强度的共同影响偏置（alpha * mean_representation），并测量对比评分是否能恢复 alpha=0 时的性能，这一实验的因果解释无歧义，且成本仅约 2 GPU 小时。这是整个 6 个 perspective 中最具操作价值的单个实验提案。

**对 30.8pp TRAK-PCA 缺口的系统分解（Experiment 3.3）是 CRA 论文必须回答的问题**。通过分别测试"仅最后层梯度 TRAK-PCA"和"余弦归一化 TRAK-PCA"，可以将缺口来源量化分配到层混合、归一化、语义特征等因素，直接回应 Contrarian 的挑战。

**对于需要诚实报告的阴性结果提供了明确列表**。毒性逆转、BM25 竞争力、H9 否证、30.8pp 缺口——Empiricist 明确要求这些都在论文中如实报告，这是学术诚信的基本要求，也是提高审稿通过率的策略。

#### 不足

**对理论框架的功用估计过低**。Empiricist 的建议是"将 phi^T M psi 作为组织工具而非预测理论"，并将全规模实验设计中的高优先级全部给了纯实证实验（2x2, eigenspectrum, dim sweep）。但 pilot 发现的正向信号（SNR-准确性相关 r=0.34, K-FAC 控制实验通过, 梯度协方差低秩性强确认）表明框架并非完全无预测能力——Empiricist 对这些正向信号的权重不足，导致建议可能过于保守。

**全规模实验总 GPU 小时估计（33h）中，约 40% 是多方法锦标赛（8-10h）和 DDA 消融（4h）**，这两个实验的优先级定为"Medium"，但其对验证"phi^T M psi 框架预测能力"至关重要。若计算资源受限，这部分实验可能被优先裁撤，但那样会使框架验证不完整。Empiricist 应更明确地区分"证伪框架所需的最小集"和"支持框架预测能力所需的完整集"。

**对 Innovator 提出的 CATCL 和 LABA 等新方法没有给出评估**。Empiricist 专注于验证现有假设，没有讨论是否值得在全规模中纳入新方法变体，这使其建议在范围上有所局限。

#### 对 full-scale 实验的指导价值

**最高**。Empiricist 提供的 9 个实验的优先级表和 13 GPU 小时最小可行集是最直接可操作的实验计划。其对评测协议缺陷的诊断（H2 需要连续指标，H7 需要 PCA 降维白化，H9 需要 N>>d 重测）是全规模实验设计的核心约束，所有其他 perspective 的提案都应在此框架内被评估优先级。

---

## 综合打分汇总

| Perspective | 对 pilot 证据的回应 | 修正方案可行性 | 对全规模实验的指导价值 | 综合评分 |
|-------------|---------------------|---------------|----------------------|---------|
| Innovator | 部分（忽视 H9/H7 根本矛盾） | 中（CATCL 有风险，LABA 需重建理论基础） | 中 | **6.5/10** |
| Pragmatist | 优秀（提前预见大多数风险） | 高（工程路线图直接可用） | 最高 | **8.5/10** |
| Theoretical | 良好（Theorem 3 有价值，但 H9 矛盾未解） | 中（需修订 M=I 有效性的理论基础） | 中高 | **7.0/10** |
| Contrarian | 卓越（挑战全部有 pilot 数据佐证） | 中（部分建议过于极端） | 高 | **8.0/10** |
| Interdisciplinary | 欠佳（对 H9/H7 的回应基本缺席） | 中（CFAR/OVB 工具有价值，IV 有风险） | 中 | **6.0/10** |
| Empiricist | 卓越（最诚实完整的 pilot 解读） | 最高（实验设计无歧义可操作） | 最高 | **9.0/10** |

---

## 关键共识（6 个 perspective 一致同意的核心结论）

1. **FM2 评测协议失效是最高优先级问题**。所有 perspective 都（隐式或显式）承认：rank 指标对均值平移不变，H2/H3 的 zero gain 结论不能被解读为"FM2 不存在"。全规模实验**必须加入连续指标**（Kendall tau、Spearman rho，或 LDS 连续版本）。

2. **毒性任务逆转定义了 CRA 的适用边界**。当归因信号主要来自梯度范数（如毒性检测），而非语义相似性时，FM1 叙事不适用。这是合理的范围限定，而非理论失败，但必须在论文中明确表述。

3. **H7 在 pilot 规模的失败是工程限制，而非理论否证**。N/d=0.049 导致协方差矩阵欠定——所有 perspective 都认可这一诊断。全规模实验（N/d ≈ 2.5-5）或 PCA 降维白化（N/k >> 1）是两条可行出路。

4. **TRAK-PCA@k=d 与 RepSim 之间的 30.8pp 缺口是 CRA 叙事的核心挑战**。FM1 是必要的但不充分的解释。需要系统分解缺口来源（层混合、归一化、语义特征质量）。

5. **BM25 竞争力需要在全规模实验中明确对标**。counterfact 任务的 BM25=1.0 可能在 N=5473 全规模下降（更多候选项时词汇重叠不够区分），但这必须实验确认，而非假设。

---

## 关键分歧（需要全规模实验解决的未决争议）

### 分歧 1：FM1 的本质是"维度"还是"特征质量"？

- **支持"维度"的一侧**（Theoretical, Pragmatist, Innovator）：r_eff=10 的极端低秩性是信号稀释的直接证据；TRAK 维度扫描的饱和证明维度是关键；B/d = 500 的维度差距是 FM1 的核心。
- **支持"特征质量"的一侧**（Contrarian, Empiricist）：RepSim PCA 在 k=64 而非 k=d 饱和；r_eff ~ 50-63 对两个空间几乎相同；TRAK-PCA@k=d 缺口 30.8pp 证明维度之外还有重要因素；AirRep 的成功说明通用表示并非天然适合归因。

**全规模实验的决定性测试**：Empiricist Experiment 3.3（残差分析）——通过"最后层梯度 TRAK-PCA"和"余弦归一化 TRAK-PCA"将缺口分解到具体因素，可以量化维度 vs 特征质量各贡献多少。

### 分歧 2：phi^T M psi 框架是"理论"还是"符号"？

- **支持"理论"的一侧**（Theoretical, Innovator, Pragmatist）：Theorem 3（N-P 最优性）、Proposition 1.1（B/d SNR 优势）、H5 的维度扫描相变点是非平凡预测，已有两个在 pilot 中获得定向支持（H4 梯度极低秩，H5 饱和）。
- **支持"符号"的一侧**（Contrarian, Empiricist, 部分 Interdisciplinary）：框架的三个最显著预测（H7 白化, H2/H3 对比评分, H9 各向同性）全部失败；非平凡预测在全规模存活的置信度 < 50%；目前尚无"白化 CRA 优于 RepSim"的实验证据。

**全规模实验的决定性测试**：(1) PCA 降维白化是否在 N/k >> 1 时优于标准 RepSim？(2) 连续指标下对比评分是否存在改善（任意任务 > 0.02 Kendall tau）？这两个测试将决定框架的地位。

### 分歧 3：新方法（CATCL、LABA、IV 归因）的优先级

- **Innovator**：这些方法是核心贡献，应进入全规模。
- **Contrarian/Empiricist**：在 pilot 证据已经提示基础叙事存疑的情况下，应优先确认核心假设，新方法是锦上添花。

**协调建议**：先执行 Empiricist 的 13 GPU 小时最小可行集（FM1 + FM2 两个核心诊断套件），根据结果决定是否纳入 CATCL 或 LABA。如果 FM1 缺口分解实验显示"非线性/语义特征质量"是主因，CATCL 则获得额外动机。

---

## 对全规模实验的综合建议

### 必做项（基于所有 perspective 的共识）

1. **连续指标（Empiricist Exp 2.1）**：Kendall tau + Spearman rho 作为 FM2 的主要评测，替代无效的 rank 指标。成本 ~1 GPU 小时。

2. **受控污染注入（Empiricist Exp 2.2）**：alpha in {0, 0.1, 0.5, 1.0, 2.0, 5.0} 的 phi_contaminated = phi + alpha * mean(phi_train)。是验证 FM2 机制的金标准。成本 ~2 GPU 小时。

3. **全规模特征谱（Empiricist Exp 1.1）**：N in {500, 1000, 2000, 5000}，确认 N/d 充足时 H9 的走向。Pythia-70M 上 2 GPU 小时。

4. **TRAK 维度扫描（全规模，Empiricist Exp 1.2）**：Pythia-1B, N=5473 (counterfact 全量)，确认维度扫描模式在 N>>d 时的行为；TRAK-PCA vs TRAK-random 的 smoking gun 测试。~8 GPU 小时（4 GPU 并行 ~2h）。

5. **检索 baseline（Contrarian Direction 1）**：Contriever 或 GTR-T5 对比 RepSim。成本 ~2 GPU 小时。如果检索模型与 RepSim 持平，CRA 的贡献需要重新定位。

6. **BM25 全规模对比（Pragmatist 建议）**：N=5473 全量 counterfact，确认词汇重叠是否在全规模下降。成本 <0.5 GPU 小时。

7. **30.8pp 缺口分解（Empiricist Exp 3.3）**：最后层梯度 TRAK-PCA + 余弦归一化 TRAK-PCA。~2 GPU 小时。

### 条件性做项（根据必做项结果决定）

- **PCA 降维白化（Empiricist Exp 3.1）**：若全规模 N/d > 2，k=64/128 的 PCA 白化可能恢复 H7；若 N/d 仍然不足，跳过。
- **OS-CFAR / 聚类对比评分（Interdisciplinary R1/R2）**：若受控污染注入显示均值减法不能完全恢复，尝试中位数或聚类条件均值。
- **CATCL（Innovator Angle 2）**：若缺口分解实验显示"非线性语义结构"是主因，CATCL 的时序对比方式提供了一个独立的 FM2 去偏路径。
- **OVB 敏感性分析（Interdisciplinary E1）**：成本极低（0.5 GPU 小时），可无条件做，为 FM2 提供因果量化框架。

### 需要放弃或降级的项

- **全维度白化 RepSim（H7 原始版本）**：在 N/d ~ 3 的全规模下仍然边缘，预期仍会失败。应替换为 PCA 降维白化。
- **过于乐观的框架预测声明**：在全规模实验结果出来之前，不应在论文中声明"白化 CRA 是 phi^T M psi 族中 N-P 最优的"，这一声明依赖 H7 和 H9 的验证。

---

## 辩论协调者总结

本轮精炼辩论呈现了一个清晰的共识核心和若干重要的未解分歧。

**最强共识**：pilot 实验中 H2/H3 的平凡满足是**评测协议失效**，而非 FM2 不存在的证据；毒性任务逆转定义了 CRA 的任务适用边界；TRAK-PCA 30.8pp 缺口需要系统解释。这三点是全规模实验必须面对的结构性挑战，且所有 perspective（包括 Contrarian 和 Empiricist）都认为 CRA 框架仍有价值，只是需要叙事修订的幅度和方向存在分歧。

**最大风险**：若全规模实验下，(1) 专用检索模型匹配 RepSim，且 (2) PCA 白化在 N/k >> 1 时仍不优于 RepSim，且 (3) 连续指标下对比评分增益 < 0.02 Kendall tau，则 CRA 框架需要从"信号处理诊断理论"降格为"系统化 benchmark + 分类学"——这仍然是有价值的贡献，但论文定位需根本性调整。

**最值得期待的发现**：受控污染注入（Empiricist Exp 2.2）是本轮辩论中评估最高的新实验设计，其结果将直接决定 FM2 是真实的信号处理缺陷还是 CRA 的理论盲区。这一实验应作为全规模实验的第一批任务之一执行。
