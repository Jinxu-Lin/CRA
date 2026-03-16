# 第 2 轮综合评估 — CRA 研究提案辩论终结报告

**综合者角色**: 辩论协调者（整合 6 份 perspective + 4 份第 1 轮批评）
**日期**: 2026-03-16
**状态**: Round 2 最终评估，直接指导实验计划制定

---

## 执行摘要

经过六份独立视角分析和四轮批评的交叉检验，CRA 研究提案的核心框架（FM1/FM2 双失效模式诊断 + phi^T × psi 双线性统一框架）具有实质性学术价值，但存在若干关键脆弱点。本报告识别跨批评者的共识、核心分歧、调整后的评分，并给出可直接执行的 Top-3 推荐实验和必须放弃的方向。

---

## 第一部分：跨批评者共识点（≥ 3 方同意）

### 共识 C1：精确 Hessian 控制实验是整个框架的必要前置条件

**支持者**: Contrarian（原 perspective 首倡）、Empiricist（列为"必须补充的实验 #1"）、Pragmatist（评为"关键控制，科学价值极高"）、Theoretical（在批评中指出 NTK 假设脆弱性，隐含了同样需求）

**共识内容**: 在 Pythia-70M 上使用精确/高质量 Hessian（K-FAC full eigendecomposition，而非对角近似）计算影响函数，与 RepSim 进行直接比较。若精确 Hessian IF ≈ RepSim 性能，则 FM1/FM2 是 Hessian 近似误差的下游症状，而非独立失效模式，整个 CRA 论证需要根本性修订。若仍然显著低于 RepSim，则 FM1/FM2 作为独立诊断成立。

**操作说明**: "精确 Hessian"在 70M 参数模型上必须理解为 K-FAC + 完整特征值分解，而非字面意义的全 Hessian 矩阵（7000 万 × 7000 万不可存储）。使用 1K DATE-LM 训练样本，成本约 2 GPU-hours。

**潜在结果的影响**:
- 精确 Hessian IF 仍然失败（P ≈ 50%）：FM1/FM2 独立成立，全面推进实验计划
- 精确 Hessian IF 接近 RepSim（P ≈ 30%）：项目从"双失效模式诊断"转向"Hessian 近似质量才是关键瓶颈"
- 中间结果（P ≈ 20%）：需要精细化论证 FM1/FM2 与 Hessian 质量之间的关系

---

### 共识 C2：BM25/k-NN 作为非参数基线是不可缺少的控制实验

**支持者**: Contrarian（原 perspective 首倡）、Empiricist（在"必须补充实验 #2"中明确要求）、Pragmatist（在"科学价值极高"清单第 1 位）、Empiricist Round 1 批评（明确指出这是"审稿人最容易提出的质疑"）

**共识内容**: DATE-LM 自身已发现 BM25（纯词汇匹配方法）在某些任务上能与归因方法持平。这是对整个 TDA 框架价值的根本性挑战。若 BM25 在事实归因任务上击败 RepSim，则 RepSim 的成功可能反映的是语义重叠检测而非真正的因果归因。

**操作说明**: BM25 实现不需要 GPU，2 分钟内可出结果。k-NN（k=10，表示空间）在提取表示后 CPU 可完成。成本极低，必须无条件加入所有实验对比。

**若 BM25 赢**: 论文需重新聚焦于数据选择和毒性过滤任务（BM25 在这两类任务上优势不明显），明确限定 TDA 的适用场景。

---

### 共识 C3：phi^T × psi 框架需要非平凡约束才有理论意义

**支持者**: Contrarian（核心挑战之一："平凡普适性"）、Theoretical（指出 Riesz 表示定理使任何线性相似度都可写成此形式）、Theoretical Round 1 批评（要求给出非平凡约束候选）、Empiricist（通过 SVD 最优基线检验框架预测力）

**共识内容**: phi^T × psi 的双线性形式对于所有线性归因方法都成立（这是代数同义反复，不是发现）。框架的真正价值在于：(a) 对 phi 和 psi 的结构约束（什么样的 phi/psi 最优？），(b) 不同方法之间的 phi/psi 关系（哪些方法在同一"等价类"中？），(c) 框架能否生成预测（给定任务，哪种 phi/psi 选择会更好？）。

**操作建议**: 采用 Theoretical perspective 的 Theorem 8 方向（SVD 最优 phi/psi）作为理论锚点，但放弃在 LOO oracle 上验证（计算不可行）。改为用小规模近似实验（Pythia-70M，500 样本）定性验证方向性正确性。

---

### 共识 C4：LOO Oracle 在时间约束下不可行，必须用替代方案

**支持者**: Pragmatist（500 次 LOO 重训练 ≈ 17 小时，超出约束 17 倍）、Empiricist Round 1（"基本上基于'LOO 重训练在 70M 模型上很快'的假设，但没有说明具体训练多少步"）、Theoretical（指出 Corollary 3.1 实验可行性风险高）、Contrarian（将 LOO 相关实验评为高风险）

**共识内容**: 在任何时间约束下（≤ 8 GPU-hours），都无法在 Pythia-1B 上计算真正意义的 LOO oracle。即使退化到 Pythia-70M + 500 样本子集，完整 LOO 重训练也需要 8-17 小时。Empiricist 的 Angle 3 中依赖 LOO oracle 的 P5 部分必须重新设计。

**替代方案**: 使用 DATE-LM 的保留集（holdout split）LDS 分数作为 oracle 近似；或使用 PBRF（Parameter-Based Resampling Filter）单步近似代替完整重训练；或仅在极小规模（100 样本，Pythia-70M，仅做定性方向验证）上运行。

---

### 共识 C5：IB/MINE 方法在高维 LLM 表征上不可靠

**支持者**: Contrarian（"实验产生大量噪声但无法得出结论"）、Pragmatist（"明确建议降优先级"）、Empiricist（MINE 在小 batch 下极易过拟合，未给出量化估计误差方案）、Pragmatist Round 1 批评（"MI 估计不稳定；幂律拟合需要更多数据点"）

**共识内容**: Innovator Angle 3（IB Lens）和 Interdisciplinary Angle 2（RG + IB）的实验核心依赖 MINE（Mutual Information Neural Estimator）在 d > 1000 维空间估计互信息。MINE 在高维空间方差极高，在小批量下存在严重的过拟合风险。LLM 的 IB 理论本身仍有争议（Shwartz-Ziv & Tishby 与 Saxe 的分歧），且 IB-RG 等价仅在高斯统计下证明，在高度非高斯的 LLM 表征上适用性存疑。

**结论**: IB/MINE 类实验应从主实验计划中移除，降级为"如果时间充裕可尝试的探索性方向"，不作为论文的主要支撑实验。

---

### 共识 C6：白化匹配滤波（Whitened Matched Filter）是最具理论保证的跨域方法改进

**支持者**: Interdisciplinary（Angle 4，65% 成功概率，最高）、Empiricist（在推荐优先级列表第 4 位）、Pragmatist（在优先级中排为"唯一接近即插即用的工程改进"）、Theoretical（"从最优线性检测理论直接推导出 M = Sigma_noise^{-1}，是对框架贡献最大的单一理论结论"）

**共识内容**: phi^T × Sigma_noise^{-1} × psi 是在加性有色噪声条件下的最优线性检测器（Neyman-Pearson 引理）。这将 phi^T × M × psi 中的最优 M 给出了显式表达，不只是描述性分类而是规范性预测。实现相对直接（协方差矩阵估计 + 矩阵求逆，Woodbury identity 可以高效处理高维情况）。

**注意事项**: 需要 Ledoit-Wolf 缩减处理样本协方差矩阵的条件数问题（d=2048，N=1K-5K 时容易病态）。成本约 1.5 GPU-hours。

---

## 第二部分：关键分歧点

### 分歧 D1：LoRA 梯度是否是公平的参数空间基线

**一方（Contrarian）**: 用 LoRA 梯度代替全参数梯度作为 TRAK 的输入，已经在做"维度约化到低秩子空间"——这是 FM1 的理论解法！使用 LoRA-TRAK 的 2x2 矩阵中，"参数空间 + 标准打分"这个 cell 不是干净的"未修复 FM1"基准。

**另一方（Pragmatist）**: LoRA 梯度是工程现实约束下的合理近似，全参数梯度在 Pythia-1B 上存在严重的显存瓶颈；用 LoRA 梯度是可接受的 mitigation，并建议同时报告 LoGra 作为第二个参数空间基线。

**裁决**: 两方都有道理。最优解是：(a) 报告 LoRA-TRAK 和 LoGra（完整梯度投影）两个参数空间变体，(b) 明确在论文中指出 LoRA 梯度本身包含维度约化成分，将其定位为"部分修复 FM1"的实验内点而非纯粹的基线。这不会削弱结论，反而能更细粒度地展示 FM1 修复的梯度效应。

---

### 分歧 D2：DiD 归因的平行趋势假设是否成立

**一方（Innovator）**: DiD 估计量提供因果保证，是 DDA 单差分的更原则性升级。

**另一方（Contrarian Round 1）**: LLM 中平行趋势假设"几乎肯定违反"——pre-training 相似度本身就是 fine-tuning 因果归因的预测因子，DiD 可能产生系统性反向偏差（比单差更差）。

**裁决**: 平行趋势假设需要在 pilot 阶段先行验证。建议设计简单检验：测量 pre-training 相似度与 fine-tuning 归因标签之间的相关性（Spearman's rho）。如果 rho > 0.5，则平行趋势假设违反，DiD 不应使用。如果 rho < 0.2，可以继续。DiD 保留在实验计划中但设置前置验证门槛。

---

### 分歧 D3：NTK 近似对 Theoretical 的 Theorem 1-2 是否有效

**一方（Theoretical perspective 本身）**: NTK 近似给出了梯度协方差有效秩约为 d 的机制性解释，是 FM1 谱理论的核心。

**另一方（Pragmatist、Contrarian、Theoretical Round 1 批评）**: NTK 近似在 LLM fine-tuning 中已知失效（fine-tuning 是典型的特征学习场景，而非内核机制）。以 NTK 假设为前提的定理在 ML 顶会中难以过 reviewer 关。

**裁决**: Theorem 1-2 仍然保留，但必须重新定位：(a) 将 NTK 部分作为理论直觉而非定理前提，(b) "r_eff ~ d 预测"单独陈述为可测假说（不依赖 NTK 推导），(c) 通过 Pythia-70M 的特征值实验直接验证该预测是否成立。即使 NTK 推导不成立，r_eff ~ d 这个实验预测本身是独立有价值的。

---

### 分歧 D4：跨域类比（神经科学/统计物理/免疫学）深度如何

**一方（Interdisciplinary perspective）**: 四个类比不只是隐喻，而是形式同构，可以产生具体算法改进。

**另一方（Contrarian Round 1）**: 评分 4.5/10，认为神经科学和 RG 类比的"不只是隐喻"声明缺乏数学论证；Transformer 的全局注意力机制与 RG 的局部性原理根本矛盾；eligibility trace 类比要求突触特异性，与 LLM 参数空间的纠缠性相违背。

**裁决**: 只保留白化匹配滤波（Angle 4）作为有实验价值的跨域导入。神经科学 ETA 方法因工程障碍（训练时记录 per-sample 快照）降至附录级别。RG 和免疫学方向放弃（详见第四部分）。

---

## 第三部分：调整后的最终评分（1-10）

| Perspective | 第 1 轮分数（批评者打分均值）| 第 2 轮调整后分数 | 调整理由 |
|---|---|---|---|
| **Empiricist** | Contrarian:8, Pragmatist:7.5, Theoretical:6 → 均值 7.2 | **8.0** | 实验设计最严格，预注册证伪标准是本次辩论质量最高的单项贡献；LOO Oracle 可行性问题可通过替代方案解决，不影响整体价值 |
| **Contrarian** | Empiricist:7, Pragmatist:7, Theoretical:7 → 均值 7.0 | **7.5** | 精确 Hessian 控制实验设计、BM25 基线要求、phi^T*psi 平凡性挑战是全局最高价值贡献；主要弱点是对 DDA debias 替代解释缺乏数学细节 |
| **Pragmatist** | Contrarian:7, Empiricist:7, Theoretical:5 → 均值 6.3 | **7.0** | 工程可行性分析最扎实，防御性工程清单是执行层面不可缺少的贡献；LoRA 梯度作为基线的设计问题经裁决可通过双路基线解决 |
| **Theoretical** | Contrarian:7.5, Empiricist:5, Pragmatist:5 → 均值 5.8 | **6.5** | Theorem 7（分类学）+ Theorem 3-4（FM2 形式化）具有实质价值；T1-T2 NTK 假设问题经裁决可通过重新定位解决；T8 因 LOO oracle 不可行降权 |
| **Innovator** | Contrarian:6, Empiricist:5, Pragmatist:6, Theoretical:7 → 均值 6.0 | **6.0** | Spectral Scalpel（Angle 1）是有价值的机制验证实验；DiD Attribution（Angle 2）需平行趋势验证门槛；IB Lens（Angle 3）已由共识 C5 排除 |
| **Interdisciplinary** | Contrarian:4.5, Empiricist:5, Pragmatist:3, Theoretical:6 → 均值 4.6 | **5.0** | 白化匹配滤波（Angle 4）理论最严谨，提升整体评分；其余三个角度工程障碍严重，评分被拉低；统一公式因破坏因果可分性被降权 |

**说明**: 评分反映的是对 CRA 项目的实际贡献价值，考虑了可行性、理论严谨性和与核心论点的相关性。

---

## 第四部分：必须放弃的方向

### 放弃方向 R1：IB/MINE 互信息估计类实验

**适用范围**: Innovator Angle 3（IB Lens）、Interdisciplinary Angle 2（RG + IB 幂律预测）

**放弃理由**:
1. MINE 在 d > 1000 维空间的估计方差无法接受，任何结论都可能是估计误差而非真实信号
2. LLM 的 IB 理论本身存在根本性争议（Saxe 等已证明 ReLU 网络不存在信息压缩相变，Transformer 的情况更复杂）
3. IB-RG 等价仅在高斯统计下证明，LLM 表征的高度非高斯性使理论不适用
4. 四位批评者（Contrarian、Pragmatist、Empiricist、Theoretical）均明确反对

**严重性**: 若强行执行，极可能产生噪声结果，损害论文可信度。

---

### 放弃方向 R2：LOO Oracle 完整计算（Pythia-70M 以上）

**适用范围**: Empiricist Angle 3 的 P5 部分、Theoretical Theorem 8 的完整验证

**放弃理由**:
1. 500 次 LOO 重训练在 Pythia-70M 上估计需要 8-17 小时，远超任何合理的时间预算
2. 使用 PBRF 近似会引入额外偏差，污染 SVD 分析的理论洁净性
3. 三位批评者（Pragmatist、Empiricist Round 1、Theoretical Round 1）均指出这个计算不可行

**替代方案**: 接受定性/方向性验证（小规模 Pythia-70M + 100 样本），或用 DATE-LM 保留集的排名相关性作为 oracle 代理。

---

### 放弃方向 R3：神经科学 Eligibility Trace Attribution（ETA）作为主实验

**适用范围**: Interdisciplinary Angle 1

**放弃理由**:
1. 需要在训练过程中记录 per-sample 表示快照，对标准 DATE-LM pipeline 是侵入式修改，需要 2-3 小时额外代码工作
2. Fine-tuning 中每个 epoch 每个 sample 都被多次访问，不存在单一的"触发时刻"t_i，使类比的基础崩溃
3. 计算成本被严重低估（Pragmatist 指出实际更接近 2-4h 含代码修改）
4. Theoretical Round 1 批评指出公式本身存在理论不一致性（phi 和 psi 来自不同参数状态）

**保留价值**: 作为附录中的概念性讨论，不需要实验验证。

---

### 放弃方向 R4：统计物理/RG 幂律衰减实验作为主实验

**适用范围**: Interdisciplinary Angle 2

**放弃理由**:
1. 幂律拟合需要 ≥ 10 个数据点，而可用的层数或模型规模不足
2. 依赖 IB 估计，受共识 C5 约束
3. IB-RG 等价只在场论系统（高斯统计）下证明，Transformer 的全局注意力机制与 RG 局部性原理相悖
4. Pragmatist Round 1 指出实际成本 2-4h 含重跑，且结果难以可靠解读

---

### 放弃方向 R5：免疫学两阶段去偏（Immunological Two-Stage Debiasing）作为独立实验

**适用范围**: Interdisciplinary Angle 3

**放弃理由**:
1. 引入两个额外超参数（β 和 τ），在 N=3 的 DATE-LM 任务上无法可靠选择超参数而不过拟合
2. Stage 2 的改进能否超过 Stage 1（简单均值减法）是开放问题，且免疫学类比并不提供此答案
3. 成功概率最低（40%），且即使成功，贡献边际
4. 作为独立实验的意义不大，内容可以并入对比打分实验（Pragmatist Angle 3）中作为消融

---

## 第五部分：实验方向优先级排序

（综合可行性、科学价值、与核心论点的相关性）

| 优先级 | 实验方向 | 来源 | GPU 成本 | 成功概率 | 不可替代性 |
|---|---|---|---|---|---|
| **P0** | Pipeline 验证 Pilot：RepSim + TRAK on Pythia-1B + DATE-LM data selection (1K subsample) | Pragmatist | 0.25h | 90% | 不可绕过 |
| **P1** | 精确 Hessian 控制实验（Pythia-70M, 1K 样本，K-FAC full eigendecomp + IF） | Contrarian | 2h | 50% | 最高 — 决定框架存亡 |
| **P2** | Hardened 2x2 Factorial（RepSim+TRAK × standard+contrastive，Pythia-1B，DATE-LM 3 tasks，含 BM25/k-NN 基线，含 DDA 强基线） | Empiricist | 4h（4GPU 并行≈1h 实际等待）| 75% | 最高 — 主结果表 |
| **P3** | 梯度协方差特征值分析（Pythia-70M，Lanczos top-500，DATE-LM 1K 样本） | Innovator/Theoretical | 1h | 65% | 高 — FM1 机制证据 |
| **P4** | 维度扫描：TRAK at k ∈ {64,128,256,512,1024,2048,4096} + RepSim-PCA at d_eff 对应值 | Pragmatist/Empiricist | 2h | 75% | 高 — Figure 2 FM1 可视化 |
| **P5** | 对比打分通用插件（36-cell：4方法 × 3对比变体 × 3任务，使用缓存表示/梯度） | Pragmatist | 1h | 80% | 高 — FM2 证据表 |
| **P6** | 白化匹配滤波（Whitened RepSim：phi^T Σ^{-1} psi，Ledoit-Wolf 正则化） | Interdisciplinary | 1.5h | 65% | 中高 — 框架规范性结论 |
| **P7** | DiD 平行趋势验证 pilot（pre/post 相似度相关性检验，决定 DiD 是否可行） | Innovator | 0.5h | 60%（假设成立） | 中 — 门控后续 DiD 实验 |
| **P8** | 多方法竞赛（不含 oracle）：所有 5 个表示空间方法 + TRAK + BM25 + k-NN，DATE-LM 全 3 任务 | Empiricist | 3h | 70% | 中 — phi^T*psi 分类学验证 |
| **P9** | 跨模型维度扫描验证（Pythia-160M，仅 Phase 1-2，d=768 验证 r_eff ~ d 预测） | Empiricist | 1h | 65% | 中 — 可扩展性检验 |

**总计（P0-P8，排除重叠）**: 约 14-16 GPU-hours，在 4x RTX 4090 上约 4-5 天。

---

## 第六部分：Top-3 推荐实验（附理由与时间估算）

### 推荐实验 #1：精确 Hessian 控制实验

**理由**: 这是 CRA 研究的基础性前置条件。五位批评者中至少四位明确提及或隐含要求这个实验。若跳过直接进行 2x2 实验，而后发现 FM1/FM2 是 Hessian 近似的下游症状，整个实验计划需要推倒重来，浪费所有已投入计算。相比之下，先用 2 GPU-hours 确认基础前提，是最高 ROI 的投入决策。

**具体设计**:
- 模型：Pythia-70M（参数量合理，K-FAC 可行）
- 数据：DATE-LM data selection task，1000 训练样本子集
- 方法：(1) 标准 TRAK（随机投影），(2) K-FAC full eigendecomp IF，(3) RepSim（last layer）
- 评估指标：LDS（Linear Datamodeling Score，DATE-LM 标准指标）
- 预注册判据：若 K-FAC IF 与 RepSim LDS 差距 < 5pp，则 FM1/FM2 框架需要修订

**时间估算**:
- Pilot（验证 K-FAC eigendecomp 可行性）：15 分钟
- 完整实验：2 GPU-hours
- 结果分析：30 分钟

**决策树**:
- K-FAC IF 仍然 >> RepSim：FM1/FM2 独立成立 → 推进 P2 完整 2x2 实验
- K-FAC IF ≈ RepSim（< 5pp 差距）：FM1 主要由 Hessian 近似质量解释 → 项目核心论点修订为"Hessian 质量是参数空间归因的根本瓶颈，表示空间方法等价于隐式精确 Hessian"

---

### 推荐实验 #2：Hardened 2x2 Factorial + BM25/DDA 对照

**理由**: 这是论文的主结果表，无论 #1 的结论如何都应执行（即使 FM1/FM2 框架修订，2x2 的实验结果本身仍然有价值）。关键是正确控制混淆因子（特别是实现不对称性和归一化不对称性），并加入 BM25 和 DDA 这两个被所有批评者认为不可缺少的控制组。

**具体设计**:
- 模型：Pythia-1B（使用 DATE-LM 官方 checkpoint）
- 数据：DATE-LM 全 3 个任务（data selection, toxicity filtering, factual attribution）
- 方法矩阵：
  - 参数 × 标准：TRAK（k=2048，LoRA 梯度）+ LoGra（全梯度投影，补充）
  - 参数 × 对比：TRAK + mean-subtraction（DDA-style）
  - 表示 × 标准：RepSim（last layer，cosine similarity）
  - 表示 × 对比：RepSim + mean-subtraction
- 控制组：BM25（词汇），k-NN（k=10，表示空间），DDA（充分调优后的强参数空间基线）
- 混淆控制：(C1) 固定投影维度 k=d=2048，(C3) 测试 last/mid/phase-transition 三个层，(C4) 2 种负样本策略（random vs. task-structured）
- 统计方法：bootstrap CI（B=1000），报告每个任务的结果（不跨任务平均）
- 预注册判据：(a) FM1 证伪：RepSim < TRAK-5pp on data selection LDS；(b) FM2 证伪：对比打分在 ≥1 方法 ≥1 任务上损害 > 3pp；(c) 正交性证伪：交互项 > 30% × min(主效应) on ≥2 tasks

**时间估算**:
- Pilot（15 分钟，仅 data selection，RepSim + TRAK，验证 pipeline 与 leaderboard 对齐）：15 分钟
- 完整 2x2 矩阵（4x RTX 4090 并行）：约 4 GPU-hours（实际等待约 1 小时）
- 控制组（BM25、k-NN、DDA）：额外 2 GPU-hours
- 混淆控制消融（层选择、负样本策略）：额外 2 GPU-hours
- **合计：约 8 GPU-hours，实际等待约 2-3 小时**

---

### 推荐实验 #3：梯度协方差谱分析 + 维度扫描（FM1 机制证据）

**理由**: 这是 FM1 诊断从"直觉叙事"升级为"可测量机制"的关键实验。提供两个互补的 FM1 证据：(a) 梯度协方差有效秩测量（定量证明信号低维性），(b) TRAK 维度扫描（直接显示信号饱和点在 k ~ d 附近）。这两个实验可以共享大量计算（Pythia-70M 上的特征值分析也是维度扫描的 pilot），是单位 GPU-hours 信息密度最高的实验组合。

**具体设计**:

*Part A：梯度协方差特征值分析*
- 模型：Pythia-70M（B=70M，d=512）
- 数据：DATE-LM data selection，1K 训练样本
- 方法：Lanczos top-500 特征值（使用 scipy.sparse.linalg.eigsh，比 PyTorch lobpcg 更稳定）
- 测量：r_eff 在 90%/95%/99% 能量阈值下的值
- 预测：r_eff(95%) ∈ [0.5d, 2d] = [256, 1024]
- 证伪条件：r_eff(95%) > 10d（约 5120），则 FM1 非秩亏问题

*Part B：维度扫描*
- 模型：Pythia-1B（d=2048）
- 数据：DATE-LM data selection，10K 训练样本
- TRAK：k ∈ {64, 128, 256, 512, 1024, 2048, 4096}（随机投影）
- RepSim-PCA：d_eff ∈ {64, 128, 256, 512, 1024, 2048}
- 关键对照：TRAK-PCA（用 Sigma_g 的 top-k 特征向量投影）vs TRAK-random（随机投影）
- 评估：各配置的 LDS，画饱和曲线
- 关键预测："TRAK-PCA 在 k=d 处接近 RepSim 性能" → 这是 FM1 的"吸烟枪"

*Part C：跨模型验证（可选）*
- Pythia-160M（d=768），重复 Part A，验证 r_eff 与 d 的线性关系

**时间估算**:
- Part A Pilot（15 分钟，Pythia-70M，计算特征值分布）：15 分钟
- Part A 完整分析：1 GPU-hour
- Part B 维度扫描（8 配置 × 3 方法，可并行）：2 GPU-hours
- Part C（可选）：1 GPU-hour
- **必要部分合计：约 3 GPU-hours**

---

## 第七部分：执行顺序与决策树

```
Day 1（建议时间安排）:
─── [P0] Pipeline Pilot（15 min）
    ├── 通过 → 继续
    └── 失败 → Debug DATE-LM 环境，修复 pipeline

─── [P1] 精确 Hessian 控制（2h，与 P0 后串行）
    ├── K-FAC IF 仍然 >> RepSim（≥10pp 差距）→
    │   FM1/FM2 独立框架成立，推进全部实验计划
    ├── K-FAC IF ≈ RepSim（<5pp 差距）→
    │   **PROJECT PIVOT**: 修订核心论点，将 P2-P5 重新配置为
    │   "Hessian 质量诊断"框架下的实验
    └── 中间结果（5-10pp）→ 在论文中细化讨论

Day 2-3（并行执行）:
─── [P2] Hardened 2x2 Factorial（4x GPU 并行，~8 GPU-hours）
─── [P3] 梯度协方差特征值分析（Part A，1h）

Day 4:
─── [P4] 维度扫描（Part B，2h）
─── [P5] 对比打分通用插件（1h，复用 P2 缓存数据）
─── [P6] 白化匹配滤波（1.5h，复用表示缓存）

Day 5（可选扩展）:
─── [P7] DiD 平行趋势验证 pilot（0.5h）
    ├── 通过（rho < 0.2）→ 实施 DiD Attribution 实验（1.5h）
    └── 失败（rho > 0.5）→ 放弃 DiD
─── [P8] 多方法竞赛（3h，复用缓存）
─── [P9] 跨模型验证（1h）
```

---

## 第八部分：对各 Perspective 的最终评价

### Empiricist (8.0/10)
在实验设计的系统性和预注册证伪标准方面是本次辩论中最高质量的贡献。四个混淆因子的识别和控制方案是直接可以放入实验设计文档的内容。主要弱点（LOO Oracle 可行性）已有明确替代方案。**强烈建议采用其实验框架作为 P2 的基础设计**。

### Contrarian (7.5/10)
在科学严谨性方面最重要的贡献：精确 Hessian 控制实验、BM25 基线要求、phi^T*psi 平凡性挑战，这三点在其他五份 perspective 中全部缺失。任何不包含这些控制组的实验计划都会在 reviewer 审查中暴露严重漏洞。**推荐实验 #1 直接来自 Contrarian perspective**。

### Pragmatist (7.0/10)
工程可行性分析是其他 perspective 的重要参照。防御性工程清单（5 个风险点）和具体的时间估算（大多数情况下比其他 perspective 更准确）对执行阶段至关重要。**P0 的 pipeline 验证和整体执行顺序建议来自 Pragmatist**。

### Theoretical (6.5/10)
Theorem 7（分类学）和 Theorem 3-4（FM2 形式化去混淆）是直接可以写进论文的理论贡献。T1-T2 在重新定位后（NTK 作为直觉而非推导前提）仍然有价值。T8（SVD 最优性）虽然最优美，但因 LOO oracle 不可行只能做定性验证。**理论框架部分应集中在 T3-T4-T7 三个定理，其余作为辅助**。

### Innovator (6.0/10)
Spectral Scalpel（Angle 1）在修正实验范围后（仅 Pythia-70M，不跨模型规模）是推荐实验 #3 的核心。DiD Attribution 有价值但需要前置验证门槛。IB Lens 已由共识 C5 排除。**净贡献主要是 Angle 1 的有效秩测量思路**。

### Interdisciplinary (5.0/10)
白化匹配滤波（Angle 4）是唯一直接可操作的贡献，理论有据（Neyman-Pearson 最优性），实现直接。其他三个角度（神经科学/RG/免疫学）在 1 小时约束下的工程可行性被系统性高估，类比的数学严格性也被过度声明。**仅白化匹配滤波进入核心实验计划（P6）**。

---

## 附录：关键参考文献清单（基于辩论共识）

### 必读（直接相关）
- DATE-LM (2507.09424) — 基准评估协议
- Hong et al. (2509.23437) — 精确 Hessian 控制的参照点
- DDA (2410.01285) — 强参数空间基线，对比打分原型
- Hu et al. (2602.10449) — 随机投影理论，FM1 的理论上界
- RepT (2510.02334) — 表示空间方法，层选择证据
- Vitel & Chhabra (2511.04715) — 中间层优于首/末层证据

### 重要（框架支撑）
- Li et al. (2409.19998) — IF vs RepSim 系统比较
- TRAK (2303.14186) — 参数空间基线
- AirRep (2505.18513) — 学习型表示归因
- Daunce (2505.23223) — 无梯度归因方法，对 FM1/FM2 框架的潜在反例
- Li et al. (2512.09103) — TRAK 几何脆弱性，FM1 机制支撑

### 注意（需要谨慎使用）
- CKA 病理学三篇（Davari 2022, Cloos 2024, Okatan 2025） — RepSim 可能继承 CKA 病理
- Shwartz-Ziv & Tishby (2017) / Saxe et al. (2019) — IB 理论争议，谨慎引用
- Stankovic & Mandic (2021) — CNN 作为匹配滤波，白化方案理论基础
