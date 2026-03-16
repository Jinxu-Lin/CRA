# Pragmatist 交叉批评 Round 1

**角色**: Pragmatist（工程可行性、时间约束、实现复杂度）
**日期**: 2026-03-16
**被批评对象**: innovator.md, theoretical.md, contrarian.md, interdisciplinary.md, empiricist.md

---

## 总体评估框架

硬约束：
- 硬件：4x RTX 4090（每卡约 24GB VRAM）
- 单实验时间上限：约 1 小时
- Pilot 目标：10-15 分钟内完成
- 外部资源：DATE-LM 数据集（需要提前确认可用性）；Pythia 系列模型（HuggingFace 可获取）

核心问题：五份 perspective 的提案在理论雄心上普遍过高，但在工程可行性的细节上存在不同程度的乐观偏差。

---

## 一、Innovator Perspective

**总分：6 / 10**

**理由：提案创新性强，但三个 Angle 的工程风险差异悬殊。**

### 可行性逐项评估

**Angle 1（Spectral Scalpel）**：工程可行性最高，优先级最高。

- 核心瓶颈：Pythia-1B 有约 10 亿参数，完整梯度向量存在 GPU 显存瓶颈。单样本梯度约需 4GB 显存（FP32），1K 样本的梯度协方差矩阵若直接计算约为 10^9 × 10^9，完全不可行。
- **实际可行方案**：只能用 Lanczos 迭代计算 top-K 特征值，但每次 Lanczos 步都需要一次矩阵向量积（即一次完整的反向传播），1K 样本 × 500 Lanczos 步 = 5×10^5 次反向传播。在 Pythia-1B 上每次反向传播约 0.1-0.2 秒，合计约 14-28 小时，远超 45 分钟上限。
- **修正方案**：必须用 Pythia-70M（7000 万参数，梯度存储约 280MB/样本，但仍需优化）。Pilot 15 分钟可行，但"Core experiment across 4 model scales"需要至少 4-6 GPU-hours，不满足约束。
- **实际建议**：仅在 Pythia-70M 上做完整分析，Pythia-160M 作为验证，更大模型只做 pilot 级别验证。

**Angle 2（DiD Attribution）**：可行性中等，但关键前提存在风险。

- 需要 pre-fine-tuning 和 post-fine-tuning 两个检查点。DATE-LM 是否提供 pre-checkpoint？提案承认需要"用 HuggingFace base models 作为 pre"——这个假设需要提前验证。
- 表示提取是轻量操作（CPU 可以跑），实验本身 1.5 GPU-hours 是合理的。
- 主要风险：parallel trends 假设验证本身就是一个需要仔细设计的实验，不是顺手能做的。若假设不成立，整个 Angle 2 实际上变成一个负结果。

**Angle 3（IB Lens）**：可行性最差，明确建议降优先级。

- MINE（Mutual Information Neural Estimator）在高维（d=2048）空间中不可靠是已知问题，不是新发现。LLM 的 IB 理论本身争议极大（Shwartz-Ziv vs. Saxe 的分歧）。
- "45% 成功概率"是作者自己承认的，加上实验结果难以解释（MI 估计误差大），极有可能产生噪声结果。
- 15 分钟 pilot 在技术上可行（MINE 在 Pythia-1B 小 batch 上可运行），但用于实验设计决策的质量不够。

### 时间风险总结

| Angle | 声称时间 | 实际估计 | 风险点 |
|-------|---------|---------|--------|
| Spectral Scalpel (all scales) | ~2h | 4-6h | Lanczos 在大模型上耗时严重低估 |
| DiD Attribution | ~1.5h | ~1.5h | pre-checkpoint 可用性需先确认 |
| IB Lens | ~2h | ~2-3h + 噪声结果 | MI 估计不稳定，实验结果难解读 |

### 优先级建议

Angle 1（限 Pythia-70M）> Angle 2（需先验证 pre-checkpoint 可用）> ~~Angle 3（降到附录级别）~~

---

## 二、Theoretical Perspective

**总分：5 / 10**

**理由：理论框架严谨，但大量定理的"可测 corollary"实验代价被严重低估，且 NTK 假设对 LLM 的适用性存在根本性风险。**

### 工程可行性逐项评估

**Angle 1（Spectral FM1 Theory，T1-T2）**：与 Innovator Angle 1 有高度重叠，存在同样的梯度协方差计算瓶颈。声称"2 GPU-hours（Lanczos top-500 eigenvalues at each model scale）"。问题：4 个模型规模 × Lanczos top-500 × 1K 训练样本，在 Pythia-1B 规模下严重低估，已在 Innovator 部分分析过。

**Angle 2（FM2 Bias Decomposition，T3-T5）**：工程实现相对简单（均值减法 + 协方差分解），1.5 GPU-hours 估计合理。但关键挑战是"定量预测的验证"（如 Corollary 2.1 声称的"参数空间 ||phi_shared||/||phi_task|| > 10×"）——这个比例需要在训练数据的子分布上计算，实现细节复杂。

**Angle 3（phi^T M psi 完备性，T6-T8）**：LOO retraining oracle（计算真实归因矩阵的 SVD）是最大的工程瓶颈。Pythia-70M 上 500 样本的 LOO oracle 需要 500 次独立 fine-tuning（或近似）。即使每次只训练 1 epoch，单次约 2 分钟，500 次约 1000 分钟 ≈ 17 小时，超出单实验 1 小时约束 17 倍。Empiricist 也注意到了这个问题，但仍列为"P5"计划。

### 最大工程风险

1. **NTK 假设**：T2 的核心论点"g_i ≈ J^T delta_i"在 LLM fine-tuning 中非常粗糙。实际梯度涉及所有层，不只是最后一层的 Jacobian。这不只是理论上的担忧——若 NTK 假设在 Pythia-1B 上误差很大，Theorems 1-2 的实验验证可能得到与预测截然相反的结果。

2. **T8（最优 phi psi）的实验验证**：完全依赖 LOO oracle，计算不可行（见上）。建议改用 PBRF（Parameter-based Resampling Filter）作为近似 oracle，但这引入额外近似误差。

3. **Corollary 3.1（SVD 对齐测试）**：同样依赖 LOO oracle。

### 代码实现复杂度

- T3-T4（均值减法分解）：低复杂度，已有代码基础
- T7（分类学表格）：不需要实验，仅数学推导
- T1-T2（特征值分析）：中等复杂度，需要 Lanczos 实现（PyTorch 有 `lobpcg` 但不稳定，`scipy.sparse.linalg.eigsh` 更可靠）
- T5（DiD）：中等复杂度
- T8（最优 SVD）：高复杂度 + 不可行的 oracle 计算

### 优先级建议

T7（分类学，不需要实验）+ T3-T4（FM2 分解，实现简单，可行）> T1-T2（限小模型）> ~~T8（LOO oracle 不可行）~~ > T5（DiD 作为附录）

---

## 三、Contrarian Perspective

**总分：7 / 10**

**理由：提出的三个 falsification experiment 都具有极高工程可行性和科学价值，是五份 perspective 中最务实的一份。主要问题是"exact Hessian"实验对资源的低估。**

### 工程可行性逐项评估

**Falsification 1（Exact-Hessian IF on Pythia-70M）**：
- Pythia-70M 有约 7000 万参数，Hessian 矩阵是 7×10^7 × 7×10^7，不可能存储（约需 4×10^10 GB）。
- **实际含义**：这里的"exact Hessian"必须理解为"更好的 Hessian 近似"，例如 K-FAC（Kronecker-Factored Approximate Curvature）加完整特征值矩阵，而不是字面意义上的 full Hessian。Pythia-70M 上 K-FAC 本身的内存开销已经很大。
- 提案声称"2 GPU-hours"——如果理解为 K-FAC exact eigendecomposition（而非 diagonal approximation），可能是低估，但数量级上合理。
- 科学价值极高：这是 CRA 整个 FM1/FM2 叙事最关键的控制实验。

**Falsification 2（DDA-debiased IF 作为强 baseline）**：
- 完全可行，2 GPU-hours 内可完成。DDA 的代码已公开（2410.01285）。
- 强烈建议作为核心实验的必要组成部分。

**Falsification 3（BM25 + k-NN 基线）**：
- BM25 实现成本极低（不需要 GPU），2 分钟内可出结果。
- k-NN 在表示空间中也非常轻量（提取表示后 CPU 可完成）。
- 唯一风险：如果 BM25 真的打平 RepSim，这个结果的解读需要非常小心，可能推翻整个 CRA 叙事。Contrarian 已经意识到这一点，并提出 45% 的概率。

### Contrarian 的主要盲点

1. **"exact Hessian"术语不精确**：文中未解释如何在 7000 万参数模型上计算"exact Hessian"。实际上必须是某种低秩近似，只是比 TRAK 的 diagonal 近似更好。这个关键细节需要在实验设计中明确。

2. **Daunce 反例的解读**：Daunce（2505.23223）不使用梯度但仍然有效，这确实挑战了"FM1/FM2 是梯度空间特有问题"的叙事。Contrarian 正确识别了这一点。但从工程角度，Daunce 的实现复杂度如何？如果它是一个计算代价很高的方法（ensemble perturbation 需要多次前向传播），在 1 小时约束下是否可行？需要评估。

3. **"Add Li et al. benchmark"的时间成本**：这要求在第二个 benchmark 上重跑所有方法，大约会使总实验时间翻倍。在资源约束下，建议仅用 DATE-LM 主实验 + Li et al. 的子集验证。

### 优先级建议

Falsification 1（exact Hessian 控制，明确操作定义后）= Falsification 2（DDA 作为强基线）> Falsification 3（BM25/k-NN，成本极低应无条件加入）

---

## 四、Interdisciplinary Perspective

**总分：3 / 10**

**理由：概念类比丰富且智识上令人兴奋，但四个 Angle 中有三个的工程可行性存在根本性问题，且多数预测难以在 1 小时内得到可靠结论。**

### 工程可行性逐项评估

**Angle 1（神经科学：Eligibility-Trace Attribution, ETA）**：

- **核心工程障碍**：需要在 fine-tuning 过程中每步记录 per-sample 的表示快照。对于 1K 训练样本、Pythia-1B，假设每步只存最后一层的表示（维度 2048），1000 样本 × 假设 100 步 × 2048 × FP16 ≈ 400MB，勉强可行。但 DATE-LM 的 fine-tuning 通常是多个 epoch，每 epoch 内的步数远超 100 步，存储开销会迅速失控。
- **更实际的问题**：DATE-LM 的标准 pipeline 不支持中间 checkpoint 保存，修改训练代码需要额外开发时间，估计需要 2-3 小时的代码工作，这不在实验时间内。
- 声称"1.5 GPU-hours"已经包含了需要修改标准 fine-tuning pipeline 的隐性工程时间，实际成本更高。

**Angle 2（统计物理：Renormalization Group）**：

- **最大问题**：IB proxy 的可靠估计在 LLM 表示空间（d=2048）中本质上困难，Innovator 已经指出 MINE 的不可靠性。Interdisciplinary 自己也承认："the IB-RG equivalence is only proven for Gaussian statistics"，而 LLM 表示高度非高斯。
- **Power-law 拟合**：4 个数据点（4 层或 4 个模型规模）无法可靠拟合幂律指数。这个分析需要至少 10+ 个数据点。
- 声称"2 GPU-hours"是理想情况，MI 估计的数值不稳定可能导致需要多次重跑。
- 科学价值很高，但这个方向更适合一篇独立的理论-实验综合论文，而不是 CRA 的子实验。

**Angle 3（免疫学：Two-Stage Debiasing）**：

- **工程可行性最高**：两阶段均值减法实现极简单，~1 GPU-hour 估计合理。
- **主要问题**：改进幅度可能极小（自己估计 40% 成功率）。从工程角度，这是一个"低风险、低回报"的实验——容易做，但很可能结果差异在误差范围内。
- 如果作为核心实验，没有太大价值；作为 ablation 补充实验可以接受。

**Angle 4（信号处理：Whitened Matched Filter）**：

- **工程可行性良好**：协方差矩阵估计是标准操作，Woodbury identity 可以避免 d×d 矩阵求逆（d=2048 时求逆约 0.1 秒，完全可行）。
- 声称"1.5 GPU-hours"合理，65% 成功率是可接受的风险。
- **主要工程风险**：样本协方差矩阵在 d=2048、N=1K-5K 样本时条件数较差，需要加正则化（Ledoit-Wolf 缩减）。如果不处理好这个问题，Sigma_noise^{-1} 会放大噪声方向，可能导致性能下降而非提升。
- 这是四个 Angle 中唯一接近"即插即用"的工程改进。

### 时间风险总结

| Angle | 声称时间 | 实际估计 | 主要风险 |
|-------|---------|---------|---------|
| 神经科学（ETA） | 1.5h | 2-4h（含代码修改） | 需修改训练 pipeline，存储 per-sample checkpoint |
| RG/统计物理 | 2h | 2-4h（含重跑） | MI 估计不稳定；幂律拟合需要更多数据点 |
| 免疫学（Two-Stage） | 1h | 0.5-1h | 可行，但改进微小 |
| 信号处理（Whitened MF） | 1.5h | 1.5h | 协方差矩阵条件数问题需要正则化 |

### 优先级建议

Angle 4（Whitened Matched Filter，工程可行，理论有据）> Angle 3（Two-Stage Debiasing，作为 ablation）> ~~Angle 1（ETA，工程代价高于声称）~~ > ~~Angle 2（RG，方法论风险太大）~~

---

## 五、Empiricist Perspective

**总分：7.5 / 10**

**理由：实验设计最严谨、最系统，与 1 小时约束的契合度最高。主要弱点是 Angle 3（Oracle SVD）的计算预算严重超出约束，以及部分 baseline 组合的总 GPU-hours 被分散低估。**

### 工程可行性逐项评估

**Angle 1（2x2 Factorial Ablation）**：工程可行性最高，设计最成熟。

- 4 GPU-hours（声称）：在 4x RTX 4090 并行条件下等价于约 1 GPU-hour 实际等待时间，完全可行。
- 统计分析方案（bootstrap CI B=1000 + two-way ANOVA）：标准且实现简单。
- **需要注意的工程细节**：
  - TRAK 官方实现（github.com/MadryLab/trak）对 LoRA 梯度的支持状态需要确认。
  - Confounder C4（negative sample strategy）会使实验矩阵扩展为 2x2x3，总实验量增加 3×，时间预算需相应调整。
  - 建议只测 Confounder C4 的两种策略（random vs. task-structured），减少为 2x2x2 设计。

**Angle 2（Dimension-Controlled Causal Isolation）**：

- Phase 1（eigenspectrum，15 min）：合理，和 Innovator/Theoretical 的 pilot 完全重合，可以共享。
- Phase 2（dimension sweep on Pythia-1B）：声称 30 分钟。这里有一个关键问题：k in {64, 128, 256, 512, 1024, 2048, 4096, 8192} 是 8 个点，加上 PCA vs. random 两个变体，加上 RepSim-PCA，是约 17 个实验配置。每个配置在 Pythia-1B 上提取一次梯度 + 计算 LDS，每次约 10-20 分钟，总计约 3-6 小时。"30 分钟"是不可能的，即使并行。
- Phase 3（cross-model scaling）：声称 30 分钟，实际更接近 1-2 小时。
- **修正后总估计**：~4-6 GPU-hours，可并行到 ~2 小时实际等待时间。

**Angle 3（Multi-Method Tournament + SVD Oracle）**：

- **P4（不含 oracle 的 multi-method tournament）**：3 GPU-hours，合理，在约束内。
- **P5（含 LOO oracle 的完整版本）**：声称 5 GPU-hours。问题：500 样本 LOO oracle 在 Pythia-70M 上需要 500 次训练（或 PBRF 近似）。即使用 PBRF 近似（单次正向传播推导归因分数的单步更新），计算量也远超 5 GPU-hours。Empiricist 在文中以"Pythia-70M 500 training samples LOO oracle feasibility"作为 pilot 目标——这个 pilot 本身就需要时间评估可行性，说明他们对实际成本也不确定。
- **实际建议**：放弃 LOO oracle，改用 DATE-LM 自带的 20% holdout split 作为"oracle"近似，用 LDS 排名相关性作为 phi/psi 对齐度的代理指标。

### 4x2 Confounder 矩阵设计的总实验量风险

Empiricist 列出了 8 个 confounders 需要控制，但如果每个都做充分控制，实验矩阵会指数级膨胀。现实的取舍：

- **必须控制**（否则结果无法解读）：C1（implementation asymmetry）、exact-Hessian control、BM25 baseline
- **应该控制但可以简化**：C3（layer selection，只测 3 层）、C4（negative strategy，只测 2 种）
- **可以降为 ablation 附录**：C2（L2 normalization）、random seed（3 seed mean）

### 优先级建议

P0（pipeline 验证）→ P1（2x2 Factorial，核心）→ P2（exact Hessian，关键控制）→ P3（dimension sweep Phase 1+2，机制验证）→ P4（multi-method tournament，不含 oracle）> ~~P5（LOO oracle，超出预算）~~

---

## 综合对比与最终优先级

### 五个 Perspective 总分排名

| Perspective | 总分 | 核心优势 | 核心风险 |
|-------------|------|---------|---------|
| Empiricist | 7.5/10 | 实验设计最严谨，baselines 完整 | P5 oracle 不可行；总实验量被低估 |
| Contrarian | 7/10 | 最关键的 falsification 实验，科学价值高 | "exact Hessian"术语需明确；Daunce 实现成本未评估 |
| Innovator | 6/10 | Angle 1 工程可行（小模型），创新性高 | Angle 1 跨模型规模超时；Angle 3 方法论不可靠 |
| Theoretical | 5/10 | 数学严谨，T7 分类学价值高 | LOO oracle 不可行；NTK 假设对 LLM 适用性存疑 |
| Interdisciplinary | 3/10 | Angle 4（Whitened MF）唯一工程可行 | 其余三个 Angle 工程代价被严重低估，MI 估计不可靠 |

### 跨 Perspective 重叠与整合建议

**实验重叠（可共享计算）**：
1. **梯度协方差特征值分析（Pythia-70M）**：Innovator Angle 1 + Theoretical T1 + Empiricist Angle 2 Phase 1 三个 perspective 都要求这个实验，只需运行一次。
2. **表示提取（DATE-LM, Pythia-1B）**：RepSim、DDA、DiD、Whitened MF 的表示计算可全部共享同一次特征提取，显著节省 GPU 时间。
3. **BM25 baseline**：Contrarian + Empiricist 都要求，成本极低，无条件加入。

**关键工程依赖（必须提前确认）**：
1. DATE-LM 数据集访问方式（HuggingFace dataset？需要特殊申请？）
2. DATE-LM 的 pre-fine-tuning checkpoint 是否提供（影响 DiD、In-the-Wild、ETA 的可行性）
3. TRAK 官方代码对 Pythia-1B + LoRA 的支持状态
4. Pythia-70M 上 exact K-FAC（全特征值分解）的显存需求估计

### 一小时约束下的推荐实验组合

**Pilot 阶段（15 分钟，1x RTX 4090）**：
- RepSim + TRAK (k=512) on Pythia-1B + DATE-LM data selection (1K subsample)
- 目标：验证 pipeline，对齐 DATE-LM leaderboard 数字，测量各方法单次运行时间

**核心实验（分批，每批 ~45-60 分钟，可并行到 4x GPU）**：
- Batch A（~1h，4 GPU 并行）：2x2 Factorial on DATE-LM 3 tasks，Pythia-1B，RepSim + TRAK + DDA baseline + BM25
- Batch B（~1h，1-2 GPU）：Pythia-70M 梯度协方差特征值分析 + dimension sweep Phase 1-2
- Batch C（~1h，1 GPU）：Whitened Matched Filter（Sigma_noise^{-1}）on DATE-LM + exact-Hessian IF 控制实验（Pythia-70M）

**降优先级（超出 1 小时约束或方法论风险高）**：
- LOO oracle SVD（Theoretical T8 + Empiricist P5）：不推荐在资源约束内执行
- IB Lens + MI estimation（Innovator Angle 3 + Interdisciplinary Angle 2）：方法论不可靠
- ETA（Interdisciplinary Angle 1）：工程代价高于声称
- RG scaling（Interdisciplinary Angle 2）：数据点不足，无法可靠拟合

---

## 实用主义总结

从工程和时间约束角度，五份 perspective 的价值集中在以下几个最重要的实验上，按 ROI（科学价值/时间成本）排序：

1. **BM25 baseline**（成本极低，科学价值极高——如果它赢了，整个框架需要重新审视）
2. **DDA 强 baseline + exact-Hessian 控制**（成本低，是 CRA 整个 FM1/FM2 叙事的关键 falsification）
3. **2x2 Factorial（Empiricist P1）**（核心实验，设计成熟）
4. **梯度协方差特征值分析（Pythia-70M, Innovator A1）**（15 分钟 pilot，高信息密度）
5. **Whitened Matched Filter（Interdisciplinary A4）**（即插即用，工程成本低，有理论保证）

避免以下高风险低回报方向：
- 任何依赖 MI/IB 估计的实验（方法论不可靠）
- LOO oracle 计算（在当前资源下不可行）
- 跨 4 个模型规模的完整梯度协方差分析（仅在 Pythia-70M 上做，其他规模降为 pilot 验证）
