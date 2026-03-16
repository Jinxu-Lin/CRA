

## Project Spec
# 项目: CRA (Contrastive Representation Attribution)

## 研究主题
诊断参数空间 TDA 的两个信号处理缺陷（信号稀释 FM1 + 公共影响污染 FM2），将 5 种表示空间 TDA 方法统一为 φ^T·ψ 双线性框架，并在 DATE-LM 标准 benchmark 上系统验证。

## 背景与动机
Training Data Attribution (TDA) 在 LLM 上系统性失败。5 种独立提出的表示空间方法（RepSim, RepT, In-the-Wild, Concept Influence, AirRep）各自在特定任务上优于参数空间方法，但：
1. 从未被识别为一个方法家族
2. 没有统一的诊断解释为什么它们有效
3. 从未在同一 benchmark 上系统评估
4. 实践者缺乏方法选择指南

核心假设：参数空间 TDA 失败源于两个独立的信号处理缺陷：
- **FM1 (Signal Dilution)**: 参数梯度在 ℝ^B 空间中近乎正交（JL 现象），任务信号 SNR 坍塌。表示空间操作通过维度约化（ℝ^B → ℝ^4096）修复。
- **FM2 (Common Influence Contamination)**: 标准 IF 被预训练知识主导。DDA 消融：移除 debias 下降 55pp，移除 denoise 仅 9pp。对比打分修复。

两个缺陷与 Hessian 近似误差正交——三者是互补瓶颈。

## 初始想法
- **攻击角度 A（核心）**: 2×2 消融矩阵 {参数空间, 表示空间} × {标准打分, 对比打分}，在 DATE-LM + Li et al. 两套 benchmark 上评估
- **攻击角度 B（可选升维）**: Fixed-IF — 用理论预测设计参数空间修复（projected IF + contrastive gradient），验证诊断框架的预测力
- **统一框架**: 所有表示空间方法可表达为 φ(z_test)^T · ψ(z_train) 双线性形式
- **信号处理理论类比**: 匹配滤波（维度约化 → 修复 FM1）⊥ 差分检测（对比打分 → 修复 FM2），70+ 年正交性理论基础

### 已完成的前期工作（来自 ~/Research/CRA）
项目已完成 Startup → Crystallize → Strategic Review 阶段：
- 6 维辩论（创新者/务实者/理论家/反对者/跨学科者/实验主义者）→ 共识 "Go with focus"
- 4 方战略审查（反对者/比较分析者/务实者/跨学科者）
- Codex 外部评审（5/10 分，要求加强机制证据）
- 完整的 problem-statement.md 和 contribution.md
- **可直接从 Probe 阶段开始**

### 关键研究问题
- **RQ1**: RepSim/RepT 在 DATE-LM 全部 3 个任务上 vs TRAK/LoGra 的表现？
- **RQ2**: 对比打分是否通用改善参数空间和表示空间方法？（≥2/3 任务 >3pp）
- **RQ3**: FM1 和 FM2 修复增益是否近似可加？（交互项 <30% min 主效应）

### 预注册否证条件
- RepSim < TRAK − 5pp on DATE-LM LDS → "系统性优于"叙事破裂
- 对比打分在 ≥1/3 方法上导致性能下降 >3pp → 通用性不成立
- 2×2 ANOVA 交互项 >30% of min(主效应) → 正交性不成立

## 关键参考文献
- Li et al. 2025 (2409.19998) — RepSim vs IF 在 LLM 上的表现差异
- DDA (2410.01285) — 对比打分使 IF 超越 BM25 (AUC 91.64%)
- RepT (2510.02334) — 表示梯度追踪 (P@10=0.97-1.00)
- DATE-LM (2507.09424) — LLM TDA 标准 benchmark (NeurIPS 2025)
- Better Hessians Matter (2509.23437) — Hessian 层级证据（核心对手）
- In-the-Wild (2602.11079) — DPO 场景下的表示空间 TDA
- Concept Influence (2602.14869) — 概念级表示空间 TDA
- AirRep (2501.12345) — 学习的表示空间 TDA
- Episteme 知识库: ~/Research/Episteme (49 篇 TDA 论文深度分析)

## 可用资源
- GPU: 4x RTX 4090 on xuchang3
- 服务器: ssh -p 8222 jinxulin@xuchang-lab3.staff.sydney.edu.au
- 远程路径: /home/jinxulin/sibyl_system
- 本地知识库: ~/Research/Episteme, ~/Research/CRA (已有前期工作)

## 实验约束
- 实验类型: 轻量训练（fine-tuning based TDA 评估，不训练新模型）
- 模型规模: 中等 — Pythia-1B (pilot), Llama-2-7B (full)
- 时间预算: Pilot ≤1 GPU-day, 核心实验 2-3 周
- 单个实验控制在 1 小时内

## 目标产出
- 论文 — 目标 NeurIPS 2026 / ICML 2027
- 贡献天花板: poster ~ spotlight（若加 Fixed-IF 可升至 oral）

## 特殊需求
- 项目已有前期工作在 ~/Research/CRA，包括完整的 problem-statement、debate 记录、strategic review。建议直接利用这些产出，从 Probe 阶段开始
- Episteme 知识库 (~/Research/Episteme) 包含 49 篇 TDA 论文的深度分析，可作为文献调研基础
- 已知核心风险：RepSim 在 DATE-LM LDS（反事实指标）上可能表现差（相关性≠因果性）


## User's Initial Ideas
- **攻击角度 A（核心）**: 2×2 消融矩阵 {参数空间, 表示空间} × {标准打分, 对比打分}，在 DATE-LM + Li et al. 两套 benchmark 上评估
- **攻击角度 B（可选升维）**: Fixed-IF — 用理论预测设计参数空间修复（projected IF + contrastive gradient），验证诊断框架的预测力
- **统一框架**: 所有表示空间方法可表达为 φ(z_test)^T · ψ(z_train) 双线性形式
- **信号处理理论类比**: 匹配滤波（维度约化 → 修复 FM1）⊥ 差分检测（对比打分 → 修复 FM2），70+ 年正交性理论基础

## Seed References (from user)
- Li et al. 2025 (2409.19998) — RepSim vs IF 在 LLM 上的表现差异
- DDA (2410.01285) — 对比打分使 IF 超越 BM25 (AUC 91.64%)
- RepT (2510.02334) — 表示梯度追踪 (P@10=0.97-1.00)
- DATE-LM (2507.09424) — LLM TDA 标准 benchmark (NeurIPS 2025)
- Better Hessians Matter (2509.23437) — Hessian 层级证据（核心对手）
- In-the-Wild (2602.11079) — DPO 场景下的表示空间 TDA
- Concept Influence (2602.14869) — 概念级表示空间 TDA
- AirRep (2501.12345) — 学习的表示空间 TDA
- Episteme 知识库: ~/Research/Episteme (49 篇 TDA 论文深度分析)

## 文献调研报告（请仔细阅读，避免重复已有工作）
# 文献调研报告

**研究主题**: 诊断参数空间 TDA 的两个信号处理缺陷（信号稀释 FM1 + 公共影响污染 FM2），将 5 种表示空间 TDA 方法统一为 phi^T * psi 双线性框架，并在 DATE-LM 标准 benchmark 上系统验证。

**调研时间**: 2026-03-16

**arXiv 搜索关键词**:
- "topological data analysis" AND "parameter space" AND ("signal dilution" OR "failure mode")
- "topological data analysis" AND ("representation space" OR "activation space") AND ("language model" OR "neural network")
- "persistent homology" AND ("transformer" OR "language model") AND ("benchmark" OR "evaluation")
- "topological data analysis" AND "neural network" AND ("parameter space" OR "weight space")
- ti:"topological data analysis" AND ti:"survey" AND ("neural network" OR "deep learning")
- "persistent homology" AND ("signal dilution" OR "common influence" OR "confound")
- "representation topology divergence" OR "topological similarity" AND "neural network"
- ti:"neural persistence" AND ("generalization" OR "complexity" OR "weight")
- "loss landscape" AND "topology" AND ("persistent homology" OR "topological")
- "data attribution" AND ("representation similarity" OR "influence function") AND "large language model"
- ti:"LESS" AND "data selection" AND "language model"
- ti:"TRAK" AND "training data attribution"

**Web 搜索关键词**:
- topological data analysis parameter space vs representation space neural networks state of the art 2025
- TDA persistent homology language model benchmark DATE-LM 2025
- bilinear framework TDA methods unification phi psi representation 2025
- TDA neural network "parameter space" topology weight persistence "loss landscape" generalization
- "persistence image" "persistence landscape" "Betti curve" vectorization TDA comparison survey
- "representation topology divergence" OR "topological complexity" language model evaluation benchmark
- "neural persistence" Rieck weight space TDA generalization complexity measure
- TDA NLP methods comparison benchmark github 2024 2025
- arxiv 2409.19998 / 2410.01285 / 2510.02334 / 2509.23437 / 2602.11079 / 2602.14869 (种子论文)

---

## 1. 领域现状摘要

Training Data Attribution (TDA) 旨在追踪模型预测回到影响它的训练数据，是 LLM 可解释性、数据估值和安全审计的核心技术。当前 TDA 方法大致分为两大范式：**参数空间方法（Parameter-space）** 和 **表示空间方法（Representation-space）**。

**参数空间方法** 以 Influence Functions (IF) 为代表，通过梯度和 Hessian 逆来估计训练样本对模型参数的影响。经典方法包括 IF、TracIn、TRAK、LESS 等。这类方法有坚实的理论基础（基于 leave-one-out 近似），但面临严重的可扩展性问题：Hessian 逆计算代价高昂，且在 LLM 尺度上近似误差显著。Li et al. (2409.19998) 在系统评估中发现，IF 方法在 LLM 上表现一致较差，而简单的表示相似度方法 (RepSim) 却能达到近 100% 的识别率。

**表示空间方法** 直接在模型的激活空间（activation/representation space）中操作，通过计算测试样本和训练样本的表示相似度来进行归因。近年涌现了多种方法：RepSim（cosine similarity）、RepT（表示梯度追踪）、AirRep（学习的表示）、Concept Influence（概念级归因）、In-the-Wild（激活差分向量）等。这些方法在效果和效率上往往同时超越参数空间方法，但缺乏统一的理论框架来理解它们为何有效、彼此之间有何关系。

**关键张力**: 参数空间方法有理论保证但实践中失败；表示空间方法实践中有效但理论基础薄弱。尚无工作系统诊断参数空间方法失败的根本原因，也无工作将多种表示空间方法统一到一个可分析的框架中。

---

## 2. 核心参考文献

### 2.1 参数空间 TDA 方法及其局限

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 1 | Do Influence Functions Work on Large Language Models? (Li et al.) | arXiv 2409.19998 | 2024 | 系统评估 IF 在 LLM 上的表现；发现 RepSim 远超所有 IF 方法（近100% vs 极低识别率） | 仅诊断现象未分析根因；未提出修复方案 |
| 2 | Enhancing TDA for LLMs with Fitting Error Consideration (DDA) | arXiv 2410.01285 | 2024 | 通过 debias + denoise 策略修复 IF 的拟合误差问题，AUC 达 91.64% | 修复策略是工程性的，未从信号处理角度分析根因 |
| 3 | Better Hessians Matter (Hong et al.) | arXiv 2509.23437 | 2025 | 分解 Hessian 近似步骤，证明更好的 Hessian 近似 → 更好的归因质量；K-FAC 特征值失配是主要误差来源 | 仅在分类任务上验证；未考虑信号稀释和公共影响问题 |
| 4 | TRAK: Attributing Model Behavior at Scale (Park et al.) | arXiv 2303.14186 | 2023 | 随机投影 + After Kernel，用少量模型匹配训练数千模型的归因精度；支持多模态 | 随机投影可能丢失关键信号方向 |
| 5 | LESS: Selecting Influential Data (Xia et al.) | arXiv 2402.04333 | 2024 | Adam 优化器感知的 IF 适配 + 低维梯度相似度搜索；5% 数据选择超越全量训练 | 仍依赖梯度计算，大规模下可扩展性有限 |
| 6 | Imperfect Influence, Preserved Rankings (TRAK theory) | arXiv 2602.01312 | 2026 | 理论证明 TRAK 近似误差大但保留排序相关性 | 确认近似有显著误差，但未分析误差的信号处理含义 |
| 7 | LoRIF: Low-Rank Influence Functions | arXiv 2601.21929 | 2026 | 低秩结构加速 IF 计算至 70B 模型规模；存储降低 20x | 聚焦效率优化，未解决 IF 本身的概念缺陷 |
| 8 | What is Your Data Worth to GPT? (LoGra) | arXiv 2405.13954 | 2024 | 利用反向传播梯度结构的高效投影策略，支持 Llama3-8B | 性能与投影维度强相关，信号稀释问题未被讨论 |

### 2.2 表示空间 TDA 方法

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 9 | Where Did It Go Wrong? (RepT) | arXiv 2510.02334 | 2025 | 表示梯度追踪框架；定位"相变层"后缓存表示+梯度；P@10 达 0.97-1.00 | 未给出与其他表示方法的理论统一 |
| 10 | AirRep: Representational Optimization for TDA | arXiv 2505.18513 | 2025 | 学习任务特定的归因表示 + attention pooling；效率比梯度方法高近两个数量级 | 需要额外训练归因编码器 |
| 11 | In-the-Wild Model Organisms (Activation-based TDA) | arXiv 2602.11079 | 2026 | 激活差分向量做 DPO 场景下的归因；发现"干扰触发服从"行为并缓解 63-78% | 方法设计与 DPO 场景耦合较紧 |
| 12 | Concept Influence | arXiv 2602.14869 | 2026 | 概念级归因（线性探针 / SAE 特征方向）；简单 probe 方法是 Concept Influence 的一阶近似 | 依赖概念探针的质量；未在标准 TDA benchmark 上评测 |
| 13 | Daunce: Data Attribution through Uncertainty Estimation | arXiv 2505.23223 | 2025 | 扰动模型集合的协方差作为归因分数；首次对 GPT 黑盒模型做归因 | 需要多个微调模型，成本仍高 |

### 2.3 标准化评测 Benchmark

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 14 | DATE-LM: Benchmarking Data Attribution Evaluation for LLMs (Jiao et al.) | arXiv 2507.09424; NeurIPS 2025 | 2025 | 统一 LLM 归因评测：训练数据选择 / 毒性过滤 / 事实归因三任务；公开 leaderboard | 发现无单一方法全面领先；非归因基线有时匹敌归因方法 |

### 2.4 TDA 用于神经网络分析（拓扑数据分析背景）

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 15 | TDA for Neural Network Analysis: A Comprehensive Survey (Ballester et al.) | arXiv 2312.05840 | 2023 | 四维度全面综述：架构/决策区域/内部表示/训练动力学 | 未区分参数空间和表示空间 TDA 的信号处理差异 |
| 16 | Unveiling Topological Structures from Language: TDA in NLP Survey (Uchendu & Le) | arXiv 2411.10298 | 2024 | 系统综述 100 篇 TDA-NLP 论文；理论 vs 非理论分类 | 领域还处于早期，标准化评测不足 |
| 17 | Neural Persistence (Rieck et al.) | arXiv 1812.09764; ICLR 2019 | 2019 | 首个基于 TDA 的网络复杂度度量；权重图上的持续同调 → 训练停止准则 | Girrbach et al. 后续工作证明其本质上等价于权重方差 |
| 18 | Addressing Caveats of Neural Persistence (Deep Graph Persistence) | arXiv 2307.10865 | 2023 | 证明 NP 主要受权重方差控制；提出跨层全网络滤过的 Deep Graph Persistence | 仍限于参数空间，信号稀释问题未解决 |
| 19 | Persistent Topological Features in LLMs (Gardinazzi et al.) | arXiv 2410.11042 | 2024 | Zigzag persistence 追踪表示空间拓扑特征跨层演化；用于层剪枝 | 聚焦模型压缩而非数据归因 |
| 20 | Hidden Holes: Topological Aspects of Language Models (Fitz et al.) | arXiv 2406.05798 | 2024 | 提出"perforation"度量表示空间拓扑复杂度；发现 Transformer vs LSTM 拓扑结构显著不同 | 描述性分析，未连接到实际下游任务 |
| 21 | HalluZig: Hallucination Detection using Zigzag Persistence | arXiv 2601.01552 | 2026 | Zigzag persistence 检测 LLM 幻觉；拓扑签名跨模型可泛化 | 专注检测，未与 TDA 归因方法对比 |
| 22 | The Shape of Adversarial Influence (Fay et al.) | arXiv 2505.20435 | 2025 | PH 分析对抗输入对 LLM 表示空间的影响；发现"拓扑压缩"普遍签名 | 聚焦对抗检测而非训练数据归因 |
| 23 | Representation Topology Divergence (RTD) (Barannikov et al.) | arXiv 2201.00058; ICML 2022 | 2022 | 多尺度拓扑发散度量；可比较不同空间中的表示 | 计算成本较高；未针对 TDA 归因场景优化 |

### 2.5 持续同调向量化方法（TDA 工具箱）

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 24 | A Survey of Vectorization Methods in TDA (Ali et al.) | arXiv 2212.09703 | 2023 | 系统比较 13 种向量化方法（PI, PL, Betti, entropy 等） | 未与神经网络归因任务结合 |
| 25 | TDAvec: Vectorization Package (Luchinsky & Islambekov) | arXiv 2411.17340 | 2024 | R/Python 统一包；完整向量化工作流 | 工具导向，未涉及方法对比 |
| 26 | Persistence Spheres: Bi-continuous Representations (Pegoraro) | arXiv 2509.16999 | 2025 | 双连续映射保持 Wasserstein 几何；SOTA 分类/回归性能 | 新方法，社区采纳度待观察 |
| 27 | Qupid: Quantized Persistence and Integral Transforms | arXiv 2312.17093 | 2023 | 对数尺度网格 + 离散变换；极低计算成本 + 竞争性能 | 网格分辨率对性能有影响 |

### 2.6 损失景观拓扑分析

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 28 | Evaluating Loss Landscapes from a Topology Perspective (Xie et al.) | arXiv 2411.09807 | 2024 | 用 merge tree + PD 量化损失景观拓扑；简单拓扑 ↔ 更好性能 | 仅限低维切片分析 |
| 29 | Landscaper: Multi-Dimensional Topological Analysis (Chen et al.) | arXiv 2602.07135 | 2026 | SMAD 指标量化景观平滑度；捕捉传统指标遗漏的训练转变 | 聚焦诊断而非归因 |
| 30 | On the Limitations of Fractal Dimension (Tan et al.) | arXiv 2406.02234 | 2024 | 揭示 PH 维度与泛化的相关性中存在超参混淆效应 | 仅分析参数空间轨迹，未讨论表示空间 |

---

## 3. SOTA 方法与基准

### 3.1 当前最佳方法

**参数空间 SOTA**:
- **TRAK** (Park et al., 2023): 随机投影 + After Kernel，当前可扩展 IF 的标杆
- **LESS** (Xia et al., 2024): Adam 感知 + 低维梯度搜索，5% 数据选择超全量
- **DDA** (Wu et al., 2024): Debias + Denoise，AUC 91.64%
- **LoRIF** (Li et al., 2026): 低秩 IF，扩展至 70B 模型

**表示空间 SOTA**:
- **RepT** (2025): P@10 = 0.97-1.00，三阶段框架（相变层定位 → 缓存 → 归因）
- **AirRep** (2025): 学习归因表示，效率提升约 100x
- **Concept Influence** (2026): 概念级归因，probe 方法是其一阶近似
- **In-the-Wild** (2026): 激活差分向量，比梯度方法便宜 10x+

### 3.2 标准化评测

- **DATE-LM** (NeurIPS 2025): 唯一面向 LLM 的统一 TDA benchmark
  - 三任务：训练数据选择、毒性/偏见过滤、事实归因
  - 核心发现：无单一方法统治所有任务；简单基线（如 BM25）有时匹敌归因方法
  - 公开 leaderboard + 模型 checkpoint

- **Li et al. Benchmark** (2409.19998): IF vs RepSim 对比评测
  - 核心发现：所有 IF 方法在 LLM 上一致表现差，RepSim 近 100% 识别率

### 3.3 主流评测指标

- AUC (Area Under Curve)
- P@K (Precision at K)
- LOO (Leave-One-Out) correlation
- LDS (Linear Datamodeling Score)
- auPRC (Area Under Precision-Recall Curve)

---

## 4. 已识别的研究空白

- **空白 1 — 参数空间失败的根因诊断**: 现有工作（Li et al., Better Hessians）观察到参数空间方法在 LLM 上失败，或分析了 Hessian 近似误差，但未从信号处理角度系统诊断失败模式。特别是：(a) **信号稀释 (FM1)**: 高维参数空间中归因信号被噪声淹没的机制未被形式化；(b) **公共影响污染 (FM2)**: 所有训练样本对模型参数的共同贡献（如通用语言知识）混入归因分数的问题未被分析。

- **空白 2 — 表示空间方法的理论统一**: RepSim、RepT、AirRep、Concept Influence、In-the-Wild 等方法各自独立提出，缺乏统一的数学框架。没有工作将它们归纳为 phi(z_test)^T * psi(z_train) 的双线性形式，也未分析各方法在此框架下对应的 phi/psi 选择。

- **空白 3 — 信号处理理论与 TDA 的桥梁**: 匹配滤波（维度约化修复 FM1）和差分检测（对比打分修复 FM2）在信号处理领域有 70+ 年的理论基础，但这些理论从未被引入 TDA 领域来解释和改进归因方法。

- **空白 4 — 参数空间 vs 表示空间的系统对比**: 尚无工作在控制变量下（同一 benchmark，同一模型，同一评测协议）系统比较 {参数空间, 表示空间} x {标准打分, 对比打分} 的 2x2 消融矩阵。

- **空白 5 — 预测性诊断框架**: 现有工作多是事后分析（观察到 IF 失败 → 提出修补），缺乏能事先预测特定方法在特定场景下会失败的诊断理论。

---

## 5. 可用资源

### 开源代码
- **TRAK**: https://github.com/MadryLab/trak (标准 IF 加速基线)
- **RepT**: https://github.com/plumprc/RepT (表示梯度追踪)
- **AirRep**: https://github.com/sunnweiwei/AirRep (学习归因表示)
- **Deep Graph Persistence**: https://github.com/ExplainableML/Deep-Graph-Persistence (参数空间 TDA)
- **LoGIX**: 伴随 LoGra 发布的数据估值工具包
- **LoRIF**: 低秩 IF 实现
- **TDAvec**: https://cran.r-project.org/web/packages/TDAvec (持续同调向量化 R/Python)
- **AwesomeTDA4NLP**: https://github.com/AdaUchendu/AwesomeTDA4NLP (TDA-NLP 论文集)

### 数据集与 Benchmark
- **DATE-LM**: NeurIPS 2025 公开 benchmark + leaderboard + checkpoint
  - 三任务：data selection, toxicity filtering, factual attribution
  - 支持模型：LLaMA, QWEN, Mistral 等多种架构/规模
- **Li et al. 评测**: LLaMA2/QWEN2/Mistral 上的 IF vs RepSim 对比数据

### 预训练模型
- DATE-LM 提供训练好的模型 checkpoint
- 各 TDA 方法论文通常基于 HuggingFace 上的公开 LLM（LLaMA-2/3, QWEN2, Mistral, OLMo-2 等）

---

## 6. 对 Idea 生成的启示

### 高价值方向

1. **双线性统一框架是核心理论贡献**: 将 5 种表示空间 TDA 方法统一为 phi^T * psi 形式的想法极具吸引力——文献中没有任何类似工作。这不仅提供理论洞察，还能自然推导出各方法的等价条件和最优选择。建议重点打磨此框架的数学严格性。

2. **信号处理类比有巨大潜力但需谨慎**: 匹配滤波 ↔ 维度约化（修复 FM1）和差分检测 ↔ 对比打分（修复 FM2）的对应关系，如果能严格建立，将是跨领域的重大洞察。但需注意：(a) 高维表示空间与经典信号处理假设（如高斯噪声、线性系统）可能不完全匹配；(b) 需要实验验证理论预测的定量准确性，而非仅定性对应。

3. **2x2 消融矩阵是实验设计的亮点**: {参数空间, 表示空间} x {标准打分, 对比打分} 在 DATE-LM 上的系统评测，目前无人做过。DATE-LM 的发现（无单一方法统治）恰好为这种系统性分析提供了动机。

4. **Better Hessians 是核心对手但也是盟友**: Hong et al. 的工作从 Hessian 近似角度分析参数空间方法，但未触及信号稀释和公共影响问题。我们的工作可以将其结论作为"已解决的近似误差"部分，然后论证"即使 Hessian 近似完美，参数空间方法仍有 FM1/FM2 两个根本缺陷"。

### 需要避免的方向

- **纯 TDA（拓扑数据分析）方法**: 文献中大量工作将 persistent homology 用于分析网络结构或检测对抗样本，但这与 Training Data Attribution 是不同的问题。需要明确区分"TDA = Training Data Attribution"和"TDA = Topological Data Analysis"，避免术语混淆。
- **仅做工程优化**: LoRIF、LoGra 等工作已在效率优化上做了大量努力。我们的贡献应聚焦在理论诊断和框架统一上，而非又一个加速方法。

### 跨域借鉴机会

- **信号处理 → TDA**: 匹配滤波和差分检测的正交性理论可直接映射到维度约化和对比打分的正交性。
- **因果推断 → 公共影响**: FM2（公共影响污染）类似于因果推断中的混淆变量问题，差分检测类似于差中差 (difference-in-differences) 方法。
- **Concept Influence 的理论连接**: Kowal et al. 的概念级归因表明，probe 方法是 Concept Influence 的一阶近似——这可能是 phi^T * psi 框架的一个特例，值得深入分析。
