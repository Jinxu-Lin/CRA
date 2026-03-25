

## Project Spec
# 项目: cross-task-influence

## 研究主题

Multi-Task VLA 的跨任务数据交互图谱：系统识别哪些任务的训练数据互助、哪些互害，诊断负迁移机制，并基于 task-to-task influence 矩阵优化数据混合策略。

## 背景与动机

### 领域痛点

2025-2026 年 VLA 领域的核心趋势是 scaling——所有头部团队（Physical Intelligence π0、Google RT-2、NVIDIA GR00T、开源 Octo/OpenVLA）都在将来自 N 个任务、M 个机器人、K 个环境的异构数据混合训练。但数据配比目前完全靠手调或均匀混合，缺乏 principled 的工具来理解和优化跨任务数据交互。

**核心问题**：往训练集里加入 task B 的数据可能 hurt task A 的性能（负迁移），但目前没有人能系统地诊断：哪些任务对互助、哪些互害、以及为什么。

### 已知相关工作及其局限

**数据归因 (TDA) 类**：
- **QoQ** (2603.09056, ICRA 2026)：归一化梯度内积做 single-task 内的样本归因。只回答"task X 的训练集里哪条轨迹有用"，不涉及跨任务交互。
- **CUPID** (2506.19121, CoRL 2025)：标准 IF 应用于 Diffusion Policy 的数据策展，同样是 single-task scope。
- **SCIZOR** (2505.22626)：自监督 transition-level 数据策展，Open-X 100 万+ 轨迹 + Octo VLA 验证。大规模但黑盒评分，不解释任务间关系。
- **DataMIL** (2505.09603)：Datamodels 框架，60+ 任务验证，但 single-task scope，不涉及跨任务。
- **MISS** (2409.18153)：证明集合影响力不可加——暗示多任务负迁移不能通过单样本 IF 简单聚合分析。

**数据混合 (Data Mixing) 类**：
- **Re-Mix** (2408.14037)：DRO 域级权重优化，Open X-Embodiment 上 +38%。但是黑盒优化，无归因——只告诉你"权重应该是 X"，不解释为什么。
- **Diversity 论文** (2507.06219, AgiBot World)：发现任务多样性 power-law、单本体可超多本体。群体级消融，无样本级或任务级归因。
- **LESS** (2402.04333, ICML 2024)：LLM 梯度相似度 data selection，5% 数据常优于全量。但是 NLP 领域，未适配 robot/VLA。

**空白**：**没有任何工作在 VLA 上做 task-to-task 级别的数据交互归因**。现有 TDA 工作全是 intra-task，现有 data mixing 工作全是黑盒。

### 来自 VITA 项目的经验教训

我们之前在 ~/Research/VITA 项目中做了两轮 Pilot 实验（Pilot v1 和 Pilot v2），尝试在 frozen backbone VLA (RDT-1B) 上做梯度 TDA。两轮都失败了（SC-1 ≈ 0）——frozen backbone 下 action head 梯度不编码 task-discriminative 信号。但关键可复用资产包括：
- AgiBot World 数据处理 pipeline
- EK-FAC / cosine scoring / MC averaging 代码
- 评估指标体系（SC-1, LOO, Precision@K）
- DPI (Data Processing Inequality) 信息瓶颈理论分析

**重要教训**：本项目应使用 LoRA fine-tuning（QoQ 已验证可行），避免重蹈 frozen backbone 的覆辙。

### 为什么这个方向不与 QoQ 竞争

QoQ 回答的是 intra-task attribution："task X 的训练集里哪条轨迹有用？"
本项目回答的是 cross-task attribution："task Y 的数据对 task X 的性能影响是什么？通过什么机制？"

这是完全正交的问题。QoQ 甚至可以作为本项目的 building block（用 QoQ/IF 计算跨任务 influence）。

## 初始想法

### 核心方法

1. **Task-to-Task Influence 矩阵**：定义 $M_{ij} = \mathbb{E}_{z \in \mathcal{D}_i, z' \in \mathcal{D}_j} [\mathcal{I}(z, z')]$，量化任务 $i$ 的训练数据对任务 $j$ 性能的平均影响。
   - 正值 → 互助（正迁移）
   - 负值 → 互害（负迁移）
   - 近零 → 独立

2. **负迁移机制诊断**：对识别出的负迁移任务对，进一步分析其机制：
   - **Representation conflict**：相似 state 但不同最优 action（e.g., 同一个物体在不同任务中需要不同操作）
   - **Optimization conflict**：梯度方向系统性矛盾
   - **Action distribution mismatch**：训练分布的 action space 不兼容

3. **Influence-guided data mixing**：基于 influence 矩阵优化数据配比——对互助任务对增加共训权重，对互害任务对降低或分离。与 Re-Mix (DRO) 和均匀混合做正面对比。

### 预期贡献

- **C1**: Task-to-task influence 矩阵框架——首次在 VLA 上提供任务间数据交互的量化图谱
- **C2**: 负迁移机制分类——不只是检测，还诊断"为什么"
- **C3**: Influence-guided data mixing 策略——实验证明优于均匀混合和 Re-Mix
- **C4** (Bonus): 与 Diversity 论文的经验发现建立因果连接——解释"为什么单本体可超多本体"

### Pilot 验证方案

在 AgiBot World 上选取 5-10 个任务子集（~1K 轨迹），用 LoRA fine-tuning 的 RDT-1B 或 OpenVLA：
1. 计算 task-to-task influence 矩阵（5×5 或 10×10）
2. 验证 influence 矩阵是否能预测"加入/移除某任务数据后的性能变化"（反事实验证）
3. 如果矩阵中确实存在显著负值（负迁移信号），则方向可行

## 关键参考文献

- QoQ: 2603.09056 (ICRA 2026) — VLA gradient TDA baseline
- CUPID: 2506.19121 (CoRL 2025) — Robot IF data curation
- SCIZOR: 2505.22626 — Self-supervised VLA data curation at scale
- DataMIL: 2505.09603 — Datamodels for robot data selection
- Re-Mix: 2408.14037 — DRO data mixing for IL
- Diversity: 2507.06219 — VLA data scaling (AgiBot World)
- MISS: 2409.18153 — Set influence is non-additive
- LESS: 2402.04333 (ICML 2024) — LLM gradient data selection
- IF-Diffusion: 2410.13850 — GGN^model framework for diffusion
- TrackStar: 2410.17413 — Large-scale gradient TDA
- AirRep: 2505.18513 (NeurIPS 2025) — Representation-based TDA
- ASTRA: 2507.14740 — EKFAC-Preconditioned Neumann IF

## 可用资源

- GPU: 4x RTX A6000 48GB（共享服务器，通过 SSH MCP）
- 服务器: default (SSH MCP connection)
- 远程路径: /home/jinxulin/sibyl_system
- 本地已有 VITA 项目代码（~/Research/VITA/Codes/）可部分复用

## 实验约束

- 实验类型: 轻量训练（LoRA fine-tuning on 1B-7B VLA）
- 模型规模: RDT-1B (1.2B) 或 OpenVLA-7B
- 数据集: AgiBot World (公开, 217 任务类别, 100 万轨迹)
- 时间预算: Pilot ~1-2 GPU-days, 中规模 ~10-15 GPU-days
- 关键约束: 必须用 LoRA 而非 frozen backbone（VITA 已证明 frozen backbone 下梯度 TDA 失败）

## 目标产出

- 论文：目标 CoRL 2026 / RSS 2026（降级目标：ICRA 2027）
- 核心产出：Task-to-task influence 矩阵 + 负迁移机制分类 + influence-guided data mixing

## 特殊需求

- 本项目与 VITA 项目（~/Research/VITA）有知识和代码继承关系，但方向完全不同
- AgiBot World 数据处理 pipeline 可从 VITA 复用
- 论文中需要正面对比 QoQ (intra-task) 和本方法 (cross-task)，说明两者互补而非替代
- 如果 Pilot 阶段发现 task-to-task influence 矩阵中没有显著的负迁移信号（所有任务都互助或独立），需要有 Plan B：转向"positive transfer 的量化和利用"


## User's Initial Ideas
### 核心方法

1. **Task-to-Task Influence 矩阵**：定义 $M_{ij} = \mathbb{E}_{z \in \mathcal{D}_i, z' \in \mathcal{D}_j} [\mathcal{I}(z, z')]$，量化任务 $i$ 的训练数据对任务 $j$ 性能的平均影响。
   - 正值 → 互助（正迁移）
   - 负值 → 互害（负迁移）
   - 近零 → 独立

2. **负迁移机制诊断**：对识别出的负迁移任务对，进一步分析其机制：
   - **Representation conflict**：相似 state 但不同最优 action（e.g., 同一个物体在不同任务中需要不同操作）
   - **Optimization conflict**：梯度方向系统性矛盾
   - **Action distribution mismatch**：训练分布的 action space 不兼容

3. **Influence-guided data mixing**：基于 influence 矩阵优化数据配比——对互助任务对增加共训权重，对互害任务对降低或分离。与 Re-Mix (DRO) 和均匀混合做正面对比。

## Seed References (from user)
- QoQ: 2603.09056 (ICRA 2026) — VLA gradient TDA baseline
- CUPID: 2506.19121 (CoRL 2025) — Robot IF data curation
- SCIZOR: 2505.22626 — Self-supervised VLA data curation at scale
- DataMIL: 2505.09603 — Datamodels for robot data selection
- Re-Mix: 2408.14037 — DRO data mixing for IL
- Diversity: 2507.06219 — VLA data scaling (AgiBot World)
- MISS: 2409.18153 — Set influence is non-additive
- LESS: 2402.04333 (ICML 2024) — LLM gradient data selection
- IF-Diffusion: 2410.13850 — GGN^model framework for diffusion
- TrackStar: 2410.17413 — Large-scale gradient TDA
- AirRep: 2505.18513 (NeurIPS 2025) — Representation-based TDA
- ASTRA: 2507.14740 — EKFAC-Preconditioned Neumann IF

## 文献调研报告（请仔细阅读，避免重复已有工作）
# 文献调研报告

**研究主题**: Multi-Task VLA 的跨任务数据交互图谱：系统识别哪些任务的训练数据互助、哪些互害，诊断负迁移机制，并基于 task-to-task influence 矩阵优化数据混合策略。

**调研时间**: 2026-03-17

**arXiv 搜索关键词**:
- `"vision-language-action" AND "multi-task"` (cs.RO, cs.CV, cs.LG)
- `"negative transfer" AND "multi-task" AND ("robotics" OR "manipulation")` (cs.RO, cs.LG)
- `"task influence" OR "task affinity" OR "task grouping" AND "multi-task learning"` (cs.LG, cs.CV)
- `"data mixing" OR "data mixture" AND ("language model" OR "foundation model")` (cs.LG, cs.CL)
- `"training data influence" OR "data attribution" AND ("multi-task" OR "transfer learning")` (cs.LG)
- `"DoReMi" AND "data mixture"` (cs.CL, cs.LG)
- `"Open X-Embodiment" OR "RT-2-X" OR "Octo"` (cs.RO)

**Web 搜索关键词**:
- `vision language action model multi-task negative transfer cross-task influence 2024 2025`
- `multi-task learning task affinity matrix data mixing optimization state of the art 2025`
- `VLA robot manipulation multi-task training data mixture benchmark 2025`
- `cross-task influence function training data influence multi-task robotics`
- `LIBERO benchmark multi-task robot manipulation negative transfer task grouping`
- `"mix data or merge models" multi-task learning optimization diverse tasks 2024`

## 1. 领域现状摘要

**Vision-Language-Action (VLA) 模型** 已成为机器人操控领域的主流范式。以 OpenVLA (7B)、RT-2-X (55B)、pi_0 等为代表的 VLA 模型，通过在 Internet 规模的视觉-语言数据和大规模机器人示范数据（如 Open X-Embodiment 的 22 种机器人、527 种技能）上联合预训练，展现了跨任务、跨具身的泛化能力。然而，当任务数量增加时，**负迁移（negative transfer）** 问题显著加剧：CORAL (2026) 明确指出 "gradients from different tasks can conflict, causing negative transfer and reducing per-task performance"；STRAP (2024) 发现 "the performance of generalist policies on any one task is often suboptimal due to negative transfer between partitions of the data"；LangGap (2026) 揭示当语义多样性增加时，"model learning capacity proves severely insufficient; even trained tasks perform poorly"。

**多任务学习优化** 领域已发展出一套成熟的工具链：从梯度操控方法（GDOD, PCGrad, CAGrad）到任务分组方法（ETAP, STG-MTL, DMTG, Grad-TAG），再到数据混合优化方法（DoReMi, RegMix, Data Mixing Laws, BiMix, Chameleon）。这些方法主要在 NLP 和 CV 领域验证，但**极少有工作将 task affinity / influence 分析系统性地应用于 VLA 机器人操控场景**。

**数据影响力估计（Data Influence Estimation）** 方法在 LLM 和扩散模型领域蓬勃发展（GPTfluence, DMin, DAS, trajectory-specific LOO），但同样缺乏在多任务机器人策略学习中的应用。这构成了一个明确的研究空白：将 task-to-task influence 分析与数据混合优化方法结合，应用于 VLA 多任务训练场景。

## 2. 核心参考文献

### 2.1 VLA 模型与多任务机器人学习

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 1 | OpenVLA: An Open-Source Vision-Language-Action Model | arXiv 2406.09246 | 2024 | 7B 开源 VLA，970k 真实示范训练，跨 29 任务泛化 | 多任务联合训练时存在任务间干扰，未诊断具体负迁移来源 |
| 2 | Open X-Embodiment: Robotic Learning Datasets and RT-X Models | arXiv 2310.08864 | 2023 | 22 种机器人、527 技能的统一数据集，展示正迁移 | 仅报告平均性能提升，未分析哪些任务组合互害 |
| 3 | CORAL: Scalable Multi-Task Robot Learning via LoRA Experts | arXiv 2603.09298 | 2026 | 每任务一个 LoRA expert 避免参数级跨任务干扰 | 参数隔离方案回避了跨任务知识共享的可能性 |
| 4 | LangGap: Diagnosing and Closing the Language Gap in VLA Models | arXiv 2603.00592 | 2026 | 揭示 VLA 忽略语言指令的问题，多任务训练加剧困难 | 聚焦语言理解缺陷，未量化任务间数据交互 |
| 5 | LangForce: Bayesian Decomposition of VLA via Latent Action Queries | arXiv 2601.15197 | 2026 | 诊断 "Information Collapse"，vision shortcut 导致忽略语言 | 关注单一退化模式，非通用跨任务影响分析 |
| 6 | HyperVLA: Efficient Inference via Hypernetworks | arXiv 2510.04898 | 2025 | 超网络为多任务保留高容量训练、低成本推理 | 未分析任务间训练动态交互 |
| 7 | STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning | arXiv 2412.15182 | 2024 | 子轨迹粒度检索解决跨任务共享低层行为 | 非参数方法，未建立 task influence 矩阵 |
| 8 | SwitchVLA: Execution-Aware Task Switching | arXiv 2506.03574 | 2025 | 动态任务切换，建模执行状态下的任务交互 | 聚焦在线切换而非训练数据层面的跨任务影响 |
| 9 | Discrete Policy: Learning Disentangled Action Space for Multi-Task | arXiv 2409.18707 | 2024 | VQ 离散化多任务动作空间，12 任务下超越 Diffusion Policy 32.5% | 缓解多模态动作分布，但未量化任务间正负迁移 |
| 10 | LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | arXiv 2306.03310 | 2023 | 130 任务、4 个 task suite 评估 lifelong 机器人学习 | 报告 forward/backward transfer 但未建立 task-pair affinity |

### 2.2 多任务学习中的负迁移与任务分组

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 11 | ETAP: Ensemble Prediction of Task Affinity for Efficient MTL | arXiv 2602.18591 | 2026 | 梯度相似度 + 非线性修正器预测 task affinity | 验证于分类任务，未在机器人控制任务验证 |
| 12 | Towards Principled Task Grouping for Multi-Task Learning | arXiv 2402.15328 | 2024 | 理论基础的任务分组方法，无限制性假设 | 通用框架，未针对时序决策/操控任务特化 |
| 13 | Grad-TAG: Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity | arXiv 2409.06091 | 2024 | 梯度低维投影估计 task affinity，仅需 3% FLOPs | 500 任务规模验证（图+NLP），未在 robot policy 验证 |
| 14 | DMTG: One-Shot Differentiable Multi-Task Grouping | arXiv 2407.05082 | 2024 | 全可微分任务分组，同时学习分组和模型权重 | 需要 KN 个 task head 初始化，对大规模任务集计算开销大 |
| 15 | STG-MTL: Scalable Task Grouping Using Data Map | arXiv 2307.03374 | 2023 | 基于训练动态 (Data Maps) 的分类任务分组，扩展到 100 任务 | 仅支持分类任务，需要适配到策略学习 |
| 16 | Efficient Task Grouping Through Samplewise Optimisation Landscape | arXiv 2412.04413 | 2024 | 无需共享模型训练即可推断 pairwise task similarity | 5 倍速度提升但精度受限于 landscape 近似 |
| 17 | Selective Task Group Updates for Multi-Task Optimization | arXiv 2502.11986 | 2025 | 自适应任务分组 + proximal inter-task affinity | 理论分析了分组对任务特定参数学习的影响 |
| 18 | Rep-MTL: Representation-level Task Saliency for MTL | arXiv 2507.21049 | 2025 | 表征空间中量化任务交互，熵惩罚 + 跨任务对齐 | 聚焦 CV dense prediction，未在 action generation 验证 |
| 19 | CMTA: Contrastive Modules with Temporal Attention | arXiv 2311.01075 | 2023 | 对比学习约束模块多样性，细粒度时序注意力组合 | Meta-World 验证，首次超越单任务学习 |
| 20 | Quantifying Task Priority for Multi-Task Optimization | arXiv 2406.02996 | 2024 | 基于连接强度量化任务优先级，新 Pareto 最优解 | 需要额外的任务优先级学习阶段 |
| 21 | "It's a Match!" -- A Benchmark of Task Affinity Scores | arXiv 2301.02873 | 2023 | 系统基准测试多种 affinity score，发现与实际 MTL 性能相关性弱 | 重要的负面结论：简单 affinity score 不可靠 |

### 2.3 数据混合优化

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 22 | DoReMi: Optimizing Data Mixtures Speeds Up LM Pretraining | arXiv 2305.10429 | 2023 | Group DRO 小代理模型学习 domain weights，30x 规模迁移有效 | 仅优化 domain 级别比例，未区分 task-level influence |
| 23 | Data Mixing Laws: Optimizing Data Mixtures by Predicting LM Performance | arXiv 2403.16952 | 2024 | 发现数据混合比例与性能的可预测函数关系 | 聚焦 LLM pretraining domains，未推广到 robot data |
| 24 | RegMix: Data Mixture as Regression for LM Pre-training | arXiv 2407.01492 | 2024 | 回归模型预测最优混合，10% DoReMi 计算即可匹配/超越 | 发现 domain 交互 "often contradicting common sense" |
| 25 | BiMix: Bivariate Data Mixing Law | arXiv 2405.14908 | 2024 | 联合建模 domain 比例和数据量的 scaling behavior | 理论优美但验证限于 LLM pretraining |
| 26 | Chameleon: Flexible Data-mixing Framework | arXiv 2505.24844 | 2025 | Leverage scores + domain affinity matrix 确定混合权重 | 引入 domain affinity matrix 概念，可借鉴到 task affinity |
| 27 | Rethinking Data Mixture for LLMs: A Comprehensive Survey | arXiv 2505.21598 | 2025 | 最全面的 data mixture 方法综述，分类为 offline/online | 综述性质，无新方法 |
| 28 | MixMin: Finding Data Mixtures via Convex Minimization | arXiv 2502.10510 | 2025 | 凸优化框架，小模型 mixture 可迁移到大模型 | 理论贡献但需要大量代理模型训练 |
| 29 | Nemotron-CLIMB: Clustering-based Iterative Data Mixture Bootstrapping | arXiv 2504.13161 | 2025 | 语义聚类 + 迭代搜索 + 代理模型预测，Llama-3.2-1B +2% | NVIDIA 工程化方案，计算资源需求高 |
| 30 | Mix Data or Merge Models? Optimizing for Diverse MTL | arXiv 2410.10801 | 2024 | 模型合并优于数据混合（general +8%, safety +10%），语言级合并有效 | 聚焦 LLM safety/general 多目标，非 robot 场景 |
| 31 | AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization | ACL 2025 | 2025 | Minimax 优化自适应数据采样/重加权 | 聚焦 LLM alignment 场景 |

### 2.4 数据影响力估计

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 32 | GPTfluence: Featurized Simulation for Training Data Influence in GPT Models | arXiv 2404.07840 | 2024 | 参数化模拟训练动态，14M-2.8B 模型泛化 | LLM 专用，未扩展到 action prediction |
| 33 | Capturing Temporal Dependence of Training Data Influence | arXiv 2412.09538 | 2024 | Data value embedding 捕捉训练轨迹特定 LOO | 发现早期和晚期数据影响更大，可指导课程学习 |
| 34 | Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples | arXiv 2405.03869 | 2024 | 无 Hessian 的梯度异常值检测，识别有害训练样本 | Hessian-free 使其可扩展到大模型，但需适配 |
| 35 | Toward Efficient Influence Function: Dropout as Compression | arXiv 2509.15651 | 2025 | Dropout 压缩梯度降低 influence function 计算成本 | 通用方法，可应用于 VLA |

## 3. SOTA 方法与基准

### 3.1 VLA 模型 SOTA

| 模型 | 参数量 | 训练数据 | 关键特性 |
|------|--------|---------|---------|
| pi_0 / pi_0.5 | -- | 大规模多任务 | Physical Intelligence 闭源 SOTA |
| OpenVLA | 7B | 970k demos (Open X-Embodiment) | 开源 SOTA，Llama 2 + DINOv2 + SigLIP |
| RT-2-X | 55B | Open X-Embodiment | 大规模跨具身，被 OpenVLA 以 7x 少参数超越 |
| Octo | -- | Open X-Embodiment | Transformer-based diffusion policy，灵活任务/观测定义 |
| GR00T N1 | -- | Heterogeneous mixture (robot + human video + synthetic) | NVIDIA，异构数据混合策略 |

### 3.2 多任务评测基准

| 基准 | 任务数 | 特点 | 评测指标 |
|------|--------|------|---------|
| LIBERO | 130 | 4 suite (Spatial/Object/Goal/100)，lifelong learning | Success rate, FWT, NBT, AUC |
| LIBERO-PRO | -- | LIBERO 改进版，更公平评测 | Robust success rate |
| VLABench | 100 类 / 2000+ objects | 语言条件操控，强随机化 | Task success rate, generalization |
| Meta-World | 50 | 经典多任务 RL 基准 | Success rate per task |
| SimplerEnv | -- | VLA 模型 OOD 评测 | Success rate (ID/OOD) |
| RoboCasa | -- | 家庭环境操控 | Task completion |

### 3.3 关键评测指标

- **Task Success Rate**: 单任务和平均成功率
- **Forward Transfer (FWT)**: 新任务从旧知识获益程度
- **Negative Backward Transfer (NBT)**: 学习新任务对旧任务的性能退化
- **Inter-task Affinity / Influence Score**: 任务对之间的训练增益/损失
- **Gradient Cosine Similarity**: 任务梯度冲突程度
- **Data Mixing Proportion Sensitivity**: 不同混合比例下的性能变化

## 4. 已识别的研究空白

- **空白 1: VLA 多任务训练中缺乏系统性的 task-to-task influence 分析**。现有 VLA 工作（OpenVLA, RT-X, CORAL）报告平均多任务性能，但未量化具体哪些任务对之间存在正迁移或负迁移。CORAL 的解决方案是参数完全隔离（每任务 LoRA），回避了问题而非解决。

- **空白 2: Task affinity / grouping 方法未在机器人操控策略学习中验证**。ETAP, Grad-TAG, DMTG 等方法主要在分类/NLP 任务验证，机器人操控任务的时序性、连续动作空间、多模态输入使其直接迁移存在挑战。

- **空白 3: Data mixing 优化方法（DoReMi, RegMix 等）未推广到 robot demonstration data**。这些方法假设训练数据可按 "domain" 自然分割，但 robot 多任务数据的 "domain" 实际上就是 "task"，且任务间存在更细粒度的子技能共享（如 STRAP 所揭示的子轨迹级共享行为）。

- **空白 4: 缺乏跨任务影响的诊断工具**。现有方法或者仅提供二元判断（"应该一起训练 vs 不应该"），或者提供梯度级的冲突信号。缺乏一个系统框架来（1）构建完整的 task-to-task influence 矩阵，（2）诊断负迁移的具体机制（梯度冲突、特征空间竞争、数据不平衡），（3）基于诊断结果优化数据混合策略。

- **空白 5: Affinity score 的可靠性存疑**。"It's a Match!" (2023) 的重要负面结论指出 "task affinity scoring does not correlate well with actual MTL performance"，这意味着简单的梯度相似度等 proxy 指标可能不足以指导 VLA 多任务优化，需要更鲁棒的 influence 估计方法。

- **空白 6: 数据影响力方法（influence function 等）未应用于多任务 robot policy**。GPTfluence、trajectory-specific LOO 等方法在 LLM 中有效，但在多任务机器人策略训练中的适用性和计算可行性尚未探索。

## 5. 可用资源

### 5.1 开源代码

- **OpenVLA**: https://github.com/openvla/openvla -- 7B VLA，支持 Open X-Embodiment 数据训练和 LoRA 微调
- **LIBERO**: https://github.com/Lifelong-Robot-Learning/LIBERO -- 130 任务 lifelong 操控基准
- **CORAL**: https://github.com/frontierrobo/CORAL -- 基于 LoRA experts 的多任务 VLA 框架
- **Grad-TAG**: https://github.com/...（ACM KDD 2024 论文，梯度估计 task affinity）
- **DMTG**: https://github.com/ethanygao/DMTG -- 可微分多任务分组
- **RegMix**: https://github.com/sail-sg/regmix -- 数据混合回归优化
- **Chameleon**: https://github.com/LIONS-EPFL/Chameleon -- Leverage scores 数据混合
- **MixMin**: https://arxiv.org/abs/2502.10510 -- 凸优化数据混合
- **HyperVLA**: https://github.com/MasterXiong/HyperVLA -- 超网络 VLA
- **Awesome Embodied VLA**: https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln -- VLA 论文集合

### 5.2 数据集

- **Open X-Embodiment**: 22 种机器人、527 种技能、160k+ 任务的标准化数据集
- **LIBERO**: 130 任务，4 个 task suite（Spatial/Object/Goal/100），含高质量 demo 数据
- **VLABench**: 100 类任务，2000+ objects，自动化数据采集
- **Meta-World**: 50 个机器人操控任务，经典 MT-RL 基准

### 5.3 预训练模型

- **OpenVLA checkpoints** (HuggingFace): openvla/openvla-7b
- **Octo** (HuggingFace): octo-base, octo-small
- **DINOv2, SigLIP**: OpenVLA 使用的视觉编码器
- **Llama 2**: OpenVLA 的语言模型骨干

## 6. 对 Idea 生成的启示

### 6.1 值得深入探索的方向

1. **构建 VLA 多任务的 task-to-task influence 矩阵**：借鉴 Grad-TAG 的梯度低维投影和 ETAP 的集成预测方法，但适配到 VLA 的连续动作空间和时序结构。可在 LIBERO-100 或 Meta-World MT50 上系统量化所有 task-pair 的互助/互害关系。

2. **基于 influence 矩阵的自适应数据混合**：将 DoReMi / RegMix 的框架从 domain-level 下推到 task-level。不同于简单的均匀混合或按数据量比例混合，根据 task influence 矩阵动态调整各任务的采样权重，抑制负迁移来源任务的数据。

3. **负迁移机制的多层次诊断**：不仅测量梯度冲突，还分析特征空间（哪些表征层上任务竞争最激烈）、动作空间（哪些动作维度上任务预测冲突）、数据特征（哪些训练样本是负迁移的主要来源）。

### 6.2 已被充分探索 / 不建议重复的方向

- **简单梯度操控方法**（PCGrad, CAGrad 等）在 VLA 规模下计算成本高且效果不稳定
- **完全参数隔离方案**（如 CORAL 的每任务 LoRA）虽有效但回避了核心科学问题
- **通用 MTL 方法的简单套用**：Azorin et al. (2023) 已表明简单 affinity score 与实际 MTL 性能相关性弱，需要更 robust 的方法

### 6.3 跨领域借鉴的潜力

- **LLM Data Mixing Laws → Robot Data Mixing Laws**: 数据混合的可预测函数关系（Ye et al. 2024）可能同样适用于 robot demonstration 数据，但需要验证
- **Trajectory-specific Data Influence (Wang et al. 2024) → 策略学习的训练动态追踪**: data value embedding 方法可追踪训练过程中不同阶段数据的影响，可用于识别 VLA 训练中负迁移出现的关键时间节点
- **Sub-trajectory Retrieval (STRAP) → 细粒度跨任务共享分析**: 不在任务级别而在子轨迹/技能级别分析跨任务数据交互，因为很多操控任务共享 "pick"、"place"、"approach" 等基本子技能
- **Chameleon's Domain Affinity Matrix → Task Affinity Matrix for VLA**: Chameleon 用 leverage scores 和 domain embedding 构建 affinity matrix，类似思路可用任务 embedding（从 VLA 的语言/视觉编码器提取）构建 task affinity matrix
