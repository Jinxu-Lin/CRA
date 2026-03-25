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
