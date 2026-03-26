## 多维辩论综合（Formalize Review Synthesis）

### 分歧地图

**共识点（4/4 视角）**：
1. **Probe 未执行是最大风险** — 核心假设 H4（RepSim 在 LDS 上与 TRAK 竞争力）零实证支撑。Probe 成本 <1 GPU-day，应立即执行。
2. **MAGIC 威胁真实** — 若 MAGIC 在 Pythia-1B/DATE-LM 上 LDS ~0.95+，FM1 论点直接被反驳。需要明确的决策规则（decision rule）。
3. **三瓶颈框架有智识价值** — Gap 真实（5个独立 repr-space 方法出现，无统一评估），根因分析有深度。
4. **文献缺失** — "Towards Unified Attribution" (arXiv:2501.18887) 是直接概念竞争者，必须引用并区分。

**主要分歧**：
- **是否应在 formalize 通过前执行 probe？**
  - Contrarian/Comparativist/Pragmatist（3/4）：Revise — probe 必须先于 design
  - Interdisciplinary（1/4）：Pass（conditional）— formalize 在智识层面已成熟，probe 是 design 的工作

**独特洞察**：
- **Interdisciplinary**：信号处理类比是深层数学对应（matched filter + differential detection），非表面隐喻。CMRR 可作为 FM2 严重度的量化指标。2×2 交互项应重新框定为"耦合发现"而非"框架失败"。
- **Contrarian**：FM1 最薄弱的支撑腿 — Li et al. 的证据完全来自 LoRA，full-FT 下 FM1 严重性是开放问题。
- **Comparativist**：AirRep (NeurIPS 2025)、LoGra+LogIX (ICLR 2026)、MDA (Jan 2026) 构成间接竞争。
- **Pragmatist**：MAGIC 在 Pythia-1B 上的内存需求可能超过 48GB A6000，probe 实际可行性高。

### 优先级排序

**必须处理（P0）**：
1. 问题陈述中添加 MAGIC 决策规则：若 MAGIC LDS > X 在 Pythia-1B/DATE-LM，论文叙事如何调整
2. 添加 "Towards Unified Attribution" (2501.18887) 到竞争格局讨论
3. 将 probe 执行设为 design 阶段第一优先（非 formalize 阻塞）

**可选处理（P1）**：
4. 引入 CMRR 作为 FM2 量化指标
5. 重新框定 2×2 交互项为"耦合发现"
6. 评估 MAGIC 在目标硬件上的可行性
7. FM1 LoRA-specificity 应升级为核心实验问题

**暂时搁置（P2）**：
8. 添加 Fisher Information metric
9. 因果推断 (IV) 视角
10. Sample-adaptive debias

### 判定

**方向确认（Pass with mandatory revisions）**

三瓶颈框架有真实智识价值，Gap 确实存在，2×2 ablation 设计独特。但 3 个必须修订项需在进入 design 前完成。

**理由**：
- CRA_old 已通过 6-agent 战略审查（Noesis V1），核心方向已验证
- Probe 执行是 design 阶段的职责，不应阻塞 formalize 通过
- 但 MAGIC 决策规则和文献覆盖是 formalize 文档本身的质量问题，必须修复

### 修订要求

1. 在 problem-statement.md §1.3 添加 MAGIC invalidation decision rule（具体数值阈值 + 叙事调整方案）
2. 在 problem-statement.md §2.2 添加 "Towards Unified Attribution" 竞争分析
3. 在 problem-statement.md §1.5 RQ1 中明确 LoRA vs full-FT 作为实验维度

### 下一阶段重点

Design 阶段第一优先：执行 DATE-LM probe（RepSim vs TRAK, <1 GPU-day）

### 未解决的开放问题

1. MAGIC 在 A6000 48GB 上对 Pythia-1B 是否可行？
2. DATE-LM data selection 任务的 contrastive reference 如何构造？
3. RepSim 层选择敏感度 — 是否需要 RepT 的自动层检测？
