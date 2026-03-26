# Pipeline Evolution Log

> 本文档记录每次 Phase 执行后的流程反思观察。
> 每条 Entry 是一个快照——记录当时的观察，不做即时修改。
> 在 Retrospective 中综合所有 Entry，正式更新框架文档。

---

## 使用说明

- 每次阶段完成后，Runner 自动注入 X-reflect prompt，agent 将反思追加于此
- Entry 编号递增，格式为 `Entry [序号]`
- 观察前的 `[ ]` 在 retrospective 综合进化时标记已处理 `[x]`

### 观察分类

| 类别 | 含义 |
|------|------|
| **改进 (Improve)** | 现有内容可以做得更好 |
| **确认 (Confirm)** | 好的实践，值得保留和强调 |
| **边界 (Boundary)** | 阶段边界模糊，需要澄清 |

---

## Entry 1 -- Assimilation -- 2026-03-25

**执行模式**: /praxis-assimilate (fusion of CRA_old + CRA)

### 观察

#### 确认 (Confirm)
- **[跨阶段]** -- CRA_old (Noesis V1) 的 Crystallize + Strategic Review 产出（project-startup.md, problem-statement.md, contribution.md）质量高，六维辩论有效覆盖了战略风险。直接迁移到 v3 research/problem-statement.md 仅需格式适配。
- **[跨阶段]** -- CRA (Sibyl) 的 falsification-first experimental design（Phase 0 pilot with pre-registered gates）是良好实践，v3 的 probe_design + probe_impl 可以借鉴这种门控结构。

#### 改进 (Improve)
- [ ] **[Prompt: assimilate] [中]** -- 融合两个不同系统（Noesis V1 + Sibyl）的项目时，最大挑战是决定主方向。当前缺乏结构化的方向比较框架（venue匹配度、理论成熟度、实验进展、竞争态势等维度的加权评估）。
- [ ] **[跨阶段] [低]** -- CRA_old 的 pipeline-evolution-log.md 中有关 C (Crystallize) 和 RS (Strategic Review) 阶段的反思，这些对 v3 的 formalize 和 formalize_review 有直接参考价值，但 assimilation 流程没有自动迁移这些反思。

#### 边界 (Boundary)
- [ ] **[跨阶段] [中]** -- Assimilation 后项目应该进入哪个阶段需要仔细判断。CRA_old 有 problem-statement + review PASS，相当于 v3 的 formalize + formalize_review 完成。但 method-design 和 experiment-design 是 assimilation 时新生成的草稿，尚未经过 design_review。建议：设置为 design 阶段（需要正式的方法+实验设计），而非 formalize（已完成问题定义）。

## Entry 2 -- Formalize -- 2026-03-25

**执行模式**: /praxis-r-formalize (problem statement refinement from assimilated v1.0 to v1.1)

### 观察

#### 确认 (Confirm)
- **[Formalize]** -- CRA_old 的六维辩论产出了明确的方向修正条件（4 conditions），这些条件为 formalize 提供了具体的改进清单而非模糊指导。formalize_review -> revise 循环的价值在于产出 actionable conditions。
- **[Formalize]** -- Episteme KB 的 Gap/Hypothesis 索引（G-RepT4, G-AR2, H-IF-LLM4, H-RF1 等）为 root cause 分析提供了结构化的外部证据引用，减少了 agent 自我引用的风险。

#### 改进 (Improve)
- [ ] **[Prompt: formalize] [高]** -- 当项目有 assimilated problem-statement（来自另一系统的 review-passed 版本），formalize prompt 应明确区分"refine existing"和"create from scratch"两种模式。当前 prompt 默认 create-from-scratch 流程，导致 agent 需要自行判断哪些内容保留、哪些需要升级。
- [ ] **[Prompt: formalize] [高]** -- MAGIC 作为 exact IF baseline（LDS ~0.99）是 CRA 的核心挑战，但 formalize prompt 没有强制要求 agent 评估"已有方法是否已经解决了声称的 Gap"。任何 gap statement 都应包含"为什么现有最强 baseline 不足"的显式论证。
- [ ] **[跨阶段] [中]** -- 探针未执行时的 formalize 本质上是"基于理论推演的 gap 定义"，所有 LDS 相关的 prediction 都是未验证假设。Prompt 应在探针未执行时自动降低 solvability 评级并标注所有 empirical predictions 为 unverified。

#### 边界 (Boundary)
- [ ] **[Formalize <-> Probe] [高]** -- CRA 的 probe 在 Init Module（probe_design + probe_impl）中定义但从未执行。formalize 阶段的 RQ formulation 依赖 probe 结果（H4 假设完全无证据支撑）。当 probe 未执行时，formalize 的 gap assessment 存在系统性过度自信风险。建议：formalize 应有一个 gate 检查："probe_result.md 是否包含实际执行结果？"如果不包含，所有 empirical claims 必须标注 [UNVERIFIED]。

## Entry 3 -- Formalize Review (Round 1) -- 2026-03-26

**执行模式**: formalize_review round-1 → revise (3 mandatory issues)
**时间分配**: 辩论 agents 并行执行占 ~50%，synthesis + report 占 ~40%，reflection ~10%

### 观察

#### 确认 (Confirm)
- **[Formalize Review]** -- 4/4 共识的 P0 issues（MAGIC decision rule, Unified Attribution, LoRA vs full-FT）是精确且 actionable 的修订要求。产出质量取决于共识检测的准确性。
- **[Formalize Review]** -- "Pass with mandatory revisions" 作为判定模式有效：方向无根本问题时，不需要重新审查整体方向，只需验证具体修订。这减少了 Round 2 的审查负担。

#### 改进 (Improve)
- [ ] **[Prompt: formalize-review] [中]** -- Round 1 的 3 个 P0 issues 和 4 个 P1 issues 之间的边界不够清晰。P1 #7（FM1 LoRA-specificity 升级为核心实验问题）实际上被 formalize 阶段处理为 P0 级别，说明 P0/P1 分类在 synthesis 中不够准确。

## Entry 4 -- Formalize Review (Round 2) -- 2026-03-26

**执行模式**: formalize_review round-2 → pass (revision verification)
**时间分配**: 辩论 agents 并行执行占 ~40%，synthesis + report 占 ~50%，reflection ~10%

### 观察

#### 确认 (Confirm)
- **[Formalize Review]** -- Round 2 作为 revision verification 高效运作。4/4 agents 确认 3 个 Round 1 issues 已解决，无新 blocking issues。审查焦点从"发现问题"转为"验证修复"，大幅减少审查时间。
- **[跨阶段]** -- Codex 外部审查（non-blocking）也返回 Pass，提供独立验证。external + internal 共识增强判定可信度。

#### 改进 (Improve)
- [ ] **[Prompt: formalize-review] [中]** -- Round 2 revision verification 模式下，4 个 debaters 的完整辩论可能过重。当 Round 1 issues 明确且修订针对性强时，可考虑精简审查流程：仅运行 2 个最相关 agents（如 Contrarian + Comparativist）+ synthesizer，节省 ~50% token 成本。
- [ ] **[跨阶段] [低]** -- Pragmatist 在 Round 2 识别出 2x2x2 compute expansion 的预算影响（72-144 GPU-days），这类信息对 design 阶段至关重要，但从 formalize_review synthesis 到 design prompt 的信息传递路径不明确。建议：design prompt 应自动读取最新 formalize_review synthesis 中的"建议改进"和"战略预判"部分。

#### 边界 (Boundary)
- [ ] **[Formalize Review <-> Design] [中]** -- 多个 agents 指出 probe 执行是 design 阶段第一优先。但 design prompt 目前不强制 probe-first 顺序。建议：当 probe_result.md 标记为"NOT YET EXECUTED"时，design prompt 应包含显式 gate："先执行 probe，再进行方法设计。"

## Entry 5 -- Design -- 2026-03-26

**执行模式**: 首次 (first entry, joint method + experiment design)
**时间分配**: 探针信号消化 + 资源预算 ~20%, 方法框架设计 ~30%, 实验矩阵设计 ~35%, 风险/失败预案 ~15%

### 观察

#### 缺失 (Missing)
- [ ] **[Prompt: design] [高]** -- Design prompt 要求"消化探针结果"，但 probe_result.md 标记为 "NOT YET EXECUTED"。当无探针结果时，prompt 未给出替代流程（如：基于间接证据的条件设计 + 探针作为 Experiment 0）。当前执行中自行发明了 "Experiment 0 probe + Experiment 0.5 mini pilot" 的两级门控，这应该成为 prompt 的标准模式。
- [ ] **[Prompt: design] [中]** -- Prompt 要求检查 Episteme Methods Bank + Experimental Patterns，但 Episteme 没有这些子目录（知识存储在论文笔记中，非结构化）。要么 Logos 应产出结构化的 methods-bank/ 目录，要么 design prompt 应改为"搜索 Episteme 论文笔记中的相关方法"。

#### 改进 (Improve)
- [ ] **[Prompt: design] [中]** -- "解空间探索"（Step 4a）对于诊断型项目（CRA 不提出新方法，而是诊断框架）有些错位。CRA 的"方法组件"是已有方法的配置组合，不是新算法设计。Design prompt 应区分"novel method design"和"diagnostic framework design"两种模式，后者的重点是实验矩阵设计而非方法创新。
- [ ] **[跨阶段] [中]** -- formalize_review synthesis 中的"建议改进"（CMRR metric, 2x2x2 compute budget, probe-first priority）是 design 的关键输入，但 design prompt 没有显式要求读取 formalize_review synthesis。当前依赖用户在 fork prompt 中注入这些信息。建议：runner 自动注入最近一次 formalize_review 的 synthesis 文档路径。

#### 确认 (Confirm)
- **[Design]** -- 方法和实验同步设计（"耦合原则"）在 CRA 中高度有效。每个方法组件（repr-space, contrastive, MAGIC, LoRA vs Full-FT）都直接映射到一个实验，反之亦然。映射表（§3）是 design 的核心交付物，确保无孤立组件或无支撑实验。
- **[Design]** -- 资源预算前置（Step 2）有效约束了设计空间。在确定 60 GPU-day 总量后，每个实验的规模自然受限，避免了"先设计再砍"的低效循环。

#### 边界 (Boundary)
- [ ] **[Design <-> Blueprint] [中] [BOUNDARY]** -- Design 和 Blueprint 之间的边界在 CRA 项目中模糊。CRA 的"方法"是已有方法的组合配置，不需要复杂的实现蓝图。Blueprint 阶段对于 CRA 可能退化为"DATE-LM codebase 集成计划"——这更像是工程规划而非研究设计。建议：对于诊断型项目，考虑合并 design + blueprint 或简化 blueprint 为"实现检查清单"。

## Entry 6 -- Design Review (Round 1) -- 2026-03-26

**执行模式**: 首次 (design_review round-1, 6 debaters + synthesizer)
**时间分配**: 文档审读 ~15%, 辩论 agents 并行执行 ~45%, synthesis + report ~35%, reflection ~5%

### 观察

#### 改进 (Improve)
- [ ] **[Prompt: design-review] [中]** -- 6 debaters 产出大量文本（each ~800-1200 words），synthesizer 需要整合 ~6000 words 的辩论记录。辩论质量高但存在冗余：Skeptic 和 Contrarian 在 TRAK projection confound 上几乎完全重合，Empiricist 和 Methodologist 在 representation extraction protocol 上也高度重叠。6 agents 对于 design review 是合理的（覆盖不同维度），但 synthesizer 应有显式的 dedup 策略。
- [ ] **[Prompt: design-review] [中]** -- 审查维度中缺少"计算预算验证"作为独立维度。Pragmatist 发现的 LDS evaluation cost uncertainty 是本轮最严重的实际问题，但它不是 10 个审查维度中任何一个的核心（最接近的是"评估协议完整性"但重点不同）。建议：添加"计算可行性"作为第 11 个审查维度，或在"评估协议完整性"维度中显式包含 compute budget audit。

#### 确认 (Confirm)
- **[Design Review]** -- 6-agent 辩论有效覆盖了技术审查的多个层面。强信号问题（3+ agents 共识）确实是最重要的问题：representation extraction protocol、LDS cost uncertainty、TRAK projection confound。独立发现（单 agent）也有高信息价值：Theorist 的 Taylor expansion 建议、Skeptic 的 random-model control。
- **[Design Review]** -- Pass 判定合理。所有 mandatory modifications 是 additions（增加控制实验、指定实现细节），不需要重新设计方法或实验框架。2x2 ablation 核心设计经受住了 6-agent 审查。

#### 边界 (Boundary)
- [ ] **[Design Review <-> Blueprint] [低] [BOUNDARY]** -- Pass 判定附带 4 个 mandatory additions。这些修改应在哪个阶段执行？Design 阶段（回写 method-design.md 和 experiment-design.md）还是 Blueprint 阶段（纳入实现计划）？当前 routing 是 Pass → Blueprint，但 Blueprint prompt 可能不知道这些 mandatory additions。建议：Runner 在 Pass with mandatory additions 时，将 additions 列表注入 Blueprint fork_prompt。

<!-- 后续 Entry 在此下方追加 -->
