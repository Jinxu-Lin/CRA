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

## Entry 7 -- Blueprint -- 2026-03-26

**执行模式**: 首次 (code architecture + experiment execution plan)
**时间分配**: 探针复用评估 ~5%, 组件-文件映射 + 架构设计 ~25%, experiment-todo 细化 ~40%, CLAUDE.md 编写 ~25%, reflection ~5%

### 观察

#### 确认 (Confirm)
- **[Blueprint]** -- 深浅解耦原则（core/ vs experiments/）在 CRA 中非常自然。所有 attribution 方法是 core/，每个实验是 experiments/ 下的薄脚本。这验证了 Entry 5 的观察：CRA 的"方法"是已有方法的配置组合，blueprint 的核心价值是 config-driven 实验矩阵而非复杂架构。
- **[Blueprint]** -- Design review 的 4 个 mandatory additions（token aggregation, LDS timing, Grad-Sim baseline, random-model RepSim）能无缝整合到 experiment-todo.md 中，因为它们都是"增加一个实验条件"而非"修改架构"。这印证了 design review Pass 判定的正确性。

#### 改进 (Improve)
- [ ] **[Prompt: blueprint] [中]** -- Blueprint prompt 要求"探针代码复用评估"（Step 1），但对 CRA 这种探针未执行、probe/ 目录为空的项目，此步骤退化为"确认无可复用代码"。更有价值的评估是"DATE-LM codebase 复用评估"——哪些功能 DATE-LM 已提供、哪些需自行实现。Prompt 应扩展 Step 1 为"已有代码资产评估"（探针代码 + 外部 codebase + 相关项目代码）。
- [ ] **[Prompt: blueprint] [中]** -- Blueprint prompt 禁止编写实际 .py 文件，但同时要求建立目录结构。创建空目录（`mkdir -p`）是合理的，但创建的空子目录（如 `core/attribution/`, `experiments/probe/`）在 git 中不会被跟踪（git 不跟踪空目录）。建议：要么允许创建 `__init__.py` 占位文件，要么明确说明目录结构仅在 CLAUDE.md 中描述、implement 阶段创建。

#### 边界 (Boundary)
- [ ] **[Design Review -> Blueprint] [中] [BOUNDARY]** -- 再次出现 Entry 6 的观察：design review 的 mandatory additions 需要在 blueprint 中集成，但 blueprint prompt 不自动接收这些信息。本次执行依赖用户在 fork prompt 中显式注入 synthesis 文档路径。建议：Runner 在 design_review Pass with mandatory additions 后，自动将 additions 列表写入 `phase-outcomes/design_review.json` 的 `mandatory_additions` 字段，blueprint prompt 显式读取此字段。

## Entry 8 -- P1 (Paper Outline) -- 2026-03-26

**执行模式**: 首次 (paper outline generation, PLACEHOLDER mode)
**时间分配**: 材料审读 + 映射 ~30%, 叙事脊柱 + 策略决策 ~20%, 章节大纲 + 图表规划 ~35%, 符号表 + 一致性检查 ~15%

### 观察

#### 确认 (Confirm)
- **[P1]** -- Problem-driven 叙事策略与 Analysis Paper 类型高度匹配。CRA 不提出新方法，核心卖点是"诊断分解 + 系统基准"，problem-driven（展示失败 -> 分析根因 -> 提出框架 -> 验证）自然对齐。Prompt 的三种策略选项（contrastive, insight-driven, problem-driven）覆盖了主要情况。
- **[P1]** -- Contribution-Evidence alignment matrix 是高价值检查。CRA 的 C0-C3 每个都映射到 Method + Experiment 位置，无悬空贡献。这种前置检查在 outline 阶段捕获叙事缺口远比在 P2 写作中发现更高效。

#### 改进 (Improve)
- [ ] **[Prompt: 30-paper-outline-prompt] [中]** -- PLACEHOLDER 模式下，outline 中的图表计划本质上是"预期结果的视觉化方案"，但 prompt 未区分"基于真实数据的图表设计"和"基于预测趋势的骨架设计"。后者需要更明确标注哪些设计决策（如 Fig.4 的 grouped bar vs heatmap）依赖于实际数据分布，可能在 P2 时需修改。建议：PLACEHOLDER 模式下，图表计划增加一列"设计稳定性"（高：不依赖数据；中：布局稳定但细节待定；低：完全依赖数据模式）。
- [ ] **[Prompt: 30-paper-outline-prompt] [中]** -- Related Work 的位置决策（Method 前 vs Method 后）是一个重要的结构选择，但 prompt 没有给出决策框架。CRA 选择 Method 后放置 RW（因为三瓶颈框架是组织文献的工具），这对于 Analysis Paper 合理，但对于 Method-Heavy Paper 可能不适用。建议：prompt 中添加 RW 位置决策指南（Method 后：当框架是组织工具时；Method 前：当方法有高理解门槛需要背景铺垫时）。

#### 边界 (Boundary)
- [ ] **[P1 <-> P2] [低] [BOUNDARY]** -- Outline 的空间预算总计 ~103%（超出 ~0.3 页），需要在 P2 写作时压缩。但压缩决策（哪个 section 削减）依赖于实际写作中各 section 的自然篇幅。P1 能做的是标注"如超空间，优先压缩 S4.7 效率分析和 S4.6 Scale-up"——这种优先级信息对 P2 有价值，但 prompt 未要求产出。

## Entry 9 -- P2 (Section Writing) -- 2026-03-26

**执行模式**: 首次 (PLACEHOLDER mode, Mode B direct writing)
**时间分配**: Method section ~25%, Experiments section (table skeletons + PENDING) ~30%, Introduction ~20%, Related Work ~10%, Conclusion + Abstract ~15%

### 观察

#### 确认 (Confirm)
- **[P2]** -- Writing order (Method -> Experiments -> Intro -> Related Work -> Conclusion -> Abstract) is highly effective. Writing Method first established the exact notation and framework terminology, which then flowed consistently through all subsequent sections. Writing Abstract last was essential since it summarizes claims that solidified during Experiments and Introduction writing.
- **[P2]** -- PLACEHOLDER mode works well for an Analysis Paper. Method and Related Work are 100% writable without experimental data. Introduction is ~95% writable (only key numbers pending). The experiment section's table skeletons with conditional analysis ("If results match expectations...") provide useful structure for the future fill phase.

#### 改进 (Improve)
- [ ] **[Prompt: 31-paper-sections-prompt] [中]** -- The prompt's Method section guidance assumes a novel algorithm paper ("Component Details" in "logical order, not code structure"). For a diagnostic framework paper like CRA, the Method section is really "Framework + Diagnostic Design." The prompt should offer an alternative structure for Analysis Papers: "Framework Formalization -> Diagnostic Methodology -> Experimental Design Rationale" rather than "Overview -> Components -> Training/Inference."
- [ ] **[Prompt: 31-paper-sections-prompt] [中]** -- In PLACEHOLDER mode, the Experiments section is the most time-consuming because each table requires careful construction of placeholder IDs, descriptions, and expected ranges. A template or naming convention for PENDING placeholders (e.g., `method_task_metric` pattern) would reduce cognitive overhead and ensure consistency for the fill phase.
- [ ] **[跨阶段] [低]** -- outline.md's Related Work placement decision ("after Method" with note "if reviewers prefer standard ordering, adjust in P2") required judgment during P2. The outline should make a definitive decision rather than deferring, or provide explicit criteria for P2 to decide.

#### 边界 (Boundary)
- [ ] **[P2 <-> P3] [中] [BOUNDARY]** -- PLACEHOLDER mode creates a tension: P3 (Critique) will evaluate sections where ~40% of content is `{{PENDING:...}}`. The critique's ability to assess argument strength, evidence sufficiency, and contribution validation is fundamentally limited. P3 should have a PLACEHOLDER-aware mode that focuses on structural and narrative critique rather than evidence critique.

## Entry 10 -- P3 (Cross-Review Critique) -- 2026-03-26

**执行模式**: 首次 (5-role independent critique, PLACEHOLDER mode)
**时间分配**: 全文通读 + 理解 ~20%, 5-role 独立审查 ~60%, Summary synthesis + priority actions ~15%, reflection ~5%

### 观察

#### 确认 (Confirm)
- **[P3]** -- 5-role 独立审查在 PLACEHOLDER 模式下仍有高信息价值。Novelty, Soundness, Presentation 三个维度完全可评（不依赖实验数据）。Experiment 维度可评设计但非结果。Reproducibility 可评实现细节完整性。Entry 9 的 BOUNDARY 观察（P3 应有 PLACEHOLDER-aware 模式）是正确的，但实践中 PLACEHOLDER 模式限制可通过调整审查焦点自然应对。
- **[P3]** -- 最高价值发现是 Soundness 维度的 FM1/JL argument 问题（Critical）：JL 适用于随机向量，但梯度是结构化的。这类根本性逻辑问题在写作中容易被忽视，P3 的独立审视有效捕获。

#### 改进 (Improve)
- [ ] **[Prompt: 32-paper-critique-prompt] [中]** -- PLACEHOLDER 模式下，Experiment Critic 的评分受限（无法评估结果质量、统计显著性、ablation 效果）。Prompt 应为 PLACEHOLDER 模式提供调整后的评分标准：Experiment 评分仅基于设计质量（baseline 选择、消融覆盖、统计计划），明确注明"结果填充后需二次评审"。
- [ ] **[Prompt: 32-paper-critique-prompt] [低]** -- Summary 的 Priority Actions 对 P4 至关重要，但 P4 prompt 是否自动读取 critique/summary.md 不确定。建议：P4 prompt 应显式要求读取 summary.md 并按 Priority 1/2/3 顺序处理。

#### 边界 (Boundary)
- [ ] **[P3 <-> P4] [中] [BOUNDARY]** -- 部分 issues 标记为 "needs-additional-analysis"（如梯度内积分布实证）或 "needs-additional-experiments"（如增加 seeds）。这些超出 P4（编辑阶段）的范围。P3 应更清晰地区分"P4 可修"和"需回到 implement 阶段"的 issues，或建议在 P4 中仅处理 rewrite-fixable issues，其余记录到 experiment-design.md 的 TODO 中。

---

## Entry 5 -- P4 (Integration & Editing) -- 2026-03-26

**执行模式**: First Integration (no prior review.md)
**时间分配**: Edit plan creation ~10%, Section edits (method critical+major) ~35%, Section edits (intro/related/experiments/conclusion) ~30%, Paper assembly ~15%, Self-check + reflection ~10%

### 观察

#### 确认 (Confirm)
- **[P3 -> P4]** -- P3 critique summary.md with prioritized action list was highly effective for P4. The Priority 1/2/3 grouping and per-issue tags (rewrite-fixable vs needs-additional-analysis) made triage immediate. Entry 4's suggestion (P4 prompt should explicitly read summary.md) is confirmed valuable.
- **[P4]** -- "Minimal invasion" editing principle worked well. Most edits were surgical: reframing claims, adding caveats, tightening language. No paragraph-level rewrites were needed except for the abstract (which was a known requirement).

#### 改进 (Improve)
- [ ] **[Prompt: 33-paper-integrate-prompt] [中]** -- The prompt says "edit precisely maps to critique items" but several critique items (e.g., CMRR rename, bilinear taxonomy demotion) require coordinated edits across 3+ files (method, experiments, notation, conclusion). A cross-file impact checklist in the prompt would reduce missed cascading updates.
- [ ] **[Prompt: 33-paper-integrate-prompt] [低]** -- Section renumbering (critique #23) affects section cross-references throughout the paper. The prompt should flag this as a high-cascade edit requiring a grep-and-replace pass, not a local fix.

#### 边界 (Boundary)
- [ ] **[P4 <-> implement] [中] [BOUNDARY]** -- Issues #8 (gradient inner product distribution), #15 (increase seeds), and #14 (add methods) are tagged "needs-additional-analysis/experiments" but P4 can only address them via reframing. The boundary between "reframe the claim" (P4 scope) and "run additional experiments" (implement scope) should be explicit in the prompt. Current approach: reframe JL as analogy + cite empirical evidence, narrow benchmark claim, add seed escalation plan. These feel like the right P4 actions but the prompt does not guide this judgment.

## Entry 6 -- P5 (Final Review) -- 2026-03-26

**执行模式**: 首次
**时间分配**: Detailed read and P3 cross-check ~50%; scoring calibration ~25%; review writing ~25%

### 观察

#### 确认 (Confirm)
- **[P3 -> P5]** -- The P3 critique summary with per-issue severity tags and section-level grouping made the cross-check highly efficient. 17 of 22 issues could be verified as fixed by targeted reading rather than full re-review. The Priority 1/2/3 grouping from P3 directly maps to what matters in P5.
- **[P5]** -- The six-dimension scoring rubric with accept references (">= 7: clear new insight") is well-calibrated. The override rules (any dimension <= 4 -> revise; novelty <= 5 -> revise) prevent gaming the composite score with one strong dimension masking a fatal weakness.

#### 改进 (Improve)
- [ ] **[Prompt: 34-paper-review-prompt] [中]** -- The prompt says "simulate reviewer reading" with a "first pass (15-minute skim)" and "second pass (detailed read)." For a PLACEHOLDER-mode paper, the first pass is less informative because tables are empty and there are no figures to skim. The prompt could have a conditional note: "For PLACEHOLDER papers, focus the first pass on structural flow and argumentation rather than numbers/figures."
- [ ] **[Prompt: 34-paper-review-prompt] [中]** -- Scoring a PLACEHOLDER paper requires an explicit policy on how to handle missing data. The current prompt says nothing about this. I adopted the user's instruction ("score structure, argumentation, and design quality; do not penalize for missing data") but this should be codified in the prompt itself -- otherwise different reviewers will apply different standards.

#### 边界 (Boundary)
- [ ] **[P5 <-> P4] [中] [BOUNDARY]** -- The pass/revise decision at P5 creates a loop back to P4. For PLACEHOLDER papers, what exactly should P4 fix on a revise? The paper's weaknesses are primarily about what happens when results fill in (length, novelty depending on interaction term). These are not P4-addressable issues. The prompt should distinguish "structural revise" (fixable by P4 editing) from "results-dependent concerns" (not fixable until experiments run).

#### 缺失 (Missing)
- [ ] **[Prompt: 34-paper-review-prompt] [低]** -- No guidance on how to weight the P3 cross-check relative to fresh review. I spent significant time verifying each P3 issue fix status (22 items), which is valuable for quality tracking but may crowd out independent review if the paper has changed substantially between P3 and P5. A time budget suggestion (e.g., "P3 cross-check should be ~20% of review effort") would help.

## Entry 13 — P6 (LaTeX Compilation) — 2026-03-26

**执行模式**: 首次
**时间分配**: Markdown-to-LaTeX conversion ~60%, bibliography compilation ~15%, PENDING placeholder handling ~15%, template/packaging ~10%

### 观察

#### 改进 (Improve)
- [ ] **[Prompt: 35-paper-latex-prompt] [中]** — The prompt instructs to "prompt user to download venue template" for the .sty file, but in automated pipeline execution there is no interactive user. The skill should check for existing .sty files in common locations (Papers/sty/, Papers/latex/) before falling back. In this case the .sty already existed at Papers/sty/neurips_2026.sty.
  - 建议: Add a file-discovery step at the beginning: search Papers/sty/, Papers/latex/, and project root for venue template files.

- [ ] **[Prompt: 35-paper-latex-prompt] [中]** — Step 4 (Figure Processing) assumes figures already exist in Codes/_Results/. For PLACEHOLDER-mode papers with no experiments run, this step is entirely vacuous but the prompt does not acknowledge this case. Time was spent checking for nonexistent figures.
  - 建议: Add a conditional: "If paper is in PLACEHOLDER mode (contains {{PENDING:...}}), skip figure processing and insert placeholder figure boxes instead."

#### 确认 (Confirm)
- **[当前阶段]** — The PENDING placeholder conversion strategy (using a \pending{} LaTeX command rendering as red text) works well for maintaining visibility of incomplete items while producing compilable LaTeX. This is a clean approach for PLACEHOLDER-mode papers.

- **[当前阶段]** — Having the neurips_2026.sty already available in Papers/sty/ made template selection trivial. The project's sty directory convention is effective.

#### 边界 (Boundary)
- [ ] **[跨阶段] [低] [BOUNDARY]** — The prompt says to produce Papers/latex/main.pdf but pdflatex is not available on this macOS machine. The compilation step is environment-dependent and cannot be guaranteed. The prompt's fallback ("suggest Overleaf") is adequate but could be more explicit about when compilation is expected to succeed vs. when it is optional.
  - 建议: Mark compilation as "best-effort" explicitly in the exit criteria, rather than listing it as a primary deliverable.

## Entry 14 — P7 (Project-Level Review) — 2026-03-26

**执行模式**: 首次
**时间分配**: Critic review ~40%, Supervisor review ~30%, Synthesis ~20%, reflection ~10%

### 观察

#### 改进 (Improve)
- [ ] **[Prompt: 36-project-review-prompt] [中]** — The P7 prompt instructs Critic and Supervisor to independently assess the paper, but in PLACEHOLDER mode (43 pending items), both reviews inevitably converge on the same dominant issue (no results). The Critic's adversarial role is underutilized because the most devastating attack is trivially obvious. For PLACEHOLDER-mode papers, the prompt could branch: Critic should focus on structural/logical vulnerabilities that persist regardless of data, and Supervisor should focus on which experimental outcomes would change the paper's framing.
  - 建议: Add a PLACEHOLDER-mode section to the P7 prompt that redirects Critic toward framework robustness analysis and Supervisor toward outcome scenario planning.

- [ ] **[跨阶段] [高]** — The Paper Module (P1-P7) ran to completion on a paper with zero experimental results. The pipeline did not gate on experiment execution. The probe is defined as a "CRITICAL GATE" in problem-statement.md but the paper module proceeds regardless. This means the pipeline can produce a polished but empty paper, which is wasted effort if the probe fails.
  - 建议: Add a soft gate between the Research Module's implement phase and the Paper Module. If no experiment_result.md exists or contains "NOT YET EXECUTED," the paper module should warn at P1 that it is operating in speculative mode and suggest running experiments first.

#### 确认 (Confirm)
- **[当前阶段]** — The multi-role review structure (Critic + Supervisor + Synthesis) is effective. Even in PLACEHOLDER mode, the two roles identified different secondary concerns: Critic focused on MAGIC invalidation and missing baselines, Supervisor focused on pivot framings and venue fit. The synthesis consensus-weighting mechanism (doubling weight for issues flagged by multiple roles) correctly elevated the two critical issues.

- **[当前阶段]** — The proxy metric gaming checklist is valuable as a forcing function. Even though no actual gaming can be detected without data, the checklist forced systematic examination of LDS reliability, single-metric dependence, and pre-specified prediction ranges -- all genuine concerns for this paper.

#### 边界 (Boundary)
- [ ] **[跨阶段] [中] [BOUNDARY]** — P7 is defined as the final paper module phase, but for this project the most urgent output is not a review document but a prioritized experiment execution plan. The P7 synthesis naturally produces action items, but these feed into the Research Module (implement phase), not back into the Paper Module. The handoff from P7 action items to actual experiment execution is not formalized in the pipeline.
  - 建议: After P7 in PLACEHOLDER mode, the pipeline should suggest returning to the Research Module's implement phase with P7's action items as input, rather than marking the paper module as complete.

<!-- 后续 Entry 在此下方追加 -->
