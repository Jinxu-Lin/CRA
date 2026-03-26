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

<!-- 后续 Entry 在此下方追加 -->
