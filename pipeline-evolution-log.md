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

<!-- 后续 Entry 在此下方追加 -->
