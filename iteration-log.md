# Iteration Log

> 本文档记录所有版本变更的结构化历史。**每次版本号变化时追加一条记录**。
> **倒序排列**（最新在最上方），方便 Agent 快速读取最近的迭代上下文。
> 排除方向必须记录——这是防止 Agent 重复已失败方向的核心数据。
> 关键洞察必须记录——失败经验是最有价值的知识资产。

---

## [1.2] -- 2026-03-26 -- Formalize Review Revisions

- **触发**: formalize_review round-1 → revise (3 mandatory revisions)
- **诊断层次**: N/A (review-driven revision, not failure)
- **变更文档**: `research/problem-statement.md` v1.1 -> v1.2
- **关键变更**:
  1. **MAGIC invalidation decision rule** (§1.3): 添加三级决策规则——LDS >= 0.90 则 FM1 弱化并转向 cost-benefit 叙事；MAGIC 不可行则 FM1 默认成立但须声明局限；LDS 0.70-0.90 则混合证据，FM1 为次要瓶颈
  2. **Towards Unified Attribution 竞争分析** (新增 §2.2): arXiv:2501.18887 是概念层统一，CRA 是 TDA-specific 实证诊断，互补而非竞争
  3. **LoRA vs full-FT 升级为核心实验维度** (§1.5 RQ1): FM1 可能是 LoRA artifact，2×2 ablation 必须包含 LoRA 和 full-FT 两种条件
- **排除方向**: None (revisions strengthen existing direction)
- **关键洞察**: formalize_review 4/4 共识 MAGIC 威胁真实，decision rule 将模糊的"需要实验回应"转化为具体的叙事调整方案。LoRA vs full-FT 维度将 2×2 扩展为 2×2×2，需在 design 阶段评估计算成本。

## [1.1] -- 2026-03-25 -- Formalize Refinement

- **触发**: `/praxis-r-formalize` -- 从 assimilated v1.0 精炼为正式问题陈述 v1.1
- **诊断层次**: N/A (refinement, not failure-triggered)
- **变更文档**: `research/problem-statement.md` v1.0 -> v1.1
- **关键变更**:
  1. 修正"Hessian 不重要"叙事为"三个互补瓶颈"（战略审查条件 #1）
  2. 正式引入 MAGIC (LDS ~0.99) 作为 FM1 thesis 的核心挑战，要求实验回应而非论证回避
  3. 降级双线性统一从"理论深度"到"分类便利"，承认 Concept IF 和 AirRep 不完全符合 phi^T psi
  4. 明确标注探针未执行，所有 LDS 预测均为未验证假设
  5. 按优先级排序不确定性：探针结果 > FM1 LoRA artifact > MAGIC invalidation > 对比打分通用性 > 竞争风险
  6. RQ1 重构为 bottleneck decomposition（定量分解三个瓶颈贡献），而非简单的"表示空间 vs 参数空间"比较
- **排除方向**: None (refinement preserves direction)
- **关键洞察**: MAGIC 的 LDS ~0.99 是最大威胁——如果精确 IF 在参数空间已近乎完美，FM1 论断的实际意义取决于 exact IF 的可扩展性边界。CRA 的价值可能从"representation space is better"转变为"representation space is cheaper and nearly as good"。

## [1.0] -- 2026-03-25 -- Assimilation Entry

- **触发**: `/praxis-assimilate` -- 将 CRA_old (Noesis V1) 和 CRA (Sibyl) 融合为 Noesis v3 项目
- **诊断层次**: N/A (assimilation, not failure)
- **变更文档**: All v3 documents created (project.md, research/problem-statement.md, research/method-design.md, research/experiment-design.md, research/contribution.md)
- **融合决策**:
  - **Primary direction**: CRA_old's TDA diagnostic framework (FM1/FM2 + unified taxonomy + DATE-LM benchmark) targeting NeurIPS 2026
  - **Secondary (preserved)**: CRA (Sibyl) VLA cross-task influence direction with LIBERO-10 pilot results, preserved as `iter_001/` and documented in probe_result.md
  - **Rationale**: CRA_old is stronger for NeurIPS 2026 (TDA is NeurIPS core area; VLA cross-task influence targets CoRL/RSS). CRA_old has complete theoretical framework + strategic review PASS. CRA's pilot validates methodology but in different domain.
- **排除方向**: None (assimilation, no direction excluded)
- **从融合中获得的关键洞察**:
  - CRA (Sibyl) pilot confirms cross-task influence is detectable in multi-task settings (dominant positive transfer in LIBERO-10), but pairwise C-LOTO needed for negative transfer detection
  - The representation-space analysis methodology (BCS, gradient projection) from CRA's proposal has conceptual overlap with CRA_old's representation-space TDA focus
  - VITA project failure (frozen backbone gradients carry no signal) reinforces FM1 thesis: signal dilution is real
- **来源项目保留**: CRA_old at ~/Research/CRA_old (read-only reference); Sibyl files (iter_001/, config.yaml, status.json, .sibyl/) untouched in current project
