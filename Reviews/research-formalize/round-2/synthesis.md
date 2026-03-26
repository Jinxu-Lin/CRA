# 问题形式化审查报告

## Round 2 — Revision Verification

**Context**: Round 1 returned "revise" with 3 mandatory issues. Problem-statement.md v1.2 addresses all three. This round evaluates whether the revisions are adequate.

## 多视角辩论摘要

**辩论 Agents**: Contrarian, Comparativist, Pragmatist, Interdisciplinary

**强信号问题（多视角共识）**:

1. **All 3 Round 1 issues are resolved** (4/4 consensus):
   - MAGIC invalidation decision rule (§1.3): Three-tier threshold with concrete narrative pivots. Contrarian confirms thresholds are reasonable; Interdisciplinary confirms alignment with detection theory; Pragmatist notes "infeasible" is the most likely branch at Pythia-1B scale.
   - Towards Unified Attribution (§2.2): Correctly positioned as complementary (conceptual taxonomy vs TDA-specific empirical). Comparativist confirms no remaining literature gaps.
   - LoRA vs full-FT (§1.5 RQ1): Elevated to core experimental dimension. Interdisciplinary provides signal-processing rationale (FM1 should be MORE severe under full-FT, LESS under LoRA -- if opposite observed, FM1 is conditioning not dimensionality problem).

2. **Probe remains the critical gate** (4/4 consensus): H4 (RepSim competitive on LDS) has zero empirical support. All agents agree this is the single highest-risk assumption. However, all 4 agents also agree this is a design-phase concern, not a formalize-phase blocker -- the probe design is well-specified and cheap (<1 GPU-day).

3. **2x2x2 compute expansion needs design-phase scoping** (Pragmatist + Contrarian): Adding LoRA vs full-FT creates 8 conditions. Pragmatist estimates 72-144 GPU-days for full evaluation (3 tasks x 8 conditions x 3 seeds). This is feasible on 4x A6000 but tight for NeurIPS 2026 timeline.

**重要独立发现**:

- **[Interdisciplinary]** The 2x2 interaction term (RQ3) may be larger than expected. Signal processing theory predicts non-zero interaction when interference is spatially structured (pre-training knowledge in representation space). The 30% threshold may be too generous.
- **[Contrarian]** If MAGIC achieves LDS 0.90-0.95 AND is feasible, the cost-benefit-only paper has a ceiling of workshop/poster level. This scenario is underspecified in the current decision rule.
- **[Pragmatist]** MAGIC's metagradient at Pythia-1B likely exceeds 48GB A6000 memory, making the "infeasible" branch most probable. This is pragmatically favorable (FM1 thesis holds) but scientifically weaker.
- **[Comparativist]** LoGra/LogIX (ICLR 2026) is a minor gap in §2.2 but does not warrant Revise -- they address efficient computation, not the representation-space vs parameter-space distinction.

**分歧议题裁判**:

- No substantive disagreements in Round 2. All 4 agents recommend Pass.

---

## 各维度评估

### 1. Gap 真实性与推导系统性
**判定**: Pass
**证据**: Five independently proposed representation-space methods with zero cross-comparison on a common benchmark is an acknowledged gap (G-RepT4, G-AR2). The three-bottleneck decomposition (Hessian, FM1, FM2) is a systematic derivation, not intuition-based. v1.2 correctly positions the bilinear unification as "taxonomic convenience, not theoretical depth" (§2.3 point 2).
**与顶会标准对标**: The gap is real and the derivation is systematic. Meets NeurIPS standards for problem motivation.

### 2. Gap 重要性与贡献天花板
**判定**: Pass
**证据**: TDA for LLMs is a high-activity area (50+ papers in KB, multiple NeurIPS/ICML 2025-2026 papers). The benchmark contribution (representation-space methods on DATE-LM) has standalone citation value. Diagnostic framework adds conceptual depth. Ceiling: solid NeurIPS poster, with oral potential if the 2x2 ablation produces clean and surprising results.
**改进建议**: Design phase should consider whether a lightweight "FixedRep" method (representation-space + contrastive = addressing both FM1 and FM2) could elevate the ceiling from diagnostic to method contribution.

### 3. Gap 新颖性 + 竞争态势
**判定**: Pass
**证据**: No direct competitor does three-bottleneck decomposition + systematic benchmark. "Towards Unified Attribution" (2501.18887) is correctly differentiated (conceptual taxonomy vs empirical diagnostics). Concurrent risk is medium -- individual method papers may fill benchmark gaps piecemeal, but the systematic decomposition study has 6-12 months of differentiation.

### 4. Root Cause 深度
**判定**: Pass
**证据**: Three-layer root cause (symptom → Hessian → FM1 → FM2) with inter-bottleneck interactions explicitly discussed. MAGIC tension honestly confronted with decision rule. LoRA-specificity of FM1 evidence acknowledged and elevated to core experimental dimension.

### 5. 攻击角度可信度
**判定**: Pass
**证据**: Diagnostic framework + benchmark evaluation is a well-understood research template. The 2x2 ablation design is information-dense (any outcome pattern yields publishable results). Attack angle limitations honestly assessed in §2.3 (5 limitations listed).

### 6. RQ 可回答性与可证伪性
**判定**: Pass
**证据**: All three RQs have explicit falsification criteria with concrete thresholds. RQ1: MAGIC within 3pp of RepSim → FM1 negligible. RQ2: RepSim LDS < TRAK LDS - 5pp → narrative refuted. RQ3: interaction > 30% of minimum main effect → independence fails. v1.2 adds LoRA vs full-FT dimension to RQ1 with clear reframing contingency.

### 7. 探针结果整合质量
**判定**: Pass (with caveat)
**证据**: Probe NOT executed -- and the problem statement is fully transparent about this throughout. §3.1 explicitly states "PROBE NOT YET EXECUTED." §1.4 rates solvability as "Medium" with "probe NOT been executed -- solvability assessment is provisional." The VLA pilot is correctly flagged as irrelevant to the TDA thesis. All 4 agents agree: probe is a design-phase gate, not a formalize-phase blocker.

---

## 竞争态势分析

Based on Comparativist's assessment:

- **直接竞争工作（近 12 月）**: None that combines three-bottleneck decomposition + systematic benchmark. Individual competitors (MAGIC, DDA, RepT, AirRep, Better Hessians Matter) each address one bottleneck.
- **差异化空间评估**: Strong. The systematic decomposition + benchmark is unique. The signal processing framing (Interdisciplinary) adds theoretical depth.
- **竞争窗口估计**: 6-12 months. Risk is piecemeal erosion (individual method papers filling benchmark gaps) rather than a single competing work.

## 贡献天花板评估

- **预期贡献级别**: Solid (diagnostic framework + benchmark). Could reach significant if the 2x2 produces clean, surprising results or if a lightweight FixedRep method emerges.
- **目标 venue 匹配度**: NeurIPS 2026 -- well-matched. TDA is a core NeurIPS topic.
- **一句话 pitch 测试**: "We decompose LLM TDA failure into three independent bottlenecks and show that representation-space methods bypass the dominant one, providing the first systematic evaluation of five independently proposed methods on DATE-LM."

---

## 问题清单

**必须修改（Revise / Abandon 级）**:
None. All Round 1 mandatory issues resolved.

**建议改进（Pass 级，可选）**:
1. [Contrarian] Specify the ceiling for cost-benefit-only paper if MAGIC achieves LDS 0.90-0.95 feasibly (§1.3 decision rule middle tier). Acknowledge this is poster-level ceiling.
2. [Interdisciplinary] Consider CMRR as secondary FM2 metric in design phase.
3. [Interdisciplinary] The 2x2 interaction term may be larger than 30% -- consider adjusting or adding interpretation guidance for interaction-heavy results.
4. [Pragmatist] Design phase must explicitly budget the 2x2x2 compute expansion (LoRA vs full-FT adds 2x multiplier).
5. [Comparativist] Minor: mention LoGra/LogIX (ICLR 2026) in related work for completeness.

---

## 战略预判

1. **进入 design 后最可能遇到的技术挑战**: DATE-LM evaluation pipeline integration. Understanding and correctly interfacing with LDS counterfactual evaluation is non-trivial (Pragmatist estimate: 1-2 days just for evaluation protocol).

2. **需换攻击角度时的备选**: If RepSim fails on LDS entirely, pivot to "correlation vs causation in representation-space TDA" -- the gap between P@K and LDS becomes the primary contribution (informative negative result + diagnostic framework).

3. **此方向最大的 unknown unknown**: Whether the 2x2 ablation produces clean results. If FM1 and FM2 effects are tangled, task-dependent, and model-size-dependent, the clean "three independent bottlenecks" narrative becomes "it's complicated" -- still publishable but less impactful.

---

## 整体判定：Pass

The v1.2 revisions are targeted, complete, and address all three Round 1 mandatory issues:

1. **MAGIC invalidation decision rule** (§1.3): Concrete three-tier rule with actionable thresholds and narrative pivots. Confirmed sound by all 4 agents.
2. **Towards Unified Attribution competitive analysis** (§2.2): Correctly positioned as complementary, not competitive. Literature coverage now comprehensive.
3. **LoRA vs full-FT as core dimension** (§1.5 RQ1): Elevated from secondary ablation to core experimental question with clear reframing contingency.

No new blocking issues were identified. The remaining risks (probe unexecuted, MAGIC invalidation, FM1-FM2 interaction magnitude, 2x2x2 compute budget) are all design-phase concerns that do not warrant further revision of the problem formalization. The problem statement is rigorous, honest about uncertainties, and provides sufficient structure for the design phase.

**Recommendation**: Proceed to design phase. First priority: execute the DATE-LM probe (RepSim vs TRAK, < 1 GPU-day) to gate H4.
