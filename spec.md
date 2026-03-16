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
