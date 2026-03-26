# 设计审查报告 — Round 1

## 多视角辩论摘要

**强信号问题**（多视角共识）：

1. **Representation extraction protocol unspecified (Empiricist + Pragmatist + Methodologist)**: Method-design.md does not specify the token aggregation strategy for RepSim (last token? mean pooling? all tokens?). For autoregressive LLMs like Pythia, this implementation detail can make or break the method. All three agents independently flagged this as a gap that could invalidate the probe results.

2. **LDS evaluation cost uncertainty threatens budget feasibility (Pragmatist + Empiricist)**: LDS requires retraining models with data subsets removed. The 60 GPU-day budget assumes ~0.28 GPU-days per condition, but if DATE-LM's LDS evaluation requires per-condition model retraining, the actual cost could be 1.5-3x higher. Pragmatist identified this as the single biggest risk to the project timeline. Empiricist reinforced that LDS timing must be measured during the probe.

3. **TRAK projection confound in the 2x2 design (Skeptic + Contrarian + Empiricist)**: The 2x2 ablation uses TRAK (with random projection) as the parameter-space representative. The FM1 main effect therefore conflates "representation space vs parameter space" with "learned compression vs random projection." Multiple agents recommended including Grad-Sim as an additional parameter-space cell to isolate the pure FM1 effect.

4. **MAGIC tension with FM1 thesis remains the core theoretical vulnerability (Theorist + Contrarian)**: MAGIC's LDS 0.95-0.99 in parameter space directly challenges the FM1 narrative. If exact IF works in parameter space, FM1 (signal dilution) may be irrelevant. The decision rules in problem-statement.md §1.3 are appropriate, but the theoretical framework cannot currently explain WHY MAGIC succeeds if FM1 is real.

**重要独立发现**：

- **[Theorist]** First-order Taylor expansion connecting RepSim to influence: I(z_test, z_train) ~ nabla_h L(z_test) . (h_ft(z_train) - h_base(z_train)). This would make the signal-processing analogy predictive rather than decorative, and specifically predicts LoRA > Full-FT approximation quality.

- **[Skeptic]** Random-model RepSim control: computing RepSim with an untrained model (< 1 GPU-hour) would distinguish "learned feature quality" from "dimensionality reduction" as the explanation for RepSim's success. Highly informative for minimal cost.

- **[Methodologist]** Parameter-matched ablation for RepT: RepT doubles feature dimension (concat[h, nabla_h L]). Need RepSim at l* with dimension-matched random features to isolate the gradient component's contribution.

- **[Contrarian]** TRAK with PCA projection (learned rather than random): if this matches RepSim, FM1 thesis collapses — the problem is random projection quality, not parameter space per se.

- **[Pragmatist]** MAGIC disk space requirements (400-800GB for checkpoints) may exceed shared server capacity.

**分歧议题裁判**：

- **Skeptic (interaction threshold 50%) vs Experiment Design (30%)**: Skeptic argues 30% is too lenient for an "independence" claim. **Judgment**: The 30% threshold in the experiment design is reasonable for a diagnostic framework that acknowledges "partial overlap." The paper should present the full interaction magnitude and let readers judge. The framing (independent vs partially separable vs tangled) should be determined by the data, not pre-committed. **Keep 30% as the threshold for narrative adjustment, but report the full interaction magnitude.**

- **Contrarian (FM1 doesn't exist) vs Theorist (FM1 has theoretical support but needs formalization)**: **Judgment**: Both positions have merit. MAGIC's success doesn't disprove FM1 — it proves that FM1 can be overcome with exact IF at enormous compute cost. The question is whether FM1 is a PRACTICAL bottleneck when using approximate methods (TRAK), which the 2x2 experiment directly tests. The experiment design correctly treats FM1 as a testable hypothesis, not an assumed truth. **No change needed, but the paper must clearly state that FM1 is a hypothesis about approximate IF, not about parameter space in general.**

## 方法侧审查

### 逻辑闭合

**Gap → Root Cause → Method → Why Solves chain**:

- Gap → Root Cause: **Adequate**. Three bottlenecks (Hessian, FM1, FM2) are identified with evidence from multiple papers. The decomposition is logically sound, though the independence of these bottlenecks is an assumption (tested by Experiment 2).
- Root Cause → Method: **Adequate with caveats**. Component A (repr-space) directly targets FM1. Component B (contrastive) directly targets FM2. Component C (MAGIC) bounds Hessian error. The mapping is complete.
- Method → Why Solves: **Partially adequate**. The "why" is articulated via signal-processing analogies but not formalized. The Theorist identified a Taylor expansion argument that could strengthen this link.
- **Hidden assumption**: FM1 and FM2 are the practical bottlenecks of approximate IF. If Hessian error alone explains TRAK's poor performance (as Better Hessians Matter suggests), the FM1/FM2 decomposition adds no explanatory power.

**Rating**: Revise-level. Direction correct, but the "why" link needs strengthening, and the MAGIC tension needs explicit handling in the method framework (not just in the decision rules).

### 组件必要性

Each component maps to a specific bottleneck: A→FM1, B→FM2, C→Hessian, D→FM1 generality. No "show-off" components detected. The 2x2 design inherently tests necessity (each cell is a condition with/without a specific fix). **Occam's Razor is satisfied** — this is a diagnostic framework, not a novel method, so the "components" are experimental conditions, not architectural additions.

**Rating**: Pass.

### 理论正确性

- JL concentration argument: Mathematically correct for random vectors; application to gradients is an unverified extrapolation (Theorist).
- Contrastive scoring: Influence linearity assumption (I_total = I_pretrain + I_finetune) is not theoretically justified for nonlinear models (Theorist).
- Numerical stability: No concerns — all operations are cosine similarities and score subtractions, numerically benign.
- Tensor shape consistency: RepSim output is N-dimensional (one score per training sample), matching DATE-LM's evaluation input. RepT concatenation is (2d)-dimensional, then reduced to N-dimensional scores. Consistent.

**Rating**: Revise-level. The influence linearity assumption underlying contrastive scoring should be explicitly stated and tested. Not a blocker since it's an empirical hypothesis.

### 与探针结果的一致性

**PROBE HAS NOT BEEN EXECUTED.** All design decisions are based on indirect evidence (Li et al., DATE-LM, RepT, MAGIC). The method-design.md correctly flags this (§1: "All design decisions below are based on theoretical analysis and indirect evidence").

The design appropriately gates the full experiment on the probe result (Experiment 0), with clear pass/fail criteria and pivot plans. The contingency for probe failure (§1.4 of experiment-design.md) is adequate.

**Rating**: Pass (given the probe-gated design). The probe is the critical gate and must be executed before full commitment.

### Scalability 评估

RepSim: O(N * d) per test sample — trivially scalable. RepT: O(N * 2d) + one backward pass per sample — moderate, ~5 min per 10K samples. TRAK: O(N * k) + full backward per sample — standard. MAGIC: O(N * n * T) — infeasible at scale, acknowledged. Contrastive: 2x base method.

Scaling to Llama-7B: LoRA fine-tuning feasible on 48GB. Full-FT infeasible. RepSim at 7B: forward pass + representation extraction, ~14GB model + ~160MB representations. Feasible.

**Rating**: Pass. Scalability is well-analyzed.

### 训练稳定性分析

Not directly applicable — CRA does not train a new model. Fine-tuning is done by DATE-LM's protocol. The only training concern is Full-FT at Pythia-1B (Experiment 3), where learning rate sensitivity and gradient checkpointing are addressed.

**Rating**: Pass.

## 实验侧审查

### RQ 覆盖度

| RQ | Experiments | Coverage |
|----|------------|----------|
| RQ1 (Bottleneck Decomposition) | Exp 2 (2x2), Exp 3 (LoRA vs FT), Exp 4 (MAGIC) | Complete |
| RQ2 (Rep-Space Benchmark) | Exp 1 | Complete |
| RQ3 (FM1-FM2 Independence) | Exp 2 (interaction term) | Complete |

No orphan experiments. No overclaimed results. Each claim has clear experimental backing.

**Rating**: Pass.

### Baseline 公平性与时效性

- All methods evaluated on same model checkpoint: fair.
- TRAK, Grad-Sim, BM25 from DATE-LM codebase: current (NeurIPS 2025).
- DDA reimplemented as TRAK_ft - TRAK_base: reasonable approximation.
- RepT reimplemented: needs verification against original results (Methodologist).
- MAGIC: best-effort reimplementation, acknowledged as potentially infeasible.
- **Missing**: EK-FAC (cite DATE-LM numbers at minimum).
- **Missing**: Grad-Sim in the 2x2 as additional parameter-space control (multiple agents).

**Rating**: Revise-level. Need Grad-Sim in the 2x2 and EK-FAC citation.

### Ablation 完整性

The 2x2 design IS the ablation. Each cell tests a specific combination of FM1/FM2 fixes.

**Missing ablations identified**:
1. Layer sweep for RepSim (Methodologist): should be promoted from "conditional" to "required."
2. Contrastive reference ablation (Methodologist): test at least one alternative reference.
3. Parameter-matched RepT ablation (Methodologist): isolate gradient contribution.
4. Random-model RepSim control (Skeptic): minimal cost, high information value.

**Rating**: Revise-level. Layer sweep and random-model control should be required.

### 探针 → 完整实验衔接

The probe-to-full transition is well-specified (experiment-design.md §1). Scale-up dimensions are clearly listed. Failure contingencies exist.

**One gap**: The probe does not measure LDS evaluation wall-clock time (Pragmatist). This must be added to the probe protocol to validate the compute budget.

**Rating**: Pass with minor addition (measure LDS evaluation time in probe).

### 评估协议完整性

- Data split: no leakage risk (low, per Methodologist).
- Hyperparameter selection: on validation set (learning rate sweep for Full-FT).
- Multiple runs: 3 seeds, mean +/- std.
- Metrics: LDS (primary), AUPRC, Recall@50, MRR, P@K. Comprehensive.
- Statistical tests: permutation test, bootstrap CI, Benjamini-Hochberg FDR. Strong.
- **GPU time within budget**: Uncertain (Pragmatist's LDS cost concern). Must verify during probe.

**Rating**: Revise-level due to budget uncertainty. Pragmatist's LDS cost concern must be resolved.

### 计算预算可行性

**This is the most serious practical concern.** Pragmatist identified that LDS evaluation cost could blow the budget. The 60 GPU-day estimate is based on scoring time only, not including LDS evaluation retraining. If DATE-LM uses leave-K-out with per-subset retraining:
- 54 conditions x retraining cost per condition could exceed 100 GPU-days.
- Budget mitigation options: reduce seeds, use subset LDS, share retraining across methods.

**Rating**: Revise-level. Budget must be recalibrated during probe. Need explicit contingency plan if LDS evaluation is expensive.

### 超参敏感度评估

Partially addressed. Need to add:
- RepSim token aggregation strategy exploration
- TRAK projection dimension sensitivity
- Full-FT learning rate sensitivity reporting

**Rating**: Revise-level (minor).

## 联合维度

### 方法-实验对齐

**Method → Experiment mapping**:

| Component | Ablation/Experiment | Claim Validated |
|-----------|-------------------|-----------------|
| A (RepSim/RepT) | Exp 1 (benchmark), Exp 2 (2x2 row) | FM1 fix effectiveness |
| B (Contrastive) | Exp 2 (2x2 column) | FM2 fix effectiveness |
| C (MAGIC) | Exp 4 | Hessian bottleneck bound |
| D (LoRA vs FT) | Exp 3 | FM1 generality |
| 2x2 interaction | Exp 2 ANOVA | FM1-FM2 independence |

**Experiment → Method mapping**:

| Experiment | Claims Tested | Method Component |
|-----------|--------------|-----------------|
| Exp 0 (Probe) | H4 (RepSim competitive) | A |
| Exp 0.5 (Pilot) | H3 (additivity) | A + B |
| Exp 1 (Benchmark) | RQ2 (rep-space evaluation) | A |
| Exp 2 (2x2) | RQ1 (decomposition), RQ3 (independence) | A + B |
| Exp 3 (LoRA vs FT) | RQ1 (FM1 generality) | D |
| Exp 4 (MAGIC) | RQ1 (Hessian bound) | C |
| Exp 5 (Scale-up) | Generalization | All |

**Alignment assessment**: Complete. Every component has at least one experiment. Every experiment validates a specific claim. Cross-references in method-design.md and experiment-design.md are consistent.

**Missing alignment**: The "correlation vs causation" gap (P@K vs LDS) is identified as an important finding but doesn't map cleanly to any component. It's a diagnostic output of the framework, not a fix. This is acceptable.

**Rating**: Pass.

## 问题清单

**必须修改**：

1. **Specify representation extraction protocol** — Source: Empiricist, Pragmatist, Methodologist — Severity: Serious — Description: Token aggregation strategy (last token / mean pooling / all tokens) for RepSim and RepT is unspecified. For autoregressive LLMs, this choice can change results by 10-20pp. — Suggested fix: Specify "last token representation at layers L/2 and L" as default; include mean-pooling as a secondary comparison in the probe. — Impact: Method-design.md §5 Component A.

2. **Add LDS evaluation timing to probe protocol** — Source: Pragmatist, Empiricist — Severity: Serious — Description: The 60 GPU-day budget does not clearly account for LDS evaluation retraining cost. If DATE-LM's LDS requires per-condition retraining, budget could be 2-3x insufficient. — Suggested fix: In Experiment 0, measure and report wall-clock time for one full LDS evaluation cycle. Include budget recalibration step before committing to Experiments 1-5. Add explicit contingency plan (reduce seeds, use AUPRC-only for some conditions, subset LDS). — Impact: Experiment-design.md §2.1, method-design.md §2 budget table.

3. **Include Grad-Sim in the 2x2 ablation** — Source: Skeptic, Contrarian, Empiricist — Severity: Moderate — Description: Using TRAK (with random projection) as the sole parameter-space representative confounds FM1 with projection quality. Grad-Sim (no projection) provides a cleaner parameter-space baseline. — Suggested fix: Run 2x2 with both TRAK and Grad-Sim as parameter-space representatives. Report FM1 main effect for both. If RepSim advantage over Grad-Sim differs substantially from RepSim advantage over TRAK, the projection quality confound is real. — Impact: Experiment-design.md §3.2, minor budget increase (~4 additional conditions x 3 tasks x 3 seeds).

4. **Add random-model RepSim control to probe** — Source: Skeptic — Severity: Moderate — Description: RepSim with untrained model representations would distinguish "learned feature quality" from "dimensionality reduction" as the explanation. Takes < 1 GPU-hour. — Suggested fix: Add to Experiment 0: compute RepSim with randomly initialized Pythia-1B. Report alongside trained RepSim and TRAK. — Impact: Experiment-design.md §2.1.

**建议改进**：

- **Layer sweep for RepSim** — Source: Methodologist — Promote from "conditional" (§8.1) to "required" ablation. Estimated cost: minimal (forward pass at all layers, ~2 GPU-hours). — Impact: Experiment-design.md §8.1 → §3.1 or new §3.6.

- **First-order Taylor expansion argument** — Source: Theorist — Formalize the connection between RepSim and influence via Taylor expansion. This would make the theoretical framework predictive (LoRA > Full-FT approximation quality). — Impact: Method-design.md §7.

- **Parameter-matched RepT ablation** — Source: Methodologist — RepSim at l* with dimension-matched random features to isolate gradient contribution. Estimated cost: ~2 GPU-days. — Impact: Experiment-design.md (new ablation).

- **TRAK with PCA projection control** — Source: Contrarian — Test TRAK with learned (PCA) projection instead of random projection. If it matches RepSim, FM1 narrative needs revision. Estimated cost: ~3 GPU-days. — Impact: Experiment-design.md (new control).

- **Verify RepT implementation against original results** — Source: Methodologist — Before using reimplemented RepT in CRA, validate it reproduces RepT's published P@10 on their original benchmark. — Impact: Implementation phase.

- **Explicitly state influence linearity assumption** — Source: Theorist — The contrastive scoring mechanism assumes approximate linearity of influence. State this in method-design.md §5 Component B and note it as a testable implication. — Impact: Method-design.md §5.

## 战略预判

1. **实现中最可能出 bug 的组件**: RepT phase-transition layer detection. The algorithm requires computing gradient norm across layers and detecting a "sharp change" — the definition of "sharp" is implementation-dependent and could silently fail (pick wrong layer) without obvious error signals.

2. **训练中最可能不稳定的环节**: Full-FT at Pythia-1B (Experiment 3). Learning rate must be carefully tuned, and the fine-tuned model quality directly affects attribution comparisons. A poorly fine-tuned Full-FT model would confound the LoRA vs Full-FT comparison.

3. **结果不达标的最可能 root cause**: RepSim LDS is low despite good AUPRC/P@K. The "correlation vs causation" gap is real — representation similarity captures topical relevance (which sample is ABOUT similar things) but not causal influence (which sample CHANGED the model's behavior). This would be the single most likely reason the FM1 thesis doesn't translate to LDS-level evidence.

4. **需要准备的 fallback plan**:
   - If RepSim LDS is low: pivot to "correlation vs causation diagnostic" paper + focus on FM2 (contrastive scoring) contribution.
   - If LDS evaluation is too expensive: use AUPRC as primary metric for toxicity filtering; use subset LDS (30 test samples instead of 100).
   - If MAGIC is infeasible: acknowledge limitation and frame FM1 thesis as "relative to approximate IF" only.
   - If budget is tight: cut Experiment 5 (scale-up) first, then reduce seeds from 3 to 2 for Experiment 3.

## 整体判定：**Pass**

The method-design and experiment-design are fundamentally sound: the three-bottleneck diagnostic framework is well-conceived, the 2x2 ablation is an elegant experimental design that yields informative results regardless of outcome, the probe-gated approach correctly manages the highest uncertainty (H4: RepSim LDS competitiveness), and the failure contingencies cover all major scenarios. The statistical plan (permutation tests, bootstrap CIs, FDR correction) exceeds the field standard.

The issues identified are real but correctable without redesigning the core approach:
- Representation extraction protocol is an implementation specification gap, not a design flaw.
- LDS cost uncertainty requires measurement (during probe), not redesign.
- Adding Grad-Sim to the 2x2 strengthens the design without changing it.
- The random-model RepSim control is a low-cost, high-value addition.

None of these issues rise to the level of "rethink the experimental approach" (Revise) or "rethink the problem framing" (Fundamental). The mandatory modifications are additions to the existing design, not corrections. The theoretical concern about MAGIC's tension with FM1 is real but explicitly handled via decision rules.

**The design is ready for blueprint with the following mandatory additions before implementation**:
1. Specify token aggregation strategy in method-design.md
2. Add LDS timing measurement + budget contingency to probe protocol
3. Add Grad-Sim to 2x2 ablation
4. Add random-model RepSim control to probe
