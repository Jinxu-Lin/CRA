## [Interdisciplinary] 跨学科者视角

### Round 2 Revision Assessment

All three Round 1 issues are addressed. The revisions strengthen the problem statement without introducing new conceptual problems.

**Issue 1 (MAGIC decision rule)**: The three-tier rule maps cleanly onto signal processing decision theory. In radar/sonar, when the "ideal detector" (matched filter with perfect channel knowledge = MAGIC's exact IF) outperforms the "adaptive detector" (representation-space = dimensionality-reduced approximate filter), the conclusion is "adaptation loss dominates." When the ideal detector is infeasible, the practical question becomes "what is the cheapest detector that achieves acceptable performance?" -- exactly the cost-benefit pivot CRA proposes. This is sound.

**Issue 3 (LoRA vs full-FT)**: From a signal processing perspective, LoRA constrains the parameter space to a low-rank subspace. This is equivalent to projecting the signal onto a predetermined subspace before detection. FM1 (signal dilution in high dimensions) is expected to be LESS severe under LoRA (already projected to low-rank) and MORE severe under full-FT (full R^B dimensionality). If the opposite is observed (FM1 severe under LoRA, mild under full-FT), this would suggest FM1 is not a dimensionality problem but a conditioning problem (LoRA's Hessian is degenerate). This distinction is scientifically important and the v1.2 correctly positions it as core.

### 跨域对应物

#### 类比 A — Signal Processing (Matched Filtering + Differential Detection)

**对应关系**: FM1 (signal dilution in R^B) ↔ detection in high-dimensional noise (low SNR per dimension); representation-space TDA ↔ dimensionality-reduced matched filter; contrastive scoring ↔ differential detection (subtract reference channel)

**类比深度**: 深层 — mathematical structure is isomorphic. The cosine similarity in representation space IS the matched filter inner product in reduced dimensions. DDA's debias (subtract base-model influence) IS differential detection. The 2x2 ablation maps exactly onto the signal processing decomposition: {full-band, narrowband filter} x {direct detection, differential detection}.

**已有解法**: 70+ years of detection theory. Key result: dimensionality reduction before detection is optimal when noise is isotropic (Johnson-Lindenstrauss); differential detection is optimal when systematic interference dominates noise. The CRA thesis predicts both conditions hold simultaneously -- this is the "independence" claim (RQ3).

**可借鉴洞察**: In signal processing, the interaction between spatial filtering and temporal filtering is well-characterized. When interference is spatially structured (as pre-training knowledge would be in representation space), spatial filtering (dimensionality reduction) partially addresses temporal interference (common-mode contamination). This predicts a NON-zero interaction in the 2x2 ANOVA -- the 30% threshold in RQ3 may be too generous.

#### 类比 B — Causal Inference (Instrumental Variables)

**对应关系**: LDS measures counterfactual influence (causal); RepSim measures representational similarity (correlational). The gap between P@K and LDS for representation-space methods is exactly the "correlation vs causation" gap in causal inference.

**类比深度**: 表面 — the concepts map but the mathematical machinery (IV, do-calculus) doesn't directly apply to TDA's counterfactual evaluation. However, the conceptual distinction is important: representation-space methods may capture "which training samples are similar" rather than "which training samples caused this output."

**可借鉴洞察**: If RepSim achieves high P@K but low LDS, this is not a failure of representation space but a measurement of a different causal quantity. The paper should frame this as a discovery (representation-space TDA captures associational influence) rather than a negative result.

### 未被利用的工具

- **Common-Mode Rejection Ratio (CMRR)**: From instrumentation engineering. Quantifies how much common-mode signal (pre-training influence) is suppressed by contrastive scoring. Could provide a scalar metric for FM2 severity across tasks. Flagged in Round 1 as P1 (optional improvement), still relevant.

- **Fisher Information as FM1 severity metric**: The Fisher Information matrix F_theta measures how much parameter perturbations affect model output. FM1's severity could be quantified as rank(F_theta) / dim(theta). Low ratio = severe FM1. This was flagged in Round 1 as P2 (deferred) -- still valid but not blocking.

### 跨域盲点与教训

- **Detection theory lesson**: When the "ideal detector" is infeasible, the performance gap between feasible approximations often depends more on the approximation method than on the theoretical framework. CRA's value may ultimately be empirical (which approximation works best on DATE-LM) rather than theoretical (the three-bottleneck framework). This is fine -- the field needs empirical guidance.

### 建议引入路径

No mandatory changes. The signal processing framing is already well-integrated. For design phase: consider using CMRR as a secondary metric alongside LDS to quantify FM2 severity across tasks. The 2x2 interaction prediction (non-zero due to spatial-temporal coupling) should inform the ANOVA interpretation.

### 继续的最强理由
The signal processing analogy provides deep mathematical structure that elevates this from "benchmark paper" to "diagnostic framework paper." The matched filter ↔ representation space and differential detection ↔ contrastive scoring correspondences are genuine isomorphisms, not surface analogies.

### 最危险的失败点
The "correlation vs causation" gap: RepSim may capture representational similarity (associational) but not counterfactual influence (causal). If this gap is large, the three-bottleneck framework explains the wrong quantity.

### 被施压的假设
H3 (FM1 and FM2 independence) -- from signal processing, spatial and temporal filtering interact when interference is spatially structured. Pre-training knowledge IS spatially structured in representation space. The interaction term may exceed the 30% threshold.

### 探针一致性检查
No probe executed. The signal processing theory predicts RepSim should work (dimensionality reduction preserves SNR when noise is isotropic), but this is a theoretical prediction, not empirical verification.

### 推荐判定：**Pass**

The v1.2 revisions are conceptually sound. The MAGIC decision rule aligns with detection theory decision frameworks. The LoRA vs full-FT distinction has clear signal-processing interpretation. The problem formalization is rigorous enough for design phase. Remaining uncertainties (probe results, FM1-FM2 interaction magnitude) are empirical questions for the design and implement phases.
