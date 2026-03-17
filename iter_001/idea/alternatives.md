# CRA: Backup Ideas for Potential Pivot (Revised After Pilot)

## Alternative 1: Attribution vs Retrieval Boundary Paper (cand_d)

**Trigger**: Retrieval baselines (Contriever/GTR) match RepSim on 2+ DATE-LM tasks.

**Pivot Thesis**: "Representation-space TDA on semantic tasks is largely equivalent to dense retrieval. The genuine value of model-internal attribution lies in behavioral detection tasks (toxicity), where gradient-based methods capture behavioral properties invisible to semantic similarity. We characterize this attribution-retrieval boundary and provide task-dependent method selection criteria."

**Key Experiments**:
1. Multi-retriever comparison: Contriever, GTR-T5, E5 vs RepSim on all 3 DATE-LM tasks
2. Gradient norm analysis on toxicity: demonstrate that gradient norms directly correlate with toxic content (Cohen's d=2.66 from pilot)
3. Hybrid method: Retrieval for semantic tasks + gradient-based for behavioral tasks
4. Task feature analysis: what makes a task "retrieval-solvable" vs "attribution-requiring"?

**Strengths**: Novel contribution defining a previously unrecognized boundary; practical impact on method selection; builds on pilot's strongest finding (toxicity reversal)
**Weaknesses**: May be perceived as negative/incremental; "retrieval = attribution" is deflating for the TDA community
**Venue fit**: EMNLP 2026 or NeurIPS 2026 poster
**Pilot support**: BM25 R@50=1.0 on counterfact; RepSim cosine similarity is structurally similar to retrieval; toxicity reversal confirms gradient-based methods capture different information

---

## Alternative 2: Representation-Space TDA Benchmark and Survey (cand_b fallback)

**Trigger**: phi^T M psi framework shows no predictive power AND FM2 is undetectable even with continuous metrics AND PCA whitening fails. The paper becomes purely empirical.

**Pivot Thesis**: "We provide the first comprehensive benchmark of representation-space TDA methods on DATE-LM, revealing task-dependent performance profiles. We organize methods under a common bilinear framework and discover that the parameter-vs-representation gap is task-type-dependent, not universal."

**Key Experiments**:
1. Full multi-method tournament: RepSim, RepT, AirRep, In-the-Wild (if feasible), TRAK, DDA, LoGra, BM25, k-NN, Contriever
2. Per-task leaderboard with confidence intervals
3. Layer selection analysis (7 layers x 3 tasks)
4. Computational cost comparison (wall-clock, memory, implementation complexity)
5. Practitioner decision tree: which method for which task type?

**Strengths**: Immediately useful; low risk; comprehensive; fills a real gap (DATE-LM itself noted "no single method dominates")
**Weaknesses**: Perceived as "just a benchmark" without theoretical depth
**Venue fit**: EMNLP 2026 or COLM 2026
**Pilot support**: All infrastructure validated; 2x2 factorial already executed; toxicity reversal adds novelty beyond pure benchmark

---

## Alternative 3: Matched Filter Theory for Data Attribution (cand_c elevated)

**Trigger**: PCA-reduced whitened attribution succeeds dramatically (+5pp on 2+ tasks) AND SNR-accuracy correlation strengthens at full scale (r > 0.4).

**Pivot Thesis**: "Representation-space TDA is mathematically equivalent to signal detection in colored noise. We derive the optimal attribution score from Neyman-Pearson theory, demonstrate practical improvement via PCA-reduced whitening, and provide the first per-query reliability estimate for TDA via output SNR."

**Key Experiments**:
1. PCA-whitened RepSim across k in {16, 32, 64, 128, 256, 512} on all 3 tasks
2. Per-query SNR_out prediction of attribution accuracy (target r > 0.4)
3. CFAR-normalized attribution for cross-query calibration (from Interdisciplinary radar angle)
4. OS-CFAR variants (median/trimmed-mean subtraction) vs global mean subtraction
5. Noise covariance structure analysis: eigenspectrum of residual after mean subtraction

**Strengths**: Strong theoretical grounding; practical improvement; per-query reliability is novel and useful
**Weaknesses**: Narrower scope; depends entirely on whitening succeeding; pilot showed only directional support (r=0.34)
**Venue fit**: NeurIPS 2026 spotlight (if results are strong)
**Pilot support**: SNR-accuracy r=0.34 on counterfact; ftrace showed +6.8pp DDA whitening gain; concept directionally validated

---

## Alternative 4: FM1 as a Window Into LLM Representation Geometry

**Trigger**: Full-scale eigenspectrum reveals surprising structure (BBP transition, sqrt(d) scaling, or multi-modal eigenvalue distribution) that goes beyond the simple "r_eff << d" story.

**Pivot Thesis**: "The gradient eigenspectrum of fine-tuned LLMs reveals a rank-10 signal subspace that captures 85% of task-relevant variation. We characterize this subspace geometry, show it corresponds to a BBP-type phase transition in the signal detection sense, and demonstrate that this low-rank structure is a fundamental property of fine-tuned representations that explains multiple phenomena beyond TDA."

**Key Experiments**:
1. Full-scale eigenspectrum at multiple N values (100, 500, 1000, 2000, 5000)
2. Marchenko-Pastur bulk fitting to identify signal eigenvalues above noise floor
3. Cross-model scaling: Pythia-{70M, 160M, 410M, 1B} eigenspectrum comparison
4. Signal subspace visualization: what do the top-10 gradient eigenvectors encode?
5. Connection to representation geometry: do the top gradient eigenvectors align with representation PCA components?

**Strengths**: Fundamental contribution to understanding LLM representations; generalizes beyond TDA; connects to statistical physics (BBP) and neuroscience (RSA) literatures
**Weaknesses**: Ambitious scope; may require extensive follow-up; eigenspectrum alone may not be sufficient for a full paper
**Venue fit**: ICLR 2027 or NeurIPS 2026 (if results are striking)
**Pilot support**: r_eff=10 with 85.6% top-5 variance is already a striking finding; pilot eigenspectrum data is the strongest part of the CRA evidence base

---

## Decision Timeline

| Decision Point | Experiment | Outcome -> Action |
|---------------|-----------|-------------------|
| After Priority 1 (~1h) | FM2 verification | FM2 detected -> keep in cand_a; undetected -> narrow to FM1 paper |
| After Priority 3 (~5h) | Retrieval baselines | Retrieval = RepSim -> promote cand_d; RepSim >> retrieval -> stay with cand_a |
| After Priority 5 (~7.5h) | PCA whitening | Whitening works -> promote cand_c elements; fails -> cand_a is final |
| After Priority 2 (~4h) | Full eigenspectrum | Surprising structure -> consider cand_e (Alt 4); expected -> cand_a confirmed |
