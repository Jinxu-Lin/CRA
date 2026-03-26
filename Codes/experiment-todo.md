# Experiment TODO

> Blueprint v1.0 | 2026-03-26
> Based on: method-design.md v1.0, experiment-design.md v1.0, design-review round-1 synthesis (Pass with 4 mandatory additions)

## Environment Preparation

- [ ] **Clone DATE-LM codebase**: `git clone <DATE-LM-repo> ~/Research/CRA/Codes/core/date-lm/`
- [ ] **Install dependencies**: `pip install -r requirements.txt` (create from DATE-LM + custom deps)
- [ ] **Lock environment**: `pip freeze > requirements-lock.txt`, record CUDA version, PyTorch version, GPU model
- [ ] **Verify GPU access**: SSH to server, `nvidia-smi` confirms 4x A6000 48GB, CUDA toolkit version
- [ ] **Prepare data paths**:
  - DATE-LM datasets: `~/Resources/Datasets/DATE-LM/` (toxicity, data selection, factual attribution)
  - Pre-trained models: `~/Resources/Models/pythia-1b/`, `~/Resources/Models/llama-7b/` (HuggingFace cache or symlink)
  - Base model checkpoints: ensure pre-fine-tuning Pythia-1B checkpoint accessible
- [ ] **Seed infrastructure**: implement `set_all_seeds(seed)` covering torch, numpy, random, CUDA deterministic, dataloader worker seed
- [ ] **Logging setup**: configure wandb project "CRA" (or tensorboard) with standard metrics: loss, gradient norm, LR, GPU utilization

## Phase 0: Sanity Checks (~0.5 day)

- [ ] **shape-check**: Forward pass through Pythia-1B, extract hidden representations at layers L/2 and L. Verify shapes: `(batch, seq_len, d_model)` where d_model=2048 for Pythia-1B. Verify token aggregation produces `(batch, d_model)`.
  - Command: `python scripts/sanity_check.py --check shape --model pythia-1b`
  - Pass: all shapes match expected dimensions, no NaN/Inf

- [ ] **representation-extraction-check**: Extract representations for 10 train + 5 test samples. Compute cosine similarity matrix. Verify: (a) self-similarity = 1.0, (b) cross-similarities in [-1, 1], (c) no constant vectors.
  - Command: `python scripts/sanity_check.py --check repr --model pythia-1b --n_train 10 --n_test 5`
  - Pass: no degenerate representations, similarity matrix has variance > 0

- [ ] **trak-integration-check**: Run DATE-LM's TRAK on 10 train + 5 test samples. Verify: attribution scores produced, LDS evaluation pipeline accepts scores.
  - Command: `python scripts/sanity_check.py --check trak --model pythia-1b --n_train 10 --n_test 5`
  - Pass: TRAK scores and LDS evaluation succeed without error

- [ ] **contrastive-scoring-check**: Compute RepSim scores with fine-tuned model and base model. Verify subtraction produces non-trivial contrastive scores.
  - Command: `python scripts/sanity_check.py --check contrastive --model pythia-1b --n_train 10 --n_test 5`
  - Pass: contrastive scores not all zero, distribution differs from standard scores

## Phase 1: Probe -- Critical Gate (~2 GPU-days)

### Experiment 0: RepSim vs TRAK on DATE-LM Toxicity

**Purpose**: Gate decision for entire project. Tests H4: whether RepSim achieves competitive LDS.

**Mandatory design-review additions integrated**:
- Token aggregation: **last token** representation as default; mean-pooling as secondary comparison
- Random-model RepSim control: RepSim with randomly initialized Pythia-1B
- LDS evaluation timing: measure and record wall-clock time for one full LDS evaluation cycle

- [ ] **probe-repsim**: Extract representations + compute RepSim scores
  - Config: `configs/probe_repsim.yaml`
  - Model: Pythia-1B, LoRA fine-tuned (DATE-LM default)
  - Task: Toxicity filtering (homogeneous)
  - Layers: L/2 (layer 12) and L (layer 24) for Pythia-1B (24 layers)
  - Token aggregation: last token (primary), mean pooling (secondary)
  - Seeds: 1
  - Command: `python experiments/probe/run_repsim.py --config configs/probe_repsim.yaml`
  - Output: `_Results/probe/repsim_scores.json`
  - GPU time: ~0.5 day

- [ ] **probe-trak**: Run TRAK via DATE-LM implementation
  - Config: `configs/probe_trak.yaml`
  - Same model, task, seed as above
  - Command: `python experiments/probe/run_trak.py --config configs/probe_trak.yaml`
  - Output: `_Results/probe/trak_scores.json`
  - GPU time: ~0.5 day

- [ ] **probe-gradsim**: Run Grad-Sim (gradient cosine similarity, no projection)
  - Config: `configs/probe_gradsim.yaml`
  - Same model, task, seed
  - Command: `python experiments/probe/run_gradsim.py --config configs/probe_gradsim.yaml`
  - Output: `_Results/probe/gradsim_scores.json`
  - GPU time: ~0.5 day

- [ ] **probe-random-model-repsim**: RepSim with randomly initialized (untrained) Pythia-1B [DESIGN-REVIEW MANDATORY #4]
  - Config: `configs/probe_random_repsim.yaml`
  - Command: `python experiments/probe/run_repsim.py --config configs/probe_random_repsim.yaml --random_init`
  - Output: `_Results/probe/random_repsim_scores.json`
  - GPU time: < 0.5 day
  - Purpose: distinguish "learned feature quality" from "dimensionality reduction"

- [ ] **probe-evaluate**: Submit all scores to DATE-LM LDS evaluation pipeline
  - Command: `python experiments/probe/evaluate.py --scores_dir _Results/probe/`
  - Metrics: LDS (primary), AUPRC (secondary), P@K (diagnostic)
  - **CRITICAL**: Record wall-clock time for LDS evaluation [DESIGN-REVIEW MANDATORY #2]
  - Output: `_Results/probe/evaluation_summary.md`

- [ ] **probe-gate-decision**: Analyze results and determine go/no-go
  - **Strong pass**: RepSim LDS >= TRAK LDS on toxicity -> proceed with high confidence
  - **Pass**: RepSim LDS >= TRAK LDS - 5pp -> proceed with caution
  - **Weak pass**: RepSim LDS < TRAK - 5pp BUT P@K competitive -> pivot to "correlation vs causation" framing
  - **Fail**: RepSim LDS < TRAK - 5pp AND P@K < TRAK P@K -> try RepT before abandoning
  - **Budget check**: If LDS evaluation takes > X GPU-hours per condition, recalibrate total budget (see contingency plan below)
  - Output: `_Results/probe_result.md` (overwrite existing placeholder)

**LDS Cost Contingency Plan** [DESIGN-REVIEW MANDATORY #2]:
If LDS evaluation requires per-condition model retraining and costs > 0.5 GPU-days per condition:
1. Option A: Use AUPRC as primary metric for toxicity (no retraining needed), LDS on subset (30 test samples)
2. Option B: Reduce seeds from 3 to 2 for Experiments 2-3
3. Option C: Share retraining across methods (same retrained models, different scoring)
4. Recalibrate total budget before committing to Phase 3+

### Experiment 0.5: Mini Pilot (2x2 Sanity Check) (~3 GPU-days)

**Purpose**: Verify 2x2 ablation produces interpretable patterns. Only if Experiment 0 passes.

**Depends on**: Experiment 0 pass

- [ ] **pilot-contrastive-repsim**: RepSim contrastive = RepSim(M_ft) - RepSim(M_base)
  - Config: `configs/pilot_contrastive_repsim.yaml`
  - Task: Toxicity filtering, 1 seed
  - Command: `python experiments/pilot/run_2x2.py --config configs/pilot_contrastive_repsim.yaml`
  - Output: `_Results/pilot/contrastive_repsim_scores.json`

- [ ] **pilot-contrastive-trak**: TRAK contrastive = TRAK(M_ft) - TRAK(M_base)
  - Config: `configs/pilot_contrastive_trak.yaml`
  - Command: `python experiments/pilot/run_2x2.py --config configs/pilot_contrastive_trak.yaml`
  - Output: `_Results/pilot/contrastive_trak_scores.json`

- [ ] **pilot-contrastive-gradsim**: Grad-Sim contrastive = GradSim(M_ft) - GradSim(M_base) [DESIGN-REVIEW MANDATORY #3]
  - Config: `configs/pilot_contrastive_gradsim.yaml`
  - Command: `python experiments/pilot/run_2x2.py --config configs/pilot_contrastive_gradsim.yaml`
  - Output: `_Results/pilot/contrastive_gradsim_scores.json`

- [ ] **pilot-evaluate**: Compute 2x2 table (extend to 2x3 with Grad-Sim)
  - Evaluate all 6 conditions: {RepSim, TRAK, Grad-Sim} x {standard, contrastive}
  - Compute FM1 main effect, FM2 main effect, interaction term
  - Command: `python experiments/pilot/evaluate_2x2.py --scores_dir _Results/pilot/`
  - **Pass**: Both FM1 and FM2 main effects positive (improve LDS)
  - **Adjust**: One main effect zero/negative -> drop that dimension from full experiment
  - **Fail**: Both main effects negative -> diagnostic framework is wrong
  - Output: `_Results/pilot/pilot_summary.md`

## Phase 2: Baseline Reproduction (~2 GPU-days, parallel with Phase 1)

- [ ] **baseline-trak-reproduce**: Verify TRAK reproduction matches DATE-LM reported numbers
  - Config: `configs/baseline_trak.yaml`
  - Task: Toxicity filtering (homogeneous), 1 seed
  - Target: DATE-LM reported TRAK LDS +/- 2pp
  - Command: `python experiments/benchmark/run_baseline.py --method trak --task toxicity`
  - Output: `_Results/baselines/trak_reproduction.md`

- [ ] **baseline-bm25-reproduce**: Verify BM25 reproduction
  - Same protocol as TRAK
  - Command: `python experiments/benchmark/run_baseline.py --method bm25 --task toxicity`

- [ ] **baseline-gradsim-reproduce**: Verify Grad-Sim reproduction
  - Same protocol as TRAK
  - Command: `python experiments/benchmark/run_baseline.py --method gradsim --task toxicity`

## Phase 3: Core Experiments (~37 GPU-days)

**Depends on**: Phase 1 pass + Phase 2 baseline reproduction verified

### Experiment 1: Systematic Benchmark (~15 GPU-days)

**Goal**: First comprehensive evaluation of representation-space TDA methods on DATE-LM.

Methods (7 total, including Grad-Sim per design review):

| # | Method | Space | Scoring | Source |
|---|--------|-------|---------|--------|
| 1 | TRAK | Parameter | Standard | DATE-LM |
| 2 | Grad-Sim | Parameter | Standard | DATE-LM |
| 3 | DDA (TRAK contrastive) | Parameter | Contrastive | Reimplemented |
| 4 | RepSim | Representation | Standard | Custom |
| 5 | RepT | Representation | Standard | Reimplemented |
| 6 | BM25 | Lexical | Standard | DATE-LM |
| 7 | Random | Lower bound | -- | Trivial |

For each of 3 tasks x 7 methods x 3 seeds:

- [ ] **bench-toxicity**: All 7 methods on toxicity filtering, 3 seeds
  - Config: `configs/benchmark_toxicity.yaml`
  - Command: `python experiments/benchmark/run_all.py --task toxicity --seeds 3`
  - Metrics: LDS, AUPRC, P@K
  - GPU time: ~5 GPU-days

- [ ] **bench-selection**: All 7 methods on data selection, 3 seeds
  - Config: `configs/benchmark_selection.yaml`
  - Command: `python experiments/benchmark/run_all.py --task selection --seeds 3`
  - Metrics: LDS, P@K
  - GPU time: ~5 GPU-days

- [ ] **bench-factual**: All 7 methods on factual attribution, 3 seeds
  - Config: `configs/benchmark_factual.yaml`
  - Command: `python experiments/benchmark/run_all.py --task factual --seeds 3`
  - Metrics: LDS, Recall@50, MRR, P@K
  - GPU time: ~5 GPU-days

- [ ] **bench-layer-sweep**: RepSim LDS across all 24 layers of Pythia-1B [DESIGN-REVIEW RECOMMENDED]
  - Config: `configs/benchmark_layer_sweep.yaml`
  - Task: Toxicity filtering, 1 seed
  - Command: `python experiments/benchmark/run_layer_sweep.py --task toxicity`
  - GPU time: ~2 GPU-hours (forward pass at all layers, minimal)
  - Output: `_Results/benchmark/layer_sweep.md`

- [ ] **bench-efficiency**: Profile all methods for GPU-hours per 1K test samples, peak memory, throughput
  - Command: `python experiments/benchmark/profile_efficiency.py`
  - Output: `_Results/benchmark/efficiency_profile.md`

- [ ] **bench-evaluate**: Statistical analysis
  - Per-sample permutation test (10K permutations)
  - Bootstrap 95% CI (1K bootstrap samples over 3 seeds)
  - Cohen's d for pairwise comparisons
  - Benjamini-Hochberg FDR correction (q=0.05)
  - Command: `python experiments/benchmark/statistical_analysis.py`
  - Output: `_Results/benchmark/benchmark_results.md`

### Experiment 2: 2x2(+) Ablation (~10 GPU-days)

**Goal**: Quantify FM1 and FM2 contributions, test independence.

Design: {RepSim, TRAK, Grad-Sim} x {standard, contrastive} = 6 conditions [DESIGN-REVIEW MANDATORY #3: Grad-Sim added]

- [ ] **ablation-run**: 6 conditions x 3 tasks x 3 seeds = 54 runs
  - Config: `configs/ablation_2x2.yaml`
  - Command: `python experiments/ablation_2x2/run_ablation.py --seeds 3`
  - GPU time: ~10 GPU-days (some runs shared with Experiment 1)

- [ ] **ablation-analyze**: Compute FM1/FM2 main effects and interactions
  - FM1 main effect (repr vs param) with BOTH TRAK and Grad-Sim as parameter baselines
  - FM2 main effect (contrastive vs standard)
  - Interaction term
  - CMRR (Common-Mode Rejection Ratio) per task
  - Interaction interpretation: <10% clean, 10-30% partial overlap, >30% tangled
  - Per-sample permutation test + bootstrap CI
  - Command: `python experiments/ablation_2x2/analyze_ablation.py`
  - Output: `_Results/ablation/ablation_results.md`

### Experiment 3: LoRA vs Full-FT (~12 GPU-days)

**Goal**: Test whether FM1 is LoRA-specific or general.

- [ ] **ft-prep**: Fine-tune Pythia-1B with full parameters on DATE-LM
  - Learning rate sweep: {1e-5, 5e-5, 1e-4} on dev set (3 quick runs, ~1 GPU-day)
  - WSD scheduler, 200-step decay, gradient checkpointing
  - Config: `configs/full_ft_sweep.yaml`
  - Command: `python experiments/lora_vs_ft/finetune_full.py --lr_sweep`
  - Output: `_Results/lora_vs_ft/ft_sweep_results.md`

- [ ] **ft-repsim-trak**: {LoRA, Full-FT} x {RepSim, TRAK} x {toxicity, selection} x 3 seeds
  - Config: `configs/lora_vs_ft.yaml`
  - Command: `python experiments/lora_vs_ft/run_comparison.py --seeds 3`
  - Key metric: RepSim advantage = RepSim LDS - TRAK LDS per FT mode
  - GPU time: ~10 GPU-days

- [ ] **ft-analyze**: Compare RepSim advantage under LoRA vs Full-FT
  - If advantage(Full-FT) > advantage(LoRA): FM1 scales with dimensionality (general)
  - If advantage(Full-FT) < advantage(LoRA): FM1 is LoRA-specific
  - Bootstrap CI on advantage difference
  - Command: `python experiments/lora_vs_ft/analyze_ft.py`
  - Output: `_Results/lora_vs_ft/ft_comparison_results.md`

## Phase 4: MAGIC Feasibility (~5 GPU-days, can parallel with Phase 3)

**Goal**: Bound Hessian error contribution via exact IF.

- [ ] **magic-feasibility-check**: Estimate disk/memory requirements before full run
  - Pythia-1B: 200 steps x checkpoint size. Estimate: 200 x ~8GB (optimizer state) = 1.6TB
  - If disk insufficient: implement gradient checkpointing at every N steps
  - Command: `python experiments/magic/feasibility_check.py`
  - Output: `_Results/magic/feasibility_report.md`

- [ ] **magic-run**: MAGIC on 5-10 test samples (if feasible)
  - Task: Toxicity filtering, deterministic training
  - Config: `configs/magic.yaml`
  - Command: `python experiments/magic/run_magic.py --n_test 5`
  - GPU time: ~3-5 GPU-days
  - Output: `_Results/magic/magic_scores.json`

- [ ] **magic-evaluate**: Compare MAGIC LDS vs TRAK LDS vs RepSim LDS (on same test subset)
  - Decision rule:
    - MAGIC LDS >= 0.90: FM1 thesis weakened, pivot to "efficiency" argument
    - MAGIC infeasible: FM1 thesis stands, acknowledged limitation
    - MAGIC LDS 0.70-0.90: FM1 secondary to Hessian error
  - Output: `_Results/magic/magic_results.md`

## Phase 5: Scale-Up (~8 GPU-days, contingent on budget)

**Goal**: Demonstrate findings generalize beyond Pythia-1B.

**Depends on**: Phase 3 results + remaining budget

- [ ] **scaleup-run**: Top 2-3 methods + TRAK on Llama-7B, LoRA only
  - Tasks: Toxicity filtering + data selection
  - 3 seeds per condition
  - Config: `configs/scaleup.yaml`
  - Command: `python experiments/scaleup/run_scaleup.py --model llama-7b --seeds 3`
  - GPU time: ~8 GPU-days

- [ ] **scaleup-analyze**: Compare FM1 main effect at Pythia-1B vs Llama-7B
  - If FM1 increases with model size: dimensionality argument validated
  - Output: `_Results/scaleup/scaleup_results.md`

**Contingency**: If budget tight, reduce to 1 task (toxicity) + 2 methods (RepSim + TRAK) = ~4 GPU-days

## Final Deliverables

- [ ] **compile-results**: Aggregate all results into `_Results/experiment_result.md`
  - Command: `python scripts/compile_results.py`
  - Include: all tables, statistical tests, figures, practitioner guidance table

- [ ] **reproducibility-package**: Verify all results reproducible
  - All configs committed to git
  - All random seeds documented
  - `requirements-lock.txt` up to date
  - Git commit hashes recorded per experiment

## Total Time Budget

| Phase | GPU-days | Cumulative | Priority |
|-------|----------|------------|----------|
| Phase 0: Sanity checks | 0.5 | 0.5 | P0 |
| Phase 1: Probe (Exp 0 + 0.5) | 5 | 5.5 | P0 |
| Phase 2: Baseline reproduction | 2 | 7.5 | P0 |
| Phase 3: Core experiments (Exp 1+2+3) | 37 | 44.5 | P1 |
| Phase 4: MAGIC feasibility (Exp 4) | 5 | 49.5 | P2 |
| Phase 5: Scale-up (Exp 5) | 8 | 57.5 | P3 |
| Buffer | 2.5 | 60 | -- |
| **Total** | **60** | | |

**Available**: 4x A6000 x ~2 months x 75% utilization = ~180 GPU-days. Budget is 33% of capacity.

**Parallelization plan**: Phase 4 (MAGIC) can run on 1 GPU while Phase 3 runs on remaining 3 GPUs. Phase 2 baseline reproduction overlaps with Phase 1 probe on separate GPUs.

**Cut order if over budget**: Exp 5 (scale-up) first -> reduce seeds to 2 for Exp 3 -> drop MAGIC if disk infeasible -> reduce Exp 1 to 2 tasks (toxicity + factual).
