# CRA Codes -- Implementation Guide

> This file guides the coding agent through the CRA implementation. Read this before writing any code.

## Project Summary

CRA is a **diagnostic framework** (not a novel method) that decomposes LLM TDA failure into three bottlenecks (Hessian error, FM1 signal dilution, FM2 common contamination) and validates the decomposition via controlled experiments on DATE-LM benchmark.

**You are NOT building a new model.** You are implementing scoring methods (RepSim, RepT, contrastive variants) and connecting them to DATE-LM's evaluation pipeline.

## Code Architecture

```
Codes/
├── core/                        # Deep kernel: reusable across experiments
│   ├── attribution/             # Attribution method implementations
│   │   ├── repsim.py            # RepSim: cosine similarity of hidden representations
│   │   ├── rept.py              # RepT: phase-transition layer + gradient augmentation
│   │   ├── contrastive.py       # Contrastive wrapper: score(M_ft) - score(M_base)
│   │   ├── gradsim.py           # Grad-Sim: gradient cosine similarity (if not in DATE-LM)
│   │   └── magic.py             # MAGIC exact IF (best-effort)
│   ├── evaluation/              # Evaluation utilities
│   │   ├── metrics.py           # LDS, AUPRC, P@K, Recall@50, MRR computation
│   │   ├── statistical.py       # Permutation test, bootstrap CI, BH-FDR, Cohen's d
│   │   └── ablation_analysis.py # 2x2 main effects, interaction, CMRR
│   ├── data/                    # Data loading and preprocessing
│   │   ├── date_lm_loader.py    # DATE-LM data pipeline wrapper
│   │   └── representation.py    # Hidden representation extraction
│   └── date-lm/                 # DATE-LM codebase (git submodule or clone)
│
├── experiments/                 # Shallow wrappers: per-experiment scripts
│   ├── probe/                   # Experiment 0: RepSim vs TRAK probe
│   │   ├── run_repsim.py
│   │   ├── run_trak.py
│   │   ├── run_gradsim.py
│   │   └── evaluate.py
│   ├── pilot/                   # Experiment 0.5: 2x2 mini pilot
│   │   ├── run_2x2.py
│   │   └── evaluate_2x2.py
│   ├── benchmark/               # Experiment 1: systematic benchmark
│   │   ├── run_all.py
│   │   ├── run_baseline.py
│   │   ├── run_layer_sweep.py
│   │   ├── profile_efficiency.py
│   │   └── statistical_analysis.py
│   ├── ablation_2x2/           # Experiment 2: FM1/FM2 ablation
│   │   ├── run_ablation.py
│   │   └── analyze_ablation.py
│   ├── lora_vs_ft/             # Experiment 3: LoRA vs Full-FT
│   │   ├── finetune_full.py
│   │   ├── run_comparison.py
│   │   └── analyze_ft.py
│   ├── magic/                   # Experiment 4: MAGIC feasibility
│   │   ├── feasibility_check.py
│   │   └── run_magic.py
│   └── scaleup/                 # Experiment 5: Llama-7B scale-up
│       ├── run_scaleup.py
│       └── analyze_scaleup.py
│
├── configs/                     # All experiment configs (YAML)
│   ├── base.yaml                # Shared defaults (model, seed, paths)
│   ├── probe_repsim.yaml
│   ├── probe_trak.yaml
│   ├── probe_gradsim.yaml
│   ├── probe_random_repsim.yaml
│   ├── pilot_contrastive_repsim.yaml
│   ├── pilot_contrastive_trak.yaml
│   ├── pilot_contrastive_gradsim.yaml
│   ├── benchmark_toxicity.yaml
│   ├── benchmark_selection.yaml
│   ├── benchmark_factual.yaml
│   ├── benchmark_layer_sweep.yaml
│   ├── ablation_2x2.yaml
│   ├── full_ft_sweep.yaml
│   ├── lora_vs_ft.yaml
│   ├── magic.yaml
│   └── scaleup.yaml
│
├── scripts/                     # Utility scripts
│   ├── sanity_check.py          # Pre-experiment sanity checks
│   ├── compile_results.py       # Aggregate results into experiment_result.md
│   └── run_experiment.sh        # Wrapper: set seed, log GPU info, run command
│
├── _Data/                       # Generated data (gitignored)
│   ├── representations/         # Cached hidden representations
│   ├── gradients/               # Cached gradient features
│   ├── scores/                  # Raw attribution scores
│   └── checkpoints/             # MAGIC training checkpoints
│
└── _Results/                    # Experiment results (git tracked)
    ├── probe_result.md          # Probe gate decision
    ├── probe/                   # Probe raw results
    ├── pilot/                   # Pilot raw results
    ├── benchmark/               # Benchmark results + statistics
    ├── ablation/                # 2x2 ablation results
    ├── lora_vs_ft/              # LoRA vs Full-FT results
    ├── magic/                   # MAGIC feasibility results
    ├── scaleup/                 # Scale-up results
    └── experiment_result.md     # Final compiled results
```

## Component-to-File Mapping

| method-design.md Component | Code File | Category | Source |
|----------------------------|-----------|----------|--------|
| Component A: RepSim | `core/attribution/repsim.py` | Deep kernel | Custom implementation |
| Component A: RepT | `core/attribution/rept.py` | Deep kernel | Reimplemented from paper |
| Component B: Contrastive Scoring | `core/attribution/contrastive.py` | Deep kernel | Custom wrapper |
| Component C: MAGIC | `core/attribution/magic.py` | Deep kernel | Reimplemented (best-effort) |
| TRAK baseline | `core/date-lm/` (DATE-LM codebase) | External | DATE-LM provided |
| Grad-Sim baseline | `core/attribution/gradsim.py` or DATE-LM | Deep kernel / External | DATE-LM or custom |
| BM25 baseline | `core/date-lm/` | External | DATE-LM provided |
| DATE-LM evaluation pipeline | `core/date-lm/` | External | DATE-LM provided |
| Representation extraction | `core/data/representation.py` | Deep kernel | Custom |
| Statistical analysis | `core/evaluation/statistical.py` | Deep kernel | Custom |
| 2x2 ablation analysis | `core/evaluation/ablation_analysis.py` | Deep kernel | Custom |

## Probe Code Reuse

**Existing probe code**: None (`Codes/probe/` is empty). All code written from scratch.

**DATE-LM codebase reuse**:
- TRAK implementation: use directly from DATE-LM
- Grad-Sim: check if DATE-LM provides it; if not, implement cosine similarity of per-sample gradients
- BM25: use directly from DATE-LM
- Evaluation pipeline (LDS, AUPRC): use directly from DATE-LM
- Data loading: wrap DATE-LM's data pipeline

**VITA project** (`~/Research/VITA/Codes/`): EK-FAC/cosine scoring utilities may provide gradient extraction patterns. Evaluate compatibility but do not depend on it.

## Config Structure

All experiments are config-driven. Ablation = change one config field.

```yaml
# base.yaml (shared defaults)
model:
  name: "pythia-1b"
  path: "~/Resources/Models/pythia-1b/"
  n_layers: 24
  d_model: 2048

fine_tuning:
  mode: "lora"           # "lora" or "full"
  lora_rank: 16
  learning_rate: 5e-5
  scheduler: "wsd"
  decay_steps: 200

attribution:
  method: "repsim"       # repsim, rept, trak, gradsim, dda, bm25, magic, random
  scoring: "standard"    # "standard" or "contrastive"
  layer: "middle"        # "middle" (L/2), "last" (L), "all" (sweep), or int
  token_aggregation: "last_token"  # "last_token" or "mean_pool"

evaluation:
  metrics: ["lds", "auprc", "pk"]
  n_seeds: 3
  task: "toxicity"       # toxicity, selection, factual

paths:
  data: "~/Resources/Datasets/DATE-LM/"
  base_model: "~/Resources/Models/pythia-1b/"  # pre-fine-tuning checkpoint
  output: "_Results/"
  cache: "_Data/"

reproducibility:
  seed: 42
  deterministic: true
  log_wandb: true
  wandb_project: "CRA"
```

**Config override for experiments**: Each experiment config imports `base.yaml` and overrides specific fields. Example:
```yaml
# probe_repsim.yaml
_base_: "base.yaml"
attribution:
  method: "repsim"
  layer: "middle"
evaluation:
  n_seeds: 1
  task: "toxicity"
```

**Ablation via config**: To switch from standard to contrastive scoring, change only `attribution.scoring: "contrastive"`. To switch from LoRA to full-FT, change `fine_tuning.mode: "full"`.

## Implementation Priorities

### Critical Path (implement first)
1. `core/data/representation.py` -- hidden representation extraction with token aggregation
2. `core/attribution/repsim.py` -- RepSim scoring (cosine similarity)
3. Integration with DATE-LM evaluation pipeline
4. `experiments/probe/` -- probe experiment scripts

### Second Priority
5. `core/attribution/contrastive.py` -- contrastive wrapper
6. `core/attribution/rept.py` -- RepT with phase-transition detection
7. `core/attribution/gradsim.py` -- gradient cosine similarity
8. `experiments/pilot/` + `experiments/benchmark/`

### Third Priority
9. `core/evaluation/statistical.py` -- statistical analysis suite
10. `core/evaluation/ablation_analysis.py` -- 2x2 analysis
11. `experiments/ablation_2x2/` + `experiments/lora_vs_ft/`

### Best-Effort
12. `core/attribution/magic.py` -- MAGIC (may be infeasible)
13. `experiments/scaleup/`

## Key Implementation Details

### Representation Extraction (`core/data/representation.py`)

```python
# Token aggregation strategy [DESIGN-REVIEW MANDATORY #1]
# Default: last token representation (for autoregressive LLMs)
# Secondary: mean pooling (compare in probe)

def extract_representations(model, dataloader, layer, aggregation="last_token"):
    """
    Extract hidden representations at specified layer.

    Args:
        model: HuggingFace model with output_hidden_states=True
        dataloader: DATE-LM data pipeline
        layer: int (0-indexed) or "middle" or "last"
        aggregation: "last_token" or "mean_pool"

    Returns:
        representations: (N, d_model) tensor
    """
    # For "last_token": h[batch, -1, :] (last position in sequence)
    # For "mean_pool": h[batch, :seq_len, :].mean(dim=1) (exclude padding)
```

### RepSim Scoring (`core/attribution/repsim.py`)

```python
def repsim_score(h_test, h_train):
    """
    I_RepSim(z_test, z_train) = cos(h(z_test), h(z_train))

    Args:
        h_test: (n_test, d_model)
        h_train: (n_train, d_model)

    Returns:
        scores: (n_test, n_train) cosine similarity matrix
    """
    # Use F.normalize + matrix multiply for efficiency
    # Cosine > inner product (DATE-LM finding)
```

### Contrastive Scoring (`core/attribution/contrastive.py`)

```python
def contrastive_score(score_ft, score_base):
    """
    I_contrastive = I(M_ft) - I(M_base)
    Generic wrapper: works with any scoring method.
    """
    return score_ft - score_base
```

### RepT Phase-Transition Detection (`core/attribution/rept.py`)

**WARNING**: Most likely component to have bugs (design review prediction).
- Compute gradient norm ||nabla_h L||_2 at each layer
- Detect "sharp change" = layer where gradient norm ratio (l)/(l-1) is maximized
- Use that layer l* for feature extraction
- Concatenate: phi(z) = [h^(l*)(z), nabla_h L(z)]
- **Validation**: compare detected l* with RepT paper's reported values on similar models

## Reproducibility Checklist

- [ ] `set_all_seeds(seed)` called before every experiment run
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `np.random.seed(seed)`
  - `random.seed(seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
- [ ] `requirements-lock.txt` generated with `pip freeze`
- [ ] CUDA version + PyTorch version + GPU model recorded in `_Results/environment.md`
- [ ] Every experiment run records: config hash, git commit hash, wall-clock time, peak GPU memory
- [ ] Wandb (or tensorboard) logging enabled for all training runs
- [ ] Checkpoints: save fine-tuned model after DATE-LM training (reuse across methods)

## Experiment Naming Convention

- Configs: `{experiment}_{method}.yaml` (e.g., `probe_repsim.yaml`)
- Score files: `_Data/scores/{experiment}/{method}_{task}_seed{N}.pt`
- Results: `_Results/{experiment}/{analysis_name}.md`
- Wandb runs: `CRA/{experiment}/{method}_{task}_seed{N}`

## Data Paths

| Data | Path | Notes |
|------|------|-------|
| DATE-LM datasets | `~/Resources/Datasets/DATE-LM/` | Download via DATE-LM scripts |
| Pythia-1B | `~/Resources/Models/pythia-1b/` | HuggingFace: `EleutherAI/pythia-1b` |
| Llama-7B | `~/Resources/Models/llama-7b/` | HuggingFace (gated, need access) |
| Cached representations | `_Data/representations/` | Gitignored |
| Cached gradients | `_Data/gradients/` | Gitignored |
| Attribution scores | `_Data/scores/` | Gitignored |
| MAGIC checkpoints | `_Data/checkpoints/` | Gitignored, potentially 1.6TB |

## Debug Guide

### NaN/Inf in Scores
1. Check representation extraction: any all-zero or constant vectors? (degenerate layers)
2. Check cosine similarity computation: division by zero when ||h|| = 0
3. Fix: add epsilon to normalization: `h / (||h|| + 1e-8)`

### OOM on A6000 (48GB)
1. Pythia-1B (~2GB fp16) should fit easily. Check batch size for representation extraction.
2. Full-FT gradient storage: ~8GB per sample. Use gradient accumulation.
3. MAGIC: checkpoint storage may fill disk. Monitor `df -h`.
4. Reduce batch size or use gradient checkpointing: `model.gradient_checkpointing_enable()`

### TRAK Integration Issues
1. Verify DATE-LM TRAK produces scores in expected format (list of scores per test-train pair)
2. Check TRAK projection dimension matches DATE-LM default
3. Ensure TRAK and RepSim use the same model checkpoint for fair comparison

### RepT Phase-Transition Detection Fails
1. Plot gradient norms across all layers -- should see clear transition
2. If no clear transition: fall back to layer L/2 (same as RepSim middle)
3. Verify backward pass through hidden states (not just parameters)

### Low LDS for All Methods
1. Check DATE-LM evaluation pipeline: are we using the correct test set and evaluation protocol?
2. Verify fine-tuned model quality: does it actually learn the task? Check task-specific metrics.
3. Check score format: DATE-LM may expect specific normalization or format.

## Version Sync

After every code modification:
```bash
cd ~/Research/CRA
git add Codes/
git commit -m "implement: <what changed>"
git push origin main
```
