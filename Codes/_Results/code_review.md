# Code Review Report

> Reviewer: DL Research Engineer + Code Review Expert
> Date: 2026-03-26
> Baseline: `Codes/CLAUDE.md` component mapping + `research/method-design.md` + `research/experiment-design.md`

---

## Overall Assessment

| Dimension | Verdict | Summary |
|-----------|---------|---------|
| 1. Architecture Faithfulness | **Pass** | File tree matches CLAUDE.md mapping exactly; clean core/experiments separation |
| 2. Component Faithfulness | **Pass (with Concerns)** | RepSim, Contrastive, GradSim, Metrics, Statistical, Ablation are faithful. RepT has a gradient-blocking bug. MAGIC is skeleton-only (expected). |
| 3. Ablation Engineering | **Pass** | Config-driven ablation via `_base_` inheritance and CLI overrides; no code changes needed to switch conditions |
| 4. DL Common Bugs | **Concern** | RepT `@torch.no_grad()` blocks gradient computation; AUPRC uses non-standard trapezoidal approximation; permutation test performance |
| 5. Reproducibility | **Pass (with Concerns)** | Seed chain complete; `worker_init_fn` available but not wired into `create_dataloader`; no `requirements-lock.txt` yet |
| 6. Compute Efficiency | **Pass** | Batched scoring implemented; representation extraction uses `@torch.no_grad()` correctly; RepT per-sample loop is slow but necessary |

---

## Dimension 1: Architecture Faithfulness

### 1.1 File Existence Check

Every file listed in CLAUDE.md Component-to-File Mapping exists:

| CLAUDE.md Entry | File | Status |
|-----------------|------|--------|
| `core/attribution/repsim.py` | Exists | OK |
| `core/attribution/rept.py` | Exists | OK |
| `core/attribution/contrastive.py` | Exists | OK |
| `core/attribution/gradsim.py` | Exists | OK |
| `core/attribution/magic.py` | Exists | OK |
| `core/evaluation/metrics.py` | Exists | OK |
| `core/evaluation/statistical.py` | Exists | OK |
| `core/evaluation/ablation_analysis.py` | Exists | OK |
| `core/data/date_lm_loader.py` | Exists | OK |
| `core/data/representation.py` | Exists | OK |
| `core/date-lm/` (DATE-LM submodule) | **Not yet cloned** | Expected (documented prerequisite) |

All experiment directories (`probe/`, `pilot/`, `benchmark/`, `ablation_2x2/`, `lora_vs_ft/`, `magic/`, `scaleup/`) exist with the scripts listed in CLAUDE.md.

### 1.2 Ghost Files

Files present but NOT in CLAUDE.md mapping:
- `config_utils.py` -- Config loading utility (reasonable infrastructure)
- `seed_utils.py` -- Seed management (reasonable infrastructure)
- `logging_utils.py` -- Experiment logging (reasonable infrastructure)
- `run_attribution.py` -- Main entry point (reasonable, mentioned in CLAUDE.md Usage)
- `evaluate.py` -- Evaluation entry point (reasonable)
- `tests/` -- Unit tests (positive, not ghost files)
- `scripts/compile_results.py` -- Aggregation utility (listed in CLAUDE.md tree)

**Verdict: No problematic ghost files.** All extra files are legitimate infrastructure.

### 1.3 Core vs Experiments Separation

- `core/` contains reusable scoring, evaluation, and data modules -- **correct**.
- `experiments/` scripts are thin wrappers that delegate to `run_attribution.py` or call `core/` functions directly -- **correct**.
- No experiment-specific logic leaks into `core/` -- **correct**.

---

## Dimension 2: Component Faithfulness

### 2.1 RepSim (`core/attribution/repsim.py`)

**method-design.md spec**: `I_RepSim(z_test, z_train) = cos(h^(l)(z_test), h^(l)(z_train))`, using `F.normalize` + matrix multiply.

**Implementation check**:
- Line 46-47: `F.normalize(h_test, p=2, dim=1, eps=eps)` -- correct L2 normalization along feature dim.
- Line 50: `h_test_norm @ h_train_norm.T` -- correct cosine similarity via normalized dot product.
- eps default 1e-8 -- matches CLAUDE.md debug guide recommendation.
- Batched variant (`repsim_score_batched`) correctly normalizes once then chunks -- **no redundant normalization**.
- Input validation (ndim, shape) -- present.

**Verdict: PASS.** Faithful to spec.

### 2.2 RepT (`core/attribution/rept.py`)

**method-design.md spec**:
- Detect phase-transition layer l* where gradient norm ratio is maximized
- phi(z) = concat[h^(l*)(z), nabla_h L(z)]
- I_RepT = cos(phi(z_test), phi(z_train))

**Implementation check**:

**CONCERN -- `@torch.no_grad()` on `compute_layer_gradient_norms` (line 52)**:
The function is decorated with `@torch.no_grad()` but internally uses `torch.enable_grad()` context manager (line 103). This works because `torch.enable_grad()` overrides the outer `@torch.no_grad()`. However, this is a fragile pattern -- if anyone removes the inner `with torch.enable_grad():`, gradient computation silently fails (all norms become 0). The code currently works but the design is misleading.

**BUG -- `@torch.no_grad()` on `extract_hidden_gradients` (line 191)**:
This function is decorated with `@torch.no_grad()` and uses `torch.enable_grad()` internally (line 237). Same pattern as above -- currently works but fragile. More importantly, at line 250-251:
```python
if h_l.requires_grad:
    grad = torch.autograd.grad(loss, h_l, create_graph=False)[0]
else:
    grad = torch.zeros_like(h_l)
```
The `h_l.requires_grad` check depends on the model's forward pass creating the hidden state with `requires_grad=True`. Under `torch.enable_grad()`, this should work for most HuggingFace models that use standard nn.Module operations. However, **if the model uses `@torch.no_grad()` internally or detaches hidden states**, `h_l.requires_grad` will be False and the function silently returns zero gradients. This is not explicitly validated.

- `detect_phase_transition_layer` (line 23-48): Correctly computes ratio of consecutive gradient norms, returns argmax+1. Edge case: if all gradient norms are near-zero (degenerate model), eps=1e-10 prevents division-by-zero but the detected layer is meaningless. No warning or fallback to L/2 as CLAUDE.md debug guide suggests.

- `extract_rept_features` (line 137-157): Simple concat, correct.
- `rept_score` (line 160-188): Same as RepSim scoring but on 2*d_model features, correct.

**Verdict: CONCERN.** The `@torch.no_grad()` + `torch.enable_grad()` pattern works but is fragile and potentially confusing. Missing fallback when phase-transition detection fails (all norms ~0).

**Files/Lines**:
- `/home/jinxulin/CRA/Codes/core/attribution/rept.py`, line 52 (`@torch.no_grad()` on `compute_layer_gradient_norms`)
- `/home/jinxulin/CRA/Codes/core/attribution/rept.py`, line 191 (`@torch.no_grad()` on `extract_hidden_gradients`)
- `/home/jinxulin/CRA/Codes/core/attribution/rept.py`, line 250 (silent zero-grad fallback)

### 2.3 Contrastive Scoring (`core/attribution/contrastive.py`)

**method-design.md spec**: `I_contrastive = I(M_ft) - I(M_base)`, generic wrapper.

- `contrastive_score` (line 18-39): Simple subtraction with shape assertion -- correct.
- `contrastive_score_from_representations` (line 42-68): Computes scoring on both models then subtracts -- correct.
- `compute_cmrr` (line 71-97): `|standard - contrastive| / (|standard| + eps)` -- matches experiment-design.md definition.

**Verdict: PASS.**

### 2.4 GradSim (`core/attribution/gradsim.py`)

**method-design.md spec**: `I_GradSim = cos(g(z_test), g(z_train))` where g = per-sample gradient.

- Scoring functions (line 21-89): Same pattern as RepSim (normalize + matmul) but on gradients -- correct.
- `extract_per_sample_gradients` (line 92-152): Per-sample forward + backward, flattens and stacks gradients. Note at line 149: `torch.cat([p.grad.flatten() for p in params if p.grad is not None])` -- this correctly skips parameters without gradients (e.g., frozen layers).
- **Minor concern**: `model.eval()` at line 115 but then `loss.backward()` is called. This is correct for extracting gradients in eval mode (dropout disabled), but `model.zero_grad()` at line 136 is called inside the per-sample loop -- correct practice.

**Verdict: PASS.**

### 2.5 MAGIC (`core/attribution/magic.py`)

**method-design.md spec**: Best-effort, may be infeasible. Skeleton expected.

- `magic_score_single_test`: Raises `NotImplementedError` with detailed explanation -- correct for best-effort.
- `magic_feasibility_check`: Reasonable estimates (model_size * 3 for checkpoint with optimizer state, linear time scaling).

**Verdict: PASS** (skeleton matches expectations).

### 2.6 Representation Extraction (`core/data/representation.py`)

**method-design.md spec**: Extract h^(l)(z) at specified layer, token aggregation via last_token or mean_pool.

- `resolve_layer_index` (line 18-38): Handles int, "middle" (L//2), "last" (L-1) -- correct.
- `aggregate_tokens` (line 41-85):
  - `last_token` with mask: Finds last non-padding position via `attention_mask.sum(dim=1) - 1` -- correct.
  - `mean_pool` with mask: Masks padding, averages remaining -- correct. Uses `.clamp(min=1.0)` to avoid division by zero.
- `extract_representations` (line 88-146):
  - Line 141: `hidden_states[layer_idx + 1]` -- correct (+1 because hidden_states[0] is embedding output).
  - `@torch.no_grad()` -- correct for inference-only extraction.
- `extract_all_layer_representations` (line 149-201): Extracts all layers in a single forward pass -- efficient.

**Verdict: PASS.**

### 2.7 DATE-LM Loader (`core/data/date_lm_loader.py`)

- Handles both `.pt` and directory-based data formats -- flexible.
- `shuffle=False` in DataLoader -- correct (attribution requires fixed ordering).
- `pin_memory=True` -- good for GPU transfer performance.
- **Missing**: `worker_init_fn=seed_worker` and `generator=get_generator(seed)` from `seed_utils.py` are NOT passed to the DataLoader at line 117-123. This means multi-worker data loading is not fully reproducible.

**Files/Lines**: `/home/jinxulin/CRA/Codes/core/data/date_lm_loader.py`, line 117-123.

### 2.8 Metrics (`core/evaluation/metrics.py`)

- `spearman_correlation`: Custom implementation using rank transform + Pearson on ranks. Handles ties via averaging -- correct.
- `lds`: Wrapper around Spearman -- correct per experiment-design.md.
- `auprc` (line 106-149):
  - **CONCERN**: Uses left-rectangle trapezoidal rule (line 147: `precision_with_one[:-1] * recall_diff`), not full trapezoidal (`(p[i] + p[i+1])/2 * dr`). For standard AUPRC computation, this is actually the **correct** convention (step function interpolation, not trapezoidal), matching sklearn's implementation. However, prepending `(recall=0, precision=1)` at line 142-143 assumes the precision at recall=0 is 1.0, which is the standard convention.
  - **Verdict**: Acceptable, matches standard practice.
- `precision_at_k`, `recall_at_k`, `mrr`: Standard implementations, correct.

**Verdict: PASS.**

### 2.9 Statistical Analysis (`core/evaluation/statistical.py`)

- `permutation_test` (line 18-60):
  - Uses sign-flip permutation on paired differences -- correct for paired test.
  - Continuity correction `(count + 1) / (n_perm + 1)` -- correct.
  - **Performance concern**: Python loop over 10K permutations is slow. For 100 samples x 10K permutations, this creates 10K random tensors. Could be vectorized.
- `bootstrap_ci`: Standard percentile bootstrap -- correct.
- `cohens_d`: Uses paired formula `mean(A-B)/std(A-B)` -- correct for paired design.
- `benjamini_hochberg`: Standard BH step-up procedure -- correct.

**Verdict: PASS** (with minor performance note on permutation test).

### 2.10 Ablation Analysis (`core/evaluation/ablation_analysis.py`)

- `compute_main_effects` (line 25-79): FM1 = mean(repr) - mean(param), FM2 = mean(contrastive) - mean(standard), interaction = standard 2x2 interaction formula -- all match experiment-design.md Section 3.2 exactly.
- `assess_independence`: Thresholds (0.10, 0.30) match formalize review thresholds in experiment-design.md Section 3.2.
- `full_ablation_analysis`: Correctly averages across scoring modes for FM1 test and across space modes for FM2 test.
- **Duplicate function**: `compute_cmrr` exists in BOTH `core/attribution/contrastive.py` (line 71) AND `core/evaluation/ablation_analysis.py` (line 100). They compute the same thing with slightly different signatures. The one in `ablation_analysis.py` returns a Python float, the one in `contrastive.py` returns a tensor. This duplication could lead to divergence.

**Files/Lines**:
- `/home/jinxulin/CRA/Codes/core/attribution/contrastive.py`, line 71 (CMRR version 1)
- `/home/jinxulin/CRA/Codes/core/evaluation/ablation_analysis.py`, line 100 (CMRR version 2)

**Verdict: PASS** (with duplication concern).

---

## Dimension 3: Ablation Engineering

### 3.1 Config Switch Mechanism

- `base.yaml` defines all defaults. Experiment configs use `_base_: "base.yaml"` and override specific fields.
- `config_utils.py` `load_config` recursively resolves `_base_` chains with deep merge -- correct.
- CLI overrides via `--override attribution.method=repsim attribution.scoring=contrastive` -- works for switching methods and conditions.

### 3.2 Ablation Conditions

All experiment-design.md conditions are achievable via config changes alone:

| Condition | Config Change | Code Change Needed? |
|-----------|--------------|-------------------|
| RepSim standard | `method=repsim, scoring=standard` | No |
| RepSim contrastive | `method=repsim, scoring=contrastive` | No |
| RepT standard | `method=rept` | No |
| GradSim standard | `method=gradsim` | No |
| TRAK/DDA/BM25 | `method=trak/dda/bm25` | **Yes** -- returns placeholder scores (DATE-LM not integrated) |
| LoRA vs Full-FT | `fine_tuning.mode=lora/full` | **Partially** -- `run_attribution.py` does not use `fine_tuning.mode` to select which model checkpoint to load. The model path in config must be manually changed to point to the correct fine-tuned checkpoint. |
| Layer sweep | `layer=int` | No |
| Random baseline | `method=random` | No |

### 3.3 Missing Ablation Configs

CLAUDE.md lists `probe_random_repsim.yaml` -- verified exists. All 18 config files listed exist.

**Verdict: PASS.** Config-driven ablation is well-designed. The LoRA vs Full-FT dimension requires the user to provide different model checkpoints via config, which is documented.

---

## Dimension 4: DL Common Bugs

### 4.1 Data Leakage

- No test-train data mixing detected. `shuffle=False` in DataLoader preserves ordering.
- `extract_representations` processes train and test separately.
- No label information leaks into feature extraction.

**Verdict: PASS.**

### 4.2 Shape/Broadcasting

- All scoring functions have explicit shape assertions (ndim, dimension match).
- `aggregate_tokens` handles both masked and unmasked cases correctly.
- No implicit broadcasting in critical paths.

**Verdict: PASS.**

### 4.3 Loss Reduction

- `run_attribution.py` line 138 and `rept.py` line 112: `CrossEntropyLoss(reduction="mean")` -- then line 113 calls `.mean()` again. This is a **no-op** (mean of a scalar is the scalar), so it's not a bug, but it's redundant.
- The loss functions in `rept.py` and `gradsim.py` are passed in by the caller. The per-sample gradient extraction in `gradsim.py` (line 144-146) uses `loss.mean()` after computing with the same loss_fn. As long as the caller provides `reduction="mean"`, this is correct.

**Verdict: PASS** (redundant `.mean()` is harmless).

### 4.4 Random Seed

- `set_seed()` in `seed_utils.py` covers all 6 required items: `random`, `numpy`, `torch`, `cuda`, `cudnn.deterministic`, `cudnn.benchmark=False`.
- All experiment entry points call `set_seed()` before computation.
- **CONCERN**: `create_dataloader` does not use `seed_worker` or `get_generator`. With `num_workers > 0` (base.yaml sets `num_workers: 4`), data loading order may not be reproducible across runs. The `seed_worker` and `get_generator` functions exist in `seed_utils.py` but are never wired in.

**Files/Lines**: `/home/jinxulin/CRA/Codes/core/data/date_lm_loader.py`, line 94-123 (missing `worker_init_fn` and `generator`).

### 4.5 Train/Eval Mode

- `model.eval()` is called in all extraction functions (`extract_representations`, `extract_per_sample_gradients`, `compute_layer_gradient_norms`, `extract_hidden_gradients`).
- `load_model` in `run_attribution.py` also calls `model.eval()`.
- No `model.train()` calls anywhere in the codebase (CRA does not train, only extracts features from pre-trained models).

**Verdict: PASS.**

### 4.6 Gradient-Related Bugs

- RepSim: No gradients used, `@torch.no_grad()` context -- correct.
- RepT: See Dimension 2.2 discussion. The `@torch.no_grad()` + `torch.enable_grad()` pattern works but is fragile.
- GradSim `extract_per_sample_gradients`: `model.zero_grad()` before each sample, `loss.backward()` accumulates into `p.grad`, then flatten and detach -- correct. No detach/`.data` issues.

**Verdict: CONCERN** (RepT fragility as noted).

---

## Dimension 5: Reproducibility

### 5.1 Seed Chain

| Requirement | Status |
|-------------|--------|
| `set_all_seeds()` before every run | **PASS** -- All entry points call `set_seed()` |
| `torch.backends.cudnn.deterministic = True` | **PASS** -- In `set_seed()` |
| `torch.backends.cudnn.benchmark = False` | **PASS** -- In `set_seed()` |
| DataLoader worker seeds | **CONCERN** -- `seed_worker` exists but not used |
| `requirements-lock.txt` | **MISSING** -- Not yet generated |
| Environment recording | **PASS** -- `logging_utils.py` records GPU, CUDA, PyTorch versions |
| Config hash + git commit | **PASS** -- `ExperimentLogger` captures both |
| Wandb logging | **PASS** -- Configurable via `reproducibility.log_wandb` |

### 5.2 Checkpoint Management

- Score files saved with naming convention: `{method}_{scoring}_seed{seed}.pt` -- consistent.
- No model checkpoint saving (CRA does not fine-tune, uses pre-existing checkpoints) -- correct.

### 5.3 Config Completeness

- All configs inherit from `base.yaml` which defines all fields.
- No config field referenced in code that isn't defined in `base.yaml`.
- `pilot.yaml` exists but was not listed in CLAUDE.md -- minor documentation gap.

**Verdict: PASS (with Concerns)** -- Missing `worker_init_fn` in DataLoader and no `requirements-lock.txt`.

---

## Dimension 6: Compute Efficiency

### 6.1 GPU Memory

- RepSim: Only stores (N, d_model) representations on CPU, scoring on CPU/GPU as float32. For N=10K, d=2048: ~80MB. Negligible.
- Batched scoring (`repsim_score_batched`, `gradsim_score_batched`): Chunks over training samples to control memory. Default batch_size=1024 for RepSim, 256 for GradSim -- reasonable.
- GradSim: Per-sample gradients stored as (N, B) tensor. For Pythia-1B with B~10^9, this is ~40GB per float32. The code creates this tensor on CPU (line 150-152), which is fine for disk storage but may cause issues during scoring. The `max_params` option mitigates this for testing.

### 6.2 Data Loading

- `pin_memory=True` in DataLoader -- good.
- `num_workers=4` in base config but `create_dataloader` hardcodes `num_workers=0` (line 121). This means the config value is IGNORED and data loading is single-threaded.

**Files/Lines**: `/home/jinxulin/CRA/Codes/core/data/date_lm_loader.py`, line 121 (`num_workers=0` ignores config).

### 6.3 Redundant Computation

- RepT `extract_hidden_gradients` processes one sample at a time (line 236 loop), which means N separate forward passes through the model. This is necessary for per-sample gradient computation and cannot be easily batched.
- `compute_layer_gradient_norms` also processes one sample at a time -- necessary for per-sample per-layer gradients.
- Representation extraction (`extract_representations`) processes full batches in a single forward pass -- efficient.

### 6.4 `extract_all_layer_representations`

This function extracts representations at ALL layers in a single forward pass, storing N x d_model per layer. For Pythia-1B (24 layers, d=2048, N=10K): 24 * 10K * 2048 * 4 bytes = ~2GB. Manageable.

**Verdict: PASS** (with `num_workers` config bypass noted).

---

## Must-Fix Items

| # | Severity | File | Line | Issue | Fix |
|---|----------|------|------|-------|-----|
| 1 | **Medium** | `core/data/date_lm_loader.py` | 121 | `num_workers=0` hardcoded, ignoring config `data.num_workers: 4` | Change to `num_workers=num_workers` (parameter already accepted) |
| 2 | **Medium** | `core/data/date_lm_loader.py` | 117-123 | Missing `worker_init_fn` and `generator` for reproducible multi-worker loading | Add `worker_init_fn=seed_worker, generator=get_generator(seed)` from `seed_utils.py`; requires passing seed to `create_dataloader` |

## Recommended-Fix Items

| # | Severity | File | Line | Issue | Fix |
|---|----------|------|------|-------|-----|
| 3 | **Low** | `core/attribution/rept.py` | 52, 191 | `@torch.no_grad()` on functions that need gradients; relies on fragile `torch.enable_grad()` override | Remove `@torch.no_grad()` decorator; keep the internal `torch.enable_grad()` context managers, or better: remove decorator and put `torch.no_grad()` only around the parts that genuinely don't need gradients |
| 4 | **Low** | `core/attribution/rept.py` | 250 | Silent fallback to zero gradients if `h_l.requires_grad` is False | Add explicit warning: `warnings.warn(f"Layer {layer} hidden state has requires_grad=False; gradient will be zero")` |
| 5 | **Low** | `core/attribution/rept.py` | 23-48 | No fallback to L/2 when gradient norms are degenerate (all near-zero) | Add check: if `gradient_norms.max() < eps`, log warning and return `n_layers // 2` |
| 6 | **Low** | `core/attribution/contrastive.py` + `core/evaluation/ablation_analysis.py` | 71, 100 | Duplicate `compute_cmrr` implementations | Remove one; import from a single canonical location |
| 7 | **Low** | `run_attribution.py` | 138 | Redundant `.mean()` on already-reduced loss | Remove the outer `.mean()` or use `reduction='none'` and apply `.mean()` once |
| 8 | **Info** | `core/evaluation/statistical.py` | 50-56 | Permutation test Python loop is slow for 10K iterations | Vectorize: generate all sign matrices at once as `(n_perm, N)` tensor, compute all permuted diffs via matrix multiplication |

---

## Ghost File / Missing File Summary

- **Missing (expected)**: `core/date-lm/` -- DATE-LM codebase not yet cloned. TRAK, BM25, DDA baselines return placeholder scores until integrated.
- **Missing (documented prerequisite)**: `requirements-lock.txt` -- to be generated before first real experiment.
- **Missing (documented prerequisite)**: `_Data/` and `_Results/environment.md` -- generated at experiment time.
- **Extra (benign)**: `configs/pilot.yaml` -- not listed in CLAUDE.md but follows convention.

---

## Key Observations

1. **The codebase is well-structured and closely tracks the design documents.** The component-to-file mapping in CLAUDE.md is accurate. The separation between `core/` (reusable) and `experiments/` (thin wrappers) is clean.

2. **DATE-LM integration is the critical remaining work.** TRAK, BM25, and DDA all depend on the DATE-LM codebase that hasn't been cloned yet. The code handles this gracefully (placeholder scores, clear warnings), but none of the parameter-space baselines can actually run.

3. **The config system is well-designed.** YAML inheritance via `_base_` + CLI overrides makes ablation experiments fully config-driven without code changes.

4. **RepT is the highest-risk component**, as predicted by the design review. The `@torch.no_grad()` decorator pattern is error-prone, and there's no fallback for degenerate gradient profiles.

5. **Dry-run support is comprehensive.** Every experiment script supports `--dry-run` with reduced data and step limits. This is excellent for pre-experiment validation.
