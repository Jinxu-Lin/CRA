# Code Baseline Verification Report

> Generated: 2026-03-26
> Environment: Local machine (no GPU, no model, no DATE-LM data)
> Purpose: Pre-deployment code verification before remote GPU execution

## Phase 0: Code-Level Sanity Checks

### Unit Tests: 143/143 PASSED

All unit tests pass across all modules:

| Module | Tests | Status |
|--------|-------|--------|
| test_representation.py | 14 | PASS |
| test_repsim.py | 11 | PASS |
| test_rept.py | 13 | PASS |
| test_gradsim.py | 10 | PASS |
| test_contrastive.py | 11 | PASS |
| test_metrics.py | 17 | PASS |
| test_statistical.py | 15 | PASS |
| test_ablation_analysis.py | 13 | PASS |
| test_date_lm_loader.py | 14 | PASS |
| test_magic.py | 7 | PASS |
| test_integration.py | 10 | PASS |
| **Total** | **143** | **ALL PASS** |

### Sanity Check Script: 6/6 PASSED

`scripts/sanity_check.py --check all` results:

| Check | Status | Details |
|-------|--------|---------|
| config | PASS | 18/18 YAML configs loaded without error |
| shape | PASS | Config structure validated (model dims, layer specs) |
| metrics | PASS | LDS, AUPRC, P@K, Recall@50, MRR all compute correctly |
| statistical | PASS | Permutation test, bootstrap CI, Cohen's d, BH-FDR all work |
| ablation | PASS | 2x2 main effects, interaction, independence assessment work |
| scoring | PASS | RepSim, GradSim, contrastive scoring produce correct shapes/ranges |

### Dry-Run Pipeline: PASSED

`evaluate.py --config configs/base.yaml --dry-run` confirms:
- All metric functions callable and return valid values
- Statistical analysis functions work end-to-end
- Ablation analysis pipeline produces correct output format
- Markdown result generation works

### Bug Found and Fixed

**Spearman rank tie handling**: `_rank()` in `core/evaluation/metrics.py` did not handle tied values correctly. For input `[1,2,2,3]`, it assigned ranks `[1,2,3,4]` instead of the correct `[1, 2.5, 2.5, 4]`. This would produce incorrect LDS values when attribution scores have ties (likely in real data). Fixed to use average ranking for ties.

## Remaining Phase 0 Checks (Require GPU Server)

The following checks from `experiment-todo.md` Phase 0 require the actual model (Pythia-1B) and DATE-LM data, which are only available on the remote GPU server:

| Check | Status | Requirement |
|-------|--------|-------------|
| shape-check (actual model) | PENDING | Pythia-1B + DATE-LM data |
| representation-extraction-check | PENDING | Pythia-1B + 15 samples |
| trak-integration-check | PENDING | DATE-LM codebase clone |
| contrastive-scoring-check | PENDING | Pythia-1B + base model |

**Commands to run on GPU server:**
```bash
cd ~/Research/CRA/Codes
python scripts/sanity_check.py --check all
python run_attribution.py --config configs/probe_repsim.yaml --dry-run
python run_attribution.py --config configs/probe_gradsim.yaml --dry-run
python evaluate.py --config configs/base.yaml --dry-run
```

## Phase 1: Probe (NOT EXECUTED)

The critical gate experiment (RepSim vs TRAK on DATE-LM toxicity) has not been run. Requires:
1. Clone DATE-LM codebase to `Codes/core/date-lm/`
2. Download DATE-LM datasets to `~/Resources/Datasets/DATE-LM/`
3. Download/link Pythia-1B to `~/Resources/Models/pythia-1b/`
4. Run probe experiments per `experiment-todo.md` Phase 1

## Phase 2: Baseline Reproduction (NOT EXECUTED)

Blocked on Phase 1 completion and DATE-LM integration.

## Placeholder Results Cleaned

All dummy/placeholder result files from the implement phase have been removed:
- `_Results/experiment_result.md` (contained random data)
- `_Results/benchmark/`, `_Results/ablation/`, `_Results/pilot/` (all dummy)
- `_Data/scores/toxicity/trak_standard_seed42.pt` (placeholder tensor)

Only `_Results/probe_result.md` (project history) and `_Results/magic/feasibility_report.md` (valid theoretical computation) retained.

## Code Architecture Verification

All components implemented and tested:

| Component | File | Status |
|-----------|------|--------|
| Representation extraction | `core/data/representation.py` | Tested (14 tests) |
| RepSim scoring | `core/attribution/repsim.py` | Tested (11 tests) |
| RepT scoring | `core/attribution/rept.py` | Tested (13 tests) |
| GradSim scoring | `core/attribution/gradsim.py` | Tested (10 tests) |
| Contrastive wrapper | `core/attribution/contrastive.py` | Tested (11 tests) |
| MAGIC feasibility | `core/attribution/magic.py` | Tested (7 tests) |
| Evaluation metrics | `core/evaluation/metrics.py` | Tested (17 tests), bug fixed |
| Statistical analysis | `core/evaluation/statistical.py` | Tested (15 tests) |
| Ablation analysis | `core/evaluation/ablation_analysis.py` | Tested (13 tests) |
| DATE-LM data loader | `core/data/date_lm_loader.py` | Tested (14 tests) |
| Config system | `config_utils.py` | 18/18 configs load |
| Seed management | `seed_utils.py` | Tested via integration |
| Logging | `logging_utils.py` | Tested via integration |
| Attribution pipeline | `run_attribution.py` | Dry-run verified |
| Evaluation pipeline | `evaluate.py` | Dry-run verified |

## Next Steps

1. **Deploy to GPU server**: `git push` and `git pull` on server
2. **Environment setup**: Install dependencies, verify GPU access
3. **Clone DATE-LM**: `git clone <DATE-LM-repo> Codes/core/date-lm/`
4. **Run Phase 0 GPU checks**: shape, representation, TRAK integration, contrastive
5. **Run Phase 1 Probe**: RepSim vs TRAK critical gate
6. **Run Phase 1.5 Pilot**: 2x2 sanity check (if probe passes)
