# Probe Results

> [ASSIMILATED: Synthesized from two sources:
> 1. CRA_old: Probe experiment DESIGNED but not executed (RepSim vs TRAK on DATE-LM)
> 2. CRA (Sibyl): LIBERO-10 pilot experiments COMPLETED (Phase 0 GO) -- different domain (VLA) but relevant methodology]

## Status: PROBE NOT YET EXECUTED

The primary probe (RepSim vs TRAK on DATE-LM toxicity filtering) from CRA_old has not been run. This is the critical next experiment.

---

## Secondary Evidence: CRA (Sibyl) Pilot Results

### Setup
- **Domain**: Multi-task VLA (LIBERO-10), NOT TDA for LLMs
- **Comparison**: 10-task joint model vs. 5-task subset model (tasks 0-4 only)
- **Volume control**: 5-task model trained with upsampled data
- **Architecture**: CNN-4layer encoder + 3-layer MLP action head (~1M params)
- **Training**: 30 epochs, batch size 256, AdamW (lr=3e-4), seed 42
- **Hardware**: 1x NVIDIA RTX 4090, 8.7 minutes

### Results

| Task | 10-task Loss | 5-task Loss | Delta% |
|------|-------------|-------------|--------|
| T0: soup+tomato->basket | 0.675 | 0.931 | -27.5% |
| T1: cheese+butter->basket | 0.866 | 1.005 | -13.8% |
| T2: stove+moka | 0.557 | 0.670 | -16.9% |
| T3: bowl->drawer+close | 0.429 | 0.535 | -19.8% |
| T4: 2mugs->2plates | 0.640 | 0.853 | -25.0% |
| T5-T9 (only in 10-task) | -- | -- | -69.5% avg |

### Key Findings
- **Dominant positive transfer** in LIBERO-10 (all deltas negative = 10-task better)
- Even for shared tasks (T0-T4), 10-task model achieves 14-27% lower prediction error
- Negative transfer (the target signal) may be masked in bulk comparison; pairwise C-LOTO needed
- **Verdict**: Phase 0 GO -- LIBERO-10 viable for cross-task influence study

### Relevance to CRA_old Direction
This pilot validates the *methodology* of cross-task data influence analysis but operates in a different domain (VLA policy learning, not LLM TDA). The pilot results do NOT directly validate the CRA_old FM1/FM2 thesis. The primary probe (RepSim vs TRAK on DATE-LM) remains the critical gate.

---

## Next Steps

1. **Execute primary probe**: RepSim vs TRAK on DATE-LM toxicity filtering task (Pythia-1B)
2. Pass/Fail criteria defined in experiment-design.md Section 1.3
3. Estimated time: 2.5 GPU-days
