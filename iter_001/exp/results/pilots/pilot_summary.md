# Phase 0 Pilot Summary: Cross-Task Influence Feasibility

## Verdict: GO

The pilot experiment confirms that task composition measurably affects per-task performance in LIBERO-10, validating the research premise.

## Setup

- **Models compared**: 10-task joint model vs. 5-task subset model (tasks 0-4 only)
- **Volume control**: 5-task model trained with upsampled data to match 10-task total volume
- **Evaluation**: Held-out action prediction MSE (15% demo holdout), 7 eval demos per task
- **Architecture**: CNN-4layer encoder + 3-layer MLP action head (~1M params)
- **Training**: 30 epochs, batch size 256, AdamW (lr=3e-4), seed 42
- **Hardware**: 1x NVIDIA RTX 4090, completed in 8.7 minutes

## Key Results

### Per-Task Loss Comparison (MSE, lower = better)

| Task | 10-task Loss | 5-task Loss | Delta | Delta% |
|------|-------------|-------------|-------|--------|
| T0: soup+tomato->basket | 0.675 | 0.931 | -0.256 | -27.5% |
| T1: cheese+butter->basket | 0.866 | 1.005 | -0.139 | -13.8% |
| T2: stove+moka | 0.557 | 0.670 | -0.113 | -16.9% |
| T3: bowl->drawer+close | 0.429 | 0.535 | -0.106 | -19.8% |
| T4: 2mugs->2plates | 0.640 | 0.853 | -0.213 | -25.0% |
| T5: book->caddy | 0.185 | 3.308 | -3.122 | -94.4% |
| T6: mug+pudding->plate | 0.707 | 1.932 | -1.225 | -63.4% |
| T7: soup+cheese->basket | 0.649 | 3.156 | -2.507 | -79.4% |
| T8: 2mokas->stove | 1.104 | 2.040 | -0.936 | -45.9% |
| T9: mug->microwave+close | 0.782 | 2.183 | -1.401 | -64.2% |

### Gate Check
- **Gate criterion**: At least 1 task shows >3% difference --> **PASSED** (all 10 tasks show >5%)
- **Strong signal**: At least 1 task shows >5% difference --> **YES** (all 10 tasks)

### Group Analysis
- **Tasks 0-4** (in both models): mean delta = **-20.6%** (10-task model is 20.6% better)
- **Tasks 5-9** (only in 10-task): mean delta = **-69.5%** (expected: not trained in 5-task model)

## Interpretation

1. **Dominant positive transfer**: All deltas are negative, meaning the 10-task model outperforms the 5-task model on ALL tasks. This is consistent with LIBERO-10 tasks being related tabletop manipulation tasks that share visual and motor primitives.

2. **Tasks 0-4 analysis** (the meaningful comparison): Even for tasks present in both models, the 10-task model achieves 14-27% lower prediction error. This means tasks 5-9 provide positive transfer to tasks 0-4. The volume control ensures this is not simply a "more diverse data" effect.

3. **Overfitting**: Both models show overfitting (train loss ~0.008, eval loss ~0.66). Future experiments should use early stopping.

4. **Caveat - proxy metric**: We use action prediction MSE, not rollout success rate. The loss differences are large enough to be confident about direction, but magnitudes may not map linearly to success rate differences.

## Implications for Full Experiments

- **LIBERO-10 is a viable benchmark** for studying cross-task interactions. The signal is strong.
- **Positive transfer dominates** in this bulk comparison. The C-LOTO design (Phase 1) is critical: removing a SINGLE task may reveal negative transfer that is masked when removing 5 tasks at once.
- **Hypothesis H1 (existence of negative transfer)** is still open -- this pilot shows positive transfer in aggregate, but pairwise C-LOTO could reveal negative transfer between specific task pairs.
- **Install mujoco/robosuite** for rollout-based evaluation before Phase 1.

## Next Steps

1. Install mujoco + robosuite for proper rollout evaluation
2. Implement ResNet-18 + MLP policy (~5M params) as specified in methodology
3. Proceed to Phase 1: C-LOTO (11 configs x 5 seeds x 200 rollouts)
