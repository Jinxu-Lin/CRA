# Pilot Summary: p1_fm2_continuous_metrics

## Task
Full method tournament with 8 methods x 2 scorings (standard, contrastive) x 3 tasks,
evaluated with both rank-based AND continuous metrics (Kendall tau, Spearman rho, NDCG).

## Key Finding: Kendall Tau is Also Invariant to Mean Subtraction

**CRITICAL**: The hypothesis that Kendall tau (a continuous metric) would detect FM2 effects
where rank-based metrics failed is WRONG. Kendall tau is a rank correlation metric -- it
measures concordance of ranks, not raw score values. Mean subtraction (contrastive scoring)
shifts all scores by the same constant per column, preserving all pairwise orderings.
Therefore Kendall tau, Spearman rho, and NDCG are all exactly invariant to contrastive
scoring, producing 0.0 gain across all 24 method-task cells.

This means the entire "continuous metrics for FM2" approach in the methodology is fundamentally
flawed. The contamination injection experiment (p1_fm2_contamination_injection) with
**non-uniform** contamination is the only viable path to test FM2.

## Full Method Tournament Results (Standard Scoring)

### Rank-Based Metrics
| Method   | toxicity(AUPRC) | counterfact(R@50) | ftrace(R@50) |
|----------|----------------|-------------------|--------------|
| DiagIF   | **0.982**      | **0.965**         | 0.655        |
| RawDotIF | 0.954          | 0.753             | 0.644        |
| TRAK     | 0.926          | 0.670             | 0.590        |
| LoGra    | 0.922          | 0.798             | 0.583        |
| DDA      | 0.876          | 0.692             | 0.651        |
| RepSim   | 0.809          | 0.949             | 0.661        |
| kNN      | 0.743          | 0.949             | 0.661        |
| BM25     | 0.510          | **1.000**         | **0.661**    |

### Continuous Metrics (Kendall tau)
| Method   | toxicity(tau) | counterfact(tau) | ftrace(tau) |
|----------|--------------|------------------|-------------|
| DiagIF   | **0.564**    | **0.147**        | 0.011       |
| RawDotIF | 0.555        | 0.074            | 0.004       |
| LoGra    | 0.534        | 0.076            | 0.014       |
| TRAK     | 0.528        | 0.056            | 0.011       |
| DDA      | 0.505        | 0.052            | 0.009       |
| RepSim   | 0.465        | 0.143            | 0.030       |
| kNN      | 0.453        | 0.161            | 0.025       |
| BM25     | 0.308        | **0.256**        | **0.088**   |

## New Findings

1. **DiagIF is surprisingly strong**: Diagonal Fisher IF achieves best AUPRC (0.982) on toxicity
   AND best R@50 (0.965) on counterfact, matching/beating RepSim. This was not tested in the
   original pilot. Implication: the Hessian diagonal captures meaningful per-parameter scaling.

2. **BM25 dominates continuous metrics on semantic tasks**: BM25 has the highest Kendall tau
   on counterfact (0.256) and ftrace (0.088). This supports the contrarian hypothesis that
   "RepSim is just retrieval" for semantic tasks.

3. **Task-type boundary confirmed again**: Parameter-space methods (DiagIF, RawDotIF, TRAK, LoGra)
   dominate toxicity; representation-space and retrieval methods dominate counterfact/ftrace.

4. **Contrastive scoring is a no-op for ALL metrics tested**: Kendall tau = Spearman rho = NDCG =
   invariant to mean subtraction. FM2 cannot be tested via contrastive scoring + any correlation metric.

## Pass Criteria
- **FAILED**: max |Kendall tau diff| = 0.0 (threshold: > 0.01)
- Reason: Mathematical invariance, not lack of FM2 effect

## Runtime
- Total: 202.8s (~3.4 min) on GPU 2 (RTX 4090, shared memory)
- Gradient extraction dominated: ~150s for 3 tasks x (100 train + refs)

## Recommendation
Proceed to p1_fm2_contamination_injection with non-uniform contamination (alpha * mu injection)
as the definitive FM2 test. The contrastive scoring approach cannot distinguish FM2 effects
from no-effect using any rank-based or rank-correlation metric.
