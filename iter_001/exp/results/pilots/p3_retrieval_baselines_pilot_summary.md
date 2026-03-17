# P3: Retrieval Baselines Pilot Summary

## Configuration
- Mode: PILOT (N=100)
- Methods: Contriever (facebook/contriever), GTR-T5 (sentence-transformers/gtr-t5-base), BM25
- Tasks: toxicity, counterfact, ftrace
- Total runtime: 55.7s
- GPU: RTX 4090 (CUDA_VISIBLE_DEVICES=2)

## Results

### Comparison Table (Retrieval vs RepSim)

| Method | Task | Primary Metric | Value | RepSim Value | Gap (pp) | Kendall tau | RepSim tau |
|--------|------|----------------|-------|-------------|----------|-------------|------------|
| Contriever | toxicity | AUPRC | 0.968 | 0.809 | **+15.9** | 0.555 | 0.465 |
| Contriever | counterfact | R@50 | 1.000 | 0.949 | **+5.1** | 0.172 | 0.143 |
| Contriever | ftrace | R@50 | 0.682 | 0.660 | +2.2 | 0.102 | 0.030 |
| GTR-T5 | toxicity | AUPRC | 0.842 | 0.809 | +3.3 | 0.465 | 0.465 |
| GTR-T5 | counterfact | R@50 | 1.000 | 0.949 | **+5.1** | 0.173 | 0.143 |
| GTR-T5 | ftrace | R@50 | 0.832 | 0.660 | **+17.2** | 0.141 | 0.030 |
| BM25 | toxicity | AUPRC | 0.509 | 0.809 | -30.0 | 0.308 | 0.465 |
| BM25 | counterfact | R@50 | 1.000 | 0.949 | +5.1 | 0.256 | 0.143 |
| BM25 | ftrace | R@50 | 0.661 | 0.660 | +0.1 | 0.088 | 0.030 |

### Contriever
- toxicity: AUPRC=0.9682 [0.9028, 1.0000]
  - Kendall tau=0.555033, Spearman rho=0.6764
- counterfact: R@50=1.0000 [1.0000, 1.0000], MRR=0.8091
  - Kendall tau=0.171884, Spearman rho=0.209469
- ftrace: R@50=0.6822 [0.5927, 0.7711], MRR=0.5593
  - Kendall tau=0.102275, Spearman rho=0.124639

### GTR-T5
- toxicity: AUPRC=0.8415 [0.6776, 0.9521]
  - Kendall tau=0.465488, Spearman rho=0.567275
- counterfact: R@50=1.0000 [1.0000, 1.0000], MRR=0.8598
  - Kendall tau=0.173108, Spearman rho=0.210961
- ftrace: R@50=0.8317 [0.7368, 0.9033], MRR=0.4726
  - Kendall tau=0.140669, Spearman rho=0.171429

### BM25
- toxicity: AUPRC=0.5095 [0.2877, 0.7263]
  - Kendall tau=0.30843, Spearman rho=0.375874
- counterfact: R@50=1.0000 [1.0000, 1.0000], MRR=0.7422
  - Kendall tau=0.256025, Spearman rho=0.275687
- ftrace: R@50=0.6609 [0.5551, 0.7579], MRR=0.5886
  - Kendall tau=0.0875, Spearman rho=0.106591

## Critical Findings

1. **Contriever MATCHES or EXCEEDS RepSim on ALL 3 tasks.** At pilot scale, a generic dense retrieval model (not trained on any attribution task) achieves comparable or better performance than model-internal representation similarity.

2. **GTR-T5 EXCEEDS RepSim on ftrace (+17.2pp) and counterfact (+5.1pp).** Only toxicity shows a modest gap (+3.3pp). GTR-T5 is a general-purpose text retrieval model with no knowledge of LLM internals.

3. **BM25 is the clear differentiator for toxicity.** BM25 AUPRC=0.509 vs RepSim=0.809 (-30.0pp). Lexical matching fails on behavioral detection, confirming that toxicity attribution requires semantic understanding. But BM25 matches RepSim on counterfact and ftrace at pilot scale.

4. **Kendall tau: Contriever >= RepSim on all tasks.** Contriever tau=0.555 vs RepSim tau=0.465 on toxicity; 0.172 vs 0.143 on counterfact; 0.102 vs 0.030 on ftrace.

5. **Pilot scale saturation caveat.** R@50=1.0 for all methods on counterfact at N=100 is a saturation artifact. Full-scale (N=5473) will differentiate methods.

## Decision Gate Evaluation

- **Condition:** Contriever or GTR-T5 matches RepSim (< 3pp gap) on >= 2 tasks
- **Result:** **TRIGGERED** -- both Contriever and GTR-T5 match or exceed RepSim on all 3 tasks
- **Action:** Paper must consider repositioning from "attribution quality" to "attribution vs retrieval boundary analysis"

## Implications for Paper

If full-scale results confirm that generic retrieval matches RepSim:
1. RepSim's advantage over parameter-space methods may be due to operating in a "retrieval-friendly" space, not genuine model-internal attribution
2. The toxicity reversal (where parameter-space methods dominate) becomes even more important
3. Paper narrative shifts to: "When is attribution more than retrieval?"

## Caveats
- Pilot scale (N=100) with R@50 saturation inflates all methods
- Full-scale experiment with N=5473 (counterfact) and N=10187 (toxicity) is essential
- Contriever and GTR-T5 use different tokenization and pooling than Pythia-1B RepSim

## Recommendation
**GO** for full-scale experiment. The pilot reveals that the retrieval baseline question is even more critical than anticipated.
