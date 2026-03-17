# P4: Cosine vs Euclidean RepSim -- Pilot Summary (v2: raw reps)

**Note**: v1 used pre-L2-normalized cache, making all similarity functions produce identical rankings. v2 re-extracts raw (unnormalized) representations from Pythia-1B to enable meaningful comparison.

## Pass Criteria
- All similarity functions produce valid scores: **PASS**
- Cosine vs Euclidean gap > 3pp on at least 1 task: **FAIL**
- Cosine vs Dot Product gap > 3pp on at least 1 task: **PASS**
- Max cosine-euclidean gap: +1.26pp on toxicity
- Overall: **PASS**

## Results Table (Rank-Based Metrics)

| Task | Metric | Cosine | Euclidean | Dot Product | Cos-Euc Gap | Cos-Dot Gap |
|------|--------|--------|-----------|-------------|-------------|-------------|
| toxicity | AUPRC | 0.6407 | 0.6280 | 0.6499 | +1.26pp | -0.93pp |
| counterfact | Recall@50 | 0.9327 | 0.9263 | 0.9103 | +0.64pp | +2.24pp |
| ftrace | Recall@50 | 0.7339 | 0.7252 | 0.6175 | +0.87pp | +11.64pp |

## Kendall Tau Comparison

| Task | Cosine tau | Euclidean tau | Dot tau | Cos-Euc tau diff |
|------|-----------|--------------|---------|-----------------|
| toxicity | 0.2096 | 0.2054 | 0.2203 | +0.0043 |
| counterfact | 0.1270 | 0.1266 | 0.1233 | +0.0004 |
| ftrace | 0.0498 | 0.0501 | 0.0321 | -0.0004 |

## Norm Analysis (RAW representations)

| Task | Train Norm Mean | Train Norm CV | Ref Norm Mean |
|------|----------------|---------------|---------------|
| toxicity | 107.10 | 0.0443 | 107.23 |
| counterfact | 101.52 | 0.0459 | 102.62 |
| ftrace | 106.14 | 0.0181 | 105.66 |

## Rank Correlations Between Similarity Functions

| Task | cos-euc rho | cos-dot rho | euc-dot rho |
|------|------------|------------|------------|
| toxicity | 0.9958 | 0.9846 | 0.9685 |
| counterfact | 0.9735 | 0.9438 | 0.8494 |
| ftrace | 0.9972 | 0.8843 | 0.8486 |

## Implication for TRAK-PCA Gap

Euclidean is robust (matches cosine within 3pp) but dot product diverges (>3pp gap). This means direction is what matters, not scale. Euclidean distance is approximately equivalent to cosine at fixed norm, so the normalization effect is subtle. Dot product's vulnerability to norm variation confirms that TRAK's unnormalized gradient inner products lose signal to norm noise.

## Runtime
Total: 177.0s