# P4: RepSim Layer Sweep -- Pilot Summary

## Configuration
- Model: EleutherAI/pythia-1b (hidden_dim=2048, num_layers=16)
- N_train: 100 (pilot)
- Layers: ['embed', 'layer_2', 'layer_4', 'layer_7', 'layer_10', 'layer_12', 'layer_15']
- Hidden states indices: [0, 3, 5, 8, 11, 13, 16]
- Metrics: R@50/AUPRC, MRR, Kendall tau, Spearman rho

## Results by Layer

### counterfact (Recall@50)
| Layer | hs_index | Recall@50 | Kendall tau |
|---|---|---|---|
| embed | 0 | 1.0000 | 0.1780 | **best**
| layer_2 | 3 | 1.0000 | 0.1776 | **best**
| layer_4 | 5 | 1.0000 | 0.1776 | **best**
| layer_7 | 8 | 1.0000 | 0.1776 | **best**
| layer_10 | 11 | 1.0000 | 0.1776 | **best**
| layer_12 | 13 | 1.0000 | 0.1768 | **best**
| layer_15 | 16 | 0.9936 | 0.1704 |

- Range: 0.6pp
- Best: embed (1.0000)
- Last layer best: no
- Depth correlation (Spearman): rho=-0.612, p=0.1438

### toxicity (AUPRC)
| Layer | hs_index | AUPRC | Kendall tau |
|---|---|---|---|
| embed | 0 | 0.5218 | 0.2969 |
| layer_2 | 3 | 0.6667 | 0.3909 |
| layer_4 | 5 | 0.6601 | 0.3951 |
| layer_7 | 8 | 0.6598 | 0.4179 |
| layer_10 | 11 | 0.6653 | 0.4051 |
| layer_12 | 13 | 0.6528 | 0.3738 |
| layer_15 | 16 | 0.6852 | 0.3461 | **best**

- Range: 16.3pp
- Best: layer_15 (0.6852)
- Last layer best: yes
- Depth correlation (Spearman): rho=0.393, p=0.3833

### ftrace (Recall@50)
| Layer | hs_index | Recall@50 | Kendall tau |
|---|---|---|---|
| embed | 0 | 0.5236 | 0.0341 |
| layer_2 | 3 | 0.4715 | 0.0415 |
| layer_4 | 5 | 0.3572 | 0.0542 |
| layer_7 | 8 | 0.5564 | 0.0861 |
| layer_10 | 11 | 0.5787 | 0.0849 |
| layer_12 | 13 | 0.6052 | 0.0910 |
| layer_15 | 16 | 0.7474 | 0.1174 | **best**

- Range: 39.0pp
- Best: layer_15 (0.7474)
- Last layer best: yes
- Depth correlation (Spearman): rho=0.857, p=0.0137

## Pass Criteria
- Varies >= 10pp on any task: **PASS**
- Last layer best on semantic: **PASS**
- Overall: **GO**

## Task-Type Boundary Analysis
- Semantic tasks depth preference: {'counterfact': -0.6124, 'ftrace': 0.8571}
- Behavioral (toxicity) depth preference: 0.393
- Semantic prefer later layers: no
- Behavioral different pattern: no

## Runtime: 390.5s