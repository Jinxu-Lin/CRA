# Discussion and Conclusion

## 5.1 Key Findings Summary

We proposed a diagnostic framework decomposing LLM-scale TDA failure into three independent bottlenecks---Hessian approximation error, signal dilution (FM1), and common influence contamination (FM2)---and validated this decomposition through systematic experiments on DATE-LM.

Our $2 \times 2$ ablation reveals that bottleneck severity is task-dependent: {{PENDING: bottleneck_profiles | which bottleneck dominates which task, e.g., FM2 dominates toxicity filtering while FM1 dominates data selection | Expected: task-dependent profiles with FM2 dominant on toxicity and FM1 dominant on data selection}}. The interaction term is {{PENDING: interaction_summary | small/moderate/large relative to main effects | Expected: small, supporting independence}}, {{PENDING: independence_conclusion | supporting/weakening the independence assumption of the three-bottleneck framework | Expected: supporting}}.

The systematic benchmark establishes that {{PENDING: benchmark_summary | key method ranking findings and practical recommendations | Expected: representation-space methods competitive on LDS for toxicity and data selection; no single method dominates all tasks}}. The LoRA vs. full fine-tuning comparison shows that FM1 is {{PENDING: fm1_generality | a general phenomenon / a LoRA-specific artifact | Expected: general, with larger RepSim advantage under full-FT}}.

## 5.2 Practitioner Guidance

Based on our findings, we provide the following recommendations for TDA method selection:

| Task Type | Compute Budget | Recommended Method |
|-----------|---------------|-------------------|
| Toxicity filtering | Low | {{PENDING: rec_tox_low | best low-compute method for toxicity | Expected: RepSim or RepSim-C}} |
| Toxicity filtering | Medium | {{PENDING: rec_tox_med | best medium-compute method for toxicity | Expected: RepSim-C}} |
| Data selection | Low | {{PENDING: rec_ds_low | best low-compute method for data selection | Expected: RepSim}} |
| Factual attribution | Low | {{PENDING: rec_fact_low | best low-compute method for factual | Expected: BM25 or RepSim}} |
| Any task (max accuracy) | Very high | MAGIC (if feasible) |

The general principle emerging from our analysis is that practitioners should first identify which bottlenecks are most severe for their task (using the Common-Mode Fraction (CMF) to diagnose FM2 severity and the parameter-space vs. representation-space gap to diagnose FM1 severity), then select methods that address the dominant bottleneck.

## 5.3 Limitations

We identify several limitations of our work, framed as conscious trade-offs:

1. **Probe not executed prior to full experiments.** All experimental predictions are based on theoretical analysis and indirect evidence from prior work. The actual LDS performance of representation-space methods on DATE-LM is unverified at the time of writing, creating a risk that our framework's predictions do not hold.

2. **FM1 evidence is primarily LoRA-based.** The strongest prior evidence for FM1 (Li et al.'s iHVP degeneracy analysis) is obtained entirely under LoRA fine-tuning. While we test FM1 under full fine-tuning (Experiment 3), the possibility remains that FM1 is primarily a LoRA artifact, reducing the three-bottleneck framework to two bottlenecks (Hessian + FM2) for the most common training regime.

3. **MAGIC feasibility may leave Hessian error unbounded.** If MAGIC is computationally infeasible at Pythia-1B scale, we cannot directly measure the Hessian error contribution, leaving the relative importance of the three bottlenecks incompletely resolved.

4. **Bilinear taxonomy is organizational, not theoretical.** Our observation that representation-space methods share a $\phi^\top \psi$ structure is a useful taxonomic tool but does not constitute a deep theoretical unification. Methods like Concept IF and AirRep do not cleanly fit this mold.

5. **Benchmark scope.** DATE-LM covers three tasks on two model scales (Pythia-1B, Llama-7B). Generalization to other tasks (e.g., instruction following, code generation), other model architectures (e.g., mixture-of-experts), and pre-training data attribution remains untested.

6. **LDS metric reliability.** Recent work raises concerns about LDS as a metric for TDA evaluation. If LDS does not faithfully capture attribution quality, all quantitative comparisons in our study are affected.

## 5.4 Future Work

Our framework opens several directions for future investigation:

1. **Pre-training data attribution.** Extending the three-bottleneck analysis to pre-training (rather than fine-tuning) data attribution, where FM2 correction is more challenging because there is no natural "before fine-tuning" reference model for contrastive scoring.

2. **Formal theoretical analysis of FM1.** Deriving tight bounds on the severity of signal dilution as a function of parameter dimensionality $B$, representation dimensionality $d$, training set size $N$, and fine-tuning regime (LoRA rank $r$ vs. full parameters), going beyond the JL-based approximation.

3. **Hybrid methods.** Developing methods that combine representation-space operation (addressing FM1), contrastive scoring (addressing FM2), and improved Hessian approximation in a principled way, potentially achieving the accuracy of exact IF at the cost of representation-space methods.

**Reproducibility.** We will release all code, including our RepSim-C and TRAK-C implementations, fine-tuning scripts, and evaluation pipelines, upon publication. All experiments use the publicly available DATE-LM benchmark and open-weight models (Pythia-1B, Llama-7B).
