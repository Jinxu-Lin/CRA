# Reproducibility Critique Report

## Overall Assessment
- **Score**: 7.0 / 10
- **Core Assessment**: The paper provides a solid foundation for reproducibility: clear experimental protocol, fair comparison design (same checkpoint, same evaluation pipeline), and a well-defined statistical plan. The main gaps are in implementation details -- several critical parameters are either implicit (learning rate, training schedule specifics) or deferred to the DATE-LM codebase without specifying which version/commit. The contrastive scoring variants (RepSim-C, TRAK-C) are novel combinations that need more explicit algorithmic descriptions. Compute requirements are well-characterized, which is commendable.
- **Would this dimension cause reject at a top venue?**: No. The reproducibility level is typical for NeurIPS empirical papers.

## Issues (by severity)

### [Major] RepSim-C and TRAK-C implementations are underspecified
- **Location**: Section 3.3 (Contrastive Scoring), Section 4.1 (Implementation Details)
- **Problem**: The contrastive variants are the paper's novel methodological combinations. RepSim-C subtracts base-model cosine similarity from fine-tuned model similarity (Eq. 4). TRAK-C subtracts base-model TRAK scores from fine-tuned TRAK scores. However, several implementation questions are unanswered: (a) Are base-model and fine-tuned representations extracted at the same layer? For RepSim, the optimal layer may differ between base and fine-tuned models. (b) For TRAK-C, are the random projection matrices shared between base and fine-tuned model runs? Different projections would add noise. (c) Is the contrastive score normalized? Raw cosine differences could have a very different scale than raw cosine similarities, potentially requiring different downstream processing.
- **Simulated reviewer phrasing**: "The contrastive variants (RepSim-C, TRAK-C) are central to the 2x2 ablation but their implementation details are insufficient for reproduction. Are the same layers used for base and fine-tuned representations? Are TRAK projection matrices shared?"
- **Suggested fix**: Add an "Implementation Details for Contrastive Variants" paragraph in S4.1 specifying: (a) same layer index for both models, (b) shared random projection for TRAK-C, (c) no additional normalization on contrastive scores, (d) whether the subtraction is applied before or after any score aggregation.

### [Major] Fine-tuning details for LoRA and Full-FT conditions are incomplete
- **Location**: Section 4.1, Section 4.4 (Experiment 3)
- **Problem**: The paper mentions "LoRA (rank 16)" for the LoRA condition and gives a learning rate sweep for Full-FT, but several training details are missing: (a) LoRA alpha, target modules (all attention? attention + MLP?), dropout; (b) Optimizer (AdamW? learning rate schedule? warmup?); (c) Training epochs/steps for each task; (d) Batch size; (e) Whether the fine-tuned models achieve comparable task performance (if LoRA and Full-FT achieve different task accuracy, the TDA comparison is confounded by model quality, not just fine-tuning regime).
- **Suggested fix**: Add a training details table or paragraph specifying all hyperparameters for both LoRA and Full-FT conditions. Include validation performance to confirm models are comparable.

### [Minor] DATE-LM codebase version not pinned
- **Location**: Section 4.1
- **Problem**: Multiple methods are sourced from "DATE-LM codebase" but no specific version, commit hash, or URL is provided. DATE-LM may update their codebase between now and publication, potentially changing results.
- **Suggested fix**: Pin the DATE-LM repository version (commit hash) and include the URL. Same for any other external codebases.

### [Minor] RepT phase-transition layer detection not described
- **Location**: Section 3.3 (RepT), Section 4.1
- **Problem**: RepT "uses automatic phase-transition layer detection" but the detection algorithm is not described in the paper. A reader cannot implement RepT from the paper alone. The method section gives only "l* detection: layer where gradient norm exhibits phase transition (sharp change)" -- which is too vague for reproduction.
- **Suggested fix**: Either (a) add a brief description of the phase-transition detection algorithm (e.g., "the layer with maximum second derivative of ||nabla_h L||_2 across layers"), or (b) cite the RepT paper and state "we use the authors' released implementation."

### [Minor] Compute costs listed as ranges rather than precise measurements
- **Location**: Section 4.7 (Table 5)
- **Problem**: All efficiency numbers are PENDING with expected ranges (e.g., "0.1-0.5 GPU-hours"). While this is expected in PLACEHOLDER mode, the final version should report precise measurements, not ranges. Also, the current table doesn't specify whether GPU-hours include preprocessing (representation extraction / gradient computation) or only scoring.
- **Suggested fix**: When filling in Table 5, distinguish preprocessing time from scoring time. Report wall-clock time on a single A6000 with specific batch size.

### [Minor] No mention of code release
- **Location**: Entire paper
- **Problem**: The paper does not mention whether code, models, or attribution scores will be released. For a benchmark/diagnostic paper, code release is particularly important.
- **Suggested fix**: Add a sentence in the introduction or conclusion: "We will release all code, fine-tuned model checkpoints, and precomputed attribution scores to facilitate reproduction and future research."

## Strengths
- The fair comparison protocol (same checkpoint, same pipeline, cosine normalization) is well-thought-out and addresses a common criticism of TDA benchmarks.
- Hardware specifications (A6000 48GB) and compute budget (60 GPU-days) are clearly stated, allowing other researchers to estimate their own feasibility.
- The statistical plan (permutation tests, bootstrap CI, FDR correction) is thorough and appropriate for the claims.

## Summary Recommendations
The paper is reasonably reproducible for a NeurIPS submission but needs more implementation details for the novel components (contrastive variants) and the fine-tuning setup (LoRA/Full-FT hyperparameters). Pin all external codebases to specific versions. Add a code release statement. The most impactful improvement would be a clear algorithmic specification of RepSim-C and TRAK-C, since these are the novel combinations that don't exist in any prior paper.
