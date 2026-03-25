---
version: "1.0"
status: "assimilated"
decision: "Go with focus"
created: "2026-03-16"
last_modified: "2026-03-25"
---

# Project: CRA (Contrastive Representation Attribution)

> [ASSIMILATED: Fused from CRA_old (Noesis V1, TDA diagnostic framework) and CRA (Sibyl, Cross-Task VLA Influence). Primary direction: CRA_old's TDA diagnostic framework targeting NeurIPS 2026. VLA cross-task influence preserved as secondary knowledge asset.]

## 1. Overview

### 1.1 Topic

Diagnosing why parameter-space Training Data Attribution (TDA) systematically fails on LLMs through a signal-processing lens, unifying 5 independently proposed representation-space TDA methods, and systematically benchmarking them on DATE-LM.

### 1.2 Initial Idea

TDA in actual LLM tasks fails systematically, not because of poor Hessian approximation, but because of two independent signal-processing defects: (1) Signal Dilution (FM1) -- parameter space dimensionality p >> d_h causes per-sample gradients to be nearly orthogonal, collapsing SNR; (2) Common Influence Contamination (FM2) -- standard IF is dominated by generic language patterns from pre-training, drowning task-specific signals. Representation-space methods fix FM1, contrastive scoring fixes FM2, and the two are complementary.

Five independently proposed representation-space methods (RepSim, RepT, In-the-Wild, Concept Influence, AirRep) can all be unified under the bilinear form phi(z_test)^T * psi(z_train), but no one has named this family or explained why it works. This project provides the diagnostic framework + unified taxonomy + first systematic benchmark evaluation on DATE-LM.

### 1.3 Baseline Papers

| # | Paper | Link | Relevance |
|---|-------|------|-----------|
| 1 | Li et al. 2025 "Do Influence Functions Work on LLMs?" | arXiv 2409.19998 | FM1 core evidence: RepSim 96-100% vs IF 0-7% |
| 2 | DDA "Enhancing Training Data Attribution for LLMs" | arXiv 2410.01285 | FM2 core evidence: debias +55pp contribution |
| 3 | RepT "Representation Gradient Tracing" | arXiv 2510.02334 | Rep-space + implicit contrastive, P@10=0.97-1.00 |
| 4 | DATE-LM Benchmark | arXiv 2507.09424 | Primary evaluation benchmark, NeurIPS 2025 |
| 5 | Better Hessians Matter | arXiv 2509.23437 | Core challenger: better Hessian -> better attribution |
| 6 | In-the-Wild | arXiv (2026) | Rep-space + explicit contrastive for DPO alignment |
| 7 | Concept Influence | arXiv | Rep-space concept-level attribution |
| 8 | AirRep | arXiv 2505.18513 | Rep-space encoder-based TDA, NeurIPS 2025 |
| 9 | MAGIC | arXiv 2504.16430 | Key competitor discovered in strategic review |

### 1.4 Available Resources

- **GPU**: 4x RTX A6000 48GB (shared server via SSH MCP)
- **Timeline / DDL**: NeurIPS 2026 (submission ~May 2026), 3-4 months
- **Existing Assets**:
  - CRA_old: Complete theoretical framework (FM1/FM2 diagnostic), strategic review PASS, probe design ready
  - CRA (Sibyl): LIBERO-10 pilot results (Phase 0 GO), remote server code at /home/jinxulin/sibyl_system
  - VITA project: AgiBot World data pipeline, EK-FAC/cosine scoring code (~/Research/VITA/Codes/)
  - Episteme KB: 50 papers, 453 Gap entries covering TDA landscape

---

## 2. Problem & Approach

### 2.1 Baseline Analysis

#### What they solved
- Li et al.: Diagnosed iHVP degeneracy under LoRA, showed RepSim dominance
- DDA: Demonstrated contrastive scoring (debias) provides +55pp on hallucination tracing
- RepT: Achieved near-perfect attribution via representation gradients with phase transition layer detection
- Better Hessians Matter: Proved Hessian quality ordering consistently improves attribution

#### What they didn't solve
- No comparative evaluation across representation-space methods on a common benchmark
- No explanation for WHY representation-space methods work (beyond LoRA-specific analysis)
- No test of contrastive scoring generality beyond 2 task types
- No resolution of tension between "better Hessians help" and "representation space bypasses Hessians"

#### Why they didn't solve it
- Research community fragmentation: parameter-space and representation-space communities use different evaluation protocols
- Each representation-space method was developed for a specific task niche, not as part of a family
- DATE-LM benchmark only recently became available (NeurIPS 2025)

### 2.2 Problem Definition

- **One-line**: Five independently proposed representation-space TDA methods each beat parameter-space methods in their niches, yet no work has recognized them as a coherent family, explained why they work, or evaluated them on a common benchmark.
- **Authenticity**: 5 independent methods in 12 months, none evaluated on DATE-LM (G-RepT4, G-AR2)
- **Importance**: TDA for LLMs is a high-activity area (50 papers in KB, multiple top-venue papers 2025-2026); practitioners lack principled guidance
- **Value level**: "Conditions changed" (LLM scale makes FM1/FM2 dominant) + "Done but fundamentally flawed" (parameter-space TDA)

### 2.3 Root Cause Analysis

**Layer 1 (symptom)**: Parameter-space IF performs poorly on LLM tasks (RepSim 96-100% vs IF 0-7%).
**Layer 2 (FM1 - Signal Dilution)**: In R^B (B~10^9), per-sample gradients are nearly orthogonal; task-relevant signal has extremely low SNR.
**Layer 3 (FM2 - Common Influence Contamination)**: Standard IF measures total influence dominated by shared pre-training knowledge (DDA debias ablation: -55.2pp).
**Layer 4 (structural root cause)**: FM1 and FM2 are structurally coupled to parameter space -- no natural decomposition separates task-specific from general influence at the parameter level. Representation space offers natural decomposition with orders-of-magnitude lower dimensionality.

These are distinct from Hessian approximation error (Better Hessians Matter operates at small scale where FM1/FM2 are mild).

### 2.4 Proposed Approach

Diagnostic framework identifying FM1 and FM2 as two independent signal-processing defects, complementary to Hessian approximation error. Systematic 2x2 ablation {parameter-space, representation-space} x {standard scoring, contrastive scoring} on DATE-LM + Li et al. benchmarks. Unified taxonomy showing 5 methods share phi^T psi bilinear structure.

Signal-processing theory provides deep support: matched filtering (dimensionality reduction maximizes SNR) corresponds to representation-space operation; differential detection (subtract reference channel) corresponds to contrastive scoring. 70+ years of orthogonality theory from signal processing.

### 2.5 Core Assumptions

| # | Assumption | Type | Source | Strength | If False |
|---|-----------|------|--------|----------|----------|
| H1 | FM1 (signal dilution) is the primary cause of parameter-space IF failure on LLMs | Theoretical | Li et al. 2025 | Medium-Strong (LoRA only) | Theoretical framework loses foundation |
| H2 | Contrastive scoring effectiveness is general, not limited to DDA's hallucination tracing | Empirical | DDA + In-the-Wild | Weak-Medium (2 task types only) | "Universal enhancement" claim downgraded |
| H3 | FM1 and FM2 repair gains are approximately additive | Theoretical | Theory conjecture | Weak (no direct verification) | 2x2 narrative needs downgrade |
| H4 | Representation-space method differences reflect different signal types | Empirical | Concept IF correlation 0.37-0.45 | Medium (indirect) | Unified framework lacks discriminability |
| H5 | Representation-space methods on DATE-LM LDS are not significantly worse than parameter-space | Empirical | None | None | "Systematic superiority" narrative broken |

---

## 3. Validation Strategy

### 3.1 Idea Type Classification

Hybrid: New diagnostic framework (conceptual) + Systematic evaluation (empirical) + Unified taxonomy (organizational). Verification focus is on the empirical validation of the diagnostic framework through 2x2 ablation.

### 3.2 Core Hypothesis

If the CRA thesis is correct, RepSim (simplest representation-space method) should achieve LDS >= TRAK - 5pp on at least one DATE-LM task (preferably toxicity filtering, most analogous to Li et al.'s harmful data identification).

### 3.3 Probe Experiment Design

**Step 1 (0.5 day)**: Clone DATE-LM, select Pythia-1B + toxicity filtering task.
**Step 2 (1 day)**: Implement RepSim on DATE-LM (extract h^(l), compute cosine similarity, submit to LDS evaluation).
**Step 3 (0.5 day)**: Run TRAK baseline using DATE-LM's existing implementation.
**Step 4 (0.5 day)**: Compare + interpret. If time permits, also run on data selection task.

Total compute: < 1 GPU-day on a single A100.

### 3.4 Pass / Fail Criteria

| Result | Condition | Action |
|--------|-----------|--------|
| Strong Pass | RepSim LDS >= TRAK LDS on toxicity filtering | Full systematic evaluation justified |
| Pass | RepSim LDS >= TRAK LDS - 5pp | Representation space viable, proceed |
| Weak Pass | RepSim < TRAK - 5pp on toxicity, but >= TRAK - 5pp on data selection | Task-dependent; CRA thesis needs scoping |
| Fail | RepSim < TRAK - 5pp on both tasks | Direction needs fundamental re-evaluation |

### 3.5 Time Budget & Resources

2.5 days (setup + implementation + evaluation + analysis). 1 A100 GPU.

### 3.6 Failure Diagnosis Plan

| Failure Mode | Characteristic | Meaning | Action |
|-------------|---------------|---------|--------|
| RepSim fails + TRAK also low | Both struggle | DATE-LM evaluation issue | Check LDS metric validity |
| RepSim fails, TRAK succeeds | Rep captures correlation not causation | Confirms H-IF-LLM4 | Reframe to nuanced comparison |
| RepSim fails due to layer | Wrong layer chosen | Layer selection critical | Follow up with RepT (auto-selects layer) |

---

## 4. Review

### 4.1 Review History

| Round | Date | Decision | Key Changes |
|-------|------|----------|-------------|
| 1 (CRA_old) | 2026-03-16 | Go with focus | 6-agent stress test passed; narrative corrections required |

### 4.2 Latest Assessment Summary

- **Innovator**: Unification defines new subfield; need Fixed-IF for oral-level ceiling
- **Pragmatist**: Core components open-source, pilot <= 1 GPU-day; 3-4 week first results
- **Theorist**: FM1 has JL lemma support; signal processing analogy provides orthogonality theory
- **Contrarian**: 25-35% success probability; must benchmark against MAGIC/DDA; Hessian could be the real bottleneck (30-40%)
- **Interdisciplinary**: Matched filtering + differential detection = deep isomorphism; z_cf ablation needed
- **Empiricist**: DATE-LM open-source provides standardized evaluation; pilot must be front-loaded

### 4.3 Decision

- **Decision**: Go with focus (direction correction)
- **Rationale**: 5+ independent works converging on unrecognized paradigm shift; infrastructure complete; signal processing theory provides deep support
- **Key Risks**: Hessian is the real bottleneck (30-40%); FM1 is LoRA artifact; concurrent competition
- **Unresolved Disputes**: Contrarian estimates 25-35% success; Innovator wants Fixed-IF; Pragmatist concerned about scope

### 4.4 Conditions for Next Module

1. Correct "Hessian doesn't matter" narrative -> "three complementary bottlenecks"
2. Downgrade orthogonality from assumption to testable hypothesis
3. Narrow experimental scope to RepSim + RepT + TRAK + DDA
4. Include MAGIC + DDA as mandatory baselines

<!-- Full debate records from CRA_old: Reviews preserved in CRA_old project -->
