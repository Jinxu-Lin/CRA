# Contribution Tracker: CRA (Contrastive Representation Attribution)

> [ASSIMILATED: Initialized from CRA_old/research/contribution.md. Adapted to Noesis v3 format.]
> 本文档跨阶段维护，记录项目贡献的演化过程。

---

## Contribution List

### Formalize Phase (from CRA_old strategic review)

| # | Contribution | Type | Source Phase | Status |
|---|-------------|------|-------------|--------|
| C0 | Signal-processing diagnostic framework: FM1 (Signal Dilution) + FM2 (Common Influence Contamination) as two independent bottlenecks complementary to Hessian error | Gap identification / Problem definition | CRA_old Crystallize | Conceptual (pending empirical verification) |
| C1 | Unified bilinear taxonomy: 5 representation-space TDA methods as phi^T psi instances | Methodological unification | CRA_old Crystallize | Conceptual complete |
| C2 | First systematic evaluation of representation-space methods on DATE-LM benchmark | Empirical study | CRA_old Crystallize | Pending (probe first) |
| C3 | 2x2 ablation verifying FM1/FM2 independence and additivity | Empirical validation | CRA_old Crystallize | Pending |
| C4 | (Optional) Fixed-IF: theory-guided parameter-space repair | Method innovation | CRA_old Crystallize | Optional, contingent on C2/C3 |

---

## Contribution Assessment

### Overall Publication Value

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| Novelty | Medium-High | No prior unification of 5 methods; no FM1/FM2 diagnostic framework |
| Significance | High | Fills acknowledged gap in DATE-LM coverage; provides practitioner guidance |
| Venue Match (NeurIPS 2026) | Good | TDA is a high-activity NeurIPS area; empirical + conceptual |

### Contribution-to-Paper Mapping

| Contribution | Introduction claim | Experiments verification |
|-------------|-------------------|------------------------|
| C0 | "Parameter-space TDA suffers from two independent signal-processing defects" | 2x2 ANOVA main effects |
| C1 | "Five representation-space methods form a coherent phi^T psi family" | Method correlation matrix |
| C2 | "First systematic evaluation on DATE-LM" | Benchmark tables |
| C3 | "FM1 and FM2 repairs are approximately additive" | ANOVA interaction term |

---

## Metadata
- **Target venue**: NeurIPS 2026
- **Last updated**: Assimilation (2026-03-25)
- **Current status**: Contributions conceptually defined, pending empirical verification
