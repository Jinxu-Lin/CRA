# Interdisciplinary Perspective: Cross-Task Influence in Multi-Task VLA

## Meta-Observation

The existing perspectives (innovator, theoretical, pragmatist, contrarian) approach the cross-task influence problem primarily from within ML -- influence functions, gradient surgery, spectral analysis of task interaction matrices. While technically sound, these approaches treat the influence matrix as a novel construct to be engineered. But the *structure* of the problem -- N interacting entities that can help or harm each other, forming a signed interaction network with emergent system-level dynamics -- is ancient and deeply studied in at least three disciplines outside ML. Each offers not just metaphorical inspiration but concrete mathematical machinery and testable predictions that can be transplanted.

---

## Analogy 1: Community Ecology -- Lotka-Volterra Interaction Matrices and the Competitive Exclusion Principle

### The Structural Correspondence

The task-to-task influence matrix $M_{ij}$ proposed in this project is structurally isomorphic to the **community interaction matrix** $\alpha_{ij}$ in generalized Lotka-Volterra (gLV) ecology, where $\alpha_{ij}$ quantifies how species $j$ affects the growth rate of species $i$:

$$\frac{dx_i}{dt} = x_i \left( r_i - \sum_j \alpha_{ij} x_j \right)$$

| Ecology | Multi-Task VLA |
|---------|---------------|
| Species $i$ | Task $i$ |
| Abundance $x_i$ | Task $i$'s performance (success rate) |
| Growth rate $r_i$ | Single-task learning rate / baseline performance |
| $\alpha_{ij} > 0$ | Task $j$ hurts task $i$ (negative transfer / competition) |
| $\alpha_{ij} < 0$ | Task $j$ helps task $i$ (positive transfer / mutualism) |
| Carrying capacity $K_i$ | Maximum achievable performance under resource constraints |
| Resource (niche) | Shared model capacity (parameters, representation bandwidth) |

This is not a surface analogy. The mathematical structure is identical: a system of $N$ interacting entities whose "fitness" (performance) depends on the composition and relative abundances (data mixing proportions) of all other entities, mediated through shared limited resources (model capacity). Recent ecological theory (Tonolo et al., arXiv 2503.20887, 2025; Giraldo & Lee, arXiv 2406.06897, 2024) studies exactly how the structure of the interaction matrix $\alpha$ determines system stability, feasibility, and species persistence -- directly analogous to asking which tasks can coexist in a multi-task model and which will be "excluded."

### Key Ecological Principles That Transfer

**1. The Competitive Exclusion Principle (Gause's Law)**: Two species competing for the exact same niche cannot stably coexist -- one will drive the other to extinction. In multi-task VLA terms: *two tasks that require mutually exclusive representations in the same parameter subspace cannot both achieve optimal performance*. This predicts that the influence matrix will show strong negative entries specifically between tasks that demand contradictory use of the same representational resources (e.g., same visual scene but opposite actions). The ecological resolution -- niche differentiation -- maps directly to the idea of giving conflicting tasks separate parameter subspaces (cf. CORAL's per-task LoRA).

**2. Mutualism and Facilitation**: In ecology, some species *facilitate* others by modifying the environment (e.g., nitrogen-fixing plants enriching soil for neighbors). Similarly, some tasks may improve shared representations that benefit other tasks. The ecological literature provides a taxonomy of interaction types (mutualism, commensalism, amensalism, parasitism, competition) that is richer than the binary "positive/negative transfer" typically used in MTL. We propose adopting this full taxonomy for VLA task interactions.

**3. Community Stability Analysis via Random Matrix Theory**: May's theorem (1972) showed that large random interaction matrices become unstable beyond a complexity threshold $\sigma \sqrt{NC} > 1$, where $\sigma$ is interaction strength variance, $N$ is species count, and $C$ is connectance. Stone (arXiv 1607.01879) extended this to structured networks using a Google-matrix reduction. **Transplant prediction**: As the number of VLA tasks grows, there exists a critical threshold beyond which the multi-task system becomes "unstable" (performance collapses). This threshold depends on the variance of cross-task interactions and the fraction of task pairs that interact -- directly testable by varying the number of tasks in multi-task training.

**4. Sparse Interaction Networks**: Tonolo et al. (2025) show that sparse (not fully connected) ecological networks exhibit fundamentally different dynamics from dense ones -- including non-Gaussian abundance distributions and a novel "topological glass" phase. For VLA: the influence matrix is likely *sparse* (most task pairs have negligible interaction), and this sparsity structure is critical. Dense interaction assumptions (as in gradient conflict methods that consider all task pairs) may be fundamentally misleading.

### Concrete Method: Ecological Stability Analysis of the Task Influence Matrix

1. **Compute $M_{ij}$ as a community matrix**: Use the task-to-task influence scores as entries in a gLV interaction matrix.
2. **Eigenvalue analysis**: The stability of the multi-task "community" is determined by the leading eigenvalue of $M$. If all eigenvalues have negative real parts, the system is stable (all tasks can coexist). Positive eigenvalues indicate instability -- the corresponding eigenvector identifies the "axis of conflict."
3. **Feasibility analysis**: Even if stable, the equilibrium may be infeasible (some tasks have negative "abundance" = performance below threshold). The fraction of feasible equilibria in the influence matrix predicts how many tasks can realistically coexist.
4. **Optimal "carrying capacity" allocation**: In ecology, adjusting resource availability changes which species persist. Analogously, adjusting data mixing proportions $w_i$ changes the effective carrying capacity for each task. The gLV framework provides closed-form solutions for the equilibrium abundances given the interaction matrix and carrying capacities, enabling *direct optimization of mixing weights*.

### References
- Tonolo et al. (arXiv 2503.20887, 2025): "Generalized Lotka-Volterra model with sparse interactions: non-Gaussian effects and topological multiple-equilibria phase"
- Giraldo & Lee (arXiv 2406.06897, 2024): "Bifurcations and multistability in empirical mutualistic networks"
- Stone (arXiv 1607.01879): "Determinants of Structural Stability in Complex Ecological and Biological Networks: the Google Matrix Approach"
- Samadder et al. (arXiv 2201.03193, 2022): "Interconnection between density-regulation and stability in competitive ecological network"
- Ding et al. (arXiv 2405.17420, 2024): "Survival of the Fittest Representation: A Case Study with Modular Addition" -- **directly uses Lotka-Volterra equations to model competing representations in neural networks**
- Fried et al. (arXiv 1703.00940, 2017): "Alternative steady states in random ecological networks"
- "Impossible ecologies: Interaction networks and stability of coexistence" (Cell Systems, 2025)

---

## Analogy 2: Statistical Physics -- Spin Glass Frustration and the Multi-Task Optimization Landscape

### The Structural Correspondence

The problem of multi-task gradient conflicts maps onto the physics of **frustrated spin systems** (spin glasses). In a spin glass, each spin wants to align with its neighbors, but the coupling constants $J_{ij}$ have mixed signs -- some pairs prefer parallel alignment, others prefer anti-parallel. When these constraints form closed loops with an odd number of negative couplings, *frustration* arises: no single spin configuration can simultaneously satisfy all pairwise preferences.

| Spin Glass | Multi-Task VLA |
|-----------|---------------|
| Spin $s_i \in \{+1, -1\}$ | Gradient direction of task $i$ at shared parameter $k$ |
| Coupling $J_{ij}$ | Cross-task gradient alignment: $J_{ij} = \cos(\nabla L_i, \nabla L_j)$ |
| Frustration (odd loop) | Three tasks where A helps B, B helps C, but C hurts A |
| Ground state degeneracy | Multiple Pareto-optimal solutions with different task trade-offs |
| Rugged energy landscape | Multi-task loss landscape with many local minima |
| Temperature $T$ | Learning rate / noise level |
| Annealing schedule | Learning rate schedule |

The key insight from spin glass physics is that frustration is a **topological** property of the interaction network, not just a pairwise one. A frustrated loop of three tasks (A helps B, B helps C, C hurts A) creates fundamentally harder optimization than three independent pairwise conflicts. The "misfit parameter" (Kobe & Klotz, cond-mat/9505020) quantifies the degree of frustration by measuring the ground-state energy increase due to frustrated loops compared to an unfrustrated reference state.

### What Spin Glass Theory Predicts for Multi-Task VLA

**1. Frustration Diagnosis Beyond Pairwise Analysis**: The influence matrix $M_{ij}$ alone is insufficient to predict optimization difficulty. What matters is the *frustration structure* -- specifically, the signed loops in the interaction graph. We propose computing the **frustration index** of the task influence graph:

$$f = 1 - \frac{E_{\text{ground}}}{E_{\text{unfrustrated}}}$$

where $E_{\text{ground}}$ is the actual multi-task loss and $E_{\text{unfrustrated}}$ is the loss achievable if all task interactions were simultaneously satisfiable. High $f$ indicates fundamental multi-task incompatibility that cannot be resolved by simple gradient manipulation (PCGrad, CAGrad) -- only architectural changes (parameter separation) or data mixing changes can help.

**2. Phase Transitions in Multi-Task Scaling**: Spin glass theory predicts sharp phase transitions as the density of frustrated interactions increases. In VLA terms: there should exist a critical number of tasks $N_c$ beyond which the multi-task optimization landscape transitions from having a few smooth basins (easy optimization) to an exponentially rugged landscape (hard optimization). The review by Tahriri et al. (arXiv 2512.19818, 2025) connects this directly to the notion that spin glasses exhibit "slow, history-dependent dynamics" -- analogous to the training-order sensitivity observed in multi-task learning.

**3. Simulated Annealing as Curriculum Learning**: The optimal strategy for navigating a frustrated landscape is simulated annealing -- start at high temperature (high learning rate, uniform mixing) and gradually cool (lower learning rate, optimized mixing). This provides a physics-principled justification for curriculum learning in multi-task VLA: start with uniform data mixing, measure the frustration structure, then gradually adjust mixing weights as training progresses.

**4. Ground State Degeneracy as Pareto Front**: In frustrated systems, the ground state is highly degenerate -- many configurations have near-equal energy. This maps to the multi-task Pareto front: there are many data mixing strategies that achieve similar total performance but with different task-level trade-offs. The spin glass perspective suggests that enumerating these degenerate solutions (e.g., via replica methods) could map the full Pareto front efficiently.

### Concrete Method: Frustration-Aware Task Grouping

1. **Construct the signed task interaction graph** from the influence matrix $M_{ij}$ (positive edges for positive transfer, negative for negative).
2. **Compute the frustration index** by finding the minimum number of edges that must be removed to make the graph balanced (no frustrated loops). This is the graph frustration index, computable via spectral methods on the signed Laplacian.
3. **Partition tasks into frustrated and unfrustrated clusters**: Tasks within an unfrustrated subgraph can be safely co-trained; frustrated loops require special handling (per-task LoRA, data separation, or curriculum scheduling).
4. **Apply annealing-inspired curriculum**: For frustrated task groups, use a temperature schedule that starts with equal mixing and gradually biases toward the less-frustrated task pairs.

### References
- Tahriri et al. (arXiv 2512.19818, 2025): "Spin Glasses: Disorder, Frustration, and Nonequilibrium Complexity"
- Kobe & Klotz (cond-mat/9505020, 1995): "Frustration -- how it can be measured"
- Seyed-allaei et al. (arXiv 0710.5403): "The energy landscape networks of spin-glasses"
- Samarakoon et al. (arXiv 1707.03086, 2017): "Aging, memory, and nonhierarchical energy landscape of spin jam"
- Dauphin et al. (2014): "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization" (loss surface = energy landscape analogy)
- Choromanska et al. (2015): "The Loss Surfaces of Multilayer Networks" (spin glass connection to neural network loss)

---

## Analogy 3: Immunology -- Original Antigenic Sin and Cross-Reactive Interference

### The Structural Correspondence

The immune system faces a problem structurally identical to multi-task learning: it must maintain effective responses (memory T/B cells) against many different pathogens (tasks) simultaneously, using a shared pool of adaptive immune cells with limited repertoire diversity (model capacity). The phenomenon of **Original Antigenic Sin (OAS)** -- where prior immune memory to one pathogen *interferes with* the response to a related but distinct pathogen -- is a near-perfect biological analogue of negative transfer.

| Immunology | Multi-Task VLA |
|-----------|---------------|
| Pathogen / antigen $i$ | Task $i$ |
| Memory B/T cells | Learned representations / LoRA parameters |
| Immune repertoire diversity | Model capacity (parameter count) |
| Cross-reactivity | Shared sub-skill representations |
| Original Antigenic Sin | Negative transfer from previously learned tasks |
| Affinity maturation | Fine-tuning / gradient descent on task loss |
| Antibody sequence space | Parameter space |

### The Deep Structural Insight

Deem & Lee (arXiv cond-mat/0308613, 2003) explained OAS as **localization in antibody sequence space**: when the immune system encounters a pathogen similar to one it has seen before, it preferentially activates memory cells tuned to the *old* pathogen rather than generating *new* cells optimally tuned to the current one. This is because the fitness landscape (antibody affinity as a function of sequence) is *rugged* -- memory cells sit in a local optimum for the old pathogen, and the gradient toward the new pathogen's optimum requires crossing a fitness valley.

The direct ML analogy: when a VLA model has been trained on task A and then encounters task B (which shares visual features but requires different actions), the model is "localized" in a parameter region optimized for task A. If tasks A and B are sufficiently similar that they share the same representation subspace but different enough that they require different action mappings, the model gets trapped -- exactly as in OAS. The immune system's solution is informative:

**1. Repertoire Diversification**: The immune system maintains a *diverse* repertoire of cells with varying specificities, ensuring that some cells are always available for novel pathogens. ML transplant: maintain a **diverse ensemble of LoRA adapters** with different initializations, so that when a new task conflicts with an existing one, there is always an adapter available that is not "locked" to the conflicting task.

**2. Germinal Center Dynamics**: The immune system uses germinal centers -- specialized microenvironments where B cells undergo rapid mutation and selection. Cells that cross-react beneficially are amplified; cells that cross-react detrimentally are eliminated. ML transplant: use a **"germinal center" selection phase** after computing the influence matrix -- amplify training data from tasks with beneficial cross-reactivity, suppress data from tasks with detrimental cross-reactivity. This is essentially what influence-guided data mixing does, but the immunological framing reveals an additional insight: the selection should be *iterative* (multiple rounds of mutation + selection), not one-shot.

**3. The Threshold Model of Cross-Reactivity**: Chen et al. (arXiv 2512.02730, 2025) formalize immune cross-reactivity using a conserved quantity from Noether's theorem -- the "immune capacity" $I$ is conserved under translations in antigenic structure space. In multi-task terms: there is a conserved quantity representing total learning capacity, and cross-task influence is governed by how much of this capacity each task consumes. If task B consumes capacity that task A needs, negative transfer results. The conservation law predicts that *improving one task necessarily degrades another unless total capacity increases* -- a form of the multi-task learning impossibility theorem.

### Concrete Method: Immune-Inspired Influence Diagnosis

1. **Cross-reactivity mapping**: For each task pair $(i, j)$, measure the "antigenic distance" in representation space -- how similar are the visual features that activate the model for tasks $i$ and $j$? High visual similarity + divergent action requirements = high OAS risk.
2. **Affinity landscape roughness**: Compute the curvature of the loss surface along the direction connecting task $i$'s and task $j$'s optimal parameters. Rough landscape (many local minima) predicts OAS-like interference; smooth landscape predicts beneficial transfer.
3. **Iterative repertoire selection**: Instead of computing the influence matrix once and fixing the data mixture, iterate:
   - Train for $k$ steps with current mixture
   - Recompute local influence scores
   - Adjust mixture to amplify beneficial and suppress detrimental cross-task interactions
   - Repeat (analogous to affinity maturation rounds)
4. **Capacity conservation diagnostic**: Track total multi-task performance $\sum_i \text{perf}_i$ over training. If this sum is approximately conserved (improvements in some tasks offset by degradation in others), the system is capacity-limited and architectural expansion (more parameters) is needed rather than data mixing optimization.

### References
- Deem & Lee (arXiv cond-mat/0308613, 2003): "Sequence Space Localization in the Immune System Response to Vaccination and Disease"
- Chen et al. (arXiv 2512.02730, 2025): "Invariance under Structure Translation as the Origin of Host Immune Capacity Conservation from Noether's Theorem"
- Frontiers in Immunology (2024): "Editorial: Quantification and prediction of T-cell cross-reactivity through experimental and computational methods"
- Nature Methods (2025): "Assessment of computational methods in predicting TCR-epitope binding recognition"

---

## Analogy 4: Neuroscience -- Lateral Inhibition, Cortical Columns, and Multi-Population Dynamics

### The Structural Correspondence

The brain faces the same multi-task challenge: multiple neural populations in the cortex process different types of information (analogous to different tasks), sharing limited neural substrate (analogous to shared model parameters). The brain's solution is **lateral inhibition** combined with **cortical column organization** -- populations that process related information excite each other, while populations that process conflicting information inhibit each other.

| Cortical Dynamics | Multi-Task VLA |
|------------------|---------------|
| Neural population $i$ | Task $i$'s gradient signal |
| Excitatory coupling $W_{ij} > 0$ | Positive transfer (aligned gradients) |
| Inhibitory coupling $W_{ij} < 0$ | Negative transfer (conflicting gradients) |
| Winner-take-all (WTA) | Dominant task monopolizing shared representations |
| Lateral inhibition | Mechanism by which one task suppresses another |
| Cortical column | Task-specific parameter subspace |
| Population dynamics | Training dynamics of multi-task model |

### Key Neuroscience Principles That Transfer

**1. Balanced Excitation-Inhibition**: Healthy cortical circuits maintain a precise balance between excitation and inhibition (E/I balance). Disruption leads to pathological states -- too much excitation causes seizures (analogous to one task dominating and destroying all others), too much inhibition causes silence (analogous to catastrophic forgetting). Schwalger et al. (arXiv 1611.00294) derive mesoscopic population equations showing how this balance emerges from microscopic neuron properties. **Transplant**: define an E/I balance metric for the influence matrix as the ratio of total positive to total negative transfer. Predict that optimal multi-task performance occurs at a specific E/I ratio, not at maximum positive transfer.

**2. Winner-Take-All as Failure Mode**: In cortical circuits, strong lateral inhibition can cause winner-take-all dynamics where one population suppresses all others. This is the neural analogue of one task dominating the shared representation. The cortical solution is **soft winner-take-all** with graded inhibition, allowing multiple populations to coexist at reduced activity. **Transplant**: instead of hard task grouping (either co-train or separate), use graded data mixing weights that implement soft competition -- tasks that conflict are not fully separated but have their influence attenuated proportionally to their conflict strength.

**3. Multi-Region Communication and Disentanglement**: Liu et al. (arXiv 2506.19094, 2025; MR-LFADS) and Xin & Kass (arXiv 2506.02263, 2025; GLM-Transformer) develop methods to disentangle *inter-regional communication* from *local population dynamics* and *unobserved inputs* in multi-region neural recordings. The key insight is that observed co-variation between brain regions conflates three sources: (a) direct causal influence, (b) shared input from a common source, and (c) independent local dynamics. **This is directly relevant to VLA task influence analysis**: observed co-variation between task performances can arise from (a) genuine cross-task data influence, (b) shared dependency on common sub-skills, or (c) independent training dynamics. Methods that fail to disentangle these sources will produce misleading influence matrices.

**4. Timescale Separation**: Maran et al. (arXiv 2506.19800, 2025) show that non-local cortical projections dominate on fast timescales (<30ms) but contribute little to slow spontaneous dynamics. **Transplant prediction**: cross-task influence may operate on different timescales -- early in training, tasks interact strongly through shared representations (fast dynamics); later, as representations specialize, tasks become more independent (slow dynamics). The influence matrix should be computed at multiple training stages, not just at convergence.

### Concrete Method: Neural Population Dynamics Model of Task Interactions

1. **Model task performance dynamics** as a coupled dynamical system inspired by Wilson-Cowan or neural mass models:
   $$\frac{d\text{perf}_i}{dt} = -\text{perf}_i + f\left(\sum_j W_{ij} \cdot \text{perf}_j + I_i(t)\right)$$
   where $W_{ij}$ is the influence matrix, $I_i(t)$ is the external input (data for task $i$), and $f$ is a nonlinear activation (sigmoid).
2. **Fit $W_{ij}$ from training trajectories**: Record per-task validation performance at each epoch. Fit the coupled dynamical system to extract the interaction weights. This provides an *alternative method* for estimating the influence matrix that does not require gradient computations -- only per-task performance trajectories.
3. **Stability analysis**: Compute the eigenvalues of the fitted $W$ matrix. Predict which tasks will eventually be "suppressed" (analogous to WTA dynamics) and which will coexist.
4. **Design intervention**: If a pathological attractor is detected (one task dominating), inject targeted "inhibition" (reduce that task's data proportion) to restore balance -- analogous to pharmacological intervention in epilepsy.

### References
- Schwalger et al. (arXiv 1611.00294, 2016): "Towards a theory of cortical columns: From spiking neurons to interacting neural populations of finite size"
- Liu et al. (arXiv 2506.19094, 2025): "Accurate identification of communication between multiple interacting neural populations" (MR-LFADS)
- Xin & Kass (arXiv 2506.02263, 2025): "Identifying interactions across brain areas while accounting for individual-neuron dynamics with a Transformer-based variational autoencoder"
- Maran et al. (arXiv 2506.19800, 2025): "Modeling the influences of non-local connectomic projections on geometrically constrained cortical dynamics"
- Rosch et al. (arXiv 2309.05939, 2023): "Spontaneous brain activity emerges from pairwise interactions in the larval zebrafish brain"

---

## Synthesis: A Unified Cross-Disciplinary Framework

The four analogies converge on a unified insight: the task-to-task influence matrix should not be treated as a static diagnostic tool, but as the **interaction matrix of a dynamical system** whose behavior is governed by well-understood principles from ecology, physics, immunology, and neuroscience.

### Proposed Framework: Ecological-Physical Task Interaction Analysis (EPTIA)

| Component | Ecology Contribution | Physics Contribution | Immunology Contribution | Neuroscience Contribution |
|-----------|---------------------|---------------------|------------------------|--------------------------|
| Influence matrix construction | gLV interaction coefficients from performance perturbation experiments | Signed graph from gradient inner products | Cross-reactivity map from representation similarity | Dynamical system fitting from training trajectories |
| Structural analysis | Community stability (eigenvalue analysis), competitive exclusion detection | Frustration index, loop analysis, ground state degeneracy | Antigenic distance landscape, localization detection | E/I balance, WTA detection, timescale separation |
| Optimization | Equilibrium abundance optimization in gLV = data mixing | Annealing schedule for frustrated systems = curriculum | Iterative repertoire selection = online mixing adaptation | Soft WTA inhibition = graded attenuation |
| Scaling predictions | May's complexity-stability threshold | Phase transition at critical frustration density | Capacity conservation law | Timescale-dependent interaction strength |

### Concrete Experimental Plan

**Phase 1: Pilot (10-15 min per experiment)**
- Train a small policy (Qwen-0.5B LoRA) on 5 LIBERO tasks
- Compute the 5x5 influence matrix using LESS-style gradient projections
- Analyze the matrix using all four lenses:
  - (Ecology) Eigenvalue stability analysis + competitive exclusion check
  - (Physics) Frustration index of the signed task graph
  - (Immunology) Cross-reactivity vs. antigenic distance scatter plot
  - (Neuroscience) Fit Wilson-Cowan dynamics to per-task performance curves

**Phase 2: Validation (30 min)**
- Counterfactual test: remove data for the task pair predicted to have strongest negative transfer, verify performance improvement on the "victim" task
- Scaling test: increase to 10 tasks, check whether May's stability threshold predicts the onset of performance degradation
- Frustration test: verify that task triplets with high frustration index are harder to co-train than those with low frustration index

**Phase 3: Influence-Guided Mixing (30 min)**
- Use the gLV equilibrium equations to compute optimal data mixing weights
- Compare against uniform mixing and DoReMi/Re-Mix baselines
- Track per-task performance trajectories and verify that the dynamical system model predicts the observed dynamics

**Computational Cost**: Total pilot budget ~1.5 GPU-hours on a single A6000 (5 tasks x ~500 trajectories x 5 LoRA fine-tuning runs for LOO validation + gradient computation).

**Success Probability**: 55%

**Key Risk**: The ecological/physical analogies assume that the influence matrix is approximately static (or slowly varying), but in practice the matrix may change substantially during training as representations evolve. The immunological analogy (iterative repertoire selection) addresses this by advocating for online re-estimation, but this increases computational cost.

### Why This Is Novel Beyond Existing Cross-Disciplinary Work

- **Ding et al. (2024)** used Lotka-Volterra for *representation competition within a single task* (modular addition). We propose using it for *cross-task competition in multi-task learning* -- a fundamentally different level of analysis.
- **Spin glass analogies to neural networks** (Choromanska et al., 2015) focused on the loss surface of a *single task*. We apply frustration theory to the *multi-task interaction structure* -- the frustration is between tasks, not within a single loss landscape.
- **No prior work** has applied immunological OAS theory to explain negative transfer in multi-task learning, despite the structural correspondence being remarkably precise.
- **Neuroscience-inspired MTL** (e.g., lateral inhibition in SNNs) has focused on architectural design. We propose using neural population dynamics *as an analysis tool* to extract the influence matrix from training trajectories without gradient computation.

### Testable Predictions

1. **Ecological prediction**: The multi-task system will show May-type instability when $\sigma(M) \cdot \sqrt{N \cdot C} > 1$, where $\sigma(M)$ is the standard deviation of influence scores, $N$ is task count, and $C$ is the fraction of non-zero task interactions. This can be tested by systematically increasing $N$.

2. **Physics prediction**: Task triplets with high frustration index (measured by the signed graph's frustrated loop count) will show worse joint performance than task triplets with the same pairwise conflict magnitudes but low frustration index. This disentangles pairwise from higher-order effects.

3. **Immunological prediction**: Tasks that share similar visual observations but require different actions (high cross-reactivity, large antigenic distance) will show the strongest negative transfer -- stronger than tasks with dissimilar observations (low cross-reactivity). Furthermore, the "localization" effect predicts that negative transfer will be asymmetric: the first-learned task will interfere with later tasks more than vice versa.

4. **Neuroscience prediction**: The influence matrix estimated from gradient inner products (instantaneous measure) will differ systematically from the matrix estimated by fitting dynamical models to performance trajectories (temporal measure), because the latter captures higher-order and delayed effects invisible to gradient analysis. Specifically, the dynamical model will reveal oscillatory task interactions (task A helps B early, hurts B late) invisible to a single-time-point gradient analysis.
