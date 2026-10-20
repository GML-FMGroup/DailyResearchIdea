# Joint Constraint Algebra for Composable Procedural Worlds

## Motivation

InfiniDream's per-asset symbolic constraints do not compose across heterogeneous procedural algorithms, so composed worlds may violate individual asset constraints; this residual gap persists across Set-the-scene, CityCraft, and SceneDreamer because no branch provides a constraint composition operator that preserves both local editability and global layout consistency.

## Key Insight

A joint global-local constraint algebra with a differentiable solver that treats asset parameters as variables and scene graph relations as differentiable constraints yields a consistent parameter assignment that preserves both local symbolic constraints and global layout, because the solver's fixed-point iteration enforces all constraints simultaneously rather than sequentially.

## Method

**What it is:** We introduce the Joint Constraint Algebra (JCA), a constraint composition operator that takes per-asset symbolic constraints (from InfiniDream's symbolically constrained asset generation) and a global scene graph, and outputs a consistent assignment of asset parameters that satisfies both local and global constraints while remaining editable. JCA consists of a constraint encoding that unifies local symbolic constraints and global layout constraints into a single differentiable objective, and a differentiable constraint solver that iteratively refines asset parameters via gradient descent to reach a fixed point.

**How it works:**
```pseudocode
Input: Assets A = {a_i} with symbolic constraints C_i^local (e.g., from InfiniDream), scene graph G = (V, E) with layout constraints C^global (e.g., non-overlap, alignment), initial parameters θ_i^0.
Output: Consistent parameters θ_i^*.

1. Encode constraints:
   For each asset a_i, convert C_i^local into differentiable penalty functions f_i^local(θ_i) = max(0, g_i(θ_i))^2, where g_i(θ_i) is the signed violation of the local constraint (e.g., scale bounds, aspect ratio, orientation limits).
   For each edge (i,j) in G, convert C^global into differentiable penalty f_{ij}^global(θ_i, θ_j) = max(0, h_{ij}(θ_i, θ_j))^2, where h_{ij} is the signed violation of the layout constraint (e.g., overlap distance, alignment offset).
   Total loss L(θ) = Σ_i f_i^local(θ_i) + λ Σ_{(i,j)∈E} f_{ij}^global(θ_i, θ_j), with λ=1.0.

2. Differentiable solver:
   Initialize θ_i = θ_i^0 (from InfiniDream's placement).
   For t = 1 to T (T=100):
     Compute gradient ∇_θ L via automatic differentiation.
     Update θ_i ← θ_i - η ∇_{θ_i} L, with η=0.01 (Adam optimizer).
     If max_i ||θ_i^{t} - θ_i^{t-1}|| < ε (ε=1e-4), break.
   Return θ_i^*.

3. Editability preservation:
   After solving, project θ_i^* back to the nearest point in the symbolic constraint set C_i^local (if any residual violation) using a projection layer: θ_i^proj = argmin_{θ ∈ C_i^local} ||θ - θ_i^*||_2, computed via a differentiable projection (e.g., clipping for box constraints, closed-form for linear constraints). This ensures local constraints are exactly satisfied.
```

**Why this design:** We chose a differentiable penalty-based solver over a discrete constraint satisfaction solver (e.g., SAT) because procedural asset parameters are continuous (e.g., scale, position, rotation) and gradient-based optimization can handle large scenes efficiently, accepting the cost that the solver may converge to local minima. We chose a single joint loss over sequential constraint enforcement (local then global) because sequential enforcement can violate earlier constraints when later ones are applied, whereas the joint loss balances all constraints simultaneously, accepting the cost that trade-offs between conflicting constraints require careful weighting (we set λ=1.0 and rely on the projection step to enforce local constraints exactly). We chose to keep the symbolic constraints as differentiable penalties rather than hard constraints because hard constraints would make the optimization non-differentiable and incompatible with gradient-based solvers, accepting the cost that the solver may produce small violations that the projection step must correct. These decisions ensure that JCA is not a variant of retrieval-based constraint matching (e.g., nearest-neighbor over constraint libraries) because it computes a joint parameter assignment via optimization, not a lookup; nor is it a two-module controller because the local and global constraints are fused into a single loss and solved jointly.

**Why it measures what we claim:** The total loss L(θ) measures joint constraint satisfaction because the penalty functions f_i^local and f_{ij}^global are designed to be zero exactly when the corresponding symbolic or layout constraint is satisfied; this assumption fails when constraints are infeasible (e.g., conflicting local and global constraints), in which case L(θ) reflects the minimal violation rather than full satisfaction. The gradient-based update measures progress toward a consistent assignment because the gradient of L points in the direction of steepest constraint violation reduction; this assumption fails when the loss landscape is non-convex and the solver gets stuck in a local minimum, in which case the gradient reflects local rather than global consistency. The projection step measures editability preservation because projecting onto the symbolic constraint set ensures local constraints are exactly satisfied; this assumption fails when the projection is not unique (e.g., multiple nearest points), in which case the projection may choose an arbitrary valid point that changes the asset's intended semantics. Together, these quantities operationalize the motivation-level concept of joint global-local consistency.

## Contribution

['A novel joint constraint algebra (JCA) that unifies per-asset symbolic constraints and scene-level layout constraints into a single differentiable objective, enabling consistent parameter assignment across heterogeneous procedural assets.', 'A differentiable constraint solver that iteratively refines asset parameters to satisfy both local and global constraints, with a projection step that preserves local editability.', 'An empirical demonstration that JCA resolves the residual gap in InfiniDream by composing assets from different procedural algorithms without violating individual constraints, as measured by constraint satisfaction rates and editability metrics.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | InfiniDream 100-object / 20-scene tasks | Matches source of local symbolic constraints |
| Primary metric | Joint constraint violation rate | Directly measures global-local consistency claim |
| Baseline | Sequential local-then-global enforcement | Tests whether joint loss beats staged enforcement |
| Baseline | Retrieval-based constraint matching | Tests whether optimization beats lookup |
| Baseline | Unconstrained InfiniDream placement | Tests whether constraint solving adds anything |
| Baseline | Generic penalty solver (same loss, no projection) | Tests whether projection step adds value |
| Ablation-of-ours | JCA without projection step | Isolates editability/exact-satisfaction contribution |

### Why this setup validates the claim

The central claim is that fusing local symbolic and global layout constraints into one differentiable objective yields more jointly consistent, still-editable scenes than alternatives. The sequential baseline tests the sub-claim that joint solving avoids the re-violation that staged enforcement causes; the retrieval baseline tests the sub-claim that optimization produces assignments lookup cannot; the unconstrained baseline establishes that any gain is attributable to constraint solving rather than to InfiniDream's underlying generation. The generic penalty solver baseline (identical loss and optimizer but no projection) isolates the contribution of the projection step to exact local-constraint satisfaction. The projection ablation tests whether exact local-constraint satisfaction is what preserves editability, or whether the gradient solver alone suffices. Joint constraint violation rate is the right primary metric because it is exactly the quantity the loss is designed to drive to zero, and it is computed on held-out constraint checks rather than on the training loss itself, so a method that merely minimizes its own objective without satisfying constraints cannot score well. Together these form a falsifiable test: the claim predicts a specific ordering of violation rates concentrated on scenes where local and global constraints interact.

### Expected outcome and causal chain

**vs. Sequential local-then-global enforcement** — On a scene where an asset's local constraint fixes its scale near a wall while the global non-overlap edge pushes it away, the sequential baseline satisfies local first then moves the asset to fix overlap, re-violating the local scale constraint because it never revisits the earlier stage. Our method instead balances both penalties in one gradient step, so the asset settles at a compromise that the projection step then snaps exactly onto the local set. We expect a noticeable gap on scenes with many local-global interactions and near-parity on scenes where constraints barely couple.

**vs. Retrieval-based constraint matching** — On a case requiring a novel combination of local symbolic constraints and a dense layout graph, the retrieval baseline returns the nearest stored assignment, which satisfies its own library entry but violates the current graph's edges because no stored scene matches this exact constraint bundle. Our method instead computes a fresh assignment by descending the joint loss for this specific instance, so it can satisfy constraint combinations absent from any library. We expect the retrieval baseline to degrade sharply as scenes grow denser while our method stays comparatively flat.

**vs. Unconstrained InfiniDream placement** — On a scene with tight alignment and non-overlap edges, the unconstrained baseline places assets plausibly per-asset but ignores pairwise layout, producing overlaps and misalignments because no global objective exists. Our method adds exactly that global penalty and refines placement, so we expect a large violation-rate gap on layout-heavy scenes and small gaps on sparse ones — the signature that the gain comes from global constraint handling.

**vs. Generic penalty solver (same loss, no projection)** — This baseline uses the identical joint loss and Adam optimizer but omits the projection step. On scenes where the joint loss leaves small residual local violations, the generic solver will have higher local violation rates, while our method's projection drives them to zero. We expect our method to outperform on local violation rate, with similar global violation rates, confirming the projection step's role in exact local satisfaction.

### What would falsify this idea

If our gain over the sequential and retrieval baselines were uniform across all scenes rather than concentrated where local-global constraints interact, or if removing the projection step left editability unchanged, the claim that joint differentiable solving is the causal mechanism would be wrong.

## References

1. InfiniDream: Open-Ended Photorealistic 3D Worlds Through Procedural Generation
