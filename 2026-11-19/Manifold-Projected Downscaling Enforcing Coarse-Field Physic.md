# Manifold-Projected Downscaling: Enforcing Coarse-Field Physical Consistency via Differentiable Closure after Residual Heads

## Motivation

Downscaling frameworks such as PROLA, SGD, and OneForecast all learn a mapping from coarse to high-resolution fields and implicitly assume that the learned output respects mass conservation and dynamical balance, yet none of them encodes such constraints: PROLA's residual heads (E3, E6) produce high-resolution corrections with no mechanism tying the fine field's column-integrated mass or geostrophic residual back to the coarse conditioning state, and SGD's diffusion sampler (E2, E5) likewise generates states whose agreement with the coarse field is only statistical. The root cause is structural: the residual/diffusion output is parameterized in an unconstrained function space, so any conservation law must be satisfied by accident of the training distribution rather than by construction, and distribution shift (extreme events, unseen regions) breaks the accident. This is a forest-level gap: every branch of the downscaling literature treats physical consistency as a property to be learned rather than an invariant to be projected onto.

## Key Insight

Physical consistency between coarse and fine fields is an invariant of a low-dimensional constraint manifold, so a differentiable projection onto that manifold applied after any residual head makes conservation and balance hold by construction rather than by distributional accident, and the projection's residual norm is itself a certified measure of how far the learned operator strayed from the coarse model's physical manifold.

## Method

**What it is.** Manifold-Projected Downscaling (MPD) is a post-head closure layer that takes the coarse conditioning field `x_c`, the backbone+residual-head high-resolution prediction `x_h`, and a fixed numerical scaffold `S` (the same scaffold family PROLA uses), and returns a corrected high-resolution field `x_h*` that lies on the coarse model's physical manifold: column mass conserved to the coarse field, and geostrophic residual bounded. It is a differentiable projection, so gradients flow back through PROLA's residual heads during training.

**Load-bearing assumption (stated explicitly).** MPD assumes that the coarse conditioning field's own column-integrated mass and geostrophic residual constitute a physically valid manifold target, so that projecting the high-resolution prediction onto consistency with the coarse field is equivalent to enforcing true physical consistency rather than propagating the coarse model's own mass and balance errors into the fine field. This assumption is load-bearing: if the coarse field is itself out of balance (as reanalysis products like ERA5 are, with budget residuals closed only through analysis increments and parameterized physics), the projection can drive the high-resolution field toward a physically wrong invariant. To make the assumption explicit and testable, MPD defines the manifold target not as the raw coarse field but as a **budget-closed reference** `x_c^bc` obtained by adding an explicit source term `R_c` to the coarse column-mass and momentum constraints, where `R_c` is the analysis-increment/parameterized-physics residual estimated from the coarse dataset's own budget closure (e.g., the difference between the coarse field's column integral and the time-tendency-plus-flux-divergence budget). The geostrophic term is reweighted by the local Rossby number `Ro = |u|/(f L)` so that the projection targets a balanced subspace only where balance is defined (`Ro < 1`), and is set to zero in the tropics where `f → 0`. The method reports a **discretization-error floor** for `consistency_score` computed on the coarse field itself, so that any score above this floor is attributable to the learned operator rather than to scaffold error.

**How it works.**

```pseudocode
Input: coarse field x_c (H_c x W_c x V), raw high-res prediction x_h (H_h x W_h x V),
       scaffold S (interpolation + finite-difference operators),
       budget-closure residual R_c (H_c x W_c x V) from coarse dataset,
       tolerance eps_m, eps_g
Hyperparams: lambda_m = 1.0 (mass weight), lambda_g = 0.1 (geostrophic weight),
             K = 3 projection iterations, rho = 0.5 (geostrophic relaxation),
             Ro_thresh = 1.0 (Rossby-number cutoff for balance projection)

# Step 1: budget-closed coarse-manifold targets from the conditioning field
x_c_bc = x_c + R_c                         # apply explicit source term to close budget
m_c    = ColumnIntegral(x_c_bc)             # per-column mass, shape (H_c, W_c)
g_c    = GeostrophicResidual(x_c_bc, S)     # per-gridpoint balance residual
Ro_c   = RossbyNumber(x_c_bc, S)            # local Rossby number, shape (H_c, W_c)
w_g    = 1.0 if Ro_c < Ro_thresh else 0.0   # balance weight mask

# Step 2: differentiable projection loop (fixed-point, K iterations)
x = x_h
for k in 1..K:
    m_x = ColumnIntegral(x)                 # (H_h, W_h)
    r_m = Downsample(m_x) - m_c             # coarse-grid mass mismatch
    x   = x - lambda_m * S.adjoint( Upsample(r_m) )   # distribute correction via scaffold adjoint
    g_x = GeostrophicResidual(x, S)         # (H_h, W_h)
    r_g = (g_x - Upsample(g_c)) * Upsample(w_g)  # mask by Rossby-number validity
    x   = x - lambda_g * rho * S.balance_adjoint(r_g) # project residual onto balanced subspace

# Step 3: certification scalar (returned alongside field)
consistency_score = ||Downsample(ColumnIntegral(x)) - m_c||_2 + ||(GeostrophicResidual(x,S) - Upsample(g_c)) * Upsample(w_g)||_2
floor_score = ||Downsample(ColumnIntegral(x_c_bc)) - m_c||_2 + ||(GeostrophicResidual(x_c_bc,S) - Upsample(g_c)) * Upsample(w_g)||_2  # discretization-error floor

Output: x_h* = x, consistency_score, floor_score
```

Training loss: `L = L_data(x_h*, y) + beta * consistency_score` with `beta = 0.05`, where `L_data` is PROLA's original data loss. The projection is differentiable because `ColumnIntegral`, `Downsample`, `Upsample`, `GeostrophicResidual`, and the scaffold adjoints are all linear operators implemented as fixed sparse matrices; the fixed-point loop is unrolled for K=3 steps so gradients propagate.

**Why this design.** We chose a fixed-point projection over a learned penalty head because a penalty head reintroduces exactly the failure mode we are fixing (the network can trade off data loss against the penalty and settle in an inconsistent region), whereas a projection makes the constraint hold to machine tolerance at the output, accepting the cost that the projection is only as correct as the scaffold's discretization of the conservation law. We chose the coarse field's own column integral and geostrophic residual as the manifold targets rather than a global climatological target because the invariant we care about is consistency *with the conditioning field*, not absolute physical realism, accepting the cost that a coarse field that is itself unbalanced will propagate that imbalance; to mitigate this, we replace the raw coarse field with a budget-closed reference `x_c^bc = x_c + R_c` and mask the geostrophic term by local Rossby number, so the target is physically valid where balance is defined. We chose the scaffold adjoint to distribute mass corrections over a per-column uniform redistribution because it is the exact adjoint of the column-integral operator and thus makes the projection a true orthogonal projection in the mass norm, accepting the cost that corrections are spatially smooth and cannot fix fine-scale mass anomalies within a column. We chose K=3 unrolled iterations over a closed-form solve because the geostrophic operator is not diagonalizable in closed form under the scaffold's finite differences, accepting the cost of a small residual that we report as `consistency_score`. A domain expert would not describe MPD as a variant of PROLA: PROLA is a low-rank adaptation of backbone weights, whereas MPD is an output-space projection layer that is orthogonal to *how* the backbone was adapted and could be applied after any downscaling head; nor is it a variant of physics-informed neural network penalties, because it projects rather than penalizes.

**Why it measures what we claim.** The quantity `consistency_score` measures physical consistency between coarse and fine fields because the assumption that the budget-closed coarse field's column integral and geostrophic residual constitute the correct manifold target makes the score zero exactly when the fine field integrates to the budget-closed coarse field and shares its balance residual; this assumption fails when the budget-closure residual `R_c` is itself inaccurate or when the scaffold's finite-difference stencil is too coarse to resolve the local geostrophic term, in which case the score reflects scaffold discretization error rather than true inconsistency. The projection residual `r_m` measures mass-conservation violation because the assumption that column integration commutes with the scaffold's downsampling operator makes `Downsample(ColumnIntegral(x)) - m_c` exactly the coarse-grid mass mismatch; this assumption fails at domain boundaries where the downsampling stencil is truncated, in which case `r_m` reflects boundary truncation rather than interior mass loss. The geostrophic residual `r_g` measures dynamical-balance violation because the assumption that the scaffold's finite-difference Coriolis and pressure-gradient operators approximate the continuous geostrophic relation to first order makes `r_g` proportional to the departure from balance; this assumption fails in the tropics where the Coriolis parameter vanishes and where the scaffold stencil cannot resolve the Rossby radius, in which case `r_g` reflects the breakdown of geostrophic scaling rather than a modeling error; the Rossby-number mask `w_g` sets `r_g` to zero in those regions, and the discretization-error floor `floor_score` computed on the coarse field itself provides a reference level below which `consistency_score` cannot distinguish genuine balance violation from scaffold error.

## Contribution

['A differentiable manifold-projection closure layer (MPD) that sits after any downscaling residual head and enforces column-mass conservation and geostrophic-balance residual matching against the coarse conditioning field by construction, with a fixed-point unrolled implementation whose adjoints are exact sparse operators.', "A certified consistency score derived from the projection residual that quantifies, per sample, how far the learned downscaling operator strayed from the coarse model's physical manifold, usable as a diagnostic and as an auxiliary training signal.", 'A design principle: physical consistency in downscaling is best enforced as an output-space projection onto a constraint manifold defined by the coarse field itself, rather than as a soft penalty or a learned consistency head, because projection makes the invariant hold independent of the training distribution.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PSWD high-res reanalysis pairs (ERA5-derived) | Same setting PROLA was evaluated in |
| Primary metric | consistency_score + data RMSE on balanced subsets | Directly tests projected-consistency claim |
| Secondary metric | Discretization-error floor `floor_score` (computed on coarse field) | Separates scaffold error from true inconsistency |
| Secondary metric | Extreme-event hit rate (e.g., precipitation > 95th percentile) | Downstream impact on forecast skill |
| Secondary metric | Spectral energy at fine scales (wavenumbers 10–50) | Tests whether projection preserves fine-scale structure |
| Baseline | PROLA (unmodified) | Tests whether projection adds beyond adaptation |
| Baseline | PROLA + learned physics penalty head | Tests projection vs. penalty failure mode |
| Baseline | PROLA + post-hoc bilinear mass rescaling | Tests whether simple correction suffices |
| Ablation-of-ours | MPD with K=0 (no projection, score only) | Isolates projection loop from scoring |
| Ablation-of-ours | MPD without budget-closure residual (R_c = 0) | Tests whether budget closure matters |
| Ablation-of-ours | MPD without Rossby-number mask (w_g = 1 everywhere) | Tests whether masking matters |

### Why this setup validates the claim

The central claim is that an output-space projection onto the coarse field's own mass and geostrophic manifold yields fields that are simultaneously more data-accurate and more physically consistent than adaptation alone, and that this advantage is *causal to projection* rather than to extra capacity or scoring. Each baseline isolates one sub-claim: unmodified PROLA tests whether adaptation without projection leaves residual mass/balance inconsistency; the learned penalty head tests our specific mechanistic claim that penalties can be traded off against data loss while projections cannot; the post-hoc bilinear rescaling tests whether a cheap non-differentiable correction captures most of the gain, which would undercut the need for a differentiable scaffold-adjoint projection. The K=0 ablation is the crucial control: it holds the consistency_score readout fixed while removing the projection, so any gap between it and full MPD is attributable to the projection loop itself, not to the auxiliary loss term. The two additional ablations (no budget closure, no Rossby mask) test the load-bearing assumption directly: if removing either does not degrade consistency_score relative to the floor, the projection is not relying on the physically valid target we claim. consistency_score is the right primary metric because the claim is about a *specific* diagnostic (coarse-grid mass mismatch plus balance residual) that the projection is designed to drive to machine tolerance; data RMSE is reported alongside to guard against the degenerate case where projection improves consistency by degrading accuracy. The secondary metrics (floor_score, extreme-event hit rate, spectral energy) ensure that the consistency gain is not an artifact of scaffold error and that it translates to quantities practitioners care about. Together these form a falsifiable test: the method must win on consistency *and* not lose on accuracy, the win must disappear under K=0, and the win must be robust to the budget-closure and Rossby-mask choices.

### Expected outcome and causal chain

**vs. PROLA (unmodified)** — On a case where the coarse conditioning field carries a column-integrated mass anomaly (e.g., a frontal zone where coarse moisture flux is under-resolved), unmodified PROLA produces a high-resolution field whose column integrals drift from the coarse field because its low-rank adaptation has no mechanism tying output integrals to the conditioning input. Our method instead applies the scaffold adjoint to redistribute the coarse-grid mass mismatch across each column, because the projection is the exact adjoint of the column-integral operator and thus zeroes the mismatch by construction. We expect a clear consistency_score gap concentrated on mass-anomaly-heavy subsets (frontal and convective regions) and near-parity on quiescent subsets; the gap should exceed the discretization-error floor `floor_score` by at least a factor of 2 on those subsets.

**vs. PROLA + learned physics penalty head** — On a case where the data loss and the penalty conflict (a sharp gradient feature where smoothing reduces the penalty but blurs the field), the penalty-head baseline settles at an intermediate point because the network can trade data loss against the penalty weight. Our method instead projects after the head, so the constraint holds to tolerance regardless of the data term, because projection is not a soft trade-off. We expect the penalty-head baseline to show moderate consistency gains but a measurable RMSE degradation on sharp-gradient subsets, while MPD shows comparable consistency gains with no such RMSE loss — a divergence in the accuracy-consistency plane.

**vs. PROLA + post-hoc bilinear mass rescaling** — On a case where the mass anomaly is spatially structured within a column (a tilted anomaly the coarse grid smears), bilinear rescaling produces a field that matches column totals but leaves a residual geostrophic imbalance because uniform rescaling ignores the balance operator. Our method instead applies the balance-adjoint projection step, which redistributes corrections onto the balanced subspace, because the geostrophic residual is projected rather than merely rescaled. We expect near-parity on the mass term but a noticeable gap on the balance term, largest in mid-latitude balanced-flow subsets and vanishing in the tropics where geostrophic scaling breaks down (and where the Rossby-number mask sets the balance term to zero).

**vs. MPD with K=0 (no projection, score only)** — The K=0 ablation holds the consistency_score readout fixed while removing the projection loop. We expect full MPD to achieve a consistency_score at or below the discretization-error floor `floor_score`, while K=0 remains above it by a margin that grows with the magnitude of coarse mass anomalies, confirming that the projection loop itself—not the auxiliary loss term—is responsible for the consistency gain.

**vs. MPD without budget-closure residual (R_c = 0)** — This ablation tests the load-bearing assumption that the coarse field's raw column integral and geostrophic residual are a valid manifold target. We expect the no-budget-closure variant to show a higher consistency_score (worse) on subsets where the coarse field's own budget residual is large (e.g., regions with strong analysis increments), and to have a larger gap to the floor, because the projection is targeting a physically wrong invariant. If the no-budget-closure variant matches full MPD, the budget-closure correction is unnecessary and the load-bearing assumption is not operative.

**vs. MPD without Rossby-number mask (w_g = 1 everywhere)** — This ablation tests whether the geostrophic term is valid in the tropics. We expect the no-mask variant to show degraded data RMSE and spurious consistency_score improvements in tropical subsets, because the projection is enforcing a balance relation that does not hold there. Full MPD should avoid this degradation by masking the balance term where Ro > 1.

### What would falsify this idea

If MPD's consistency_score gain over unmodified PROLA is uniform across all subsets rather than concentrated where coarse mass anomalies and balance residuals are largest, the projection is not doing what we claim. Equally, if the K=0 ablation matches full MPD on consistency_score, the projection loop is inert and the improvement comes from the auxiliary loss alone, falsifying the central mechanism. Additionally, if the no-budget-closure or no-Rossby-mask ablations match full MPD, the load-bearing assumption about the validity of the coarse manifold target is not supported.

## References

1. PROLA: Principal-Orthogonal Low-rank Adaptation for Predictive Spatiotemporal Weather Downscaling
