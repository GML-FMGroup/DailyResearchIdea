# Feedback-Calibrated Low-Curvature Projection for Preservation-Constrained Adaptation

## Motivation

Preservation-constrained adaptation methods such as CurvDelta construct a single fixed low-curvature subspace from K-FAC factors around the pre-adaptation model and then project every update into it, so the strength of preservation is set once at construction time and never re-examined as training proceeds. Structurally, this is a mismatch between a static geometric object and a non-stationary optimization trajectory: as parameters move away from the linearization point, the local curvature approximation that justified the projection degrades, yet the projection strength stays constant, so the realized preservation loss can drift arbitrarily far from the near-invariance the construction promised. The field therefore lacks any per-update mechanism that reads realized preservation loss and adjusts projection strength, leaving the acquisition–retention trade-off unquantified at the granularity where it is actually decided.

## Key Insight

The realized preservation-loss increment along a proposed update is a monotone function of the update's component inside the curvature-approximated preservation-sensitive subspace, so a scalar feedback gain on that component — driven by the measured loss increment — makes projection strength and preservation loss monotonically consistent without ever reconstructing the full curvature matrix.

## Method

**What it is.** FCLP (Feedback-Calibrated Low-Curvature Projection) is an optimizer wrapper that takes a base optimizer's proposed update Δθ, a preservation loss L_pres, and a low-curvature basis built once by K-FAC, and returns a scaled, projected update Δθ̃ together with an updated projection gain γ_t. It augments CurvDelta's fixed projection with a per-step multiplicative controller on the projection strength. The load-bearing assumption is stated explicitly: **the realized preservation-loss increment at step t is dominated by the update's component inside the fixed top-r K-FAC subspace U**, so that a scalar gain on Δθ_par alone can drive L_pres monotonically toward τ even though U is built once at θ_0 and never refreshed. This assumption is operationalized and tested per step (Phase 3b below) rather than assumed silently.

**How it works.**

```pseudocode
Inputs: base optimizer step Δθ_t, preservation loss L_pres(θ_t),
        K-FAC factors (A, B) at θ_0, rank r, gains η (base lr),
        target tolerance τ, controller rate β, gain bounds [γ_min, γ_max],
        finite-difference probe ε (default 1e-3), refresh threshold κ (default 0.1)

# --- Phase 1: build low-curvature basis ONCE (inherited from CurvDelta) ---
U  = top-r eigenvectors of K-FAC preconditioner  # preservation-sensitive dirs
P  = I - U U^T                                   # projection onto low-curvature space

# --- Phase 2: decompose the proposed update ---
Δθ_par  = U U^T Δθ_t          # component inside preservation-sensitive subspace
Δθ_perp = P Δθ_t              # component in low-curvature subspace

# --- Phase 3: per-update calibration of projection strength ---
# 3a. Measured directional sensitivity of L_pres along the actual proposed update
#     (finite-difference / JVP estimate; no full curvature reconstruction)
ΔL_meas = (L_pres(θ_t - η * Δθ_t) - L_pres(θ_t)) / (η * ||Δθ_t|| + ε)
# 3b. Per-step parallel-vs-perpendicular contribution to ΔL_pres (diagnostic + refresh trigger)
ΔL_par  = (L_pres(θ_t - η * Δθ_par)  - L_pres(θ_t)) / (η * ||Δθ_par||  + ε)
ΔL_perp = (L_pres(θ_t - η * Δθ_perp) - L_pres(θ_t)) / (η * ||Δθ_perp|| + ε)
# 3c. Controller driven by the MEASURED sensitivity, not the parallel-component magnitude
γ_t = clamp(γ_{t-1} * exp(-β * (ΔL_meas - τ) / (|ΔL_meas| + ε)), γ_min, γ_max)
Δθ_tilde = Δθ_perp + γ_t * Δθ_par
# 3d. Cheap periodic refresh of U when perpendicular leakage dominates
if |ΔL_perp| > κ * |ΔL_meas|:
    U = top-r eigenvectors of K-FAC preconditioner recomputed at θ_t  # O(d^2) amortized
    P = I - U U^T

# --- Phase 4: apply and record ---
θ_{t+1} = θ_t - η * Δθ_tilde
record (γ_t, L_pres(θ_t), ΔL_meas, ΔL_par, ΔL_perp, ||Δθ_par||, ||Δθ_perp||) for diagnostics
```

The controller in Phase 3c is a multiplicative integral controller on the **measured** directional derivative of the preservation loss along the actual proposed update: when ΔL_meas exceeds tolerance τ, γ_t shrinks geometrically (attenuating the preservation-sensitive component), and when ΔL_meas falls below τ, γ_t relaxes back toward 1, restoring acquisition capacity. β and τ are the only new hyperparameters; γ is bounded so the update never inverts sign. The finite-difference probe in 3a costs one extra forward evaluation of L_pres per step (no backward pass, no full curvature matrix). The refresh in 3d is triggered only when the measured perpendicular leakage exceeds κ times the total measured sensitivity, so the O(d²) K-FAC recomputation is amortized over many steps rather than paid every step.

**Why this design.** We chose a scalar multiplicative gain on the parallel component over recomputing K-FAC factors every step because full curvature refresh costs O(d²) per step and destroys the amortized efficiency that makes CurvDelta practical, accepting that the gain only corrects first-order drift in the curvature estimate rather than tracking it exactly; the cheap finite-difference probe in Phase 3a and the thresholded refresh in Phase 3d are the two concrete concessions that keep the controller reading the true loss increment without paying full-refresh cost every step. We chose an integral (multiplicative) controller over a learned gating network because the preservation loss is directly observable at every step and a learned gate would need its own training signal, which reintroduces the very acquisition–retention coupling we are trying to expose; the cost is that β and τ must be set per task family. We chose to attenuate only the parallel component rather than rescaling the whole update because the perpendicular component is, by construction, the part the low-curvature space certifies as preservation-safe, so scaling it would sacrifice acquisition for no preservation benefit; the Phase 3b diagnostic and Phase 3d refresh are what keep this choice honest when the certification is stale. We chose hard clamping of γ over unconstrained adaptation because a single anomalous loss spike would otherwise collapse the gain to zero and stall learning. A domain expert would not describe this as 'CurvDelta plus a scheduler': CurvDelta's projection is a fixed linear map, whereas here the projection strength is a state variable driven by a measured loss signal, so the effective update operator changes every step — the object being studied is the closed loop, not the projection.

**Why it measures what we claim.** The quantity γ_t measures the calibrated projection strength because the multiplicative controller makes γ_t a monotone decreasing function of the measured preservation-loss sensitivity excess (ΔL_meas − τ) under the assumption that the finite-difference probe ΔL_meas is a faithful estimate of the true directional derivative of L_pres along Δθ_tilde; this assumption fails when the probe step ε is too large relative to the local curvature radius (the finite difference then averages over a region where the derivative has changed sign), in which case γ_t reflects a smeared sensitivity rather than the instantaneous one. The quantity ΔL_par / ΔL_meas measures the parallel-dominance assumption because it directly compares the preservation-loss contribution of the parallel component to the total measured increment under the assumption that the two finite-difference probes are evaluated at the same θ_t and with the same ε; this assumption fails when the parallel and perpendicular probes have very different norms (the normalization by ||Δθ_par|| and ||Δθ_perp|| then amplifies noise differently), in which case the ratio is a noisy indicator and the refresh trigger in Phase 3d may fire spuriously. The quantity ||Δθ_par|| measures the preservation-sensitive update mass because U spans the top-r eigenvectors of the K-FAC preconditioner under the assumption that preservation curvature is dominated by those r directions; this assumption fails when the preservation Hessian has a heavy tail beyond rank r, in which case ||Δθ_par|| underestimates the true preservation-relevant update — the Phase 3b diagnostic is what detects this failure empirically. The tolerance τ measures the target preservation budget because it is compared directly against the measured ΔL_meas under the assumption that ΔL_meas is on a comparable scale across steps; this assumption fails when the preservation loss is non-stationary (e.g., distribution shift in the retention probe), in which case τ encodes a stale budget and the controller chases a moving target.

## Contribution

(1) A per-update feedback calibration mechanism, FCLP, that turns a fixed low-curvature projection into a closed-loop operator whose strength is a monotone function of realized preservation loss, requiring only two new hyperparameters (tolerance τ and rate β) on top of CurvDelta. (2) The design principle that preservation strength should be controlled on the parallel (curvature-sensitive) component only, leaving the certified perpendicular component untouched, which decouples retention control from acquisition capacity. (3) A diagnostic trace (γ_t, L_pres, parallel/perpendicular norms) that exposes the acquisition–retention trade-off at single-update granularity, enabling per-task tuning of τ without retraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Knowledge unlearning (TOFU-style forget/retain split) + math reasoning (GSM8K) + code generation (HumanEval) | Exercises both retention constraint and acquisition across two distinct preservation-constrained task families (unlearning and capability learning) |
| Primary metric | Retention-loss excess vs. downstream gain, reported as a Pareto frontier across τ ∈ {quantiles 0.25, 0.5, 0.75, 0.9 of L_pres at θ_0} | Directly exposes acquisition–retention trade-off at the granularity where it is decided |
| Secondary metric | Per-step ΔL_par / ΔL_meas ratio (parallel-dominance diagnostic) | Empirically validates the load-bearing assumption |
| Baseline | CurvDelta (fixed projection) | Tests whether closed-loop gain beats fixed projection |
| Baseline | CurvLoRA (low-rank variant) | Tests gain controller against cheaper fixed geometry |
| Baseline | LoRA (no preservation constraint) | Tests whether constraint is needed at all |
| Baseline | Per-step line-search / trust-region on L_pres (adaptive strength without a controller) | Tests whether closed-loop gain is reproducible by a simpler adaptive-strength scheme |
| Ablation-of-ours | FCLP with γ frozen at 1 | Isolates the controller from the projection |
| Ablation-of-ours | FCLP with γ frozen at time-averaged γ̄ (computed post-hoc from a full run) | Separates per-step calibration from constant attenuation |
| Ablation-of-ours | FCLP with Phase 3d refresh disabled | Isolates the contribution of the periodic U refresh |
| Hyperparameter sweep | β ∈ {0.01, 0.1, 1.0}, τ ∈ {quantiles of L_pres at θ_0}, ε ∈ {1e-4, 1e-3, 1e-2}, κ ∈ {0.05, 0.1, 0.2} | Pre-specified sensitivity sweep; report compute/memory overhead per step vs. CurvDelta |

### Why this setup validates the claim

The central claim is that replacing CurvDelta's fixed projection strength with a measured-loss-driven multiplicative gain improves the acquisition–retention trade-off, and that this improvement comes specifically from attenuating the preservation-sensitive (parallel) component when the preservation budget is exceeded. CurvDelta is the right primary baseline because it holds the low-curvature basis fixed, so any gap between it and FCLP isolates the contribution of the closed loop rather than the geometry. CurvLoRA tests whether the same benefit is obtainable from a cheaper fixed low-rank geometry, guarding against the possibility that the gain is doing nothing a smaller rank could not. LoRA without any preservation constraint establishes the unconstrained upper bound on acquisition and the lower bound on retention, so we can see whether FCLP lands strictly between the constrained and unconstrained extremes. The per-step line-search / trust-region baseline is the decisive test that the closed-loop gain is not reproducible by a simpler adaptive-strength scheme that also adjusts projection strength without a controller. The frozen-γ-at-1 ablation shares FCLP's basis and projection but disables the controller, attributing any effect to the feedback loop; the frozen-γ-at-γ̄ ablation further separates per-step calibration from constant attenuation. The retention-loss-excess-versus-downstream-gain Pareto frontier across τ values is chosen because it is the two-sided quantity the claim is about; a one-sided accuracy number could be matched by simply tuning a global learning rate and would not distinguish calibrated attenuation from blunt regularization. The per-step ΔL_par / ΔL_meas diagnostic directly tests the load-bearing assumption: if the ratio is near 1 on most steps, the parallel component dominates and the controller reads the right signal; if it is near 0, the controller is regulating a proxy and the Phase 3d refresh is doing the work.

### Expected outcome and causal chain

**vs. CurvDelta (fixed projection)** — On an unlearning case where the retention probe distribution shifts mid-run so the fixed projection under-attenuates the parallel component, CurvDelta keeps a constant gain and lets preservation loss accumulate past budget because its projection cannot respond to realized loss. Our method instead shrinks γ_t geometrically once ΔL_meas exceeds τ, so we expect a visible retention-excess reduction on the shifted-probe subset with roughly matched downstream unlearning accuracy, and near-parity on stationary subsets where the fixed projection was already sufficient. The per-step ΔL_par / ΔL_meas diagnostic should show the ratio dropping below κ on the shifted-probe steps, coinciding with Phase 3d refresh events.

**vs. CurvLoRA (low-rank variant)** — On a math-reasoning case where the preservation-relevant curvature has mass beyond the chosen rank, CurvLoRA's fixed low-rank geometry leaks preservation loss that no rank reduction fixes because it has no per-step correction. Our method attenuates the parallel component adaptively and refreshes U when perpendicular leakage dominates, so we expect FCLP to hold retention excess closer to budget on the heavy-tail-curvature subset while CurvLoRA shows a persistent floor, with the gap widening as rank decreases. The ΔL_par / ΔL_meas diagnostic should show the ratio falling below κ more often at lower rank, triggering more refreshes.

**vs. LoRA (no preservation constraint)** — On a code-generation case where unconstrained adaptation achieves high downstream gains but degrades general capability, LoRA shows maximal acquisition with retention loss far above τ because nothing constrains it. Our method should approach LoRA's downstream gain while keeping retention excess near τ, so we expect FCLP to sit between LoRA and CurvDelta on the acquisition axis but clearly below both on retention excess, with γ_t settling below 1 on the high-drift steps.

**vs. per-step line-search / trust-region on L_pres** — On both unlearning and math-reasoning cases, the line-search baseline adjusts projection strength per step but without a controller state, so it can match FCLP on stationary subsets where the optimal strength is roughly constant. We expect FCLP to outperform it on the shifted-probe and heavy-tail-curvature subsets, where the controller's memory (γ_{t-1}) and the Phase 3d refresh provide adaptation that a memoryless line-search cannot, with the gap visible as a lower retention-excess at matched downstream gain on the Pareto frontier.

**vs. FCLP with γ frozen at 1** — This ablation shares FCLP's basis and projection but disables the controller, so it should match CurvDelta's behavior on stationary subsets and underperform FCLP on shifted-probe and heavy-tail subsets. The gap between FCLP and this ablation is the direct measure of the closed loop's contribution.

**vs. FCLP with γ frozen at γ̄** — This ablation uses the time-averaged gain from a full FCLP run, so it captures constant attenuation but not per-step calibration. We expect it to close part of the gap to FCLP on stationary subsets but to leave a residual gap on shifted-probe and heavy-tail subsets, where per-step calibration is what matters. The residual gap is the evidence that per-step calibration, not constant attenuation, drives the benefit.

**vs. FCLP with Phase 3d refresh disabled** — This ablation isolates the refresh contribution. We expect it to match FCLP on subsets where the parallel-dominance assumption holds (ΔL_par / ΔL_meas near 1) and to underperform on subsets where perpendicular leakage dominates (ratio below κ), confirming that the refresh is doing the work the controller alone cannot.

### What would falsify this idea

The claim is wrong if the frozen-γ-at-1 ablation matches FCLP's retention-excess-versus-gain frontier, meaning the controller contributes nothing beyond the fixed projection; it is also wrong if the frozen-γ-at-γ̄ ablation matches FCLP, meaning per-step calibration adds nothing beyond constant attenuation. It is further wrong if the per-step ΔL_par / ΔL_meas diagnostic stays near 1 on all steps (so the refresh never fires) yet FCLP still underperforms the line-search baseline, since that would show the controller is not tracking the preservation signal it claims to measure.

## References

1. CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning
