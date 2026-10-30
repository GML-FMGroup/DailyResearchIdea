# Curvature-Certified Adaptive Step Allocation for Flow-Matching Diffusion Transformers

## Motivation

Across the flow-matching diffusion-transformer lineage — from the Oboro text-to-image model to the Matcha-TTS and Lumina-Next families — the probability-flow ODE is integrated on a fixed grid, so the number and placement of velocity-field evaluations is a hyperparameter decoupled from the actual local difficulty of the trajectory. Oboro inherits this structural defect: its decoder cannot spend more solver evaluations on the high-curvature early phase near the noise manifold or on atypically curved samples, and cannot spend fewer on the smooth late phase, which means inference cost is allocated by a global constant rather than by the geometry of the specific sample being generated.

## Key Insight

A cheap second-order Taylor residual of the flow-matching velocity field, evaluated at the current latent, is simultaneously the error estimate and the controller, so the step schedule is a deterministic function of the trajectory's local curvature rather than a hyperparameter or a separately trained router.

## Method

**What it is.** Curvature-Certified Flow Integration (CCFI) replaces the fixed-grid ODE solver in a flow-matching diffusion transformer with a per-sample, per-step adaptive integrator. Its inputs are the velocity field `v_θ(x, t)` (the Oboro transformer), the current latent `x`, the current time `t`, and a target per-step error tolerance `ε`. Its outputs are the next latent `x'`, the next time `t'`, and the realized local error certificate `r`.

**How it works.**

```pseudocode
def ccfi_sample(v_theta, x, t_grid, eps=1e-3, h_max=0.05, h_min=1e-4, alpha_init=0.5):
    t = t_grid[0]
    h = h_max                       # initial step
    alpha = alpha_init              # local scaling exponent, updated empirically
    while t < t_grid[-1]:
        h = min(h, t_grid[-1] - t)
        # --- curvature certificate: embedded Heun/RK pair ---
        v0   = v_theta(x, t)                       # 1 NFE
        v1   = v_theta(x + h*v0, t + h)            # 1 NFE (Euler predictor)
        x_h  = x + h * (0.5*v0 + 0.5*v1)           # Heun (trapezoidal) update
        v2   = v_theta(x_h, t + h)                 # 1 NFE (third-order correction)
        x_3  = x + h * (v0 + 4*v1 + v2) / 6.0      # third-order step (embedded pair)
        r    = || x_3 - x_h || / (|| x_h || + 1e-8) # relative embedded error estimate
        # --- step-allocation policy (deterministic, closed-form) ---
        if r <= eps:                               # accept full step
            x = x_h
            t = t + h
            # update local scaling exponent from last accepted step
            if previous accepted step exists:
                alpha = log(r / r_prev) / log(h / h_prev)
                alpha = clip(alpha, 0.1, 1.0)
            h = min(h_max, h * (eps / (r + 1e-8))**(1.0 / (alpha + 1.0)))
        else:                                      # half-step, same t
            h = max(h_min, h * 0.5)
        # loop continues; NFE count is emergent, not chosen
    return x
```

The certificate `r` is the relative difference between the Heun (trapezoidal) update and a third-order step computed from the same velocity evaluations. This embedded pair estimates the local truncation error of the propagated solution (the Heun result) rather than a predictor-predictor defect. The step size is grown or halved by the closed-form rule `h ← h·(ε/r)^{1/(α+1)}`, where `α` is the local scaling exponent estimated empirically from the last accepted step pair (as in standard step-doubling / PI controllers). The loop terminates when `t` reaches the end of the requested time grid. The total number of ODE evaluations is therefore an emergent property of the trajectory's curvature profile for that sample.

**Why this design.** We chose an embedded Heun/RK pair over a learned step-count router because the embedded difference is a direct numerical-error estimate available at inference with no training signal, accepting the cost that it adds one additional NFE per accepted step (the third-order correction) beyond the two used by the Heun update. We chose a deterministic closed-form growth rule with an empirically updated exponent over a learned controller because a learned controller would need its own training objective and would re-introduce the hyperparameter problem at a higher level, accepting the cost that the schedule is conservative and may under-spend steps on unusually smooth samples. We chose to halve rather than reject-and-restart on a failed certificate because halving keeps the latent on the same trajectory branch and avoids discarding the predictor evaluation, accepting the cost that a badly overestimated step incurs one wasted half-step. Crucially, a domain expert would not describe this as a variant of the Oboro fixed-grid decoder or of the Matcha-TTS / Lumina-Next fixed-schedule solvers: those methods commit to a step count before seeing the sample, whereas CCFI's step count is a derived quantity whose value is produced by the same embedded error estimate that certifies its own error — the schedule and the error estimate are the same computation, not two cooperating modules.

**Why this measures what we claim.** The relative embedded error `r = ||x_3 − x_h|| / ||x_h||` measures local integration error because the assumption that the velocity field is smooth enough that the difference between a second-order and third-order step is a faithful proxy for the true local truncation error of the second-order solution holds; this assumption fails when the velocity field has a near-discontinuity (e.g., a sharp prompt-conditioning switch mid-trajectory), in which case `r` reflects the discrepancy between the two embedded solutions rather than true truncation error and the policy over-refines. The step-growth exponent `1/(α+1)` measures trajectory difficulty because the assumption that local error scales as `h^{α+1}` with `α` estimated from the last step pair makes the closed-form step the unique size achieving tolerance `ε`; this assumption fails when the trajectory is stiff (very high curvature over a short interval), in which case the empirical exponent under-corrects and the half-step branch dominates, which is exactly the intended conservative behavior. The half-step decision measures the need for finer resolution because the assumption that halving reduces the residual by at least a factor of two holds for smooth fields; this assumption fails near the noise manifold where curvature changes faster than the step, in which case the loop may halve repeatedly and the realized NFE count reflects manifold proximity rather than a fixed budget.

## Contribution

(1) A deterministic curvature-certified step-allocation policy for flow-matching diffusion transformers in which the second-order Taylor residual of the velocity field simultaneously serves as the local error estimate and the step-size controller, eliminating the fixed-grid step count as a hyperparameter. (2) A design principle showing that the number of ODE evaluations can be made an emergent function of per-sample trajectory geometry at inference time with no additional training signal and only one amortized predictor evaluation per accepted step. (3) An open integrator compatible with any flow-matching transformer velocity field, demonstrated as a drop-in replacement for the fixed-grid decoder in the Oboro architecture.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Oboro limited-data T2I set | Matches the transformer velocity field under test |
| Primary metric | NFE-to-target-FID efficiency curve | Directly exposes emergent step allocation vs quality |
| Baseline | Oboro fixed-grid Euler | Isolates the value of adaptive stepping |
| Baseline | Oboro fixed-grid Heun | Tests whether adaptivity beats a stronger fixed solver |
| Baseline | Learned step-count router | Tests the deterministic certificate against a learned controller |
| Baseline | Standard embedded adaptive solver (dopri5) on the same velocity field | Tests whether CCFI beats a general-purpose adaptive solver at matched NFE |
| Ablation-of-ours | CCFI with fixed h (no growth rule) | Removes step-allocation while keeping the certificate |
| Ablation-of-ours | CCFI with Euler-vs-Euler residual (original r) | Tests whether the embedded Heun/RK pair is necessary for reliable error control |

### Why this setup validates the claim

The central claim is that CCFI's step count is an emergent, per-sample consequence of the trajectory's curvature, and that this buys quality-per-NFE over schedules fixed before seeing the sample. The Oboro fixed-grid Euler baseline tests the first sub-claim: if adaptivity is doing anything, CCFI should reach a given FID with fewer mean NFEs, and the gap should widen on samples whose curvature profile is heterogeneous. The fixed-grid Heun baseline tests whether the win is merely from higher order rather than from adaptivity, so CCFI must beat a second-order fixed solver, not just Euler. The learned-router baseline tests the design argument that a deterministic certificate avoids re-introducing a hyperparameter; if the router matches CCFI, the certificate's simplicity is not earning its keep. The dopri5 baseline tests whether CCFI's specific embedded pair and exponent update offer any advantage over a standard general-purpose adaptive solver; if dopri5 matches CCFI at matched NFE, the novelty must be reframed around per-sample curvature profiling rather than solver design. The fixed-h ablation removes the growth rule while keeping the certificate, isolating whether the gain comes from step allocation or from the embedded update alone. The Euler-vs-Euler ablation tests whether the original predictor-predictor residual is sufficient or whether the embedded Heun/RK pair is necessary for reliable error control. The NFE-to-FID efficiency curve is the right metric because it is the only one that reads out the derived quantity (NFE) against quality, which is exactly the trade the method claims to improve; a single FID number at a fixed budget would hide the effect entirely.

### Expected outcome and causal chain

**vs. Oboro fixed-grid Euler** — On a sample with a sharp prompt-conditioning switch mid-trajectory, fixed-grid Euler commits to a uniform step before seeing the curvature, so it either overshoots through the switch (visible as structural artifacts) or wastes steps on the smooth early segment. CCFI detects the elevated embedded error at the switch, halves the step there, and spends its budget locally because the embedded error and the schedule are the same computation. We expect a noticeable FID gap concentrated on high-curvature samples with near-parity on smooth ones, and a lower mean NFE at matched FID.

**vs. Oboro fixed-grid Heun** — On a stiff trajectory where curvature changes faster than the step, fixed-grid Heun applies its second-order correction uniformly and still steps over the stiff interval, producing a small but systematic bias that shows up as texture smearing. CCFI's half-step branch dominates exactly in that interval because the empirical exponent under-corrects under stiffness, which is the intended conservative behavior. We expect CCFI to match Heun on smooth samples but pull ahead on the stiff subset, with the gain appearing as fewer NFEs to reach Heun's best FID rather than a uniformly better FID.

**vs. Learned step-count router** — On an unusually smooth sample, the learned router, trained on average difficulty, allocates a step count near the training mean and over-spends, because its controller was fit to a global objective rather than this trajectory. CCFI's closed-form growth rule lets `h` expand toward `h_max` on that same sample because the embedded error stays below `ε`, so the budget collapses naturally. We expect CCFI to show a clear NFE reduction on the smooth tail of the distribution with quality parity, and to trail the router only on the rare sample where the router's learned prior happens to be better calibrated.

**vs. Standard embedded adaptive solver (dopri5)** — On trajectories where the velocity field is smooth and well-behaved, dopri5's higher-order embedded pair may achieve comparable FID at similar NFE, because both methods adapt to local error. However, on samples with heterogeneous curvature (e.g., prompt switches, stiff intervals), CCFI's per-sample curvature profiling and empirical exponent update may yield a better NFE-to-FID trade-off because the controller is tuned to the specific trajectory's scaling behavior. We expect CCFI to match or slightly beat dopri5 on the high-curvature subset, with parity on smooth samples; if dopri5 dominates uniformly, the novelty claim must be reframed around the diagnostic value of the curvature profile.

### What would falsify this idea

If CCFI's FID gain over fixed-grid Euler is uniform across smooth and high-curvature subsets rather than concentrated where the predicted curvature failure occurs, the claim that emergent step allocation is the causal mechanism is wrong. Likewise, if the fixed-h ablation matches full CCFI, the growth rule is not doing the work and the central argument collapses. Additionally, if the Euler-vs-Euler residual ablation matches full CCFI, the embedded Heun/RK pair is unnecessary and the original predictor-predictor certificate is sufficient.

## References

1. oboro: Text-to-Image Synthesis on Limited Data using Flow-based Diffusion Transformer with MMH Attention
2. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
3. eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers
