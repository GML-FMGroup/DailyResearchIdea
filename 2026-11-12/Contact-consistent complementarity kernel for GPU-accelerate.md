# Contact-consistent complementarity kernel for GPU-accelerated legged MPC

## Motivation

FAME-QP accelerates dense QPs for linear MPC via fused kernels and a cached operator bank, but its operator bank has no representation for complementarity constraints from switched contact dynamics or friction cones. Because legged-robot MPC pipelines linearize contact dynamics and fix the contact schedule before the QP is formed, the solver never sees the complementarity structure, so the returned iterate can be accelerated yet contact-inconsistent — a structural gap, not a tuning issue, that recurs across every operator-splitting QP solver that treats constraints as simple projections.

## Key Insight

A proximal splitting on the normal-cone indicator of the friction/contact set is itself a closed-form, per-contact, GPU-parallel operator whose fixed-point residual is exactly the complementarity violation, so inserting it into the fused iteration makes contact-consistent feasibility the quantity the solver drives to zero rather than a post-hoc repair.

## Method

**What it is.** C3-QP (Contact-Consistent Complementarity QP) is a GPU operator-bank extension for FAME-QP-style fused iterations. Input: QP data (H, g, equality/inequality operators) plus a per-contact normal-cone descriptor (normal direction, friction coefficient, contact-active mask). Output: primal-dual iterate whose contact forces lie in the friction cone and whose normal velocity/force product satisfies complementarity to a tolerance.

**How it works.**

```pseudocode
# Bank construction (once per contact-set change)
for each contact c in contacts:
    build_normal_cone_operator(c, mu_c)        # projection onto K_c = {f: ||f_t|| <= mu_c f_n, f_n >= 0}
    # sign convention: f_n >= 0 is compression, v_n >= 0 is separation; complementarity is 0 <= f_n ⟂ v_n >= 0
    # under this convention, projection onto K_c equals projection onto the normal cone of the contact set
cache cone operators keyed by contact-set hash      # lazy update, as in FAME-QP

# Outer loop: re-linearize dynamics and update contact-set hash
repeat until outer convergence (max 10 outer iterations):
    v_c = contact_velocity(z, c)               # linearized velocity at current iterate
    hash = hash_contact_set(active_mask)       # contact-set descriptor
    if hash changed: rebuild cone operators for changed contacts

    # Inner loop: freeze velocity linearization v_c and cone operators
    repeat until inner max residual < eps_inner (default 1e-5) or max 50 inner iterations:
        x_half   = ADMM_x_step(z, u; H, g, rho)          # existing FAME-QP operator
        z_eq     = project_equalities(x_half)            # existing operator
        z_ineq   = project_box(x_half)                   # existing operator
        for each contact c (parallel across contacts):
            f_c        = extract_contact_forces(z_ineq, c)
            f_c_proj   = prox_normal_cone(f_c - sigma*v_c, mu_c)  # closed-form cone projection, v_c frozen
            r_comp[c]  = ||f_c_proj - f_c|| + |<f_c, v_c>|        # complementarity residual (inner)
        z        = fuse(z_eq, z_ineq, {f_c_proj})
        u        = u + (x_half - z)                       # dual update
        if max_c r_comp[c] > eps_c (default 1e-4):        # contact-consistency guard
            reweight_penalty(rho, contacts=argmax r_comp) # raise rho only on violating contacts

    # After inner loop: compute linearization error bound
    v_c_new = contact_velocity(z, c)
    lin_err = max_c ||v_c_new - v_c||
    report r_comp and lin_err together
```

**Why this design.** We chose a closed-form proximal operator on the friction-cone normal cone over a mixed-integer QP relaxation because the MIQP branch-and-bound would serialize the per-contact combinatorial search and destroy the fused-kernel latency that FAME-QP was built to deliver, accepting the cost that the splitting converges only asymptotically rather than certifying global optimality of the contact schedule. We chose to fuse the cone projection into the existing ADMM iteration rather than run a separate complementarity post-pass because a post-pass would leave the primal-dual pair inconsistent with the equality/inequality operators already applied, accepting the cost that the operator bank must be rebuilt when the contact-set hash changes. We chose per-contact adaptive penalty reweighting over a single global rho because contact violations are spatially localized and a global reweight would slow convergence on the well-behaved contacts, accepting the cost of one extra reduction kernel per iteration. A domain expert would not describe this as 'FAME-QP plus a projection' — FAME-QP's operator bank is closed under linear equality/inequality projections, whereas the normal-cone operator is nonlinear and per-contact, so the cached bank's lazy-update invariant must be extended with a contact-set hash; that extension, not the projection itself, is the mechanism.

**Why this measures what we claim.** The quantity `r_comp[c] = ||f_c_proj - f_c|| + |<f_c, v_c>|` measures contact-consistent feasibility because the fixed-point residual of the normal-cone proximal operator equals the complementarity violation under the assumption that the velocity operator `contact_velocity` is the exact linearization of the dynamics at the current iterate; this assumption fails when the linearization is stale (large velocity error between iterations), in which case `r_comp` reflects linearization error rather than true complementarity violation. To make this assumption explicit and verifiable, we hold the velocity linearization fixed across an inner ADMM loop (the 'linearization-freeze' inner iteration) so that the cone operator and velocity map are jointly fixed and the fixed-point residual is a genuine complementarity measure; we re-linearize only in an outer loop and report the linearization-error bound `lin_err = max_c ||v_c_new - v_c||` alongside `r_comp`. The contact-set hash measures contact-schedule consistency because two iterates with the same hash share the same active-set structure under the assumption that the contact-set descriptor is a sufficient statistic for the cone operator; this fails when a contact switches within an iteration, in which case the hash lags the true active set by one iteration. The per-contact penalty reweighting measures localized violation because raising rho only on argmax contacts is equivalent to a per-contact dual ascent step under the assumption that contact violations are separable; this fails when contacts are dynamically coupled (e.g., a shared torso), in which case the reweighting under-corrects the coupled direction. The sign convention under which the friction-cone projection equals the normal-cone indicator is stated in the pseudocode: `f_n >= 0` is compression, `v_n >= 0` is separation, and complementarity is `0 <= f_n ⟂ v_n >= 0`.

## Contribution

(1) A complementarity-aware operator-bank extension for GPU QP solvers that represents friction-cone normal-cone projections as cached, per-contact parallel operators with a contact-set hash for lazy updates. (2) A fused-iteration design in which the complementarity residual is the convergence quantity and drives per-contact penalty reweighting, replacing fixed-schedule linearization in legged MPC pipelines. (3) An open CUDA implementation and contact-set descriptor format that integrates with FAME-QP-style operator banks without changing the existing equality/inequality operators.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Contact-rich humanoid/quadrotor MPC QPs with friction cones | Matches FAME-QP benchmark and exposes contact complementarity |
| Primary metric | Contact-complementarity residual r_comp at fixed wall-clock latency | Directly measures the claimed contact-consistent feasibility |
| Secondary metric | Linearization-error bound lin_err = max_c ||v_c_new - v_c|| | Quantifies contamination of r_comp by stale linearization |
| Baseline | FAME-QP (no cone projection) | Isolates the contribution of the normal-cone operator |
| Baseline | GPU interior-point QP solver | Tests whether splitting-with-projection beats certified solvers |
| Baseline | MIQP contact-schedule relaxation | Tests the serialization cost of combinatorial contact search |
| Ablation-of-ours | C3-QP with global rho (no per-contact reweight) | Isolates the localized penalty-reweighting mechanism |
| Ablation-of-ours | C3-QP without inner linearization-freeze (single-loop) | Isolates the effect of freezing velocity linearization |

### Why this setup validates the claim

The central claim is that fusing a per-contact normal-cone proximal operator into FAME-QP's fused ADMM iteration yields contact-consistent iterates at real-time latency, and that the contact-set-hash extension plus per-contact reweighting are the mechanisms responsible. FAME-QP without the cone projection is the decisive baseline: it shares the same operator bank and CUDA-graph latency profile, so any r_comp gap between it and C3-QP is attributable to the cone operator and hash extension rather than to solver architecture. The GPU interior-point baseline tests whether asymptotic splitting convergence is an acceptable substitute for certified optimality at the latency FAME-QP targets, and the MIQP relaxation tests the serialization argument directly. The ablation with global rho tests the localization sub-claim: if per-contact reweighting is the mechanism, removing it should degrade r_comp specifically on scenes with heterogeneous contact violations (e.g., one slipping foot among stable contacts), not uniformly. The ablation without inner linearization-freeze tests the load-bearing assumption: if r_comp is contaminated by stale linearization, the single-loop variant should show higher r_comp and higher lin_err, and the gap should correlate with lin_err. r_comp is the right primary metric because it is the fixed-point residual of the exact operator whose feasibility we claim, so it detects the predicted effect at the point of the mechanism rather than at a downstream proxy; reporting lin_err alongside bounds the metric's fidelity.

### Expected outcome and causal chain

**vs. FAME-QP (no cone projection)** — On a case where a humanoid foot contacts the ground at an oblique angle so the unconstrained QP solution places tangential force outside the friction cone, FAME-QP produces iterates whose box and equality projections are satisfied but whose contact forces violate ||f_t|| <= mu f_n, because its operator bank is closed only under linear projections and has no mechanism to enforce the cone. Our method instead applies the closed-form normal-cone proximal step per contact inside the same fused iteration, so the cone constraint is enforced at every iterate rather than never, and we expect a large r_comp gap on oblique-contact frames with near-parity on frames where the unconstrained solution already lies inside the cone.

**vs. GPU interior-point QP solver** — On a case where the contact set is large and the QP is solved to high accuracy under a tight latency budget, the interior-point baseline produces accurate cone-feasible solutions but its per-iteration dense factorizations and centralized step computation inflate latency beyond the real-time budget, because its mechanism is not structured around the fused operator-bank replay that FAME-QP exploits. Our method instead replays cached cone and linear operators in a CUDA graph with per-contact parallelism, so we expect comparable r_comp at the target latency but a clear latency advantage, with the gap widening as contact count grows.

**vs. MIQP contact-schedule relaxation** — On a case where the contact schedule is ambiguous (e.g., a foot near lift-off), the MIQP baseline enumerates active-set branches and serializes the per-contact combinatorial search, producing a certified schedule but at latency that scales with the number of ambiguous contacts, because branch-and-bound cannot be fused into a single kernel launch. Our method instead relaxes the schedule via the cone proximal operator and the contact-set hash, accepting asymptotic rather than certified convergence, so we expect r_comp to converge within tolerance on most frames while latency stays flat across contact counts, with a small residual tail on the hardest ambiguous frames.

**vs. C3-QP with global rho (no per-contact reweight)** — On scenes with heterogeneous contact violations (e.g., one slipping foot among stable contacts), the global-rho variant raises the penalty uniformly, slowing convergence on well-behaved contacts and under-correcting the violating one, because it cannot localize the penalty increase. Our method instead reweights rho only on argmax contacts, so we expect lower r_comp on the violating contact and faster overall convergence, with the gap concentrated on heterogeneous-violation scenes.

**vs. C3-QP without inner linearization-freeze (single-loop)** — On frames where the velocity linearization drifts across iterations, the single-loop variant computes r_comp against a moving velocity target, so its r_comp conflates linearization error with complementarity violation and its fixed point need not be contact-consistent. Our method instead freezes v_c across the inner loop and reports lin_err, so we expect lower r_comp and a measurable correlation between the single-loop r_comp gap and lin_err, confirming that the freeze is necessary for r_comp to be a faithful measure.

### What would falsify this idea

If C3-QP's r_comp advantage over FAME-QP is uniform across contact subsets rather than concentrated on oblique and near-lift-off contacts where the predicted cone-violation failure mode occurs, the claim that the normal-cone operator is the causal mechanism is wrong. Likewise, if removing per-contact reweighting (global-rho ablation) leaves r_comp unchanged on heterogeneous-violation scenes, the localization sub-claim is falsified; and if the single-loop variant without linearization-freeze shows no r_comp degradation correlated with lin_err, the load-bearing assumption that r_comp measures true complementarity violation is falsified.

## References

1. FAME-QP: A GPU-Accelerated Quadratic-Programming Solver for Real-Time Model Predictive Control
