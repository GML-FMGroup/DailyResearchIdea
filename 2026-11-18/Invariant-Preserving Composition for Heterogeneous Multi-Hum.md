# Invariant-Preserving Composition for Heterogeneous Multi-Humanoid Teams

## Motivation

Multi-robot coordination methods increasingly rely on a learned high-level planner issuing directives to a post-trained low-level controller, and Heave-Ho exemplifies this: its low-level multi-agent tracking policy is post-trained on two identical humanoids, yet the paper claims scaling to heterogeneous teams of up to ten robots 'without retraining', with no structural guarantee that the physical-coupling invariants the low-level policy learned survive when team composition changes. The root cause is compositional: the high-level vision-language directive is expressed in task space, while the low-level coordination manifold is defined in joint/contact space under coupling constraints, and nothing in the pipeline maps one onto the other in a way that respects the invariants. EMOS shows the complementary half of the problem—embodiment-aware capability descriptions (robot resumes) let LLM agents plan for heterogeneous hardware—but capability description alone does not constrain execution to a feasible coordination manifold.

## Key Insight

Physical coupling constraints among a heterogeneous team induce a low-dimensional coordination manifold whose tangent space can be computed per-composition from embodiment descriptors, so projecting every high-level directive onto that tangent space before execution preserves the invariants the low-level policy was post-trained to satisfy, independent of which robots are on the team.

## Method

**What it is.** CIL (Composition-Invariant Layer) is a runtime composition layer between Heave-Ho's vision-language planner and its post-trained multi-humanoid tracking policy. It takes as input (i) a task-space directive from the high-level planner, (ii) the current joint state and local observations of each team member, and (iii) a per-robot embodiment descriptor (a structured capability vector derived from URDF, following EMOS's robot-resume idea), and it outputs per-robot joint-space reference trajectories that lie on the team's learned coordination manifold.

**Load-bearing assumption (stated explicitly).** CIL assumes that the analytic coupling Jacobian J_c built from URDF-derived embodiment descriptors spans exactly the constraint set under which Heave-Ho's post-trained low-level tracking policy learned coordinated control, so that a reference with zero residual r_t lies on the policy's learned coordination manifold for any team composition. This assumption is not asserted from physics alone: CIL calibrates J_c against the frozen policy per team composition (step 1b below) and reports r_t against the calibrated set. The assumption fails when heterogeneous constraint rows (e.g., force balance with a much heavier robot) were never active during post-training on two identical humanoids; in that case r_t reflects feasibility with respect to the analytic constraint model rather than membership in the policy's learned manifold.

**How it works.**

```pseudocode
# Per control step t, team of N robots, joint dim d_i per robot
# Inputs: task directive g_t (from VLM planner), states s_t=(s_t^1..s_t^N),
#         descriptors D=(D_1..D_N) from URDF (mass, reach, DoF, contact sites)

1. Build analytic coupling Jacobian:
   J_c^ana(s_t, D) = d(contact_constraints)/d(q)   # rows = physical coupling constraints
                 # (shared-object pose agreement, force balance, non-penetration)

1b. Calibrate J_c against the frozen policy for this composition:
   # Offline, per composition class (N robots, descriptor bins), on a held-out
   # calibration set of 512 rollouts from the post-trained policy in the
   # 2-identical-humanoid training distribution and in target heterogeneous
   # compositions:
   #   - For each analytic constraint row j, compute the empirical residual
   #     distribution of the policy's own executed trajectories under row j.
   #   - Mark row j as ACTIVE if its 95th-percentile residual < tau_active=1e-3
   #     in the training distribution; otherwise mark INACTIVE.
   #   - Fit a linear correction term C_j (d x d) per row by least squares:
   #       min_C  sum_rollouts || (J_c^ana_j + C_j) q_dot - 0 ||^2
   #     subject to C_j = 0 for rows marked ACTIVE (preserve analytic rows
   #     the policy already satisfies).
   #   - Store per-composition calibrated Jacobian J_c^cal = J_c^ana + C.
   #   - Report per-row ACTIVE/INACTIVE flags alongside all metrics.

2. Compute coordination manifold tangent space:
   T_t = nullspace(J_c^cal(s_t, D))            # dim = sum(d_i) - rank(J_c^cal)

3. Map directive to task-space target:
   x*_t = PlannerDecoder(g_t)                  # VLM decoder, frozen

4. Project target onto manifold:
   q*_t = argmin_q ||FK(q) - x*_t||^2  s.t.  J_c^cal(s_t,D) q = 0
         # solved by one damped-least-squares step, damping lambda=1e-2

5. Invariant residual check:
   r_t = ||J_c^cal(s_t,D) q*_t||
   if r_t > tau (=1e-3): shrink step by alpha=0.5 and re-solve (max 3 iters)

6. Emit q*_t to the post-trained tracking policy as reference; policy
   outputs joint torques from local observations only.

7. Log invariant residual r_t, per-robot constraint violation, and per-row
   ACTIVE/INACTIVE flags for audit.
```

**Why this design.** We chose to compute the coupling Jacobian analytically from the embodiment descriptors rather than learning an invariance head over planner outputs, because the coupling constraints are known physical equalities (shared-object pose agreement, force balance) that a learned head cannot manufacture when the underlying variables are operationally coupled, accepting the cost that the layer requires a URDF-derived constraint model per robot. We chose a single damped-least-squares projection step over an iterative nonlinear solver because the control loop runs at policy rate and a closed-form tangent-space projection has bounded per-step cost, accepting the cost that large directives are shrunk rather than fully realized in one step. We chose to keep the low-level post-trained policy frozen and inject the projection as a reference-trajectory filter rather than retraining the policy per team composition, because retraining would defeat Heave-Ho's without-retraining scaling claim and because the manifold is defined by the constraints, not by the policy weights, accepting the cost that the filter can only reject infeasible references, not repair a policy that is itself badly post-trained. We added the per-composition calibration step (1b) because the analytic rows alone are not guaranteed to coincide with the policy's learned manifold under composition shift; the calibration fits a correction only to rows the policy does not already satisfy in its training distribution, so the nullspace is defined by the policy's empirical manifold where the analytic model is unverified. A domain expert would not describe CIL as a variant of EMOS's robot-resume planning, because EMOS uses capability descriptions to *select* actions at plan time, whereas CIL uses them to *construct a constraint set* whose nullspace filters every executed reference at control rate—the descriptor is an input to a geometric projection, not to a retrieval or prompting step.

**Why it measures what we claim.** The invariant residual r_t = ||J_c^cal(s_t,D) q*_t|| measures preservation of the low-level coordination invariants because the assumption is that the calibrated coupling constraints J_c^cal are exactly the constraints under which the post-trained policy learned coordinated control, so a reference with zero residual lies on the same manifold; this assumption fails when the post-training distribution covered only two identical humanoids and the heterogeneous constraint rows (e.g., a force-balance row involving a much heavier robot) were never active during training, in which case r_t reflects feasibility with respect to the calibrated constraint model rather than with respect to the policy's learned manifold. The per-row ACTIVE/INACTIVE flags measure which constraints the policy actually learned because the assumption is that a row with low empirical residual on the training distribution is part of the learned manifold, so gains on ACTIVE rows are attributable to the policy's learned coordination while gains on INACTIVE rows are attributable to the calibration correction; this assumption fails when a row is spuriously marked ACTIVE due to insufficient calibration rollouts, in which case the flag reflects sampling noise rather than training-distribution activity. The per-composition nullspace dimension dim(T_t) measures team-composition adaptivity because the assumption is that heterogeneous capability differences manifest as changes in the rank of J_c^cal, so a team whose descriptors yield a lower-dimensional tangent space is genuinely more constrained; this assumption fails when two robots have different descriptors but identical coupling rows (e.g., same contact geometry, different mass), in which case dim(T_t) is unchanged and the metric reflects geometric rather than dynamic heterogeneity.

## Contribution

(1) A composition-invariant layer (CIL) that constructs a per-team coordination manifold from URDF-derived embodiment descriptors and projects high-level planner directives onto its tangent space before execution, keeping the post-trained low-level policy frozen. (2) The design principle that heterogeneous-team generalization without retraining is achievable by filtering references through an analytically constructed constraint nullspace rather than by retraining or by capability-conditioned prompting, with the invariant residual r_t as a run-time audit signal. (3) An audit protocol that logs per-step invariant residuals and nullspace dimensions across team compositions, enabling detection of composition regimes where the frozen policy's learned manifold diverges from the analytic one.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HH-Bench (2–10 humanoids, heterogeneous teams) | Directly tests scaling and composition claims |
| Primary metric | Invariant residual r_t (against calibrated J_c^cal) and per-robot constraint violation | Measures manifold preservation, not task success alone |
| Diagnostic metric | Per-row ACTIVE/INACTIVE flags and residual on each row | Attributes gains to rows the policy actually learned |
| Baseline | Heave-Ho unmodified | Tests whether CIL adds anything at all |
| Baseline | EMOS-style robot-resume planner | Tests plan-time capability routing vs control-rate projection |
| Baseline | Single-robot diffusion motion planner (MMD) composed per robot | Tests whether naive per-robot planning breaks coupling |
| Baseline | Standard whole-body controller with the same analytic J_c (Khatib-style nullspace projection) | Isolates descriptor-driven per-composition calibration from textbook nullspace projection |
| Ablation-of-ours | CIL with identity J_c (no projection step) | Isolates the projection from the descriptor plumbing |
| Ablation-of-ours | CIL with analytic J_c only (no calibration step 1b) | Isolates the calibration correction from the analytic constraint model |

### Why this setup validates the claim

The central claim is that a control-rate geometric projection onto the nullspace of calibrated coupling constraints preserves the coordination manifold across team compositions without retraining. HH-Bench is the right testbed because it spans 2–10 humanoids including heterogeneous teams, so it exercises exactly the composition shift the claim targets. The unmodified Heave-Ho baseline tests the sub-claim that the frozen policy alone drifts off the manifold under composition shift; if CIL's gain over it is concentrated on heterogeneous teams rather than uniform, the mechanism story holds. The EMOS-style baseline tests the sub-claim that plan-time capability routing is insufficient because it cannot enforce per-step physical coupling; a gap here separates descriptor-as-constraint from descriptor-as-retrieval. The per-robot diffusion baseline tests whether coupling is genuinely needed versus independent planning plus collision avoidance. The standard whole-body controller baseline tests whether the descriptor-driven per-composition calibration adds anything beyond textbook nullspace projection with the same analytic J_c; a gap here isolates the calibration contribution. The identity-J_c ablation is the decisive control for the projection: it keeps all plumbing and removes only the projection, so any residual gap between it and full CIL is attributable to the projection itself rather than to the descriptor or logging machinery. The analytic-J_c-only ablation is the decisive control for the calibration: it keeps the projection but removes the policy-conditioned correction, so any residual gap between it and full CIL is attributable to the calibration step. The invariant residual is the right primary metric because it is defined on the same calibrated constraint set the method claims to preserve, making it a direct rather than proxy test, and the per-row ACTIVE/INACTIVE diagnostic attributes gains to rows the policy actually learned versus rows the calibration corrected.

### Expected outcome and causal chain

**vs. Heave-Ho unmodified** — On a case where a 6-robot heterogeneous team lifts a shared object and one member is much heavier, the baseline produces references that drift off the coupled manifold because its post-training covered two identical humanoids and its tracking policy has no explicit coupling term. Our method instead projects each reference onto the nullspace of the calibrated coupling Jacobian, so we expect a large residual gap on heterogeneous teams of 4+ robots but near-parity on the 2-identical-humanoid teams where the baseline's training distribution already covers the active constraints. The per-row diagnostic should show the gain concentrated on rows marked INACTIVE in the training distribution (e.g., heavy-robot force balance).

**vs. EMOS-style robot-resume planner** — On a case where the planner selects a physically sensible action for each robot's embodiment but the resulting joint references violate shared-object pose agreement mid-execution, the baseline produces per-robot feasible but team-infeasible trajectories because capability descriptions only shape plan-time action selection. Our method instead uses the descriptors to construct constraint rows enforced at every control step, so we expect constraint violations to drop sharply on tasks requiring sustained physical coupling while remaining comparable on loosely-coupled navigation subtasks.

**vs. Per-robot diffusion planning (MMD composed independently)** — On a case where two robots must jointly maintain force balance while moving a heavy object through a narrow passage, the baseline produces collision-free but dynamically inconsistent trajectories because independent planners cannot represent the force-balance coupling. Our method instead folds force-balance into J_c^cal and filters references through its nullspace, so we expect a success-rate gap concentrated on the force-balance and shared-pose tasks, with parity on pure collision-avoidance tasks where independent planning suffices.

**vs. Standard whole-body controller with analytic J_c** — On a case where the analytic J_c includes a force-balance row that was never active during post-training on two identical humanoids, the baseline projects onto a nullspace that does not match the policy's learned manifold, so its references may be analytically feasible but policy-infeasible. Our method instead calibrates J_c against the frozen policy per composition, fitting a correction only to rows marked INACTIVE, so we expect a residual gap on heterogeneous teams that is attributable to the calibration step, with parity on homogeneous teams where all rows are ACTIVE and the correction is zero.

### What would falsify this idea

If CIL's gain over unmodified Heave-Ho is uniform across homogeneous and heterogeneous teams rather than concentrated where the predicted coupling-constraint failure occurs, or if the identity-J_c ablation matches full CIL, or if the analytic-J_c-only ablation matches full CIL (meaning the calibration step adds nothing), then the projection or the calibration is not doing the claimed work and the central mechanism story is wrong.

## References

1. Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration
