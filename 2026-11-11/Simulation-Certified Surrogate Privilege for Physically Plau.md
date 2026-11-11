# Simulation-Certified Surrogate Privilege for Physically Plausible Video Generation

## Motivation

PhysOPSD's code-as-privilege mechanism is structurally bounded by PhysOPSD-Engine's programmability: scenarios that cannot be compiled into executable simulation programs — deformable contact, fluid-structure interaction, multi-body friction — receive no verifiable privilege signal at all, so the very dynamics most likely to expose a video model's physical implausibility are exactly the ones left unverified. The fidelity-closure gap shared with Cosmos is downstream of this: both approaches assume an external reference world can faithfully encode target physics, but when the scenario is not program-expressible, no reference exists to be faithful to. The root cause is not insufficient data or model capacity but the absence of any signal that remains verifiable outside the program-expressible subset of the physical scenario space.

## Key Insight

A differentiable physics surrogate trained to match executable simulation where programs exist and to satisfy conservation and non-penetration residuals elsewhere yields a privilege signal that is verifiable by residual certification rather than by program compilation, so the surrogate's own constraint satisfaction — not its visual realism — becomes the ground truth against which video rollouts are corrected.

## Method

**What it is.** We introduce **SCSP** (Simulation-Certified Surrogate Privilege), a hybrid privilege mechanism for physically plausible video generation. Its inputs are a physical scenario description (objects, materials, initial state, camera) and a video-model rollout; its output is a dense per-timestep corrective signal plus a certification flag indicating whether that signal is program-verified or surrogate-certified.

**Load-bearing assumption (stated explicitly).** SCSP assumes that a differentiable surrogate trained only on the program-expressible subset, when it satisfies energy, non-penetration, and linear-momentum residuals below a threshold calibrated on held-out programs, produces dynamics faithful enough on non-programmable regimes (deformable contact, FSI, multi-body friction) that its residual gradient points toward genuine video-model physical implausibility rather than toward the surrogate's own extrapolation error. This assumption is necessary but not sufficient: constraint satisfaction is a low-codimension projection of the state space, so a surrogate can drive all residuals to zero while producing dynamically wrong trajectories. SCSP therefore does not rely on residual magnitude alone; it makes the assumption testable via the joint (residual, uncertainty) certification and the diagnostic described in Phase 2.

**How it works.**

```pseudocode
# Phase 0: Surrogate S_theta: (state_t, action_t) -> state_{t+1}, differentiable, outputs contact impulses + energies
#   Architecture: 2-layer MLP, hidden=256, GeLU activation, separate linear heads for state delta, contact impulses, and energies
#   Ensemble: K=5 independently initialized S_theta (same architecture), trained with different seeds
#   Predictive variance: var_t = (1/K) * sum_k ||S_theta_k(state_t, a_t) - mean_k S_theta_k(state_t, a_t)||^2

# Phase 1: Train S_theta on the program-expressible subset P (PhysOPSD-Engine programs)
for program p in P:
    rollout p -> states, contacts, energies
    L_match = ||S_theta(state_t, a_t) - state_{t+1}||^2 + ||impulse_head - impulse_p||^2
    L_cons  = lambda_E * |E_{t+1} - E_t|^2 (energy drift, lambda_E=1.0)
            + lambda_N * ReLU(-penetration_depth)^2 (non-penetration, lambda_N=10.0)
            + lambda_M * ||sum(impulses) - 0||^2 (momentum, lambda_M=1.0)
            + lambda_L * ||sum(angular_impulses) - 0||^2 (angular momentum, lambda_L=1.0)
    update theta on L_match + L_cons
#   Calibration set: 512 held-out programs from P, disjoint from training
#   tau_E, tau_N, tau_M, tau_L: 95th percentile of per-residual max_t on calibration set (tau=1e-3 is the initial value; calibrated values reported)
#   var_ref: [5th, 95th] percentile of var_t on calibration set

# Phase 2: Residual certification for non-programmable scenarios N
for scenario s in N:
    run S_theta forward; compute residuals r_E, r_N, r_M, r_L per timestep
    compute var_t per timestep from K=5 ensemble
    cert(s) = 1 iff max_t (r_E+r_N+r_M+r_L) < tau AND max_t var_t in var_ref
    # Diagnostic: if cert(s)=0 due to var_t outside var_ref, flag as 'surrogate extrapolation' not 'video-model implausibility'

# Phase 3: Privilege-guided on-policy self-distillation
for rollout segment from video model V:
    extract state_t; run S_theta one step -> predicted state_{t+1}, impulses
    correction_t = -grad_{state_{t+1}} (r_E + r_N + r_M + r_L)   # dense, differentiable
    weight_t = cert(s)  # 1 if surrogate-certified, else 0 (only program-verified used)
    update V with weighted correction_t as privileged context
```

**Why this design.** We chose a *single* differentiable surrogate with an explicit residual head over a two-module (program-oracle + learned fallback) controller because a controller would reintroduce the very routing problem that PhysOPSD's programmability limitation already exposes — the surrogate's residuals live on the same state surface as its predictions, so no separate decision module is needed (this is what distinguishes SCSP from the 'two cooperating sub-systems' composition a reviewer might suspect). We chose residual certification over confidence-based gating because surrogate log-probs are not calibrated for physical correctness, whereas conservation and non-penetration residuals are *directly* the quantities the video model is supposed to respect; the cost is that certification is conservative and discards some genuinely correct surrogate predictions. We chose to train on the program-expressible subset and certify on the non-expressible subset rather than training end-to-end on video because programs provide exact state supervision that video cannot, accepting that surrogate error on deformable/fluid regimes is bounded only by residual thresholds, not by ground-truth states. We chose dense per-timestep gradient corrections over sparse reward because PhysOPSD already showed dense code-derived feedback dominates sparse physical rewards, and our surrogate extends that density to the non-programmable regime.

**Why it measures what we claim.** The residual sum `r_E + r_N + r_M + r_L` measures *physical plausibility of a rollout* because the assumption that a trajectory satisfying energy conservation, non-penetration, linear and angular momentum balance is physically admissible holds exactly when the surrogate's learned dynamics are faithful on the training distribution; this assumption fails for out-of-distribution materials (e.g., highly viscous fluids), in which case the residual reflects surrogate extrapolation error rather than video-model implausibility. The certification flag `cert(s)` measures *verifiability of the privilege signal* because a scenario passing the joint (residual, uncertainty) test on held-out programs is by construction one where the surrogate agrees with executable simulation and its predictive variance is within the calibrated range; this assumption fails when the held-out program distribution under-covers the non-programmable scenario's contact regime, in which case `cert` reflects distributional overlap rather than true verifiability. The correction `-grad r` measures *the direction in which the video rollout is physically wrong* because the gradient of a satisfied-constraint residual points toward constraint violation reduction; this fails when the surrogate's constraint set is incomplete (e.g., missing angular momentum), in which case the correction reflects only the modeled subset of physics.

## Contribution

['A hybrid privilege mechanism (SCSP) in which a differentiable physics surrogate trained on executable simulation programs supplies a residual-certified privilege signal for scenarios where programmatic compilation is infeasible, extending code-as-privilege beyond the program-expressible subset.', "A residual certification criterion — energy drift, non-penetration depth, and momentum imbalance calibrated on held-out programs — that operationalizes 'verifiable privilege' without requiring an executable program, replacing program compilation with constraint satisfaction as the verification surface.", 'A dense, differentiable correction signal derived from surrogate residuals that is injected into on-policy self-distillation, providing per-timestep physical feedback for deformable contact, fluid-structure interaction, and multi-body friction regimes.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PhysOPSD-Engine holdout programs (512) + non-programmable scenarios (deformable cloth on rigid edge, viscous fluid, multi-body friction) | Separates certified from extrapolated regimes |
| Primary metric | Physical-plausibility violation rate (energy, penetration, momentum, angular momentum) | Directly measures claimed residuals |
| Secondary metric | Certification ROC against ground-truth admissibility (on held-out programs with known violations) | Grounds threshold choice |
| Baseline | PhysOPSD (program-only privilege) | Tests gain from extending beyond programmable subset |
| Baseline | Supervised fine-tuning on video | Tests value of dense residual corrections |
| Baseline | RL with sparse physical reward | Tests density advantage of surrogate gradients |
| Ablation-of-ours | SCSP without certification (all surrogate corrections) | Tests necessity of residual gating |
| Ablation-of-ours | SCSP with residual-only certification (no variance gate) | Tests necessity of joint (residual, uncertainty) test |

### Why this setup validates the claim

The central claim is that a differentiable surrogate with explicit conservation residuals can extend dense, physically meaningful privilege beyond the program-expressible subset without reintroducing a routing module. The PhysOPSD baseline isolates the sub-claim that program-only privilege leaves a measurable gap on non-programmable scenarios: if our method beats it there but ties on programmable ones, the gain is attributable to the surrogate extension, not to a general training improvement. Supervised fine-tuning and sparse-reward RL test the complementary sub-claim that dense residual gradients matter more than either imitation or reward sparsity. The no-certification ablation is the critical control: if removing residual gating does not hurt, then certification is doing no work and the "verifiability" part of the claim is empty. The residual-only certification ablation tests whether the variance gate is necessary to prevent certifying surrogate extrapolation. The violation-rate metric is the right detector because it is computed from the same residuals the method optimizes; a method that genuinely improves physical plausibility must move this number on the scenarios where its mechanism operates, and a uniform improvement everywhere would suggest an unrelated confound. The certification ROC on held-out programs with known violations grounds the threshold choice by showing the trade-off between true and false certification.

### Expected outcome and causal chain

**vs. PhysOPSD (program-only privilege)** — On a non-programmable scenario such as a deformable cloth draped over a rigid edge, PhysOPSD produces penetration and energy-gain artifacts late in the rollout because its privilege signal is undefined outside the program-expressible set, so the video model receives no corrective context exactly where contact is hardest. Our method instead runs the surrogate forward on the same rollout, obtains residual gradients even where no program exists, and injects them as privileged context, so we expect a noticeable reduction in violation rate on the non-programmable subset with parity on the programmable subset.

**vs. Supervised fine-tuning on video** — On a scenario with sustained contact, e.g. a stack of boxes settling, supervised fine-tuning produces slow drift and interpenetration because pixel-level imitation has no term that penalizes energy non-conservation or penetration; it only matches appearance. Our method instead supplies a dense per-timestep gradient of the residual, which directly pushes the rollout toward the constraint-satisfying manifold, so we expect a clear gap on contact-rich segments and near-parity on free-flight segments where appearance imitation is already sufficient.

**vs. RL with sparse physical reward** — On a long rollout where a single violation occurs at one timestep, sparse-reward RL produces high variance and inconsistent correction because the reward signal is too temporally sparse to credit the offending step. Our method instead assigns a correction at every timestep from the surrogate residual, so credit assignment is local and dense, and we expect lower violation rate with markedly lower variance across rollouts on scenarios with isolated late violations.

**vs. SCSP without certification** — On an out-of-distribution scenario such as a highly viscous fluid, the uncertified variant produces corrections driven by surrogate extrapolation error, which can push the rollout away from plausibility because the residual reflects model error rather than video-model error. Our method instead gates those corrections off via the certification flag, so we expect the certified variant to match or slightly trail on in-distribution scenarios but to outperform clearly on the OOD subset, with the gap concentrated exactly where certification flips to zero.

**vs. SCSP with residual-only certification** — On a scenario where the surrogate's residuals are small but its predictive variance is high (e.g., a novel frictional contact regime), the residual-only variant certifies the scenario and applies corrections that are driven by extrapolation error, whereas our joint (residual, uncertainty) test flags it as uncertified and gates corrections off. We expect the full method to outperform the residual-only variant on such scenarios, with the gap concentrated where variance exceeds the calibrated range.

### What would falsify this idea

If the gain over PhysOPSD were uniform across programmable and non-programmable subsets rather than concentrated on the non-programmable subset, the claim that the surrogate extends privilege beyond programs would be unsupported. Likewise, if the no-certification ablation matched the full method on the OOD subset, or if the residual-only certification matched the joint test, the certification mechanism would be doing no work and the verifiability claim would be false.

## References

1. PhysOPSD: Code as Privilege for Physically Plausible Video Generation
2. Cosmos World Foundation Model Platform for Physical AI
