# Parameter-Robust Surprise: Invariance Training for Latent Physical Surprise Detection

## Motivation

The framework of When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models? shows that latent prediction error is only modestly aligned with physical possibility and is strongly driven by non-physical factors such as appearance and action perturbation; its frozen-representation analysis characterizes this but cannot fix it. Structurally, self-supervised latent prediction is trained to minimize error under the observed data distribution, so the error signal carries no constraint that it be invariant to physical parameters like mass, friction, and restitution — any parameter shift that changes dynamics is free to change the error as much as a genuine physical violation does. This is the recurring meta-level gap: surprise signals in video world models are evaluated only in-distribution, so their diagnostic meaning collapses under the parameter shifts that real embodied deployment guarantees.

## Key Insight

Surprise becomes a parameter-invariant physical-violation detector only if the model is trained under a domain-randomized parameter distribution whose support strictly contains the evaluation distribution, because the error signal is then forced to vary with kinematic-geometric inconsistency rather than with the specific dynamics parameters that generated the motion.

## Method

**What it is.** PRS (Parameter-Robust Surprise) trains a latent video world model on domain-randomized physical parameters and derives a surprise score from a *parameter-conditioned residual*: the model receives the rollout's physical parameters as a conditioning vector, and surprise is the prediction error after the model's parameter-conditioned predictor has been asked to explain the clip under the clip's own declared parameters. Inputs: video clip, declared parameter vector; outputs: a scalar surprise score and a parameter-attribution vector.

**Load-bearing assumption (stated explicitly).** The method assumes the parameter-conditioned predictor can be trained so that, for any physically possible clip generated with its declared parameters, `r_cond` is small (near-zero residual), while for physically impossible clips no parameter setting in the grid yields a small residual — i.e., the predictor's parameter input is causally and sufficiently used to fully explain all valid dynamics, so that the residual reflects only kinematic-geometric inconsistency and not model capacity or parameter-encoding failure. Because the training distribution is deliberately widened (`theta*(1+U(-0.5,0.5))`), rare-parameter underfitting is possible by construction, so `r_cond` can be confounded with capacity exactly on the shift clips the method must separate. The calibration and conditioning-fidelity checks below are specified to make this assumption testable and to keep the statistic from degenerating into a coverage/underfitting statistic.

**How it works.**

```pseudocode
# Phase 1: Domain-randomized pretraining
for each training clip x_{1:T} with simulator parameters theta ~ p_train:
    theta_train = theta * (1 + U(-0.5, 0.5))      # widen support beyond theta
    z_t = encoder(x_t)
    z_hat_{t+1} = predictor(z_t, theta_train)      # FiLM-conditioned on theta_train
    L = ||z_hat_{t+1} - z_{t+1}||^2 + lambda * L_align(z_hat, z)   # lambda=0.1
    update encoder + predictor

# Phase 1b: Per-parameter calibration baseline (held-out valid clips)
for each held-out valid clip x_{1:T} with declared theta:
    r_cond_cal(theta) = mean_t || predictor(z_t, theta) - z_{t+1} ||^2
sigma(theta) = std of r_cond_cal over held-out valid clips with declared theta
              (pooled over nearest-neighbor theta bins if fewer than 32 clips per exact theta)

# Phase 2: Surprise scoring at test time
for clip x_{1:T} with declared parameters theta_declared:
    # (a) parameter-conditioned residual
    r_cond = mean_t || predictor(z_t, theta_declared) - z_{t+1} ||^2
    # (b) parameter-marginalized residual: MIN over a fixed grid G of K=8 parameter draws
    r_marg = min_{theta' in G} mean_t || predictor(z_t, theta') - z_{t+1} ||^2
    # (c) parameter attribution: finite-difference sensitivity
    A = [ (r_cond(theta_declared + d*e_j) - r_cond(theta_declared)) / d ]_j   # d=0.05
    # (d) standardized excess (capacity-normalized difference)
    surprise = (r_cond - r_marg) / max(sigma(theta_declared), eps)   # eps=1e-6
    # (e) two-sided decision rule (see block D for sign derivation)
    is_violation = (surprise > tau_hi) AND (r_cond > tau_cond)
    # tau_hi from held-out in-distribution quantile (95th pct of surprise on valid clips)
    # tau_cond from held-out in-distribution quantile (95th pct of r_cond on valid clips)

# Conditioning-fidelity check (run before trusting any score)
# On held-out valid clips, require r_cond(theta_declared) < r_cond(theta') for theta' != theta_declared
# in at least 90% of clips; otherwise flag conditioning under-use and report the score as unreliable.
```

The key quantity is `(r_cond - r_marg) / sigma(theta_declared)`: the parameter-conditioned residual minus the best residual achievable by re-explaining the clip under alternative parameters, normalized by the predictor's own held-out residual scale at the declared parameters. If a clip is physically impossible, *no* parameter setting explains it, so both `r_cond` and `r_marg` are large and the difference is small; if a clip is merely from an out-of-distribution parameter regime, some grid draw explains it, so `r_marg` is small while `r_cond` is large, and the difference is large — the sign of the difference therefore separates parameter shift from physical violation, and the two-sided rule requires both a large standardized difference and a large absolute `r_cond` so that violations (small difference, large `r_cond`) are not missed.

**Why this design.** We chose a *conditioned predictor with a widened parameter support* over a parameter-agnostic predictor because a parameter-agnostic model's error is dominated by whichever parameters were most common in pretraining, so out-of-distribution parameters produce errors indistinguishable from violations; the cost is that the conditioning vector must be observable at test time, which we assume via simulator metadata or a parameter estimator. We chose the *standardized difference* `(r_cond - r_marg)/sigma(theta_declared)` over raw error because raw error already fails in the framework of When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models? (its own analysis shows error is driven by appearance and action perturbation); the cost is that the grid `G` must cover the shifted regime, so a shift outside the grid's support degrades the marginal estimate, and the per-parameter calibration requires held-out valid clips at or near each declared parameter. We chose a *finite-difference parameter attribution* over a learned attribution head because a learned head would re-introduce the appearance/physics coupling the framework documents, whereas finite differences directly query the predictor's parameter sensitivity; the cost is `K * d` extra forward passes, which is cheap for frozen encoders. An expert would not describe this as a variant of domain randomization alone (Tobin et al.) or of the frozen-representation probing of the framework, because neither produces the *standardized difference* statistic that requires the predictor to be conditioned on parameters and queried counterfactually over them.

**Why this measures what we claim.** `r_cond` measures *parameter-conditioned physical inconsistency* because it is the residual of a predictor that has been told the clip's own parameters, under the assumption that a parameter-conditioned predictor can explain any physically possible motion generated with those parameters; this assumption fails when the predictor is underfit on rare parameters, in which case `r_cond` reflects model capacity rather than physical violation. `r_marg` measures *explainability under alternative parameters* because it takes the best (minimum) residual over a parameter grid, under the assumption that the grid covers the shifted regime; this assumption fails when the true shift lies outside `G`, in which case `r_marg` overestimates explainability and the difference `r_cond - r_marg` reflects grid coverage rather than parameter shift. The standardized excess `(r_cond - r_marg)/sigma(theta_declared)` measures *violation-relative-to-expected-error* because it divides by the predictor's own held-out residual scale at the declared parameters, under the assumption that `sigma(theta_declared)` is estimated from enough held-out valid clips (≥32 per exact theta or nearest-neighbor bin); this assumption fails when calibration clips are scarce, in which case `sigma` is noisy and the standardized excess is unstable. The finite-difference attribution `A` measures *parameter sensitivity of the error* because it holds the clip fixed and perturbs only the conditioning vector, under the assumption that the predictor's parameter input is causally used; this assumption fails when the predictor ignores its conditioning (conditioning collapse), in which case `A` is near zero and the difference statistic degenerates to raw error — we detect this failure by the conditioning-fidelity check (requiring `r_cond(theta_declared) < r_cond(theta')` on ≥90% of held-out valid clips) rather than by an `||A||` threshold alone. The thresholds `tau_hi` and `tau_cond` measure *in-distribution surprise calibration* because they are set from held-out in-distribution quantiles (95th percentile), under the assumption that in-distribution clips contain no violations; this assumption fails when the held-out set is contaminated, in which case the thresholds are inflated and the detector under-fires.

## Contribution

(1) A parameter-conditioned latent world model with a widened training parameter support and a surprise statistic defined as the difference between the parameter-conditioned residual and the parameter-marginalized residual, which separates physical violation from parameter shift. (2) A finite-difference parameter-attribution diagnostic that detects conditioning collapse and gates the surprise score, providing a concrete failure mode check for parameter-conditioned world models. (3) An evaluation protocol on IntPhys2 and Physion++ with held-out out-of-distribution parameter ranges (mass, friction, restitution) that reports surprise separation under parameter shift rather than only in-distribution.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Physion++ with parameter metadata, plus IntPhys2 violation clips | Provides declared physics parameters and impossible events |
| Primary metric | AUC separating parameter-shift from physical-violation clips | Directly tests the sign-of-difference claim |
| Baseline | Raw latent prediction error (V-JEPA2 probe) | Tests whether conditioning-difference beats unconditioned error |
| Baseline | Parameter-agnostic predictor residual | Tests whether conditioning removes OOD-parameter confound |
| Baseline | r_cond alone (no marginal subtraction) | Tests whether subtraction is what isolates violation |
| Baseline | Conditional-vs-marginal likelihood ratio on a non-physics latent model | Tests whether physics-conditioning is the novel ingredient |
| Ablation-of-ours | PRS with conditioning-collapse detector disabled | Tests whether attribution gate is load-bearing |
| Ablation-of-ours | PRS with mean-over-grid instead of min-over-grid r_marg | Tests whether best-explanation marginalization is load-bearing |
| Ablation-of-ours | PRS with estimated parameters (regression head) instead of declared | Tests robustness to missing metadata |

### Why this setup validates the claim

The central claim is that `(r_cond - r_marg)/sigma(theta_declared)` separates a *physical violation* from a mere *parameter shift*, and that this separation is what raw prediction error cannot do. The design is falsifiable because each baseline isolates one sub-claim. Raw latent prediction error (the framework's own representation) tests whether the standardized difference beats the quantity the framework showed is appearance/action-driven; if raw error matches PRS, the difference adds nothing. The parameter-agnostic predictor tests whether conditioning on declared parameters is necessary; if an unconditioned model already separates the two regimes, conditioning is superfluous. The `r_cond`-alone baseline tests whether the marginal subtraction is the active ingredient or whether conditioned error alone suffices. The conditional-vs-marginal likelihood ratio on a non-physics latent model tests whether the physics-conditioning is the novel ingredient rather than the subtraction itself. The ablation (disabling the conditioning-collapse gate) tests whether the attribution diagnostic is load-bearing rather than decorative. The mean-vs-min ablation tests whether the best-explanation marginalization matches the 'no parameter explains it' claim. The estimated-parameter ablation tests the realistic no-metadata regime. AUC over parameter-shift versus violation clips is the right metric because the claim is precisely about *separation of two categories* along the surprise axis — not absolute error magnitude, which the framework already showed is confounded. A method that scores low absolute error but fails to order the two categories would be correctly judged a failure by this metric, and a method that orders them is exactly what the causal story predicts.

### Expected outcome and causal chain

**vs. Raw latent prediction error (V-JEPA2 probe)** — On a clip where a ball is launched with an out-of-distribution but physically valid restitution, the raw-error baseline produces a large residual because its predictor was dominated by common restitution values and cannot explain the fast rebound, so it flags a valid clip as surprising. Our method instead computes a small `r_marg` because some grid draw of restitution explains the rebound, and the standardized difference `(r_cond - r_marg)/sigma(theta_declared)` stays near zero. We expect a large AUC gap on the parameter-shift subset, with parity on the violation subset where both methods fire.

**vs. Parameter-agnostic predictor residual** — On a violation clip where a block passes through a wall, the parameter-agnostic predictor produces an error that is *smaller* than on the OOD-but-valid clip, because its error is dominated by appearance and motion statistics rather than physical consistency. Our method instead produces a large `r_cond` (no parameter explains penetration) and a large `r_marg` (grid draws also fail), so the standardized difference is small but the raw conditioned residual is high — and the reverse ordering holds on the shift clip. We expect the two categories to be ordered correctly by PRS but inverted or overlapping for the agnostic baseline.

**vs. r_cond alone (no marginal subtraction)** — On an OOD-parameter clip, `r_cond` alone is large because the predictor is underfit at that parameter, so it fires on a valid clip exactly as it would on a violation. Our method subtracts `r_marg`, which is small for this clip because a nearby grid draw explains it, driving the standardized difference down. We expect `r_cond` alone to show high false-positive rate on the shift subset while PRS concentrates its detections on the violation subset, yielding a visible separation gap that is absent without subtraction.

**vs. Conditional-vs-marginal likelihood ratio on a non-physics latent model** — On both shift and violation clips, the non-physics latent model's conditional-vs-marginal ratio is driven by appearance and action statistics rather than physical parameters, so its AUC is near chance. Our method's physics-conditioned predictor produces a standardized difference that orders the two categories because the conditioning vector carries the physical parameters. We expect PRS to outperform this baseline by a clear margin on both subsets, confirming that physics-conditioning, not the subtraction operation, is the novel ingredient.

**vs. PRS with mean-over-grid instead of min-over-grid r_marg** — On a violation clip where a block passes through a wall, the mean-over-grid `r_marg` is inflated by grid draws that also fail, so the difference `r_cond - r_marg` is artificially small and the detector misses the violation. The min-over-grid `r_marg` correctly reflects the best explanation, keeping the difference small but the absolute `r_cond` large, so the two-sided rule fires. We expect the min-over-grid variant to have higher AUC on the violation subset.

**vs. PRS with estimated parameters (regression head) instead of declared** — On clips where the parameter regression head has estimation error, `r_cond` is computed under a noisy `theta_declared`, inflating the residual and degrading separation. We expect a measurable AUC drop relative to declared-parameter PRS, bounding the significance claim in the no-metadata regime.

### What would falsify this idea

If the AUC gain of PRS over raw error is uniform across the parameter-shift and violation subsets rather than concentrated on the shift subset (where the marginal subtraction is supposed to act), the central claim that the standardized difference separates the two regimes is wrong. Equivalently, if conditioning-collapse ablation leaves performance unchanged, the attribution gate is not doing the work the causal story assigns it; and if the conditioning-fidelity check fails on held-out valid clips (r_cond(theta_declared) not lower than r_cond(theta') for theta' != theta_declared in ≥90% of clips), the load-bearing assumption of causal conditioning use is violated.

## References

1. When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models?
