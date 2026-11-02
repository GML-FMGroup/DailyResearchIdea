# Regime-Indexed Preference Learning for Multi-Robot Teams with Joint Change-Point and Reward Estimation

## Motivation

Across the multi-robot learning-from-human-feedback literature (e.g., the frontier thesis 'Multi-Robot Learning from Human Feedback'), the supervisor's reward model is fit once and frozen, inheriting a stationarity assumption that treats any drift in intent as noise. This is a recurring structural failure: swarm-parameter optimization (Optimising autonomous robot swarm parameters) and behavior-discovery pipelines (Discovery and Deployment of Emergent Robot Swarm Behaviors) likewise assume a fixed objective during deployment, so when the supervisor's intent shifts the team's alignment silently diverges with no principled trigger for recalibration. The root cause is that the field has no object to estimate that represents non-stationary intent — there is no regime-indexed reward family, hence no bounded-error recalibration rule.

## Key Insight

If supervisor intent is piecewise-stationary, the change-point posterior and per-regime reward parameters are jointly identifiable from the same preference-comparison stream, and the posterior concentrates exactly when the cross-regime likelihood ratio becomes decisive — so the recalibration trigger is a derived quantity of the reward estimate itself, not an external drift detector.

## Method

**What it is.** RIPE (Regime-Indexed Preference Estimation) takes a stream of pairwise preference comparisons over team behavior rollouts from a decentralized multi-robot team and outputs, at every step, a posterior over the current change-point location, a per-regime reward parameter estimate, and a recalibration flag that fires when the posterior mass on a new regime crosses a threshold. It replaces the frozen reward model of the frontier thesis with a regime-indexed reward family whose regime boundaries are inferred, not assumed.

**Load-bearing assumption (stated explicitly).** The method assumes that the change-point posterior marginal `pi_new`, computed from Bradley-Terry preference likelihoods under a linear team-return model, is a calibrated and decisive signal of a true intent change: the cross-regime Bayes factor concentrates at the true boundary and stays quiet otherwise, so a threshold `tau` on `pi_new` yields a reliable recalibration trigger. This assumption is load-bearing because the entire recalibration mechanism is derived from `pi_new`; if `pi_new` is not calibrated, the trigger fires on Laplace-approximation scale artifacts or hazard-prior bias rather than on intent change. The method therefore includes an explicit gauge-fixing and calibration procedure (steps 2a–2c below) that makes the trigger scale-invariant and controls its false-alarm rate by construction on held-out stationary streams.

**How it works.**

```pseudocode
Input: preference stream D = {(traj_a, traj_b, label) at times t}, decay-free
Hyperparams: hazard lambda (expected change rate; default 1/200 comparisons),
             prior precision sigma0 (default 1.0),
             trigger threshold tau (posterior mass on post-change regime;
               calibrated per step 2c, default set so stationary false-alarm
               rate <= 0.05 per 100 comparisons),
             refit window W (comparisons retained per regime; default 500),
             max regimes K_max (default 8; prune regimes with posterior mass < 1e-3),
             coordinate-ascent tolerance eps (default 1e-4 on theta change),
             max coordinate-ascent iterations I_max (default 20)

Initialize: regime index k=0, change-point set C={}, reward params theta_0 ~ N(0, sigma0^-1 I)

for each new comparison (traj_a, traj_b, y) at time t:
    # 1. Regime-conditional preference likelihood (Bradley-Terry on team return)
    for each candidate regime r in {0..k} with params theta_r:
        p_r = sigmoid( (R_r(traj_a) - R_r(traj_b)) * y )   # y in {+1,-1}
        R_r(traj) = theta_r^T phi(traj), phi = [mean pairwise distance,
                    formation error, coverage] (fixed feature map)

    # 2a. Gauge fixing: normalize each theta_r to unit L2 norm before computing
    #     any cross-regime likelihood ratio, so the BT scale ambiguity does not
    #     enter the Bayes factor. (BT identifies theta only up to positive scale;
    #     unit-norm normalization fixes the gauge.)
    theta_r <- theta_r / ||theta_r||_2   for all r

    # 2b. Change-point posterior via explicit forward-backward recursion over
    #     the constant-hazard run-length process (not a closed-form sigmoid).
    #     Run-length posterior gamma_t(r) = P(run length = r at time t | D_{1:t}).
    #     Recursion (Fearnhead 2006 form):
    #       gamma_t(0)   = (1/Z) * lambda * sum_{r=0}^{k} gamma_{t-1}(r) * m_r(t)
    #       gamma_t(r+1) = (1/Z) * (1-lambda) * gamma_{t-1}(r) * m_r(t)
    #     where m_r(t) is the marginal likelihood of comparison t under regime r,
    #     integrated over the Laplace posterior (not the plug-in MAP):
    #       m_r(t) = ∫ p(y | traj_a, traj_b, theta) N(theta; mu_r, Sigma_r) dtheta
    #              ≈ p(y | traj_a, traj_b, mu_r) * |I + Sigma_r^{1/2} H_r Sigma_r^{1/2}|^{-1/2}
    #     with H_r the Hessian of the negative log-likelihood at mu_r (Laplace
    #     covariance term included explicitly in the likelihood ratio).
    #     Z normalizes gamma_t to sum to 1.
    #     pi_new = gamma_t(0)  (posterior mass on a boundary at t)

    # 2c. Trigger calibration: tau is set on a held-out stationary preference
    #     stream (same feature map, no change-points) so that the empirical
    #     false-alarm rate of (pi_new > tau) is <= 0.05 per 100 comparisons.
    #     This calibration is run once before deployment; tau is then fixed.

    # 3. Joint MAP update of (change-point set C, {theta_r}) via coordinate ascent
    #    with explicit convergence criterion:
    #    repeat up to I_max iterations:
    #      (a) assign each comparison in the window W to its MAP regime given C
    #      (b) update each theta_r by Laplace-approximated Bayesian logistic
    #          regression on its assigned comparisons (window=W)
    #      stop when max_r ||theta_r^{new} - theta_r^{old}||_2 < eps
    #    if pi_new > tau and k < K_max:
    #        k += 1; C = C union {t}; theta_k = theta_{k-1} (warm start)
    #    prune regimes with posterior mass < 1e-3 (renumber)

    # 4. Recalibration trigger is pi_new itself (posterior-derived, no drift detector)
    if pi_new > tau: emit RECALIBRATE(regime=k, theta_k)

    # 5. Team uses theta_{current regime} for decentralized policy update
```

**Complexity.** Per comparison: step 1 is O(k * d) for d=3 features and k<=K_max=8 regimes; step 2b forward-backward is O(k * W) with W=500; step 3 coordinate ascent is O(I_max * k * W * d) worst case, I_max=20. Total per-comparison cost is bounded by O(K_max * W * d * I_max) ≈ 8 * 500 * 3 * 20 = 240k flops, feasible for online use on a decentralized team's onboard compute.

**Why this design.** We chose to estimate the change-point process jointly with the per-regime reward rather than bolting on a separate CUSUM/Page-Hinkley drift detector because the two are coupled: the change-point posterior depends on cross-regime likelihood ratios, which depend on theta_r, which depends on which comparisons are assigned to which regime — a separate detector would freeze one side and lose the identifiability. We chose a constant-hazard piecewise-stationary prior over a sliding-window heuristic because the hazard gives a principled prior on regime duration and makes the posterior concentrate (rather than oscillate) when comparisons are decisive, accepting the cost that a misspecified lambda biases the inferred regime length. We chose a linear Bradley-Terry team-return model over a deep preference model because the Laplace posterior covariance is what feeds the change-point likelihood in closed form; a neural preference model would force approximate inference and break the trigger's calibration story. This is not a variant of standard preference-based RL with a drift detector (e.g., the frozen-reward pipeline in the Multi-Robot Learning from Human Feedback thesis) because the recalibration signal is the change-point posterior marginal, a quantity that does not exist in a single-reward model. The gauge-fixing (step 2a) and the held-out calibration of tau (step 2c) are included because BT preferences identify theta only up to scale, so without gauge-fixing the cross-regime Bayes factor would depend on the arbitrary scale of theta_r rather than on intent change; the calibration step controls the false-alarm rate by construction rather than relying on the hazard prior alone.

**Why it measures what we claim.** The change-point posterior marginal `pi_new` measures the latent regime boundary because the piecewise-stationary hazard model makes the posterior mass on a boundary equal to the Bayes factor between the current-regime and fresh-regime preference models, computed with the Laplace covariance term included and with theta normalized to unit norm to fix the BT gauge; this assumption fails when the supervisor's intent shifts gradually rather than abruptly, in which case `pi_new` reflects accumulated likelihood-ratio evidence that lags the true drift, or when the Laplace covariance misstates the fresh-regime evidence, in which case `pi_new` reflects the approximation's scale calibration rather than intent change. The per-regime reward `theta_r` measures the regime-conditional supervisor intent because Bradley-Terry comparisons identify theta up to scale under the assumption that the feature map phi spans the return; this assumption fails when two distinct intents induce identical phi-projections, in which case theta_r reflects the projection rather than the intent. The recalibration trigger `pi_new > tau` measures the decision to refit because it is the posterior probability that the current comparison stream is better explained by a new regime, with tau calibrated on held-out stationary streams to control the false-alarm rate; this assumption fails when the hazard lambda is misspecified, in which case the trigger reflects the prior's regime-duration bias rather than evidence of intent change.

## Contribution

(1) A regime-indexed preference-learning framework (RIPE) for multi-robot teams that jointly estimates a change-point posterior and per-regime reward parameters from a single preference stream, with the recalibration trigger derived as the posterior marginal rather than from an external drift detector. (2) A design principle: coupling the change-point likelihood to the reward posterior covariance makes the recalibration trigger concentrate exactly when cross-regime comparisons are decisive, eliminating the separate detector module that prior multi-robot human-feedback pipelines require. (3) A concrete Bayesian inference procedure (constant-hazard forward-backward over Bradley-Terry comparisons with Laplace-approximated per-regime posteriors) that is directly implementable on decentralized team rollouts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Synthetic non-stationary preference stream with scripted intent change-points (10 seeds, 5 change-points per stream, 2000 comparisons per stream) | Ground-truth regime boundaries enable direct falsification |
| Dataset (high-fidelity) | Swarm formation/coverage simulator (e.g., ARGoS or a ROS2 multi-robot formation task) with scripted preference labels from two alternating intent profiles (formation-tightness vs. coverage) | Tests downstream team alignment beyond synthetic boundary recovery |
| Primary metric | Change-point detection lag (comparisons between true boundary and first trigger) and boundary-F1 of inferred regimes | Directly measures the claimed mechanism, not downstream return |
| Secondary metric | False recalibrations per true boundary; downstream team alignment error (formation error or coverage gap) on the high-fidelity simulator | Tests trigger reliability and downstream impact |
| Baseline | Frozen single-reward Bradley-Terry model | Tests claim that regime indexing is necessary |
| Baseline | CUSUM/Page-Hinkley drift detector on preference log-likelihood | Tests claim that joint inference beats decoupled detection |
| Baseline | Sliding-window refit (heuristic forgetting; window sizes 100, 500, 1000) | Tests claim that hazard prior beats window heuristic |
| Baseline | BOCPD applied to the preference likelihood with fixed theta (run-length posterior over hazard process, theta frozen at pre-change estimate) | Isolates the joint (theta, boundary) coordinate-ascent as the novel contribution |
| Ablation-of-ours | RIPE with fixed misspecified hazard lambda (lambda set to 1/50 and 1/1000, vs. default 1/200) | Tests whether gains depend on hazard calibration |
| Ablation-of-ours | RIPE without gauge-fixing (theta not normalized) | Tests whether the BT scale ambiguity corrupts the trigger |
| Ablation-of-ours | RIPE with plug-in MAP likelihood (no Laplace covariance term in m_r(t)) | Tests whether the Laplace covariance term is necessary for calibration |

### Why this setup validates the claim

The central claim is that coupling change-point inference with per-regime reward estimation produces a calibrated recalibration trigger that a frozen reward model or a bolted-on detector cannot. A synthetic stream with scripted boundaries is the only setting where the true regime index is observable, so boundary-F1 and detection lag become a direct, falsifiable test of the mechanism rather than a proxy. The frozen-reward baseline isolates the sub-claim that a single theta cannot represent multiple intents; the CUSUM baseline isolates the sub-claim that decoupled detection loses identifiability because it freezes theta while testing for drift; the sliding-window baseline isolates the sub-claim that the constant-hazard prior concentrates the posterior where a window oscillates; the BOCPD-with-fixed-theta baseline isolates the sub-claim that the joint (theta, boundary) coordinate ascent — not the hazard recursion alone — is what produces the gain. The hazard-misspecification and gauge-fixing ablations are the crucial controls: if RIPE's advantage survives a deliberately wrong lambda and survives removing gauge-fixing, the gain comes from joint inference; if it collapses, the gain was prior tuning or scale calibration, which would undercut the design rationale. The high-fidelity simulator experiment tests whether the trigger changes downstream team alignment, since significance hinges on demonstrating the mechanism matters beyond synthetic boundary recovery. Boundary-F1 and lag are the right metrics because they measure the quantity the method actually outputs (posterior mass on boundaries) rather than a downstream return that could be confounded by policy optimization; the high-fidelity alignment error is a secondary check that the trigger has downstream consequence.

### Expected outcome and causal chain

**vs. Frozen single-reward Bradley-Terry model** — On a case where the scripted intent shifts from formation-tightness to coverage at a known step, the frozen baseline keeps fitting one theta to both segments, so its preferences become an average of two incompatible objectives and its implied policy drifts toward a compromise formation. Our method instead opens a new regime when the fresh-regime Bayes factor dominates, fitting separate thetas that each match their segment. We expect a large boundary-F1 gap concentrated exactly at the scripted change-points, with near-parity on the stationary segments before and after.

**vs. CUSUM/Page-Hinkley drift detector on preference log-likelihood** — On a case where the intent change is subtle (small rotation of theta) and comparisons are noisy, the CUSUM baseline fires late or oscillates because its reference distribution is frozen to the pre-change theta and cannot adapt its test statistic to the new regime's scale. Our method instead re-estimates theta jointly with the boundary posterior, so the likelihood ratio sharpens as evidence accumulates. We expect RIPE to show shorter detection lag on subtle shifts while matching CUSUM on large abrupt shifts, i.e. a gap that grows as the intent-rotation magnitude shrinks.

**vs. Sliding-window refit** — On a case where a regime is long and comparisons are sparse, the window baseline either forgets decisive early evidence (short window) or dilutes the new regime with stale comparisons (long window), producing a trigger that fires repeatedly or never. Our method instead retains per-regime comparisons up to W and lets the hazard prior concentrate posterior mass, so the trigger fires once per true boundary. We expect RIPE to show far fewer false recalibrations per true boundary, with the gap widening as regime duration grows relative to window length.

**vs. BOCPD with fixed theta** — On a case where the intent change is a rotation of theta that preserves the feature-projection scale, the BOCPD baseline's run-length posterior is computed against a frozen pre-change theta, so its Bayes factor understates the fresh-regime evidence and its trigger lags or misses the boundary. Our method instead re-estimates theta jointly with the boundary posterior via coordinate ascent, so the fresh-regime likelihood reflects the new theta. We expect RIPE to show shorter detection lag and higher boundary-F1 than BOCPD-with-fixed-theta specifically on subtle-rotation cases, isolating the joint-inference contribution from the hazard recursion itself.

**vs. RIPE without gauge-fixing** — On a case where the two regimes have different reward scales (e.g., coverage intent has larger feature magnitudes than formation intent), the unnormalized variant's cross-regime Bayes factor is dominated by the scale difference rather than the intent change, producing spurious triggers at scale transitions. We expect the gauge-fixed RIPE to show fewer false recalibrations and higher boundary-F1, confirming that the BT scale ambiguity corrupts the trigger without normalization.

**vs. RIPE with plug-in MAP likelihood** — On a case where comparisons are sparse within a regime, the plug-in MAP variant's marginal likelihood ignores the Laplace covariance term, so the fresh-regime evidence is overstated (the plug-in likelihood is higher than the integrated likelihood) and the trigger fires early. We expect the full RIPE with the Laplace covariance term to show better-calibrated trigger timing (lower false-alarm rate at matched detection lag), confirming that the covariance term is necessary for the calibration claim.

**High-fidelity simulator** — On the swarm formation/coverage task with alternating intent profiles, we expect RIPE's recalibration trigger to fire within 50 comparisons of each scripted intent switch, and the downstream team alignment error (formation error or coverage gap) to be lower than the frozen-reward baseline by at least 20% in the post-switch segments, demonstrating that the trigger has downstream consequence beyond synthetic boundary recovery.

### What would falsify this idea

If RIPE's boundary-F1 advantage over the CUSUM, sliding-window, and BOCPD-with-fixed-theta baselines is uniform across all segments rather than concentrated at scripted change-points and on subtle-shift cases, the claimed coupling mechanism is not what produces the gain. Equally, if the misspecified-hazard ablation or the no-gauge-fixing ablation matches full RIPE, the design rationale that the hazard prior and gauge-fixing drive posterior concentration and trigger calibration is wrong.

## References

1. Multi-Robot Learning from Human Feedback
2. Discovery and Deployment of Emergent Robot Swarm Behaviors via Representation Learning and Real2Sim2Real Transfer
3. Zespol: A Lightweight Environment for Training Swarming Agents
4. Optimising autonomous robot swarm parameters for stable formation design
5. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
6. Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks
7. Training Language Models to Self-Correct via Reinforcement Learning
8. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
