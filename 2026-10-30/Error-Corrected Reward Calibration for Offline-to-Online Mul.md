# Error-Corrected Reward Calibration for Offline-to-Online Multimodal Agent Policies

## Motivation

Step-level reward benchmarks such as Agent-RewardBench score reward models against static manual labels, implicitly treating those labels as a sufficient proxy for agent capability. But once the policy acts, the state distribution shifts away from the label distribution and the world model's next-state prediction errors compound, so a reward score that was calibrated off-policy becomes an invalid proxy exactly where the policy diverges. Across the reward-modeling and model-based-planning trees (e.g., WebDreamer's reliance on LLM-predicted next states), no mechanism ties step-level reward scores to the on-policy state distribution induced by the policy's own actions while carrying the world-model prediction-error term into the score.

## Key Insight

A step-level reward score is a valid closed-loop proxy for policy improvement if and only if it is evaluated under the policy-induced state distribution and corrected by the world-model's next-state prediction error through an identity that makes the corrected score equal to the on-policy reward expectation up to that error, so calibration degrades gracefully with prediction error rather than silently off-policy.

## Method

**What it is.** ECRC (Error-Corrected Reward Calibration) takes as input a step-level reward model r_phi, a world model f_psi with per-step next-state prediction error, and a target policy pi, and outputs a calibrated step reward r_hat(s,a) plus a scalar calibration certificate kappa that bounds how far r_hat is from the true on-policy reward expectation given the measured world-model error.

**Load-bearing assumption (stated explicitly).** The correction term lambda * sigma(s,a) * grad_a r_phi(s,a)^T * delta_a is assumed to be a monotone and sign-correct proxy for the true on-policy calibration gap, so that kappa = |r_hat - r_phi| / (sigma + tau) bounds the actual gap and downweighting by exp(-kappa) removes miscalibrated steps rather than well-calibrated ones. This assumption holds when r_phi is locally Lipschitz with bounded second derivative along the error direction; it fails near sharp reward-model decision boundaries where the Taylor remainder O(||delta_a||^2 * sup|Hessian|) dominates. To make the assumption testable rather than asserted, we validate kappa against a held-out on-policy reward estimate (see step 4b) and report the empirical discrepancy.

**How it works.**

```pseudocode
Input: reward model r_phi, world model f_psi, policy pi, rollouts D_on from pi,
       error estimator sigma (per-step next-state residual), horizon H, batch B=256,
       certificate-validation rollouts M=8 per flagged step
Output: calibrated step reward r_hat, certificate kappa

1. On-policy rollouts: sample B=256 trajectories from pi in the world model;
   record (s_t, a_t, s_{t+1}, r_phi(s_t,a_t), eps_t = ||s_{t+1} - f_psi(s_t,a_t)||).
2. Error model: fit sigma(s_t,a_t) = E[eps_t | s_t,a_t] with a small MLP
   (2 hidden layers, 128 units, GeLU activation, Adam lr=1e-3, batch=64, 2000 steps)
   on the rollout buffer; hold out 20% for validation.
3. Calibration identity per step:
     r_hat(s_t,a_t) = r_phi(s_t,a_t) + lambda * sigma(s_t,a_t) * grad_a r_phi(s_t,a_t)^T * delta_a
   where delta_a is the finite-difference action perturbation defined as
     delta_a = (grad_a sigma(s_t,a_t) / ||grad_a sigma(s_t,a_t)||) * h,
   with step size h=1e-3 and grad_a sigma computed by central finite differences
   (perturb each action dimension by +/-h, 2*d_a forward passes).
   lambda in [0,1] annealed linearly 1.0 -> 0.1 over the first 10k policy updates.
4. Certificate: kappa = max_t |r_hat - r_phi| / (sigma_t + tau), tau=1e-3;
   flag any step with kappa > kappa_max=2.0 as uncalibrated and downweight it
   by w_t = exp(-kappa_t) in the downstream policy objective.
4b. Certificate validation (empirical check of the load-bearing assumption):
   for each flagged step (kappa > kappa_max), draw M=8 Monte-Carlo rollouts from
   (s_t, a_t) in the world model and compute the empirical on-policy reward
   expectation r_mc = (1/M) * sum_{m=1}^M r_phi(s_t, a_t, s_{t+1}^{(m)}).
   Record the discrepancy d_t = |r_hat(s_t,a_t) - r_mc|. Report mean d_t on the
   flagged subset; if mean d_t exceeds 0.5 * sigma_t, mark kappa as a heuristic
   reliability score rather than a bound in the results table.
5. Return r_hat and kappa; downstream PPO/DPO consumes w_t * r_hat.
```

**Why this design.** We chose to condition the correction on the *policy-induced* state distribution (step 1) rather than on the offline label distribution because the label distribution is exactly the off-policy proxy that Agent-RewardBench and similar benchmarks already provide, and reusing it would reproduce the meta-gap; the cost is that we must roll out the policy in the world model, which is expensive and amplifies world-model error, so we compensate by fitting an explicit per-step error model (step 2) instead of assuming homoscedastic error. We chose a first-order Taylor correction along the predicted-error gradient (step 3) over a full second-order or learned correction because the identity is exact to first order in the perturbation and the residual is absorbed into the certificate, accepting that for large world-model error the correction under-corrects and the certificate must downweight those steps. We chose to expose the certificate kappa as a per-step weight (step 4) rather than a global scalar because calibration is heterogeneous across states, accepting the cost of a second hyperparameter kappa_max that must be tuned per domain. We added step 4b not as a new mechanism but as a direct empirical check of the load-bearing assumption: it measures whether kappa actually tracks the on-policy calibration gap on the flagged subset, so the certificate's validity is reported rather than asserted. This is not a Process Reward Model or VinePPO variant: those use Monte-Carlo rollouts to *estimate* step values, whereas ECRC keeps the existing reward model fixed and instead *corrects its calibration* using the world-model error term, which is a quantity PRM/VinePPO never measure.

**Why it measures what we claim.** The on-policy rollout distribution (step 1) measures *on-policy state coverage* because sampling trajectories from pi in the world model is equivalent to the policy-induced distribution under the assumption that the world model is an unbiased simulator; this assumption fails when world-model error is state-dependent and correlated with action, in which case the rollouts reflect the world model's bias rather than the true policy distribution. The per-step error model sigma (step 2) measures *world-model next-state prediction error* because the residual ||s_{t+1} - f_psi(s_t,a_t)|| is equivalent to the prediction error under the assumption that observed next states are noiseless; this assumption fails under stochastic environments, in which case sigma reflects aleatoric environment noise rather than model error. The correction term lambda*sigma*grad_a r_phi^T delta_a (step 3) measures *reward-score sensitivity to world-model error* because the first-order Taylor expansion along the error-gradient direction is equivalent to the change in reward score under the actual next-state perturbation under the assumption that r_phi is locally Lipschitz with bounded second derivative; this assumption fails near sharp reward-model decision boundaries, in which case the correction reflects local curvature rather than the global calibration gap. The certificate kappa (step 4) measures *per-step calibration validity* because the normalized correction magnitude is equivalent to the calibration gap under the assumption that the correction is monotone in the true gap; this assumption fails when the reward model is already miscalibrated off-policy, in which case kappa flags residual miscalibration rather than world-model-induced error alone. Step 4b directly measures *the discrepancy between kappa and the on-policy reward expectation* because the Monte-Carlo estimate r_mc is equivalent to the true on-policy reward expectation under the assumption that M=8 rollouts suffice to average out world-model stochasticity; this assumption fails when the world model has high variance, in which case d_t reflects MC noise rather than certificate error, and we report the MC standard error alongside d_t.

## Contribution

(1) A prediction-error-corrected calibration identity that couples step-level reward scores to the on-policy state distribution and the world-model next-state error term, realized as the ECRC procedure with a per-step calibration certificate. (2) A per-step error model and certificate-based reweighting scheme that downweights steps where reward calibration is invalid, providing a concrete design principle for offline-to-online reward validity. (3) An evaluation protocol that reports reward calibration under policy-induced rollouts rather than static labels, exposing where static step-label benchmarks such as Agent-RewardBench overstate reward validity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Agent-RewardBench step-level tasks (held-out split) plus on-policy rollouts from pi in a WebDreamer-style LLM world model on the WebArena subset | Agent-RewardBench provides step-level reward labels across perception, planning, and safety; the on-policy rollouts provide the policy-induced state distribution the method targets |
| Primary metric | Calibration error (ECE) of step rewards against on-policy reward estimates from M=8 Monte-Carlo rollouts in the world model (not static labels) | Directly measures the on-policy calibration gap the method claims to reduce |
| Secondary metric | ECE against static Agent-RewardBench labels (off-policy reference) | Tests whether on-policy calibration gains transfer to the benchmark's original target |
| Baseline | Uncalibrated r_phi (raw reward model) | Tests whether correction reduces miscalibration at all |
| Baseline | Global temperature scaling of r_phi (single scalar T fit on 512 held-out steps) | Tests whether a uniform scalar suffices vs. per-step certificate |
| Baseline | PRM/VinePPO-style Monte-Carlo step value (M=8 rollouts per step) | Tests whether MC estimation beats error-corrected calibration |
| Ablation-of-ours | ECRC without certificate weighting (w_t=1) | Isolates contribution of per-step downweighting |
| Ablation-of-ours | ECRC with kappa replaced by |r_hat - r_phi| / (sigma + tau) without the exp(-kappa) downweight (i.e., report kappa but do not use it) | Tests whether kappa itself is informative even if not used as a weight |
| Hyperparameter grid (pre-registered) | lambda in {1.0, 0.5, 0.1} annealed linearly over 10k updates; kappa_max in {1.0, 2.0, 4.0}; tau in {1e-4, 1e-3, 1e-2}; h=1e-3 fixed | Bounds tuning burden and ensures reproducibility |

### Why this setup validates the claim

The central claim is that conditioning reward calibration on measured world-model error, and exposing a per-step certificate, reduces on-policy reward miscalibration beyond what a global post-hoc correction or a Monte-Carlo step-value estimator can achieve. Agent-RewardBench provides step-level reward labels across perception, planning, and safety, but those labels are off-policy; to test the on-policy claim we additionally compute ECE against on-policy reward estimates from M=8 Monte-Carlo rollouts in the WebDreamer-style world model on the WebArena subset, which is the closest available proxy to the true on-policy reward expectation. The raw r_phi baseline establishes the initial miscalibration; global temperature scaling tests the sub-claim that calibration is heterogeneous across states and cannot be fixed by a single scalar; PRM/VinePPO tests the sub-claim that correcting a fixed reward model using world-model error is preferable to replacing it with MC estimates. The no-certificate ablation isolates whether the per-step weight w_t is doing real work or whether the Taylor correction alone accounts for any gain. The kappa-informative ablation tests whether the certificate carries signal even when not used as a weight. If ECRC's advantage is real, it should appear as lower ECE concentrated on steps where the error model sigma is high, not uniformly across all steps, and the step-4b discrepancy d_t should be small on the flagged subset (mean d_t < 0.5 * sigma_t), confirming that kappa tracks the on-policy gap rather than being a heuristic sensitivity score.

### Expected outcome and causal chain

**vs. Uncalibrated r_phi** — On a planning step where the world model's next-state prediction is off in a direction that shifts the reward-model score, the raw r_phi produces a step reward that is systematically biased because it never observes the world-model residual. Our method adds the first-order correction along the error-gradient direction and downweights steps with large kappa, so we expect a clear ECE reduction on the high-sigma subset of planning steps, with near-parity on low-sigma perception steps where the correction term is small. The step-4b check should show mean d_t < 0.5 * sigma_t on flagged steps, confirming the correction moves r_hat toward the MC on-policy estimate.

**vs. Global temperature scaling** — On a safety-critical step where the reward model is overconfident but the world model is accurate, temperature scaling shrinks all rewards uniformly and cannot fix a step-specific bias because it has no per-state signal. Our method's certificate kappa is computed per step from the local error model, so it flags exactly those overconfident steps and downweights them, yielding lower ECE on the safety subset while temperature scaling remains at baseline; we expect the gap to be concentrated where sigma is state-dependent, not where it is constant. The kappa-informative ablation should show that reporting kappa without downweighting yields intermediate ECE, isolating the weight's contribution.

**vs. PRM/VinePPO-style Monte-Carlo step value** — On a step deep in a trajectory where MC rollouts from that step have high variance due to stochastic environment transitions, the MC estimator produces a noisy step value because it averages over many sampled futures rather than correcting the existing reward model. Our method instead keeps r_phi fixed and corrects it using the measured world-model error, so we expect lower ECE on deep-horizon steps where MC variance is largest, with the two methods converging on shallow steps where MC variance is small. Because our on-policy ECE target is itself an M=8 MC estimate, we report the MC standard error and only claim a gap when it exceeds 2 standard errors.

**vs. ECRC without certificate weighting** — On a step near a sharp reward-model decision boundary, the Taylor correction under-corrects because the local Lipschitz assumption fails, and without the certificate weight that step's residual error propagates into the policy objective. Our full method flags it via kappa > kappa_max and downweights it, so we expect the ablation to show higher ECE specifically on the high-curvature subset, confirming that the certificate, not just the Taylor term, is responsible for the calibration gain. The step-4b discrepancy d_t should be largest on this high-curvature subset, and we report it explicitly as the empirical test of the load-bearing assumption.

### What would falsify this idea

If ECRC's ECE gain over global temperature scaling is uniform across all steps rather than concentrated on the high-sigma, high-kappa subset, then the per-step certificate is not the causal mechanism and the central claim about heterogeneous calibration is wrong. Similarly, if the no-certificate ablation matches full ECRC everywhere, the certificate weighting contributes nothing and the method reduces to a fixed Taylor correction. Additionally, if the step-4b discrepancy d_t on flagged steps exceeds 0.5 * sigma_t, then kappa does not track the on-policy calibration gap and must be reported as a heuristic reliability score rather than a bound.

## References

1. Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents
2. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
