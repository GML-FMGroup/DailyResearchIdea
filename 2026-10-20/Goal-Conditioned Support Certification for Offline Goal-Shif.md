# Goal-Conditioned Support Certification for Offline Goal-Shifted Policy Learning

## Motivation

Offline world models and policies are increasingly evaluated on goal-shifted task variants where individual state factors and actions stay in-distribution but their recombinations are not, as shown by the AnyGoal benchmark, yet AnyGoal only diagnoses this failure without providing a training-time or evaluation-time mechanism that certifies whether a candidate goal-conditioned policy's actions remain inside the behavior support for each composed goal. The root cause is structural: support constraints in offline RL are defined relative to the marginal behavior distribution over actions, which is invariant to goal shift, so a policy can satisfy every marginal support test while acting in regions that are jointly unsupported under the shifted goal, and no existing method conditions the support set on the goal itself.

## Key Insight

For a fixed composed goal, the behavior policy's support over actions is a goal-conditioned set whose indicator can be estimated from the joint (goal, action) occupancy of the offline data, so a conservative action-value bound built from that goal-conditioned indicator certifies the policy exactly where marginal support constraints are provably silent.

## Method

**What it is.** Goal-Conditioned Support Certification (GCSC) takes as input an offline dataset of (state, action, goal, reward, next-state) tuples, a learned world model used only for goal-conditioned rollouts, and a candidate goal-conditioned policy, and outputs a certified conservative Q-value estimate together with a per-goal support certificate flag indicating whether the policy's action distribution at each visited state lies inside the behavior support conditioned on the composed goal.

**How it works.**

```pseudocode
Input: offline dataset D = {(s,a,g,r,s')}, world model M, policy pi(a|s,g), target goals G* (composed)
Hyperparams: kappa (support slack, default 0.05), n_rollout (default 8), lambda_sup (default 1.0), tau (certificate threshold, default 0.9),
             B (bootstrap classifiers, default 10), alpha (LCB confidence level, default 0.05), sigma_smooth (goal-kernel bandwidth, default 0.1)

Phase 1 - Goal-conditioned support estimation
  For each (s,g) pair in D, fit a goal-conditioned density ratio
      w(s,a,g) = p_beh(a | s, g) / p_ref(a | s)
  using a classifier q_phi(s,a,g) trained to separate D actions from
  reference actions sampled from a goal-agnostic behavior marginal p_ref(a|s),
  then w = q_phi / (1 - q_phi).
  To obtain a lower-confidence-bound (LCB) support indicator, train B=10 bootstrap
  classifiers {q_phi^b} on resampled D; for each (s,a,g) compute w^b = q_phi^b/(1-q_phi^b)
  and define
      LCB(w(s,a,g)) = quantile_{alpha}( {w^b(s,a,g)}_{b=1..B} )
  (alpha=0.05, i.e., the 5th percentile).
  Define the goal-conditioned support set
      S_g(s) = { a : LCB(w(s,a,g)) >= kappa }.
  Reference distribution: p_ref(a|s) is the goal-marginalized behavior action distribution
  estimated from D; to avoid thin-tail instability, when evaluating w for a target goal g*
  we use the smoothed goal-conditional proposal p_ref(a|s,g*) = sum_{g' in D} K_sigma(g*,g') p_beh(a|s,g')
  with Gaussian kernel K_sigma (bandwidth sigma_smooth=0.1 in normalized goal space),
  so the ratio is not evaluated where p_ref has negligible mass.

Phase 2 - Conservative goal-shifted value bound
  For each target goal g* in G*:
      sample n_rollout trajectories from M starting at s_0 ~ D under pi(·|·,g*)
      for each visited (s_t, a_t):
          compute indicator I_t = 1[ LCB(w(s_t, a_t, g*)) >= kappa ]
          compute conservative target
              Q_target = r_t + gamma * ( min_{j=1..n_rollout} Q_bar(s_{t+1}, pi(s_{t+1}, g*), g*) )
          update Q_theta(s_t, a_t, g*) toward Q_target - lambda_sup * (1 - I_t) * |Q_theta|
  end

Phase 3 - Per-goal certificate
  For each g*: estimate support violation rate
      V(g*) = E_{s ~ d_pi, a ~ pi(·|s,g*)} [ 1 - I(s,a,g*) ]
  where d_pi is the state distribution induced by n_rollout world-model rollouts.
  Certificate: CERT(g*) = 1[ V(g*) <= tau ].
  Return Q_theta, {CERT(g*)}.
```

**Why this design.** We chose a goal-conditioned density ratio w(s,a,g) over a goal-agnostic marginal support indicator because the marginal indicator is provably blind to goal shift (it returns the same value for every g), accepting the cost that w must be estimated from finite data and is noisier for rarely composed goals. We chose a classifier-based density-ratio estimator over a generative action model because the classifier's logit directly yields a calibrated ratio under the logistic loss and requires no likelihood evaluation over a high-dimensional action space, accepting the cost that the reference distribution p_ref(a|s) must be specified (we use the goal-marginalized behavior action distribution, which is available in D). We chose a penalized conservative update (lambda_sup * (1 - I_t) * |Q_theta|) over hard action clipping because hard clipping changes the policy's action distribution and destroys the very certificate we want to report, accepting the cost that the penalty introduces a bias in Q that must be traded against the certificate threshold tau. We chose rollouts from the world model over pure offline Q-learning because goal-shifted evaluation requires visiting composed goals that are underrepresented in D, accepting the cost that world-model rollout error propagates into the certificate; the min-over-rollouts target mitigates this. A domain expert would not describe GCSC as a variant of CQL or IQL: those methods regularize toward the goal-marginal behavior distribution and cannot distinguish two goals that share the same marginal action support, whereas GCSC's penalty is indexed by the goal and its certificate is a per-goal quantity.

**Why this measures what we claim.** The density ratio w(s,a,g) measures goal-conditioned behavior support because the logistic-classifier assumption makes q_phi/(1-q_phi) equal the true likelihood ratio p_beh(a|s,g)/p_ref(a|s); this assumption fails when the classifier overfits to goal-correlated nuisance features, in which case w reflects goal-conditional action frequency rather than support and the certificate becomes over-optimistic for rare goals. The LCB indicator I_t = 1[LCB(w) >= kappa] measures membership in the behavior support set because kappa is a slack threshold on the lower confidence bound of the likelihood ratio, which is equivalent to a conservative level-set of p_beh(a|s,g) up to the p_ref factor; this assumption fails when the bootstrap classifiers are correlated (e.g., due to shared training data) so the LCB is not a valid lower bound, in which case I_t can misclassify unsupported actions as supported. The min-over-rollouts target measures conservatism under world-model error because the minimum over n_rollout independent rollouts is a lower confidence bound on the expected return under the assumption that rollout errors are independent across samples; this assumption fails when world-model errors are correlated across rollouts (e.g., systematic dynamics bias), in which case the minimum reflects the shared bias rather than a valid lower bound. The violation rate V(g*) measures the certificate because it is the expected fraction of policy actions outside S_g(s) under the policy's own state distribution, which is exactly the quantity a support certificate should bound; this assumption fails when the state distribution d_pi is estimated from too few rollouts, in which case V(g*) reflects rollout sampling noise rather than the policy's true support violation.

## Contribution

(1) A goal-conditioned support certification framework, GCSC, that constructs a conservative action-value bound indexed by the composed goal and emits a per-goal certificate flag, providing a training-time and evaluation-time mechanism that the AnyGoal benchmark lacks. (2) A design principle: goal-conditioned density ratios estimated via logistic classification yield a support indicator that is strictly more discriminative than marginal support constraints, because marginal indicators are invariant to goal shift by construction. (3) An open-source implementation of GCSC integrated with the AnyGoal task library, including the certificate computation and the goal-conditioned density-ratio estimator.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Goal-shifted variants of world-model benchmark envs | Contains composed goals underrepresented in behavior data |
| Primary metric | Support violation rate V(g*) vs. realized return gap | Directly measures certificate validity and its value consequence |
| Baseline | Goal-agnostic marginal support (CQL/IQL-style) | Tests whether goal-conditioning is necessary |
| Baseline | World-model rollout planning without support penalty | Tests whether the certificate/penalty is necessary |
| Baseline | Hard action clipping to behavior support | Tests whether penalized update beats distribution-altering fix |
| Ablation-of-ours | GCSC with goal-agnostic w(s,a) (drop g) | Isolates contribution of goal-conditioned density ratio |
| Ablation-of-ours | GCSC with point-estimate w (no LCB) | Isolates contribution of LCB support indicator |
| Ablation-of-ours | GCSC with unsmoothed p_ref | Isolates contribution of goal-kernel smoothing |

### Why this setup validates the claim

The central claim is that goal-conditioned support certification is both necessary and sufficient to predict where a goal-conditioned policy will fail under goal shift, and that a goal-indexed conservative penalty converts that prediction into a usable value estimate. The design is falsifiable because each baseline removes exactly one component of the mechanism. The goal-agnostic marginal baseline tests the necessity of goal-conditioning: if support is truly goal-blind, our per-goal certificate should show no advantage. The no-penalty world-model planner tests whether the certificate is merely diagnostic or actually corrective. The hard-clipping baseline tests whether the penalized update is required to preserve the certificate rather than distort it. The goal-agnostic ablation isolates the density-ratio index. The LCB ablation isolates the effect of conservative ratio estimation. The unsmoothed-p_ref ablation isolates the effect of avoiding thin-tail instability. The primary metric — support violation rate paired with realized return gap — is the right detector because the claim is a monotone relationship: higher V(g*) should predict larger return degradation, and only a method whose certificate tracks this relationship has measured what it claims.

### Expected outcome and causal chain

**vs. Goal-agnostic marginal support (CQL/IQL-style)** — On a case where two goals share identical marginal action support but one requires an action region absent from behavior data for that goal, the marginal baseline assigns equal conservatism to both and produces over-optimistic values for the unsupported goal because its penalty cannot distinguish goals. Our method instead computes w(s,a,g) conditioned on the composed goal, flags the unsupported goal's actions as out-of-support, and penalizes them, so we expect a clear separation in V(g*) between supported and unsupported goals that the baseline collapses to near-zero for both.

**vs. World-model rollout planning without support penalty** — On a case where the world model hallucinates a high-reward trajectory through an action region the behavior policy never took for the target goal, the unpenalized planner exploits this hallucination and produces inflated Q-values because nothing constrains it to supported actions. Our method instead applies the goal-indexed penalty at each visited out-of-support (s,a), driving Q down along that path, so we expect the unpenalized planner to show a large optimistic bias concentrated on out-of-support rollouts while our estimate tracks realized return more closely there.

**vs. Hard action clipping to behavior support** — On a case where the policy's natural action distribution is slightly outside S_g(s) but recoverable, hard clipping projects actions onto the boundary and changes the state distribution the policy visits, so the reported certificate reflects the clipped policy rather than the candidate policy, producing a certificate that is self-fulfilling and uninformative. Our method instead leaves the policy distribution intact and reports V(g*) on the true policy, so we expect clipping to show artificially low violation rates that do not predict realized return, while our V(g*) retains its monotone relationship with return gap.

**vs. GCSC with goal-agnostic w(s,a) (drop g)** — On a case where two goals share identical marginal action support but one requires an action region absent from behavior data for that goal, the goal-agnostic ablation assigns equal conservatism to both and produces over-optimistic values for the unsupported goal because its penalty cannot distinguish goals. Our method instead computes w(s,a,g) conditioned on the composed goal, flags the unsupported goal's actions as out-of-support, and penalizes them, so we expect a clear separation in V(g*) between supported and unsupported goals that the ablation collapses to near-zero for both.

**vs. GCSC with point-estimate w (no LCB)** — On a case where the classifier is noisy in the thin-tail region for a rare composed goal, the point-estimate w can overestimate the ratio and misclassify unsupported actions as supported, leading to an over-optimistic certificate. Our method uses the LCB over bootstrap classifiers, which is conservative under ratio-estimation error, so we expect the point-estimate ablation to show lower V(g*) (more optimistic) and a weaker monotone relationship with return gap, while our LCB-based V(g*) retains its monotone relationship.

**vs. GCSC with unsmoothed p_ref** — On a case where p_ref has thin tails exactly where p_beh(a|s,g*) is thin, the unsmoothed ratio is unstable and the indicator I_t can misclassify unsupported actions as supported. Our method uses a smoothed goal-conditional proposal p_ref(a|s,g*) to avoid the thin-tail regime, so we expect the unsmoothed ablation to show higher variance in V(g*) and a less reliable certificate, while our smoothed version yields a more stable and predictive V(g*).

### What would falsify this idea

If our per-goal certificate's advantage over the goal-agnostic baseline is uniform across all goal subsets rather than concentrated on goals whose action support genuinely differs from the marginal, then goal-conditioning is doing no real work and the central claim is wrong. Equivalently, if V(g*) fails to predict realized return gap while the marginal baseline's violation rate predicts it equally well, the certificate measures the wrong quantity.

## References

1. AnyGoal: On World Model Generalization Beyond Expert Rollouts
