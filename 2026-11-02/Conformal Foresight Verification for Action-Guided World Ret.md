# Conformal Foresight Verification for Action-Guided World Retrieval in VLA Models

## Motivation

Across world-model-augmented VLA systems, predicted future latents are fed into action policies or retrieval modules with no distribution-free guarantee that a wrong prediction is rejected before it contaminates control. HiFOR's action-guided factorized retrieval is a concrete instance: it retrieves action-conditioned latents from a hierarchical foresight memory, but has no verifier or uncertainty estimator, so a mispredicted milestone or goal latent is silently integrated into the policy; Mantis and DUST similarly expose predicted futures to downstream action learning without a calibrated rejection mechanism. The structural root cause is that prediction and verification are two separate loops that are never closed at the latent level — the predictor's nonconformity against retrieved latents is never quantified, so retrieval robustness under predictor error remains unestablished.

## Key Insight

Exchangeability of per-horizon nonconformity scores between calibration and deployment trajectories makes a split-conformal threshold a finite-sample, distribution-free gate on whether a predicted latent is admissible for retrieval, turning the predictor-retriever pair into a testable hypothesis rather than an unchecked feed-forward path.

## Method

## Conformal Foresight Verifier (CFV)

**(A) What it is.** CFV wraps HiFOR's hierarchical foresight predictor and action-guided factorized retrieval with a per-horizon split-conformal verifier. Inputs: the predicted multi-horizon latents $\hat{z}_{h}$ for $h \in \{\text{short}, \text{milestone}, \text{goal}\}$, the action-conditioned retrieved latents $r_{h}(a)$ from HiFOR's factorized memory, and a calibration set of trajectories. Output: a binary admission decision and a down-weighting coefficient $w_h \in [0,1]$ per horizon, used to gate or attenuate the retrieved latent before it enters the action policy.

**Load-bearing assumption (stated explicitly).** CFV assumes that the normalized latent residual $s_h = \|\hat{z}_h - r_h(a)\|_2 / \sigma_h$ is a monotone proxy for downstream action error: larger residual implies proportionally larger policy degradation, so that a conformal threshold on $s_h$ gates harmful latents rather than merely measuring predictor-retriever self-consistency. This assumption is load-bearing because the conformal guarantee covers only the score $s_h$; if $s_h$ is uncorrelated with true error (e.g., a systematically biased retriever that the predictor also reproduces), the gate admits wrong latents at full weight. We therefore (i) calibrate $s_h$ against an external correctness signal available at calibration time, and (ii) report the empirical monotonicity (Spearman rank correlation) between $s_h$ and realized downstream action error as a precondition; if monotonicity is weak (Spearman $\rho < 0.3$), we replace the linear down-weight with a learned monotone map fit on the calibration curve, as specified in (B).

**(B) How it works.**

```pseudocode
# Offline calibration (one-time, held-out trajectories)
for each calibration trajectory tau and horizon h:
    z_hat_h, r_h(a) = HiFOR.predict_and_retrieve(tau, h)
    s_h(tau) = nonconformity(z_hat_h, r_h(a))          # see Eq. below
    # external correctness signal for monotonicity check:
    e_h(tau) = || z_hat_h - z_h^GT(tau) ||_2           # ground-truth future latent
    # or, if GT latent unavailable: e_h(tau) = realized action error / task failure indicator
    store s_h(tau), e_h(tau)
q_h = Quantile( {s_h(tau)}, level = ceil((n+1)(1-alpha_h))/n )   # alpha_h per horizon
# Monotonicity precondition (reported, not assumed):
rho_h = SpearmanRankCorr( {s_h(tau)}, {e_h(tau)} )
if rho_h < 0.3:
    fit monotone map g_h: s -> w via isotonic regression on (s_h, e_h) calibration pairs
    use g_h for down-weighting instead of linear rule below

# Deployment, per timestep t
z_hat_h, r_h(a) = HiFOR.predict_and_retrieve(obs_t, a_t)
for h in {short, milestone, goal}:
    s_h = nonconformity(z_hat_h, r_h(a))
    if s_h <= q_h:
        w_h = 1.0                                   # admit at full weight
    else:
        if rho_h >= 0.3:
            w_h = max(0, 1 - (s_h - q_h)/ (q_h * kappa))  # kappa = 2.0, linear down-weight
        else:
            w_h = g_h(s_h)                          # isotonic map fit on calibration curve
    r_h_gated = w_h * r_h(a)
policy_input = concat(r_short_gated, r_milestone_gated, r_goal_gated, obs_t)
a_t = policy(policy_input)

# Nonconformity (per horizon): normalized latent-space residual
nonconformity(z_hat_h, r_h(a)) =
    || z_hat_h - r_h(a) ||_2 / ( sigma_h + eps )
# sigma_h = median absolute deviation of ||z_hat_h - r_h(a)|| over calibration set, eps=1e-6
# alpha_h = {short: 0.10, milestone: 0.15, goal: 0.20}  (stricter for short horizon)
# Calibration set size: n = 200 trajectories per horizon (split conformal requires n >= 1/alpha_h - 1; for alpha=0.10, n>=9; we use 200 for stable quantiles)
```

**(C) Why this design.** We chose split conformal over a learned uncertainty head (e.g., a variance-predicting MLP or ensemble) because conformal calibration provides a finite-sample coverage guarantee under only exchangeability, whereas a learned head would require distributional assumptions that HiFOR's hierarchical latents do not satisfy; the cost is that the guarantee is marginal over calibration trajectories, not conditional per state. We chose a normalized residual against the *retrieved* action-conditioned latent rather than against a ground-truth future because the retrieved latent is exactly the quantity that will be injected into the policy, so the nonconformity score directly measures the discrepancy that matters downstream; the cost is that a systematically biased retriever could produce low residuals despite wrong futures, which we mitigate by also reporting raw residual quantiles and by the monotonicity precondition above. We chose per-horizon thresholds $\alpha_h$ rather than a single global $\alpha$ because the goal latent is inherently higher-variance and a global threshold would either over-reject short-horizon latents or under-reject goal latents; the cost is three calibration sets and slightly more hyperparameter tuning. We chose linear down-weighting over hard rejection because hard rejection of all three horizons collapses to no retrieval, defeating HiFOR's purpose; the cost is that down-weighted but still-wrong latents can leak into the policy, which we bound by the conformal coverage on the admitted set. A domain expert would not describe this as a variant of standard conformal prediction applied to a new domain because the nonconformity score is defined against the *retrieved action-conditioned latent* — a quantity that exists only inside HiFOR's factorized retrieval — not against a held-out label or a model's own predictive distribution; the verifier is structurally coupled to the retrieval module it gates.

**(D) Why it measures what we claim.** The normalized residual $\|\hat{z}_h - r_h(a)\|_2 / \sigma_h$ measures *predictor-retriever disagreement* because we assume that under exchangeability, calibration and deployment trajectories are drawn from the same joint distribution of (observation, action, future latent), so the residual distribution is invariant; this assumption fails when deployment involves novel objects or action spaces not represented in calibration, in which case the residual reflects distribution shift rather than genuine prediction error. The conformal quantile $q_h$ measures *admissibility at confidence $1-\alpha_h$* because we assume the nonconformity scores are exchangeable between calibration and test, which makes the rank of the test score uniformly distributed and thus the coverage guarantee exact in finite samples; this assumption fails under temporal correlation within a trajectory (consecutive latents are not independent), in which case $q_h$ reflects a trajectory-level rather than step-level guarantee. The down-weighting coefficient $w_h$ measures *trust in the retrieved latent* because we assume that larger nonconformity implies proportionally larger downstream action error, i.e., the mapping from latent residual to policy degradation is monotone; this assumption fails when the policy is robust to small latent perturbations but brittle to specific directions, in which case $w_h$ reflects a generic distance rather than task-relevant trust. **Sub-block: the missing equivalence.** The equivalence that normalized latent residual measures downstream action degradation is not automatic; it requires assumption A: the residual-to-policy-error map is monotone and direction-agnostic. Failure mode F: the policy is robust to small residuals but brittle to specific latent directions (e.g., a small residual in a task-critical dimension causes failure while a large residual in an irrelevant dimension does not). We back this with the residual-vs-action-error calibration curve (Spearman $\rho_h$) and, if $\rho_h < 0.3$, the isotonic map $g_h$ replaces the linear rule, making the down-weighting empirically grounded rather than assumed.

## Contribution

["A conformal foresight verifier (CFV) that defines a per-horizon nonconformity score between predicted and action-conditioned retrieved latents and gates HiFOR's factorized retrieval with a distribution-free calibrated threshold, closing the prediction-verification loop without discarding the hierarchical predictor.", "A design principle for uncertainty-aware gating in world-model-augmented VLA systems: the verifier's nonconformity must be computed against the exact quantity injected into the policy (the retrieved latent), not against a proxy, so that the gate's coverage guarantee transfers to the policy input.", 'An analysis of when conformal coverage on latent residuals does and does not imply downstream action reliability, identifying the monotonicity assumption and its failure modes as the key boundary condition for verifier-gated retrieval.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HiFOR manipulation benchmark suite (sim + Franka real) | HiFOR's own retrieval latents are the gated quantity |
| Primary metric | Success rate on retrieval-sensitive task subsets | Directly reflects whether gated latents help actions |
| Baseline | Vanilla HiFOR (no verifier) | Tests whether gating adds value over raw retrieval |
| Baseline | Learned variance-head gate | Tests conformal vs learned uncertainty under shift |
| Baseline | Global-alpha conformal gate | Tests whether per-horizon thresholds matter |
| Ablation-of-ours | CFV with hard rejection (w_h ∈ {0,1}) | Isolates contribution of linear down-weighting |
| Ablation-of-ours | CFV with isotonic map (w_h = g_h(s_h)) | Tests monotonicity precondition fallback |
| Diagnostic | Spearman rank correlation between s_h and realized action error | Checks load-bearing monotonicity assumption |

### Why this setup validates the claim

The central claim is that a per-horizon conformal verifier coupled to HiFOR's retrieved action-conditioned latents improves downstream action success specifically where retrieval is unreliable, without harming reliable cases. Vanilla HiFOR tests the sub-claim that gating is needed at all: if raw retrieval already succeeds everywhere, the verifier has nothing to fix and the idea is dead. The learned variance-head gate tests the sub-claim that conformal calibration beats a distribution-assuming uncertainty estimate, which should show up as a gap under distribution shift where the learned head miscalibrates. The global-alpha gate tests the sub-claim that per-horizon thresholds are necessary, which should show up as degradation on short-horizon-heavy tasks (over-rejection) or goal-horizon tasks (under-rejection). The hard-rejection ablation tests the sub-claim that linear down-weighting preserves useful retrieval where hard rejection would collapse it. The isotonic-map ablation tests the fallback when the monotonicity precondition fails, ensuring the method remains grounded. The diagnostic Spearman correlation directly checks the load-bearing assumption that residual is monotone with action error; if $\rho_h < 0.3$, the isotonic map is used and we report both. Success rate on retrieval-sensitive subsets is the right metric because the predicted effect is localized: CFV should not change performance where retrieval is already accurate, so a metric averaged over all tasks would dilute the signal. The falsifiable structure is a predicted interaction — gain concentrated on the subset the mechanism targets — not a uniform lift.

### Expected outcome and causal chain

**vs. Vanilla HiFOR (no verifier)** — On a case where the retrieved milestone latent is drawn from a memory entry that superficially matches the observation but encodes a different object configuration, vanilla HiFOR injects that latent at full weight and the policy commits to a trajectory that misaligns with the true scene, producing a grasp or placement failure. Our method computes a high normalized residual against the predicted latent, exceeds the milestone quantile, and down-weights the retrieved latent, so the policy leans more on the observation and short-horizon signal. We expect a noticeable success-rate gap on retrieval-mismatch subsets but near-parity on subsets where retrieval is clean, since gating is inert when residuals fall below threshold.

**vs. Learned variance-head gate** — On a case involving a novel object category absent from calibration, the variance head was trained on the original distribution and outputs confidently low variance for the out-of-distribution retrieval, so it admits a wrong latent at full weight and the policy fails. Our method's conformal quantile is fixed from calibration and the residual for this novel case lands above it, triggering down-weighting regardless of any learned confidence. We expect the gap to concentrate on the distribution-shift subset and to be small in-distribution, since both gates behave similarly when the calibration distribution holds.

**vs. Global-alpha conformal gate** — On a case where the goal-horizon latent is inherently high-variance but correct, a global threshold tuned for short horizons over-rejects the goal latent, starving the policy of long-horizon guidance and causing drift on multi-stage tasks. Our per-horizon alpha leaves the goal latent admitted while still rejecting genuinely bad short-horizon latents. We expect the global gate to underperform specifically on long-horizon, multi-stage tasks while matching CFV on short-horizon tasks, revealing the horizon-interaction signature.

**vs. CFV with hard rejection (w_h ∈ {0,1})** — On a case where the retrieved latent is slightly above threshold but still useful, hard rejection discards it entirely, collapsing to no retrieval for that horizon and degrading performance on tasks that rely on partial guidance. Our linear down-weighting retains a fraction of the latent, preserving useful signal. We expect hard rejection to underperform CFV on tasks with moderate residuals, while matching on tasks with either very low or very high residuals.

**vs. CFV with isotonic map (w_h = g_h(s_h))** — On a case where the monotonicity precondition fails (Spearman $\rho_h < 0.3$), the linear rule may mis-weight latents because the residual-to-error map is non-linear or direction-dependent. The isotonic map, fit on calibration pairs, provides a monotone but flexible mapping. We expect the isotonic variant to match or slightly outperform the linear rule when $\rho_h$ is low, and to be comparable when $\rho_h$ is high, with the diagnostic Spearman correlation reported per horizon to justify the choice.

### What would falsify this idea

If the success-rate gain over vanilla HiFOR is uniform across retrieval-clean and retrieval-mismatch subsets rather than concentrated on the mismatch subset, the verifier is not doing what we claim and the central mechanism is wrong. Equivalently, if the learned variance-head gate matches CFV even under distribution shift, the conformal-calibration argument fails. Additionally, if the Spearman rank correlation between residual and realized action error is near zero across all horizons and the isotonic fallback does not recover a gain, the load-bearing monotonicity assumption is false and the gating mechanism is invalid.

## References

1. HiFOR: Hierarchical Foresight with Action-Guided World Grounding for Vision-Language-Action Model
2. Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight
3. Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model
