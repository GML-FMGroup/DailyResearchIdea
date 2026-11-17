# Acceptance-Profile-Aware Scheduling for Speculative Decoding

## Motivation

Existing speculative decoding theory assumes stationary, token-independent acceptance rates, while adaptive mechanisms like MMFlash's visual-dependence upweighting and PEARL's adaptive draft length change the acceptance statistics on the fly without any cost model to predict when they help. This structural gap means that polybasic's optimal multi-model schedule (the theorem characterizing optimal inference time across model configurations) is computed under assumptions that adaptive drafting violates, so practitioners cannot know whether a token-level adaptive intervention will improve or degrade the theoretically optimal schedule. The root cause is that the optimal-time characterization treats acceptance as a single stationary scalar, whereas real adaptive pipelines produce a non-stationary per-token acceptance profile that must be integrated into the schedule computation.

## Key Insight

The optimal multi-model schedule under adaptive drafting is a functional of the per-token acceptance profile rather than a scalar, so a schedule computed from the profile's upper envelope is optimal exactly when the profile is stationary, and the gap between profile-based and scalar-based schedules is a computable quantity that predicts whether token-level adaptivity helps.

## Method

**What it is.** We introduce APAS (Acceptance-Profile-Aware Scheduling), a scheduler that takes a per-token acceptance profile estimated online from draft-verify outcomes and outputs a model schedule (which draft model to invoke at each step, how many draft tokens to propose, and whether to upweight visually dependent tokens) that minimizes expected wall-clock latency under a stated cost model. Inputs: a set of draft models with per-token forward costs, a target model cost, and a stream of (token position, visual-dependence score, draft model id, accepted?) tuples. Output: a schedule policy.

**Load-bearing assumption (stated explicitly).** APAS assumes that the per-token acceptance probability of draft model m is a function of a single observable scalar—the MMFlash visual-dependence bin g—that is stable enough to be estimated online from a 32-step window with a shrinkage prior, such that the resulting per-bin profile predicts future acceptance at positions t+1..t+L and the profile gap predicts the true acceptance gain from upweighting. This assumption is load-bearing: if it fails, the profile-aware schedule collapses to the scalar baseline. We therefore make the assumption testable by (i) replacing future-bin indexing with an explicit transition-kernel forecast, (ii) replacing the constant-shift gap with a measured causal estimate of upweighting's effect on acceptance, and (iii) reporting calibration diagnostics that directly measure the violation of each sub-assumption.

**How it works.**

```pseudocode
# APAS: Acceptance-Profile-Aware Scheduling
# Hyperparameters: window W (default 32), shrinkage kappa (default 5.0),
#                  profile grid G (default 16 bins over visual-dependence score),
#                  transition-kernel smoothing lambda (default 1.0),
#                  A/B exploration fraction rho (default 0.1),
#                  gap threshold tau (default 0.02)

Initialize: for each draft model m and each grid bin g in G:
    alpha_hat[m][g] = prior_mean (0.5), n[m][g] = 0
    # transition counts: T[m][g][g'] = number of observed transitions g -> g'
    T[m][g][g'] = 0 for all g'
    # causal upweighting effect: delta_hat[m][g] = measured acceptance delta
    delta_hat[m][g] = 0.0, n_delta[m][g] = 0

For each decoding step t:
    # 1. Estimate per-token acceptance profile
    g_t = bin(visual_dependence_score_t)          # from MMFlash-style KL metric
    for each candidate draft model m:
        alpha_hat[m][g_t] = (n[m][g_t]*alpha_hat[m][g_t] + kappa*prior_mean)
                            / (n[m][g_t] + kappa)   # shrinkage estimator

    # 2. Compute profile-aware expected latency for each (m, draft_length L)
    #    using transition-kernel forecast instead of realized future bins
    for each m and L in 1..L_max:
        # forecast distribution over future bins via empirical transition kernel
        # P_hat(g_{t+1} | g_t) = (T[m][g_t][g_{t+1}] + lambda) / (sum_g' T[m][g_t][g'] + lambda*|G|)
        # expected accepted tokens under profile and kernel:
        E_acc[m][L] = sum_{i=1..L} prod_{j=1..i} E_{g_{t+j-1} ~ P_hat(.|g_{t+j-2})}[ alpha_hat[m][g_{t+j-1}] ]
        E_lat[m][L] = cost_draft[m]*L + cost_target*(1 + E_acc[m][L])
        # upweighting option: if g_t is high, MMFlash-style weight w_t>1
        #   modifies alpha_hat[m][g_t] -> alpha_hat_up[m][g_t] (see step 3)

    # 3. Decide upweighting via measured causal gap (interleaved A/B)
    #    With probability rho, apply upweighting regardless of gap (exploration);
    #    otherwise, apply upweighting only if measured delta_hat[m][g_t] > tau.
    if random() < rho:
        apply_upweight = True
        record_exploration = True
    else:
        apply_upweight = (delta_hat[m*][g_t] > tau)
        record_exploration = False
    if apply_upweight:
        apply MMFlash upweighting to draft model m*
        alpha_hat[m*][g_t] = alpha_hat_up[m*][g_t]

    # 4. Choose schedule minimizing expected latency
    (m*, L*) = argmin_{m,L} E_lat[m][L]
    propose L* tokens with m*, verify with target model

    # 5. Update statistics from verify outcomes
    for each proposed token j at position t+j-1:
        n[m*][g_{t+j-1}] += 1
        if accepted: alpha_hat[m*][g_{t+j-1}] = running update
        # update transition counts
        if j > 1:
            T[m*][g_{t+j-2}][g_{t+j-1}] += 1
        # update causal upweighting effect if this step was an A/B trial
        if record_exploration or (not record_exploration and apply_upweight):
            # compare acceptance of this token to counterfactual baseline
            # (baseline acceptance estimated from non-upweighted steps in same bin)
            delta_obs = accepted - alpha_hat_baseline[m*][g_{t+j-1}]
            n_delta[m*][g_{t+j-1}] += 1
            delta_hat[m*][g_{t+j-1}] = (n_delta[m*][g_{t+j-1}]*delta_hat[m*][g_{t+j-1}] + delta_obs)
                                       / (n_delta[m*][g_{t+j-1}] + 1)
```

**Why this design.** We chose a shrinkage estimator over a full Bayesian posterior because the per-bin sample counts are small in early decoding and a posterior would require specifying a conjugate prior family whose misspecification would propagate into the schedule; the shrinkage estimator with kappa=5 gives a bias-variance trade-off that is transparent and cheap to update, accepting the cost that it is not fully Bayesian and may under-explore rare bins. We chose to compute the profile gap and gate upweighting on it rather than always applying MMFlash's upweighting because MMFlash applies upweighting unconditionally, which can help drafting quality but may increase draft cost without a corresponding acceptance gain when visual dependence is low; the gap gate trades off the risk of missing a beneficial upweight against wasted draft computation. We chose a per-bin profile over a single scalar because polybasic's optimal-time theorem relies on a stationary acceptance scalar, and the profile is the minimal generalization that makes the theorem's schedule computation valid under non-stationary acceptance; the cost is that the profile must be estimated online and introduces estimation noise. We replaced future-bin indexing with an explicit transition-kernel forecast because the visual-dependence score of a not-yet-generated token is unknown at scheduling time; the kernel is estimated from the same stream with Laplace smoothing (lambda=1.0) and integrated into E_acc, making the forecast testable. We replaced the constant-shift gap with a measured causal estimate via interleaved A/B (exploration fraction rho=0.1) because MMFlash frames upweighting as a loss-reweighting intervention on the drafter's training objective, not an inference-time acceptance shift; the A/B directly measures the acceptance delta and calibrates the gate against observed data. A domain expert would not describe APAS as 'polybasic plus MMFlash plus PEARL' because APAS does not simply layer adaptivity onto polybasic's schedule: it replaces the scalar acceptance input to the optimal-time computation with a profile functional, uses a transition-kernel forecast for future bins, and uses a measured causal gap as the decision variable for when adaptivity helps, which is a different mechanism from either composition.

**Why it measures what we claim.** The per-bin acceptance estimate alpha_hat[m][g] measures the conditional acceptance probability of draft model m given visual-dependence bin g because the binning assumption makes the acceptance probability a function of the bin (i.e., tokens within a bin are exchangeable with respect to acceptance); this assumption fails when visual dependence interacts with token position or context length, in which case alpha_hat reflects a bin-average rather than the true conditional acceptance. The transition-kernel forecast P_hat(g_{t+1}|g_t) measures the distribution of future visual-dependence bins because it is estimated from observed transitions in the stream with Laplace smoothing; this assumption fails when the transition dynamics are non-stationary (e.g., a prompt switch), in which case the forecast reflects a historical average rather than the current local dynamics. The measured causal gap delta_hat[m][g] measures the expected acceptance improvement from upweighting because it is estimated from interleaved A/B trials that directly compare acceptance with and without upweighting within the same bin; this assumption fails when the A/B exploration fraction rho is too small to yield sufficient samples in rare bins, in which case delta_hat reflects a high-variance estimate rather than the true causal effect. The expected latency E_lat[m][L] measures the wall-clock cost of a schedule because the cost model assumes draft and target forward costs are additive and independent of the token content; this assumption fails when target-model verification cost depends on the number of accepted tokens (e.g., due to attention over accepted prefixes), in which case E_lat reflects a lower bound on true latency.

## Contribution

["APAS, a scheduler that computes the optimal multi-model speculative decoding schedule from an online-estimated per-token acceptance profile, replacing the stationary scalar acceptance assumed in polybasic's optimal-time characterization with a profile functional and using the profile gap as a decision rule for when token-level upweighting (MMFlash-style) and adaptive draft length (PEARL-style) improve the schedule.", 'A design principle: token-level adaptivity helps exactly when the profile gap exceeds a threshold that is computable from the cost model, yielding a predictor of when adaptive interventions improve latency without requiring a full re-derivation of the schedule.', 'An online estimation procedure with shrinkage that makes the profile-based schedule robust to small per-bin sample counts, enabling deployment without a separate offline calibration phase.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multimodal VQA + long-video benchmarks (MMFlash suite) | Contains visual-dependence variation the profile exploits |
| Primary metric | Wall-clock speedup at matched output quality | Directly reflects the latency-minimization claim |
| Baseline | MMFlash (unconditional upweighting) | Tests whether gap-gated upweighting beats always-on |
| Baseline | PEARL (adaptive draft length) | Tests profile-aware length vs. reactive length adaptation |
| Baseline | polybasic / EAGLE-3 scalar-acceptance scheduler | Tests whether profile beats stationary scalar acceptance |
| Ablation-of-ours | APAS with scalar acceptance (no profile, no gate) | Isolates the profile-plus-gate contribution |
| Ablation-of-ours | APAS with profile but no transition-kernel forecast (uses realized future bins) | Isolates the transition-kernel contribution |
| Ablation-of-ours | APAS with profile and kernel but constant-shift gap (no A/B) | Isolates the measured causal gap contribution |
| Calibration diagnostic | Bin-internal acceptance variance, empirical distribution of upweighting-induced acceptance deltas, measured vs. predicted latency residuals | Directly measures violation of each assumption |

### Why this setup validates the claim

The central claim is that replacing a stationary scalar acceptance estimate with an online per-bin acceptance profile, and gating upweighting on the resulting profile gap, yields lower wall-clock latency than schedulers that assume stationarity or apply upweighting unconditionally. Each baseline isolates one sub-claim: MMFlash tests the upweighting-gate sub-claim, PEARL tests the draft-length sub-claim, and the scalar-acceptance scheduler tests the profile sub-claim. The ablation removes both the profile and the gate, so any gain over it must come from those two mechanisms specifically rather than from draft-model quality or verification parallelism. The additional ablations isolate the transition-kernel forecast and the measured causal gap, ensuring that any gain from those components is attributable to the specific repair of the load-bearing assumption. Wall-clock speedup at matched quality is the right metric because the claim is about latency under a cost model, not acceptance length in isolation; a method could raise acceptance while inflating draft cost, and only wall-clock latency detects that failure. The visual-dependence stratification inherited from MMFlash gives the natural subset on which the predicted effect should concentrate. The calibration diagnostics directly measure the violation of each assumption (bin-internal variance, constancy of upweighting shift, additive cost), so the cost model's error is bounded rather than assumed away.

### Expected outcome and causal chain

**vs. MMFlash** — On a case where visual dependence is low (a text-heavy prompt in a multimodal benchmark), MMFlash still applies upweighting, so it spends extra draft computation on tokens whose acceptance would not improve, inflating draft cost with no acceptance gain. Our method computes the measured causal gap and, finding it below threshold, skips upweighting, so draft cost stays low. We expect near-parity on high-visual-dependence subsets but a measurable wall-clock advantage on low-visual-dependence subsets, where the wasted draft cost is eliminated.

**vs. PEARL** — On a case where acceptance drops sharply across a few high-visual-dependence tokens, PEARL's reactive pre/post-verify adjusts draft length only after observing the mutual-waiting symptom, so it over-proposes tokens that the target then rejects, wasting verification slots on the transition. Our method anticipates the drop from the transition-kernel forecast of upcoming bins and shortens the proposal before the transition, so we expect a gap concentrated at visual-dependence transitions and parity on homogeneous stretches.

**vs. scalar-acceptance scheduler (polybasic/EAGLE-3)** — On a long video input where acceptance falls from roughly the mid-40s to low-20s percent between low and high visual-dependence quartiles, a stationary scalar acceptance averages these, so its chosen draft length is too long in high-dependence regions and too short in low-dependence ones. Our per-bin profile tracks the two regimes separately, so we expect a uniform-across-quartiles advantage that is largest where the scalar average is most wrong, i.e., at the extremes of the visual-dependence range.

**vs. APAS with scalar acceptance (no profile, no gate)** — This ablation removes both the profile and the gate, so it should match the scalar-acceptance scheduler. Any gain of full APAS over this ablation is attributable to the profile-plus-gate mechanisms.

**vs. APAS with profile but no transition-kernel forecast** — This ablation uses realized future bins (which are unknown at scheduling time) as an oracle, so it upper-bounds the benefit of the transition-kernel forecast. We expect full APAS to approach but not exceed this oracle, with the gap measuring the forecast error.

**vs. APAS with profile and kernel but constant-shift gap** — This ablation uses the original constant-shift approximation for the gap. We expect full APAS with measured causal gap to outperform it when the true upweighting effect varies across bins, as indicated by the calibration diagnostic.

### What would falsify this idea

If APAS's wall-clock gain over the scalar-acceptance baseline is uniform across visual-dependence quartiles rather than concentrated at the extremes where the scalar average is most misestimated, the profile mechanism is not doing the work claimed and the gain is attributable to something else. Likewise, if the gap-gated variant matches unconditional upweighting even on low-dependence subsets, the gate is not buying back draft cost and the central scheduling claim fails. Additionally, if the calibration diagnostic shows bin-internal acceptance variance is so large that the profile is prior-dominated (most bins at 0.5 prior), or if the measured causal gap is indistinguishable from zero across all bins, the load-bearing assumption is falsified and APAS collapses to the scalar baseline.

## References

1. MMFlash: Faster Speculative Decoding for Vision-Language Models
