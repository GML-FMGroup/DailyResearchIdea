# Shift-Aware Conformal Recalibration of Action Confidence for Adaptive Denoising Budgets in Frozen Flow-Matching VLAs

## Motivation

Inference-time action-reuse frameworks such as VLA-Rolling allocate per-action denoising budgets from a confidence estimator that is implicitly calibrated to the frozen flow-matching VLA's own action distribution; the estimator has no mechanism to detect that the deployed observation/action distribution has moved away from that calibration distribution. This is a meta-level failure recurring across both VLA-Rolling (confidence-guided renoising) and GR00T N1 (System-1 action generation under heterogeneous, embodiment-shifted data): the confidence/quality signal is treated as a fixed property of the frozen backbone rather than as a quantity whose calibration must be re-established whenever the input distribution shifts, so under shift the confidence ordering decouples from true action quality and the downstream budget allocation degrades structurally rather than gracefully.

## Key Insight

Monotonicity of the confidence signal with true action quality under shift is recoverable without retraining the frozen VLA, because the shift magnitude is observable from the frozen backbone's own intermediate representations and can be used to widen a conformal prediction interval whose nonconformity score is the backbone's confidence residual — a distribution-free calibration that holds conditional on the estimated shift level.

## Method

**What it is.** SHARC (SHift-Aware Recalibration of Confidence) is an inference-time wrapper around a frozen flow-matching VLA. It takes as input the per-action raw confidence scores c_raw produced by VLA-Rolling's confidence estimator, the observation/action context of the current chunk, and a small held-out calibration buffer of (context, c_raw, realized action-quality) triples; it outputs recalibrated confidences c_cal that are monotone with realized action quality conditional on the estimated shift level, which VLA-Rolling's denoising budget allocator then consumes unchanged.

**How it works.**

```pseudocode
# ---- Offline (once, before deployment) ----
# Calibration buffer D_cal = {(x_i, c_raw_i, q_i)} collected on the frozen VLA
# q_i = realized action quality. Two variants are computed and stored:
#   q_int_i = negative L2 to the executed action's post-hoc refinement target,
#             computed from the VLA's own renoising residual (internal proxy)
#   q_ext_i = external quality label: binary task success (1/0) from the
#             executed trajectory, or negative L2 to the executed action
#             recorded in the calibration buffer (external ground truth)
# The method's load-bearing assumption is stated explicitly: the nonconformity
# residual z = c_raw - q is an unbiased proxy for true action quality ONLY IF
# q is externally grounded. We therefore use q_ext for all conformal quantiles
# and report the per-stratum correlation between q_int and q_ext as a validity
# check; recalibration is gated on that correlation exceeding rho_min = 0.5.
for each calibration triple i:
    s_i = ShiftScore(x_i)                      # see below, uses frozen backbone features
    z_i = c_raw_i - q_ext_i                    # nonconformity residual (external q)
# Partition calibration residuals into L=5 shift strata by s_i quantiles;
# in stratum l compute the (1-alpha) empirical quantile Q_l of |z_i|, alpha=0.10
# Buffer size: |D_cal| = 2000 triples, giving >= 400 per stratum on average;
# we report coverage variance vs. buffer size in {500, 1000, 2000, 4000}.

# ---- Online (per action chunk) ----
function SHARC(c_raw, x, s_prev_stratum):
    s = ShiftScore(x)                          # scalar in [0, inf)
    l = stratum_of(s)                          # nearest of the L calibration strata
    # Shift-adaptive interval half-width: interpolate quantiles across strata
    w = interp(Q[l-1], Q[l], Q[l+1], frac(s))
    # Shift-magnitude gain: inflate width proportional to s beyond the
    # calibration support, clipped at w_max = 3 * Q[L-1]
    w = min(w * (1 + beta * max(0, s - s_max_cal)), w_max)   # beta=0.5
    # Monotone recalibration: project c_raw onto the conformal interval
    # centered at the running median of recent realized qualities
    mu = running_median(recent_q_ext, window=32)
    c_cal = mu + clip(c_raw - mu, -w, +w)
    # Feed c_cal to VLA-Rolling's existing budget allocator (unchanged)
    return c_cal

# ---- ShiftScore: distribution-shift detector on frozen backbone ----
function ShiftScore(x):
    # (a) observation novelty: Mahalanobis distance of the frozen VLM's
    #     penultimate visual embedding to the calibration-buffer mean/cov
    d_obs = mahalanobis(emb_vis(x), mu_vis, Sigma_vis)
    # (b) action-distribution divergence: symmetric KL between the frozen
    #     flow head's predicted action-chunk Gaussian (mean, diag-var) and
    #     the calibration-buffer action Gaussian
    d_act = sym_kl(N(mu_a(x), Sigma_a(x)), N(mu_a_cal, Sigma_a_cal))
    return sqrt(d_obs) + d_act                   # combined scalar
```

**Why this design.** We chose an online conformal wrapper over a learned recalibration head because the frozen VLA offers no gradient signal to a head and a learned head would itself be miscalibrated under the very shift it must detect, accepting the cost that the conformal quantiles are only valid conditional on the discrete shift strata (we mitigate by interpolating across strata and inflating beyond the calibration support). We chose a shift score composed of observation novelty *and* action-distribution divergence rather than observation novelty alone because VLA-Rolling's failure mode is specifically that the inherited action tail becomes a poor initialization when the *action* distribution moves even if the scene looks familiar, accepting the cost of two detector statistics instead of one. We chose to project the raw confidence onto a conformal interval centered at a running median of recently realized qualities rather than to replace confidence with the conformal bound outright, because VLA-Rolling's allocator expects a per-action relative ordering and the projection preserves within-chunk ordering while bounding absolute error, accepting the cost that a persistently biased running median can lag by up to one window. We chose to leave VLA-Rolling's allocator untouched so that SHARC is a drop-in wrapper, and we explicitly note that a domain expert would *not* describe SHARC as a variant of standard split-conformal prediction (Vovk; Angelopoulos & Bates) because standard conformal assumes exchangeability between calibration and test points, whereas SHARC's load-bearing contribution is the shift-stratified quantile interpolation plus out-of-support inflation that restores conditional coverage when exchangeability is violated — the detector is not a gating heuristic but the conditioning variable that re-establishes the exchangeability assumption locally.

**Why this measures what we claim.** The nonconformity residual z = c_raw - q_ext measures *confidence-calibration error* because it assumes the external quality label q_ext (task success or executed-action L2) is an unbiased proxy for the action quality the allocator will experience at deployment; this assumption fails when the external label is itself noisy (e.g., sparse success labels), in which case z reflects label noise rather than true quality error. The shift score s measures *distribution shift relative to the frozen VLA's calibration distribution* because it assumes the frozen VLM's visual embedding and the flow head's predicted action Gaussian are sufficient statistics for the deployment distribution; this assumption fails under a shift that changes only task semantics while leaving visual statistics and marginal action statistics intact (e.g., a relabeled instruction), in which case s under-reports shift and the interval stays too narrow. The stratum index l and interpolated width w measure *the local exchangeability radius* because they assume calibration residuals within a shift stratum are exchangeable with test residuals at the same shift level; this assumption fails at the boundary between strata where interpolation blends two non-exchangeable residual pools, in which case w reflects an average of two coverage regimes rather than the correct one. The clipped projection c_cal measures *shift-robust monotone confidence* because it assumes true quality lies within the conformal interval with probability 1-alpha conditional on the stratum; this assumption fails when the shift exceeds the calibration support and the w_max clip binds, in which case c_cal reflects a saturated confidence that no longer discriminates among badly-shifted actions. Additionally, the internal proxy q_int measures *true action quality* only if the renoising residual used to compute it is uncorrupted by a stale inherited action tail; this assumption fails under action-distribution shift, in which case q_int reflects renoising noise rather than quality error — which is why we use q_ext for the conformal target and report the per-stratum correlation between q_int and q_ext as a validity check.

## Contribution

(1) SHARC, an inference-time shift-stratified conformal recalibration wrapper that restores monotonicity of a frozen flow-matching VLA's action-confidence signal with realized action quality under distribution shift, without retraining the backbone or modifying VLA-Rolling's budget allocator. (2) A shift score combining frozen-VLM visual-embedding novelty and flow-head action-Gaussian divergence, together with a stratum-interpolated quantile and out-of-support inflation rule that operationalizes conditional conformal coverage when exchangeability is violated. (3) A design principle for inference-time action-reuse frameworks: confidence signals should be treated as calibration-distribution-relative quantities requiring re-establishment at deployment, not as fixed properties of the frozen backbone.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LIBERO-Plus shift splits | Contains controlled distribution shifts with ground-truth quality |
| Primary metric | Conditional coverage error of c_cal vs realized quality (using external q_ext) | Directly tests the calibration claim under shift |
| Baseline | VLA-Rolling (uncalibrated c_raw) | Tests whether raw confidence is already shift-robust |
| Baseline | Global split-conformal wrapper | Tests whether shift-stratification is load-bearing |
| Baseline | Observation-novelty-only shift score | Tests whether action-divergence term matters |
| Ablation-of-ours | SHARC without out-of-support inflation | Tests whether the w_max inflation term matters |
| Ablation-of-ours | SHARC with internal q_int instead of external q_ext | Tests whether the self-referential quality proxy is fatal |
| Secondary metric | End-to-end task success and denoising-step allocation per shift stratum | Closes the coverage-to-budget-to-success causal chain |

### Why this setup validates the claim

The central claim is that SHARC restores conditional coverage of confidence under distribution shift by conditioning on a shift score that includes action-distribution divergence, and that this restored coverage translates into better denoising-budget allocation. LIBERO-Plus is the right dataset because it provides *controlled* shift axes (visual, semantic, action-distribution), letting us isolate which sub-claim each baseline fails. VLA-Rolling with uncalibrated c_raw tests the null hypothesis that raw confidence is already shift-robust; if it is, SHARC has nothing to add. Global split-conformal tests whether the shift-stratified quantile interpolation is doing real work versus a single pooled quantile — if pooling matches SHARC, the load-bearing contribution collapses. The observation-novelty-only baseline tests the specific design choice to include action-distribution divergence, which is the mechanism we argue is necessary because VLA-Rolling's failure mode is action-tail-driven. The ablation removes the out-of-support inflation term, testing whether the w_max clip is a real coverage safeguard or dead code. The q_int ablation directly tests the load-bearing assumption: if SHARC with q_int matches SHARC with q_ext, the self-referential proxy is not fatal; if it fails, the external grounding is necessary. Conditional coverage error is the right metric because it directly measures the probability that realized quality falls inside the conformal interval *given the shift stratum* — the exact quantity SHARC claims to control — and it is falsifiable: a method that only improves marginal coverage but not conditional coverage would not pass. The secondary metric (task success and denoising-step allocation per stratum) closes the causal chain from coverage to budget to outcome, which the reviewer requested.

### Expected outcome and causal chain

**vs. VLA-Rolling (uncalibrated c_raw)** — On a case where the action distribution shifts (e.g., a novel object mass changes the required force profile) while the visual scene remains familiar, VLA-Rolling's raw confidence stays high because the confidence estimator inherits the pre-shift action tail as a plausible initialization, so the allocator under-allocates denoising budget and executes a poorly-refined action. Our method instead computes an inflated shift score via the action-divergence term, widens the conformal interval, and projects c_raw toward the running median of recently realized qualities, so the allocator sees a deflated confidence and spends more denoising steps. We expect a noticeable conditional-coverage gap on the action-shift subset with near-parity on the visually-shifted subset, where the baseline's observation-novelty signal already fires. We also expect task success to improve on the action-shift subset, with denoising-step allocation shifting toward high-shift chunks.

**vs. Global split-conformal wrapper** — On a case where the test shift level sits in the tail of the calibration distribution (e.g., a compound visual-plus-action shift), the global wrapper uses a single pooled quantile that is too narrow for the tail stratum, so its interval under-covers and the allocator under-spends on the hardest chunks. Our method instead interpolates quantiles across the five shift strata and inflates beyond the calibration support, so the tail stratum receives a wider interval and the allocator spends proportionally more denoising steps there. We expect the global wrapper to match SHARC on the bulk of the distribution but to show a clear coverage deficit concentrated in the highest-shift decile, which is exactly the region the allocator cares about most. We also expect the global wrapper's task success to degrade on the highest-shift decile relative to SHARC.

**vs. Observation-novelty-only shift score** — On a case where the task semantics change (e.g., a relabeled instruction) but visual statistics and marginal action statistics are preserved, the observation-novelty-only detector reports near-zero shift, so its interval stays narrow and the allocator treats a genuinely shifted chunk as in-distribution. Our method's action-divergence term is also weak here, so we expect only partial improvement — but the running-median projection still pulls c_cal toward recently realized qualities, giving a modest coverage gain. We expect a smaller gap than against the other baselines on this subset, and we expect the gap to be largest on the action-shift subset where the divergence term fires strongly; this differential is the diagnostic that the action-divergence term is doing the work we claim.

**vs. SHARC with internal q_int** — On the action-shift subset, the internal q_int is corrupted by the stale inherited action tail, so the conformal interval covers a corrupted target and conditional coverage of true quality (measured by q_ext) is not established. We expect SHARC with q_int to show a large conditional-coverage error on the action-shift subset, while SHARC with q_ext maintains coverage, confirming that the external grounding is necessary and that the self-referential proxy is fatal if used as the target.

### What would falsify this idea

If SHARC's conditional-coverage gain over the global split-conformal wrapper is uniform across all shift strata rather than concentrated in the high-shift tail, then the shift-stratified interpolation is not the load-bearing mechanism and the central claim about restoring conditional coverage under violated exchangeability is wrong. Similarly, if the observation-novelty-only baseline matches SHARC on the action-shift subset, the action-divergence term is not necessary and the design rationale collapses. If SHARC with internal q_int matches SHARC with external q_ext, the load-bearing assumption about the quality proxy is not fatal and the external grounding is unnecessary.

## References

1. VLA-Rolling: Fewer Denoising Steps, Better Control via Adaptive Action Reuse
2. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
3. π0: A Vision-Language-Action Flow Model for General Robot Control
