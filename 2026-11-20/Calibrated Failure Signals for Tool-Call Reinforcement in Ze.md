# Calibrated Failure Signals for Tool-Call Reinforcement in Zero-Shot VLN

## Motivation

Vision-language navigation agents such as ZeroVLN rely on a hard binary success/failure oracle to drive closed-loop RL over semantic tool calls and to arbitrate memory retention, but that oracle is itself an imperfect estimator: a single mislabel silently misdirects both the reward and the memory gate. The same structural gap recurs across the field — LH-VLN's ISR/CSR metrics and GSA-R2R's environment-specific training both consume success as ground truth without modeling the evaluator's own uncertainty, so genuine navigation failure cannot be distinguished from evaluator mislabeling. The root cause is that no existing VLN method treats its own success oracle as a noisy observation with an estimable error rate, leaving the reward and memory-arbitration signals uncalibrated.

## Key Insight

The success oracle is not ground truth but a noisy measurement whose error rate can be estimated from paired repeated evaluations and held-out human-verified labels, and once that error rate is known the posterior probability of genuine success is identifiable — a quantity that hard labels and raw model confidence both fail to provide, enabling reward and memory gating to be corrected for evaluator bias rather than inheriting it.

## Method

**What it is.** CaliFail is a calibrated failure-signal layer wrapped around ZeroVLN's closed-loop RL. It takes as input the trajectory episode, the frozen success oracle's binary verdict, and a human-verified calibration set, and outputs a posterior failure probability `p_fail` with an explicit uncertainty interval that replaces the hard label in both the RL reward and the memory-arbitration check.

**Load-bearing assumption (stated explicitly).** CaliFail assumes the frozen success oracle's error is *stationary and estimable* from a human-verified calibration set, and that a logistic map over four episode features (path length, instruction length, number of tool calls, final-view distance-to-goal) fitted on that set yields a posterior failure probability that remains valid on the RL-drifted episode distribution. This assumption is load-bearing: if it fails, `p_fail` reflects extrapolated calibration rather than measured failure, and both the reward and memory corrections inherit that error. The method therefore includes an explicit drift-aware calibration procedure (Phase 0b) and a support-aware uncertainty term (Phase 1) that grows, rather than shrinks, when an episode leaves the calibration support.

**How it works.**

```python
# Phase 0a: Offline calibration of the frozen oracle (once, before RL)
# Calibration set D_cal = {(episode_i, oracle_verdict_i, human_label_i)}
# Fit a logistic calibration map from oracle verdict + episode features f_i
# (path length, instruction length, number of tool calls, final-view distance-to-goal)
# to human label. Fit by maximum likelihood; store coefficients w and a
# bootstrap ensemble {w_b}_{b=1..B} (B=50) for coefficient sampling variance.
# Calibration set size: 200 human-verified episodes (pre-registered audit protocol).

def calibrate(D_cal):
    w = fit_logistic(features(D_cal), human_labels(D_cal))   # point estimate
    W = [fit_logistic(bootstrap_sample(D_cal)) for b in range(50)]
    # Also store the empirical feature distribution of D_cal for density-ratio diagnostics
    D_cal_features = features(D_cal)
    return w, W, D_cal_features

# Phase 0b: Online drift-aware recalibration (every K=500 RL episodes)
# Draw a small stream of human-audited episodes from the CURRENT policy's rollout
# distribution: N_audit = 50 episodes per recalibration round.
# Re-fit the logistic map on D_cal UNION the audited stream, with importance weights
# w_i = density_ratio(f_i) = p_current(f_i) / p_cal(f_i) estimated by a logistic
# classifier distinguishing current-policy features from calibration-set features.
# This keeps the map anchored to the original human labels while tracking drift.

def recalibrate(D_cal, audited_stream):
    D_aug = D_cal + audited_stream
    weights = density_ratio_weights(features(D_aug), D_cal_features)
    w = fit_logistic_weighted(features(D_aug), human_labels(D_aug), weights)
    W = [fit_logistic_weighted(bootstrap_sample(D_aug), weights) for b in range(50)]
    return w, W

# Phase 1: Online, inside each RL episode of ZeroVLN
def calibrated_failure_signal(episode, oracle_verdict, w, W, D_cal_features):
    f = episode_features(episode)            # same features as Phase 0
    # posterior failure probability under each bootstrap map
    p_b = [1 - sigmoid(dot(w_b, f)) for w_b in W]   # per-bootstrap P(human=success)
    p_fail = 1 - mean(p_b)                   # point estimate of failure prob
    u_coef = std(p_b)                        # coefficient sampling variance
    # Support-aware uncertainty: density ratio of this episode's features
    # relative to the calibration set. High ratio => out-of-support => u grows.
    dr = density_ratio(f, D_cal_features)    # estimated by logistic classifier
    u_support = max(0.0, dr - 1.0)           # 0 inside support, grows outside
    u = u_coef + u_support                   # total epistemic uncertainty
    return p_fail, u

# Phase 2: Reward and memory gating consume the calibrated signal
def rl_step(episode, oracle_verdict):
    p_fail, u = calibrated_failure_signal(episode, oracle_verdict, w, W, D_cal_features)
    # calibrated reward: expected success minus uncertainty penalty
    reward = (1 - p_fail) - beta * u        # beta = 0.5, uncertainty penalty
    # memory arbitration: retain a memory entry only if calibrated success
    # probability exceeds a threshold that scales with u
    keep = (1 - p_fail) > tau + gamma * u   # tau = 0.6, gamma = 1.0
    return reward, keep
```

The calibration map is refit only on the fixed human-verified calibration set plus the small audited drift stream, never on un-audited RL rollouts, so the reward signal cannot be gamed by the policy. The uncertainty `u` combines coefficient sampling variance with an out-of-support density-ratio term, so that ambiguous or drifted episodes are conservatively discarded rather than committed to long-term memory.

**Why this design.** We chose a bootstrap-ensemble logistic calibration map over a temperature-scaled softmax on the VLM's own log-probs because the oracle's error is a property of the evaluator, not of the brain's token likelihood, and bootstrap spread gives an explicit epistemic uncertainty that a single temperature scalar cannot; the cost is that calibration requires a small human-verified set (200 episodes) and assumes the oracle's error is stationary across the RL distribution. We chose to gate memory retention with a threshold that scales with `u` rather than a fixed threshold because uncertain episodes are exactly the ones whose hard labels are least trustworthy, accepting the cost that some genuinely successful but ambiguous episodes are never retained. We chose to penalize reward by `beta * u` rather than to abstain on high-`u` episodes because abstention removes training signal from the hardest cases, accepting the cost that high-uncertainty episodes contribute a smaller but nonzero gradient. We added the density-ratio support term to `u` because bootstrap coefficient spread alone measures sampling variance of the calibration set, not evaluator error under distribution shift, and would report low uncertainty precisely where the map is extrapolating; the cost is that the density-ratio classifier adds a second model to fit and a small audited stream (50 episodes per 500 RL episodes) to maintain. A domain expert would not describe CaliFail as a variant of ZeroVLN or of standard process-reward-model RL (e.g., Lightman 2023) because the calibrated quantity is a property of the *evaluator*, estimated from paired oracle/human labels, not a property of the policy's intermediate reasoning steps.

**Why it measures what we claim.** The posterior failure probability `p_fail` measures *genuine navigation failure* because the logistic calibration map is fit on paired oracle verdicts and human labels, which makes `p_fail` an estimate of P(human=failure | oracle verdict, episode features) under the assumption that the calibration set is exchangeable with the RL episode distribution; this assumption fails when the RL policy drifts into episode feature regions absent from the calibration set, in which case `p_fail` reflects extrapolated calibration rather than measured failure. The uncertainty `u` measures *evaluator uncertainty* because it combines bootstrap resampling of the calibration set (coefficient sampling variance) with a density-ratio term that grows when the episode leaves the calibration support; this assumption fails when the human labels themselves are biased toward easy episodes, in which case `u` reflects label-sampling variance rather than evaluator error. The uncertainty-scaled retention threshold measures *conservative memory arbitration* because it operationalizes the decision rule 'retain only when calibrated success probability exceeds the uncertainty-inflated bar' under the assumption that memory contamination by mislabeled successes is costlier than discarding ambiguous successes; this assumption fails in regimes where memory is the bottleneck, in which case the threshold reflects an over-conservative retention policy rather than calibrated arbitration. The feature-based logistic map measures *evaluator error* under the feature-sufficiency assumption that oracle mislabeling is a function only of the four listed episode features; this assumption fails when oracle error depends on unobserved factors (e.g. semantic landmark violations), in which case `p_fail` tracks task difficulty rather than evaluator error.

## Contribution

(1) A calibrated failure-signal layer, CaliFail, that wraps a frozen success oracle with a bootstrap-ensemble logistic calibration map and produces a posterior failure probability with an explicit epistemic uncertainty estimate, replacing the hard binary label in ZeroVLN's RL reward and memory-arbitration check. (2) A design principle: the success oracle in VLN should be treated as a noisy measurement whose error rate is estimable from paired oracle/human labels, and the resulting uncertainty should inflate memory-retention thresholds rather than be collapsed into a point estimate. (3) An uncertainty-penalized reward formulation that keeps high-uncertainty episodes in the training loop at reduced weight instead of abstaining on them.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VLN-CE (R2R val-unseen + GSA-R2R OOD split) | In-distribution plus drift to test calibration stationarity |
| Primary metric | Success rate on human-audited episodes | Directly measures genuine navigation, not oracle agreement |
| Baseline | ZeroVLN (uncalibrated hard-label RL) | Isolates effect of calibration layer alone |
| Baseline | SFT-based hierarchical VLN | Tests gain vs. fine-tuned-VLM paradigm |
| Baseline | Hard-label + fixed memory threshold | Isolates uncertainty-scaled retention from calibration |
| Baseline | Process-reward-model / label-smoothing on same episodes | Tests whether evaluator calibration is just relabeled smoothing |
| Ablation-of-ours | CaliFail without uncertainty penalty (beta=0, gamma=0) | Separates calibration from uncertainty gating |
| Ablation-of-ours | CaliFail with constant base-rate calibration (no features) | Tests feature-sufficiency assumption |
| Diagnostic | Calibration reliability (ECE, Brier score) and interval coverage on held-out human-labeled drift set | Grounds exchangeability and uncertainty claims empirically |
| Measurement study | Oracle-human disagreement rate and correlation with episode features on VLN-CE and GSA-R2R | Tests whether disagreement is rare or feature-independent |

### Why this setup validates the claim

The central claim is that replacing the oracle's hard verdict with a calibrated, uncertainty-aware failure posterior improves genuine navigation success and memory quality. The ZeroVLN baseline holds the architecture and RL loop fixed and swaps only the reward label, so any gap is attributable to calibration rather than to the slow-fast design. The SFT baseline checks whether the gain is merely a ZeroVLN artifact or a broader correction of oracle mislabeling that also benefits fine-tuned agents. The hard-label-plus-fixed-threshold baseline isolates the uncertainty-scaled retention rule from the calibration map itself: if CaliFail beats it, the uncertainty term is doing work, not just the posterior. The process-reward-model / label-smoothing baseline tests whether evaluator-error modeling produces a qualitatively different reward ordering than policy-confidence smoothing, demonstrating the mechanism is not just a relabeled smoothing prior. The no-uncertainty ablation tests the converse. The constant-base-rate ablation tests the feature-sufficiency assumption: if it matches CaliFail, the four episode features are not carrying evaluator-error information and `p_fail` is tracking task difficulty. The primary metric is success rate on human-audited episodes rather than oracle success, because the whole premise is that oracle success is a biased proxy; scoring against audited labels is the only metric that can detect the predicted correction. The OOD split from GSA-R2R is included because the method's own stated assumption—calibration stationarity—predicts where gains should shrink, giving the design a built-in falsification gradient. The calibration reliability diagnostic (ECE, Brier score, interval coverage) on a held-out human-labeled drift set directly tests whether `p_fail` and its uncertainty interval remain valid as the policy drifts, rather than assuming exchangeability. The measurement study reports the empirical oracle-human disagreement rate and its correlation with episode features; if disagreement is rare or feature-independent, the significance claim collapses and is reported as a negative result.

### Expected outcome and causal chain

**vs. ZeroVLN (uncalibrated hard-label RL)** — On a case where the VLM reaches the goal region by an unintended shortcut and the oracle returns success, ZeroVLN reinforces the shortcut trajectory with a full positive reward because it treats the binary verdict as ground truth. Our method instead assigns a high `p_fail` from episode features (short path, large final-view distance-to-goal) and shrinks the reward, so the shortcut receives weak reinforcement. We expect a noticeable success gap on episodes with oracle-human disagreement in the calibration set and near-parity on episodes where oracle and human agree, i.e. a gain concentrated in the disagreement subset.

**vs. SFT-based hierarchical VLN** — On a case where the agent predicts a geometric goal that lands near the target but the instruction's landmark reference is violated, the SFT baseline commits to the geometric goal and is rewarded, because its action head is trained to match oracle-verified goals. Our method instead discounts that episode's reward through the calibrated posterior and gates its memory entry out via the uncertainty-inflated threshold, so the violated-landmark pattern is not memorized. We expect CaliFail to show a modest but consistent advantage on instruction-following-heavy episodes while trailing SFT on raw oracle success, since the SFT baseline is optimized directly for that proxy.

**vs. Hard-label + fixed memory threshold** — On a case where the oracle is confidently wrong on an ambiguous episode, the fixed-threshold baseline retains the mislabeled success in memory because its hard label clears the bar, contaminating later retrieval. Our method instead inflates the retention bar by `gamma * u`, so the ambiguous episode is discarded and does not pollute memory. We expect the gap to widen with episode count—small early, larger after many RL iterations—and to be concentrated on long-horizon episodes where memory retrieval matters most.

**vs. Process-reward-model / label-smoothing** — On a case where the oracle returns success but the episode features indicate a shortcut (short path, large final-view distance-to-goal), label smoothing assigns a fixed softened reward independent of the evaluator's error structure, while a process-reward model scores intermediate steps by policy confidence. Our method instead assigns a high `p_fail` because the calibration map was fit on paired oracle/human labels where such features predicted human-labeled failure, producing a reward ordering that label smoothing cannot represent. We expect CaliFail to outperform both on the human-audited disagreement subset, with the gap largest on episodes where the oracle is confidently wrong and policy confidence is high.

**vs. CaliFail with constant base-rate calibration** — On a case where the oracle mislabels a long-horizon episode with many tool calls, the constant-base-rate ablation assigns the same `p_fail` to every episode regardless of features, so it cannot distinguish this mislabel from a genuine success. Our method uses the four episode features to assign a higher `p_fail` to this episode. We expect CaliFail to outperform the constant-base-rate ablation on the human-audited disagreement subset if the features carry evaluator-error information; if the ablation matches CaliFail, the feature-sufficiency assumption is false and `p_fail` is tracking task difficulty rather than evaluator error.

### What would falsify this idea

If CaliFail's gain over ZeroVLN is uniform across the oracle-human disagreement subset and the agreement subset rather than concentrated where mislabeling occurs, the calibration mechanism is not the cause of any improvement. Likewise, if the uncertainty-scaled retention baseline matches CaliFail exactly, the uncertainty term is inert and the claim reduces to ordinary label smoothing; if the constant-base-rate ablation matches CaliFail, the feature-sufficiency assumption fails and `p_fail` measures task difficulty rather than evaluator error.

## References

1. ZeroVLN: Learning to Navigate without Supervised Fine-Tuning of the VLM Brain
