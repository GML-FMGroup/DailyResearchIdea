# Recursive Improvement Gain: A Measurable Carry-Forward Criterion for Physical Experience

## Motivation

The frontier of recursive self-improvement in physical AI inherits an unvalidated carry-forward condition: as noted in 'Toward Recursive Self-Improvement in Physical AI', the four conditions (feedback relevance, usability, robustness, safety/affordability) are proposed as criteria but never operationalized or empirically validated as a complete framework. Without a quantitative metric, it is impossible to determine when a change in physical experience qualifies as recursively self-improving, leaving both the influence-score estimator and the distributed preference aggregator (as in RoboArena) unable to distinguish genuine process improvement from mere behavioral gains. This structural gap—the absence of a measurable carry-forward criterion—is the root cause that prevents the field from validating RSI claims across branches.

## Key Insight

The recursive improvement gain—defined as the ratio of post-change to pre-change improvement in the receiving component's performance on held-out physical tasks—is a necessary and sufficient statistic for the carry-forward condition because it directly measures whether a change in physical experience increases the component's capacity to improve, not just its current performance.

## Method

```python
# Recursive Improvement Gain (RIG) Estimator — multi-step learning-curve variant
# Inputs: pre-change experience buffer E0, post-change buffer E1, receiving component C
#         (influence-score estimator or preference aggregator), held-out task set T_hold,
#         improvement mechanism M (gradient update on C), number of improvement steps k=5,
#         bootstrap samples n_bootstrap=100
# Output: RIG score, bootstrap CI, carry-forward verdict

# Assumption A (explicit): The improvement mechanism M's held-out gain is monotone and
# step-count-invariant over the first k=5 steps, so the slope of the learning curve is a
# stable proxy for improvement capacity. Failure mode F: saturation (slope→0) or
# non-monotone learning curves (e.g., overfitting after step 3) break this equivalence.

def compute_RIG(E0, E1, C, T_hold, M, k=5, n_bootstrap=100):
    # Phase 1: Measure baseline improvement trajectory of C with E0 over k steps
    C0 = clone(C)
    perf_traj_E0 = []
    for step in range(k):
        C0 = M(C0, E0)  # one gradient step on E0
        perf_traj_E0.append(evaluate(C0, T_hold))  # accuracy / preference agreement
    # Fit learning rate (slope) via linear regression on steps 1..k
    slope0 = fit_slope(perf_traj_E0)  # ordinary least squares slope

    # Phase 2: Measure improvement trajectory of C with E1 over k steps
    C1 = clone(C)
    perf_traj_E1 = []
    for step in range(k):
        C1 = M(C1, E1)
        perf_traj_E1.append(evaluate(C1, T_hold))
    slope1 = fit_slope(perf_traj_E1)

    # Phase 3: Compute RIG as ratio of fitted learning rates with regularized denominator
    # Shrinkage: add empirical-Bayes prior on slope0 (mean 0, variance estimated from
    # the k-step trajectory residuals) to stabilize when slope0 is near zero.
    slope0_reg = shrink(slope0, prior_var=var(perf_traj_E0 residuals))
    rig_scores = []
    for _ in range(n_bootstrap):
        T_sample = bootstrap_sample(T_hold)
        # Recompute trajectories on bootstrap sample (k steps each)
        s0 = fit_slope([evaluate(M(clone(C), E0), T_sample) for _ in range(k)])
        s1 = fit_slope([evaluate(M(clone(C), E1), T_sample) for _ in range(k)])
        s0_reg = shrink(s0, prior_var=var(perf_traj_E0 residuals))
        rig_scores.append(s1 / (s0_reg + 1e-8))  # epsilon for numerical stability
    rig_mean = mean(rig_scores)
    rig_ci = percentile(rig_scores, [2.5, 97.5])

    # Phase 4: Carry-forward verdict (threshold tau=1.0: improvement must not degrade)
    carry_forward = (rig_mean > 1.0) and (rig_ci[0] > 0.95)  # strict improvement with 95% CI lower bound > 0.95
    return rig_mean, rig_ci, carry_forward

# Cross-branch validation: apply to influence-score estimator (C_influence) and distributed preference aggregator (C_pref)
# For C_influence: E0 = pre-change influence scores, E1 = post-change influence scores from physical rollouts
# For C_pref: E0 = pre-change preference pairs, E1 = post-change preference pairs from distributed evaluators
```

**Why this design**
We chose a ratio of fitted learning rates (slope over k=5 steps) over a single-step gain because the ratio directly captures the *recursive* aspect: a value >1 means the change in physical experience has increased the component's capacity to improve, not just its current performance, which is the core of the carry-forward condition. We chose a bootstrap confidence interval over a point estimate to account for the high variance in physical robot evaluation, accepting the cost of additional computation (100 bootstrap samples × 5 steps) to avoid false positives. We chose a fixed threshold (tau=1.0) with a strict lower confidence bound (0.95) over a learned threshold because the carry-forward condition is a qualitative criterion that should not be tuned to specific tasks; this sacrifices some flexibility but ensures the metric is interpretable and comparable across branches. We chose to validate on two distinct receiving components (influence-score estimator and preference aggregator) rather than a single one because the carry-forward condition is claimed to be general, and cross-branch validation is the only way to test that generality. This design is not a variant of retrieval-based or log-prob gating methods (e.g., Toolformer, Gorilla) because it does not select from a library or use confidence scores; instead, it intervenes on the improvement mechanism itself by swapping the experience buffer and measuring the causal effect on the component's learning capacity, which is a counterfactual operation that retrieval cannot perform.

**Why this measures what we claim**
The computational quantity `slope1 / slope0_reg` (RIG) measures the motivation-level concept of *recursive improvement gain* because the assumption that the slope of a k-step learning curve on a held-out task set is a valid proxy for the component's improvement capacity holds when the task set is representative of the component's deployment distribution and the learning curve is monotone over k steps; this assumption fails when the held-out tasks are not representative (e.g., distribution shift) or when the component saturates within k steps, in which case RIG reflects the component's ability to overfit to the held-out set or numerical noise rather than its general improvement capacity. The bootstrap confidence interval measures *robustness of the carry-forward verdict* because the assumption that bootstrap resampling of tasks approximates the sampling distribution of the fitted learning rate holds when tasks are exchangeable; this assumption fails when tasks are correlated (e.g., sequential manipulation tasks), in which case the CI underestimates variance and may yield false carry-forward verdicts. The threshold tau=1.0 measures the *qualitative carry-forward condition* because the assumption that a ratio >1 indicates genuine recursive improvement (not just noise) holds when the improvement gain is positive and the ratio is estimated with sufficient precision; this assumption fails when slope0 is near zero (e.g., component already saturated), in which case the shrinkage regularizer stabilizes the ratio but RIG may reflect the prior rather than improvement capacity.

## Contribution

['A novel metric, the Recursive Improvement Gain (RIG), that operationalizes the carry-forward condition as a ratio of post-change to pre-change improvement gains, with a bootstrap confidence interval and a fixed threshold for verdict.', 'A cross-branch validation protocol that applies RIG to two distinct receiving components—an influence-score estimator and a distributed preference aggregator—demonstrating that the metric yields consistent carry-forward verdicts across branches.', 'An empirical finding that the RIG metric identifies which changes in physical experience qualify as recursively self-improving, and that the four carry-forward conditions (feedback relevance, usability, robustness, safety/affordability) are necessary but not sufficient without the quantitative RIG criterion.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Held-out manipulation task set (pre/post physical change) | Directly probes improvement capacity under changed experience |
| Primary metric | RIG ratio (post/pre fitted learning rate over k=5 steps) | Ratio of slopes captures recursive capacity change |
| Baseline | Digital-agent RSI (code-rewrite + tests) | Tests whether executable-test RSI transfers to physical components |
| Baseline | Conventional robot learning (behavior-only improvement) | Tests whether improving behavior ≠ improving improvement process |
| Baseline | Difference-based gain metric (slope1 − slope0) | Tests whether ratio is needed over simple additive gain |
| Baseline | Forward transfer ratio (FTR) from transfer learning literature | Tests whether RIG is isomorphic to existing relative-transfer measures |
| Ablation-of-ours | RIG with matched-marginal resampled E1 (content-preserving permutation) | Isolates whether gain comes from experience content vs. gradient magnitude |

### Why this setup validates the claim

The central claim is that RIG detects a *causal* increase in a receiving component's improvement capacity when physical experience changes, and that this signal generalizes across two distinct components (influence-score estimator, preference aggregator). The digital-agent RSI baseline tests the sub-claim that executable-test-driven self-improvement does not transfer to physical components whose improvement mechanism is a learned estimator rather than code: if RIG on that baseline's components sits at or below 1.0 while ours exceeds it, the carry-forward condition is doing real work rather than tracking generic progress. The conventional robot-learning baseline tests the sub-claim that behavioral improvement alone is insufficient — its components improve current performance but not their own improvement gain, so RIG should hover near 1.0. The difference-based baseline tests whether the *ratio* formulation is necessary: if slope1 − slope0 is positive but RIG is not, then the additive metric is confounded by baseline capacity, confirming the ratio is the right detector. The forward transfer ratio (FTR) baseline tests whether RIG is merely a reparameterization of existing relative-transfer measures: FTR is computed as (post-change performance − pre-change performance) / pre-change performance on the same buffers; if RIG and FTR yield identical carry-forward verdicts across all conditions, then RIG adds no discriminative power beyond FTR. The matched-marginal resampled E1 ablation is the falsification anchor: it destroys the informational content of post-change experience while preserving its marginal distribution and gradient magnitude (by resampling task features from E1's marginal but shuffling their association with outcomes), so any RIG lift under this ablation would mean the metric responds to distributional artifacts rather than causal experience content. RIG is the right primary metric because it is scale-free across components with different absolute accuracies and directly operationalizes "improvement capacity" rather than "performance."

### Expected outcome and causal chain

**vs. Digital-agent RSI (code-rewrite + tests)** — On a case where the physical change alters contact dynamics (e.g., a gripper swap) so that pre-change influence scores no longer predict rollout outcomes, the digital-agent RSI baseline produces a component that still passes its executable tests because those tests encode pre-change dynamics, so its post-change improvement gain collapses (slope1 ≈ 0). Our method instead swaps the experience buffer and measures the causal effect on the influence estimator's learning capacity over k=5 steps, so we expect RIG noticeably above 1.0 on the dynamics-shifted subset but parity on the unaffected subset.

**vs. Conventional robot learning (behavior-only improvement)** — On a case where the component's held-out accuracy is already high, the conventional baseline produces a policy that improves current task success but leaves the estimator's improvement gain unchanged, because its update targets behavior not the improvement mechanism; slope0 and slope1 are both small and roughly equal. Our method intervenes on the improvement mechanism directly, so we expect RIG concentrated above 1.0 specifically where pre-change gain was low (unsaturated components) and near 1.0 where the component was saturated, revealing the capacity signature rather than a uniform lift.

**vs. Difference-based gain metric (slope1 − slope0)** — On a case where the component has large baseline capacity (slope0 large), the difference metric produces a positive score from a small relative change, falsely signaling recursive improvement, because additive gains scale with baseline capacity. Our method normalizes by slope0_reg, so we expect the difference metric to flag carry-forward on saturated components where RIG stays near 1.0, and the two metrics to agree only on low-capacity components — the divergence pattern is the diagnostic.

**vs. Forward transfer ratio (FTR)** — On a case where the component's pre-change performance is high but its improvement capacity is low (saturated), FTR will be near zero (small numerator) and may falsely indicate no transfer, while RIG (ratio of slopes) will correctly show near 1.0 (no capacity change). Conversely, on a case where pre-change performance is low but improvement capacity is high, FTR will be large (large relative performance gain) while RIG will be near 1.0 if capacity is unchanged. We expect RIG and FTR to diverge systematically on these cases, demonstrating that RIG captures improvement capacity rather than relative performance gain.

### What would falsify this idea

If RIG lift is uniform across the dynamics-shifted and unaffected subsets, or if the matched-marginal resampled E1 ablation produces the same RIG as true E1, then the metric is responding to distributional artifacts rather than causal experience content, and the central claim that RIG measures recursive improvement capacity is wrong. Additionally, if RIG and FTR yield identical carry-forward verdicts across all conditions, then RIG is isomorphic to existing relative-transfer measures and adds no new discriminative power.

## References

1. Toward Recursive Self-Improvement in Physical AI
