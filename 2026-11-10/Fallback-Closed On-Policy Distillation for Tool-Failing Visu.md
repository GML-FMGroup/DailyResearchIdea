# Fallback-Closed On-Policy Distillation for Tool-Failing Visual Agents

## Motivation

On-policy multi-teacher distillation as instantiated by Reframe assumes a fixed action vocabulary: when a tool call is deficient it substitutes a canonical action supplied by an oracle, so the student's action set is only closed under substitution by external repair, never by the student itself. This means the deployed policy is a partial function — on genuinely unavailable tools (no canonical action exists), the rollout has no defined continuation, and distillation gradients are only collected on trajectories that the repair machinery rescued. The structural gap is that closure of the action set under failure substitution is unestablished, so the learned policy is undefined exactly on the states that failure produces.

## Key Insight

If every tool action a is paired with a learned fallback generator that maps the failed-observation state to a substitute action drawn from the same vocabulary, then the augmented action set is closed under failure substitution by construction, making the student policy a total function on the reachable state space rather than a partial one rescued by an oracle.

## Method

**What it is.** FARA (Fallback-Augmented Reframed Agents) augments a Reframe-style student with a fallback generator g_φ and trains it by on-policy distillation on failure-substituted trajectories. Inputs: a student policy π_θ over tool actions A, a set of specialist teachers {π_T}, and a failure-substitution operator. Output: a student π̃_θ whose action set A ∪ {g_φ(s, a_failed)} is closed under failure, plus the trained generator g_φ.

**Load-bearing assumption (stated explicitly).** The method assumes that the teacher's substitute action a*_t recorded on failure states is a valid, in-vocabulary target that a small supervised head g_φ can learn to reproduce from (s_t, a_t), so that the student's action set is genuinely closed under failure substitution by construction. This assumption is load-bearing and is verified, not asserted: (i) D_fail is filtered to states where the teacher's substitute was verified executable and non-degenerate (see Phase 1 filter), and (ii) g_φ is refreshed each iteration on the student's own failure states via DAgger-style aggregation (Phase 3 refresh), so g_φ is optimized for executable substitutes on the student's reachable failure states rather than imitating teacher labels on the teacher's state distribution. The vocabulary of g_φ is A ∪ {ABORT/REPLAN}, where ABORT/REPLAN is an explicit action added to the action set so that closure is achieved by construction even when no in-A substitute exists; success is reported conditional on ABORT/REPLAN being available.

**How it works.**

```pseudocode
# Phase 1: failure-substitution data collection (no SFT cold start)
for each task x, teacher T in {spatial, temporal}:
    rollout π_T on x with stochastic visual-input removal (Reframe's p_rem=0.3)
    at each tool call a_t, with prob p_fail=0.25 mark tool unavailable
    if unavailable: record state s_t, failed action a_t, and teacher's
        subsequent substitute action a*_t (from A) plus returned observation o*_t
    -> raw failure corpus D_raw = {(s_t, a_t, a*_t, o*_t)}
# Filter: keep only states where a*_t was verified executable and non-degenerate
#   (executable = tool(a*_t) available and returned o*_t without error;
#    non-degenerate = a*_t != a_t and a*_t != no-op)
-> filtered failure corpus D_fail = {(s_t, a_t, a*_t, o*_t) in D_raw : valid(a*_t, o*_t)}

# Phase 2: train fallback generator g_φ (supervised on D_fail)
# g_φ: 2-layer MLP head over the student's hidden state (hidden=256, GeLU), vocab = A ∪ {ABORT/REPLAN}
loss_g = -E_{(s,a,a*,o*)~D_fail}[ log g_φ(a* | s, a) ]

# Phase 3: on-policy distillation with closed action set
for iteration:
    rollout student π_θ on x; at each step:
        sample a_t ~ π_θ(·|s_t)
        if tool(a_t) unavailable:
            a_t' ~ g_φ(·|s_t, a_t)          # fallback substitution
            execute a_t', observe o_t'
        else: execute a_t, observe o_t
        # stochastic tool-action removal (extension of Reframe's visual removal)
        with prob p_tool_rem=0.2 replace a_t by g_φ sample anyway
    distill: L = KL( π_T(·|s_t) || π_θ(·|s_t) ) averaged over teachers,
             weighted by evidence-aware reward r_t from Reframe
    update θ
    # DAgger-style aggregation: collect student failure states into D_fail
    for each student failure state (s_t, a_t) in this iteration's rollouts:
        query teacher for substitute a*_t; if valid(a*_t, o*_t): add to D_fail
    refresh g_φ on updated D_fail (1 epoch, Adam lr=1e-3, batch=64)
```

**Why this design.** We chose a learned generator g_φ over Reframe's oracle canonical-action substitution because the oracle presupposes a canonical action always exists, which is false when the tool itself is unavailable; accepting the cost that g_φ can emit a suboptimal substitute, we gain a policy defined on every reachable state. We chose to train g_φ by supervised fitting on teacher failure trajectories rather than by an RL fallback reward because the teacher's substitute action is a dense, low-variance target, accepting the cost that g_φ inherits teacher bias. We chose to extend Reframe's stochastic visual-input removal to stochastic tool-action removal (p_tool_rem=0.2) rather than only substituting on true failures, because the student must learn to invoke g_φ before failures dominate the rollout distribution, accepting the cost of noisier distillation targets. A domain expert would not describe this as a variant of Reframe's OG-MOPD: OG-MOPD repairs with an external canonical action and resumes distillation, whereas FARA makes the student's own action set closed via a learned generator, so the repair is internal to π̃_θ and no oracle is needed at deployment.

**Why it measures what we claim.** The fallback-generator training loss measures closure-under-substitution because the supervised target a*_t is exactly the teacher's substitute action on a state where the original action failed and was verified executable and non-degenerate by the Phase 1 filter; this assumption fails when the teacher itself had no valid substitute (then a*_t is filtered out, so the loss is not computed on degenerate no-ops) or when real failures are correlated across steps (then i.i.d. removal under-represents cascading failures). The stochastic tool-action removal rate p_tool_rem measures the student's robustness to action unavailability because removal simulates the failure distribution the deployed policy faces; this assumption fails when real tool failures are correlated across steps (then i.i.d. removal under-represents cascading failures). The KL distillation term measures observation-aligned teacher agreement because it is computed on states reached after fallback substitution, so the student is distilled on the same observation distribution it will see at deployment; this assumption fails when g_φ's substitutes diverge from the teacher's, in which case the KL measures agreement on off-teacher states rather than aligned ones.

## Contribution

(1) A fallback-augmented action set construction in which a learned generator g_φ, trained on teacher failure trajectories, makes the distilled student policy total over the reachable state space under tool unavailability. (2) An extension of stochastic visual-input removal to stochastic tool-action removal, yielding a failure-substitution augmentation that trains the student to invoke fallback before failures dominate the rollout distribution. (3) An empirical design principle that internalizing repair into the student's action set removes the deployment-time dependence on oracle canonical actions that Reframe's OG-MOPD requires.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 4 representative image/video agentic benchmarks (subset of Reframe's 14) | Matches Reframe's eval surface for comparability while keeping compute within academic resources |
| Primary metric | Task success under injected tool unavailability | Directly probes closure-under-failure claim |
| Baseline | Reframe (OG-MOPD) | Tests oracle-canonical-substitution vs learned fallback |
| Baseline | Mix-RL | Tests multi-teacher distillation without failure closure |
| Baseline | Reframe minus tool-removal | Tests value of proactive fallback invocation |
| Baseline | Rule-based fallback (repeat last successful action) | Tests whether a simple rule-based fallback suffices vs learned g_φ |
| Ablation-of-ours | FARA with frozen g_φ (no refresh) | Isolates benefit of on-policy generator updates |
| Ablation-of-ours | FARA with unfiltered D_fail (includes degenerate no-ops) | Isolates benefit of the Phase 1 validity filter |
| Ablation-of-ours | FARA without ABORT/REPLAN action | Tests whether explicit closure action is necessary when no in-A substitute exists |

### Why this setup validates the claim

The central claim is that making the student's action set closed under tool failure via a learned generator yields better deployment performance than repairing with an external oracle action. Reframe tests the sub-claim that oracle canonical substitution is insufficient when no canonical action exists: on tasks where the unavailable tool has no in-A canonical replacement, Reframe's repair either fails or substitutes a semantically wrong action, so any success gap concentrated on those tasks isolates the oracle assumption as the bottleneck. Mix-RL tests whether failure closure matters at all versus plain multi-teacher distillation by exposing it to the same injected unavailability without a fallback mechanism. The rule-based fallback baseline tests whether the learned generator provides a unique benefit over a simple heuristic (repeat last successful action), particularly on the no-canonical-substitute subset where the heuristic has no valid action to repeat. The frozen-g_φ ablation tests whether the generator must track the student's evolving state distribution or whether a one-shot supervised fit suffices; if the frozen version matches full FARA, the on-policy refresh is unmotivated. The unfiltered-D_fail ablation tests whether the Phase 1 validity filter (removing degenerate no-ops) is necessary for closure. The no-ABORT/REPLAN ablation tests whether the explicit closure action is necessary when no in-A substitute exists. Success under injected tool unavailability is the right primary metric because it is the exact deployment condition the method claims to handle, and it is measured on the same observation distribution the student is distilled on, so a gain cannot be attributed to distribution shift artifacts.

### Expected outcome and causal chain

**vs. Reframe (OG-MOPD)** — On a case where the student attempts a tool that is unavailable and the task has no canonical in-A substitute (e.g. a spatial tool fails but only temporal actions are valid), Reframe's oracle repair emits a degenerate or mismatched action because it presupposes a canonical replacement exists. Our method instead samples from g_φ, which was trained on teacher failure states to map (s, a_failed) to a valid substitute (including ABORT/REPLAN when no in-A substitute exists), so the rollout continues on a reachable state. We expect a noticeable success gap concentrated on the no-canonical-substitute subset, with near-parity on tasks where canonical substitutes exist.

**vs. Mix-RL** — On a case where a tool becomes unavailable mid-trajectory, Mix-RL has no fallback operator, so the rollout either aborts or the policy emits an invalid action and the trajectory terminates with a failure. Our method keeps the action set closed via g_φ, so the student can recover and continue distilling on the post-failure observation stream. We expect Mix-RL to degrade sharply as injected unavailability rises while FARA degrades gracefully, producing a widening gap as failure rate increases.

**vs. Reframe minus tool-removal** — On a case where the student would only encounter failure states late in deployment, the no-removal variant never trains g_φ to be invoked proactively, so when a true failure occurs the generator is queried on out-of-distribution states and emits poor substitutes. Our p_tool_rem=0.2 forces early and frequent g_φ invocation during distillation, so the generator is exercised on the states it will actually see. We expect the no-removal variant to show a larger gap on early-step failures than on late-step failures.

**vs. Rule-based fallback (repeat last successful action)** — On a case where the failed tool has no valid in-A substitute and the last successful action is also unavailable or semantically wrong, the rule-based fallback repeats an invalid action and the rollout terminates. Our learned g_φ instead maps (s, a_failed) to a valid substitute (including ABORT/REPLAN), so we expect the rule-based baseline to fail specifically on the no-canonical-substitute subset while FARA succeeds.

**vs. FARA with frozen g_φ (no refresh)** — On a case where the student's failure states drift from the teacher's failure-state distribution as θ updates, the frozen g_φ is queried on out-of-distribution states and emits poor substitutes. Our on-policy refresh (DAgger-style aggregation of student failure states into D_fail each iteration) keeps g_φ aligned with the student's reachable failure states, so we expect the frozen variant to show a larger gap on late-iteration failure states than on early-iteration ones.

**vs. FARA with unfiltered D_fail** — On a case where the teacher had no valid substitute (a*_t is a no-op), the unfiltered corpus trains g_φ to imitate degenerate actions, so g_φ emits no-ops on exactly the hard states the method claims to fix. Our Phase 1 filter removes these degenerate labels, so we expect the unfiltered variant to show lower success on the no-canonical-substitute subset.

**vs. FARA without ABORT/REPLAN action** — On a case where no action in A is a valid substitute, g_φ's vocabulary is restricted to A and cannot close the set, so it must emit an invalid in-A action. Our method augments the vocabulary with ABORT/REPLAN, so closure is achieved by construction; we expect the no-ABORT/REPLAN variant to fail on the subset where no in-A substitute exists.

### What would falsify this idea

If FARA's gain over Reframe is uniform across all task subsets rather than concentrated on tasks lacking a canonical substitute, then the learned-generator mechanism is not what drives the improvement and the closure-under-failure claim is unsupported. Likewise, if the frozen-g_φ ablation matches full FARA, the on-policy refresh is doing no work and the design's central training loop is unmotivated.

## References

1. Reframe: Observation-Aligned Multi-Teacher On-Policy Distillation for Agentic Visual Reasoning
