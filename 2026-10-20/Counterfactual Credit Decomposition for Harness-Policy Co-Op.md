# Counterfactual Credit Decomposition for Harness-Policy Co-Optimization

## Motivation

Self-HPO's P-step trains the policy under a selected harness using a single monolithic scalar reward, so when a harness edit improves benchmark performance the gradient cannot identify which harness component (prompt scaffold, tool interface, retry policy, output validator) actually caused the improvement, and conversely a component that is individually harmful can be retained because it co-occurs with a helpful sibling. This is a structural credit-assignment failure, not a data-scarcity failure: the harness is a compositional object edited as a bundle, so the aggregate reward is a sum over an unknown interaction structure that Self-Play Fine-Tuning-style scalar objectives treat as a black box. The consequence is that harness search drifts toward components that are merely correlated with reward, and the H-step's failure introspection has no signal telling it which component to revise.

## Key Insight

A harness component's marginal contribution is identified by the counterfactual reward difference between the realized harness and a harness with that component ablated, evaluated on the same task batch with the same policy, so component credit is a contrast between two executions that differ in exactly one component rather than a decomposition of a shared scalar.

## Method

**What it is.** C3 (Counterfactual Component Credit) is a reward-decomposition layer that sits inside Self-HPO's P-step. Its inputs are a task batch B (|B| = 64 tasks per round), the current policy pi_theta, and the selected harness H = (c_1, ..., c_K) with K = 6 components: c_1 = prompt scaffold, c_2 = tool interface, c_3 = retry/repair policy, c_4 = output validator, c_5 = context-compaction rule, c_6 = test-runner invocation. Its output is a vector of per-component credits (r_1, ..., r_K) plus a per-component, per-task advantage that drives K separate policy-gradient updates.

**Load-bearing assumption (stated explicitly).** The method assumes that replacing a harness component c_k with its null version null_component(c_k) is a valid intervention that changes only c_k's scaffolding/decision contribution while leaving the task distribution and the policy's task-relevant observation set unchanged. In code-generation harnesses this assumption is not automatic: removing the test-runner or output validator also removes the agent's observation of failing tests/errors, which is a change to the POMDP observation function, not a null edit. To make the assumption testable and to separate scaffolding value from information value, C3 uses two ablation modes per component and reports both:
- **Information-preserving ablation** (primary credit signal): null_component(c_k) is replaced by a no-op that still surfaces the same observation channel but does not gate or alter the agent's decisions. Concretely: for c_4 (output validator), the no-op still prints the validator's error messages to the agent but never blocks or retries on them; for c_6 (test-runner), the no-op still runs the tests and shows the output but does not feed failures into the retry policy; for c_3 (retry policy), the no-op keeps test output visible but disables the retry decision rule (single attempt); for c_1 (prompt scaffold), the no-op replaces the scaffold with a minimal task statement that preserves all task-relevant information; for c_2 (tool interface), the no-op exposes the same tool signatures but removes any wrapper-level argument validation; for c_5 (context-compaction), the no-op keeps all context (no compaction) so no information is lost.
- **Information-destroying ablation** (control): the original null_component(c_k) that removes the observation channel entirely (e.g., drop the test-runner output).
The difference between the two deltas is the information term; the information-preserving delta is the scaffolding term used for credit.

**How it works.**

```pseudocode
Input: policy pi_theta, harness H=(c_1..c_K) with K=6, task batch B (|B|=64), ablation budget m=8 permutations, baseline harness H0
for each round of P-step:
    # Phase 1: realized execution (cached for reuse across ablations)
    for task t in B:
        rollout tau_t = execute(pi_theta, H, t)          # full harness
        R_t = verifier(tau_t)                             # scalar 0/1 or test-pass fraction
        cache[t] = (tau_t, R_t)
    # Phase 2: counterfactual ablations (same policy, same batch)
    for component k in 1..K:
        H_abl_k = H with c_k replaced by information_preserving_null(c_k)   # see assumption block
        H_destroy_k = H with c_k replaced by information_destroying_null(c_k)  # control
        for task t in B:
            tau_t^k = execute(pi_theta, H_abl_k, t)
            R_t^k = verifier(tau_t^k)
            tau_t^k_destroy = execute(pi_theta, H_destroy_k, t)
            R_t^k_destroy = verifier(tau_t^k_destroy)
        delta_k(t) = R_t - R_t^k                          # per-task scaffolding marginal contribution
        delta_k_destroy(t) = R_t - R_t^k_destroy          # per-task information+scaffolding contribution
        info_k(t) = delta_k_destroy(t) - delta_k(t)       # information term (reported, not used for credit)
        delta_k = mean_t delta_k(t)                       # batch-mean scaffolding credit
    # Phase 3: interaction correction via Shapley-style sampling (m permutations)
    for s in 1..m:
        perm = random_permutation(1..K)
        for prefix j in 1..K:
            H_prefix = H restricted to perm[1..j] with information-preserving nulls for omitted components
            R_prefix = mean_t verifier(execute(pi_theta, H_prefix, t))
        phi_k += (1/m) * shapley_increment(perm, R_prefix, k)
    # Phase 4: component-specific policy updates with task-conditional credit
    for k in 1..K:
        credit_k(t) = 0.5 * delta_k(t) + 0.5 * phi_k      # per-task credit (phi_k is batch-level, broadcast)
        A_k(t) = credit_k(t) * (R_t - baseline_k)         # per-task credit-weighted advantage
        theta <- theta + lr * grad log pi_theta(tau_t) * A_k(t)
    # Phase 5: hand credit back to H-step
    send credit vector (mean_t credit_k(t)) and per-task credit_k(t) to H-step; components with mean_t credit_k(t) < 0 become revision targets
```

Hyperparameters: ablation budget m = 8 permutations per round; credit blend 0.5/0.5 between single-ablation delta and Shapley phi; baseline_k is an exponential moving average of R_t^k with decay 0.9; lr follows the base RL recipe of Self-HPO; |B| = 64 tasks per round; K = 6 components. Compute budget: per round, Phase 1 uses |B| = 64 rollouts, Phase 2 uses 2 * K * |B| = 768 rollouts (information-preserving + information-destroying), Phase 3 uses m * K * |B| = 8 * 6 * 64 = 3072 rollouts, total ~3904 rollouts per round versus Self-HPO's ~64 rollouts per round (a ~61x rollout multiplier). To keep wall-clock feasible, Phase 3 Shapley sampling is run on a stratified subset of |B_shapley| = 16 tasks (stratified by task difficulty quartile from the previous round's pass rate), reducing Phase 3 to 8 * 6 * 16 = 768 rollouts and total to ~1600 rollouts per round (~25x Self-HPO). Realized-execution rollouts from Phase 1 are cached and reused as the full-harness condition in Phase 3 prefixes where the prefix equals H.

**Why this design.** We chose per-component single ablation as the primary credit signal over a pure Shapley estimator because ablation gives an exact, low-variance contrast for the common case where components are near-additive, accepting the cost that it under-credits components whose value is only realized jointly with another (e.g., a retry policy that only pays off with a validator). We therefore add Shapley-style permutation sampling as a correction term rather than replacing ablation, accepting m extra rollouts per round because the interaction term is exactly what the H-step needs to avoid deleting a component that is only useful in combination. We chose to keep the policy fixed during all counterfactual executions within a round over re-training between ablations, because a fixed policy makes the contrast a true intervention on the harness (the only variable that changes is c_k); the cost is that credits are stale with respect to the policy after Phase 4, which we accept because the H-step consumes credits before the next round anyway. We chose to send the credit vector back to the H-step as revision targets over letting the H-step re-derive failures from scratch, because a negative credit_k is a direct causal signal that a specific component hurt, whereas Self-HPO's failure introspection is a free-form LLM judgment with no counterfactual grounding. We chose information-preserving nulls over information-destroying nulls for the credit signal because the latter confounds scaffolding value with information value, and we report the information term separately as a diagnostic. A domain expert would not describe this as a variant of VinePPO or process reward models: those assign credit to *steps within a single trajectory* under a fixed environment, whereas C3 assigns credit to *environment components* by intervening on the environment itself, which requires the harness to be a compositional object and the ablation to be a valid intervention — a structure absent from step-level MC value estimation.

**Why this measures what we claim.** The quantity delta_k(t) = R_t - R_t^k measures the causal scaffolding marginal contribution of harness component c_k on task t because the only difference between the two executions is c_k's decision rule, under the assumption that the information-preserving null keeps the observation set fixed; this assumption fails when the null still changes the information available (e.g., a validator no-op that prints errors but the agent ignores them), in which case delta_k(t) reflects residual information value rather than pure scaffolding value. The Shapley term phi_k measures the component's average marginal contribution across orderings because permutation sampling over component subsets is the Shapley value estimator, under the assumption that the m=8 sampled permutations are representative of the component interaction structure; this assumption fails when K=6 and m=8 is small relative to the number of orderings (720), in which case phi_k reflects only the interactions captured by the sampled orderings. The per-task credit-weighted advantage A_k(t) = credit_k(t) * (R_t - baseline_k) measures per-component policy improvement because weighting the trajectory advantage by the component's per-task credit attributes the policy update to the component responsible for the reward change on that task, under the assumption that credit is task-conditional (which we enforce by using delta_k(t) rather than the batch mean); this assumption fails when the per-task ablation delta is itself noisy (single rollout per task), in which case A_k(t) reflects rollout noise rather than true task-conditional credit.

## Contribution

(1) A counterfactual reward-decomposition layer for harness-policy co-optimization that replaces Self-HPO's monolithic P-step scalar with a per-component credit vector computed from single-component ablations blended with Shapley-style permutation sampling, yielding K component-specific policy updates instead of one. (2) A design principle: negative per-component credit serves as a causal revision target for the H-step, converting free-form failure introspection into a counterfactually grounded harness-edit signal. (3) An ablation-budget analysis identifying when single-ablation credit suffices versus when interaction correction is necessary, expressed as a measurable condition on component additivity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-bench Lite (300 test tasks) with composable harnesses (tests, validators, retries) | Exercises joint-component interactions the credit must resolve |
| Dataset (transfer) | HumanEval+ (164 tasks) with a second harness decomposition (prompt scaffold, tool interface, retry policy, output validator) | Tests whether credit vector generalizes across harness decompositions |
| Primary metric | Pass@1 on held-out tasks after equal harness-policy rounds (10 rounds) | Detects whether credit improves both harness selection and policy |
| Secondary metric | Per-task credit variance and information-term magnitude (info_k) | Diagnoses whether credit is task-conditional and information-confounded |
| Baseline | Self-HPO (free-form LLM failure introspection) | Tests whether counterfactual credit beats judgment-based revision |
| Baseline | Harness-only optimization | Tests whether credit-weighted policy updates add value |
| Baseline | Policy-only optimization | Tests whether harness credit is the missing signal |
| Baseline | DSPy-style prompt optimization (harness search without policy credit) | Tests against environment-design prior work |
| Ablation-of-ours | C3 with ablation term only (no Shapley correction) | Isolates whether interaction correction drives any gain |
| Ablation-of-ours | C3 with information-destroying nulls only (no information-preserving nulls) | Isolates whether information-preserving ablation is necessary |
| Ablation-of-ours | C3 with batch-mean credit (no per-task credit) | Isolates whether task-conditional credit drives gain |

### Why this setup validates the claim

The central claim is that per-component counterfactual credit lets the H-step revise the harness causally rather than by free-form judgment, and that this yields better harness-policy co-adaptation. Self-HPO is the right primary foil: it shares the identical alternating-round loop but replaces counterfactual credit with LLM failure introspection, so any gap isolates the credit mechanism rather than the outer loop. The harness-only and policy-only baselines decompose the two sub-claims — that credit improves harness revision, and that credit-weighted policy updates improve the policy — so a gain cannot be attributed to one arm alone. The DSPy-style prompt-optimization baseline tests whether C3's credit adds value over environment-design search that does not use policy credit. The ablation-only variant (no Shapley) is the diagnostic for joint-component interactions: if interactions matter, removing permutation correction should specifically hurt on tasks where a component (e.g., retry) only pays off with another (e.g., validator). The information-destroying-null ablation tests the load-bearing assumption directly: if information-preserving nulls are necessary, the information-destroying variant should show systematically different (over-credited information-bearing) credit vectors and worse harness revisions. The batch-mean-credit ablation tests whether task-conditional credit is necessary: if a component helps on some tasks and hurts on others, batch-mean credit should cancel and underperform. Pass@1 after 10 equal rounds is the right metric because it reflects the compounding of both arms across rounds, which is exactly where stale or misattributed credit should surface as divergence rather than a one-step effect. The HumanEval+ transfer experiment tests whether the credit vector generalizes across harness decompositions, so the significance claim is not tied to a single Self-HPO instantiation.

### Expected outcome and causal chain

**vs. Self-HPO** — On a case where a retry/repair policy is only useful in combination with the validator, Self-HPO's free-form introspection attributes the failure to the prompt scaffold and revises it, because it has no counterfactual grounding to see that the retry-validator pair carries the value. Our method's Shapley correction assigns positive joint credit to that pair, so the H-step retains both and revises elsewhere. We expect a noticeable pass@1 gap concentrated on tasks requiring multi-attempt repair, with near-parity on single-shot tasks.

**vs. Harness-only optimization** — On a case where the harness is already near-optimal but the policy keeps emitting malformed tool calls, harness-only search keeps proposing harness variants that cannot fix a policy-level defect, because it has no component-level gradient into the policy. Our method's per-task credit-weighted advantage drives per-component policy updates that target the malformed-call behavior directly. We expect harness-only to plateau early while our method continues improving in later rounds, with the gap widening as rounds progress.

**vs. Policy-only optimization** — On a case where the policy is competent but the harness lacks an output validator, policy-only RL receives noisy rewards it cannot cleanly attribute, because the harness defect is invisible to a fixed environment. Our method's information-preserving ablation identifies the validator as high-credit and the H-step inserts it, cleaning the reward signal for subsequent rounds. We expect policy-only to show high reward variance and slow gains, while our method shows lower variance and faster early-round improvement.

**vs. DSPy-style prompt optimization** — On a case where the harness prompt scaffold is already well-tuned but the retry policy is harmful, DSPy-style search keeps revising the prompt because it has no component-level credit and no policy gradient. Our method's per-task credit identifies the retry policy as negative-credit and the H-step removes it. We expect DSPy-style to show flat pass@1 after early rounds while our method continues to improve.

**vs. C3 ablation-only (no Shapley)** — On interaction-heavy tasks (multi-attempt repair), ablation-only under-credits the retry-validator pair because single ablation cannot see the joint value. We expect ablation-only to match full C3 on single-shot tasks but to underperform on multi-attempt repair tasks, with the gap concentrated there.

**vs. C3 information-destroying nulls only** — On tasks where the test-runner is information-bearing, information-destroying nulls over-credit the test-runner (because removing it collapses pass rate for information reasons) and under-credit pure-scaffolding components. We expect the information-destroying variant to show systematically higher credit for c_4 and c_6 and worse harness revisions (e.g., retaining a harmful validator because its information value masks its scaffolding harm).

**vs. C3 batch-mean credit** — On tasks where a component helps on some tasks and hurts on others (e.g., context-compaction helps short tasks but hurts long ones), batch-mean credit cancels to near zero and the H-step neither retains nor removes the component. We expect batch-mean credit to show lower per-task credit variance and worse pass@1 on the mixed-effect subset.

### What would falsify this idea

If the ablation-only variant matches full C3 on the interaction-heavy subset (multi-attempt repair tasks), the Shapley correction is doing nothing and the central claim about joint-component credit is wrong. Likewise, if our gain over Self-HPO is uniform across subsets rather than concentrated where the predicted failure mode (joint-component retention) occurs, or if the information-preserving and information-destroying nulls yield the same credit vectors, the credit mechanism is not the cause.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
