# Retention-Gated Policy Refinement for Multimodal Tool-Use Agents

## Motivation

OpenVLHarness and the broader family of reflective agent harnesses (hierarchical-ReAct, spatial-tool-RL, prompt-evolution branches) refine their tool-use policy — prompts, tool schemas, and program templates — as multimodal memory accumulates, but no branch enforces that a refinement preserves competence on tasks the agent had already solved. The root cause is structural: the coding interface executes generated programs over a growing addressable memory without any record of which past traces depended on the current policy's specific behaviors, so a revision that improves new tasks can silently corrupt solved ones and nothing detects the regression before deployment. This is the same monotonicity gap that recurs across every branch — memory growth and policy refinement are coupled but never certified against each other.

## Key Insight

If every candidate policy revision must re-derive its outputs on a retention buffer of solved-task traces and is admitted only when a paired, decision-level non-degradation check passes on that buffer, then the deployed policy sequence is monotone non-decreasing on the retained solved set by construction — a property that reflective prompt evolution and RL tool-policy refinement do not establish because they optimize only forward performance on new tasks.

## Method

**What it is.** We introduce MPRI (Monotone Policy-Refinement Invariant), a gate wrapped around OpenVLHarness's coding interface. Inputs: a current deployed policy π_t (prompt + tool schema + program template), a candidate reflective revision π_{t+1} produced by the harness's own reflection loop, and a retention buffer B of solved-task traces. Output: either π_{t+1} is deployed (invariant holds) or π_{t+1} is rejected and π_t remains, with the rejection logged as a memory entry.

**Load-bearing assumption (stated explicitly).** MPRI's paired delta is valid only if the two sides of the comparison are drawn from the same sampling process. Concretely: the cached final answer of π_t on each buffer task must be a valid, variance-free counterfactual for a fresh rollout of π_t under the same conditions as the candidate rollout. If π_cand is re-executed as a stochastic tool-use rollout while π_t is represented by a single cached trace, the delta conflates policy revision with rollout stochasticity and artifact drift — the classic paired-comparison validity failure (pass@k vs. pass^k, Chen et al. 2021; AgentBench, Liu et al. 2023; WebArena, Zhou et al. 2023). With τ=0.0 and a strict n_degraded==0 rule, a single unlucky candidate rollout produces a spurious REJECT, and a lucky cached trace biases toward rejection. The repair below makes the delta symmetric and variance-controlled by re-executing BOTH policies on the buffer with matched sampling (same seeds, temperature, k>1 samples per task) and comparing distributions rather than point estimates.

**How it works.**

```pseudocode
# Retention buffer entry: (task q, frozen multimodal artifacts A_q, reference program p*_q, gold answer y_q)
# B is populated whenever a task is first solved by the deployed policy.
# Sampling config for matched re-execution: k=5 samples per task, temperature=0.7, fixed seed set S={s1..s5}.

MPRI_gate(pi_t, pi_cand, B):
    # Phase 1: matched re-execution of BOTH policies on retained solved traces
    for (q, A_q, p*_q, y_q) in B:
        for seed in S:
            out_cand[q, seed] = execute_coding_interface(pi_cand, q, A_q, seed=seed, temperature=0.7)
            out_curr[q, seed] = execute_coding_interface(pi_t,     q, A_q, seed=seed, temperature=0.7)

    # Phase 2: decision-level non-degradation check (paired, per-task, distributional)
    for q in B:
        s_cand[q, seed] = task_scorer(out_cand[q, seed], y_q)   # binary or graded task metric
        s_curr[q, seed] = task_scorer(out_curr[q, seed], y_q)
        delta[q, seed]  = s_cand[q, seed] - s_curr[q, seed]
        # per-task paired sign test over k samples: count seeds where delta < 0
        n_neg[q] = count(delta[q, seed] < 0 for seed in S)
        # task is degraded if the majority of matched samples show negative delta
        degraded[q] = (n_neg[q] > k/2)   # strict majority; k=5 so threshold is 3

    # Phase 3: invariant test
    n_degraded = count(degraded[q] for q in B)
    n_improved = count(n_neg[q] == 0 for q in B)   # all k samples non-negative
    if n_degraded == 0:
        deploy(pi_cand); B.extend(newly_solved_traces(pi_cand))   # buffer grows monotonically
        return ACCEPT
    else:
        log_rejection(pi_cand, degraded_tasks={q: n_neg[q] for q in B if degraded[q]}); return REJECT

# Retention buffer maintenance: capacity C = 512 traces, reservoir-sampled by task-cluster
# so that B stays representative as memory grows; traces never evicted once admitted
# unless a task is re-solved by a strictly better policy under the same gate.
# Compute budget: each gate call costs 2 * |B| * k rollouts. With |B|=512, k=5, that is 5120 rollouts per revision.
# Cheaper variant: stratified subsample of B of size C_sub=128 per revision, with a 95% Wilson upper bound on n_degraded;
# accept only if the upper bound is 0. This reduces cost to 1280 rollouts per revision.
```

**Why this design.** We chose a *paired per-task* non-degradation check over a *global aggregate* score comparison because aggregate means can hide single-task regressions that matter for safety-critical tool use, accepting the cost that the gate is stricter and rejects more revisions. We chose *matched re-execution of both π_t and π_cand* over a single cached π_t trace because the cached trace is a high-variance single-sample estimate that contaminates the delta with rollout stochasticity; matched seeds and k=5 samples per task make the paired delta a distributional comparison, accepting the cost of doubling rollout compute per gate call. We chose a *strict* threshold τ=0.0 over a tolerance band because the invariant's value is exactly that it cannot silently trade a solved task for a new one, accepting the cost of lower revision throughput. We chose a *reservoir-sampled, cluster-stratified* buffer over a recency window because recency windows drift toward recent task types and would let the invariant be satisfied on an unrepresentative set, accepting the cost of extra bookkeeping. A domain expert would not describe MPRI as a variant of Reflexion or self-refine: those loops accept a revision whenever the *new* task succeeds and never re-check the solved set, whereas MPRI's load-bearing object is the paired delta on a certified retention buffer.

**Why it measures what we claim.** The computational quantity `n_degraded = count(degraded[q])` where `degraded[q] = (n_neg[q] > k/2)` measures *monotone non-degradation of the deployed policy on previously solved tasks* because the assumption that `task_scorer(out_cand[q, seed], y_q)` is a faithful per-task competence signal AND that the matched-seed re-execution of π_t provides a valid counterfactual makes the paired delta equivalent to a competence change on that task; this assumption fails when the scorer is a weak proxy (e.g., exact-match on a task with multiple valid answers) or when rollout variance exceeds the majority threshold, in which case the delta reflects scorer brittleness or sampling noise rather than a real regression. The retention buffer's cluster-stratified reservoir sample measures *representativeness of the solved-task distribution* because the assumption that task clusters are stable across memory growth makes stratified sampling equivalent to an unbiased estimate of solved-set competence; this assumption fails when new memory entries introduce a genuinely new task cluster, in which case the buffer under-covers the new cluster and the invariant is certified on a stale distribution. The gate's binary ACCEPT/REJECT on `n_degraded == 0` measures *safety of the reflective revision* because the assumption that the candidate's behavior on buffer tasks is predictive of its behavior on the deployed distribution (i.e., the buffer is exchangeable with deployment traffic) makes zero buffer degradation equivalent to zero expected deployment degradation; this assumption fails under distribution shift between buffer and live traffic, in which case the gate certifies monotonicity on the buffer but not on deployment.

## Contribution

(1) MPRI, a retention-buffer-gated non-degradation invariant wrapped around OpenVLHarness's coding interface that makes reflective policy refinement monotone on the solved-task set by construction. (2) A design principle — paired per-task delta on cached traces, strict τ=0, cluster-stratified reservoir buffer — showing that reflective prompt evolution can be made safe under multimodal memory growth without re-executing the incumbent policy. (3) A rejection-log memory artifact that records which candidate revisions degraded which solved tasks, providing a diagnostic surface for the harness's reflection loop.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OpenVLHarness 22-benchmark suite, sequential task streams | Matches deployment where solved tasks accumulate over time |
| Primary metric | Solved-task retention rate under revision (per-task paired delta, k=5 matched samples) | Directly measures the invariant MPRI claims to enforce |
| Baseline | Base MLLM, increased inference budget (matched to MPRI's rollout count) | Tests whether raw capability subsumes guarded refinement |
| Baseline | Reflexion / self-refine loop, unguarded | Isolates the value of the non-degradation gate |
| Baseline | Aggregate-score acceptance gate (global mean delta > 0) | Tests paired-per-task vs global-mean decision rule |
| Baseline | Naive replay-finetuning (replay buffer of solved traces, no decision-level gate) | Isolates what the decision-level gate adds beyond data replay |
| Ablation-of-ours | MPRI with tolerance band tau > 0 (tau=0.1) | Tests whether strict zero-degradation is load-bearing |
| Ablation-of-ours | MPRI with k=1 (single-sample, cached pi_t) | Tests whether variance control is load-bearing |
| Ablation-of-ours | MPRI with buffer size C_sub=128 stratified subsample + 95% Wilson bound | Tests cheaper variant's retention vs full buffer |
| Diagnostic | Live-traffic probe set (held-out 10% of tasks) | Measures buffer-deployment exchangeability: correlation between buffer paired-delta and probe paired-delta per revision; false-accept rate vs buffer size C |

### Why this setup validates the claim

The central claim is that an unguarded reflective revision loop silently trades solved tasks for new ones, and that a paired per-task non-degradation check on a retention buffer prevents this without a capability ceiling. The dataset must therefore present sequential task streams where the policy is revised after solves, so that previously solved tasks remain available for re-checking — the OpenVLHarness suite provides exactly this mix of tool-use tasks with scored gold answers. The primary metric, retention rate on previously solved tasks, is the only metric that can detect the predicted failure: an aggregate accuracy number would rise even while individual solved tasks regress, so it cannot falsify or confirm the invariant. The unguarded Reflexion baseline tests the sub-claim that reflection loops cause regression; the aggregate-gate baseline tests the sub-claim that per-task pairing rather than mean comparison is what catches it; the inference-budget baseline tests the sub-claim that the gain is not just more compute; the naive replay-finetuning baseline tests whether the decision-level gate adds anything beyond simply replaying solved traces. The tau>0 ablation tests whether the strict threshold is the load-bearing design choice or merely a conservative setting; the k=1 ablation tests whether variance control via matched re-execution is load-bearing; the C_sub=128 ablation tests the cheaper variant's retention. The live-traffic probe set directly measures the exchangeability assumption: if buffer paired-delta correlates poorly with probe paired-delta, the gate's certification does not transfer to deployment. Together these form a falsifiable test: if MPRI's advantage appears only where the predicted regression mechanism operates, the causal story holds.

### Expected outcome and causal chain

**vs. Base MLLM, increased inference budget** — On a case where a task requires chaining a visual grounding step with a code-edit tool call, the baseline produces a plausible but unverified answer because it has no mechanism to freeze and re-check previously solved traces. Our method instead deploys revisions only after re-executing them on the retention buffer, because the paired delta exposes any solved-task regression before deployment. We expect parity on first-attempt tasks but a clear retention-rate gap on the sequential stream, concentrated on multimodal tool-chaining subsets.

**vs. Reflexion / self-refine, unguarded** — On a case where reflection over-corrects a working program template after a single failure, the baseline produces a revision that fixes the new task but breaks an earlier solved one, because it accepts any revision whose new task succeeds without re-checking the solved set. Our method instead rejects that revision because the buffer re-execution shows a negative paired delta on the earlier task. We expect the unguarded loop to show higher new-task success but visibly lower solved-task retention, while MPRI holds retention near the pre-revision level. We will also report the base rate of silent solved-task regression in the unguarded loop across the 22 benchmarks; if this rate is low, the gate's value scales with it.

**vs. Aggregate-score acceptance gate** — On a case where a candidate improves three easy tasks and regresses one hard solved task, the baseline accepts because the global mean rises, because an aggregate comparison cannot see the single regression. Our method instead rejects because the per-task paired check flags that one degraded task. We expect MPRI to show a retention advantage concentrated on hard-task subsets, with near-parity on easy subsets — a subset-localized gap rather than a uniform one.

**vs. Naive replay-finetuning** — On a case where replaying solved traces updates the policy but does not certify per-task non-degradation, the baseline may still regress on tasks whose traces were not replayed or whose artifacts drifted. Our method instead gates every revision on the full retention buffer with matched re-execution, so we expect MPRI to show higher retention on tasks not covered by the replay sample, isolating the decision-level gate's contribution beyond data replay.

**vs. MPRI with k=1 (single-sample, cached π_t)** — On a case where rollout stochasticity produces a spurious negative delta, the k=1 variant rejects a valid revision or accepts a lucky one. Our method with k=5 matched samples and majority vote reduces this variance, so we expect the k=1 variant to show lower revision throughput and higher false-reject rate, with retention differences concentrated on tasks with high run-to-run variance.

**vs. MPRI with tolerance band τ=0.1** — On a case where a small but real regression (delta=-0.05) is tolerated, the τ>0 variant accepts a revision that degrades a solved task. Our method with τ=0.0 rejects it. We expect the τ>0 variant to show slightly higher new-task success but lower solved-task retention, confirming strict zero-degradation is load-bearing.

**vs. MPRI with C_sub=128 stratified subsample + Wilson bound** — On a case where the subsample misses a degraded task, the cheaper variant accepts a revision that the full buffer would reject. We expect the cheaper variant to show retention close to full-buffer MPRI but with a measurable false-accept rate that increases as C_sub decreases, quantifying the compute-retention trade-off.

### What would falsify this idea

The claim is wrong if MPRI's retention advantage over the unguarded loop is uniform across all subsets rather than concentrated where the predicted regression mechanism operates, or if the aggregate-gate baseline matches MPRI's per-task retention — either pattern would mean the paired per-task check is not what produces the safety, and the invariant's causal story collapses. Additionally, if the live-traffic probe set shows low correlation between buffer paired-delta and probe paired-delta (e.g., Pearson r < 0.3) or a high false-accept rate that does not decrease with buffer size C, then the exchangeability assumption fails and the gate's certification does not transfer to deployment.

## References

1. OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory
