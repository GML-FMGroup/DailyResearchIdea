# Closed-Loop Task Distribution Tracking for Harness-Induced Failure Shift

## Motivation

Automated task pipelines such as SWE-rebench supply fresh tasks but treat the evaluation distribution as stationary, so when the harness (prompt scaffold, tool interface, retry policy) co-evolves with the coding agent, the failure modes the task pool covers silently drift away from the failure modes the agent now exhibits; Self-HPO compounds this by selecting harnesses against a hard-task set that is itself never re-derived after the harness changes, so its next harness search is seeded with failures induced by the previous harness. The root cause is structural: no existing loop closes over the joint (harness, failure-mode) distribution — tasks are collected once, harnesses are optimized against that fixed set, and the resulting shift is never fed back into task selection.

## Key Insight

Harness-induced failure shift is observable as a change in the conditional distribution of execution-trace signatures given harness, so re-weighting the task pool by the density ratio between post-update and pre-update trace distributions restores coverage of the new failure modes without requiring the harness designer or policy to introspect on their own weaknesses.

## Method

**What it is.** HATRIC (Harness-Aware Task Re-weighting for Induced Closure) is a layer wrapped around an existing harness-optimization loop (e.g., Self-HPO). Its inputs are (i) the current task pool with per-task execution traces, (ii) the pre-update harness h_{t-1} and post-update harness h_t, and (iii) a trace-signature extractor. Its output is a re-weighted task mixture and a small synthesized supplement targeted at under-covered failure modes, which is fed back into both harness search and policy training.

**How it works.**

```pseudocode
# Called once per harness update t
INPUT: task pool T, traces under h_{t-1} and h_t, signature map phi
HYPER: signature vocabulary V (from failure taxonomy), bandwidth k=5, cap M=200 synthesized tasks, weight floor eps=0.02,
       collision threshold tau_coll=0.10, secondary discriminator D (exit code + tool-call error type)

1. For each task tau in T and each harness h in {h_{t-1}, h_t}:
     s(tau,h) = phi(trace(tau,h))     # signature = (failure class, tool-call pattern, test-diff locus)
     # phi is constructed as follows: failure class is the label from a fixed 12-class taxonomy
     # (compile error, test assertion, timeout, wrong API call, missing import, syntax error,
     #  runtime exception, resource exhaustion, permission error, network error, flaky test, other);
     # tool-call pattern is the ordered sequence of tool names invoked (e.g., [read_file, edit_file, run_tests]),
     # discretized by collapsing repeated calls and truncating to the first 5 distinct tools;
     # test-diff locus is the set of test file paths whose pass/fail status changed, discretized
     # to the top-3 most frequent loci in the pool plus an 'other' bucket.

2. Estimate pre/post signature distributions over V:
     p_pre(v)  = (1/|T|) * sum_tau 1[s(tau,h_{t-1}) = v]
     p_post(v) = (1/|T|) * sum_tau 1[s(tau,h_t)     = v]

3. Density ratio per signature:  r(v) = p_post(v) / (p_pre(v) + delta),  delta=1e-3
   Per-task importance weight:    w(tau) = r(s(tau,h_t)) / Z,  Z = sum_tau r(s(tau,h_t))
   Clip: w(tau) <- max(w(tau), eps)
   # Fallback for p_pre(v)=0: set r(v) = 1.0 (uniform weight) and flag v for mandatory synthesis in step 4.

3b. Signature collision diagnostic (assumption A test):
     For each signature v in V with at least 5 tasks, compute the collision rate
     c(v) = 1 - max_{failure class c} (count of tasks in v with class c) / (count of tasks in v).
     If c(v) > tau_coll, refine phi for that cell by appending the secondary discriminator D
     (exit code bucket: 0, 1-127, 128-255; tool-call error type: none, timeout, permission, not-found, other)
     to the signature, splitting v into sub-signatures v' = (v, D). Recompute p_pre, p_post, r(v') for the split cells.
     Report the pool-level collision rate C = (1/|V|) * sum_v c(v) as a diagnostic.

4. Identify under-covered signatures: U = { v : p_post(v) > 0 and w(tau)=eps for all tau with s(tau,h_t)=v }
   For each v in U (up to M total):
     synthesize a task tau_new by mutating the nearest existing task whose post-harness
     trace shares v's failure class, using the harness's own tool schema as the mutation grammar
     (k=5 nearest by signature embedding; mutation = swap the failing call for the alternative
     call that the harness exposes but the trace did not use)
     add tau_new to T with weight w(tau_new)=mean(w of its k neighbors)

5. Return re-weighted pool {(tau, w(tau))} union synthesized supplement.
```

**Why this design.** We chose a density-ratio re-weighting over retraining a classifier to predict failure modes because the ratio is a closed-form functional of the trace counts and does not require labels for 'which failure mode matters', accepting the cost that signatures must be discretized into a finite vocabulary V and rare modes can be under-estimated. We chose to synthesize tasks by mutating the harness's own tool schema rather than by LLM-generated free-form tasks because mutations stay inside the harness's action space (so the synthesized task is reachable under h_t), accepting the cost that synthesis cannot discover failure modes outside the current tool vocabulary. We chose a per-signature ratio rather than a per-task ratio because the motivation is about distribution shift over failure modes, not over individual tasks, accepting the cost that two tasks with the same signature receive the same weight even if their difficulty differs. We cap the synthesized supplement at M=200 to bound the cost of the next harness-search round, accepting that extremely rare signatures may remain uncovered for one round. This differs from Self-HPO, which carries forward hard tasks from the previous harness's training set without re-deriving them; a domain expert would not describe HATRIC as a variant of Self-HPO because HATRIC never introspects on failures — it estimates a density ratio over trace signatures and mutates the pool, leaving harness proposal to the existing loop.

**Why it measures what we claim.** The per-signature density ratio r(v) measures the harness-induced shift in failure-mode prevalence because the assumption that the trace signature is a sufficient statistic for the failure mode makes p_post(v)/p_pre(v) equal to the likelihood ratio of the failure mode under the two harnesses; this assumption fails when two distinct failure modes collapse to the same signature (e.g., a timeout and a wrong-API-call both produce an empty test diff), in which case r(v) reflects the aggregate shift of those modes rather than any one mode. The per-task weight w(tau) measures a task's coverage contribution to the post-update failure distribution because the assumption that tasks are exchangeable within a signature makes the re-weighted pool an unbiased estimate of the post-harness failure distribution; this assumption fails when a signature contains tasks of very different difficulty, in which case w(tau) over-weights easy tasks that happen to share the signature. The synthesized supplement measures coverage of under-covered signatures because the assumption that the harness's unused tool calls are the interventions that would exercise the missing mode makes the mutation a valid intervention; this assumption fails when the missing mode requires a tool the harness does not expose, in which case the supplement reflects only within-harness coverage. The collision diagnostic c(v) measures the degree to which assumption A (signature sufficiency) holds for cell v because c(v) is the fraction of tasks in v whose failure class differs from the majority class; this diagnostic fails to detect collapse when the colliding modes have identical failure class labels but different root causes (e.g., two distinct timeout causes both labeled 'timeout'), in which case c(v) underestimates the true collision rate and the secondary discriminator D may not separate them.

## Contribution

['A closed-loop task re-weighting layer (HATRIC) that estimates the density ratio between post- and pre-harness-update trace-signature distributions and converts it into per-task importance weights plus a schema-bounded synthesized supplement, providing the first mechanism that closes the task pool over harness-induced failure shift.', "A design principle: harness-induced failure shift is measurable as a density ratio over execution-trace signatures, and coverage closure is achieved by re-weighting rather than by introspective failure analysis, which decouples task selection from the harness proposer's self-knowledge.", 'An open trace-signature taxonomy and mutation grammar derived from harness tool schemas, enabling reproduction of the re-weighting layer on top of any harness-optimization loop that emits execution traces.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-round coding task pool with execution traces (e.g., SWE-rebench, 500 tasks, 5 harness-update rounds) | Provides per-task failure signatures across harness updates |
| Primary metric | Post-update pass rate on held-out failure-mode subsets (defined by the 12-class taxonomy, evaluated on 100 held-out tasks per class) | Directly tests coverage of shifted failure distribution |
| Baseline | Self-HPO (harness-policy alternating loop) | Tests whether trace-signature reweighting beats hard-task carryforward |
| Baseline | Harness-only optimization | Tests whether policy-side reweighting adds value beyond harness search |
| Baseline | Policy-only optimization | Tests whether harness-induced shift matters at all |
| Baseline | Classifier-based failure-mode predictor retrained after each harness update and used to resample tasks (reviewer-requested) | Tests whether closed-form density ratio differs from a learned resampler, especially when p_pre(v)=0 |
| Ablation-of-ours | HATRIC without synthesized supplement | Isolates contribution of mutation-based coverage |
| Ablation-of-ours | HATRIC without collision diagnostic (fixed phi, no secondary discriminator) | Isolates effect of assumption-A test and fallback |

### Why this setup validates the claim

The central claim is that harness updates shift the failure-mode distribution, and that reweighting tasks by a trace-signature density ratio plus targeted mutation covers that shift better than carrying forward hard tasks. Self-HPO is the right primary baseline because it embodies the exact mechanism gap we target: it reuses hard tasks without re-deriving them from the post-update trace distribution, so if HATRIC's gain comes from density-ratio reweighting, the gap should appear specifically on subsets whose signature prevalence shifted most between h_{t-1} and h_t. The harness-only and policy-only baselines bracket the two components HATRIC feeds, testing whether the reweighted mixture matters on both sides rather than only one; if HATRIC only beats one of them, the claim about feeding both search and training is unsupported. The classifier-based baseline directly tests whether the closed-form ratio is merely a relabeling of standard importance weighting: we include a regime where p_pre(v)=0 for an emerging signature, where the ratio is undefined and our fallback assigns uniform weight plus mandatory synthesis, while the classifier still assigns mass; if HATRIC matches or beats the classifier there, the mechanism is not a relabeling. The no-synthesis ablation isolates whether mutation-based supplement adds coverage beyond reweighting alone, which is the second sub-claim. The no-collision-diagnostic ablation tests whether the assumption-A test and fallback matter: if removing it does not change results, the collision rate is low in this pool and the diagnostic is unnecessary. Post-update pass rate on held-out failure-mode subsets is the right metric because it detects the predicted effect directly: a uniform gain would suggest generic improvement, whereas a gap concentrated on shifted signatures is what the density-ratio mechanism predicts. We also report the pool-level collision rate C (mean c(v) over V) and the fraction of signatures with c(v) > 0.10, to quantify how often assumption A fails in this pool.

### Expected outcome and causal chain

**vs. Self-HPO** — On a case where the new harness fixes the dominant API-misuse failures but introduces a rarer tool-sequencing failure mode, Self-HPO carries forward the previously hard API-misuse tasks, so the policy over-trains on a now-saturated mode and the sequencing mode stays under-sampled. Our method computes r(v) from pre/post trace signatures, down-weights the collapsed API-misuse signature, and up-weights the emerging sequencing signature, so we expect a noticeable gap (e.g., +8 to +12 percentage points) on the shifted-signature subset but near-parity (within 2 points) on the stable subset.

**vs. Harness-only optimization** — On a case where the harness search finds a configuration that is good on average but the policy is still trained on the stale pre-update mixture, harness-only optimization selects against a task distribution that no longer matches post-update failures, so it plateaus on the emerging mode. Our method feeds the reweighted pool into policy training as well, so the policy sees the shifted signature before the next harness round, and we expect the harness-only baseline to lag specifically in rounds immediately following a large harness change (e.g., a 5-point gap in the round after a harness update that changes the tool interface).

**vs. Policy-only optimization** — On a case where the harness exposes an unused alternative tool call that would resolve a failure class, policy-only optimization cannot discover this because the action space is fixed and the failure mode never appears in its training mixture. Our method's mutation swaps the failing call for the harness-exposed alternative, surfacing the mode as a reachable task, so we expect the supplement to yield gains on signatures whose resolution requires the unused tool (e.g., +6 points), with parity where no such tool exists.

**vs. Classifier-based failure-mode predictor** — On the regime where an emerging signature has p_pre(v)=0, the density ratio is undefined and our fallback assigns uniform weight plus mandatory synthesis, while the classifier (trained on pre-update traces) assigns near-zero probability to the new signature and thus resamples it rarely. We expect HATRIC to outperform the classifier on the emerging-signature subset (e.g., +5 points) but to be comparable on signatures with p_pre(v)>0, where both methods assign similar weights. If HATRIC does not beat the classifier on the p_pre(v)=0 subset, the closed-form ratio is not adding value beyond a learned resampler.

### What would falsify this idea

If HATRIC's gain over Self-HPO is uniform across signature subsets rather than concentrated on subsets whose pre/post prevalence shifted most, the density-ratio mechanism is not what produces the improvement. If removing the synthesized supplement leaves the gain unchanged, the mutation-based coverage sub-claim is wrong. If the pool-level collision rate C is high (e.g., >0.20) and the collision diagnostic does not improve results over the no-diagnostic ablation, then assumption A fails in this pool and the density-ratio mechanism is invalid.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
