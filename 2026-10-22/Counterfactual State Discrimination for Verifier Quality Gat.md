# Counterfactual State Discrimination for Verifier Quality Gates in Synthesized Computer-Use Environments

## Motivation

Environment-synthesis pipelines such as WildAWM gate task quality on execution success plus cross-judge agreement, but this proxy never establishes that success is causally tied to correct task completion: a verifier can pass every rollout and still be blind to off-by-one state changes or wrong operation orderings. R2E-Gym independently reports that execution-based verifiers suffer from low distinguishability, and SWE-Gym's unit-test verifiers inherit the same weakness because tests were not constructed to separate near-miss solutions from correct ones. The recurring structural failure is that verifier quality is measured by agreement with other judges over the same successful trajectories rather than by the verifier's ability to reject causally adjacent wrong outcomes.

## Key Insight

A verifier's discriminative power over the environment's state space is measurable by constructing paired traces that differ only in a causally necessary precondition and checking whether the verifier's final-state score separates them, which converts verifier quality from a judge-agreement question into a local sensitivity test on the state transition function.

## Method

# CUA-VerifyGate: counterfactual discrimination gate for synthesized tasks
Input: task t with environment E (GUI/MCP/CLI), reference solution trace tau*, execution verifier V, replay count R=5, noise epsilon=0.02, margin threshold delta=0.1
Output: accept / reject / repair signal for t

0. DETERMINISM AUDIT (explicit load-bearing assumption)
   The separation margin is attributable to the mutated precondition ONLY IF E is deterministic and fully observable. We do not assume this; we measure it.
   For the reference trace tau*, replay R=5 times in E and record final-state scores {score_V(F_r(tau*))}_{r=1..R}.
   Compute sigma_ref = max_r |score_V(F_r(tau*)) - mean_r score_V(F_r(tau*))|.
   If sigma_ref > epsilon, flag task as NOISY and route to REPAIR (cannot attribute margins); log sigma_ref.

1. TRACE DECOMPOSITION
   tau* = [(a_1, s_1), ..., (a_n, s_n)]   # actions and resulting states
   For each step i, record the precondition predicate pre_i(s_{i-1}) that
   a_i requires (e.g., field selected, file exists, cursor at position k).

2. NEAR-MISS MUTATION GENERATION (LLM mutator, open-source model)
   For each step i with a causally necessary precondition, generate K=4 mutants:
     M1: off-by-one state change     (s_i' = s_i with index/offset shifted by 1)
     M2: operation reorder           (swap a_i, a_{i+1} when order matters)
     M3: precondition skip           (execute a_i without pre_i satisfied)
     M4: boundary perturbation       (s_i' at the nearest valid/invalid boundary)
   Execute each mutant in E to obtain final state F(tau_m).
   STATE-DIFF CHECK: compute the state diff D_m = F(tau_m) \ F(tau*) and require that D_m is exactly the intended predicate violation (no extra fields changed). Discard mutants whose D_m contains unintended differences or whose final state equals F(tau*) (non-discriminative by construction).

3. SEPARATION MARGIN COMPUTATION (paired, repeated-execution estimator)
   For each surviving mutant m:
     Replay tau* and tau_m R=5 times each in E, recording scores {score_V(F_r(tau*))} and {score_V(F_r(tau_m))}.
     Compute paired mean difference: margin_m = mean_r score_V(F_r(tau*)) - mean_r score_V(F_r(tau_m)).
     Compute 95% CI of the paired difference (bootstrap over R pairs, 1000 resamples).
     Discard m if the CI contains 0 (noise-driven difference).
   Aggregate over surviving mutants:
     M_sep(t) = min_m margin_m   # worst-case separation
     M_cons(t) = fraction of mutants with margin_m > delta (=0.1)

4. GATE DECISION
   if M_sep(t) >= delta and M_cons(t) >= 0.75: accept t
   elif M_sep(t) < 0: reject t (verifier passes a wrong outcome)
   else: route t to REPAIR: regenerate verifier assertions targeting the
         failing mutant's state predicate, then re-run steps 0-4 once.

5. LOG per-task (sigma_ref, M_sep, M_cons, mutant-type breakdown, fraction of mutants discarded by state-diff/CI) into the WildAWM quality-gate record alongside existing rollout and cross-judge signals.

**Why this design.** We chose constructed counterfactual mutants over additional agent rollouts because rollouts sample the same success distribution the verifier was already fitted to, so they cannot probe the boundary; the cost is that mutants must be executable, which we accept by restricting mutation to state-level edits that the environment's own transition function can replay. We chose worst-case separation (min over mutants) over mean margin because a verifier that passes 90% of near-misses still admits a systematic failure mode, and the cost is noisier rejection of tasks with one hard-to-construct mutant, which the repair loop absorbs. We chose a single open-source LLM mutator with fixed mutation templates over free-form mutation generation because template-constrained mutants stay causally adjacent to the reference trace (they differ in exactly one precondition), accepting lower mutant diversity in exchange for a clean causal interpretation. A domain expert would not describe this as a variant of SWE-Gym's unit-test filtering or APIGen's execution checking: those check that the reference solution passes, whereas CUA-VerifyGate checks that wrong solutions fail, which is the complementary and previously unmeasured property. The determinism audit (step 0) and paired repeated-execution estimator (step 3) make the causal-attribution assumption explicit and measurable: if the environment is non-deterministic, sigma_ref exceeds epsilon and the task is flagged rather than silently mis-scored.

**Why it measures what we claim.** The separation margin `mean_r score_V(F_r(tau*)) - mean_r score_V(F_r(tau_m))` measures verifier discriminative power because the mutant construction guarantees tau_m differs from tau* only in a causally necessary precondition, so any score gap is attributable to that precondition; this assumption fails when the environment's transition function is non-deterministic or has hidden state, in which case the margin reflects execution noise rather than causal sensitivity — which is why step 0 measures sigma_ref and step 3 uses a paired CI to discard noise-driven margins. The mutant-type breakdown (`M1` off-by-one, `M2` reorder, `M3` precondition skip) measures causal necessity of each precondition class because each template targets exactly one precondition type; this assumption fails when a task has redundant preconditions, in which case a skipped precondition may be compensated elsewhere and the mutant's final state matches tau*, which is why step 2 discards equal-final-state mutants and applies the state-diff check. `M_cons(t)` measures the robustness of the verifier's decision boundary because it counts the fraction of causally adjacent wrong outcomes that are rejected by more than delta; this assumption fails when the verifier's score is bimodal, in which case M_cons can be high while M_sep is negative for a small subset, which is why the gate requires both.

## Contribution

(1) CUA-VerifyGate, a counterfactual state-discrimination gate that augments execution-based task synthesis with constructed near-miss traces and a worst-case separation margin, providing a verifier-quality signal that existing rollout-plus-judge-agreement gates cannot produce. (2) A mutation taxonomy (off-by-one, reorder, precondition-skip, boundary) for computer-use environments that yields executable near-miss traces differing from the reference in exactly one causally necessary precondition, enabling attribution of verifier failures to specific precondition classes. (3) An empirical design principle: verifier acceptance should be conditioned on rejection of causally adjacent wrong outcomes, not only on agreement over successful trajectories, with a repair loop that regenerates verifier assertions targeting failing mutant predicates.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WildAWM synthesized CUA tasks (GUI/MCP/CLI), scoped first to CLI/MCP with explicit state | Native home of the quality gate; tractable state injection |
| Primary metric | Verifier false-accept rate on near-miss mutants | Directly measures the complementary property claimed |
| Secondary metric | Fraction of tasks flagged NOISY by determinism audit (sigma_ref > epsilon=0.02) | Quantifies how often causal-attribution assumption breaks |
| Baseline | Rollout-based execution gate (WildAWM default) | Tests whether rollouts probe the decision boundary |
| Baseline | Cross-judge consistency gate | Tests whether judge agreement substitutes for discrimination |
| Baseline | APIGen/SWE-Gym pass-only execution check | Tests whether reference-pass checking is sufficient |
| Ablation-of-ours | Mean-margin gate (drop worst-case min) | Isolates the worst-case aggregation sub-claim |
| Ablation-of-ours | Single-run margin (no paired R=5 estimator, no CI filter) | Isolates the noise-robustness sub-claim |
| Held-out mutant class | M5: multi-step interaction error (e.g., correct actions in wrong order across >2 steps) not used in gating | Estimates template-cover assumption A |

### Why this setup validates the claim

The central claim is that verifier quality in synthesized CUA tasks is limited by a property no existing gate measures — whether wrong solutions are rejected — and that constructed counterfactual mutants expose this where rollouts and judges cannot. The rollout-based baseline directly tests the sub-claim that sampling the success distribution cannot probe the boundary: if it matched our false-accept detection, the mutant machinery would be unnecessary. The cross-judge baseline tests whether an independent LLM judge already supplies discrimination, which would make our gate redundant. The pass-only execution baseline (APIGen/SWE-Gym style) tests whether checking that the reference solution passes is enough — the exact assumption we claim is insufficient. The mean-margin ablation isolates whether worst-case aggregation matters or whether any mutant signal suffices. The single-run ablation isolates whether the paired repeated-execution estimator and CI filter are necessary to avoid noise-driven rejections. The determinism audit reports the fraction of tasks where sigma_ref > epsilon, directly bounding the load-bearing assumption. The held-out M5 mutant class estimates whether the four gating templates cover the causally adjacent wrong-outcome space: if M5 false-accept rate is high on tasks accepted by our gate, the template-cover assumption A fails. The primary metric — false-accept rate on near-miss mutants — is the right detector because it is defined on the wrong-outcome side of the boundary, the region all baselines structurally avoid; a gain concentrated there, with parity on reference-pass checks, is precisely the signature the causal story predicts.

### Expected outcome and causal chain

**vs. Rollout-based execution gate (WildAWM default)** — On a task where the verifier checks only a final file's existence but not its contents, rollouts sample the same successful distribution the verifier was fitted to, so every rollout that reaches the goal state passes and the gate accepts the task. Our method instead constructs a precondition-skip mutant that reaches the same goal state with wrong contents, and the margin exposes the verifier's blindness. We expect a large false-accept gap on content-sensitive tasks and near-parity on tasks whose verifiers already assert terminal state, with the gap concentrated exactly where rollouts are most confident.

**vs. Cross-judge consistency gate** — On a task with a subtle off-by-one in an index or offset, two judges agree the reference trace is correct because both read the same surface-level success signal, so consistency is high while the verifier still passes a shifted-index mutant. Our method generates the off-by-one mutant directly and measures separation against it, so the failure surfaces even when judges concur. We expect judge-consistency to be near-ceiling on accepted tasks while our false-accept rate on the off-by-one mutant class stays low, producing a visible divergence on the M1 mutant breakdown specifically.

**vs. APIGen/SWE-Gym pass-only execution check** — On a task where the reference solution passes but the verifier's assertions are too permissive (e.g. checking one artifact among several), the pass-only gate accepts because the reference passes, exactly as designed. Our method instead asks whether wrong solutions fail, and the reorder mutant (M2) reaches a partially-correct state the permissive verifier accepts, driving a negative margin and triggering repair. We expect the pass-only baseline to accept nearly all tasks while our gate rejects a substantial fraction, with the rejected set enriched for multi-artifact tasks and the repair loop recovering most of them.

**vs. Mean-margin gate (ablation)** — On a task with three easy mutants and one hard boundary mutant, the mean margin is comfortably positive because the easy mutants dominate, so the mean-margin ablation accepts a task whose worst-case mutant slips through. Our method's min-over-mutants aggregation flags that single failure and routes to repair. We expect the ablation to show a higher false-accept rate than full CUA-VerifyGate, concentrated on tasks with heterogeneous mutant difficulty — the signature that worst-case aggregation, not mutant generation, carries the gain.

**vs. Single-run margin (ablation)** — On a non-deterministic CLI/MCP task where replaying tau* yields score variance sigma_ref > epsilon, the single-run margin conflates execution noise with causal sensitivity, producing spurious rejections. Our paired R=5 estimator with CI filter discards noise-driven margins, so the gate decision is stable. We expect the single-run ablation to show higher false-reject rate on tasks with sigma_ref near epsilon, and the determinism audit to flag those tasks explicitly.

**vs. Held-out M5 mutant class** — On a task accepted by our gate, an M5 multi-step interaction error (correct actions in wrong order across >2 steps) may still be passed by the verifier if the four gating templates do not cover that failure mode. We expect the M5 false-accept rate to be lower than the rollout baseline's but nonzero, quantifying the template-cover assumption A; a high M5 false-accept rate on accepted tasks would indicate the gate certifies verifiers blind to unmutated failure modes.

### What would falsify this idea

If our false-accept reduction were uniform across mutant classes and task types rather than concentrated on the M1/M2/M3 classes and on tasks where rollouts or judges are most confident, the claim that mutants probe a boundary rollouts cannot reach would be wrong. Equally fatal: if the mean-margin ablation matched full CUA-VerifyGate on false-accept rate, the worst-case aggregation sub-claim would collapse; if the single-run ablation matched the paired estimator, the noise-robustness sub-claim would collapse; and if the held-out M5 false-accept rate on accepted tasks were as high as the rollout baseline's, the template-cover assumption A would fail.

## References

1. Agent World Model in the Wild: Computer-Use Agent Environment Synthesis
2. R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents
3. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
4. Training Software Engineering Agents and Verifiers with SWE-Gym
5. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
6. FireAct: Toward Language Agent Fine-tuning
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
