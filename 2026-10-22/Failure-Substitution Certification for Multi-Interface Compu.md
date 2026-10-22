# Failure-Substitution Certification for Multi-Interface Computer-Use Agent Environments

## Motivation

WildAWM's execution-based quality gates verify that a synthesized task is solvable and that its verifier is judge-consistent, but they never establish that the environment's action set is closed under failure substitution: a task may pass the gate while its GUI path has no equivalent MCP or CLI action that reaches the same final state, silently assuming cross-interface functional equivalence instead of measuring it. This is a meta-level structural gap — the same unmeasured equivalence assumption recurs across synthesized-environment pipelines (WildAWM's multi-interface tasks, R2E-Gym's procedurally curated executables, and CodeGym's extracted tool environments), where solvability is filtered but substitutability is not. The result is environments whose training signal rewards interface-specific memorization rather than the interface-agnostic goal achievement that general computer-use agents require.

## Key Insight

For a synthesized task to certify cross-interface substitutability, every forced GUI-action failure must admit an MCP or CLI action whose execution reaches a final environment state that the execution verifier judges equivalent to the GUI-success reference state — a property that is measured by differential state comparison, not assumed from task solvability.

## Method

```pseudocode
Input: synthesized task T with environment E (GUI, MCP, CLI surfaces), execution verifier V, GUI reference trajectory tau_GUI that V judges successful
Output: certification label cert(T) in {CERTIFIED, GUI_IRREPLACEABLE, NO_EQUIVALENT, VERIFIER_AMBIGUOUS, GUI_RECOVERABLE}

# Load-bearing assumption (stated explicitly):
# Forcing a GUI action a_i to return an error with no state change is a valid, isolating
# intervention on a_i's causal contribution, so that V(s_fail_i) != PASS reliably identifies
# a_i as load-bearing and the subsequent single-action substitution test
# (tau_GUI[0:i] + [c] + tau_GUI[i+1:]) is a well-posed test of cross-interface substitutability.
# This assumption is verified per task by the re-planning-aware intervention in step 2:
# after forcing a_i to fail, the GUI driver continues to completion under its normal policy
# (re-observation after each action), and the terminal state is compared to s_ref. If the
# driver recovers via a different GUI path and V(s_terminal) = PASS, a_i is labeled
# GUI_RECOVERABLE (not NON-CRITICAL) and excluded from substitution search; only actions
# whose failure leaves V(s_terminal) != PASS are CRITICAL. For substitution, candidate c is
# executed by the same driver loop with re-observation, so the test measures reachability of
# s_ref under the substituted action set rather than assuming the spliced trajectory is executable.

1. Reference state capture:
   s_ref <- execute(tau_GUI) in E
   assert V(s_ref, T) = PASS   # only certify tasks whose GUI path already passes

2. Failure-injection variant generation (re-planning-aware):
   for each GUI action a_i in tau_GUI at step i:
       tau_fail_i <- inject_failure(tau_GUI, i)   # force a_i to return error, no state change
       s_terminal_i <- execute_with_replanning(tau_fail_i) in E   # driver continues to completion under normal policy, re-observing after each action
       if V(s_terminal_i, T) = PASS:
           if s_terminal_i != s_ref:   # driver recovered via a different GUI path
               mark a_i as GUI_RECOVERABLE; continue
           else:
               mark a_i as NON-CRITICAL; continue   # action was not load-bearing
       else:
           mark a_i as CRITICAL and retain (T, a_i, s_ref, s_terminal_i)

3. Cross-interface substitution search (per critical a_i):
   candidates <- enumerate_actions(E.MCP) union enumerate_actions(E.CLI)   # bounded by action schema, not free-form generation
   for each candidate c in candidates:
       tau_sub <- tau_GUI[0:i] + [c] + tau_GUI[i+1:]
       s_sub <- execute_with_replanning(tau_sub) in E   # same driver loop with re-observation
       if V(s_sub, T) = PASS and state_equiv(s_sub, s_ref):
           record substitute (a_i -> c); break
   if no substitute found:
       label T as GUI_IRREPLACEABLE if E.MCP and E.CLI lack any action touching a_i's target resource
       else label T as NO_EQUIVALENT

4. Verifier ambiguity check:
   if state_equiv(s_sub, s_ref) but V(s_sub,T) != V(s_ref,T):
       label T as VERIFIER_AMBIGUOUS   # verifier is not a valid oracle for this substitution

5. Certification aggregation:
   cert(T) <- CERTIFIED iff every critical a_i has a substitute c with V-pass and state_equiv
```

**Why this design.** We chose differential failure injection over solvability filtering because solvability only tests whether *some* trajectory passes V, whereas substitutability requires that the *same final state* be reachable after removing a specific GUI action — a strictly stronger condition that WildAWM's gate never checks, accepting the cost that certification is O(|tau_GUI| * |candidates|) verifier calls per task. We chose bounded schema enumeration of MCP/CLI actions over free-form LLM-generated substitutes because generation would confound substitution failure with generation failure, accepting the cost that we may miss exotic substitutes outside the schema. We chose state_equiv as an explicit differential comparison of the environment's residual state (filesystem, database, service config) in addition to V, because V may be task-outcome-oriented and blind to side effects, accepting the cost that state_equiv requires environment instrumentation. We chose to retain only critical actions (those whose failure changes V's verdict) rather than all actions, because non-critical actions are not load-bearing and certifying them would inflate the certified fraction; this is the same load-bearing distinction that separates our certification from a naive replay of R2E-Gym-style execution checks, and a domain expert would not describe it as a variant of R2E-Gym's hybrid verifier because R2E-Gym never intervenes on the action set to test substitutability.

**Why this measures what we claim.** The quantity `V(s_sub, T) = PASS and state_equiv(s_sub, s_ref)` measures *cross-interface substitutability* because the named assumption is that the execution verifier V is a sound oracle for task outcome and that state_equiv captures all environment state relevant to the task; this assumption fails when V is under-specified (e.g., checks only a log line) or state_equiv omits a resource the task depends on, in which case the metric reflects verifier-permitted state coincidence rather than true functional equivalence — which is exactly why step 4 explicitly flags VERIFIER_AMBIGUOUS when V disagrees with state_equiv. The quantity `a_i is CRITICAL` measures *load-bearing action identity* because the named assumption is that forcing a_i to fail with no state change is a valid intervention isolating a_i's causal contribution; this assumption fails when a_i's failure triggers a compensating retry inside the GUI driver, in which case the metric reflects driver resilience rather than action necessity — which is why step 2 now uses re-planning-aware execution and labels such cases GUI_RECOVERABLE rather than NON-CRITICAL. The quantity `cert(T) = CERTIFIED` measures *action-set closure under failure substitution* because the named assumption is that the enumerated MCP/CLI candidate set is complete for the resources a_i touches; this assumption fails when the environment exposes a resource only through GUI, in which case the metric reflects schema coverage rather than true irreplaceability — the GUI_IRREPLACEABLE label is defined precisely to separate this case from NO_EQUIVALENT.

## Contribution

['A failure-substitution certification layer for multi-interface environment synthesis that, for each synthesized task, generates forced-action-failure variants and searches for MCP/CLI substitutes whose executed final state the verifier and a state-equivalence oracle both judge equivalent to the GUI reference state.', 'A four-way certification taxonomy (CERTIFIED, GUI_IRREPLACEABLE, NO_EQUIVALENT, VERIFIER_AMBIGUOUS) that separates genuine cross-interface substitutability from schema-coverage gaps and verifier under-specification, enabling synthesis pipelines to filter tasks by measured substitutability rather than assumed solvability.', 'An analysis of the certified fraction of WildAWM-synthesized tasks and the per-interface substitutability rate, exposing which task families are interface-specific and which are interface-agnostic.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | WildAWM synthesized GUI tasks with MCP/CLI surfaces | Tasks already carry GUI reference trajectories and verifiers |
| Primary metric | Fraction of tasks labeled CERTIFIED vs GUI_IRREPLACEABLE vs NO_EQUIVALENT vs GUI_RECOVERABLE | Directly exposes the substitutability partition, not just pass rate |
| Baseline | Solvability-only gate (WildAWM-style) | Tests whether passing V implies substitutability |
| Baseline | Free-form LLM substitute generation | Tests whether bounded enumeration is necessary |
| Baseline | All-action certification (no criticality filter) | Tests whether load-bearing filtering matters |
| Ablation-of-ours | Drop state_equiv, certify on V alone | Isolates the verifier-ambiguity contribution |
| Ablation-of-ours | Single-action failure injection (no re-planning) | Isolates the contribution of re-planning-aware intervention |

### Why this setup validates the claim

The central claim is that certification requires differential failure injection plus bounded cross-interface search, and that a solvability gate cannot recover this partition. The solvability baseline tests the first sub-claim: if V-pass implied substitutability, its CERTIFIED fraction would match ours; any divergence on tasks where critical actions lack substitutes is direct evidence the gate is insufficient. The free-form generation baseline tests the second sub-claim: if LLM-generated substitutes matched bounded enumeration, our schema restriction would be unnecessary; instead generation failures should appear as spurious NO_EQUIVALENT labels on tasks where a schema action exists. The all-action baseline tests the criticality filter: certifying non-load-bearing actions should inflate CERTIFIED on tasks whose only substitutable actions are incidental, and the gap between it and ours localizes exactly that inflation. The V-only ablation isolates state_equiv: any task that flips from CERTIFIED to VERIFIER_AMBIGUOUS or NO_EQUIVALENT when state_equiv is removed is precisely where V is an unsound oracle. The single-action ablation isolates the re-planning-aware intervention: if the load-bearing assumption were valid, the single-action and re-planning-aware variants would produce identical labels; any divergence (e.g., tasks labeled NON-CRITICAL by single-action but GUI_RECOVERABLE by re-planning-aware) directly measures the confound between action necessity and driver resilience. The partition metric, rather than raw pass rate, is the right detector because the claim is about the *structure* of substitutability, not aggregate success.

### Expected outcome and causal chain

**vs. Solvability-only gate (WildAWM-style)** — On a case where the GUI trajectory passes V only because a GUI-only settings panel writes a config file that MCP and CLI cannot touch, the baseline labels the task CERTIFIED because V passes, conflating solvability with substitutability. Our method instead injects failure at that panel action, observes V flip to FAIL, finds no MCP/CLI action touching the config resource, and emits GUI_IRREPLACEABLE. We expect a noticeable gap concentrated on tasks whose reference trajectory includes GUI-only resource writes, with near-parity on tasks whose actions all have MCP/CLI counterparts.

**vs. Free-form LLM substitute generation** — On a case where the correct substitute is an obscure CLI flag combination outside the LLM's priors, the baseline generates a plausible but non-executing command, V fails, and the task is mislabeled NO_EQUIVALENT despite a schema action existing. Our method instead enumerates the bounded MCP/CLI schema, finds the flag-bearing action, executes it, and certifies on V-pass plus state_equiv. We expect the baseline to show excess NO_EQUIVALENT labels on tasks with deep or non-idiomatic CLI surfaces, while our labels track the actual schema coverage.

**vs. All-action certification (no criticality filter)** — On a case where the trajectory includes a cosmetic GUI click whose failure does not change V, the baseline treats it as substitutable-or-not and inflates the certified fraction when a trivial MCP no-op matches. Our method instead marks it NON-CRITICAL via failure injection and excludes it from aggregation, so certification reflects only load-bearing actions. We expect the baseline's CERTIFIED fraction to exceed ours by a margin concentrated on trajectories with many incidental actions, with parity on minimal trajectories.

**vs. Single-action failure injection (no re-planning)** — On a case where forcing a GUI action to fail triggers a compensating retry inside the GUI driver that reaches s_ref via a different GUI path, the single-action baseline observes V(s_fail_i) = PASS and labels the action NON-CRITICAL, thereby excluding it from substitution search and potentially certifying the task even though the action is load-bearing. Our re-planning-aware method instead detects that the terminal state differs from s_ref (or that V passes only after recovery) and labels the action GUI_RECOVERABLE, excluding it from certification aggregation. We expect the single-action baseline to show a higher CERTIFIED fraction on tasks with driver-resilient actions, with parity on tasks whose actions have no recovery path.

### What would falsify this idea

If our CERTIFIED fraction matches the solvability gate uniformly across all task subsets rather than diverging specifically on tasks with GUI-only resource writes, the claim that substitutability is strictly stronger than solvability is wrong. If removing state_equiv changes no labels, the verifier-ambiguity mechanism is inert and the differential-state component is unjustified. If the single-action and re-planning-aware variants produce identical labels across all tasks, the load-bearing assumption is empirically valid and the re-planning-aware intervention is unnecessary.

## References

1. Agent World Model in the Wild: Computer-Use Agent Environment Synthesis
2. R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents
3. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
4. Training Software Engineering Agents and Verifiers with SWE-Gym
5. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
6. FireAct: Toward Language Agent Fine-tuning
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
