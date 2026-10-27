# Trace-Anchored Differential Probing for Precondition-Level Skill Invalidation

## Motivation

Incremental skill maintenance systems such as Steward (in Δτ-Bench) accumulate skills across successive capability-preserving ecosystem states, but they have no principled invalidation trigger: a transformation that preserves a tool's task-level capability can silently falsify a procedural precondition inside a stored skill (e.g., an argument schema change), and neither end-to-end success per state nor context-matching retrieval can localize which step broke, because the skill's natural-language description still matches. The field's convergent trajectory toward continuously maintained skill stores therefore hits a new bottleneck — the store accumulates stale skills that pass retrieval but fail execution — and this bottleneck is structural, not a tuning issue, since the only signal distinguishing 'retained' from 'silently broken' is a differential test over the skill's own trace.

## Key Insight

A stored skill's execution precondition set is a finite, probeable object: each precondition is witnessed by a specific tool-call in the skill's own trace, so a transformation's effect on preconditions can be isolated by re-executing only the minimal counterfactual suffixes of those witness calls rather than re-running full tasks.

## Method

**What it is.** PRISM (Precondition-Rewrite via Isolated Step Mutation) takes as input a stored skill s = (trace τ_s of tool-call steps, natural-language description d_s), an ecosystem transition Δ, and the new tool registry R'; it outputs either 'retained' or a rewritten skill s' with the broken step localized and patched. It never re-runs the full task; it probes only counterfactual suffixes anchored on τ_s.

**Load-bearing assumption (stated explicitly).** The suffix probe attributes a mismatch to step i only if the recorded downstream state s_i is reconstructible and the suffix's expected observations are deterministic given s_i. Because s_i is not a self-contained replayable object (it is the product of the full prefix execution, including tool outputs, hidden environment state, and agent context), we replace 'resume from recorded s_i' with a **prefix-replay probe**: re-execute τ_s[:i] in R' to reconstruct s_i, then execute the suffix. A mismatch is attributed to step i only if the prefix replay reproduces τ_s[:i] exactly; otherwise the step is marked 'unprobeable' and falls back to full re-execution. This makes the state-restoration dependency explicit and testable, at the cost of O(i+k) rather than O(k) calls.

**How it works.**

```pseudocode
INPUT: skill s=(τ_s, d_s), transition Δ, new registry R'
HYPER: MAX_PROBES=3 per step, ARG_TOL=exact, PREFIX_REPLAY=True

# Phase 1: extract witness steps
W = []
for step i in τ_s:
    if step_i calls tool t with args a:
        W.append((i, t, a, s_i))   # s_i = downstream state dependency

# Phase 2: per-step differential probe (prefix-replay)
broken = []
unprobeable = []
for (i, t, a, s_i) in W:
    if t not in R':                      # tool removed
        broken.append((i, 'missing', t)); continue
    schema = R'[t].arg_schema
    if not schema_accepts(schema, a):    # schema falsified
        broken.append((i, 'schema', t)); continue
    # prefix replay to reconstruct s_i
    if PREFIX_REPLAY:
        replay_obs = execute_prefix(τ_s[:i], registry=R')
        if replay_obs != τ_s[:i].expected_obs:
            unprobeable.append((i, 'prefix_mismatch', t)); continue
        s_i_reconstructed = replay_obs[-1].state
    else:
        s_i_reconstructed = s_i
    # semantic probe: minimal counterfactual suffix from step i
    suffix = τ_s[i:]
    obs = execute_suffix(suffix, registry=R', start_state=s_i_reconstructed)
    if obs != τ_s[i].expected_obs:       # behavior changed
        broken.append((i, 'behavior', t, obs))

# Phase 3: conditional rewrite
if broken == [] and unprobeable == []: return 'retained'
for (i, kind, t, *rest) in broken:
    if kind == 'schema':  s = repair_args(s, i, R'[t].arg_schema)
    if kind == 'behavior': s = repair_obs(s, i, rest[0])
    if kind == 'missing': s = substitute_tool(s, i, R', Δ)
for (i, kind, t) in unprobeable:
    s = full_reexecute_and_repair(s, i, R', Δ)   # fallback
return s'
```

**Why this design.** We chose a *trace-anchored differential probe* over end-to-end re-execution because the skill's own trace already contains the witness for every precondition, so probing a suffix of length k costs O(k) calls instead of O(task length), accepting the cost that a precondition not witnessed in the trace is invisible to us. We chose *per-step isolation* over a confidence gate (the log-prob routing of prior work) because a precondition's falsification is a discrete, observable event — `schema_accepts` returns a boolean and `obs != expected_obs` is a concrete mismatch — so no calibration story is needed, accepting the cost that we cannot detect a precondition whose failure is only visible after a full task. We chose *conditional rewrite* (only broken steps) over full skill regeneration because regeneration discards procedural knowledge that is still valid, accepting the cost that a broken step whose repair requires cross-step reasoning will be patched locally rather than holistically. A domain expert would not describe PRISM as a variant of Steward's incremental maintenance: Steward updates skills by re-running tasks and aggregating success, whereas PRISM never re-runs a task and instead intervenes on the trace itself — the unit of analysis is the step, not the task.

**Why it measures what we claim.** `schema_accepts(schema, a)` measures *precondition falsification* because the assumption that a tool's argument schema is a faithful encoding of its callable interface makes a schema rejection equivalent to a broken precondition; this assumption fails when a tool accepts the old schema but silently ignores a field, in which case the metric reflects only syntactic compatibility rather than semantic precondition. `execute_suffix(suffix, R', s_i_reconstructed) != τ_s[i].expected_obs` measures *behavioral drift localized to step i* because the assumption that the suffix's expected observations are deterministic given the downstream state s_i makes any mismatch attributable to step i rather than to a downstream step; this assumption fails when the tool is stochastic, in which case the metric reflects sampling noise and MAX_PROBES=3 majority-vote mitigates but does not eliminate it. The conjunction of the two probes measures *silent breakage* because the assumption that a capability-preserving transformation leaves the task-level goal reachable while altering at least one witness step makes a per-step mismatch a necessary and sufficient indicator of silent breakage; this assumption fails when a transformation preserves all witness steps but breaks an unwitnessed precondition, in which case the metric reflects 'retained' for a skill that will fail at execution — the boundary we state in limitation. The prefix-replay step adds a third assumption: that τ_s[:i] can be re-executed in R' to reproduce the recorded observations exactly; this assumption fails when the prefix itself is affected by Δ or when environment state is not replayable, in which case the step is marked 'unprobeable' and the metric reflects a fallback to full re-execution rather than a localized detection.

## Contribution

(1) PRISM, a differential-probe operator that localizes which step of a stored skill a capability-preserving ecosystem transformation has silently falsified, using only counterfactual suffixes of the skill's own trace rather than full task re-execution. (2) The design principle that a skill's execution precondition set is witnessable by its trace, which converts skill invalidation from a task-level success question into a per-step boolean test and eliminates the need for a calibrated confidence gate. (3) A rewrite procedure that patches only broken steps, preserving valid procedural knowledge across ecosystem states.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Δτ-Bench six-state ecosystem stream | Native capability-preserving transitions with validated multi-turn tasks |
| Primary metric | Silent-breakage detection rate at step granularity | Directly measures the localized-falsification claim |
| Baseline | Steward (incremental maintenance) | Tests task-aggregation against step-level intervention |
| Baseline | End-to-end skill re-execution | Tests cost/coverage tradeoff of full re-runs |
| Baseline | Full skill regeneration | Tests whether local patching preserves valid procedure |
| Ablation-of-ours | PRISM without differential probe (schema-only) | Isolates the behavioral-drift probe's contribution |
| Ablation-of-ours | PRISM without prefix-replay (resume from recorded s_i) | Isolates the state-restoration dependency and measures unprobeable rate |
| Diagnostic | Determinism check: re-run identical suffixes N=10 times on unchanged registry | Measures baseline mismatch rate (false-positive floor) for the behavioral probe |

### Why this setup validates the claim

Δτ-Bench is the right substrate because its transitions are explicitly capability-preserving, so any task-level failure after a transition is by construction a *silent* breakage — exactly the phenomenon PRISM claims to localize. The primary metric, step-granular silent-breakage detection, is chosen over task success because task success conflates detection with repair quality and with downstream stochasticity; step detection isolates the falsification mechanism the paper argues for. Steward tests the sub-claim that task-level success aggregation cannot see per-step precondition falsification, since it only observes task outcomes. End-to-end re-execution tests the cost/coverage sub-claim: it should be more thorough but far more expensive, so the comparison is on detection-per-probe rather than raw detection. Full regeneration tests the conditional-rewrite sub-claim: if regeneration matched PRISM on retained-skill preservation, local patching would be unmotivated. The schema-only ablation is the decisive control — if removing the behavioral probe does not degrade detection, then the core differential-probe mechanism is doing no work and the design collapses to schema checking. The prefix-replay ablation isolates the load-bearing state-restoration assumption: if PRISM without prefix-replay shows a high unprobeable rate or spurious detections, the replay step is necessary; if it matches full PRISM, the assumption is empirically benign on this stream. The determinism diagnostic measures the false-positive floor of the behavioral probe on an unchanged registry, so any detection advantage can be compared against the noise level rather than assumed.

### Expected outcome and causal chain

**vs. Steward** — On a case where a tool's argument schema is unchanged but its return semantics shift (e.g., a search tool now returns ranked IDs instead of documents), Steward re-runs the task, observes aggregate failure, and either discards or wholesale-replaces the skill because it has no step-level witness to localize the fault. Our method instead probes the suffix anchored at the affected step, sees `obs != expected_obs`, and patches only that step's observation handling. We expect a large gap in detection on behavior-drift transitions and near-parity on tool-removal transitions, where Steward's aggregate signal is also sufficient.

**vs. End-to-end re-execution** — On a case where the skill is long (many steps) and only one mid-trace step is broken, re-execution pays full task cost per transition and, if the broken step is early, cascades failures that obscure which step actually broke. Our method probes only the suffix from the suspected step, so cost scales with suffix length and the mismatch is attributable to step i by construction. We expect PRISM to match re-execution's detection rate at a small fraction of the probe budget, with the gap in cost-per-detection widening as task length grows.

**vs. Full skill regeneration** — On a case where a transition breaks one argument field but the rest of the procedure remains valid, regeneration discards correct procedural knowledge and produces a skill that may fail on unwitnessed-but-still-valid steps because the model re-derives them from description alone. Our method patches only the broken step and leaves the validated trace intact. We expect regeneration to show higher variance in retained-skill preservation and a detectable drop on the subset where the transition touches only one step, while PRISM holds retention steady.

**vs. PRISM without prefix-replay** — On a case where the recorded state s_i is not reconstructible (e.g., environment state mutated by prior calls), resuming from s_i produces a mismatch caused by missing prefix rather than by step i, leading to spurious detections or unprobeable steps. Our prefix-replay probe reconstructs s_i by re-executing τ_s[:i] in R' and attributes a mismatch only if the prefix replay reproduces τ_s[:i] exactly. We expect the no-replay variant to show a higher false-positive rate on behavior-drift transitions and a non-trivial unprobeable rate, while full PRISM maintains localization at the cost of O(i+k) calls.

### What would falsify this idea

If PRISM's detection advantage over the schema-only ablation is uniform across transition types rather than concentrated on behavior-drift transitions where the differential probe is the only signal, then the probe is not measuring what we claim. Equally, if PRISM matches re-execution on detection but at comparable probe cost, the trace-anchored cost argument is wrong and the method reduces to expensive re-execution. Additionally, if the determinism diagnostic shows a baseline mismatch rate comparable to the detection rate, the behavioral probe's signal is indistinguishable from noise.

## References

1. Δτ-Bench: Pushing Agents Toward a Moving Capability
2. Towards General Agentic Intelligence via Environment Scaling
3. T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step
4. AutoGuide: Automated Generation and Selection of Context-Aware Guidelines for Large Language Model Agents
5. ReAct: Synergizing Reasoning and Acting in Language Models
6. xLAM: A Family of Large Action Models to Empower AI Agent Systems
7. ExpeL: LLM Agents Are Experiential Learners
8. AgentTuning: Enabling Generalized Agent Abilities for LLMs
