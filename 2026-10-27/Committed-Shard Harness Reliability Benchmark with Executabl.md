# Committed-Shard Harness Reliability Benchmark with Executable Violation Oracles

## Motivation

Across agent-evaluation paper-trees, the evaluation unit is a fixed, expert-curated task suite with executable ground truth, and constraint checking is predefined allow/block policy enforcement — neither the contamination status of the suite nor a runtime-verifiable causal link between a constraint violation and a harness failure is established. TraceTrial-Bench (Trial by Trace) decomposes harness reliability into seven components and collects traces, but cannot prove its 95-task suite is uncontaminated nor that a predefined constraint violation is the cause of a harness failure; SWE-Bench+ shows that 32.67% of apparent successes are solution leakage and 31.08% are weak-test artifacts, confirming that fixed-suite contamination is structurally unprovable rather than a one-off defect. The root cause is that the evaluation unit is a (task, ground-truth) pair rather than a (task, constraint-oracle, harness-component) triple whose contamination status and violation-to-failure link are checkable invariants.

## Key Insight

If each task is sealed under a cryptographic Merkle commitment and each constraint oracle is an executable runtime monitor that emits a signed certificate over the trace, then contamination status becomes a proof-checkable predicate and violation-to-failure causality becomes a replay-testable invariant on the same trace — turning two assumptions into two verifiable equations without changing the trace-collection substrate.

## Method

**What it is.** HARNESS-SEAL is a benchmark harness that ingests an agent trace and outputs, per evaluation unit, a triple `(task_id, oracle_id, harness_component_id)` together with two verifiable artifacts: a **contamination proof** (a Merkle inclusion/non-inclusion path against a sealed shard root) and a **violation certificate** (a signed record binding an oracle firing to a specific harness component, validated by a counterfactual replay against a perturbed harness). Inputs: task suite, sealed commitment, harness configuration, trace. Outputs: `(unit, contamination_status, certificate, replay_verdict)`.

**How it works.**

```pseudocode
# Phase 0: one-time sealing (benchmark author)
for task in suite:
    leaf = H(canonical_serialize(task.prompt, task.ground_truth, task.oracle_spec))
    tree = MerkleTree(leaves)                    # binary, SHA-256, domain-separated leaf/node tags
public_root  = tree.root                       # published
sealed_root  = tree.root_of(sealed_shard)      # published, task bodies withheld
public_shard = {task_i : i in public_index}    # calibration shard, opened

# Phase 1: contamination proof (per evaluation run)
def contamination_proof(task_id):
    path = tree.inclusion_path(task_id)
    if task_id in public_index: return (OPEN, path)
    return (SEALED, path)                      # verifier recomputes root from path + leaf hash

# Phase 2: executable oracle monitors (per trace, per constraint c)
monitors = {c: RuntimeMonitor(c.predicate, c.scope, c.signing_key) for c in constraints}
for event in trace.events:                     # events carry (component_id, action, state_delta)
    for c, m in monitors.items():
        if m.scope.matches(event):
            v = m.check(event)                 # deterministic predicate over event + harness state
            if v.fired:
                cert = sign(m.key, {task_id, oracle_id: c, component_id: event.component_id,
                                    trace_hash: H(trace), event_idx, witness: v.witness})
                certificates.append(cert)

# Phase 3: causal verification by counterfactual replay
# Assumption (A): REPLACE_WITH_NOOP on exactly one component is a valid do-intervention
# with no compensating component masking the perturbation. Failure mode (F): a compensating
# component restores the outcome despite the no-op, yielding spurious NOT_CAUSAL or UNDETERMINED.
# We measure the masking rate and detect compensating components as specified below.
def replay_verdict(cert, trace, harness):
    toggled = harness.perturb(cert.component_id, mode=REPLACE_WITH_NOOP)  # same trace, one component perturbed
    replay  = toggled.replay(trace)            # deterministic replay of identical event stream
    oracle_replay = monitors[cert.oracle_id].check_all(replay)
    if oracle_replay.fired_at(cert.event_idx) and replay.outcome == trace.outcome:
        return NOT_CAUSAL                    # violation survives perturbation -> not harness-caused
    if not oracle_replay.fired_at(cert.event_idx) and replay.outcome != trace.outcome:
        return CAUSAL                        # violation disappears AND outcome changes
    return UNDERDETERMINED                   # one of the two conditions fails

# Phase 3b: compensating-component detector (runs only when replay_verdict == UNDERDETERMINED)
def detect_compensator(cert, trace, harness):
    # For each other component d != cert.component_id, perturb d alone with REPLACE_WITH_NOOP
    # and check if the original violation (cert.oracle_id at cert.event_idx) still fires and
    # the outcome remains unchanged. If so, d is a candidate compensator.
    candidates = []
    for d in harness.components:
        if d == cert.component_id: continue
        toggled_d = harness.perturb(d, mode=REPLACE_WITH_NOOP)
        replay_d  = toggled_d.replay(trace)
        oracle_d  = monitors[cert.oracle_id].check_all(replay_d)
        if oracle_d.fired_at(cert.event_idx) and replay_d.outcome == trace.outcome:
            candidates.append(d)
    return candidates  # empty list means no compensator found; verdict stays UNDERDETERMINED

# Phase 4: evaluation unit
def evaluate(unit=(task_id, oracle_id, component_id), trace):
    status = contamination_proof(task_id)
    certs  = [c for c in certificates if c.oracle_id==oracle_id and c.component_id==component_id]
    verdicts = [replay_verdict(c, trace, harness) for c in certs]
    # For any UNDERDETERMINED verdict, run compensator detection and record the result.
    comp_info = [detect_compensator(c, trace, harness) for c, v in zip(certs, verdicts) if v == UNDERDETERMINED]
    return {unit, status, certs, verdicts, comp_info}   # a unit is VALID iff status==SEALED and any verdict==CAUSAL
```

Hyperparameters: Merkle arity 2 (SHA-256); oracle scope = the seven TraceTrial-Bench components (context, execution, governance, lifecycle, observability, tools, verification), one monitor per (component, constraint) pair; replay perturbation mode fixed to `REPLACE_WITH_NOOP` (the only perturbation that isolates a component without altering the event stream); `UNDETERMINED` is reported, never silently dropped. The masking rate is measured as the fraction of UNDERDETERMINED verdicts where `detect_compensator` returns a non-empty candidate list, computed over the calibration shard (public_index) and reported per component.

**Why this design.** We chose a Merkle commitment over a held-out private set (as in SWE-Bench Pro) because a commitment yields a *checkable* non-inclusion proof for every sealed task, whereas a private set yields only an access-control promise that the benchmark author must be trusted to keep — accepting the cost that sealing prevents per-task difficulty inspection until declassification. We chose executable runtime monitors over LLM-based evidence acquisition (as in TraceTrial-Bench's Multi-source Evidence Acquisition) because a deterministic predicate over `(event, harness state)` produces a reproducible, signable witness, whereas an LLM judge produces a non-replayable score — accepting the cost that monitors must be hand-specified per constraint and cannot cover open-ended failure modes. We chose the counterfactual replay (`REPLACE_WITH_NOOP` on exactly one component) over correlational attribution (as in Who&When's step-level annotation) because replay holds the trace fixed and varies only the component, giving a single-variable intervention, whereas correlation across traces confounds component with task — accepting the cost that replay requires a deterministic harness and is therefore inapplicable to nondeterministic tool backends. A domain expert would not describe HARNESS-SEAL as a variant of TraceTrial-Bench: TraceTrial-Bench *collects* traces and attributes failure via manually designed mechanism chains, whereas HARNESS-SEAL *verifies* a causal link by replaying the same trace against a perturbed harness and requires the signed certificate to survive the perturbation.

**Why it measures what we claim.** The Merkle inclusion path measures *contamination status* because the collision-resistance of SHA-256 makes a valid path to the published sealed root equivalent to membership in the sealed shard; this assumption fails when the author commits the same task under two leaves (duplicate-leaf attack), in which case the proof reflects shard membership but not uniqueness, so HARNESS-SEAL additionally publishes a leaf-multiset hash. The signed oracle certificate measures *violation occurrence* because the monitor's predicate is deterministic over the event stream and the signature binds the witness to the trace hash; this assumption fails when the monitor's scope misses an event (scope under-approximation), in which case the certificate reflects a partial violation rather than a complete one. The replay verdict measures *causal necessity of the harness component* because replacing exactly one component with a no-op while holding the trace fixed is a single-variable intervention, and the verdict requires both that the violation disappear and the outcome change; this assumption fails when the harness has a compensating component that masks the perturbation, in which case the verdict reflects `UNDETERMINED` rather than `NOT_CAUSAL`, and we report it as such rather than collapsing it. The compensating-component detector measures *masking rate* because it checks whether any other component, when no-oped, leaves the violation and outcome unchanged; this assumption fails when the detector's candidate list is empty despite a true compensator (e.g., due to multiple simultaneous compensators), in which case the masking rate is underestimated and the verdict remains `UNDETERMINED`.

## Contribution

['HARNESS-SEAL, a benchmark harness whose evaluation unit is a (task, constraint-oracle, harness-component) triple, with a Merkle-committed public/sealed shard split that makes contamination status a proof-checkable predicate and executable runtime monitors that emit signed violation certificates.', 'A counterfactual replay protocol that validates the violation-to-failure link by perturbing exactly one harness component under a fixed trace, with a three-valued verdict (CAUSAL / NOT_CAUSAL / UNDETERMINED) that separates genuine harness-caused failures from compensating-component masking.', 'An open sealing-and-oracle specification (canonical serialization, leaf-multiset hash, per-component monitor scopes) that ports onto any trace-collecting substrate, demonstrated by re-sealing the TraceTrial-Bench task suite.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TraceTrial-Bench 95-task suite | Sealed multi-component traces with known failure modes |
| Primary metric | Causal-verdict precision on seeded violations | Directly tests single-variable attribution claim |
| Secondary metric | Masking rate (fraction of UNDERDETERMINED with detected compensator) | Quantifies assumption (A) failure frequency |
| Baseline | TraceTrial-Bench mechanism-chain attribution | Closest attribution competitor, same traces |
| Baseline | Who&When step-level annotation | Correlational attribution alternative |
| Baseline | SWE-Bench Pro held-out private set | Access-control contamination baseline |
| Ablation-of-ours | HARNESS-SEAL minus counterfactual replay | Isolates replay's contribution to verdicts |
| Ablation-of-ours | HARNESS-SEAL minus compensator detection | Isolates detector's contribution to masking rate |

### Why this setup validates the claim

The central claim is that a signed certificate surviving a single-variable no-op replay yields verifiable causal attribution that correlational and access-control baselines cannot match. TraceTrial-Bench supplies traces with author-seeded, component-localized violations, so we have ground truth to check verdicts against. The mechanism-chain baseline tests the sub-claim that manual chain construction under-attributes when the failing component is masked by a compensating one; the Who&When baseline tests the sub-claim that correlational step annotation confounds component with task; the SWE-Bench Pro private-set baseline tests the sub-claim that a commitment yields checkable non-inclusion where access control yields only trust. The ablation removes replay, leaving certificates without verdicts, so any gain attributable to causal verification disappears — this isolates the mechanism rather than the harness plumbing. Causal-verdict precision is the right metric because the method's whole value proposition is that a VALID unit means a certified violation, not a scored guess; precision on seeded violations detects exactly the predicted effect while remaining insensitive to how many violations exist. The masking rate directly measures the frequency of assumption (A) failure (compensating components), and the compensator-detection ablation tests whether our detector correctly identifies those cases, bounding the causal claim by a measured validity condition.

### Expected outcome and causal chain

**vs. TraceTrial-Bench mechanism-chain attribution** — On a case where a governance constraint fires but a lifecycle component silently repairs the state, the baseline attributes failure to the visible governance step because its manually designed mechanism chain follows the firing order, not the counterfactual dependence. Our method instead no-ops the lifecycle component, observes the violation vanish and the outcome change, and returns CAUSAL because the replay isolates that component. We expect a noticeable precision gap concentrated on traces with compensating components, with near-parity on traces lacking them. Additionally, we expect the masking rate to be non-zero (e.g., 10–20% of UNDERDETERMINED verdicts have a detected compensator), confirming that assumption (A) fails in practice and that our detector captures it.

**vs. Who&When step-level annotation** — On a case where the same component fails across many tasks, the baseline's correlational annotation marks the component as responsible whenever it co-occurs with failure, even on tasks where the trace would have failed without it. Our method instead requires both violation disappearance and outcome change under perturbation, so it withholds CAUSAL where the component is merely correlated. We expect the baseline to over-attribute on high-co-occurrence subsets while our verdicts stay stable, producing a gap that widens with component reuse frequency.

**vs. SWE-Bench Pro held-out private set** — On a case where a sealed task leaks into an agent's training data, the private-set baseline cannot prove non-inclusion and must rely on the author's access-control promise, so contamination status is unverifiable. Our method instead returns a SEALED non-inclusion path recomputable against the published root, making status checkable without revealing task bodies. We expect our contamination-status audit to be decisive on every sealed task while the baseline is decisive only on opened tasks, a gap that is total on the sealed shard.

### What would falsify this idea

If our causal-verdict precision gain over mechanism-chain attribution is uniform across compensating and non-compensating traces rather than concentrated where compensating components exist, the replay mechanism is not doing the causal work and the central claim fails. Likewise, if the replay ablation shows no precision drop, certificates alone suffice and counterfactual verification is unnecessary. Additionally, if the masking rate is zero (no compensators detected) across the calibration shard, assumption (A) holds trivially and the detector is unnecessary; if the masking rate is high (>50%) and the detector fails to identify compensators, the causal claim is severely bounded.

## References

1. Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution
2. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
3. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
4. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems
5. OpenAgents: An Open Platform for Language Agents in the Wild
6. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
7. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
8. SWE-Bench+: Enhanced Coding Benchmark for LLMs
