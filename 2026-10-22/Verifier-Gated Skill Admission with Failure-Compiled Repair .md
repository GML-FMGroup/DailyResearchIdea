# Verifier-Gated Skill Admission with Failure-Compiled Repair Constraints for Embodied GUI Agents

## Motivation

Across multiple agent-tooling trees, the externalized skill/evidence store is populated once and then consumed read-only: VISUALSKILL builds its multimodal skill index through authored documents plus a single UI-exploration pass and never revises it from downstream execution outcomes, while CONFLICTGUARD consumes a filtered HTML snapshot and the agent's own grounding confidence without writing back what it learned. The structural consequence is that the store is never closed under execution-grounded validation — a skill can be admitted on documentation plausibility alone, and a failed grounding or planning episode leaves no trace in the index, so the store cannot converge toward execution-correct skills. The root cause is the absence of a shared execution-grounded invariant that simultaneously gates admission and channels failure into write-back.

## Key Insight

If and only if a skill's admission certificate is a verifier-produced execution trace whose preconditions are themselves derived from previously certified traces, the index becomes monotonically tightening under a partial order of execution reachability, so every write-back either widens no unsound skill or is rejected — making the store converge to execution-correct skills rather than drift with documentation priors.

## Method

**What it is.** VGSA (Verifier-Gated Skill Admission) maintains a skill index `I` of entries `(s, C_s, P_s)` where `s` is a GUI skill or feasibility rule, `C_s` is a verifier-backed execution certificate, and `P_s` is the set of admission preconditions derived from other certified entries. Inputs: candidate skills from authored docs / UI exploration / failure repairs. Outputs: a monotonically tightening index and, at inference, only entries whose certificates are still valid under the current environment probe.

**Load-bearing assumption (stated explicitly).** A precondition `p` is treated as satisfiable in the live environment iff some previously-certified entry's postcondition witnesses `p`. This is a closed-world coverage assumption on the reachable state space: it holds only when every reachable state is witnessed by a certified trace. It is structurally false for open-ended GUI environments (e.g., a fresh login screen reachable by no certified entry). To keep the gate measuring soundness rather than index coverage, the method includes an explicit coverage/exploration channel: a skill whose preconditions are unwitnessed may still be admitted if a verifier-executed state-construction probe reaches and certifies the precondition state; that probe is recorded as a first-class certified entry. The partial order is therefore closed under probe-witnessed states, not only under prior skill postconditions.

**How it works.**

```pseudocode
# ---- Admission phase ----
def admit(s, I, env):
    # 1. Precondition grounding: every precondition p in P_s must be
    #    witnessed by an already-certified entry in I (partial order check)
    #    OR by a probe-witnessed state recorded as a certified entry.
    for p in P_s:
        if not exists (s', C_s', _) in I with p in postconditions(C_s'):
            # coverage/exploration channel: try to construct and certify p
            probe = env.probe_state(p)          # verifier-executed state-construction trace
            if probe.success and probe.postcondition_match:
                I.add((probe.skill, probe.certificate, probe.preconditions))
            else:
                return REJECT(p, reason="unwitnessed_precondition",
                              coverage_attempt=probe)
    # 2. Execution certificate: run verifier V on s in env, N=3 trials
    traces = [V.execute(s, env) for _ in range(3)]
    if not all(t.success and t.postcondition_match for t in traces):
        return REJECT(s, reason="execution_failure", traces=traces)
    # 3. Certificate = canonical trace + environment fingerprint
    C_s = (canonicalize(traces), env.fingerprint())
    I.add((s, C_s, P_s))
    return ADMIT(C_s)

# ---- Failure-compiled repair ----
def on_failure(episode, I, env):
    # episode = failed grounding/planning trace with a diagnosed step
    r = compile_minimal_repair(episode)          # smallest edit to skill/rule
    new_constraint = derive_constraint(r)        # e.g. "button X only clickable after dialog Y"
    # Write-back is a NEW admission, not an edit — monotone by construction
    return admit(new_constraint, I, env)         # rejected repairs leave I unchanged

# ---- Inference ----
def select(task, I, env):
    # Only entries whose fingerprint matches the live env probe are eligible
    eligible = [e for e in I if e.C_s.fingerprint == env.probe().fingerprint]
    return top_k_by_execution_reachability(eligible, task, k=5)
```

**Why this design.** We chose a *partial-order precondition check* over a flat admission list because a flat list lets two individually-plausible skills jointly violate the environment (each is executable alone but their preconditions conflict), accepting the cost that admission becomes O(|I|) per candidate rather than O(1). We chose *execution certificates over documentation scores* because VISUALSKILL's authored-doc pipeline admits skills on textual plausibility, which we found does not transfer to verifier-passing skills; the trade-off is that every admission costs N=3 live executions, so cold-start is slower. We chose *write-back as a new admission rather than an in-place edit* because in-place edits break monotonicity (an edit can silently invalidate previously-certified entries), accepting the cost that the index grows with failures rather than shrinking. We chose *fingerprint-gated inference eligibility* over always-on retrieval because a certificate issued against a stale UI is unsound; the trade-off is that UI drift temporarily shrinks the eligible set. A domain expert would not describe this as a variant of ASI's program-induction loop: ASI verifies individual programs in isolation and stores them, whereas VGSA's admission predicate is defined over the *joint* precondition lattice of already-admitted entries, so a new skill is only admitted if it extends a certified reachability chain — a property ASI's library does not maintain.

**Why it measures what we claim.** The `unwitnessed_precondition` rejection measures *execution-grounded soundness* because we assume a precondition is satisfiable in the current environment iff some previously-certified entry's postcondition witnesses it; this assumption fails when the environment has a state reachable by no certified entry (e.g., a fresh login screen), in which case the rejection reflects index incompleteness rather than unsoundness. The N=3 execution certificate measures *skill reliability* because we assume the environment is deterministic enough that three successes imply the postcondition holds under the fingerprint; this assumption fails under nondeterministic latency or A/B UI variants, in which case the certificate reflects a lucky sample rather than a stable skill. The `canonicalize` step measures *repair minimality* because we assume the diagnosed failing step is the unique edit point; this assumption fails when a failure is caused by an upstream grounding error, in which case the compiled repair constrains the wrong step and is rejected by the next admission attempt — which is exactly the monotone gate doing its job. The fingerprint match measures *certificate validity* because we assume the UI fingerprint is a sufficient statistic for skill applicability; this fails on partial UI drift, in which case eligibility reflects a conservative over-rejection rather than unsound admission. To make the incompleteness-vs-unsoundness distinction measurable, every `unwitnessed_precondition` rejection is logged with the outcome of the coverage/exploration probe: if the probe succeeds, the rejection is attributable to index incompleteness (the state was reachable but not yet certified); if the probe fails, the rejection is attributable to genuine unsoundness (the precondition is not satisfiable in the live environment). The per-rejection decomposition is reported as the fraction of rejections in each category.

## Contribution

(1) A closed-loop skill index in which admission is gated by a verifier-backed execution certificate whose preconditions must be witnessed by previously-certified entries, yielding a monotone partial order over admitted skills. (2) A failure-compilation operator that converts each failed grounding/planning episode into a minimal repair constraint admitted through the same gate, so the index tightens under execution outcomes rather than drifting with documentation priors. (3) A fingerprint-gated inference-eligibility rule that restricts retrieval to certificates valid under the live environment probe, with an explicit account of when this over-rejects.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OSWorld + CUA-World | Live GUI drift stresses certificate validity |
| Primary metric | Task success rate | Directly reflects sound skill admission |
| Secondary metric | Incompleteness-vs-unsoundness rejection decomposition | Measures whether gate reflects soundness or coverage |
| Baseline | VISUALSKILL | Tests doc-scored admission against verifier-gated |
| Baseline | ASI program induction | Tests isolated verification against joint lattice |
| Baseline | Agent S2 | Tests generalist agent without certified skill index |
| Ablation-of-ours | VGSA w/o partial-order check | Isolates precondition-lattice contribution |
| Ablation-of-ours | VGSA w/o coverage/exploration probe | Isolates probe-witnessed-state channel contribution |

### Why this setup validates the claim

The central claim is that execution-certified, precondition-grounded, fingerprint-gated skill admission yields sound skills that transfer, where textual plausibility and isolated program verification do not. VISUALSKILL tests the sub-claim that documentation scores fail to predict verifier-passing skills: if our method beats it, execution certificates are doing real work. ASI tests the sub-claim that joint precondition-lattice admission beats isolated program induction: if our gain concentrates on tasks requiring chained skills, the lattice is causal. Agent S2 tests whether a generalist without any certified index leaves performance on the table. The ablation (dropping the partial-order precondition check) is the sharpest test: if removing it does not increase joint-violation failures, the lattice is decorative. The second ablation (dropping the coverage/exploration probe) tests whether the probe-witnessed-state channel is doing the work of closing the partial order under reachable states rather than only under prior skill postconditions; if removing it does not increase `unwitnessed_precondition` rejections attributable to index incompleteness, the channel is not contributing to soundness. Success rate is the right primary metric because the claim is about soundness of admitted skills under live environments, not efficiency; a method that admits unsound skills should show success-rate collapse specifically on drift and chained-precondition subsets. The rejection decomposition is the right secondary metric because it directly measures the load-bearing assumption: the fraction of rejections attributable to index incompleteness (probe succeeds) versus genuine unsoundness (probe fails) tells us whether the gate is measuring soundness or coverage.

### Expected outcome and causal chain

**vs. VISUALSKILL** — On a case where a skill is documented as "click the export button" but the live UI requires a preceding format-selection dialog, VISUALSKILL admits the skill on textual plausibility and the agent clicks a disabled button, stalling. Our method instead rejects the skill at admission because no certified entry's postcondition witnesses the dialog precondition, then admits the repair as a new entry after N=3 verifier passes. We expect a noticeable success-rate gap on chained-precondition tasks and near-parity on single-step tasks where textual plausibility happens to coincide with executability. The rejection decomposition should show that most `unwitnessed_precondition` rejections on chained tasks are attributable to genuine unsoundness (probe fails), confirming the gate is measuring soundness rather than coverage.

**vs. ASI program induction** — On a case where two individually-verified skills each pass in isolation but jointly conflict (skill A leaves a modal open that skill B's click assumes closed), ASI stores both and the composed plan fails at execution. Our method instead rejects the second skill at admission because its precondition is unwitnessed by any certified postcondition, keeping the index sound. We expect our gain to concentrate on multi-skill composition tasks, with parity on single-skill tasks where isolated verification suffices. The rejection decomposition should show that joint-conflict rejections are attributable to genuine unsoundness (the probe cannot construct a state satisfying the conflicting precondition), not to index incompleteness.

**vs. Agent S2** — On a case where the UI drifts (a button relocates after an A/B variant rolls out), Agent S2's generalist policy re-grounds from scratch each episode and wastes steps re-discovering the interaction, occasionally clicking stale coordinates. Our method instead gates retrieval on the live fingerprint, so stale certificates are excluded and only re-admitted entries are eligible, trading temporary coverage loss for soundness. We expect a modest but consistent gap on drift-affected episodes and parity on stable-UI episodes. The rejection decomposition should show that drift-induced rejections are attributable to index incompleteness (the probe can reach the relocated button but no certified entry yet witnesses it), which the coverage/exploration channel then repairs by admitting the probe as a certified entry.

### What would falsify this idea

If our gain over VISUALSKILL is uniform across single-step and chained-precondition subsets rather than concentrated on chained tasks, the precondition-lattice mechanism is not doing the claimed work. If the ablation without the partial-order check shows no increase in joint-violation failures, the central soundness claim is wrong. If the rejection decomposition shows that most `unwitnessed_precondition` rejections are attributable to index incompleteness (probe succeeds) rather than genuine unsoundness (probe fails), the gate is measuring coverage rather than soundness, and the load-bearing assumption is false.

## References

1. VISUALSKILL: Multimodal Skills for Computer-Use Agents
2. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
3. Inducing Programmatic Skills for Agentic Tasks
4. Agent S: An Open Agentic Framework that Uses Computers Like a Human
5. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
6. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
7. Agent Workflow Memory
8. Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents
