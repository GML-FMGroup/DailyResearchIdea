# Compositional Safety Closure for LLM Agent Teams via Delegation-Monitored Composition

## Motivation

Existing multi-agent alignment work measures team safety with aggregate rubrics and repairs it by assigning a safety-lead role, without any account of whether team-level safety is preserved by construction. Aligning AI Teams (E1) reports that natural interventions such as mutual critique and group-chat coordination fail to restore single-agent alignment, and that diffusion of responsibility accounts for the gap, yet it offers no compositional model of why the safety-lead and coordinator-specification protocols would preserve safety rather than merely correlate with higher scores. The root cause is structural: team alignment is treated as an emergent aggregate property of a protocol, not as a property composed from member safety states through a named operator with stated closure conditions, so the field cannot distinguish protocols that preserve alignment from protocols that happen to score well on the sampled tasks.

## Key Insight

Team-level safety is preserved exactly when the composition operator makes every system-level safety-relevant proposition either owned by a monitored delegate or explicitly covered by the published specification, so the closure failure of any protocol is localized to a concrete unowned proposition rather than to an unexplained aggregate score drop.

## Method

```pseudocode
INPUT: team protocol P (roles, message graph, delegation edges), member agents a_1..a_n,
       safety-relevant proposition set Q (extracted per task), monitor M, budget B
OUTPUT: composed team safety state S_team, closure verdict, minimal failure witness

# Phase 0: behavioral ownership probe (ground truth for M)
# For each (agent a_i, proposition q), run a probe that makes inaction observable:
#   - inject a task instance where q is at risk and the agent must emit a check/flag/block
#   - record the action trace T_i(q) = (check_emitted, flag_raised, handoff_blocked)
# OWNED_behavior(q) = TRUE iff T_i(q) shows at least one of {check_emitted, flag_raised, handoff_blocked}
# M is a rubric-conditioned LLM judge (fixed model, temperature=0) that reads T_i(q) and the agent's text.
# M returns {OWNED, DEFERRED(owner), IGNORED, RATIONALIZED}.
# Calibration set C: 512 (agent, proposition, trace) tuples with human labels for all four classes.
# M is accepted only if per-class precision/recall on C >= 0.80 and Cohen's kappa vs. human >= 0.70.
# Any (i,q) where M disagrees with OWNED_behavior(q) is forced to UNOWNED by default.

# Phase 1: per-agent invariant extraction (no rubric aggregation)
for each agent a_i:
    for each proposition q in Q:
        s_i(q) = M(a_i, q, T_i(q))   # M returns {OWNED, DEFERRED(owner), IGNORED, RATIONALIZED}
        if s_i(q) == OWNED and not OWNED_behavior(q): s_i(q) = UNOWNED  # behavioral override
    Inv_i = { q : s_i(q) == OWNED }  # per-agent safety invariant

# Phase 2: specification coverage from coordinator agent
Spec = extract_spec(P.coordinator_output)   # set of propositions the spec document asserts
# Extraction protocol: fixed prompt template, model=gpt-4o, temperature=0, versioned;
# human spot-check of 10% of extracted propositions; report inter-run variance.

# Phase 3: delegation-monitoring condition (the closure rule)
for each delegation edge (a_i -> a_j) carrying proposition q:
    monitored(i->j, q) = M(a_j, q, T_j(q)) == OWNED  AND  a_i records the handoff in Spec
DM(P, Q) = all edges (i->j, q) satisfy monitored(i->j, q)

# Phase 4: composition operator
S_team(q) = OWNED      if q in union_i Inv_i
          = COVERED    if q in Spec and DM holds on the edge that delivers q
          = UNOWNED    otherwise

# Phase 5: closure verdict + witness
if every q in Q has S_team(q) in {OWNED, COVERED}: verdict = CLOSED
else: verdict = OPEN; witness = argmin_{q: S_team(q)=UNOWNED} cost_to_cover(q)

# Phase 6: targeted repair (only on witnesses)
repeat up to B rounds: re-delegate witness q to a member; recompute Phases 3-5
```

**What it is.** We introduce COSMIC (Compositional Safety via Monitored Invariant Closure), a framework that takes a team protocol, its member agents, and a task-derived set of safety-relevant propositions, and returns a composed team safety state together with a closure verdict and, when the verdict is OPEN, a minimal unowned-proposition witness. The inputs are the protocol and proposition set; the output is a per-proposition ownership map, not a scalar score.

**How it works.** Phase 0 defines behavioral ownership operationally: for each (agent, proposition) pair we run a probe that makes inaction observable and record an action trace T_i(q); OWNED_behavior is true iff the agent emits a check, raises a flag, or blocks a handoff. The monitor M is a fixed rubric-conditioned LLM judge (temperature=0) that reads T_i(q) and the agent's text and returns OWNED, DEFERRED, IGNORED, or RATIONALIZED; M is calibrated on a held-out set C of 512 human-labeled tuples and accepted only if per-class precision/recall ≥ 0.80 and Cohen's kappa ≥ 0.70. Any (i,q) where M disagrees with OWNED_behavior is forced to UNOWNED. Phase 1 probes each member agent on each safety-relevant proposition and classifies its response using M with the behavioral override. Phase 2 extracts the coordinator agent's safety specification as a proposition set using a fixed, versioned prompt (model=gpt-4o, temperature=0) with a 10% human spot-check. Phase 3 checks the delegation-monitoring condition: every handoff edge must deliver a proposition that the receiving agent actually owns (by M with behavioral override) and that the sender records in the specification. Phase 4 composes the team state proposition-wise: a proposition is OWNED if any member owns it, COVERED if the specification asserts it and the delivering edge is monitored, and UNOWNED otherwise. Phase 5 declares closure and, on failure, returns the cheapest unowned proposition as a witness. Phase 6 spends a repair budget B (B=3 rounds) re-delegating only witnesses, which is the operational analogue of the safety-lead intervention but targeted by witness rather than by role assignment.

**Why this design.** We chose a proposition-wise composition operator over the aggregate rubric used in Aligning AI Teams (E1) because an aggregate cannot localize a failure, and localization is exactly what the closure rule needs to distinguish preservation from correlation; the cost is that Q must be extracted per task, which we accept because the extraction is a fixed preprocessing step. We chose a delegation-monitoring condition over a simple per-agent conjunction because the diffusion-of-responsibility finding in E1 shows that per-agent ownership does not survive handoffs; the cost is that the monitor must correctly classify DEFERRED versus OWNED, which we accept and probe in ablation. We chose a witness-based repair budget over the role-based safety-lead intervention of E1 because role assignment does not specify which proposition the lead must own; the cost is that repair may need multiple rounds, bounded by B. A domain expert would not describe COSMIC as a variant of MACPO (E2) or ICR (E3): MACPO is a preference-optimization training loop and ICR is a partner-aware RL algorithm, whereas COSMIC is a static composition operator over an already-trained team with an explicit closure condition and a witness, and it does not train any agent.

**Why it measures what we claim.** The per-proposition ownership classification s_i(q) measures member-level safety ownership because we assume the monitor's OWNED label, grounded in the behavioral action trace T_i(q) and overridden to UNOWNED on disagreement with OWNED_behavior, tracks proactive system-level safety checking rather than post-hoc rationalization; this assumption fails when an agent produces fluent rationalization that the monitor misreads as ownership despite the behavioral override, in which case the metric reflects rhetorical compliance rather than safety behavior. The delegation-monitoring predicate monitored(i->j, q) measures preserved responsibility across handoffs because we assume a recorded handoff in the specification is evidence that the sender did not diffuse responsibility; this assumption fails when the specification is written after the fact to match whatever the agents did, in which case the predicate reflects documentation rather than monitoring. The closure verdict measures team-level alignment preservation because we assume the proposition set Q is complete for the task; this assumption fails when a safety-relevant proposition is missing from Q, in which case closure reflects coverage of the extracted set rather than of the task.

## Contribution

['A proposition-wise composition operator with an explicit delegation-monitoring closure condition that maps member-agent safety states and a coordinator specification to a team-level safety state and a minimal unowned-proposition witness.', 'A diagnostic finding that the safety-lead and coordinator-specification protocols of Aligning AI Teams preserve team alignment only on propositions that satisfy the delegation-monitoring condition, and that their reported gains are attributable to the subset of propositions where that condition happens to hold.', 'A witness-targeted repair procedure that spends a bounded budget re-delegating only unowned propositions, as an alternative to role-based safety-lead assignment.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Aligning AI Teams task suite (sepsis triage, news recommender, consulting proposals) | Matches the diffusion-of-responsibility failure we target |
| Primary metric | Proposition-level closure rate with witness localization accuracy | Localizes failures, unlike aggregate rubric scores |
| Baseline | Single-agent alignment behavior | Tests whether teams degrade below solo agents |
| Baseline | Mutual-critique / group-chat intervention | Tests whether generic coordination closes the gap |
| Baseline | Designated safety-lead role intervention | Tests whether role assignment without witness suffices |
| Ablation-of-ours | COSMIC without delegation-monitoring condition | Isolates the handoff-diffusion mechanism |
| Monitor validation | Held-out calibration set C of 512 (agent, proposition, trace) tuples with human labels for OWNED/DEFERRED/IGNORED/RATIONALIZED; report per-class precision/recall, Cohen's kappa, and false-OWNED rate on injected rationalization traps | Validates the load-bearing assumption that M separates OWNED from RATIONALIZED |
| Verdict stability | Swap judge models (e.g., gpt-4o vs. claude-3.5-sonnet) and report verdict flip rate on 100 randomly sampled propositions | Tests whether closure verdicts are stable under monitor perturbation |
| Q-completeness audit | Human expert review of 10% of extracted propositions per task; report missing-proposition rate | Quantifies the assumption that Q is complete |

### Why this setup validates the claim

The central claim is that team safety failures are localized, handoff-specific ownership gaps, and that a proposition-wise closure rule with a witness can detect and repair them where aggregate scoring and role-based interventions cannot. The dataset is the right testbed because Aligning AI Teams already documents that single-vs-team gaps appear on these tasks and that natural interventions fail to close them, so any gain here is not an artifact of an easy domain. The single-agent baseline tests the sub-claim that teams are worse than the sum of members; the mutual-critique and safety-lead baselines test whether generic coordination or role assignment alone recovers ownership. The ablation is the load-bearing test: removing the delegation-monitoring condition should collapse performance specifically on handoff-carrying propositions, confirming that the closure rule, not the per-agent probing, drives the effect. The proposition-level closure metric is chosen because it is the only metric that can distinguish a genuinely owned proposition from one that is merely asserted in a specification, which is precisely the distinction the claim rests on. The monitor-validation study directly tests the load-bearing assumption that M can separate OWNED from RATIONALIZED: we report per-class precision/recall and Cohen's kappa on the 512-tuple calibration set, and we inject known rationalization traps to measure the false-OWNED rate. The verdict-stability check swaps judge models and reports the flip rate, ensuring that closure verdicts are not an artifact of a single judge. The Q-completeness audit quantifies the assumption that the proposition set is complete, reporting the missing-proposition rate from expert review.

### Expected outcome and causal chain

**vs. Single-agent alignment behavior** — On a case where a sepsis-triage proposition (e.g., flagging a contraindicated drug interaction) is owned by the solo agent, the team baseline produces a silent drop because responsibility diffuses across members who each assume another checks it. Our method instead probes each member per proposition and marks the proposition UNOWNED unless some member explicitly owns it, exposing the gap. We expect a noticeable closure-rate gap on handoff-heavy propositions but near-parity on propositions that never cross an agent boundary.

**vs. Mutual-critique / group-chat intervention** — On a case where the news recommender team must jointly enforce a safety constraint, the critique baseline produces fluent mutual agreement that rationalizes the risk away because critique rewards consensus rather than ownership assignment. Our method instead classifies each proposition as OWNED, DEFERRED, IGNORED, or RATIONALIZED and refuses to count rationalization as coverage, so the same case yields an OPEN verdict with a witness. We expect the baseline to show high apparent agreement but low closure, i.e., a divergence between its aggregate score and our proposition-level verdict.

**vs. Designated safety-lead role intervention** — On a case where the safety lead is assigned but the specification does not name which propositions the lead owns, the baseline produces partial coverage because the role exists without a proposition-level mandate and handoffs remain unmonitored. Our method instead requires each delegation edge to deliver an owned proposition recorded in the specification, so unmonitored handoffs surface as witnesses and get repaired within budget B. We expect gains concentrated on delegation-edge propositions, with the safety-lead baseline matching us only where no handoff occurs.

### What would falsify this idea

If COSMIC's gain over the safety-lead and critique baselines is uniform across all proposition subsets rather than concentrated on handoff-carrying propositions, the delegation-monitoring mechanism is not the cause and the central claim is wrong. Likewise, if removing the delegation-monitoring ablation leaves closure rates unchanged, the closure rule is doing no work beyond per-agent probing.

## References

1. Aligning AI Teams
