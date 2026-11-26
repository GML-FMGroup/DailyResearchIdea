# Revision-Closed Action Spaces for Egocentric Computer-Use Agents

## Motivation

Existing egocentric agent benchmarks treat the action space as fixed: EgoUseBench (arXiv:2611.34466) explicitly asks agents to 'reassess whether prior digital actions remain valid' as new first-person observations arrive, but its action model offers no operator for invalidating or rolling back an already-executed digital action — validity is a property agents are merely asked to reason about, never a property of the action set itself. The same fixed-action-space assumption recurs in COMBO (venue:iclr:2025:c42f9a9010e7), whose compositional world model plans over joint actions drawn from a static per-agent set. Structurally, this is because both benchmarks define the action space a priori and evaluate consistency post hoc, so no closure rule exists under which belief revision can shrink or expand the legal action set; an agent can only be scored on whether it 'should have' revised, never constrained to act within a revised set.

## Key Insight

If every executed digital action carries a machine-checkable precondition grounded in first-person evidence, then belief revision becomes an operator on the action set — retracting an action's evidence support closes the valid action set under revision by construction, so consistency is enforced by the legal-action frontier rather than scored after the fact.

## Method

**What it is.** RCAS (Revision-Closed Action Space) is a benchmark harness plus agent protocol in which each digital action is a tuple `(op, args, precondition, evidence_refs)` and the legal action set at any step is the closure of actions whose preconditions are satisfied by the agent's current, revisable belief state. Inputs: a stream of first-person observations and an evolving task request. Outputs: a sequence of executed actions plus a revision trace, and per-episode validity/consistency scores.

**Load-bearing assumption (stated explicitly).** RCAS assumes that every executed digital action has (i) a machine-checkable precondition grounded in first-person evidence and (ii) a clean, precondition-checkable inverse, so that retracting evidence support deterministically shrinks the legal action set and LIFO compensating rollback restores a consistent state. This assumption does not hold for all digital actions (e.g., an email already read by a recipient is IRREVERSIBLE; first-person observations may be incomplete or ambiguous). RCAS therefore declares an explicit action-class partition in the benchmark spec: only actions whose preconditions and inverses are provably machine-checkable from the fixed ontology enter the closure and the primary legality metric; all others are quarantined as IRREVERSIBLE, excluded from the legality metric, and scored separately. The primary metric is defined over the certified subset so the contribution is measured only where the closure is well-defined.

**How it works.**

```pseudocode
# Belief state B: map from proposition -> (support_set, status)
#   status in {ASSERTED, RETRACTED}; support_set = observation ids
# Action schema: a = (op, args, pre(a), ev(a))
#   pre(a): conjunction of propositions that must be ASSERTED
#   ev(a): observation ids the agent cited when asserting pre(a)
# Action-class partition (declared in benchmark spec):
#   CERTIFIED: pre(a) and inverse(a) both machine-checkable from ontology
#   IRREVERSIBLE: no clean inverse, or pre(a) not certifiable from egocentric frames
#   Only CERTIFIED actions enter legal(A) and the primary legality metric.

init B from first-person observations O_1..O_t (Offline Ego Use)
legal(A) = { a in A : a.class == CERTIFIED and all p in pre(a) has B[p].status == ASSERTED }

def step(B, O_new, A):
    # 1. Ingest new evidence; detect contradictions
    B = assert_from(B, O_new)
    conflicts = {(p, q) : p in B, q in B, contradicts(p, q)}
    # 2. Belief revision: retract the weaker-supported side
    for (p, q) in conflicts:
        weaker = p if support_weight(B[p]) < support_weight(B[q]) else q
        B[weaker].status = RETRACTED
    # 3. Revision-closure: any executed action whose ev() cited a
    #    retracted proposition is marked INVALIDATED
    invalidated = { a_exec : exists p in ev(a_exec), B[p].status == RETRACTED }
    # 4. Rollback: emit compensating actions for invalidated CERTIFIED actions
    #    (undo stack, LIFO), each compensating action itself precondition-checked
    for a in reverse(invalidated):
        if a.class == IRREVERSIBLE:
            log(a, status=IRREVERSIBLE_FLAGGED); continue
        c = compensate(a)              # op-specific inverse from published inverse table
        if all(p in pre(c) has B[p].status == ASSERTED):
            execute(c); log(c)
    # 5. Recompute legal set and let agent act
    return legal(A), B, invalidated
```

`support_weight(B[p])` is a fixed integer count of independent observation ids in `support_set` (no learned score). Contradiction detection uses a hand-specified proposition ontology per task domain (e.g., `object_present(x, location)` vs `object_absent(x, location)`); this ontology is part of the benchmark spec, not learned. The initial implementation is scoped to a fixed, enumerated action vocabulary of 18 ops (e.g., `open_app`, `click_element`, `type_text`, `send_message`, `book_slot`, `read_field`, `scroll`, `navigate`, `confirm`, `cancel`, `undo_last`, `copy`, `paste`, `select`, `submit_form`, `close`, `search`, `share`), with a published ontology and inverse table; ops without a certified inverse are declared IRREVERSIBLE in that table.

**Why this design.** We chose a *symbolic, hand-specified proposition ontology with integer support counts* over a learned belief tracker because the benchmark's purpose is to measure revision behavior, not belief-estimation accuracy — a learned tracker would confound the two, accepting the cost that RCAS cannot score domains whose propositions cannot be hand-annotated. We chose *precondition-tagged actions with explicit evidence references* over post-hoc consistency scoring because post-hoc scoring (the EgoUseBench approach) cannot constrain the legal action set, accepting the cost that agents must expose their evidence citations, which adds protocol overhead. We chose *LIFO compensating rollback* over re-planning from scratch because rollback preserves the audit trail that makes revision traceable, accepting the cost that not every digital action has a clean inverse (e.g., an email already read by a recipient), which we handle by marking such actions IRREVERSIBLE and scoring them separately. A domain expert would not describe RCAS as a variant of EgoUseBench's Online Ego Use setting: EgoUseBench asks whether an agent *revises its understanding*, whereas RCAS makes revision an *operator that changes the legal action set*, so the two differ in what is being measured, not in dataset or hyperparameters.

**Why it measures what we claim.** The quantity `|legal(A) after revision| / |legal(A) before revision|` measures *revision responsiveness* because we assume that a correctly grounded agent's preconditions are exactly the propositions its cited evidence supports; this assumption fails when an agent cites evidence it did not actually use, in which case the ratio reflects citation honesty rather than revision. The count of `invalidated` executed actions measures *action-invalidation exposure* because we assume each executed action's `ev()` is the complete evidential basis for its execution; this fails when an action was executed for reasons outside the ontology, in which case it reflects ontology coverage. The rollback success rate (compensating actions that passed their own precondition check) measures *recoverability* because we assume compensating actions are exact inverses; this fails for IRREVERSIBLE actions, in which case it reflects the fraction of reversible task structure. To quantify these assumptions rather than assert them, we run a calibration study on a held-out annotated subset of 200 episodes: (a) we measure the Spearman correlation between integer `support_weight` and human-judged evidential strength (5-point Likert, 3 annotators); (b) we report the citation-honesty rate — the fraction of executed actions whose `ev()` matches the evidence a human judge identifies as actually used — and we use this rate as a diagnostic covariate when interpreting the revision-responsiveness ratio.

## Contribution

(1) A revision-closed action-space formulation in which digital actions carry first-person-evidence-grounded preconditions, so belief revision mechanically shrinks and expands the legal action set — a closure rule absent from EgoUseBench and COMBO. (2) A benchmark protocol and scoring suite (revision responsiveness, invalidation exposure, rollback recoverability) that separates an agent's belief-update quality from its action-validity maintenance. (3) An open harness with a hand-specified proposition ontology and compensating-action library for a set of digital task domains.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | EgoUseBench Online Ego Use episodes (primary); HASARD navigation episodes with a second ontology (secondary) | Evolving first-person context is exactly the revision trigger; second domain tests generality of the paradigm |
| Primary metric | Revision-closure validity rate over the CERTIFIED action subset (legal-set + invalidation consistency) | Directly scores whether revision changes action legality, measured only where the closure is well-defined |
| Secondary metrics | Citation-honesty rate; Spearman correlation between `support_weight` and human-judged evidential strength; rollback success rate on reversible subset; IRREVERSIBLE-flag rate | Quantifies the grounding assumptions rather than asserting them |
| Baseline | EgoUseBench Computer-Use (explicit-instruction) agent | Tests whether first-person revision matters at all |
| Baseline | EgoUseBench Online Ego Use (post-hoc consistency scoring) | Tests whether constraining legality beats scoring after the fact |
| Baseline | Re-plan-from-scratch agent (no rollback) | Tests whether closure+rollback beats naive replanning |
| Ablation-of-ours | RCAS with post-hoc scoring, no precondition gating | Isolates the contribution of closure over scoring |
| Calibration set | 200 held-out episodes, 3 annotators, 5-point evidential-strength Likert | Bounds ontology-construction cost and quantifies citation honesty |

### Why this setup validates the claim

The central claim is that making revision an operator over the legal action set — rather than a post-hoc consistency judgment — produces measurable differences in how agents behave under evolving first-person context. EgoUseBench's three settings give a controlled ladder: the Computer-Use baseline has no first-person context to revise, so it tests whether revision is needed at all; the Online Ego Use baseline has the same context stream as RCAS but scores consistency after execution, so it isolates the effect of constraining legality before execution; the replan-from-scratch baseline has RCAS's preconditions but discards the audit trail, isolating the rollback contribution. The ablation removes only precondition gating while keeping the ontology and evidence citations, so any gap between it and full RCAS is attributable to closure rather than to annotation quality. Revision-closure validity rate is the right metric because it is computed from the same proposition ontology the benchmark specifies and is restricted to the CERTIFIED action subset, so it detects the predicted effect — legality changing when evidence is retracted — without a learned belief tracker confounding the measurement and without IRREVERSIBLE actions inflating or deflating the score. The calibration study on 200 held-out episodes measures the two equivalences the metric depends on: the Spearman correlation between integer `support_weight` and human-judged evidential strength, and the citation-honesty rate (fraction of actions whose `ev()` matches the evidence a human judge identifies as actually used). A uniform gain across reversible and IRREVERSIBLE subsets would be uninformative; the design is built to produce a concentrated signal, and the HASARD secondary domain with a second ontology tests whether the effect is a single-benchmark artifact.

### Expected outcome and causal chain

**vs. EgoUseBench Computer-Use** — On an episode where the wearer's activity changes mid-task (say, an object moves out of frame after being asserted present), the Computer-Use baseline produces actions grounded in the stale initial instruction because it never ingests the first-person stream as revisable evidence. Our method instead retracts the contradicted proposition and removes dependent actions from the legal set because preconditions are checked against the current belief state, so we expect near-parity on episodes with no mid-task contradiction and a clear gap concentrated on the contradiction-bearing subset.

**vs. EgoUseBench Online Ego Use** — On an episode where the agent acts on an object it misidentified and later evidence corrects the identification, the post-hoc-scoring baseline produces a low consistency score but has already executed the dependent action, because scoring cannot constrain the legal set. Our method instead invalidates the executed action through revision-closure and emits a compensating action, because evidence references are attached to preconditions at execution time, so we expect the gap to appear specifically in invalidation-exposure and rollback-success terms rather than in raw task completion. The citation-honesty audit on the 200-episode calibration set tells us whether this gap is driven by genuine revision or by post-hoc rationalization in `ev()`.

**vs. Re-plan-from-scratch agent** — On an episode where a booking or message is executed and then a contradicting observation arrives, the replanning baseline produces a fresh plan that re-derives downstream actions without undoing the stale one, because it has no undo stack tied to cited evidence. Our method instead rolls back in LIFO order with precondition-checked compensations, because the audit trail records which evidence justified each action, so we expect RCAS to show higher recoverability on the reversible subset and explicitly flagged divergence on the IRREVERSIBLE subset rather than silent failure.

### What would falsify this idea

If RCAS's advantage over the post-hoc-scoring baseline is uniform across episodes with and without mid-task contradictions, or if its gain over re-plan-from-scratch appears equally on reversible and IRREVERSIBLE subsets, then the predicted causal mechanism — revision changing the legal action set — is not what produces the effect and the central claim is wrong. If the citation-honesty rate on the calibration set is low enough that the revision-responsiveness ratio tracks citation honesty rather than revision, the effectiveness argument is not supported.

## References

1. EgoUseBench: Benchmarking Computer-Use Agents under Evolving First-Person Contexts
