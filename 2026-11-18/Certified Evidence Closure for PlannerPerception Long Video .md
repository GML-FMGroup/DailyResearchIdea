# Certified Evidence Closure for Planner–Perception Long Video Agents

## Motivation

Existing long-video agent loops — DrVideo's iterative document augmentation and VideoNexus's planner-driven perception sub-agents — both terminate on an unformalized heuristic of 'sufficient evidence gathered,' inherited from the assumption that a question-driven search loop naturally accumulates what it needs. Structurally they fail because the planner has no representation of what remains unexamined or unresolved: VideoNexus's planner decides where to inspect next but carries no state over the temporal extent of the stream, so it cannot distinguish 'I have resolved this hypothesis' from 'I have not looked there yet,' and repeated or redundant inspections are indistinguishable from coverage. The gap is a missing completeness invariant over the evidence space, shared across both branches, that no single branch's retrieval or agentic machinery supplies.

## Key Insight

If the planner maintains a question-conditioned partition of the video timeline into hypothesis cells whose status is monotone (unexamined → resolved/out-of-scope) and only terminates on an empty unresolved set, then 'sufficient evidence' becomes a checkable closure condition rather than a planner intuition — because termination is decoupled from the planner's self-assessment and grounded in the evidence ledger the perception sub-agents actually populate.

## Method

**What it is.** CEC (Certified Evidence Closure) wraps VideoNexus's planner–perception loop with an explicit evidence ledger: a question-conditioned partition of the native audio-video timeline into temporal hypothesis cells, each carrying a status in {UNEXAMINED, RESOLVED, OUT_OF_SCOPE_PROVISIONAL, OUT_OF_SCOPE_CONFIRMED}. Inputs are the question Q and the native audio-video stream; outputs are a final answer plus a closure certificate listing every cell's terminal status and the sub-agent evidence that discharged it. The planner's inspection directives are chosen to discharge specific UNEXAMINED cells; termination requires the ledger to have no UNEXAMINED cells and no OUT_OF_SCOPE_PROVISIONAL cells.

**How it works.**

```pseudocode
# Phase 1: Hypothesis cell construction (planner, one call)
input: question Q, stream metadata (duration T, scene-change timestamps)
cells = planner.decompose(Q, T)   # returns list of (t_start, t_end, hypothesis_text, hypothesis_type)
# hypothesis_type in {SINGLE_CELL, CROSS_CELL, GLOBAL}
# cells partition [0,T]; granularity = scene-change boundaries, merged until |cells| <= 32
ledger = {c: UNEXAMINED for c in cells}

# Phase 2: Certified inspection loop
while any(status == UNEXAMINED for status in ledger) or any(status == OUT_OF_SCOPE_PROVISIONAL for status in ledger):
    target = planner.select_unexamined(ledger, Q)   # lowest-index UNEXAMINED cell; if none, lowest-index OUT_OF_SCOPE_PROVISIONAL cell
    density = planner.choose_density(target, Q)     # 1 of {sparse, dense, audio-focused}
    evidence = subagent.inspect(target.t_start, target.t_end, density, Q)
    # subagent returns compact temporally-grounded evidence + a verdict field
    if target.hypothesis_type == SINGLE_CELL:
        ledger[target] = classify(evidence, Q)
        # classify: RESOLVED if evidence bears on hypothesis_text and answers Q-part;
        #           OUT_OF_SCOPE_PROVISIONAL if evidence shows hypothesis_text is not answerable from
        #           this span (e.g., speaker absent, event never occurs);
        #           else UNEXAMINED with refined sub-cells (split target in half, |cells| cap 64)
        if ledger[target] == UNEXAMINED: ledger = split_and_reinsert(ledger, target)
    else:  # CROSS_CELL or GLOBAL
        # subagent returns evidence for the cell's local span; the planner must integrate across cells
        # before any status can be assigned. We store the evidence and mark the cell as PENDING_INTEGRATION.
        ledger[target] = PENDING_INTEGRATION
        # After all cells of a relational hypothesis have PENDING_INTEGRATION evidence, the planner
        # performs a Reduce step: synthesize an answer for the relational hypothesis from the collected
        # evidence and assign RESOLVED or OUT_OF_SCOPE_PROVISIONAL to each participating cell.
        if all(c in ledger for c in target.related_cells) and all(ledger[c] == PENDING_INTEGRATION for c in target.related_cells):
            relational_evidence = {c: evidence[c] for c in target.related_cells}
            verdict = planner.reduce(relational_evidence, target.hypothesis_text, Q)
            for c in target.related_cells:
                ledger[c] = verdict  # RESOLVED or OUT_OF_SCOPE_PROVISIONAL

    # Confirmation step for OUT_OF_SCOPE_PROVISIONAL: require a second, independently sampled inspection
    if ledger[target] == OUT_OF_SCOPE_PROVISIONAL:
        density2 = planner.choose_density(target, Q, independent=True)  # forced different from density
        evidence2 = subagent.inspect(target.t_start, target.t_end, density2, Q)
        if evidence2 confirms absence (same specific absence reported):
            ledger[target] = OUT_OF_SCOPE_CONFIRMED
        else:
            ledger[target] = UNEXAMINED  # revert and split if needed
            ledger = split_and_reinsert(ledger, target)

# Phase 3: Certified answer
answer = planner.synthesize(Q, {c: evidence[c] for c in ledger if ledger[c] == RESOLVED})
certificate = {c: (status, evidence_id) for c in ledger}
return answer, certificate
```

**Why this design.** We chose a *discretized partition with monotone cell statuses* over a free-form planner scratchpad (as in VideoNexus) because a partition makes 'what remains unexamined' a set-membership query rather than a planner judgment, accepting the cost that coarse cells can force unnecessary splits when a hypothesis straddles a boundary. We chose *scene-change-anchored cells capped at 32* over uniform fixed-length windows because scene boundaries align cell edges with semantic events, accepting that scene detection can miss gradual transitions and leave a cell spanning two topics. We chose *sub-agent-returned verdicts* (RESOLVED / OUT_OF_SCOPE_PROVISIONAL / UNEXAMINED) over planner-assigned statuses because the sub-agent has actually seen the native audio-video span and the planner has not, accepting that a sub-agent can mislabel a cell and thereby close the loop prematurely. We chose *lowest-index-first selection among UNEXAMINED cells* over planner-scored prioritization because it makes the loop's progress monotone and auditable, accepting reduced efficiency when a late cell is decisive. We chose *hypothesis typing (SINGLE_CELL, CROSS_CELL, GLOBAL)* because the load-bearing assumption that a sub-agent inspecting a single temporal cell can correctly and locally decide whether that cell discharges its hypothesis (RESOLVED) or is provably irrelevant (OUT_OF_SCOPE) holds only for SINGLE_CELL hypotheses; for CROSS_CELL and GLOBAL hypotheses, local verdicts do not compose into question-level resolution, so we require a planner Reduce step over the participating cells' evidence before any status is assigned. We chose *OUT_OF_SCOPE_PROVISIONAL* as a non-terminal status requiring a second independent inspection because absence detection from sparse sampling is false-negative-prone; only after confirmation does a cell become OUT_OF_SCOPE_CONFIRMED and count toward closure. A domain expert would not describe CEC as a variant of VideoNexus or DrVideo: those systems' loops are driven by planner heuristics with no ledger, whereas CEC's termination is a set-emptiness condition over a state object the sub-agents populate — the planner's role shrinks from 'decide when to stop' to 'pick the next unexamined cell,' which is a different decision surface.

**Why it measures what we claim.** The ledger's UNEXAMINED count measures *evidence incompleteness* because the partition is question-conditioned and the sub-agent verdict is grounded in native audio-video inspection of that exact span; this assumption (that a sub-agent inspecting a span can correctly classify whether that span discharges its hypothesis) fails when a hypothesis requires cross-cell integration (e.g., 'does the protagonist return to the same room?'), in which case the verdict reflects local-span relevance rather than question-level resolution and the certificate overstates closure. The OUT_OF_SCOPE_CONFIRMED status measures *provable irrelevance* because the sub-agent reports the specific absence (speaker not present, event not occurring) that makes the hypothesis unanswerable from the span, and this report is confirmed by a second independent inspection; this assumption fails when both inspections miss the same evidence due to correlated sampling bias, in which case OUT_OF_SCOPE_CONFIRMED reflects a persistent sampling miss rather than genuine irrelevance. The split-on-UNEXAMINED operation measures *hypothesis refinement* because halving a cell whose evidence is ambiguous localizes the unresolved sub-question; this assumption fails when the ambiguity is inherent to the question wording rather than the temporal span, in which case splitting produces ever-finer cells without progress and the |cells| cap 64 becomes the binding constraint.

## Contribution

['A certified evidence-closure mechanism (CEC) that replaces heuristic agent-loop termination with a checkable set-emptiness condition over a question-conditioned temporal hypothesis ledger, emitting a per-cell closure certificate alongside the answer.', "A design principle: decoupling the planner's 'where to look next' decision from the 'when to stop' decision by grounding termination in a sub-agent-populated state object rather than planner self-assessment.", "An analysis of when sub-agent verdicts are valid discharge certificates and the cross-cell-integration and sparse-sampling failure modes that break the certificate's soundness."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LVBench (full split) | Long, evidence-sparse; stresses closure and irrelevance detection |
| Dataset | MLVU (multi-task long video benchmark) | Provides cross-span and global-context questions to stress the CROSS_CELL and GLOBAL hypothesis types |
| Primary metric | Answer accuracy + closure-certificate soundness (per hypothesis type) | Accuracy alone cannot distinguish certified from lucky closure; per-type soundness exposes the local-decidability assumption |
| Baseline | VideoNexus (same backbone) | Tests whether ledger closure beats planner-judged termination |
| Baseline | DrVideo (retrieval-document loop) | Tests whether partition closure beats retrieval-driven stopping |
| Baseline | Static ingestion (same backbone) | Tests whether any agentic closure helps at all |
| Ablation-of-ours | CEC without OUT_OF_SCOPE status | Isolates the contribution of provable-irrelevance closure |
| Ablation-of-ours | CEC without hypothesis typing (all cells treated as SINGLE_CELL) | Isolates the contribution of the Reduce step for cross-cell/global hypotheses |

### Why this setup validates the claim

The central claim is that replacing a planner-judged stopping heuristic with a set-emptiness condition over a sub-agent-populated evidence ledger yields more complete evidence coverage and therefore higher answer accuracy on long-video questions where decisive evidence is sparse. LVBench is the right stress test because its questions require locating specific temporal evidence inside long native streams, so a system that stops early — or that never proves irrelevance — will visibly lose accuracy rather than merely spend more tokens. MLVU adds a second benchmark with explicit cross-span and global-context questions, which is where the local-decidability assumption is most likely to break; including it lets us measure whether the hypothesis-typing and Reduce-step extensions actually recover accuracy on relational questions rather than only on single-cell ones. VideoNexus is the direct ancestor: it shares the backbone and the planner–perception decomposition, so any accuracy gap isolates the ledger-and-closure mechanism rather than backbone capacity. DrVideo tests a competing closure philosophy (retrieval saturation instead of partition exhaustion), and static ingestion establishes that the effect is not just "more agentic calls." The ablation without OUT_OF_SCOPE forces cells that can never be resolved to remain UNEXAMINED, so if provable irrelevance is doing real work, the ablation should stall or over-split and lose accuracy specifically on questions whose answers require ruling spans out. The ablation without hypothesis typing collapses all cells to SINGLE_CELL, so if the Reduce step is doing real work, this ablation should lose accuracy specifically on CROSS_CELL and GLOBAL questions. Accuracy is the right primary metric because the claim is about answer correctness; the certificate-soundness check (auditing RESOLVED, OUT_OF_SCOPE_CONFIRMED, and UNEXAMINED-at-termination cells against human-verified evidence) is what makes the accuracy gain attributable to closure rather than to more inspection budget.

### Expected outcome and causal chain

**vs. VideoNexus (same backbone)** — On a case where the decisive evidence sits in a late, low-salience segment and early segments are uninformative, VideoNexus's planner stops once its scratchpad looks sufficient, producing a confident wrong answer because nothing forces it to account for unexamined spans. CEC instead cannot terminate while any cell is UNEXAMINED, so the loop is driven into that late segment, and we expect a noticeable accuracy gap concentrated on questions with late-lying evidence and rough parity on questions whose evidence is front-loaded.

**vs. DrVideo (retrieval-document loop)** — On a case where the question requires proving that something does *not* occur ("does the speaker ever mention X?"), DrVideo's retrieval loop keeps returning weakly related frames and either saturates prematurely or exhausts budget without a definitive negative, because it has no representation of provable irrelevance. CEC's sub-agent can mark spans OUT_OF_SCOPE_PROVISIONAL and confirm them with a second independent inspection, closing the loop cleanly, so we expect a gap concentrated on negative/absence-style questions and near-parity on straightforward locate-and-answer questions.

**vs. Static ingestion (same backbone)** — On a case where relevant evidence occupies a small fraction of a long stream, static ingestion dilutes it across the context and answers from the dominant but irrelevant content, because it has no mechanism to direct attention. CEC's cell-targeted inspection spends density only where the ledger is open, so we expect a uniform-but-smaller gain over static ingestion, with the largest margins on the longest videos and the smallest margins on short videos where dilution is mild.

### What would falsify this idea

If CEC's accuracy gain over VideoNexus is uniform across question types rather than concentrated on late-evidence and absence questions, or if the no-OUT_OF_SCOPE ablation matches full CEC, then the ledger's closure condition is not what produces the improvement and the central claim is wrong. A second falsifier: if certificate-soundness auditing shows RESOLVED cells frequently lack the claimed discharging evidence, or if OUT_OF_SCOPE_CONFIRMED cells are frequently contradicted by the second inspection, the ledger is overstating closure and its termination signal is not measuring evidence completeness.

## References

1. VideoNexus: Video Perception Beyond the Reasoning Context
