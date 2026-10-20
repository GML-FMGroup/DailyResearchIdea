# Irreversibility-Aware Plan Validation via Side-Effect Completion Graphs

## Motivation

Interlock and BUTTON both assume a resettable synthetic environment: Interlock compiles typed completion graphs into a resettable workspace and BUTTON collects trajectories in a simulated multi-agent sandbox. This shared resettable-state assumption prevents any guarantee of execution validity in real, non-resettable environments where actions like sending emails, deleting records, or making payments cannot be undone. The structural root cause is that completion graphs encode only preconditions and effects of successful execution, not the irreversible side effects that persist even when a plan is abandoned or fails mid-way.

## Key Insight

Plan validity in non-resettable environments is a monotonicity property over a partial order of irreversible effects: a plan is valid iff every irreversible effect is either explicitly authorized by the task specification or is a necessary precondition for a later authorized effect, and this can be checked by a fixpoint computation on the completion graph without ever executing the plan.

## Method

**What it is.** We introduce SIREN (Side-effect IRreversibility-aware Execution semaNtics), a framework that augments Interlock's typed completion graph with two annotation layers: (1) irreversible-effect nodes and (2) pre/post-condition pairs that encode which state transitions are non-resettable. SIREN takes as input a completion graph G and a task specification T (a set of authorized irreversible effects), and outputs a three-valued verdict (valid / invalid / indeterminate) plus a minimal set of violated constraints. It never executes the plan; validation is a static three-valued (must/may) reachability analysis over the annotated graph.

**Load-bearing assumption (stated explicitly).** SIREN's verdict is sound and complete for authorization validity only under the closed-world assumption that the graph's pre/post annotations are complete: every literal that any plan consistent with G can make true is derivable from the graph, and every literal not in a post_e is unchanged. This assumption is exactly what resettability in Interlock's workspace repairs at execution time; SIREN cannot rely on that repair, so it makes the assumption explicit and, when it is violated, abstains (indeterminate) rather than returning a false verdict.

**How it works.** We annotate each realization edge e in G with a triple (pre_e, post_e, irr_e), where pre_e and post_e are sets of state literals and irr_e is a set of irreversible effect literals. Grounding edges and coupling hyperedges are left as in Interlock. We first apply a well-formedness check: reject any graph where some post_e contains the negation of a literal in any irr_e (annotation not closed under negation). We then compute a three-valued closure over the graph, tracking for each literal whether it is must-true (in S_must), may-true (in S_may), or absent:

```pseudocode
Input: completion graph G=(V,E), task spec T=(A, R) where A = authorized irr effects, R = required final state
Output: verdict in {valid, invalid, indeterminate}, violations (list)

0. Well-formedness: if exists e, l with l in irr_e and neg(l) in post_e for some e, return (indeterminate, ["annotation not closed under negation"])
1. Initialize S_must = {}, S_may = {}  # empty sets of literals
2. For each grounding edge g in G:
     S_must = S_must ∪ post_g
     S_may  = S_may  ∪ post_g
3. Repeat until fixpoint:
     For each realization edge e in G:
       if pre_e ⊆ S_must and post_e ∩ S_must = ∅:
         S_must = S_must ∪ post_e
         S_may  = S_may  ∪ post_e
         if irr_e ≠ ∅:
           record (e, irr_e) in applied_irr_must
       elif pre_e ⊆ S_may and post_e ∩ S_may = ∅:
         S_may = S_may ∪ post_e
         if irr_e ≠ ∅:
           record (e, irr_e) in applied_irr_may
     For each coupling hyperedge h in G:
       if all preconditions of h ⊆ S_must:
         S_must = S_must ∪ post_h
         S_may  = S_may  ∪ post_h
       elif all preconditions of h ⊆ S_may:
         S_may = S_may ∪ post_h
4. If R ⊈ S_must:
     if R ⊆ S_may: return (indeterminate, ["required state only may-reached"])
     else: return (invalid, ["required state not reached"])
5. For each (e, irr_e) in applied_irr_must:
     if irr_e ⊈ A:
       # backward reachability: does irr_e enable any authorized effect via a chain in applied_irr_must?
       if not backward_reachable(irr_e, A, applied_irr_must):
         violations.append((e, irr_e))
6. If any (e, irr_e) in applied_irr_may but not in applied_irr_must and irr_e ⊈ A:
     return (indeterminate, violations + ["unauthorized may-irreversible effect"])
7. Return (valid if len(violations)==0 else invalid, violations)

backward_reachable(irr_e, A, applied_irr_must):
  # BFS over applied_irr_must edges: start from any edge e' with irr_e' ⊆ A,
  # traverse backwards along preconditions that are irr literals of other applied edges.
  frontier = {e' in applied_irr_must : irr_e' ⊆ A}
  visited = set(frontier)
  while frontier not empty:
    e' = pop(frontier)
    for each l in pre_e':
      if l in irr_e for some e'' in applied_irr_must and e'' not in visited:
        if irr_e'' == {l}: return True  # irr_e is a prerequisite of an authorized effect
        visited.add(e''); frontier.add(e'')
  return False
```

Hyperparameters: none; the three-valued fixpoint is deterministic. The necessity check in step 5 is a backward reachability query over applied_irr_must from authorized effects, replacing the previous one-step lookahead; this captures multi-hop prerequisite chains where an authorized effect's prerequisites are themselves irreversible but not directly in any authorized edge's precondition.

**Why this design.** We chose a static three-valued reachability analysis over graph annotations rather than dynamic execution-time monitoring because the resettable-state assumption in Interlock is precisely what makes execution-time monitoring possible, and we cannot rely on it. The trade-off is that SIREN cannot validate plans whose preconditions depend on runtime values not present in the graph; we accept this cost because the graph is the only artifact available before execution, and we make the cost explicit by returning indeterminate when the closure only may-reaches the required state. We chose a backward reachability necessity check rather than a one-step lookahead because irreversible effects are by definition not reorderable after the fact, so any effect not justified by a chain of authorized downstream effects is a violation regardless of ordering; the backward query over applied_irr_must captures exactly those chains. We chose to leave coupling hyperedges unchanged rather than annotate them with irreversibility because hyperedges encode joint constraints that are checked atomically; adding irreversibility to them would require a separate partial-order analysis that we defer. We chose a three-valued verdict rather than a graded score because the task specification defines a binary authorization relation, and a graded score would require a cost model for irreversible effects that the task does not provide; the third value (indeterminate) is not a grade but an abstention when the closed-world assumption is violated.

**Why this measures what we claim.** The three-valued closure computation in step 3 measures *execution validity under irreversible effects* because the must/may fixpoint over pre/post conditions is equivalent to the set of states reachable by any plan consistent with the graph, under the assumption that the graph's pre/post annotations are complete; this assumption fails when a realization edge has a precondition that is satisfied only by an unmodeled environmental side effect, in which case the closure under-approximates reachability and SIREN returns indeterminate rather than a false verdict. The backward reachability check in step 5 measures *authorization necessity* because an irreversible effect is necessary iff it appears in a backward chain from an authorized downstream effect in applied_irr_must, under the assumption that the task specification's authorized set A is closed under prerequisite; this assumption fails when a task authorizes an effect without listing its prerequisites, in which case SIREN flags a false violation. The applied_irr_must set measures *irreversible side effects* because every realization edge with non-empty irr_e that fires in the must-closure contributes its effect permanently, under the assumption that irr_e literals are never removed from S_must; this assumption fails when a later edge's post_e explicitly negates an irr_e literal, in which case the well-formedness check in step 0 rejects the graph as indeterminate, and the annotation was incorrect.

## Contribution

["SIREN, an environment-agnostic execution validity framework that augments Interlock's typed completion graph with irreversible-effect annotations and a static fixpoint validator that checks authorization and necessity of irreversible side effects without executing the plan.", 'A design principle for non-resettable plan validation: irreversible effects must be justified by downstream authorized effects, formalized as a one-step necessity check over the applied irreversible set, which distinguishes SIREN from execution-time monitoring approaches that require resettable state.', 'A set of annotated completion graphs for non-resettable task domains (email, file deletion, payment) with ground-truth validity labels, enabling evaluation of plan validators without environment resets.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Interlock completion-graph task suite | Provides typed graphs with realization edges to annotate |
| Dataset | Real non-resettable tool-use benchmark (email/calendar/file-system) with graphs constructed from actual tool schemas | Tests transfer beyond synthetic resettable suite; reports agreement with human authorization judgments |
| Primary metric | Violation-detection F1 vs. ground-truth authorization, computed over valid/invalid verdicts only (indeterminate excluded from F1, reported separately as abstention rate) | Directly measures the binary verdict claim while accounting for abstention |
| Secondary metric | Abstention rate (fraction of plans returned indeterminate) | Quantifies the cost of the closed-world assumption |
| Baseline | Interlock (resettable-state validator) | Tests whether irreversibility annotations add signal |
| Baseline | Unguarded closure (no irr annotations) | Tests whether the irr layer is doing the work |
| Baseline | One-step lookahead without closure | Tests whether the fixpoint closure matters |
| Baseline | PDDL3 state-trajectory-constraint planner (Gerevini & Long) with authorization encoded as constraints | Tests whether classical constraint formalisms can express authorization-necessity without full task-spec encoding |
| Ablation-of-ours | SIREN with necessity check removed | Isolates the authorization-necessity sub-claim |
| Ablation-of-ours | SIREN with one-step necessity check (original) | Isolates the backward-reachability contribution |

### Why this setup validates the claim

The central claim is that SIREN's static three-valued fixpoint over pre/post/irr annotations yields a correct binary authorization verdict where resettable-state assumptions break, and abstains when the closed-world assumption is violated. Interlock as a baseline tests the sub-claim that irreversibility annotations add discriminating signal beyond typed completion graphs: if Interlock's verdicts already match ground truth on irreversible-effect tasks, SIREN's annotation layer is unmotivated. The unguarded-closure baseline tests whether the irr layer specifically (rather than the closure itself) produces the verdict, isolating the annotation contribution. The no-closure one-step baseline tests whether the fixpoint in step 3 is doing real work versus a shallow scan, addressing our "why this design" argument that the closure captures realized dependencies. The PDDL3 baseline tests whether classical state-trajectory-constraint formalisms can express the authorization-necessity property without encoding the full task spec as constraints; if PDDL3 matches SIREN, the contribution is a lightweight validator rather than a new framework. The ablation removing the necessity check isolates the step-5 sub-claim: if removing it does not change verdicts, the necessity logic is inert. The ablation with one-step necessity isolates the backward-reachability contribution: if one-step matches backward reachability, the multi-hop chain case is not exercised. Violation-detection F1 is the right metric because the method's output is a boolean plus minimal violations, and the task spec defines a binary authorization relation; a graded score would require a cost model the task does not supply. The abstention rate is reported separately because indeterminate is not a grade but a signal that the closed-world assumption failed. Together these form a falsifiable grid: each baseline/ablation targets one sub-claim, so a uniform gain would be uninformative while a subset-concentrated gain confirms the causal story.

### Expected outcome and causal chain

**vs. Interlock (resettable-state validator)** — On a case where a plan reaches the required state but does so by firing an unauthorized irreversible effect (e.g., deleting a record to satisfy a downstream precondition), Interlock produces a valid verdict because its resettable-state assumption lets it treat the deletion as undoable and thus outside the authorization check. Our method instead records the effect in applied_irr_must during the closure and flags it in step 5 because its irr_e literal is not backward-reachable from any authorized effect. We expect a noticeable F1 gap concentrated on the irreversible-effect subset, with parity on purely resettable tasks where Interlock's assumption holds.

**vs. Unguarded closure (no irr annotations)** — On a case where the closure reaches R only through an edge whose irr_e is empty in the annotation but whose post_e permanently removes a literal, the unguarded baseline produces a valid verdict because it never tracks permanence and treats every post as ordinary state. Our method instead fires the edge only when post_e ∩ S_must = ∅ and records the irreversibility, so it flags the unauthorized permanent removal. We expect a gap concentrated on tasks with negating post_e edges, and parity where all effects are additive.

**vs. One-step lookahead without closure** — On a case where an authorized irreversible effect is enabled only through a two-hop chain of realization edges, the one-step baseline produces a false violation because it checks necessity against the immediate precondition set rather than the closure-reached state. Our method instead computes the fixpoint first, so the downstream authorized effect appears in applied_irr_must and the backward reachability query justifies the upstream irreversible effect. We expect a gap concentrated on multi-hop dependency tasks, with parity on single-hop tasks.

**vs. PDDL3 state-trajectory-constraint planner** — On a case where the task specification authorizes an irreversible effect without listing its prerequisites, the PDDL3 baseline must encode the full task spec as constraints to express authorization-necessity, which either over-constrains (rejecting valid plans) or under-constrains (accepting invalid plans) depending on the encoding. Our method instead uses the authorized set A directly and the backward reachability query over applied_irr_must, so it flags only effects not reachable from A. We expect a gap concentrated on tasks where A is not closed under prerequisites, with parity where A is closed.

**vs. SIREN with necessity check removed** — On a case where an unauthorized irreversible effect is fired but the required state is reached, the ablation produces a valid verdict because it never checks authorization. Our method flags the effect in step 5. We expect a large gap on all tasks with unauthorized irreversible effects, confirming the necessity check is load-bearing.

**vs. SIREN with one-step necessity check** — On a case where an authorized irreversible effect's prerequisite is itself irreversible but not directly in any authorized edge's precondition (multi-hop chain), the one-step ablation produces a false violation because it checks only immediate preconditions. Our method's backward reachability query traverses the chain and justifies the upstream effect. We expect a gap concentrated on multi-hop prerequisite tasks, with parity on single-hop tasks.

### What would falsify this idea

If SIREN's violation-detection gain over Interlock is uniform across resettable and irreversible-effect subsets rather than concentrated on the irreversible subset, the annotation layer is not measuring what we claim and the central mechanism is wrong. Likewise, if removing the necessity check or the closure leaves verdicts unchanged, or if the abstention rate on the real non-resettable benchmark is so high that F1 is computed on a small minority of plans, the corresponding sub-claim is falsified.

## References

1. Interlock: From Local Correctness to Task Completion in Long-Horizon Agents
2. Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning
3. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
4. TaskBench: Benchmarking Large Language Models for Task Automation
