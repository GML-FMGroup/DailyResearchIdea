# Saga Algebra: Recursive Compensation Contracts for Nested Tool-Call Recovery

## Motivation

Agent recovery methods treat an execution's effect history as a flat sequence, so when a partial execution spans nested tool calls the adjudicator can only reason about top-level compensations and cannot lift a sub-saga's local recovery guarantee into the enclosing transaction. Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution makes this concrete: its effect-history adjudicator validates key expiry, authorization timing, dependencies, and residual effects over a single flattened history, which means a compensation that is valid for an inner tool call is never checked against the outer call's invariants, and a globally safe completion may be certified while an inner sub-saga remains partially applied with no admissible undo. The structural root cause is that compensation semantics are defined per-saga rather than being closed under nesting, so no composition rule exists to propagate a local guarantee upward.

## Key Insight

Compensation contracts are closed under nesting when each sub-saga exports a guard-conditioned residual-effect summary, so a parent saga's recovery guarantee is exactly the conjunction of its children's summaries plus its own frame condition, making upward guarantee propagation a well-defined algebraic operation rather than a heuristic flattening.

## Method

**What it is.** Saga Algebra (SA) is a recovery layer for an agent's tool-call executor. Its input is the runtime trace of nested tool calls (a tree of sagas, each saga a transaction with pre/post conditions and a compensation per effectful step) plus the observed outcomes; its output is, for every saga node, a *recovery contract* — a guard-conditioned residual-effect summary — and a single top-level verdict: `certified_complete`, `certified_recoverable` (with a concrete compensation schedule), or `unrecoverable` (with the innermost violating saga named).

**Load-bearing assumption (stated explicitly).** SA's composition operator `⊗` is sound and complete because a parent saga's recovery guarantee is exactly the conjunction of its children's guard-conditioned residual summaries plus its own frame condition — i.e., the exported summary `(guard, residual)` is a sufficient statistic for the child's recoverability, so no information needed to adjudicate the parent is lost by summarizing the child as `(guard, residual)`. This assumption is load-bearing: if it fails, `⊗` can certify a parent as recoverable when the composed schedule is infeasible. The method below carries the smallest repair that makes the summary an actual sufficient statistic for schedule feasibility: `residual` is an ordered poset of pending effects (not a set), each contract carries a `contention` multiset of exclusive resources, `CONFLICTS` checks contention across siblings, and `admissible` is a topological feasibility check over the composed poset. The completeness claim is therefore bounded to the stated assumptions on `discharged`, `ENTAILS`, and `CONFLICTS`; outside them SA is a sound-but-incomplete over-approximator.

**How it works.**

```pseudocode
# Runtime trace node: saga S = (steps, children, frame_cond, pre, post)
# Each effectful step s has compensation comp(s), guard g(s) (key expiry, auth window, dependency),
# and consumes a multiset of exclusive resources res(s) (e.g., single-use auth token, lock id).
# residual is a POSET (partial order) over pending effects, not a set.
# contention is a multiset of exclusive resources each pending compensation consumes.

function CONTRACT(S, evidence):
    child_contracts = [CONTRACT(c, evidence) for c in S.children]   # bottom-up
    # 1. Local residual: effects of S.steps not yet discharged by compensation, as a poset
    local_residual = POSET({s.effect for s in S.steps if not discharged(s, evidence)},
                           order = S.step_order)
    # 2. Compose children: conjunction of guards, union of residuals as poset union,
    #    multiset union of contention
    child_guard    = AND(c.guard for c in child_contracts)
    child_residual = POSET_UNION(c.residual for c in child_contracts)
    child_contention = MULTISET_UNION(c.contention for c in child_contracts)
    # 3. Frame condition: parent post must hold modulo child residuals
    frame_ok = ENTAILS(S.post, child_residual, S.frame_cond)
    # 4. Admissibility of each compensation against ALL enclosing guards and sibling contention
    admissible = [s for s in S.steps
                  if g(s) holds under evidence AND child_guard
                  AND not CONFLICTS(comp(s), child_residual, child_contention)]
    # 5. Topological feasibility over the composed poset
    schedule = TOPO_SORT(child_residual ∪ local_residual, admissible)
    feasible = schedule is not None
    guard = AND(S.pre, child_guard, frame_ok, feasible)
    residual = (local_residual ∪ child_residual) \ UNION(comp(s).effect for s in admissible)
    contention = child_contention ∪ MULTISET(res(s) for s in admissible)
    return Contract(guard, residual, contention, admissible, frame_ok, feasible)

function ADJUDICATE(root, evidence):
    C = CONTRACT(root, evidence)
    if C.residual == {} and C.guard:  return certified_complete
    if C.residual != {} and C.guard:  return certified_recoverable(schedule=topological(C.admissible))
    return unrecoverable(innermost_violating(C))
```

The composition operator `⊗` is `CONTRACT`'s steps 2–5: `C(S) = C_local(S) ⊗ (⊗_{c∈children} C(c))`, where `⊗` conjoins guards, unions residuals as posets, unions contention multisets, checks the parent frame condition against the union, and runs a topological feasibility check over the composed poset. The adjudicator is a single bottom-up pass; no flattening occurs. Guards are evaluated against the same public evidence bundle the executor already records (key timestamps, auth tokens, dependency tokens).

**Why this design.** We chose a bottom-up contract pass over the flat effect-history adjudicator used in Evidence-Guided Recovery because a flat history cannot express that an inner compensation is only valid while an outer guard still holds, accepting the cost that SA must maintain a saga tree at runtime rather than a single list. We chose a guard-conditioned residual summary as the exported interface over replaying each child's full trace at the parent, because replay is exponential in nesting depth while a summary is linear, accepting the cost that the summary must be conservative (it may over-approximate residuals and trigger a recoverable verdict where replay would certify complete). We chose to fold the parent frame condition into the guard (`frame_ok`) rather than checking it only at the root, because a child that violates the parent's frame condition must invalidate the parent's guarantee locally, accepting the cost that frame conditions must be written per saga rather than once globally. We chose to carry an ordered, resource-annotated residual (poset + contention multiset) rather than a set-union residual, because two children whose residuals are individually compensable but jointly non-compensable (e.g., both require the same single-use auth token, or one's compensation invalidates the other's precondition) compose to a residual that looks recoverable under set union; the poset and contention fields make `⊗` compose schedules rather than sets, accepting the cost that the summary is larger (linear in pending effects plus contended resources). The closest named prior method is SagaLLM's per-saga compensation: a domain expert would not describe SA as a variant of it because SagaLLM assumes a flat action set and never defines a composition operator that lifts a child's guarantee into the parent — SA's `⊗` is the load-bearing object, and it is what makes nested recovery checkable.

**Why it measures what we claim.** `local_residual` measures *unrecovered effect* because the assumption that `discharged(s, evidence)` is sound (a step is discharged iff its compensation effect is observed in evidence) makes the poset difference equal to the true residual; this assumption fails when a provider reports a compensation as applied but the effect is deferred, in which case `local_residual` under-counts and the verdict is optimistic. `child_guard` measures *local recovery guarantee* because the assumption that guards are conjunctive and mutually satisfiable (a child is recoverable iff all its guards hold) makes the conjunction equal to the composed guarantee; this fails when two guards are mutually exclusive (e.g., a key expires exactly when a dependency token is issued), in which case the conjunction is unsatisfiable and the verdict is pessimistic. `frame_ok` measures *compositional safety* because the assumption that the parent post-condition is expressible as an entailment over child residuals plus the frame condition makes the entailment check equivalent to global safety; this fails when the post-condition references a variable no child exports, in which case `frame_ok` reflects only the exported projection. `admissible` measures *executable recovery* because the assumption that `CONFLICTS(comp(s), child_residual, child_contention)` captures all interference (including sibling contention over exclusive resources) makes the filtered set equal to the compensations that can be run without breaking a child; this fails when a compensation has a side effect not modeled in the residual or contention, in which case the schedule is feasible on paper but unsafe in execution. `feasible` measures *schedule feasibility* because the assumption that the composed poset plus contention multiset captures all ordering and resource constraints makes the topological check equivalent to the existence of a valid compensation order; this fails when a compensation's precondition depends on a resource not listed in `contention`, in which case `feasible` is optimistic.

## Contribution

(1) A recursive compensation-contract algebra for nested tool-call sagas, comprising a composition operator that conjoins child guards, unions residual effects, and checks the parent frame condition, yielding a bottom-up adjudicator that never flattens the execution tree. (2) A design principle — guard-conditioned residual summaries as the exported interface between saga levels — that makes recovery guarantees composable at cost of conservatism, with the conservatism localized to the summary rather than the whole verdict. (3) An open adjudicator implementation and saga-tree trace format that any observation-contingent planner can emit, enabling recovery checks over nested executions that flat effect histories cannot express.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Nested-saga tool-call traces with recorded evidence bundles, generated by a pre-registered procedure (see below) with maximum nesting depth 4 and maximum 12 effectful steps per saga | Exercises composition and guard expiry directly; bounded for reproducible oracle |
| Primary metric | Verdict correctness vs. ground-truth recoverability computed by exhaustive replay oracle (bounded to depth 4, 12 steps/saga) | Directly tests the certification claim |
| Baseline | Evidence-Guided Recovery (flat effect-history adjudicator) | Tests whether nesting composition is load-bearing |
| Baseline | SagaLLM per-saga compensation | Tests whether `⊗` beats flat per-saga compensation |
| Baseline | Naive replay of child traces at parent (same bounded oracle, used as cost reference) | Tests whether summaries avoid exponential cost |
| Ablation-of-ours | SA without `frame_ok` fold (root-only frame check) | Isolates the compositional-safety contribution |
| Ablation-of-ours | SA with set-union residual (no poset, no contention) | Isolates the schedule-feasibility contribution of the ordered, resource-annotated summary |
| Trace class | Mutually-exclusive-guard traces (key expiry coinciding with dependency issuance) | Measures pessimism rate from unsatisfiable guard conjunction |

**Pre-registered trace-generation procedure.** Traces are generated by a script with fixed seed 42. Each trace is a tree of sagas with depth uniformly in {1,2,3,4} and 1–12 effectful steps per saga. Each effectful step is assigned a compensation and a guard drawn from three families: key-expiry (timestamp window), auth-window (token validity interval), and dependency-token (issued only after a named predecessor). Guard expiry is injected by setting the evidence bundle's timestamp past the guard's expiry for a uniformly chosen 30% of steps in the nested-guard-expiry class. Frame violations are injected by making a child's post-condition contradict the parent's frame condition for a uniformly chosen 20% of child sagas in the child-frame-violation class. Mutually-exclusive guards are injected by scheduling a key expiry at the same evidence timestamp as a dependency-token issuance for a uniformly chosen 15% of steps in the mutually-exclusive-guard class. The exhaustive-replay oracle enumerates all compensation orders over the bounded trace (depth ≤ 4, ≤ 12 steps/saga) and returns the ground-truth verdict; traces exceeding the bound are discarded and regenerated. The generation script and oracle are released with the paper.

### Why this setup validates the claim

The central claim is that a bottom-up, guard-conditioned residual contract with a real composition operator certifies nested recovery more accurately and more cheaply than flat adjudication. The dataset must therefore contain nesting and guard expiry, otherwise the claim is untestable; traces with recorded evidence bundles let us compute ground-truth recoverability by exhaustive replay offline, bounded to depth 4 and 12 steps/saga so the oracle is reproducible within the 3–6 month timeline. The primary metric — agreement between the method's verdict and that ground truth — is the right detector because the method's entire output is a verdict plus a schedule, so any mechanism gap shows up as a mis-certification rather than as a diffuse score change. Evidence-Guided Recovery tests the sub-claim that a flat history cannot express inner-compensation-valid-only-under-outer-guard; SagaLLM tests whether `⊗` adds anything over per-saga compensation; naive replay is the cost reference that establishes the summary is not merely an accuracy trade. The ablation (dropping the `frame_ok` fold) is the sharpest test: if the compositional-safety argument is correct, removing the local frame check should produce over-optimistic `certified_complete` verdicts precisely on traces where a child violates a parent frame condition, and nowhere else. The set-union-residual ablation tests whether the ordered, resource-annotated summary is load-bearing: if the completeness claim is correct, dropping the poset and contention fields should produce over-optimistic `certified_recoverable` verdicts precisely on traces where two children contend for the same exclusive resource. The mutually-exclusive-guard trace class measures the pessimism rate from unsatisfiable guard conjunction, bounding the conservativeness claim.

### Expected outcome and causal chain

**vs. Evidence-Guided Recovery (flat effect-history adjudicator)** — On a case where an inner saga's compensation is only valid while an outer key guard still holds, the baseline produces `certified_complete` because its flat history has no representation of the enclosing guard and sees the compensation as applied. Our method instead conjoins `child_guard` up the tree, so the expired outer guard propagates and the verdict becomes `certified_recoverable` or `unrecoverable`. We expect a noticeable accuracy gap concentrated on nested-guard-expiry traces and near-parity on flat, single-level traces.

**vs. SagaLLM per-saga compensation** — On a case where a child saga's post-condition is violated but the child's own compensations all ran, the baseline produces a per-saga "recovered" verdict because it never lifts the child's guarantee into the parent. Our method instead runs `⊗` and the `frame_ok` entailment, so the parent's guarantee is locally invalidated and the residual surfaces at the parent. We expect the gap to concentrate on traces where child-local recovery succeeds but parent post-conditions fail, with parity elsewhere.

**vs. Naive replay of child traces at parent** — On a deeply nested trace, the baseline produces the same verdicts as ours but at cost exponential in nesting depth, because it re-expands each child's full trace at every parent. Our method instead composes linear-size summaries, so we expect verdict parity with replay on shallow traces and a widening cost gap — with occasional over-approximation (recoverable where replay certifies complete) — that grows with nesting depth.

**vs. SA without `frame_ok` fold (root-only frame check)** — On a trace where a child violates a parent frame condition, the ablation produces `certified_complete` because the frame check is deferred to the root and the child's violation is not locally visible. Our method instead folds `frame_ok` into the guard at each saga, so the parent's guarantee is locally invalidated and the verdict becomes `certified_recoverable` or `unrecoverable`. We expect the ablation to change verdicts only on frame-violating traces, with parity elsewhere.

**vs. SA with set-union residual (no poset, no contention)** — On a trace where two children's residuals are individually compensable but jointly non-compensable (e.g., both require the same single-use auth token), the ablation produces `certified_recoverable` because set union loses ordering and resource contention. Our method instead carries the poset and contention multiset, so `⊗` detects the infeasibility and the verdict becomes `unrecoverable` or a different schedule. We expect the ablation to change verdicts only on contention traces, with parity elsewhere.

**Mutually-exclusive-guard trace class** — On a trace where a key expires exactly when a dependency token is issued, the guard conjunction is unsatisfiable and our method yields a pessimistic verdict (`unrecoverable` where replay certifies recoverable). We expect the pessimism rate to be bounded and reported per trace class, quantifying the conservativeness of the conjunctive guard.

### What would falsify this idea

If our accuracy gain over the flat baselines is uniform across nested and flat traces rather than concentrated on traces where outer guards expire or child residuals violate parent frame conditions, the composition operator is not doing the work we claim. Likewise, if the `frame_ok` ablation changes nothing on frame-violating traces, or the set-union-residual ablation changes nothing on contention traces, the corresponding component is not load-bearing and the central compositional-safety argument is wrong.

## References

1. Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution
