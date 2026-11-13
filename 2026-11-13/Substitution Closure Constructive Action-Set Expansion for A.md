# Substitution Closure: Constructive Action-Set Expansion for Agent Failure Recovery

## Motivation

Existing recovery planners assume a pre-enumerated admissible action set and enforce static admissibility predicates, so when ambiguous tool outcomes and partial execution invalidate the original actions, the planner can be left with no admissible action at all. Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution exemplifies this: its shared admissibility check enforces key expiry, authorization timing, dependencies, and residual effects but never establishes closure of the action set under failure substitution, so recovery is only as good as the enumerated set. The same residual assumption recurs across the shielding and SagaLLM branches, making action-set closure a forest-level structural gap rather than a single-tree limitation.

## Key Insight

Failure recovery is a closure problem over the lattice of effect-substitution obligations, not a filtering problem over a fixed action set: every infeasibility certificate emitted by the admissibility check names a residual obligation whose substitution generators can be composed, and the least fixed point of that composition is guaranteed non-empty whenever the goal remains reachable at all.

## Method

**What it is.** SCRATCH (Substitution Closure Recovery) takes as input the current failure context `(state, residual obligations, ambiguity set, admissibility certificate)` and outputs an expanded admissible action set plus a recovery plan over it, computed as the least fixed point of a substitution-closure operator. It replaces the static admissibility filter with a constructive operator `Cl` that both filters and generates.

**Load-bearing assumption (stated explicitly).** SCRATCH assumes that the obligation-indexed substitution generator `G(o, s, A)` is a *sufficient statistic for substitutability*: that obligation class plus public evidence uniquely determines the set of admissible substitute actions, so that the least fixed point of `Cl` is non-empty whenever the goal remains reachable. This is a frame/qualification assumption on the authored generator library: a generator whose postcondition retroactively invalidates an earlier action will be missed unless it is declared as a compensating generator. Closure completeness is therefore assumed by construction, not derived. The method makes this assumption checkable rather than hidden: the generator library is a partial, open-ended set `G`, and when the fixed point is empty or the frontier is non-empty at depth `D`, the loop emits a verifiable `INFEASIBLE` certificate and then searches over an LLM-proposed generator that is admitted only after passing the same admissibility re-check `adm(.)`. The contribution is thus *closure with a checkable generator-admission rule*, not a guarantee that closure is complete.

**How it works.**

```pseudocode
inputs: state s, goal g, obligation set O, ambiguity set A,
        admissibility predicate adm(.), substitution generator G(.),
        max_depth D=4, max_actions N=64
output: expanded action set Sigma*, plan pi over Sigma*

1. Sigma0 = original enumerated actions
2. Front = {o in O : adm(o, s) = false}          # infeasible obligations
3. Sigma* = Sigma0 ; depth = 0
4. while Front nonempty and depth < D and |Sigma*| < N:
5.     o = pop(Front)                             # one infeasibility certificate
6.     for each substitution sigma in G(o, s, A): # generators keyed by obligation+context
7.         if sigma not in Sigma*:
8.             Sigma* = Sigma* + {sigma}
9.             for each new obligation o' in post(sigma, s):
10.                if adm(o', apply(sigma,s)) = false: push(Front, o')
11.    depth += 1
12. if Front nonempty:
13.     sigma_llm = LLM_propose_generator(o, s, A)   # open-ended fallback
14.     if adm(post(sigma_llm, s), apply(sigma_llm,s)) holds for all new obligations:
15.         Sigma* = Sigma* + {sigma_llm}; goto 9
16.     else: return INFEASIBLE(certificate)
17. pi = observation-contingent_plan(Sigma*, s, g, A)
18. return (Sigma*, pi)
```

Generators `G(o, s, A)` are typed rewrite rules indexed by obligation class (e.g., re-authorize, re-key, re-order dependency, compensate-residual); each carries a precondition over `s` and `A`, so ambiguity narrows which generators fire rather than blocking closure. The loop computes the least fixed point of `Sigma -> Sigma0 ∪ ⋃_{o infeasible in Sigma} G(o)`, terminating at depth `D` or size `N`. The generator library is scoped to the 3-4 obligation classes with highest coverage in the evaluation suite (re-authorize, re-key, re-order dependency, compensate-residual); per-class coverage is reported explicitly.

**Why this design.** We chose a constructive fixed-point operator over a static admissibility filter because a filter can only shrink the action set while failure substitution requires growth, accepting the cost that closure may fail to terminate and needs depth/size caps. We chose obligation-indexed substitution generators over a generic action-repair LLM because generators are auditable rewrite rules whose preconditions are checkable against public evidence, accepting the cost that the generator library must be authored per obligation class. We chose to push only *newly created* infeasible obligations onto the frontier rather than re-checking all of `Sigma*`, because monotone growth makes older obligations' admissibility stable under substitution, accepting the cost that a generator whose postcondition retroactively invalidates an earlier action will be missed unless it is declared as a compensating generator. A domain expert would not describe SCRATCH as a variant of the admissibility-filter approach in the Evidence-Guided Recovery card, because that card's check is a pure predicate over a fixed set, whereas SCRATCH's operator is a generator that enlarges the set as a function of the failure certificate itself — the fixed point, not the predicate, is the object of study.

**Why it measures what we claim.** The frontier size `|Front|` measures *residual infeasibility* because the assumption that every inadmissible obligation emits exactly one certificate makes the frontier a complete encoding of remaining failure; this assumption fails when a generator's postcondition is only partially observable under ambiguity, in which case the frontier under-counts and closure may be declared prematurely. The closure depth `depth` measures *substitution compositionality* because the assumption that generators compose associatively over obligation classes makes depth a faithful count of composed repairs; this assumption fails when two generators interact non-commutatively, in which case depth reflects a particular repair order rather than the minimal composition. The generator set `G(o,s,A)` measures *context-conditional substitutability* because the assumption that obligation class plus public evidence determines the substitutable action set makes the generator a sufficient statistic for substitution; this assumption fails when two distinct states share an obligation class but differ in unobserved residual effects, in which case the generator over-generates and the admissibility re-check is the only guard. To test these assumptions directly, the loop is instrumented to log, per assignment: (i) the number of generators whose postconditions were only partially observable, and (ii) the number of non-commutative generator pairs encountered, defined as pairs `(sigma1, sigma2)` whose application order changes the resulting admissible set `Sigma*`.

## Contribution

(1) A constructive substitution-closure operator SCRATCH that replaces static admissibility filtering with a least-fixed-point expansion of the action set, driven by infeasibility certificates and obligation-indexed rewrite generators. (2) A design principle: recovery planners should be evaluated on closure of the reachable action set under substitution, not on the size or quality of a pre-enumerated set, with a termination certificate that distinguishes genuine infeasibility from depth-cap exhaustion. (3) An open generator library schema for obligation classes (re-authorize, re-key, re-order, compensate-residual) with checkable preconditions over public evidence and ambiguity sets.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Ambiguous-tool-outcome recovery suite (37 assignments, 19 task families) | Matches the failure class SCRATCH targets. |
| Primary metric | Completion rate on assignments with residual infeasibility | Directly measures closure success. |
| Secondary metric | Count of non-commutative generator pairs per assignment (pairs whose application order changes the resulting admissible set) | Tests the associativity assumption behind closure depth. |
| Secondary metric | Count of generators with partially observable postconditions per assignment | Tests the frontier-completeness assumption. |
| Baseline | Evidence prompting | Tests whether generation beats predicate-only guidance. |
| Baseline | Static admissibility filter | Tests whether growth beats shrinking-only filtering. |
| Baseline | Generic action-repair LLM | Tests whether auditable generators beat unconstrained repair. |
| Ablation-of-ours | SCRATCH with generators disabled (filter-only fixed point) | Isolates the contribution of substitution generation. |
| Ablation-of-ours | SCRATCH with LLM-proposed generator fallback disabled | Isolates the contribution of the checkable generator-admission rule. |

### Why this setup validates the claim

The central claim is that a constructive substitution-closure operator recovers from failures that a static admissibility predicate cannot, because the predicate can only shrink an action set while substitution requires enlarging it. The static-filter baseline tests exactly this sub-claim: if SCRATCH beats it, growth is doing work the predicate cannot. Evidence prompting tests a weaker sub-claim — that structured, obligation-indexed generation beats unstructured prompting advice — and the generic action-repair LLM tests whether auditability of generators matters versus unconstrained repair. The generators-disabled ablation is the decisive control: it runs the same fixed-point loop with `G` empty, so any gain over it is attributable to the substitution generators rather than to the loop or the frontier bookkeeping. The LLM-fallback-disabled ablation isolates the checkable generator-admission rule from the rest of the loop. Completion rate on assignments with residual infeasibility is the right metric because those are precisely the cases where the frontier is nonempty and closure must either succeed or emit an INFEASIBLE certificate; on assignments with no residual infeasibility all methods should tie, giving a built-in parity check that separates mechanism from noise. The two secondary metrics operationalize the acknowledged failure modes: non-commutative-pair count tests whether depth reflects minimal composition or repair order, and partially-observable-postcondition count tests whether the frontier under-counts residual infeasibility. We report the correlation between each count and the corresponding error type (premature closure for partial observability; over-generation for non-commutativity).

### Expected outcome and causal chain

**vs. Evidence prompting** — On a case where an assignment's residual obligation is an inadmissible re-authorization under partial observability, evidence prompting produces a plausible-sounding repair suggestion that is not checked against postconditions, so it either repeats the failed action or proposes an action already ruled out. Our method instead fires the obligation-class generator, re-checks the newly created obligations, and composes a concrete plan over the expanded set. We expect a noticeable completion gap concentrated on assignments whose frontier depth exceeds one, with near-parity on single-obligation assignments.

**vs. Static admissibility filter** — On a case where the only admissible action set is empty because every enumerated action violates an obligation, the filter returns infeasible and stops, since it can only shrink `Sigma0`. Our method instead grows `Sigma*` by applying the substitution generator for that obligation class and re-checking only the newly created obligations. We expect the filter to certify only the subset of assignments whose original enumeration already contained a substitute, and SCRATCH to close the remainder, producing a gap that widens with the number of distinct obligation classes per assignment.

**vs. Generic action-repair LLM** — On a case where two obligations interact non-commutatively (re-keying before re-authorizing versus after), the generic repair LLM proposes a single plausible repair without tracking which postconditions it invalidates, so it silently drops a residual obligation and the plan fails at execution. Our method instead pushes each newly infeasible postcondition onto the frontier and composes generators in an order the admissibility re-check validates. We expect SCRATCH to match the LLM on shallow single-step repairs but pull ahead on multi-step compositions, with the gap tracking closure depth and with the non-commutative-pair count predicting where the LLM fails.

### What would falsify this idea

If SCRATCH's gain over the generators-disabled ablation is uniform across all assignments — including those with empty frontiers and no residual infeasibility — rather than concentrated on assignments with deep or multi-class frontiers, then the substitution generators are not doing the causal work and the closure story is wrong. A second falsifier: if closure depth fails to predict which assignments SCRATCH recovers (recovered and unrecovered cases indistinguishable by depth), or if the non-commutative-pair count does not correlate with over-generation errors, then depth is not measuring substitution compositionality as claimed.

## References

1. Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution
