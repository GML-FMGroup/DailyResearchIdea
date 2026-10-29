# Failure-Closed Action Substitution for Causal Embodied Planning

## Motivation

CausGround and similar experience-distilled planners model actions only through preconditions and effects observed in successful interactions, so when a precondition fails at execution the action set offers no guaranteed alternative whose causal profile is consistent with the current grounded state; the planner must either abandon the subgoal or retry, because substitutability was never a structural property of the action model. This is the shared residual gap across experience-derived planners, MLLM planners, and training-based planners: none defines a closure of the action set under precondition failure, so adaptive planning is brittle exactly where the world violates distilled success statistics.

## Key Insight

If the action set is closed under a substitution rule that requires each substitute to share the failed action's effect signature and to have preconditions that are either satisfied or recursively substitutable in the current grounded state, then every precondition failure has a causally consistent repair path by construction rather than by search luck.

## Method

**What it is.** F-CAS (Failure-Closed Action Substitution) is a planning-time closure operator layered on CausGround's distilled action model. Its input is the grounded causal state (object attributes and inter-object relations from the current observation), the distilled action set with preconditions/effects, and a currently failing action; its output is either a substitute action whose grounded preconditions hold, or a substitution chain terminating in such an action, or a certified closure failure that prunes the subgoal.

**Load-bearing assumption (stated explicitly).** F-CAS assumes the distilled effect model is complete and exact over the attributes and relations it represents, so that `eff(b) == eff(a)` is a sound certificate of true causal substitutability and the forward effect check `eff(x) satisfies pre(y)` is a sound certificate of chain consistency. This assumption is not verified by the closure rule itself; it is calibrated offline (Step 0b) and stress-tested in the experiment via side-effect injection. When the assumption fails (omitted side effect, over-specified precondition, non-additive interaction), the certificate degrades to surface effect equality and local rather than global consistency, exactly as described in 'Why it measures what we claim'.

**How it works.**

```pseudocode
# Inputs: grounded state s, distilled action set A (each a -> (pre(a), eff(a)))
#         failing action a0 with pre(a0) not subset of s
#         substitution relation Sub built offline (see step 0)
#         effect-completeness flag complete(a) in {True, False} per distilled effect (see step 0b)
# Output: executable substitute chain or CLOSURE_FAIL

# Step 0 (offline, once per action set): build Sub
for each pair (a, b) in A x A:
    if eff(b) == eff(a) and pre(b) != pre(a):
        Sub[a].add(b)                      # same effect signature, different preconditions
    if eff(b) == eff(a) and pre(b) subset pre(a):
        mark b as a strict substitute of a

# Step 0b (offline calibration, once per action set): set complete(a)
#   For each action a, sample N_CAL=50 grounded states from the training distribution.
#   Execute a in the ground-truth simulator; compare the observed post-state delta
#   against eff(a) over the represented attributes/relations.
#   complete(a) = True iff the observed delta equals eff(a) on all N_CAL samples
#   (no unmodeled side effect on any represented attribute/relation); else False.
#   Sub is restricted to candidates b with complete(b) == True; actions with
#   complete(a) == False are still executable but are never certified as substitutes.

# Step 1 (online): ground preconditions of a0 in s
def close(a, s, depth=0, visited=set()):
    if depth > D_MAX (=3): return CLOSURE_FAIL
    if a in visited: return CLOSURE_FAIL
    visited.add(a)
    unmet = pre(a) \ s
    if unmet is empty: return [a]
    for b in Sub[a]:                       # candidates sharing a's effect signature
        if pre(b) subset s: return [b]     # direct substitute
    for b in Sub[a]:
        chain = close(b, s, depth+1, visited)
        if chain != CLOSURE_FAIL: return chain + [a]   # substitute then original
    return CLOSURE_FAIL

# Step 2: verify causal consistency of the returned chain
for each adjacent (x, y) in chain:
    require eff(x) satisfies pre(y) \ s   # forward effect check
    require eff(y) == eff(a0)             # effect preservation along chain
return chain
```

Sub is a relation, not a lookup table: it is defined by the effect-equality / precondition-difference rule, so it is closed under composition (if b substitutes a and c substitutes b, c substitutes a whenever effect signatures match). D_MAX bounds the recursion so closure is decidable at planning time. The `complete(a)` flag from Step 0b gates certification: a substitute is only returned if every action on the chain has `complete == True`, so the effect-signature certificate is only issued where the completeness assumption was empirically checked.

**Why this design.** We chose effect-signature equality over embedding similarity for building Sub because the planner needs a causal guarantee (the substitute must achieve the same grounded effect), accepting the cost that actions with no effect-equal partner are not substitutable and must be handled by subgoal repair. We chose recursive precondition closure with a depth bound over a single-hop substitution table because a substitute's own preconditions may themselves fail, and a one-hop table would reproduce CausGround's brittleness one level down; the cost is a branching factor bounded by |Sub[a]|^D_MAX, which we keep tractable by D_MAX=3 and by pre-filtering Sub by satisfied-precondition count. We chose a certified CLOSURE_FAIL return over silent fallback to the original action because a planner that knows closure failed can prune the subgoal and replan at a higher level, whereas a silent retry hides the inconsistency; the cost is that some subgoals are declared unreachable earlier than a search-based planner would. We chose offline Sub construction over learning a substitution policy because the closure rule must hold for any grounded state, and a learned policy would only approximate it; the cost is that Sub is static and cannot exploit state-specific action equivalences that were never distilled. We chose the Step 0b completeness calibration over trusting the distilled effect model outright because the causal guarantee is only as sound as the effect model it reads; the cost is N_CAL=50 simulator rollouts per action at build time, which we accept to make the certificate's scope explicit. This differs from Toolformer/Gorilla-style tool retrieval: those select the nearest tool by descriptor, whereas F-CAS requires the selected action to preserve the failed action's effect signature and to discharge the failed precondition through a compositional chain — a property retrieval cannot certify.

**Why this measures what we claim.** The effect-signature equality check `eff(b) == eff(a)` measures causal substitutability because we assume the distilled effect model is complete over the attributes and relations it represents, making equal effect signatures equivalent to achieving the same grounded post-state; this assumption fails when the distilled model omits an effect (e.g., a side effect on an unmodeled attribute), in which case the check reflects surface effect equality rather than true causal equivalence. The recursive precondition closure `close(b, s, depth+1)` measures failure-repair reachability because we assume each distilled precondition is a necessary condition for execution, so discharging unmet preconditions recursively is equivalent to making the action executable; this assumption fails when a precondition is over-specified (a sufficient but unnecessary condition), in which case the closure prunes an action that would in fact have executed. The forward effect check `eff(x) satisfies pre(y) \ s` measures chain causal consistency because we assume the factorized world-state representation composes effects additively across steps, so satisfying the next action's preconditions from the previous action's effects is equivalent to a causally consistent plan; this assumption fails under non-additive interactions (e.g., an effect that removes a relation another action's precondition depends on), in which case the check reflects local rather than global consistency. The `complete(a)` flag from Step 0b measures whether the effect model is complete for action a because we assume N_CAL=50 sampled executions are representative of the action's effect distribution; this assumption fails when the action has rare or state-dependent side effects not exercised by the 50 samples, in which case the flag reflects sampled rather than universal completeness.

## Contribution

['A failure-closure operator over distilled action models that defines substitutability by effect-signature preservation and recursive precondition discharge, turning precondition failure from a planning dead-end into a decidable repair search with a certified CLOSURE_FAIL outcome.', "A design principle: effect-signature equality, not descriptor similarity, is the correct substitutability criterion for causal planners, because it is the only criterion that preserves the grounded post-state the planner's backward reasoning depends on.", "A planning-time algorithm (F-CAS) that composes with CausGround's grounded backward/forward reasoning without retraining, and a characterization of when closure is decidable within a bounded substitution depth."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Adapted VirtualHome + RoboCas | Causally grounded, matches CausGround's distilled action model |
| Primary metric | Subgoal success rate under injected precondition failures | Directly detects closure-repair vs. silent-fail divergence |
| Secondary metric | Effect-model omission rate vs. closure accuracy (calibration curve) | Tests the load-bearing completeness assumption directly |
| Baseline | CausGround (distilled planner, no closure) | Isolates whether closure adds value over base distilled model |
| Baseline | MLLM planner (descriptor retrieval) | Tests retrieval-by-similarity against causal effect-preservation |
| Baseline | ReAct-style replanner (search fallback) | Tests certified pruning against unconstrained search |
| Baseline | Classical goal-regression planner (means-ends analysis) | Isolates closure from regression-based repair |
| Ablation-of-ours | F-CAS with D_MAX=1 (single-hop only) | Tests whether recursive closure drives the predicted gain |
| Ablation-of-ours | F-CAS without Step 0b completeness gate (certify all actions) | Tests whether the completeness calibration is load-bearing |
| Ablation-of-ours | F-CAS with side-effect injection at rates {0%, 5%, 10%, 20%} | Quantifies degradation under the stated failure mode F |

**Injected-failure generator (pre-specified, released as a fixed benchmark split).** For each episode we sample a perturbation from three families at a fixed rate: (i) attribute drift — flip one boolean attribute of one object in the grounded state (e.g., container open→closed) with probability p_drift=0.3 per episode; (ii) relation drift — remove or add one inter-object relation (e.g., `inside(obj, container)`) with p_rel=0.2; (iii) precondition over-specification — mark one satisfied precondition as unmet in the distilled model with p_overspec=0.1. Perturbations are applied only to preconditions of the currently selected action, so the failure is localized and the subgoal is otherwise reachable. The generator is seeded (seed=0) and the split is frozen before any baseline is run.

**Side-effect injection (stress test of the load-bearing assumption).** To quantify failure mode F, we additionally inject unmodeled side effects into the simulator at rates {0%, 5%, 10%, 20%} of executed actions: the simulator removes one relation that the distilled effect model does not represent as a delete effect. This makes `eff(b) == eff(a)` surface-equal but not causally equal, and lets us measure how closure accuracy degrades as a function of effect-model omission rate.

### Why this setup validates the claim

The central claim is that certified effect-preserving action substitution with recursive precondition closure recovers subgoals that a distilled planner would silently fail on, while pruning ones that are genuinely unreachable. CausGround is the right backbone because F-CAS is layered on its distilled action model, so the base baseline tests whether closure adds anything beyond the distilled preconditions/effects themselves. The MLLM planner tests the specific contrast the method draws against Toolformer/Gorilla-style retrieval: if retrieval-by-descriptor were sufficient, it would match us on effect-preserving substitutions, so any gap isolates the value of the causal guarantee. The ReAct-style replanner tests the pruning sub-claim: an unconstrained search should eventually find some substitute but at the cost of exploring closures that F-CAS certifies as failed, so the metric must separate recovered subgoals from wasted exploration. The classical goal-regression planner is the sharpest novelty control: if F-CAS were reducible to means-ends analysis, regression would match us on the injected-failure subset; any gap isolates the closure operator from regression-based repair. The D_MAX=1 ablation is the sharpest test of the recursive mechanism: if the predicted mechanism is recursive closure (not just one-hop substitution), then the ablation should capture only the shallow substitutions and lose precisely the deep-chain cases, while matching us elsewhere. The no-completeness-gate ablation tests whether Step 0b is load-bearing: if the gate does no work, removing it should not change results on the side-effect-injection subset. The side-effect-injection sweep directly operationalizes the load-bearing assumption: closure accuracy should degrade monotonically with omission rate, and the calibration curve is the empirical test of whether `eff(b) == eff(a)` is a sound certificate. Subgoal success rate under injected precondition failures is the right primary metric because it is measured on exactly the state distribution where closure matters; a metric like overall task success would dilute the signal across unaffected subgoals. This combination is falsifiable: it predicts a specific localization of gain, not a uniform uplift.

### Expected outcome and causal chain

**vs. CausGround (distilled planner, no closure)** — On a case where a grounded precondition is unmet because an object attribute drifted (e.g., a container is closed when the distilled action assumed it open), CausGround returns the failing action with no repair path and the subgoal dies silently, because its distilled model has no mechanism to reach an effect-equal action with different preconditions. Our method instead walks Sub to a substitute sharing the failed action's effect signature whose preconditions hold, or recursively discharges the unmet precondition, because the closure operator is defined exactly over effect-equality plus precondition difference. We expect a large gap concentrated on the injected-failure subset and near-parity on clean states, since clean states never invoke closure.

**vs. MLLM planner (descriptor retrieval)** — On a case where the failing action's descriptor is closest to a tool that shares surface wording but not effect signature (e.g., "open" retrieved for a latch vs. a lid), the MLLM planner substitutes the wrong tool and produces a post-state that does not satisfy the downstream subgoal, because retrieval matches descriptors rather than grounded effects. Our method instead rejects that candidate at the effect-equality check and only returns a substitute whose `eff(b) == eff(a)` holds, because Sub is built by the effect-signature rule rather than by similarity. We expect a noticeable gap on the ambiguous-descriptor subset and parity on unambiguous ones, with the gap widening as descriptor space gets more crowded.

**vs. ReAct-style replanner (search fallback)** — On a case where the subgoal is genuinely unreachable given the grounded state (no effect-equal action has satisfiable preconditions), the replanner explores a branching search and either times out or returns a spurious plan that violates a precondition on re-execution, because it has no certificate of failure and cannot distinguish exhausted closure from unexplored search. Our method instead returns CLOSURE_FAIL at bounded depth and prunes the subgoal for higher-level replanning, because the depth-bounded recursion is decidable and the failure is certified. We expect the replanner to match us on recoverable subgoals but to show wasted steps and spurious-plan rate on the unreachable subset, where we show clean pruning.

**vs. Classical goal-regression planner (means-ends analysis)** — On a case where the failing action's effect signature is shared by a substitute whose own preconditions are unmet, regression back-chains from the subgoal and may select a different action entirely, because regression is not constrained to preserve the failed action's effect signature. Our method instead restricts candidates to Sub[a0] and recurses only within that relation, because the closure rule is defined over effect-equality. We expect regression to match us on simple one-hop cases but to diverge on cases where the effect-preserving substitute is not the regression-preferred action, isolating the closure operator from regression-based repair.

**vs. F-CAS with D_MAX=1 (ablation)** — On a case where the only viable substitute itself has an unmet precondition that needs a second-level substitute, the single-hop ablation returns CLOSURE_FAIL because its Sub is queried once and cannot recurse, reproducing CausGround's brittleness one level down. Our full method instead composes the chain `[c, b, a0]` because the closure recurses with the depth bound, and the forward effect check confirms each link. We expect the ablation to match us on direct-substitute cases and to lose precisely the deep-chain cases, so the gain should be localized to instances requiring depth ≥2 rather than spread uniformly.

**vs. F-CAS without Step 0b completeness gate** — On a case where an action has an unmodeled side effect (injected at 10% rate), the ungated variant certifies a substitute whose effect signature is surface-equal but not causally equal, and the downstream subgoal fails on re-execution, because the gate that would have excluded that action is absent. Our method instead excludes actions with `complete(a) == False` from Sub, because Step 0b calibrated completeness against the simulator. We expect the gate to matter only on the side-effect-injection subset and to be neutral on the clean and attribute-drift subsets, localizing its contribution to failure mode F.

**vs. F-CAS under side-effect-injection sweep (0%, 5%, 10%, 20%)** — As the omission rate rises, the fraction of actions with `complete(a) == False` rises, Sub shrinks, and closure accuracy should degrade monotonically, because the certificate is only as sound as the effect model it reads. We expect the calibration curve to show closure accuracy near the no-injection level at 0% and to fall toward the CausGround level as omission rate approaches the fraction of actions whose effects are unmodeled, quantifying the load-bearing assumption empirically rather than asserting it.

### What would falsify this idea

If the gain over CausGround were uniform across clean and injected-failure subsets rather than concentrated where precondition failures occur, or if the D_MAX=1 ablation matched full F-CAS on deep-chain instances, the claim that recursive effect-preserving closure is the causal driver would be wrong. A further falsifier: if the MLLM retrieval baseline matched us on the ambiguous-descriptor subset, the effect-signature guarantee would not be doing the work we attribute to it; and if closure accuracy did not degrade with the side-effect-injection rate, the completeness assumption would not be load-bearing in the way claimed.

## References

1. CausGround: Causal Interaction Grounding for Adaptive Embodied Planning
