# Obligation-Closed Playtest Editing for LLM Game Generation

## Motivation

Existing agentic game-generation loops treat playtest feedback as free-form natural language: Play2Evolve diagnoses functionality, visual, and gameplay issues and then lets coding agents propose arbitrary edits, with no guarantee that a proposed edit actually addresses the diagnosed defect or that the defect is fully covered. Unbounded demonstrates the same structural gap from the other branch — its LLM generates mechanics and narratives in real time but has no playtest channel at all, so there is no diagnosis-to-edit closure to audit. The recurring failure is that the mapping from observed defect to code change is an open-ended generation act rather than a constrained, checkable transformation, which makes recursive refinement unauditable and prone to silent non-fixes.

## Key Insight

If every diagnosed defect is compiled into a typed obligation with a declared, checkable satisfaction predicate and the edit space is restricted to operations whose precondition/postcondition signatures are known, then closure becomes a decidable coverage check over a finite obligation set rather than a judgment about generated text.

## Method

**What it is.** OBLIGE (OBLigation-Gated Infinite-game Editing) is a wrapper around Play2Evolve's Refinement/Evolution stages. Inputs: the current game artifact (code + asset manifest), a playtest trace, and the previous obligation ledger. Outputs: an accepted edit patch, an updated obligation ledger with per-obligation discharge status, and a rejection/repair record for any edit that fails closure.

**Load-bearing assumption (stated explicitly).** OBLIGE assumes that every playtest-diagnosed defect can be faithfully compiled by an LLM into a predicate drawn from the fixed library (FUNC/VIS/PLAY) such that satisfying that predicate is both necessary and sufficient for the defect being fixed, so that closure over the finite obligation set is a valid proxy for true diagnosis-to-edit closure. This assumption is load-bearing and is not guaranteed: the library is fixed and small, whereas the defect space of open-ended LLM-generated games is unbounded and largely semantic (e.g., 'the game is not fun', 'the mechanic is registered but degenerate', 'the goal is reachable but trivial'). To make the assumption checkable rather than silent, every obligation carries a paired adequacy witness (Phase 1b) and the primary reported quantity is the gap between predicate-satisfied and probe-confirmed closure (Phase 3b).

**How it works.**

```pseudocode
# Phase 1: diagnosis -> typed obligations
D = playtest_diagnose(game, trace)          # Play2Evolve-style, free-form text
O = compile_obligations(D)                  # LLM emits JSON, schema-validated
#   each o in O: {id, kind in {FUNC,VIS,PLAY}, target_selector,
#                 predicate P_o, evidence_span, severity,
#                 adequacy_witness W_o}
#   predicate P_o is chosen from a fixed predicate library:
#     FUNC: exit_code==0 | unit_test(name) | invariant(expr)
#     VIS : asset_present(id) | palette_dist(region, ref) < tau_vis (tau_vis=0.15)
#     PLAY: reachable(goal) | mechanic_registered(name) | progress_span >= k
#         (k=3 distinct progress events in a scripted 60-second rollout)

# Phase 1b: adequacy witness (paired with every obligation)
#   W_o is a held-out behavioral probe, not a predicate in the library:
#     FUNC: differential test between pre-edit and post-edit traces on the
#           defect-relevant code path (same seed, same input script)
#     VIS : scripted screenshot diff of the defect region against a reference
#           palette, reporting mean per-pixel L2 in CIELAB
#     PLAY: scripted playtest rollout (fixed seed, 60s) recording the
#           defect-relevant behavioral signal (e.g., progression counter,
#           mechanic invocation count)
#   W_o must change in the defect-relevant direction (pre-specified sign and
#   minimum effect size delta_o; default delta_o = 0.10 normalized units).

# Phase 2: edit-space restriction
E = declare_edit_space(O)                   # LLM emits candidate edit ops
#   each e in E: {op in {ADD_FUNC, PATCH_FUNC, ADD_ASSET, REGISTER_MECHANIC,
#                        TUNE_PARAM}, preconditions, postconditions}
#   postconditions are drawn from the SAME predicate library as P_o

# Phase 3: closure verification (the gate)
for each proposal batch B (LLM agent edits, <=1 op per obligation):
    for o in O:
        if not exists e in B with P_o in e.postconditions:
            reject(o, reason="uncovered")          # completeness
    for e in B:
        if not preconditions_hold(e, game):
            reject(e, reason="unsound_precondition")
        if not postcondition_implies(e, obligations(e)):
            reject(e, reason="unsound_postcondition")  # soundness
    B = repair_loop(B, O, max_rounds=3)   # agent re-proposes only rejected slots
    if closure(B, O): apply(B); update_ledger(O)

# Phase 3b: adequacy-witness confirmation (runs after apply)
for o in O with applied edit e:
    if not witness_confirms(W_o, game_before, game_after, delta_o):
        mark o as predicate_satisfied_probe_unconfirmed
#   closure fraction is reported decomposed:
#     predicate_closure = |{o: P_o satisfied}| / |O|
#     probe_confirmed_closure = |{o: P_o satisfied AND W_o confirmed}| / |O|
#     closure_gap = predicate_closure - probe_confirmed_closure

# Phase 4: Evolution gating
new_goals = propose_goals(ledger)            # only after all FUNC obligations discharged
O <- O + compile_obligations(new_goals)      # new obligations enter the same gate
```

**Checker specifications (postcondition_implies and preconditions_hold).** For each of the five op types, the checkers are concrete and mechanical:
- ADD_FUNC: preconditions_hold checks that the target module exists and the new function name is not already defined (AST lookup); postcondition_implies checks that the added function's body contains the predicate's referenced symbol (e.g., the unit test name or invariant expression) and that the function is reachable from the game's entry point via static call-graph analysis.
- PATCH_FUNC: preconditions_hold checks that the patched function exists and the diff applies cleanly; postcondition_implies checks the postcondition by executing the patched function's unit test or invariant expression in a sandbox and confirming exit_code==0 or the invariant holds on the post-edit artifact.
- ADD_ASSET: preconditions_hold checks that the asset id is absent from the manifest; postcondition_implies checks asset_present(id) by manifest lookup and palette_dist(region, ref) < tau_vis by computing the CIELAB palette distance on the rendered asset.
- REGISTER_MECHANIC: preconditions_hold checks that the mechanic name is not already registered; postcondition_implies checks mechanic_registered(name) by registry lookup and, for PLAY obligations, that the scripted rollout invokes the mechanic at least once.
- TUNE_PARAM: preconditions_hold checks that the parameter exists and the new value is within the declared range; postcondition_implies checks the postcondition by re-running the affected unit test or invariant expression and confirming the predicate holds.

**Why this design.** We chose a fixed predicate library over free-form LLM-written assertions because the gate must be mechanically checkable against the artifact, accepting the cost that some subtle aesthetic defects cannot be expressed and must be dropped or escalated to a human — a deliberate trade of expressiveness for auditability. We chose a declared edit space with explicit pre/postconditions over unconstrained code generation because soundness (an edit's postcondition actually implying the obligation predicate) can only be checked when the edit's effect surface is declared, accepting that the agent must spend tokens on a declaration step and that some legitimate edits fall outside the five op types. We chose a repair loop bounded at 3 rounds over open-ended re-generation because unbounded retry masks whether the edit space itself is inadequate, accepting that after 3 rounds we must surface an `edit_space_insufficient` signal rather than silently succeed. We added the paired adequacy witness and the decomposed closure report because the fixed library alone cannot establish necessity/sufficiency for semantic defects; the witness is a held-out behavioral probe that must move in the defect-relevant direction, so the gate is validated against an artifact-level signal rather than the predicate alone. A domain expert would not describe OBLIGE as a variant of Play2Evolve or of Reflexion-style self-refinement: those methods let the agent both diagnose and decide when it is done, whereas here the termination condition is a coverage check over a finite obligation set computed outside the agent, and the agent's freedom is restricted to a declared edit algebra.

**Why it measures what we claim.** The predicate closure fraction measures diagnosis-to-edit closure because we assume that every playtest-diagnosed defect is faithfully compiled into a predicate in the library; this assumption fails when a defect is real but inexpressible in the library, in which case predicate closure reflects library coverage rather than true closure. The probe-confirmed closure fraction measures true diagnosis-to-edit closure because we assume the adequacy witness W_o changes in the defect-relevant direction if and only if the defect is actually fixed; this assumption fails when the witness is itself gameable (e.g., a no-op mechanic that increments the progression counter), in which case probe-confirmed closure reflects witness coverage rather than true closure. The closure gap (predicate minus probe-confirmed) measures the specification-gaming margin because we assume the gap is zero when the library predicate is necessary and sufficient; this assumption fails when the predicate is weaker than the intended property, in which case the gap reflects the degree of specification gaming. The soundness rejection rate measures edit-space soundness because we assume `postcondition_implies` is a faithful check of the declared postcondition against the obligation predicate; this assumption fails when the artifact contains side effects outside the declared effect surface, in which case the rate reflects declaration honesty rather than actual soundness. The `edit_space_insufficient` count measures the adequacy of the declared edit algebra because we assume the repair loop exhausts the space before declaring insufficiency; this assumption fails when the LLM fails to find a valid edit that exists, in which case the count reflects agent search failure rather than space inadequacy.

## Contribution

(1) OBLIGE, a diagnosis-to-edit contract layer comprising a typed defect schema over a fixed predicate library, a declared edit-space algebra with pre/postconditions, and a closure verifier that gates each Refinement/Evolution step on soundness and completeness. (2) A design principle: restricting the LLM agent's edit surface to declared operations with checkable postconditions converts recursive game refinement from an open-ended generation act into an auditable transformation with a finite termination condition. (3) An obligation ledger artifact that records per-defect discharge status, enabling post-hoc audit of which playtest diagnoses were actually closed versus dropped.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | GameCraft-Bench playtest traces (all 20 traces used in the month-3 prototype; full benchmark for final evaluation) | Same benchmark as Play2Evolve baselines |
| Primary metric | Probe-confirmed closure fraction (obligations with predicate satisfied AND adequacy witness confirmed) | Directly measures diagnosis-to-edit closure, not library coverage |
| Secondary metric | Closure gap = predicate closure fraction − probe-confirmed closure fraction | Quantifies specification-gaming margin |
| Baseline | Play2Evolve (Refinement/Evolution) | Tests whether gating beats open-ended self-refinement |
| Baseline | Game Initialization (no iteration) | Tests whether any iteration helps at all |
| Baseline | Play2Code | Tests whether feedback loop alone suffices |
| Baseline | Free-form-assertion gate (LLM-written assertions mechanically executed, no fixed library) | Isolates whether the gain comes from the fixed typed library or merely from any external check |
| Ablation-of-ours | OBLIGE w/o closure gate (free-form edits) | Isolates the gate from the edit-space declaration |
| Ablation-of-ours | OBLIGE w/o adequacy witness (predicate closure only) | Isolates the witness from the predicate library |
| Human audit | 100 diagnosed defects annotated for library expressibility | Validates the faithfulness of compile_obligations |
| Checker audit | 50 hand-labeled edits for false-accept/false-reject rate of postcondition_implies and preconditions_hold | Validates the soundness checkers |

### Why this setup validates the claim

The central claim is that restricting the agent's edit freedom to a declared, mechanically checkable edit algebra — and gating acceptance on obligation closure — produces more reliable diagnosis-to-edit closure than letting the agent both diagnose and decide when it is done. GameCraft-Bench playtest traces are the right substrate because they expose the exact failure modes the gate targets: runnable-but-unplayable games, missing assets, and shallow gameplay. Game Initialization tests the trivial sub-claim that iteration is needed at all; Play2Code tests whether a feedback loop without a closure check is sufficient; Play2Evolve tests whether the strongest existing self-refinement loop already captures the gain. The free-form-assertion gate tests whether the gain comes from the fixed typed library or merely from any external check, which is the specific novelty claim over generic contract-based generation. Probe-confirmed closure fraction is the right primary metric because it is computed outside the agent and requires an artifact-level behavioral signal, so it cannot be inflated by the agent's own judgment of completion — which is precisely the mechanism gap the baselines share. The closure gap is the right secondary metric because it directly measures the load-bearing assumption: if the fixed library is necessary and sufficient, the gap is zero; if the library is weaker than the intended property, the gap is positive and concentrated on the semantic subsets. The ablation without the closure gate isolates the gate's contribution from the declaration step's contribution; the ablation without the adequacy witness isolates the witness's contribution from the predicate library's contribution. The human audit of 100 diagnosed defects reports the fraction expressible in the library, making the faithfulness of compile_obligations an empirical quantity rather than an assumption. The checker audit of 50 hand-labeled edits reports false-accept/false-reject rates for postcondition_implies and preconditions_hold, making the soundness checkers auditable.

### Expected outcome and causal chain

**vs. Play2Evolve** — On a case where the playtest trace shows a character that can be fed but never progresses, the baseline produces a plausible-looking narrative patch that the agent judges sufficient because it has no external termination condition; its self-refinement loop stops when the agent's own diagnosis text sounds resolved. Our method instead rejects any batch lacking a `progress_span >= k` postcondition, forcing the agent to register a progression mechanic before acceptance, and the adequacy witness requires the scripted rollout to show at least k=3 distinct progress events. We expect a noticeable gap on the progression-and-mechanics subset of GameCraft-Bench, with rough parity on pure bug-fix subsets where free-form edits already suffice.

**vs. Play2Code** — On a case where a visual defect (off-palette assets) is diagnosed, the baseline applies a code edit that fixes the symptom the agent narrates but leaves the asset manifest inconsistent, because its feedback is free-form text with no predicate tied to the artifact. Our method requires an `asset_present` or `palette_dist < tau_vis` postcondition, so the patch is rejected as unsound until the manifest is repaired, and the adequacy witness requires the scripted screenshot diff to move in the defect-relevant direction. We expect the largest relative gain on visual-consistency and asset-integrity subsets, and a modest gain elsewhere.

**vs. Game Initialization** — On a case where the initial game is runnable but has no exploration or progression, the baseline produces that game and stops, because there is no refinement stage at all. Our method's Evolution gating only opens after all FUNC obligations are discharged, then compiles new PLAY obligations into the same gate, so exploration goals are added under closure pressure rather than ad hoc. We expect a clear gap on gameplay-richness subsets, with the gap concentrated where initialized games lack registered mechanics.

**vs. Free-form-assertion gate** — On a case where the defect is semantic (e.g., 'the mechanic is registered but degenerate'), the free-form-assertion gate can write an assertion that is mechanically executed but not tied to the fixed library, so it may capture the defect more expressively but with no declared effect surface for soundness checking. Our method's fixed library plus adequacy witness should show a smaller closure gap on the semantic subsets if the witness is doing the work, or a larger gap if the fixed library is the binding constraint. We expect the comparison to isolate whether the gain comes from the typed library or from any external check.

**vs. OBLIGE w/o closure gate** — On a case where the playtest trace shows a missing asset, the no-gate ablation applies a free-form edit that the agent judges sufficient, with no postcondition check. Our method rejects the edit as unsound until the manifest is repaired. We expect the no-gate ablation to match OBLIGE only on pure bug-fix subsets, and to diverge on visual-integrity and progression subsets.

**vs. OBLIGE w/o adequacy witness** — On a case where the defect is semantic (e.g., 'the goal is reachable but trivial'), the no-witness ablation accepts any edit satisfying `reachable(goal)` even if the goal remains trivial, so predicate closure is high but probe-confirmed closure is low. Our method's adequacy witness requires the scripted rollout to show a defect-relevant behavioral change, so the closure gap is reported explicitly. We expect the no-witness ablation to show a larger closure gap on the semantic subsets, confirming that the witness is doing the work the fixed library cannot.

### What would falsify this idea

If OBLIGE's gain over Play2Evolve is uniform across all subsets rather than concentrated on the progression, visual-integrity, and gameplay-richness subsets where the predicted mechanism gap occurs — or if the no-gate ablation matches OBLIGE, or if the closure gap is near zero on the semantic subsets (meaning the fixed library is already necessary and sufficient) — then the closure gate and the adequacy witness are not doing the causal work the claim assigns to them.

## References

1. Playfulness as Feedback for Agentic Game Development
