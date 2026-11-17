# Closure-Checked Scene Editing: A Spatial-Fact Invariant Contract Between Generation and Evaluation

## Motivation

Across the six branches of 3D scene generation represented in the Idea Cards — factored latent diffusion (SceneFactor), latent-tree diffusion (LT3SD), image-intermediary assembly (ArtiScene), single-image interactive generation (WonderWorld), cuboid-arrangement synthesis (CASAGPT), and agentic Blender reconstruction (VibePrevisBench) — editing operations are never checked against a formal invariant that the edited scene still satisfies the spatial constraints of the original brief. VibePrevisBench's Rule-based Scene Check re-runs agent code and compares an exported scene against hidden ground-truth facts in nine dimensions, but it is a one-shot pass/fail: it does not expose which facts a given edit preserves or violates, so controllability (the edit) and evaluation (the check) remain structurally decoupled. The root cause is that no shared machine-checkable object — a fact-closure state — is threaded through both the editing interface and the verifier, so an edit can silently trade one satisfied fact for another without either side recording the trade.

## Key Insight

If the hidden spatial facts are treated as a constraint set over a typed scene state and every edit is required to emit a signed delta of satisfied/violated facts that is re-verified by re-running the exported scene, then the same closure state becomes a compositional certificate shared by generation and evaluation, turning a one-shot pass/fail into a monotone, per-edit verifiable contract.

## Method

**What it is.** CLOSED (Closure-Led Scene EDiting) is a fact-closure contract layer over a scene-editing pipeline. Its inputs are (i) a typed scene state S (objects with cuboid/geometry, pose, camera, time), (ii) a fact set F = {f_1..f_n} of ground-truth spatial predicates in the nine VibePrevisBench dimensions (existence, count, category, position, size, orientation, containment, adjacency, camera-motion), and (iii) a proposed edit e (add/remove/resize box, cuboid rearrangement, or Blender code patch). Its outputs are the edited state S', a closure record C(e) = (preserved, violated, newly_satisfied) over F, and a verdict from re-running the exported scene.

**Load-bearing assumption (stated explicitly).** CLOSED assumes that the nine-dimension typed predicate set F, evaluated on a re-exported scene, is a faithful and complete oracle for the hidden spatial ground-truth facts, such that a predicate flipping true→false between sat0 and sat1 is attributable to the single edit e and the preserved/violated split is a sound measure of spatial-fact preservation. This assumption is not asserted; it is measured and calibrated in Phase 0 below, and the fold in Phase 6 is made group-aware so that coupled facts are not double-counted. Where calibration shows the oracle is unreliable in a dimension, that dimension's reliability weight is reduced and the certificate reports the calibrated (not nominal) closure.

**How it works.**

```pseudocode
# Phase 0: Oracle calibration (once per benchmark, before any edit)
# 0a. Exporter determinism: re-export each of the 242 base scenes N=5 times
flip_rate[f] = mean over scenes, over 5 re-exports of [predicate f flips true<->false]
# 0b. Predicate-vs-hidden-GT agreement per dimension d in {1..9}
agree[d] = (# tasks where all predicates in d match hidden GT) / 242
# 0c. Reliability weight per fact (replaces default w=1; camera-motion prior w=2 retained as multiplier)
r_f = agree[dim(f)] * (1 - flip_rate[f]) * w_prior[dim(f)]
# 0d. Coupling detection: for each fact pair (f_i, f_j) sharing a primitive or pair,
#      compute empirical co-flip rate over the 242 tasks; flag pair as coupled if co-flip > 0.8
G = {joint predicate groups}  # coupled pairs merged into atomic groups; singletons stay

# Phase 1: Fact compilation (once per brief)
F = compile_facts(brief, hidden_gt)          # typed predicates, one per dimension
W = {r_f for f in F}                          # calibrated weights from Phase 0

# Phase 2: Pre-edit closure snapshot
sat0 = evaluate_facts(F, export(S))           # re-run exporter, check predicates

# Phase 3: Edit application with predicted delta
e -> patch(S) -> S'                            # edit op or Blender code patch
delta_pred = predict_delta(e, F, sat0)        # which facts the edit touches

# Phase 4: Closure-checking pass (mandatory)
sat1 = evaluate_facts(F, export(S'))          # re-run exporter on edited scene
C(e) = diff(sat0, sat1)                       # preserved / violated / new

# Phase 5: Admissibility rule (closed-form)
if violated(C(e)) == empty:  accept(S')
else:                        reject(S') unless edit declared as intentional_violation
                             and records violated(C(e)) in the edit log

# Phase 6: Compositional certificate (group-aware fold)
# Fold over joint groups G, not individual facts, so coupled pairs count once
cert = fold_over_groups(C(e_1)..C(e_k), G)
score = sum_{g in preserved(cert)} r_g / sum_{g in G} r_g   # r_g = mean r_f over f in g
```

A `predict_delta` step is a deterministic dependency map from the edit's touched primitives to the facts whose predicates reference those primitives (e.g., resizing box b touches existence(b), size(b), and any adjacency/containment predicate naming b); it is not learned. The closure-checking pass in Phase 4 is the load-bearing step: it re-executes the exporter and re-evaluates every predicate, so `delta_pred` is a hint, not a substitute for verification. The exporter is required to be deterministic; if Phase 0a reports flip_rate[f] > 0.05 for any f, Phase 4 averages `evaluate_facts` over 3 re-exports and reports the majority verdict for that fact, and the fact's reliability weight is reduced accordingly.

**Why this design.** We chose a typed predicate set over a free-form VLM judge because the closure invariant must be machine-checkable per edit, accepting the cost that fact coverage is bounded by the nine dimensions VibePrevisBench already defines. We chose a closed-form admissibility rule (violated = empty, or declared-and-recorded) over a learned accept/reject head because the contract must be auditable by a human reviewer, accepting the cost that edits which legitimately trade one fact for another must be explicitly declared rather than silently allowed. We chose to re-run the exporter after every edit rather than trust the predicted delta because the predicted delta is a dependency heuristic, accepting the cost of one exporter execution per edit. We chose per-edit signed records folded into a scene-level certificate over a single end-state score because composition is what makes the check a compositional verifier rather than a one-shot pass; the closest named prior is VibePrevisBench's Rule-based Scene Check, and a domain expert would NOT describe CLOSED as a variant of it because the check is no longer a terminal pass/fail on the final scene — it is an invariant threaded through the edit sequence, with the violated/preserved split as the unit of evidence. We added Phase 0 calibration and the group-aware fold because the oracle-fidelity and fact-independence assumptions are load-bearing and must be measured, not assumed.

**Why this measures what we claim.** `preserved(C(e))` measures spatial-fact preservation under an edit because the assumption that `evaluate_facts` on the re-exported scene is a faithful predicate oracle for the hidden ground-truth facts holds when the exporter is deterministic and the predicate set covers the nine dimensions; this assumption fails when an edit changes a fact in a dimension not represented in F (e.g., a material change that alters perceived containment in the proxy video but not the geometric predicate), in which case `preserved` reflects geometric-fact preservation only and overstates closure. Phase 0b quantifies this per dimension via `agree[d]`, and Phase 0a quantifies exporter non-determinism via `flip_rate[f]`; both enter the reliability weight `r_f`, so the reported score is calibrated rather than nominal. `violated(C(e))` measures the edit's factual cost because the assumption that a predicate flipping from true to false between `sat0` and `sat1` is attributable to `e` holds when edits are applied one at a time to a fixed base state and the exporter is deterministic; this assumption fails under batched edits or non-deterministic exporters, in which case `violated` reflects aggregate state drift rather than the specific edit's cost, and Phase 4's 3-re-export majority vote plus the `flip_rate[f]` weight bound this failure. `score` measures scene-level closure because the assumption that per-edit records compose additively under the fold holds when facts are independent across dimensions; this assumption fails when two facts are coupled (e.g., containment and adjacency of the same pair), in which case `score` double-counts the coupled pair and should be read as an upper bound on closure. Phase 0d detects coupled pairs by empirical co-flip rate > 0.8 and merges them into atomic groups `G`, so the Phase 6 fold counts each coupled pair once and the score is group-aware rather than naively additive.

## Contribution

['A fact-closure contract layer (CLOSED) that threads a typed spatial-fact state through every scene edit and emits a signed preserved/violated/newly-satisfied record per edit, with a closed-form admissibility rule and a compositional certificate folded from per-edit records.', "A design principle: the same fact set that defines the benchmark's rule-based check can serve as the editing-time invariant, making controllability and evaluation share one machine-checkable object rather than two decoupled pipelines.", 'An edit-log schema and closure-certificate format that makes the VibePrevisBench Scene Check compositional — every accepted scene carries the per-edit records that justify its fact status.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VibePrevisBench 242-task 4D briefs | Only source with hidden nine-dimension spatial ground truth |
| Primary metric | Calibrated group-aware closure score (sum of preserved group weights / total group weight, using r_f from Phase 0) | Directly measures per-edit spatial-fact preservation with measured oracle reliability |
| Calibration metric | Per-dimension predicate-vs-hidden-GT agreement `agree[d]` and per-fact exporter flip rate `flip_rate[f]` over N=5 re-exports | Grounds the oracle-fidelity assumption empirically |
| Coupling metric | Empirical co-flip rate per fact pair; report fraction of pairs flagged coupled (>0.8) | Operationalizes the fact-independence assumption |
| Baseline | Frontier coding agent, single-pass | Tests whether terminal check suffices |
| Baseline | Frontier coding agent + Rule-based Scene Check reject-retry | Tests whether final-scene pass/fail catches edit drift |
| Baseline | Frontier coding agent + free-form VLM judge gate | Tests whether learned judgment replaces typed closure |
| Ablation-of-ours | CLOSED without Phase 4 re-export (trust delta_pred) | Isolates load-bearing verification step |
| Ablation-of-ours | CLOSED with naive additive fold (no Phase 0d coupling groups) | Isolates the group-aware fold |
| Ablation-of-ours | CLOSED with nominal weights (w=1, camera-motion w=2; no Phase 0 calibration) | Isolates the calibration contribution |

### Why this setup validates the claim

The central claim is that threading a machine-checkable closure invariant through the edit sequence, rather than applying a terminal pass/fail to the final scene, yields higher spatial-fact preservation and an auditable per-edit cost record. VibePrevisBench is the right dataset because it is the only source whose hidden ground truth is expressed in exactly the nine typed spatial dimensions CLOSED's predicate set covers, so `evaluate_facts` has a well-defined oracle to agree with — and Phase 0 makes that agreement a measured quantity `agree[d]` rather than an assumption. The single-pass agent baseline tests the sub-claim that a terminal check is insufficient: if an agent can simply produce a correct final scene, closure adds nothing. The reject-retry baseline tests the sub-claim that end-state pass/fail cannot localize drift: it has a check, but only on the final export, so it should catch some violations and miss others depending on whether the drift is visible at the end. The VLM-judge baseline tests the sub-claim that a learned judge is a weaker oracle than typed predicates for this fact set: it should agree on gross layout but disagree on precise predicates like count and containment. The Phase 4 ablation tests the load-bearing claim that the re-export, not the predicted delta, is what makes the check sound: removing it should degrade the score specifically on edits whose dependency map is incomplete. The naive-fold ablation tests the compositionality claim: if coupled facts are not merged, the score should inflate on scenes containing containment/adjacency pairs of the same object pair. The nominal-weight ablation tests the calibration claim: if per-dimension agreement is ignored, the score should overstate closure in dimensions where `agree[d]` is low. The calibrated group-aware closure score is the right primary metric because it is the quantity the method claims to optimize, it is defined per-edit and per-scene, and its weights are measured rather than assumed, so it can detect a localized effect that a scene-level aesthetic or VLM score would average away.

### Expected outcome and causal chain

**vs. Frontier coding agent, single-pass** — On a case where the brief requires an object to be added inside a shelf while preserving adjacency to a wall, the single-pass agent produces a final scene that looks plausible in the proxy video but silently moves the wall-adjacent object, because it has no per-edit fact check and no record of what it changed. Our method instead rejects the edit at Phase 5 because `violated(C(e))` contains the adjacency predicate, forcing the agent to re-attempt with a constrained patch, so we expect a noticeable gap on the containment-and-adjacency subset and near-parity on single-object existence tasks where no fact is coupled.

**vs. Frontier coding agent + Rule-based Scene Check reject-retry** — On a multi-edit sequence where an early resize breaks a containment fact and a later add restores a superficially valid final layout, the reject-retry baseline accepts the final scene because its terminal check passes, even though the intermediate violation was never recorded and the scene's provenance is unauditable. Our method instead records the violation in the per-edit certificate at the moment it occurs, so the folded certificate exposes the violation even when the end state passes, and we expect the gap to concentrate on multi-edit briefs with three or more sequential edits and to vanish on single-edit briefs.

**vs. Frontier coding agent + free-form VLM judge gate** — On a case where the brief specifies an exact object count and a precise orientation, the VLM judge baseline accepts a scene with the wrong count because the proxy video renders overlapping objects and the judge's visual evidence is ambiguous, whereas our typed predicate `count(b)` is evaluated against the exported scene graph and is unambiguous. We expect our method to show a large gap on count and orientation dimensions and near-parity on existence and category, where the VLM judge is already reliable, producing a characteristic dimension-dependent gap profile rather than a uniform improvement.

**vs. CLOSED without Phase 4 re-export (trust delta_pred)** — On an edit that patches Blender code touching a parent empty rather than a named box, the predicted delta misses the facts that reference the child primitives, so the ablated variant accepts an edit that silently violates a containment predicate. Our method's mandatory re-export catches the violation because it re-evaluates every predicate on the actual exported scene, so we expect the ablation to lose score specifically on code-patch edits and on edits whose touched primitives are indirect, with parity on direct box-resize edits where `delta_pred` is complete.

**vs. CLOSED with naive additive fold (no Phase 0d coupling groups)** — On scenes containing a containment-and-adjacency pair of the same object pair, the naive fold counts both facts independently, so a single edit that breaks both is charged twice and a scene that preserves both is credited twice, inflating the score relative to the group-aware fold. We expect the naive-fold score to exceed the group-aware score by an amount proportional to the fraction of coupled pairs flagged in Phase 0d, with parity on scenes with no coupled pairs.

**vs. CLOSED with nominal weights (no Phase 0 calibration)** — In dimensions where `agree[d]` is low (expected: camera-motion and containment, per VibePrevisBench's own finding that multimodal understanding of space and camera motion separates agents), nominal weights overstate the contribution of predicates that do not faithfully track hidden ground truth, so the nominal-weight score should exceed the calibrated score by an amount concentrated in those low-agreement dimensions, with parity in high-agreement dimensions (existence, category).

### What would falsify this idea

If the calibrated closure score gain over the reject-retry baseline is uniform across all edit types and dimensions rather than concentrated on multi-edit sequences and on the count, orientation, and containment dimensions where the predicted failure modes live, then the per-edit invariant is not doing the work and the terminal check is sufficient. Likewise, if removing the Phase 4 re-export does not degrade score on indirect code-patch edits, or if the group-aware fold does not differ from the naive fold on scenes with flagged coupled pairs, or if Phase 0 reports `agree[d]` uniformly high and `flip_rate[f]` uniformly near zero, then the load-bearing steps and the calibration story are not load-bearing and the central mechanism claim is wrong.

## References

1. VibePrevisBench: Evaluating Agentic Coding for 4D Scene Generation in Blender
