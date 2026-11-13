# World Closure Auditing via Mutation-Guided Coverage of 4D World Specifications

## Motivation

WorldDraft claims its 48 executable 4D world families span six dynamic challenge categories, yet it provides no rule for when a code-built world fully specifies the dynamics it purports to test; an under-specified world can satisfy world preservation vacuously because the video model is never forced to render a behavior the code fails to constrain. This is the same residual gap that appears in BlenderAlchemy's vision-language editing loop (where an edited scene graph is assumed to fully specify the scene) and in Cosmos/VPO's reliance on code-generated worlds as ground truth: across all four paper-trees, executable-world benchmarks lack a closure criterion that certifies the code captures every dynamic behavior the benchmark claims to evaluate.

## Key Insight

A world family is closed with respect to a dynamic challenge category exactly when every behavior distinguishable by the benchmark's own preservation metric is forced by the code — so closure can be measured as the fraction of metric-distinguishable mutants that the code's execution rejects, turning an informal 'coverage' claim into a mutation-kill rate over the specification's own observable surface.

## Method

**What it is.** We introduce **CLOSURE-AUDIT**, a mutation-guided coverage procedure that takes as input a WorldDraft-style executable 4D world family (a Python program producing per-frame object poses and camera poses) plus the benchmark's world-preservation metric, and outputs a per-(family, challenge-category) closure score together with a ranked list of under-specified behaviors the code fails to constrain.

**How it works.**
```pseudocode
Input: family F (code), challenge category C, preservation metric M,
       mutation operators O = {O_kin, O_cam, O_contact, O_occl, O_light, O_topo},
       budget N=200 mutants per (F,C), kill threshold tau=0.05,
       independent behavior probes P = {P_contact, P_occl, P_light, P_topo}
       (contact-event detector, occlusion-flag change detector, light-change detector, topology-change detector),
       calibration sample size S=50 mutants per (F,C) for hand-labeling
Output: closure score kappa(F,C), ranked under-specification witnesses,
        sensitivity bound kappa_uncovered(F,C)

1. Reference rollout: execute F, record trajectory T_ref = {(p_i, q_i, c_i)}_{i=1..T}
   (object poses p, camera poses q, visibility/contact flags c).
2. For each mutation operator o in O:
     a. Sample N_o = N/6 mutant programs F' = o(F) that alter a code-level
        dynamic primitive (e.g., swap a keyframe, delete a contact constraint,
        perturb a camera spline control point by sigma=0.1 of scene scale).
     b. Execute F'; if F' fails to run, discard (syntactic, not semantic).
     c. Compute delta = M(T_ref, T_{F'}).
     d. Compute probe signals: for each probe P_j, a binary flag probe_j(F') = 1 if P_j detects a change
        from T_ref (e.g., contact event count differs, occlusion flag flips, light intensity changes by >0.1,
        topology (number of connected components) changes).
     e. Label F' as *distinguishable* if delta > tau AND at least one probe_j(F') = 1
        (agreement between M and an independent behavior probe).
     f. Label F' as *metric-invisible* if delta <= tau BUT at least one probe_j(F') = 1
        (behavior real but M misses it).
     g. Label F' as *killed* if F's specification surface (declared constraints,
        asserted invariants, and the family's documented challenge-category
        predicates) rejects F' — i.e., F' violates a constraint that F explicitly
        encodes for category C.
3. Closure score: kappa(F,C) = (# distinguishable AND killed) / (# distinguishable).
   Sensitivity bound: kappa_uncovered(F,C) = (# metric-invisible AND killed) / (# metric-invisible),
   reported separately as the fraction of metric-invisible behaviors that are nevertheless constrained.
4. Witness extraction: rank surviving distinguishable mutants by delta; emit the
   top-5 as executable counterexample scripts showing a behavior the benchmark
   claims to test but F does not constrain.
5. Aggregate: report kappa per family, per category, and the 48x6 closure matrix;
   also report kappa_uncovered per cell.
6. Calibration: draw a stratified sample of S=50 mutants per (F,C) (stratified by operator and by
   distinguishable/metric-invisible label). Two independent annotators label each mutant as
   *truly-under-specified* (behavior is a genuine dynamic event in category C that F fails to constrain)
   or *metric-artifact* (behavior is an artifact of M's insensitivity or noise).
   Compute precision = TP/(TP+FP) and recall = TP/(TP+FN) of kappa's low-closure cells against this gold set.
```

**Why this design.** We chose mutation-kill rate over a static code-coverage proxy (e.g., line or branch coverage of the world program) because line coverage cannot distinguish a constraint that is *executed* from one that is *load-bearing* for the benchmark's metric, accepting the cost that mutation sampling is stochastic and requires a budget N per cell. We chose to define distinguishability using the benchmark's own preservation metric M rather than a hand-authored behavior taxonomy because this makes closure relative to what WorldDraft actually claims to measure, accepting the cost that closure scores are metric-dependent and not comparable across different preservation metrics. We chose six mutation operators aligned one-to-one with WorldDraft's six dynamic challenge categories rather than a generic program-mutation operator set because category-aligned mutations produce witnesses that are directly interpretable as under-specified behaviors in that category, accepting the cost that operators must be re-authored if the category taxonomy changes. We chose to discard non-running mutants rather than count them as killed because a syntactic failure is not evidence of semantic constraint, accepting the cost that some genuinely under-specified worlds may receive inflated closure scores if their mutants frequently crash. Unlike WorldDraft's fixed 432-case evaluation, which reports world-preservation error without asking whether the code could have constrained the behavior at all, CLOSURE-AUDIT treats the world program as a specification under test and measures its constraint strength directly; a domain expert would not describe this as a variant of WorldDraft because WorldDraft evaluates *video models* against worlds, whereas CLOSURE-AUDIT evaluates *worlds* against their own claimed challenge categories.

**Why it measures what we claim.** The closure score kappa(F,C) measures the *specification completeness* of family F for category C because we assume that a mutant which the preservation metric M can distinguish from the reference is a behavior the benchmark claims to test; this assumption fails when M is insensitive to a genuine dynamic behavior (e.g., a contact event that produces no large pose delta), in which case kappa reflects metric sensitivity rather than specification completeness. To make this assumption explicit and testable, we define the equivalence assumption A: *M-distinguishability equals benchmark-claimed behavior* (i.e., M is sensitive to every dynamic behavior in category C). We operationalize A by requiring agreement between M and at least one independent behavior probe before labeling a mutant distinguishable, and we report the metric-invisible mutants separately as kappa_uncovered. The kill label measures *explicit constraint* because we assume the family's declared constraints and category predicates are the operational definition of what the code specifies; this assumption fails when a constraint is enforced only implicitly by the code's execution order rather than by a declared predicate, in which case the kill label reflects declaration style rather than actual constraint. The mutation operators O measure *dynamic challenge coverage* because we assume each operator targets exactly one of WorldDraft's six categories; this assumption fails when an operator (e.g., camera-spline perturbation) incidentally changes occlusion behavior, in which case the per-category attribution of a surviving mutant is ambiguous. The witness ranking by delta measures *severity of under-specification* because we assume larger metric-distinguishable deviations correspond to more consequential unconstrained behaviors; this assumption fails when M is nonlinear in behavior magnitude, in which case the ranking reflects metric geometry rather than behavioral importance. The calibration study (step 6) directly quantifies how much of kappa reflects specification completeness versus metric sensitivity by comparing kappa's low-closure cells to a hand-labeled gold set of truly-under-specified vs. metric-artifact mutants.

## Contribution

['CLOSURE-AUDIT, a mutation-guided coverage procedure that formalizes world-family closure as a metric-relative mutation-kill rate over category-aligned dynamic mutation operators, producing a 48x6 closure matrix and executable under-specification witnesses for the WorldDraft benchmark.', "An empirical finding characterizing which of WorldDraft's six dynamic challenge categories are systematically under-specified across the 48 world families, and whether under-specification concentrates in categories where world preservation and visual realization are in tension.", 'A reusable audit tool and witness format that exposes under-specified worlds as executable counterexample scripts, enabling benchmark maintainers to patch world families and enabling video-model evaluators to report closure-adjusted preservation scores.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WorldDraft 48 families × 6 challenge categories | Native category structure matches per-cell closure claim. |
| Primary metric | Closure score kappa(F,C) per (family, category) | Directly quantifies constraint strength, not render quality. |
| Secondary metric | kappa_uncovered(F,C) (metric-invisible kill rate) | Quantifies constraint on behaviors M misses. |
| Baseline | Static line/branch coverage of world programs | Tests whether execution coverage substitutes for constraint load-bearing. |
| Baseline | WorldDraft fixed 432-case preservation error | Tests whether fixed-case evaluation already exposes under-specification. |
| Baseline | Random program mutation (untyped operators) | Tests whether category alignment is necessary for interpretable closure. |
| Ablation-of-ours | CLOSURE-AUDIT with kill-label removed (distinguishability only) | Isolates contribution of declared-constraint kill labeling. |
| Ablation-of-ours | CLOSURE-AUDIT with single-metric distinguishability (no probes) | Isolates contribution of multi-signal distinguishability oracle. |
| Calibration | Hand-labeled sample of S=50 mutants per (F,C) by two annotators | Quantifies precision/recall of kappa vs. gold standard. |

### Why this setup validates the claim

The central claim is that closure score kappa(F,C) measures specification completeness of a world program relative to a challenge category, and that this quantity is not recoverable from either static code coverage or fixed-case behavioral error. The static coverage baseline tests the sub-claim that execution is not constraint: if line/branch coverage already predicts which categories are under-specified, then kappa adds nothing and the mutation machinery is unnecessary. The fixed 432-case WorldDraft baseline tests the sub-claim that behavioral error alone does not reveal whether the code could have constrained a behavior — a world can show low preservation error simply because the fixed cases never probe the unconstrained region. The untyped random-mutation baseline tests the sub-claim that category-aligned operators are what make witnesses interpretable and per-category attribution meaningful. The ablation removing the kill label isolates the load-bearing-constraint mechanism: if distinguishability alone reproduces the full closure matrix, the declared-constraint half of the definition is doing no work. The ablation removing independent probes isolates the contribution of the multi-signal distinguishability oracle: if single-metric distinguishability reproduces the full closure matrix, the probes are unnecessary. The calibration study directly tests the load-bearing assumption A (M-distinguishability equals benchmark-claimed behavior) by comparing kappa's low-closure cells to a hand-labeled gold set of truly-under-specified vs. metric-artifact mutants, reporting precision and recall. Closure score is the right metric because it is defined exactly over the object we claim to audit — the world program as a specification — and the predicted effect is a structured, non-uniform pattern across the 48×6 matrix, not a single aggregate win.

### Expected outcome and causal chain

**vs. Static line/branch coverage** — On a family where a contact constraint is declared but immediately overwritten by later keyframe assignment, static coverage reports the constraint line as executed and predicts closure because the mechanism cannot see load-bearingness. Our method instead mutates the contact primitive, observes a metric-distinguishable deviation, and finds no declared predicate rejecting it, so the mutant survives and lowers kappa. We expect the gap between coverage-predicted and measured closure to concentrate in exactly the families with declared-but-overwritten or declared-but-unreachable constraints, with near-parity on families whose constraints are cleanly enforced.

**vs. WorldDraft fixed 432-case preservation error** — On a family whose unconstrained behavior lies outside the fixed case distribution (e.g., an occlusion regime never sampled by the 432 cases), the fixed-case evaluation reports low preservation error because the mechanism only measures error on pre-chosen cases, never asking whether the code could constrain the regime at all. Our method instead generates mutants that drive the world into that regime and checks whether the specification surface rejects them, so a surviving distinguishable mutant appears as a low-kappa cell with an executable witness. We expect the fixed-case error to be near-uniform across the matrix while kappa shows a sparse, category-structured pattern of low-closure cells that the fixed cases missed.

**vs. Random untyped program mutation** — On a family with a subtle camera-spline under-specification, random untyped mutants mostly crash or perturb unrelated primitives because the mechanism has no category alignment, so the surviving mutants are uninterpretable and cannot be attributed to the camera category. Our method instead applies the camera-spline operator, whose mutants are syntactically valid and semantically localized, so surviving mutants are directly readable as camera-category witnesses. We expect random mutation to yield a higher discard rate and a flatter, less category-discriminative closure matrix, while CLOSURE-AUDIT concentrates low-kappa cells along the diagonal of its category attribution.

**vs. CLOSURE-AUDIT with single-metric distinguishability (no probes)** — On a family where a contact event produces no large pose delta, the single-metric version labels the mutant as not distinguishable and excludes it from the denominator, inflating kappa. Our method requires agreement between M and at least one independent behavior probe (e.g., contact-event detector), so the mutant is labeled metric-invisible and reported separately in kappa_uncovered. We expect the single-metric version to yield higher kappa on families with metric-invisible behaviors, while CLOSURE-AUDIT shows lower kappa and non-zero kappa_uncovered, revealing the constraint gap.

**vs. CLOSURE-AUDIT with kill-label removed (distinguishability only)** — On a family where a constraint is declared but not enforced, the kill-label-removed version counts all distinguishable mutants as killed (since it only checks distinguishability), inflating kappa. Our method checks whether the specification surface rejects the mutant, so unenforced constraints yield surviving mutants and lower kappa. We expect the kill-label-removed version to produce a uniformly high closure matrix, while CLOSURE-AUDIT shows a structured pattern of low-kappa cells corresponding to unenforced constraints.

**vs. Calibration gold set** — We expect kappa's low-closure cells to achieve precision ≥0.7 and recall ≥0.6 against the hand-labeled gold set of truly-under-specified mutants, with the remaining errors attributable to metric-artifact mutants (where M is insensitive but the behavior is real). The kappa_uncovered metric should capture a substantial fraction of these metric-artifact mutants, reducing the false-negative rate of kappa.

### What would falsify this idea

If CLOSURE-AUDIT's low-kappa cells were uniformly distributed across the 48×6 matrix rather than concentrated in families where declared constraints are overwritten, unreachable, or absent, and if the static-coverage and fixed-case baselines predicted those cells equally well, then kappa would be measuring sampling noise rather than specification completeness and the central claim would be wrong. A second falsifier: if the kill-label ablation reproduced the full closure matrix, the declared-constraint definition would be doing no work and the method would reduce to metric-sensitivity measurement. A third falsifier: if the calibration study showed precision and recall near chance (e.g., <0.3), then kappa would not reliably distinguish specification completeness from metric sensitivity.

## References

1. WorldDraft: If Code Builds the World, Can Video Models Just Render It?
