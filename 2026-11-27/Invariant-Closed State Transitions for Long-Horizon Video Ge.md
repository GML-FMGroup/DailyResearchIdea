# Invariant-Closed State Transitions for Long-Horizon Video Generation

## Motivation

Long-horizon video generators built on agentic story-state inheritance — StateWeaver being the clearest instance — validate each shot's Active State locally against its references, then commit it to hierarchical memory and chain to the next shot, with no operator that certifies the composition of chained transitions preserves the narrative invariants the story depends on. The same unstated assumption appears in StreamingT2V, whose conditional-attention and appearance-preservation modules enforce local chunk consistency but provide no composition rule, so a sequence of individually consistent chunks can still drift on identity, wardrobe, or story-time ordering. Structurally, per-shot validation is a local predicate, and local predicates do not compose unless the state space is equipped with a closure condition; without one, errors that are invisible at the shot boundary accumulate across the chain.

## Key Insight

If each shot's inherited asset state is represented as a typed invariant set over a fixed attribute vocabulary, then narrative consistency across a chain of shots becomes a closure condition on the composition of typed transitions — a condition that can be checked before commitment rather than assumed after generation.

## Method

**What it is.** We introduce **ClosureWeaver**, a wrapper around StateWeaver's review-driven generation loop. Its inputs are (i) the story specification as an ordered shot list with per-shot narrative obligations, (ii) the current global invariant ledger `L`, and (iii) the generator/reviewer agents already used by StateWeaver. Its outputs are either a committed shot whose Active State has been verified to close against `L`, or a typed repair request naming the invariant that failed. The mechanism is a closure check on typed state transitions, not a new generator.

**Load-bearing assumption (stated explicitly).** ClosureWeaver assumes that the typed attribute vocabulary `A` is a sufficient statistic for the narrative invariants the story specification names: if every scoped predicate over `A` holds along the chain, then the story's narrative obligations are met. This assumption is load-bearing and is not assumed silently. To make it measurable, the method includes (i) an open, per-story invariant schema compiled from the story specification rather than a fixed six-attribute list, and (ii) an explicit coverage/abstention check: any narrative obligation that cannot be compiled into a typed predicate over the extensible attribute set is marked `UNCOVERED` in the ledger, and the closure gate abstains (does not report closure) on shots whose obligations are uncovered. The rate of `UNCOVERED` obligations is reported as the out-of-vocabulary obligation rate.

**How it works.**

```pseudocode
# Attribute vocabulary A is open and per-story: compiled from the story specification.
# Base types: {identity, wardrobe, location, prop, story_time_order, emotional_state}
# Extensible: the compiler may introduce new typed attributes (e.g., relationship_state,
# causal_precondition) when a shot obligation names a property not expressible in base types.
# Invariant ledger L: list of (scope, predicate, covered) where predicate is over typed attribute values
#   and covered in {TRUE, FALSE}; covered=FALSE marks an UNCOVERED obligation.

function COMPILE_LEDGER(story_spec):
    L = []
    for each shot s in story_spec:
        for each narrative obligation o in s.obligations:
            (pred, covered) = COMPILE_TO_TYPED_PREDICATE(o, A)
            L.append((scope=o.scope, pred=pred, covered=covered))
    return L

function GENERATE_SHOT(shot s, ledger L, prior_state S_{i-1}):
    # Phase 1: propose
    draft = StateWeaver.generate(s, references=refs(S_{i-1}))
    S_i   = StateWeaver.extract_active_state(draft)        # typed dict over A

    # Phase 2: local review (unchanged from StateWeaver)
    local_ok, local_repairs = StateWeaver.review(draft, s)
    if not local_ok: return REPAIR(local_repairs)

    # Phase 3: closure check against the global ledger
    T_i = TYPED_TRANSITION(S_{i-1}, S_i)                    # map attr -> (old, new)
    violations = []
    uncovered  = []
    for (scope, pred, covered) in L:
        if scope covers shot i:
            if not covered:
                uncovered.append((scope, pred))
            elif not pred.eval(S_i):
                violations.append((scope, pred, pred.witness(S_i)))
    # Phase 4: composition check — does the CHAIN preserve invariants?
    # For each covered invariant with scope [j..k], require pred holds on S_j..S_k jointly.
    for (scope=[j,k], pred, covered) in L:
        if covered and j <= i <= k and not pred.chain_holds(S_{j:i}):
            violations.append((scope, pred, pred.chain_witness(S_{j:i})))

    # Phase 4b: coverage/abstention gate
    if uncovered:
        # Gate abstains: does not report closure. Emits an UNCOVERED record for the
        # held-out annotator check and for the out-of-vocabulary obligation rate metric.
        return ABSTAIN(uncovered, draft)

    if violations:
        # Translate each violation into a typed repair request for StateWeaver
        return REPAIR(TYPED_REPAIR_REQUESTS(violations))

    # Phase 5: commit
    L = L.extend(INVARIANTS_DERIVED_FROM(S_i, s))          # new invariants this shot introduces
    StateWeaver.memory.commit(S_i, T_i, L)
    return draft

# Typed repair request: (attribute a, required_value, scope, source_shot)
# StateWeaver's Active States machinery consumes this directly as a visual asset requirement.
```

Hyperparameters: the base attribute vocabulary `A` has 6 types (identity, wardrobe, location, prop, story_time_order, emotional_state) and is extensible per story by the ledger compiler; ledger scope granularity is per shot-interval, defaulting to the full story for identity and wardrobe and to local windows of length 3 for location and emotional state; the closure check runs on every shot with no sampling; the abstention gate fires whenever any in-scope obligation is `UNCOVERED`.

**Why this design.** We chose a typed invariant ledger over StateWeaver's free-form review because free-form review cannot distinguish a locally-fine shot that breaks a global story invariant, and we accept the cost that the attribute vocabulary must be fixed in advance and may under-cover narrative properties (e.g., tone). We chose to make the closure check a *pre-commit* gate rather than a post-hoc filter over generated videos because a post-hoc filter cannot feed repairs back into StateWeaver's Active States loop, and we accept the cost that a failed closure check forces regeneration rather than reuse. We chose typed repair requests over natural-language repair prompts because typed requests are directly consumable by StateWeaver's reference-construction step and can be checked for satisfaction, and we accept the cost that some narrative failures cannot be expressed as a typed attribute violation. A domain expert would not describe ClosureWeaver as a variant of StateWeaver: StateWeaver's contribution is review-driven reference updating, whereas ClosureWeaver's contribution is a composition operator on the state space that StateWeaver's per-shot validation does not define. Nor is it a variant of StreamingT2V's CAM/APM: those are memory modules over features, whereas ClosureWeaver operates on a symbolic invariant ledger and gates commitment.

**Why it measures what we claim.** The computational quantity `pred.chain_holds(S_{j:i})` measures *global narrative invariant preservation* because we assume that the typed attribute vocabulary `A` is a sufficient statistic for the narrative properties the story specification names — i.e., if identity, wardrobe, location, prop, story-time order, and emotional state all satisfy their scoped predicates along the chain, then the story's narrative obligations are met; this assumption fails when a narrative obligation depends on a property outside `A` (for example, a relationship reveal that is not reducible to any single attribute), in which case the closure check reflects only attribute-level consistency and reports closure even though the narrative invariant is broken. The `UNCOVERED` abstention path and the out-of-vocabulary obligation rate are the operational measurement of this assumption: they count how often the story specification names an obligation that the compiler cannot express as a typed predicate. The quantity `TYPED_TRANSITION(S_{i-1}, S_i)` measures *state inheritance fidelity* because we assume that StateWeaver's `extract_active_state` faithfully reads the generated shot's attributes; this assumption fails when the extractor hallucinates an attribute value that the pixels do not support, in which case the transition records a state change that did not occur and closure is checked against a fictitious state. The quantity `violations` measures *actionable repair signal* because we assume that every invariant violation has a typed repair request that StateWeaver's reference-construction step can satisfy; this assumption fails when an invariant is violated by an interaction between two attributes (e.g., wardrobe and location jointly), in which case the repair request names one attribute and the regeneration may fix the wrong one.

## Contribution

["A compositional state-closure operator over StateWeaver's Active States: a typed invariant ledger plus a pre-commit closure check that certifies chained shot transitions preserve scoped narrative invariants, with violations translated into typed repair requests consumable by StateWeaver's existing reference-construction loop.", 'A design principle for long-horizon video generation: narrative consistency across a chain of shots is a closure condition on typed state transitions, not a per-shot property, and enforcing it before commitment changes the failure mode from silent accumulation to explicit, repairable violation.', "A typed attribute vocabulary and scoped-invariant ledger specification that makes the closure condition checkable and falsifiable, together with the closure-vs-local-review ablation protocol that isolates the composition operator's effect."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MovieBench (multi-scene, character-consistency splits) | Long coherent storylines with cross-scene identity obligations |
| Primary metric | Cross-shot invariant violation rate, counted by (a) the ledger's own predicates and (b) a held-out annotator judgment of narrative-obligation satisfaction that is not defined by the ledger | Directly measures the closure claim without circularity: the annotator check can register violations outside the typed vocabulary |
| Secondary metric | Out-of-vocabulary obligation rate: fraction of story-specification obligations that the ledger compiler marks `UNCOVERED` | Bounds how often the sufficiency assumption fails on MovieBench |
| Secondary metric | Extractor fidelity: accuracy of `StateWeaver.extract_active_state` against hand-annotated ground-truth typed states on a calibration set of 40 MovieBench shots | Quantifies the dependence of the closure check on extractor correctness |
| Baseline | StateWeaver (unmodified) | Tests whether free-form review alone already closes invariants |
| Baseline | StreamingT2V (autoregressive CAM/APM) | Tests whether feature-memory consistency suffices for story invariants |
| Baseline | LongDiff (training-free single-pass) | Tests whether global-consistency tricks substitute for a state ledger |
| Baseline | Generic runtime-assertion baseline: the same scoped predicates checked as plain assertions at each shot boundary, with no chain/composition operator | Tests whether the composition operator buys something a plain assertion library does not |
| Ablation-of-ours | ClosureWeaver minus chain check (per-shot only) | Isolates the composition operator from local review |
| Ablation-of-ours | ClosureWeaver with abstention disabled (reports closure on `UNCOVERED` obligations) | Isolates the coverage gate from the closure check |

### Why this setup validates the claim

The central claim is that a typed, pre-commit closure check over a global invariant ledger catches narrative-invariant breaks that per-shot review and feature-memory consistency miss. MovieBench is the right testbed because it supplies multi-scene, consistent-character stories with explicit shot-level obligations, so invariant violations are observable rather than subjective. The primary metric is deliberately split: the ledger's own predicates measure whether the closure operator fires as designed, while the held-out annotator judgment measures whether narrative obligations are actually satisfied, so the sufficiency of the attribute vocabulary is measured rather than assumed. The out-of-vocabulary obligation rate and the extractor-fidelity calibration set of 40 hand-annotated shots bound the two assumptions the closure check rests on. Each baseline isolates a different sub-claim: unmodified StateWeaver tests whether free-form review already closes global invariants; StreamingT2V tests whether feature-level memory is sufficient for symbolic story invariants; LongDiff tests whether training-free global-consistency machinery substitutes for an explicit ledger; the generic runtime-assertion baseline tests whether the composition operator is more than a plain assertion library. The per-shot-only ablation is the decisive control for the composition operator, and the abstention-disabled ablation isolates the coverage gate. Together, if the gain appears only where the predicted failure mode lives (long-scope identity/wardrobe invariants spanning many shots) and vanishes in the per-shot-only ablation, the claim survives; if the gain is uniform or the ablation matches full ClosureWeaver, the claim is wrong.

### Expected outcome and causal chain

**vs. StateWeaver (unmodified)** — On a case where a character's wardrobe is established in shot 1 and must persist through a scene change at shot 9, StateWeaver's free-form review approves each shot locally because nothing in that shot contradicts its own description, and the drift accumulates silently across the intervening shots. Our method instead evaluates a scoped wardrobe predicate against the chain S_1..S_9 at commit time and raises a typed repair request naming wardrobe before the drift compounds. We expect a noticeable gap on long-scope identity/wardrobe subsets and near-parity on single-scene shorts.

**vs. StreamingT2V** — On a case where a prop is introduced, abandoned, and must reappear at a specified story time, StreamingT2V's CAM/APM carry appearance features but have no representation of story-time ordering, so the prop reappears with plausible appearance at the wrong narrative moment or not at all. Our method instead holds story_time_order as a typed attribute with a scoped predicate and blocks commit until the transition satisfies it. We expect a concentrated gap on prop-reappearance and story-time subsets, with parity on motion-dynamics measures where StreamingT2V is strong.

**vs. LongDiff** — On a case where a relationship reveal depends on two characters' co-presence across several shots, LongDiff's single-pass global consistency preserves visual smoothness but has no mechanism to detect that a narrative obligation was never discharged, so it produces a seamless but narratively broken sequence. Our method instead records the obligation as a scoped invariant and reports a typed violation at the shot where closure fails; if the obligation cannot be compiled into a typed predicate, the abstention gate marks it `UNCOVERED` rather than reporting closure. We expect a gap on multi-shot narrative-obligation subsets but essentially no gap on short, single-scene clips.

**vs. Generic runtime-assertion baseline** — On the same long-scope wardrobe case, per-shot assertions fire only when a single shot's state violates a predicate, so a gradual drift that never violates a single-shot predicate passes every assertion while the chain drifts. Full ClosureWeaver's chain check evaluates the scoped predicate over S_1..S_9 jointly and catches it. We expect the assertion baseline to match ClosureWeaver on short-scope location/emotion invariants and to lose most of the gain on long-scope identity/wardrobe invariants.

**vs. ClosureWeaver minus chain check** — On the same long-scope wardrobe case, the ablation checks only that each shot's state satisfies predicates at that shot, so a gradual drift that never violates a single-shot predicate passes every local check while the chain drifts. Full ClosureWeaver's chain check catches it. We expect the ablation to recover most of the gain on short-scope location/emotion invariants but to lose most of it on long-scope identity/wardrobe invariants — the diagnostic that the composition operator, not the typing, is doing the work.

**vs. ClosureWeaver with abstention disabled** — On stories containing relationship-reveal obligations that the compiler marks `UNCOVERED`, the abstention-disabled variant reports closure on those shots, so the held-out annotator check registers narrative-obligation violations that the ledger's own predicates do not. We expect the abstention-disabled variant to show a higher annotator-judged violation rate on the out-of-vocabulary subset, confirming that the coverage gate is what prevents false closure.

### What would falsify this idea

If ClosureWeaver's improvement over unmodified StateWeaver is uniform across short-scope and long-scope invariant subsets rather than concentrated on long-scope identity/wardrobe/story-time cases, the composition-operator claim is wrong. Equally, if the per-shot-only ablation matches full ClosureWeaver on long-scope subsets, the chain check is not the causal mechanism; and if the out-of-vocabulary obligation rate on MovieBench is high while the held-out annotator check shows no difference between abstention-enabled and abstention-disabled variants, the sufficiency assumption is not the binding constraint.

## References

1. StateWeaver: Multi-Agent Coordination of Story-State Transitions for Long-Horizon Video Generation
