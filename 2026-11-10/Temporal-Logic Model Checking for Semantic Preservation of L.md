# Temporal-Logic Model Checking for Semantic Preservation of Lottie Motion Edits

## Motivation

MotionGraphicsBench verifies each prompt requirement with a separate deterministic program that checks spatial, appearance, and temporal properties independently, so its oracle inherits an independent-tween assumption: causal timing structure across properties (e.g., 'the icon must finish fading before the badge begins sliding') stays implicit and is never checked as a joint constraint. This is a recurring structural gap across programmatic-verification approaches to motion graphics, not a local bug — any per-property final-frame equality check can pass while the compositional temporal ordering that makes the animation semantically correct is violated, because no first-class representation of cross-property causal invariants exists to test against.

## Key Insight

Encoding cross-property causal timing as explicit temporal-logic formulas over the animation's property traces turns semantic preservation under a motion edit into a decidable model-checking problem, so an edit is accepted exactly when every invariant holds over the whole trace rather than at sampled frames.

## Method

**What it is.** We introduce TL-Lottie, a verification layer that takes a Lottie animation (or an edited variant) and a set of cross-property causal invariants written in a bounded linear temporal logic (BLTL) over property traces, and returns a per-invariant pass/fail with a minimal violating frame interval. Inputs: a Lottie JSON, an invariant specification; output: a verification report.

**Load-bearing assumption (stated explicitly).** Every downstream verdict rests on the threshold event abstraction being faithful: each atomic proposition (e.g. `fade_done(icon)`, `slide_start(badge)`) is true at frame t iff the underlying property has genuinely crossed its semantic threshold by t, so that BLTL satisfaction over the boolean abstraction is equivalent to the causal ordering holding in the animation. This assumption is not hidden in a global constant; it is made a checkable precondition per verdict via the per-property hysteresis abstraction and the per-invariant abstraction-stability flag described below. It fails when a property oscillates around its threshold (slow fade tails, scale/position overshoot from Bezier easing), in which case the verdict would reflect threshold-crossing noise rather than a genuine causal violation — which is exactly why the abstraction is a Schmitt trigger with a derivative-derived deadband and why any witness inside the deadband is reported as `abstraction-unstable` rather than pass/fail.

**How it works.**

```pseudocode
def verify(lottie_json, invariants, fps=60, horizon=None):
    # Phase 1: trace extraction
    traces = {}
    for prop in PROPERTIES:            # opacity, position.x, position.y, scale, rotation, color
        traces[prop] = sample_property(lottie_json, prop, fps)   # dense per-frame series
    # Phase 1b: per-property derivative bound for deadband sizing
    for prop in PROPERTIES:
        dmax[prop] = max_t |traces[prop][t+1] - traces[prop][t]|   # max per-frame step
    # Phase 2: event abstraction (atomic propositions) via Schmitt trigger
    aps = {}
    for inv in invariants:
        for ap in inv.atomic_props:    # e.g. fade_done(icon), slide_start(badge)
            # two thresholds: enter = tau_enter[prop], exit = tau_exit[prop]
            # deadband margin derived from derivative bound: margin[prop] = k * dmax[prop], k=2
            aps[ap] = schmitt_event(traces, ap,
                                    tau_enter=tau_enter[prop],
                                    tau_exit=tau_exit[prop],
                                    margin=k*dmax[prop])  # boolean per frame
    # Phase 3: bounded model checking
    H = horizon if horizon else len(next(iter(traces.values())))
    report = []
    for inv in invariants:
        holds, witness = bmc_check(inv.formula, aps, H)   # standard BLTL bounded semantics
        # abstraction-stability check: is the witness (or any frame in the witness interval)
        # within the deadband of any atomic proposition's threshold?
        unstable = any(abs(traces[prop][t] - tau_enter[prop]) < margin[prop]
                       for t in witness_interval(witness)
                       for prop in inv.properties)
        report.append({"invariant": inv.id,
                       "holds": holds,
                       "witness": witness,
                       "abstraction_stable": not unstable})
    return report
```

Invariants are written as BLTL formulas with bounded until and globally operators over atomic propositions derived from property thresholds, e.g. `G[0,H](fade_done(icon) -> F[0,12](slide_start(badge)))`. `bmc_check` evaluates the formula over the finite trace by dynamic programming on the bounded semantics, returning the earliest frame index at which the formula is falsified. The Schmitt trigger uses two thresholds per property: `tau_enter[prop]` (the value at which the AP becomes true) and `tau_exit[prop]` (the value at which it becomes false), separated by a deadband `margin[prop] = k * dmax[prop]` with `k=2`, where `dmax[prop]` is the maximum per-frame step of that property's trace. This makes the abstraction faithful under the explicit separation condition that the signal's per-frame variation is bounded by `dmax[prop]`; any invariant whose witness interval contains a frame within the deadband is flagged `abstraction_stable: false` rather than reported as a clean pass/fail.

**Why this design.** We chose dense per-frame trace extraction over keyframe-only inspection because Lottie interpolates between keyframes and causal violations can occur mid-interpolation, accepting the cost that trace extraction is O(frames x properties) and requires choosing an fps. We chose BLTL with explicit bounds over unbounded LTL because animation edits are evaluated on a finite horizon and bounded until gives a decidable, frame-exact witness, accepting the cost that bounds must be specified per invariant and are not inferred. We chose threshold-based atomic propositions over learned event detectors because thresholds make the oracle deterministic and human-auditable, accepting the cost that threshold values must be calibrated per property scale and can misclassify slow transitions; to make this cost explicit and checkable, we replace the fixed global `tau=1e-3` with a per-property Schmitt trigger whose deadband is derived from the property's own derivative bound, and we report an `abstraction_stable` flag alongside every verdict so that a witness inside the deadband is surfaced as an abstraction-stability caveat rather than silently changing the pass/fail. Unlike MotionGraphicsBench's per-requirement programs, which evaluate each property in isolation, and unlike a generic temporal-logic runtime monitor that would treat properties as independent streams, TL-Lottie's atomic propositions are defined over pairs of properties so that the formula itself encodes the coupling; a domain expert would not describe this as a variant of per-requirement checking because the unit of verification is a joint temporal constraint, not a single-property predicate.

**Why it measures what we claim.** The bounded model-checking verdict `bmc_check(inv.formula, aps, H)` measures semantic preservation of causal timing because, under the assumption that the threshold event abstraction is faithful (each atomic proposition is true at frame t iff the underlying property has crossed its semantic threshold by t, enforced by the Schmitt trigger with derivative-derived deadband), the BLTL formula is satisfied over the trace iff the causal ordering holds at every frame; this assumption fails when a property oscillates around its threshold, in which case the verdict reflects threshold-crossing noise rather than a genuine causal violation, and the `abstraction_stable` flag marks that case. The minimal witness frame interval measures the location of the causal break because, under the assumption that the bounded semantics' earliest falsification point corresponds to the first frame where the consequent becomes unreachable within its bound, the witness localizes the violation; this assumption fails when the bound is set too tight, in which case the witness reflects an over-constrained deadline rather than an actual ordering failure. The per-invariant pass/fail measures cross-property semantic preservation because, under the assumption that the invariant set is complete for the edit's requirements, conjunction of all invariants is equivalent to the edit being semantically correct; this assumption fails when an edit introduces a requirement not covered by any invariant, in which case the report reflects only the specified subset of semantics.

## Contribution

(1) TL-Lottie, a temporal-logic verification layer for Lottie animations that lifts cross-property causal timing invariants to first-class BLTL formulas and checks them by bounded model checking with frame-exact witnesses. (2) A demonstration that joint temporal invariants detect semantic-preservation failures that per-property final-frame equality checks in MotionGraphicsBench-style verification miss, establishing compositional temporal constraints as a necessary oracle component. (3) An invariant specification format and trace-extraction pipeline that makes Lottie motion edits auditable at the level of causal ordering rather than sampled frames.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MotionGraphicsBench 180-animation balanced test set | Programmatic ground truth enables frame-exact comparison |
| Primary metric | Agreement with human causal-ordering judgments | Directly tests semantic preservation, not surface similarity |
| Baseline | GPT-6 Astra VLM judge | Tests whether holistic visual judging misses temporal coupling |
| Baseline | MotionGraphicsBench per-requirement programs | Tests whether isolated property checks miss cross-property coupling |
| Baseline | Generic BLTL runtime monitor (independent streams) | Tests whether joint atomic propositions are essential |
| Ablation-of-ours | TL-Lottie with keyframe-only trace extraction | Tests whether dense interpolation sampling is necessary |
| Ablation-of-ours | TL-Lottie with fixed global tau=1e-3 (no Schmitt trigger) | Tests whether the per-property hysteresis abstraction is necessary |
| Calibration study | Sweep tau_enter in {1e-4, 1e-3, 1e-2} and k in {1, 2, 4} | Measures verdict sensitivity to threshold calibration |

### Why this setup validates the claim

The central claim is that verification of causal timing in motion graphics requires joint temporal constraints over dense, cross-property traces, and that TL-Lottie's BLTL-over-paired-atomic-propositions mechanism captures this where alternatives do not. Each baseline isolates one mechanism gap: the VLM judge tests whether holistic perception suffices (predicted to fail on subtle mid-interpolation ordering); the per-requirement programs test whether single-property predicates suffice (predicted to fail on coupled constraints); the generic runtime monitor tests whether joint atomic propositions matter (predicted to fail on cross-property formulas). The keyframe-only ablation isolates the value of dense sampling, so any gain attributable to interpolation-aware traces is separated from the BLTL machinery itself. The fixed-tau ablation isolates the value of the per-property hysteresis abstraction, so any gain attributable to faithful event abstraction is separated from the BLTL machinery itself; the calibration study reports how verdicts move as tau_enter and k vary, making the load-bearing faithfulness assumption an explicit, measured precondition rather than a hidden constant. Agreement with human causal-ordering judgments is the right metric because the claim is about semantic preservation of ordering, not pixel fidelity or aesthetic quality; a metric that rewards surface similarity could not distinguish a temporally-correct animation from a visually-similar-but-causally-broken one. Together these form a falsifiable design: if TL-Lottie's advantage is real, it must be concentrated on the subsets where each baseline's specific mechanism gap bites, and near-parity elsewhere.

### Expected outcome and causal chain

**vs. GPT-6 Astra VLM judge** — On a case where the badge slides in before the icon's fade completes but both events are visually plausible in isolation, the VLM judge produces a pass because it reasons over sampled frames holistically and lacks a frame-exact ordering oracle. Our method instead evaluates the paired BLTL formula `G[0,H](fade_done(icon) -> F[0,12](slide_start(badge)))` frame by frame, so we expect a noticeable agreement gap on coupled-timing subsets but parity on single-event subsets.

**vs. MotionGraphicsBench per-requirement programs** — On a case where each property individually satisfies its own requirement (fade completes, slide starts) but the causal dependency between them is violated, the per-requirement baseline produces a pass because it checks each property in isolation and never encodes the coupling. Our method instead treats the joint temporal constraint as the unit of verification, so we expect a marked gap on cross-property-coupling instances and parity on independent-property instances.

**vs. Generic BLTL runtime monitor (independent streams)** — On a case where the violation only manifests as a relation between two property traces rather than within either trace, the generic monitor produces a pass because its atomic propositions are single-stream predicates with no coupling. Our method instead defines atomic propositions over property pairs, so we expect a gap concentrated on relational-violation subsets and parity on single-stream subsets.

**vs. TL-Lottie with fixed global tau=1e-3 (no Schmitt trigger)** — On a case where a slow fade tail oscillates within 1e-3 of the threshold, the fixed-tau variant flips the `fade_done(icon)` AP on sub-threshold noise and produces a spurious pass or fail, whereas the Schmitt trigger with derivative-derived deadband holds the AP stable and the `abstraction_stable` flag marks any witness inside the deadband. We expect the fixed-tau ablation to show lower agreement on slow-transition subsets and parity on sharp-transition subsets, isolating the value of the hysteresis abstraction from the BLTL machinery.

**Calibration study** — Sweeping tau_enter over {1e-4, 1e-3, 1e-2} and k over {1, 2, 4} reports the distribution of property values around thresholds and the sensitivity of verdicts to these choices; we expect verdicts to be stable for k≥2 on sharp transitions and to show the `abstraction_stable` flag concentrating on slow-transition instances, making the faithfulness assumption empirically checkable.

### What would falsify this idea

If TL-Lottie's agreement gain over the per-requirement and generic-monitor baselines were uniform across all subsets rather than concentrated on the coupled-timing and relational-violation subsets where those baselines' mechanism gaps predict failure, the central claim that joint cross-property BLTL is the operative mechanism would be wrong. Likewise, if the keyframe-only ablation matched full dense-trace TL-Lottie, the claim that mid-interpolation violations require dense sampling would be falsified; and if the fixed-tau ablation matched the Schmitt-trigger TL-Lottie on slow-transition subsets, the claim that faithful event abstraction is load-bearing would be falsified.

## References

1. MotionGraphicsBench: Evaluating Motion Graphics Generation with Programmatic Verification
