# Cross-Format Semantic Equivalence for Motion Graphics via Temporal Constraint Normalization

## Motivation

Motion graphics verification is currently format-specific: MotionGraphicsBench constructs deterministic verification programs that operate on Lottie-rendered animation state, so a semantically identical edit expressed in an imperative JS engine (e.g., a requestAnimationFrame loop with GSAP-style tweens) cannot be checked by the same program and cannot be compared to its Lottie counterpart. The root cause is structural: verification programs are written against a concrete animation object's property paths and frame semantics, so the same visual intent produces different observable state traces depending on the authoring format, which means semantic preservation of a motion edit is never actually tested across formats — only within one. This is the shared residual gap of all branches: representation-independent semantic preservation.

## Key Insight

If every motion graphic is compiled to a common set of timestamped visual-event constraints over (object identity, property, value) tuples sampled on a shared time grid, then two animations are semantically equivalent exactly when their constraint sets are mutually satisfiable — a property that is preserved by construction under any format-preserving translator, independent of the authoring engine.

## Method

**What it is.** We introduce TEMPO (Temporal Event Model for Programmatic Output), a format-agnostic semantic model in which a motion graphic is a finite set of temporal constraints over visual events, plus two translators — `LottieToTEMPO` and `JSToTEMPO` — that map Lottie JSON and imperative JS animation scripts into TEMPO. Given a motion-edit prompt, a TEMPO-level verifier checks requirements directly on the constraint set, so the same check runs on both formats and cross-format equivalence becomes a satisfiability question between two TEMPO models.

**Load-bearing assumption (stated explicitly).** TEMPO assumes that two animations are perceptually equivalent if and only if their TEMPO constraint sets are mutually satisfiable under a per-property perceptual tolerance and a continuous-time event representation. This is the sampling/aliasing equivalence claim: it holds only if the event extraction preserves sub-frame timing and the tolerance is calibrated to human just-noticeable differences (JNDs) per property. It fails when (i) a difference is purely sub-frame and the grid aliases it, or (ii) the tolerance is a fixed absolute value that violates Weber's law for low-magnitude properties (e.g., opacity near zero, color near gamut boundary). We therefore make the grid continuous for CHANGE/ORDER events and replace the fixed tolerance with per-property JND-calibrated tolerances, as specified below.

**How it works.**

```pseudocode
# Phase 1: Event extraction (format-agnostic surface)
def extract_events(anim, format):
    if format == "lottie":
        return lottie_walker(anim)      # resolves keyframe interpolation -> (t, obj, prop, value)
    if format == "js":
        return js_instrumenter(anim)    # wraps tween/easing calls, records (t, obj, prop, value)

# Phase 2: Temporal constraint construction
def build_TEMPO(events, grid_dt=1/60, tol_map=None):
    # tol_map: per-property JND tolerance dict; if None, use defaults below.
    # Default tol_map (normalized units):
    #   opacity: Weber-scaled, tol_opacity(v) = max(0.02 * max(v, 0.05), 0.005)
    #   position (x,y): 0.002 normalized units
    #   scale: 0.005 normalized units
    #   rotation: 0.5 degrees
    #   color (RGB): CIEDE2000 ΔE = 2.0 (just-noticeable difference)
    # CHANGE and ORDER events retain continuous timestamps (no grid snapping).
    # HOLD and RATE constraints are sampled on grid_dt=1/60 s for interval bounds.
    # Emit constraints of four kinds:
    #   HOLD(obj, prop, v, [t0,t1])   -- value constant on interval
    #   CHANGE(obj, prop, v0, v1, t)  -- transition at continuous timestamp t
    #   ORDER(e1, e2)                 -- event e1 strictly precedes e2
    #   RATE(obj, prop, lo, hi, [t0,t1]) -- bounded rate of change
    return ConstraintSet(snap_holds_and_rates(events, grid_dt), continuous_changes_and_orders(events))

# Phase 3: Requirement verification (runs identically on both formats)
def verify(T, requirement):
    # requirement is a human-readable predicate over TEMPO constraints
    return requirement.check(T)   # returns pass/fail + witness constraint

# Phase 4: Cross-format equivalence
def equivalent(T_lottie, T_js, tol_map):
    # Mutual satisfiability with per-property JND tolerances.
    # For each constraint c in T_lottie, check has_witness(c, T_js, tol_map[c.prop]);
    # and symmetrically for each c in T_js against T_lottie.
    # has_witness for CHANGE/ORDER uses continuous timestamps with a timing tolerance
    # of 1/120 s (half a 60 Hz frame) to avoid aliasing sub-frame differences.
    return all(has_witness(c, T_js, tol_map[c.prop]) for c in T_lottie) and \
           all(has_witness(c, T_lottie, tol_map[c.prop]) for c in T_js)
```

Hyperparameters: `grid_dt = 1/60 s` for HOLD/RATE interval sampling (matches Lottie's default 60 fps and typical rAF cadence); CHANGE and ORDER events use continuous timestamps with a timing tolerance of `1/120 s` to avoid grid aliasing of sub-frame easing differences. Per-property tolerances `tol_map` are JND-calibrated: opacity uses a Weber-scaled tolerance `max(0.02 * max(v, 0.05), 0.005)`; position and scale use `0.002` and `0.005` normalized units; rotation uses `0.5°`; color uses CIEDE2000 `ΔE = 2.0`. Rate bounds `[lo, hi]` are derived per-property from the source animation's observed extrema, not hand-set.

**Why this design.** We chose a *constraint set* over a *canonical animation object* because a canonical object forces a lossy choice of one interpolation scheme (Lottie's bezier vs. JS easing functions do not agree), accepting the cost that some sub-frame timing detail is abstracted away. We chose *mutual satisfiability with witnesses* over *exact constraint-set equality* because exact equality is brittle to floating-point and to semantically irrelevant ordering differences, accepting the cost that equivalence becomes a tolerance-parameterized judgment rather than a binary identity. We chose *instrumentation of the JS engine* over *static analysis of JS source* because static analysis cannot resolve runtime-computed tween targets, accepting the cost that the translator requires executing the script in a sandbox. We chose *continuous CHANGE/ORDER timestamps* over grid-snapped events because a 60 Hz grid aliases sub-frame easing differences that are perceptually visible as stutters or pops, accepting the cost that the constraint set is no longer a finite grid sample. We chose *per-property JND tolerances* over a fixed absolute tolerance because a fixed tolerance violates Weber's law for low-magnitude properties (opacity near zero, color near gamut boundary), accepting the cost that the tolerance map must be calibrated from human judgments. A domain expert would not describe TEMPO as a variant of the verification-program approach in MotionGraphicsBench because that work writes checks against one concrete animation object's property paths — TEMPO's contribution is the constraint algebra that makes the *same* check well-defined across two engines, which no format-specific verifier provides.

**Why it measures what we claim.** The mutual-witness satisfiability check measures *cross-format semantic equivalence* because we assume that any visually observable difference between two animations must manifest as a violated HOLD, CHANGE, ORDER, or RATE constraint under continuous CHANGE/ORDER timestamps and per-property JND tolerances; this assumption fails when a difference is purely sub-frame and below the 1/120 s timing tolerance (e.g., two easings that differ only within one 8 ms interval), in which case the check reflects timing-resolution equivalence rather than perceptual equivalence. The per-property JND tolerances measure *semantic* rather than *syntactic* agreement because we assume value differences below the calibrated JND are imperceptible; this assumption fails for high-contrast properties like opacity near zero, where the Weber-scaled tolerance `max(0.02 * max(v, 0.05), 0.005)` may still under-report divergence if the human JND is smaller than 0.005, in which case the check under-reports divergence. The ORDER constraints measure *temporal intent preservation* because we assume the relative sequencing of events, not their absolute timestamps, carries the prompt's temporal requirements; this fails when a prompt specifies absolute timing (e.g., 'at 2 seconds'), in which case ORDER alone is insufficient and the verifier must fall back to timestamped HOLD intervals.

## Contribution

(1) TEMPO, a format-agnostic semantic model of motion graphics as timestamped temporal constraints over visual events, with a constraint algebra supporting HOLD/CHANGE/ORDER/RATE and a mutual-witness equivalence relation. (2) Two translators — LottieToTEMPO and JSToTEMPO — that map Lottie JSON and imperative JS animation scripts into the same constraint surface, enabling a single verification program to run unchanged on both formats. (3) A cross-format equivalence checker that reports per-constraint witnesses and divergence witnesses, providing a diagnostic of *where* two format renderings of the same edit semantically diverge rather than a binary pass/fail.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MotionGraphicsBench 180-animation balanced test set | Paired Lottie/JS renders with human judgments available |
| Primary metric | Agreement with human judgments on motion-edit prompts | Directly tests whether TEMPO checks track perception |
| Baseline | GPT-6 Astra VLM judge | Tests whether visual-judge alone suffices |
| Baseline | Format-specific verification programs (MotionGraphicsBench) | Tests whether constraint algebra beats per-format checks |
| Baseline | Exact constraint-set equality (no witnesses) | Tests whether mutual satisfiability is the right equivalence |
| Ablation-of-ours | TEMPO without RATE constraints | Tests whether rate bounds carry measurable signal |
| Ablation-of-ours | TEMPO with fixed global tol=1e-3 (no JND calibration) | Tests whether per-property JND tolerances are necessary |
| Ablation-of-ours | TEMPO with grid-snapped CHANGE/ORDER (no continuous timestamps) | Tests whether sub-frame timing matters |

### Why this setup validates the claim

The central claim is that a shared constraint algebra makes the *same* check well-defined across two engines, so verification judgments should track human judgments more closely than any single-format mechanism. The VLM judge baseline tests the sub-claim that perception-only evaluation is insufficient when a prompt specifies temporal intent (ordering, holds, rates) that a single frame cannot expose. The format-specific verification baseline tests the sub-claim that per-format checks fragment: it can only be written against one concrete animation object, so on the JS side its coverage silently drops. The exact-equality baseline tests the sub-claim that mutual satisfiability with witnesses is the correct equivalence notion rather than brittle set identity. The RATE ablation removes RATE constraints, isolating whether bounded rate-of-change is doing real work or is redundant with HOLD/CHANGE. The fixed-tolerance ablation tests whether the per-property JND calibration is necessary by replacing it with the original fixed `tol=1e-3`; if agreement drops, the calibration is load-bearing. The grid-snapped ablation tests whether continuous CHANGE/ORDER timestamps are necessary by snapping them to the 60 Hz grid; if agreement drops on sub-frame-sensitive prompts, the continuous representation is load-bearing. Human agreement is the right primary metric because the claim is about semantic equivalence and temporal-intent preservation — both are perceptual notions, and only human judgments can arbitrate whether a sub-frame or sub-tolerance difference is actually observable. A metric like constraint-count or syntactic match would be circular: it would reward TEMPO for agreeing with itself.

### Expected outcome and causal chain

**vs. GPT-6 Astra VLM judge** — On a case where a prompt says 'the logo fades in before the tagline slides up,' the VLM judge inspects sampled frames and often rates a reversed-order render as acceptable because both elements are present and the frames look plausible in isolation. Its mechanism gap is that frame-level perception does not resolve strict precedence. Our method instead checks the ORDER constraint directly on the TEMPO set, so a reversed render produces a violated ORDER with a witness constraint. We expect a noticeable agreement gap concentrated on prompts with explicit sequencing or 'before/after' language, with near-parity on prompts about static appearance.

**vs. Format-specific verification programs** — On a case where the same motion is authored in Lottie and in a JS tween script, the format-specific baseline writes its check against one object's property paths and, when applied to the other format, either fails to resolve the path or silently passes on a missing property. Its mechanism gap is that checks are bound to one concrete representation. Our method instead translates both into TEMPO and runs the identical requirement check, so cross-format divergence surfaces as a failed mutual-witness match. We expect the gap to be largest on cross-format pairs and to shrink toward parity on single-format cases where the baseline's paths happen to align.

**vs. Exact constraint-set equality** — On a case where Lottie's bezier interpolation and the JS easing function produce the same visual motion but differ in floating-point samples and in irrelevant event ordering, exact equality flags the pair as non-equivalent. Its mechanism gap is brittleness to numerically meaningless differences. Our method instead requires each constraint to have a tolerance-bounded witness in the other set, so semantically identical motion matches while genuine divergences still fail. We expect the baseline to produce a high false-divergence rate on equivalent pairs and our method to preserve true divergences, showing a clean separation on the equivalence subset.

**vs. TEMPO without RATE constraints** — On a case where a prompt says 'the element accelerates smoothly into place,' removing RATE constraints means the verifier only sees HOLD and CHANGE events and cannot distinguish a smooth ease-in from a linear or stepped motion that reaches the same endpoint. Its mechanism gap is that bounded rate-of-change is the only constraint that encodes acceleration profile. Our method with RATE constraints checks that the observed rate stays within the per-property bounds derived from the source animation, so a stepped motion that violates the upper rate bound produces a failed RATE witness. We expect the ablation to show lower agreement on prompts with 'smooth,' 'accelerate,' or 'ease' language, with near-parity on prompts that only specify endpoints or ordering.

**vs. TEMPO with fixed global tol=1e-3** — On a case where opacity fades from 0.0 to 0.1, a fixed absolute tolerance of 1e-3 is below the human JND near zero (which is larger in relative terms), so the fixed-tolerance variant may flag imperceptible differences as divergences, while near mid-range opacity it may miss visible differences. Its mechanism gap is that a single absolute tolerance cannot satisfy Weber's law across the property range. Our method uses the Weber-scaled opacity tolerance `max(0.02 * max(v, 0.05), 0.005)`, which adapts to the value magnitude. We expect the fixed-tolerance ablation to show lower agreement on prompts involving low-opacity or high-contrast color changes, with near-parity on mid-range position or scale changes.

**vs. TEMPO with grid-snapped CHANGE/ORDER** — On a case where two easings differ only within a single 16 ms interval (e.g., a stutter that is visible as a pop), grid-snapping to 60 Hz aliases the difference and the CHANGE/ORDER witnesses match, so the variant reports equivalence. Its mechanism gap is that the 60 Hz grid is not a perceptual Nyquist rate for high-contrast temporal structure. Our method retains continuous CHANGE/ORDER timestamps with a 1/120 s timing tolerance, so the sub-frame difference produces a timing mismatch that fails the witness check. We expect the grid-snapped ablation to show lower agreement on prompts with rapid or high-contrast motion, with near-parity on slow, low-contrast motions.

### What would falsify this idea

If TEMPO's agreement gain over the VLM judge and format-specific baselines were uniform across all prompt subsets rather than concentrated on prompts with explicit ordering, hold, or rate requirements, the claim that constraint algebra captures temporal intent would be wrong. Likewise, if removing RATE constraints or replacing per-property JND tolerances with a fixed global tolerance changed nothing on any subset, the constraint vocabulary and calibration would be over-claimed.

## References

1. MotionGraphicsBench: Evaluating Motion Graphics Generation with Programmatic Verification
