# Oracle-Certified Multimodal Grading: Replacing LLM Verdicts with Executable and Formal Oracles for Co-Design Bench

## Motivation

Co-Design Bench grades visual and interaction correctness through a judge-then-refute protocol in which both the initial criterion judgment and the adversarial verifier are LLM opinions, so a positive verdict is never grounded in an executable artifact and its soundness is unestablished. The same structural gap recurs in SWE-bench, where correctness is certified only by whatever existing test suite happens to exist, meaning the verdict is sound only relative to the test suite's coverage rather than to the specification the task actually states. In both benchmarks the post-hoc external grader is the sole arbiter, and no mechanism ties a per-criterion verdict to a deterministic check that could falsify it, which is why the field cannot distinguish a genuinely satisfied criterion from a plausible-sounding but wrong LLM judgment.

## Key Insight

A multimodal criterion verdict is sound exactly when it is backed by a deterministic oracle whose observed execution trace on the candidate application is bisimilar to the trace produced by a reference implementation on the same criterion, and this bisimulation can be checked mechanically for rendered-UI diffs, interaction traces, and property assertions without ever consulting an LLM for the final verdict.

## Method

## Oracle-Certified Multimodal Grading (OCMG)

**What it is.** OCMG takes as input a Co-Design Bench task (normalized request, runnable pre-edit application, one hidden acceptance criterion) plus a candidate application produced by an agent, and outputs a per-criterion verdict in {CERTIFIED_PASS, CERTIFIED_FAIL, UNRESOLVABLE} together with a machine-checkable certificate. The LLM judge and adversarial verifier are demoted from arbiters to failure-localizers: they propose which oracle to invoke and where the failure lies, but the verdict is emitted only by a deterministic oracle.

**Load-bearing assumption (stated explicitly).** OCMG's soundness claim is conditional on the criterion oracle specification (COS) being a sound and complete operationalization of the natural-language criterion: a deterministic oracle passing on the COS implies the criterion is genuinely satisfied. The COS is authored by an LLM and frozen, so its correctness is not certified by construction. We therefore make COS correctness itself a certified object via a mutation-adequacy gate (Phase 0 below): a COS is admitted only if it passes the reference implementation and fails every mutant in a per-criterion mutant set. Criteria whose COS fails the gate are routed to UNRESOLVABLE, and all reported soundness is conditioned on COS-adequacy.

**How it works.** For each criterion we first compile the criterion's natural-language claim into a *criterion oracle specification* (COS) — a triple (setup, action script, assertion set) — using a one-time offline pass by an LLM that is then frozen and versioned. The COS is the only place an LLM touches grading, and it is authored against the *reference implementation* (a human-written solution kept hidden from agents), not against the candidate. Each COS is then subjected to a mutation-adequacy gate before it may emit any verdict.

```pseudocode
# Phase 0: COS mutation-adequacy gate (offline, once per criterion)
function admit_cos(criterion c, reference R, COS spec S, mutant_set M_c):
    # M_c = 5 hand-authored mutants per criterion, each breaking exactly one
    # aspect of c (e.g. wrong handler, wrong color, missing validation)
    if not execute_oracles(R, S).all_pass:
        return REJECT   # COS must pass the reference
    for m in M_c:
        if execute_oracles(m, S).all_pass:
            return REJECT   # COS must fail every mutant
    return ADMIT

function grade(criterion c, candidate app A, reference impl R, COS spec S):
    if not S.admitted:
        return UNRESOLVABLE, certificate(reason='COS_FAILED_MUTATION_GATE')

    # Phase 1: deterministic execution on reference and candidate
    trace_R = execute(R, S.setup, S.action_script)   # browser + DOM + event log
    trace_A = execute(A, S.setup, S.action_script)

    # Phase 2: three oracle families, all deterministic
    verdicts = []
    # (a) rendered-UI diff oracle: perceptual hash + pixel-region diff
    ui = pixel_diff(render(trace_R.final_state), render(trace_A.final_state),
                    region_mask=S.assertion_regions, tol=2/255)
    verdicts.append(('ui', ui.max_region_delta <= 2/255))
    # (b) interaction-trace replay oracle: bisimulation on event sequences
    inter = trace_bisim(trace_R.event_seq, trace_A.event_seq,
                        obs_fn=S.observable_projection, max_depth=len(S.action_script))
    verdicts.append(('interaction', inter.is_bisimilar))
    # (c) property-based assertion oracle: Hypothesis-style, 200 samples
    props = run_property_tests(A, S.assertions, n_samples=200, seed=0)
    verdicts.append(('property', props.all_pass))

    # Phase 3: LLM localizer runs ONLY on failing oracles, never on verdict
    if all(v for _, v in verdicts):
        return CERTIFIED_PASS, certificate(verdicts, traces)
    failing = [name for name, v in verdicts if not v]
    localized = llm_localize(c, A, failing, traces)   # returns a diff region
    # Phase 4: adversarial verifier tries to refute CERTIFIED_PASS only
    if not failing and llm_verifier_refutes(c, A, traces):
        return UNRESOLVABLE, certificate(verdicts, traces, refutation=True)
    return CERTIFIED_FAIL, certificate(verdicts, traces, localized)
```

The certificate is a JSON record containing the two traces, the pixel-diff region map, the bisimulation witness (or counterexample pair), the property-test seed and failing sample, and the mutation-gate result (which mutants were killed), replayable by any third party without an LLM.

**Why this design.** We chose to compile criteria into a frozen COS authored against a reference implementation rather than let the judge free-form the check at grading time, because a per-grading-time LLM-authored check cannot be audited or replayed and its verdicts are not comparable across candidates, accepting the cost that COS authoring is a one-time human-in-the-loop expense and that criteria whose COS cannot be compiled fall into UNRESOLVABLE. We added the mutation-adequacy gate because the COS is itself an LLM artifact and its correctness would otherwise be an unverified assumption; the gate converts 'the COS is correct' from an assumption into a per-criterion certified property, accepting the cost of 5 hand-authored mutants per criterion. We chose bisimulation on the projected event sequence over exact trace equality because exact equality is brittle to irrelevant timing and DOM-node reordering, accepting the cost that the observable projection must be specified per criterion family (we use a fixed projection: visible text, focus target, and DOM role of the active element). We chose property-based testing with 200 Hypothesis samples over a single hand-written assertion because the latter is exactly the SWE-bench test-suite-coverage limitation reappearing inside the criterion, accepting the cost of a fixed random seed budget per criterion. We chose to restrict the LLM verifier to refuting only CERTIFIED_PASS and never to overturning CERTIFIED_FAIL, because the motivation's failure mode is unsound positive verdicts, accepting the cost that a wrong CERTIFIED_FAIL must be caught by the UNRESOLVABLE bucket rather than by the verifier. A domain expert would not describe OCMG as a variant of the Co-Design Bench judge-then-refute protocol: in that protocol the verifier's opinion *is* the verdict, whereas here the verifier is a refutation attempt that can only move a verdict from PASS to UNRESOLVABLE, and the PASS itself is certified by bisimulation and pixel-diff oracles that never consult an LLM.

**Why it measures what we claim.** `trace_bisim` measures *interaction correctness* because the assumption is that two applications are interaction-equivalent on a criterion iff their observable projections agree at every step of the action script; this assumption fails when the action script is shorter than the behavior the criterion implicitly requires, in which case bisimulation reflects only prefix-equivalence and a candidate that diverges after the script ends is falsely certified. `pixel_diff` with region masking measures *visual correctness* because the assumption is that the criterion's assertion regions are exactly the pixels the criterion constrains; this assumption fails when a candidate achieves the same visual outcome through an unintended DOM structure (e.g., an image standing in for a styled div), in which case pixel-diff reflects surface appearance rather than the structural change the criterion intended. `run_property_tests` measures *functional correctness under input variation* because the assumption is that the assertion set is a complete characterization of the criterion over the sampled input distribution; this assumption fails when the criterion constrains behavior outside the sampled distribution, in which case property pass reflects in-distribution conformance only. The mutation-adequacy gate measures *COS soundness* because the assumption is that a COS that passes the reference and kills all 5 mutants is a faithful operationalization of the criterion; this assumption fails when the mutant set under-covers the criterion's failure modes, in which case the gate certifies a COS that is bisimilar to R while both diverge from the criterion. The UNRESOLVABLE bucket measures *the fraction of criteria for which no sound oracle exists* because the assumption is that COS compilation failure and mutation-gate rejection are the only sources of unsoundness; this assumption fails when a COS passes the gate but is still wrong, in which case UNRESOLVABLE reflects residual COS-authoring error rather than genuine unverifiability.

## Contribution

(1) OCMG, a grading framework that replaces LLM verdicts with three deterministic oracle families (region-masked pixel diff, observable-projection bisimulation, seeded property-based assertions) and emits a replayable certificate per criterion, with the LLM demoted to failure localization and PASS-only refutation. (2) The finding that per-criterion verdict soundness is separable from per-criterion verdict completeness: criteria partition into a certified subset (sound verdicts) and an UNRESOLVABLE subset, and the certified fraction is itself a measurable property of the benchmark rather than of the model. (3) A criterion-oracle-specification (COS) compilation procedure and a certificate format that makes Co-Design Bench verdicts independently auditable without re-running any LLM.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Stratified subset of Co-Design Bench: 72 tasks (36 ME, 36 RE), sampled to span the three oracle families (ui, interaction, property) with 24 tasks per family | Runnable scope; spans tracks and oracle families |
| Primary metric | Soundness rate: fraction of CERTIFIED_PASS verdicts that are truly correct, reported overall and conditioned on COS-adequacy | Directly measures the failure mode the motivation names |
| Secondary metric | UNRESOLVABLE rate per criterion family and per track (ME/RE) | First-class result: how often no sound oracle exists |
| Secondary metric | COS-authoring cost: human-hours per admitted COS, and mutation-gate rejection rate | Lets community judge soundness gain vs. overhead |
| Baseline | Co-Design Bench judge-then-refute protocol (strongest endpoint) | The protocol OCMG claims to improve on |
| Baseline | Judge-only LLM grading (no verifier) | Isolates whether certification, not the verifier, drives soundness |
| Baseline | Exact-trace-equality oracle (no bisimulation) | Tests whether the projection choice matters or is incidental |
| Baseline | Metamorphic-testing oracle (fixed metamorphic relations per criterion family, no COS) | Tests whether oracle substitution alone, without COS compilation, suffices |
| Baseline | Differential-testing oracle (compare candidate vs. reference on random inputs, no COS) | Tests whether reference-anchored differential testing suffices |
| Ablation-of-ours | OCMG with LLM verifier allowed to overturn CERTIFIED_FAIL | Tests the PASS-to-UNRESOLVABLE-only restriction |
| Ablation-of-ours | OCMG without the mutation-adequacy gate (all compiled COS admitted) | Tests whether the gate is doing causal work |
| Ablation-of-ours | OCMG with observable projection replaced by exact trace equality (already listed as baseline; reported as ablation for projection sensitivity) | Tests projection choice |

**COS-authoring protocol.** Each COS is authored by one LLM pass (frozen model, temperature 0) against the hidden reference implementation, then reviewed by one of two human annotators who either accept it or edit it. A second annotator independently labels each COS as correct/incorrect against the criterion; we report Cohen's κ on this binary label. Mutant sets (5 per criterion) are authored by the same annotators. Human-hours per admitted COS are logged.

**Projection validation.** For a sample of 24 criteria (8 per oracle family), two human annotators independently judge whether the reference and a candidate are interaction-equivalent on the criterion; we report the disagreement rate between human judgment and `trace_bisim` under the fixed projection (visible text, focus target, active DOM role).

### Why this setup validates the claim

The central claim is that demoting the LLM from arbiter to failure-localizer, and emitting verdicts only from deterministic oracles whose COS has passed a mutation-adequacy gate, converts unsound positive verdicts into either certified-correct passes or honest UNRESOLVABLEs. That claim is falsifiable only if we measure *soundness of positive verdicts* rather than raw pass rate, since a method can raise pass rate by being more permissive. Co-Design Bench supplies hidden expert-authored criteria, so a CERTIFIED_PASS can be checked against ground truth without the grader seeing it. The 72-task stratified subset makes the study runnable and lets us report UNRESOLVABLE and soundness per oracle family and per track, so the COS-error confound is measured (via the mutation-gate rejection rate and the no-gate ablation) rather than assumed away. The judge-then-refute baseline tests the sub-claim that verifier-as-arbiter admits unsound passes; the judge-only baseline tests whether the verifier contributes anything at all; the exact-trace-equality baseline tests whether bisimulation is doing real work; the metamorphic- and differential-testing baselines test whether COS compilation adds anything beyond oracle substitution. The no-gate ablation tests whether the mutation-adequacy gate is the mechanism that makes certification sound. Soundness rate is the right metric because the predicted effect is a *shift in the composition of passes*, not a uniform lift; a permissive grader would show high pass rate and low soundness, which this metric separates.

### Expected outcome and causal chain

**vs. Co-Design Bench judge-then-refute protocol** — On a case where a candidate renders the correct final UI but wires the interaction to a stale event handler, the baseline's verifier reads the criterion, sees the right pixels, and lets the pass stand, because the verifier's opinion *is* the verdict and it never replays the event sequence. Our method replays the action script on both reference and candidate and the bisimulation witness diverges at the handler step, so the verdict is CERTIFIED_FAIL with a counterexample pair. We expect a noticeable soundness gap concentrated on interaction-heavy criteria, with near-parity on purely visual criteria.

**vs. Judge-only LLM grading** — On a case where the criterion constrains a property over many inputs (e.g. "the filter stays consistent as rows are added"), the judge inspects one or two plausible states, finds them consistent, and certifies, because a free-form judge samples no input distribution. Our method runs 200 seeded property samples against the compiled assertion set and returns a failing sample in the certificate. We expect the gap to concentrate on criteria whose COS compiles to a property oracle, with parity where the criterion is a single static assertion.

**vs. Exact-trace-equality oracle** — On a case where a candidate reorders DOM nodes or attaches an equivalent listener later in the same tick, exact equality reports a mismatch and the criterion is marked failed, because raw event sequences are brittle to irrelevant ordering. Our method's fixed observable projection (visible text, focus target, active DOM role) makes the two traces bisimilar, so the candidate is correctly certified. We expect this baseline to show *lower* pass rate alongside higher soundness — a diagnostic pattern confirming bisimulation buys recall without buying unsoundness.

**vs. Metamorphic-testing oracle** — On a case where the criterion constrains a specific visual outcome (e.g. a particular color on a particular element), a fixed metamorphic relation (e.g. "changing the input changes the output monotonically") does not pin the exact color, so the metamorphic oracle passes a candidate with the wrong color. Our method's pixel-diff oracle with the criterion's assertion regions fails it. We expect the metamorphic baseline to show lower soundness on visual criteria and comparable soundness on property criteria, confirming that COS compilation (which localizes the assertion to the criterion) adds beyond generic metamorphic relations.

**vs. Differential-testing oracle** — On a case where the criterion constrains behavior on a specific input the random differential-testing distribution rarely samples, the differential oracle sees agreement on sampled inputs and passes, because it has no criterion-specific input distribution. Our method's COS-authored assertion set and 200-sample property budget target the criterion's input space. We expect the differential baseline to show lower soundness on property criteria and comparable soundness on interaction criteria where the action script is the input.

**Ablation: OCMG with LLM verifier allowed to overturn CERTIFIED_FAIL** — On a case where the deterministic oracles correctly fail a candidate but the verifier's opinion is that the failure is spurious, the ablation flips the verdict to PASS, reintroducing the unsound-positive channel. We expect a measurable soundness drop relative to full OCMG, confirming the defended asymmetry.

**Ablation: OCMG without the mutation-adequacy gate** — On a case where the LLM-authored COS under-approximates the criterion (e.g. asserts only the final color, not the hover state), the no-gate variant admits the COS and certifies a candidate that fails the hover requirement, because the COS is bisimilar to R while both diverge from the criterion. We expect the no-gate variant to show lower soundness and a lower UNRESOLVABLE rate, confirming the gate converts COS-authoring error into honest UNRESOLVABLEs.

### What would falsify this idea

If OCMG's soundness gain over the judge-then-refute baseline is uniform across criteria rather than concentrated on the interaction- and property-oracle subsets where the predicted failure mode lives, then certification is not doing the causal work and the improvement is an artifact of the harness. A second falsifier: if the no-gate ablation shows no soundness drop, the mutation-adequacy gate is unmotivated; a third: if the ablation allowing the verifier to overturn CERTIFIED_FAIL shows no soundness drop, the defended asymmetry is unmotivated.

## References

1. Co-Design Bench: Multimodal Collaborative Coding Evaluation for Agentic Models
2. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
