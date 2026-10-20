# Contract-Certified Tool-Program Reuse Across Heterogeneous Visual Backends

## Motivation

OpenVLHarness wraps heterogeneous visual tool backends behind task-level interfaces but stores grown tool-use programs without any compositional contract, so a program grown against one backend can be silently re-invoked against another with different rendering, resolution, or coordinate conventions. The same structural gap recurs across the hierarchical-ReAct, spatial-tool-RL, and prompt-evolution branches: none establishes an equivalence relation or precondition/postcondition layer over the evidence-rendering interface, so 'reuse' there means re-running a program, not preserving its semantics. This is a meta-level failure of compositional reuse, not a single-tree bug: the harness has no typed surface on which cross-backend preservation can even be stated.

## Key Insight

A tool-use program's visual semantics are fully captured by a typed pre/postcondition contract over the evidence-rendering interface, so cross-backend preservation of that program reduces to a decidable contract-refinement check between the source backend's rendering algebra and the target backend's, rather than to re-executing the program and hoping.

## Method

**A. What it is.** TPCL (Tool-Program Contract Layer) is a layer inside OpenVLHarness that, for every grown tool-use program, synthesizes a typed contract `{P_pre, P_post}` over the evidence-rendering interface — a small algebra of evidence types (Image, Region, Mask, Table, Scalar, WebRef) with operations (resize, crop, threshold, OCR, spatial-join, unit-convert) and coordinate/unit metadata. At reuse time against a different backend, TPCL runs a static refinement check plus a bounded dynamic witness search to certify or reject the re-invocation. The load-bearing assumption is stated explicitly: **the rendering algebra is a complete signature for evidence semantics**, i.e., two renderings with the same declared type produce interchangeable evidence. This assumption is not taken on faith: at backend registration (amortized, not at reuse time), TPCL runs a fixed probe battery to derive a behavioral fingerprint for each operation, and `refines` checks that fingerprint rather than the declared type name. The probe battery is specified in Phase 0 below.

**B. How it works.**

```pseudocode
# Phase 0: backend registration — behavioral fingerprint (amortized, once per backend)
probe_battery(B):
    # fixed 8-probe battery per op; each probe is a deterministic input with known ground-truth behavior
    fp = {}
    for op in {resize, crop, threshold, OCR, spatial-join, unit-convert}:
        fp[op] = {
            'interp_kernel':   probe_interp(B, op),      # nearest | bilinear | bicubic | area
            'antialias':       probe_antialias(B, op),   # bool
            'padding_mode':    probe_padding(B, op),     # zero | replicate | reflect | silent-pad
            'coord_origin':    probe_coord(B, op),       # corner | center (half-pixel)
            'unit_scale':      probe_unit(B, op),        # e.g. 1.0, 255.0, 1000.0
            'ocr_norm':        probe_ocr_norm(B, op),    # none | whitespace | unicode-NFKC
            'bbox_convention': probe_bbox(B, op),        # xyxy | xywh | normalized
        }
    return fp

# Phase 1: contract synthesis (at grow time, on backend B_src)
contract_synth(program p, backend B_src, grow_traces T):
    P_pre  = infer_pre(p, B_src)          # typed assertions on input evidence
    P_post = infer_post(p, T)             # assertions on output evidence, from traces
    P_post = abstract_quantifiers(P_post, k=3)   # collapse to 3 witness classes
    return Contract(P_pre, P_post, B_src.fingerprint)

# Phase 2: backend refinement relation (static, over behavioral fingerprints)
refines(B_tgt, B_src, P_pre, P_post):
    for each op o in p.ops:
        if not fp_refines(B_tgt.fingerprint[o], B_src.fingerprint[o]):
            return REJECT(reason=o, fp_diff=diff(B_tgt.fingerprint[o], B_src.fingerprint[o]))
    if not implies(B_tgt.evidence_assumptions, P_pre):
        return REJECT(reason=P_pre)
    return ACCEPT_STATIC

# fp_refines: field-wise check; a field refines if equal, or if target is strictly more precise
# (e.g. antialias=True refines antialias=False for resize; coord_origin must match exactly;
#  padding_mode must match exactly; unit_scale must be equal up to 1e-6)

# Phase 3: dynamic witness search (bounded, only if static ACCEPT)
certify(p, C, B_tgt, budget=12 witnesses):
    W = sample_witnesses(C.P_pre, budget)      # inputs satisfying P_pre
    for w in W:
        y_src = execute(p, w, B_src)
        y_tgt = execute(p, w, B_tgt)
        if not satisfies(y_tgt, C.P_post):     # postcondition violation
            return REJECT(witness=w, y_tgt=y_tgt)
        if distance(y_src, y_tgt) > tau_equiv(witness_class(w)):  # per-class calibrated threshold
            return REJECT(witness=w, drift=distance(y_src,y_tgt))
    return CERTIFIED

# tau_equiv calibration: for each witness class c, from grow traces T, compute the
# distribution of distance(y_src, y_src') over re-executions of p on B_src with
# perturbed inputs; set tau_equiv(c) = 95th percentile of that distribution.
# For exact-match tasks (e.g. OCR), tau_equiv(c) = 0 (any nonzero distance rejects).
```

The contract is stored alongside the program in persistent multimodal memory; reuse against a new backend is gated by `refines` then `certify`. Rejected programs are quarantined with the failing witness attached, which becomes a repair hint for re-growing.

**C. Why this design.** We chose a *typed contract over the evidence-rendering interface* rather than free-form natural-language pre/postconditions because the rendering algebra is the only surface shared by all backends in OpenVLHarness, so typed assertions are the only ones that can be checked without re-running the program; the cost is that contracts cannot express backend-specific quirks outside the algebra. We chose *static subtyping first, dynamic witnesses second* over pure dynamic testing because static refinement rejects most incompatible backends at zero execution cost, accepting the cost that some semantically compatible backends with non-subtype renderings are falsely rejected and must be re-grown. We chose *bounded witness search (12 samples) with a per-witness-class calibrated equivalence threshold* over exhaustive or learned testing because certification must be cheap at reuse time, accepting the cost that rare postcondition violations on unsampled inputs escape certification — this is why the contract carries a `witness_class` annotation so downstream callers know the certification is class-conditional. A domain expert would not describe this as a variant of Toolformer, Gorilla, or standard retrieval-based tool selection: those select tools by descriptor similarity and never state or check a postcondition, whereas TPCL's load-bearing object is the refinement relation between two backends' rendering algebras, which retrieval has no representation for.

**D. Why it measures what we claim.** `fp_refines(B_tgt.fingerprint[o], B_src.fingerprint[o])` measures *cross-backend semantic preservation of operation o* because the behavioral fingerprint (interpolation kernel, antialiasing, padding mode, coordinate origin, unit scale, OCR normalization, bbox convention) is assumed to be a complete operational signature for evidence semantics — i.e., two renderings with matching fingerprints produce interchangeable evidence; this assumption fails when a backend's renderer has behavior outside the probe battery (e.g., a crop that silently pads only for non-square inputs), in which case the fingerprint check reflects probed rather than actual semantics. `satisfies(y_tgt, P_post)` measures *postcondition preservation* because `P_post` was abstracted from grow traces to the witness classes that cover the program's input distribution; this assumption fails when the target backend's failure mode lives in a witness class not sampled, in which case the check reflects class-conditional rather than global preservation. `distance(y_src, y_tgt) <= tau_equiv(witness_class(w))` measures *task-level semantic preservation* because the per-class threshold is calibrated from grow traces to the 95th percentile of same-backend re-execution distance (and set to 0 for exact-match tasks), under the assumption that normalized output distance is monotone in downstream task correctness within a witness class; this assumption fails for tasks whose scoring is discontinuous in the output (e.g., exact-match OCR), in which case the distance reflects perceptual rather than task-level agreement — which is why exact-match classes get tau_equiv=0. Together these three quantities operationalize the motivation's 'preserved pre/postconditions' as a conjunction of fingerprint refinement, class-conditional postcondition satisfaction, and per-class calibrated observational equivalence.

## Contribution

['TPCL, a typed contract layer over the evidence-rendering interface of OpenVLHarness that synthesizes pre/postconditions for grown tool-use programs and certifies their preservation across heterogeneous backends via a static refinement relation plus bounded dynamic witness search.', 'A quarantine-and-repair mechanism in which certification failures emit a concrete failing witness that seeds re-growth, turning cross-backend incompatibility into a structured signal rather than a silent semantic drift.', "A design principle for compositional tool reuse in multimodal agents: reuse safety is a property of a checkable contract on the shared evidence interface, not of the program's textual similarity or its grow-time backend identity."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OpenVLHarness 22-benchmark suite, cross-backend reuse split | Native source/target backend pairs expose refinement failures |
| Primary metric | Reuse success rate: certified programs completing task correctly | Directly measures preserved pre/postconditions under reuse |
| Baseline | Base MLLM with inflated inference budget | Tests whether extra compute substitutes for contract gating |
| Baseline | Retrieval-based tool selection (descriptor similarity) | Tests whether descriptor matching captures rendering semantics |
| Baseline | Ungated program reuse (no TPCL check) | Isolates the contribution of refinement+certification |
| Ablation-of-ours | TPCL with static refinement only (no witness search) | Tests whether dynamic certification catches static-check escapees |
| Ablation-of-ours | TPCL with declared-type subtyping only (no fingerprint probes) | Tests whether behavioral fingerprint is load-bearing |
| Calibration metric | Declared-vs-actual mismatch rate per op across 22 backends | Quantifies completeness assumption of rendering algebra |
| Calibration metric | Manual-vs-synthesized contract agreement rate on 4-6 backends, 3-4 ops | Bounds contract synthesis effort |

### Why this setup validates the claim

The central claim is that gating cross-backend reuse with a typed evidence-rendering contract preserves pre/postconditions that retrieval and ungated reuse cannot. The 22-benchmark suite is the right substrate because it contains genuine backend heterogeneity: the same task can be served by backends whose renderers differ in coordinate conventions, unit scaling, or masking semantics. The base-MLLM-budget baseline tests the sub-claim that raw inference scaling does not recover reuse correctness—if it did, contracts would be redundant. The retrieval baseline tests the sub-claim that descriptor similarity is blind to rendering-algebra mismatch, so it should certify programs that then fail. The ungated-reuse baseline isolates whether the gain comes from gating at all. The static-only ablation isolates whether bounded witness search catches the residual class of semantically compatible but non-subtype renderings, which is precisely the cost the design acknowledges. The declared-type-only ablation isolates whether the behavioral fingerprint (Phase 0) is doing work beyond the declared type name, directly testing the load-bearing assumption. The two calibration metrics quantify the completeness assumption (declared-vs-actual mismatch rate) and bound the synthesis effort (manual-vs-synthesized agreement), turning the unstated assumptions into measured quantities. Reuse success rate is the right metric because it is evaluated at the task level, so it detects both false certifications (program runs but produces wrong evidence) and false rejections (program quarantined unnecessarily)—the two failure modes the design trades against each other.

### Expected outcome and causal chain

**vs. Base MLLM with inflated inference budget** — On a case where a grown OCR-plus-spatial-join program is reused against a backend whose crop silently pads, the base MLLM with more budget still emits the same tool call because extra sampling does not expose the renderer's coordinate drift; it produces a table with misaligned rows. Our method instead rejects or certifies via `refines` on the crop op's fingerprint (padding_mode mismatch) and a witness that exercises the padding region, because the contract carries coordinate metadata and a postcondition on row alignment, so we expect a noticeable gap on the cross-backend reuse subset with parity on single-backend tasks.

**vs. Retrieval-based tool selection** — On a case where two backends expose similarly described `threshold` ops but one returns a mask in normalized coordinates and the other in pixel coordinates, retrieval selects the target backend by descriptor similarity and produces a mask that downstream `spatial-join` misplaces; its mechanism gap is that descriptors carry no rendering-algebra type. Our method instead fails the fingerprint check on `bbox_convention` or `coord_origin` or rejects at certification with a witness showing coordinate mismatch, so we expect retrieval to show high apparent reuse volume but low task success, while our method shows lower reuse volume with markedly higher success on the heterogeneous-backend subset.

**vs. Ungated program reuse** — On a case where a grown program's postcondition requires exact-match OCR output, the ungated baseline re-invokes against a target backend whose OCR normalizes whitespace differently, producing near-miss strings that fail exact-match scoring while the program appears to run cleanly. Our method instead runs `certify` with witnesses sampled from the OCR witness class and rejects on `distance > tau_equiv(c)` (with tau_equiv=0 for exact-match classes) or postcondition violation, attaching the failing witness as a repair hint, so we expect ungated reuse to exhibit silent failures concentrated in exact-match tasks and our method to convert those into quarantines that later re-grow successfully.

**vs. TPCL with static refinement only (no witness search)** — On a case where the target backend's fingerprint matches the source on all probed fields but a rare input triggers a postcondition violation (e.g., a crop that pads only for non-square inputs, outside the probe battery), the static-only ablation accepts the program and it fails silently. Our method runs `certify` and rejects on the failing witness, so we expect the static-only ablation to show higher false-certification rate concentrated on inputs outside the probe battery, while full TPCL converts those into quarantines.

**vs. TPCL with declared-type subtyping only (no fingerprint probes)** — On a case where two backends both declare `crop: Image -> Region` but differ in padding_mode (zero vs. silent-pad) and coord_origin (corner vs. center), the declared-type-only ablation passes the subtype check and certifies a program that then misaligns spatial-join. Our method's fingerprint check rejects on `padding_mode` or `coord_origin` mismatch, so we expect the declared-type-only ablation to show a higher false-certification rate on the heterogeneous-backend subset, with the gap concentrated on ops whose fingerprints diverge despite matching declared types.

### What would falsify this idea

If our gain over ungated reuse were uniform across all benchmark subsets rather than concentrated on tasks whose backends differ in rendering-algebra fingerprints, the claim that refinement over the evidence interface is the load-bearing mechanism would be wrong. Likewise, if the static-only ablation matched full TPCL everywhere, bounded witness search would be doing no work and the certification argument would collapse to fingerprint refinement alone; and if the declared-type-only ablation matched full TPCL everywhere, the behavioral fingerprint would be doing no work and the completeness assumption would be false.

## References

1. OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory
