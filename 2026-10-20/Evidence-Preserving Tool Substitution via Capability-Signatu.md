# Evidence-Preserving Tool Substitution via Capability-Signature Equivalence in Multimodal Agent Harnesses

## Motivation

Existing multimodal tool-use harnesses such as OpenVLHarness assume that every tool backend named in a hierarchical plan will remain available and error-free, so a single backend failure collapses the plan's visual evidence contract and forces the agent to restart or hallucinate. This failure is structural: the harness exposes task-level interfaces but never records which capability signature each interface satisfies, so it cannot know that a substitute tool would preserve the evidence the plan's downstream steps depend on. The same gap recurs across hierarchical-ReAct planners, spatial-tool RL policies, and prompt-evolution branches — none establishes an equivalence relation over tools, so no branch can close the action set under substitution.

## Key Insight

Two tool backends are substitutable exactly when their capability signatures agree on the evidence-contract fields that the downstream plan steps consume, so an equivalence class over signatures — not a similarity score over descriptions — is the invariant that makes failure substitution evidence-preserving.

## Method

**What it is.** ASER (Action-Set Equivalence Relation) is a harness-level layer that, for each tool backend, computes a capability signature from its interface and a declared evidence-contract record, partitions backends into equivalence classes by signature agreement on the contract fields a plan step consumes, and applies a failure-substitution closure operator that rewrites a failed plan step to an equivalent backend while asserting the evidence contract is unchanged. Inputs: a hierarchical plan whose leaves are tool calls annotated with required evidence-contract fields, and a registry of backends with interfaces. Outputs: a rewritten plan with substituted backends plus a substitution certificate per rewrite.

**Load-bearing assumption (stated explicitly).** ASER assumes that a backend's declared evidence-contract metadata (`evidence_fields`, `coord_frame`, `output_artifact`) is a faithful, runtime-accurate summary of the fields it actually emits, and that downstream plan steps consume only the fields listed in their `required_evidence` annotation. Under this assumption, set-equality on declared fields is equivalent to evidence preservation. The method below includes a runtime conformance monitor that checks this assumption per backend and demotes backends whose emitted fields drift from their declaration, and a consumer-instrumentation step that derives `R[p]` from logged field reads rather than trusting plan annotations alone.

**How it works.**

```pseudocode
# Phase 0: runtime conformance monitor (per backend b, over calibration calls)
# Calibration set: 64 calls per backend drawn from the harness's recorded tool-call log,
# stratified by input modality; re-run every 200 live calls or on registry update.
for b in registry:
    emitted = [parse_output_schema(call(b, x)) for x in calibration_inputs[b]]  # set of field names present in output artifact
    b.drift_rate = mean_over_calls( 1[ declared_fields(b) - emitted_fields(call) != set() ] )
    b.conformant = (b.drift_rate <= 0.05)   # 5% tolerance; backends above are demoted
    if not b.conformant:
        b.evidence_fields = intersection_over_calls(emitted_fields(call))  # shrink to fields actually emitted in all calibration calls

# Phase 1: signature extraction (per backend b)
for b in registry:
    S[b] = {
      input_modalities:  set(b.interface.input_types),      # e.g. {image, text}
      output_artifact:   b.interface.output_type,            # e.g. image_region_set
      evidence_fields:   set(b.contract.declared_fields),    # e.g. {bbox, label, confidence}; shrunk by Phase 0 if non-conformant
      determinism:       b.contract.determinism_class,       # {deterministic, sampled}
      coord_frame:       b.contract.coordinate_frame,        # e.g. pixel_xy
      cost_tier:         b.contract.cost_tier,               # ordinal
      conformant:        b.conformant                        # from Phase 0
    }

# Phase 2: equivalence relation (per plan step p with required fields R[p])
# R[p] is derived empirically: downstream consumer of p is instrumented to log field reads
# over the same 64-call calibration set; R[p] = union of logged reads, unioned with the
# plan annotation. If logged reads include fields outside the annotation, those are added.
def equivalent(b1, b2, R):
    return (S[b1].conformant and S[b2].conformant
        and S[b1].input_modalities  >= R.input_modalities
        and S[b1].output_artifact   == S[b2].output_artifact
        and (S[b1].evidence_fields & R) == (S[b2].evidence_fields & R)
        and S[b1].coord_frame       == S[b2].coord_frame)
    # determinism and cost_tier are NOT in the equivalence test;
    # they are tie-breakers inside a class (Phase 3).

# Phase 3: failure-substitution closure operator
def close(plan, failed_step, registry):
    R = failed_step.required_evidence   # annotation ∪ logged consumer reads (Phase 2)
    cls = [b for b in registry if equivalent(failed_step.backend, b, R)
                             and b.available]
    if not cls: return FAIL(reason="no equivalent backend", R=R)
    # deterministic tie-break: prefer same determinism_class, then lowest cost_tier,
    # then lexicographic backend id (fully specified, no learned router)
    cls.sort(key=lambda b: (S[b].determinism != S[failed_step.backend].determinism,
                            S[b].cost_tier, b.id))
    sub = cls[0]
    cert = {step: failed_step.id, from: failed_step.backend, to: sub.id,
            preserved_fields: R & S[sub].evidence_fields,
            dropped_fields:   R - S[sub].evidence_fields,
            sub_drift_rate:   S[sub].drift_rate}
    assert cert.dropped_fields == set()   # closure guarantees empty
    return rewrite(plan, failed_step, sub), cert

# Phase 4: harness loop
for step in plan.leaves:
    try: obs = call(step.backend, step.args)
    except BackendError:
        plan, cert = close(plan, step, registry)
        obs = call(plan.step(step).backend, step.args)
    # post-call conformance check: validate emitted fields against R
    if (R - emitted_fields(obs)) != set():
        registry.demote(plan.step(step).backend)   # contract violation; backend leaves its class
        plan, cert = close(plan, step, registry)
        obs = call(plan.step(step).backend, step.args)
    memory.write(obs, contract=cert.preserved_fields)
```

**Why this design.** We chose a signature-equality equivalence relation over a description-embedding nearest-neighbor matcher because the latter (the mechanism used by Gorilla-style tool retrieval) cannot certify that a substitute preserves the specific evidence fields a downstream step consumes, accepting the cost that ASER only substitutes among backends whose interfaces have been declared with contract metadata. We chose a deterministic tie-break (determinism class, then cost tier, then id) over a learned router because a router would reintroduce the very failure mode — an unavailable or miscalibrated component — that substitution is meant to survive, accepting the cost that ASER cannot exploit soft preferences among equivalent backends. We chose to exclude determinism and cost from the equivalence test and keep them as intra-class tie-breakers because folding them into equivalence would fragment classes and shrink the substitutable set exactly when failures occur, accepting the cost that a sampled backend may substitute for a deterministic one when no deterministic equivalent is available. We added a runtime conformance monitor (Phase 0) and consumer instrumentation (Phase 2) because the declared-field assumption is load-bearing and can fail via contract/implementation drift or implicit consumption; the monitor shrinks a backend's declared field set to what it actually emits over a 64-call calibration set and demotes backends whose drift rate exceeds 5%, and the instrumentation derives `R[p]` from logged field reads rather than trusting annotations alone. A domain expert would not describe ASER as a variant of Gorilla or Toolformer because those systems rank tools by relevance to a query; ASER partitions tools by a contract-equality invariant and certifies evidence preservation, which retrieval never computes.

**Why it measures what we claim.** The signature-agreement predicate `(S[b1].evidence_fields & R) == (S[b2].evidence_fields & R)` measures evidence-contract preservation because the declared-field assumption makes contract metadata a faithful summary of what a backend emits; this assumption fails when a backend's implementation drifts from its declared contract (e.g., a detector silently stops emitting confidence), in which case the predicate reflects the declaration rather than the runtime output — which is why Phase 0 shrinks `evidence_fields` to the intersection of fields actually emitted over 64 calibration calls and demotes backends with drift rate > 5%. The `dropped_fields == set()` assertion in the closure certificate measures closure under substitution because the empty-difference assumption makes field-set equality equivalent to evidence preservation; this assumption fails when downstream steps consume evidence implicitly (e.g., via a learned reader that uses unlisted fields), in which case the certificate reflects declared rather than consumed evidence — which is why Phase 2 derives `R[p]` from logged consumer field reads unioned with the plan annotation. The `determinism` tie-break measures substitution safety because the same-determinism assumption makes a deterministic substitute behaviorally interchangeable with the failed backend for a planning step; this assumption fails when a step's correctness depends on sampling diversity, in which case the tie-break reflects interface parity rather than distributional parity.

## Contribution

(1) A harness-level Action-Set Equivalence Relation and failure-substitution closure operator that partitions tool backends by capability-signature agreement on evidence-contract fields and certifies evidence preservation on every rewrite. (2) A design principle: evidence-contract fields, not tool descriptions, are the correct carrier of substitutability in multimodal agent harnesses, yielding a deterministic substitution rule with no learned router. (3) An open registry schema and substitution-certificate format that make backend failures auditable at the plan level.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OpenVLHarness tool-use suite (22 benchmarks) | Real multimodal tool calls with backend failures |
| Primary metric | Evidence-preserving task success under injected backend failures | Directly measures substitution correctness |
| Baseline | Base MLLM, increased inference budget | Tests whether reasoning alone absorbs failures |
| Baseline | Gorilla-style embedding retrieval substitution | Tests signature-equality vs. nearest-neighbor matcher |
| Baseline | Toolformer-style relevance-ranked tool selection | Tests contract partitioning vs. relevance ranking |
| Ablation-of-ours | ASER without evidence-field predicate (artifact+coord only) | Isolates the contract-preservation contribution |
| Ablation-of-ours | ASER without Phase 0 conformance monitor (declared fields only) | Isolates the runtime-verification contribution |
| Diagnostic | Consumer-instrumentation log of field reads | Measures fraction of consumed fields outside R |

### Why this setup validates the claim

The central claim is that contract-equality substitution preserves the evidence fields downstream steps consume, and that this preservation—not raw tool availability or extra inference—drives recovery from backend failures. The base-MLLM baseline tests whether simply thinking harder absorbs failures; if it did, substitution machinery would be unnecessary. The Gorilla-style retrieval baseline tests the specific sub-claim that nearest-neighbor matching cannot certify field preservation: it will substitute plausibly similar tools that silently drop required evidence. The Toolformer-style relevance baseline tests whether ranking by query relevance rather than contract partitioning suffices. The ablation—dropping the evidence-field predicate while keeping artifact and coordinate-frame checks—isolates the contract-preservation mechanism from generic interface matching. The conformance-monitor ablation isolates the runtime-verification contribution from the declared-field assumption. The consumer-instrumentation diagnostic measures the fraction of consumed fields outside R, directly testing the load-bearing assumption that downstream steps read only annotated fields. The primary metric, evidence-preserving task success under injected failures, is the right detector because it counts a task as successful only when downstream consumers receive the fields they declared, so a substitution that swaps in an incompatible backend registers as failure even if the tool call itself returns. This combination makes the claim falsifiable: if contract equality is doing the causal work, gains must concentrate on failure-injected subsets where evidence fields diverge across backends, not on failure-free subsets.

### Expected outcome and causal chain

**vs. Base MLLM, increased inference budget** — On a case where a detection backend fails mid-plan and the next step needs a bounding box plus confidence, the baseline produces a plausible but unverified answer because its mechanism gap is that extra inference cannot restore a missing artifact—it can only re-reason over what it has. Our method instead rewrites the failed step to a contract-equivalent backend and asserts preserved fields because the closure operator checks field-set equality before substitution, so we expect a clear gap concentrated on failure-injected steps with parity on failure-free steps.

**vs. Gorilla-style embedding retrieval substitution** — On a case where the failed backend emitted `{bbox, label, confidence}` and a retrieval-nearest substitute emits only `{bbox, label}`, the baseline produces a downstream answer with missing confidence because its mechanism gap is that embedding similarity ranks interface descriptions, not emitted-field sets. Our method instead rejects that substitute and selects one whose declared fields cover the required set because equivalence is a set-containment test on contract fields, so we expect a noticeable gap on evidence-divergent substitutions and near-parity where substitutes happen to share fields.

**vs. Toolformer-style relevance-ranked tool selection** — On a case where a high-relevance substitute uses a different coordinate frame than the failed backend, the baseline produces spatially misaligned downstream output because its mechanism gap is that relevance ranking ignores coordinate-frame compatibility. Our method instead requires coordinate-frame equality in the equivalence predicate and certifies preserved fields, so we expect the gap to concentrate on cross-frame substitution instances while remaining small where frames coincide.

**vs. ASER without evidence-field predicate (ablation)** — On a case where two backends share artifact type and coordinate frame but differ in declared evidence fields, the ablation produces substitutions that drop required fields because its mechanism gap is that it cannot distinguish field-divergent backends. Our method instead filters on field-set agreement over the required fields, so we expect the full method to outperform the ablation specifically on the subset where required fields diverge, with parity elsewhere—confirming the predicate, not generic matching, drives recovery.

**vs. ASER without Phase 0 conformance monitor (ablation)** — On a case where a backend's declared fields include `confidence` but its runtime output omits it in 12% of calls, the ablation treats the backend as conformant and may substitute it in, producing downstream answers with missing confidence because its mechanism gap is that it trusts declarations without runtime verification. Our method instead shrinks the backend's evidence_fields to the intersection of actually emitted fields and demotes it if drift rate exceeds 5%, so we expect the full method to outperform the ablation specifically on backends with drift rate > 5%, with parity on conformant backends.

**Diagnostic: consumer-instrumentation log** — We instrument downstream consumers to log which fields they read over the 64-call calibration set per step and report the fraction of consumed fields outside the annotated `R[p]`. If this fraction is near zero, the load-bearing assumption holds; if it is non-trivial, the method's gains should degrade gracefully because Phase 2 unions logged reads into `R[p]`.

### What would falsify this idea

If the full method's gain over the Gorilla-style retrieval baseline were uniform across failure-injected and failure-free subsets, or if the evidence-field ablation matched the full method on the field-divergent subset, the claim that contract-equality preservation is the causal mechanism would be wrong. A diagnostic pattern of parity on field-divergent instances alongside gains only on generic interface-matching instances would indicate the mechanism is not doing what we claim.

## References

1. OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory
