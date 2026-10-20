# Provenance-Weighted Contradiction Reconciliation for Persistent Multimodal Agent Memory

## Motivation

Persistent multimodal memory in agent harnesses such as OpenVLHarness stores images, structured data, and web references as addressable artifacts, but it provides no belief-revision rule: when two heterogeneous backends return conflicting visual evidence (e.g., a rendered crop vs. a structured datum), the memory simply accumulates both entries and downstream tool calls condition on an inconsistent state. This is a forest-level structural gap — the frontier-model, neuro-symbolic-reuse, and reward-modeling branches all inherit it — because no branch defined an operator that decides which stored multimodal entry survives a contradiction, so consistency is left to the model's incidental reasoning rather than enforced by the memory substrate.

## Key Insight

Contradiction reconciliation over multimodal memory can be made a commutative, idempotent, provenance-weighted lattice join, so the final memory state is invariant to the order in which tool observations arrive and repeated observations are no-ops — a confluence property that no accumulate-only memory (OpenVLHarness) or retrieval-based tool selector provides.

## Method

**(A) What it is.** CRO (Contradiction-Reconciliation Operator) is a total function `reconcile: Memory × Observation → Memory` inserted between OpenVLHarness's capability layer and its persistent memory. It takes a new tool observation `o = (modality, content, source, timestamp)` and the current memory `M`, and returns `M'` in which every entry contradicting `o` is superseded, merged, or flagged, with a contradiction label attached; downstream tool calls are gated on `M'`.

**(B) How it works.**

```pseudocode
# Each memory entry e = (id, modality, content, source, prov, status, label)
# prov = provenance weight in [0,1]; status in {ACTIVE, SUPERSEDED, MERGED, FLAGGED}
# PROV(source, o) is a content-conditioned reliability estimate, not a fixed constant:
#   PROV(source, o) = CALIBRATE(source, FEATURES(o))
# where FEATURES(o) = (resolution, hash_distance_to_conflict, extraction_length, rotation_angle)
# and CALIBRATE is a logistic regression model per backend, trained on a held-out calibration set
# of 512 observations per backend with binary correctness labels (human-verified).
# The calibration set is disjoint from the evaluation suite.

function RECONCILE(M, o):
    conflicts = [e for e in M if CONTRADICTS(e, o)]        # detection
    if conflicts is empty:
        return M + [NEW_ENTRY(o, prov=PROV(o.source, o))]
    for e in conflicts:
        tau = PROV(o.source, o) - PROV(e.source, e)        # calibrated provenance gap
        if tau > THETA_SUPERSEDE:      # e loses
            e.status = SUPERSEDED; e.label = ("superseded_by", o.id)
        elif tau < -THETA_SUPERSEDE:   # o loses
            o.status = SUPERSEDED; o.label = ("superseded_by", e.id)
        elif |tau| <= THETA_MERGE and SAME_MODALITY(e, o):
            e.content = MERGE_CONTENT(e.content, o.content)  # deterministic join
            e.status  = MERGED; e.prov = MAX(e.prov, PROV(o.source, o))
        else:                          # incomparable calibrated provenance
            e.status = FLAGGED; e.label = ("unresolved", o.id)
            o.status = FLAGGED; o.label = ("unresolved", e.id)
    return M + [o]

function CLOSURE(M):   # invariant enforced before every tool call
    assert all(not CONTRADICTS(a, b) for a, b in ACTIVE_PAIRS(M))
    return ACTIVE_VIEW(M)   # only ACTIVE/MERGED entries visible downstream
```

`CONTRADICTS` is a modality-specific exact check: for structured data, key-value disagreement on a shared key; for images, a perceptual-hash Hamming distance above `HASH_TAU=10` on the same crop region; for web references, differing extracted spans for the same query. `PROV(source, o)` is a content-conditioned reliability estimate: for each backend, a logistic regression model maps observable features (resolution, hash distance to the conflicting entry, extraction length, rotation angle) to a calibrated probability of correctness. The model is trained on a held-out calibration set of 512 observations per backend with human-verified correctness labels, disjoint from the evaluation suite. `THETA_SUPERSEDE=0.15`, `THETA_MERGE=0.05`.

**(C) Why this design.** We chose a provenance-weighted total order over a learned contradiction classifier because the classifier's errors would silently corrupt memory with no audit trail, accepting the cost that fixed provenance weights cannot adapt to a genuinely more reliable low-weight backend. We chose supersede/merge/flag as three explicit statuses over a single 'overwrite' because flagging preserves the incomparable case for downstream inspection, accepting the cost of extra memory entries and a larger closure check. We chose a commutative, idempotent update (max-join on provenance, deterministic content merge) over last-write-wins because last-write-wins makes memory order-dependent and non-reproducible, accepting the cost that a late-arriving high-provenance observation must trigger re-reconciliation of earlier entries. A domain expert would not describe CRO as a variant of retrieval-based tool selection (Gorilla, Toolformer) or of OpenVLHarness's accumulate-only memory, because those systems never define a revision rule at all — CRO's contribution is the closure invariant and confluence, not a selector.

**(D) Why it measures what we claim.** The calibrated provenance gap `tau` measures contradiction resolution correctness because the assumption that backend reliability is conditionally stationary given observable features makes the higher-provenance entry the expected-correct one; this assumption fails when a normally reliable backend returns a systematically stale crop that the calibration features do not capture, in which case `tau` reflects source reputation rather than current evidence quality. The closure invariant `all(not CONTRADICTS(a,b))` measures memory consistency because the assumption that `CONTRADICTS` is a sound exact check for the three modalities makes the absence of active contradictions equivalent to a consistent state; this assumption fails for images where perceptual-hash collisions hide a real contradiction, in which case the invariant reflects hash agreement rather than visual agreement. The commutativity/idempotence of `RECONCILE` measures order-invariance because the assumption that reconciliation is a join on a lattice with provenance as the ordering key makes any permutation of observations reach the same fixpoint; this assumption fails when `MERGE_CONTENT` is non-associative, in which case the operator reflects arrival order rather than a true join.

## Contribution

['A Contradiction-Reconciliation Operator (CRO) specified as a commutative, idempotent, provenance-weighted update rule over multimodal memory entries, with explicit supersede/merge/flag contradiction labels and a closure invariant that gates downstream tool calls.', 'The design principle that persistent multimodal memory consistency should be enforced by the memory substrate rather than left to model reasoning, demonstrated through the confluence property that makes agent memory order-invariant under heterogeneous backend observations.', "A modality-specific exact contradiction detector (key-value, perceptual-hash, extracted-span) integrated into OpenVLHarness's capability layer as a drop-in reconciliation stage."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OpenVLHarness 22-benchmark suite, contradiction-heavy subset | Native multimodal memory conflicts with known provenance |
| Primary metric | Contradiction-resolution accuracy on gated downstream calls | Directly measures whether closure yields correct tool use |
| Baseline | Base MLLM, increased inference budget | Tests whether budget alone substitutes for revision |
| Baseline | OpenVLHarness accumulate-only memory | Tests whether revision rule is necessary vs. accumulation |
| Baseline | Last-write-wins memory update | Tests whether provenance weighting beats recency |
| Ablation-of-ours | CRO with flagging disabled (overwrite-on-conflict) | Isolates the incomparable-case preservation claim |
| Diagnostic | Downstream accuracy conditioned on memory-state consistency (closure invariant holds vs. violated) | Isolates CRO's effect from the downstream model's ability to resolve residual contradictions |

### Why this setup validates the claim

The central claim is that a provenance-weighted, commutative revision operator with a closure invariant produces more correct downstream tool calls than systems that never define a revision rule. The accumulate-only baseline tests the necessity sub-claim: if memory never revises, contradictions persist into the active view and downstream calls inherit them. The last-write-wins baseline tests the provenance sub-claim: it defines a revision rule but ignores source reliability, so it should fail specifically where a stale high-provenance entry is overwritten by a fresh low-provenance one. The inference-budget baseline tests whether the effect is merely a compute artifact — if budget scaling matched CRO, the mechanism would be uninteresting. The flagging-disabled ablation isolates the incomparable case: if flagging contributes nothing, the ablation should match full CRO, and any gap between them localizes the value of preserving unresolved conflicts. Contradiction-resolution accuracy on gated downstream calls is the right metric because it is the only observable that directly reflects whether memory state at call time was consistent; end-task accuracy alone would confound contradiction handling with perception and planning, while memory-entry counts would reward the accumulate-only baseline trivially. The diagnostic that conditions downstream accuracy on memory-state consistency tests the missing equivalence: if gains are attributable to CRO, they should appear only when the closure invariant holds and the downstream model cannot re-derive the correct value from an inconsistent active view; if a strong MLLM re-derives the correct value despite an inconsistent memory, the diagnostic will show parity on violated-consistency instances, confirming that the effect is not merely downstream reasoning. Together these form a falsifiable test: the predicted effect must be concentrated on the contradiction-heavy subset and absent where no conflicts arise.

### Expected outcome and causal chain

**vs. Base MLLM, increased inference budget** — On a case where a rendered crop and an OCR span disagree on a numeric field, the baseline produces a downstream call using whichever reading its context favors because more inference budget only re-reads the same conflicting evidence without a revision rule to break the tie. Our method instead supersedes the lower-provenance OCR entry because the calibrated provenance gap exceeds the supersede threshold, so the gated call sees a single consistent value. We expect a noticeable gap on the contradiction-heavy subset but near parity on conflict-free benchmarks, since budget scaling already suffices where no contradiction exists.

**vs. OpenVLHarness accumulate-only memory** — On a case where a web-scrape reference and a rendered crop give differing extracted spans for the same query, the baseline produces an active memory containing both spans because accumulation never marks either as superseded, so the closure invariant is violated and downstream selection is arbitrary. Our method instead flags or supersedes one entry per the calibrated provenance gap and exposes only the active view, so the downstream call is deterministic. We expect a large gap concentrated exactly on instances with cross-backend conflicts, with parity on single-source instances.

**vs. Last-write-wins memory update** — On a case where a late-arriving low-provenance web-scrape contradicts an earlier high-provenance rendered crop, the baseline produces the wrong value because recency overrides reliability. Our method instead keeps the crop active because the negative calibrated provenance gap triggers supersession of the new observation. We expect the gap to appear specifically on late-low-provenance-arrival instances and to shrink to parity on instances where arrival order matches provenance order, which is the diagnostic signature that provenance, not order, drives the gain.

### What would falsify this idea

If CRO's gain over the accumulate-only and last-write-wins baselines were uniform across conflict-free and contradiction-heavy subsets rather than concentrated on instances with cross-backend contradictions, the claim that the closure invariant and provenance weighting cause the improvement would be wrong. Equally, if the flagging-disabled ablation matched full CRO everywhere, the incomparable-case preservation argument would be unsupported. Additionally, if the diagnostic showed that downstream accuracy is equally high on instances where the closure invariant is violated (i.e., the downstream model re-derives the correct value despite inconsistent memory), then the attribution of gains to CRO's memory-state consistency would be falsified.

## References

1. OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory
