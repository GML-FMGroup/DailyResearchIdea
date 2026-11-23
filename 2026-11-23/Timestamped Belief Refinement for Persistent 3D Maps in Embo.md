# Timestamped Belief Refinement for Persistent 3D Maps in Embodied World Models

## Motivation

Persistent embodied memory systems — the 3D map aggregation in the persistent embodied world model (E3/E7), 3D-LLM-Mem's episodic fusion (E1/E6), and 3D-Mem's incremental snapshots (E2/E9) — all update memory by aggregating new observations into a stored representation without any invariant that ties a downstream query to the most recent observation of the object it asks about. The structural root cause is that these maps are written as commutative aggregations (averaging, token fusion, incremental insertion) whose stored value is a function of the multiset of observations, not of their timestamps; a stale observation therefore retains equal or greater weight than a later revisit, and ReSpaceBench's cross-session object-change tasks (E4/E8) expose exactly this, with the strongest zero-shot model reaching only 50.5% and agentic frameworks showing no improvement over direct inference because execution traces show evidence omissions and incorrect cross-visit reasoning.

## Key Insight

Replacing commutative map aggregation with a timestamp-ordered last-writer-wins resolution rule makes every downstream query provably resolve against the latest observation of each object, so stale cached representations are structurally invalidated rather than merely down-weighted.

## Method

**What it is.** TRACE-Map (Timestamp-Resolved Aggregation with Consistency Enforcement) is a persistent 3D map module for embodied world models. Its inputs are the per-step RGB-D observation stream, the agent pose, and a detected set of object instances with predicted states; its output is a queryable map in which every object record carries a resolved state and a timestamp, and every planning query is answered by resolving against that record rather than against an aggregated cache.

**Load-bearing assumption (stated explicitly).** The design assumes that a later *valid* observation timestamp corresponds to a more valid object state, so that timestamp-ordered resolution on valid observations yields the correct resolved belief for every downstream query. This assumption fails when the agent re-observes an object from a viewpoint that cannot resolve its state (occlusion, partial view, detector miss): a later timestamp then reflects observation recency, not state validity. To keep the ordering invariant sound under a noisy perception stream, resolution is gated on observation validity (Phase 2 below), and invalidation keys on the timestamp of the last *valid* observation, not the last observation.

**How it works.**

```pseudocode
# Phase 1: observation ingestion (per timestep t)
for each frame (RGB_t, D_t, pose_t):
    instances = detector(RGB_t, D_t)                  # object id, bbox, state label
    for inst in instances:
        key = place_identity(inst, pose_t)            # room id + object id, ReSpaceBench-style
        record = map.get(key)
        new_obs = Observation(state=inst.state, t=t, pose=pose_t,
                              confidence=inst.score,
                              visible=visibility_check(inst, D_t))  # bbox depth coverage >= 0.5
        # Phase 2: timestamped, validity-gated last-writer-wins resolution
        if record is None:
            map[key] = Record(state=new_obs.state, t=t, t_valid=t,
                              history=[new_obs], unverified_since=None)
        else:
            record.history.append(new_obs)
            if new_obs.t > record.t and new_obs.confidence >= 0.5 and new_obs.visible:
                record.state = new_obs.state
                record.t = new_obs.t
                record.t_valid = new_obs.t
                record.unverified_since = None
            elif new_obs.t > record.t:
                # low-confidence or occluded late write: retain prior resolved state
                record.unverified_since = new_obs.t
            # else: observation is older; append to history, do not overwrite

# Phase 3: stale-representation invalidation (keyed on last VALID observation)
def invalidate(record, t_query):
    # any cached downstream artifact derived from record with source_t < record.t_valid is dropped
    for artifact in cache.derived_from(record.key):
        if artifact.source_t < record.t_valid:
            cache.drop(artifact)
    return record.state, record.t_valid, record.unverified_since

# Phase 4: query resolution
answer(query, t_query):
    for key in query.referenced_objects:
        state, t_valid, unverified = invalidate(map[key], t_query)
        if t_valid > t_query: raise InconsistentQuery      # future evidence
    return planner.resolve(query, resolved_states)
```

The map is a keyed store `(place_identity, object_id) -> Record(state, t, t_valid, history, unverified_since)`. Resolution is validity-gated last-writer-wins on timestamp `t`: a new observation overwrites the resolved state only if `t` is strictly greater than the current `t` AND its detector confidence is at least 0.5 AND its bbox depth coverage is at least 0.5 (the visibility check); otherwise the prior resolved state is retained and the record is marked `unverified_since = t`. Ties on `t` are broken by pose distance to the query viewpoint (closer wins). The invalidation pass runs at every query, not on a schedule, and keys on `t_valid` (the timestamp of the last valid observation), so a noisy late write cannot invalidate correct derived artifacts. No downstream planner artifact can outlive the valid observation it was derived from.

**Why this design.** We chose validity-gated last-writer-wins over weighted averaging because averaging makes the stored state a function of the observation multiset, so a stale observation keeps nonzero weight after a revisit — accepting the cost that a single noisy late observation can flip a state, which we mitigate by gating resolution on confidence and visibility rather than by smoothing, and by keeping the full history for audit. We chose per-query invalidation over periodic cache flushing because a periodic schedule leaves a window in which a planner can consume a stale artifact — accepting the cost of a linear scan over derived artifacts per query, bounded by the number of objects the query references. We chose place-identity keying (room + object, following ReSpaceBench's physical-room-entry visit definition) over raw coordinate proximity because coordinate proximity cannot distinguish same-type room distractors, which ReSpaceBench deliberately includes — accepting the cost that place identity must be maintained as a separate bookkeeping surface. A domain expert would not describe this as retrieval-augmented memory or as a variant of 3D-Mem's incremental snapshot pipeline, because those systems store and retrieve observations; TRACE-Map instead imposes an ordering invariant on the map and enforces it at query time.

**Why it measures what we claim.** The computational quantity `record.t_valid` (the timestamp of the last valid resolved state) measures *belief freshness* because the assumption that a later valid timestamp corresponds to a later physical observation holds under the walkthrough protocol in which visits are defined by physical room entry and observations pass the confidence/visibility gate; this assumption fails when the agent re-observes an object from a viewpoint that cannot resolve its state (occlusion), in which case the observation is retained as `unverified_since` and `record.t_valid` reflects the last valid observation rather than the latest observation. The quantity `cache.drop` on artifacts with `source_t < record.t_valid` measures *stale-representation invalidation* because the assumption that every planner artifact is derived from exactly the record version it read holds when the planner is deterministic given resolved states; this assumption fails when the planner caches intermediate geometry across objects, in which case invalidation reflects only direct dependencies. The quantity `InconsistentQuery` measures *monotone consistency under belief refinement* because the assumption that queries are issued at a timestamp no earlier than their referenced evidence holds under online execution; this assumption fails under offline replay of past queries, in which case the flag reflects replay order rather than belief inconsistency.

## Contribution

['TRACE-Map, a timestamp-resolved persistent 3D map module that replaces commutative aggregation with last-writer-wins resolution plus per-query invalidation of derived planner artifacts, giving a structural freshness invariant rather than a learned or weighted one.', 'A design principle for persistent embodied memory: recency must be enforced at query resolution rather than at write time, because write-time smoothing is what allows stale observations to survive revisits — demonstrated by contrasting last-writer-wins against averaging on cross-session object-change tasks.', 'An audit surface (full per-object observation history retained alongside the resolved state) that lets cross-visit failures be attributed to either missing evidence or incorrect resolution, addressing the evidence-omission failure mode reported for agent execution traces.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ReSpaceBench (590 scenes, 3,244 questions) | Deliberate same-type room distractors and cross-session changes |
| Primary metric | Accuracy on cross-visit state-tracking subset | Directly probes persistent state resolution after revisits |
| Baseline | Strongest zero-shot video model (direct inference) | Tests whether ordering invariant beats raw context |
| Baseline | Agentic video-understanding framework | Tests whether per-query invalidation beats agentic retrieval |
| Baseline | 3D-Mem-style incremental snapshot memory | Tests place-identity keying against snapshot retrieval |
| Ablation-of-ours | TRACE-Map with periodic cache flushing (flush interval = 10 queries) | Isolates per-query invalidation from timestamp resolution |
| Ablation-of-ours | TRACE-Map with strict last-writer-wins (no validity gate) | Isolates the validity gate from timestamp resolution |
| Diagnostic | Occlusion rate on ReSpaceBench revisits (fraction of revisit observations with bbox depth coverage < 0.5) | Measures how often the load-bearing assumption's failure mode occurs |

### Why this setup validates the claim

The central claim is that imposing a strict timestamp-ordering invariant on a place-keyed map, gated on observation validity and enforced at query time, yields correct persistent state after revisits where observation-storing systems fail. ReSpaceBench is the right dataset because its controlled walkthroughs define visits by physical room entry and deliberately include same-type room distractors and cross-session object changes — exactly the conditions under which a stale or averaged belief survives. The primary metric, accuracy on the cross-visit state-tracking subset, is the right detector because it is scored on questions whose answer depends on the resolved state after a revisit, so any residual staleness shows up as error rather than being masked by aggregate accuracy. The three baselines each test a distinct sub-claim: direct inference tests whether the ordering invariant adds anything over full-context observation; the agentic framework tests whether per-query invalidation beats retrieval orchestration; the 3D-Mem-style snapshot baseline tests whether place-identity keying beats snapshot retrieval for distractor rooms. The two ablations isolate the two mechanisms: periodic flushing (interval = 10 queries) tests whether per-query invalidation matters, and strict last-writer-wins without the validity gate tests whether the gate matters. The occlusion-rate diagnostic measures how often the load-bearing assumption's failure mode occurs, bounding the conditions under which the gain is attributable to the validity gate rather than to timestamp ordering alone.

### Expected outcome and causal chain

**vs. strongest zero-shot video model** — On a case where the agent revisits a same-type room and an object's state changed between visits, the baseline produces the pre-change state because it reasons over the observation multiset in context with no ordering invariant, so the earlier and later observations compete. Our method instead resolves the record by validity-gated last-writer-wins on `t` and invalidates derived artifacts, so the post-change state is returned by construction. We expect a noticeable gap concentrated on the cross-visit state-tracking and same-type-distractor subsets, with near-parity on single-visit room-association questions.

**vs. agentic video-understanding framework** — On a case where the agent must answer a query referencing an object whose state was updated two visits ago, the baseline produces an answer built from an intermediate artifact derived from the older observation because its retrieval orchestration has no per-query invalidation step, so a stale derivation outlives its source. Our method instead runs `invalidate` at every query, dropping any artifact with `source_t < record.t_valid`, so the planner only sees the resolved state. We expect the gap to widen with the number of intervening visits and to be largest on downstream-reasoning questions chained on a changed object.

**vs. 3D-Mem-style incremental snapshot memory** — On a case where two visually similar rooms are visited in sequence, the baseline produces the wrong room's object state because snapshot retrieval keys on visual similarity rather than physical place identity, so the distractor room's snapshot is retrieved. Our method instead keys on `(place_identity, object_id)` with room identity maintained as separate bookkeeping, so the distractor cannot alias the target record. We expect a gap concentrated on the same-type-distractor subset and parity on unambiguous single-room scenes.

**vs. TRACE-Map with periodic cache flushing (interval = 10 queries)** — On a case where a query lands inside the flush window after a valid state update, the periodic-flush ablation consumes a stale artifact because its invalidation cadence is decoupled from query arrival; the equivalence assumption is that the flush interval exceeds query inter-arrival so no stale artifact is consumed, and the failure mode is a query landing inside the flush window, in which case the ablation silently matches per-query invalidation. We expect the ablation to match per-query invalidation on the fraction of queries outside the flush window and to underperform on the fraction inside it; we will report the fraction of queries falling inside the flush window as a diagnostic.

**vs. TRACE-Map with strict last-writer-wins (no validity gate)** — On a case where a revisit observation is occluded or low-confidence, the ungated variant overwrites the correct earlier state with a noisy late write and then locks it in via invalidation, because it treats timestamp order as state validity. Our method instead retains the prior resolved state and marks the record `unverified_since`, so the correct derived artifacts survive. We expect the gap to scale with the measured occlusion rate on ReSpaceBench revisits.

### What would falsify this idea

If the gain over the strongest baseline is uniform across subsets rather than concentrated on cross-visit state-tracking and same-type-distractor questions, the ordering-invariant explanation is wrong and the method is merely a generic memory improvement. Likewise, if the periodic-flush ablation matches per-query invalidation on the queries outside the flush window, or if the ungated last-writer-wins variant matches the validity-gated variant despite a non-negligible occlusion rate, then the claimed mechanisms are not what produce the effect.

## References

1. Lost in the Hallway: Benchmarking Persistent Spatial Memory Across Revisits
