# Delayed Render-Grounded Memory Auditing for Long Video Generation

## Motivation

Existing long-video generators such as Agentic LongLive maintain entity memory on a separate GPU and prefill it into the generator's attention sink, but the memory agent explicitly never reads the generated video, so memory errors are never corrected and the memory-to-generator interface remains open-loop. This structural decoupling is deliberate — it removes memory from the latency-critical path — but it also means that any drift in the prefilled entity representation propagates unchallenged through all subsequent shots, and no existing branch (LongDiff's training-free positional fix, StreamingT2V's autoregressive chunk conditioning, WorldWeaver's perceptual memory bank) closes this loop without reintroducing blocking synchronization.

## Key Insight

A delayed auditor that compares prefilled memory images against actually rendered frames after streaming, and revises memory only when a per-entity drift invariant is violated, restores closed-loop correction while preserving latency decoupling because the correction is applied to future shots rather than the shot that triggered it.

## Method

**What it is.** DRAMA (Delayed Render-grounded Memory Auditor) is an asynchronous correction layer wrapped around an entity-memory long-video generator. Its inputs are (i) the per-entity memory images that the memory agent prefilled into the generator's attention sink for shot t, and (ii) the rendered frame(s) of shot t once they are streamed out. Its outputs are (a) a per-entity drift residual, (b) a binary invariant-violation flag, and (c) when violated, a revised entity memory image plus a re-prefill request for shots t+1 onward.

**Load-bearing assumption (stated explicitly).** DRAMA assumes that a rendered frame which disagrees with the prefilled entity memory is evidence that the *memory* is wrong and should be revised toward the frame, rather than evidence that the *render* is wrong. The auditor never observes ground truth, so memory error and render error are confounded by construction. To make this assumption testable rather than hidden, the revision operator below is gated by a multi-frame consensus requirement: memory is overwritten only when the same entity appearance is observed consistently across ≥2 independent shots, so a one-off bad render cannot poison memory and a legitimate appearance change is not silently reverted. The confidence weight w is a secondary safeguard, not the primary attribution mechanism.

**How it works.**

```pseudocode
# Runs on a separate auditor GPU, consuming streamed output only.
# tau_id = 0.35 (identity-embedding L2 threshold), tau_pose = 0.20,
# W = 3 (consecutive violation window), K = 5 (revision steps),
# C = 2 (minimum independent shots corroborating a revision)

for each streamed shot t:
    F_t = rendered frames of shot t                    # arrives after streaming
    for each entity e named in shot t's prompt:
        M_e = memory image prefilled for e at shot t   # from memory agent
        # 1. Extract comparable views
        x_M = crop_and_align(M_e, bbox(F_t, e))
        x_F = crop_and_align(F_t, bbox(F_t, e))
        # 2. Drift residual: identity + pose components
        r_id  = || E_id(x_M) - E_id(x_F) ||_2          # E_id: face/person ReID encoder
        r_pose= || E_pose(x_M) - E_pose(x_F) ||_2      # E_pose: body-keypoint encoder
        r_e   = max(r_id / tau_id, r_pose / tau_pose)  # normalized drift
        # 3. Drift-bounded invariant with hysteresis
        if r_e > 1.0:
            viol_count[e] += 1
            # record this shot's rendered crop as a corroboration candidate
            evidence[e].append((t, x_F, r_e))
        else:
            viol_count[e] = 0
            evidence[e].clear()
        # 4. Trigger revision only on sustained violation AND multi-frame consensus
        if viol_count[e] >= W:
            # consensus gate: require >= C independent shots whose rendered
            # crops agree with each other (pairwise r_id <= tau_id and
            # r_pose <= tau_pose) before treating the frame as memory evidence
            consistent = count_consistent_pairs(evidence[e], tau_id, tau_pose)
            if consistent >= C:
                # Revision: blend rendered evidence into memory, weighted by
                # how far the rendered frame is from the *other* entities' memories
                # (so a single bad frame does not overwrite good memory)
                w = clip(1.0 / (1.0 + mean_{e'!=e} r_{e'}), 0.1, 0.9)
                x_F_consensus = mean of the C most recent consistent crops in evidence[e]
                M_e_new = (1 - w) * M_e + w * x_F_consensus
                # 5. Re-prefill request for shots t+1..t+K
                enqueue_reprefill(e, M_e_new, horizon=K)
            # if consensus not reached, treat as render error: suppress, do not write back
            viol_count[e] = 0
            evidence[e].clear()
    # Auditor never blocks: enqueue only; generator drains queue at shot boundary
```

**Why this design.** We chose a *delayed* auditor that reads streamed frames rather than a synchronous verifier that gates each shot, because Agentic LongLive's entire latency argument rests on the memory path being off the critical path, and a synchronous verifier would reintroduce the blocking that paper was designed to remove — accepting the cost that a violated invariant is corrected only for future shots, not the shot that revealed it. We chose a *normalized max* of identity and pose residuals over a single combined embedding distance because identity and pose drift have different scales and failure modes (a face swap vs. a body teleport), and a max-normalized residual makes the invariant interpretable as a single threshold across entities — accepting the cost that the two thresholds must be calibrated per encoder. We chose *hysteresis* (W consecutive violations) over single-shot triggering because single rendered frames are noisy and a single bad frame would overwrite good memory, which is exactly the failure Agentic LongLive avoided by never writing video back — accepting the cost that a genuinely fast drift is corrected W shots later. We chose a *confidence-weighted* memory revision (weight w from other entities' residuals) over naive replacement because it prevents one corrupted frame from poisoning entity memory — accepting the cost that a coordinated multi-entity drift is down-weighted and corrected more slowly. We added the *multi-frame consensus gate* (C=2) because the auditor cannot distinguish memory error from render error without ground truth, and requiring the same appearance across independent shots is the only attribution signal available to it — accepting the cost that a genuine but one-shot appearance change is not written back until it recurs. A domain expert would not describe this as a variant of Agentic LongLive because the auditor is a *new feedback surface* — a per-entity invariant monitor with a revision operator — that the original system's design explicitly forecloses; nor is it a variant of StreamingT2V's appearance-preservation module, which conditions on the *first* chunk and never revises it.

**Why it measures what we claim.** The normalized drift residual `r_e = max(r_id/tau_id, r_pose/tau_pose)` measures *entity consistency* because the ReID and pose encoders are assumed to be invariant to lighting, background, and camera motion while sensitive to identity and body configuration; this assumption fails when a shot legitimately changes an entity's appearance (a costume change, a time jump), in which case `r_e` reflects intended narrative change rather than memory error. The hysteresis counter `viol_count[e]` measures *sustained drift* because it assumes that genuine memory errors persist across consecutive shots while rendering noise is i.i.d.; this assumption fails when a memory error is intermittent (e.g., only under certain camera angles), in which case the counter under-triggers. The multi-frame consensus gate (C=2) measures *attribution of drift to memory vs. render* because it assumes that a true memory error will produce a consistent rendered appearance across independent shots while a render error will not; this assumption fails when the generator consistently renders the same wrong appearance (a systematic render bias), in which case consensus is reached on a wrong frame and memory is revised toward it. The revision weight `w` measures *reliability of the rendered frame as memory evidence* because it assumes that frames consistent with other entities' memories are more trustworthy; this assumption fails when all entities drift together, in which case `w` stays high and the revision still fires — which is the intended behavior, but it means `w` does not protect against correlated drift.

## Contribution

['A delayed, render-grounded memory auditor (DRAMA) that closes the open-loop memory-to-generator interface in asynchronous entity-memory long-video generation via a per-entity drift-bounded invariant with hysteresis and confidence-weighted memory revision.', 'A design principle for latency-decoupled correction: feedback can be restored to a streaming generator without blocking by applying corrections to future shots rather than the triggering shot, with a drift-bounded invariant as the trigger condition.', 'An analysis of the failure modes of render-grounded memory revision, including correlated multi-entity drift and legitimate appearance change, that delineates when delayed correction helps versus harms consistency.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LongStory-Online (Agentic LongLive benchmark) | Matches the asynchronous entity-memory regime we augment |
| Primary metric | Entity-consistency drift rate over streamed shots | Directly detects sustained per-entity invariant violations |
| Baseline | Agentic LongLive (unmodified) | Tests whether delayed correction adds anything to the base memory loop |
| Baseline | StreamingT2V (autoregressive, first-chunk APM) | Tests against non-revising appearance preservation |
| Baseline | Synchronous verifier gating every shot | Tests whether delayed auditing preserves latency while correcting |
| Baseline | Naive single-shot write-back (W=1, no consensus gate) | Tests whether the multi-frame consensus gate is doing real work |
| Ablation-of-ours | DRAMA without hysteresis (W=1, consensus gate on) | Isolates the sustained-drift assumption |
| Ablation-of-ours | DRAMA without consensus gate (C=1) | Isolates the attribution assumption |
| Calibration set | 512 ground-truth consistent entity pairs + 512 deliberately corrupted pairs from LongStory-Online | Justifies tau_id=0.35 and tau_pose=0.20 and reports false-positive/false-negative rates |
| Stress test | Correlated-drift subset (all entities drift together) | Tests the failure mode of w under correlated drift |

### Why this setup validates the claim

The central claim is that a delayed, streamed-frame auditor can correct entity-memory drift in an asynchronous long-video generator without reintroducing blocking, and that this correction is concentrated where sustained drift occurs. LongStory-Online is the right dataset because it is the exact benchmark where the base generator's memory path is off the critical path, so any gain we observe cannot be attributed to a different generator or a synchronous design. The primary metric — entity-consistency drift rate aggregated per streamed shot — is the right detector because our mechanism is predicted to reduce *persistent* violations rather than per-frame noise, so a metric aggregated across consecutive shots should separate our method from baselines that only fix isolated frames. Agentic LongLive (unmodified) tests the sub-claim that the base system leaves correctable drift on the table because it never reads rendered video. StreamingT2V tests the sub-claim that first-chunk conditioning without revision cannot recover from drift that appears mid-stream. The synchronous verifier baseline tests the sub-claim that our gain is not merely the gain of *any* verification — if a blocking verifier matches our consistency but loses on latency, the delayed design is validated as a distinct point. The naive single-shot write-back baseline (W=1, no consensus) tests the load-bearing attribution assumption: if it performs comparably to full DRAMA, the multi-frame consensus gate is decorative and the auditor cannot distinguish memory error from render error. The W=1 ablation (consensus on) tests whether hysteresis is doing real work: if W=1 performs comparably, our sustained-drift argument is undermined. The C=1 ablation tests whether the consensus gate is doing real work: if C=1 performs comparably, the attribution assumption is not load-bearing. The calibration set of 512 consistent + 512 corrupted pairs justifies the thresholds and reports false-positive/false-negative rates of the invariant at tau_id=0.35 and tau_pose=0.20. The correlated-drift stress test exercises the stated failure mode of w.

### Expected outcome and causal chain

**vs. Agentic LongLive (unmodified)** — On a case where an entity's appearance slowly drifts across several shots (e.g., a character's clothing color shifting under changing lighting), the baseline produces a monotonically worsening identity residual because its memory agent never reads rendered video and only refreshes memory from prompts. Our method instead detects the sustained residual via the hysteresis counter, confirms it across ≥2 independent shots via the consensus gate, and blends the rendered evidence back into the entity memory with a confidence weight, so we expect a visible reduction in entity-drift rate concentrated on long multi-shot sequences, with near-parity on short sequences.

**vs. StreamingT2V** — On a case where a character re-enters after many intermediate shots and the first-chunk appearance memory no longer matches the current scene, the baseline produces a hard appearance mismatch because its APM conditions only on the first chunk and never revises. Our method instead treats the consensus-corroborated rendered frame as evidence and re-prefills a revised memory for the next K shots, so we expect a gap on re-entry-heavy subsets and parity on subsets without re-entry.

**vs. Synchronous verifier** — On a case where a shot reveals a violation, the synchronous baseline produces correct consistency but blocks the generator at every shot boundary because it gates each shot before streaming. Our method instead enqueues a re-prefill for future shots and never blocks, so we expect comparable entity-consistency with a clear latency advantage, and a small consistency deficit only on the shot that first reveals the violation.

**vs. Naive single-shot write-back (W=1, no consensus)** — On a case where a single bad render produces a high residual (e.g., a motion-blurred or occluded frame), the naive baseline writes that frame directly into memory and poisons subsequent shots, producing a *higher* drift rate than unmodified Agentic LongLive on corrupted-frame subsets. Our method instead requires ≥2 independent shots to agree before writing back, so we expect a clear gap on corrupted-frame subsets and parity on clean subsets.

**vs. DRAMA without hysteresis (W=1, consensus on)** — On a case where a single noisy frame triggers a false violation, the W=1 ablation fires revision immediately (subject only to consensus), producing more frequent but less stable memory updates. Our method requires W=3 consecutive violations, so we expect a lower false-revision rate and a smoother drift trajectory on noisy subsets, with a small deficit on genuinely fast drift.

**vs. DRAMA without consensus gate (C=1)** — On a case where a systematic render bias produces a consistent wrong appearance, the C=1 ablation writes that appearance into memory and compounds the error. Our method requires independent-shot agreement, so we expect a lower drift rate on the correlated-drift stress subset, with parity on clean subsets.

**vs. Correlated-drift stress test** — On a case where all entities drift together, the confidence weight w stays high (because mean_{e'!=e} r_{e'} is large for all e), so revision still fires. We expect DRAMA to reduce drift relative to unmodified Agentic LongLive but not to eliminate it, and we report this as the stated failure mode of w rather than as a success.

### What would falsify this idea

If our gain in entity-consistency drift rate is uniform across short and long multi-shot sequences rather than concentrated on the long, re-entry-heavy subsets where the predicted failure mode lives, the delayed-revision mechanism is not doing what we claim. If the W=1 ablation matches full DRAMA, the sustained-drift argument is wrong; if the C=1 ablation matches full DRAMA, the attribution assumption is not load-bearing; if the naive single-shot write-back matches full DRAMA, the consensus gate is decorative.

## References

1. Agentic LongLive: Asynchronous Agentic Memory for Long Video Generation
