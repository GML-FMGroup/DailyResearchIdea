# Residual-Contracted Re-Anchoring for Drift-Bounded Autoregressive Script-to-Video Rollouts

## Motivation

Autoregressive script-to-video systems built on shift-window KV-cache denoisers (e.g., the simulation module of Yan: Foundational Interactive Video Generation) and evaluated by per-request state-consistency metrics (e.g., StreamEval's incremental request scoring) share a structural defect: every objective is chunk-local, so nothing couples the latent world state at chunk boundary k to the state at boundary k+1. Because the shift-window denoiser re-anchors its KV cache at each boundary using only the current chunk's context, and StreamEval scores consistency per request rather than across the full unbounded horizon, small per-chunk latent errors have no mechanism that prevents them from compounding — the field measures drift locally and therefore cannot bound it globally.

## Key Insight

If each chunk boundary computes a state-consistency residual in the denoiser's own latent space and re-anchors the KV shift-window by a contraction-mapped correction whose magnitude is bounded by that residual, then the accumulated latent drift over an unbounded rollout is bounded by a telescoping sum of per-step residuals, converting chunk-local scores into a horizon-level invariant without retraining the backbone.

## Method

**What it is.** We introduce **DRIFT-ANCHOR** (Drift-bounded Residual-Injected Fixed-point re-ANCHORing), a wrapper around any shift-window KV-cache denoiser that (i) computes a per-boundary state-consistency residual in the denoiser's latent space, (ii) applies a contraction-mapped correction to the shift-window anchor, and (iii) exposes a telescoping drift budget that StreamEval-style per-request scores cannot provide. Inputs: a stream of script/action tokens, a frozen shift-window denoiser D with latent space Z, and a contraction modulus ρ ∈ (0,1). Outputs: a rollout of latents z_1, z_2, … and per-step residuals r_k with a running drift budget B_k.

**How it works.**

```pseudocode
# Frozen backbone: shift-window denoiser D, latent space Z, window length W
# Hyperparameters: contraction modulus rho = 0.7, residual clip tau = 0.05, budget cap B_max = 0.5
# Calibration: fraction of tau-clipped steps f_clip tracked; manifold check via nearest decoded-valid latent

initialize anchor a_0 = encode(initial_frame)
B_0 = 0
f_clip = 0
for k = 1, 2, ... :
    # (1) Propose: run the frozen shift-window denoiser from the current anchor
    z_k_raw = D(z_{k-1}, action_k, KV_window(a_{k-1}))

    # (2) Residual: state-consistency residual in Z, computed on the OVERLAP
    #     between the previous chunk's tail latents and the new chunk's head latents
    overlap_prev = tail(z_{k-1}, W)
    overlap_new  = head(z_k_raw, W)
    r_k = || overlap_new - overlap_prev ||_2 / sqrt(W)      # per-step residual
    r_k_raw = r_k
    r_k = min(r_k, tau)                                     # clip for stability
    if r_k_raw > tau: f_clip += 1

    # (3) Contraction-mapped re-anchoring: pull the new anchor toward the
    #     previous anchor by a step whose size is bounded by r_k
    delta_k = clip( rho * r_k / (1 - rho), -tau, tau )      # correction magnitude
    a_k = a_{k-1} + delta_k * unit( overlap_prev - overlap_new )

    # (4) Re-run the denoiser with the corrected anchor (single extra step)
    z_k = D(z_{k-1}, action_k, KV_window(a_k))

    # (5) Telescoping drift budget (exposed, not used to gate)
    B_k = B_{k-1} + r_k
    if B_k > B_max:  emit warning; B_k = B_max  (rollout continues; budget saturates)
```

**Why this design.** We chose a *contraction-mapped* correction over a learned correction head because the contraction modulus ρ < 1 is what makes the telescoping bound hold without retraining the backbone, accepting the cost that the correction is conservative and may under-correct when the true drift is larger than τ. We chose to compute the residual on the *overlap window* (tail of previous chunk vs. head of new chunk) rather than on a full-chunk reconstruction because the overlap is the only region where the two chunks are directly comparable in the same latent coordinates, accepting that drift occurring entirely inside a chunk is invisible to r_k. We chose a *single* re-anchoring step rather than iterating to a fixed point because iterating would multiply inference cost per boundary and break the real-time 1080P/60FPS regime that Yan's shift-window denoiser targets, accepting a residual fixed-point error that the contraction modulus bounds geometrically. We chose to *expose* the budget B_k rather than gate on it because gating on an uncalibrated residual would repeat the log-prob-as-gate anti-pattern; instead the budget is a diagnostic that StreamEval-style per-request scoring cannot produce. A domain expert would not describe this as a variant of Process Reward Models or per-step MC value estimation: those assign a scalar reward to a step, whereas DRIFT-ANCHOR establishes a *coupling* between the residual and the anchor update — the correction is a function of the residual, so the residual is not a score but a control input, and the bound is a property of the coupled map, not of any single score.

**Why it measures what we claim.** The residual r_k = ||overlap_new − overlap_prev||/√W measures *cross-boundary state inconsistency* because the overlap window is the unique region where consecutive chunks share latent coordinates, so any deviation there is attributable to boundary drift rather than to new content; this assumption fails when the action at step k legitimately changes content inside the overlap region (e.g., an edit request that rewrites the tail), in which case r_k reflects intended change rather than drift and the correction will resist a legitimate edit. The contraction-mapped correction δ_k measures *the portion of drift that the anchor can absorb without retraining* because ρ < 1 guarantees the map a ↦ a + δ(‖overlap_prev − overlap_new‖) is a contraction on Z, so the post-correction overlap error is at most ρ times the pre-correction error; this assumption fails when the denoiser's latent manifold is not convex in the overlap region, in which case the contraction holds in ambient Z but the corrected anchor may leave the manifold and r_{k+1} will spike. The telescoping budget B_k = Σ r_j measures *horizon-level accumulated drift* because each r_j is the per-boundary contraction residual and the contraction map guarantees the true drift after k boundaries is at most Σ ρ^{k−j} r_j ≤ B_k/(1−ρ); this assumption fails when the clip τ activates on more than a fraction (1−ρ) of steps, in which case the saturated budget B_max under-reports true drift and the bound becomes vacuous — which is exactly why the budget is exposed as a warning rather than used as a gate. To make the load-bearing assumption explicit: we assume (A) that overlap-region latent change is dominated by drift rather than intended content, and (F) that the failure mode occurs when a legitimate edit rewrites the overlap tail or the manifold is non-convex so r_{k+1} spikes; we track the fraction of τ-clipped steps f_clip as a first-class calibration diagnostic and report it alongside B_k, and we verify manifold adherence by measuring the distance of the corrected anchor a_k to the nearest decoded-valid latent.

## Contribution

(1) DRIFT-ANCHOR, a backbone-agnostic wrapper for shift-window KV-cache denoisers that couples a per-boundary latent residual to a contraction-mapped anchor correction, turning chunk-local consistency into a horizon-level drift budget without retraining. (2) The design principle that the residual must be a *control input* to the anchor update rather than a score, which is what distinguishes the method from per-step reward estimation and is what makes the telescoping budget meaningful. (3) A diagnostic protocol, the drift-budget trace B_k, that exposes where StreamEval-style per-request state-consistency scores saturate and how much of the horizon-level drift is attributable to boundary vs. intra-chunk error.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | StreamEval 300-case incremental-request suite | Multi-boundary streaming edits stress cross-boundary drift |
| Primary metric | Boundary state-consistency error vs. drift budget B_k | Directly tests the coupled residual-anchor claim |
| Baseline | StreamEval per-request scoring (no gating) | Tests whether exposing budget adds diagnostic power |
| Baseline | Frozen shift-window denoiser, no re-anchoring | Tests whether contraction correction reduces drift |
| Baseline | Residual-as-gate (threshold on r_k) | Tests coupling vs. score-as-control distinction |
| Baseline | Test-time KV-cache correction (FIFO/rolling re-anchoring) | Tests whether contraction-mapped overlap residual is distinct from generic feedback control |
| Ablation-of-ours | Full-chunk residual instead of overlap residual | Tests whether overlap is the right measurement locus |
| Calibration diagnostic | Fraction of τ-clipped steps f_clip; post-correction overlap error ratio vs. ρ; manifold-violation rate (distance of corrected anchor to nearest decoded-valid latent) | Verifies the bound's premises empirically |

### Why this setup validates the claim

The central claim is that DRIFT-ANCHOR's contraction-mapped correction reduces horizon-level drift *because* the residual is a control input coupled to the anchor, not merely a score. StreamEval's 300 incremental-request cases are the right dataset because they force many chunk boundaries with legitimate content change, which is exactly where the overlap-residual assumption is stressed. The primary metric — boundary state-consistency error tracked against the telescoping budget B_k — is the right one because it lets us observe whether the contraction actually bounds post-correction overlap error at roughly ρ times pre-correction error across boundaries, rather than just averaging per-request scores. The no-re-anchoring baseline isolates the contribution of the correction step; the residual-as-gate baseline isolates the coupling claim (control input vs. threshold); the test-time KV-cache correction baseline isolates whether our contraction-mapped overlap residual is distinct from a generic feedback controller; the full-chunk-residual ablation tests whether the overlap window is the correct measurement locus, since drift inside a chunk should be invisible to our r_k but visible to a full-chunk residual. The calibration diagnostic directly measures the fraction of τ-clipped steps f_clip, the post-correction overlap error ratio vs. the claimed ρ, and the manifold-violation rate, verifying the bound's premises rather than assuming them. Together these form a falsifiable test: if the gain is uniform across subsets rather than concentrated where boundary drift is high and overlap content is stable, the mechanism story is wrong.

### Expected outcome and causal chain

**vs. Frozen shift-window denoiser, no re-anchoring** — On a case where a long script stream accumulates small per-boundary latent offsets across many chunks (e.g., a slow camera pan rewritten in successive requests), the baseline produces visibly compounding state drift because each chunk is anchored only to the previous raw latent with no contraction toward a consistent anchor. Our method instead pulls each new anchor toward the previous one by a residual-bounded step, so per-boundary overlap error shrinks by roughly ρ each time and B_k grows sublinearly. We expect a clear gap on long-horizon cases (many boundaries) but near-parity on single-boundary cases, since the correction has nothing to accumulate against.

**vs. Residual-as-gate (threshold on r_k)** — On a case where a legitimate edit rewrites the overlap tail (e.g., an object enters the frame at a chunk boundary), the baseline produces either a false halt or a silently accepted drift because a scalar threshold cannot distinguish intended change from drift and cannot act on it. Our method instead applies a bounded correction whose magnitude scales with r_k but is capped by τ, so legitimate edits are resisted only weakly while true drift is absorbed. We expect the baseline to show a bimodal error profile (either frozen or drifting) while ours stays within budget, with the gap concentrated on edit-heavy subsets.

**vs. StreamEval per-request scoring (no gating)** — On a case where drift accumulates gradually and each per-request score looks acceptable individually, the baseline produces no warning because per-request scores cannot telescope into a horizon-level quantity, so the failure is invisible until a visible artifact appears. Our method instead exposes B_k, which grows monotonically with each boundary residual, so the warning fires before the artifact. We expect the baseline to show low per-request variance but high end-of-rollout error, and ours to show a rising B_k that correlates with the eventual error, with the diagnostic gap largest on the longest streams.

**vs. Test-time KV-cache correction (FIFO/rolling re-anchoring)** — On a case where a generic feedback controller (e.g., FIFO re-anchoring that refreshes the KV cache without a contraction-mapped residual) is applied, the baseline may reduce drift but lacks a horizon-level bound and can over-correct on edit-heavy cases because it does not distinguish drift from intended change. Our method instead couples the correction magnitude to the overlap residual via a contraction map with modulus ρ, so the post-correction error is bounded by ρ times the pre-correction error and the telescoping budget B_k provides a horizon-level invariant. We expect our method to match or exceed the baseline on long-horizon drift reduction while showing lower over-correction on edit-heavy subsets, with the gap attributable to the contraction-mapped coupling rather than generic feedback.

**vs. Full-chunk residual ablation** — On a case where drift originates entirely inside a chunk (e.g., a mid-chunk action change with a stable overlap), the ablation produces a nonzero residual and an unnecessary correction because the full-chunk comparison conflates new content with drift, while the true overlap is clean. Our method instead reports r_k near zero and leaves the anchor untouched, preserving the legitimate content change. We expect the ablation to show over-correction artifacts on intra-chunk-edit subsets and parity elsewhere, confirming the overlap window is the correct measurement locus.

### What would falsify this idea

If the boundary state-consistency gain over the no-re-anchoring baseline is uniform across short- and long-horizon subsets rather than concentrated on long horizons, the contraction-accumulation story is wrong. If the residual-as-gate baseline matches our method on edit-heavy cases, the coupling claim (residual as control input, not score) is not doing the work we attribute to it. If the calibration diagnostic shows f_clip > 1−ρ on more than 10% of steps or the post-correction overlap error ratio deviates significantly from ρ, the bound's premises are violated and the telescoping budget is vacuous.

## References

1. Yan: Foundational Interactive Video Generation
2. StreamEval: Evaluating Interactive Streaming Video Generation Models
3. Video-Bench: Human-Aligned Video Generation Benchmark
4. DreamBench++: A Human-Aligned Benchmark for Personalized Image Generation
5. T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation
6. VBench: Comprehensive Benchmark Suite for Video Generative Models
7. FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation
