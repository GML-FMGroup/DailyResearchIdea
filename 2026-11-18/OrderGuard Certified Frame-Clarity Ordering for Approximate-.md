# OrderGuard: Certified Frame-Clarity Ordering for Approximate-Cache Autoregressive Video Diffusion

## Motivation

Autoregressive video diffusion relies on the structural assumption that earlier frames are clearer than later frames — AR-Diffusion enforces this by a non-decreasing corruption-timestep constraint plus temporal causal attention, and FastFrames achieves interactive latency by replacing exact KV-cache updates with lightweight approximations. Neither work tests whether the ordering survives approximation: FastFrames's own profiling shows KV-cache handling dominates latency, so its latency win is purchased by silently discarding the exactness that AR-Diffusion's ordering constraint presupposes. The field's trajectory toward real-time AR diffusion therefore creates a new bottleneck — an unverified invariant — that becomes binding precisely when caches are approximated, and no existing method provides a per-step certificate that the approximation preserves it.

## Key Insight

The clarity ordering that AR-Diffusion assumes is exactly a comparison between two per-frame scalars (corruption timestep and its implied noise level), so it can be checked at every denoising step against the approximate cache's own outputs, and any violation is localizable to a specific frame whose exact-attention refresh restores it — turning a latency/quality trade-off into a bounded, verifiable correction.

## Method

**What it is.** OrderGuard is a wrapper around an autoregressive video diffusion sampler (e.g., AR-Diffusion-style with temporal causal attention) that uses a lightweight KV cache (FastFrames-style replacement) and, at every reverse-diffusion step, computes a per-frame clarity score, checks that it is non-increasing in frame index, and when the check fails applies a bounded correction: exact-attention refresh for the offending frame's cache entries, followed by a re-check. Inputs: the current noisy latent batch, the approximate KV cache, the per-frame corruption timesteps. Outputs: a corrected cache and a per-step certificate flag (number of refreshed frames and whether the ordering holds).

**Load-bearing assumption (stated explicitly).** The certificate is valid only if the per-frame clarity score is computed against an independent, exact-attention reference that the correction itself cannot modify. Concretely: (i) the clarity score for frame i is the negative reconstruction error between the approximate-cache denoised output and an exact-attention reference denoised output for that frame, computed on a fixed random subset of frames (size K=4 per step, sampled once per video with seed 0); (ii) the corruption timesteps t_i are frozen during certification — the re-scheduling fallback is applied only after the certificate for the step is emitted and is reported as a separate, uncertified intervention; (iii) the ordering check compares the reference-based clarity scores across frames, not the timestep anchor alone. If this assumption fails (e.g., the reference pass is itself approximated, or t_i is mutated before certification), the certificate is not a valid measure of ordering preservation.

**How it works.**

```pseudocode
# Precompute once per video: fixed random subset of frames for reference pass
ref_frames = random.sample(range(N), K=4, seed=0)

# Per denoising step
for step in reverse_diffusion_steps:
    x_hat = denoise(x, cache_approx)          # approximate-cache forward
    # Exact-attention reference pass on the fixed subset (no cache approximation)
    x_ref = denoise(x[ref_frames], cache_exact)   # exact attention over full history
    for i in 0..N-1:
        if i in ref_frames:
            # clarity score = negative reconstruction error vs exact reference
            s_hat[i] = -||x_hat[i] - x_ref[i]||^2 / sigma_t^2
        else:
            # for non-reference frames, use the timestep-implied anchor only
            s_hat[i] = 1 - t_i / T
    # Ordering check: monotone non-increasing in frame index, tolerance delta
    violations = { i : s_hat[i] > s_hat[i-1] + delta for i in 1..N-1 }
    if violations nonempty:
        for i in sorted(violations, descending):
            # bounded correction: refresh only frame i's KV entries from exact attention
            cache_exact = exact_attention_kv(x_hat[max(0,i-w):i+1])   # window w=2
            cache_approx.replace(i, cache_exact)
            # recompute clarity score for frame i using the same reference
            if i in ref_frames:
                s_hat[i] = -||recompute_frame(x_hat[i], cache_approx[i]) - x_ref[i]||^2 / sigma_t^2
            else:
                s_hat[i] = 1 - t_i / T
        # re-check after refresh; if still violated, record for uncertified re-scheduling
        remaining = { i : s_hat[i] > s_hat[i-1] + delta }
        # re-scheduling applied AFTER certificate emission (uncertified intervention)
        for i in remaining:
            t_i = min(T, t_i + eta)          # eta = 1 step, push frame later in schedule
    emit(cache_approx, certificate={len(violations), len(remaining)})
```

Hyperparameters: tolerance `delta = 0.02` (clarity units); refresh window `w = 2` frames; re-schedule step `eta = 1`; reference subset size `K = 4` frames (fixed seed 0). The certificate is emitted per step and aggregated per video (fraction of steps with zero violations after correction).

**Why this design.** We chose a *reference-based clarity score* (negative reconstruction error against an exact-attention pass on a fixed subset) over a timestep-anchored residual-energy proxy because the anchor term is constant across the check (AR-Diffusion enforces non-decreasing timesteps by construction), so the ordering signal would reduce to comparing predicted-noise norms, which are systematically inflated by cache staleness in a frame-index-dependent way; the reference pass provides an independent ground truth that the correction cannot modify. We chose *exact-attention refresh for only the offending frame with a small window* over a full-cache refresh because FastFrames's profiling shows cache work is the latency bottleneck, so a per-violation refresh keeps the amortized cost proportional to the violation rate; the cost is that a violation caused by a distant frame's cache entry may need repeated refreshes. We chose a *re-scheduling fallback* over unbounded refresh because a persistent violation signals that the corruption schedule itself is inconsistent with the approximate cache, and re-scheduling is the only correction that changes the constraint rather than the approximation; the cost is a small latency increase on those steps. A domain expert would not describe this as 'FastFrames plus a checker' because the checker's output *changes the cache and the schedule*, i.e., the certificate is a control signal, not a post-hoc diagnostic — this is not the shape of a monitoring add-on.

**Why it measures what we claim.** The computational quantity `s_hat[i] = -||x_hat[i] - x_ref[i]||^2 / sigma_t^2` (for reference frames) measures *frame clarity* because it directly quantifies the deviation of the approximate-cache output from an exact-attention reference, under the assumption that the exact reference is a valid proxy for the clean frame; this assumption fails when the exact reference itself is noisy (e.g., at high noise levels), in which case `s_hat` reflects reference noise rather than cache error — which is why we restrict the certificate's validity claim to steps where the reference pass uses the same noise level as the approximate pass. The per-step `certificate` count measures *ordering preservation under approximation* because the check compares `s_hat` against the monotone ordering with tolerance `delta`, under the assumption that `delta` absorbs sampling stochasticity but not systematic cache bias; this fails when a biased cache shifts all frames uniformly (ordering preserved but quality degraded), in which case the certificate reports success while quality drops — a known blind spot we expose by reporting the residual-energy mean alongside the certificate.

## Contribution

['A per-step certified ordering check for approximate-cache autoregressive video diffusion, with a bounded correction (exact-attention refresh on offending frames plus timestep re-scheduling) that converts the latency/quality trade-off into a verifiable invariant.', 'A design principle: the corruption-timestep schedule itself supplies a free clarity anchor, so ordering violations under cache approximation can be detected without a learned quality model, and the certificate doubles as a control signal for cache refresh.', 'An evaluation protocol that reports violation rate, post-correction certificate rate, and residual-energy drift as separate axes, exposing the uniform-bias blind spot of ordering-only checks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Long benchmark videos requiring >N frames (e.g., 128 frames) | Exposes ordering drift over long horizons |
| Primary metric | Fraction of steps with zero post-correction violations (using reference-based clarity score) | Directly measures ordering preservation claim |
| Secondary metric | Per-frame LPIPS to clean reference (on held-out frames) | Closes the ordering-to-quality link |
| Baseline | AR-Diffusion (exact attention) | Reference for ordering without cache approximation |
| Baseline | FastFrames-style approximate KV cache | Tests whether approximation is the failure source |
| Baseline | AR-Diffusion + post-hoc clarity monitor | Tests certificate-as-control vs diagnostic |
| Ablation-of-ours | OrderGuard without re-scheduling fallback | Isolates contribution of schedule correction |
| Ablation-of-ours | OrderGuard with timestep-anchored residual-energy proxy (original proxy) | Tests whether reference-based clarity score is necessary |

### Why this setup validates the claim

The central claim is that approximate KV caching in autoregressive video diffusion breaks the non-decreasing clarity ordering that AR-Diffusion's corruption schedule assumes, and that a bounded, violation-triggered cache refresh plus schedule re-scheduling restores it. The exact-attention AR-Diffusion baseline establishes the ordering ceiling: if our method approaches it, the cache approximation was the culprit. The FastFrames-style approximate-cache baseline is the direct test of the failure hypothesis: it should show ordering violations concentrated where cache error accumulates, and our method should recover those without its latency. The post-hoc monitor baseline is the discriminating control: it observes the same violations but cannot correct them, so any gap between it and OrderGuard isolates the value of the certificate-as-control-signal design rather than detection alone. The no-re-scheduling ablation tests whether persistent violations genuinely require changing the schedule or whether refresh alone suffices. The timestep-anchored proxy ablation tests whether the reference-based clarity score is necessary for valid certification. The primary metric is the right one because it is exactly the quantity the method certifies and controls, making the test falsifiable rather than a proxy for generic quality; the secondary LPIPS metric closes the loop by linking ordering preservation to perceptual quality.

### Expected outcome and causal chain

**vs. AR-Diffusion (exact attention)** — On a long video where late frames need information from early frames, the exact-attention baseline maintains non-increasing clarity because every frame attends to true cached keys. Our method runs with an approximate cache, so its raw clarity score (reference-based) should show occasional early-frame inflation from cache noise; after correction we expect the zero-violation step fraction to sit close to the exact baseline, with the residual gap concentrated on steps where a distant-frame dependency exceeds the refresh window w=2. The observable signal is near-parity on ordering, with a small deficit on the longest-range dependency subset.

**vs. FastFrames-style approximate KV cache** — On a case where cache entries for an early frame drift, the approximate baseline produces a clarity inversion: a later frame appears clearer than an earlier one because the drifted cache inflates the early frame's reconstruction error against the exact reference, and no mechanism exists to notice or fix it. Our method detects the inversion via the monotone check and refreshes only the offending frame's window, so we expect a large gap in zero-violation step fraction on the drifted-cache subset and near-parity on steps where the cache is already accurate — a gap concentrated where the failure trigger lives, not uniform.

**vs. AR-Diffusion + post-hoc clarity monitor** — On the same drifted-cache instance, the monitor flags the inversion but leaves the cache and schedule untouched, so the violation persists across subsequent steps and accumulates. Our method converts the flag into a control action, refreshing the cache and re-scheduling persistent offenders, so we expect the monitor to show a rising violation count over steps while OrderGuard shows a flat, near-zero post-correction count — the diagnostic signature that the certificate is doing control work, not just reporting.

**vs. OrderGuard with timestep-anchored residual-energy proxy** — On the same drifted-cache instance, the timestep-anchored proxy reduces to comparing predicted-noise norms, which are systematically inflated by cache staleness in a frame-index-dependent way, so it should report violations that are not true clarity inversions and miss true inversions that do not manifest as residual-energy spikes. Our reference-based score should show higher agreement with the exact-attention ordering (measured by rank correlation between s_hat and the exact-reference clarity) and lower false-positive rate on steps where the cache is accurate.

### What would falsify this idea

If the zero-violation step fraction gain over the approximate-cache baseline is uniform across all subsets rather than concentrated on the drifted-cache and long-range-dependency subsets where the predicted failure mode occurs, the causal story is wrong and the correction is acting as generic smoothing rather than targeted repair. Likewise, if the no-re-scheduling ablation matches full OrderGuard, the schedule-correction half of the mechanism is unnecessary and the central claim about schedule inconsistency is unsupported. Additionally, if the reference-based clarity score does not correlate better with exact-attention ordering than the timestep-anchored proxy, the load-bearing assumption about the need for an independent reference is false.

## References

1. FastFrames: Video Generation at Interactive Latency
