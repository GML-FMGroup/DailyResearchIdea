# Bake-Time Integrated Positional Encoding and Multi-Resolution Prefiltering for Alias-Free Baked Neural Rendering

## Motivation

Test-time baking (TTB) compresses cross-view attention into a lightweight MLP that decodes novel views, but it bakes only the point-sampled mapping and discards Zip-NeRF's scale-aware machinery (integrated positional encoding and prefiltering), so the baked operator is aliased at resolutions and camera distances it was not baked for. This is a concrete instance of a field-level residual gap: frequency- and scale-aware rendering consistency is not preserved when a scene representation is distilled into a rendering operator, because distillation targets point samples rather than the continuous radiance integral. No single branch has made anti-aliasing a property of the baked rendering operator itself rather than of the underlying scene representation.

## Key Insight

If the baked MLP is trained to reproduce the original decoder's output averaged over the conical frustum footprint of each ray at every training resolution, then the baked operator learns the scale-conditioned integral of the radiance field rather than a point-sampled approximation, so alias-free rendering becomes intrinsic to the operator and transfers across resolutions without re-baking.

## Method

**What it is.** We introduce BIPE (Baked Integrated Positional Encoding), a test-time baking procedure that produces an alias-free lightweight MLP. Inputs: the frozen decoder-only view-synthesis model (cross-view attention decoder), a set of training rays with multi-resolution pixel footprints, and the per-scene context tokens. Output: a baked MLP `f_θ` that maps (ray origin, ray direction, conical footprint) to RGB and renders alias-free at arbitrary resolution.

**Load-bearing assumption (stated explicitly).** The baked MLP can learn a footprint-conditioned, scale-integrated radiance operator from a finite set of training footprints and generalize to held-out resolutions (2×, 4×) it never saw, because IPE(d, footprint) analytically encodes the scale integral rather than merely interpolating between sampled footprints. To make this assumption testable rather than asserted, we (i) supervise the student at a continuous distribution of footprints (randomly sampled per ray each epoch, including footprints larger and smaller than any baking resolution) so the operator is constrained across the full scale domain, and (ii) verify that the student's input carries the bandlimit information by comparing against a footprint-blind student in the experiment.

**How it works.**

```pseudocode
# Phase 1: footprint-aware teacher queries
for each training view v at resolution r in {r_1/4, r_1/2, r_1}:
    for each pixel p in view v:
        ray o,d = camera_ray(v, p)
        footprint = pixel_solid_angle(v, p, r)   # conical frustum half-angle
        # Zip-NeRF-style integrated positional encoding (IPE) of the footprint
        ipe = integrated_pe(d, footprint, L=16, r_max=0.5)   # analytic Gaussian approx
        # Multi-sample the teacher over the footprint to form the anti-aliased target
        target = (1/K) * sum_{k=1..K} Decoder(o, d + δ_k, context_tokens)
                 with δ_k ~ uniform_in_cone(footprint), K=8
        store (o, d, footprint, ipe, target)

# Phase 2: bake the alias-free operator
initialize MLP f_θ with input [o, ipe(d, footprint)] and hidden width 256, 4 layers
for epoch in 1..E (E=2000):
    batch = sample stored tuples
    # Continuous footprint supervision: for each ray, resample footprint from a log-uniform
    # distribution over [footprint_min/4, footprint_max*4] each epoch, where footprint_min
    # and footprint_max are the smallest and largest footprints in the stored set.
    footprint_sampled = log_uniform(footprint_min/4, footprint_max*4)
    ipe_sampled = integrated_pe(d, footprint_sampled, L=16, r_max=0.5)
    # Recompute target for the sampled footprint via K=8 Monte-Carlo teacher queries
    target_sampled = (1/K) * sum_{k=1..K} Decoder(o, d + δ_k, context_tokens)
                     with δ_k ~ uniform_in_cone(footprint_sampled)
    pred = f_θ(concat(o, ipe_sampled))
    loss = ||pred - target_sampled||^2 + λ * ||f_θ(o, ipe(d, footprint_j)) - f_θ(o, ipe(d, footprint_k))||^2
           # λ=0.1: cross-resolution consistency between footprints j,k of the same ray
           # (j,k are two footprints sampled from the same log-uniform distribution)
    update θ by Adam(lr=1e-3)

# Phase 3: rendering
render(o,d,footprint) = f_θ(concat(o, integrated_pe(d, footprint)))
```

**Why this design.** We chose to place integrated positional encoding (IPE) in the input of the baked MLP rather than only in the teacher because the baked MLP must itself be a function of the footprint; putting IPE only in the teacher would leave the student footprint-blind, accepting the cost that the input dimensionality grows and the MLP must learn a scale-conditioned function. We chose to train against a Monte-Carlo average over K=8 samples inside the conical frustum rather than a single point sample because the target must be the anti-aliased integral the original decoder would produce; this accepts the cost of K× teacher forward passes per training ray. We chose a cross-resolution consistency loss between two footprints of the same ray rather than only per-footprint regression because per-footprint regression does not constrain the operator to be scale-consistent between sampled resolutions; this accepts the cost of pairing rays across resolutions during baking. We chose to resample footprints continuously from a log-uniform distribution over [footprint_min/4, footprint_max*4] each epoch rather than training only on the three discrete baking resolutions because the load-bearing assumption requires the operator to be constrained across the full scale domain, not interpolated between three fixed resolutions; this accepts the cost of recomputing the K=8 Monte-Carlo target for each sampled footprint. A domain expert would not describe this as "Zip-NeRF + TTB": Zip-NeRF modifies the scene representation (hash grid features) to be prefiltered, whereas BIPE modifies the distillation target and the student's input encoding so that the baked rendering operator itself carries the scale integral, which is a property neither Zip-NeRF nor TTB establishes.

**Why it measures what we claim.** The conical footprint half-angle `footprint` measures the scale of the rendering operator because the assumption that the pixel solid angle equals the cone half-angle holds for small angles and pinhole cameras; this assumption fails at extreme wide-angle or fisheye cameras, in which case the footprint reflects a geometric approximation rather than the true pixel support. The integrated positional encoding `ipe(d, footprint)` measures frequency-aware scale conditioning because the assumption that the Gaussian approximation of the frustum integral equals the true integral holds when the positional encoding frequency is high relative to the footprint; this assumption fails when the footprint is so large that the encoding's highest frequencies are fully attenuated, in which case IPE reflects only the low-frequency band. The Monte-Carlo average `target` measures the anti-aliased radiance integral because the assumption that K uniform cone samples are an unbiased estimator of the frustum integral holds for any K; this assumption fails when the decoder is non-linear at sub-pixel scale (e.g., sharp specular edges), in which case the target reflects a noisy estimate rather than the exact integral. The cross-resolution consistency loss measures scale-consistency of the baked operator because the assumption that two footprints of the same ray should map to the same underlying radiance holds when the scene is locally smooth at the coarser footprint; this assumption fails at occlusion boundaries, in which case the loss reflects a smoothing bias rather than true consistency. The continuous footprint resampling measures coverage of the scale domain because the assumption that a log-uniform distribution over [footprint_min/4, footprint_max*4] spans the held-out test footprints (2×, 4×) holds when the test footprints fall within this range; this assumption fails if the test resolution exceeds 4× the largest baking footprint, in which case the operator is extrapolating beyond its training distribution.

## Contribution

["A test-time baking procedure (BIPE) that embeds integrated positional encoding and conical-frustum prefiltering directly into the baked MLP's input and training target, making anti-aliasing a property of the rendering operator rather than of the scene representation.", 'A multi-resolution anti-aliasing loss combining footprint-conditioned regression with a cross-resolution consistency term, which yields a baked operator that renders consistently across resolutions without re-baking.', "An empirical demonstration that the baked operator's frequency response matches the teacher's prefiltered response across camera distances and resolutions, whereas point-sampled baking produces aliasing."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Multi-resolution synthetic scenes (e.g., Blender, NSVF) | Controlled pixel footprints across resolutions test scale conditioning. |
| Primary metric | PSNR at held-out resolutions (2×, 4× up/down) | Directly measures alias-free rendering at novel scales. |
| Baseline | Decoder-only LSVM (TTB teacher) | Tests whether baking alone suffices without footprint-aware inputs. |
| Baseline | TTB with TTT layers | Tests whether test-time training already handles resolution changes. |
| Baseline | Zip-NeRF-style prefiltered hash grid | Tests whether prefiltering the scene representation is necessary. |
| Ablation-of-ours | BIPE without cross-resolution consistency loss | Isolates the effect of scale-consistency regularization. |
| Ablation-of-ours | BIPE with footprint-blind student (IPE only in teacher) | Tests whether operator-level footprint conditioning is distinct from representation prefiltering. |
| Ablation-of-ours | BIPE with discrete footprint training (three baking resolutions only) | Tests whether continuous footprint resampling is necessary for held-out scale generalization. |
| Ablation-of-ours | BIPE with K-sweep (K=1,4,8,16) | Shows accuracy/compute trade-off of Monte-Carlo target. |
| Metric | Occlusion-boundary-stratified PSNR | Tests the smoothness assumption of the consistency loss. |
| Metric | Target variance vs. K and footprint size | Grounds the anti-aliased-integral claim in measured error. |
| Metric | Baking wall-clock time and teacher forward-pass budget | Judges feasibility of the baking cost. |

### Why this setup validates the claim

This setup forms a falsifiable test because it isolates the central claim—that placing integrated positional encoding (IPE) in the baked MLP's input, trained on footprint-aware Monte-Carlo targets with continuous footprint resampling, yields an alias-free operator that generalizes across resolutions. The decoder-only LSVM baseline (TTB teacher) tests whether baking alone, without footprint-aware student inputs, can achieve scale generalization; if it matches BIPE, our input-encoding claim is unnecessary. The TTB with TTT baseline tests whether test-time training already provides resolution robustness, which would undermine the need for explicit footprint conditioning. The Zip-NeRF-style prefiltered hash grid baseline tests whether prefiltering the scene representation, rather than the rendering operator, is the key mechanism. The ablation without cross-resolution consistency tests whether the consistency loss is necessary for scale-consistent behavior. The footprint-blind student ablation (IPE only in teacher) directly tests whether operator-level footprint conditioning is a distinct mechanism from representation prefiltering, as the reviewer requested. The discrete-footprint ablation tests whether continuous footprint resampling is necessary to constrain the operator across the full scale domain, addressing the load-bearing assumption. The K-sweep and target-variance measurements ground the anti-aliased-integral claim in measured error rather than assertion. PSNR at held-out resolutions is the right metric because aliasing manifests as high-frequency errors that PSNR penalizes; a method that is truly alias-free should maintain PSNR as resolution changes, while footprint-blind methods will show a sharp drop when tested at resolutions not seen during baking. The occlusion-boundary-stratified PSNR breakdown tests the smoothness assumption of the consistency loss: if the loss induces smoothing bias at occlusion boundaries, we expect lower PSNR there for BIPE with consistency loss than without. This combination allows us to attribute any gain to the specific mechanism of footprint-aware input encoding and consistency training, rather than to baking or prefiltering alone.

### Expected outcome and causal chain

**vs. Decoder-only LSVM (TTB teacher)** — On a case where the test resolution is 2× the highest baking resolution (e.g., rendering a high-frequency texture at close range), the baseline produces aliased outputs (jagged edges, moiré) because its baked MLP never sees the footprint and thus cannot adapt its frequency content to the pixel scale. Our method instead conditions on IPE(d, footprint), which analytically attenuates high frequencies as the footprint grows, so the baked operator inherently low-pass filters at the correct scale. We expect a noticeable PSNR gap (e.g., 1–2 dB) on high-frequency regions at 2× resolution, but near parity at the baking resolution.

**vs. TTB with TTT layers** — On a case where the scene contains sharp specular highlights that vary at sub-pixel scale, the baseline produces noisy or blurred highlights when rendering at a different resolution because its test-time training adapts to the training resolution but does not explicitly model the pixel footprint integral. Our method instead trains against a Monte-Carlo average over K=8 samples within the conical footprint, so the target itself is the anti-aliased integral, and the student learns to reproduce it. We expect a visible reduction in highlight aliasing (e.g., 0.5–1 dB PSNR gain) on specular subsets, with parity on diffuse regions.

**vs. Zip-NeRF-style prefiltered hash grid** — On a case where the footprint is large (e.g., rendering at 1/4 resolution), the baseline produces over-blurred results because prefiltering the hash grid features attenuates all frequencies uniformly, losing detail that should be preserved at the coarse scale. Our method instead uses IPE to condition the MLP on the footprint, allowing it to learn a scale-dependent mapping that preserves low-frequency structure while suppressing aliasing. We expect our method to outperform the baseline on coarse-resolution PSNR (e.g., 0.5–1 dB) while being comparable at fine resolutions, demonstrating that operator-level footprint conditioning is more effective than representation prefiltering.

**vs. BIPE with footprint-blind student (IPE only in teacher)** — On held-out 2× and 4× resolutions, the footprint-blind student cannot adapt its frequency content to the pixel scale because its input lacks the footprint, so it aliases similarly to the decoder-only LSVM baseline. Our method conditions on IPE(d, footprint) in the student input, so we expect a PSNR gap of 1–2 dB at held-out resolutions, establishing that operator-level footprint conditioning is a distinct mechanism from representation prefiltering.

**vs. BIPE with discrete footprint training (three baking resolutions only)** — On held-out 2× and 4× resolutions, the discrete-footprint variant interpolates between three fixed footprints and may fail to extrapolate, producing aliasing or over-blurring. Our method resamples footprints continuously from a log-uniform distribution over [footprint_min/4, footprint_max*4], constraining the operator across the full scale domain. We expect a PSNR gain of 0.5–1 dB at held-out resolutions for continuous over discrete training, supporting the load-bearing assumption.

**vs. BIPE without cross-resolution consistency loss** — On scenes with locally smooth radiance, the ablation may match our full method, but at occlusion boundaries the consistency loss could induce smoothing bias. We expect the full method to show a small PSNR gain (0.2–0.5 dB) on smooth regions and a small PSNR drop (0.1–0.3 dB) at occlusion boundaries, as measured by the occlusion-boundary-stratified PSNR breakdown, revealing the trade-off.

**K-sweep (K=1,4,8,16)** — We expect target variance to decrease with K, with diminishing returns beyond K=8, and PSNR at held-out resolutions to improve with K up to 8, then plateau, showing K=8 is a reasonable accuracy/compute trade-off.

**Target variance vs. K and footprint size** — We expect variance to increase with footprint size and decrease with K, grounding the Monte-Carlo integral claim in measured error.

**Baking wall-clock time and teacher forward-pass budget** — We expect baking to require approximately 2000 epochs × batch size 4096 rays × K=8 teacher forward passes, totaling on the order of 10^7 teacher forward passes per scene, and wall-clock time of a few GPU hours, demonstrating feasibility.

### What would falsify this idea

If our method's PSNR gain over the decoder-only LSVM baseline is uniform across all frequency subsets and resolutions, rather than concentrated on high-frequency regions and held-out resolutions where aliasing is predicted to occur, then the central claim that footprint-aware input encoding produces alias-free rendering is wrong. Additionally, if the ablation without cross-resolution consistency matches our full method on held-out resolutions, then the consistency loss is not necessary for scale generalization, falsifying the claimed mechanism; and if the footprint-blind student matches our full method at held-out resolutions, then operator-level footprint conditioning is not a distinct mechanism.

## References

1. TTB: Test-time Baking for Efficient Rendering of Decoder-only View Synthesis Models
