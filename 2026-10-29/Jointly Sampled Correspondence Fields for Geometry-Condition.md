# Jointly Sampled Correspondence Fields for Geometry-Conditioned Video Diffusion

## Motivation

Existing geometry-conditioned video diffusion methods, such as OmniShowcase, treat reconstructed surface-normal maps as a static, temporally unaligned conditioning signal: normals come from multi-view reconstruction (VGGT, Depth Anything 3) with no per-frame correspondence to the video being generated. This structural assumption—that static synthetic multi-view assets are adequate ground truth for a dynamic generation task—means the model is never forced to reconcile the geometry it renders with the geometry it was conditioned on, so temporal consistency and physical plausibility of the conditioned frames remain unestablished. The meta-gap recurs across paper-trees: geometry is always an oracle, never a variable that the generative process must explain.

## Key Insight

Treating the per-frame geometry-to-video correspondence as a latent variable optimized jointly with the video latent under a cycle-consistency invariant turns temporal alignment from a post-hoc registration step into an identifiability condition: the generated frame must be the unique video frame whose differentiable normal render matches the conditioned normal map under the inferred correspondence.

## Method

**What it is.** We introduce **CycleField**, a geometry-conditioned video diffusion sampler that, at every denoising timestep, jointly optimizes the video latent $z_t$ and a per-frame correspondence field $\phi_t$ (a dense 2D warp from the conditioned normal-map turntable to each generated frame) under a cycle-consistency loss between the frame's rendered normal field and the conditioned normal map. Inputs: reference views, a turntable of surface-normal maps $\{N_k\}$, and a text prompt. Output: a video $\{x_i\}$ plus the inferred per-frame correspondences $\{\phi_i\}$.

**How it works.**

```pseudocode
# Precompute: latent geometry proxy G from VGGT on reference views
G = VGGT(reference_views)                      # differentiable mesh + normals
N_turntable = {N_k}_{k=1..K}                   # conditioned normal maps

# Initialize video latent z_T ~ N(0,I), correspondence field phi_T = identity
for t = T, T-1, ..., 1:
    # (1) Predict clean video estimate via the diffusion transformer
    x0_hat = DTM(z_t, t, cond=text, N_turntable)

    # (2) Differentiable normal render of the latent geometry at each frame
    for each frame i:
        N_hat_i = RenderNormals(G, camera_i(phi_t))   # camera pose warped by phi_t

    # (3) Cycle-consistency loss: rendered normals vs. conditioned normals
    #     under the current correspondence phi_t
    L_cycle = sum_i || Warp(N_turntable, phi_t,i) - N_hat_i ||_1

    # (4) Joint update: one gradient step on phi_t only (z_t frozen)
    phi_t <- phi_t - eta * grad_{phi_t} L_cycle      # eta = 0.01, 5 inner steps

    # (5) Denoising step on z_t using the *updated* correspondence
    z_{t-1} = DDIMStep(z_t, x0_hat, t, cond=phi_t)

return x_0 = Decode(z_0), phi_0
```

**Why this design.** We chose to optimize $\phi_t$ with a frozen $z_t$ at each timestep rather than jointly updating both latents in a single gradient step, because decoupling the two updates makes the cycle-consistency loss a well-posed registration problem at every step (the geometry proxy is fixed), accepting the cost that the correspondence cannot correct gross errors in the video latent until the next denoising step. We chose a differentiable normal renderer over a learned normal predictor because only the renderer enforces that the normal field is *induced* by the latent geometry—a learned predictor would let the cycle loss be satisfied by hallucinated normals, accepting the cost of requiring an explicit mesh from VGGT. We chose to condition the denoiser on $\phi_t$ (as a warped camera pose) rather than on the raw normal maps, because this forces the denoiser to consume the *aligned* geometry, accepting the cost that the denoiser must learn to interpret warped poses. A domain expert would not describe this as OmniShowcase plus a registration module: OmniShowcase conditions on a static normal turntable with no per-frame correspondence, whereas CycleField makes the correspondence a latent variable whose value is determined by the cycle-consistency invariant at sampling time—the alignment is an output of the generative process, not an input.

**Why it measures what we claim.** The cycle-consistency loss $L_{cycle}$ measures **temporal geometric alignment** because the assumption that the latent geometry proxy $G$ is a faithful reconstruction of the subject makes the rendered normal field $N_hat_i$ equivalent to the true surface normals of the generated frame; this assumption fails when the reference views under-constrain a surface (e.g., occluded regions), in which case $L_{cycle}$ reflects the proxy's hallucinated normals rather than the frame's true geometry. The correspondence field $\phi_t$ measures **per-frame temporal correspondence** because the assumption that the turntable's camera trajectory is a smooth warp of the generated video's implicit camera path makes $\phi_t$ equivalent to the frame-to-turntable alignment; this fails when the generated video violates the turntable's viewpoint ordering, in which case $\phi_t$ reflects a best-fit warp to an inconsistent trajectory. The joint optimization of $(z_t, \phi_t)$ measures **identifiability of the generated video under the geometry condition** because the assumption that the cycle loss has a unique minimum at the correct correspondence makes the joint optimum equivalent to the unique video- correspondence pair consistent with the geometry; this fails when the normal map is locally constant (e.g., flat regions), in which case the joint optimum is a family of correspondences and identifiability is only up to that ambiguity.

## Contribution

(1) A geometry-conditioned video diffusion sampler that treats per-frame geometry-to-video correspondence as a latent variable jointly optimized with the video latent under a cycle-consistency invariant, turning temporal alignment into an identifiability condition rather than post-hoc registration. (2) A differentiable normal-rendering pipeline that closes the loop between the latent geometry proxy and the conditioned normal turntable at every denoising timestep, providing a per-frame geometric consistency signal that static conditioning cannot supply. (3) An empirical demonstration that joint correspondence optimization improves temporal geometric consistency and physical plausibility of generated frames over static-oracle conditioning, with the inferred correspondence fields serving as a diagnostic of where the geometry condition is under-constrained.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-view subject scenes with turntable normal maps (e.g., OmniShowcase-style) | Provides ground-truth normals and reference views for cycle consistency. |
| Primary metric | Per-frame normal-cycle error under inferred correspondence | Directly measures geometric alignment claim without proxy hallucination. |
| Baseline | OmniShowcase | Static normal conditioning tests need for per-frame correspondence. |
| Baseline | Wan + post-hoc VGGT registration | Tests whether alignment can be added after generation. |
| Baseline | Single-reference S2V diffusion | Tests whether multi-view normals are necessary for alignment. |
| Baseline | Test-time optimization on frozen Wan: per-frame camera pose optimized via the same normal-render cycle loss | Isolates in-loop joint latent-correspondence coupling from post-hoc pose optimization. |
| Ablation-of-ours | CycleField without φ optimization (identity warps) | Isolates the contribution of latent correspondence optimization. |
| Ablation-of-ours | CycleField with φ optimized every N timesteps (N=2,4) | Tests accuracy-cost tradeoff of correspondence update frequency. |
| Ablation-of-ours | CycleField with φ optimized at coarse resolution (1/2, 1/4) | Tests accuracy-cost tradeoff of correspondence resolution. |

### Why this setup validates the claim

The central claim is that making per-frame correspondence a latent variable optimized by cycle consistency yields better temporal geometric alignment than conditioning on static normal maps or post-hoc registration. The dataset supplies both the normal turntable (as condition) and reference views from which VGGT builds the geometry proxy, so the cycle error can be computed against the same normals the method consumes—this is the right metric because it directly quantifies the alignment invariant, not a perceptual proxy. OmniShowcase tests the sub-claim that static conditioning is insufficient for per-frame alignment; Wan + post-hoc VGGT tests whether alignment can be recovered after generation; single-reference S2V tests whether multi-view geometry is necessary at all. The test-time optimization baseline on frozen Wan tests whether the gain comes from in-loop joint latent-correspondence coupling rather than from optimizing a per-frame camera pose via the same normal-render cycle loss on a frozen model. The ablation—freezing φ to identity—tests whether the gain comes from optimization rather than from the normal condition alone. The frequency and resolution ablations test the accuracy-cost tradeoff of the correspondence update. If the claim is correct, the gap to baselines should concentrate on frames where the subject rotates or the turntable viewpoint ordering is violated, not be uniform across all frames.

### Expected outcome and causal chain

**vs. OmniShowcase** — On a case where the subject undergoes a 90° turntable rotation and the normal map changes across frames, OmniShowcase produces frames whose surface normals drift from the conditioned turntable because it conditions on a static normal map with no per-frame correspondence, so the denoiser cannot know which normal map applies to which frame. Our method instead optimizes φ_t at each timestep so that the rendered normals from the latent geometry are cycle-consistent with the warped turntable, forcing the denoiser to consume the aligned geometry. We expect a noticeable gap in normal-cycle error on rotation-heavy frames but parity on near-frontal frames.

**vs. Wan + post-hoc VGGT registration** — On a case where the generated video violates the turntable's viewpoint ordering (e.g., a jump cut in the implicit camera path), the baseline produces a best-fit registration that warps the normal map to an inconsistent trajectory because post-hoc registration cannot alter the video latent. Our method instead updates φ_t inside the denoising loop, so each step's correspondence corrects the latent's geometry before the next step. We expect the baseline to show high cycle error on the jump-cut segment and our method to reduce it substantially, with a smaller gap on smooth segments.

**vs. Single-reference S2V diffusion** — On a case where the reference view under-constrains a surface (e.g., an occluded back side), the baseline produces hallucinated normals that are locally plausible but globally inconsistent with the turntable because it lacks multi-view geometry. Our method instead uses VGGT's mesh to render normals from the latent geometry, so the cycle loss penalizes hallucination even in occluded regions. We expect the baseline to show high cycle error on occluded surfaces and our method to match the turntable's normal ordering there, with a smaller gap on well-observed surfaces.

**vs. Test-time optimization on frozen Wan** — On a case where the generated video requires a per-frame camera pose that deviates from the turntable's ordering, the baseline optimizes a per-frame camera pose via the same normal-render cycle loss but on a frozen pretrained video model, so it cannot alter the video latent to correct geometric inconsistencies. Our method instead jointly updates φ_t and z_t inside the denoising loop, so the correspondence and the latent co-adapt. We expect the baseline to show higher cycle error on frames requiring latent correction, with a smaller gap on frames where pose alone suffices.

### What would falsify this idea

If our gain in normal-cycle error over OmniShowcase is uniform across rotation-heavy and near-frontal frames rather than concentrated where the predicted failure mode occurs, then the correspondence optimization is not doing the claimed work. If the ablation with identity φ performs as well as full CycleField, the central claim that per-frame correspondence must be a latent variable is wrong.

## References

1. OmniShowcase: Omnidirectional Subject-Consistent Video Generation from Multi-View Images
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. VGGT: Visual Geometry Grounded Transformer
4. MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement
