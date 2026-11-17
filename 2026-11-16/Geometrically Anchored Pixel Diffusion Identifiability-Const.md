# Geometrically Anchored Pixel Diffusion: Identifiability-Constrained RGB-Disparity Video Generation

## Motivation

Pixel-space video diffusion with joint modality generation, as demonstrated by JVT (arxiv:2611.18549), produces RGB and disparity outputs that are only loosely coupled through a shared denoising trajectory; there is no structural guarantee that the learned score corresponds to a unique geometric latent. This is a meta-level problem: across video-diffusion branches, camera-conditioned generation is evaluated by rendering quality and camera-following metrics, not by whether the underlying 3D structure is identifiable from the generated pixels. The root cause is that the diffusion objective is defined over pixel residuals alone, so the score can fit the data while remaining invariant to a one-parameter family of depth-disparity scalings that produce identical RGB under a fixed camera — JVT's own camera-conditioned demonstration inherits this ambiguity.

## Key Insight

A per-step scale-and-shift consistency residual between the RGB-predicted and disparity-predicted geometry, enforced during denoising, removes the one-parameter ambiguity of monocular scale from the score function, so the shared pixel-space trajectory is forced onto a unique geometric latent rather than a depth-equivalent orbit.

## Method

**What it is.** GeomAnchor-JVT is a training-time regularizer and inference-time projection added to the JVT pixel-space DiT. Inputs: a batch of camera-conditioned RGB-disparity video clips with per-frame camera intrinsics and extrinsics, plus ground-truth disparity maps for the training clips (used only for the anchor residual, not for the diffusion loss). Outputs: a diffusion score that, at every denoising step, is consistent with a single scale-consistent disparity field, plus a per-step anchor loss used during training.

**How it works.**

```pseudocode
# Training step for GeomAnchor-JVT
for each clip (x_rgb, x_disp, cameras, gt_disp) in loader:
    t ~ Uniform(1, T)
    eps_rgb, eps_disp ~ N(0, I)
    x_t_rgb   = sqrt(abar_t)*x_rgb  + sqrt(1-abar_t)*eps_rgb
    x_t_disp  = sqrt(abar_t)*x_disp + sqrt(1-abar_t)*eps_disp
    # Single shared DiT forward; output splits into two heads
    eps_hat_rgb, eps_hat_disp = DiT(x_t_rgb, x_t_disp, t, cameras)
    # x0-estimates via the standard DDPM posterior mean
    x0_rgb  = (x_t_rgb  - sqrt(1-abar_t)*eps_hat_rgb ) / sqrt(abar_t)
    x0_disp = (x_t_disp - sqrt(1-abar_t)*eps_hat_disp) / sqrt(abar_t)
    # Geometry anchor: lift RGB to a point cloud, project into disparity space
    cloud   = backproject(x0_rgb, x0_disp, cameras)          # per-pixel 3D points
    disp_proj = project_to_disparity(cloud, cameras)          # analytic re-render
    # Scale-and-shift identifiability residual against ground-truth disparity
    a, b = lstsq_scale_shift(disp_proj, gt_disp)              # closed-form per frame
    anchor = mean( (disp_proj - (a*gt_disp + b))^2 )
    loss = MSE(eps_hat_rgb, eps_rgb) + MSE(eps_hat_disp, eps_disp) + lambda_anchor * anchor
    # lambda_anchor = 0.1, annealed linearly from 0 over first 5k steps

# Inference: per-step anchor projection
for t in T..1:
    eps_hat_rgb, eps_hat_disp = DiT(...)
    x0_rgb, x0_disp = posterior_mean(...)
    a, b = lstsq_scale_shift(project(backproject(x0_rgb, x0_disp)), x0_disp)
    x0_disp = (x0_disp - b) / a          # project onto the identifiable manifold
    x_{t-1} = ddpm_step(x_t, eps_hat, t) using projected x0
```

**Why this design.** We chose a closed-form least-squares scale-and-shift residual over a learned geometric-consistency discriminator because the ambiguity being removed is exactly a two-parameter group action (scale and shift in disparity) on the depth field, so the residual can be computed analytically and has no training signal of its own to collapse; the cost is that non-affine distortions (lens distortion, rolling shutter) are not captured and must be handled by camera preprocessing. We chose to apply the anchor at every denoising step rather than only at t=0 because the score function is step-indexed and an end-only penalty leaves early-step trajectories free to drift into a different depth orbit; the cost is roughly 1.4x wall-clock per training step from the backproject/project pass. We chose a single shared DiT with two heads over two separate networks because the motivation is that the shared pixel-space score must be tied to one geometric latent — separate networks would reintroduce the very independence we are removing; the cost is that the two heads compete for capacity and disparity quality degrades slightly in the first 10k steps. We chose to anneal lambda_anchor from zero rather than fixing it because at high noise the x0-estimate is too unreliable for the backprojection to be meaningful; the cost is a longer warmup. This is not the same as a learned disentanglement head (which assumes the two modalities are statistically separable) nor as the standard joint-generation setup of JVT: JVT trains two heads on a shared trajectory with no constraint tying them to a common 3D structure, so a domain expert would describe that as multi-task diffusion, whereas here the anchor is a group-theoretic projection that makes the shared score identifiable.

**Why this measures what we claim.** The anchor residual `mean((disp_proj - (a*gt_disp + b))^2)` measures geometric identifiability because the least-squares fit of the affine group (a,b) is the maximal symmetry of the monocular depth-disparity map under a fixed camera, so any residual orthogonal to that group is structure the score cannot explain away by rescaling; this assumption fails when the camera intrinsics are misestimated or when the scene contains a dominant planar surface whose disparity is itself affine, in which case the residual reflects planar-fit error rather than true geometric inconsistency. The per-step projection `x0_disp <- (x0_disp - b)/a` measures trajectory anchoring because it enforces that the denoised disparity at every step lies on the same affine orbit as the RGB-derived point cloud, which is equivalent to a single geometric latent only under the assumption that the camera model is exact and the RGB-to-depth backprojection is invertible; this fails under occlusion or disocclusion, where pixels visible in one frame have no RGB correspondence, and the metric then reflects occlusion noise rather than depth ambiguity. The `anchor` term in the loss measures the degree to which the shared score has collapsed onto a unique depth orbit, because minimizing it is equivalent to maximizing the likelihood of the disparity head under the RGB-implied geometry up to the affine group; this equivalence fails when the two heads are trained with different noise schedules, in which case the residual reflects schedule mismatch rather than geometric disagreement.

## Contribution

["GeomAnchor-JVT: a training-time scale-and-shift identifiability regularizer and per-step inference projection that ties JVT's shared pixel-space score to a unique geometric latent for camera-conditioned RGB-disparity video generation.", 'A design principle: monocular depth ambiguity in joint pixel-space diffusion is a two-parameter group action that can be removed analytically by a least-squares projection at every denoising step, without a learned discriminator or separate geometry network.', 'An evaluation protocol measuring geometric identifiability (cross-view depth consistency under synthetic camera motion) alongside standard camera-following and generation-quality metrics.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Camera-conditioned RGB-disparity video clips (JVT protocol) with ground-truth disparity | Matches the joint pixel-aligned modality JVT already defines. |
| Primary metric | Geometric identifiability residual (affine-orthogonal disparity error vs. ground-truth) | Directly measures the group-theoretic quantity the anchor targets. |
| Baseline | Vanilla JVT joint heads (no anchor) | Isolates the effect of the anchor term itself. |
| Baseline | JVT + end-only anchor (t=0 penalty) | Tests whether per-step anchoring, not just any anchoring, matters. |
| Baseline | JVT + learned consistency discriminator | Tests analytic group projection against learned separability. |
| Baseline | JVT + standard scale-invariant depth loss (BerHu/Eigen) on disparity head | Tests whether the cross-modal projection adds beyond scale-invariant depth supervision. |
| Ablation-of-ours | GeomAnchor-JVT without inference projection | Separates training-time regularizer from inference-time projection. |

### Why this setup validates the claim

The central claim is that a shared pixel-space score is only identifiable up to an affine disparity group, and that anchoring the score to the RGB-implied point cloud at every denoising step removes exactly that ambiguity. The dataset choice is forced: camera-conditioned RGB-disparity clips with ground-truth disparity are the only setting where a reference disparity and a known camera model coexist, so the affine-orthogonal residual is computable against an external reference rather than assumed. Vanilla JVT tests the sub-claim that unconstrained joint heads drift across the affine orbit; the end-only anchor baseline tests the sub-claim that the constraint must be step-indexed rather than terminal; the learned-discriminator baseline tests the sub-claim that closed-form group projection beats a learned separability prior; the scale-invariant depth loss baseline tests whether the cross-modal projection is the novel ingredient beyond standard scale-invariant depth supervision. The ablation isolates whether the gain comes from the training regularizer or from the inference projection, which is the difference between "the score learned it" and "the sampler forced it." The primary metric is the right one because it is orthogonal by construction to the two-parameter symmetry we claim is the only unidentifiable direction: if the method works, residual should shrink on the non-affine component while leaving the affine component untouched; a metric like PSNR or FVD would conflate both and cannot distinguish our mechanism from generic quality improvement.

### Expected outcome and causal chain

**vs. Vanilla JVT joint heads** — On a clip where the camera translates laterally through a scene with strong depth variation, vanilla JVT produces disparity that is locally plausible but globally rescaled frame-to-frame, because nothing ties the two heads' x0-estimates to a common 3D structure and the shared trajectory is free to drift along the affine orbit. Our method instead backprojects RGB using the disparity head's x0, re-projects into disparity space, and penalizes the affine-orthogonal residual against ground-truth disparity, so the score is pushed onto one orbit. We expect a clear residual gap on clips with camera motion and depth gradients, but near-parity on near-frontal, shallow-depth clips where the orbit is already tight.

**vs. JVT + end-only anchor** — On a clip with a fast pan, the end-only baseline produces early denoising steps that drift into a wrong depth orbit and a final frame that is self-consistent but inconsistent with the RGB trajectory, because a terminal penalty cannot correct a trajectory that has already left the manifold. Our method applies the projection at every step, so the x0-estimate is re-anchored before the error compounds. We expect the end-only baseline to show a residual that is small at t=0 yet large in mid-trajectory consistency, while ours stays flat across steps — a diagnostic step-indexed residual profile, not just a lower mean.

**vs. JVT + learned consistency discriminator** — On a clip with a dominant planar background, the learned discriminator baseline produces a disparity field that satisfies the discriminator yet is systematically biased, because the discriminator absorbs the affine ambiguity into its own weights and cannot separate scale from structure. Our closed-form least-squares residual has no parameters to absorb the ambiguity, so the affine component is projected out exactly. We expect the discriminator baseline to match ours on planar-heavy subsets but to lag on subsets with mixed planar and non-planar structure, where the group action and the scene structure are separable.

**vs. JVT + standard scale-invariant depth loss** — On a clip with mixed depth structure, the scale-invariant depth loss baseline reduces affine-orthogonal error on the disparity head alone but cannot enforce cross-modal consistency, because it operates only on the disparity output and ignores the RGB-implied point cloud. Our method uses the RGB backprojection as an independent geometric signal, so the residual is tied to the shared trajectory rather than to the disparity head in isolation. We expect the scale-invariant baseline to match ours on disparity-only metrics but to lag on the cross-modal residual, isolating the projection as the novel ingredient.

### What would falsify this idea

If the residual gain over vanilla JVT is uniform across all subsets — including near-frontal, shallow-depth clips where the affine orbit is already tight — rather than concentrated on clips with camera motion and mixed depth structure, then the anchor is acting as a generic smoothness prior rather than removing a specific group ambiguity, and the central claim is wrong. Likewise, if the ablation without inference projection matches the full method, the claimed per-step trajectory anchoring is doing no work and the effect is entirely a training-time regularizer.

## References

1. JVT: Just Video Transformer makes pixel-space video diffusion simple
