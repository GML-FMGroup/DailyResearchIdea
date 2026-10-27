# Differentiable Uncertainty-Aware Geometry Guidance for Video Diffusion

## Motivation

Existing controllable video generation methods such as WorldCast and Uni3C treat estimated depth and camera pose as fixed conditioning signals, computing geometric guidance (scaled point clouds, coverage masks) in a single open-loop forward pass that is never corrected by the discrepancy between rendered guidance and generated frames. This shared assumption that geometry is reliable enough to be open-loop is structurally flawed because depth estimation errors and pose misalignment propagate directly into the guidance without any gradient path from generation error back to the geometry estimate, leaving the model to compensate for bad geometry rather than correct it.

## Key Insight

Making the geometry a latent variable inferred by the same denoising objective that generates frames creates a closed-loop system where depth correction, scale, and coverage confidence are identified by generation error rather than by depth supervision alone, because the guidance operator is differentiable and its parameters receive gradients through the rendered guidance.

## Method

**What it is**: We introduce DIG (Differentiable Inferable Geometry), a guidance operator whose per-region depth correction, global scale, and coverage confidence are latent variables inferred jointly with a video diffusion transformer. DIG takes as input the estimated depth map, target camera trajectory, and noisy video latents, and outputs a rendered guidance tensor and coverage mask that are differentiable with respect to all geometry parameters.

**Load-bearing assumption (stated explicitly)**: DIG assumes that the gradient of the video diffusion denoising loss with respect to the geometry parameters (ΔD, s, c) carries a task-specific, geometrically meaningful signal that identifies true depth/scale/coverage errors, rather than being absorbed by the transformer as arbitrary conditioning noise. This assumption is not taken for granted: we add a calibration step (step 2b below) that verifies the gradient direction correlates with ground-truth depth error on a held-out calibration set before trusting the closed-loop updates, and we gate geometry updates by a learned reliability estimate trained against a proxy geometric residual (step 9b).

**How it works**:
```pseudocode
Input: initial frame I0, target camera trajectory T (poses), noisy video latents x_t, timestep t
Output: rendered guidance G, coverage mask M, updated geometry parameters

1. Estimate base depth D0 = DepthEstimator(I0)  # frozen, differentiable
2. Initialize latent geometry parameters:
   - per-region depth correction ΔD ∈ R^{H/8×W/8} (zero-initialized, coarse grid upsampled bilinearly to H×W)
   - global scale s ∈ R (initialized from point cloud alignment)
   - coverage confidence c ∈ R^{H×W} (initialized to 1)
2b. Calibration (one-time, before closed-loop training): on a held-out calibration set of 512 frames from CLACT with ground-truth depth, compute the gradient of the denoising loss w.r.t. ΔD at initialization; verify that its spatial correlation with ground-truth depth error exceeds a threshold (Pearson r > 0.3) before enabling closed-loop updates. If not, freeze ΔD and use only s and c.
3. Corrected depth D = D0 + upsample(ΔD)
4. Back-project D to point cloud P = s * K^{-1} * D * pixel_coords
5. Render P along trajectory T to obtain guidance G and coverage mask M (differentiable renderer)
6. Modulate guidance by confidence: G' = M * c * G
7. Feed G' into video diffusion transformer as conditioning (via adapter)
8. Compute denoising loss L = ||ε - ε_θ(x_t, t, G')||^2
9. Backpropagate L through G' to ΔD, s, c, and optionally to DepthEstimator
9b. Gate geometry updates by a learned reliability estimate r ∈ [0,1]^{H×W} (2-layer CNN, hidden=64, sigmoid output) trained against a proxy geometric residual (reprojection error of generated frames); update ΔD, s, c only where r > 0.5.
10. Update ΔD, s, c with gradient descent (learning rate 1e-4 for geometry, 1e-5 for diffusion transformer; weight decay 1e-4 on ΔD)
```

**Why this design**: We chose to parameterize depth correction as a per-region additive term ΔD on a coarse H/8×W/8 grid upsampled bilinearly rather than a free per-pixel field because a low-dimensional, identifiable parameterization prevents the correction from absorbing appearance/pose errors and keeps the latent space tractable (memory: ΔD has H*W/64 parameters, e.g., 1,024 for 256×256 frames), while still capturing local depth estimation errors that global transforms miss. We chose to infer geometry parameters jointly with the denoising objective rather than through a separate depth refinement loss because the generation error provides a task-specific signal that depth supervision alone cannot, accepting the risk of unstable gradients which we address by using a smaller learning rate for geometry parameters, stopping gradients to the depth estimator after a warm-up period of 1,000 steps, and gating updates by a learned reliability estimate. We chose to modulate guidance by a learned coverage confidence c rather than using binary masks because soft confidence allows the model to downweight unreliable regions continuously, accepting the cost that c must be regularized (entropy penalty weight 0.01) to avoid collapsing to zero. These decisions ensure DIG is not a simple combination of WorldCast's point cloud rendering with a learnable scale; rather, it closes the loop by making geometry a function of generation error, a structural property absent in prior open-loop pipelines.

**Why it measures what we claim**: The per-region depth correction ΔD measures the discrepancy between estimated depth and the depth implied by the generated video because the gradient of the denoising loss with respect to ΔD is non-zero only when the rendered guidance misaligns with the generated frames under the assumption that the diffusion model's denoising objective is sensitive to geometric inconsistencies; this assumption fails when the model ignores guidance (e.g., due to weak conditioning), in which case ΔD reflects noise rather than meaningful correction. The global scale s measures the alignment between the point cloud and the camera trajectory because the gradient of the loss with respect to s captures how scaling affects the projected guidance's consistency with generated motion, under the assumption that the trajectory is accurate; this fails when pose estimation is erroneous, in which case s absorbs pose errors. The coverage confidence c measures the reliability of guidance per pixel because c is optimized to minimize the denoising loss, under the assumption that unreliable regions contribute more to loss when trusted; this fails when the model overfits to noisy guidance, in which case c may remain high in unreliable areas. Together, these quantities operationalize the motivation-level concept of closed-loop geometry inference.

## Contribution

['A differentiable, uncertainty-aware guidance operator (DIG) that infers per-pixel depth correction, global scale, and coverage confidence as latent variables jointly with a video diffusion transformer via gradients from the denoising loss.', 'A training procedure that enables gradient flow from generation error back to geometry parameters through a differentiable renderer, making geometry a function of generation error rather than a one-shot forward pass.', 'An empirical demonstration that closed-loop geometry inference improves geometric consistency and camera control accuracy compared to open-loop pipelines like WorldCast and Uni3C.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | CLACT benchmark | Large-angle camera control with ground-truth trajectories. |
| Primary metric | Camera trajectory error (ATE) | Directly measures geometric consistency of generated video. |
| Baseline | WorldCast | Open-loop point cloud rendering with fixed scale. |
| Baseline | Uni3C | Precomputed 3D guidance without generation-error feedback. |
| Baseline | Wan (no geometry guidance) | Tests whether geometry conditioning helps at all. |
| Ablation-of-ours | DIG with frozen geometry (no ΔD, s, c updates) | Isolates effect of closed-loop inference. |
| Ablation-of-ours | DIG with depth-refinement loss (no denoising gradient) | Isolates whether generation-error signal, not just extra parameters, drives gain. |
| Ablation-of-ours | DIG with per-pixel ΔD (free H×W field) | Tests whether low-dimensional parameterization is necessary for identifiability. |
| Diagnostic | Gradient-direction correlation with ground-truth depth error | Verifies the load-bearing assumption that denoising gradient carries correction signal. |

### Why this setup validates the claim

This design forms a falsifiable test because each baseline probes a different link in the causal chain from closed-loop geometry inference to improved camera control. WorldCast tests the sub-claim that per-region depth correction matters: its fixed point cloud cannot adapt to local depth errors, so if our method's gain concentrates on scenes with depth estimation noise, the correction mechanism is validated. Uni3C tests the sub-claim that generation-error feedback matters: its precomputed 3D guidance is open-loop, so if our method outperforms it specifically when initial depth is biased, the closed-loop inference is validated. The frozen-geometry ablation tests whether the latent geometry parameters actually absorb the generation error signal; if performance matches the full model, the inference is inert. The depth-refinement-loss ablation tests whether the denoising gradient itself—not merely the presence of extra parameters—drives any gain: if it matches the full model, the generation-error signal is not the cause. The per-pixel ΔD ablation tests whether the low-dimensional parameterization is necessary for identifiability: if it matches or exceeds the full model, the coarse grid is not needed. Camera trajectory error is the right metric because it directly measures the geometric inconsistency that DIG is designed to reduce, and it is sensitive to both depth misalignment and scale errors, unlike perceptual metrics that can obscure geometric drift. The gradient-direction diagnostic directly verifies the load-bearing assumption: we compute the spatial correlation between the gradient of the denoising loss w.r.t. ΔD and the ground-truth depth error on CLACT; if the correlation is near zero, the gradient does not carry correction signal and the closed-loop updates are noise-fitting.

### Expected outcome and causal chain

**vs. WorldCast** — On a case where the initial depth map has a locally biased region (e.g., a thin structure misestimated as farther than it is), WorldCast renders a fixed point cloud that produces ghosting and trajectory drift because its global scale cannot correct per-pixel depth errors. Our method instead adjusts ΔD in that region via gradients from the denoising loss, so the rendered guidance aligns with generated frames. We expect a noticeable gap in trajectory error on scenes with high depth-estimation variance but parity on scenes with clean depth.

**vs. Uni3C** — On a case where the camera trajectory is accurate but the initial depth is globally mis-scaled (e.g., due to monocular scale ambiguity), Uni3C's precomputed 3D guidance remains misaligned because it never updates geometry from generation feedback. Our method infers the global scale s jointly with denoising, correcting the misalignment. We expect our trajectory error to be substantially lower on mis-scaled scenes, while both methods perform similarly when depth scale is already correct.

**vs. Wan (no geometry guidance)** — On a case requiring large-angle camera motion, Wan without geometry guidance produces plausible but geometrically inconsistent video because it has no explicit 3D signal to enforce view consistency. Our method supplies differentiable rendered guidance that penalizes geometric drift. We expect our method to show a large improvement in trajectory error on large-angle subsets, with smaller gains on small-angle subsets where the baseline already performs well.

**vs. DIG with depth-refinement loss** — On a case where the initial depth is biased, the depth-refinement variant updates ΔD using a separate depth loss (e.g., L1 to ground-truth depth on the calibration set) but receives no gradient from the denoising loss. If our method outperforms this variant, the generation-error signal—not just the presence of extra parameters—drives the gain, validating the closed-loop claim. We expect our method to show lower trajectory error on scenes with depth bias, while the depth-refinement variant may match on scenes with clean depth.

**vs. DIG with per-pixel ΔD** — On a case with local depth errors, the per-pixel variant has a free H×W field that can absorb appearance/pose errors, potentially overfitting. If our coarse-grid method matches or outperforms the per-pixel variant, the low-dimensional parameterization is sufficient and more identifiable. We expect the per-pixel variant to show higher variance and worse trajectory error on scenes with pose noise, while our method remains stable.

### What would falsify this idea

If our method's improvement over baselines is uniform across scenes with clean and noisy depth estimates, rather than concentrated where the predicted failure modes occur, then the closed-loop geometry inference is not the cause of any gain. Additionally, if the frozen-geometry ablation matches the full model's performance, the latent geometry parameters are not absorbing the generation error signal, falsifying the central claim. If the gradient-direction diagnostic shows near-zero correlation between the denoising gradient w.r.t. ΔD and ground-truth depth error, the load-bearing assumption fails and the closed-loop updates are noise-fitting.

## References

1. WorldCast: World-Consistent Character Video Generation with Large-Angle Camera Control
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation
