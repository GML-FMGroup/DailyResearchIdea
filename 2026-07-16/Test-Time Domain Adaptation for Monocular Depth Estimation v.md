# Test-Time Domain Adaptation for Monocular Depth Estimation via Self-Supervised Geometric Consistency

## Motivation

Existing feed-forward geometry networks such as MetaView assume training and test domains are similar, causing failure on out-of-distribution scenes. The root cause is that these networks are frozen at test time, lacking a mechanism to adjust to the specific target image statistics. MetaView's reliance on pre-trained monocular depth priors limits generalization to unseen lighting, texture, or structure. We propose to bridge this gap by online fine-tuning using self-supervised consistency losses that exploit geometric constraints inherent in the depth estimation task.

## Key Insight

Because depth induces a geometric transformation between nearby views, photometric consistency across synthesized small-baseline views provides a domain-invariant supervisory signal that does not require ground truth or paired data, enabling adaptation to any target scene.

## Method

We propose Test-Time Geometry Adaptation (TTGA).

### (A) What it is
TTGA is a lightweight optimization procedure that fine-tunes a pre-trained feed-forward geometry network (e.g., the depth predictor from MetaView) on a single target image by minimizing a self-supervised consistency loss between the original image and a synthetic view generated via the network's own depth estimate.

### (B) How it works
```python
# Input: target image I_t, pre-trained depth network f_θ (MetaView checkpoint)
# Hyperparameters: pose perturbation range (rot=10°, trans=0.1*scene_scale), λ=0.5, lr=1e-4, N=50
# Mask: M = forward-backward consistency mask (exclude pixels where round-trip error > 0.1)  # NEW
# Mask: M_nonlam = 1 if |∇I_t| < 0.5 else 0  # exclude likely non-Lambertian regions (specularities)  # NEW

for step in range(N):
    δP = sample_random_pose(rotation=10°, translation=0.1 * median_depth(f_θ(I_t)))
    d_t = f_θ(I_t)  # current depth estimate
    I_syn = warp(I_t, d_t, δP)  # inverse warp with bilinear sampling
    # Compute forward-backward mask
    I_recon = warp(I_syn, d_t', -δP)  # d_t' is depth from I_syn (same network)  # NEW
    M_fb = (|I_t - I_recon|_1 < 0.1)  # NEW
    # Compute non-Lambertian mask
    M_grad = (|∇I_t| < 0.5)  # NEW
    M = M_fb & M_grad  # NEW
    L_photo = mean(|I_t - I_syn|_1 * M)  # masked mean  # MODIFIED
    L_geo = edge_aware_smoothness(d_t, I_t)  # penalty on depth gradients not aligned with image gradients
    L = L_photo + λ * L_geo
    θ ← Adam(θ, L, lr=1e-4)
```
We rely on the assumption that the scene is locally Lambertian and that the pose perturbation is small enough to avoid disocclusions; these assumptions are reasonable for many scenes but may fail. To mitigate, we add a forward-backward consistency mask to exclude disoccluded pixels and a gradient-based mask to exclude likely non-Lambertian regions. The photometric loss L_photo measures domain invariance because it penalizes inconsistencies in the warped image that arise from incorrect depth under the assumed pose perturbation; the key assumption is that the scene is Lambertian and the pose perturbation is small enough that the warping is a good model of appearance change. This assumption fails when the scene contains non-Lambertian surfaces (e.g., specularities) or large pose changes causing disocclusions, in which case the loss reflects non-geometric appearance mismatch rather than depth error. Specifically, specularities cause large photometric error even with correct depth, so the loss no longer correlates with depth accuracy. Similarly, disoccluded pixels produce large errors regardless of depth. Therefore, our loss is a valid proxy only when these failure modes are absent, and our masking strategy attempts to exclude such pixels. The geometric smoothness L_geo measures structural plausibility under the assumption that depth is piecewise smooth at edges; this fails at depth discontinuities aligning with texture edges, where smoothness incorrectly penalizes boundaries. The optimization directly updates network parameters θ to minimize these losses, which by construction reduces the domain gap between pre-training and target scene because the losses are self-supervised and do not rely on any labeled data from the target domain.

## Contribution

(1) A novel test-time adaptation framework for monocular depth estimation that uses self-supervised geometric consistency from synthetic small-baseline views to adapt a pre-trained network to individual target images. (2) The finding that even a few optimization steps (50) on a single image significantly improves depth accuracy on out-of-distribution scenes without requiring any ground truth or additional data. (3) An empirical demonstration that the adaptation is robust to the choice of random pose perturbations, as long as the pose magnitude is constrained to avoid disocclusions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | RealEstate10K | Large viewpoint changes, ground truth views. |
| Primary metric | PSNR | Primary measure of image fidelity. |
| Baseline 1 | MetaView (no adaptation) | Pre-trained depth without test-time adaptation. |
| Baseline 2 | Depth Anything v3 (no adaptation) | Pre-trained depth from different model. |
| Baseline 3 | TTGA with smoothness only (no photometric) | Ablation to isolate geometric consistency. |
| Ablation | TTGA w/o smoothness | Removes geometric regularization. |

### Why this setup validates the claim
This experimental design creates a falsifiable test of the central claim that test-time geometry adaptation improves novel view synthesis over feed-forward models. RealEstate10K provides diverse scenes with large viewpoint changes and ground truth novel views, making it ideal to measure depth warping quality. Using PSNR as the primary metric captures pixel-level fidelity, directly reflecting depth accuracy. Comparing against MetaView (same backbone) isolates the effect of our adaptation; comparing against Depth Anything v3 tests generality across different pre-trained features. The additional baseline (TTGA with smoothness only) demonstrates that the geometric warp loss is critical—without it, adaptation reduces to simple smoothing. The ablation without geometric smoothness identifies whether the regularization is critical. To further quantify assumption violations, we conduct a failure-case analysis by partitioning the test set into scenes with specularities (e.g., mirrors, windows) and scenes with significant disocclusions (e.g., foreground objects with large motion relative to background). We report PSNR separately on these subsets to measure the impact of our masking strategy.

### Expected outcome and causal chain

**vs. MetaView (no adaptation)** — On a case where the target scene has large low-texture regions (e.g., a blank wall), MetaView’s pre-trained depth is often inaccurate because its training set may have different depth statistics, leading to blurry or incorrect warps. Our method instead leverages self-supervised photometric consistency from small-baseline warps, which forces the depth to align with the observed image edges; this refines the depth locally. Consequently, we expect a noticeable PSNR gain (>1 dB) on such problematic regions, whereas on well-textured scenes our gains will be smaller (<0.5 dB).

**vs. Depth Anything v3 (no adaptation)** — On a case with out-of-distribution appearance (e.g., indoor scene with unusual furniture), DA3’s pre-trained depth may still be plausible but not perfectly scale-consistent for warping. Our method adapts to the exact scene geometry using the same photometric loss, correcting scale and local structures. We expect our method to outperform DA3 across the board, with a larger margin (>2 dB) on scenes far from the DA3 training distribution, and a smaller margin (<1 dB) on typical scenes.

**vs. TTGA with smoothness only** — On a case with strong texture where geometric warp is informative, the smoothness-only baseline cannot correct depth errors because it lacks the photometric cue. We expect >3 dB gap in favor of full TTGA, confirming that geometric consistency is essential.

### What would falsify this idea
If our PSNR improvement over MetaView is uniform across all scene types rather than concentrated on low-texture or out-of-distribution cases, then the photometric loss is not specifically correcting depth errors but merely overfitting to random pose perturbations. Alternatively, if the ablation without smoothness performs identically to the full model, then geometric regularization is unnecessary, contradicting our rationale. Furthermore, if the PSNR on specular/disocclusion subsets does not improve relative to MetaView (or if the mask does not help), then our masking strategy fails to address the assumption violations.

## References

1. MetaView: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors
2. Depth Anything 3: Recovering the Visual Space from Any Views
