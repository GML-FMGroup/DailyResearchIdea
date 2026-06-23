# SelfMV: Self-Supervised 3D Scene Representation for Trajectory-Conditioned Video Prediction

## Motivation

DriveDreamer-2 relies on an HDMap generator trained with expensive 3D annotations, creating a scalability bottleneck for world models. While self-supervised depth methods like ManyDepth show 3D can be inferred from video alone, they are not integrated into generative world models. The root cause is that world models treat structure as a separately supervised module, preventing learning from the vast availability of unlabeled driving video. We need a method that jointly learns a 3D scene representation and trajectory-conditioned future prediction without any 3D labels.

## Key Insight

Temporal photometric consistency from a moving camera enforces a multi-view geometry constraint that, when combined with a differentiable renderer, allows a latent 3D scene representation to be learned self-supervisably and directly used for trajectory-conditioned video generation.

## Method

### (A) What it is
**SelfMV** is a self-supervised framework that learns a latent 3D scene representation from unlabeled driving videos by enforcing photometric consistency across frames, then conditions a latent video diffusion model on a target trajectory for future frame prediction. Input: a sequence of posed video frames, a target trajectory (future camera poses). Output: a video of future frames from the trajectory.

### (B) How it works
```python
# Training phase for 3D representation
for each batch of video clips with known ego-motion:
    # Encode frames into latent 3D volume representation
    V = 3D_Volume_Encoder(clip_frames)  # outputs voxel grid with features per cell
    for each frame i in clip:
        # Render depth and appearance from volume using differentiable ray marching
        depth_i, rgb_i = NeuralRenderer(V, cam_pose_i)
        # Warp frame i to frame j using depth and relative pose
        warped_j = backward_warp(rgb_j, depth_i, relative_pose_ij)
        # Photometric loss between warped and actual frame
        L_photo = L1(warpped_j, rgb_i) + SSIM_loss(warpped_j, rgb_i)
        # Optional: smoothness loss on depth
    self_supervised_loss += L_photo

# After 3D volume is learned, train trajectory-conditioned diffusion
# Latent video diffusion model (e.g., based on Stable Video Diffusion)
# Condition on V and target trajectory T
# Training objective: epsilon_prediction loss conditioned on (V, T)
# Inference: sample future latent frames, decode with NeuralRenderer conditioned on poses from T
```

### (C) Why this design
We chose a voxel-based 3D volume over a NeRF-style MLP because the voxel grid can be updated incrementally from multiple frames and naturally handles large-scale driving scenes, accepting the cost of fixed spatial resolution. The photometric loss with SSIM was selected over direct L1 alone because SSIM better preserves structural details (edges, textures) which are critical for downstream trajectory planning, at the cost of slightly higher computation. We use ray marching rather than volume rendering with learned opacity to avoid dependence on density prediction, which often requires fine-tuning; this trades some flexibility for numerical stability. Conditioning the diffusion model on the 3D volume (rather than on a separate depth map) ensures that the video generator can access full 3D structure, which prevents depth inconsistencies across frames. Compared to prior work like DriveDreamer-2 that uses a separate HDMap generator, our joint representation eliminates the annotation bottleneck. The design is not a simple combination of ManyDepth and Stable Video Diffusion — the 3D volume unifies self-supervised depth cues and generative conditioning, which prior work treats as independent pipelines.

### (D) Why it measures what we claim
- The photometric loss <computational quantity> measures **multi-view consistency** <motivation concept> because the assumption that the scene is static and Lambertian makes the warped frame match the target frame; this assumption fails when objects are dynamic (e.g., moving cars), in which case the loss reflects the sum of photometric error and motion error. To mitigate, we apply an automasking that ignores regions with high flow residuals, ensuring the loss still measures static scene consistency. The 3D volume encoding <computational quantity> measures **scene geometry** because the assumption that a single voxel grid can represent all observed views via differentiable rendering; this assumption fails when the scene is severely occluded, in which case the volume may hallucinate plausible features. The trajectory-conditioned diffusion loss <computational quantity> measures **predictive fidelity** because the assumption that future frames are determined by scene geometry and trajectory; this assumption fails when the future contains unseen objects or weather changes, in which case the loss only captures the distribution of training data. By tying each computational component to a named assumption and its failure mode, we ensure the method does not rely on unstated proxies.

## Contribution

(1) A self-supervised framework that learns a voxel-based 3D scene representation from unlabeled driving videos using temporal photometric consistency and ego-motion, without any 3D annotations. (2) A trajectory-conditioned latent video diffusion model that conditions future frame generation on the learned 3D representation, enabling annotation-free video prediction for world models. (3) A demonstration that the learned 3D representation improves video consistency over 2D-based conditioning as measured by FVD and downstream perception tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | nuScenes | Large-scale, diverse driving scenes with poses. |
| Primary metric | FVD | Measures video distribution similarity, task-agnostic. |
| Baseline 1 | DriveDreamer-2 | State-of-the-art driving video generation. |
| Baseline 2 | Stable Video Diffusion | Strong general video diffusion model. |
| Ablation | SelfMV w/o 3D volume | Uses depth map conditioning instead. |

### Why this setup validates the claim

This combination forms a falsifiable test of our central claim: that a self-supervised 3D volume representation enables better future frame prediction than methods lacking explicit geometry or relying on external annotations. nuScenes provides realistic driving scenarios with varied geometry and motion. FVD captures both per-frame quality and temporal consistency, directly reflecting predictive fidelity. Comparing to DriveDreamer-2 tests the advantage of removing the HD-map annotation bottleneck. Comparing to Stable Video Diffusion tests the benefit of explicit 3D structure over pure latent diffusion. The ablation (removing the volume) isolates the contribution of the 3D representation. If our method outperforms baselines only on geometry-dependent cases (e.g., consistent depth across turns), the claim is supported.

### Expected outcome and causal chain

**vs. DriveDreamer-2** — On a case where the road layout changes (e.g., a sudden curve) and the HD map is outdated or missing, DriveDreamer-2 produces unrealistically straight roads because it relies on map-based priors that lack scene-specific geometry. Our method instead dynamically reconstructs the 3D scene from video frames, capturing the actual curvature via the voxel volume, so we expect a noticeable FVD gap (e.g., 10% lower on curve-heavy subsets) but parity on straight highways.

**vs. Stable Video Diffusion** — On a case with a sharp turn requiring consistent depth across frames, Stable Video Diffusion produces temporal flickering or shape distortion because it lacks explicit 3D reasoning; it treats each frame independently in latent space. Our method enforces multi-view consistency through the volume renderer, so we expect a clear improvement in FVD (e.g., 15% better) on sequences with complex ego-motion, while on static scenes performance is similar.

### What would falsify this idea

If the FVD improvement over baselines is uniform across all scene types (e.g., static scenes and dynamic turns alike) rather than concentrated on geometry-dependent subsets, then the central claim that the 3D volume representation is the key driver is falsified.

## References

1. DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation
2. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
3. VideoPoet: A Large Language Model for Zero-Shot Video Generation
4. SparseFusion: Distilling View-Conditioned Diffusion for 3D Reconstruction
5. Latent Video Diffusion Models for High-Fidelity Long Video Generation
