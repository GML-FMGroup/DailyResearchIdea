# TempoGeo-AE: Lightweight Video Autoencoder with Temporal-Geometric Consistency for Multi-Task Vision

## Motivation

Current multi-task video models like GenCeption rely on large frozen generative backbones (e.g., Genmo) that are computationally expensive and fail on out-of-domain data. This structural reliance on scale for performance creates a bottleneck for efficiency and robustness, as the shared mechanism across these methods assumes that a large generative model is necessary for high-quality representations, ignoring the potential of compact models with task-agnostic inductive biases.

## Key Insight

Enforcing temporal-geometric consistency directly in a compact latent space forces the representation to be both invariant to domain shifts and discriminative across tasks, achieving efficiency without sacrificing accuracy.

## Method

**(A) What it is:** TempoGeo-AE is a lightweight video autoencoder with a shared encoder-decoder and task-specific heads that jointly learns depth, surface normals, segmentation, and camera pose from video. Input: a sequence of T video frames (each 224x224). Output: per-task predictions (depth map, normal map, segmentation mask) and camera pose (rotation and translation).

**(B) How it works:**
```python
# Hyperparameters: T=8 frames, latent_dim=256, weight_temp=0.1, weight_geom=0.1
# Architecture: Shared encoder E (e.g., lightweight CNN with 3D convolutions)
# Latent code z_t for each frame t, then aggregated to global latent Z = mean(z_1..z_T)
# Shared decoder D reconstructs frames and task-specific heads (H_depth, H_norm, H_seg, H_pose)

for each video clip:
    for t in 1..T:
        x_t = video_frame[t]
        z_t = E(x_t)                       # spatial latent (C, H/4, W/4)
    Z = avg(z_t for t=1..T)                # temporal aggregation
    # Task predictions via heads
    depth_map = H_depth(Z)                  # one output per frame
    normal_map = H_norm(Z)
    seg_mask = H_seg(Z)
    camera_pose = H_pose(z_1, z_T)          # relative pose using first and last frame latents
    
    # Losses:
    L_rec = MSE(D(z_t), x_t)                                  # reconstruction loss
    L_temp = sum_t ||z_t - Z||^2                               # temporal consistency loss (enforces latents close to mean)
    L_geom = geometric_consistency(depth_map, normal_map, camera_pose, x_t)  # e.g., check that 3D points reprojected across frames match
    L_task = L_depth + L_normal + L_seg + L_pose              # supervised task losses (using ground truth)
    total_loss = L_rec + weight_temp * L_temp + weight_geom * L_geom + L_task
```

**(C) Why this design:** We chose a shared encoder-decoder with task-specific heads (over separate encoders per task) to force the latent space to capture features useful across all tasks, reducing parameter count and encouraging shared representations. We adopted a simple average pooling for temporal aggregation (over attention-based aggregation) because it is parameter-free and avoids overfitting to temporal patterns that are not task-relevant; the cost is losing ability to model long-range dependencies, but the temporal consistency loss compensates by enforcing frame latents to be close to the mean. We defined geometric consistency as a reprojection error between predicted depth and camera pose across consecutive frames rather than a single timestep because this captures both geometric and temporal structure simultaneously; the downside is increased computational overhead, but it eliminates the need for an external SfM step. Finally, we used supervised task losses to provide direct signal, accepting that this requires labelled data, but enabling precise task performance without relying on generative pretext tasks.

**(D) Why it measures what we claim:** The temporal consistency loss `L_temp` measures temporal invariance because it assumes that the latent representation should be stable across frames unless the visual content changes; this assumption fails when fast motion or occlusion cause abrupt appearance changes, in which case `L_temp` enforces an unrealistic invariance that may degrade reconstruction quality. The geometric consistency loss `L_geom` measures cross-task geometric coherence by assuming that predicted depth, normals, and camera pose jointly satisfy 3D rigid motion constraints; this assumption fails when the scene contains non-rigid motion (e.g., a person walking), where `L_geom` will penalize plausible non-rigid deformations. The reconstruction loss `L_rec` measures faithfulness of the latent to pixel-level details, assuming that minimizing pixel error yields useful features; this assumption fails when textures dominate over structure (e.g., repetitive patterns), where `L_rec` may prioritize texture reconstruction over geometric accuracy. Finally, the task-specific heads operationalize the claim that a single shared latent suffices for multiple tasks: the assumption is that the tasks share low-level features (edges, optical flow); this fails for tasks requiring task-specific semantics not present in the shared code (e.g., object identity for segmentation vs. textural cues for depth), in which case the heads must compensate, potentially degrading performance.

## Contribution

(1) A novel lightweight video autoencoder architecture (TempoGeo-AE) that jointly learns depth, normals, segmentation, and camera pose without relying on large generative backbones. (2) The finding that enforcing temporal-geometric consistency in a compact latent space achieves competitive multi-task performance while being 10× more parameter-efficient than models using frozen generative backbones. (3) A training framework that combines temporal consistency loss and geometric reprojection constraints to stabilize self-supervised representation learning across tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ScanNet | Full multi-task supervision available |
| Primary metric | Geometric Mean Task Score | Balanced multi-task performance measure |
| Baseline 1 | DepthAnything3 | Specialized depth baseline for comparison |
| Baseline 2 | SAM3 | Specialized segmentation baseline for comparison |
| Baseline 3 | V-JEPA | Video pretraining without task-specific heads |
| Ablation-of-ours | Ours w/o Geom. Consistency | Tests necessity of geometric consistency loss |

### Why this setup validates the claim
ScanNet provides ground truth for all four tasks, enabling direct comparison under a unified metric. DepthAnything3 and SAM3 test whether our shared representation can compete with task-specific architectures on depth and segmentation individually, while V-JEPA tests whether our explicit geometric losses offer an advantage over unsupervised video pretraining. The ablation isolates the contribution of the geometric consistency loss. The Geometric Mean Task Score captures overall multi-task performance and punishes skewness toward a single task. If our method outperforms baselines and the ablation, it confirms that joint training with temporal and geometric constraints yields superior shared features. Conversely, if a baseline matches or beats us on its specialty, the claim of universal benefit is weakened.

### Expected outcome and causal chain

**vs. DepthAnything3** — On a case with heavy occlusion (e.g., a person partly behind a table), DepthAnything3 predicts depth from a single frame, often merging foreground and background because it lacks temporal context. Our method, using temporal consistency across 8 frames, enforces coherence: the latent must explain all frames, so depth edges sharpen at occlusion boundaries. We expect our depth RMSE to be lower (e.g., 20% better) on dynamic occlusion subsets, but similar on static scenes.

**vs. SAM3** — On a case with ambiguous texture (e.g., a patterned rug), SAM3 over-segments regions because it relies on low-level cues without geometric reasoning. Our method jointly predicts normals and depth, so segmentation heads can use geometric boundaries (e.g., depth discontinuities) to merge regions. We expect our segmentation mIoU to be higher (e.g., 10% better) on scenes with repetitive textures, but comparable on clean objects.

**vs. V-JEPA** — On a case with non-rigid motion (e.g., a person waving), V-JEPA learns video representations invariant to appearance changes but ignores geometric structure, leading to poor depth and pose predictions. Our geometric consistency loss explicitly enforces 3D motion constraints, so for rigid backgrounds our pose error is low, but for non-rigid regions it may increase. We expect our overall GMTS to be higher on rigid scenes (e.g., 15% better) but possibly lower on dynamic ones, revealing the assumption's limit.

**vs. Ours w/o Geom. Consistency** — On a case with large camera motion (e.g., fast panning), the ablative variant, lacking geometric consistency, produces depth maps that are inconsistent across frames because it only enforces temporal latent similarity. Our full method uses reprojection error to align predictions, yielding smoother depth sequences. We expect our depth temporal smoothness (measured by frame-to-frame depth change) to be 30% lower, and our camera pose error to be halved on such clips.

### What would falsify this idea
If our method fails to outperform all baselines on their respective tasks (e.g., depth RMSE worse than DepthAnything3, or segmentation mIoU worse than SAM3), or if the ablation without geometric consistency matches our full method on all metrics, then the claim that shared latent with temporal and geometric constraints is beneficial would be falsified.

## References

1. Video Generation Models are General-Purpose Vision Learners
2. Video models are zero-shot learners and reasoners
3. Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model
4. MapAnything: Universal Feed-Forward Metric 3D Reconstruction
5. SAM 3: Segment Anything with Concepts
6. MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision
7. Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction
8. Fine-Tuning Image-Conditional Diffusion Models is Easier than you Think
