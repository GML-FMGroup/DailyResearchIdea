# Epipolar Confidence-Guided Consistency for Occlusion-Robust Novel View Synthesis

## Motivation

MVTrack4Gen uses attention-based point tracking as geometric supervision, but attention correspondences become unreliable under occlusion or fast motion, causing geometric artifacts. The root cause is that attention lacks explicit geometric priors and confidence estimation, making it invariant to occlusion. We need a supervision that incorporates known epipolar geometry and learned confidence to handle occlusion robustly.

## Key Insight

Epipolar constraints provide an invariant geometric relationship between views that holds regardless of scene content, and learned confidence from feature matching consistency allows the method to downweight occluded regions, ensuring that supervision remains valid where geometry is observable.

## Method

### Epipolar Confidence-Guided Consistency (ECGC)

(A) **What it is**: The method, Epipolar Confidence-Guided Consistency (ECGC), produces a weighted epipolar consistency loss that supervises a diffusion model for novel view synthesis. Inputs are multi-view video frames, camera poses, and estimated depth maps. Output is a scalar loss that penalizes violations of epipolar geometry, weighted by a learned confidence score that indicates likelihood of visibility.

(B) **How it works**:
```python
def ecgc_loss(views, poses, depths, features, confidence_net):
    N = len(views)
    total_loss = 0
    for i in range(N):
        for j in range(i+1, N):
            # 1. Unproject each pixel in view i to 3D using depth and pose
            points_3d = unproject_all(views[i], depths[i], poses[i])
            # 2. Project 3D points into view j
            proj_points = project_all(points_3d, poses[j])
            # 3. Compute feature similarity (cosine) between source pixel features and projected pixel features
            sims = cosine_similarity(features[i], features[j][proj_points])
            # 4. Compute depth consistency: ratio of depths at projected point
            depth_consistency = depths[i] / depths[j][proj_points]
            # 5. Compute epipolar angular error: angle between epipolar line and projected direction
            epi_angular = angular_error(proj_points, fundamental_matrix(poses[i], poses[j]), views[i].shape)
            # 6. Concatenate cues: [sims, depth_consistency, epi_angular]; MLP with 2 hidden layers (256 units, GeLU), output sigmoid
            confs = confidence_net(torch.cat([sims, depth_consistency, epi_angular], dim=-1))
            # 7. Compute epipolar error: distance of projected point to epipolar line (using fundamental matrix)
            epi_errors = epipolar_distance(proj_points, fundamental_matrix(poses[i], poses[j]), views[i].shape)
            # 8. Weighted loss with regularization (lambda_reg=0.1)
            total_loss += (confs * epi_errors).mean() + 0.1 * (1 - confs).mean()
    return total_loss
```
Training hyperparameters: learning rate=1e-4, Adam optimizer, batch size=8, total training steps=200k, GPU memory~24GB, training time~5 days on single RTX 3090.

(C) **Why this design**: We chose a learned confidence net (2-layer MLP, hidden=256, GeLU) over a fixed threshold because occlusion patterns vary; learned confidence adapts per-pixel. The confidence net assumes feature similarity is a reliable proxy for visibility. This assumption holds in textured regions but fails under repetitive textures or motion blur. To mitigate, we extend inputs to include depth consistency and epipolar angular error, providing additional cues. We use epipolar error rather than reprojection error because epipolar error is less sensitive to depth inaccuracies. We integrate this loss as an auxiliary supervision alongside the denoising loss because consistency provides structural guidance while denoising ensures high-frequency detail. The regularization term (0.1 * (1-confs).mean()) prevents predicting all zeros.

(D) **Why it measures what we claim**: The epipolar error **epi_errors** measures geometric consistency because it enforces that corresponding points lie on epipolar lines; this assumes depth estimates are accurate enough to project into the correct epipolar region. The confidence **confs** measures occlusion likelihood because it is trained via the loss to be high when feature similarity, depth consistency, and epipolar angular error are all low (indicating visible match). The assumption that feature matching consistency implies visibility fails under F (repetitive textures), causing overconfidence in occluded regions. The product **confs * epi_errors** ensures the loss penalizes geometric inconsistency only for likely visible matches.

## Contribution

(1) A novel consistency loss for novel view synthesis that combines epipolar geometry with a learned confidence weighting derived from feature matching consistency, enabling occlusion-robust geometric supervision. (2) An empirical design principle demonstrating that explicit geometric constraints with learned occlusion handling outperform attention-based supervision under occlusion and fast motion (validated on multi-view video datasets). (3) (Optional) A confidence-labeled dataset for training occlusion-aware supervision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | DTU dataset (multi-view, depth maps) | Tests geometric consistency across viewpoints. |
| Primary metric | PSNR | Measures image quality after synthesis. |
| Baseline 1 | Standard diffusion (no consistency) | Isolates effect of ECGC loss. |
| Baseline 2 | Fixed-threshold epipolar loss (threshold=0.8) | Tests adaptive confidence vs. fixed. |
| Baseline 3 | RANSAC-based robust epipolar loss (inlier threshold=1px) | Compares to classical robust estimator. |
| Ablation | ECGC w/o confidence net (uniform weight) | Isolates learned confidence contribution. |

### Why this setup validates the claim
This setup validates that ECGC's learned confidence weighting improves robustness to occlusion compared to fixed thresholds and RANSAC, and that epipolar error is less sensitive to depth noise than reprojection error. The standard diffusion baseline tests if any consistency loss helps; the fixed-threshold baseline tests if adaptive weighting is beneficial; the RANSAC baseline tests if learned adaptation outperforms classical robust estimation. The ablation (uniform weight) isolates the confidence net's effect. PSNR is chosen because it directly measures pixel-level accuracy. Additionally, we compute Spearman correlation between confidence scores and ground-truth occlusion masks on the validation set to verify that confidence reflects occlusion. Together, these components form a falsifiable test: if ECGC outperforms baselines primarily on difficult cases (high motion, textureless regions), the claim is supported; if not, the idea fails.

### Expected outcome and causal chain

**vs. Standard diffusion (no consistency)** — On a case with large camera motion causing severe occlusion/disocclusion, the baseline produces blurry or ghosting artifacts. Our method uses epipolar consistency, so we expect a noticeable PSNR gap (e.g., +3 dB) on high-motion frames, but parity on static frames.

**vs. Fixed-threshold epipolar loss** — On a case with textureless regions (e.g., white wall) where fixed similarity threshold mislabels occluded pixels as visible, the baseline incorrectly penalizes those areas. Our method learns to assign low confidence to ambiguous matches, so we expect better performance (e.g., higher PSNR by 1.5 dB) on such regions, but similar on richly textured areas.

**vs. RANSAC-based robust epipolar loss** — On a case with repetitive textures (e.g., brick wall) where RANSAC struggles to distinguish inliers/outliers, the baseline produces artifacts due to incorrect weighting. Our method uses learned confidence with multi-cue inputs, so we expect a PSNR gain of 2 dB on such scenes, but similar on scenes with simple geometry.

### What would falsify this idea
If ECGC's PSNR gain over fixed-threshold and RANSAC is uniform across all pixel subsets rather than concentrated on occluded or textureless regions, the central claim (learned confidence adapts to occlusion) is falsified. Additionally, if Spearman correlation between confidence and ground-truth occlusion is below 0.3, the confidence net does not reliably measure occlusion.

## References

1. MVTrack4Gen: Multi-View Point Tracking as Geometric Supervision for 4D Video Generation
2. GS-DiT: Advancing Video Generation with Dynamic 3D Gaussian Fields through Efficient Dense 3D Point Tracking
