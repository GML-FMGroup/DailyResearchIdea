# CrossView4D: Direct 4D Human Reconstruction from Sparse Views via Cross-View Attention

## Motivation

Existing methods for 4D human reconstruction from sparse multi-view videos rely on explicit 2D keypoint detection and triangulation to initialize human models, but these steps fail under occlusion because keypoints become unreliable or invisible. For instance, StudioRecon (4D Human-Scene Reconstruction) assumes reliable multi-view keypoint correspondences, which degrades in highly dynamic scenes or with only two cameras. This structural reliance on keypoint quality is a bottleneck across current approaches.

## Key Insight

Cross-view attention over image features inherently establishes view-consistent 3D correspondences without explicit detection, because the attention mechanism aggregates information from visible views in a permutation-equivariant manner, making the representation robust to missing or occluded keypoints.

## Method

We propose CrossView4D, a feed-forward neural field that directly reconstructs a 4D human representation (3D Gaussians with temporal motion) from sparse multi-view images (2-4 views) and time.

**Load-bearing assumption (explicit):** Cross-view attention over image features inherently establishes view-consistent 3D correspondences robust to occlusions, because the attention mechanism aggregates information from visible views. To handle cases where attention weights could be ambiguous under extreme occlusions, we incorporate an explicit visibility mask from SMPL geometry to suppress attention weights from occluded views before aggregation.

**How it works (pseudocode)**
```python
# Input: Multi-view images {I_v} for v=1..V at time t; time index t
# Output: Set of 3D Gaussians {G_i} with (pos, cov, color, opacity) and per-Gaussian motion MLP parameters

# 1. Feature extraction
F_v = CNN(I_v)  # multi-scale feature maps, e.g., ResNet18, stride 4, output channels 256, ~11M params

# 2. Sample 3D points on SMPL mesh (coarse template)
P_init = sample_points(SMPL_pose, num_points=10000)

# 3. For each point p in P_init:
#    Compute visibility mask from SMPL geometry (ray-mesh intersection, binary)
vis_mask = compute_visibility(p, SMPL_mesh, cameras)  # shape (V,)

for each p:
    # Project to all views, get features
    feat = []
    for v in 1..V:
        u = project(p, K_v, Rt_v)  # camera params
        if inside_image(u) and vis_mask[v]:
            f = bilinear_interpolate(F_v, u)
        else:
            f = zeros
        feat.append(f)
    # Cross-view attention (permutation-equivariant)
    # Use transformer encoder with L=4 layers, d_model=256, 4 heads
    # Also output attention weights for analysis
    agg, weights = CrossViewTransformer(feat, return_weights=True)  # output: aggregated feature vector (256-dim) and attention weights matrix (V x V)
    # Decode to Gaussian parameters (position offset, cov, color, opacity)
    offset, cov, color, opacity = MLP(agg, p_initial)  # 3-layer MLP, hidden=256, GeLU activation, ~0.5M params
    # Store Gaussian with position = p + offset
    G_i = (p+offset, cov, color, opacity)

# 4. Temporal motion (per Gaussian)
for each G_i:
    motion_offset = MotionMLP(G_i.pos, t)  # small MLP, 2 layers, hidden=128, predicts delta position (3-dim)
    G_i.pos += motion_offset

# 5. Differentiable rasterization (3DGS renderer)
rendered = rasterize(Gaussians, camera params)
loss = L1(rendered, target) + SSIM_loss (λ_SSIM=0.2)

# Hyperparameters: attention layers=4, hidden=256, MLP width=256, num_gaussians=10000, lr=1e-3, total params ~14M
# Training: 100K steps, batch size 4 (2 views each), resolution 512x512, ~3 days on A100 GPU
```

(C) **Why this design**: We chose cross-view attention over view pooling (e.g., max-pooling) because attention dynamically weights each view based on feature similarity, enabling the network to ignore occluded views; the trade-off is O(V²) complexity but V is small (2-4). We conditioned point sampling on a parametric human mesh (SMPL) rather than a dense voxel grid, because it focuses capacity on regions of interest and reduces memory; the cost is dependence on a coarse pose prior, which may fail for loose clothing. We adopted Gaussian splatting as output over NeRF because it enables real-time rendering and direct temporal deformation via MLPs; the downside is limited geometric detail compared to implicit surfaces. Unlike MV-DUSt3R+ which predicts pointmaps for general scenes, our method leverages a human-specific mesh prior and outputs Gaussian parameters, enabling efficient and accurate dynamic human reconstruction with temporal modeling.

(D) **Why it measures what we claim**: The cross-view attention weights w_{v,u} measure the confidence of correspondence between views v and u for a given 3D point, because attention computes similarity of projected features; this equivalence assumes that consistent appearance across views implies a shared 3D point, which fails when specularities or lighting changes cause feature mismatch, in which case attention becomes uncertain and the aggregated feature averages inconsistent appearances. The per-point position offset predicted by the MLP measures the residual between the SMPL template and true surface, because the template provides a rough initialization; this assumes the true geometry lies near the template, which fails for non-rigid deformations (e.g., clothing), in which case the offset attempts to model the deviation but may be inaccurate. The temporal motion MLP's output magnitude measures per-Gaussian movement across frames, because it is trained to minimize photometric error; this assumption holds under smooth motion, which fails for abrupt motions, causing the motion prediction to blur across frames. To validate occlusion handling, we record attention weights and correlate them with ground-truth per-point visibility (derived from SMPL mesh), expecting high attention to visible views and low to occluded views.

## Contribution

(1) A feed-forward architecture that directly reconstructs 3D Gaussians for humans from sparse multi-view images using cross-view attention, eliminating the need for keypoint detection and triangulation. (2) The finding that cross-view attention can implicitly learn robust correspondences under occlusion, outperforming methods that rely on explicit keypoint matching. (3) Introduction of a temporal extension with per-Gaussian motion MLPs for 4D reconstruction.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | ZJU-MoCap (synthetic) | Covers varied dynamic poses and sparse views |
| Primary metric | PSNR | Directly measures reconstruction fidelity |
| Baseline 1 | NeuralBody | Explicit SMPL-based 4D reconstruction |
| Baseline 2 | MultiNeRF | Generalizable NeRF from sparse views |
| Baseline 3 | Vanilla 3DGS (per-frame) | No temporal modeling |
| Ablation of ours | w/o cross-view attention | Mean-pooling aggregation instead |
| Additional controlled test | Synthetic occlusion (mask one view) | Isolates attention's robustness to missing views |
| Additional analysis metric | Attention weight vs. visibility correlation | Validates occlusion handling mechanism |
| Additional analysis metric | Per-Gaussian offset magnitude | Validates template deviation capability |

### Why this setup validates the claim

The ZJU-MoCap dataset provides multi-view video sequences with ground truth geometry, enabling controlled evaluation of dynamic human reconstruction from sparse inputs (2-4 views). Comparing against NeuralBody tests our ability to handle deviations from SMPL template; MultiNeRF tests generalization without human priors; Vanilla 3DGS tests temporal coherence. The ablation of cross-view attention isolates its effect on handling occlusions. PSNR is chosen because it directly measures per-frame pixel accuracy, essential for novel view synthesis in dynamic scenes, and is sensitive to both geometric and appearance errors. Additionally, we perform a controlled experiment with synthetic occlusions (masking one view entirely) to directly test attention's robustness: for each frame, we randomly mask one of the input views and measure PSNR drop relative to the unmasked case. We also record attention weights per 3D point and compute their correlation with ground-truth visibility (derived from SMPL mesh) to verify that the attention mechanism correctly downweights occluded views. Finally, we measure per-Gaussian offset magnitude (L2 norm of predicted offset) on frames with loose clothing to confirm that our method can deviate from the SMPL template. This combination creates a falsifiable test: if our method succeeds, we expect systematic gains where prior methods fail due to their inherent assumptions, and our internal analyses should confirm the intended mechanisms.

### Expected outcome and causal chain

**vs. NeuralBody** — On a frame with loose clothing (e.g., a dress fluttering), NeuralBody assumes the surface lies on SMPL, producing over-smoothed geometry and blurry renders because it cannot model large offsets. Our method instead predicts per-Gaussian residuals and temporal motion, explicitly allowing deviations from the template, so we expect a noticeable PSNR gain (e.g., +2 dB) on such challenging regions, but parity on tight clothing where SMPL fits well. Furthermore, the per-Gaussian offset magnitude should be significantly larger on loose clothing regions (e.g., >5 cm) than on tight regions (<1 cm), confirming template deviation.

**vs. MultiNeRF** — On a novel view with occluded limbs (e.g., arm behind back), MultiNeRF averages across views without attention, leading to ghosting artefacts because it assumes consistent features. Our cross-view attention dynamically weights visible views and ignores occluded ones, so we expect higher PSNR on these complex poses, possibly +1.5 dB overall, with larger gaps on highly occluded frames. In the synthetic occlusion test, our method should degrade gracefully (PSNR drop <0.5 dB) while MultiNeRF degrades severely (drop >2 dB), because our visibility mask and attention suppress the masked view.

**vs. Vanilla 3DGS** — On a sequence with fast limb motion (e.g., punching), per-frame 3DGS reconstructs each frame independently, causing severe flickering and temporal inconsistency when rendered from novel views, as no motion prior exists. Our motion MLP enforces smooth temporal deformation, so we expect higher and more stable PSNR across frames (e.g., 0.5 dB higher mean and 30% lower variance), especially on dynamic regions.

**Ablation (w/o cross-view attention)** — Using mean-pooling instead of cross-view attention should reduce performance on occluded frames (PSNR drop ~1 dB) and show lower correlation between features and ground-truth visibility, confirming the attention mechanism's role.

### What would falsify this idea

If our method shows no significant PSNR improvement over NeuralBody on loose clothing (≤0.5 dB), or if the ablation w/o cross-view attention matches the full model (ΔPSNR < 0.2 dB), or if attention weights do not correlate with ground-truth visibility (Pearson correlation < 0.3), then the central claims about template deviation handling and attention-based occlusion handling are invalid.

## References

1. 4D Human-Scene Reconstruction from Low-Overlap Captures
2. $\pi^3$: Permutation-Equivariant Visual Geometry Learning
3. Echoes of the Coliseum: Towards 3D Live streaming of Sports Events
4. FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction
5. Kineo: Calibration-Free Metric Motion Capture From Sparse RGB Cameras
6. MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds
7. 3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos
8. SplattingAvatar: Realistic Real-Time Human Avatars With Mesh-Embedded Gaussian Splatting
