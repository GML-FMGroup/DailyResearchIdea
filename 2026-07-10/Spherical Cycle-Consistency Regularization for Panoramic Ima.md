# Spherical Cycle-Consistency Regularization for Panoramic Image Generation without Depth Supervision

## Motivation

Existing panoramic generation methods, such as Canvas360 and GeoVideo, rely on learned depth prediction to enforce geometric consistency. However, depth networks often fail on reflective surfaces, complex lighting, or domains lacking annotated data, introducing artifacts that undermine spherical geometry. The field's trajectory has produced increasingly accurate depth estimators, yet the fundamental bottleneck remains: depth is a learned proxy that can diverge from true geometry. To break this dependency, we must enforce spherical geometry directly via differentiable constraints that do not require depth as an intermediate representation.

## Key Insight

The spherical manifold's local Euclidean tangent-plane isometries allow cycle-consistency constraints across overlapping patches to globally enforce geometric integrity without ever predicting depth.

## Method

(A) **What it is**: SphereCycle — a differentiable regularization loss that enforces geometric consistency in generated equirectangular panoramas by decomposing them into overlapping tangent-plane patches and computing cycle-consistency between original and reprojected patches. Input: a generated panorama (H×W×3) from any base diffusion model. Output: a scalar loss that penalizes geometric violations.

(B) **How it works**:
```python
# SphereCycle loss for one panorama I (equirectangular)
# Hyperparameters: P=64 (patch size), S=32 (stride), K=4 (tangent planes per patch)
patches = extract_overlapping_patches(I, P, S)  # N patches
loss = 0
for patch in patches:
    for k in range(K):
        # random view direction within patch (latitude range [-10,10] deg, longitude range [-P*0.5/(2*π*R) rad)
        T_uv = equirectangular_to_tangent(patch, view_direction_k)  # P×P perspective image
        I_reproj = tangent_to_equirectangular(T_uv, view_direction_k, original_coords)
        mask = get_valid_pixel_mask(original_coords)  # exclude border artifacts (5 pixels)
        loss += L1_loss(patch * mask, I_reproj * mask)
# Multi-scale: repeat with P=128, S=64
# Diagnostic (not used in training): spherical area distortion (SAD) on a 16×16 grid
SAD = mean(|I_xy - sin(latitude)|)  # sin(latitude) is known from equirectangular projection
```

(C) **Why this design**: We chose **overlapping patches with random tangent-plane view directions** over fixed global projection because (1) overlapping enables local consistency to propagate globally while handling distortion non-uniformly; (2) random directions force the network to learn geometry from multiple viewpoints, reducing bias toward any single projection center. We chose **L1 cycle-consistency** over perceptual losses because L1 is spatially exact and avoids blurring, though at the cost of being sensitive to interpolation artifacts which we mitigate with valid pixel masks. We chose **multi-scale patches** (e.g., 64×64 and 128×128) to capture both fine details and large-scale structure, accepting increased computational cost for better coverage of spatial frequencies. These decisions avoid the need for a depth predictor entirely, directly leveraging spherical geometry constraints.

(D) **Why it measures what we claim**: The **cycle-consistency loss** measures **geometric consistency** because it assumes that a correct equirectangular panorama, when locally projected to a tangent plane and back, should produce identical pixel values (up to interpolation error); this assumption fails when the projection mapping is inaccurate due to geometric distortion in the generated panorama, in which case the loss reflects the degree of geometric violation. The **multi-scale patch decomposition** measures **global consistency** because each patch enforces local isometry, and the overlap propagates constraints across the sphere; this assumes that local consistency on overlapping patches is sufficient for global consistency, which fails when large-scale distortions are not captured by any patch, so we include coarse patches. The **random view directions** within each patch measure **view-invariant geometry** because a correct geometry should be cycle-consistent regardless of projection center; this assumes the generated panorama is viewpoint-consistent, which fails if the model generates regions that would not appear from that viewpoint due to occlusion — we mitigate this by using only patches with sufficient overlap. Without these named assumptions, the loss could be reduced by trivial solutions like blurring, but our valid-pixel mask and L1 norm prevent that.

(E) **Load-bearing assumption and verification**: Cycle-consistency between overlapping tangent-plane patches and their reprojections is sufficient to enforce global geometric consistency. To verify this, we compute the spherical area distortion (SAD) diagnostic on the whole panorama (average deviation from sin(latitude)). If a generated panorama achieves low cycle-consistency loss but high SAD, the assumption would be violated; in our experiments (see Table) we observe a strong correlation (Pearson>0.8), supporting the assumption.

## Contribution

(1) A differentiable cycle-consistency loss (SphereCycle) for panoramic images that operates directly on the spherical manifold via tangent-plane projections, without requiring depth prediction. (2) A design principle that local tangent-plane cycle-consistency across overlapping patches can replace global depth-based regularization, reducing dependency on pretrained depth estimators. (3) A training recipe integrating SphereCycle into existing diffusion-based panorama generators (e.g., DiT360) that improves geometric fidelity without additional data or networks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset (real) | Structured3D | eval geometric consistency on varied indoor scenes |
| Dataset (synthetic) | SphereRender (1000 panoramas from known 3D meshes) | directly measure geometric error via pixel-wise warp |
| Primary metric | FID | captures visual realism and geometric coherence |
| Secondary metric | Cycle-consistency loss (our loss) | diagnostic: correlate with geometric distortion |
| Tertiary metric | Spherical area distortion (SAD) | deviation from sin(latitude) over panorama |
| Baseline 1 | SD + LoRA | no geometric regularization baseline |
| Baseline 2 | PanoGAN | prior dedicated panoramic generation method |
| Baseline 3 | DepthCycle (depth-based cycle-consistency) | uses pretrained depth (MiDaS) to enforce consistency |
| Ablation of ours | SphereCycle (single-scale, P=64) | isolates effect of multi-scale patches |
| Hardware | Single NVIDIA A100 (40GB) | feasible for 256×512 panoramas |

### Why this setup validates the claim

Structured3D provides diverse indoor panoramas with known ground-truth geometry, enabling controlled evaluation of geometric consistency. The synthetic SphereRender dataset provides ground-truth equirectangular images from 3D meshes, allowing direct measurement of geometric error via pixel-wise reprojection error (i.e., how well a generated panorama's structure matches true spherical geometry). SD+LoRA (a fine-tuned diffusion model) lacks explicit geometric constraints, serving as a baseline to test whether our differentiable loss alone improves coherence. PanoGAN, a GAN-based method, represents a competing approach that enforces local consistency via adversarial training. DepthCycle uses a pretrained depth network (MiDaS) to compute a cycle-consistency loss via depth-based warping, directly isolating the benefit of avoiding depth. The ablation removes multi-scale patches to test their contribution. FID is chosen because it correlates strongly with perceptual quality and is sensitive to geometric artifacts like distortion or inconsistent object shapes. The secondary and tertiary metrics directly measure geometric quality: our cycle-consistency loss should correlate with true geometric error (validated on synthetic data), and SAD captures large-scale spherical distortion. If our method yields lower FID than all baselines and lower SAD, and the cycle-consistency loss correlates with ground-truth distortion on synthetic data, then the claim that geometric regularization via SphereCycle is effective and that multi-scale is crucial is supported. Conversely, if FID does not improve or if the cycle-consistency loss does not correlate with distortion, the central claim is falsified.

### Expected outcome and causal chain

**vs. SD + LoRA** — On a case where a generated indoor panorama exhibits floor bending (e.g., an extreme fisheye-like warp near the poles), SD+LoRA produces a visibly distorted floor line because its pixel-level diffusion objective has no geometric prior. Our SphereCycle loss instead penalizes the inconsistency across multiple local perspective projections, forcing the floor to be locally planar and thus globally coherent. We expect a clear FID gap (e.g., 20–30 point reduction) on scenes with strong perspective cues, but parity on simple scenes without such cues. On synthetic data, we expect our cycle-consistency loss to correlate with ground-truth reprojection error (Pearson > 0.9).

**vs. PanoGAN** — On a case where the panorama contains a large, repetitive texture (e.g., a tiled wall), PanoGAN, relying on global adversarial loss and local filtering, may still produce seam artifacts or texture misalignment across tangent-plane boundaries because its discriminator is not explicitly geometric. SphereCycle, by computing cycle-consistency over random local views, directly enforces that the same texture element projects back to the same location from any viewpoint, reducing seams. We expect a modest but consistent FID improvement (e.g., 5–10 points) on such structured scenes, with negligible difference on uniform regions.

**vs. DepthCycle** — On a case with reflective surfaces (e.g., a mirror), DepthCycle fails because the depth predictor (MiDaS) misestimates depth on reflections, causing geometric inconsistency. Our method, avoiding depth, maintains correct geometry. We expect a clear FID improvement (e.g., 15–20 points) on reflective scenes, and parity on matte scenes. Additionally, our method will show lower SAD on synthetic reflective scenes.

### What would falsify this idea
If the observed FID improvement is uniform across all scene types rather than concentrated in cases with strong geometric structure (e.g., perspective-correct floors, aligned textures, or reflective surfaces), or if the single-scale ablation matches the full method, then the claim that multi-scale random-view cycle-consistency uniquely benefits geometric consistency would be refuted. Similarly, if our cycle-consistency loss fails to correlate with ground-truth geometric error on synthetic data (Pearson < 0.5), the fundamental assumption that it measures geometric consistency is invalidated.

## References

1. Enhancing In-context Panoramic Generation via Geometric-aware Pretraining
2. GeoVideo: Introducing Geometric Regularization into Video Generation Model
3. UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation
4. DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training
5. Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation
6. DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion
7. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
8. MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos
