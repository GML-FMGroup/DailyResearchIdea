# GeoLatent: End-to-End Single-View 3D Reconstruction via Jointly Trained Latent Encoder with Depth Priors

## Motivation

Existing single-view 3D reconstruction methods such as FLAT and Lyra rely on pre-trained video diffusion models to generate latent codes, which are not optimized for geometric accuracy and introduce distributional biases. This dependence on an external frozen model prevents end-to-end learning of the latent space for 3D reconstruction, limiting generalization and geometric fidelity.

## Key Insight

Because depth estimation provides a strong, view-consistent geometric prior that is complementary to appearance, integrating depth-derived geometric constraints into the training of a latent encoder aligns the latent space with 3D structure, enabling end-to-end optimization without a separate diffusion model.

## Method

### GeoLatent Training Loop
```python
# GeoLatent Training Loop
Input: Image I, ground truth multi-view images {I_v}, camera poses {P_v} from dataset
Optional: Pre-trained monocular depth estimator D_est (MiDaS v3.1, frozen)
Initialize: Encoder E (ResNet-50 backbone + global avg pooling + linear to 512-dim latent), 
            Decoder Dec (6-layer transformer + Gaussian head for differentiable Gaussian splatting), 
            optimizer Adam (lr=1e-4, weight decay=1e-5)

for each batch (batch size 8, views per scene=6):
    # Compute depth and confidence map from I
    D, conf = D_est(I)  # confidence from MiDaS internal uncertainty
    # Encode image to latent
    z = E(I)  # 512-dim vector
    # Decode 3D representation (set of Gaussians: mean, scale, rotation, opacity)
    rep = Dec(z)  # initialized as isotropic Gaussians per pixel
    # Render images from multiple training views
    rendered_views = {differentiable_rasterize(rep, P_v) for all v}
    L_recon = sum_v ||I_v - rendered_views[v]||^2
    # Render depth maps from the same views
    rendered_depths = {differentiable_depth_rasterize(rep, P_v) for all v}
    # Warp the estimated depth D to each target view using inverse warping with bilinear sampling
    D_warped, conf_warped = {warp_with_confidence(D, conf, P_0, P_v) for all v}  # P_0 is reference view
    # Weighted depth consistency loss with confidence masking
    weighted_depth = conf_warped * (rendered_depths[v] - D_warped[v])**2
    L_depth = sum_v (sum(weighted_depth) / (sum(conf_warped) + ε))
    # Smoothness regularizer on rendered depths (total variation)
    L_smooth = sum_v TV(rendered_depths[v])  # TV = sum of absolute gradients
    L = L_recon + λ_depth * L_depth + λ_smooth * L_smooth
    # Hyperparameters: λ_depth = 0.1, λ_smooth = 0.01
    # Confidence threshold: regions with conf < 0.2 (10th percentile on calibration set of 512 examples) are masked (weight=0)
    optimize L w.r.t. E, Dec parameters
```

**Why this design:** We chose to train a dedicated encoder instead of using a frozen video diffusion model because the latter's latent distribution is tuned for 2D video consistency, not 3D geometry. This introduces a distributional gap that degrades reconstruction accuracy. To bridge this gap, we incorporate depth consistency as an auxiliary loss. A design trade-off is using a pre-trained, frozen depth estimator rather than jointly training a depth network: we accept that depth estimates may be imperfect, but gain training stability and avoid competing objectives. Another choice is to inject depth features via cross-attention in the encoder (as opposed to simple concatenation), which allows the model to selectively attend to depth cues while preserving RGB information, at the cost of increased parameters. Finally, we use differentiable Gaussian splatting as the decoder because it offers fast rendering and explicit geometry, similar to FLAT, but note that our decoder is trained from scratch without any diffusion priors. We explicitly assume the monocular depth estimator is sufficiently reliable for the target domain; to verify this, we compute a calibration set of 512 examples to measure the depth estimator's error distribution and set a confidence threshold (10th percentile) — samples below this threshold are masked out (weight=0) in L_depth.

**Why it measures what we claim:** The reconstruction loss L_recon directly measures appearance fidelity, but appearance alone does not guarantee geometric accuracy. The depth consistency loss L_depth measures the agreement between rendered depths and a monocular depth estimate. This operationalizes the concept of geometric prior: L_depth measures the alignment of latent-driven geometry with a view-consistent depth cue, assuming the monocular depth estimator is sufficiently reliable for the target domain. **Equivalence:** L_depth measures geometric accuracy under the assumption **A** that the monocular depth estimator is accurate in the target domain. **Failure mode F:** when the depth estimator is biased (e.g., on out-of-distribution scenes), L_depth misaligns geometry. To mitigate, we use confidence masking as described. The smoothness term L_smooth measures local depth continuity, capturing the prior that surfaces are piecewise smooth; this fails at depth discontinuities (object boundaries), where smoothness induces blurring. Together, these losses ensure that the latent z encodes geometry that is consistent with a strong external prior, effectively replacing the role of frozen diffusion latents with a jointly optimized latent space.

## Contribution

(1) Introduces GeoLatent, a framework that replaces the frozen video diffusion latents with a jointly trained latent encoder guided by depth priors, enabling end-to-end optimization for single-view 3D reconstruction. (2) Shows that depth consistency loss during training effectively aligns the latent space with geometric structure, achieving competitive reconstruction accuracy without dependence on pre-trained video models. (3) Proposes a depth-aware encoder architecture that fuses monocular depth features via cross-attention, enhancing geometric awareness.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ShapeNet (single-view) | Standard benchmark for geometry evaluation |
| Primary metric | Chamfer Distance | Measures geometric accuracy directly |
| Baseline 1 | FLAT | Feedforward Gaussian splatting without depth prior |
| Baseline 2 | Lyra | Uses video diffusion latents for 3D reconstruction |
| Ablation of ours | w/o depth consistency | Isolates effect of depth loss on geometry |

### Why this setup validates the claim

This setup tests whether our depth consistency loss bridges the distributional gap between latent codes and 3D geometry, compared to alternatives that use frozen video latents (Lyra) or no geometric prior (FLAT). ShapeNet provides ground-truth 3D shapes, allowing Chamfer Distance to quantify geometric fidelity. The ablation removes the depth loss to isolate its contribution. If the depth loss is critical, we expect the full method to outperform both baselines and the ablation, especially on geometrically complex scenes where monocular depth is reliable. This forms a falsifiable test: if our method fails to improve over FLAT on such scenes, the claim that depth consistency helps is false.

### Expected outcome and causal chain

**vs. FLAT** — On a case with thin structures (e.g., chair legs), FLAT produces fragmented or incomplete geometry because its viewpoint-agnostic latent space lacks 3D awareness, causing misalignment across views. Our method instead enforces depth consistency from a monocular estimator, which provides coarse shape cues that align the latent code with a plausible 3D layout. Consequently, we expect a 10-20% improvement in Chamfer Distance on objects with fine details, but parity on simple, blob-like shapes where FLAT already reconstructs well.

**vs. Lyra** — On a case with significant viewpoint changes (e.g., large rotation between training views), Lyra produces blurry renderings and distorted geometry because its video diffusion latent is optimized for temporal smoothness, not 3D consistency across large angle gaps. Our method directly optimizes latent codes via multi-view reconstruction loss and depth consistency, which is agnostic to video priors. We expect our method to achieve lower Chamfer Distance by 15-25%, especially when camera poses are spread widely (e.g., 60° apart), while Lyra may catch up when views are small (e.g., 10° apart).

### What would falsify this idea

If the full method performs similarly to the ablation without depth consistency, or if the improvement over FLAT is uniform across all object categories (rather than concentrated on geometrically complex cases), then the depth loss is not providing the claimed geometric benefit and the central hypothesis is invalid.

## References

1. FLAT: Feedforward Latent Triangle Splatting for Geometrically Accurate Scene Generation
2. Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation
3. Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation
4. I2VControl-Camera: Precise Video Camera Control with Adjustable Motion Strength
5. Motion Prompting: Controlling Video Generation with Motion Trajectories
6. HunyuanVideo: A Systematic Framework For Large Video Generative Models
7. Wonderland: Navigating 3D Scenes From a Single Image
8. AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers
