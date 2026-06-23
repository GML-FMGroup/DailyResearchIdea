# Joint Implicit Neural Field Diffusion for Globally-Consistent Novel View Synthesis from a Single Image

## Motivation

Existing methods like SceneCompleter generate novel views independently via diffusion, without enforcing global 3D consistency, leading to view-dependent artifacts and misalignments. The root cause is the lack of a shared 3D representation during generation; each view is synthesized in isolation from the conditioning image, preventing geometric and appearance coherence across viewpoints.

## Key Insight

By jointly denoising a continuous 3D implicit neural field across multiple camera views in a single diffusion process, the generated field inherently enforces global consistency, as all views are rendered from the same underlying representation.

## Method

### (A) What it is
**JointFieldDiffusion** is a latent diffusion model that generates a neural implicit field (parameterized by a compact latent code of dimension 64) from a single input image. The field can be rendered at arbitrary novel viewpoints via differentiable volume rendering, guaranteeing multi-view consistency.

### (B) How it works
```python
# Training
for each scene with input image I and ground-truth views {V_j}:
    # encode scene into a latent code z (shared across views)
    z = encode_scene(I)  # ResNet-50 without final FC, output 2048-D global feature, projected to 64-D via linear layer; multi-scale local features (from layers 2 and 3) are also extracted for conditioning
    # add noise to z: z_t = sqrt(alpha_t)*z + sqrt(1-alpha_t)*epsilon, alpha_t from linear schedule beta (1e-4 to 0.02, T=1000)
    # denoise z_t conditioned on I and target camera poses {P_j}
    # predict noise epsilon_hat using a transformer that processes camera rays
        # transformer: 6 layers, 8 heads, hidden dim 512, cross-attends to image features (global and local grids)
        # each ray: positional encoding of origin and direction (6 frequencies), concatenated with camera pose (flattened 4x4 matrix)
        # sample 1024 rays per batch, 256 points per ray
    # compute loss on rendered views: render z_0_hat (predicted clean z) via NeRF decoder
        # NeRF decoder: 8-layer MLP, hidden dim 256, ReLU, skip connection at layer 4
        # outputs RGB (3) and density (1); uses positional encoding of 3D points (10 frequencies) and view direction (4 frequencies)
    # apply L1+perceptual loss on rendered vs ground-truth views (L1 weight 1.0, LPIPS via VGG weight 0.1)
    # optimize diffusion loss (simple MSE on epsilon) and reconstruction loss jointly

# Inference
Input: single image I
1. Encode I to conditioning embedding c (global+local features)
2. Sample random noise z_T ~ N(0,I) of dimension 64
3. Iteratively denoise z_t using the learned denoiser (transformer) conditioned on c, with DDIM scheduler (50 steps) for speed
4. Obtain predicted clean latent z_0
5. Decode z_0 into NeRF weights (MLP)
6. Render novel views from desired camera poses via volume rendering (512 rays per view, 128 points per ray)
```
Load-bearing assumption: A single latent code z of dimension 64 can capture all 3D scene information necessary for globally consistent novel view synthesis from a single image. We calibrate this by monitoring the average depth consistency across rendered views during training: we compute depth maps from two random poses and measure the alignment of visible surfaces; if the assumption holds, depth maps should be geometrically consistent. During training, we log this depth consistency metric (L1 difference in depth after warping) to verify that the latent code is not overfitting to training views.

### (C) Why this design
We chose an implicit neural field (NeRF) over explicit point clouds or meshes because it provides a continuous, differentiable representation that is naturally suited for joint diffusion and volume rendering. The trade-off is slower rendering due to ray marching, but we accept this for consistency. We adopted a latent diffusion framework (inspired by Stable Diffusion) over pixel-space diffusion to reduce computational cost and leverage pretrained image encoders; the cost is loss of high-frequency detail that can be recovered via a lightweight decoder (the NeRF MLP itself). For conditioning, we encode the input image into a global embedding and multi-scale local features that are injected via cross-attention into the transformer, rather than concatenating to the noisy latent, because this allows the model to attend to relevant image features without spatial misalignment; the drawback is that spatial details may be less localized, but we compensate with the multi-scale features. We use a transformer architecture to process ray-based samples because it naturally handles varying numbers of cameras and provides global context; the downside is higher memory usage, which we mitigate by sampling a fixed number of rays (1024) during training.

### (D) Why it measures what we claim
The latent code z is a shared representation from which all views are rendered, so the consistency across views is a direct consequence of the single code—this measures **global consistency** because the assumption is that z captures all 3D scene information; if z is insufficient (e.g., low capacity), views may be blurry but still geometrically consistent. The denoising step with conditioning on the input image ensures **scene fidelity** because the diffusion prior learned from multi-view data encourages plausible 3D structures; the assumption is that the training data covers a diverse set of scenes, and failure occurs when the input image contains unseen configurations (e.g., non-Lambertian materials), causing the generated field to be inconsistent with the input. The multi-view reconstruction loss (L1 + LPIPS, weights 1.0 and 0.1) during training explicitly enforces that the rendered views match ground truth, which measures **appearance accuracy**; the assumption is that the rendering equation is invertible, and failure mode is when occlusions or transparency break the loss (e.g., reflective surfaces), leading to averaged-out appearances.

**Resource estimate**: Training requires ~5 days on 4 V100 GPUs (32GB each). Model has ~45M parameters. Memory usage ~12GB per GPU. Latent dimension 64. Number of rays per batch 1024. Training set: 80k scenes from RealEstate10K, validation 5k, test 5k.

## Contribution

(1) A novel diffusion framework that generates a neural implicit field from a single image, enabling globally-consistent novel view synthesis by design. (2) A training strategy that jointly optimizes a shared latent representation with a multi-view reconstruction loss, ensuring both consistency and fidelity. (3) Empirical demonstration on RealEstate10K and LLFF showing superior consistency over per-view synthesis baselines (e.g., SceneCompleter) while matching or exceeding surface quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | RealEstate10K (80k train, 5k val, 5k test) | Diverse indoor/outdoor scenes with camera poses |
| Primary metric | LPIPS (lower better), also PSNR, SSIM | Perceptual fidelity; PSNR/SSIM for completeness |
| Baseline 1 | PixelNeRF (ResNet-34 encoder, 8-layer NeRF) | Generalizable NeRF with pixel-aligned features |
| Baseline 2 | latentSplat (50k 3D Gaussians, fast rendering) | Explicit 3D Gaussians with fast rendering |
| Baseline 3 | 3DILG (Chan et al., 2022) | Concurrent 3D diffusion model, independent fields |
| Ablation | JointFieldDiffusion w/o diffusion (direct NeRF decoder from image) | Isolates contribution of diffusion prior |

### Why this setup validates the claim
RealEstate10K provides a wide range of viewpoint changes (up to 90° baseline), allowing us to test multi-view consistency and generalization. LPIPS measures perceptual similarity, which is critical for generative NVS; it captures both appearance and structural fidelity better than PSNR. PixelNeRF tests the necessity of a global 3D prior (it uses local features), while latentSplat tests the advantage of implicit neural fields over explicit representations. 3DILG tests the advantage of joint denoising over generating per-view fields independently. The ablation isolates the contribution of the diffusion process. This combination forms a falsifiable test: if our method's improvements are concentrated on scenes with large viewpoint gaps or complex appearances (specular, thin structures), the causal chain (diffusion prior enables consistent field inference) is supported.

### Expected outcome and causal chain

**vs. PixelNeRF** — On a scene with thin structures (e.g., a staircase), PixelNeRF produces blurred and inconsistent novel views because it linearly interpolates pixel-aligned features without inferring occluded geometry. Our method instead reconstructs a complete neural field via diffusion, which fills in occluded parts plausibly. We expect a noticeable gap in LPIPS on such scenes (e.g., 0.05 lower) but parity on simple scenes with small baselines.

**vs. latentSplat** — On a scene with specular reflections (e.g., a car hood), latentSplat's Gaussian representation cannot capture view-dependent radiance, leading to washed-out highlights. Our method's NeRF decoder naturally models directional effects. We expect our LPIPS to be substantially lower (e.g., 0.08 gap) on reflective subsets, but similar on diffuse scenes.

**vs. 3DILG** — On a scene with large viewpoint change (e.g., 90°), 3DILG generates independent fields for each view, leading to misaligned geometry (e.g., double walls). Our joint latent enforces a single consistent field. We expect our LPIPS to be 0.1 lower on large-baseline subsets, but comparable on small baselines where 3DILG can leverage per-view consistency.

### What would falsify this idea
If our method's LPIPS improvement over baselines is uniform across all scene types (e.g., a constant 0.01 gain) rather than concentrated on cases with large occlusions, thin structures, or view-dependent effects, then the hypothesis that diffusion prior drives consistency is invalidated.

## References

1. SceneCompleter: Dense 3D Scene Completion for Generative Novel View Synthesis
2. latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction
3. CameraCtrl: Enabling Camera Control for Text-to-Video Generation
4. ReconX: Reconstruct Any Scene From Sparse Views With Video Diffusion Model
5. 3D Reconstruction with Spatial Memory
6. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
7. MagicVideo: Efficient Video Generation With Latent Diffusion Models
8. Latent Video Diffusion Models for High-Fidelity Long Video Generation
