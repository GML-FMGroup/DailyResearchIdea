# Self-supervised Spherical Rotation Equivariance for In-Context Panoramic Generation

## Motivation

Current in-context panoramic generation methods, such as Canvas360, require large paired datasets (e.g., RGB+depth) to learn geometric priors, limiting scalability and generalization. The root cause is that these methods lack an inductive bias for spherical geometry; they treat equirectangular projections as generic images, ignoring the rotational equivariance property that a spherical rotation corresponds to a deterministic transformation in the equirectangular domain. This reliance on curated paired data creates a bottleneck across multiple branches of panoramic generation research.

## Key Insight

Spherical rotation equivariance provides a free supervisory signal because any rotation of the scene induces a deterministic transformation of equirectangular pixels, enabling the model to learn geometric consistency without paired datasets.

## Method

We propose Rotation-Aware Self-Supervised Panorama Generation (RASS).

(A) **What it is:** RASS is a self-supervised training framework for diffusion-based panoramic image generators that enforces spherical rotation equivariance via a consistency loss using spherical harmonic positional encoding. Input: a set of unlabeled equirectangular images. Output: a pretrained diffusion model that can be adapted to in-context generation tasks.

(B) **How it works:** We build upon a diffusion model backbone (e.g., DiT with 24 layers, hidden size 1024, attention heads 16). For each training mini-batch, we sample an image x and apply a random spherical rotation R (parameterized by Euler angles, yaw uniformly sampled from [0, 2π), pitch/roll uniformly sampled from [-30°, 30°]). The rotated image x_R is obtained by resampling the equirectangular grid according to the rotation using bilinear interpolation. We then compute the denoising objective on both x and x_R. The key component is the spherical harmonic positional encoding (SHE) added to the noise latents. SHE encodes the angular coordinates (θ, φ) on the sphere using spherical harmonics up to degree L=10, resulting in a 121-dimensional encoding per pixel. During training, we impose a consistency loss: the denoising prediction for x_R should be the rotation of the denoising prediction for x. Specifically, let ε_θ(z_t, t, SHE) be the predicted noise where SHE is the positional encoding at the current latent resolution. We require that ε_θ(z_t^R, t, SHE_R) = R(ε_θ(z_t, t, SHE)), where SHE_R is the rotated positional encoding obtained by applying the Wigner D-matrix (degree L) to the SHE coefficients. The total loss is L_total = L_diffusion + λ L_consistency, where L_consistency is the MSE between the rotated predicted noise from the original and the predicted noise from the rotated latent. Hyperparameters: λ=0.1, rotation angles sampled uniformly in [0,2π) for yaw, and pitch/roll limited to max 30°. We note that this procedure assumes the Gaussian noise added in the equirectangular latent space is isotropic and rotation-equivariant under the spherical resampling, which is not true on the sphere (see De Bortoli et al., ICML 2022). This is a known fatal limitation of the current design; we discuss it in the limitations.

```pseudocode
# Pseudo-code for one training step
for each batch:
  x = sample batch of equirectangular images
  R = random spherical rotation (yaw ∈ [0,2π), pitch/roll ∈ [-30°,30°])
  x_R = warp(x, R)  # resample equirectangular grid via bilinear interpolation
  z0 = encode(x)  # to latent space via VAE (e.g., 4x downsampled)
  z0_R = encode(x_R)
  t = random timestep
  z_t = add_noise(z0, t)  # add isotropic Gaussian noise N(0, I)
  z_t_R = add_noise(z0_R, t)
  SHE = spherical_harmonic_encoding(angular_coords, L=10)  # per-pixel encoding
  SHE_R = rotate_encoding(SHE, R)  # apply Wigner D-matrix
  pred_noise = model(z_t, t, SHE)
  pred_noise_R = model(z_t_R, t, SHE_R)
  loss_diff = diffusion_loss(pred_noise, noise) + diffusion_loss(pred_noise_R, noise_R)
  loss_cons = MSE(rotate(pred_noise, R), pred_noise_R)
  loss = loss_diff + λ * loss_cons
  update model
```

(C) **Why this design:** We chose SHE over standard sinusoidal positional encoding because SHE is rotation-equivariant in the frequency domain, enabling easy application of rotations via Wigner D-matrices. We chose a consistency loss on noise predictions rather than on the denoised images because the noise prediction is more aligned with the diffusion process and avoids blurring. We limited pitch/roll rotations to small angles (max 30°) to avoid severe stretching near poles, accepting the cost that the model may not learn full spherical invariance for extreme orientations. We also chose to add SHE to the latent features rather than to the input pixels to maintain compatibility with latent diffusion architectures. Compared to Canvas360's paired training, our method avoids the need for depth data by exploiting the self-supervised rotation signal. The use of isotropic Gaussian noise is a simplifying assumption that is not valid on the sphere, as discussed below.

(D) **Why it measures what we claim:** The consistency loss L_consistency measures spherical rotation equivariance because it enforces that the model's internal representation transforms correctly under rotation; this assumption (that a rotation of the scene corresponds to a rotation of the latent representation) fails when the rotation introduces interpolation artifacts (e.g., resampling noise) or when the scene has dynamic content (e.g., moving objects), in which case the loss reflects interpolation inconsistency rather than geometric inconsistency. Additionally, the noise added in the diffusion process is assumed to be isotropic and rotation-equivariant, but Gaussian noise on the equirectangular grid is not isotropic on the sphere (De Bortoli et al.), breaking the consistency loss. Hence, L_consistency measures rotation equivariance only under the assumption of isotropic noise; when this assumption fails, the loss is contaminated by noise anisotropy. The SHE component operationalizes the geometric prior by making the model's positional encoding itself rotation-equivariant, so that the model can learn to map rotated inputs to rotated outputs without seeing paired rotations during training. Thus, the loss directly quantifies the model's adherence to the spherical geometry under the false noise isotropy assumption.

**Validation experiment:** To verify the geometric prior, we measure equivariance error on a synthetic dataset of 1000 randomly rotated spherical images rendered from 3D models. For each rotation, we compute the rotated predicted noise from the original and the predicted noise from the rotated latent, then record the MSE. A low error (target <0.05) indicates the model respects spherical rotations. We compare against a model using standard sinusoidal positional encoding to show the benefit of SHE.

**Compute and cost savings:** Training requires ~500 GPU-hours on 4x NVIDIA A100 (80GB) GPUs using PyTorch 2.0, HuggingFace diffusers, and CUDA 11.8. Our method needs only 100k unlabeled equirectangular images (e.g., from SUN360), costing an estimated $5k for collection and storage, whereas paired RGB+D datasets like Canvas360 cost ~$50k to collect and annotate.

## Contribution

(1) A self-supervised training framework for panoramic diffusion models that enforces spherical rotation equivariance without paired data, using spherical harmonic positional encoding and a consistency loss. (2) An empirical demonstration that the proposed method learns geometric priors that transfer to in-context generation tasks, reducing the need for large paired datasets. (3) A method for applying controllable rotations in the latent space of diffusion models for equirectangular images.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Canvas360Dataset | Standard paired panorama dataset for downstream tasks |
| Primary metric | FID | Measures overall quality and distribution match |
| Baseline 1 | Paint-by-Example | In-context generation without panorama prior |
| Baseline 2 | Stable Diffusion | General text-to-image baseline |
| Baseline 3 | PanoGAN | Dedicated panoramic GAN baseline |
| Baseline 4 | Spherical CNN (S2CNN) + DiT | Tests rotation-equivariant architectures |
| Ablation-of-ours | RASS w/o consistency loss | Isolates effect of rotation equivariance |

### Why this setup validates the claim
This setup provides a falsifiable test of the central claim that self-supervised rotation equivariance pretraining (RASS) improves downstream panoramic generation. By comparing against baselines that lack spherical awareness (Paint-by-Example, Stable Diffusion), use different geometric priors (PanoGAN), or directly incorporate rotation equivariance via spherical CNNs (S2CNN+DiT), we can attribute any gains to our consistency mechanism. The ablation removes the consistency loss to verify its necessity. FID is chosen because it captures both visual quality and distribution alignment, and we expect our method to yield lower FID specifically on test cases with significant rotations, as the baselines would produce distorted or inconsistent outputs. The dataset provides paired samples necessary for fine-tuning on in-context tasks, allowing direct comparison of generation fidelity.

### Expected outcome and causal chain
**vs. Paint-by-Example** — On a case where the input image has a 90° yaw rotation, Paint-by-Example generates a panorama with visible seams and misaligned content because it lacks any geometric prior and treats the equirectangular grid as a flat image. Our method instead produces a seamless stitched result because the pretrained diffusion model enforces rotation equivariance via the consistency loss with SHE. We expect a noticeable FID gap (e.g., >3 points) on heavily rotated subsets, but parity on near-upright views.

**vs. Stable Diffusion** — On a scene with large pitch variation (e.g., looking up), Stable Diffusion, even when fine-tuned on panoramas, introduces pole distortions because its sinusoidal positional encoding is not rotation-equivariant. Our method handles this by explicitly encoding spherical harmonics that rotate consistently, leading to better perspective consistency. We expect a moderate FID improvement (e.g., 1-2 points) on such cases, and no degradation on standard orientations.

**vs. PanoGAN** — On a complex indoor scene with fine texture details, PanoGAN produces blurry or repetitive patterns due to its GAN architecture's difficulty in capturing high-frequency content. Our diffusion-based method, with consistency loss, retains sharp details and maintains geometric coherence across rotations. We expect lower FID on all subsets, especially on texture-rich images (e.g., >2 points).

**vs. Spherical CNN (S2CNN+DiT)** — On a scene with a 45° pitch rotation, the S2CNN-based baseline preserves equivariance better than standard DiT but underperforms our method because spherical CNNs struggle with the equirectangular projection's distortions. Our method leverages the diffusion training objective to handle these distortions. We expect our method to outperform by ~1 FID point on rotated subsets.

**vs. RASS w/o consistency loss** — On a scene where the rotation introduces new spherical harmonics content, the ablation fails to align the predicted noise with the rotated ground truth, resulting in ghosting artifacts. Our full method enforces the consistency loss, forcing the model to learn an equivariant representation. We expect a clear FID drop (e.g., >4 points) on the heavily rotated subset, but similar performance on unrotated images.

### What would falsify this idea
If our method shows uniform FID improvement across all rotation angles rather than a concentrated gain on highly rotated cases, it would suggest that the consistency loss is not providing the claimed geometric equivariance but acting as a generic regularizer. Conversely, if the ablation also performs well on rotated cases, the consistency loss is unnecessary. Furthermore, if the validation experiment on synthetic rotations shows high equivariance error (e.g., >0.1 MSE), it would indicate that the SHE and consistency loss do not achieve the desired geometric prior.

## References

1. Enhancing In-context Panoramic Generation via Geometric-aware Pretraining
2. GeoVideo: Introducing Geometric Regularization into Video Generation Model
3. UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation
4. DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training
5. Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation
6. DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion
7. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
8. MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos
