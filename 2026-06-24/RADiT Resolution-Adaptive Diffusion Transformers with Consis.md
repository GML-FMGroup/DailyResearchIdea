# RADiT: Resolution-Adaptive Diffusion Transformers with Consistency Training

## Motivation

Diffusion transformers (DiTs) have shown promise for image generation, but prior work such as Scaling Rectified Flow Transformers assumes transformer backbones scale to high resolutions without empirical validation at resolutions beyond 256×256. DiffusionBench further reveals that existing DiT evaluations are limited to low resolutions, leaving a gap in understanding how DiT behavior changes across scales. The root cause is the lack of mechanisms to adapt latent representations and enforce functional invariance across resolutions, leading to potential instability and degraded performance when naively upscaling.

## Key Insight

Enforcing functional invariance of the diffusion model across resolutions via a consistency loss, combined with resolution-adaptive latent normalization that stabilizes the latent distribution, is the key structural property that enables principled resolution scalability.

## Method

## (A) What it is
**RADiT** (Resolution-Adaptive Diffusion Transformer) is a training framework that augments a standard DiT with (i) a resolution-adaptive latent normalization (RALN) layer that uses a learned affine transformation to adjust the latent feature statistics based on the input resolution, and (ii) a resolution consistency loss (RCL) that encourages the model to produce similar outputs for the same image at different resolutions after appropriate scaling.

## (B) How it works
```python
# Pseudocode for one training step
# Input: image x at resolution R, noise level t, text condition c
# Hyperparameters: batch_size=256, 200K steps, 8 A100 GPUs (~72 hours)

# 1. Encode image to latent space (e.g., using VAE encoder)
z = encoder(x)  # shape: (C, H/R_lat, W/R_lat)

# 2. Resolution-aware scaling: normalize latent statistical moments
#    RALN: learned affine transformation based on resolution
#    R_emb = sinusoidal encoding of R (dim=128)
#    learned_scale = MLP(R_emb)  # 2-layer MLP, hidden=64, output shape (C,)
#    learned_shift = MLP(R_emb)  # same MLP structure, output shape (C,)
z_norm = (z - mean(z, dims=(1,2,3), keepdim=True)) / (std(z, dims=(1,2,3), keepdim=True) + 1e-6)
z_adapt = z_norm * learned_scale + learned_shift  # element-wise per channel

# 3. Forward pass through DiT with adaptive parameters
#    DiT takes z_adapt, t, c, and resolution embedding R_emb
noise_pred = DiT(z_adapt, t, c, R_emb=R_emb)

# 4. Standard diffusion loss (e.g., flow matching or DDPM)
l_diff = mse_loss(noise_pred, noise_target)

# 5. Resolution consistency loss (RCL): enforce functional invariance
#    Create a downscaled version of z (to lower resolution R_low = R/2)
z_low = downsample(z, scale_factor=0.5)  # bilinear spatial downsampling
z_low_norm = (z_low - mean(z_low)) / (std(z_low) + 1e-6)
with torch.no_grad():
    # Use same learned affine transformation for low-res (with R_low embedding)
    learned_scale_low = MLP(sinusoidal(R_low))
    learned_shift_low = MLP(sinusoidal(R_low))
    z_low_adapt = z_low_norm * learned_scale_low + learned_shift_low
    noise_pred_low = DiT(z_low_adapt, t, c, R_emb=sinusoidal(R_low))  # teacher
# Upsample noise_pred_low to match noise_pred spatial size
noise_pred_low_up = upsampl(noise_pred_low, scale_factor=2.0)  # bilinear
# Consistency loss: MSE between upsampled low-res prediction and high-res prediction
l_rcl = mse_loss(noise_pred, noise_pred_low_up.detach())

# 6. Total loss
loss = l_diff + lambda_rcl * l_rcl  # lambda_rcl = 0.1
```

## (C) Why this design
We chose a learned affine transformation over a heuristic scaling factor because the heuristic sqrt(R/256) assumes linear variance scaling with resolution, which is unverified and may over-normalize for low-variance images; a learned transformation adapts to the actual data distribution, trading off slight training overhead for robustness. We chose RCL as an explicit consistency loss rather than relying on data augmentation because it directly ties the model’s behavior across scales, ensuring that the model learns resolution-invariant features (trade-off: RCL requires an additional forward pass for the downscaled version, doubling the training cost per step—but it avoids the need for multi-resolution training data). We used a teacher-student setup with detached teacher to stabilize training; a symmetric consistency (both directions) would enforce stricter invariance but can lead to trivial solutions like zero output (trade-off: one-directional is simpler and empirically effective).

## (D) Why it measures what we claim
RCL measures functional invariance across resolutions because it minimizes the difference between the model’s output at resolution R and the upsampled output at resolution R/2. The underlying assumption is that downsampling preserves sufficient coarse structure for the teacher to provide a valid target; this assumption fails when high-frequency details are aliased, causing the loss to penalize the model for correctly generating fine details. To mitigate, we use a low-resolution teacher with detach so the model learns to add details at high resolution while aligning with the coarse structure from low resolution. The RALN component measures the stability of latent distributions: it uses a learned affine transformation to bound latent statistics across resolutions. The assumption is that the latent statistics can be stably mapped across resolutions via a learned transformation; this assumption fails if the transformation is underparameterized or if there is extreme distribution shift beyond the learned capacity. Even then, the learned parameters adapt during training to prevent gradient explosion. The combination provides diagnostic (RCL loss indicates cross-resolution consistency) and corrective (RALN prevents divergence) mechanisms for resolution scalability.

## Contribution

(1) A resolution-adaptive latent normalization (RALN) layer that stabilizes the latent distribution of diffusion transformers across different image resolutions, enabling stable training at high resolutions. (2) A resolution consistency loss (RCL) that enforces functional invariance of the diffusion model across scales, providing a principled way to transfer learned knowledge from low to high resolutions. (3) The RADiT framework that integrates RALN and RCL into a standard DiT, demonstrating that transformer-based diffusion models can be empirically scaled to resolutions beyond 256×256 without degradation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet 256x256 | Standard benchmark for DiT evaluation. |
| Primary metric | FID (Frechet Inception Distance) | Measures generation quality vs. training set. |
| Baseline 1 | Vanilla DiT (no RALN, no RCL) | Tests need for resolution adaptation. |
| Baseline 2 | DiT + RALN only (no RCL) | Isolates effect of latent normalization. |
| Baseline 3 | DiT + RCL only (no RALN) | Isolates effect of consistency loss. |
| Ablation of ours | RADiT with learned scaling | Tests heuristic vs. learned scaling factor. |
| High-res evaluation | Additional FID on ImageNet 1024x1024 | Validate scalability beyond training resolution. |

### Why this setup validates the claim
This setup validates that RADiT’s resolution adaptation (RALN) and consistency (RCL) jointly enable stable training and high-quality generation across resolutions. ImageNet 256x256 is a standard benchmark where DiT models are trained at fixed resolution. By comparing to vanilla DiT, we test whether our modifications improve performance. The two ablations (RALN-only and RCL-only) disentangle the contributions: if both are needed, neither ablation alone will match full RADiT. FID is the standard metric for generation quality and directly reflects the model’s ability to produce realistic images. The ablation with learned scaling tests whether the learned affine transformation is more effective than the previous heuristic sqrt(R/256). The high-resolution evaluation at 1024x1024 tests whether RADiT can generate plausible images at resolutions not seen during training, using FID computed on a held-out set of 1024x1024 ImageNet images downsampled to 256x256 for comparison. This combination creates a falsifiable test: if RADiT outperforms all baselines on both resolutions, the claim holds; if RALN-only or RCL-only already match RADiT, the other component is redundant; if learned scaling fails to beat heuristic, our design choice is suboptimal; if high-resolution FID does not improve over upsampled baselines, scalability is limited.

### Expected outcome and causal chain

**vs. Vanilla DiT** – On a case of an image with high-frequency details (e.g., a tiger) at 256x256, vanilla DiT produces blurry results because its latent variance grows with resolution, causing unstable training and poor detail capture. Our method instead normalizes latent statistics via RALN, keeping variance bounded, which stabilizes training and preserves details. We expect RADiT to achieve a significantly lower FID (e.g., 10% improvement) while maintaining comparable performance on low-texture images.

**vs. DiT + RALN only** – On a case where the same content appears at two different resolutions (e.g., a cityscape at 128x128 and 256x256), the RALN-only model may produce inconsistent outputs (different noise predictions) because it lacks cross-resolution constraints. RADiT’s RCL enforces consistency between predictions at different scales, leading to more coherent multi-resolution behavior. We expect RADiT to show a smaller difference in noise predictions across resolutions (measured by lower RCL loss) and a slightly better FID, especially on datasets with varied object scales.

**vs. DiT + RCL only** – On a case of a high-resolution input (e.g., 512x512) that causes latent variance to spike, the RCL-only model may produce distorted outputs because RCL alone cannot prevent variance explosion; the consistency loss may force alignment with a noisy low-resolution prediction. Our method with RALN stabilizes the latent distribution, preventing variance explosion, so the model can learn scaling-invariant features without instability. We expect RADiT to achieve better FID on high-resolution subsets while the RCL-only baseline may even degrade.

**vs. Ablation with learned scaling** – We expect RADiT with learned scaling to outperform the heuristic version on both FID and RCL loss, especially for high-resolution images where the linear assumption fails. If the heuristic performs comparably, the learned transformation may have overfitted; if it underperforms, the heuristic is sufficient.

### What would falsify this idea
If RADiT’s FID improvement over vanilla DiT is uniform across all image types (not concentrated in high-resolution or variable-resolution cases), or if the ablation with learned scaling significantly outperforms the heuristic, then the central claim that RALN and RCL specifically address resolution-induced instability and inconsistency would be falsified. Additionally, if high-resolution generation FID is worse than simple upsampling baselines, the scalability claim is unsupported.

## References

1. DiffusionBench: On Holistic Evaluation of Diffusion Transformers
2. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
3. Classifier-Free Diffusion Guidance
4. Mean Flows for One-step Generative Modeling
5. Emu Edit: Precise Image Editing via Recognition and Generation Tasks
6. CogVLM: Visual Expert for Pretrained Language Models
7. LAION-5B: An open large-scale dataset for training next generation image-text models
8. SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model
