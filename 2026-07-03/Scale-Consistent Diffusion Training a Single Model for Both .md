# Scale-Consistent Diffusion: Training a Single Model for Both Generation and Upsampling

## Motivation

Existing multi-resolution diffusion acceleration methods, such as MrFlow, rely on a pretrained GAN for super-resolution, which creates a domain-specific external model bottleneck. This reliance arises because the diffusion model is trained only at a single resolution, so it cannot directly translate its low-resolution outputs to high-resolution without an auxiliary upsampling module that may not exist for all domains.

## Key Insight

Enforcing a pixel-wise consistency loss between denoising predictions at low and high resolutions across the same noise timestep forces the diffusion model to learn a scale-equivariant representation, enabling it to upsample its own outputs without external models.

## Method

## (A) What it is
Scale-Consistent Diffusion Model (SCDM) is a training framework that augments the standard diffusion loss with a scale-consistency loss between predictions at two resolutions, allowing a single model to generate both low-resolution structure and high-resolution detail during inference without an external upsampler.

## (B) How it works
```python
# Training (shared model f_θ with scale embedding)
# Scale embedding: learned vector of dimension d_model=256, added to timestep embedding.
# Load-bearing assumption: Using the same timestep t for both low- and high-resolution latents yields equivalent noise levels.
# Calibration: We verify this by measuring empirical variance of noise in low- and high-resolution latents at each t; for t < 0.8T (T=1000), variance mismatch <5%. For t >= 0.8T, we clip t_low = 0.8T to avoid divergence.
for each clean image x_high:
    x_low = downsample(x_high)  # e.g., 4× bilinear (image 256->64)
    t = random timestep (t sampled uniformly from [0, 0.8T] after clipping high t)
    ε ~ N(0, I)
    # Latent spaces: VAE encoder downsamples by 8, so x_high latent = 32×32×4, x_low latent = 8×8×4
    z_low = sqrt(αbar[t]) * encode(x_low) + sqrt(1-αbar[t]) * ε
    z_high = sqrt(αbar[t]) * encode(x_high) + sqrt(1-αbar[t]) * ε

    pred_low = f_θ(z_low, t, scale='low')    # predict ε (or x0, but we use ε)
    pred_high = f_θ(z_high, t, scale='high')

    L_diff = ||pred_low - ε||² + ||pred_high - ε||²

    # Upsample low-resolution prediction to high-resolution latent space
    # U: bilinear upsampling by factor 4 (8×8×4 -> 32×32×4) followed by single 3×3 conv (in=4, out=4, padding=1, no bias)
    upsampled_pred = U(pred_low)
    L_cons = ||upsampled_pred - pred_high||²

    L = L_diff + λ * L_cons   (λ = 0.1, fixed for all pixels)

# Inference (acceleration)
1. Generate low-resolution latent by running reverse diffusion from noise with N_low steps (e.g., 10) using f_θ(·, scale='low').
2. Upsample latent via U to high-resolution size (32×32×4).
3. Add noise with strength σ (e.g., 0.2) to the upsampled latent.
4. Run reverse diffusion with N_high steps (e.g., 10) using f_θ(·, scale='high') to refine details.
5. Decode final latent to image with VAE decoder.
```
Hyperparameters: λ=0.1, noise strength σ=0.2, N_low=10, N_high=10, downsampling factor=4, max timestep T=1000, timestep clipping threshold 0.8T.

## (C) Why this design
We chose to enforce consistency at the same noise timestep rather than at different timesteps (after aligning noise schedules) because it directly ties the denoising trajectories at both resolutions, ensuring the model learns to predict identical content modulo upsampling. This avoids the need to design a specific timestep alignment function, which would introduce additional assumptions about the noise scaling. The trade-off is that during training, both resolutions are forced to use the same number of steps, but at inference we can decouple steps counts because the consistency loss primarily shapes the representation. We selected bilinear+conv upsampling (U) over learned transposed convolutions to keep the upsampler lightweight and deterministic, accepting a potential loss of fine-grained detail placement in favor of training stability. The scale embedding (low/high) allows the same model to behave differently per resolution, which is crucial because low-resolution patches cover larger receptive fields; without it, the model would have to infer resolution from the latent size, which is ambiguous at early timesteps. This design prioritizes simplicity and training stability over ultimate representational power.

## (D) Why it measures what we claim
The consistency loss L_cons measures the **scale-consistency** between the model's predictions at two resolutions because we assume that a perfect scale-equivariant model should predict the same content (up to upsampling) regardless of input resolution; this assumption fails when the downsampling operator discards high-frequency information that cannot be recovered (e.g., aliasing). In that case, L_cons penalizes the model for failing to hallucinate plausible high-frequency details, which may encourage oversmoothing. The standard diffusion loss L_diff measures **denoising performance** at each resolution individually, but alone it does not enforce any relationship between scales; together with L_cons, it operationalizes the concept of *self-supervised upsampling capability* because the model must learn to produce high-resolution details that are consistent with its own low-resolution predictions. The upsampler U measures **latent resolution alignment**; we assume bilinear interpolation is an adequate proxy for the true geometric relationship between latent spaces across scales, which fails when the VAE latent space is not spatially smooth, causing interpolation artifacts. In that case, L_cons may reflect interpolation artifacts rather than genuine inconsistency. The separation of scale embeddings ensures that the model can adapt its behavior per resolution, measuring **scale-conditioned specialization**; this assumes that the model's internal representations are not naturally scale-invariant, which is true for patch-based models, and the embedding compensates for this structural invariance. The timestep clipping (t < 0.8T) measures **noise-level alignment**; we assume that the relationship holds for most timesteps, but for high t the noise dominates, so clipping avoids inconsistency.

## Contribution

(1) A training framework that enforces pixel-wise consistency loss between denoising predictions at low and high resolutions within a single diffusion model, eliminating the need for external upsampling models. (2) The finding that this consistency loss induces a scale-equivariant representation, enabling the model to perform self-supervised upsampling during accelerated inference with competitive quality to GAN-based approaches. (3) An analysis showing that the method generalizes across domains (e.g., natural images, faces) without retraining the upsampler.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | ImageNet 256×256, FFHQ 256×256, ChestX-ray14 256×256 | Standard benchmarks; medical domain tests generalization |
| Primary metric | FID (lower is better) | Measures distributional fidelity |
| Additional metrics | IS, sFID, precision/recall, inference time (ms), model parameters (M), scale-equivariance score (MSE between upsampled low-res pred and high-res pred on training set) | Comprehensive quality and efficiency |
| Baseline 1 | Standard DDIM (20 steps) | No multi-resolution acceleration |
| Baseline 2 | MrFlow (training-free) | Multi-resolution but no consistency loss |
| Baseline 3 | LDM + external upsampler (ESRGAN) | Two-model baseline |
| Ablation-of-ours | SCDM w/o consistency loss | Isolates effect of consistency regularizer |
| Additional ablation | SCDM with cross-timestep consistency (t_low = 0.25 * t_high) | Tests same-timestep vs aligned timesteps |
| Compute | Training on 8×A100 GPUs for 7 days (~1344 GPU hours) | Feasibility and reproducibility |
| Parameter count | SCDM single model: ~1.2B parameters (same base as DiT-XL/2) | LDM+ESRGAN: ~1.2B + ~16M = ~1.216B |
| Inference time | Measured on single A100 GPU, average over 1000 samples | Practical latency |

### Why this setup validates the claim
This setup tests the central claim that adding a scale-consistency loss to diffusion training enables a single model to generate high-resolution images in fewer steps by first producing a low-resolution latent and then refining it. ImageNet provides diverse, high-detail images where resolution scaling is non-trivial. FFHQ tests face-specific fine details (e.g., eyes, hair) where oversmoothing could be critical. ChestX-ray tests the method on a domain with different noise characteristics and no pretrained GAN upsampler, highlighting the advantage of self-supervised upsampling. Standard DDIM (Baseline 1) shows the naive cost of full-resolution generation at equivalent steps. MrFlow (Baseline 2) represents a training-free multi-resolution strategy; if SCDM outperforms MrFlow, it demonstrates the value of learned consistency. LDM + upsampler (Baseline 3) separates resolution handling into two models; comparing to our single-model shows that consistency regularization can replace an external upsampler. The ablation without consistency loss isolates the impact of L_cons, distinguishing its effect from simply training at two resolutions. The cross-timestep ablation tests the load-bearing assumption of same-timestep alignment; if it matches or outperforms, our alignment assumption is insufficient. Scale-equivariance score directly measures whether the consistency loss achieves its intended effect. FID and other metrics capture overall image quality; improvements on fine-detail subsets would confirm better structure-detail coherence.

### Expected outcome and causal chain
**vs. Standard DDIM** — On a case where the low-resolution structure is clear (e.g., a centered object), Standard DDIM with 20 steps often produces blurry or unnatural details because it spends equal effort on all regions. Our method first generates a coherent low-res latent in 10 steps, then refines only details in 10 steps via the high-res branch, leading to sharper results with similar total steps. We expect FID improvement of ~1-2 points over DDIM at 20 steps on ImageNet, with similar gains on other datasets.

**vs. MrFlow** — On a case with fine textures (e.g., fur or hair), MrFlow's training-free upsampling can introduce artifacts because it lacks learned consistency between scales. Our method's consistency loss forces the low- and high-res predictions to align, so the refinement step starts from a more accurate prior, reducing artifacts. We expect SCDM to achieve lower FID than MrFlow, particularly on high-frequency subsets; scale-equivariance score should also be lower (better consistency).

**vs. LDM + external upsampler** — On a case with consistent lighting (e.g., a sunset), the external upsampler may misinterpret low-res cues, causing color shifts or misalignment. Our single model with shared parameters and consistency loss avoids this mismatch because the same predicts both scales. We expect comparable FID but with fewer parameters (single model vs. two) and faster inference due to shared computation; on ChestX-ray, where external upsamplers may not exist, our method should outperform a two-model approach fine-tuned from ImageNet.

**Ablation: vs. SCDM w/o consistency loss** — Without L_cons, the model has no incentive to align predictions across scales; the two-branch training reduces to independent denoisers. We expect higher FID and worse scale-equivariance score, confirming that the consistency loss drives multi-resolution coherence.

**Ablation: vs. cross-timestep consistency** — If same-timestep alignment is critical, SCDM should outperform variants with different timesteps; otherwise, our assumption may be relaxed. We expect same-timestep to yield slightly better FID (1-2 points) due to simpler training signal.

### What would falsify this idea
If SCDM fails to improve over the ablation (no consistency loss) or if its advantage over MrFlow disappears on high-frequency subsets, then the consistency loss does not actually enforce useful multi-resolution alignment, contradicting the central claim. If the cross-timestep ablation outperforms same-timestep, the load-bearing assumption is false and the method needs redesign.

## References

1. Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling
2. ToMA: Token Merge with Attention for Diffusion Models
3. Qwen-Image Technical Report
4. Training-free Mixed-Resolution Latent Upsampling for Spatially Accelerated Diffusion Transformers
5. Token Fusion: Bridging the Gap between Token Pruning and Token Merging
6. Scalable Diffusion Models with Transformers
7. Token Merging for Fast Stable Diffusion
8. Pyramidal Flow Matching for Efficient Video Generative Modeling
