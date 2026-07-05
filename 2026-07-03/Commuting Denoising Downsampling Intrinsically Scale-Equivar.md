# Commuting Denoising Downsampling: Intrinsically Scale-Equivariant Diffusion without External Super-Resolution

## Motivation

Current diffusion acceleration methods, such as MrFlow (Multi-Resolution Flow Matching), rely on external pretrained super-resolution modules (e.g., GANs) to upscale low-resolution outputs, limiting generality across domains and introducing acquisition cost. The root cause is that standard diffusion models are not scale-equivariant: the denoising operator at different resolutions is independent, forcing the use of separate upsampling networks. By architecturally enforcing a commuting property between denoising and downsampling, we can eliminate the need for any external module.

## Key Insight

Enforcing that the denoising operator commutes with downsampling (i.e., D∘S = S∘D) ensures that the learned denoising trajectory is consistent across scales, allowing the same network to generate at high resolution after initial low-resolution inference without additional training.

## Method

**(A) What it is:** Commuting Denoising Downsampling (CDD) is a training objective that regularizes a diffusion model to satisfy D(S(x, t), t) ≈ S(D(x, t), t), where D is the denoising step (predicted clean image), S is a fixed downsampling operator (bilinear, scale factor s=2), and x is the high-resolution input at noise level t. The output is a model whose denoising operator is approximately scale-equivariant, enabling self-contained multi-resolution generation.

**(B) How it works:**
```pseudocode
Input: high-res image x_H (e.g., 32x32 for CIFAR-10), noise level t (uniformly sampled from 1,...,1000),
       downsampling operator S (bilinear 2×), noise schedule sigma(t) (linear, beta_1=1e-4, beta_T=0.02)

# Standard diffusion outputs (epsilon prediction)
eps_H = denoise_network(x_H + sigma(t)*epsilon, t)   # predict noise at high res
eps_L = denoise_network(S(x_H) + sigma(t)*epsilon', t)  # predict noise at low res (different noise sample)

# Denoised clean images (DDPM formulation)
clean_H = x_H - sigma(t) * eps_H
clean_L = S(x_H) - sigma(t) * eps_L

# Commuting constraint: denoise then downsample vs. downsample then denoise
clean_H_down = S(clean_H)               # denoise first, then downsample
clean_L_direct = clean_L                 # downsample first, then denoise (already computed)

# Commuting loss
L_commute = MSE(clean_H_down, clean_L_direct)

# Total loss (lambda is hyperparameter, default 0.1)
L = L_diffusion(x_H, t) + L_diffusion(S(x_H), t) + lambda * L_commute

# Calibration: After training, compute average L_commute on 1000 validation images across 20 evenly spaced t.
# If average > 0.01, increment lambda by 0.05 and retrain for 100k additional steps.
```
Hyperparameters: lambda (commuting weight, default 0.1), S (bilinear downsampling with scale factor s=2), training steps 500k, batch size 64, Adam lr=1e-4.

**(C) Why this design:** We chose to enforce the commuting property via explicit regularization rather than architectural hard-wiring because it allows the model to learn equivariance from data while maintaining flexibility for non-ideal conditions. The trade-off is that the regularization may not be perfectly satisfied, but it can be improved with larger lambda and calibration. We use separate denoising paths for high and low resolution to compute the constraint, accepting the additional forward pass cost during training. We use MSE for L_commute because it corresponds to the expected L2 norm of the difference, which aligns with the diffusion objective. The downsampling operator is fixed (bilinear) to avoid additional learnable components, even though a learned downsampler could better adapt to data; the trade-off is that fixed downsampling may introduce artifacts but ensures consistency and simplicity. Finally, we include L_diffusion on both resolutions to ensure the model generates meaningfully at each scale, which is necessary for the commuting constraint to be meaningful.

**(D) Why it measures what we claim:** The commuting loss L_commute measures the extent to which the denoising operator D commutes with downsampling S, which operationalizes the concept of scale-equivariance because if D is truly equivariant, then D(S(x)) = S(D(x)); this assumption holds when the model's internal representations are consistent across scales and the downsampling operator preserves the relevant structure. Specifically, L_commute measures internal consistency Y (low-res denoising trajectory matches high-res refinement) because we assume approximate equivariance. This assumption fails when downsampling is non-invertible (loses high frequencies) or noise corrupts the signal to the point that the commuting constraint becomes trivial; in that case, L_commute may be low without guaranteeing good high-res refinement. During inference, we exploit the commuting property to generate at low resolution (saving compute) and then apply the same denoising steps at high resolution using the low-res output as initialization; the commuting loss during training directly targets the condition that the high-resolution refinement is consistent with the low-resolution trajectory, eliminating the need for a separate upsampler. Thus, L_commute is a direct proxy for the internal consistency that enables self-contained acceleration.

## Contribution

(1) We introduce Commuting Denoising Downsampling (CDD), a training objective that enforces scale-equivariance in diffusion models without architectural modifications. (2) We demonstrate that CDD enables self-contained acceleration: the model can generate at high resolution by first generating at low resolution and then applying the same denoising steps at full resolution, eliminating the need for external super-resolution modules. (3) We provide a training recipe that combines standard diffusion loss with a commuting regularization term, applicable to any diffusion architecture.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | CIFAR-10 (32x32) | Proof-of-concept; small size enables rapid iteration. Later scale to ImageNet 256x256 for final comparison. |
| Primary metric | FID | Measures image quality and diversity |
| Baseline1 | Full-resolution DDPM | Upper bound for quality without acceleration |
| Baseline2 | Low-res + bilinear upsample | Tests necessity of high-res refinement |
| Baseline3 | Low-res + separate SR (SR3) | Compares to two-stage pipeline |
| Baseline4 | E(2)-equivariant U-Net diffusion (Xu et al., 2024) | Compares to hard-architectural equivariance alternative |
| Ablation | CDD without commuting loss | Isolates effect of commuting constraint |

### Why this setup validates the claim
This combination directly tests the central claim that the commuting regularization endows a single diffusion model with the ability to generate at low resolution and refine consistently at high resolution without an external upsampler. By comparing to full-resolution generation, we measure the quality cost of acceleration. Comparing to naive bilinear upsampling reveals whether the model's high-res refinement actually adds meaningful detail. Comparing to a separate super-resolution model (SR3) assesses if our single-model approach can match or exceed a dedicated two-stage pipeline. Adding an equivariant network baseline (E(2)-U-Net) contextualizes whether soft regularization is competitive with hard architectural guarantees. The ablation (no commuting loss) isolates whether the commuting constraint is responsible for any improvement, or simply training on multiple resolutions is sufficient. FID is chosen because it captures both fidelity and diversity, making it sensitive to the subtle inconsistencies that the commuting constraint aims to prevent.

### Expected outcome and causal chain

**vs. Full-resolution DDPM** — On a case with fine texture (e.g., animal fur), full-resolution DDPM produces sharp details from many steps, while our method uses fewer steps at high res after low-res initialization. The baseline suffers from long runtime. Our method, due to the commuting constraint, ensures the low-res trajectory is consistent with high-res denoising, so the high-res refinement can recover fine details efficiently. We expect our FID to be close to full-resolution DDPM (e.g., within 1-2 points) while achieving significant speedup (e.g., 2-4x).

**vs. Low-res + bilinear upsample** — On a case with high-frequency patterns (e.g., checkered fabric), bilinear upsampling produces blurry artifacts because it cannot generate missing high frequencies. Our method, by rerunning denoising at high resolution with the low-res output as initialization, can synthesize novel high-frequency details consistent with the low-res semantics because the commuting loss forces the denoising operator to be scale-equivariant. We expect a large FID gap (e.g., >5 points) favoring our method, especially on images rich in texture.

**vs. Low-res + separate SR (SR3)** — On a case with rare object shapes (e.g., unusual car), a separate SR model may introduce hallucinations or inconsistencies because it is trained on generic upsampling and does not share the diffusion model's prior. Our method uses the same denoising operator across scales, so the high-res refinement is consistent with the low-res generative process. We expect our FID to be competitive with or slightly worse than the two-stage pipeline (within 1-2 points), but our method is simpler (single model) and may be faster if SR is costly.

**vs. E(2)-equivariant U-Net** — If our soft-regularization approach achieves comparable FID to the hard architecture, it validates the feasibility of learning equivariance without architectural constraints. If not, it highlights remaining gaps.

### What would falsify this idea
If the FID of CDD is not significantly better than the ablation (no commuting loss) on high-frequency subsets, or if the gap to full-resolution DDPM persists even when commuting loss is low after calibration, then the commuting constraint is not providing the expected equivariance benefit and the idea is falsified.

## References

1. Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling
2. ToMA: Token Merge with Attention for Diffusion Models
3. Qwen-Image Technical Report
4. Training-free Mixed-Resolution Latent Upsampling for Spatially Accelerated Diffusion Transformers
5. Token Fusion: Bridging the Gap between Token Pruning and Token Merging
6. Scalable Diffusion Models with Transformers
7. Token Merging for Fast Stable Diffusion
8. Pyramidal Flow Matching for Efficient Video Generative Modeling
