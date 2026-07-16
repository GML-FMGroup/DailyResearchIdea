# Adaptive Register Modulation for Diffusion Transformers: Automating Hyperparameter Selection via Feature Entropy

## Motivation

Register tokens improve feature map quality in diffusion transformers (Registers Matter for Pixel-Space Diffusion Transformers), but they introduce hyperparameters—the number of registers and guidance scale—that must be manually tuned per dataset or sample. This contradicts the goal of invariance under hyperparameter variation; suboptimal choices lead to inconsistent generation quality, and the structural root cause is that the optimal configuration depends on per-sample feature complexity, which varies across noise levels and image content.

## Key Insight

The optimal register configuration (number of tokens and guidance strength) is a function of the local feature complexity, which can be quantified by the entropy of the attention weights between patches and register tokens.

## Method

### (A) What it is
We propose **Adaptive Register Modulation (ARM)**, a lightweight hypernetwork that predicts the number of register tokens and the guidance scale for each sample based on the entropy of intermediate attention maps. ARM takes the feature map from a specific DiT block and the diffusion timestep embedding, and outputs a scalar guidance multiplier and a selection of register tokens.

### (B) How it works
```python
def adaptive_register_modulation(feature_map, register_tokens, timestep_embed, num_total_registers, MAX_GUIDANCE=2.0):
    """
    feature_map: (B, N, D) from a DiT block, N = number of patches
    register_tokens: (B, num_total_registers, D)
    timestep_embed: (B, D_t) diffusion timestep embedding (e.g., from a sinusoidal embedding or from the DiT timestep MLP)
    Returns: modulated feature map, guidance_scale, num_registers
    """
    # Step 1: Compute entropy of attention between patches and registers
    attn = torch.softmax(feature_map @ register_tokens.transpose(-2, -1) / sqrt(D), dim=-1)  # (B, N, R)
    entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean(dim=-1)  # (B,)

    # Step 2: Concatenate entropy with timestep embedding and predict guidance scale (sigmoid scaled to [0, MAX_GUIDANCE])
    guidance_input = torch.cat([entropy.unsqueeze(-1), timestep_embed], dim=-1)  # (B, 1 + D_t)
    guidance_scale = torch.sigmoid(Linear(guidance_input)) * MAX_GUIDANCE  # (B,)

    # Step 3: Predict number of registers (discretized to integer) via a separate linear layer also conditioned on timestep
    count_input = torch.cat([entropy.unsqueeze(-1), timestep_embed], dim=-1)  # (B, 1 + D_t)
    raw_count = Linear(count_input).squeeze(-1)  # (B,)
    num_registers = torch.clamp(torch.round(raw_count), 1, num_total_registers).long()  # (B,)

    # Step 4: Select top-k registers based on average attention weight
    avg_attn = attn.mean(dim=1)  # (B, num_total_registers)
    _, topk_idx = torch.topk(avg_attn, k=num_registers.max().item(), dim=-1)  # (B, k_max) per sample
    selected_registers = torch.gather(register_tokens, 1, 
                         topk_idx.unsqueeze(-1).expand(-1, -1, register_tokens.size(-1)))  # (B, k_max, D)
    # Zero out registers beyond each sample's predicted count (masked aggregation not needed here)
    # For simplicity, we use mask to avoid influence of extra registers in output
    mask = torch.arange(selected_registers.size(1)).unsqueeze(0).to(selected_registers.device) < num_registers.unsqueeze(-1)  # (B, k_max)
    selected_registers = selected_registers * mask.unsqueeze(-1).float()

    # Apply Register Guidance: add scaled selected registers
    output = feature_map + guidance_scale.unsqueeze(-1).unsqueeze(-1) * selected_registers.sum(dim=1).unsqueeze(1)  # (B, N, D)
    return output, guidance_scale, num_registers
```
*Hyperparameters*: Two separate linear layers (input dim = 1 + D_t, output dim = 1, no hidden layer), each with bias. D_t = 512 (DiT timestep embedding dimension). MAX_GUIDANCE=2.0. Training uses a perceptual loss (LPIPS with AlexNet backbone, as in the original LPIPS paper) plus L1 penalty on guidance_scale (weight λ_g=0.01) and num_registers (weight λ_r=0.01). The straight-through estimator is used for gradients through the rounding operation in num_registers prediction.

### (C) Why this design
We chose to predict hyperparameters from attention entropy rather than from a learned embedding of the full sample because entropy is a lightweight, differentiable proxy that correlates with feature complexity; the trade-off is that entropy may not capture all relevant structure, leading to suboptimal predictions in cases where complexity is better captured by other statistics (e.g., variance of feature norms). We used a linear layer for guidance scale prediction over a recurrent or attention-based predictor to keep overhead minimal and avoid overfitting, accepting that the mapping may be less expressive if the relationship is highly nonlinear. We discretize the number of registers via rounding and top-k selection rather than using a differentiable relaxation (e.g., Gumbel-softmax) because exact integer counts are needed for register indexing and the rounding operation is simple; the cost is that gradients through rounding must be approximated via straight-through estimator, potentially introducing bias in early training. Using top-k selection based on average attention (instead of random or learned selection) ensures that the most attended registers are preserved, which aligns with the original finding that registers with high attention are most beneficial; the trade-off is that this may discard registers that are useful at later noise levels if attention shifts. Conditioning the linear layers on the diffusion timestep embedding addresses the documented inverse correlation between entropy and register importance across timesteps, ensuring that the predictions adapt to noise level.

### (D) Why it measures what we claim
The entropy of attention weights between patches and registers measures the uniformity of attention, which captures the complexity of the feature map: high entropy implies distributed attention requiring more registers, and low entropy implies focused attention needing fewer registers, under the assumption that register tokens serve to clean up feature maps by aggregating information from multiple patches and that the monotonicity holds: higher entropy → higher register utility. This assumption fails when the feature map is already clean but has high entropy due to noise (e.g., early timesteps), in which case entropy overestimates the required registers, leading to unnecessary computational cost but not quality degradation; the timestep conditioning mitigates this by allowing the model to downweight entropy in early steps. The predicted guidance scale from entropy measures the amplification needed for register contributions; the linear mapping from entropy to scale assumes that the benefit of registers increases with complexity, which holds when registers provide complementary information to existing features, but fails when registers are redundant (e.g., at very high entropy they may add noise), in which case the scale may be too high, causing over-amplification. The number of registers predicted via rounding and top-k selection measures the subset of registers that are actually needed; the assumption that top-k attention weight sum approximates total register utility is valid when attention weights correlate with register importance, but fails if some registers have low attention but high utility (e.g., at low noise levels), leading to underutilization. The L1 penalty in training acts as a prior that encourages minimal register usage, which operationalizes invariance to hyperparameter over-specification; the assumption that smaller hyperparameters generalize better holds when registers are not strictly necessary, but fails when all registers are needed for quality, forcing the model to trade off between penalty and perceptual loss.

## Contribution

(1) A novel adaptive method (ARM) that automatically determines the number of register tokens and guidance scale per sample based on feature entropy, eliminating manual hyperparameter tuning in diffusion transformers. (2) A design principle that local feature complexity, as measured by attention entropy, is a reliable predictor of optimal register configuration, validated through qualitative and quantitative analysis.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | ImageNet 256×256 | Standard benchmark for class-conditional generation |
| Primary metric | FID (Frechet Inception Distance) | Captures fidelity and diversity |
| Baseline 1 | Standard DiT (no register tokens) | Tests the core value of register tokens |
| Baseline 2 | PixelDiT | Tests against a prior pixel diffusion transformer |
| Baseline 3 | Fixed Register DiT (non-adaptive, register count = 4, guidance scale = 1.0) | Tests the necessity of adaptivity |
| Ablation 1 | ARM w/o adaptive guidance (fixed guidance scale = 1.0, but adaptive register count) | Isolates the benefit of guidance scaling |
| Ablation 2 | ARM w/o timestep conditioning (entropy-only prediction) | Quantifies the contribution of timestep embedding |
| Correlation Study | Sweep over register count (1-8) and guidance scale (0.5, 1.0, 1.5, 2.0) on 512 validation samples | Verifies that entropy correlates with optimal configuration

### Why this setup validates the claim

This combination of dataset, baselines, and metric forms a falsifiable test of ARM's central claim: that predicting register count and guidance from attention entropy (conditioned on timestep) improves image quality. ImageNet 256×256 is the de facto standard for class-conditional generation. FID is the primary metric because it jointly measures fidelity and diversity, which are the intended effects of register modulation. The baselines isolate each sub-claim: Standard DiT tests whether any register tokens help; PixelDiT tests whether ARM outperforms a strong pixel-space alternative; Fixed Register DiT tests whether adaptive prediction beats a fixed configuration; Ablation 1 quantifies the specific contribution of the learned guidance scale; Ablation 2 isolates the effect of timestep conditioning; the Correlation Study directly tests the load-bearing assumption that entropy predicts optimal hyperparameters. If ARM fails to outperform Fixed Register DiT, adaptivity is unwarranted; if it fails against PixelDiT, the overall approach is not competitive; if Ablation 2 performs comparably, timestep conditioning is unnecessary; if the correlation between entropy and optimal configurations is low, the fundamental premise is flawed.

### Expected outcome and causal chain

**vs. Standard DiT (no registers)** — On a high-complexity sample (e.g., a scene with many objects and textures), the standard DiT produces blurry or artifact-ridden features because a fixed number of registers would be needed to clean the feature maps, but none are present. Our method instead predicts a higher register count and guidance scale from the high entropy of attention, adding amplified register contributions that sharpen features. We expect a noticeable FID improvement on high-entropy samples (e.g., >0.3 FID drop) but parity on low-entropy ones.

**vs. PixelDiT** — On a sample with intricate fine-grained texture (e.g., fur or foliage), PixelDiT uses a fixed architecture without adaptive register modulation, so it may miss local details or over-apply uniform processing. Our method adaptively selects registers based on attention entropy, providing targeted amplification where needed. We expect ARM to achieve lower FID overall (e.g., 0.2–0.4 FID improvement), with gains concentrated on high-entropy images.

**vs. Fixed Register DiT** — On a simple sample (e.g., a blank background with a single object), a fixed large number of registers wastes computation and may introduce noise additive guidance, degrading quality. Our method predicts low entropy, leading to few registers and low guidance, thus preserving clean features. On complex samples, Fixed Register DiT under-modulates because its configuration is a compromise. ARM adapts, so we expect a FID improvement (e.g., 0.15–0.3 FID drop) that is larger on high-entropy and low-entropy extremes than on moderate entropy.

**Correlation Study** — We expect that across the 512 validation samples, the entropy of attention weights correlates with the optimal register count (Pearson r > 0.5) and optimal guidance scale (r > 0.4), validating the proxy. Failure of correlation would falsify the load-bearing assumption.

### What would falsify this idea
If ARM's FID gains are uniform across all entropy levels rather than concentrated on high-and-low extremes, then the entropy-based mechanism fails as a causal predictor. Alternatively, if ARM does not outperform Fixed Register DiT, then adaptivity is not beneficial. Additionally, if the correlation between entropy and optimal hyperparameters is weak (r < 0.3), the entire premise is unsupported.

## References

1. Registers Matter for Pixel-Space Diffusion Transformers
2. PixelDiT: Pixel Diffusion Transformers for Image Generation
3. Scalable Diffusion Models with Transformers
