# Locally-Aware Sparse Register Attention for Scalable Pixel-Space Diffusion Transformers

## Motivation

Register tokens improve visual structure in pixel-space DiTs (Darcet et al., 2024) but add quadratic computational cost in image resolution, violating scalability under high-resolution generation. Existing sparse attention methods (e.g., Sparse Transformer) do not account for the global role of registers, while full attention remains prohibitive. The root cause is that standard self-attention scales with total token count, and adding registers makes this worse.

## Key Insight

The global role of register tokens and the local nature of patch tokens are structurally separable, allowing a sparse attention pattern that preserves register benefits while reducing cost to linear in the number of patches.

## Method

**Locally-Aware Sparse Register Attention (LASRA)**

(A) **What it is**: LASRA is a sparse self-attention mechanism for pixel-space DiTs that limits each patch token's attention to a local window of patches and all register tokens, while register tokens attend globally. **Load-bearing assumption: patch tokens in pixel-space DiTs only need to attend to a local spatial window of patches (size w×w) plus all register tokens to achieve comparable quality to global attention.** It takes as input patch tokens, register tokens, and conditioning; outputs updated tokens.

(B) **How it works** (pseudocode):
```python
# hyperparameters: window_size w=7, num_registers R=8
# x_patch: (B, H, W, D), x_reg: (B, R, D)
B, H, W, D = x_patch.shape
N = H * W
# flatten patches: x_patch_flat = x_patch.view(B, N, D)
# concatenate: x = cat([x_patch_flat, x_reg], dim=1)  # (B, N+R, D)
# compute Q, K, V with adaptive layer norm (conditioning)
Q = linear(x)  # (B, N+R, D)
K = linear(x)
V = linear(x)

# Efficient implementation: use block-sparse attention with a custom mask
# Pre-compute attention mask for all query positions
mask = torch.zeros(N+R, N+R, dtype=torch.bool)  # True to attend
for i in range(N):
    h = i // W
    w_pos = i % W
    # local window indices (spatial neighbors within H,W, clamped at boundaries)
    for dh in range(-w//2, w//2+1):
        for dw in range(-w//2, w//2+1):
            nh, nw = h+dh, w_pos+dw
            if 0 <= nh < H and 0 <= nw < W:
                j = nh * W + nw
                mask[i, j] = True
    # all register tokens
    for r in range(R):
        mask[i, N+r] = True
for i in range(R):
    # register tokens attend to all tokens
    mask[N+i, :] = True

# Perform masked attention using PyTorch's scaled_dot_product_attention with mask
# Use memory_efficient_backend if available
attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0)

# Output: attn_output contains updated patch and register tokens
```
(C) **Why this design**: We chose local window attention for patches over global attention because patches capture texture details that are spatially local (PixelDiT, 2024), accepting the trade-off that long-range dependencies are only captured via register tokens; this preserves register benefits while reducing per-query keys from O(N+R) to O(w^2+R). We let registers attend to all tokens to maintain their global structural role (Registers Matter), despite a marginal O(R(N+R)) cost, because R is small (e.g., 8) and registers are few; alternative sparsification of registers would risk losing global context. A fixed window size of 7 was chosen over learned dynamic windows to avoid overhead from index computation and to leverage hardware-friendly dense blocks; the hyperparameter w can be tuned per resolution to balance quality and cost. Compared to prior sparse attention in diffusion (e.g., Sparse DiT), our design is explicitly coupled with register tokens, whose existence justifies the asymmetric pattern—a domain expert would not describe LASRA as a variant of generic sparse attention because its pattern is driven by register-patch separation rather than generic sparsity heuristics.

(D) **Why it measures what we claim**: The computational quantity "per-query number of attended keys" (w^2 + R for patches, N+R for registers) measures reduction in attention complexity from O((N+R)^2) to O(N(w^2+R) + R(N+R)) ≈ O(N) for constant w,R, which operationalizes "computational scalability" because total FLOPs are linear in N, under the assumption that sparse attention can be efficiently implemented on GPUs (e.g., using block-sparse MatMul with tiling). Failure mode: if block-sparse operations are not hardware-optimized (e.g., on A100, dense matmul uses Tensor Cores while sparse masks fall back to slower routines), the actual speedup may be limited; we verify with wall-clock timings. The quantity "register attention includes all patches" measures preservation of register benefits because registers need to aggregate global information to guide structure; this assumes registers operate as global descriptors, which fails if registers degenerate to local features (countered by their design in Darcet et al.), in which case attending all patches over-expresses spurious correlations.

## Contribution

(1) Locally-Aware Sparse Register Attention (LASRA), a sparse self-attention mechanism for pixel-space DiTs that asymmetricly handles patch and register tokens to reduce computational cost while maintaining register benefits. (2) A design principle that separates global and local token roles in diffusion transformers, enabling scalable high-resolution generation without sacrificing quality. (3) An analysis of the trade-off between window size and generation fidelity under fixed register count.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet 256×256 | Standard benchmark for class-conditional generation |
| Primary metric | FID | Measures distributional fidelity and mode coverage |
| Baseline 1 | DiT (no registers, global attn) | Tests need for registers in pixel-space DiT |
| Baseline 2 | DiT + registers (full attn) | Tests overhead of full global attention |
| Baseline 3 | PixelDiT (global attn, pixel-space) | Tests pixel-space baseline without sparsity |
| Ablation 1 | LASRA w/o registers | Tests role of registers within our sparse pattern |
| Ablation 2 | LASRA with registers attending only locally (window size 7) | Tests importance of global register attention |
| Additional baseline | Sparse DiT (Child et al., 2019) with sparse factorisation | Tests generic sparse attention in same setting |

### Why this setup validates the claim
This setup isolates the two key innovations: sparse attention and register tokens. By comparing against DiT (no registers) and PixelDiT, we test whether registers improve quality in pixel-space DiTs. The ablation (LASRA w/o registers) directly measures the contribution of registers to our method's performance. The full attention + registers baseline (DiT+registers) establishes the upper bound on quality and the computational cost we aim to reduce. FID is chosen because it captures both fidelity and diversity, which are essential for image generation. If our method matches DiT+registers in FID while requiring fewer FLOPs, the claim of efficient quality is validated. Additionally, we include an ablation where registers attend only locally to test whether the global role of registers is necessary, and a baseline using existing sparse attention (Sparse DiT) to contextualize our novelty.

### Expected outcome and causal chain

**vs. DiT (no registers, global attention)** — On images with complex global structure (e.g., multiple objects), DiT without registers produces artifacts like missing parts because local patch tokens cannot capture long-range dependencies. Our method uses register tokens that aggregate global information, so on such cases it generates coherent structures. We expect a noticeable FID gap (e.g., 0.5–1.0 points) favoring LASRA over DiT, especially on categories requiring global consistency.

**vs. DiT + registers (full attention)** — On high-resolution images (e.g., 256×256 with 65536 patches), full attention has O(N²) cost, leading to slower sampling. Our method reduces per-query keys to w²+R, so on such large inputs we achieve lower latency while maintaining similar quality. We expect LASRA to match DiT+registers in FID (within 0.1 points) but with significantly fewer FLOPs (measured in TeraFLOPs per step) and faster wall-clock time (measured in seconds per batch on an NVIDIA A100).

**vs. PixelDiT (global attention)** — PixelDiT uses global attention without registers, so it may suffer from attention distraction in cluttered scenes (e.g., overlapping objects) where many patches attend to irrelevant ones. Our method's local windows limit noise, and registers capture essential global cues. On such challenging samples, LASRA should achieve better FID (by ∼0.3–0.5) than PixelDiT.

**Ablation: LASRA with registers attending only locally** — If this variant achieves FID worse than full-register LASRA by >0.3, it confirms that registers need global attention. We expect a gap of ~0.5-1.0 FID, justifying the global design.

### What would falsify this idea
If LASRA w/o registers achieves FID comparable to LASRA (within 0.1), then registers are unnecessary. If LASRA has FID worse than DiT+registers by >0.5, then our sparsity pattern degrades quality irrecoverably. If the wall-clock speedup is negligible (<1.2×) compared to full attention, the computational claim is invalid in practice.

## References

1. Registers Matter for Pixel-Space Diffusion Transformers
2. PixelDiT: Pixel Diffusion Transformers for Image Generation
3. Scalable Diffusion Models with Transformers
