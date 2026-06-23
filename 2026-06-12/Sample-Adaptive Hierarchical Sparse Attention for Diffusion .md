# Sample-Adaptive Hierarchical Sparse Attention for Diffusion Transformers via Local Concentration

## Motivation

Existing hierarchical sparse attention for diffusion transformers (e.g., LLSA) uses a fixed token selection hierarchy that is agnostic to per-sample complexity, leading to either wasted computation on samples with uniform structure or loss of important tokens on samples with highly variable saliency. The local concentration module from L²ViT produces attention distributions sharply peaked around informative tokens, providing a natural per-token saliency score. However, this module has only been used to enhance linear attention, not to guide hierarchical sparsity patterns.

## Key Insight

When attention distributions are concentrated on a subset of tokens, the cumulative sum of their scores provides a monotonic measure of information coverage that can be thresholded to obtain sample-adaptive token budgets at each hierarchy level.

## Method

We propose **Scalable Adaptive Hierarchical Sparse Attention (SAHSA)**.

### (A) What it is
SAHSA is a drop-in replacement for the attention mechanism in diffusion transformers. It takes as input the query, key, and value tensors (shape B×N×D) and outputs an aggregated representation. The core idea is to compute a per-token saliency score via a learned local concentration module, then use those scores to dynamically construct a hierarchical token selection process that adapts the sparsity pattern to each sample.

### (B) How it works
```python
import torch
import torch.nn.functional as F

def sahsa(q, k, v, saliency_mlp, hierarchy_levels=3, tau=0.1, rho=0.8):
    B, N, D = q.shape
    # Step 1: Learned local concentration (replacing fixed convolution)
    # saliency_mlp: 2-layer MLP, hidden=256, GeLU activation, output 1 scalar per token
    # Input: q (B,N,D) -> output: BxNx1 -> squeeze to BxN
    score = torch.sigmoid(saliency_mlp(q)).squeeze(-1)  # B x N
    # Step 2: Sort tokens by score descending
    scores_sorted, indices = torch.sort(score, dim=1, descending=True)
    # Step 3: Compute cumulative saliency
    cum_scores = torch.cumsum(scores_sorted, dim=1)
    # Step 4: Determine per-level token counts via thresholds
    level_bounds = [int(N * (1 - (1 - rho)**(l+1))) for l in range(hierarchy_levels)]
    # Step 5: Hierarchical Top-K selection (like LLSA but adaptive)
    outputs = []
    q = q / (D**0.5)
    for l in range(hierarchy_levels):
        # Use cumulative threshold to select K_l = argmin(|cum_scores - threshold_l|)
        K_l = min(torch.searchsorted(cum_scores, 
                                     torch.tensor(level_bounds[l]/N).to(q.device)), N)
        topk_indices = indices[:, :K_l]  # sample-adaptive K_l
        gathered_k = torch.gather(k, 1, topk_indices.unsqueeze(-1).expand(-1,-1,D))
        gathered_v = torch.gather(v, 1, topk_indices.unsqueeze(-1).expand(-1,-1,D))
        attn_weights = F.softmax(q @ gathered_k.transpose(1,2) * tau, dim=-1)
        out = attn_weights @ gathered_v
        outputs.append(out)
    # Step 6: Combine (average) outputs from all levels
    combined = torch.stack(outputs).mean(dim=0)
    return combined
```
Hyperparameters: `saliency_mlp` is a 2-layer MLP with hidden size 256 and GeLU activation, trained end-to-end with the diffusion objective; `hierarchy_levels=3`, `tau=0.1` (temperature), `rho=0.8` (base retention rate).

### (C) Why this design
We chose to derive token budgets from the cumulative saliency distribution rather than fixing the budgets per level (as in LLSA) because the cumulative distribution naturally concentrates on informative regions, enabling sample-adaptive budgets without a learned controller. This avoids the overhead and brittleness of a separate routing network (Anti-pattern 4). We used a lightweight learned MLP-based saliency module (inspired by L²ViT but now trainable) instead of a fixed convolution because the static score may be noisy for diffusion models; the learned module can adapt to varying timesteps and noise levels. The trade-off is increased parameter count (negligible, ~2% of total); we accept this cost because the MLP is small and trained end-to-end, ensuring accurate saliency. We also chose a geometric decay for the base thresholds (rho) to reflect that early levels capture coarse structure while later levels refine details. This design avoids the complexity of tuning per-level budgets manually and ensures the method is plug-and-play into existing diffusion transformers.

### (D) Why it measures what we claim
The learned saliency score measures **per-token importance for denoising** because it is trained end-to-end with the diffusion objective via backpropagation through the attention mechanism; this assumption fails when the MLP overfits to specific patterns, in which case the score may not generalize across timesteps. The cumulative saliency bound (threshold position) measures **the proportion of saliency retained** under the assumption that tokens with highest scores are most informative for attention; this assumption fails when the saliency scores are uniform (e.g., early steps), resulting in fixed budgets (degenerating to fixed hierarchy). The hierarchical averaging in Step 6 measures **multi-granularity context** by assuming that each level captures a different scale of dependency; this assumption fails when there is no scale separation, in which case all levels produce similar outputs and the extra computation is wasted. Each computational quantity is thus causally linked to the motivation-level concept of sample-adaptive sparsity, with explicit failure modes that bound the scope.

## Contribution

(1) A sample-adaptive hierarchical sparse attention mechanism that uses local concentration scores to dynamically determine token budgets per hierarchy level, eliminating the fixed hierarchy limitation of existing sparse diffusion transformers. (2) A design principle that the cumulative saliency distribution from a lightweight module can serve as a principled guide for hierarchical sparsity, unifying two previously separate stages (saliency map and token selection). (3) A reduction in algorithmic complexity by avoiding a separate controller or per-head pattern mining, making the method directly applicable to any diffusion transformer with minimal modification.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | ImageNet 256x256 | Common benchmark for diffusion |
| Primary metric | FID | Standard for generation quality |
| Baseline: L^2ViT | L^2ViT | Efficient attention baseline |
| Baseline: LLSA | LLSA | Sparse attention baseline |
| Baseline: Hourglass DiT | Hourglass DiT | SOTA diffusion transformer |
| Ablation: SAHSA no hierarchy | SAHSA w/o hier. avg | Isolates hierarchy benefit |
| Ablation: Saliency ground-truth correlation | Pearson r between saliency score and gradient-based token importance (mean absolute gradient w.r.t. output) | Validates monotonicity assumption |

### Why this setup validates the claim
This setup directly tests SAHSA's central claim: sample-adaptive hierarchical sparsity improves generation efficiency and quality over fixed or uniform sparsity. ImageNet offers diverse images with varying complexity, making it an ideal dataset to expose adaptivity benefits. FID captures both fidelity and diversity, so improved sample-adaptive attention should lower FID, especially on images with heterogeneous regions. L^2ViT tests whether the local saliency module alone suffices; LLSA tests whether adaptive budgets improve over fixed ones; Hourglass DiT tests whether hierarchical selection outperforms spatial downscaling. The ablation isolates the hierarchy component. The correlation ablation validates the core assumption that learned saliency correlates with gradient-based importance. If SAHSA's gains are concentrated where adaptivity matters (e.g., diverse scenes), the logic is confirmed.

### Expected outcome and causal chain

**vs. L^2ViT** — On a landscape image with fine grass detail and large uniform sky, L^2ViT's linear attention blurs grass because it averages all tokens. Our method uses learned saliency to retain high-information tokens (grass edges) while downplaying sky, preserving detail. We expect SAHSA to yield a ~0.5–1.0 FID improvement on images with high local variance, and parity on uniform ones.

**vs. LLSA** — On an image with a small salient object (e.g., a bird) against a plain background, LLSA with fixed token budgets at each hierarchy level may allocate many tokens to background, wasting capacity. Our method adaptively selects only the bird's tokens at fine levels, capturing its detail. We expect SAHSA to show ~0.3–0.8 FID improvement on scenes with sparse salient objects, and similar performance on uniformly textured images.

**vs. Hourglass DiT** — On a face image requiring both global symmetry (eyes alignment) and local texture (skin pores), Hourglass DiT loses spatial resolution in downsampling, causing blurry fine details. Our method keeps high-resolution tokens for salient regions while hierarchical levels capture structure. We expect SAHSA to achieve ~0.2–0.5 FID improvement on high-resolution generation tasks with fine details, and parity on low-frequency images.

### What would falsify this idea
If the FID gain is uniform across all image subsets (e.g., equally effective on uniform and complex scenes), then adaptivity is not the cause. If the ablation without hierarchy matches full SAHSA performance, hierarchy is unnecessary.

## References

1. The Linear Attention Resurrection in Vision Transformer
2. Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers
3. Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers
4. MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
5. PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis
6. eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers
7. Scalable Diffusion Models with Transformers
