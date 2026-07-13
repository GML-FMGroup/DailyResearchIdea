# Lightweight Global Context Module for Efficient Domain-Adaptive Dense Prediction with Frozen Diffusion Transformers

## Motivation

ReChannel (From RGB Generation to Dense Field Readout) achieves efficient dense prediction by freezing a DiT and using per-token linear heads, but its token-local readout cannot capture long-range dependencies and lacks mechanisms for domain adaptation. This structural limitation arises because the linear head operates on individual token features without global aggregation, and the frozen backbone cannot adapt to distribution shifts. Our LGC module injects global context and enables domain adaptation without breaking the frozen backbone or linear complexity.

## Key Insight

A single learned query can aggregate global context from frozen DiT tokens via cross-attention because the tokens are already spatially arranged and encode local patterns; the resulting context vector modulates the linear head via a low-rank affine transformation, enabling domain adaptation with minimal extra parameters because the frozen backbone's generality suffices across domains.

## Method

(A) **Lightweight Global Context (LGC) module**: A plug-in module for ReChannel-style frozen DiT dense predictors. It takes the set of frozen DiT token features (N tokens, each d-dim) and outputs a global context vector that is used to modulate the shared linear head via a low-rank affine transformation. For domain adaptation, it accepts a domain-specific context query and additionally applies a per-domain feature modulation (a learnable affine transformation on frozen tokens) to mitigate the risk that domain shift alters internal token representations. (B) **How it works**:
```python
def LGC_module(frozen_tokens, domain_id=None):
    # frozen_tokens: shape (N, d) from DiT
    # Learnable global query (shared across domains) and learned domain-specific queries
    # Per-domain feature modulation (smallest repair for load-bearing assumption):
    if domain_id is not None:
        # domain_scale: learnable vector per domain, shape (d,)
        # domain_shift: learnable vector per domain, shape (d,)
        modulated_tokens = frozen_tokens * domain_scale[domain_id] + domain_shift[domain_id]  # shape (N, d)
        query = domain_queries[domain_id]  # shape (1, d), learned per domain, initialized from shared query
    else:
        modulated_tokens = frozen_tokens
        query = shared_global_query  # shape (1, d), learned
    # Cross-attention to aggregate tokens
    attn_weights = softmax(query @ modulated_tokens.T / sqrt(d))  # attention over tokens
    global_context = attn_weights @ modulated_tokens  # shape (1, d)
    # Low-rank affine modulation for linear head
    # Linear head originally: output = frozen_tokens @ W_head + b_head (W_head: d x k, b_head: k)
    # We learn low-rank factors A (d x r), B (r x k), and a bias delta (k)
    # Apply affine transformation to global context and add to head params
    modulation = global_context @ A @ B  # shape (1, k)
    # Adjusted linear head for each token: output_token = token @ W_head + b_head + modulation
    outputs = modulated_tokens @ W_head + b_head + modulation  # (N, k)
    return outputs
```
Hyperparameters: r=8 (low-rank rank), d=1024 (DiT feature dim), shared_global_query dimension d, domain_queries count = number of training domains, domain_scale and domain_shift each of dimension d per domain. (C) **Why this design**: We chose cross-attention over global pooling (e.g., average pooling) because pooling loses token-level spatial information critical for tasks like segmentation; cross-attention learns which tokens are relevant for the global context, preserving detailed interactions. We chose a low-rank affine modulation over a fully separate head per domain because the latter would multiply the parameter count linearly with domains and violate parameter efficiency; low-rank (r=8) adds only 2rd + k parameters per domain (context query plus A,B), which is far less than retraining the head (d*k). We chose to learn domain-specific queries rather than fine-tuning the backbone because fine-tuning risks catastrophic forgetting of the generative prior, while a new query only requires ~d parameters per domain and training is stable. The per-domain feature modulation (scale and shift) is added to adjust token features for domain shift without unfreezing the backbone, addressing the load-bearing assumption that domain shift only affects output statistics; this modulation adds 2d parameters per domain, still parameter-efficient. The trade-off is that the global context is limited to a single vector, which may not capture highly multi-modal global structure; for tasks like panoptic segmentation with many object categories, a multi-query extension (e.g., 8 queries) could be used at the cost of linear increase in parameters. (D) **Why it measures what we claim**: The cross-attention output `global_context` measures global context relevance because it is a weighted average of all token features, where weights are assigned by the learned query based on similarity to each token. This assumes that the query can learn to attend to tokens that encode global structure (e.g., sky regions, large objects) while ignoring local noise; this assumption fails when the DiT tokens themselves do not contain enough spatial coherence (e.g., severe occlusions or overlapping objects), in which case the attention may be misaligned. The attention weight sum over a region of interest (e.g., a large object) measures global context relevance; we assume large objects correspond to global context. This assumption fails when large objects are background clutter (e.g., a large wall), in which case attention may be dominated by irrelevant features. The low-rank affine modulation `modulation` measures domain adaptation because it adjusts the linear head's output distribution as a function of the global context; this assumes that domain shift primarily affects the output statistics (e.g., depth scale, label distribution) rather than the internal token representations; if the domain shift alters the DiT features (e.g., image style changes), the modulation cannot compensate and full backbone adaptation would be needed. The per-domain feature modulation (scale+shift) is designed to compensate for such shifts in token features while keeping the backbone frozen.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | ADE20K semantic segmentation | Diverse scenes, tests global context aggregation. |
| Additional tasks | NYUv2 depth estimation, ScanNet surface normal | Broaden significance across dense tasks. |
| Additional domain shift | GTA5→Cityscapes (synthetic to real) | Test domain adaptation on style shift. |
| Primary metric | mean Intersection over Union (mIoU) for seg; RMSE for depth; mean angle error for normals | Standard metrics for each task. |
| Baseline 1 | ReChannel (frozen DiT + linear head) | Direct comparison without our modulation. |
| Baseline 2 | VPD (diffusion-based dense predictor) | Representative of backbone-modifying methods. |
| Baseline 3 | Set Transformer (with frozen DiT tokens as set inputs) | Compare to other attention-based aggregation. |
| Ablation-of-ours | LGC with shared query only | Isolates effect of domain-specific queries. |
| Ablation-of-ours | LGC without per-domain feature modulation | Tests necessity of token-level adaptation. |
| Quantitative analysis | Spearman correlation between region attention sum and ground-truth object area | Validates global context claim. |

### Why this setup validates the claim

ADE20K's semantic segmentation task requires dense pixel-wise predictions in complex scenes with multiple object categories. Our method claims that cross-attention aggregation of frozen DiT tokens into a global context vector, followed by low-rank modulation of the linear head, improves accuracy by capturing global structure and enabling efficient domain adaptation. Using ReChannel as a baseline directly tests whether the modulation adds value over a frozen head. VPD tests against a method that fine-tunes the backbone, so if ours matches or exceeds it, we support parameter efficiency. The ablation with only a shared query tests the necessity of domain-specific queries for adaptation. The Set Transformer baseline tests whether our simpler cross-attention design is as effective as a more complex set aggregation. The additional tasks (depth, normals) and domain shift (GTA5→Cityscapes) test generalization across tasks and domains. The correlation analysis operationalizes the global context claim: we compute for each image the attention weight sum over the region of each large object (area > 10% of image) and correlate with the object's actual area; a positive correlation indicates the query attends to large objects, supporting the global context interpretation. mIoU, RMSE, and mean angle error are sensitive to both coarse and fine errors, making them ideal for detecting improvements from global context and domain adaptation.

### Expected outcome and causal chain

**vs. ReChannel** – On a case with a cluttered kitchen scene (many small objects like utensils, cabinets), ReChannel's average pooling dilutes local token information, leading to confusion between semantically distinct but visually similar regions (e.g., sink vs. counter). Our cross-attention mechanism learns to assign higher weights to tokens representing large surfaces (e.g., countertop) and lower weights to noisy details, producing a global context that correctly biases the linear head. We expect a noticeable gap (e.g., +3-5 mIoU) on highly varied scenes but near parity on simple uniform scenes (e.g., a solid wall).

**vs. VPD** – On a night-time image from a different domain (not in training set), VPD fine-tunes the backbone, risking catastrophic forgetting of the generative prior and producing lower-quality features for any semantic category. Our method keeps the backbone frozen, only learning a domain-specific query and low-rank modulation, which adjusts the output distribution without altering feature representations. We expect our method to maintain stable mIoU (within 1-2% of daytime performance) while VPD drops significantly (5-10% mIoU) due to domain shift.

**vs. Set Transformer** – On a scene with many small objects (e.g., a crowded street), Set Transformer's full self-attention among tokens may capture complex interdependencies but incurs O(N^2) cost. Our LGC uses a single query, which is O(N) and may be less expressive. We expect Set Transformer to slightly outperform on such scenes (e.g., +1-2 mIoU) but at a higher computational cost; our method will be preferred for efficiency. The correlation analysis should show that our query's attention weights positively correlate with object area (Spearman ρ > 0.3), whereas Set Transformer's induced attention is harder to interpret.

### What would falsify this idea

If our gain over ReChannel is uniform across all scene types instead of concentrated on cluttered scenes where global context is critical, or if VPD outperforms our method on the target domain despite our adaptation, then the central claim—that cross-attention global context and parameter-efficient modulation are effective—would be invalidated. Additionally, if the correlation between attention weight sum and object area is zero or negative, the global context claim would be unsupported.

## References

1. From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models
2. Edit2Perceive: Image Editing Diffusion Models Are Strong Dense Perceivers
3. Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction
