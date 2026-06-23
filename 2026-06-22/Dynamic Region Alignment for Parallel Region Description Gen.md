# Dynamic Region Alignment for Parallel Region Description Generation

## Motivation

Existing parallel region description generation models (e.g., PerceptionDLM) rely on fixed-resolution vision encoders that resize images to a standard resolution, causing information loss for non-standard aspect ratios. This structural limitation is inherited from the standard ViT backbone used in these models, which discards the dynamic-resolution benefits shown in recent vision-language models like Qwen2.5-VL.

## Key Insight

The geometric fidelity of region features is maintained because the attention mechanism uses relative positions normalized by region size, making it invariant to region aspect ratio.

## Method

(A) **What it is**: Dynamic Region Alignment (DRA) is a lightweight attention module that takes as input a dynamic-resolution image feature map and a set of region bounding boxes, and outputs region-aligned features that preserve aspect ratio information. Inputs: dynamic-resolution feature map F (H' x W' x D) and region boxes B (n x 4). Outputs: region tokens R (n x d).

(B) **How it works** (pseudocode):
```python
# Step 1: Extract dynamic-resolution features
F = dynamic_res_encoder(I)  # shape: (H', W', D), where H'≈H/16, W'≈W/16
# Step 2: RoIAlign with fixed output size (7x7)
F_regions = RoIAlign(F, B, output_size=7)  # shape: (n, 7, 7, D)
# Step 3: Flatten and project
region_tokens = MLP(flatten(F_regions))  # shape: (n, d), hidden_dim=1024, d=768
# Step 4: Region-aware cross-attention with scaled relative position bias
# For each region bi with center (cx,cy) and size (rw,rh), compute bias:
for each patch at (i,j) in F:
    dx = (i - cx) / rw, dy = (j - cy) / rh
    pos_bias[bi, i, j] = MLP([dx, dy])  # MLP with hidden_dim=128, output single scalar
# Multi-head attention (8 heads): queries=region_tokens, keys=Wk*F, values=Wv*F
attn_logits = Q @ K^T / sqrt(d) + pos_bias
R = softmax(attn_logits) @ V  # shape: (n, d)
# Step 5: R is fed into the diffusion language model for parallel generation
```

(C) **Why this design**: We chose RoIAlign over fixed crop+resize because RoIAlign preserves spatial correspondences by using bilinear interpolation, avoiding misalignment from resizing to a fixed square. We chose relative position bias scaled by region size over absolute position encoding because it allows the attention to adapt to varying region aspect ratios: a larger region should have a broader attention focus. We chose to extract region features before attention rather than attending directly to whole image with region queries because the initial RoI extract provides a strong local prior, reducing the attention's need to search globally, which is more efficient. The trade-off is that the RoI extraction may miss context outside the region; we accept this under the assumption that local region details are most important for description generation, and the subsequent attention can incorporate broader context if needed via region-aware attention. This design avoids the fixed-resolution bottleneck of prior work (e.g., PerceptionDLM) by using dynamic-resolution encoding from Qwen2.5-VL, ensuring no aspect-ratio distortion at the feature level.

(D) **Why it measures what we claim**: The computational quantity `relative position bias scaled by region size` measures aspect-ratio-aware spatial alignment because it ensures that the attention distribution is proportional to the region's geometry: for a tall narrow region, the bias encourages attending to distant patches along the height dimension while staying narrow in width. This assumption fails when the region contains objects that heavily depend on far-away context (e.g., a small object with large background), in which case the scaled bias may overly restrict attention and miss global context. The computational quantity `RoIAlign with fixed output size` measures region feature uniformity because it normalizes all regions to the same spatial resolution, enabling batch processing and fixed-size input to downstream layers; this assumption fails when regions have extreme aspect ratios (e.g., very thin or very wide), causing severe interpolation artifacts, in which case the feature downsampling distorts shape-sensitive details.

## Contribution

(1) Dynamic Region Alignment (DRA) module that integrates dynamic-resolution image encoders with region-aware attention for parallel region description generation. (2) The design insight that aspect-ratio-aware positional biases in cross-attention preserve geometric fidelity without requiring fixed-resolution resizing. (3) An open-source implementation for reproducing DRA on top of PerceptionDLM.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | ParaDLC-Bench | Parallel region captioning with varied aspect ratios |
| Primary metric | CIDEr | Measures caption quality, sensitive to detail |
| Baseline | PerceptionDLM | Fixed-resolution region features, absolute positions |
| Baseline | AbsolutePosAttn | RoIAlign + standard cross-attention (no scaled bias) |
| Ablation-of-ours | NoRegionAttn | RoIAlign+MLP only, no cross-attention |

### Why this setup validates the claim
ParaDLC-Bench contains regions with diverse aspect ratios, directly testing the central claim that aspect-ratio-aware alignment improves caption quality. CIDEr captures n-gram overlap and is standard for captioning. Comparing against PerceptionDLM isolates the effect of dynamic resolution and scaled bias; AbsolutePosAttn isolates the scaled bias component; the NoRegionAttn ablation tests whether the cross-attention refinement is necessary. If our method's gains are concentrated on extreme aspect ratio subsets, the claim is supported; uniform gains across all regions would suggest a different mechanism.

### Expected outcome and causal chain

**vs. PerceptionDLM** — On a tall narrow region (e.g., a standing person), PerceptionDLM resizes to a fixed square, distorting aspect ratio, so it produces a generic caption like "a person" because spatial cues are lost. Our method preserves the aspect ratio via dynamic-resolution features and scaled bias that focuses attention vertically, capturing the elongated shape. We expect a noticeable CIDEr gap on tall or wide regions, but near-parity on square regions.

**vs. AbsolutePosAttn** — On a wide region (e.g., a landscape), AbsolutePosAttn uses absolute positions that do not adapt to region size, so attention spread is uniform regardless of region extent, missing global context. Our scaled relative bias spreads attention proportionally to region width, capturing the wide context. We expect our method to outperform specifically on wide regions, while on small compact regions performance is similar.

### What would falsify this idea
If our method's CIDEr gains are uniform across all aspect ratio subsets rather than concentrated on extreme aspect ratio regions, or if the NoRegionAttn ablation matches our full method, then the central claim that aspect-ratio-aware alignment drives improvement would be falsified.

## References

1. PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models
2. SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation
3. LLaDA2.0: Scaling Up Diffusion Language Models to 100B
4. Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning
5. Scaling Diffusion Language Models via Adaptation from Autoregressive Models
6. Code Llama: Open Foundation Models for Code
7. Auto-Regressive Next-Token Predictors are Universal Learners
8. Graph of Thoughts: Solving Elaborate Problems with Large Language Models
