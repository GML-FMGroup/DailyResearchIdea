# COAC: Coordinate-Aware Conditioning for Unified Multimodal Layout Control

## Motivation

Existing unified multimodal models (e.g., SenseNova-Vision) achieve broad task coverage via instruction tuning but lose fine-grained coordinate-level layout controllability, because they replace explicit spatial guidance with generic language prompts. ConsistCompose showed the benefit of layout-specific conditioning, yet its approach is not integrated into a unified generative framework that handles diverse tasks without task-specific heads. COAC addresses this by injecting learnable coordinate maps directly into the visual encoder's self-attention, preserving both task breadth and precise spatial control.

## Key Insight

Coordinate maps provide a dense, spatially-aligned representation that can be seamlessly fused with visual features via attention biases, enabling the model to condition on arbitrary layouts without architectural modification, because the bias term operates on the same token grid as self-attention.

## Method

### (A) What it is
COAC (Coordinate-Aware Conditioning) is a module that transforms a layout specification (e.g., bounding boxes with category labels) into a per-pixel coordinate map and integrates it into the self-attention of a unified multimodal model via learned additive biases. Input: image tokens $X \in \mathbb{R}^{H_f W_f \times d}$ and layout $L = \{(b_i, c_i)\}$ (bounding boxes and class labels). Output: layout-conditioned visual features $X' \in \mathbb{R}^{H_f W_f \times d}$.

### (B) How it works
```python
# Given visual tokens X (reshaped to H_f x W_f x d) and layout L
# Step 1: Render layout into per-pixel coordinate channels
coordinate_map = zeros(H_f, W_f, 2)  # (dx, dy) relative to nearest object center
for each (bbox, class_id) in L:
    # Compute object center (cx, cy) in feature grid coordinates
    cx, cy = bbox_to_grid_center(bbox, H_f, W_f)  # H_f, W_f are feature map dimensions
    # For each pixel (i,j), compute offset to (cx, cy) normalized by feature stride
    for i in range(H_f):
        for j in range(W_f):
            coordinate_map[i,j,0] = (j - cx) / W_f   # normalized horizontal offset
            coordinate_map[i,j,1] = (i - cy) / H_f   # normalized vertical offset
# Step 2: Encode coordinate map with a small MLP (2 hidden layers, 64 dim each)
coord_features = MLP(coordinate_map)  # output: H_f x W_f x d
# Step 3: Compute attention bias from coordinate features
# Use a learned linear projection to get a scalar bias per token pair
query = X.view(H_f*W_f, d)
key = X.view(H_f*W_f, d)
coord_bias = coord_features.view(H_f*W_f, d).mm(coord_features.view(H_f*W_f, d).T)  # H_f*W_f x H_f*W_f
coord_bias = Linear(1)(coord_bias.unsqueeze(-1)).squeeze(-1)  # scalar per pair
# Step 4: Add bias to standard self-attention logits
attention_weights = softmax(QK^T / sqrt(d) + coord_bias * alpha)  # alpha=0.1 (hyperparameter)
X' = attention_weights @ V
```

### (C) Why this design
We chose a coordinate map with per-pixel normalized offsets over token-level positional embeddings because the map retains fine spatial granularity across all tokens, whereas token embeddings would force a fixed assignment and lose relative layout structure; the cost is additional memory for the map and MLP encoding. We opted for additive attention biases (inspired by ALiBi) rather than cross-attention to a separate layout encoder because biases are parameter-efficient and integrate directly into existing attention without increasing the sequence length; the trade-off is that biases only modulate existing interactions rather than introducing new query-key pairs, which may limit expressiveness for complex layouts. We used a learnable MLP to encode the coordinate map instead of a fixed sinusoidal function because the MLP can adapt to different layout densities and class-specific patterns, but it requires more training data to avoid overfitting to common layouts. Finally, we scaled the bias by a learnable α (initialized 0.1) to control the influence of coordinates; setting α too high might suppress visual content, while too low renders conditioning ineffective.

### (D) Why it measures what we claim
The computational quantities in (B) are the coordinate map encoder, attention bias, and the bias scale α. The coordinate map encoder measures **coordinate-level controllability** because it converts explicit spatial instructions (bounding boxes) into a dense offset field that the model can attend to; this assumption fails when the MLP generalizes poorly to extreme aspect ratios or out-of-distribution layouts, in which case the encoder measures memorized coordinate patterns instead. The attention bias measures **integration of spatial constraints** because it directly modulates attention logits based on coordinate offsets, making certain tokens more likely to attend to layout-relevant regions; this assumption fails when the model learns to ignore the bias for tasks where layout is not informative (e.g., pure classification), in which case the bias measures residual noise but does not harm performance. The scale α measures the **strength of conditioning** because it gates the bias magnitude; this assumption fails when α collapses to 0 during training (e.g., due to large gradients), in which case α measures a degenerate solution that disables layout control. Together, these components ensure that when the model successfully uses layout information, the attention bias is the causal mechanism for spatial control, and we can verify this by ablating α or the coordinate map.

## Contribution

(1) A novel coordinate-aware conditioning module (COAC) that integrates layout specifications into unified multimodal models via learnable coordinate maps and attention biases, requiring no task-specific heads. (2) Empirical demonstration that COAC restores layout controllability in tasks like object detection and image generation while maintaining performance on general vision-language benchmarks, showing that coordinate-level conditioning can coexist with instruction-based versatility. (3) Analysis of design choices for spatial conditioning in unified models, including the trade-off between coordinate map resolution and computational cost.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | COCO detection | Standard layout evaluation. |
| Primary metric | AP@0.5 | Measures spatial localization accuracy. |
| Baseline 1 | Flamingo | No explicit layout conditioning. |
| Baseline 2 | Unified-IO | Text-only layout conditioning. |
| Baseline 3 | Mask R-CNN | Task-specific detector. |
| Ablation | COAC with α=0 | Isolates effect of bias. |

### Why this setup validates the claim

This setup tests the central claim that COAC's coordinate-aware attention biases improve layout controllability. By comparing to Flamingo (no layout conditioning) we test if any conditioning is beneficial. Against Unified-IO (text-only layout) we test the advantage of per-pixel coordinate offsets. Mask R-CNN sets a strong task-specific baseline for detection performance. The ablation with α=0 nullifies the bias, directly measuring its causal contribution. The primary metric AP@0.5 is chosen because it reflects precise bounding box overlap, which requires accurate spatial relationship modeling – the exact capability COAC aims to enhance. Together, these comparisons form a falsifiable test: if COAC truly enables better spatial control, it should outperform baselines specifically on layout-sensitive subsets.

### Expected outcome and causal chain

**vs. Flamingo** — On a case where the layout specifies two overlapping objects (e.g., a cup on a table), Flamingo with no layout conditioning likely misses the overlap because it lacks explicit spatial constraints; its attention may blend features. Our method uses coordinate offsets to separate objects, so attention is biased toward each object's center. We expect our method to achieve significantly higher AP on scenes with object overlap (e.g., +5 AP) while similar on simple scenes.

**vs. Unified-IO** — On a case where the layout requires precise spatial relationships (e.g., "the cat to the left of the dog"), Unified-IO's text-only conditioning fails to capture continuous offsets; it may place objects roughly correct but not pixel-aligned. Our coordinate map provides per-pixel offsets, leading to precise localization. We expect a noticeable gap in AP on directional queries (e.g., +3 AP on left/right subsets).

**vs. Mask R-CNN** — On a case with novel object categories not in its training set, Mask R-CNN fails because it requires class-specific bounding box regression heads. Our method, as a unified model, uses layout conditioning to adapt to new classes via the coordinate map. We expect our AP on rare categories to be higher (e.g., +2 AP) than Mask R-CNN, but lower on common ones due to task specialization.

**vs. COAC with α=0** — On any layout case, disabling the bias removes the spatial bias, so the model reverts to standard attention. We expect AP to drop to near the base model level, confirming that the bias is causal.

### What would falsify this idea

If our method shows uniform improvement across all layout types (including random layouts) or if the ablation α=0 performs similarly, then the central claim (that coordinate bias is the causal mechanism) is wrong. Also if the gain is larger on non-spatial tasks, it suggests the bias is helping for other reasons.

## References

1. Vision as Unified Multimodal Generation
2. Scaling Spatial Intelligence with Multimodal Foundation Models
3. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
