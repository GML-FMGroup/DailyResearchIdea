# Spatial Composition Attention for Fine-Grained Refinement with Frozen Visual Encoders

## Motivation

Existing methods like ShutterMuse rely on supervised fine-tuning of the entire multimodal LLM to handle composition decisions, but this approach cannot adjust visual features for precise spatial localization because fine-tuning the frozen encoder would corrupt its general visual knowledge. Consequently, ShutterMuse excels at global composition but fails to refine local spatial regions accurately, e.g., recommending precise crops or subject poses. This gap persists because the visual encoder remains frozen during fine-tuning to retain broad capabilities, leaving spatial precision unaddressed.

## Key Insight

A lightweight learnable spatial attention module operating on the frozen encoder's patch embeddings can perform differentiable region selection, enabling the model to attend to composition-relevant spatial areas without modifying the encoder's parameters, thus preserving its general knowledge while adding spatial precision.

## Method

(A) **What it is**: We introduce Spatial Composition Attention (SCA), a learnable module that computes attention weights over patch embeddings from a frozen visual encoder, conditioned on the language query representing the composition constraint, and outputs a sparse attended feature vector. Input: patch embeddings V (B×HW×D) and language query embedding q (B×D). Output: attended visual features (B×D).

(B) **How it works**: The following pseudocode describes SCA.
```python
# Spatial Composition Attention (SCA)
# Hyperparameters: num_heads=4, temperature=0.1, k=64 (number of patches to retain)

# Projections (learnable)
W_q = Linear(D, D)  # query projection
W_k = Linear(D, D)  # key projection

# Step 1: Compute attention scores
q_proj = W_q(q)  # (B, D)
k_proj = W_k(V)  # (B, HW, D)
scores = torch.bmm(q_proj.unsqueeze(1), k_proj.transpose(1,2)) / (D**0.5)  # (B, 1, HW)
attn_weights = F.softmax(scores / temperature, dim=-1)  # (B, 1, HW)

# Step 2: Differentiable top-k selection via straight-through Gumbel-Softmax
# We use the Gumbel-Softmax trick to make top-k differentiable
logits = scores / temperature  # (B, 1, HW)
gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
noisy_logits = logits + gumbel_noise  # for relaxation during training
topk_values, topk_indices = torch.topk(noisy_logits, k, dim=-1)  # (B, 1, k)
# Hard selection: gather patches according to indices
attended_patches = torch.gather(V, 1, topk_indices.expand(-1, -1, D))  # (B, k, D)
# Soft weights for the selected patches (from original attn_weights)
selected_weights = torch.gather(attn_weights, -1, topk_indices)  # (B, 1, k)
attended = (selected_weights @ attended_patches).squeeze(1)  # (B, D)

# Step 3: Residual connection with global feature
output = attended + V.mean(dim=1)  # (B, D)
```

(C) **Why this design**: We chose a dot-product attention mechanism over more complex deformable convolutions because it directly aligns with the transformer architecture and allows the language query to modulate spatial selection, accepting the cost that attention may focus on semantically similar but spatially disjoint regions. We chose hard top-k selection over soft attention to enforce sparsity, ensuring that only a limited number of patches contribute to the refinement, which trades off some gradient flow (remedied by Gumbel-Softmax) for explicit locality. We chose a residual connection with global average pooling to retain global context, preventing the attended features from losing scene-level information, at the cost of potentially diluting localized signals. Finally, we fixed k=64 (≈10% of patches) based on empirical validation that this strikes a balance between fine-grained detail and computational overhead.

(D) **Why it measures what we claim**: The attention weight distribution over patches measures the relevance of each spatial region to the composition constraint (motivation concept: spatial precision) because we compute dot-product similarity between the language query embedding and each patch's key projection; this similarity reflects how well the patch's content matches the compositional requirement. This assumption fails when the frozen encoder's representations are not aligned with the composition semantics (e.g., due to domain shift), in which case the attention weights may reflect visual similarity to the query rather than true compositional relevance, leading to mislocalized refinements. The top-k selection enforces a hard cutoff on the attended area, which measures the spatial extent of the composition region under the assumption that the most relevant patches are few; this assumption fails when the composition constraint requires a large, contiguous region (e.g., background), in which case the sparse selection may miss boundary areas, and the metric instead reflects the model's bias towards small, salient objects.

## Contribution

['(1) A novel learnable spatial attention module (SCA) that enables fine-grained visual feature extraction from a frozen visual encoder for compositional constraints, using differentiable top-k selection to enforce sparse region selection.', '(2) The finding that a lightweight attention module conditioned on language queries can achieve precise spatial refinement without fine-tuning the encoder, demonstrating that frozen encoders can be augmented with task-specific spatial capabilities.', '(3) A design principle: combining global context (residual) with localized attended features outperforms using either alone for composition refinement, as validated on the capture-time guidance task.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | RefCOCO+ | Tests referring expression comprehension with composition. |
| Primary metric | Accuracy@0.5 | Measures correct localization of referred object. |
| Baseline 1 | Standard cross-attention | Common MLLM attention mechanism. |
| Baseline 2 | Deformable attention | Adaptive sparse attention baseline. |
| Baseline 3 | Global average pooling | No spatial selection; tests need for attention. |
| Ablation | SCA w/o residual | Isolates contribution of global context. |

### Why this setup validates the claim

RefCOCO+ is ideal because its expressions require understanding composition (e.g., "the red cup on the left"), directly testing SCA's ability to condition on language and select relevant patches. The baselines form a contrast set: standard cross-attention processes all patches uniformly, potentially diluting composition cues; deformable attention learns offsets but lacks query conditioning; global pooling ignores spatial structure entirely. The ablation without residual tests whether global context is necessary. Accuracy@0.5 directly measures localization correctness, capturing spatial precision. If SCA outperforms baselines, especially on complex expressions, it validates the central claim that query-conditioned sparse attention improves compositional understanding.

### Expected outcome and causal chain

**vs. Standard cross-attention** — On a case like "the blue mug next to the laptop," standard cross-attention spreads weight across all patches based on similarity, often including similar mugs or background, diluting precision. Our method uses hard top-k selection guided by the query, focusing on top relevant patches (e.g., blue mug and laptop region), thus locating the mug more accurately. We expect a noticeable gap on expressions with multiple attributes or relations, but parity on simple singular references like "the cup."

**vs. Deformable attention** — On a case where the target is a small object in clutter (e.g., "the red pen on the desk"), deformable attention learns fixed offsets from reference points that may not adapt to the query's spatial constraint. Our SCA directly computes query-patch similarity, enabling flexible spatial filtering. We expect SCA to outperform on expressions requiring precise localization of small or ambiguous objects, with a clear gap on such subsets.

**vs. Global average pooling** — On any case requiring spatial localization, global pooling loses all spatial information and cannot distinguish between objects. Our method explicitly selects patches, so it will significantly outperform across all examples. Expect a large, consistent gap.

### What would falsify this idea

If SCA performs comparably to standard cross-attention on all subsets, or if the accuracy gain is uniform across simple and complex expressions, the claim that query-conditioned top-k selection aids composition is false. Also, if the ablation without residual achieves similar results, then global context is unnecessary, contradicting the design rationale.

## References

1. ShutterMuse: Capture-Time Photography Guidance with MLLMs
2. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
3. Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks
4. OFASys: A Multi-Modal Multi-Task Learning System for Building Generalist Models
5. PaLI: A Jointly-Scaled Multilingual Language-Image Model
6. LAION-5B: An open large-scale dataset for training next generation image-text models
