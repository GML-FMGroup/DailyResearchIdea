# CVP-Net: Compositional Visual Primitives for Open-Vocabulary Multi-Image Attribute Extraction

## Motivation

Existing benchmarks like IndustryBench-MIPU and ImplicitAVE assume a fixed attribute vocabulary, limiting real-world applicability. Current multi-image attribute extraction methods fail to generalize to unseen attribute compositions because they treat each attribute as a closed category rather than decomposing it into reusable primitives that capture invariant visual properties across images.

## Key Insight

Attribute-specific representations are compositional over shared visual primitives, and cross-view consistency across multiple images provides a self-supervised signal to learn these primitives without attribute-level annotations.

## Method

### (A) What it is
CVP-Net learns a set of $M$ visual primitives (learnable vectors) and a differentiable composition function that assembles them into an attribute representation conditioned on a textual query. **Load-bearing assumption:** The cross-view contrastive loss on unlabeled multi-image product sets learns visual primitives that capture invariant attribute-relevant properties across different images of the same product. Input: a set of $K$ images of the same product and a textual attribute query (e.g., "color"). Output: the predicted attribute value string.

### (B) How it works
```python
# Phase 1: Primitive Learning via Cross-View Consistency (self-supervised)
# Input: unlabeled multi-image product sets {I_1,...,I_K} for many products
# Shared CNN backbone (ResNet-50) extracts feature map f_k from image I_k
# Learnable primitives: P = {p_1, ..., p_M} in R^d, M=64, d=512
for each product:
    for each image I_k:
        # Attention over spatial locations i in f_k
        A_k[i][m] = softmax_i( (W*f_k[i]) dot p_m )  # W is projection (d x d)
        v_k[m] = sum_i A_k[i][m] * f_k[i]  # primitive activation vectors
    # Cross-view consistency loss (contrastive)
    L_cons = -log( sum_{m} exp( cos(v_1[m], v_2[m])/tau ) / (sum_{neg} exp(cos(v_1[m], v_neg)/tau) ) )
    # tau=0.1, negatives from different products

# Phase 1.5: Primitive Verification via Linear Probe (supervised probe)
# After Phase 1, freeze primitives P and backbone weights.
# Allocate a small calibration set: 1% of training products with attribute labels.
# For each attribute (e.g., color, material), train a linear classifier (single linear layer)
# that takes as input the mean-pooled primitive activations over all images of a product
# (v_prod = mean_k v_k) and predicts the attribute value.
# Evaluate accuracy on a held-out validation set. If accuracy < 10% over random, 
# the primitives are considered misaligned; then use weak attribute supervision:
# add a cross-entropy loss from the probe to guide primitive learning in Phase 1.
# Otherwise, proceed with original Phase 2.

# Phase 2: Composition Function for Attribute Extraction (supervised)
# Input: product image set, query q (text), ground-truth attribute value y
# Text encoder (BERT) produces q_enc (768-dim)
# Aggregate primitives across images: v_prod[m] = mean_k v_k[m]
# Composition network Comp: MLP with two hidden layers (512->256->512, GeLU activation)
# input = concat( [v_prod[1],...,v_prod[M], q_enc] ) -> r (512-dim)
# Decoder: GRU with attention over r, hidden_dim=512, outputs token sequence y_pred
# Loss: cross-entropy between y_pred and y

# Hyperparameters: M=64, d=512, tau=0.1, learning rate 1e-4, batch_size=32, optimizer=Adam
# Training: Phase 1 for 100 epochs, Phase 2 for 50 epochs (if probe passes else 100+50 joint)
# GPU hours: ~48 on single A100 for both phases
```

### (C) Why this design
We chose to learn primitives via cross-view contrastive loss rather than supervised attribute labels because attribute annotations are expensive and the multi-image setting naturally provides positive pairs. This decision accepts the cost that primitives may not align perfectly with human-interpretable semantics, but it enables zero-shot generalization to unseen attribute queries. We use attention-based primitive extraction instead of global pooling because attention allows primitives to localize relevant image regions, trading off spatial invariance for discriminability across attributes. We condition the composition function on the textual query via concatenation rather than cross-attention because it is simpler and empirically faster; the trade-off is that query-conditioning may lose fine-grained interactions between primitives and query, but we mitigate this with a deep MLP. Finally, we use an autoregressive GRU decoder for open-vocabulary generation rather than a fixed classifier to allow arbitrary attribute values, at the cost of slower inference and exposure bias.

### (D) Why it measures what we claim
The cross-view consistency loss $L_{cons}$ measures **invariance** of primitives across images of the same product because it maximizes agreement between primitives from different views under the assumption that the product identity is the only common factor; this assumption fails when images contain transient occlusions, lighting changes, or backgrounds that also yield high agreement, in which case primitives may capture those irrelevant factors instead of attribute-specific features. The attention weights $A_k[i][m]$ measure **relevance** of a spatial location to primitive $m$ because they are computed via dot-product similarity between the feature and the primitive; this assumption fails when the feature space is not isotropic, in which case the weights may be dominated by magnitude rather than content. The composition function output $r$ measures **attribute-specific representation** because it is a nonlinear combination of primitives selected by the query embedding; this assumption fails when the query is ambiguous or the primitives are not discriminative for that attribute, in which case $r$ may collapse to a generic representation. The decoder's token probabilities measure **attribute value confidence** because they are derived from a well-calibrated softmax over a large vocabulary; this assumption fails for rare attributes, where the decoder may assign high probability to plausible but incorrect tokens due to training set imbalance.

## Contribution

(1) A novel framework for open-vocabulary multi-image attribute extraction that learns reusable visual primitives via cross-view consistency without attribute-level annotations. (2) A differentiable composition function that combines primitives conditioned on a textual query, enabling generalization to unseen attribute compositions. (3) Empirical demonstration that primitives learned from multi-image consistency capture invariant properties that transfer across attributes, validated on a variant of IndustryBench-MIPU adapted for open-vocabulary evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | IndustryBench-MIPU | Standard multi-image attribute benchmark. |
| Primary metric | Exact Match | Measures exact attribute value correctness. |
| Baseline 1 | Single-Image CLIP | Tests multi-image vs single-image benefit. |
| Baseline 2 | Avg-Pool Multi-Image CLIP | Tests primitive learning vs average pooling. |
| Baseline 3 | Supervised ResNet-50 | Tests self-supervised primitive pre-training. |
| Ablation of ours | CVP-Net w/o consistency loss | Isolates impact of cross-view consistency. |
| Additional verification | Linear probe accuracy on frozen primitives | Tests alignment with attribute semantics; threshold 10% above random. |

### Why this setup validates the claim

The experimental design directly tests the central claim that cross-view consistency learning of visual primitives and query-conditioned composition improve attribute extraction over alternatives. Using IndustryBench-MIPU, which contains multiple images per product and diverse attributes, allows evaluation of multi-image reasoning. Single-Image CLIP baseline isolates the multi-image advantage; if our method outperforms it, multi-image consistency is beneficial. Avg-Pool Multi-Image CLIP baseline tests whether primitive learning provides better aggregation than simple average pooling of global features. Supervised ResNet-50 baseline tests the advantage of self-supervised pre-training of primitives versus full supervision from attribute labels. The ablation removes the consistency loss to assess its contribution. The additional linear probe verification directly tests the load-bearing assumption that primitives capture semantic attribute properties; if probe accuracy is low (e.g., <10% above random), it would indicate primitives are learning shortcuts like background or lighting, thereby falsifying the core assumption. Exact Match metric directly measures the correctness of predicted attribute values, which is the task objective. This combination ensures falsifiability: if our method does not beat these baselines under expected failure modes, the core mechanisms are unsupported.

### Expected outcome and causal chain

**vs. Single-Image CLIP** — On a product with varying lighting across images, single-image CLIP may misclassify color due to shadow; it only sees one view and cannot correct. Our method aggregates primitive activations across images, learning an invariant color primitive via cross-view consistency. Therefore, we expect a noticeable gap on color attribute accuracy (e.g., 10-15% higher) but parity on attributes like material that are less view-dependent.

**vs. Avg-Pool Multi-Image CLIP** — On a product with multiple textures (e.g., patterned shirt), average pooling blends features, losing discriminative spatial detail. Our attention-based primitives localize relevant regions per primitive, preserving texture attributes. Thus, we expect our method to outperform on attributes involving local patterns (e.g., pattern type) by a larger margin (e.g., 20%) while similar on global attributes.

**vs. Supervised ResNet-50** — On a rare attribute value unseen in supervised training, the supervised model predicts the nearest seen class; it cannot generalize. Our primitives are learned without attribute labels and composed via query-conditioned decoder, enabling zero-shot generation. Hence, we expect our method to achieve non-zero exact match on rare attributes whereas supervised baseline fails completely.

### What would falsify this idea

If our method fails to outperform baselines on attributes where multi-image consistency is crucial (e.g., color under varying illumination) or where primitive localization is needed (e.g., fine-grained patterns), then the central claim that cross-view primitive learning improves multi-image attribute extraction is invalid. Additionally, if the linear probe accuracy on frozen primitives is low (e.g., <10% above random), it would indicate that primitives are not learning attribute-relevant features, directly falsifying the load-bearing assumption.

## References

1. IndustryBench-MIPU: Benchmarking Multi-Image Attribute Value Extraction for Industrial Products
2. VideoAVE: A Multi-Attribute Video-to-Text Attribute Value Extraction Dataset and Benchmark Models
3. MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models
4. ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction
5. MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark
6. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
