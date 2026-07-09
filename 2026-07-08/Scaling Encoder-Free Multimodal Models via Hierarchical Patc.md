# Scaling Encoder-Free Multimodal Models via Hierarchical Patch Aggregation with Reconstructability Constraint

## Motivation

Existing encoder-free multimodal architectures (e.g., Gemma 4's 12B model) fail to scale to larger sizes due to quadratic growth in token count and training instability. Larger models (beyond 12B) rely on modality-specific encoders, breaking the architectural consistency. We identify that patches become increasingly redundant at larger scales, enabling a hierarchical aggregation that preserves information via a reconstructability constraint.

## Key Insight

Patch redundancy scales superlinearly with model size, so hierarchical aggregation with a reconstructability objective can compress token count to a fixed budget without information loss.

## Method

### Method: Hierarchical Patch Aggregation with Reconstructability (HPAR)

**Assumption (load-bearing):** Patch redundancy scales superlinearly with model size, such that a fixedbudget compression becomes increasingly lossless. Verified via effective rank of patch features on calibration set (512 examples).

```python
def hpar(patch_sequence, budget, model_scale):
    # patch_sequence: (N, D_patch) where D_patch=64, patches are 32x32 RGB tiles
    # budget: fixed token count (e.g., 256)
    # model_scale: parameter count of LM (used for redundancy calibration)
    
    # Step 0: Redundancy calibration (once on calibration set)
    if first_call:
        eff_rank = compute_effective_rank(patch_features)  # use SVD of feature covariance
        assert eff_rank < 0.8 * N, "Redundancy insufficient for fixed budget"

    # Step 1: Patch feature extraction via small CNN (2 conv layers, 3x3 kernel, stride 2, hidden 128, output 64, ReLU)
    patch_features = small_cnn(patch_sequence)  # (N, 64)
    
    # Step 2: Initial grouping via k-means (n_clusters=budget, n_init=1, max_iter=100)
    clusters = kmeans(patch_features, n_clusters=budget)
    
    # Step 3: Aggregate each cluster via attention pooling (single-head, learnable query vector, scaled dot-product)
    aggregated_tokens = []
    for c in range(budget):
        mask = (clusters == c)
        cluster_feats = patch_features[mask]  # (m_c, 64)
        query = self.query_vector  # learnable (1, 64)
        attn = torch.softmax((query @ cluster_feats.T) / sqrt(64), dim=1)  # (1, m_c)
        agg = attn @ cluster_feats  # (1, 64)
        aggregated_tokens.append(agg)
    aggregated_tokens = torch.stack(aggregated_tokens, dim=0)  # (budget, 64)
    
    # Step 4: Reconstructability constraint via small decoder (2-layer transposed CNN, upsample to 32x32)
    reconstructions = decoder(aggregated_tokens)  # (budget, 3, 32, 32)
    loss = MSE(reconstructions, patch_sequence.reshape(budget, 3, 32, 32))
    
    # Step 5: Iterative refinement (5 iterations, reassign top 10% high-error patches)
    for _ in range(5):
        errors = per_patch_reconstruction_error
        worst = torch.topk(errors, k=int(0.1 * N)).indices
        for idx in worst:
            # Reassign to nearest aggregated token (L2 distance in feature space)
            dists = torch.norm(patch_features[idx] - aggregated_tokens, dim=1)
            clusters[idx] = torch.argmin(dists)
        aggregated_tokens = [attention_pool(cluster) for cluster in clusters]
        loss = MSE(decoder(aggregated_tokens), patch_sequence)
    return aggregated_tokens
```

**Why this design:** We chose hierarchical aggregation over single-level pooling because it adapts to varying patch complexity, accepting additional computational cost. We used a reconstructability constraint instead of a fixed compression ratio to ensure information preservation, trading off compression rate for fidelity. We used k-means for grouping instead of learned grouping to avoid training instability, accepting that grouping may not be optimal. The iterative refinement corrects grouping errors from initial k-means, trading off iteration count for accuracy.

**Why it measures what we claim:** The reconstructability constraint measures information preservation because low reconstruction error implies aggregated tokens contain sufficient information to recover original patches; this assumption fails when the decoder is too weak (e.g., 1-layer decoder can reconstruct trivial patterns), in which case low error may reflect decoder artifacts rather than token content. Additionally, reconstruction error measures pixel-level fidelity, but task-relevant information may be high-level; we assume pixel-level fidelity implies task-level sufficiency, which holds if the decoder's receptive field covers relevant features. Failure mode: a decoder that memorizes a few trivial patterns could achieve low error without preserving semantic content. The aggregation mechanism produces fixed-length tokens by design, directly addressing scaling.

## Contribution

(1) A hierarchical patch aggregation method with reconstructability constraint that compresses a variable number of patches to a fixed token budget. (2) The empirical finding that patch redundancy scales with model size, enabling this approach for large multimodal models. (3) An encoder-free architecture that scales consistently across model sizes without separate encoders.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | VQA v2 (train/val split) | Tests multimodal understanding with complex images |
| Primary metric | VQA accuracy | Directly measures information preservation |
| Baseline | Direct patch sequence (all patches) | Variable-length no compression upper bound |
| Baseline | Fixed-length average pooling (token count = budget) | Fixed length without reconstructability |
| Baseline | HPAR with random grouping (same budget) | Isolates effect of k-means grouping |
| Ablation-of-ours | HPAR without iterative refinement | Tests contribution of refinement |
| Scaling experiment | Compare token count vs accuracy at model sizes 1B, 4B, 8B, 12B | Test scaling efficiency |

### Why this setup validates the claim
This setup validates the claim because VQA v2 requires detailed visual understanding from patch sequences; accuracy is a direct measure of whether aggregated tokens preserve task-relevant information. The direct patch sequence baseline provides an upper bound on achievable accuracy, showing that compression does not severely degrade performance if our method works. The fixed-length average pooling baseline isolates the effect of the reconstructability constraint: if our method outperforms it, that demonstrates the constraint's value. The random grouping baseline tests the importance of k-means clustering. The ablation of iterative refinement tests whether the refinement step is necessary for adaptivity. The scaling experiment tests the load-bearing assumption that redundancy grows with model size: we expect the optimal budget (where accuracy matches direct baseline) to be smaller for larger models. Together, these comparisons form a falsifiable test: if our method matches or exceeds the fixed-length baseline and approaches the direct baseline on complex images, the claim is supported.

### Expected outcome and causal chain

**vs. Direct patch sequence** — On a complex image with many small objects (e.g., a busy street scene), the direct baseline processes all patches and achieves high accuracy (e.g., 72% on VQA). Our method aggregates patches into fewer tokens, potentially losing some details. However, because of the reconstructability constraint that forces preservation of information, we expect to retain most of the critical details. Thus, on such images, our method should achieve accuracy within 1-2% of the direct baseline, while on simple images (e.g., single object) accuracy is similar. The observable signal is a small accuracy gap on the complex subset.

**vs. Fixed-length average pooling** — On a complex scene, fixed-length pooling compresses without regard to importance, so fine details are lost and accuracy drops (e.g., 60% on complex subset). Our method, via k-means grouping and reconstructability, adaptively allocates tokens to informative regions, preserving key features. Therefore, we expect a noticeable accuracy gap (~5-10% on complex subset) between our method and fixed-length pooling, while on simple images both perform similarly.

**vs. HPAR with random grouping** — Random grouping allocates tokens uniformly, ignoring structure. We expect random grouping to perform worse than k-means grouping on complex scenes (e.g., 3-5% lower accuracy), confirming that k-means captures patch groupings that are semantically meaningful.

**Scaling experiment** — As model size increases from 1B to 12B, we expect the optimal budget (where accuracy matches direct baseline) to decrease (e.g., from 512 to 128 tokens), supporting the superlinear redundancy assumption.

### What would falsify this idea
If our method shows no accuracy advantage over fixed-length average pooling on any image subset, or if the ablation without refinement performs equally well as the full method, then the claim that reconstructability and iterative refinement help would be falsified. More specifically, if the gain over fixed-length pooling is uniform across all subsets rather than concentrated on complex images, that would contradict the adaptive aggregation claim. Additionally, if the scaling experiment shows that larger models require more tokens (not fewer) to match direct baseline, the core assumption of superlinear redundancy is falsified.

## References

1. Gemma 4 Technical Report
2. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
3. Generalizing Verifiable Instruction Following
4. Qwen2.5 Technical Report
5. TÜLU 3: Pushing Frontiers in Open Language Model Post-Training
