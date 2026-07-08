# Self-Supervised Object-guided Centroid Pooling for Annotation-Free Late-Interaction Retrieval

## Motivation

Existing annotation-free token merging methods like WARP compress visual tokens without object awareness, causing loss of query-relevant object evidence and degrading retrieval accuracy. SaMer achieves object-aware merging but requires costly object annotations during training, limiting applicability to domains without such supervision. The root cause is that prior work either ignores object boundaries or relies on external annotations, failing to leverage the inherent structure of token features to infer object regions.

## Key Insight

Object boundaries in visual token feature space are naturally reflected by discontinuities in feature similarity across spatial neighbors, enabling a self-supervised objectness score from local feature consistency to replace annotation-based object priors.

## Method

**Self-Supervised Object-guided Centroid Pooling (SOCP)**

**(A) What it is:** SOCP computes a self-supervised objectness score per token from local feature consistency, then uses spatially-constrained k-means clustering to group tokens into object-preserving centroids, all without object annotations. Input: post-projector image tokens with spatial positions; Output: K centroid tokens for late-interaction retrieval.

**(B) How it works:**
```python
def socp_merge(tokens, positions, K, temp=1e-4, sigma=0.5, n_neighbors=8):
    # tokens: (N, d), positions: (N, 2)
    sim = cosine_similarity(tokens)  # (N, N)
    sp_dist = spatial_distance(positions)  # (N, N), normalized by image size
    # Compute self-supervised objectness score
    obj = torch.zeros(N)
    for i in range(N):
        neighbors = torch.argsort(sp_dist[i])[:n_neighbors]
        obj[i] = sim[i, neighbors].mean()
    obj = torch.softmax(obj / temp, dim=0)  # normalize
    # Initialize centroids: top-K tokens by objectness, with distance penalty to avoid duplicates
    idxs = torch.topk(obj, K, dim=0).indices
    centroids = tokens[idxs]  # (K, d)
    # Iterative clustering with spatial and objectness weights
    for _ in range(10):
        # Assignment: soft assignment with feature similarity, spatial kernel, and objectness multiplier
        spatial_kernel = torch.exp(-sp_dist / sigma)  # (N, N)
        # For each centroid, we compute weight using the centroid's feature and spatial position
        centroid_spatial = positions[idxs]
        sp_to_centroid = torch.cdist(positions, centroid_spatial)  # (N, K)
        spatial_weight = torch.exp(-sp_to_centroid / sigma)  # (N, K)
        featsim = tokens @ centroids.T  # (N, K)
        weight = featsim * spatial_weight * obj.unsqueeze(1)  # (N, K)
        assignments = torch.softmax(weight / temp, dim=1)  # (N, K)
        # Update centroids as weighted average of tokens
        centroids = (assignments.T @ tokens) / (assignments.sum(0, keepdim=True).T + 1e-8)
        idxs = torch.topk(assignments.sum(0), K).indices  # re-select top-K clusters
    return centroids
```
Hyperparameters: K=64, temp=0.1 for softmax scaling, sigma=0.5 for spatial distance, n_neighbors=8 for objectness.

**(C) Why this design:** We chose spatially-constrained clustering over simple feature-only k-means because feature-only groups often merge spatially disjoint objects with similar appearance (e.g., two cars on opposite sides), violating object integrity; the trade-off is added computation from spatial distances and sensitivity to sigma. We initialized centroids with high-objectness tokens rather than random, because random initialization yields unstable convergence and often splits objects; the cost is dependence on objectness estimation quality. We included objectness scores as a multiplier in assignment weights rather than only as initialization, because this continuously refines centroids toward object interiors, improving boundary accuracy, at the cost of slightly biasing against low-objectness tokens (e.g., object edges or small objects). We used soft assignments during clustering rather than hard k-means, enabling differentiability and smoother centroids, but requiring a temperature hyperparameter.

**(D) Why it measures what we claim:** The objectness score (mean similarity to nearest spatial neighbors) measures the degree to which a token is surrounded by visually similar tokens in space; we assume this corresponds to being inside an object rather than on boundaries or background—this assumption fails when objects have high internal feature variation (e.g., zebra stripes), in which case the score reflects texture diversity instead of objectness. The assignment weight w_ij = sim_ij * exp(-sp_dist_ij / sigma) * obj_j measures the joint likelihood that tokens i and j belong to the same object under feature similarity, spatial proximity, and token j being an object interior; we assume objects are spatially contiguous and feature-coherent—this assumption fails for articulated objects (e.g., a person with spread limbs) where spatial proximity underweights long-range same-object connections, and the weight then reflects local patch similarity rather than global object identity. The iterative centroid update computes the average feature of tokens assigned to each centroid; we assume this average is a representative summary of the object region—this assumption fails when assignments are inaccurate due to ambiguous boundaries, causing the centroid to mix multiple objects or object and background.

## Contribution

(1) A novel self-supervised objectness score for visual tokens based on local feature consistency, enabling annotation-free object-aware token merging. (2) A spatially-constrained clustering method that uses this score to preserve object boundaries during centroid pooling for late-interaction retrieval. (3) A demonstration that object-aware centroid pooling can be achieved without object annotations, bridging the gap between annotation-free and object-aware methods.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Flickr30K | Standard retrieval benchmark with diverse objects. |
| Primary metric | Recall@1 | Measures top-1 retrieval accuracy. |
| Baseline 1 | ColPali (full tokens) | Upper bound without compression. |
| Baseline 2 | RandomPrune (K=64) | Naive compression baseline. |
| Baseline 3 | FeatureKMeans (K=64) | Tests feature-only clustering. |
| Ablation-of-ours | SOCP w/o objectness | Removes objectness score guidance. |

### Why this setup validates the claim

The central claim is that SOCP's object-guided spatial pooling preserves object integrity during token compression, improving retrieval over methods that ignore object boundaries. Flickr30K offers many multi-object scenes where this matters. Recall@1 directly tests whether the compressed representation retains discriminative object-level features. Comparing against ColPali (full tokens) quantifies the cost of compression. RandomPrune shows the naive alternative, while FeatureKMeans isolates the role of spatial constraints. The ablation (without objectness) tests the contribution of the objectness score specifically. If SOCP outperforms both RandomPrune and FeatureKMeans, and the ablation performs worse, it validates that both spatial grouping and objectness are beneficial. The metric is sensitive because a single correct match requires fine-grained object representation.

### Expected outcome and causal chain

**vs. ColPali** — On an image containing two cats and a dog, ColPali uses all patches so retrieval is accurate. SOCP compresses to 64 centroids, which may blend some details but preserves each animal as separate centroids due to spatial separation. We expect SOCP to achieve recall within 1–2% of ColPali, with a small gap concentrated on scenes with many small objects.

**vs. RandomPrune** — On the same multi-object image, RandomPrune discards patches uniformly, likely removing key object parts and mixing tokens across objects. Its retrieved image may show only one animal or a background. SOCP explicitly groups tokens by spatial proximity and objectness, creating distinct centroids per object. We expect a noticeable gap of 5–10% in recall on multi-object queries, with parity on single-object scenes.

**vs. FeatureKMeans** — On an image with two red cars on opposite sides of the street, FeatureKMeans clusters by feature similarity alone, merging both cars into one centroid because of similar color. This loses the distinction between the two objects, leading to a wrong match. SOCP's spatial weight keeps them separate. We expect SOCP to outperform FeatureKMeans by 3–7% on queries requiring discriminating between spatially separated but visually similar objects.

### What would falsify this idea
If SOCP's recall is no better than RandomPrune across  all query types, or if the improvement is uniform across single-object and multi-object queries rather than concentrated on multi-object cases, then the assumptions about object-preservation are false.

## References

1. Do All Visual Tokens Matter Equally? Object-Evidence Preserving Token Merging for Vision-Language Retrieval
2. Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings
3. ToSA: Token Merging with Spatial Awareness
4. WARP: An Efficient Engine for Multi-Vector Retrieval
5. Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers
6. SpatialBot: Precise Spatial Understanding with Vision Language Models
7. AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark
8. LLaVA-OneVision: Easy Visual Task Transfer
