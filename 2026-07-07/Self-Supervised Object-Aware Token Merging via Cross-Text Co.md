# Self-Supervised Object-Aware Token Merging via Cross-Text Consistency

## Motivation

Existing token merging methods for vision-language retrieval, such as SaMer, require object annotations during training to prevent cross-instance mixing, limiting applicability to domains lacking such supervision. The root cause is the reliance on external object boundaries that cannot be derived from raw image-text pairs alone.

## Key Insight

Token relevance profiles across multiple captions of the same image correlate with object membership because different captions describe the same objects in varied language, causing tokens from the same object to exhibit consistent response patterns.

## Method

(A) **CrosSight** (Cross-Text Consistency Token Merging) is a training-free method that produces object-aware merge priors for vision-language retrieval without any object annotations. It takes an image and its associated captions as input and outputs a set of merged centroids that preserve object evidence. (B) **How it works** (pseudocode):

```python
# Input: image tokens V (N x d), list of K captions C (each with variable length)
# Output: merged centroids (K x d)

profiles = []
for v_i in V:
    p_i = []
    for caption_j in C:
        T_j = text_encoder(caption_j)  # (M_j x d)
        # compute max dot product between v_i and text tokens of caption_j
        s = max(v_i @ T_j.T)  # scalar
        p_i.append(s)
    profiles.append(p_i)  # K-dim vector
profiles = stack(profiles)  # (N x K)
# Normalize profiles
profiles = normalize(profiles, dim=1)

# Cluster tokens into K groups based on profile similarity
labels = kmeans(profiles, n_clusters=K)  # using cosine similarity

# Merge tokens within each cluster
centroids = []
for l in range(K):
    mask = (labels == l)
    centroid = mean(V[mask], dim=0)
    centroids.append(centroid)
centroids = stack(centroids)  # (K x d)
```

(C) **Why this design**: We chose average relevance max over text tokens to obtain a scalar per caption because it is robust to caption length and focuses on the best-matching word, trading off some fine-grained signal for simplicity. We opted for k-means clustering on profile vectors over alternative methods like spectral clustering because it scales linearly with N and K, accepting the cost of assuming convex clusters. The choice of K equal to the number of captions avoids hyperparameter tuning but may produce suboptimal granularity; we accept this because it ties the merge granularity to the available caption diversity. We specifically avoided learning a separate merge prior network (as in DTEM) to maintain zero-shot applicability, though this limits adaptation to domain-specific patterns. (D) **Why it measures what we claim**: The relevance score `s = max(v_i @ T_j.T)` measures the fitness of token i to caption j under the assumption that the highest-scoring text token captures the object described; this assumption fails when captions have high lexical overlap or when a token matches a distractor word, in which case s reflects accidental similarity rather than object relevance. The profile vector p_i measures the token's response pattern across different captions under the assumption that captions vary sufficiently in how they refer to objects; this assumption fails when captions are nearly identical (e.g., in synthetic datasets), where profiles become collinear and clustering degenerates. The k-means clustering on profiles assigns group labels that correspond to object membership under the assumption that intra-object profile similarity exceeds inter-object similarity; this assumption fails when two distinct objects are described synonymously across all captions, causing them to be merged, yielding centroids that mix object evidence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | MSCOCO | Diverse captions per image test cross-text variation. |
| Primary metric | Recall@1 (image→text) | Standard metric for retrieval quality. |
| Baseline 1 | No merging (full tokens) | Upper bound on retrieval performance. |
| Baseline 2 | Visual-only k-means merging | Tests need for cross-text guidance. |
| Baseline 3 | Random token merging | Tests if clustering structure matters. |
| Ablation-of-ours | Profile w/o normalization | Tests importance of profile normalization. |

### Why this setup validates the claim
MSCOCO provides five diverse captions per image, making it ideal to test the central assumption that cross-text consistency reveals object membership. Comparing against no merging sets an accuracy ceiling, while visual-only merging isolates the value of text guidance, and random merging tests whether structured clustering is necessary. The ablation of normalization checks if the profile scaling is critical. Recall@1 directly measures retrieval accuracy; if our method maintains accuracy while reducing tokens, it validates that cross-text merging preserves object evidence. The setup is falsifiable because specific failure patterns (e.g., uniform gain across subsets) would disprove the underlying mechanism.

### Expected outcome and causal chain

**vs. No merging** — On a case with many redundant background tokens, no merging uses all tokens, causing high memory without accuracy benefit. Our method merges such tokens into few centroids, preserving object clusters. We expect minimal R@1 drop (<2%) with >50% token reduction.

**vs. Visual-only k-means merging** — On a case with two similarly colored dogs, visual-only merging groups them together because visual tokens are alike. Our method’s profile vectors capture caption differences (e.g., "dog on left" vs "dog on right"), keeping tokens separate. We expect a 5-10% R@1 gap on such images.

**vs. Random token merging** — On a multi-object scene (person, car, tree), random merging mixes object tokens, diluting evidence. Our method clusters by cross-text similarity, preserving distinct object centroids. We expect a 10-15% R@1 advantage on multi-object images.

### What would falsify this idea
If our method’s gain over visual-only merging is uniform across images rather than concentrated on cases with visually similar but differently described objects, then the cross-text consistency mechanism is not functioning as intended. Also, if the ablation without normalization matches the full method, the normalization step is unnecessary.

## References

1. Do All Visual Tokens Matter Equally? Object-Evidence Preserving Token Merging for Vision-Language Retrieval
2. Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings
3. ToSA: Token Merging with Spatial Awareness
4. WARP: An Efficient Engine for Multi-Vector Retrieval
5. Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers
6. SpatialBot: Precise Spatial Understanding with Vision Language Models
7. AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark
8. LLaVA-OneVision: Easy Visual Task Transfer
