# GradCache: Motion-Adaptive Cache Invalidation for Real-Time Causal Streaming Video Editing

## Motivation

Current streaming video editing methods like LiveEdit reuse cached features across frames under the assumption of temporal smoothness, causing temporal coherence breakdown when fast motion invalidates the cached content. This structural limitation arises because cache validity is binary and motion-blind, leading to feature mismatch and artifacts in dynamic regions. By treating cached features as parameters that can be updated via gradient descent on a temporal coherence loss, we can adaptively decide when to recompute based on a direct motion signal.

## Key Insight

The gradient magnitude of a temporal coherence loss with respect to cached features quantifies the local motion-induced change, providing a principled and efficient signal for cache invalidation.

## Method

# GradCache

(A) **What it is:** GradCache is a motion-adaptive cache invalidation module for causal streaming video editing. It takes as input the cached feature map F_{t-1} from the previous frame and the current frame's intermediate representation (e.g., noisy latent z_t in a diffusion model). It outputs an invalidation mask M_t indicating which spatial regions of the cache should be recomputed.

(B) **How it works:**

```python
# GradCache pseudocode
# Input: cached features F_{t-1} (C x H x W), current latent z_t, optical flow warp function W
# Hyperparameters: threshold τ (default 0.05), learning rate α (default 1.0), confidence threshold γ (default 0.5)
# Note: all ops are spatial (per-pixel) except where noted

# Step 1: Warp cached features to current frame using optical flow u_{t->t-1} (estimated via RAFT with default settings)
F_warpped = warp(F_{t-1}, u_{t->t-1})   # shape (C x H x W)

# Step 2: Compute temporal coherence loss L = MSE(F_warpped, F_t_initial)
# F_t_initial is a lightweight feature from early layers of the editing model (e.g., from a skip connection)
L = mean((F_warpped - F_t_initial)**2, axis=0)   # shape (H x W) per-pixel loss

# Step 3: Compute gradient of L w.r.t. cached features F_{t-1} (using chain rule through warp)
grad = ∇_{F_{t-1}} L   # shape (C x H x W) — spatial gradient magnitude per channel

# Step 4: Aggregate gradient magnitude per pixel
G = sqrt(sum(grad**2, axis=0))   # shape (H x W)

# Step 5: Obtain flow confidence map C from RAFT (output of the flow network, range [0,1])
C = flow_confidence(u_{t->t-1})   # shape (H x W), higher is more reliable

# Step 6: Combine gradient threshold and confidence to decide invalidation
M_t = ((G > τ) | (C < γ)).float()   # 1 means recompute, 0 means reuse

# Step 7: Apply cache update: reuse or recompute
F_t = M_t * recompute(z_t, region) + (1 - M_t) * F_warpped
```

(C) **Why this design:** We chose gradient magnitude over simpler motion magnitude (e.g., optical flow magnitude) because gradient magnitude directly measures how much the cached features contribute to the temporal coherence error, capturing texture and content changes beyond pure motion. We warp cached features rather than recomputing from scratch to get a reference, accepting the cost that flow estimation adds computation (but it is lightweight and amortized across frames). We use per-pixel thresholding instead of global threshold to adapt to spatially varying motion, though this increases memory for the mask. We set τ as a hyperparameter rather than learning it to avoid overfitting, trading optimality for simplicity. We compute gradients only on a lightweight feature (early layer) rather than the full model output to reduce computation, accepting that early features may miss high-level semantic changes. We additionally incorporate a flow confidence mask to guard against warping failures, forcing recomputation in regions where flow is unreliable.

(D) **Why it measures what we claim:** The per-pixel gradient magnitude G measures the **motion-induced change** in cached features because the temporal coherence loss L directly penalizes the discrepancy between warped past and current features; a high gradient indicates that the cached feature, if reused, would cause a large inconsistency. This assumption holds when the warping is accurate (i.e., flow captures pixel motion); when flow is inaccurate (e.g., occlusions, rapid appearance changes), G may instead reflect warping error rather than true motion, leading to false recomputation (conservative behavior). We mitigate this by also requiring a minimum flow confidence (γ=0.5) for a pixel to be considered valid; low-confidence pixels are always recomputed. The threshold τ operationalizes the **cache validity** concept: regions where G > τ are considered invalid because the cost of mismatch exceeds the tolerance; this mapping is valid under the assumption that temporal coherence error is monotonic in feature mismatch, which holds for smooth loss landscapes in neural features. Furthermore, using MSE on early-feature space to measure perceptual coherence relies on the assumption that early feature space is approximately Euclidean with respect to perceptual similarity. This is reasonable for low-level texture and motion but may miss high-level semantic changes (e.g., object identity) where early features are invariant; in such cases, G may remain low despite semantic inconsistency, leading to false cache reuse. We accept this failure mode as a conservative bias (reuse when uncertain about semantics) and leave it as future work.

## Contribution

(1) A motion-adaptive cache invalidation mechanism that uses gradient magnitude of a temporal coherence loss to decide when to recompute features in streaming video editing. (2) A principle that the gradient of a warped-feature loss directly quantifies the need for cache update, enabling adaptive trade-off between speed and temporal consistency. (3) Integration into a causal streaming pipeline building on LiveEdit, with no additional training required.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DAVIS-2017 (streaming subset) | Classic video segmentation benchmark adapted for editing |
| Primary metric | Temporal Coherence (LPIPS) | Per-pixel perceptual consistency between consecutive frames |
| Baseline 1 | LiveEdit | State-of-the-art realtime streaming editing method |
| Baseline 2 | Flow magnitude cache | Uses optical flow magnitude for cache invalidation |
| Baseline 3 | Global threshold cache | Single threshold for all pixels |
| Baseline 4 | Attention-based cache | Uses attention map difference instead of gradient magnitude |
| Ablation 1 | GradCache w/o warp | Directly compare features without optical flow |
| Ablation 2 | GradCache w/o confidence | Removes flow confidence masking |

### Why this setup validates the claim

The chosen dataset provides diverse motion patterns (camera pan, object motion, occlusion) to stress-test motion adaptation. Comparing against LiveEdit (full streaming method) tests whether gradient-based caching matches or exceeds a dedicated realtime system. Flow magnitude cache isolates the innovation of using gradient over pure motion. Global threshold cache tests the per-pixel adaptation claim. Attention-based cache tests whether the gradient measure is specifically beneficial compared to another feature-change signal. The ablation (no warp) verifies the necessity of warping cached features. The ablation (no confidence) isolates the contribution of flow confidence masking. Temporal Coherence (LPIPS) directly measures the claimed benefit: smooth frame-to-frame consistency. Together, these baselines and metrics form a falsifiable test: if GradCache’s advantage is solely from motion adaptation, it should outperform flow magnitude and global threshold on high-motion scenes but be comparable on static scenes; if warping is critical, the ablation should underperform. We also plan to analyze warping failure rates by measuring how often flow confidence is below γ and how gradient magnitude behaves in those regions, providing groundedness on when the assumption holds.

### Expected outcome and causal chain

**vs. LiveEdit** — On a scene with rapid camera pan and a moving object, LiveEdit’s global recomputation or simpler caching may introduce flicker because it does not selectively invalidate. Our method uses per-pixel gradient magnitude to detect regions where warped past features poorly match current features, recomputing only those. Thus, we expect lower LPIPS on high-motion frames (e.g., >0.1 improvement) while matching LiveEdit on static frames (within 0.01).

**vs. Flow magnitude cache** — On a static textured object partially occluded, flow magnitude may flag occlusion-boundary pixels as high motion, triggering wasteful recomputation. Our gradient measure focuses on actual feature mismatch, reducing false positives. Consequently, we expect similar LPIPS (within 0.02) but significantly fewer recomputed pixels (e.g., 30% less) on occlusion-heavy scenes.

**vs. Global threshold cache** — On a scene with moving foreground and static background, global threshold either over-recomputes static regions or under-recomputes motion regions, causing artifacts or wasted compute. Our per-pixel threshold adapts to local motion, so we expect equal LPIPS on moving regions (within 0.01) and lower recomputation on static regions (e.g., 50% less) compared to a global threshold tuned for overall quality.

**vs. Attention-based cache** — On a scene with moving texture but static semantics (e.g., water ripples), attention maps may change slowly while gradient magnitude on early features captures texture change, leading to better temporal coherence. We expect at least 0.05 lower LPIPS on such scenes. Conversely, on scenes with semantic changes (e.g., object appearing), attention may detect the change earlier; we expect comparable performance. This validates that gradient magnitude is specifically beneficial for motion-induced texture change.

**Ablation: GradCache w/o warp** — Without warping, the temporal coherence loss directly compares features at the same spatial locations without motion compensation. This will cause high mismatch on moving objects, leading to near-global recomputation. We expect LPIPS degradation of at least 0.05 on moving scenes, confirming the necessity of warping.

**Ablation: GradCache w/o confidence** — Without confidence mask, in occlusion regions the gradient signal may be dominated by warping errors, leading to false recomputation (conservative) or false reuse (if threshold too low). We expect a slight increase in recomputation rate (e.g., 10%) but similar LPIPS on average. Analysis of warping failures: we will measure the fraction of pixels with C<γ across the dataset and report gradient magnitude statistics in those regions. We expect that in low-confidence regions, gradient magnitude is high due to warping error (not true motion), confirming the need for confidence masking. This analysis provides groundedness for the load-bearing assumption.

### What would falsify this idea

If GradCache’s LPIPS gain over flow magnitude cache is uniform across all frames (no concentration in high-gradient regions), then our gradient measure is not capturing motion-induced changes but merely reflecting random noise or warp inaccuracies. If the confidence mask does not improve performance (e.g., recomputation rate unchanged), then the load-bearing assumption about warping accuracy may be unproblematic in practice, weakening the claimed need for mitigation.

## References

1. LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing
2. EgoEdit: Dataset, Real-Time Streaming Model, and Benchmark for Egocentric Video Editing
3. Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset
4. StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation
5. OminiControl: Minimal and Universal Control for Diffusion Transformer
6. Timestep Embedding Tells: It’s Time to Cache for Video Diffusion Model
7. xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism
8. Kosmos-G: Generating Images in Context with Multimodal Large Language Models
