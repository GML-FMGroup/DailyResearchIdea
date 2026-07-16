# Noise-Robust Perceptual Hashing for Training-Free Memory Deduplication in Egocentric Video

## Motivation

Existing training-free memory deduplication methods (e.g., EgoMemo in Vinci2) rely on a single perceptual hash to filter repetitive frames, but this is brittle to noise and lighting changes that are common in egocentric video. A single hash can cause false positives (storing noisy duplicates) or false negatives (forgetting rare events due to hash collisions), leading to inefficient memory usage or loss of important context. The structural limitation is that no single hash function can reliably distinguish semantically identical but visually perturbed frames without training, and the field's trajectory toward unbounded memory indexing makes this robustness gap a bottleneck for proactive assistance in very long videos.

## Key Insight

Multiple uncorrelated perceptual hash functions, combined with a temporal consistency check, provide a noise-invariant signature because the probability that all hashes simultaneously fail due to noise is exponentially small, while true scene transitions affect all hashes consistently.

## Method

### (A) What it is:
We propose MultiHashDedup, a training-free memory deduplication module for egocentric video. Input: a stream of frames. Output: a subset of frames to store in memory.

### (B) How it works (pseudocode):
```python
# Hyperparameters:
# K = 3 (number of hash functions: aHash, dHash, pHash)
# T = 5 (initial frequency cap threshold)
# D = 0.9 (cosine similarity threshold for temporal consistency, computed on MoCo v2 features)
# Decay_factor = 0.9 (per-frame decay of bucket counts)
# backbone = MoCo v2 ResNet-18 (output 256-dim features, frozen, no gradients)

counts = {}  # bucket_key -> count
last_stored_frame = None

for each frame f:
    # Step 1: Compute composite key
    hashes = [hash_i(f) for hash_i in [aHash, dHash, pHash]]
    bucket_key = tuple(hashes)
    
    # Step 2: Check frequency cap
    if bucket_key in counts:
        if counts[bucket_key] >= T:
            skip f
            continue
    else:
        # New bucket: perform temporal consistency check using deep features
        if last_stored_frame is not None:
            feat_f = backbone(f)  # 256-dim vector
            feat_last = backbone(last_stored_frame)
            sim = cosine_similarity(feat_f, feat_last)
            if sim > D:
                skip f  # Likely noise, do not create new bucket
                continue
    
    # Step 3: Store frame and update counts
    store f
    counts[bucket_key] = counts.get(bucket_key, 0) + 1
    last_stored_frame = f
    
    # Step 4: Decay all counts (for very long videos)
    for key in counts:
        counts[key] = max(1, int(counts[key] * Decay_factor))
```

### (C) Why this design:
We chose K=3 hash functions (aHash, dHash, pHash) over a single hash to reduce noise-induced false new buckets; the trade-off is a 3× increase in per-frame computation, but hashing is fast and on-device acceptable. The load-bearing assumption is that these three perceptual hash functions produce sufficiently independent outputs under the noise and lighting variations typical in egocentric video, such that the probability of all three simultaneously failing due to noise is negligible. We introduced a temporal consistency check using deep feature similarity (MoCo v2) instead of pixel difference to more robustly distinguish between semantic scene changes and transient perturbations (e.g., motion blur, illumination flicker) that affect all hashes similarly. The trade-off is a small overhead for feature extraction (~10 ms per frame on GPU), but it avoids false positive storage under structured noise. We added a decay factor for bucket counts to prevent memory saturation in very long videos; the trade-off is that periodic repetitions (e.g., stirring every 5 minutes) may be forgotten after a long gap, but this aligns with the goal of prioritizing rare events over long-run statistical patterns.

### (D) Why it measures what we claim:
The composite bucket key (K hashes) measures perceptual similarity with high precision under the assumption that the hash functions are independent and natural noise (e.g., sensor noise, slight motion) affects each differently; this assumption fails under structured noise (e.g., global illumination change or motion blur affecting all hashes similarly), in which case the composite key may produce false collisions (new buckets for same scene). The temporal consistency check using deep feature similarity measures visual content similarity as a proxy for semantic novelty, assuming that high feature similarity (>0.9) indicates the same scene; this assumption fails under rapid motion with changing viewpoint, causing false positive storage (storing frames that are semantically the same but visually different), but the deep features are more robust to pixel-level changes than raw pixel difference. The decaying count operationalizes forgetting of repetitive content by exponentially reducing old bucket counts; the assumption is that older repetitions are less relevant for proactive assistance, which fails for periodic tasks where old occurrences provide context for future interventions, but the decay is slow enough (factor 0.9) to retain recent repetitions.

## Contribution

(1) A training-free memory deduplication module (MultiHashDedup) that combines multiple perceptual hash functions with temporal consistency and decaying frequency caps to robustly filter redundant frames in egocentric video. (2) The design principle that noise robustness can be achieved without training by exploiting the independence of multiple hash functions and temporal continuity, validated through empirical analysis on egocentric video benchmarks. (3) A parameterized framework for evaluating deduplication quality in terms of memory savings and rare event retention, enabling systematic comparison of deduplication strategies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Ego4D | Large-scale egocentric benchmark for diverse activities |
| Primary metric | Proactive task accuracy (next-action prediction) | Measures semantic preservation in memory for downstream use |
| Baseline 1 | Uniform subsampling (every 30 frames) | Simple frame selection baseline without hashing |
| Baseline 2 | Single-hash dedup (aHash only) | Tests incremental benefit of multiple hashes |
| Ablation 1 (Ours) | No temporal consistency (skip deep feature check) | Ablation: omit feature-based gate, rely solely on hash |
| Ablation 2 (Ours) | Pixel-diff temporal consistency (replace deep features with mean_abs_diff, D=10) | Ablation: compare feature-based consistency to pixel-diff |
| Synthetic validation | Controlled noise dataset (Ego4D frames with synthetic Gaussian noise, motion blur, illumination shift) | Quantify hash collision probability under various distortions |

### Why this setup validates the claim
This experimental design directly tests the central claim that MultiHashDedup selectively stores semantically distinct frames without harming downstream task performance. Ego4D provides diverse egocentric activities with varying motion and noise levels. Proactive task accuracy (next-action prediction) assesses whether deduplicated memory retains task-relevant content. Uniform subsampling serves as a naive baseline; single-hash dedup isolates the effect of composite hashing. Ablation 1 (no temporal consistency) reveals the role of deep feature verification in filtering noise, while Ablation 2 (pixel-diff) compares the robustness of feature-based vs. pixel-based consistency. The synthetic validation provides a controlled test of the hash independence assumption: by measuring the probability that all three hashes change simultaneously under known distortions, we can falsify the assumption if collision rates exceed a threshold (e.g., >10% for mild noise). Together, these comparisons create a falsifiable test: if our method improves accuracy over baselines at lower memory usage, and if synthetic collision rates are low (<5% under typical egocentric noise), the claim is supported. Conversely, if accuracy drops on structured-noise clips or collision rates are high, the assumptions about hash independence and feature robustness are invalid.

### Expected outcome and causal chain

**vs. Uniform subsampling** — On a case of rapid cooking actions, uniform subsampling stores frames at fixed intervals, often missing key moments (e.g., adding ingredient) or storing redundant frames during idle stirring. Our method instead selects frames only when hashes change sufficiently and the deep feature similarity is low, capturing the ingredient addition while skipping the stir. We expect a substantial improvement in task accuracy (e.g., +15%) on dynamic clips, with no significant difference on static clips.

**vs. Single-hash dedup** — On a case with light flicker, single aHash produces many false new buckets, flooding memory with near-identical frames. Our composite hash (aHash + dHash + pHash) plus feature verification reduces false positives. We expect our method to achieve comparable accuracy to single-hash (within 2%) but with 30-40% lower memory usage on noisy clips.

**vs. Ablation 1 (No temporal consistency)** — On a case of global illumination change (e.g., turning on a light), hash values change simultaneously for all three functions, causing false new buckets. Without temporal consistency, our method stores many near-identical frames, wasting memory. With deep feature verification, such frames are skipped because features remain similar. Thus our full method will have lower memory footprint and similar accuracy on illumination-change clips; the ablation will show inflated memory and possibly degraded accuracy due to memory saturation.

**vs. Ablation 2 (Pixel-diff temporal consistency)** — On a case of fast camera motion (e.g., walking), pixel difference is high even for the same scene, causing the pixel-diff gate to miss true scene changes or allow false positives. Deep features are more robust, so our full method maintains accuracy while the pixel-diff variant may either store too many frames (false negatives) or miss genuine transitions (false positives).

**Synthetic validation** — We generate controlled distortions (Gaussian noise σ=0.1, motion blur radius=3, illumination shift +20) on a set of 1,000 frames from Ego4D. For each frame, we compute the three hashes and measure whether all three produce a binary output different from the original (i.e., a change). The expected probability of simultaneous change is low (<5%) for mild noise; if it exceeds 10%, the independence assumption is questionable.

### What would falsify this idea
If our method shows no accuracy advantage over uniform subsampling on any subset, or if its memory savings are uniform across all clips rather than concentrated on noisy and repetitive scenes, then the core assumptions about hash independence and feature-based verification are invalid. Additionally, if synthetic collision rates exceed 10% under typical noise, the hash independence assumption is falsified and the method's robustness claims collapse.

## References

1. Vinci2: Providing Proactive Assistance in Continuous Egocentric Videos
2. Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video
3. Sensible Agent: A Framework for Unobtrusive Interaction with Proactive AR Agents
4. StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding
5. VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format
6. MiniCPM-V: A GPT-4V Level MLLM on Your Phone
7. VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation
8. EgoPlan-Bench2: A Benchmark for Multimodal Large Language Model Planning in Real-World Scenarios
