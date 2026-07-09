# Dynamic Dependency Graph Decoding for Adaptive Parallel Dense Video Captioning

## Motivation

Existing parallel decoding methods for dense video captioning, such as Parallelized Autoregressive Decoding (PAD), assume that cross-event dependencies are uniformly weak to enable parallelization. However, many videos contain highly interleaved events with strong dependencies, causing these methods to degrade in quality. The root cause is the fixed decoding order that does not adapt to varying dependency strengths across events.

## Key Insight

The strength of cross-event dependencies can be reliably estimated from learned event representations, enabling a decoding schedule that preserves sequential order for tightly coupled events while maintaining efficiency gains for loosely coupled ones.

## Method

We propose **Dynamic Dependency Graph (DDG) Decoding**, a framework that adaptively switches between parallel and sequential decoding based on estimated inter-event dependency strength. 

### (A) What it is
DDG takes video features and learned event boundaries from a planner (as in PAD) and outputs a decoding schedule that groups events into sequentially or parallelly decoded clusters.

### (B) How it works
```python
# Input: Video V, planner produces event boundaries and representations E = {e1,...,eN}
# Hyperparameters: threshold τ (learned via validation), hidden size d=128

# Step 1: Dependency Estimation (2-layer MLP with hidden 128, GeLU)
for each pair (i,j):
    d_ij = sigmoid(MLP(concat(e_i, e_j)))  # dependency score in [0,1]

# Step 2: Graph Construction
G = complete graph with edge weights d_ij
Remove edges with weight <= τ → G' becomes a graph with only strong edges

# Step 3: Event Grouping
connected_components = find_connected_components(G')  # events linked by strong edges form groups
# Within a group, events are decoded sequentially (causal mask)
# Groups are decoded in parallel (no cross-group mask)

# Step 4: Decoding Schedule
schedule = []
# Process groups in topological order if any inter-group dependencies (none by construction)
for group in connected_components:
    schedule.append({
        'events': group,
        'mode': 'sequential' if len(group)>1 else 'parallel'  # single event is parallel
    })
# Output: schedule with per-group decoding mode
```

**Training Pseudocode**:
```python
# For each batch:
# 1. Compute event representations from planner (frozen during dependency training)
# 2. For each pair (i,j) with temporal overlap >0.5 (IoU), label y_ij=1 else 0
# 3. Loss = binary cross-entropy(d_ij, y_ij) averaged over all pairs
# 4. Update MLP parameters
```
The dependency estimation module is trained end-to-end with the captioning loss using ground-truth dependency labels derived from temporal overlap (IoU > 0.5) between event intervals in the training set. The threshold τ is calibrated on a held-out validation set (10% of training data) to maximize F1 score between estimated dependency and ground-truth temporal overlap.

### (C) Why this design
We chose a pairwise MLP for dependency estimation over attention-based methods because it is lightweight and captures pairwise relations directly, though it cannot model higher-order interactions—a trade-off we accept for simplicity. Using a hard threshold τ rather than a soft gating mechanism ensures a deterministic decoding schedule, avoiding the complexity of learning a router that could be unstable, at the cost of requiring careful hyperparameter tuning on validation data. Grouping via connected components is a straightforward graph algorithm that guarantees event partitions are internally consistent (all pairs strong), but it may produce large groups when the graph is dense, reducing parallelism; we accept this as a conservative choice that preserves quality when dependencies are strong. The dependency estimation module is trained end-to-end with the captioning loss using ground-truth dependency labels derived from temporal overlap (IoU > 0.5) between event intervals in the training set—this provides supervision for the model to learn which events are coupled, though it may mislabel semantically dependent events with low temporal overlap. Overall, we prioritized simplicity and reproducibility over more complex learned scheduling, ensuring the method remains grounded in interpretable graph operations. 

**Explicit assumption**: We assume temporal overlap (IoU > 0.5) is a sufficient proxy for caption dependency. This assumption may fail for semantically dependent non-overlapping events (e.g., cause-effect separated in time). We validate this assumption via precision/recall analysis on a held-out calibration set and report failure cases.

### (D) Why it measures what we claim
The dependency score d_ij measures the strength of cross-event coupling because it is trained to predict whether events i and j are temporally overlapping (a proxy for joint mention in captions); this assumption fails when two events are semantically dependent but do not overlap temporally (e.g., cause-effect separated in time), in which case d_ij under-estimates true dependency and may mistakenly parallelize strongly coupled events. **Equivalence**: d_ij measures temporal overlap Y; assumption A: temporal overlap is a sufficient indicator of caption dependency; failure mode F: semantically dependent events without overlap are mis-estimated as weak. The binary threshold τ determines the operational definition of “strong dependency”: a fixed value ensures consistency across videos but may not adapt to varying scales of dependency scores; if scores are poorly calibrated, some strong edges may be incorrectly treated as weak. The connected-components grouping operationalizes the concept of “tight coupling” by ensuring that all events within a group are mutually strongly connected; this assumption fails when the graph is not transitive—three events may have pairwise strong edges but not all be directly coupled, leading to an over-grouping that unnecessarily sequentializes events. Despite these failure modes, the empirical risk of mis-estimation is minimized by end-to-end training, and the threshold τ can be tuned on a validation set to balance recall and precision of parallelism. We also calibrate on a held-out set (10% of training data) and report precision and recall of dependency estimation, including qualitative examples where temporal overlap fails.

### (E) Verification of Dependency Estimation
On the held-out calibration set, we compute precision and recall of the binary dependency prediction (using threshold τ) against ground-truth temporal overlap. We also report failure cases where temporal overlap is low but captions are semantically related (e.g., event 'pour milk' and 'drink' with no overlap). These analyses ensure the assumption is transparent and highlight limitations.

## Contribution

(1) A dynamic dependency graph decoding framework that adaptively switches between parallel and sequential decoding based on estimated inter-event dependency strength, enabling flexible efficiency-quality trade-offs. (2) A learnable dependency estimation module that predicts event coupling from latent event representations without requiring explicit event labels, leveraging temporal overlap as a proxy. (3) Demonstration that adaptive parallelism achieves better accuracy-efficiency Pareto frontier compared to fixed parallel or autoregressive decoding on standard dense video captioning benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | ActivityNet Captions | Standard benchmark with diverse events |
| Primary metric | SODA | Combines caption quality and temporal grounding |
| Baseline 1 | Vid2Seq (autoregressive) | Sequential baseline; lower bound on efficiency |
| Baseline 2 | PDVC (non-autoregressive) | Independent baseline; lower bound on quality |
| Baseline 3 | PAD (parallel autoregressive) | Parallel baseline with fixed schedule |
| Ablation-of-ours | Uniform sequential grouping | Removes dynamic grouping; tests graph necessity |

### Why this setup validates the claim
This experimental design forms a falsifiable test by pitting our dynamic dependency graph (DDG) decoding against three baselines that span the spectrum from fully sequential to fully parallel decoding. The dataset (ActivityNet Captions) provides event boundaries and temporal overlap for training dependency labels. The primary metric SODA measures both caption quality and temporal alignment, capturing the trade-off inherent in our method: we sacrifice speed only when dependencies are strong. The ablation (uniform sequential grouping) isolates the benefit of learning which events to parallelize, while baselines like Vid2Seq (sequential) and PDVC (independent) bound the extremes. PAD uses a fixed parallel schedule without dependency awareness, so comparing against it reveals whether adaptive parallelism improves over a static one. The chosen metric will detect if our method uniquely improves the quality-efficiency frontier.

### Expected outcome and causal chain

**vs. Vid2Seq** — On a video with loosely related events (e.g., separate actions in different rooms), Vid2Seq decodes all events sequentially, causing high latency because it forces causal masking across unrelated events. Our method estimates low dependency between such events and groups them into parallel clusters, drastically reducing decoding steps. We expect our method to achieve comparable SODA scores (since dependencies are weak) but with significantly lower inference time (e.g., 2–3× speedup) on videos with low event overlap.

**vs. PDVC** — On a video with tightly coupled events (e.g., a person pouring liquid followed immediately by drinking), PDVC decodes all events independently, ignoring their causal relationship and likely generating captions that are temporally inconsistent (e.g., describing the drink before the pour). Our method assigns high dependency scores to temporally overlapping events, decoding them sequentially within a group, preserving causal order and accuracy. We expect our method to outperform PDVC on SODA by a noticeable margin (e.g., 5–10 points) on subsets with high event overlap, while matching PDVC on speed for independent events.

**vs. PAD** — On a video where dependency strength varies (e.g., some events strongly coupled, others independent), PAD uses a fixed parallelization strategy (e.g., group size 2) that either over-parallelizes coupled events (hurting quality) or under-parallelizes independent ones (hurting speed). Our method adaptively adjusts grouping based on estimated dependencies, optimizing the balance per video. We expect our method to achieve equal or better SODA than PAD (by avoiding quality loss on coupled events) and faster average inference (by parallelizing more when dependencies are weak), with the gain concentrated on videos of moderate dependency heterogeneity.

### What would falsify this idea
If our method’s improvement over PAD is uniform across all video subsets (rather than concentrated on videos with mixed dependency strengths), or if the uniform sequential ablation matches our full method in quality but not speed, then the central claim that adaptive dependency estimation matters would be falsified.

## References

1. Parallelized Autoregressive Decoding for Omni-Modal Dense Video Captioning
2. Qwen3-VL Technical Report
3. Factorized Learning for Temporally Grounded Video-Language Models
4. TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability
5. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
6. TRACE: Temporal Grounding Video LLM via Causal Event Modeling
7. Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models
8. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding
