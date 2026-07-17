# SwapConsistency: A Self-Supervised Metric for Compositional Evaluation of Multi-Reference Audio-Video Generation

## Motivation

Existing evaluation benchmarks for multi-reference audio-video generation, such as MultiRef-Compass, rely on static sets of reference combinations and fixed automated metrics that do not test whether models truly compose references compositionally. They assume a fixed-sample evaluation suffices, but models may overfit to specific reference pairings, leading to spurious consistency. The structural gap is that no metric dynamically probes compositional generalizability by systematically varying reference assignments, which is essential for assessing whether the model treats references as independent, composable components.

## Key Insight

A model that compositionally understands references will maintain output coherence when references are randomly swapped between generated samples, because coherence under swap is invariant to the particular assignment of each reference to its slot.

## Method

(A) **What it is:** SwapConsistency is a self-supervised evaluation metric that measures the compositional consistency of a multi-reference audio-video generation model by computing the average coherence of outputs produced after swapping references between pairs of generated samples. The metric outputs a single scalar score between 0 and 1, where higher values indicate stronger compositional treatment.

(B) **How it works (pseudocode):**
```pseudocode
# Input: Multi-reference generation model M that takes audio ref a and video ref v → output (video, audio) pair.
# Pretrained coherence models: SyncNet (audio-video sync), CLIP (video-video similarity), CLAP (audio-audio similarity)
# Hyperparameters: K = number of swap pairs, α=0.5 (weight for sync), β=0.25 (weight for video consistency), γ=0.25 (weight for audio consistency)

function coherence(output, a_ref, v_ref):
    sync_score = SyncNet(output.video, output.audio)   # range [0,1]
    v_sim = CLIP_similarity(output.video.keyframe, v_ref.keyframe)  # [0,1]
    a_sim = CLAP_similarity(output.audio, a_ref)       # [0,1]
    return α*sync_score + β*v_sim + γ*a_sim

SwapConsistency(M, test_reference_pairs):
    # test_reference_pairs: list of (a_i, v_i) pairs
    original_scores = []
    swapped_scores = []
    for i in range(K):
        (a1,v1), (a2,v2) = random.sample(test_reference_pairs, 2)
        O1 = M(a1,v1)
        O2 = M(a2,v2)
        O_swap1 = M(a1,v2)
        O_swap2 = M(a2,v1)
        orig_score = (coherence(O1,a1,v1) + coherence(O2,a2,v2)) / 2
        swap_score = (coherence(O_swap1,a1,v2) + coherence(O_swap2,a2,v1)) / 2
        original_scores.append(orig_score)
        swapped_scores.append(swap_score)
    avg_orig = mean(original_scores)
    avg_swap = mean(swapped_scores)
    return avg_swap / avg_orig   # clamp to [0,1]
```

(C) **Why this design:** We chose to compute coherence using a weighted combination of three pretrained models (SyncNet, CLIP, CLAP) over a single proxy model because each covers a different necessary aspect of coherence: sync captures cross-modal alignment, video similarity ensures the visual reference is preserved, and audio similarity ensures the acoustic reference is preserved. The weights α, β, γ are balanced (0.5, 0.25, 0.25) to prioritize cross-modal sync, which is the most challenging dimension, accepting that intra-modal consistencies receive lower weight; this trade-off is motivated by the observation that multi-reference models often fail more severely on audio-visual sync than on single-modal consistency. We sample K=1000 random swap pairs from the test set to obtain a stable estimate without exhaustive enumeration, accepting a small variance in exchange for computational efficiency. The ratio form (avg_swap / avg_orig) normalizes for model quality, so that a high-quality model that originally produces high coherence also needs to maintain that under swaps; an alternative absolute difference would unfairly penalize strong models. We commit to fixed pretrained models (SyncNet, CLIP, CLAP) rather than training a custom coherence network because it keeps the metric self-supervised and reproducible across models; the cost is that the metric inherits any biases or inaccuracies of those pretrained models, but this is acceptable given their wide adoption.

(D) **Why it measures what we claim:** The computational quantity `avg_swap / avg_orig` measures compositional consistency because it captures the relative coherence of the model when forced to reassign references to different slots. The assumption linking this ratio to compositional treatment is that a model which composes references independently (i.e., treats audio and video references as separate, slot-based inputs) will produce outputs whose coherence is invariant to reference permutation; this assumption fails when the model relies on spurious correlations between specific reference pairs (e.g., overfitting to training co-occurrences), in which case the ratio drops below 1, indicating compositional failure. Within the coherence computation, `SyncNet(output.video, output.audio)` measures cross-modal alignment under the assumption that a correctly composed output should have synchronized audio and video; this assumption fails when sync depends on the specific identity of references (e.g., a dog video with a cat audio still being synced), but such scenarios are rare in practice. `CLIP_similarity(output.video.keyframe, v_ref.keyframe)` measures visual reference consistency under the assumption that the generated video should resemble the reference video; this assumption fails when the generation style differs from the reference style (e.g., cartoon vs. realistic), but the metric still reflects whether the model can preserve the content. `CLAP_similarity(output.audio, a_ref)` measures audio reference consistency analogously. Thus, every component of the metric operationalizes a facet of the motivation-level concept 'compositional consistency', and the ratio explicitly tests invariance to reference reassignment.

## Contribution

(1) A self-supervised evaluation metric, SwapConsistency, that dynamically generates compositional test cases by swapping references between generated samples, eliminating the need for static benchmarks or human annotations. (2) A design principle for compositional evaluation: the ratio of coherence under swapped references to original coherence quantifies the degree to which a model treats references as independent composable slots. (3) An operationalization of coherence via a weighted combination of SyncNet, CLIP, and CLAP, enabling reproducible and interpretable scoring.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MultiRef-AV dataset (from related papers) | Contains diverse audio-video reference pairs |
| Primary metric | SwapConsistency score (ours) | Directly quantifies compositional consistency |
| Baseline 1 | FVD (Fréchet Video Distance) | Standard video quality metric, ignores composition |
| Baseline 2 | FAD (Fréchet Audio Distance) | Standard audio quality metric, ignores composition |
| Baseline 3 | AV-Align (SyncNet-based sync score) | Measures cross-modal sync but not reference swapping |
| Ablation-of-ours | SwapConsistency without sync component (α=0) | Tests contribution of cross-modal alignment |

### Why this setup validates the claim

This design tests the central claim that SwapConsistency measures compositional consistency by contrasting it with standard quality metrics (FVD, FAD) that are blind to reference permutation, and with a sync-only metric (AV-Align) that captures alignment but not reference fidelity. The dataset includes multiple reference pairs with known compositional relationships, enabling controlled swap experiments. Our ablation removes the sync term to isolate its role. If SwapConsistency genuinely captures compositional treatment, it should (a) drop significantly when models use spurious correlations (e.g., overfit to training co-occurrences) while FVD/FAD remain unchanged, and (b) outperform AV-Align on cases where sync is preserved but content is mismatched. The ablation helps confirm that cross-modal sync is a necessary component. This setup creates a falsifiable test: if SwapConsistency correlates perfectly with FVD across all models, the claim fails because it would not detect compositional failures.

### Expected outcome and causal chain

**vs. FVD** — On a case where a model memorizes training pairs (e.g., a specific dog video with a barking audio often co-occur), swapping references with a cat video and meow audio yields a low-SwapConsistency output because coherence drops (sync fails despite good video/audio individually). FVD would rate the video as visually realistic (high quality), missing the composition error. Our method instead detects the mismatch via averaged swap coherence, so we expect SwapConsistency to be significantly lower for such models, while FVD shows no drop.

**vs. FAD** — On the same memorization case, FAD would rate the audio as realistic (clean meow), ignoring that it was swapped from the wrong reference. Our metric penalizes the audio-reference mismatch via CLAP similarity, so SwapConsistency drops. Expected signal: models with high FAD but low SwapConsistency on swapped pairs, indicating compositional weakness.

**vs. AV-Align** — On a case where the model produces well-synced but content-mismatched output (e.g., synced cat video with dog audio, both realistic), AV-Align gives a high sync score because mouth movements match audio—but the audio content is wrong. Our metric includes CLIP and CLAP similarities to penalize the mismatch. We expect AV-Align to remain high on such swaps, while SwapConsistency drops, showing the gap.

### What would falsify this idea

If SwapConsistency remains high (close to 1) for all models regardless of their composition ability, or if it correlates perfectly with any single baseline (e.g., FVD or AV-Align), then the metric fails to measure compositional consistency beyond existing measures.

## References

1. MultiRef-Compass: Towards Comprehensive Evaluation of Multi-Reference-to-Audio-Video Generation
2. OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation
3. T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation
4. VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models
5. VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation
6. VBench: Comprehensive Benchmark Suite for Video Generative Models
7. VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation
8. Diffusion Model Alignment Using Direct Preference Optimization
