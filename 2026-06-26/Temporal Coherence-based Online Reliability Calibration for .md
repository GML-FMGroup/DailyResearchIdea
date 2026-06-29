# Temporal Coherence-based Online Reliability Calibration for Tool-Orchestrated Video Understanding

## Motivation

Existing systems like Robust-TO assume tool-provided reliability scores are calibrated, but in practice, scores can be miscalibrated and tools may silently fail under distribution shift or perturbation, causing erroneous evidence weighting. The root cause is that reliability estimation is performed once using offline calibration, which cannot adapt to online changes. Temporal consistency across video frames provides a free, online supervisory signal: reliable tools produce temporally coherent outputs, while unreliable ones exhibit inconsistency, enabling online calibration without ground truth.

## Key Insight

Temporal consistency of tool outputs across consecutive video frames is a monotonic indicator of tool reliability because continuous video inherently requires prediction coherence for correct perception, and this coherence can be computed without any external labels.

## Method

```pseudocode
Algorithm: Temporal Coherence Online Reliability Calibrator (TCO-TRC)
Input: Stream of video frames f_t, set of tools T = {t_1,...,t_n}
Output: Calibrated reliability weights w_i(t) for each tool at each frame

Parameters:
  - window_size W = 5 (number of frames for coherence computation)
  - update_rate alpha = 0.7 (exponential moving average smoothing)
  - threshold gamma = 0.3 (minimum coherence to consider tool reliable)

Initialize: for each tool i,
  - coherence_ema_i = 0.5 (initial neutral)
  - last_K_outputs_i = deque(maxlen=W)  // store last W outputs

For each frame f_t:
  For each tool i:
    o_i_t = tool_i(f_t)  // tool's prediction (e.g., bounding boxes, action logits, or text)
    Append o_i_t to last_K_outputs_i
    
    // Compute temporal coherence for tool i based on its output type
    if tool_i is object detector:
       // average IoU of tracked objects between consecutive frames
       coherence_score_i = mean(IoU(bbox_{t-1}, bbox_t) over all matched objects in window)
    elif tool_i is action recognizer:
       // average cosine similarity of logit vectors
       coherence_score_i = mean(cosine_sim(logits_{t-1}, logits_t) over window)
    elif tool_i is text recognizer:
       // negative normalized edit distance
       coherence_score_i = 1 - mean(edit_distance(text_{t-1}, text_t) / max_len)
    else:
       // generic: cosine similarity of output embeddings (frozen encoder)
       coherent_score_i = mean(cosine_sim(embed_{t-1}, embed_t))
    
    // Update exponential moving average of coherence
    coherence_ema_i = alpha * coherence_score_i + (1 - alpha) * coherence_ema_i
    
    // Detect silent failure: if tool produced no output for this frame but had recent high coherence, flag
    if o_i_t is None and coherence_ema_i > gamma:
       coherence_ema_i = coherence_ema_i * 0.5  // penalize
    
    // Map EMA to calibrated weight (linear scaling from 0 to 1)
    w_i(t) = max(0, min(1, (coherence_ema_i - gamma) / (1 - gamma)))
    
  Use weights w_i(t) to aggregate tool evidence (e.g., Robust-TO's three-tier synthesis)
  Return w_i(t) for all tools
```

**Why this design** (≥80 words, 3+ design decisions with trade-offs): We chose a sliding window of W=5 frames over frame-by-frame comparison because it reduces sensitivity to single-frame noise while maintaining low latency (trade-off: larger W stabilizes but delays detection). We used an exponential moving average (EMA) with α=0.7 rather than a simple moving average to give more weight to recent observations, enabling faster adaptation to sudden tool failures (trade-off: EMA can overshoot under high variance). We adopted a linear scaling from coherence to weight (instead of sigmoid) because it provides an interpretable, monotonic mapping with clear zero region below threshold γ (trade-off: less flexibility than learned calibration but avoids overfitting). We chose task-specific coherence metrics (IoU, cosine similarity, edit distance) rather than a single embedding-based metric to align with each tool's output structure, accepting engineering overhead for better signal (trade-off: embedding-based method works for any tool but may lose modality-specific cues). We detect silent failures by penalizing missing outputs only when recent coherence is high, distinguishing between tools that are genuinely idle vs. failing (trade-off: may miss failures if tool never produced outputs historically).

**Why it measures what we claim** (≥60 words, specify assumption and failure mode): The computational quantity `coherence_ema_i` measures **temporal reliability** because we assume that a reliable tool's outputs for adjacent frames should be consistent given the continuous nature of video; this assumption fails when the scene undergoes rapid changes (e.g., cuts, fast motion) causing even reliable tools to produce inconsistent outputs, in which case `coherence_ema_i` reflects **scene instability** rather than tool reliability. The threshold γ operationalizes the decision boundary **reliable vs. unreliable** under the assumption that coherence significantly above chance indicates correct operation; this assumption fails when a tool is barely coherent (e.g., consistently wrong but stable) leading to false positive reliability. The silent failure detection mechanism (penalty on `coherence_ema_i` when no output) relies on the assumption that a previously reliable tool should produce outputs continually; this fails when the tool legitimately has nothing to report (e.g., no objects in frame), causing false negative detection. Each component in the method thus carries an explicit assumption that must hold for the proxy to align with the concept of reliability.

## Contribution

['(1) A novel online temporal consistency calibration mechanism (TCO-TRC) that estimates tool reliability without requiring tool-provided scores, using only the coherence of outputs across video frames.', '(2) A practical silent failure detection scheme integrated into the coherence update, enabling automatic down-weighting of tools that produce no output despite prior reliable behavior.', '(3) A design principle that temporal coherence can serve as a free supervision signal for tool reliability in video understanding, which can be plugged into existing orchestration frameworks like Robust-TO.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Embodied video benchmarks with common perturbations | Tests robustness under realistic tool failures |
| Primary metric | Task completion rate on perturbed frames | Measures ability to handle unreliable tool outputs |
| Baseline 1 | Frontier Video Reasoning Models (VideoLLaMA) | Assumes uniform frame reliability; baseline for need of calibration |
| Baseline 2 | Standard Video MLLM (VideoChat-R1) | Lacks temporal coherence for tool outputs |
| Ablation-of-ours | TCO-TRC without silent failure detection | Isolates effect of silent failure detection |

### Why this setup validates the claim

The chosen dataset includes frames with motion blur, glare, and occlusion that cause intermittent tool failures. By measuring task completion on these perturbed frames, we directly test whether our temporal coherence calibration can down-weight unreliable tool outputs. The baselines (Frontier models and standard MLLMs) lack such calibration, so they serve as counterexamples that should fail on perturbed frames. The ablation removes silent failure detection, isolating whether that component is necessary for handling tools that suddenly stop producing outputs. This design creates a falsifiable test: if our method improves performance relative to baselines, but only on perturbed frames (not clean ones), it supports the claim that temporal coherence calibration is the cause. If improvements are uniform, the claim is falsified.

### Expected outcome and causal chain

**vs. Frontier Video Reasoning Models (VideoLLaMA)** — On a frame with heavy motion blur causing the object detector to miss objects, the baseline uses uniform frame weighting, so the blurry frame contaminates the aggregation and produces incorrect predictions. Our TCO-TRC detects low temporal coherence (low IoU between adjacent frames) and assigns a low weight to that tool, so the aggregation relies on other tools or clean frames. We expect a noticeable gap in accuracy on perturbed frames (e.g., >10% relative improvement) while maintaining parity on clean frames.

**vs. Standard Video MLLM (VideoChat-R1)** — On a case where an action recognizer outputs erratic logits due to occlusion, the standard MLLM treats each frame independently and may misclassify. Our method computes cosine similarity between logits over a sliding window, penalizes the inconsistent outputs, and reduces their weight. We expect our method to achieve higher accuracy on frames with occlusion (e.g., >15% relative improvement) while performing similarly on unperturbed frames.

### What would falsify this idea

If our method shows similar gains on clean frames as on perturbed frames, then the calibration is not specifically addressing temporal reliability but rather providing a uniform benefit (e.g., better base feature extraction). This would indicate that the coherence metric does not isolate tool failures as intended.

## References

1. Confidence-Aware Tool Orchestration for Robust Video Understanding
2. VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding
3. VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning
4. Video-R1: Reinforcing Video Reasoning in MLLMs
5. LLaVA-Video: Video Instruction Tuning With Synthetic Data
6. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
7. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection
8. Text-Conditioned Resampler For Long Form Video Understanding
