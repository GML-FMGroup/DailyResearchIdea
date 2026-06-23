# Temporal Adaptive Chunking via Predictive Uncertainty for Video Generation and Retrieval

## Motivation

Existing video generation and retrieval methods (e.g., MAGI-1, VideoRAG) rely on fixed temporal windows or chunks that are agnostic to content structure, leading to suboptimal handling of variable event durations and missed long-range dependencies. For instance, MAGI-1 uses a fixed chunk length, causing oversmoothing at event boundaries and unnecessary segmentation within coherent segments. VideoRAG encodes videos with fixed temporal windows, limiting its ability to capture long-range context across diverse event lengths. The root cause is the lack of an online, content-aware boundary detection mechanism that adapts to video dynamics without external supervision.

## Key Insight

The generative model's own prediction error serves as a natural and online indicator of content boundaries because the model systematically fails to anticipate transitions between distinct events, providing a self-supervised signal that aligns chunk boundaries with perceptual event boundaries.

## Method

## Method: TAC-PU (Temporal Adaptive Chunking via Predictive Uncertainty)

We propose TAC-PU, an online algorithm that dynamically segments video into content-aware chunks by monitoring per-frame prediction uncertainty from an autoregressive video model.

**(A) What it is**: TAC-PU takes an autoregressive video model (e.g., a next-frame predictor outputting a Gaussian distribution per pixel) and processes video frames sequentially, outputting a list of frame indices that mark chunk boundaries.

**(B) How it works**:

```python
import numpy as np

def tac_pu_generate(model, initial_context, max_frames, window_size=10, alpha=0.1, threshold_mult=2.0, calibration_videos=16):
    # Calibrate threshold_mult using held-out validation set (16 videos) to optimize boundary F1 against ground-truth
    threshold_mult = calibrate_threshold(model, calibration_videos, initial_threshold=2.0)
    
    context = [initial_context]
    chunk_boundaries = [0]
    error_ema = None
    error_var = None
    for t in range(max_frames):
        # Predict next frame distribution from last window_size frames
        pred_dist = model.predict_next(context[-window_size:])
        actual = model.sample(pred_dist)  # actual frame (ground truth during training, sampled during generation)
        # Compute negative log-likelihood of actual under predicted distribution
        mean = pred_dist.mean()
        logvar = pred_dist.logvar()
        nll = 0.5 * (logvar + (actual - mean)**2 / np.exp(logvar) + np.log(2*np.pi))
        error = np.mean(nll)
        # Update EMA statistics
        if error_ema is None:
            error_ema = error
            error_var = np.array([0.0])
        else:
            delta = error - error_ema
            error_ema = alpha * error + (1 - alpha) * error_ema
            error_var = alpha * delta**2 + (1 - alpha) * error_var
            error_std = np.sqrt(error_var)
        # Trigger boundary if error exceeds adaptive threshold
        if error > error_ema + threshold_mult * error_std:
            chunk_boundaries.append(t)
            context = [actual]  # reset context to current frame
        else:
            context.append(actual)
    return chunk_boundaries

def calibrate_threshold(model, num_videos, initial_threshold):
    # For each video, run TAC-PU with a set of candidate thresholds and pick the one maximizing average F1
    candidate_thresholds = np.linspace(1.0, 4.0, 7)
    best_f1 = -1
    best_thresh = initial_threshold
    for thresh in candidate_thresholds:
        f1_scores = []
        for video in validation_set[:num_videos]:
            pred_boundaries = tac_pu_generate(model, video[0], len(video), threshold_mult=thresh)
            gt_boundaries = video['boundaries']
            f1 = boundary_f1(pred_boundaries, gt_boundaries, tolerance=5)
            f1_scores.append(f1)
        avg_f1 = np.mean(f1_scores)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_thresh = thresh
    return best_thresh
```

**(C) Why this design**: We chose to use negative log-likelihood (NLL) as the error metric rather than mean squared error (MSE) because NLL accounts for both prediction accuracy and the model's own uncertainty (via logvar), making it a proper scoring rule; the cost is slightly higher computation but yields more reliable boundary detection when model confidence varies. The exponential moving average (EMA) with adaptive threshold is used instead of a fixed global threshold to handle non-stationary video statistics (e.g., gradual illumination changes); the trade-off is increased sensitivity to sudden noise bursts, which we mitigate with a higher threshold_mult (calibrated to 2.0 via held-out validation). After a boundary, we reset the context to the current frame rather than keeping the entire past, ensuring that each chunk starts with a clean slate and avoids error accumulation across events; this sacrifices temporal continuity at boundaries but is acceptable since boundaries represent content changes. The window_size of 10 frames balances responsiveness and smoothing: small windows may over-segment due to transient errors, while large windows delay boundary detection. Compared to the fixed chunking in MAGI-1, our method dynamically adapts chunk lengths to content, and unlike confidence-based routing (Anti-pattern 3), NLL measures the model's predictive incompleteness rather than self-assessed confidence.

**(D) Why it measures what we claim**: The per-frame NLL ε_t measures the generative model's inability to predict the actual frame given past context. Under the assumption that the model is well-trained on coherent video segments, ε_t spikes at event boundaries where the visual input changes abruptly; this assumption fails for scenes with gradual transitions or model misspecification, in which case ε_t reflects model uncertainty rather than a boundary. To mitigate this, we calibrate the threshold_mult on a small validation set (16 videos) to ensure the detector is tuned to true boundaries rather than noise. The EMA statistics (μ and σ) adaptively normalize ε_t, so a deviation beyond threshold_mult*σ signals a boundary. Thus, ε_t operationalizes "content novelty" while the adaptive threshold operationalizes "online event detection". The context reset ensures that error does not propagate across boundaries, aligning with the motivation of content-aware segmentation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | ActivityNet Captions | Dense captions with temporal event boundaries. |
| Primary metric | Answer F1 on video QA | Direct measure of RAG performance. |
| Additional metrics | ROUGE-L on generated captions, Recall@1 on chunk-level event retrieval | Capture generation fidelity and retrieval accuracy. |
| Baseline1 | Uniform fixed-length (32 frames) | Simple non-adaptive chunking baseline. |
| Baseline2 | Confidence-based chunking | Uses model confidence to find boundaries. |
| Baseline3 | Random chunking | Naive baseline without any adaptation. |
| Ablation | TAC-PU w/o adaptive threshold (fixed threshold=2.0) | Isolates adaptation effect. |
| Additional ablation | window_size in {5,10,15} | Assess sensitivity to window size. |

### Why this setup validates the claim

This experimental design tests whether TAC-PU's adaptive segmentation improves retrieval-augmented video QA over non-adaptive baselines. ActivityNet Captions provides ground-truth event boundaries, enabling evaluation of both segmentation quality and downstream QA accuracy. Uniform chunking tests if content-aware splitting is necessary; confidence-based chunking tests if NLL (our metric) outperforms raw model confidence; random chunking establishes a lower bound. The ablation isolates the adaptive threshold component. The primary metric, Answer F1, directly captures the end-to-end benefit for RAG, ensuring that any segmentation improvement translates to better retrieval and generation. Additional metrics (ROUGE-L and Recall@1) provide finer-grained diagnostics: ROUGE-L on generated captions tests whether coherent chunks lead to better caption sequences, and Recall@1 on chunk-level event retrieval tests whether boundaries align with ground-truth events.

### Expected outcome and causal chain

**vs. Uniform fixed-length** — On a video with varied event durations (e.g., a cooking tutorial with short steps and long waits), uniform chunks either split long events mid-action or merge short events with adjacent background. This causes retrieval to return incomplete or overcombined segments, degrading answer quality. TAC-PU instead adapts chunk boundaries to content changes, so each chunk aligns with a coherent event. Thus, we expect a noticeable gap in Answer F1 on videos with high event duration variance, but parity on videos with uniform events.

**vs. Confidence-based chunking** — On a scene with gradual lighting change (e.g., sunrise), the model's confidence may decrease slowly, causing confidence-based chunking to fragment the scene into many tiny chunks. This introduces noise and irrelevant context. TAC-PU uses NLL, which spikes only at actual boundaries because the prediction error increases sharply when the content genuinely changes, not when uncertainty rises gradually. Hence, we expect TAC-PU to produce fewer but more accurate boundaries, leading to higher Answer F1 on gradual transition videos.

**vs. Random chunking** — Random chunking often splits key events arbitrarily, causing retrieval to miss critical context or include irrelevant frames. TAC-PU's content-aware boundaries ensure each chunk is semantically coherent, so retrieved segments are more relevant. The gap in Answer F1 should be large across all video types, with TAC-PU always outperforming random chunking.

### What would falsify this idea

If TAC-PU's advantage over uniform chunking is uniform across all video types rather than concentrated on those with high event duration variance, then the adaptive mechanism is not exploiting content changes as intended, and the central claim is wrong. Additionally, if calibration on 16 videos does not yield robust threshold_mult (e.g., same threshold works for all videos), the calibration step may be ineffective.

## References

1. VideoRAG: Retrieval-Augmented Generation over Video Corpus
2. Streaming Autoregressive Video Generation via Diagonal Distillation
3. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
4. Long-Context Autoregressive Video Modeling with Next-Frame Prediction
5. Wan: Open and Advanced Large-Scale Video Generative Models
6. MAGI-1: Autoregressive Video Generation at Scale
7. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
8. Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens
