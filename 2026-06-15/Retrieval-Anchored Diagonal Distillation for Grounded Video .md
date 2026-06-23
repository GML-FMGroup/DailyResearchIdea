# Retrieval-Anchored Diagonal Distillation for Grounded Video Generation

## Motivation

Existing video generation methods such as Diagonal Distillation produce coherent videos but lack grounding in external content, resulting in hallucinations and factual inconsistencies when generating videos about specific entities or events. VideoRAG demonstrates successful retrieval-augmented video QA, but its retrieval process is designed for question answering, not for temporally aligned conditioning during generation. The core limitation is that diagonal distillation processes video chunks with a fixed schedule, but retrieval from a corpus requires dynamic temporal alignment; without synchronizing retrieval units with generation steps, the conditioning signal is misaligned with the content being produced, causing underutilization of retrieved information.

## Key Insight

By anchoring each generation step to a retrieved video chunk that shares the same temporal position in a canonical representation, we convert the generative process from open-loop prediction to closed-loop constrained generation, where the retrieved chunk serves as a fixed point that prevents autoregressive drift.

## Method

### Retrieval-Anchored Diagonal Distillation (RADD)

(A) **What it is**: We propose **Retrieval-Anchored Diagonal Distillation (RADD)**, a method for retrieval-augmented video generation that conditions each diagonal distillation step on a retrieved video clip aligned in temporal position. Input is a text query and a video corpus; output is a generated video of specified length.

(B) **How it works** (pseudocode):
```python
def RADD(query, corpus, T_chunks=4, chunk_len=16, steps_per_chunk=[16,12,8,4]):
    # T_chunks: number of video chunks to generate
    # chunk_len: frames per chunk
    # steps_per_chunk: list of denoising steps decreasing from first to last chunk
    
    # Step 1: Retrieve anchor clips for each chunk position
    anchor_clips = []
    for t in range(T_chunks):
        # Encode query and corpus clips using LVLM (e.g., Video-LLaMA)
        # Retrieve top-1 clip from corpus that is temporally relevant to chunk position
        # Use temporal encoding of query (e.g., "first 1 second", "next 1 second")
        clip = retrieve_clip(query, corpus, temporal_position=t, chunk_len=chunk_len)
        anchor_clips.append(clip)
    
    # Step 2: Initialize first chunk from random noise
    x0 = random_noise([chunk_len, H, W, C])
    
    # Step 3: Diagonal distillation loop
    # --- Load-bearing assumption: The pre-trained diffusion model can effectively utilize concatenated anchor frames as conditioning without any fine-tuning or adaptation. ---
    # To address this, we fine-tune the diffusion model on a calibration set of 512 video pairs with anchor conditioning before generation.
    for t in range(T_chunks):
        num_steps = steps_per_chunk[t]
        anchor = anchor_clips[t]
        # Concatenate noisy chunk and anchor along channel dimension
        x0 = denoise_with_anchor(x0, anchor, num_steps, condition='concatenate')  # fine-tuned model weights
        generated_chunks.append(x0)
        if t < T_chunks - 1:
            # Shift frames: drop first frame, add noise to last frame
            x0 = shift_and_noise(x0, shift=1)
    return concatenate(generated_chunks)
```

(C) **Why this design**: We chose to anchor each chunk to a retrieved clip rather than conditioning on a single global retrieval because temporal alignment is critical: without it, retrieved content may not match the temporal context of the generated chunk, causing inconsistency. We use the fixed decreasing step schedule {16,12,8,4} from Diagonal Distillation to maintain motion coherence, but modify it to incorporate the anchor at each step; this increases computational cost per chunk due to anchor processing, but grounding reduces error accumulation, potentially allowing fewer total steps. We assume the retrieval function returns clips of exact chunk length (16 frames); if not, we resize to 16×H×W using bilinear interpolation, accepting a resolution mismatch that may blur details. Concatenation conditioning is simple and efficient, but risks feature dominance; we choose it for initial simplicity over a cross-attention mechanism, which would add parameters and training cost. This trade-off prioritizes ease of adaptation over potential accuracy gains. To mitigate the load-bearing assumption that concatenation works zero-shot, we fine-tune the pre-trained diffusion model (a 1.5B parameter video diffusion model from Ho et al., 2022) on a calibration set of 512 video pairs, each consisting of a noisy chunk and its corresponding anchor clip, using the same denoising objective. This fine-tuning requires approximately 24 GPU-hours on 4 NVIDIA A100 GPUs (80GB each) and adds 0.1M training steps with batch size 8.

(D) **Why it measures what we claim**: The anchor clip at each chunk position operationalizes **temporal alignment** because it is retrieved such that its temporal position in the corpus matches the generation step; this assumes the retrieval function accurately identifies clips that are both semantically and temporally relevant. This assumption fails when the corpus lacks a relevant clip for a given position, in which case the anchor is irrelevant and the method reverts to standard diagonal distillation, losing grounding. The denoising step count per chunk measures **coherence** because fewer steps are allocated to later chunks assuming early chunks provide sufficient context; this fails when later events are independent of early context, causing under-generation. The concatenated condition measures **groundedness** because the anchor's features directly influence denoising; this assumes the model learns to use the anchor rather than ignore it, which fails if the anchor is noisy or the model overfits to noise, leading to ignored conditioning. Additionally, retrieval accuracy (measured by temporal IoU between retrieved and ground-truth clips) directly quantifies grounding quality. When temporal IoU is low (e.g., <0.5), the anchor is likely misaligned, and we expect generation quality to drop, effectively reverting to the non-anchored baseline. This creates a direct link: higher temporal IoU should correlate with lower FVD—if this correlation is weak, the assumption that alignment improves generation is falsified.

## Contribution

(1) We introduce the concept of retrieval-anchored generation for video, where each chunk is grounded in an externally retrieved clip aligned by temporal position. (2) We design a method that integrates retrieval into the diagonal distillation framework, enabling grounded video generation without retraining the entire model from scratch. (3) We provide a mechanism to synchronize retrieval granularity with generation chunk boundaries, addressing a key mismatch between retrieval and generation in existing RAG-for-video approaches.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | UCF-101 (primary), Diving48 (secondary) | UCF-101: diverse actions; Diving48: complex temporal events, factual consistency |
| Primary metric | FVD (overall), temporal IoU (retrieval quality) | FVD: distributional video quality; temporal IoU: quantifies grounding mismatch |
| Baseline 1 | Diagonal Distillation | No retrieval, tests temporal coherence gain |
| Baseline 2 | VideoRAG | Retrieval but no temporal alignment |
| Baseline 3 | Autoregressive diffusion | Standard sequential generation |
| Ablation-of-ours | RADD w/o temporal anchors (random anchor) | Random anchor, isolates alignment effect |

### Why this setup validates the claim

This setup forms a falsifiable test of RADD's central claim: that temporally aligned retrieval anchors improve video generation quality. UCF-101 provides a diverse set of actions with clear temporal structure, making it suitable to reveal alignment benefits. Diving48 is added to evaluate factual consistency because it contains fine-grained temporal segments (e.g., different dive types) where correct retrieval and temporal alignment are critical for generating realistic sequences. FVD is the primary metric because it captures both frame-level and temporal distributional similarity, directly reflecting the coherence and realism improvements we hypothesize. Temporal IoU serves as a secondary metric to measure retrieval accuracy and its correlation with generation quality; if high IoU does not correspond to lower FVD, the grounding mechanism is ineffective. The baselines isolate key components: Diagonal Distillation tests the benefit of retrieval alone, VideoRAG tests the effect of temporal alignment (since it retrieves a single global clip), and standard autoregressive generation tests the overall advantage of diagonal distillation with anchors. The ablation of using random anchors (instead of retrieved ones) pinpoints whether temporal alignment is the critical factor. Together, they allow us to attribute any gains specifically to our temporal anchoring mechanism, rather than to retrieval or architectural changes in general.

### Expected outcome and causal chain

**vs. Diagonal Distillation** — On a video of a person walking, Diagonal Distillation may generate flickering limbs because it lacks grounding for each chunk, leading to error accumulation over time. Our method retrieves a clip of the same action from the corpus, providing a visual anchor that stabilizes each chunk's generation. We expect a noticeable FVD improvement (e.g., ~10% reduction) on motion-heavy sequences, with the gap concentrated on videos requiring consistent object appearance.

**vs. VideoRAG** — VideoRAG retrieves a single global clip, causing temporal misalignment when the action changes (e.g., a person first running then falling). Our method retrieves a separate anchor for each chunk, allowing the model to adapt to changing contexts. On such multi-stage videos, VideoRAG will produce blurry transitions, while RADD maintains clarity. We expect our method to outperform VideoRAG primarily on videos with distinct temporal segments, with FVD gap larger for complex actions.

**vs. Autoregressive diffusion** — Autoregressive generation suffers from error propagation, especially for long videos, where early errors compound. Our diagonal distillation with anchors reduces drift by grounding each chunk to retrieved frames. On longer sequences (e.g., 8+ chunks), we expect RADD to achieve lower FVD and higher motion coherence, as the autoregressive baseline will show increased divergence from the true distribution.

### What would falsify this idea

If RADD fails to outperform Diagonal Distillation on motion coherence or FVD, or if its gains are uniform across all videos rather than concentrated on temporally complex ones, the central claim that temporal anchoring improves quality would be falsified. Specifically, observing similar improvement on simple, static videos as on multi-stage actions would indicate that retrieval alone (without alignment) drives performance. Additionally, if the correlation between temporal IoU and FVD is weak (e.g., Pearson r < 0.2), the assumption that alignment directly improves generation is unsupported.

## References

1. VideoRAG: Retrieval-Augmented Generation over Video Corpus
2. Streaming Autoregressive Video Generation via Diagonal Distillation
3. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
4. Long-Context Autoregressive Video Modeling with Next-Frame Prediction
5. Wan: Open and Advanced Large-Scale Video Generative Models
6. MAGI-1: Autoregressive Video Generation at Scale
7. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
8. Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens
