# Depth-Guided Occlusion Inpainting for Autoregressive Video World Models

## Motivation

Existing video world models like AlayaWorld generate future frames without explicit reasoning about occlusions, leading to artifacts when objects become occluded or reappear. The root cause is the lack of inductive bias for occlusion handling: the model treats all pixels uniformly, ignoring depth ordering and temporal consistency that dictate which content is revealed. This gap is structural across current autoregressive video generators, as they rely only on learned temporal dynamics without geometric priors.

## Key Insight

Depth and temporal consistency provide a geometric invariant that constrains the inpainting of occluded regions, ensuring coherence with both past and future observations.

## Method

### (A) What it is
**Occlusion-Aware Video World Model (OAVWM)** extends an autoregressive video world model (e.g., AlayaWorld) with an occlusion reasoning module that jointly predicts depth maps and inpaints occluded regions using temporal consistency. Input: frames up to time *t*, action *aₜ*, and a recurrent memory of past inpainted content. Output: next frame *Fₜ₊₁* with occluded regions inpainted.

### (B) How it works
```python
def OAVWM_next_frame(F_1:t, a_t, memory):
    # 1. Estimate depth for last frame
    D_t = DepthNet(F_t)  # pretrained depth estimator
    # 2. Predict optical flow conditioned on action
    flow = FlowNet(F_t, a_t)  # e.g., RAFT, fine-tuned on action-conditioned data
    # 3. Warp depth to next frame candidate
    D_next_warp = warp(D_t, flow)  # bilinear sampling
    # 4. Detect occlusion regions: discontinuity in warped depth or flow inconsistency
    occlusion_mask = (warp_consistency_error(F_t, flow) > τ) | (edge(D_next_warp) > 0.5)
    # τ=0.1 (pixel-wise flow error threshold)
    # 5. Copy non-occluded pixels via flow
    F_next_partial = warp(F_t, flow) * (1 - occlusion_mask)
    # 6. Inpaint occluded regions using diffusion conditioned on depth and memory
    # 4-step diffusion with teacher distillation (as in Fast Autoregressive Video Diffusion)
    F_next_inpaint = DiffusionInpainter(
        mask=occlusion_mask, 
        context=F_next_partial, 
        depth=D_next_warp, 
        memory_state=memory
    )
    # 7. Update memory with inpainted region (store both RGB and depth)
    memory.update(F_next_inpaint, D_next_warp)
    return F_next_inpaint
```

### (C) Why this design
We chose depth-guided occlusion detection over learned attention masks because depth provides a geometric prior that generalizes to unseen scenes without additional training; the cost is reliance on a pretrained depth estimator that may fail on stylized domains. We use warping-based motion flow rather than learning flow from scratch to leverage robust optical flow methods (e.g., RAFT), accepting that flow errors at motion boundaries can introduce false positives in occlusion masks. Diffusion-based inpainting is chosen over single-step generative fill because it enables iterative refinement conditioned on depth and memory, but it is computationally heavy; we mitigate by using 4-step distilled diffusion, inheriting the speed-quality trade-off from Fast Autoregressive Video Diffusion. A recurrent memory (inspired by VideoSSM’s hybrid memory) stores both inpainted RGB and depth to propagate temporal consistency; this risks memory saturation for very long sequences (>10 minutes), where we accept gradual drift as a bound on application horizon.

### (D) Why it measures what we claim
The depth map *Dₜ* **measures geometric scene structure** because, under the assumption that the world is piecewise rigid and surfaces are smooth, depth encodes ordering; this assumption fails for non-rigid objects like cloth, where depth still provides soft ordering but may be inexact. The occlusion mask computed from flow consistency and depth edges **measures regions where new content must be generated** under the assumption that flow is locally accurate and depth discontinuities correspond to occlusion boundaries; this assumption fails at fine-grained motion boundaries (e.g., thin fences), where the mask may include spurious pixels, leading to unnecessary inpainting. The diffusion inpainter conditioned on depth and recurrent memory **measures content that is consistent with estimated geometry and prior frames** under the assumption that occlusion areas are largely determined by geometry; this assumption fails when multiple plausible inpaintings exist (e.g., ambiguous texture behind an occluder), in which case the model produces a blurred average. The recurrent memory **measures long-term temporal consistency of inpainted regions** under the assumption that occluded areas persist across frames; this assumption fails for fast-moving occluders, where memory becomes stale and introduces ghosting artifacts.

## Contribution

(1) A depth-guided occlusion reasoning module integrated into an autoregressive video world model, combining warping, depth estimation, and diffusion-based inpainting with recurrent memory. (2) A design principle demonstrating that explicit depth ordering improves occlusion recovery in dynamic scenes, reducing artifacts in generated long videos. (3) An extension of hybrid memory architectures (e.g., VideoSSM) to store both RGB and depth for temporally consistent inpainting.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Davis 2017 (occlusion-heavy) | Tests occlusion handling and temporal consistency.
| Primary metric | FVD on occlusion subsets | Measures video quality where occlusion matters.
| Baseline | AlayaWorld | Autoregressive baseline with no occlusion reasoning.
| Baseline | VideoSSM | Tests long-term memory without depth guidance.
| Baseline | Fast Autoregressive Video Diffusion | Strong diffusion baseline without geometric prior.
| Ablation | OAVWM w/o depth-guided masking | Isolates benefit of depth-guided occlusion detection.

### Why this setup validates the claim

Davis 2017 provides natural occlusion scenarios (object boundaries, camera motion), directly testing the core claim that depth-guided occlusion detection improves next-frame generation. By comparing against AlayaWorld (no occlusion handling) and VideoSSM (memory-based but geometric-agnostic), we isolate whether depth cues add value beyond temporal memory. Fast Autoregressive Video Diffusion tests whether iterative diffusion alone (without explicit depth conditioning) suffices. The ablation on depth-guided masking confirms the specific contribution of geometric prior. FVD on occlusion subsets is the right metric because it aggregates perceptual quality exactly where failure modes (blur, ghosting) are most likely under the baseline mechanisms. This combination forms a falsifiable test: if our method excels precisely on occlusion-heavy frames but not uniformly, the causal chain is supported.

### Expected outcome and causal chain

**vs. AlayaWorld** — On a case where a foreground object moves across a static background, AlayaWorld produces blurry or repeated background texture in newly occluded regions because its autoregressive transformer lacks explicit occlusion reasoning and averages over ambiguous possibilities. Our method instead warps depth to detect the occlusion boundary, then inpaints with diffusion conditioned on depth and memory, so we expect a significant gap in FVD on frames with large occluded areas (e.g., ≥30% pixels), but parity on static scenes.

**vs. VideoSSM** — On a case where an occluder reveals a background region that was hidden for many frames, VideoSSM may generate a stale or distorted background from its memory because it relies on latent state without geometric grounding. Our method uses warped depth to detect the newly visible region and inpaints with depth context, so we expect better texture reconstruction on long-range occlusions (e.g., after 50 frames), observable as lower FVD on that subset.

**vs. Fast Autoregressive Video Diffusion** — On a case with thin occluding structures (e.g., fence), this baseline may incorrectly inpaint fence texture into background because its diffusion process, while iterative, lacks depth ordering to separate layers. Our depth edges help mask precisely at occlusion boundaries, so we expect fewer artifacts around fine occluders, measurable as a noticeable but smaller gap on those specific frames.

### What would falsify this idea

If our method fails to outperform baselines specifically on occlusion-heavy subsets (or shows uniform improvement across all frames) it would indicate that depth guidance provides no structural advantage over learned temporal consistency, contradicting the geometric prior claim.

## References

1. AlayaWorld: Long-Horizon and Playable Video World Generation
2. VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory
3. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
4. Autoregressive Video Generation without Vector Quantization
