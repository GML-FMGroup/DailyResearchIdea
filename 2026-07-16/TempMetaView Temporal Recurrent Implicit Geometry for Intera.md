# TempMetaView: Temporal Recurrent Implicit Geometry for Interactive Novel View Synthesis from Video

## Motivation

MetaView achieves consistent novel view synthesis from a single image but processes each frame independently, ignoring temporal dynamics essential for video streams. Its feed-forward geometry perception network has no memory, and the diffusion model cannot enforce frame-to-frame consistency. Depth Anything 3 recovers geometry from multiple views but neither targets novel view synthesis nor supports user interaction. The root cause is the absence of a temporal latent state that can propagate implicit geometry priors across frames, which prevents real-time interactive applications like live video editing.

## Key Insight

Temporal coherence in video scenes can be captured by a lightweight recurrent module operating in the latent space of the geometry perception network, enabling efficient propagation of implicit geometry priors without explicit optical flow or per-frame optimization.

## Method

TempMetaView extends MetaView with a Temporal Recurrence Module (TRM) and a Click Interaction Module (CIM). Inputs: video frames {I_t}, camera poses {P_t}, optional user clicks c_t = (x,y,action). Output: novel view images at requested viewpoints. The TRM maintains a hidden state h_t that aggregates geometry latents across frames, while CIM encodes a click into a spatial conditioning signal.

**How it works**:
```python
# Training (single clip of T=16 frames)
for t in 1..T:
    g_t = GPN(I_t)  # geometry latent (depth, normals) from ResNet-50 backbone
    h_t = GRU(g_t, h_{t-1})  # temporal recurrence (hidden size 128, 1 layer)
    # keyframe reset every 32 frames or upon scene cut detection (feature change > 0.5)
    if t % 32 == 0 or scene_cut > 0.5: h_t = 0
    # diffusion conditioning: x_t = [I_t, g_t, h_t, c_t_encoding]
    # supervised on ground-truth novel views with L1 reconstruction loss + temporal warping loss (weight 0.1)

# Inference (streaming, real-time)
h = 0
for each new frame I_t with requested viewpoint V_t:
    g = GPN(I_t)
    h = GRU(g, h)
    if user click c_t received:
        c_enc = CIM(c_t, I_t)  # 2D position + action one-hot -> spatial mask (bilinear interpolated)
    else:
        c_enc = zeros(1, H, W)
    sample novel view x_t from diffusion model (DDIM, 20 steps) conditioned on [I_t, g, h, c_enc]
```
Hyperparameters: GRU hidden=128, diffusion steps=20 (DDIM), training clips length=16, keyframe reset interval=32.

**Why this design**: We chose a GRU over a transformer attention across frames for efficiency, accepting reduced long-range memory to maintain real-time throughput (30 fps). We propagate geometry latents rather than pixel features because geometry latents encode scale-aware structure (from metric depth), enabling temporal consistency without drift. We integrate user clicks as a spatial conditioning mask instead of post-hoc optimization (e.g., NeRF editing) to achieve sub-millisecond response, though limiting edit granularity to point-based changes. Compared to a naive frame-by-frame MetaView+flow-warping baseline, our recurrent latent avoids optical flow computation (which fails under large motion) and uses learned temporal dynamics from video datasets.

**Why it measures what we claim**: The temporal hidden state h_t aggregates geometry priors across frames via the GRU recurrence. h_t measures **temporal consistency** because it encodes a compressed representation of past geometry that biases the diffusion toward frame-coherent outputs under the assumption that scene geometry changes smoothly; this assumption fails at sudden cuts or teleports, in which case h_t would lag and cause temporary inconsistency until reset. The click encoding c_enc measures **interactive controllability** because it directly modulates the diffusion conditioning via a spatial mask that forces the model to attend to the clicked region; this relies on the assumption that the diffusion model generalizes to out-of-distribution conditioning (e.g., moving an object), which fails for edits requiring physically implausible changes (e.g., removing a large foreground object). The geometry prior g_t from GPN measures **geometric consistency** because it provides a per-frame scale-aware depth estimate that anchors the novel view to metric space; this assumption fails in textureless or highly reflective regions where monocular depth is ambiguous, leading to potential 3D distortion in those areas. The warp error (average photometric error across warped frames) measures temporal consistency only under the assumption that scene motion is smooth and occlusions are minor; this assumption fails under large occlusion or cuts, where warp error becomes unreliable; to cover this failure, we also report temporal flicker frequency (mean per-pixel variance across consecutive novel views).

## Contribution

(1) A lightweight temporal recurrence module (TRM) that propagates implicit geometry latents across video frames, enabling temporally consistent novel view synthesis without explicit flow or global optimization. (2) A click-based interaction interface (CIM) integrated into the diffusion conditioning, allowing real-time user control over the generated novel views. (3) The first framework to unify temporal consistency and interactive editing within a diffusion-based implicit geometry approach for video streams.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | RealEstate10K | Large-scale indoor video, diverse motions. |
| Primary metric | Novel View PSNR (dB) | Standard measure of view synthesis fidelity. |
| Baseline 1 | MetaView (frame-by-frame) | No temporal aggregation, tests geometric prior. |
| Baseline 2 | MetaView+Flow (optical flow) | Tests flow-based temporal consistency. |
| Baseline 3 | MetaView+Transformer (temporal attention) | Tests necessity of recurrent design for efficiency. |
| Ablation | TempMetaView w/o CIM | Isolates effect of click interaction. |
| Resource budget | 4× NVIDIA A100, 2 days training | Estimated from MetaView scaling. |

### Why this setup validates the claim
This experimental setup is designed to falsify the central claim by testing each sub-claim independently. The baseline MetaView (frame-by-frame) tests whether temporal aggregation improves quality; if our method outperforms it on temporally coherent scenes but not on static scenes, the temporal claim is supported. MetaView+Flow tests whether learned temporal dynamics (GRU) beat explicit optical flow; our method should show gains on large motion where flow fails. MetaView+Transformer tests whether the GRU's efficiency is necessary for real-time; we expect comparable quality but higher latency for the transformer. The ablation w/o CIM tests whether click conditioning adds value; we expect improvement on tasks with user edits. PSNR captures overall fidelity, but we also evaluate temporal consistency via warp error and temporal flicker frequency to detect temporal artifacts under failures. RealEstate10K provides diverse camera motions and occlusions, enabling stress-testing of temporal mechanisms. Hyperparameter tuning on temporal clip length (8,16,32) is performed on a validation split of 10% of RealEstate10K.

### Expected outcome and causal chain

**vs. MetaView** — On a video with fast camera rotation, the baseline produces jittery novel views because it lacks temporal context, leading to inconsistent geometry across frames. Our method instead propagates geometry latents via GRU, smoothing transitions due to learned temporal dynamics, so we expect lower temporal warp error (e.g., 20% reduction) and higher PSNR (0.5 dB) on high-motion clips.

**vs. MetaView+Flow** — On a scene with large occlusion (e.g., object disappearing behind another), optical flow fails and causes ghosting artifacts because pixel correspondences break. Our method avoids explicit flow by relying on recurrent latent aggregation from geometry priors, which is robust to occlusion, so we expect fewer artifacts—measurable as a 0.5 dB PSNR gain on occlusion-heavy subsets.

**vs. MetaView+Transformer** — On a long video clip, the transformer baseline achieves similar PSNR (within 0.1 dB) but at 10× higher latency (300 ms vs 30 ms per frame), confirming that the GRU's efficiency is mandatory for real-time interactive applications.

**vs. TempMetaView w/o CIM** — On a user click to move a small object, the ablation ignores the click because no spatial conditioning is provided, producing the original view. Our method encodes the click into a spatial mask that biases the diffusion, altering the region accordingly, so we expect a significant PSNR improvement (e.g., 3 dB) on clicked regions but parity elsewhere.

### What would falsify this idea
If the PSNR gain over MetaView is uniform across all clips regardless of motion magnitude, the temporal recurrence claim is unsupported. If removing CIM does not degrade performance on click-aligned regions, the interaction claim fails. If the transformer baseline outperforms our method while maintaining real-time latency (30 fps), then the necessity of recurrent design is falsified.

## References

1. MetaView: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors
2. Depth Anything 3: Recovering the Visual Space from Any Views
