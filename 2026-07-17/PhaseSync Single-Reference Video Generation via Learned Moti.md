# PhaseSync: Single-Reference Video Generation via Learned Motion Phase Priors

## Motivation

Existing single-reference video generation methods produce temporally incoherent animations because they lack an explicit temporal structure such as motion phase. Audio-driven methods like those in Kling-Omni use audio frequency envelopes to synchronize motion, but in the absence of audio, there is no such signal. Benchmarks like VBench and VideoScore evaluate video quality but do not address the specific challenge of achieving temporal coherence from a single static image. We identify that motion phase can be predicted from visual appearance because natural motion rhythms are correlated with content (e.g., speaking rate in faces, gait in bodies), enabling a learned phase prior that replaces the role of audio.

## Key Insight

The motion phase of a rhythmic movement can be predicted from a single reference image because the image encodes implicit cues about the motion’s rhythm (e.g., facial expression, body posture), and a deep network trained on video data can learn to extract these phases via a phase-invariant representation.

## Method

### (A) What it is

**PhaseSync** is a two-stage framework that first predicts a motion phase sequence from a single reference image using a **PhasePredictor**, then generates a video by conditioning a video diffusion model on both the reference image and the predicted phase sequence.

### (B) How it works

```python
# PhaseSync: Single-Reference Video Generation
# Input: reference image I_ref, number of frames N=32
# Output: video V of N frames

# 1. Feature extraction (pre-trained DINOv2, output dim 768)
F_ref = DINOv2(I_ref)  # shape: (768,)

# 2. Phase prediction (PhasePredictor: MLP with 2 hidden layers, 256 neurons each, ReLU activation, output dim 2N=64)
# Output: sin/cos pairs for each frame
sin_cos = PhasePredictor(F_ref)  # shape: (N, 2)
# Normalize each pair to unit vector to enforce valid sine/cosine
sin_cos = sin_cos / torch.norm(sin_cos, dim=1, keepdim=True)

# Loss for PhasePredictor training: L2 loss between predicted sin_cos and ground truth sin_cos (from video motion)
# Ground truth phase extracted via optical flow: compute displacement of central point, apply Hilbert transform to get instantaneous phase, then sin/cos.

# 3. Video generation (conditional denoising diffusion model, e.g., AnimateDiff)
# For each frame t, phase embedding e_t = (sin_cos[t,0], sin_cos[t,1]) (2-dim)
# Video diffusion model takes noisy video x_t, reference image I_ref, and per-frame phase embeddings e as conditioning.
# Denoising process: predict noise from conditioned UNet
# Training: Two-stage
#   Stage 1: Train PhasePredictor alone on video clips with ground truth phase (extracted via optical flow or landmark motion)
#   Stage 2: Train video diffusion model with frozen PhasePredictor, using both ground truth phases (as condition) and predicted phases (via teacher forcing)
```

**Assumption:** A single static reference image contains sufficient information to uniquely determine the motion phase sequence for the entire generated video. This is operationalized by training PhasePredictor to output a deterministic sequence from DINOv2 features.

**Calibration/Verification:** To check this assumption, we measure the circular correlation between predicted phase (arctan2(sin,cos)) and ground truth phase on a held-out set. A low correlation (<0.3) would indicate ambiguity, prompting a switch to a generative phase predictor (e.g., diffusion-based) in future work.

### (C) Why this design

We chose a sine/cosine representation instead of raw phase angles because it eliminates the circular boundary discontinuity at 0/2π, allowing standard L2 loss; the trade-off is doubled output dimension (2N vs N), which increases parameter count and risk of overfitting on small datasets. We use a fixed output length of N=32 frames rather than variable length to keep the predictor architecture simple (MLP with fixed output size); this sacrifices flexibility for scenes requiring longer videos, but a sliding-window post-processing can extend length at inference time. We separate training of the PhasePredictor from the video diffusion model to reduce optimization difficulty, accepting that predicted phases may deviate from the distribution seen by the diffusion model during training; we mitigate this by fine-tuning the diffusion model with predicted phases in stage 2. Using DINOv2 features provides robust semantic encoding of the reference image but may not capture motion-specific details; we chose it over a motion encoder (e.g., a tiny video model) because it is pre-trained on diverse images and reduces data requirements, at the cost of potentially ignoring subtle pose cues that affect phase.

### (D) Why it measures what we claim

The output of PhasePredictor (normalized sine/cosine per frame) is interpreted as a **motion phase** that should temporally synchronize the generated video. This interpretation rests on two assumptions: (1) that the reference image's visual content (encoded by DINOv2) uniquely determines the rhythmic phase sequence for a plausible motion, and (2) that the diffusion model will use the phase embeddings to align motion across frames. Assumption (1) is operationalized via the L2 loss between predicted and ground-truth sine/cosine pairs; the ground-truth phase is computed from video motion (e.g., landmark displacement), which itself assumes a dominant periodic motion component. This assumption fails when the video contains non-periodic or multiple superimposed motions—in that case the ground-truth phase is ambiguous, and the predictor learns an average phase that may not correspond to any real motion. Assumption (2) is operationalized via conditioning the denoising UNet with per-frame phase embeddings; the hope is that the model learns to correlate these embeddings with temporal position. This assumption fails if the diffusion model overpowers the phase signal by focusing on the reference image alone—then the phase condition is ignored, and temporal coherence degrades. By measuring the video's temporal smoothness (e.g., via optical flow consistency) and comparing to a baseline without phase conditioning, we can verify whether the phase signal affects output; however, a positive effect does not prove that the predicted phase is correct, only that it provides a useful signal. The key metric for our claim is the correlation between predicted phase and actual motion phase in generated videos, which requires an external motion tracker to evaluate. We therefore compute the **circular correlation** (ρ) between predicted phase and ground truth phase (from optical flow on the generated video) as a direct measure of phase correctness. Additionally, we report **Average Variance of Motion Magnitude (AVMM)** across frames to isolate temporal coherence from visual quality: lower variance indicates smoother motion.

## Contribution

(1) We introduce PhaseSync, a framework for single-reference video generation that uses a learned motion phase prior to achieve temporal coherence without external audio signals. (2) We demonstrate that motion phase can be predicted from a single static image using a lightweight MLP trained on video data, and that this prediction effectively conditions a video diffusion model to produce rhythmic, coherent animations. (3) We provide an analysis of how phase prediction quality affects video temporal coherence, revealing that the phase prior is most beneficial for motions with clear periodicity (e.g., talking, walking).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TikTok dancing videos (500 clips) + FaceForensics (facial expressions, 300 clips) + NTU RGB+D (body actions, 200 clips) | Diverse periodic motions; test generalizability.
| Primary metric | FVD (Fréchet Video Distance) | Standard for video quality and motion.
| Secondary metric (phase correctness) | Circular correlation (ρ) between predicted & ground truth phase | Directly measures phase prediction accuracy.
| Secondary metric (temporal coherence) | Average Variance of Motion Magnitude (AVMM) | Isolates temporal smoothness from visual quality.
| Baseline | Stable Video Diffusion (SVD) | State-of-the-art image-to-video.
| Baseline | AnimateDiff + IP-Adapter | Strong motion module baseline.
| Baseline | VideoCrafter2 (image-to-video) | Competitive recent model.
| Ablation (ours) | PhaseSync with random phases | Isolates effect of predicted phase.
| Ablation (ours) | PhaseSync with phase predicted from random reference image | Tests content-specificity of phase prediction.

**Compute budget:** PhasePredictor (10k params) training: 2 hours on 1 RTX 3090. Video diffusion fine-tuning (AnimateDiff with LoRA): 12 hours on 4 RTX 3090s.

### Why this setup validates the claim

This setup tests the core claim that predicted motion phase sequences improve temporal coherence in single-image-to-video generation. We use periodic datasets (TikTok, FaceForensics, NTU) where phase is meaningful. Comparing against three strong baselines (SVD, AnimateDiff, VideoCrafter) that lack explicit phase conditioning reveals whether our method yields better FVD, ρ, and AVMM. The random-phase ablation controls for any arbitrary periodic signal; if our predicted phase outperforms random on ρ and AVMM, the predictor must capture real motion rhythms. The random-reference ablation tests whether prediction relies on content-specific features; if it performs worse than the true reference, phase prediction is content-dependent. FVD captures overall quality, while ρ and AVMM isolate phase correctness and temporal coherence. This combination forms a falsifiable test: if our method shows significant improvement on periodic subsets versus baselines on all three metrics, and beats both ablations, then phase prediction works.

### Expected outcome and causal chain

**vs. SVD** — On a case where the reference shows a person starting a dance move, SVD often generates a static frame or random jitter because it lacks temporal priors for rhythmic motion. PhaseSync instead uses the predicted phase to schedule arm and leg positions over frames, producing a smooth dance loop. We expect a noticeable FVD gap (e.g., 50+ points lower) on periodic action clips, but smaller differences on static scenes. For ρ, we expect >0.7 correlation on periodic motions; SVD has no phase, so ρ near 0. AVMM should be lower for PhaseSync (e.g., 0.02 vs 0.05).

**vs. AnimateDiff** — On a fast-spinning dancer, AnimateDiff may blur or misalign due to its coarse motion modeling. PhaseSync’s per-frame sine/cosine embeddings enforce continuous cyclic motion, reducing blur. We expect lower FVD (e.g., ~40 points) on high-periodicity clips, with baseline parity on slow motions. ρ should be higher for PhaseSync (>0.6 vs ~0.2 for AnimateDiff). AVMM lower (e.g., 0.03 vs 0.06).

**vs. VideoCrafter** — On a repetitive jumping jack sequence, VideoCrafter may produce erratic speed changes because it only sees the starting pose. Our method’s phase sequence prescribes exact timing, yielding consistent jumps. The expected gap is widest on periodic actions (e.g., >60 FVD points), narrowing for deformable objects. ρ should be >0.8 vs <0.3. AVMM lower (e.g., 0.01 vs 0.04).

**vs. Ablation (random phase)** — With random phases, the generated video shows unnatural pacing (e.g., sudden stops or speed-ups). Our predicted phase yields temporally smooth motion. Expected FVD gap is large (e.g., ~80 points) on all periodic samples, confirming phase prediction’s value. ρ is near 0 for random, >0.7 for ours. AVMM higher for random (e.g., 0.08 vs 0.02).

**vs. Ablation (random reference)** — Using a random reference image (e.g., a blank scene) yields ρ near 0 and FVD similar to random phase, confirming that phase prediction relies on content-specific features.

### What would falsify this idea

If PhaseSync’s FVD improvement over baselines is uniform across all video types (periodic and random) and ρ is low (<0.3) on periodic clips, then the predicted phases are not capturing meaningful motion structure. Alternatively, if the random-phase ablation performs comparably to the predicted-phase version on ρ and AVMM, then any periodic signal (even noise) helps, undermining the claim that we predict correct phases.

## References

1. MultiRef-Compass: Towards Comprehensive Evaluation of Multi-Reference-to-Audio-Video Generation
2. OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation
3. T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation
4. VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models
5. VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation
6. VBench: Comprehensive Benchmark Suite for Video Generative Models
7. VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation
8. Diffusion Model Alignment Using Direct Preference Optimization
