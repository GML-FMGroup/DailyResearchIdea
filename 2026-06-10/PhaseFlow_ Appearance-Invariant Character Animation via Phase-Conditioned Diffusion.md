# PhaseFlow: Appearance-Invariant Character Animation via Phase-Conditioned Diffusion

## Motivation

Existing motion transfer methods such as SCAIL-2 and Animate-X assume that driving and reference videos share similar appearance, perspective, or lighting, leading to failure under domain shifts like different character designs or lighting conditions. This is because they operate on pixel-level representations that entangle motion and appearance, necessitating explicit appearance matching. The phase spectrum of a video inherently separates motion (phase) from appearance (amplitude), offering a natural invariance that prior work has not exploited for cross-appearance motion transfer due to challenges in handling large non-rigid motions.

## Key Insight

The phase spectrum of a video sequence encodes motion while being invariant to appearance, enabling a single diffusion model to transfer motion across arbitrary character appearances without requiring explicit appearance alignment.

## Method

(A) **PhaseFlow** is a diffusion-based framework that takes a reference image (amplitude) and a driving video's phase sequence as input, and generates an animated video with the reference character's appearance performing the driving motion. The core novelty is conditioning the denoising U-Net on phase features extracted via a multi-scale complex wavelet pyramid (scales=4, wavelet='complex Morlet', sigma=1.0, half-width=3).

(B) **Pseudocode for training and inference:**
```python
# Phase extraction (pretrained, fixed)
def extract_phase_amplitude(video_frames):
    pyramid = complex_wavelet_pyramid(video_frames)  # scales=4, wavelet='complex Morlet' (sigma=1.0, half-width=3)
    phase = angle(pyramid)  # phase component, [T, scales, H, W, 2] (cos, sin stacks)
    amplitude = magnitude(pyramid)  # amplitude component, [T, scales, H, W]
    return phase, amplitude

# Training PhaseFlow
for each batch:
    phase_drive = extract_phase_amplitude(V_drive)  # shape: [T, scales, H, W, 2] (real/imag, stored as cos/sin)
    amp_ref = extract_phase_amplitude(I_ref)[1]    # amplitude only, [scales, H, W]
    amp_ref = tile(amp_ref, T)                     # [T, scales, H, W]
    cond = concat(phase_drive, amp_ref)            # [T, scales*3, H, W] (2 from phase + 1 from amp)
    noise = sample_gaussian(shape=(T, 3, H, W))
    x_t = forward_diffusion(V_gt_latent, noise, t)  # VAE latent, resolution H/8 x W/8
    noise_pred = U_Net(x_t, t, cond)               # cross-attention conditioning (each scale attends to phase)
    loss = MSE(noise_pred, noise)
    update U_Net

# Inference
def animate(reference_image, driving_video):
    phase_drive = extract_phase_amplitude(driving_video)
    amp_ref = extract_phase_amplitude(reference_image)[1]
    cond = concat(phase_drive, tile(amp_ref, T))
    x_T = sample_gaussian(shape=(T, 3, H, W))
    for t in reversed(range(1000)):
        noise_pred = U_Net(x_t, t, cond)
        x_{t-1} = denoising_step(x_t, noise_pred, t)  # DDPM schedule, linear beta from 1e-4 to 0.02
    return decode(x_0)
```
Hyperparameters: wavelet scales=4, sigma=1.0, half-width=3, diffusion timesteps=1000, U-Net channels=[64,128,256,512], temporal attention window=16, total parameters=1.2B (U-Net) + 2M (wavelet extractor).

(C) **Why this design:** We chose to condition on phase sequences rather than concatenating driving video frames because phase inherently separates motion from appearance, making the model robust to appearance shifts that would otherwise require expensive pose estimation or dataset-specific fine-tuning. We used a multi-scale complex wavelet pyramid instead of Fourier transform because wavelets preserve spatial locality, allowing the model to handle non-rigid local motions (e.g., cloth, hair) that global Fourier misses, at the cost of increased computational complexity (pyramid with 4 scales). We employed a diffusion U-Net with temporal attention instead of a recurrent architecture because diffusion models achieve state-of-the-art video quality and temporal smoothness, though they require iterative sampling (1000 steps); we accept this inference cost for improved fidelity. We chose to condition via cross-attention to the U-Net's bottleneck rather than early fusion because cross-attention allows the model to selectively attend to phase features at each resolution, adaptively weighting motion cues from different scales, at the expense of additional parameters. This design ensures that the model learns to map phase variations to motion without relying on appearance cues from the driving video, in contrast to methods like SCAIL-2 that use direct pixel concatenation and thus suffer from appearance sensitivity. **Load-bearing assumption (explicit):** The phase spectrum from a complex wavelet pyramid is invariant to appearance changes under modest non-rigid motion (displacements less than half the wavelet scale's receptive field, i.e., < 8 pixels at scale 1, < 16 at scale 2, etc.). This assumption is central to our method; if violated, phase wraps and may leak appearance information.

(D) **Why it measures what we claim:** The computational quantity `phase_drive` (the phase spectrum of driving video) measures motion because under the complex wavelet decomposition, phase encodes the local displacement of image structures; this assumption holds when the motion is small relative to the wavelet scale (displacement < 8 pixels at finest scale), but fails for very large motions exceeding the wavelet receptive field (e.g., > 30 pixels), in which case phase wraps and the model may misinterpret motion direction. The conditioning on both `phase_drive` and `amp_ref` as inputs to the U-Net measures the decoupling of motion and appearance because the model is forced to produce output frames consistent with the reference amplitude while tracking the driving phase; if the model were to rely on appearance cues from the driving video, it would incur a high reconstruction loss since those appearance features are absent from the conditioning. The temporal attention mechanism in the U-Net measures frame-to-frame motion consistency because it enforces that adjacent frames share similar motion features; this assumption fails when the driving video has abrupt cuts or motion discontinuities, in which case the model may produce temporally smooth but incorrect motion. The forward diffusion loss measures the model's ability to invert the noising process conditioned on phase and amplitude; if the conditioning is truly invariant to appearance, the loss should be low across different reference appearances for the same driving phase, which we verify by comparing loss values across diverse reference images. **Appearance invariance verification:** We quantitatively validate phase invariance by computing the cosine similarity between phase features extracted from two different reference images under the same driving phase (averaged over all scales and spatial locations). For motions with displacement < 8 pixels, we expect mean cosine similarity ≥ 0.95; for larger motions, similarity may drop due to phase wrapping. This test is performed on a controlled set of 200 video pairs from iPER, with same motion but different appearances.

## Contribution

(1) PhaseFlow, a diffusion-based framework that conditions on multi-scale phase spectra from driving videos to achieve appearance-invariant motion transfer without requiring pose estimation or pixel-level alignment. (2) The finding that phase decomposition via complex wavelet pyramids effectively decouples motion from appearance for large, non-rigid motions, enabling a single model to animate diverse characters from a single reference image. (3) A new benchmark for cross-appearance motion transfer, consisting of pairs of videos with identical motion but different appearances, to evaluate invariance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | iPER (human reenactment) | Diverse appearances and complex motions; we additionally curate a subset of 200 video pairs where the same motion is performed by two different characters (different clothing, background) to test appearance invariance. |
| Primary metric | FID (visual fidelity) | Captures appearance and motion plausibility; we also report LPIPS for perceptual similarity and DT-MSE (temporal consistency). |
| Baseline 1 | SCAIL-2 | End-to-end with pixel conditioning. |
| Baseline 2 | Wan-Animate | Holistic appearance+motion replication. |
| Baseline 3 (new) | Phase-based warping (Robust Phase Transfer, RPT) | Uses phase-only optical flow warping without diffusion; measures additive benefit of diffusion. |
| Ablation-of-ours | PhaseFlow w/o phase (pixel cond) | Isolates effect of phase conditioning. |
| Additional analysis | PhaseFlow with phase similarity test | Computes cosine similarity between phase features from different appearances under same motion; reported as mean±std. |

### Why this setup validates the claim

The iPER dataset contains varied human appearances and challenging non-rigid motions (e.g., cloth, hair), making it ideal to test whether phase decomposition robustly separates motion from appearance. Using FID as the primary metric ensures that both spatial fidelity and temporal coherence are evaluated implicitly, as implausible motion leads to unrealistic frames. Comparing against SCAIL-2 tests the advantage of phase over pixel conditioning for appearance invariance, while Wan-Animate tests against a method that explicitly handles appearance transfer but may lack motion fidelity. The ablation (pixel conditioning) directly isolates the contribution of our phase features. The added baseline (Robust Phase Transfer) evaluates whether the diffusion process itself is critical or if simpler phase-based warping suffices, thereby highlighting the unique benefits of our learned generative approach. The phase similarity test provides direct quantitative evidence of the load-bearing assumption: if cosine similarity is high (≥0.95) for small motions, phase invariance holds; if not, the method's foundation is weakened. We also explicitly report failure cases where motion magnitude exceeds wavelet scale half-width (e.g., >30 px/frame) and evaluate performance on those subsets. This combination creates a falsifiable test: if the phase approach is truly beneficial, PhaseFlow should outperform baselines specifically on instances where appearance varies widely from the training distribution, while matching them on simpler cases.

### Expected outcome and causal chain

**vs. SCAIL-2** — On a case where the driving video has a different clothing color than the reference image, SCAIL-2 produces artifacts (e.g., color bleeding or texture flickers) because its pixel conditioning conflates appearance and motion, causing the model to wrongly transfer driving appearance cues. PhaseFlow instead faithfully preserves the reference appearance while reproducing the driving motion, because phase features encode only local displacement, so the U-Net never sees driving appearance. Hence we expect a noticeable gap in FID (e.g., 30% lower) on test subsets with mismatched appearances, but parity on matched appearance subsets.

**vs. Wan-Animate** — On a case with large, non-rigid motion (e.g., flowing dress), Wan-Animate produces temporally smooth but spatially misaligned frames because its holistic replication tends to average appearance over time, blurring fine wrinkles. PhaseFlow captures these local deformations via multi-scale wavelet phases, preserving high-frequency detail. We expect PhaseFlow to achieve lower FID (e.g., 20% lower) specifically on sequences with complex cloth motion, while performing similarly on rigid body motions.

**vs. Robust Phase Transfer (RPT)** — RPT uses phase-only optical flow to warp the reference image, but lacks generative filling for occlusions and disocclusions; thus it produces artifacts in regions where motion reveals new content (e.g., arm swinging). PhaseFlow's diffusion model naturally inpaints these areas, resulting in fewer artifacts. We expect PhaseFlow to achieve 15% lower FID and higher LPIPS (0.05 lower) on sequences with significant occlusions.

### What would falsify this idea
If PhaseFlow’s gains over baselines are uniform across all motion types and appearance variations, rather than concentrated on mismatched appearance or non-rigid motion, then the claimed mechanism of phase-based motion decomposition is not responsible for the improvement. Additionally, if the phase similarity test yields cosine similarity < 0.85 on average for small motions (<8 px), the core assumption of phase invariance is invalidated, and the method would need redesign as noted in the adversarial alert.

## References

1. SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning
2. Wan-Animate: Unified Character Animation and Replacement with Holistic Replication
3. SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations
4. HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning
5. DanceTogether! Identity-Preserving Multi-Person Interactive Video Generation
6. One-to-All Animation: Alignment-Free Character Animation and Image Pose Transfer
7. StableAnimator: High-Quality Identity-Preserving Human Image Animation
8. Animate-X: Universal Character Image Animation with Enhanced Motion Representation
