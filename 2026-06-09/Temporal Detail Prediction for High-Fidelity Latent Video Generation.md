# Temporal Detail Prediction for High-Fidelity Latent Video Generation

## Motivation

Existing latent compression methods for video generation (e.g., the VAE used in MALT Diffusion) discard high-frequency details such as textures and edges because they treat each frame independently, ignoring strong temporal correlations. This loss of detail reduces visual fidelity, especially in long videos where compression artifacts accumulate. To preserve high-frequency content without increasing bitrate, we need a mechanism that exploits temporal redundancy.

## Key Insight

High-frequency details are temporally coherent and can be predicted from low-frequency base latents and motion vectors, eliminating the need to store details for every frame.

## Method

### Temporal Detail Prediction for High-Fidelity Latent Video Generation

#### (A) What it is
Temporal Detail Prediction Network (TDPN) takes a base latent (low-frequency content) and a motion field (optical flow) as input, and outputs a detail latent (high-frequency residuals) for each frame. **Core assumption:** High-frequency details in each video frame can be deterministically predicted from the low-frequency base latent of that frame and the optical flow motion field computed from previous frames.

#### (B) How it works (pseudocode)
```python
# Architecture of detail predictor D: 4-layer convolutional network, channels [32,64,128,256], kernel_size=3, ReLU activation, final output 16-channel detail latent.
# Training: Given video of T frames, compute:
#   - z_base (shape T x 64 x H/8 x W/8) via low-pass VAE (compression ratio=64)
#   - z_detail_gt (shape T x 16 x H/8 x W/8) via high-pass VAE (detail latent dim=16)
#   - m (shape T-1 x 2 x 32 x 32) via RAFT (motion field size=32x32)
# Pad first motion field with zeros for frame 0.
# Detail predictor: D_encoder: z_base -> h (shape T x 256 x H/32 x W/32); D_decoder: conv3d on concatenated h and m -> z_pred_detail (shape T x 16 x H/8 x W/8)
# Loss: L = L2(z_pred_detail, z_detail_gt) + 0.1 * LPIPS(decoded(frame from z_base + z_pred_detail), decoded(frame from z_base + z_detail_gt))
# Optimizer: Adam, lr=1e-4, batch_size=32, steps=200k on UCF-101.
# Calibration: During training, compute prediction error upper bound E_upper = L2(z_pred_detail, z_detail_gt) on held-out validation set, stratified by motion magnitude (high/low) and occlusion ratio. Monitor if E_upper exceeds a threshold (e.g., 0.05) indicating assumption violation.
# At generation: Given z_base and motion m (from past generated frames), predict z_pred_detail, add to z_base, decode.
```
Hyperparameters: VAE compression ratio = 64, detail latent dimension = 16, motion field size = 32x32, TDPN parameters = 2.1M, total GPU memory ~8GB for batch size 8.

#### (C) Why this design
We chose to separate base and detail latents via two VAE branches (low-pass and high-pass) rather than using a single VAE with higher resolution, because this forces the model to explicitly handle details. We used optical flow as motion cue over simple frame differences, because flow captures non-local motion more accurately, accepting the cost of a pretrained flow estimator (RAFT). We predict details autoregressively from past base and motion rather than all past frames, because this reduces memory footprint, but risks error accumulation if motion is incorrect.

#### (D) Why it measures what we claim
The base latent z_base captures low-frequency structure (large-scale motion, colors) because it is encoded with a low-pass filter that discards high frequencies; this operationalizes the concept of 'base structure'. The motion field m measures temporal displacement of pixels, operationalizing 'motion cues'. The predicted detail latent z_pred_detail's L2 distance to ground truth z_detail measures how well high-frequency details are preserved, under the assumption that details are a deterministic function of base and motion; this assumption fails when details are caused by new objects or lighting changes (not explained by motion), in which case the loss reflects prediction uncertainty rather than detail loss. We further validate this assumption by computing prediction error upper bound E_upper and reporting its distribution across motion and occlusion scenarios.

## Contribution

(1) A two-level latent representation (base + detail) that decouples low-frequency structure from high-frequency residuals for video generation. (2) A temporal detail prediction mechanism that uses optical flow to infer details from base latents, reducing storage of detail latents per frame. (3) A training procedure that jointly optimizes the VAE and detail predictor, enabling reconstruction of high-fidelity videos from compressed latents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | UCF-101 | Standard benchmark with diverse motion. |
| Primary metric | HF-PSNR (PSNR on high-pass filtered frames) | Measures detail preservation directly. |
| Baseline 1 | Direct Latent Diffusion (standard VAE + diffusion, no detail decomposition) | Tests benefit of detail decomposition. |
| Baseline 2 | Frame-Difference Motion (use ∆I instead of RAFT flow) | Tests need for accurate flow. |
| Baseline 3 | Base-Only Generation (skip detail prediction, decode z_base) | Tests importance of predicting details. |
| Ablation of ours | No Perceptual Loss (L2 only); Motion Corruption (add Gaussian noise σ=0.1 to flow) | Isolates effect of perceptual regularization; tests robustness to flow errors. |

### Why this setup validates the claim

This setup forms a falsifiable test of the central claim that explicit detail prediction from base and motion improves detail fidelity. UCF-101 provides varied motion and textures, covering both dynamic and static scenes. HF-PSNR directly quantifies high-frequency accuracy, isolating the target effect. The baseline Direct Latent Diffusion tests whether decomposition into base and detail is beneficial; if our method outperforms it, the decomposition is justified. Frame-Difference Motion tests if sophisticated optical flow is necessary; if flow improves over frame difference, motion accuracy is key. Base-Only Generation tests whether predicting details adds value beyond base; if our method beats it, details matter. The ablation of perceptual loss tests its importance for detail realism. The Motion Corruption ablation tests sensitivity to flow errors: if a small perturbation drastically drops HF-PSNR, the deterministic assumption is fragile. Additionally, we compute the prediction error upper bound E_upper on validation clips and stratify by motion magnitude (high vs low) and occlusion ratio to quantify when the assumption holds or fails.

### Expected outcome and causal chain

**vs. Direct Latent Diffusion** — On a fast-motion clip like a tennis swing, Direct Latent Diffusion blurs the racket because it compresses motion and detail into a single latent, losing high frequencies. Our method uses optical flow to warp the base and predict residuals, preserving sharp edges. We expect a large HF-PSNR gap (e.g., >3 dB) on high-motion clips, but near parity (<0.5 dB) on static clips.

**vs. Frame-Difference Motion** — On a scene with large non-rigid motion like a galloping horse, frame difference fails to capture pixel correspondences, causing incorrect detail prediction (e.g., blurry mane). Our method uses RAFT flow which handles large displacements, so details remain accurate. We expect a significant HF-PSNR drop (e.g., >2 dB) for the baseline specifically on frames with large motion magnitudes.

**vs. Base-Only Generation** — On a texture-rich scene like a field of grass waving, Base-Only produces smooth output lacking fine grass blades. Our method predicts high-frequency details from motion, giving a realistic texture. We expect the gap to be largest (e.g., >4 dB) on clips with high spatial frequency content, and minimal on uniform surfaces.

**Ablation (No Perceptual Loss)** — On a scene with semantically important details (e.g., a face), removing perceptual loss leads to overly smooth textures due to L2-only training. Our method with perceptual loss retains facial features. We expect a moderate drop (e.g., 1-2 dB HF-PSNR) on faces, but not on random high-freq noise.

**Ablation (Motion Corruption)** — On a scene with simple translational motion (e.g., a car moving left), corrupting flow by Gaussian noise (σ=0.1) will cause small detail errors, but on scenes with complex motion (e.g., a dancer), the same corruption will significantly degrade detail prediction. We expect a larger HF-PSNR drop (e.g., >2 dB) for complex motion clips compared to simple motion (<0.5 dB drop), indicating sensitivity to flow accuracy.

### What would falsify this idea
If our method's HF-PSNR improvement over Direct Latent Diffusion is uniform across high-motion and low-motion clips, then the decomposition into base and detail is not the key factor, contradicting our claimed mechanism. Additionally, if Motion Corruption ablation shows no correlation between motion complexity and performance drop, the deterministic assumption is unsupported.

## References

1. MALT Diffusion: Memory-Augmented Latent Transformers for Any-Length Video Generation
2. Taming Teacher Forcing for Masked Autoregressive Video Generation
3. WorldSimBench: Towards Video Generation Models as World Simulators
4. Emu3: Next-Token Prediction is All You Need
5. EvalCrafter: Benchmarking and Evaluating Large Video Generation Models
6. Generative Multimodal Models are In-Context Learners
7. VideoPoet: A Large Language Model for Zero-Shot Video Generation
8. Video World Models with Long-term Spatial Memory
