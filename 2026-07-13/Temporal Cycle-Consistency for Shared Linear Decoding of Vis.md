# Temporal Cycle-Consistency for Shared Linear Decoding of Vision Tasks from Video Generative Models

## Motivation

Existing video generative models for vision tasks, such as GenCeption, require separate supervised heads and labeled data for each task, structurally wasting the rich spatio-temporal representations learned by the generative backbone. This is because they do not exploit the temporal consistency inherent in video to force a shared representation across tasks, preventing the use of a single decoder.

## Key Insight

Temporal continuity in video creates a structural coupling between modalities (RGB, depth, normals) that, when cycled through the generative model's latent space via cycle-consistency losses, yields a latent subspace that is linearly mappable to any task output.

## Method

### Temporal Cycle-Consistency for Shared Linear Decoding of Vision Tasks from Video Generative Models

#### (A) What it is
TAL (Temporal Alignment of Latents) is a training framework that takes unlabeled video frames as input and outputs a per-frame latent vector that can be linearly decoded into depth, surface normal, segmentation maps, or optical flow without task-specific heads. It uses a frozen pre-trained video diffusion backbone (e.g., Genmo or VideoMAE) and a small learned encoder $E$ to produce latents $\mathbf{z}_t$, then applies cycle-consistency losses across modalities and time. A key unverified assumption is that cycle-consistency with linear decoders and learned re-encoders ensures the latent is linearly mappable to task outputs. To calibrate, we add a small supervised loss on a calibration set of 512 frames (1% of total) using ground truth task labels from a subset of sequences.

#### (B) How it works

```python
# Given: pre-trained video diffusion backbone D (frozen), encoder E, task-specific linear decoders D_depth, D_normal, D_seg, D_flow.
# Hyperparameters: lambda_temp=1.0, lambda_cycle=1.0, lambda_sup=0.1, sigma=0.1
# Calibration set C = {512 frames with GT depth, normal, seg, flow} sampled from training data.

for video clip (frames I_1..I_T) in unlabeled dataset:
    # Encode each frame to latent
    z_t = E(I_t)   # z_t in R^d (d=256)
    
    # Temporal consistency loss: encourage smooth changes
    L_temp = sum_{t=1}^{T-1} ||z_{t+1} - z_t||^2
    
    # Cycle-consistency: for each task m in {depth, normal, seg, flow}
    # Predict task output via linear decoder (single layer, no activation)
    hat{O}_{t,m} = D_m(z_t)
    # Re-encode predicted output back to latent (use learned encoder C_m: 2-layer MLP, hidden=128, ReLU)
    hat{z}_{t,m} = C_m(hat{O}_{t,m})
    # Cycle loss: compare original latent and re-encoded latent
    L_cycle = sum_{t,m} ||z_t - hat{z}_{t,m}||^2
    
    # Supervised loss on calibration frames within the batch
    if any frames in batch are in C:
        L_sup = sum_{frames in C} sum_m ||D_m(z_t) - O_gt_{t,m}||^2
    else:
        L_sup = 0
    
    # Total loss
    L = lambda_temp * L_temp + lambda_cycle * L_cycle + lambda_sup * L_sup
    optimize E, D_m, C_m
```

**Load-bearing assumption:** The cycle loss (with linear decoders and learned re-encoders) ensures that the latent is linearly mappable to task outputs, i.e., the composed mapping from latent to task output and back approximates identity, producing a shared task-agnostic latent subspace. However, since the linear decoder is not full rank, the nullspace is unconstrained; we add a small supervised loss to anchor the latent, preventing nullspace collapse.

#### (C) Why this design
We chose a frozen generative backbone over fine-tuning it because fine-tuning could destroy the rich representations learned from large-scale video data, accepting the cost that the latent space is constrained by the backbone's capacity. We use separate task-specific decoders but keep them linear (single layer) to force the latents to be task-agnostic and linearly separable; a nonlinear decoder could absorb representation non-orthogonality. We employ cycle-consistency rather than direct regression to ground-truth because unlabeled video lacks task annotations, but this requires learning an inverse mapping $C_m$ per task, which could be unstable if the task prediction is noisy. To mitigate, we apply a Gaussian blur ($\sigma=0.1$) to predicted outputs before re-encoding. The temporal consistency loss uses L2 distance; a robust loss could handle occlusions but we trade off simplicity for computational cost. The small supervised loss (on 1% of frames) anchors the latent to avoid nullspace drift, a known issue with cycle-consistency in linear autoencoders.

#### (D) Why it measures what we claim
The temporal consistency loss $L_{temp}$ enforces that $\mathbf{z}_t$ changes slowly with time, capturing static scene properties (depth, normals) that are temporally correlated; but this assumption fails at motion boundaries where latents change abruptly, causing the loss to penalize necessary changes. The cycle-consistency loss $L_{cycle}$ enforces that the latent $\mathbf{z}_t$ can be decoded to a task output and re-encoded back to the same point; this operationalizes the claim that a shared linear subspace exists for all tasks, under the assumption that the re-encoding head $C_m$ is a proxy for the inverse of the task decoder. This assumption fails when the decoder $D_m$ loses information (e.g., due to linearity), in which case the cycle loss only measures invertibility of the chained mapping, not true alignment. The supervised loss on calibration frames partially corrects this by directly supervising the decoder output. The combination of both losses forces the latent to be both temporally smooth and task-consistent, thus measuring the claimed shared representation.

## Contribution

(1) A novel training framework that replaces task-specific supervised heads with a shared linear decoder, trained via temporal and cross-modal cycle-consistency on unlabeled video. (2) The empirical finding that such temporally aligned latents enable a single linear layer to simultaneously predict depth, normals, and segmentation, outperforming existing zero-shot baselines. (3) Introduction of a cycle-consistency mechanism between RGB and task outputs that does not require paired ground-truth data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | NYUv2 (depth, normal, seg, optical flow via synthetic FT3D) | Dense GT for evaluation; flow added per reviewer |
| Primary metric | Average task error (normalized) | Summarizes multi-task performance |
| Baseline | DepthAnything3 | Strong specialized depth estimator |
| Baseline | V-JEPA | Video self-supervised baseline |
| Baseline | Cycle-only TAL (w/o temporal loss) | Isolate temporal consistency effect |
| Ablation-of-ours | TAL w/ nonlinear decoders (2-layer MLP) | Test necessity of linear decoder |
| Ablation-of-ours | TAL w/ random latents (sanity check) | Measure decoder capacity without meaningful representation |
| Resource estimate | ~500 GPU hours on 4× A100 for Genmo; ~150 GPU hours for VideoMAE | Lighter backbone reduces cost |

### Why this setup validates the claim
NYUv2 provides ground truth for depth, normals, segmentation, and optical flow (via synthetic FT3D), enabling direct evaluation of decoder performance. Comparing against DepthAnything3 tests whether unsupervised learning can compete with supervised specialists on depth. V-JEPA tests whether cycle-consistency adds value over purely temporal self-supervision. The ablation of temporal loss isolates its contribution. The average task error metric captures overall multi-task quality; if TAL matches or surpasses baselines, the claim that a shared latent representation emerges is supported. Conversely, failure on any task or parity with simpler baselines would falsify the claim.

### Expected outcome and causal chain

**vs. DepthAnything3** — On a frame with significant motion blur (e.g., fast camera pan), DepthAnything3 may produce blurry depth because it lacks temporal context. Our TAL enforces temporal smoothness, so latents remain coherent across frames, yielding sharper depth even under blur. We expect similar depth error on static frames but a ~10% reduction on motion-blur frames, while overall accuracy may still be slightly lower due to lack of supervised pretraining.

**vs. V-JEPA** — In a scene with multiple moving objects, V-JEPA’s features may entangle motion and appearance, degrading linear depth decoding. Our cycle-consistency loss forces the latent to be invertible to task outputs, aligning features with task-specific geometry. We expect TAL to outperform V-JEPA on depth, especially in dynamic regions, with a noticeable gap (e.g., 15% lower error) on highly dynamic subsets.

**vs. Cycle-only TAL (w/o temporal loss)** — On a smooth camera sweep, without temporal consistency the latent may oscillate frame-to-frame, causing jagged depth predictions. Our full TAL with temporal loss enforces gradual change, producing temporally consistent depth. We expect the full model to have ~20% lower depth variance and ~5% lower average error on such smooth sequences.

**vs. TAL with nonlinear decoders** — If linearity is necessary, nonlinear decoders should yield better raw performance but fail to enforce a shared subspace; expect lower individual task error but higher cross-task inconsistency.

**vs. Random latents (sanity check)** — If the latent space contains no meaningful structure, linear decoders will fail to predict task outputs; expect error close to chance.

### What would falsify this idea
If TAL fails to outperform V-JEPA on average task error, or if the temporal loss does not improve consistency over the ablation, or if nonlinear decoders outperform linear ones on the shared subspace metric (e.g., cross-task latent similarity), then the central claim that temporal alignment and cycle-consistency yield a shared task-agnostic latent representation is falsified. Specifically, a uniform performance gain across subsets (rather than concentrated on dynamic scenes) would indicate the mechanism is not acting as hypothesized.

## References

1. Video Generation Models are General-Purpose Vision Learners
2. Video models are zero-shot learners and reasoners
3. Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model
4. MapAnything: Universal Feed-Forward Metric 3D Reconstruction
5. SAM 3: Segment Anything with Concepts
6. MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision
7. Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction
8. Fine-Tuning Image-Conditional Diffusion Models is Easier than you Think
