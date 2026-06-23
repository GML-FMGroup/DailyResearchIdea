# OccLearn: Learning Occlusion-Aware Layered 3D Representations from Unconstrained Video

## Motivation

Current video world models (e.g., FantasyWorld) rely on frozen video backbones that collapse depth ordering because they lack explicit occlusion modeling. In dynamic scenes, the pre-trained features cannot distinguish between foreground objects and background when they overlap, leading to inconsistent geometry over time. This limitation prevents geometry-consistent long-term video generation without per-scene optimization.

## Key Insight

Occlusion boundaries produce a measurable signal in optical flow discontinuities, which are a sufficient statistic for inferring depth ordering without 3D supervision.

## Method

Input: video frames V = {I_t}, t=1..T
Output: For each frame, per-pixel depth ordering D_t and per-layer 3D feature bank L

1. For each adjacent pair (I_t, I_{t+1}), compute optical flow F_t (using RAFT) and occlusion mask O_t (using forward-backward consistency, threshold 0.5). Also compute a confidence map C_t = exp(-|forward-backward error|) normalized to [0,1].
2. For each frame, compute a segmentation map S_t using Mask2Former (pretrained on COCO) to obtain object-level masks.
3. For each pixel, define depth ordering prior from optical flow: if pixel at (x,y) in I_t is occluded in I_{t+1} (i.e., O_t(x,y) > 0.5), then it is likely behind the occluding pixel. Construct a pairwise depth ordering matrix R_t where R_t(i,j)=1 if pixel i is in front of pixel j based on flow discontinuities.
4. Train a lightweight depth ordering network D_θ (a 4-layer CNN with output dimension 1 per pixel) that takes I_t and S_t as input and predicts per-pixel depth ordering scores Z_t. Use a confidence-weighted contrastive loss:
   L_contrast = E_{i,j from R_t} [ -log( exp(Z_t(i) - Z_t(j)) / (exp(Z_t(i)-Z_t(j)) + exp(Z_t(j)-Z_t(i)) ) ] * C_ij
   where C_ij = min(C_t(i), C_t(j)) is the confidence weight for pixel pair (i,j). Only enforce pairs with C_ij >= 0.5; ignore pairs below threshold.
5. To learn 3D representations, also train a representation network R_ϕ (a ResNet-18 encoder) that encodes each segmentation mask into a 3D feature vector (dim 128). For each segment in S_t, extract its feature and store in a memory bank M. Use a contrastive loss (InfoNCE, temperature τ=0.07) between features of the same segment across frames (tracked via optical flow) to ensure consistency, and enforce dissimilarity between features of mutually occluded segments.
6. The depth ordering network and representation network are trained jointly with L_total = L_contrast + λ * L_feat_consistency, where λ=0.1, on unconstrained video. Training uses Adam optimizer, learning rate 1e-4, batch size 8, for 100k iterations on 4 NVIDIA A100 GPUs (estimated 48 hours).

**Assumption A (load-bearing):** Flow-based occlusion (forward-backward consistency) is a sufficient and reliable signal for inferring depth ordering in video frames. This holds under rigid motion but fails for lighting changes, transparency, or textureless regions. The confidence weighting in step 4 mitigates unreliable pairs.

(C) **Why this design**: We chose flow-based occlusion as the primary depth signal over stereo or depth sensors because it is available for any video without specialized hardware, accepting the cost that flow may be noisy in textureless regions or large motions. We used a pretrained segmentation model (Mask2Former) to aggregate pixels into meaningful objects, trading off the assumption of good segmentation for reduced learning complexity; if segmentation fails, depth ordering might group disparate objects. The confidence-weighted contrastive loss formulation enforces relative ordering only on reliable pixel pairs, avoiding spurious gradients from flow errors. We opted for a separate representation network rather than jointly predicting depth and features to allow modular improvement, though this may miss cross-task interactions. The memory bank with contrastive tracking ensures temporal consistency while avoiding explicit 3D reconstruction.

(D) **Why it measures what we claim**: The flow-based occlusion mask O_t measures occlusion boundaries (a proxy for depth ordering) because optical flow consistency is physically violated at occlusion edges; this assumption (Assumption A) fails when motion is due to lighting changes or transparent objects, in which case O_t captures appearance changes rather than geometry. The segmentation map S_t measures object-level consistency because it groups pixels likely belonging to the same 3D entity; this assumption fails when segmentation oversegments or undersegments, leading to incorrect layer assignments. The confidence-weighted contrastive loss on predicted depth scores Z_t measures depth ordering consistency because it directly enforces that the relative scores match the flow-derived ordering on reliable pairs; the loss fails when both pixels have low confidence and are ignored, reducing effective supervision. The feature bank contrastive loss measures 3D consistency across time because it encourages features of the same tracked segment to be similar; this assumption fails when tracking drifts, causing different objects to be matched incorrectly.

## Contribution

(1) A self-supervised framework OccLearn that learns occlusion-aware layered 3D representations from unconstrained video without any 3D supervision or per-scene optimization. (2) A training objective that jointly optimizes depth ordering and per-layer 3D features using flow-derived occlusion cues and segmentation-based grouping. (3) Demonstrates that flow discontinuities are a sufficient signal for depth ordering in dynamic scenes, enabling geometry-consistent long-term generation without pre-trained 3D features.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CLEVRER with object occlusions | Synthetic ground truth depth and object masks. |
| Primary metric | Ordinal Depth Accuracy (OWARP) | Directly measures depth ordering performance. |
| Baseline1 | Autoregressive Video Predictor (VideoGPT) | Tests necessity of explicit 3D structure. |
| Baseline2 | Pixel-wise Depth Ordering (4-layer CNN, same architecture as D_θ, no segmentation) | Tests benefit of segmentation grouping. |
| Ablation-of-ours | OccLearn w/o feature consistency (λ=0) | Isolates effect of 3D feature constraints. |
| Compute | 4 NVIDIA A100 GPUs, 48 hours | Ensures reproducibility. |

### Why this setup validates the claim

The proposed method claims that learning depth ordering from flow occlusion and enforcing 3D feature consistency produces a world model that understands 3D structure. CLEVRER provides ground truth depth and object masks, enabling direct evaluation of depth ordering accuracy. The Autoregressive Video Predictor (VideoGPT) baseline tests whether explicit 3D reasoning is necessary, while Pixel-wise Depth Ordering (PixDO) tests the value of segmentation grouping. The ablation without feature consistency isolates the contribution of temporal 3D feature constraints. OWARP is chosen because it measures ordinal depth errors, which directly reflect the quality of the learned depth ordering. If our method outperforms VideoGPT and PixDO, and the ablation shows degradation, then the causal role of each component is validated. Additionally, we compute correlation between flow-based ordering (from O_t) and ground truth depth ordering to empirically confirm Assumption A on CLEVRER.

### Expected outcome and causal chain

**vs. Autoregressive Video Predictor (VideoGPT)** — On a case where a small object is occluded by a larger one, VideoGPT predicts the occluded object's location by extrapolating appearance, often failing to infer its true depth order because it lacks occlusion reasoning. Our method uses flow-based occlusion to infer depth ordering, so we expect a significant gap in OWARP on occluded regions (e.g., >20% improvement), with near-parity on unoccluded ones.

**vs. Pixel-wise Depth Ordering (PixDO)** — On a case where an object's pixels have varying flow due to articulation, pixel-wise ordering may produce inconsistent depth scores for the same object, while our method aggregates via segmentation, enforcing object-level consistency. We expect our method to have higher accuracy on object boundaries and internal consistency, leading to better OWARP (e.g., >10% improvement) on scenes with articulated motion.

### What would falsify this idea

If OccLearn's OWARP improvement is uniform across all scene configurations, rather than concentrated on scenes with reliable flow and segmentation (e.g., high confidence pairs), then the central claim of targeted benefit from confidence-weighted occlusion reasoning is wrong.

## References

1. Video World Models with Long-term Spatial Memory
2. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
3. VidTok: A Versatile and Open-Source Video Tokenizer
4. FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction
5. MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion
6. Wonderland: Navigating 3D Scenes From a Single Image
7. World-consistent Video Diffusion with Explicit 3D Modeling
8. Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning
