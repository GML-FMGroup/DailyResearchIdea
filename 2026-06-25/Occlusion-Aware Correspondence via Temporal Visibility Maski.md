# Occlusion-Aware Correspondence via Temporal Visibility Masking for Video Diffusion Models

## Motivation

MVTrack4Gen assumes that attention layers in video diffusion models encode reliable correspondences regardless of occlusion, leading to failures under severe occlusion or fast motion. This structural gap persists because no prior work introduces a mechanism to detect or handle occluded regions before using attention features for correspondence. We address this by explicitly masking occluded regions during correspondence extraction.

## Key Insight

Occluded regions can be identified by comparing forward-backward optical flow consistency in feature space, enabling a differentiable mask that isolates visible correspondences for attention.

## Method

## Method: OCC-Mask (Occlusion-Conscious Correspondence Mask)

**Components:**
- **Input:** Attention features \(F_t, F_{t+1}\) from layer 6 of the U-Net decoder of the video diffusion model (size [H/8, W/8, C]), and self-supervised optical flow fields \(\text{flow}_{t \to t+1}, \text{flow}_{t+1 \to t}\) computed from the same attention features using a DIFT-like procedure: we extract patch-level correspondences from attention maps (4th block of the U-Net), compute coarse offsets, then interpolate to full resolution via bilinear upsampling (yielding flow at stride 1).
- **Output:** Occlusion mask \(M_t \in [0,1]^{H \times W}\) (visibility weight per pixel).

**Procedure (pseudocode):**
```
# OCC-Mask: Occlusion-Aware Correspondence Masking
Input: F_t, F_{t+1}: attention features at frames t, t+1 (from U-Net layer 6, stride 8)
       flow_t->t+1, flow_{t+1}->t: self-supervised optical flow (stride 1, from DIFT-like procedure)
Output: mask M_t (visibility weight per pixel)

1. Warp F_t to frame t+1 using flow_t->t+1: F'_t = warp(F_t, flow_t->t+1)   # bilinear sampling
2. Compute feature similarity: S = cosine_similarity(F'_t, F_{t+1})  # [H,W] at stride 8
3. Compute forward-backward flow consistency error (at stride 1):
   E = || flow_t->t+1 + warp(flow_{t+1}->t, flow_t->t+1) ||_2  # [H,W]
4. Combine (at stride 1, then downsample to stride 8 using average pooling):
   M_t = sigmoid( α * (1 - E/τ_E) + β * (S - τ_S) )  # α=1.0, β=1.0, τ_E=0.5, τ_S=0.8
5. Apply mask to attention output (or tracking features) at stride 8:
   Z_t_masked = M_t ⊙ Z_t  # element-wise multiply
6. Temporal consistency loss: L_temp = mean( M_t * ||predicted_track - ground_truth_track||_2 )
```

**(C) Why this design:** We chose a flow-based occlusion cue over learned visibility prediction because flow consistency is a geometric invariant independent of the diffusion model's semantic features, avoiding the need for additional training data or supervision. The trade-off is that flow errors can occur at motion boundaries or fast motion, which may misclassify some visible regions as occluded; we compensate by also using feature similarity, which is robust to small misalignments. We combine both signals via a weighted sigmoid rather than a learned classifier because the thresholds (τ_E, τ_S) can be set heuristically without extra parameters, accepting a slight sensitivity to scene dynamics. We apply the mask directly to attention features rather than to tracking head outputs because masking early prevents corrupted features from propagating through the tracking head, at the cost of potentially discarding useful context from occluded regions—but in correspondence tasks, only visible points are reliable. Finally, we use a per-pixel mask rather than a binary hard mask because soft weighting allows the loss to gracefully handle ambiguous cases (e.g., partial occlusion). The complexity of managing two heuristics (flow consistency and feature similarity) and their thresholds is a design cost; however, we believe the simplicity and interpretability outweigh the need for learning.

**(D) Why it measures what we claim:** 
- **Feature similarity \(S\)** measures **correspondence reliability** because high similarity between warped and target features indicates a stable, geometrically consistent match when the flow is correct; this assumption fails when the flow itself is inaccurate (e.g., large disocclusion), in which case S reflects texture similarity rather than geometric correspondence.
- **Flow consistency error \(E\)** measures **occlusion** because a violation of forward-backward consistency implies that a pixel visible in t is not visible in t+1 (or vice versa); this assumption holds for rigid scenes with Lambertian appearance, but fails when there is specular reflection or transparency, where flow consistency may be violated even without occlusion, producing false positive occlusion signals.
- **Combined mask \(M = \text{sigmoid}(\alpha(1-E/\tau_E) + \beta(S-\tau_S))\)** measures **visibility for correspondence** because it gives high weight only when both flow consistency is high (E small) and feature similarity is high (S large), thus only pixels that are both geometrically consistent and radiometrically similar are considered visible; this joint assumption is violated when camera motion causes large appearance changes (e.g., lighting changes), where a visible point may have low S but still be correspondable, leading to false negatives.
- **Temporal consistency loss \(\mathcal{L}_{\text{temp}}\)** measures **correspondence accuracy only on visible points** because it multiplies the per-pixel loss by M, ensuring occluded points do not contribute; this relies on the mask correctly identifying occlusion, and fails when the mask erroneously labels visible points as occluded, in which case the loss is prematurely reduced for those points, potentially allowing geometric drift.

**Load-bearing assumption (explicit):** The self-supervised flow computed from attention features (DIFT-like) provides accurate correspondences aligned with the diffusion model's feature space. We verify this by checking that the flow warps attention features to have high cosine similarity on non-occluded regions (S > 0.8) on a calibration set of 100 randomly selected video frames from the training dataset. If the average cosine similarity across calibration falls below 0.6, we retrain the flow interpolation (bilinear upsampling parameters) with a learning rate of 1e-4 for 500 steps using a self-supervised photometric loss on RGB frames.

## Contribution

(1) A novel occlusion-aware attention masking mechanism (OCC-Mask) that identifies visible correspondences by combining forward-backward flow consistency and feature similarity, without requiring additional training or supervision. (2) A temporal consistency loss that selectively enforces constraints only on visible points, preventing corrupted gradients from occluded regions during training. (3) Empirical demonstration that OCC-Mask improves correspondence accuracy and video synthesis quality under occlusion compared to baseline MVTrack4Gen.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Multi-view video dataset (e.g., MVSynth, 10 scenes, 100 frames each, 512x512) | Common benchmark for NVS with ground truth |
| Primary metric | Temporal warping error (TWE) | Measures correspondence consistency across views |
| Baseline 1 | MVTrack4Gen (tracking supervision) | Uses tracking but no occlusion mask |
| Baseline 2 | GS-DiT (3D tracking with SpatialTracker) | Explicit 3D-based tracking supervision |
| Ablation 1 | OCC-Mask w/o feature similarity (only flow consistency) | Tests contribution of similarity term |
| Ablation 2 | OCC-Mask w/o flow consistency (only feature similarity) | Tests contribution of geometric cue |
| Additional metric | Percentage of correctly tracked points (PCT) at 1-pixel threshold | Measures spatial accuracy of correspondences |

### Why this setup validates the claim

The central claim is that occlusion-aware masking improves correspondence supervision in video diffusion for novel view synthesis. By comparing against baselines without occlusion handling (MVTrack4Gen) and with explicit 3D tracking (GS-DiT), we test whether our lightweight plug-in can match or exceed specialized tracking. The ablations isolate the contributions of the flow consistency and feature similarity components. TWE directly measures how well correspondences hold under warping, making it sensitive to occlusion-induced errors. Additionally, PCT provides a finer-grained assessment of tracking accuracy. This combination of dataset, baselines, and metrics allows us to attribute any performance gain specifically to the occlusion mask design.

### Expected outcome and causal chain

**vs. MVTrack4Gen (tracking supervision)** — On a case where occlusion is common (e.g., fast motion with self-occlusion), the baseline produces noisy gradients from occluded points because it uses all correspondences equally. Our method instead masks out occluded points using flow consistency and feature similarity, so we expect a noticeable gap on high-occlusion scenes but parity on static scenes.

**vs. GS-DiT (3D tracking with SpatialTracker)** — On a case where 3D tracking fails due to sparse views, the baseline produces inaccurate 3D correspondences; our method uses dense 2D attention features and flow, avoiding 3D reconstruction failure. We expect comparable performance when 3D tracking is accurate, but our method should outperform on challenging multi-view setups.

**vs. OCC-Mask w/o feature similarity** — On a case where flow consistency is ambiguous (e.g., textureless regions), the ablation misclassifies some visible points as occluded due to low flow error; our full method uses feature similarity to rescue those points. We expect a small but consistent improvement on textureless areas.

**vs. OCC-Mask w/o flow consistency** — On a case where feature similarity is ambiguous (e.g., repetitive patterns), the ablation misclassifies some occluded points as visible due to high similarity; our full method uses flow consistency to reject those points. We expect improved precision in repetitive regions.

### What would falsify this idea

If our method shows no improvement over all ablations on any subset, or if it performs worse on static scenes where occlusion is rare, then the occlusion masking idea is not beneficial. Additionally, if the self-supervised flow produces consistently low consistency error even when occlusion is present (e.g., due to flow failure), the mask would fail to identify occlusion.

## References

1. MVTrack4Gen: Multi-View Point Tracking as Geometric Supervision for 4D Video Generation
2. GS-DiT: Advancing Video Generation with Dynamic 3D Gaussian Fields through Efficient Dense 3D Point Tracking
