# Occlusion-Aware Confidence-Guided Flow for Robust Motion Transfer

## Motivation

Motion4Motion (2026) extracts motion flow via off-the-shelf optical flow, but such flow is unreliable under occlusions and complex backgrounds, leading to artifacts in transferred motion. The root cause is that standard optical flow methods assign equal weight to all pixels, including occluded regions, causing erroneous warping. This gap persists because prior methods treat optical flow as a deterministic estimate without reliability modeling.

## Key Insight

By learning an auxiliary confidence prediction head that estimates per-pixel flow uncertainty, we can modulate the contribution of each flow vector during motion transfer, effectively suppressing unreliable regions while preserving accurate motion in visible areas.

## Method

**Title**: Occlusion-Aware Confidence-Guided Flow for Robust Motion Transfer

**Motivation**: Motion4Motion (2026) extracts motion flow via off-the-shelf optical flow, but such flow is unreliable under occlusions and complex backgrounds, leading to artifacts in transferred motion. The root cause is that standard optical flow methods assign equal weight to all pixels, including occluded regions, causing erroneous warping. This gap persists because prior methods treat optical flow as a deterministic estimate without reliability modeling.
**Key insight**: By learning an auxiliary confidence prediction head that estimates per-pixel flow uncertainty, we can modulate the contribution of each flow vector during motion transfer, effectively suppressing unreliable regions while preserving accurate motion in visible areas.

**[Current method]**
(A) **What it is**: We propose OCFlow (Occlusion-Aware Confidence Flow), a neural network that takes two consecutive video frames as input and outputs both a dense optical flow field and a per-pixel confidence map indicating the reliability of each flow vector. The confidence map is then used to weight the flow during warping in training-free motion transfer frameworks like Motion4Motion.

(B) **How it works**:
```python
# OCFlow architecture
import torch
import torch.nn as nn

def ocflow(frames: torch.Tensor) -> tuple:
    """
    frames: [B, 2, 3, H, W] (two consecutive frames)
    returns: flow [B, 2, H, W], conf [B, 1, H, W]
    """
    # Shared encoder (ResNet-18 output stride 8, pretrained on ImageNet, frozen except last block)
    feat = encoder(frames)  # [B, 512, H/8, W/8]
    # Two heads
    flow_head = nn.Conv2d(512, 2, kernel_size=3, padding=1)  # [B, 2, H/8, W/8]
    conf_head = nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, padding=1),
                              nn.Sigmoid())  # [B, 1, H/8, W/8]
    # Upsample to full resolution via bilinear interpolation
    flow = nn.functional.interpolate(flow_head, scale_factor=8, mode='bilinear', align_corners=False)
    conf = nn.functional.interpolate(conf_head, scale_factor=8, mode='bilinear', align_corners=False)
    return flow, conf

# Training loss
# Synthetic dataset: FlyingThings3D frames with ground-truth optical flow and occlusion masks.
# Occlusion masks are derived from depth maps: pixels that are visible in first frame but not in second (or vice versa) are occluded.
# We also apply random rectangular occlusions (size 10-50px) to 20% of frames to mimic object occlusions.
# Losses:
# L_flow = endpoint_error(flow * mask_non_occluded, gt_flow * mask_non_occluded) / sum(mask_non_occluded)
# L_conf = binary_cross_entropy(conf, occlusion_mask)  # occlusion_mask = 1 for non-occluded, 0 for occluded
# L = L_flow + 0.1 * L_conf
```
**Hyperparameters**: 
- Encoder: ResNet-18 pretrained on ImageNet, frozen except last residual block (Block4, stride 1).
- Learning rate: 1e-4, Adam optimizer (β1=0.9, β2=0.999).
- λ = 0.1 for confidence loss weight.
- Batch size: 16.
- Training iterations: 200k steps on FlyingThings3D (with random occlusions applied on the fly).
- Implementation: PyTorch 2.0, 4× NVIDIA A100 GPUs, training time ~8 hours.

**Inference**: During motion transfer, the source video frames are passed through OCFlow. The obtained flow and conf are used in Motion4Motion's warping: target motion = sum over source pixels (conf * flow) / sum(conf), with normalization to preserve energy.

(C) **Why this design**: We chose a two-headed architecture sharing a common encoder over separate encoders to reduce parameters and encourage shared feature learning, accepting a small capacity penalty for independent head specialization. We used synthetic data with known occlusion masks for training because real occlusion labels are expensive; this allows supervised confidence learning but introduces a domain gap to real videos. We placed a sigmoid on the confidence head to output probabilities between 0 and 1, enabling direct probabilistic interpretation, but this forces the network to be calibrated; we rely on data augmentation (random occlusions, noise) to bridge the domain gap. The training loss is a weighted combination: we use endpoint error only on non-occluded pixels for flow to avoid penalizing flow at occlusion boundaries, while confidence is trained with binary cross-entropy against the occlusion mask; the weighting λ=0.1 balances the two tasks. We chose λ=0.1 empirically to prioritize flow accuracy while ensuring confidence learning; a higher λ would cause the network to focus too much on confidence at the expense of flow quality.

(D) **Why it measures what we claim**: The confidence map output by conf_head measures the reliability of the flow estimate (i.e., the probability that the flow is accurate) because we train it to predict the ground-truth occlusion mask; this equivalence relies on the assumption that occlusion is the primary source of flow unreliability on real videos. **Load-bearing assumption**: The confidence map learned from synthetic occlusion masks generalizes to real-world videos where flow unreliability arises from multiple sources (occlusion, motion blur, large displacements). This assumption holds if synthetic occlusions share edge statistics and motion boundaries with real ones; it fails when real occlusions are caused by motion blur or large displacements beyond the optical flow search range. To verify this assumption, we evaluate confidence calibration on two real occlusion-heavy datasets: DAVIS (object-level occlusion) and YouTube-VOS (complex occlusion). For each dataset, we compute the Spearman rank correlation between predicted confidence and forward-backward consistency error (as a proxy for flow reliability) on real video frames. If ρ > 0.5, the assumption is supported; if not, we note that confidence may reflect synthetic bias rather than true reliability. In that case, the confidence map still acts as a learned attention mechanism that suppresses regions with low confidence, but its grounding in occlusion is weaker.

**Implementation details**: Code and pretrained models will be released on GitHub upon publication.

## Contribution

(1) A novel occlusion-aware optical flow estimator (OCFlow) that jointly predicts dense flow and per-pixel confidence, trained end-to-end on synthetic data with occlusion labels. (2) A confidence-guided motion transfer pipeline that weights flow contributions by estimated reliability, compatible with training-free frameworks like Motion4Motion, enabling robust motion extraction under partial observability. (3) A synthetic dataset with occlusion annotations for training and evaluation, together with design principles for learning confidence from occlusion masks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Human3.6M with artificial occlusions | Tests occlusion robustness in motion transfer. |
| Additional datasets | DAVIS (2017), YouTube-VOS (2018) | Real occlusion-heavy videos for confidence calibration. |
| Primary metric | Fréchet Video Distance (FVD) | Captures video realism and motion fidelity. |
| Additional metric | Spearman ρ between confidence and forward-backward consistency error | Measures confidence calibration on real occlusions. |
| Baseline 1 | Motion4Motion (original) | Standard flow warping without confidence. |
| Baseline 2 | Skeleton-based method | Requires skeleton supervision, no occlusion reasoning. |
| Baseline 3 | RAFT (state-of-the-art flow) | High accuracy but no confidence prediction. |
| Baseline 4 | FlowNetC with confidence head (trained same way) | Existing confidence-aware flow method. |
| Ablation-of-ours | OCFlow w/o confidence head | Confidence weighting removed. |

### Why this setup validates the claim
This experimental design provides a falsifiable test of whether OCFlow's confidence-aware warping improves motion transfer under occlusion. The Human3.6M dataset with artificial occlusions simulates common real-world failures, allowing us to isolate occlusion-induced artifacts. Comparing against Motion4Motion (same framework but raw flow) directly tests the benefit of confidence weighting. The skeleton-based baseline probes the advantage of bypassing structural priors, while RAFT represents the best optical flow without confidence. The ablation variant confirms that the confidence head itself drives any gains, not the shared encoder. FVD is the correct metric because it measures both perceptual quality and temporal consistency, which are degraded by incorrect warping at occlusions. The additional calibration metric (Spearman ρ) on DAVIS and YouTube-VOS grounds the load-bearing assumption that synthetic confidence generalizes to real occlusions. If OCFlow reduces FVD selectively on occluded sequences and shows ρ > 0.5 on real data, the core claim is supported.

### Expected outcome and causal chain

**vs. Motion4Motion (original)** — On a case where a person’s hand occludes their torso during motion, Motion4Motion uses raw RAFT flow that is noisy at occlusion boundaries, causing ghosting artifacts in the transfer. Our method predicts low confidence for the occluded hand region, downweighting its flow during warping, so the torso appears clean. We expect a noticeable FVD gap on frames containing occlusions (e.g., 10-15% lower FVD) but parity on frames without occlusions.

**vs. Skeleton-based method** — On a case where the source is a non-human object (e.g., a flag waving), skeleton-based methods fail because they cannot extract a skeletal structure. Our method relies only on dense flow, so it transfers motion naturally. We expect OCFlow to achieve near-human-level FVD while the skeleton baseline collapses (very high FVD) on such sequences.

**vs. RAFT (state-of-the-art flow)** — On a case of large motion with occlusions (e.g., a dancer spinning), RAFT estimates flow for occluded pixels via interpolation, but these estimates are unreliable. Our confidence head assigns low weight to those pixels, preventing distortion. We expect OCFlow to outperform RAFT only on occluded regions (e.g., lower per-pixel error in occluded areas) while being similar in non-occluded areas.

**vs. FlowNetC with confidence** — FlowNetC also has a confidence head but is trained on synthetic data without specific occlusion supervision. OCFlow's confidence is directly supervised on occlusion masks, leading to better calibration. We expect OCFlow to achieve higher Spearman ρ (>0.6) on DAVIS compared to FlowNetC (<0.4), and correspondingly lower FVD on occluded sequences.

### What would falsify this idea
If OCFlow’s FVD improvement over Motion4Motion is uniform across all frames (not concentrated on frames with occlusions), the confidence map is not acting as intended—suggesting the gain comes from better flow, not occlusion handling. Alternatively, if OCFlow performs worse on non-occluded frames, confidence weighting harms accuracy. Additionally, if Spearman ρ on DAVIS or YouTube-VOS is below 0.3, the confidence is not calibrated to real occlusions, undermining the occlusion-aware claim.

## References

1. Motion4Motion: Motion Transfer Across Subjects at Inference
2. MotionStream: Real-Time Video Generation with Interactive Motion Controls
3. MotionV2V: Editing Motion in a Video
4. Motion2Motion: Cross-topology Motion Transfer with Sparse Correspondence
5. Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance
6. Motion Prompting: Controlling Video Generation with Motion Trajectories
7. Pose‐to‐Motion: Cross‐Domain Motion Retargeting with Pose Prior
8. Skinned Motion Retargeting with Dense Geometric Interaction Perception
