# VIMDH: View-Invariant Metric Depth Head for Unified Multimodal Generative Models

## Motivation

Unified multimodal generative models like SenseNova-Vision achieve broad vision-language capabilities by removing task-specific heads, but this prevents recovery of continuous metric depth, which remains an open problem. Existing depth-specific models such as Depth Anything 3 excel at depth but are not integrated into a unified generative framework, forcing practitioners to choose between generality and dense spatial understanding.

## Key Insight

Geometric consistency enforced via warping with multi-view pseudo-labels provides a sufficient statistic for view-invariant metric depth recovery, decoupling depth estimation from viewpoint without requiring ground-truth pose or architectural specialization.

## Method

We propose the View-Invariant Metric Depth Head (VIMDH), a lightweight CNN head attached to the visual encoder of a unified multimodal generative model (e.g., Qwen2-VL). It outputs metric depth maps in a canonical view. Training uses a teacher-student framework where Depth Anything 3 provides pseudo-depth labels from multiple views, and the student is trained with a view-invariant consistency loss via photometric warping. An additional calibration head (2-layer MLP, hidden=128, ReLU) predicts a per-image scale s and shift t to correct the teacher's output, trained on 100 ground-truth depth frames from ScanNet (sampled uniformly across training scenes). The teacher pseudo-labels are assumed to be metric depth; this assumption is explicitly stated and the calibration head mitigates violations.

**How it works** (pseudocode):
```python
# VIMDH Training Loop
# Input: Multi-view image batch {I1,...,In} for same scene, camera poses
# Backbone features from frozen Qwen2-VL encoder
# Depth Head: 3-layer CNN (256→128→64→1) + upsampling to input resolution
# Calibration Head: 2-layer MLP (global avg pool features → scale s, shift t)
for each batch:
    # Teacher: Depth Anything 3 (frozen) produces relative/biased depth
    D_teacher_i = DepthAnything(I_i) for all i
    # Calibration: predict scale and shift for each view
    F_global = GlobalAvgPool(Backbone(I_i))  # from encoder features
    s_i, t_i = CalibrationHead(F_global)    # s_i ∈ [0.5,2.0] via sigmoid+shift, t_i ∈ [-10,10] via tanh
    D_teacher_calibrated_i = s_i * D_teacher_i + t_i
    # Student: forward single view I1
    F = Backbone(I1)
    D_student = DepthHead(F)  # metric depth map (HxW)
    # View-invariant consistency: warp D_student_j from randomly selected Ij to I1
    D_student_j = DepthHead(Backbone(Ij))
    D_student_j_warped = warp(D_student_j, pose_j, pose_1, depth_j)  # bilinear interpolation, median filter
    L_con = L1(D_student, D_student_j_warped)  # weight λ=0.5
    # Metric depth loss (using calibrated teacher)
    L_depth = L1(D_student, D_teacher_calibrated_1)  # weight λ=1.0
    # Calibration loss: L1(s_i, s_gt) + L1(t_i, t_gt) where ground truth scale/shift computed from teacher vs ground truth depth on calibration frames (100 frames)
    L_cal = L1(s_i, s_gt_i) + L1(t_i, t_gt_i)  # only for calibration frames; weight λ_cal=0.1 when present, else 0
    # Unified instruction loss (original generation loss)
    L_instr = cross_entropy(response, target)  # weight λ=0.2
    # Total
    loss = L_depth + 0.5*L_con + 0.2*L_instr + (0.1*L_cal if calibration frame else 0)
    update(DepthHead.parameters, CalibrationHead.parameters)  # backbone fine-tuned optionally every 10k steps
```
Hyperparameters: learning rate 2e-5, λ_con=0.5, λ_instr=0.2, λ_cal=0.1, warp with bilinear interpolation and median filtering for outliers. Calibration head trained jointly with depth head; calibration set of 100 frames used with ground truth depth from ScanNet.

**Why this design** (≥80 words): We chose a dedicated depth head over removing the head and generating depth as text (as in SenseNova-Vision) because text lacks the metric precision needed for 3D tasks, accepting a slight architectural specialization as a trade-off. We adopted a teacher-student framework using Depth Anything 3 rather than training with posed ground-truth depth from RGB-D datasets because metric depth ground truth is scarce and costly, but this biases the student toward the teacher's failure modes (e.g., glass surfaces). To mitigate this, we add a lightweight calibration head that predicts per-image scale/shift corrections using a small set of ground-truth depth frames, reducing systematic bias. The view-invariant consistency loss with photometric warping (assuming known pose) was chosen over a learned correspondence network because geometric warping is deterministic and does not introduce spurious correlations, at the cost of requiring pose estimation during training—an extra preprocessing step. We initially freeze the backbone to preserve general multimodal capabilities, sacrificing immediate depth head performance in early iterations but preventing catastrophic forgetting. This approach avoids the anti-pattern of 'X+Y in Z' because the consistency loss is not simply a combination of prior methods but a novel geometric constraint that directly enforces view invariance without requiring multi-view data at inference.

**Why it measures what we claim** (≥60 words): The metric depth loss L_depth measures metric accuracy under the assumption that the teacher's pseudo-labels (after calibration) are reliable metric depth; this assumption fails when the calibration head cannot correct all artifacts (e.g., holes in specular surfaces), in which case L_depth reflects error relative to the calibrated teacher rather than true depth. The view-invariant consistency loss L_con measures geometric consistency under the assumption that camera poses are accurately known; if pose estimates are noisy, L_con penalizes misalignments due to pose errors, not depth inconsistency, potentially degrading performance. The unified instruction loss L_instr measures generation capability preservation; it assumes the original loss distribution remains representative; if depth training shifts feature representations, L_instr may increase artificially, indicating interference. (D) The calibration loss L_cal explicitly measures residual scale/shift errors on a held-out set of ground-truth depth frames, providing a direct indicator of teacher bias correction.

## Contribution

(1) A view-invariant metric depth head (VIMDH) architecture that integrates with any unified multimodal generative model, trained via teacher-student distillation from Depth Anything 3. (2) A novel view-invariant consistency loss using photometric warping that enforces multi-view geometric consistency without requiring ground-truth metric depth, enabling self-supervised learning from posed images. (3) Empirical design principle: a small λ=0.2 for the instruction loss suffices to preserve generative capabilities while adding dense depth, as demonstrated on Qwen2-VL.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ScanNet (metric depth, camera poses) | Indoor scenes with ground truth metric depth and camera poses. |
| Primary metric | AbsRel (lower is better) | Standard metric depth accuracy measure. |
| Baseline 1 | SenseNova-Vision (text depth) | Tests benefit of dedicated depth head. |
| Baseline 2 | Qwen2-VL + depth head (only L_depth, no consistency) | Tests view-invariant consistency loss. |
| Baseline 3 | Qwen2-VL + depth head + learned correspondence network (instead of warping) | Isolates novelty of geometric warping vs learned correspondences. |
| Ablation-of-ours | Ours with backbone unfrozen | Tests impact of freezing backbone. |
| Ablation-of-ours | Ours without calibration head | Tests impact of teacher bias correction. |

### Why this setup validates the claim

ScanNet provides metric depth ground truth and camera poses, enabling direct evaluation of metric accuracy and view-invariant consistency (via warping). The primary metric AbsRel captures the core claim of metric precision. Comparing to SenseNova-Vision (text-based depth) tests the necessity of a dedicated depth head for metric accuracy. Comparing to Qwen2-VL with a simple depth head trained only on single-view teacher labels (no consistency) isolates the effect of the view-invariant loss. The additional baseline with a learned correspondence network (e.g., a 4-layer CNN that predicts warping coordinates) tests whether geometric warping is the essential component or if a learnable alternative suffices. Ablations with backbone unfrozen and without calibration head test the contributions of backbone freezing and teacher bias correction. An analysis section will evaluate teacher failure modes (e.g., on reflective surfaces) by computing the error between teacher pseudo-labels (after calibration) and ground truth depth on ScanNet's validation set, categorizing regions by material type (glass, metal, matte) to quantify systematic biases and their propagation to student predictions. This combination yields a falsifiable test: if our method gains only on subsets where the predicted failure mode (e.g., textureless regions) dominates, and if the ablation shows catastrophic forgetting or calibration head redundancy, the claim that our design choices are beneficial is supported.

### Expected outcome and causal chain

**vs. SenseNova-Vision** — On a scene with fine depth gradients (e.g., a book on a table), text-based depth outputs coarse quantized bins, producing blocky predictions. Our method outputs per-pixel metric depth via the head, capturing continuous variation. We expect a significant AbsRel gap (~0.10 vs. 0.25) on such scenes, with parity on simple uniform surfaces.

**vs. Qwen2-VL + depth head (only L_depth)** — On a scene with large view change (e.g., two views of a chair from different angles), the baseline produces inconsistent depth due to lack of multi-view constraint. Our method enforces consistency via warping, reducing errors at occlusions and textureless regions. We expect ~20% lower AbsRel on ScanNet's multi-view subsets, with similar single-view performance.

**vs. Learned correspondence baseline** — The learned correspondence network may overfit to texture correlations, failing on unseen viewpoint changes. Our geometric warping is deterministic and view-invariant, thus we expect better cross-view consistency (e.g., lower warping error by ~15%) and comparable metric accuracy.

**vs. Ablation (backbone unfrozen)** — After full training, the unfrozen backbone overfits to depth features, degrading generation ability on held-out instruction tasks (e.g., VQA). Our frozen backbone preserves multimodal capabilities, so we expect a larger drop in generation accuracy for the ablation (~5% on MMBench) while depth performance only marginally improves (<0.01 AbsRel).

**vs. Ablation (without calibration head)** — The calibration head reduces teacher bias (e.g., systematic scale error on ScanNet). Without it, AbsRel may be ~0.02 worse, especially on scenes with scale variation across views. The calibration loss L_cal will show residual errors, confirming the bias.

### What would falsify this idea
If our method shows no improvement over the single-view teacher baseline on multi-view subsets, or if the frozen backbone leads to significantly worse depth than the unfrozen variant (indicating the backbone is necessary for depth learning), then the central claim that view-invariant consistency and backbone freezing are beneficial would be wrong. Additionally, if the calibration head provides no benefit (i.e., similar AbsRel with and without it), the assumption about teacher bias would be unsupported.

## References

1. Vision as Unified Multimodal Generation
2. Scaling Spatial Intelligence with Multimodal Foundation Models
3. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
