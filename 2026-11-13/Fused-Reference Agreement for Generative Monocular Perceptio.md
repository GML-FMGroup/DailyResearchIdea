# Fused-Reference Agreement for Generative Monocular Perception

## Motivation

Generative visual perception models such as OVB (Open Vision Banana) are trained by collapsing multi-sensor annotations into a single-view target before training, so the model is never explicitly asked to match a LiDAR–camera fused reference; modality consistency is therefore an uncontrolled side effect rather than a trained property. This structural gap means that under adverse conditions (glare, rain, low light) the generated mono-modal output can silently drift from the fused geometric evidence without any training signal penalizing the drift. The missing mechanism is a fused-reference agreement term that projects the LiDAR–camera fused reference into the camera view and supervises the generated output against it.

## Key Insight

Projecting the LiDAR–camera fused reference into the camera view and penalizing agreement between the generated mono-modal output and that projected reference makes the fused geometry an explicit training target, so modality consistency becomes a quantity the loss directly controls rather than an emergent byproduct of single-view supervision.

## Method

**What it is:** FARA (Fused-Reference Agreement) is a training-time auxiliary loss and augmentation scheme for generative visual perception editors. It takes as input a camera image, its paired LiDAR points with per-point semantic labels, and the editor's generated mono-modal output (e.g., a semantic map or depth map); it produces an agreement loss that is added to the standard generative reconstruction loss.

**Load-bearing assumption (stated explicitly):** FARA assumes that the projected LiDAR–camera fused reference F(u) is a cleaner, more accurate supervision target than the camera-view labels the generator already sees, so penalizing agreement with it strictly improves the generated output rather than teaching the editor to reproduce LiDAR projection artifacts (sparsity, occlusion/see-through, misregistration, reflective-surface noise). This assumption is verified before training (step 2d) and enforced during training by a confidence-weighted, artifact-aware target (step 2e) that down-weights pixels where projection artifacts are likely.

**How it works:**

```pseudocode
Input: camera image I, LiDAR points P with semantic labels L_P, camera intrinsics K, extrinsics T_cam_lidar
Editor E_theta (generative perception model)
Hyperparams: lambda_agree = 0.5, tau = 0.7 (agreement threshold), p_aug = 0.3 (adverse-condition prob),
             r = 2px (projection radius), min_pts = 3 (validity threshold), delta_depth = 0.5m (discontinuity threshold),
             intensity_lo = 0.1, intensity_hi = 0.3 (LiDAR return intensity ramp), w_min = 0.1 (minimum confidence weight)

1. Generate mono-modal output:  O_hat = E_theta(I)
2. Fuse reference:
   a. Project P into camera frame: p_cam = K * T_cam_lidar * P
   b. For each pixel u with projected LiDAR points, compute fused label
      F(u) = majority_vote({L_P(p) : p projects to u within radius r=2px})
      and fused depth D_f(u) = median({depth(p) : p projects to u})
   c. Mark valid pixels V = {u : |{p -> u}| >= min_pts=3}
   d. Verification of load-bearing assumption (run once on a held-out calibration set of 512 Cityscapes frames before training):
      compute accuracy of F(u) vs. camera-view label on V; if F(u) is not more accurate than the camera label on V, abort and report.
   e. Artifact-aware confidence weight (applied to every valid pixel u in V):
      w_depth(u) = 1 if local depth range within 3x3 window < delta_depth=0.5m else w_min=0.1
      w_int(u)   = clip((mean LiDAR return intensity at u - intensity_lo) / (intensity_hi - intensity_lo), w_min, 1)
      w(u)       = w_depth(u) * w_int(u)
      (Pixels near depth discontinuities, occlusion boundaries, and low-return specular/transparent surfaces receive w(u) <= w_min.)
3. Adverse-condition augmentation:
   With probability p_aug, apply a photometric corruption A (rain/glare/low-light) to I, producing I_aug; also run E_theta on I_aug to get O_hat_aug.
4. Agreement loss (on both clean and augmented outputs):
   For each valid pixel u in V:
     agree(u) = 1[O_hat(u) == F(u)]   (for segmentation)
              = exp(-|O_hat(u) - D_f(u)| / tau)  (for depth)
   L_agree = - (1/sum_{u in V} w(u)) * sum_{u in V} w(u) * log(agree(u) + eps)
   L_agree_aug = same on O_hat_aug
5. Total loss: L = L_gen + lambda_agree * (L_agree + L_agree_aug)
6. Update theta via gradient descent on L.
```

**Why this design:** We chose a projection-and-majority-vote fused reference over simply using the camera-view semantic labels because the fused reference injects LiDAR geometric evidence that is independent of the camera appearance, accepting the cost that projection introduces sparsity and requires a validity mask (V) to avoid penalizing unobserved pixels. We chose a soft agreement score (exp(-|Δ|/τ)) over a hard 0/1 match for depth because depth is continuous and hard thresholds produce zero gradients for small errors, accepting that τ must be tuned to the depth scale. We chose to apply the agreement loss to both clean and adversarially augmented outputs rather than only clean outputs because robustness must be a controlled property, accepting the extra forward pass cost during training. We chose a fixed λ_agree=0.5 rather than a learned weighting because a learned weight could down-weight the agreement term early in training and defeat its purpose, accepting that λ may need dataset-specific tuning. We added the artifact-aware confidence weight w(u) and the pre-training verification step (2d) because the load-bearing assumption—that F(u) is a cleaner target than camera labels—is not guaranteed: projection produces boundary bleeding and reflective-surface noise, so we down-weight exactly those pixels and verify the assumption empirically before training. A domain expert would not describe FARA as a variant of OVB because OVB trains only on single-view targets and has no fused-reference projection or adverse-condition agreement term; FARA adds a structurally different supervision signal that couples the generated output to multi-sensor geometry.

**Why it measures what we claim:** The projected fused label F(u) measures modality-consistent ground truth because the assumption that LiDAR points projecting to a pixel within radius r=2px carry the correct semantic/depth label for that pixel holds when calibration is accurate and the scene is rigid; this assumption fails under calibration drift or on highly reflective/transparent surfaces, in which case F(u) reflects LiDAR noise rather than true scene semantics. The agreement score agree(u) measures modality consistency between the generated mono-modal output and the fused reference because the assumption that a pixel-wise match implies the generator has internalized the fused geometry holds when the generator's output is pixel-aligned with the input image; this assumption fails when the generator produces spatially shifted outputs, in which case agree(u) reflects misalignment rather than inconsistency. The adverse-condition augmentation measures robustness as a controlled property because the assumption that the corruption A is representative of real adverse conditions holds when A is drawn from a realistic corruption distribution; this assumption fails when A is too synthetic, in which case L_agree_aug reflects overfitting to the synthetic corruption rather than genuine robustness. The confidence weight w(u) measures the reliability of the fused reference because the assumption that low LiDAR return intensity and local depth discontinuities flag projection artifacts holds when the artifact model (specular/transparent surfaces, occlusion boundaries) is correct; this assumption fails when artifacts occur at high-intensity rigid surfaces, in which case w(u) fails to down-weight them and the loss still rewards reproducing projection errors.

## Contribution

['A fused-reference agreement loss (FARA) that projects LiDAR–camera fused annotations into the camera view and explicitly supervises a generative perception editor to match the fused reference, turning modality consistency into a directly optimized training objective.', 'An adverse-condition augmentation scheme coupled to the agreement loss that makes robustness a controlled training property rather than an uncontrolled side effect of single-view supervision.', 'A concrete demonstration that adding FARA to a generative perception editor (OVB-style) improves agreement with fused multi-sensor reference under both clean and adverse conditions, without degrading standard generative perception performance.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cityscapes + adverse-condition corruption suite | Zero-shot semantic/depth with controllable robustness stress |
| Dataset (transfer) | ACDC (Adverse Conditions Dataset) | Real adverse conditions to validate corruption representativeness |
| Primary metric | mIoU on fused-reference-valid pixels | Directly measures agreement with multi-sensor geometry |
| Secondary metric | Depth RMSE on fused-reference-valid pixels | Tests depth sub-claim against metric reference |
| Baseline | Vision Banana (reported) | Establishes the generative-perception frontier to beat |
| Baseline | Affine-invariant depth SOTA | Tests depth sub-claim against strong mono-modal geometry |
| Baseline | OVB-style single-view training | Isolates value of fused-reference supervision |
| Baseline | Cross-modal distillation (LiDAR-projected pseudo-labels, no editor) | Tests whether gains require the generative editor |
| Baseline | Late-fusion segmentation network (camera + LiDAR) | Tests whether gains require the generative editor |
| Ablation-of-ours | FARA w/o adverse-condition agreement term | Tests whether robustness comes from the augmented branch |
| Ablation-of-ours | FARA w/o artifact-aware confidence weight w(u) | Tests whether down-weighting projection artifacts matters |
| Ablation-of-ours | FARA w/o pre-training verification (2d) | Tests whether the load-bearing assumption holds in practice |

### Why this setup validates the claim

The central claim is that coupling a generative editor to a LiDAR-fused reference, on both clean and corrupted inputs, yields outputs that are more modality-consistent and more robust than editors trained on single-view targets alone. Cityscapes with a corruption suite is the right testbed because it supplies calibrated camera-LiDAR pairs whose fused labels can be projected into the camera view, and it lets us separate clean-scene consistency from robustness under rain, glare, and low light. ACDC provides a held-out real adverse-condition test set to validate that the synthetic corruption distribution A is representative; if FARA's gains on ACDC match its gains on the synthetic suite, the robustness claim is grounded in real conditions rather than synthetic overfitting. Vision Banana tests the frontier claim: if our method beats a strong reported generative-perception model, the fused signal is doing real work. The affine-invariant depth estimator tests the depth sub-claim specifically, since it is a strong mono-modal geometry baseline that has never seen LiDAR fusion. The OVB-style baseline is the critical control: it shares our editor and training budget but lacks the projection-and-majority-vote reference, so any gap is attributable to the fused signal rather than to architecture or data. The cross-modal distillation and late-fusion baselines test whether the generative-editor-plus-agreement-loss combination yields gains not achievable by simply using fused labels as targets or by a discriminative fusion network. The ablation removing the augmented branch isolates whether robustness gains come from the adverse-condition agreement term rather than from the clean branch alone; the ablation removing w(u) tests whether the artifact-aware weighting is necessary to avoid reproducing projection artifacts; the ablation removing verification (2d) tests whether the load-bearing assumption holds empirically. mIoU on fused-valid pixels is the right metric because it directly measures agreement with the multi-sensor reference we are supervising toward, and because restricting to valid pixels prevents sparsity from masquerading as error.

### Expected outcome and causal chain

**vs. Vision Banana (reported)** — On a Cityscapes frame where a thin pole or distant sign is poorly lit, the baseline produces a smeared or missing class because its single-view supervision never received geometric evidence that the object occupies those pixels. Our method instead receives a majority-voted fused label from LiDAR points projecting within 2px, weighted by artifact-aware confidence w(u) so that only reliable projections drive the loss, so the agreement loss pushes the editor to preserve that structure. We expect a noticeable mIoU gap concentrated on thin, distant, and low-contrast objects, with parity on large well-lit regions.

**vs. Affine-invariant depth SOTA** — On a case where a reflective car hood or wet road creates a depth discontinuity, the baseline produces a smooth affine-invariant surface because it has no metric anchor and no fused geometric reference. Our method instead matches against the median projected LiDAR depth within the validity mask, but only where the artifact-aware confidence weight w(u) is high (i.e., away from specular/transparent surfaces and depth discontinuities), so the soft agreement term supplies a metric gradient exactly where the mono-modal estimator is ambiguous and LiDAR is reliable. We expect a clear improvement on reflective and transparent subsets, with parity on texture-rich rigid surfaces.

**vs. OVB-style single-view training** — On a rainy frame where glare washes out lane markings, the baseline produces confident but wrong segmentation because it was trained only on clean single-view targets and never saw a corrupted input paired with a geometry-anchored label. Our method instead runs the editor on the corrupted image and applies the same fused-reference agreement loss, so the LiDAR geometry acts as a corruption-invariant anchor. We expect a robustness gain that grows with corruption severity, and near-parity on clean frames.

**vs. Cross-modal distillation (LiDAR-projected pseudo-labels, no editor)** — On a frame where LiDAR projection produces boundary bleeding at object edges, the distillation baseline reproduces those artifacts because it treats every projected label as ground truth. Our method instead down-weights boundary and low-return pixels via w(u), so the editor learns to trust fused geometry only where it is reliable. We expect FARA to outperform distillation on boundary-heavy and reflective subsets, with parity on dense well-textured regions.

**vs. Late-fusion segmentation network (camera + LiDAR)** — On a frame where the camera is degraded but LiDAR is clean, the late-fusion network may still fail because it lacks a generative prior to fill in missing appearance. Our method instead uses the generative editor to produce a mono-modal output and supervises it against the fused reference, so the editor can hallucinate plausible structure where the camera fails while being anchored by LiDAR. We expect FARA to match or exceed late fusion on corrupted frames, with late fusion potentially stronger on clean frames where both sensors are reliable.

### What would falsify this idea

If the gain over the OVB-style baseline is uniform across clean and corrupted subsets rather than concentrated under adverse conditions, the augmented agreement branch is not delivering robustness and the central claim collapses. Likewise, if the ablation without the augmented term matches the full method everywhere, the adverse-condition agreement term is inert and the design rationale is wrong. If the pre-training verification (2d) shows F(u) is not more accurate than camera labels on V, the load-bearing assumption is false and the method should not proceed. If the ablation without w(u) matches the full method, the artifact-aware weighting is unnecessary and the projection-artifact concern is unfounded.

## References

1. Open Vision Banana: A Fully Open Recipe for Generative Visual Perception
