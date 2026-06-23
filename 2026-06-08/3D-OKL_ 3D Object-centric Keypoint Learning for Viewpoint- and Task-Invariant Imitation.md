# 3D-OKL: 3D Object-centric Keypoint Learning for Viewpoint- and Task-Invariant Imitation

## Motivation

Existing representations in imitation learning are either task-specific (e.g., SKIL's fixed semantic keypoints) or view-specific (e.g., Motion Tracks' 2D trajectories tied to calibrated cameras). This forces per-task retraining or recalibration when tasks or viewpoints change. The root cause is the absence of a representation that is simultaneously invariant to both camera viewpoint and task variation, which would allow a single model to transfer zero-shot across both axes.

## Key Insight

Multiview geometric consistency and cross-task contrastive learning, when jointly optimized over a shared 3D keypoint representation, force the keypoints to be both viewpoint-invariant (via differentiable triangulation consistency) and task-invariant (via pulling same-object keypoints from different tasks together), decoupling object structure from nuisance factors.

## Method

### 3D-OKL: 3D Object-centric Keypoint Learning

(A) **What it is**: 3D-OKL is a self-supervised framework that, given uncalibrated multiview RGB videos of multiple manipulation tasks, outputs a fixed set of 3D object-centric keypoints. These keypoints encode demonstrations as sequences of relative 3D positions, enabling zero-shot transfer to novel tasks and viewpoints.

(B) **How it works** (pseudocode):
```python
# Phase 1: Keypoint prediction per view
for each time step t and view v:
  features = ResNet50(image_vt)  # output 512 feature maps
  heatmaps_vt = MLP(features, 2)  # predicts K=10 heatmaps (each 2D Gaussian), hidden=256, GeLU

# Phase 2: Differentiable triangulation with learnable camera calibration
# For each view v, learnable intrinsics K_v (3x3) and extrinsics R_v (3x3), t_v (3)
# CalibNet outputs calibration parameters from view embedding (average of image features)
for each view v:
  embed = mean_pool(features_v)
  K_v, R_v, t_v = CalibNet(embed)  # 2-layer MLP, hidden=128, output 7 (focal, principal points, rotation axis-angle, translation)
for each keypoint k:
  3D_keys[k] = Triangulate(heatmaps_1..V, k, {K_v, R_v, t_v})  # weighted triangulation using heatmap confidences as weights
  reproj_loss += MSE(Project(3D_keys[k], K_v, R_v, t_v), heatmap_vt[k].argmax) over v,t,k

# Phase 3: Contrastive learning across tasks
# Pull same-object keypoints from different tasks; push different-object keypoints
for each object O, tasks T1,T2:
  sim = cosine_similarity(key_T1_O, key_T2_O)  # KxK matrix
  pos_pairs = diag(sim)
  neg_pairs = sim[~diag]  # all off-diagonal (different keypoints) plus other objects
  contrastive_loss = -log(exp(pos_pairs/τ) / (exp(pos_pairs/τ) + sum(exp(neg_pairs/τ))))
  # τ=0.07

# Phase 4: Temporal cycle-consistency
# Use RAFT-small optical flow network to predict 2D flow between consecutive frames per view
for each view v, time t:
  flow_2D = RAFT-small(image_vt, image_{v,t-1})
for each keypoint k and time t:
  # Lift 2D flow to 3D using learnable calibration
  proj_2D = Project(3D_keys[t-1], K_v, R_v, t_v)
  warped_2D = proj_2D + flow_2D[proj_2D]
  depth = 3D_keys[t-1][2]  # approximate depth from previous 3D keypoint
  warped_3D = Lift(warped_2D, depth, K_v, R_v, t_v)
  cycle_loss += ||3D_keys[t] - warped_3D||^2

Total loss = λ1 * reproj_loss + λ2 * contrastive_loss + λ3 * cycle_loss, λ1=1.0, λ2=0.1, λ3=0.5
```

(C) **Why this design**: We chose differentiable triangulation with learnable calibration over explicit calibration because it allows handling uncalibrated multiview setups during training, but at the cost of increased optimization difficulty and sensitivity to initial heatmap quality. We added a learnable calibration module to resolve projective ambiguity, accepting increased parameters but ensuring metric consistency. We used contrastive learning with a cosine similarity loss over object-specific keypoint sets rather than supervised keypoint annotation because it scales to diverse objects and tasks without human labeling, accepting that the contrastive objective may conflate distinct object parts if tasks are too few or object segments poorly defined. We added temporal cycle-consistency to enforce smooth keypoint trajectories, which is critical for imitation policy learning but adds a dependency on accurate optical flow estimation; this trade-off weights robustness against flow quality. These three design choices collectively enforce the dual invariance: geometric consistency anchors keypoints to 3D structure, contrastive learning aligns across tasks, and temporal consistency stabilizes the representation over time.

(D) **Why it measures what we claim**: The reprojection loss `reproj_loss` measures *multiview geometric consistency* because it assumes the true 3D keypoint projects to a consistent 2D location across views when using the learned calibration; this assumption fails under severe occlusion or non-Lambertian surfaces, causing the loss to enforce consistency on spurious correspondences. The contrastive loss `contrastive_loss` measures *task invariance* because it assumes that keypoints should be closer for the same object across tasks than for different objects; this assumption fails when tasks involve fundamentally different object interaction regions (e.g., grasping handle vs. pressing button), potentially pulling task-specific keypoints incorrectly toward the common mean. The cycle loss `cycle_loss` measures *temporal coherence* because it assumes that 3D keypoints change smoothly over time; this assumption fails during rapid motion or occlusions, forcing the model to rely on flow predictions that may be inaccurate. Each quantity operationalizes a specific invariance, and the named failure modes indicate when the metric may not correspond to the intended property.

## Contribution

(1) A self-supervised framework that learns 3D object-centric keypoints from uncalibrated multiview videos, achieving simultaneous viewpoint and task invariance via joint geometric and contrastive objectives. (2) An empirical demonstration that the learned keypoint representation enables zero-shot transfer of imitation policies to both novel tasks and novel camera viewpoints without any fine-tuning or calibration, outperforming per-task keypoint baselines and view-specific track representations. (3) A publicly released dataset of multiview manipulation demonstrations with varying tasks and camera positions (to be made available).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Multiview manipulation benchmark (4 cams, 10 tasks) | Tests generalization across tasks/views |
| Primary metric | Task success rate on novel tasks & viewpoints | Direct measure of zero-shot transfer |
| Baseline 1 | Behavior cloning from single-view RGB | No 3D or cross-task alignment |
| Baseline 2 | SKIL (supervised keypoints) | Requires annotations, not zero-shot |
| Baseline 3 | 3D-OKL w/o contrastive loss | Isolates effect of task alignment |
| Ablation-of-ours | 3D-OKL w/o temporal cycle loss | Measures importance of temporal consistency |

### Why this setup validates the claim
The design evaluates the central claim that 3D-OKL enables zero-shot transfer to novel tasks and viewpoints. The multiview dataset includes multiple tasks, providing a test bed for generalization. The primary metric directly captures success on unseen configurations. Comparing against behavior cloning (no 3D structure) tests whether 3D keypoints help; against SKIL (supervised) tests if self-supervision suffices. The ablation without contrastive loss isolates the role of cross-task alignment, while the temporal ablation tests smoothness. This combination creates a falsifiable test: if our method outperforms on novel conditions but matches on near-distribution cases, the claim is supported.

### Expected outcome and causal chain

**vs. Behavior cloning from single-view RGB** — On a case where the camera viewpoint shifts 30 degrees, behavior cloning fails because it relies on pixel correlations from training views. Our method uses 3D keypoints that are view-invariant, so we expect high success on novel viewpoints (e.g., >80%) versus baseline's <50% on those subsets, but parity on training viewpoints.

**vs. SKIL (supervised keypoints)** — On a case where a novel object (e.g., different color) appears, SKIL fails because its keypoints are tied to visual features of the training object. Our method learns object-invariant keypoints via contrastive learning across tasks, so we expect success on novel objects (e.g., ~70%) while SKIL drops below 40%. On familiar objects, performance may be similar.

**vs. 3D-OKL w/o contrastive loss** — On a case where two tasks involve the same object but different interaction regions (e.g., grasping handle vs. pressing button), the ablation fails because keypoints collapse to task-specific locations. Our method aligns keypoints across tasks, so we expect consistent performance across tasks (e.g., >75%) while the ablation shows a gap (e.g., 60% on one task and 80% on another).

### What would falsify this idea
If the success rate of 3D-OKL on novel tasks/viewpoints is not significantly higher than behavior cloning, or if the contrastive ablation matches the full method, then the central claim of zero-shot transfer via self-supervised object-centric keypoints is incorrect.

## References

1. SKIL: Semantic Keypoint Imitation Learning for Generalizable Data-efficient Manipulation
2. Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning
3. FlowRetrieval: Flow-Guided Data Retrieval for Few-Shot Imitation Learning
4. Flow as the Cross-Domain Manipulation Interface
5. Distilling and Retrieving Generalizable Knowledge for Robot Manipulation via Language Corrections
6. Deep Imitation Learning for Humanoid Loco-manipulation Through Human Teleoperation
7. XSkill: Cross Embodiment Skill Discovery
8. FlowBot++: Learning Generalized Articulated Objects Manipulation via Articulation Projection
