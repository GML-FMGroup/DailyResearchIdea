# Kinematic Warp Fields: Lifting 2D Motion Tracks to 3D without Multi-View Calibration

## Motivation

Existing methods like MT-π (Motion Tracks) recover 3D trajectories from 2D motion tracks by triangulating across multiple calibrated cameras, requiring cumbersome calibration and limiting deployment. This dependence arises because 2D tracks alone lack depth information. By exploiting the robot's known kinematics as a structural prior, we can self-supervise a warp field that maps 2D tracks to a canonical 3D frame, eliminating multi-view calibration entirely.

## Key Insight

The robot's kinematic chain provides a diffeomorphic mapping from joint space to end-effector pose, so enforcing that a single 3D track, when projected under different joint configurations, reproduces observed 2D tracks yields a self-supervised signal for 3D recovery without ground-truth 3D.

## Method

(A) **What it is**: Kinematic Warp Field (KWF) — a neural network that takes a set of 2D motion tracks (pixel trajectories) and the robot's joint angles, and outputs a canonical 3D trajectory in the robot's base frame. (B) **How it works**:
```python
# Core training loop for KWF
for each training timestep t with observed 2D tracks T_t and joint angles θ_t:
    # Encode tracks into latent features
    h = cross_attention_encoder(T_t, θ_t)  # cross-attention heads=4
    # Predict 3D point in robot base frame
    P_3d = MLP_decoder(h)  # output: (x, y, z) in meters
    # Self-supervision via synthetic viewpoint perturbation
    δθ = random_perturbation(σ=0.1 rad)  # 8 perturbations per timestep
    P_pert = FK(θ_t + δθ) @ P_3d  # forward kinematics applies bone transforms
    # Project to 2D using known intrinsic matrix K
    p_2d = project(K, P_pert)  # homogeneous projection
    # Loss: reprojection error + smoothness
    L = MSE(p_2d, T_t) + λ * ||P_3d - previous_P_3d||^2  # λ = 0.01
```
Hyperparameters: perturbation magnitude σ=0.1 rad, perturbations per timestep=8, cross-attention heads=4, smoothness weight λ=0.01. (C) **Why this design**: We chose a cross-attention encoder over simple concatenation because it allows the network to dynamically weight which 2D track points are most informative given joint state, accommodating occlusions (trade-off: increased compute). We use synthetic perturbations of joint angles rather than multi-view data because it requires no additional hardware, at the cost of assuming the kinematic model is accurate. We project using only camera intrinsic matrix K (which is easier to obtain than extrinsic calibration) and rely on forward kinematics, avoiding any explicit pose estimation. The smoothness regularization prevents high-frequency jitter that would violate robot constraints, but may oversmooth fast motions. (D) **Why it measures what we claim**: The reprojection loss MSE(p_2d, T_t) operationalizes "3D consistency" because it enforces that the predicted 3D point, when transformed by the true forward kinematics and projected, matches the observed pixel track; this equivalence holds under the assumption that the kinematic model is exact and K is known, an assumption that fails when joint backlash or flexure introduces unmodeled offsets, in which case the loss may be minimized by a 3D point that is geometrically inconsistent with the real robot. The smoothness term operationalizes "physical plausibility" by penalizing large displacements between consecutive time steps; this assumption fails during rapid ballistic movements where true 3D velocity is high, causing the smoothness term to bias predictions toward lower speeds. The cross-attention mechanism operationalizes "robustness to partial occlusion" by allowing the encoder to downweight occluded track points via learned attention weights; this assumption fails when all tracks are simultaneously occluded (e.g., arm behind an obstacle), in which case the network falls back to a prior from smoothness.

## Contribution

(1) Kinematic Warp Fields (KWF), a self-supervised framework that lifts 2D motion tracks to 3D trajectories using only the robot's known kinematics and a single camera with known intrinsics, eliminating the need for multi-view calibration. (2) A demonstration that synthetic viewpoint perturbations (via joint angle variation) provide sufficient supervisory signal for 3D recovery, a principle that can generalize to other kinematic chains. (3) Empirical evidence that KWF enables zero-shot task generalization from 2D tracks on novel tasks without fine-tuning, outperforming baselines that require multi-view calibration.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Simulated robot arm with ground truth 3D | Direct measurement of 3D error |
| Primary metric | Mean 3D position error (cm) | Quantifies central claim accuracy |
| Baseline 1 | Concat: MLP on concatenated tracks+angles | Tests cross-attention benefit |
| Baseline 2 | NoPerturb: KWF without synthetic perturbation | Tests self-supervision via perturbation |
| Ablation-of-ours | KWF w/o smoothness regularization | Tests smoothness term necessity |

### Why this setup validates the claim

The simulated environment provides ground-truth 3D joint positions, enabling direct measurement of the central claim: that KWF reconstructs accurate 3D trajectories from 2D tracks and joint angles. The Concat baseline tests the cross-attention mechanism's role in dynamic feature weighting; if KWF outperforms Concat, it supports the claim that attention handles occlusions. The NoPerturb baseline tests whether synthetic perturbations provide a useful self-supervision signal; outperformance would confirm that perturbations improve 3D consistency by enforcing reprojection invariance under kinematic variations. The ablation without smoothness tests whether the regularization is necessary for physically plausible trajectories; if removed leads to jittery predictions, it validates that smoothness operationalizes physical plausibility. The metric directly measures the target—a clear falsifiable test.

### Expected outcome and causal chain

**vs. Concat** — On a case where a track point is occluded (e.g., tool behind workpiece), the Concat encoder treats all tracks equally, causing the network to average over unreliable points and yield a biased 3D position toward the occluded region. Our KWF instead uses cross-attention to downweight the occluded track based on joint state contextual cues, focusing on visible tracks. We expect a noticeable gap in mean 3D error on occlusion-rich subsets (e.g., 30% improvement) but near parity on unoccluded cases.

**vs. NoPerturb** — On a case where the joint angles vary significantly across demonstrations (e.g., reaching from different initial poses), the NoPerturb variant overfits to the specific joint configuration seen during training, producing high reprojection error when tested on unseen poses. Our method applies random joint perturbations at each timestep, forcing the network to learn a viewpoint-invariant 3D point. We expect KWF to generalize to novel joint configurations (e.g., 50% lower error on held-out poses) while NoPerturb suffers high variance.

### What would falsify this idea

If KWF shows equal error on occlusion-heavy and occlusion-free subsets, or if it fails to outperform NoPerturb on novel joint configurations, then the core claim that cross-attention and synthetic perturbation enable robust 3D inference is invalid.

## References

1. Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning
2. FlowRetrieval: Flow-Guided Data Retrieval for Few-Shot Imitation Learning
3. Flow as the Cross-Domain Manipulation Interface
4. Distilling and Retrieving Generalizable Knowledge for Robot Manipulation via Language Corrections
5. Deep Imitation Learning for Humanoid Loco-manipulation Through Human Teleoperation
6. XSkill: Cross Embodiment Skill Discovery
7. FlowBot++: Learning Generalized Articulated Objects Manipulation via Articulation Projection
8. VideoDex: Learning Dexterity from Internet Videos
