# Visibility-Conditioned SE(3) Pose Uncertainty for Monocular TCP Tracking and Trajectory Generation

## Motivation

Monocular video-based manipulation pipelines recover end-effector motion as a point estimate, so the recovered action carries no signal about whether the TCP pose is actually identifiable at a given timestep. TCP4D (E4) recovers metric TCP poses in SE(3) from monocular RGB and initial 2D queries and feeds the recovered tracks directly into its generator, but under gripper occlusion, motion blur, or self-occlusion the tracker's output is equally confident whether the TCP is visible or not; the generator and downstream policy therefore condition on trajectories whose reliability varies silently across time. The structural gap is that pose identifiability under occlusion is never made an explicit, per-timestep property of the recovered action, so errors from unidentifiable frames propagate into generated trajectories and retargeted demonstrations without any mechanism to down-weight them.

## Key Insight

Occlusion-induced pose ambiguity in monocular TCP tracking is not isotropic noise but lies in a low-dimensional, geometrically determined subspace (the viewing-ray/depth direction and the rotational null direction of the visible keypoint set), so a visibility-conditioned covariance that is constrained to this subspace is identifiable from the same 4D representation TCP4D already builds, making per-timestep pose uncertainty a recoverable quantity rather than an estimated scalar confidence.

## Method

**What it is.** VISTA-TCP (Visibility-Informed SE(3) Tracking with Ascertained Pose-covariance) augments a TCP4D-style monocular tracker with a per-timestep SE(3) pose covariance conditioned on visibility evidence, and propagates that covariance into the trajectory generator as a precision-weighted conditioning signal. Inputs: monocular video, initial 2D TCP queries, language instruction. Outputs: per-timestep SE(3) pose mean, a 6x6 pose covariance, a visibility mask over TCP keypoints, and generated future TCP trajectories with per-waypoint covariance.

**Load-bearing assumption (stated explicitly).** The method assumes that the visibility-conditioned reprojection covariance `Sigma_t = (J_t^T Sigma_pix^{-1} J_t + Lambda_prior)^{-1}` is a calibrated, task-relevant measure of TCP pose identifiability: that its trace (and hence the precision weight `w_t`) is monotone in actual pose error, and that its dominant direction aligns with the true unidentifiable subspace. This assumption is only valid when (i) the visibility head `v_t` is calibrated so that `Sigma_pix` reflects realized reprojection residuals, (ii) the SE(3) tangent-space linearization at `pose_t` is locally valid (small inter-frame rotation and depth error), and (iii) the visible keypoint set is non-degenerate so that `J_t` is full rank. The method below makes these conditions explicit and adds a calibration objective and a rank gate so that `Sigma_t` is tied to measured error rather than to an assumed noise model.

**How it works.**

```pseudocode
# Phase 1: visibility evidence from the shared 4D representation
for each frame t:
    F_t = 4D_representation(video, t)          # inherited from TCP4D backbone
    K_t = project_TCP_keypoints(F_t, query)    # K=8 canonical TCP keypoints
    v_t = visibility_head(F_t)[K_t]            # per-keypoint visibility in [0,1], sigmoid
    # visibility_head: 2-layer MLP (hidden=256, GeLU) on F_t features, sigmoid output

# Phase 1b: calibration of visibility head and pixel-noise scale
# Calibration set: 512 held-out frames from DROID/RoboLab with ground-truth keypoint visibility
# (from mocap or manual annotation) and realized reprojection residuals.
# Loss: L_cal = BCE(v_t, v_gt) + lambda_res * || (sigma_px^2 / max(v_t, eps)) - r_t^2 ||^2
# where r_t is the realized reprojection residual on visible keypoints.
# lambda_res = 0.1, sigma_px = 2.0 pixels (initial), optimized jointly with visibility_head.
# Report ECE and reliability diagram on calibration set.

# Phase 2: geometry-conditioned pose covariance (NOT a free 6x6 head)
J_t = jacobian_of_reprojection(K_t, pose_t)   # 2K x 6, standard SE(3) reprojection Jacobian
Sigma_pix = diag(sigma_px^2 / max(v_t, eps))    # 2K x 2K, visibility-inflated pixel noise
H_t = J_t^T Sigma_pix^{-1} J_t + Lambda_prior  # 6x6 information matrix, Lambda_prior = 1e-3 * I
# Rank gate: compute condition number kappa(J_t). If kappa > 1e4 or rank(J_t) < 6,
# set Sigma_t = Lambda_prior^{-1} (prior-dominated) and flag frame as 'unidentifiable'.
# Otherwise, Sigma_t = H_t^{-1}.

# Phase 3: covariance-weighted conditioning of the generator
# Task-sensitivity-weighted precision (replaces scalar trace-based w_t):
# g = empirical pose-error sensitivity of the generator, estimated as the gradient
# of generator output w.r.t. pose input on a validation set (512 frames), averaged
# over se(3) directions. g is a 6-vector.
w_t = 1 / (1 + g^T Sigma_t g)                  # direction-aware precision weight
c_t = concat(F_t_sampled_along_track, w_t, flatten(Sigma_t))
future = generator(c_t, robot_obs, language)   # predicts TCP states + per-waypoint covariance

# Phase 4: confidence-weighted execution / retargeting
for each generated waypoint j:
    execute_weight_j = 1 / (1 + g^T Sigma_future_j g)
```

**Why this design.** We chose a *geometry-conditioned* covariance (Phase 2, built from the reprojection Jacobian and visibility-inflated pixel noise) over a free learned 6x6 covariance head because a free head has no structural reason to place uncertainty along the viewing ray where monocular depth is genuinely unidentifiable, accepting the cost that we must commit to a keypoint reprojection model and a pixel-noise scale rather than learning the covariance end-to-end. We chose to inflate *per-keypoint* pixel noise by inverse visibility (Phase 1) rather than to mask out invisible keypoints entirely, because masking discards the partial evidence a low-visibility keypoint still carries about the in-plane pose components, accepting the cost that visibility estimates must be roughly calibrated. We chose a scalar precision weight plus flattened covariance as generator conditioning (Phase 3) rather than feeding the full 6x6 matrix into the generator's cross-attention, because the generator was pretrained on point-estimate tracks and a scalar-plus-matrix token is a minimal perturbation, accepting the cost that some off-diagonal covariance structure is compressed. A domain expert would not describe this as 'TCP4D + an uncertainty head': TCP4D's tracker treats the recovered trajectory as a point estimate and its generator conditions on that point estimate directly (E4), whereas here the covariance is *derived from the same reprojection geometry that determines identifiability* and enters the generator as a precision weight, so the generator's conditioning distribution is a function of the tracker's identifiability rather than of a separately trained confidence predictor.

**Why it measures what we claim.** The quantity `Sigma_t = (J_t^T Sigma_pix^{-1} J_t + Lambda_prior)^{-1}` measures *pose identifiability under occlusion* because the assumption that monocular reprojection noise is the dominant error source and that the SE(3) tangent-space linearization is locally valid makes the inverse information matrix equal the Cramer-Rao-type covariance of the pose estimate; this assumption fails when the TCP undergoes fast rotation between frames (linearization error dominates) or when the visibility head is miscalibrated (Sigma_pix is wrong), in which case Sigma_t reflects reprojection-geometry conditioning rather than true pose uncertainty. The precision weight `w_t = 1/(1 + g^T Sigma_t g)` measures *how much the generator should trust this timestep's track* because the assumption that the generator's sensitivity to pose error is captured by the empirical gradient `g` (estimated on validation data) makes `w_t` monotone in task-relevant identifiability; this assumption fails when `g` is poorly estimated or when the generator's sensitivity is highly nonlinear, in which case `w_t` reflects the linearized sensitivity rather than true execution risk. The per-waypoint execution weight `execute_weight_j` measures *confidence in generated future TCP states* because the assumption that the generator's predicted covariance is calibrated against realized tracking error makes the weight monotone in execution reliability; this assumption fails when the generator's covariance is overconfident on out-of-distribution instructions, in which case the weight reflects generator self-consistency rather than true execution risk.

## Contribution

['A visibility-conditioned SE(3) pose covariance estimator for monocular TCP tracking that constrains uncertainty to the geometrically identifiable subspace via the reprojection Jacobian and visibility-inflated pixel noise, rather than learning a free covariance head.', 'A covariance-as-precision-weight conditioning scheme that lets a TCP4D-style trajectory generator consume per-timestep pose reliability, making end-effector pose identifiability an explicit property of the recovered action rather than an implicit one.', 'A confidence-weighted retargeting and execution protocol that down-weights generated waypoints with high propagated covariance, together with an evaluation protocol that stratifies tracking and manipulation success by occlusion level.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DROID + RoboLab monocular manipulation video | TCP4D reports trajectory error here; occlusion-rich real footage |
| Primary metric | Success rate on occlusion-stratified task subsets | Directly tests whether uncertainty helps where identifiability fails |
| Secondary metric | Calibration of Sigma_t: ECE of visibility head, reliability diagram, and correlation between trace(Sigma_t) and realized SE(3) pose error (stratified by inter-frame rotation rate) | Verifies the load-bearing assumption that Sigma_t is calibrated and monotone in error |
| Secondary metric | Subspace alignment error between Jacobian null-space and learned-head covariance dominant direction on held-out occlusion frames | Tests whether geometry-derived covariance captures a subspace a free head cannot |
| Baseline | TCP4D point-estimate tracker + generator | The method we augment; isolates covariance contribution |
| Baseline | TCP4D with post-hoc learned 6x6 covariance head | Tests geometry-derived vs free covariance |
| Baseline | Uniform-confidence execution (no precision weighting) | Tests whether conditioning on w_t matters at all |
| Ablation-of-ours | VISTA-TCP with visibility mask instead of noise inflation | Isolates the partial-evidence design choice |
| Ablation-of-ours | VISTA-TCP with scalar trace-based w_t (original) instead of task-sensitivity-weighted w_t | Tests whether direction-aware precision matters |
| Ablation-of-ours | VISTA-TCP without rank gate (prior-dominated covariance on degenerate frames) | Tests whether explicit unidentifiable flag helps |

### Why this setup validates the claim

The central claim is that deriving pose covariance from reprojection geometry under visibility, and feeding it into the generator as precision, improves manipulation where monocular identifiability degrades. The design is falsifiable because each baseline removes exactly one link in the causal chain. TCP4D point-estimate tests whether covariance matters at all; the free learned covariance head tests whether the *geometry-derived* structure (rather than any uncertainty signal) is what helps; uniform-confidence execution tests whether the generator actually uses w_t. Stratifying success rate by occlusion severity is the right primary metric because the claim predicts a *selective* gain concentrated on high-occlusion, fast-depth-ambiguity instances, not a uniform lift. A uniform lift would be consistent with generic regularization or extra parameters, not with the identifiability story, so the metric must resolve where the effect lands, not just its average. The added calibration metrics directly test the load-bearing assumption: if Sigma_t is not calibrated (ECE high, low correlation with realized error) or if the Jacobian null-space does not align with the true ambiguity direction, the mechanism is not what produces any observed gain. The subspace alignment metric distinguishes geometry-derived from learned covariance at the level of the covariance structure itself, not just downstream success.

### Expected outcome and causal chain

**vs. TCP4D point-estimate tracker + generator** — On a case where the TCP recedes along the viewing ray under partial occlusion, the baseline commits to a point track with no notion that depth is weakly constrained, so the generator is conditioned as if the track were reliable and produces waypoints with confident but depth-erroneous motion. Our method inflates pixel noise on low-visibility keypoints, producing a large covariance along the viewing ray and a low precision weight w_t, so the generator is conditioned to hedge. We expect a clear success-rate gap concentrated on high-occlusion subsets, with near-parity on unoccluded ones. Additionally, we expect trace(Sigma_t) to correlate positively with realized pose error on held-out frames (Spearman rho > 0.5) and the visibility head to achieve ECE < 0.1 after calibration.

**vs. TCP4D with post-hoc learned 6x6 covariance head** — On a case where occlusion is brief and the TCP rotates quickly, a free learned head has no structural reason to place uncertainty along the unidentifiable ray; it tends to predict covariance from appearance correlates and can be overconfident exactly where geometry is degenerate. Our Jacobian-based Sigma_t places uncertainty along the null direction of the reprojection by construction. We expect our method to show better-calibrated waypoint covariance and higher success on the fast-rotation-plus-occlusion subset, while the learned head may match us on easy frames. We also expect the subspace alignment error between the Jacobian null-space and the learned head's dominant covariance direction to be significantly larger than between the Jacobian null-space and our Sigma_t (e.g., principal angle > 30 degrees for learned head vs < 10 degrees for ours).

**vs. Uniform-confidence execution (no precision weighting)** — On a case where the generator produces a long horizon mixing well-tracked and poorly-tracked segments, uniform execution treats all waypoints as equally trustworthy, so early depth-ambiguous waypoints corrupt downstream retargeting. Our per-waypoint execute_weight_j down-weights the ambiguous segments, preserving the reliable in-plane motion. We expect a moderate but consistent gain on long-horizon tasks with intermittent occlusion, and parity on short, fully visible tasks.

**vs. VISTA-TCP with scalar trace-based w_t** — On a case where the covariance is highly anisotropic (depth-dominant), the scalar trace over-weights the depth direction, so the generator may over-hedge on in-plane motion that is actually well-constrained. Our task-sensitivity-weighted w_t = 1/(1 + g^T Sigma_t g) uses the generator's empirical sensitivity to down-weight only the task-relevant directions. We expect a small but measurable improvement on tasks where depth ambiguity is high but in-plane motion is critical, and parity elsewhere.

**vs. VISTA-TCP without rank gate** — On a case where gripper self-occlusion makes J_t rank-deficient, the prior-dominated covariance (Lambda_prior = 1e-3*I) produces a large but isotropic Sigma_t that does not reflect the true unidentifiable subspace. Our rank gate flags such frames as 'unidentifiable' and falls back to an explicit flag, preventing the generator from conditioning on a misleading covariance. We expect fewer catastrophic failures on degenerate-keypoint frames and better calibration of the unidentifiable flag (e.g., flag precision > 0.8).

### What would falsify this idea

If the gain over the point-estimate baseline is uniform across occlusion strata rather than concentrated where visibility is low and depth is ambiguous, the identifiability mechanism is not what produces the improvement. Likewise, if the free learned covariance head matches or beats the Jacobian-derived covariance on the high-occlusion subset, or if the calibration study shows that Sigma_t is not monotone in realized pose error (e.g., Spearman rho < 0.2) or that the Jacobian null-space does not align with the true ambiguity direction, the geometry-conditioning claim is wrong.

## References

1. TCP4D: A Unified Tool-Center-Point Tracking and Generation Model for Robot Manipulation
