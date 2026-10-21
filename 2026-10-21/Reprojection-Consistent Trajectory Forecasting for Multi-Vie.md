# Reprojection-Consistent Trajectory Forecasting for Multi-View Video Prediction

## Motivation

Existing multi-view video predictors such as VPM (Learning the 4D World through Point Trajectory Forecasting) supervise 3D point trajectories with a video-prediction loss and only score multi-view metric consistency post-hoc, so the forecast trajectories are never forced to be geometrically coherent across viewpoints. This is a structural gap: the training objective is defined per-view in pixel space, and no differentiable term ties the forecast 3D points to the known camera poses during optimization, so errors in depth and cross-view correspondence are invisible to the loss until evaluation. As a result, generated multi-view videos can look plausible per view while their 3D trajectories fail to reproject consistently, which breaks the 4D world-modeling requirement that a single metric scene be recovered from all views.

## Key Insight

Enforcing differentiable reprojection consistency of the forecast 3D point trajectories across all generated views, using known camera poses, turns multi-view metric coherence into a training invariant of the trajectory generator rather than a post-hoc score, because the reprojection residual is exactly zero for any trajectory that is a rigid 3D scene observed by the given cameras.

## Method

**What it is.** We introduce GeoTraj, a training objective and lightweight trajectory-head modification that replaces the per-view video-prediction loss on trajectories with a differentiable multi-view geometric consistency loss. GeoTraj takes as input the multi-view video history, a language instruction, the known camera intrinsics/extrinsics for all views, and the forecast 3D point trajectories produced by a VPM-style trajectory head, and outputs a scalar training loss whose gradient couples the trajectory generator to the camera geometry.

**How it works.**

```pseudocode
# Inputs per training sample:
#   H: multi-view video history (V views, T_h frames)
#   L: language instruction
#   K_v, R_v, t_v: intrinsics, rotation, translation for each view v in 1..V
#   P: set of persistent scene points, each with a forecast 3D trajectory
#      p_i(t) for t in 1..T_f, produced by the trajectory head

1. Forecast trajectories:
     {p_i(t)} = TrajectoryHead(WorldModel(H, L))
     # p_i(t) in camera-agnostic world coordinates

2. For each forecast timestep t and each view v:
     x_hat_i,v(t) = pi( K_v * ( R_v * p_i(t) + t_v ) )
     # pi is perspective projection; differentiable w.r.t. p_i(t)

3. Obtain observed 2D locations:
     x_i,v(t) = TrackHead(GeneratedView_v(t), p_i(t))
     # the same point i is tracked in every generated view at time t
     # TrackHead is a frozen, pretrained CoTracker-style point tracker
     # (weights fixed; gradients flow through the generated pixels and the
     #  projected trajectory points but not through the tracker weights).
     # To prevent co-adaptation between the tracker input and the trajectory
     # head, the generated-view branch is DETACHED: x_i,v(t) is treated as a
     # fixed target, so the residual r_i,v(t) is a function of p_i(t) only.

4. Differentiable reprojection residual (per point, view, time):
     r_i,v(t) = || x_hat_i,v(t) - x_i,v(t) ||_2

5. Robust geometric consistency loss:
     L_geo = mean_{i,v,t}  rho( r_i,v(t) / sigma )
     # rho = Charbonnier (smooth L1-like), sigma = 2 px

6. Visibility gating (occlusion-aware):
     w_i,v(t) = 1[ z_i,v(t) > 0 ] * 1[ x_hat_i,v(t) in image bounds ]
     L_geo = sum w_i,v(t) * rho(...) / sum w_i,v(t)

7. Total objective:
     L = L_video + lambda_geo * L_geo + lambda_traj * L_traj_3D
     # lambda_geo = 1.0, lambda_traj = 0.1 (3D GT supervision kept as anchor)
```

The trajectory head is unchanged in architecture; only its supervision changes. The track head is a frozen, pretrained point tracker (CoTracker-style) applied to the generated views, so gradients flow through the generated pixels and the projected trajectory points but not through the tracker weights. To break the degenerate co-adaptation where the tracker drifts to match a wrong projection, the generated-view branch is detached: x_i,v(t) is treated as a fixed target, so the residual r_i,v(t) is a function of p_i(t) only. The load-bearing assumption is that this frozen tracker, applied to the generator's own generated views, yields cross-view correspondences x_i,v(t) that identify the same physical point i in every view with sub-pixel accuracy, so that r_i,v(t) reflects geometric inconsistency of the forecast trajectory rather than tracker error. This assumption is made explicit and is verified by a tracker-accuracy diagnostic (see experiment).

**Why this design.** We chose a differentiable reprojection residual over a post-hoc consistency metric (as in MEt3R) because a metric cannot shape the generator's gradients, accepting the cost that the tracker introduces noise into the training signal. We chose to keep a 3D ground-truth trajectory loss as an anchor rather than dropping it, because pure reprojection consistency is invariant to a global similarity transform (scale, rotation, translation of the whole scene), so without the anchor the forecast trajectories could be self-consistent yet metrically wrong; the cost is that simulated data remains necessary. We chose a robust Charbonnier penalty with visibility gating over a plain L2 loss because occluded points and dynamic appearance changes produce large residuals that would otherwise dominate and destabilize training, accepting the cost that the gate is a hard threshold and may discard some valid points near image borders. We chose to apply the tracker on generated views rather than on ground-truth views because the loss must constrain the generator's own output distribution, accepting the cost that early in training the tracker is unreliable and the loss is noisy; this is mitigated by the 3D anchor and by annealing lambda_geo from 0.1 to 1.0 over the first 10k steps. An expert would not describe GeoTraj as a variant of VPM because VPM's trajectory supervision is a per-point 3D regression loss with no cross-view coupling; the reprojection residual is a new term that ties every view's forecast to the same world-coordinate point through the known poses.

**Why it measures what we claim.** The reprojection residual r_i,v(t) measures multi-view metric consistency of the forecast trajectory for point i at time t because the named assumption is that the camera poses (K_v, R_v, t_v) are exact and that the tracker x_i,v(t) identifies the same physical point across views; under that assumption a zero residual is equivalent to the forecast 3D point reprojecting to the same physical location in every view, which is exactly cross-view metric coherence. This assumption fails when the tracker associates the wrong pixel (e.g., under heavy occlusion or repetitive texture), in which case the residual reflects tracker error rather than geometric inconsistency. The visibility gate w_i,v(t) measures whether a point is observable in view v at time t because the assumption is that a point is visible iff its projected depth is positive and its projection falls inside the image; this fails for points occluded by other geometry despite being inside the image, in which case the gate admits an occluded point and the residual reflects occlusion rather than inconsistency. The 3D anchor loss measures metric correctness of the trajectory because the assumption is that the simulated ground-truth trajectories are metrically accurate; this fails on real data without GT trajectories, in which case the anchor is unavailable and only relative consistency is enforced.

## Contribution

['A differentiable multi-view geometric consistency loss (GeoTraj) that replaces the per-view video-prediction loss on trajectories with a cross-view reprojection residual computed through known camera poses and a frozen point tracker on generated views.', 'A design principle showing that reprojection consistency alone is invariant to global similarity transforms, and that a 3D anchor loss is required to pin metric scale; we provide the visibility-gated robust formulation that makes this trainable.', 'An occlusion-aware visibility gating scheme and a lambda_geo annealing schedule that stabilize training when the tracker is unreliable early in optimization.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VP-WorldBench multi-view split | Simulated poses+GT trajectories plus real views |
| Primary metric | Cross-view reprojection error (px) | Directly measures claimed geometric coherence |
| Baseline | VPM (per-view 3D traj loss) | Tests whether cross-view coupling is necessary |
| Baseline | Multi-view video predictor, no traj supervision | Tests whether trajectories help at all |
| Baseline | VPM + post-hoc MEt3R filtering | Tests differentiable loss vs metric-only |
| Baseline | GeoTraj with epipolar/pairwise cross-view consistency instead of full 3D reprojection | Tests whether the specific 3D-world-coordinate coupling is the novel ingredient rather than generic geometric regularization |
| Ablation-of-ours | GeoTraj w/o 3D anchor loss | Tests whether anchor prevents similarity drift |
| Ablation-of-ours | GeoTraj with tracker-accuracy diagnostic | Tests whether residual reflects geometry rather than tracker noise |

### Why this setup validates the claim

The central claim is that a differentiable multi-view reprojection loss couples the trajectory generator to camera geometry in a way per-view 3D regression cannot, producing forecast trajectories that are metrically coherent across views. The primary metric, cross-view reprojection error on held-out views, is chosen because it is exactly the quantity the loss penalizes: if the mechanism works, this metric must drop, and if it drops only because of the 3D anchor, the ablation isolates that. VPM tests the necessity of cross-view coupling: it has trajectory supervision but no coupling, so any gain over it must come from the new term. The no-trajectory multi-view predictor tests whether explicit world-coordinate trajectories matter at all, guarding against the possibility that strong pixel prediction alone yields coherence. The post-hoc MEt3R baseline tests the differentiable-vs-metric distinction: same signal, but no gradient shaping, so if our gain over it is large the causal story about gradients is supported. The epipolar/pairwise baseline tests whether the specific 3D-world-coordinate coupling is the novel ingredient rather than generic geometric regularization. The anchor ablation tests the similarity-invariance argument: without it, we predict coherent-but-mis-scaled trajectories, visible as low reprojection error but high absolute 3D trajectory error. The tracker-accuracy diagnostic measures the fraction of points whose tracked pixel matches the GT reprojection within a threshold across training checkpoints, and reports how tracker error correlates with L_geo magnitude to substantiate that the residual reflects geometry rather than tracker noise.

### Expected outcome and causal chain

**vs. VPM (per-view 3D traj loss)** — On a case where a point drifts in depth under a moving camera, VPM produces a trajectory that fits per-view 3D targets but reprojects inconsistently across views because its loss never ties views together through the known poses. Our method instead penalizes the cross-view reprojection residual of the same world point, so the gradient pulls the trajectory toward a single consistent 3D location, and we expect a clear reprojection-error gap concentrated on dynamic, depth-changing points, with parity on static near-field points.

**vs. Multi-view video predictor, no traj supervision** — On a case with a thin structure crossing views, the baseline produces plausible pixels per view but no persistent world-coordinate point, so cross-view correspondence drifts because nothing enforces a shared 3D anchor. Our method instead carries persistent points through the trajectory head and enforces their reprojection consistency, so we expect lower reprojection error and better long-horizon coherence, with the gap widening as forecast horizon grows.

**vs. VPM + post-hoc MEt3R filtering** — On a case where early predictions are geometrically wrong, the post-hoc baseline can only select or reject samples after generation because the metric does not shape gradients, so errors persist in the generator. Our method instead backpropagates the reprojection residual into the trajectory head, so we expect a training-time reduction in reprojection error that the post-hoc filter cannot match, with the gap largest on the hardest occluded/dynamic subset.

**vs. GeoTraj with epipolar/pairwise cross-view consistency** — On a case where a point is consistent pairwise but drifts in absolute 3D location, the epipolar baseline enforces only that the projection lies on the epipolar line, not that it matches the same world point, so it can be satisfied by a family of inconsistent 3D trajectories. Our method instead enforces full 3D reprojection to the same world coordinate, so we expect lower cross-view reprojection error and lower absolute 3D trajectory error, with the gap concentrated on points where pairwise consistency is ambiguous.

### What would falsify this idea

The idea is falsified if GeoTraj's gain over VPM is uniform across static and dynamic subsets rather than concentrated where cross-view inconsistency should occur, or if the anchor ablation shows no increase in absolute 3D trajectory error despite low reprojection error — either pattern would mean the reprojection term is not doing the geometric work we claim. Additionally, the idea is falsified if the tracker-accuracy diagnostic shows that tracker error on generated views is high and correlates strongly with L_geo magnitude, indicating that the residual reflects tracker noise rather than geometric inconsistency.

## References

1. Learning the 4D World through Point Trajectory Forecasting
