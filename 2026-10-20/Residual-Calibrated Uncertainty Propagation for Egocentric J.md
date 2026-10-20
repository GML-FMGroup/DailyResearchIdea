# Residual-Calibrated Uncertainty Propagation for Egocentric Joint Reconstruction

## Motivation

Egocentric joint reconstruction pipelines such as EgoXperience's EgoX-Engine treat SLAM-based world alignment as a reliable oracle: they generate supervision for camera, scene, hands, and objects under the assumption that pose-graph alignment succeeded, and provide no channel through which tracking degradation can reach the downstream joint estimator. When SLAM silently drifts in dynamic or low-texture egocentric scenes, the resulting annotations are confidently wrong, and the unified model (EgoX) inherits this miscalibration without any signal that its world-grounded outputs should be distrusted — a structural gap, not a data-scaling gap, since the frontier's response was to scale SLAM-aligned supervision (EgoX-Data-6M) rather than to challenge the oracle assumption. Related geometry work such as DUSt3R and MonST3R likewise produces pointmaps and poses without calibrated per-frame reliability, so the field's trajectory compounds the same unchecked-oracle bottleneck across paper-trees.

## Key Insight

SLAM residual statistics are a sufficient statistic for alignment error under the pose-graph's own noise model, so a variance head trained on these residuals with a heteroscedastic negative-log-likelihood objective yields uncertainty that is monotone in true alignment error rather than merely correlated with image appearance.

## Method

**What it is.** We introduce **RCPU** (Residual-Calibrated Propagation of Uncertainty), a lightweight module that sits between EgoX-Engine's SLAM alignment and the joint hand/object/camera/scene estimator. It consumes per-frame SLAM residuals and outputs a low-dimensional alignment-uncertainty descriptor that is fed as an auxiliary input to the joint estimator, which is trained with a heteroscedastic NLL loss so that every entity prediction carries a calibrated variance reflecting SLAM tracking quality.

**How it works.**

```pseudocode
INPUT: frames I_{1..T}, SLAM pose graph G with optimized poses {P_t},
       per-frame residuals r_t = {reprojection residuals, loop-closure residuals,
       pose-graph edge residuals}, joint estimator f_theta

# Phase 0: dynamic-outlier masking (DynaSLAM-style geometric mask)
#   For each frame t, compute a static mask M_t over residual correspondences:
#     M_t = { i : |r_t[i]| < tau_dyn = 3.0 * median(|r_t|) }   # robust threshold
#   Split residuals into static (r_t^static = r_t[M_t]) and dynamic (r_t^dyn = r_t[~M_t]).

# Phase 1: residual aggregation (per frame, per residual type, on static and dynamic sets)
for t in 1..T:
    s_t^static = [ median(r_t^static), IQR(r_t^static), count(r_t^static > tau_inlier=2.0px),
                   log det(Cov(r_t^static)), ||P_t - P_{t-1}||_2 ]   # 5-dim static stat
    s_t^dyn    = [ median(r_t^dyn), IQR(r_t^dyn), count(r_t^dyn > tau_inlier=2.0px),
                   log det(Cov(r_t^dyn)), ||P_t - P_{t-1}||_2 ]      # 5-dim dynamic stat
    s_t = [s_t^static, s_t^dyn]   # 10-dim raw stat (static + dynamic channels)

# Phase 2: learned variance head (small MLP, 2 hidden layers, 64 units, GeLU)
mu_t, log_sigma_t^2 = MLP_psi(s_t)          # scalar mean + scalar log-variance
sigma_t^2 = softplus(log_sigma_t^2) + eps   # eps=1e-4

# Phase 3: uncertainty descriptor (low-dim, fed to f_theta)
#   Propagated pose uncertainty via first-order covariance propagation with cross-frame correlation:
#     Sigma_t = J_t Sigma_{t-1} J_t^T + Q_t, where J_t = I (identity for pose increment),
#     Q_t = diag(sigma_t^2, sigma_t^2, sigma_t^2) (isotropic from variance head),
#     and Sigma_0 = 0.  The propagated scalar is sqrt(trace(Sigma_t)).
#   This replaces the ad hoc product sigma_t^2 * ||P_t - P_{t-1}||_2.
u_t = [sigma_t^2, mu_t, sqrt(trace(Sigma_t))]   # 3-dim

# Phase 4: joint estimator with heteroscedastic heads
for each entity e in {camera, scene, hands, objects}:
    yhat_e, log_var_e = f_theta(I_t, u_t)   # u_t concatenated to f_theta input
    L_e = 0.5 * ( exp(-log_var_e) * ||yhat_e - y_e||^2 + log_var_e )

# Phase 5: training
L_total = sum_t sum_e L_e + lambda_cal * L_cal(mu_t, sigma_t^2; e_t)
# lambda_cal = 0.1; L_cal = Gaussian NLL of true alignment error e_t
#   e_t computed from held-out ground-truth poses on a calibration split
#   (calibration split: 512 frames from EgoX-Bench/EgoHu with GT poses, disjoint from eval frames)
```

**Why this design.** We chose a *learned variance head on SLAM residuals* over a hand-designed Bayesian filter over the pose graph because the pose graph's own noise model is misspecified in dynamic egocentric scenes (moving objects violate static-world assumptions), so a filter would be confidently wrong; the learned head can absorb this misspecification, accepting the cost that it requires a calibration split with ground-truth poses. We chose a *low-dimensional scalar uncertainty descriptor* over feeding the full residual vector to the joint estimator because the joint estimator is trained on EgoX-Data-6M scale and a high-dimensional auxiliary input would overfit to spurious residual patterns, accepting the cost that some residual structure is discarded. We chose a *heteroscedastic NLL loss with a separate calibration term* over a plain variance head because the joint estimator's own NLL alone can collapse to predicting large variance everywhere (the classic NLL degeneracy), so the calibration term anchors sigma_t^2 to the true alignment error, accepting the cost of an extra loss weight to tune. A domain expert would not describe this as a variant of DUSt3R or MonST3R: those regress pointmaps without any per-frame reliability channel, whereas RCPU's mechanism is the residual-to-variance mapping that makes the joint estimator's uncertainty *monotone in SLAM error*, a property neither DUSt3R nor EgoX provides.

**Why it measures what we claim.** The computational quantity `sigma_t^2` measures *SLAM alignment uncertainty* because the Gaussian NLL loss with the calibration term makes `sigma_t^2` equal the conditional variance of the true alignment error given the residual statistics `s_t` under the assumption that `s_t` is a sufficient statistic for alignment error — an assumption that holds when the pose-graph's residuals capture all sources of misalignment; this assumption fails when dynamic objects produce large residuals that are *not* alignment errors, in which case `sigma_t^2` reflects *scene dynamism* rather than tracking degradation. The quantity `u_t`'s third component `sqrt(trace(Sigma_t))` measures *propagated pose uncertainty* because it equals the first-order variance of the pose increment under the assumption that consecutive-frame errors are independent; this fails under loop closures that correlate errors across frames, in which case it under-estimates propagated uncertainty. The joint estimator's `log_var_e` measures *per-entity prediction reliability* because the heteroscedastic NLL makes it the conditional variance of the entity error given `u_t` under the assumption that the entity error is Gaussian in `u_t`; this fails for heavy-tailed errors from occlusions, in which case `log_var_e` reflects *occlusion frequency* rather than SLAM-induced uncertainty.

## Contribution

(1) RCPU, a residual-calibrated uncertainty propagation module that converts SLAM pose-graph residuals into a low-dimensional uncertainty descriptor and feeds it to a joint egocentric estimator via heteroscedastic NLL, making per-entity uncertainty monotone in alignment error. (2) The finding that a scalar variance head trained on SLAM residual statistics with a separate calibration term avoids the NLL variance-collapse degeneracy and produces uncertainty that tracks true alignment error better than a Bayesian pose-graph filter in dynamic scenes. (3) A calibration protocol and evaluation split for egocentric joint reconstruction that measures uncertainty-error monotonicity under controlled SLAM degradation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | EgoX-Bench + EgoHu | Egocentric, SLAM-annotated, joint human/scene/camera ground truth |
| Primary metric | Per-entity NLL calibration on misaligned frames | Directly tests uncertainty tracks alignment error |
| Baseline | EgoX-Engine joint estimator (no uncertainty channel) | Tests whether residual signal helps at all |
| Baseline | MonST3R-style pointmap regression | Tests per-frame reliability channel necessity |
| Baseline | Hand-designed Bayesian pose-graph filter | Tests learned head vs misspecified noise model |
| Baseline | DROID-SLAM per-frame covariance | Tests learned head vs generic variance regression |
| Baseline | Deep ensemble over poses (5 models) | Tests learned head vs ensemble uncertainty |
| Ablation-of-ours | RCPU with fixed sigma (no residual head) | Isolates residual-to-variance mapping contribution |
| Ablation-of-ours | RCPU with raw residuals (no dynamic mask) | Tests sufficiency assumption under dynamism |

### Why this setup validates the claim

The central claim is that a learned residual-to-variance mapping makes the joint estimator's uncertainty monotone in SLAM alignment error, improving calibration precisely where tracking degrades. EgoX-Bench and EgoHu are the right testbeds because they carry joint hand/object/camera/scene ground truth *and* per-frame SLAM poses, so we can compute the true alignment error e_t and stratify frames by it. The primary metric — per-entity NLL on frames binned by true alignment error — is chosen because it directly exposes whether predicted variance co-varies with actual error; a method that merely improves mean accuracy would not move this metric, while a method that produces confidently-wrong predictions exactly on high-error frames will be penalized. Each baseline tests a distinct sub-claim: the plain joint estimator tests whether any uncertainty channel is needed; the MonST3R-style pointmap regressor tests whether the reliability channel must be residual-derived rather than implicit in pointmap confidence; the Bayesian pose-graph filter tests whether a hand-designed noise model, confidently wrong under dynamic-scene misspecification, is beaten by a learned head; the DROID-SLAM per-frame covariance and deep ensemble baselines test whether the learned head's advantage comes specifically from absorbing dynamic-scene misspecification rather than from generic variance regression. The fixed-sigma ablation is decisive: it holds the heteroscedastic loss and architecture constant while removing the residual mapping, so any calibration gain concentrated on high-alignment-error frames must be attributable to the residual head rather than to the NLL loss alone. The raw-residual ablation directly tests the sufficiency assumption: if the dynamic mask is necessary, the raw-residual variant will show inflated variance on dynamic-but-well-aligned frames.

### Expected outcome and causal chain

**vs. EgoX-Engine joint estimator (no uncertainty channel)** — On a case where the camera undergoes fast rotation past a moving person, the baseline's SLAM poses drift and the joint estimator produces hand and object placements with no variance signal, so its errors are confidently wrong and its NLL on high-alignment-error frames is dominated by the exp(-log_var) term with log_var near a constant. Our method instead reads the inflated reprojection and pose-graph residuals through the variance head, raising sigma_t^2 and hence u_t, so the heteroscedastic head widens log_var_e on exactly those frames. We expect a large NLL gap concentrated in the top alignment-error quintile, with near-parity on the lowest-error quintile.

**vs. MonST3R-style pointmap regression** — On a case where a loop closure retroactively corrects a long drift, the baseline regresses pointmaps per frame with no channel tying its confidence to the corrected pose graph, so its per-frame reliability is blind to the closure and it stays confident on frames whose alignment was just revised. Our method instead consumes loop-closure residuals in s_t, so a closure event shifts sigma_t^2 and propagates into u_t across the affected window, widening entity variance where the pose graph changed. We expect a visible calibration improvement in the frames bracketing loop closures, with parity on loop-free segments.

**vs. Hand-designed Bayesian pose-graph filter** — On a case where a large moving object dominates the residual set, the baseline's static-world noise model interprets those residuals as tracking degradation and inflates pose covariance, producing over-wide uncertainty on frames that are actually well-aligned but dynamic. Our method instead learns from the calibration split that high-residual dynamic frames do not imply large alignment error, so sigma_t^2 stays moderate and the entity variances do not over-widen. We expect the filter to over-cover on dynamic-but-well-aligned frames while our method's coverage stays near nominal there, with the gap reversing only on genuinely degraded frames.

**vs. DROID-SLAM per-frame covariance** — On a case where the camera moves through a low-texture corridor, DROID-SLAM's per-frame covariance is derived from its own dense bundle adjustment and may not reflect the pose-graph alignment error used by EgoX-Engine. Our method instead trains sigma_t^2 directly against the true alignment error e_t on the calibration split, so it is calibrated to the specific SLAM pipeline. We expect our method to achieve lower NLL on high-error frames, with parity on low-error frames.

**vs. Deep ensemble over poses (5 models)** — On a case where dynamic objects cause ensemble disagreement, the ensemble's variance reflects model disagreement rather than alignment error, and may be overconfident if all models share the same static-world bias. Our method instead uses residual statistics that are directly tied to the pose graph's own noise model, and the dynamic mask disentangles dynamism from tracking degradation. We expect the ensemble to be overconfident on high-error frames where all models agree, while our method's calibration remains monotone in alignment error.

### What would falsify this idea

If our calibration gain over the fixed-sigma ablation is uniform across alignment-error strata rather than concentrated in the high-error quintile and around loop closures, then the residual-to-variance mapping is not the causal driver and the central claim is wrong. Likewise, if the learned head's sigma_t^2 fails to correlate with held-out true alignment error e_t on the calibration split, the sufficiency assumption underlying the mechanism is violated.

## References

1. EgoXperience: Learning World-Grounded Egocentric Representations from Human Experience
2. Human3R: Everyone Everywhere All at Once
3. Reconstructing People, Places, and Cameras
4. MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion
5. Generative Proxemics: A Prior for 3D Social Interaction from Images
6. DUSt3R: Geometric 3D Vision Made Easy
7. PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment
