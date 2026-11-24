# Covariance-Propagated Geometric Lifting with Calibrated Fusion for 3D Hand Pose

## Motivation

Geometry-driven hand reconstruction methods such as ViDiHand2 treat per-frame 2D joint predictions from the VDM backbone as reliable anchors: Fine-Grained Geometric Lifting enforces consistency between these 2D joints and MANO metric structure, and Confidence-Gated Fusion arbitrates between lifted joints and MANO predictions using an uncalibrated localization confidence. Because the 2D localization error is neither bounded nor propagated through the lifting solver, and because the fusion gate is not tied to true joint error, errors in the anchor silently propagate into metric depth with no reliability guarantee. This is a structural failure of treating a noisy 2D observation as a noiseless geometric constraint, not a data or capacity limitation.

## Key Insight

Propagating a per-joint 2D localization covariance through the lifting solver converts each metric depth estimate into a first-order error-attenuated quantity whose predicted variance is monotone in true depth error, and calibrating the fusion gate on a held-out set makes the arbitration threshold a reliability guarantee rather than a heuristic.

## Method

**What it is.** We introduce CALIPER (Covariance-Aware Lifting with Implicit Propagation and Error-calibrated Reliability), a module that replaces ViDiHand2's point-estimate lifting and uncalibrated confidence gating with (i) a covariance-propagation step through the Fine-Grained Geometric Lifting solver and (ii) a held-out-calibrated fusion gate. Inputs: per-frame 2D joint predictions with per-joint covariance, MANO parameter predictions, and camera intrinsics. Outputs: metric 3D joints with per-joint depth variance, a calibrated fusion weight per joint, and a fused 3D hand mesh.

**Load-bearing assumption (stated explicitly).** CALIPER assumes that the per-joint 2D localization covariance fed into the lifting solver is a faithful, well-specified uncertainty estimate whose first-order propagation through the lifting Jacobian yields a depth variance that is monotone in true depth error, so that inverse-variance weighting and isotonic calibration on this signal produce a genuine reliability guarantee. This assumption is load-bearing: if the 2D covariance is anti-correlated with true 2D error on the occluded/truncated joints that ViDiHand2 targets, no monotone recalibration can recover the ordering, and the propagation merely launders a bad ranking into a confident-looking variance. To make the assumption testable rather than assumed, the 2D covariance is not taken from the VDM's self-reported head; it is an empirically validated per-joint 2D error model estimated on the held-out set as a function of observable covariates (joint type, occlusion/visibility flag, foreshortening angle from the MANO pose, image truncation), as specified in Phase 1 below. The calibration map then operates on a signal whose ranking is grounded in measured error rather than a possibly anti-correlated learned variance.

**How it works.**

```pseudocode
# Phase 1: per-joint 2D covariance from an empirically validated error model
# Covariates per joint j in frame t:
#   c_type[j] in {wrist, MCP, PIP, DIP, tip}          (joint type)
#   c_occ[t,j] in [0,1]                                (visibility/occlusion flag from VDM)
#   c_foreshort[t,j] = angle between MANO bone axis and camera ray (from MANO pose)
#   c_trunc[t,j] in {0,1}                              (image-truncation flag)
# Fit on held-out set D_cal (same set used in Phase 3):
#   For each joint type, fit a heteroscedastic Gaussian NLL model
#   Sigma_2d[t,j] = exp( f_theta(c_type[j], c_occ[t,j], c_foreshort[t,j], c_trunc[t,j]) )
#   where f_theta is a 2-layer MLP (hidden=64, GeLU) outputting log-variance,
#   trained with loss L = 0.5 * exp(-f_theta) * ||mu_2d - mu_2d_gt||^2 + 0.5 * f_theta
#   (Kendall & Gal heteroscedastic NLL) on D_cal 2D joint errors.
#   This replaces the VDM head's self-reported Sigma_2d with a covariance whose
#   ranking is grounded in measured 2D error on the deployment distribution.
#   mu_2d[t,j] remains the VDM backbone's 2D joint prediction.

# Phase 2: covariance propagation through the lifting solver
for each frame t:
    # Fine-Grained Geometric Lifting: solve for metric depth z_j s.t.
    #   pi(K, z_j * [u_j, v_j, 1]) aligns with MANO metric bone lengths
    # Linearize the lifting residual r(z; mu_2d) around z_hat:
    J_z = d r / d z      # Jacobian of geometric residual wrt depth
    J_u = d r / d mu_2d  # Jacobian wrt 2D joint position
    Sigma_z[t,j] = (J_z^T J_z)^{-1} J_z^T J_u Sigma_2d[t,j] J_u^T J_z (J_z^T J_z)^{-1}
    z_hat[t,j]   = argmin_z || r(z; mu_2d) ||^2     # existing lifting solve
    # Clip-Level Ray Fitting: fuse per-frame (z_hat, Sigma_z) across the clip
    #   via inverse-variance weighting:
    w[t,j] = 1 / Sigma_z[t,j]
    z_clip[j] = sum_t w[t,j] z_hat[t,j] / sum_t w[t,j]
    Sigma_clip[j] = 1 / sum_t w[t,j]

# Phase 3: calibration of the fusion gate on a held-out set
# Held-out set D_cal: (2D joints, MANO params, ground-truth 3D joints)
# Fit isotonic regression g: confidence -> P(|z_lifted - z_gt| < tau)
#   where confidence = 1 / Sigma_clip[j] (normalized), tau = 5mm
for each joint j in D_cal:
    conf[j] = normalize(1 / Sigma_clip[j])
    err[j]  = |z_lifted[j] - z_gt[j]|
g = IsotonicRegression().fit(conf, err < tau)

# Phase 4: calibrated fusion at inference
# Fusion weight is derived from the propagated variances directly (minimum-variance
#   convex blend under independence), NOT from the calibrated probability:
#   w_fuse[j] = Sigma_mano[j] / (Sigma_clip[j] + Sigma_mano[j])
# The calibrated probability p_reliable = g(conf[j]) is reported as a reliability
#   signal and used only to flag joints for downstream decisions, not to set w_fuse.
for each joint j:
    p_reliable = g(conf[j])                    # calibrated probability of being within tau
    w_fuse     = Sigma_mano[j] / (Sigma_clip[j] + Sigma_mano[j])
    z_final[j] = w_fuse * z_lifted[j] + (1 - w_fuse) * z_mano[j]
    # Propagate fusion variance:
    Sigma_final[j] = w_fuse^2 * Sigma_clip[j] + (1 - w_fuse)^2 * Sigma_mano[j]
```

**Why this design.** We choose first-order covariance propagation through the existing lifting Jacobian over a learned depth-regression head because the lifting solver already encodes the geometric constraint (MANO bone lengths + camera ray), so the Jacobian is the exact sensitivity of the constraint to 2D noise; a learned head would re-introduce the appearance-based regression that ViDiHand2 explicitly replaces, accepting the cost that the propagation is only first-order accurate and degrades when the residual is strongly nonlinear (e.g., near-singular J_z under extreme foreshortening). We choose inverse-variance weighting for Clip-Level Ray Fitting over a learned temporal aggregator because the lifting variance Sigma_z is already a metric of reliability, so the optimal linear unbiased combination is closed-form; a learned aggregator would require training signal that the held-out calibration set cannot provide, accepting the cost that the weighting is only optimal under the linear-Gaussian approximation. We choose isotonic regression on a held-out calibration set over the raw VDM confidence score because isotonic regression is the non-parametric monotone recalibration that guarantees the calibrated probability is monotone in the raw confidence, directly addressing the known failure of uncalibrated log-probs as gates (Kadavath et al.); the cost is that calibration requires a held-out set whose distribution matches deployment, and the guarantee is only as good as that match. We choose a reliability-weighted convex blend over a hard switch because a hard gate discards the MANO prediction entirely and produces discontinuities at the threshold, accepting the cost that the blend can be dominated by a miscalibrated p_reliable. A domain expert would not describe CALIPER as a variant of ViDiHand2 because ViDiHand2's fusion gate is a heuristic on uncalibrated confidence and its lifting is a point estimate; CALIPER's contribution is the covariance propagation operator and the calibration map, neither of which exists in ViDiHand2.

**Why it measures what we claim.** The per-joint variance Sigma_clip[j] measures the *reliability of the lifted metric depth* because the first-order propagation assumes the lifting residual is locally linear in the 2D joint position, which makes the delta-method variance an accurate estimate of the depth error variance; this assumption fails when J_z is near-singular (extreme foreshortening or joint at the camera axis), in which case Sigma_clip reflects the linearized sensitivity rather than the true error and can be over-optimistic. The calibrated probability p_reliable = g(conf[j]) measures the *probability that the lifted depth is within tau of ground truth* because isotonic regression assumes the held-out calibration distribution matches the deployment distribution (exchangeability of the calibration and test joints), which makes the monotone fit a valid conditional probability; this assumption fails under distribution shift (e.g., a new camera or a new object class not represented in D_cal), in which case p_reliable reflects the calibration-set base rate rather than the test-time reliability. The fusion weight w_fuse measures *how much to trust the lifted joint over the MANO prediction* because the convex blend assumes the two error sources are independent, which makes the reliability-weighted combination the minimum-variance unbiased blend; this assumption fails when the MANO prediction and the lifted joint share the same VDM backbone errors, in which case w_fuse over-weights the lifted joint relative to the true minimum-variance blend. The error-attenuation invariant — that Sigma_clip is monotone in the 2D covariance Sigma_2d under fixed Jacobians — measures *the claimed attenuation of 2D noise through lifting* because the propagation formula assumes J_u and J_z are evaluated at the true depth, which makes the variance map order-preserving; this assumption fails when the linearization point z_hat is far from the true depth, in which case the monotonicity can invert and the invariant is only approximate. The empirical 2D error model Sigma_2d measures *the true 2D localization error* because it is fit on held-out 2D joint errors as a function of observable covariates, which makes its ranking grounded in measured error; this assumption fails when the covariates do not capture the dominant error mode (e.g., a new occlusion pattern), in which case Sigma_2d reflects the calibration-set error distribution rather than the test-time error.

## Contribution

['A covariance-propagation operator through the Fine-Grained Geometric Lifting solver that yields per-joint metric depth variance and an explicit error-attenuation invariant under fixed lifting Jacobians.', 'A held-out isotonic calibration map from lifting confidence to the probability that lifted depth is within a tolerance, converting the fusion gate from a heuristic into a reliability-weighted convex blend with propagated fusion variance.', 'An empirical demonstration that calibrated fusion variance is monotone in true depth error and that the calibrated gate outperforms the uncalibrated confidence gate under occlusion and truncation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | In-the-wild egocentric hand video (ViDiHand2 benchmark) | Matches deployment: occlusion, truncation, unseen cameras |
| Primary metric | Per-joint metric depth error, split by localization confidence | Isolates reliability signal from raw accuracy |
| Baseline | ViDiHand2 (uncalibrated confidence gate) | Direct predecessor; tests calibration sub-claim |
| Baseline | Point-estimate lifting, no covariance propagation | Tests covariance-propagation sub-claim |
| Baseline | Hard-threshold fusion gate | Tests blend-vs-switch sub-claim |
| Ablation-of-ours | CALIPER with raw VDM confidence (no isotonic calibration) | Isolates the calibration map's contribution |

### Why this setup validates the claim

The central claim is that replacing ViDiHand2's point-estimate lifting and uncalibrated confidence gate with covariance propagation plus a held-out-calibrated fusion gate yields more reliable metric depth, specifically where 2D localization is uncertain. The dataset choice is forced: ViDiHand2's own evaluation is in-the-wild egocentric video with occlusion and truncation, which is exactly where the covariance and calibration machinery is supposed to matter, and where a distribution-shifted camera stresses the calibration assumption. The primary metric is deliberately not aggregate MPJPE but per-joint depth error stratified by predicted confidence, because an aggregate number cannot distinguish "our method is uniformly slightly better" (which would not support the mechanism story) from "our method is much better precisely on the joints it flags as unreliable" (which would). Each baseline tests one sub-claim: ViDiHand2 tests whether calibration beats raw confidence; the point-estimate-lifting variant tests whether covariance propagation adds anything beyond the existing solver; the hard-gate variant tests whether the convex blend beats a threshold switch. The ablation with raw VDM confidence inside our own pipeline isolates the isotonic map from the covariance machinery, so a gain cannot be attributed to either alone. Together these form a falsifiable design: the mechanism predicts a *heterogeneous* gain concentrated on low-confidence, high-foreshortening joints, not a uniform improvement.

### Expected outcome and causal chain

**vs. ViDiHand2 (uncalibrated confidence gate)** — On a frame where the hand is partially occluded and the VDM's 2D log-probs are confidently wrong (a known failure of uncalibrated confidence), ViDiHand2's gate trusts the lifted joint and inherits the depth error, because its gate is a heuristic on a score that was never trained to predict error. Our method instead maps that raw confidence through an isotonic fit to the empirical probability of being within tau of ground truth, so an overconfident-but-wrong joint receives a low calibrated weight and the MANO prediction dominates. We expect a visible gap concentrated on the occluded/truncated subset, with near-parity on clean, well-localized joints.

**vs. Point-estimate lifting without covariance propagation** — On a joint near the camera axis or under extreme foreshortening, the lifting residual is ill-conditioned; a point-estimate solver returns a depth with no indication that its sensitivity to 2D noise is large, so downstream fusion treats it as equally trustworthy as a well-conditioned joint. Our method linearizes the lifting residual and propagates the 2D covariance through the Jacobian, so the near-singular J_z inflates Sigma_z and the joint is down-weighted. We expect the largest error reduction on the high-foreshortening joint subset and roughly unchanged error elsewhere, which is the signature of a variance-aware rather than accuracy-only improvement.

**vs. Hard-threshold fusion gate** — On joints whose confidence sits near the threshold, a hard gate flips discretely between lifted and MANO predictions frame to frame, producing temporal jitter and occasional large errors when the switch is wrong. Our method uses a calibrated probability as a convex weight, so near-threshold joints are blended rather than abruptly reassigned, and the fused variance shrinks smoothly. We expect reduced temporal variance of per-joint depth across consecutive frames and fewer large-error outliers, with similar mean error on confidently-lifted joints.

### What would falsify this idea

If the gain over ViDiHand2 is uniform across confidence strata and foreshortening subsets rather than concentrated on the low-confidence, high-foreshortening joints where the mechanism predicts its effect, the covariance-and-calibration story is wrong and the improvement (if any) comes from something else. Likewise, if the raw-confidence ablation matches full CALIPER, the isotonic calibration map contributes nothing and the central claim about calibrated reliability fails.

## References

1. ViDiHand2: Unifying Video Diffusion Priors and Hand Geometry for 3D Hand Motion Reconstruction in the Wild
