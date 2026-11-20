# Warp-Residual Certification for Temporally Consistent Metric Monocular Depth

## Motivation

Depth Pro 2 (arxiv:2611.26603) reports that its temporal consistency is an emergent property of jointly training one network on images and video, and its abstract offers no structural constraint that can be checked independently of the training objective; the same is true of Align3R (arxiv:2412.03079), which attains consistent video depth by fine-tuning DUSt3R and running a post-hoc optimization rather than by imposing an estimator-level invariant. The root cause is shared across this lineage: consistency is treated as something to be learned from multi-frame data, so at test time nothing forbids a per-pixel depth from drifting between frames under camera motion the model has already implicitly estimated. This is a meta-level gap — no branch of the tree turns 'the depth field must be geometrically coherent across frames' into a quantity the estimator itself computes and is held to.

## Key Insight

Because a metric depth map together with an estimated focal length and a relative pose fully determines where every pixel must land in the next frame, the cross-frame depth-warp residual is a self-supervised, pose-conditioned consistency functional whose value is an observable invariant of the estimator rather than a statistic of the training distribution.

## Method

**What it is.** We introduce WARDen (Warp-Residual Depth-invariant Estimator), a training-and-inference wrapper around a Depth Pro 2-style unified image/video backbone. Inputs: a short clip of frames, the backbone's native-resolution metric depth maps, and the backbone's own focal-length and relative-pose estimates. Outputs: depth maps trained under an explicit warp-consistency objective, plus, at inference, a per-pixel certified drift field and a pass/fail certificate against a user-specified bound.

**Load-bearing assumption (stated explicitly).** The certificate and the training loss both rely on the assumption that the backbone's own focal-length and relative-pose estimates are accurate enough that the backproject–rigid-transform–reproject identity holds for static pixels, so any nonzero warp residual r_t is attributable to depth error rather than to pose/focal error. This assumption is not taken for granted: we add a joint (depth, pose, focal) refinement step inside the warp objective and a calibration check that residual magnitude tracks injected pose error, so the certificate is reported only on the depth-attributable component of the residual. The refinement and calibration are specified in Phases 2b and 2c below.

**How it works.**

```pseudocode
# Phase 1 — geometric quantities (from the backbone, no new network)
for frame t in clip:
    D_t, f_t   = backbone.depth_and_focal(I_t)          # metric depth, focal length
    P_t        = backbone.pose(I_t, I_{t-1})            # relative pose, 6-DoF

# Phase 2 — differentiable cross-frame warp residual (training signal)
for each adjacent pair (t-1, t):
    X_t   = backproject(D_t, f_t)                        # pixel -> 3D in frame t
    X_hat = T_{t<-t-1} @ X_t                             # rigid transform into frame t-1
    u_hat = project(X_hat, f_{t-1})                      # predicted pixel coords
    r_t   = | D_{t-1}[u_hat] - X_hat.z |                 # depth-warp residual per pixel
    r_t  *= visibility_mask(u_hat, occlusion_thresh=0.1) # drop out-of-frame / occluded

# Phase 2b — joint (depth, pose, focal) refinement inside the warp objective
#   For each pair, run K=3 Gauss-Newton iterations over the 6-DoF pose and the
#   two focal lengths (f_{t-1}, f_t), holding D_t and D_{t-1} fixed at their
#   current values, to minimize the robust warp residual on the static mask.
#   The refined pose/focal are used to recompute r_t; the depth-attributable
#   residual is r_t^depth = r_t evaluated at the refined geometry.
#   This mirrors the affine-correction solvers of arXiv:2501.05446, which
#   explicitly model scale/shift ambiguity rather than trusting a single
#   forward pass.
for each adjacent pair (t-1, t):
    (T_refined, f_refined) = gauss_newton_warp_refine(
        D_t, D_{t-1}, T_{t<-t-1}, f_t, f_{t-1},
        iters=3, robust_kernel='huber', delta=0.05)
    r_t^depth = warp_residual(D_t, D_{t-1}, T_refined, f_refined)  # depth-attributable
    r_t^geom  = | r_t - r_t^depth |                                  # geometry-attributable

# Phase 2c — pose/focal-error calibration check (no gradient)
#   Inject known perturbations into f and T on a held-out calibration set of
#   512 clip-pairs; record how r_t^geom responds. Fit a scalar sensitivity
#   s = d(mean r_t^geom)/d(pose_error) and report it alongside the certificate.
#   If s exceeds a pre-registered threshold (s > 0.5 m per 1% pose error),
#   the certificate is flagged as geometry-dominated and withheld.

# Phase 3 — bounded-drift regularizer (training loss term)
L_warp  = mean( huber(r_t^depth, delta=0.05) )
L_drift = max(0, mean(r_t^depth) - tau)^2                # tau = drift budget, e.g. 0.02 m
L_total = L_depth + lambda_w * L_warp + lambda_d * L_drift   # lambda_w=0.5, lambda_d=1.0

# Phase 4 — inference-time certification (no gradient)
for each pixel p and pair (t-1,t):
    drift[p] = r_t^depth[p]                              # measured on refined geometry
certificate = PASS  if  quantile_95(drift) <= B  and  s <= 0.5   # B = user bound, e.g. 0.03 m
              FAIL  otherwise, with failing pixels returned
```

**Why this design.** We chose to derive the warp residual from the backbone's *own* focal-length and pose estimates rather than from ground-truth camera intrinsics or external SfM, because the certification claim is about the estimator's internal geometric coherence, and importing external geometry would make the certificate a statement about the external solver rather than about the depth model; the cost is that focal-length and pose errors propagate into the residual and inflate it, which is why Phase 2b jointly refines pose and focal inside the warp objective and Phase 2c calibrates the geometry-attributable component before the certificate is issued. We chose a differentiable residual as a *training* loss in addition to a *test-time* measurement, rather than using it only for evaluation, because a measurement-only residual cannot shape the estimator and would leave the invariant emergent — exactly the gap in Depth Pro 2 (arxiv:2611.26603); the cost is that the training signal is only as good as the pose head, which the joint refinement in Phase 2b is designed to mitigate. We chose a hinge-form bounded-drift term (`max(0, mean(r) - tau)^2`) over a plain L1 warp loss, because a plain loss keeps pushing residuals toward zero on pixels where the pose is wrong and can degrade sharpness, whereas the hinge leaves residuals below the budget untouched; the cost is a non-smooth objective and a budget hyperparameter that must be set per dataset scale. This is not a variant of Align3R (arxiv:2412.03079): Align3R fine-tunes DUSt3R and then runs a post-hoc optimization to align depth maps, producing consistency as an output of optimization, whereas WARDen installs the residual as a term in the estimator's own objective and reports its value as a certificate; a domain expert would not describe a training-time invariant with an inference-time bound as 'Align3R with extra steps'.

**Why it measures what we claim.** The computational quantity `r_t^depth` measures *temporal geometric consistency* because the assumption that the backbone's focal length and relative pose are accurate — after the Phase 2b joint refinement — makes backprojection, rigid transform, and reprojection an exact identity for a static scene, so any nonzero residual is attributable to depth error; this assumption fails when a pixel belongs to a moving object or a specular surface, in which case `r_t^depth` reflects scene motion rather than depth drift and the visibility mask only partially removes it. The quantity `mean(r_t^depth)` measures the *estimator-level drift* that the motivation names because the assumption that residuals are spatially uncorrelated across a clip makes the mean an unbiased summary of systematic drift rather than of per-pixel noise; this assumption fails under rolling shutter or strong lens distortion, in which case the mean reflects a global geometric bias shared by all pixels. The quantity `quantile_95(drift)` measures the *certified per-pixel bound* because the assumption that the residual distribution is stationary within a clip makes the 95th percentile a stable estimate of the worst-case drift a user will encounter; this assumption fails on clips shorter than roughly ten frames, in which case the percentile is dominated by sampling noise. The quantity `r_t^geom` measures the *geometry-attributable residual* because the assumption that the joint refinement in Phase 2b absorbs pose and focal error makes the difference between the raw and refined residuals a proxy for geometry error; this assumption fails when the refinement converges to a wrong local minimum, in which case `r_t^geom` under-reports geometry error and the Phase 2c sensitivity check `s` is the backstop.

## Contribution

(1) A warp-residual training objective with a hinge-form bounded-drift regularizer that turns cross-frame geometric coherence into a term in the depth estimator's own loss, using the model's own focal-length and pose estimates. (2) An inference-time certification procedure that returns a per-pixel drift field and a pass/fail verdict against a user-specified bound, making temporal consistency an auditable property rather than a reported average. (3) A design principle — that the same residual should be both trained on and certified against — with an analysis of where the certificate is and is not valid (moving objects, short clips, distortion).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Dynamic video clips (e.g. Sintel/Bonn-style) | Contains motion and occlusion to stress the residual. |
| Primary metric | Warp-residual quantile-95 drift (m), computed on `r_t^depth` after Phase 2b refinement | Directly measures the certified quantity claimed. |
| Secondary metric | Geometry-attributable residual `mean(r_t^geom)` and sensitivity `s` (Phase 2c) | Quantifies how much residual is pose/focal error. |
| Baseline | Depth Pro 2 | Tests whether residual-as-loss beats consistency-by-architecture. |
| Baseline | Align3R | Tests whether training-time invariant beats post-hoc optimization. |
| Baseline | Depth Pro (single-frame) | Tests whether temporal objective beats sharpness-only estimation. |
| Baseline | WARDen with ground-truth pose/intrinsics (oracle geometry) | Upper-bounds the depth-attributable residual. |
| Ablation-of-ours | WARDen without L_drift hinge | Tests whether the bounded-drift term is doing the work. |
| Ablation-of-ours | WARDen without Phase 2b joint refinement | Tests whether refinement is needed for attribution. |
| Ablation-of-ours | WARDen with injected pose/focal perturbations (0.5%, 1%, 2%) | Calibrates residual response to geometry error. |

### Why this setup validates the claim

The central claim is that installing the warp residual as a term in the estimator's own objective yields measurable, bounded temporal geometric drift, and that this drift can be reported as a certificate. The primary metric is chosen to be exactly the quantity the certificate reports, so the evaluation is a direct test rather than a proxy: if the residual shrinks under training, the certificate tightens, and if it does not, the claim fails. The secondary metrics and the oracle-geometry baseline address the load-bearing assumption directly: by comparing `r_t^depth` under estimated versus ground-truth pose/intrinsics, and by measuring the sensitivity `s` of `r_t^geom` to injected pose/focal perturbations, we quantify how much of the residual is depth error versus geometry error. Depth Pro 2 tests the sub-claim that a unified backbone alone does not install the invariant; Align3R tests the sub-claim that post-hoc optimization cannot substitute for a training-time term; Depth Pro tests the sub-claim that the temporal objective, not just sharpness, drives the gain. The ablations isolate the bounded-drift hinge from the plain warp loss and the joint refinement from the raw residual, so a gain attributable to the hinge or to the refinement is distinguishable from a gain from any cross-frame signal at all. Together these form a falsifiable test: the predicted pattern is a residual drop concentrated on dynamic, occluded, and boundary regions, with parity on static flat regions, and a geometry-attributable residual that stays below the pre-registered sensitivity threshold.

### Expected outcome and causal chain

**vs. Depth Pro 2** — On a clip with a moving foreground object and a receding background, Depth Pro 2 produces sharp but temporally inconsistent depth because its single-frame sharpness objective and multi-frame consistency objective are traded off rather than jointly constrained by an explicit geometric identity. Our method instead backprojects, rigidly transforms, and reprojects using the backbone's own focal length and pose, refined jointly in Phase 2b, so the residual penalizes exactly the depth values that violate the identity. We expect a noticeable drop in quantile-95 drift on dynamic and boundary subsets, with near-parity on static flat regions where both already agree.

**vs. Align3R** — On a clip where the estimated depth maps drift gradually across frames, Align3R produces consistency only after a post-hoc alignment optimization, so the underlying estimator is unchanged and the aligned depth can still violate the warp identity at occlusion boundaries. Our method instead installs the residual in the estimator's objective, so the network itself is shaped to satisfy the identity, and the residual value is reported rather than optimized away. We expect Align3R to show low mean drift but higher quantile-95 drift at boundaries, while ours reduces both, with the gap concentrated where alignment cannot correct per-pixel error.

**vs. Depth Pro (single-frame)** — On a clip with camera translation, the single-frame model produces per-frame depth that is sharp but mutually inconsistent, because it has no cross-frame constraint at all. Our method adds the warp residual as a training term on the unified backbone, so the same instance's depth is pulled toward the rigid-transform identity across adjacent frames. We expect a large quantile-95 drift gap on translating clips and a smaller gap on near-static clips, confirming the effect is tied to the geometric constraint rather than to general depth quality.

**vs. WARDen with oracle geometry** — On the same clips, replacing the backbone's estimated pose and focal length with ground-truth values removes the geometry-attributable component `r_t^geom` by construction. We expect the oracle-geometry variant to show a lower `mean(r_t^geom)` and a lower sensitivity `s`, and the gap between the two variants to bound the pose/focal contribution to the certified residual. If the gap is large relative to the depth-attributable residual, the load-bearing assumption is not met and the certificate is flagged.

### What would falsify this idea

If the residual gain is uniform across static, dynamic, and boundary subsets rather than concentrated where the predicted failure mode occurs, the mechanism is not the warp identity and the central claim is wrong. Likewise, if removing the bounded-drift hinge leaves the certificate unchanged, the hinge is not doing the work the argument attributes to it; and if the geometry-attributable residual `r_t^geom` or the sensitivity `s` exceeds the pre-registered threshold, the certificate is measuring the pose head rather than depth drift and the attribution claim fails.

## References

1. Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth
