# Covariance-Propagated Surfel Voting for Pointmap-Based Surface Reconstruction

## Motivation

Across the current tree of pointmap- and splat-based surface reconstruction methods, the primitive-selection and carving stages are treated as deterministic even though they consume geometric priors from upstream models whose errors are spatially correlated and uncalibrated. FoundSurface (arxiv:2611.34128) selects reference views via a 'robust surfel voting mechanism' and carves high-frequency detail with a raylet module, but the abstract reports no estimate of pointmap-induced geometric bias, so consistency of the selected surfels or the carved surface under pointmap noise is never established; PlanarSplatting (E1) and 3D Gaussian Flats (E6) exhibit the same structural gap in their plane/surfel selection. The root cause is that no method in this tree propagates the pre-trained pointmap's error structure into the quantities that drive primitive selection, so any downstream 'robustness' is incidental rather than calibrated.

## Key Insight

The covariance of a pre-trained pointmap is a transportable sufficient statistic: propagating it analytically through the surfel-voting score and the raylet carving integral yields a per-surfel posterior that is provably invariant to the correlated component of pointmap distortion, which no deterministic selection score can achieve because it discards exactly the second-moment information that identifies correlated error.

## Method

**What it is.** We introduce **CPS-Vote** (Covariance-Propagated Surfel Voting), a module that sits between FoundSurface's pre-trained pointmap model and its surfel-voting / raylet stages. It takes the pointmap's per-pixel means and an estimated spatial error covariance field, and outputs (i) a probabilistic surfel-vote score per candidate reference view and (ii) a covariance-weighted raylet carving weight, together with a consistency loss that enforces surface agreement under perturbed pointmap inputs.

**Load-bearing assumption (stated explicitly).** CPS-Vote assumes that the pointmap's error structure can be captured by a single stationary, scene-agnostic Matérn covariance field Sigma(x) calibrated once on a small set of GT-depth scenes, and that this same Sigma — transported through a first-order Jacobian — correctly predicts the per-surfel uncertainty that drives view selection and raylet carving on unseen zero-shot scenes. This assumption is load-bearing because the entire precision-weighting story depends on Sigma being calibrated in the high-residual regimes (planar boundaries, thin structures) where the claimed gain appears. We state it plainly so that the calibration and verification steps below can be read as direct tests of it.

**How it works.**

```pseudocode
# Phase 1: estimate pointmap error structure (offline, one calibration pass)
# Calibration set: 512 held-out scenes with GT depth (from ScanNet++ train split, disjoint from zero-shot eval).
for each of N_cal = 512 scenes with GT depth:
    residual r_i = pointmap_i - gt_point_i
fit a parametric spatial covariance  Sigma(x) = sigma0^2 * K_theta(x, x')   # K_theta = Matern(nu=3/2, length=ell), ell, sigma0 fit by marginal likelihood
# Calibration verification (proper scoring rule): on a held-out 128-scene calibration split,
# report negative log-likelihood (NLL) and 90% interval coverage of residuals under the fitted Sigma.
# If coverage < 0.85, refit with a discontinuity-aware kernel (see Phase 1b).
# Phase 1b (fallback if stationarity fails): heteroscedastic/non-stationary GP with a
# discontinuity-aware kernel: Sigma(x) = sigma0^2(x) * K_theta(x, x'), where sigma0^2(x) is a
# per-pixel variance predicted from the pointmap's own features (2-layer MLP, hidden=256, GeLU),
# and K_theta uses a local length-scale ell(x) predicted by the same MLP. Calibrate by NLL on the
# 128-scene held-out split; accept only if 90% coverage >= 0.85.
# Phase 2: propagate covariance to surfels
for each surfel s with anchor pixel p_s and normal n_s:
    J_s = d(surfel_params)/d(pointmap_pixels)         # analytic Jacobian of surfel fit
    C_s = J_s @ Sigma(p_s) @ J_s^T + eps*I             # per-surfel covariance (eps=1e-6)
    # Near grazing surfels (angle between n_s and view ray < 15 deg), replace the first-order
    # transport with a few-sample Monte-Carlo estimate: draw M=8 perturbations delta ~ Sigma(p_s),
    # refit the surfel for each, and set C_s = empirical covariance of the M refitted surfels.
    # probabilistic vote: instead of a deterministic score, use a Gaussian likelihood
    for each candidate reference view v:
        d_{s,v} = signed distance of surfel s to view v's ray bundle
        w_{s,v} = exp( -0.5 * d_{s,v}^T (n_s^T C_s n_s + tau_v^2)^{-1} d_{s,v} )   # tau_v = view noise, tau_v=0.01
    normalize {w_{s,v}} over v; select top-K views (K=8) with cumulative weight >= 0.95
# Phase 3: covariance-weighted raylet carving
for each query ray q with raylet features f_q and ray direction n_q:
    weight_q = 1 / (1 + n_q^T C_s(q) n_q)               # directional uncertainty along the carving ray
    carved_surface = sum_q weight_q * f_q / sum_q weight_q
# Phase 4: consistency loss under perturbed pointmaps
train with L = L_geom + lambda_cons * E_{delta ~ Sigma} || S(P) - S(P + delta) ||^2   # lambda_cons=0.1, delta sampled from Sigma
```

**Why this design.** We chose a *parametric Matern covariance* over a learned per-pixel variance head because the calibration pass has ground-truth depth available (a small calibration set) and a parametric form gives a closed-form Jacobian transport, accepting that the covariance is stationary and may under-fit scene-specific anisotropy. We chose *analytic Jacobian propagation* (J_s @ Sigma @ J_s^T) over Monte-Carlo sampling of perturbed pointmaps because the surfel-fit Jacobian is available in closed form and Monte-Carlo would require many forward passes per surfel, accepting that the linearization is only first-order accurate near the surfel. We chose a *Gaussian likelihood vote* over a learned attention score because the Gaussian form makes the vote weights directly interpretable as inverse-covariance precision, accepting that the Gaussian assumption may be violated at occlusion boundaries. A domain expert would not describe this as a variant of FoundSurface's surfel voting: FoundSurface's vote is a deterministic geometric score with no second-moment information, whereas CPS-Vote's vote is a precision-weighted likelihood whose weights change when the pointmap's correlated error changes, which is the property that makes perturbation-invariance measurable.

**Why it measures what we claim.** The per-surfel covariance C_s measures *calibrated geometric uncertainty* because the Matern fit on held-out calibration scenes makes C_s equal to the empirical residual covariance under the assumption that pointmap residuals are stationary and Gaussian; this assumption fails at depth discontinuities, where C_s reflects the smooth-region covariance rather than the true boundary uncertainty. The probabilistic vote weight w_{s,v} measures *perturbation-invariance of view selection* because the Gaussian likelihood is exactly the probability that the surfel's signed distance is zero under the propagated covariance, so a view whose score is stable under Sigma-perturbation is one whose selection is invariant, under the assumption that the linearization J_s is accurate; this assumption fails for surfels near grazing angles, in which case w reflects the linearized precision rather than true invariance. The consistency loss term ||S(P)-S(P+delta)||^2 measures *surface agreement under pointmap noise* because delta is sampled from the calibrated Sigma rather than an arbitrary perturbation, so the loss is zero exactly when the reconstructed surface is invariant to the calibrated error distribution; this fails when Sigma under-estimates tail errors, in which case the loss measures agreement under a narrower noise model than reality. The carving weight 1/(1+n_q^T C_s(q) n_q) measures *directional uncertainty suppression along the carving ray* because n_q^T C_s n_q is the variance of the surfel position projected onto the ray direction, under the assumption that the ray direction is the relevant axis for carving; this assumption fails when C_s is strongly anisotropic and the ray direction is not aligned with the dominant uncertainty axis, in which case the weight reflects the projected variance rather than the full uncertainty ellipsoid.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ScanNet++ zero-shot split | Indoor scenes with GT depth and pose-free evaluation |
| Primary metric | Surface F-score at 5cm threshold | Sensitive to both accuracy and boundary detail |
| Secondary metric | NLL and 90% interval coverage of residuals under Sigma on 128 held-out calibration scenes | Directly tests the load-bearing stationarity assumption |
| Baseline | FoundSurface (deterministic surfel vote) | Tests whether precision-weighting beats geometric scoring |
| Baseline | OffsetOPT | Tests explicit reconstruction without normals on same inputs |
| Baseline | 3D Gaussian Flats | Tests hybrid planar/freeform alternative for flat surfaces |
| Baseline | Learned per-pixel variance head (heteroscedastic reweighting, no spatial correlation) | Isolates whether correlated Matern component is necessary |
| Ablation-of-ours | CPS-Vote w/o covariance (uniform weights) | Isolates the contribution of covariance propagation |
| Ablation-of-ours | CPS-Vote with trace-based carving weight 1/(1+tr(C_s)) | Tests directional vs isotropic carving weight |
| Ablation-of-ours | CPS-Vote with stationary Matern only (no Phase 1b fallback) | Tests whether discontinuity-aware kernel is needed |

### Why this setup validates the claim

The central claim is that propagating calibrated pointmap covariance into surfel voting and raylet carving yields surfaces that are more stable under the pointmap's actual error distribution. ScanNet++ zero-shot is the right arena because FoundSurface is trained elsewhere and evaluated here, so any gain cannot come from dataset-specific overfitting; it must come from better handling of the pointmap residuals that remain on unseen scenes. FoundSurface tests the sub-claim that a precision-weighted likelihood vote beats a deterministic geometric score: same backbone, same inputs, only the vote changes. OffsetOPT tests whether the covariance-weighted carving helps even when the pipeline is explicit and normal-free, probing whether the gain is specific to the surfel/raylet route. 3D Gaussian Flats tests the flat-surface regime where covariance should matter most, since texture-less planes are where pointmap residuals are largest and most correlated. The learned per-pixel variance head baseline is the decisive control for novelty: if a heteroscedastic reweighting with no spatial correlation matches CPS-Vote, then the gain is generic variance weighting and the correlated Matern component is not doing the work. The ablation (uniform weights, no covariance) is the decisive control for the mechanism: if covariance propagation is doing the work, removing it should collapse the gain precisely on high-residual regions. The trace-based carving ablation tests whether the directional weight n_q^T C_s n_q is necessary or whether isotropic trace suffices. The stationary-only ablation tests the load-bearing assumption directly: if the discontinuity-aware fallback is needed to reach 90% coverage, the stationary Matern is insufficient. Surface F-score at 5cm is chosen because it rewards both completeness and boundary sharpness, which is exactly where a precision-weighted vote should show up rather than a uniform accuracy shift. The NLL/coverage metric on held-out calibration scenes directly verifies the load-bearing assumption before any zero-shot evaluation is run.

### Expected outcome and causal chain

**vs. FoundSurface** — On a case where a texture-less wall meets a doorframe, the pointmap residuals are large and spatially correlated along the wall, so FoundSurface's deterministic surfel vote admits reference views whose rays graze the wall at a shallow angle, producing a smeared, slightly bowed surface near the frame. Our method instead down-weights those views because the propagated covariance along the wall normal inflates the signed-distance variance, so the Gaussian likelihood suppresses grazing views and selects views with tight normal-direction precision. We expect a noticeable F-score gap concentrated on planar-boundary subsets, with parity on cluttered, high-texture regions where residuals are small. The high-residual vs low-residual subset analysis is the causal test: if the gain is uniform across subsets, the covariance is not the mechanism.

**vs. OffsetOPT** — On a case where a thin chair leg sits against a wall, OffsetOPT predicts triangles from local point geometry and, without normals, tends to bridge the leg to the wall where the point sampling is sparse, producing a webbed artifact. Our method instead carries a per-surfel covariance that is anisotropic at the leg (tight across the leg, loose along it), so the raylet carving weight suppresses contributions from rays whose uncertainty ellipsoid spans the gap. We expect fewer spurious bridging triangles and a higher F-score on thin-structure subsets, with comparable results on large open surfaces.

**vs. 3D Gaussian Flats** — On a case where a flat ceiling has a subtle slope and a light fixture, the hybrid representation fits constrained planar Gaussians that over-regularize the slope and leave a residual shell around the fixture, because its planar detection is photometrically driven and the fixture breaks the flatness assumption. Our method instead keeps the covariance-based vote, which does not assume planarity, so the fixture's surfels retain their own precision and are carved rather than absorbed into the plane. We expect a modest but consistent F-score advantage on sloped-plane and fixture-adjacent subsets, with parity on pure planar regions.

**vs. learned per-pixel variance head (no spatial correlation)** — On the same texture-less wall/doorframe case, the variance-head baseline reweights rays by per-pixel variance but cannot distinguish correlated along-wall error from independent noise, so it admits the same grazing views as FoundSurface. CPS-Vote suppresses them because the Matern covariance inflates the signed-distance variance along the wall normal. We expect CPS-Vote to outperform the variance-head baseline specifically on planar-boundary subsets, with parity on high-texture regions; if the variance-head baseline matches CPS-Vote, the correlated component is not necessary and the novelty claim fails.

**vs. CPS-Vote with trace-based carving weight** — On a thin chair leg against a wall, the trace-based weight 1/(1+tr(C_s)) is isotropic and cannot distinguish uncertainty across the leg from uncertainty along it, so it under-suppresses rays that span the gap. The directional weight 1/(1+n_q^T C_s n_q) suppresses exactly those rays. We expect the directional weight to yield fewer bridging artifacts and a higher F-score on thin-structure subsets; if trace and directional weights match, the anisotropy of C_s is not being exploited.

**vs. CPS-Vote with stationary Matern only** — On scenes with many depth discontinuities, the stationary Matern under-estimates uncertainty at boundaries (coverage < 0.85), so the vote admits grazing views near edges. The discontinuity-aware fallback (Phase 1b) restores coverage to >= 0.85 and suppresses those views. We expect the fallback to improve F-score on boundary subsets; if stationary-only matches the fallback, the load-bearing stationarity assumption holds on ScanNet++ and the fallback is unnecessary.

### What would falsify this idea

If the gain over FoundSurface is uniform across high-residual and low-residual subsets rather than concentrated where the calibrated covariance predicts large uncertainty, then the covariance is not the causal mechanism and the claim is wrong. Equivalently, if the no-covariance ablation matches the full method on planar-boundary and thin-structure subsets, or if the learned per-pixel variance head (no spatial correlation) matches CPS-Vote on those subsets, the precision-weighting story fails.

## References

1. FoundSurface: Feedforward Ray-Based 3D Scene Surface Reconstruction from Unposed Images
