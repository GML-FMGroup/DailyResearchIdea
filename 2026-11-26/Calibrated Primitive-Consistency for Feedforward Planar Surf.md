# Calibrated Primitive-Consistency for Feedforward Planar Surface Reconstruction

## Motivation

Feedforward surface reconstruction (FoundSurface) selects reference views via surfel voting on a pre-trained pointmap, while planar-primitive pipelines (PlanarSplatting, 3D Gaussian Flats) detect planar regions from photometric depth; both selection modules are treated as deterministic and are never tied to a calibrated noise model of their upstream priors, so consistency under pointmap error or plane-detection misclassification is unestablished. This shared structural gap means a small perturbation to the pointmap or depth map can silently change which primitives are selected, and neither branch can detect or correct it because the selection objective has no error-propagation term. The field's trajectory toward unposed, feedforward reconstruction makes this bottleneck binding: as priors become more heterogeneous (pointmaps, monocular depth, planar fits), the absence of a shared calibrated consistency objective turns primitive selection into an unmonitored failure point.

## Key Insight

A shared perturbation-invariance loss, weighted by each branch's own calibrated noise model, makes primitive selection a measurable function of prior uncertainty rather than of the prior's raw output, so a selection is consistent exactly when it is stable under the noise the upstream prior is known to produce.

## Method

**What it is.** We introduce CalibPC (Calibrated Primitive Consistency), a training objective and inference-time selection rule that couples FoundSurface's surfel-voting reference-view selection with planar-region detection in a planar-primitive branch. Inputs: unposed multi-view images, a pre-trained pointmap model, and photometric depth maps. Outputs: a set of selected reference views, a set of planar primitives, and a continuous surface reconstructed by FoundSurface's ray-based regressor.

**Load-bearing assumption (stated explicitly).** CalibPC assumes that the perturbation distribution used to fit sigma_b is distributionally representative of the upstream prior's actual error at test time, including its spatial correlation structure and tail behavior. If this holds, minimizing L_cons under perturbations drawn from sigma_b yields selection that is stable under the errors that actually occur. If it fails — e.g., pointmap/depth errors are heavy-tailed or spatially correlated on textureless walls — then L_cons reflects sensitivity to the fitted perturbation model rather than to real prior error, and the selected primitives can be stable under the perturbation but biased at test time. The calibration and verification steps below are designed to test this assumption rather than assert it.

**How it works.**

```pseudocode
# Phase 1: Calibrate each branch's prior error model from held-out multi-view geometric consistency
# (no surface labels used; calibration scenes are disjoint from training and test)
for branch b in {pointmap, photometric_depth}:
    # Collect per-pixel/per-point prior error proxies from cross-view agreement:
    #   e_b(x) = prior_b(view_i, x) - warp(prior_b(view_j, x), T_ij)   for overlapping view pairs (i,j)
    # Fit a spatially correlated, heavy-tailed error model to e_b:
    #   - estimate spatial correlation length l_b via variogram fit on e_b (range parameter)
    #   - estimate tail index alpha_b via Hill estimator on |e_b|
    #   - fit scale sigma_b(d) as a monotone function of local prior disagreement d
    #     using a 2-parameter model: sigma_b(d) = s_b * (d / d_ref)^gamma_b, fit by least squares
    # Store (l_b, alpha_b, s_b, gamma_b) per branch; these define the perturbation sampler.

# Phase 2: Shared perturbation-invariance objective (training)
for each training scene:
    # Cache prior features once per scene (pointmap/depth forward pass done once)
    F_pm = pointmap_features(images)      # cached
    F_pl = photometric_depth(images)      # cached
    S_pm  = surfel_vote(F_pm)             # FoundSurface selection scores
    S_pl  = planar_region_scores(F_pl)    # planar-primitive detection scores
    for k in 1..K (K=4 perturbations):
        # Sample structured perturbations from the fitted error model, not i.i.d. Gaussian:
        #   delta_k^b ~ correlated heavy-tailed field with correlation length l_b and tail index alpha_b,
        #   scaled pointwise by sigma_b(d) from Phase 1
        delta_k^pm = sample_correlated_field(l_pm, alpha_pm, sigma_pm(d))
        delta_k_pl = sample_correlated_field(l_pl, alpha_pl, sigma_pl(d))
        # Perturb cached features only (lightweight selection head rerun; no prior re-forward)
        S_pm^k = surfel_vote(F_pm + delta_k^pm)
        S_pl^k = planar_region_scores(F_pl + delta_k_pl)
    L_cons = mean_k [ (S_pm - S_pm^k)^2 / sigma_pm^2 + (S_pl - S_pl^k)^2 / sigma_pl^2 ]
    L_task = FoundSurface reconstruction loss (coarse surface + raylet detail)
    L = L_task + lambda * L_cons        # lambda = 0.1, K = 4 fixed a priori (no sweep)

# Phase 3: Inference-time calibrated selection with OOD gate
# Compute test-time disagreement statistic D_test = mean local prior disagreement on test scene
# Compare to calibration distribution of D (from Phase 1 held-out scenes):
if D_test > tau_OOD (95th percentile of calibration D):
    # Fall back to deterministic selector (uncalibrated) when calibrated model is not trusted
    S_pm*, S_pl* = surfel_vote(F_pm), planar_region_scores(F_pl)
else:
    S_pm*, S_pl* = argmin_S [ L_task(S) + lambda * L_cons(S) ]   # selection is a function of sigma_b
surface = raylets_regress(selected_views=S_pm*, planar_primitives=S_pl*)
```

**Calibration verification (diagnostic, no new mechanism).** On held-out calibration scenes, we compute the correlation between L_cons (evaluated at the selected S) and the actual selection error, defined as the change in selected primitive set when the prior is replaced by a ground-truth-derived prior (available only for diagnostic scenes with GT geometry). We report Spearman rank correlation between L_cons and selection error; a high correlation supports the load-bearing assumption, a low correlation falsifies it. We also plot sigma_b(d) against actual held-out prior error |e_b| to check monotonic calibration.

**Why this design.** We chose a *shared* perturbation-invariance loss over two branch-specific regularizers because the motivation is that the two branches fail for the same structural reason (uncalibrated selection), and a shared loss forces the planar branch and the feedforward branch to agree on what counts as a stable primitive under their respective noise; the cost is that a mis-specified sigma on one branch can dominate the shared term. We chose to fit sigma_b from held-out multi-view geometric consistency rather than from reconstruction error because reconstruction error is downstream of selection and would confound the calibration signal, accepting that sigma_b is scene-family-specific rather than universal. We chose to apply perturbations to the *prior features* (pointmap/depth) rather than to the selection scores directly because perturbing scores would test the selector's smoothness, not the error propagation the motivation names; the cost is that we must sample K structured perturbation fields per scene, but we cache prior features once so the added cost is only the lightweight selection-head reruns (K=4). We chose a structured, empirically estimated error model (correlation length l_b, tail index alpha_b) over an isotropic Gaussian because the load-bearing assumption requires the perturbation distribution to match the actual prior error structure; the cost is that the calibration phase requires overlapping view pairs for cross-view consistency, which are available in multi-view indoor datasets. A domain expert would not describe this as 'FoundSurface + PlanarSplatting' because the load-bearing object is the calibrated error model (l_b, alpha_b, sigma_b) and the shared invariance term, neither of which appears in either prior system: FoundSurface's surfel voting and PlanarSplatting's plane fitting are both deterministic and would be unchanged if sigma_b were removed.

**Why it measures what we claim.** The computational quantity L_cons measures *selection consistency under prior noise* because the assumption that perturbations drawn from the fitted correlated heavy-tailed model (l_b, alpha_b, sigma_b) are distributionally representative of the upstream prior's actual error distribution makes L_cons equivalent to the expected selection change under realistic prior error; this assumption fails when the fitted error model misestimates the spatial correlation length or tail index (e.g., calibration views lack the textureless-wall statistics of test scenes), in which case L_cons reflects sensitivity to the fitted perturbation field rather than to the errors that actually occur. The quantity sigma_b(d) measures *calibrated prior uncertainty* because the assumption that held-out cross-view geometric disagreement is a sufficient statistic for prior error makes sigma_b a valid per-branch noise scale; this assumption fails when the calibration views are not representative of test scenes, in which case sigma_b reflects calibration-set difficulty rather than test-time error. The inference-time argmin over S measures *calibrated primitive selection* because the assumption that the same L_cons is monotone in selection error at test time (established during training and checked by the L_cons-vs-selection-error correlation diagnostic) makes the selected S the most noise-stable one; this assumption fails when test-time priors are out-of-distribution, in which case the OOD gate falls back to the deterministic selector and the argmin reflects training-scene stability rather than test-scene correctness.

## Contribution

['A shared calibrated perturbation-invariance objective (CalibPC) that couples feedforward surfel-voting selection and planar-region detection through per-branch noise models fitted from held-out prior disagreement, making primitive selection a function of prior uncertainty rather than of raw prior output.', 'A design principle: primitive-selection consistency in surface reconstruction should be defined relative to a calibrated noise model of the upstream prior, not as a deterministic selector, because selection instability under prior noise is the shared failure mode across feedforward and planar-primitive branches.', 'A calibration procedure that fits per-branch noise scales from held-out prior disagreement without surface labels, enabling the shared objective to be trained on unposed multi-view data.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ScanNet++ zero-shot split | Unposed indoor scenes with textureless planar regions |
| Primary metric | F-score of reconstructed surface (τ=5cm) | Directly measures geometric accuracy of surface |
| Secondary metric | Spearman correlation between L_cons and actual selection error on held-out diagnostic scenes | Tests load-bearing assumption of calibrated error model |
| Baseline | FoundSurface (unmodified) | Tests whether calibration adds over deterministic surfel voting |
| Baseline | PlanarSplatting | Tests whether shared invariance beats deterministic plane fitting |
| Baseline | FoundSurface + PlanarSplatting fusion (no sigma) | Isolates the calibrated noise model from mere fusion |
| Baseline | Single-branch perturbation-invariance (FoundSurface-only with L_cons) | Tests whether cross-branch coupling is necessary |
| Baseline | Single-branch perturbation-invariance (PlanarSplatting-only with L_cons) | Tests whether cross-branch coupling is necessary |
| Ablation-of-ours | CalibPC with sigma fixed constant | Tests whether calibrated sigma_b is load-bearing |
| Ablation-of-ours | CalibPC with isotropic Gaussian perturbation (original design) | Tests whether structured error model is load-bearing |
| Ablation-of-ours | CalibPC without OOD gate | Tests whether test-time fallback is load-bearing |

### Why this setup validates the claim

The central claim is that a calibrated, shared perturbation-invariance objective improves primitive selection under realistic prior noise, and that the gain comes specifically from the calibrated sigma_b rather than from combining two branches. The unmodified FoundSurface baseline tests the sub-claim that deterministic surfel voting is not already noise-robust; PlanarSplatting tests whether deterministic plane fitting alone suffices on planar-heavy scenes; and the no-sigma fusion baseline is the critical control, because if fusion alone matches CalibPC, the calibrated noise model is not load-bearing and the motivation collapses. The single-branch perturbation-invariance baselines test whether the shared, cross-branch coupling is necessary or whether the same loss applied to one branch suffices; if single-branch regularizers match CalibPC, the novelty is not in the cross-branch agreement constraint. The fixed-sigma and isotropic-Gaussian ablations isolate calibration and error-model structure from the invariance term. The L_cons-vs-selection-error correlation diagnostic directly tests the load-bearing assumption that the fitted error model is representative of actual prior error. F-score at a fixed distance threshold is the right metric because the predicted effect is concentrated on textureless planar surfaces where selection errors displace the surface, so a thresholded geometric metric should show a localized gap rather than a diffuse global shift.

### Expected outcome and causal chain

**vs. FoundSurface (unmodified)** — On a case where a textureless wall yields spatially correlated pointmap errors, FoundSurface's surfel voting produces a reference-view set that is stable to Gaussian noise but biased by the correlated error, because its voting is deterministic and uncalibrated. Our method instead down-weights views whose prior disagreement exceeds the calibrated sigma_pm, because the shared invariance term penalizes selection changes under perturbations drawn from the fitted correlated heavy-tailed error model. We expect a noticeable F-score gap concentrated on textureless-wall subsets but near parity on textured regions.

**vs. PlanarSplatting** — On a case where a planar region is detected from noisy photometric depth, PlanarSplatting fits planes deterministically and inherits the depth noise as plane-offset error, because it has no noise model to reject unstable primitives. Our method instead selects planar primitives that remain stable under perturbations drawn from the fitted sigma_pl with correlation length l_pl and tail index alpha_pl, because the shared invariance loss couples plane selection to calibrated depth uncertainty. We expect a gap on planar subsets with high depth disagreement and parity on well-textured planar surfaces.

**vs. FoundSurface + PlanarSplatting fusion (no sigma)** — On a case where one branch's prior is much noisier than the other, the uncalibrated fusion lets the noisy branch dominate selection, because it has no per-branch noise scale to reweight the two branches. Our method instead normalizes each branch's consistency term by its calibrated sigma_b, because the shared loss is scale-aware. We expect the fusion baseline to be close to ours on balanced-noise scenes but clearly worse where branch noise is asymmetric, a signature that only calibration can produce.

**vs. Single-branch perturbation-invariance (FoundSurface-only with L_cons)** — On a case where planar primitives are unstable but pointmap selection is stable, the single-branch regularizer cannot correct planar selection because it only penalizes pointmap selection changes. Our method instead couples both branches through the shared loss, because the cross-branch agreement constraint propagates stability requirements across branches. We expect the single-branch baseline to match CalibPC on pointmap-dominated errors but to lag on planar-dominated errors, a signature that the shared coupling is load-bearing.

**vs. Single-branch perturbation-invariance (PlanarSplatting-only with L_cons)** — Symmetrically, on a case where pointmap selection is unstable but planar primitives are stable, the planar-only regularizer cannot correct pointmap selection. Our method instead couples both branches, because the shared loss requires agreement on stable primitives across branches. We expect the planar-only baseline to match CalibPC on planar-dominated errors but to lag on pointmap-dominated errors.

**vs. CalibPC with isotropic Gaussian perturbation** — On a case where prior error is spatially correlated (textureless walls), the isotropic Gaussian perturbation model underestimates the correlation length and thus selects primitives that are stable to i.i.d. noise but biased by correlated error. Our method instead samples perturbations from the fitted correlated heavy-tailed model, because the calibration phase estimates l_b and alpha_b from cross-view consistency. We expect the isotropic-Gaussian ablation to be closer to the no-sigma fusion baseline on textureless-wall subsets, confirming that the structured error model is load-bearing.

**vs. CalibPC without OOD gate** — On a test scene whose prior disagreement exceeds the calibration 95th percentile, the ungated method applies the calibrated argmin even though the error model is not trusted, potentially selecting primitives that are stable under the fitted model but wrong under the OOD error. Our method instead falls back to the deterministic selector when D_test > tau_OOD, because the gate detects distribution shift. We expect the ungated ablation to be worse on OOD scenes but equal on in-distribution scenes, confirming the gate is load-bearing only under shift.

### What would falsify this idea

If CalibPC's gain over the no-sigma fusion baseline is uniform across all subsets rather than concentrated on the high-prior-disagreement subsets the calibration is supposed to target, the calibrated noise model is not doing the causal work claimed. Equivalently, if the fixed-sigma ablation matches full CalibPC, or if the Spearman correlation between L_cons and actual selection error is near zero, then sigma_b is not load-bearing and the central claim is wrong.

## References

1. FoundSurface: Feedforward Ray-Based 3D Scene Surface Reconstruction from Unposed Images
