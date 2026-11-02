# Metric-Calibration Probing of Latent Spatial Representations

## Motivation

Probing work such as the spatial-state probing study of visual foundation models establishes that distances, angles, and object positions are linearly decodable from frozen features, but decodability is measured in arbitrary probe-regression units with no link to physical meters or degrees. Because a probe can achieve low regression error in its own rescaled space while its predicted distances are systematically compressed, inflated, or anisotropically distorted, the field currently cannot distinguish a representation that encodes a metrically faithful spatial map from one that merely encodes ordinal or rank-consistent spatial information. This is a meta-level gap: across spatial-VQA training, decodability profiling, and predictive-feature transfer, no shared constraint ties decoded quantities to physical units, so claims of 'spatial understanding' remain untested against the calibration property that any downstream metric reasoning requires.

## Key Insight

A representation is metrically calibrated only if its decoded distances and angles, after a single global affine rescaling fit on held-out data, produce residuals whose distribution is independent of true magnitude and direction; calibration error therefore isolates the geometric distortion structure of the latent space rather than the probe's fitting capacity.

## Method

**What it is.** We introduce **MC-Probe (Metric-Calibration Probe)**, a frozen-feature readout procedure that decodes pairwise distances and relative angles from a visual foundation model's latent states and reports expected calibration error in physical units (meters, degrees), together with a distortion decomposition that separates global scale error from magnitude-dependent and direction-dependent error.

**Load-bearing assumption (stated explicitly).** MC-Probe assumes that a metrically faithful latent spatial representation requires only a *single global affine rescaling* — one scale `a`, one offset `b`, one angular offset `c` — to align decoded distances and angles with physical units, so that any residual conditional bias after that one global fit is attributable to geometric distortion in the representation rather than to legitimate scene-, object-, or depth-dependent scale variation. This assumption is the classic 'up-to-scale' problem of egocentric geometry: scale is unobservable from a single image and is only recoverable per-scene or per-trajectory, so a representation can be perfectly metric *within* each scene yet have no shared global scale. To keep this assumption testable rather than baked in, the calibration step is *hierarchical*: we fit a global affine plus an explicit per-scene scale term, and report ECE_m on the residuals after removing the per-scene scale (i.e., we test for a shared metric frame *conditional on* allowing scene-varying scale). The per-scene scale dispersion is reported as a separate, explicitly-labeled quantity `sigma_scene` (std. dev. of per-scene log-scale) rather than folded into 'miscalibration'. The distortion coefficients `(alpha, beta, gamma)` are therefore identifiable separately from the scene-scale nuisance parameter.

**How it works.**

```pseudocode
Input: frozen encoder f, egocentric image stream, ground-truth poses P (metric)
Output: ECE_m (m), ECE_deg (deg), sigma_scene, distortion profile

1. Extract features: z_t = f(o_t) for each observation o_t
2. For each pair (i,j) with known metric displacement d_ij, theta_ij:
     build pair descriptor phi_ij = [z_i ; z_j ; z_i - z_j ; z_i * z_j]
3. Split scenes into fit / calibration / eval sets at the SCENE level (no frame leakage);
   scene counts: fit=60%, calibration=20%, eval=20% (min 30 eval scenes; see power note)
4. Fit readout heads on fit set:
     d_hat = MLP_d(phi_ij)      # 2-layer, hidden 256, ReLU
     theta_hat = MLP_a(phi_ij)  # predicts (cos, sin), atan2-decoded
5. Hierarchical recalibration on the held-out calibration split:
     d_cal = a * d_hat + b + s_scene     # a>0 constrained, s_scene = per-scene scale term
     theta_cal = theta_hat + c           # circular mean offset
   Fit by least squares with a>0; s_scene is a per-scene intercept on the log-scale
   (i.e., d_cal = exp(log a + s_scene) * d_hat + b), regularized toward 0 with
   L2 penalty lambda=1e-2 so that s_scene is identified only when the data demand it.
   Report sigma_scene = std(s_scene) across calibration scenes.
6. Bin eval predictions by predicted magnitude (deciles) and by
   predicted angle (12 sectors of 30 deg); minimum per-bin count = 50
   (bins below 50 merged with neighbor; pre-registered).
7. Compute per-bin calibration error on residuals after removing s_scene:
     ECE_m = sum_b (n_b/N) * |mean(d_cal - d_true) in bin b|
     ECE_deg = sum_b (n_b/N) * |circ_mean(theta_cal - theta_true) in bin b|
8. Distortion decomposition (on residuals after s_scene removal):
     fit d_cal - d_true = alpha + beta*d_true + gamma*d_true^2
     report (alpha, beta, gamma) as offset / scale / nonlinear distortion
     fit angular residual vs. true heading; report anisotropy term
9. Report also Pearson r and MAE for comparison with decodability baselines.
```

**Why this design.** We chose a *hierarchical recalibration* (step 5: global affine plus explicit per-scene scale term, with ECE_m computed on residuals after removing the per-scene scale) over a single global affine because the motivation is whether the latent space carries a shared metric frame *conditional on* allowing the scene-varying scale that egocentric geometry makes unavoidable; a single global fit would absorb scene-dependent scale variation into the failure signal, conflating a geometric necessity with a representational defect. We chose *binned expected calibration error* over a single MAE because MAE conflates probe capacity with distortion: a high-capacity probe can drive MAE down while leaving systematic magnitude compression intact, whereas binning by predicted magnitude exposes the conditional bias that MAE averages away, accepting the cost of noisier estimates in sparse bins (mitigated by decile binning with a minimum count of 50). We chose *scene-level splits* over frame-level splits because egocentric frames from one trajectory are highly correlated, and frame-level leakage would let the probe memorize scene-specific offsets and inflate apparent calibration, accepting the cost of fewer effective training scenes. We chose *cos/sin angle regression with atan2 decoding* over direct angle regression because direct regression has a discontinuity at the wrap-around and would produce spurious large errors near 0/360 degrees, accepting the cost of a two-output head. This differs from the decodability profiling of the spatial-state probing study, which reports probe accuracy in normalized units and never fits a global physical-unit rescaling or bins residuals by magnitude; an expert would not describe MC-Probe as a variant of that probing pipeline because the load-bearing quantity here is the *conditional residual distribution after physical-unit recalibration*, not the probe's fit quality.

**Why it measures what we claim.** `ECE_m` measures *metric calibration in meters* because the hierarchical map in step 5 (global affine plus per-scene scale) is assumed to be the only degrees of freedom a metrically faithful latent space should require to align with physical units, with the per-scene scale explicitly removed so that ECE_m reflects distortion conditional on scene-varying scale; this assumption fails when the representation encodes distance only up to a per-object or per-depth scale not captured by the per-scene term, in which case `ECE_m` reflects that residual scale inconsistency rather than a global metric frame. The distortion coefficients `(alpha, beta, gamma)` measure *geometric faithfulness of the latent map* because a linear latent encoding of Euclidean position yields `beta` near zero residual slope after hierarchical correction; this assumption fails under a nonlinear (e.g., log-compressed) encoding, in which case `gamma` reflects the compression exponent rather than a failure of metric structure. `ECE_deg` measures *angular calibration* because a rotation-equivariant latent frame produces angular residuals independent of true heading; this assumption fails when the encoder's reference frame drifts with camera motion, in which case `ECE_deg` reflects frame-drift rather than angular miscalibration. The scene-level split ensures the calibration estimate measures *generalization of the metric frame* because held-out scenes share no trajectory-specific offset; this assumption fails if scenes share a dataset-wide bias, in which case the estimate reflects that shared bias rather than per-representation calibration. `sigma_scene` measures *per-scene scale dispersion* because it is the std. dev. of the fitted per-scene log-scale terms; this assumption fails when scenes have too few pairs to identify s_scene, in which case `sigma_scene` reflects estimation noise rather than true scale variation.

## Contribution

(1) MC-Probe, a frozen-feature calibration protocol that reports expected calibration error in physical meters and degrees plus a magnitude/direction distortion decomposition, converting decodability claims into testable metric-calibration claims. (2) A design principle separating global affine scale error from magnitude-dependent nonlinear distortion and from direction-dependent anisotropy, which distinguishes representations that are metrically faithful from those that are merely ordinally consistent. (3) A scene-level calibration benchmark and analysis showing how calibration error varies across encoder layers and representation families, providing a reusable diagnostic for spatial representation evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Egocentric streams with metric pose ground truth | Supplies physical-unit distances and headings for calibration |
| Primary metric | Binned ECE in meters and degrees (after per-scene scale removal) | Exposes conditional residual bias MAE averages away |
| Secondary metric | `sigma_scene` (std. dev. of per-scene log-scale) | Separates scene-scale nuisance from distortion |
| Baseline | Probe MAE / Pearson r (normalized units) | Tests whether decodability alone suffices |
| Baseline | Per-scene affine recalibration (no global term) | Tests whether scene-varying scale is the failure |
| Baseline | Direct angle regression head | Tests whether wrap-around handling matters |
| Ablation-of-ours | No hierarchical recalibration (raw d_hat) | Isolates contribution of physical-unit alignment |
| Ablation-of-ours | Single global affine (no per-scene term) | Isolates contribution of the per-scene scale term |
| Synthetic control | Encoders with known injected distortion (log-compressed, per-scene scaled, anisotropic) | Validates MC-Probe recovers known distortion parameters |

### Why this setup validates the claim

The central claim is that a metrically faithful latent space should need only a hierarchical rescaling (global affine plus per-scene scale) to align with physical units, so residual conditional bias after that rescaling is the diagnostic quantity. The normalized-unit MAE/Pearson baseline tests the sub-claim that decodability is insufficient: if a high-capacity probe matches our MAE while our binned ECE still reveals magnitude compression, the calibration framing is doing real work. The per-scene affine baseline tests the sub-claim that scene-varying scale is precisely the failure being detected: it should absorb scene offsets and look well-calibrated by construction, so any gap between it and our hierarchical fit localizes the inconsistency. The no-recalibration ablation isolates whether the affine step is load-bearing rather than incidental, and the single-global-affine ablation isolates whether the per-scene scale term is load-bearing. The synthetic control with known injected distortion (log-compressed, per-scene scaled, anisotropic) validates that MC-Probe recovers the injected distortion parameters, establishing it as a diagnostic instrument rather than a metric variant. Binned ECE is the right metric because the predicted effect is a conditional, magnitude-dependent bias that aggregate error hides; if the representation is globally metric, residual bias should be flat across deciles and sectors. Power note: with 30 eval scenes and decile binning (min 50 pairs/bin), a two-sided test at alpha=0.05 detects an ECE_m gap of 0.15 m with power 0.8 when per-scene pair count ≥ 200; we pre-register this minimum.

### Expected outcome and causal chain

**vs. Probe MAE / Pearson r (normalized units)** — On a case where the encoder compresses far distances (log-like scaling), the normalized-unit baseline reports strong Pearson r and low MAE because monotone compression preserves rank order, so its fit-quality view calls the representation metric. Our method instead bins residuals by predicted magnitude after hierarchical recalibration because compression produces magnitude-dependent bias, so we expect a clear ECE_m elevation concentrated in the far-distance deciles while near-distance deciles stay near zero — a gap that is invisible to the baseline's aggregate score.

**vs. Per-scene affine recalibration (no global term)** — On a case where each trajectory carries its own scale offset, the per-scene baseline produces near-zero residuals because it fits one affine per scene, absorbing exactly the scene-dependent variation whose presence is the failure signal. Our method instead fits a hierarchical model (global affine plus per-scene scale) on a held-out calibration split and reports ECE_m on residuals after removing the per-scene scale, so we expect our ECE_m to be markedly higher than the per-scene baseline on multi-scene evaluation only when the distortion is not explained by scene scale, with the gap shrinking when scenes share consistent scale — precisely the diagnostic pattern. The `sigma_scene` quantity is reported separately to show how much of the apparent miscalibration is scene-scale nuisance.

**vs. Direct angle regression head** — On a case where true headings sit near the 0/360 wrap-around, the direct-regression baseline produces spurious large angular errors because its scalar output has a discontinuity at the boundary, inflating ECE_deg near those sectors. Our method instead regresses cos/sin and decodes with atan2 because the circular parameterization is continuous across the wrap, so we expect our ECE_deg to be flat across all 12 sectors while the baseline shows a pronounced spike in the sectors adjacent to zero degrees.

**vs. Synthetic control (known injected distortion)** — On synthetic encoders where we inject log-compression (gamma known), per-scene scale (sigma_scene known), or anisotropy (known angular distortion), MC-Probe should recover the injected parameters within the pre-registered tolerance (gamma within ±0.1, sigma_scene within ±0.05, anisotropy within ±2 deg) and produce flat residuals on a control encoder with no injected distortion, confirming the load-bearing assumption that a genuinely metric latent yields flat residuals across deciles and sectors.

### What would falsify this idea

If our binned ECE_m and ECE_deg were uniformly low and flat across all deciles and sectors — matching the per-scene baseline and the no-recalibration ablation — then the claimed magnitude-dependent and scene-dependent distortions do not exist and the calibration framing adds nothing over decodability. Likewise, if our gain over the normalized-unit baseline were uniform across subsets rather than concentrated in the far-distance deciles and wrap-around sectors where the predicted failure modes live, or if the synthetic control failed to recover injected distortion parameters, the causal story would be wrong.

## References

1. Probing Spatial State Representations in Visual Foundation Models
