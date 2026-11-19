# Scale-Consistent Downscaling with Certified Resolution-Invariant Residual Heads

## Motivation

Existing predictive spatiotemporal downscaling methods such as PROLA and SGD claim resolution flexibility but provide no formal invariance property: PROLA's fixed numerical scaffolds and SGD's patch-based sampling can produce inconsistent predictions when the same coarse field is processed at different scaffold resolutions, and neither eliminates boundary artifacts at regional edges (a problem OneForecast's neural nested grid only partially addresses). Structurally, these methods treat resolution as a hyperparameter rather than a constraint, so the downscaling operator is not certified to commute with resampling—a property essential for reliable high-resolution forecasting.

## Key Insight

Enforcing scale-consistency during training via a resolution-invariance regularizer on residual heads, combined with a resolution-adaptive graph hierarchy whose pooling operators are boundary-artifact-free by construction, yields a downscaling operator that provably commutes with resampling across scaffold resolutions.

## Method

```python
# Certified Resolution-Invariant Downscaling (CRID)
# Inputs: coarse field x_coarse (any resolution r in R), pretrained backbone B, residual head H_theta, graph hierarchy G_r for each r
# Output: high-resolution field y_high (fixed target resolution R_target)

# Precondition (stated explicitly): CRID's invariance claim holds only on the band-limited subspace resolvable at the coarsest scaffold resolution r_min in R. For each scaffold ratio r -> r' we apply a fixed anti-aliasing low-pass filter F_{r->r'} (a separable Hamming-windowed sinc, cutoff = 0.5 * min(1, r'/r) cycles per coarse-grid cell, 15-tap) before any pooling or resampling. L_scale is computed after projecting each prediction onto this common band-limited subspace.

1. For each scaffold resolution r in R (R = {r_min, 2*r_min, 4*r_min}, three dyadic levels):
   a. Build resolution-adaptive graph hierarchy G_r via boundary-aware pooling: nodes = grid points; edges connect 4-neighbors; pooling operator P_{r->r'} = weighted average with weights from a fixed boundary-corrected kernel (no learned parameters). The kernel is a 3x3 stencil with weights [1/16, 1/8, 1/16; 1/8, 1/4, 1/8; 1/16, 1/8, 1/16] on interior nodes; on boundary nodes the stencil is renormalized to sum to 1 over available neighbors (divergence-consistent renormalization).
   b. Apply the anti-aliasing filter F_{r->r'} to x_coarse before encoding at resolution r' (for r' < r).
   c. Encode x_coarse at resolution r through pretrained backbone B to get latent z_r = B(F_{r->r_min}(x_coarse), G_r).
   d. Apply residual head H_theta to z_r to produce high-res prediction y_r = H_theta(z_r).
2. Scale-consistency regularizer L_scale = sum_{r1,r2 in R} || Proj(y_{r1}) - Proj(y_{r2}) ||^2 / |R|^2, where Proj projects a high-res prediction onto the band-limited subspace resolvable at r_min (via the same fixed low-pass filter F_{r_min->R_target} applied in the frequency domain), and y_{r1}, y_{r2} are predictions from the same coarse field processed at different scaffold resolutions.
3. Total loss L = L_task (MSE against high-res target) + lambda * L_scale, with lambda = 0.1 (tuned on validation).
4. At inference, average predictions across all r in R to obtain final y_high = (1/|R|) sum_r y_r.
```

**Why this design**
We chose a fixed boundary-corrected pooling kernel over a learned pooling operator because learned pooling can introduce resolution-dependent biases that violate the invariance property, accepting the cost that the hierarchy may be less expressive for complex terrain. We chose to enforce scale consistency via an explicit regularizer over a post-hoc calibration step because the regularizer directly shapes the residual head's function during training, ensuring the property holds for unseen coarse fields rather than only for the calibration set. We chose to average predictions across scaffold resolutions at inference over selecting a single resolution because averaging reduces variance from resolution-specific artifacts, accepting a modest increase in compute. A domain expert would not describe this as a variant of PROLA because PROLA's fixed numerical scaffolds are replaced by a resolution-adaptive graph hierarchy with boundary-corrected pooling, and the training objective includes a scale-consistency term absent from PROLA; the mechanism is not merely adding a regularizer to PROLA but changing the scaffold construction to make the regularizer well-posed.

**Why it measures what we claim**
The scale-consistency loss L_scale measures resolution invariance of the downscaling operator on the band-limited subspace resolvable at r_min because we assume that after applying the fixed anti-aliasing filter F_{r->r'} the boundary-corrected pooling kernel commutes with resampling on that subspace; this assumption fails when the coarse field contains high-frequency features above the r_min cutoff that alias differently at different scaffold resolutions, in which case L_scale reflects aliasing noise rather than true invariance violation. The boundary-corrected pooling kernel measures boundary-artifact-free consistency because we assume the kernel's weights are derived from a conservative interpolation scheme that satisfies the divergence theorem on the domain boundary; this assumption fails when the domain boundary is irregular (e.g., coastlines), in which case the kernel may still introduce small artifacts. The averaging at inference measures robustness to resolution choice because we assume predictions from different scaffold resolutions are unbiased estimators of the same high-res field; this assumption fails when one resolution is systematically biased (e.g., due to backbone pretraining distribution), in which case averaging propagates that bias.

## Contribution

['A resolution-invariant downscaling framework (CRID) that combines a scale-consistency regularizer on residual heads with a resolution-adaptive graph hierarchy using boundary-corrected pooling operators.', "An empirical demonstration that scale-consistency training reduces prediction discrepancies across scaffold resolutions by a measurable margin compared to PROLA's fixed-scaffold baseline.", 'A diagnostic protocol for quantifying boundary artifacts and resolution-invariance violations in downscaling models, applicable to future work.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ERA5 coarse-to-high-res pairs with GridSat conditions | Standard downscaling benchmark with multi-resolution structure |
| Primary metric | RMSE on high-frequency residual band (wavelengths 2–8 grid cells at R_target) | Isolates resolution-dependent artifacts, not bulk bias |
| Secondary metric | Commutation error: ||Proj(y_{r1}) - Proj(y_{r2})|| / ||Proj(y_{r1})|| on synthetic fields with known analytic spectra | Directly tests the band-limited invariance claim |
| Baseline | PROLA low-rank adaptation | Tests scaffold-construction gap vs. fixed scaffolds |
| Baseline | SGD diffusion downscaler | Tests against satellite-guided generative downscaling |
| Baseline | OneForecast nested-grid regional | Tests boundary-handling and multi-scale claims |
| Ablation-of-ours | CRID without L_scale | Isolates the scale-consistency regularizer's contribution |
| Ablation-of-ours | CRID without anti-aliasing filter F_{r->r'} | Tests whether the band-limiting precondition is load-bearing |

### Why this setup validates the claim

The central claim is that CRID's resolution-adaptive graph hierarchy plus scale-consistency regularizer yields a downscaling operator whose predictions are stable across scaffold resolutions on the band-limited subspace resolvable at r_min and accurate on high-frequency structure. The dataset choice matters because ERA5 with GridSat conditions supplies both coarse states at multiple resolutions and an independent observational signal, so resolution-invariance can be probed rather than assumed. PROLA is the key baseline because it shares the predictive-spatiotemporal framing but relies on fixed numerical scaffolds; if CRID's scaffold construction is doing real work, the gap should appear on high-frequency residual RMSE, not on bulk-field parity. SGD tests whether satellite-guided diffusion already absorbs the boundary and aliasing effects CRID targets, and OneForecast tests whether nested-grid regional forecasting already solves the boundary-loss problem. The ablation (CRID minus L_scale) is essential: it separates the regularizer's effect from the hierarchy's, so a gain cannot be misattributed. The second ablation (CRID minus anti-aliasing filter) tests the stated precondition: if removing the filter collapses the commutation-error advantage, the band-limiting step is doing the work the method claims. The high-frequency residual band is the right metric because it is precisely where aliasing and boundary artifacts concentrate; a bulk RMSE would mask the predicted effect. The commutation-error metric on synthetic fields with known analytic spectra provides a direct, assumption-light test of the invariance property itself, independent of forecast accuracy.

### Expected outcome and causal chain

**vs. PROLA low-rank adaptation** — On a case where the coarse field contains sharp coastal gradients and the scaffold resolution is coarser than the target, PROLA's fixed numerical scaffold pools across the gradient without boundary correction, producing smeared high-frequency structure and a systematic lag in where the gradient is placed. Our method instead builds a boundary-corrected hierarchy at each scaffold resolution and enforces agreement across them, so the residual head sees consistent coarse statistics. We expect a noticeable RMSE gap on high-frequency residual band concentrated on coastal and terrain-gradient subsets, with near-parity on smooth interior regions. On the synthetic commutation-error test, we expect CRID's error to be lower than PROLA's by a factor of at least 2 at the coarsest scaffold ratio, with the gap shrinking as the ratio approaches 1.

**vs. SGD diffusion downscaler** — On a case where the coarse field is sampled at a resolution different from the one SGD was pre-trained on, SGD's zero-shot guided sampling and patch-based generation drift toward the training-resolution statistics, producing plausible but resolution-mismatched fine structure. Our method instead processes the same field through multiple scaffold resolutions and averages, so resolution-specific sampling artifacts are damped rather than amplified. We expect SGD to look competitive on in-distribution resolutions but to degrade sharply on off-distribution scaffold resolutions, while CRID's gap stays roughly flat across the resolution sweep. On the commutation-error test, we expect SGD's error to grow with the resolution ratio while CRID's stays bounded by the band-limiting cutoff.

**vs. OneForecast nested-grid regional** — On a case where the target region intersects an irregular coastline, OneForecast's neural nested grid mitigates boundary loss but its densified target region still assumes a well-behaved domain boundary, so boundary-adjacent high-frequency features are under-resolved. Our method instead uses a divergence-consistent pooling kernel whose weights are fixed and boundary-corrected, so the hierarchy does not inject boundary artifacts into the latent. We expect a localized RMSE advantage in a narrow band along irregular boundaries, with parity elsewhere, which is the diagnostic signature of boundary handling rather than a global accuracy gain.

### What would falsify this idea

If CRID's gain over the ablation is uniform across smooth and high-frequency subsets rather than concentrated on the aliasing- and boundary-sensitive subsets, or if removing L_scale leaves high-frequency residual RMSE unchanged, or if removing the anti-aliasing filter does not increase commutation error on synthetic fields, then the claimed mechanism (scale consistency via boundary-corrected hierarchies on a band-limited subspace) is not what produces the result and the central claim is wrong.

## References

1. PROLA: Principal-Orthogonal Low-rank Adaptation for Predictive Spatiotemporal Weather Downscaling
