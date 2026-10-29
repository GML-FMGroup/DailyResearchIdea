# Scale-Calibrated Normal Conditioning for Subject-Driven Video Diffusion

## Motivation

Geometry-conditioned subject-driven video generators such as OmniShowcase inject feed-forward surface-normal maps as if they were oracle control, but those normals are derived from depth that is only defined up to an affine scale (as in pi^3 / Depth Anything 3). The resulting normal field carries a structured, spatially varying error that the diffusion backbone cannot distinguish from correct geometry, so the model trusts unreliable normals and drifts in viewpoint or breaks subject consistency. This is a field-wide structural gap: every geometry branch treats a feed-forward 3D proxy as oracle control with no calibrated error bound propagated from depth-scale ambiguity into the conditioning interface.

## Key Insight

The affine-invariant scale ambiguity of monocular depth induces a closed-form per-pixel covariance on the surface normal, so the identifiability limit of the normal is computable in closed form and can be used as a deterministic, error-bounded gate rather than a learned router.

## Method

**What it is.** We introduce SCAN-C (Scale-Calibrated Normal Conditioning), a deterministic conditioning interface that sits between a feed-forward geometry estimator and a video diffusion transformer. Its inputs are the reference views, the estimated depth map with its affine scale parameters, and the estimated surface normals; its output is a per-pixel, per-normal-component confidence field that multiplicatively gates the normal-conditioning tokens inside the diffusion transformer.

**Load-bearing assumption (stated explicitly).** SCAN-C assumes that the affine scale ambiguity (a,b) of monocular depth is the dominant source of error in the feed-forward surface normals, so that a covariance propagated only from Cov(a,b) through a locally linear normal operator yields a calibrated per-pixel confidence that correctly ranks normal reliability. This assumption is load-bearing: if feed-forward normal errors are dominated by monocular priors, cross-view inconsistency, or boundary/occlusion artifacts rather than by scale ambiguity, the gate will under-attenuate biased normals. We therefore (i) add a multi-view reprojection-consistency residual term to the covariance so that conf reflects total normal uncertainty rather than only the affine component, and (ii) validate that conf is calibrated (rank correlation with ground-truth normal error) before claiming it measures identifiability.

**How it works.**

```pseudocode
# Inputs: reference views I_1..I_K, backbone video diffusion model D
# Geometry branch: VGGT-style feed-forward estimator G
(depth d, normal n, affine scale (a,b)) = G(I_1..I_K)
# d is affine-invariant: true depth = a * d + b, with (a,b) estimated per view

# Step 0: obtain Cov(a,b) by bootstrapping the affine fit
#   For B=32 bootstrap resamples of the K reference views (sampled with replacement),
#   refit (a,b) by least-squares on the affine-invariant depth; report the empirical
#   covariance of the B estimates as Cov(a,b).
#   Fallback when the estimator reports no covariance: fixed prior
#   diag(sigma_a^2, sigma_b^2) with sigma_a = 0.05, sigma_b = 0.05 (normalized depth units).

# Step 1: propagate affine-scale ambiguity into a normal covariance
for each pixel p:
    # Jacobian of the normal w.r.t. depth, evaluated at estimated depth
    J_p = d(n_p) / d(depth neighborhood)          # 2x2 finite-difference Jacobian, 3x3 neighborhood
    Sigma_scale_p = J_p @ Cov(a,b) @ J_p^T        # closed-form covariance of n_p
    # Multi-view reprojection-consistency residual term (bias correction):
    #   For each reference view k, warp n_p into view k via the estimated pose and depth,
    #   compute the angular residual r_{p,k} = angle(n_p, n_{p->k}); set
    #   Sigma_res_p = (1/K) * sum_k r_{p,k}^2 * I_3x3.
    Sigma_p = Sigma_scale_p + Sigma_res_p

# Step 2: per-component confidence gate (deterministic, no learned router)
for each normal component c in {x,y,z}:
    conf[p,c] = 1 / (1 + lambda * Sigma_p[c,c])   # lambda = 1.0

# Step 3: gate the normal-conditioning signal inside the DiT
normal_tokens = PatchEmbed(n) * conf                 # element-wise, per component
for each DiT block l:
    h = h + CrossAttn(h, normal_tokens) * conf_gate   # conf_gate = spatial mean of conf

# Step 4: aleatoric residual loss — the backbone learns to treat
#         unidentifiable normals as noise, not as control error
L = L_diffusion + beta * || (n_pred - n) * (1 - conf) ||^2   # beta = 0.1
```

**Why this design.** We chose a closed-form covariance propagation over a learned confidence head because the affine-scale ambiguity has an exact analytic form (the normal is a function of depth gradients, and the scale enters linearly), so a learned head would only re-approximate a quantity we can compute exactly, accepting the cost that we must assume a local linearization of the normal operator. We chose to gate the conditioning tokens multiplicatively rather than to mask them out entirely because soft gating preserves gradient flow to the backbone in low-confidence regions, accepting the cost that residual error still enters the model but is attenuated in proportion to its bound. We chose to add an aleatoric residual loss weighted by (1 - conf) rather than a standard L2 normal loss because the diffusion backbone must learn that low-confidence normals are noisy observations, not targets to fit, accepting the cost that the loss magnitude shrinks in exactly the regions where geometry is hardest. We added the multi-view reprojection-consistency residual term Sigma_res_p to the covariance because the scale-only term is near-degenerate (a global affine transform leaves the normal direction unchanged up to the multiplicative factor a), so the residual term is what makes conf reflect total normal uncertainty rather than only the affine component. A domain expert would not describe this as OmniShowcase with an extra confidence head, because the gate is not learned and not a router: it is a deterministic reparameterization whose value is fixed by the depth estimator's own scale covariance plus a multi-view consistency residual, and the backbone is trained against the residual rather than against the normal.

**Why it measures what we claim.** The per-component confidence `conf[p,c] = 1/(1 + lambda * Sigma_p[c,c])` measures the identifiability of the normal component because the closed-form covariance `Sigma_p` equals the variance of the normal induced by the affine-scale ambiguity plus the multi-view reprojection-consistency residual under the assumption that the normal operator is locally linear in depth over the finite-difference neighborhood and that the residual term captures the non-scale error; this assumption fails when the depth surface has high curvature relative to the finite-difference neighborhood or when the reprojection residual is dominated by pose error, in which case `conf` reflects local linearization or pose error rather than scale ambiguity. The gated token `PatchEmbed(n) * conf` measures the reliable portion of the normal-conditioning signal because multiplying by `conf` attenuates exactly the components whose total variance exceeds the identity threshold under the assumption that scale ambiguity plus reprojection residual are the dominant error sources; this assumption fails when the depth estimator has bias uncorrelated with both scale and reprojection consistency, in which case the gate under-attenuates biased normals. The aleatoric residual loss `||(n_pred - n)*(1 - conf)||^2` measures the backbone's treatment of unidentifiable normals as noise because the (1 - conf) weight zeroes the gradient where the normal is unidentifiable under the assumption that the diffusion model's prediction error decomposes into control error and aleatoric noise; this assumption fails when the backbone systematically overfits low-confidence normals, in which case the loss reflects residual control error instead.

## Contribution

['A deterministic, closed-form scale-ambiguity-aware normal-conditioning interface (SCAN-C) that propagates the affine-invariant depth scale into a per-pixel, per-normal-component confidence field and gates normal tokens inside the diffusion transformer without any learned router.', 'An aleatoric residual training objective that teaches the video diffusion backbone to treat unidentifiable normal components as observation noise rather than as control targets, derived from the same closed-form covariance.', 'A design principle for geometry-conditioned subject-driven video generation: geometry proxies should be injected only up to their identifiability limit, with the residual modeled as aleatoric uncertainty.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-view subject scenes with 360-degree orbits | Exercises scale-ambiguous geometry across wide baselines |
| Primary metric | Subject consistency under novel-view orbit | Directly reflects whether gated normals improve fidelity |
| Baseline | OmniShowcase | Strong multi-view S2V without scale-aware gating |
| Baseline | Single-image S2V conditioning | Tests value of multi-view geometry signal |
| Baseline | VGGT normals, ungated conditioning | Isolates effect of confidence gating alone |
| Ablation-of-ours | SCAN-C with fixed uniform confidence | Removes scale-covariance signal, keeps architecture |
| Ablation-of-ours | SCAN-C with learned per-pixel variance head | Isolates analytic scale-identifiability from generic uncertainty weighting |
| Ablation-of-ours | SCAN-C with scale-only covariance (no Sigma_res) | Tests whether the multi-view residual term is necessary |
| Calibration check | Synthetic ground-truth depth and affine scale | Validates that conf predicts empirical normal error |

### Why this setup validates the claim

The central claim is that propagating the affine-scale ambiguity of feed-forward depth into a per-component normal confidence, and using that confidence to gate normal-conditioning tokens and weight the residual loss, improves subject consistency precisely where scale ambiguity corrupts normals. The multi-view orbit dataset is the right testbed because wide-baseline 360-degree views are exactly where affine scale estimation is least constrained and where normal errors would propagate into inconsistent subject appearance. OmniShowcase tests whether our gating beats a strong multi-view S2V method that conditions on geometry without scale-aware confidence; the single-image baseline tests whether the geometry signal matters at all; the ungated VGGT baseline isolates the gating contribution from the geometry contribution. The uniform-confidence ablation is the decisive control: it preserves our architecture and loss but discards the scale-covariance signal, so any gain over it must come from the deterministic confidence computation rather than from added parameters or training. The learned-variance-head ablation isolates the analytic scale-identifiability claim from generic uncertainty weighting: if the deterministic version wins specifically on high-scale-ambiguity scenes, the gain is attributable to the closed-form covariance rather than to learned uncertainty. The scale-only ablation tests whether the multi-view reprojection-consistency residual term is necessary for calibration. The synthetic ground-truth check validates that conf is calibrated (rank correlation with empirical normal error) before we claim it measures identifiability. Subject consistency under orbit is the right primary metric because the predicted effect is a reduction in view-dependent subject drift caused by corrupted normals, which manifests as consistency failures rather than global quality degradation.

### Expected outcome and causal chain

**vs. OmniShowcase** — On a case where the subject is viewed from a near-profile angle far from the reference views, the baseline produces drifting texture and shape on the subject's occluded side because its normal conditioning carries scale-ambiguous normals with no mechanism to attenuate them. Our method instead down-weights exactly those normal components whose scale-induced variance is large, because the closed-form covariance propagation flags them, so we expect a noticeable consistency gap concentrated on wide-baseline orbit segments and near-parity on near-frontal segments.

**vs. Single-image S2V conditioning** — On a case where a single reference leaves the subject's back unobserved, the baseline produces plausible but identity-inconsistent geometry because it has no multi-view geometric anchor. Our method instead injects gated normals from multiple references, preserving the reliable geometric signal while suppressing ambiguous components, so we expect a broad consistency advantage that is largest on unobserved-side views and shrinks where the single reference already covers the subject.

**vs. VGGT normals, ungated conditioning** — On a case where affine scale is poorly estimated, the baseline produces normal-conditioning tokens that are confidently wrong, because it treats all estimated normals as equally reliable. Our method instead gates those tokens by the propagated scale covariance plus the multi-view reprojection-consistency residual and trains against the confidence-weighted residual, so we expect a gap that scales with the magnitude of scale ambiguity, appearing on high-ambiguity scenes and vanishing on well-constrained ones.

**vs. SCAN-C with fixed uniform confidence** — On a high-scale-ambiguity scene, the uniform-confidence ablation applies the same gate everywhere, so it cannot attenuate the corrupted normals that our method flags. Our method instead uses the deterministic confidence field to down-weight exactly those components, so we expect the gain over this ablation to be concentrated on high-scale-ambiguity scenes rather than uniform across scenes; a uniform gain would falsify the claim that conf measures scale identifiability.

**vs. SCAN-C with learned per-pixel variance head** — On a high-scale-ambiguity scene, the learned head must infer uncertainty from data and may not recover the analytic scale-identifiability signal, especially in low-data regimes. Our method instead computes the covariance in closed form from the estimator's own scale covariance, so we expect the deterministic version to win specifically on high-scale-ambiguity scenes while the learned head may match or slightly exceed it on well-constrained scenes where generic uncertainty weighting suffices.

**vs. SCAN-C with scale-only covariance (no Sigma_res)** — On scenes with boundary/occlusion artifacts or monocular-prior bias, the scale-only covariance is near-degenerate and under-attenuates biased normals. Our method instead adds the multi-view reprojection-consistency residual, so we expect the full covariance to improve calibration (higher rank correlation with ground-truth normal error) and subject consistency on scenes with large non-scale normal error.

**Calibration check (synthetic ground-truth)** — On synthetic scenes where ground-truth depth and affine scale are known, we perturb (a,b) by the reported covariance, recompute normals exactly, and measure empirical per-pixel normal variance against the closed-form Sigma_p. We expect high rank correlation (Spearman rho > 0.7) between conf and empirical normal error, with the correlation degrading in high-curvature regions where the local linearization assumption fails; this validates that conf measures identifiability before we claim it does.

### What would falsify this idea

If the gain over the uniform-confidence ablation is uniform across scenes rather than concentrated on high-scale-ambiguity cases, the confidence field is not measuring scale identifiability and the central claim is wrong. Likewise, if the ungated VGGT baseline matches our method on high-ambiguity scenes, the gating mechanism provides no benefit over simply conditioning on normals; and if conf has low rank correlation with empirical normal error in the synthetic check, the calibration claim is false.

## References

1. OmniShowcase: Omnidirectional Subject-Consistent Video Generation from Multi-View Images
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. VGGT: Visual Geometry Grounded Transformer
4. MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement
