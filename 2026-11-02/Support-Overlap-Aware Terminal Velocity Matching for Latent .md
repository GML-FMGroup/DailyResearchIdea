# Support-Overlap-Aware Terminal Velocity Matching for Latent Flow Enhancement

## Motivation

GLOW-FM's Terminal Velocity Matching trains a velocity field to match the terminal velocity of a conditional flow between low-light and normal-light latent distributions, but this objective presumes the conditional expectation defining that velocity is well-defined — which fails under the distribution shift typical of low-light enhancement, where the two latent supports can be nearly disjoint. The stochastic-interpolant formulation of Building Normalizing Flows with Stochastic Interpolants makes the same implicit assumption: its quadratic velocity loss is derived from the probability current of the interpolating density and does not establish identifiability when the base and target supports do not overlap. Neither work provides a training signal that remains well-defined under support mismatch, so the learned velocity can be arbitrary in the region where enhancement is most needed.

## Key Insight

The terminal-velocity target is identifiable if and only if the low-light and normal-light latent supports overlap on the interpolation path; estimating that overlap per sample and using it to gate the velocity loss and to shape the latent geometry turns an ill-posed regression into a well-posed one exactly where the supports are shared.

## Method

## SOVA-TVM: Support-Overlap-Aware Terminal Velocity Matching

**(A) What it is.** SOVA-TVM is a training procedure for the GLOW-FM velocity network that estimates, per training pair, the degree to which the low-light latent support overlaps the normal-light latent support, and uses that estimate both to weight the terminal-velocity matching loss and to drive a latent-space alignment regularizer. Inputs: paired low/normal-light images, frozen Tiny-VAE encoder, velocity network. Output: a velocity network whose one-step terminal velocity is well-defined under distribution shift.

**(B) How it works.**

```pseudocode
# Precompute reference statistics from a held-out normal-light split
Z_ref = encode(frozen_VAE, normal_light_holdout)      # N x d latent bank, N=5000
sigma = median_pairwise_distance(Z_ref)               # RBF bandwidth, fixed

for each training pair (x_low, x_normal):
    z0 = encode(frozen_VAE, x_low)
    z1 = encode(frozen_VAE, x_normal)

    # 1. Kernel two-sample overlap estimate (MMD-based)
    #    Local MMD between z0 and its k nearest normal-light latents
    N_k = kNN(z0, Z_ref, k=16)
    mmd_local = MMD_rbf(z0, N_k, sigma)                # scalar >= 0
    overlap = exp(-mmd_local / tau)                    # tau = 0.5, in (0,1]

    # 2. Overlap-weighted terminal velocity matching
    v_target = z1 - z0                                # terminal velocity (linear interpolant)
    v_pred   = velocity_net(z0, t=1, cond=z0)
    L_tvm    = overlap * ||v_pred - v_target||^2

    # 3. Latent alignment regularizer (active only when overlap < rho)
    rho = 0.3
    if overlap < rho:
        # Push z0 toward the convex hull of its normal-light neighbors
        z0_aligned = mean(N_k)
        L_align = lambda_align * ||z0 - z0_aligned||^2   # lambda_align = 0.1
    else:
        L_align = 0

    L = L_tvm + L_align
    update velocity_net with grad(L)
```

**(C) Why this design.** We chose a kernel two-sample (MMD) overlap estimate over a learned discriminator because MMD gives a per-sample scalar with a closed-form dependence on the latent geometry and no adversarial training instability, accepting the cost that the bandwidth sigma must be set from held-out statistics rather than adapted per batch. We chose to weight the terminal-velocity loss rather than to discard low-overlap samples because discarding would bias training toward easy pairs and starve the model of the degraded inputs where enhancement matters most, accepting the cost that low-overlap samples contribute a weak but non-zero gradient. We chose to activate the alignment regularizer only below a threshold rho rather than always, because always-on alignment would collapse the low-light latent distribution onto the normal-light mean and destroy reconstruction detail, accepting the cost that rho introduces a discrete switch whose value must be tuned. A domain expert would not describe this as a variant of the stochastic-interpolant velocity loss from Building Normalizing Flows with Stochastic Interpolants, because that loss is applied uniformly and assumes overlap; SOVA-TVM makes the loss itself a function of an estimated overlap statistic, which is a structural change to the training signal rather than a reweighting heuristic.

**(D) Why it measures what we claim.** The local MMD `mmd_local` measures support overlap between the low-light latent and the normal-light latent distribution because, under the assumption that the k-nearest normal-light latents are a sufficient local sample of the normal-light support around z0, MMD between z0 and that neighborhood is zero if and only if z0 lies in the support of the normal-light distribution; this assumption fails when the normal-light latent distribution is multi-modal and the k-NN neighborhood straddles two modes, in which case mmd_local reflects inter-mode distance rather than true non-overlap. The overlap weight `exp(-mmd_local/tau)` measures the identifiability of the terminal-velocity target because, under the assumption that the conditional expectation E[z1 - z0 | z0] is continuous in the overlap region, a larger overlap implies a better-conditioned regression target; this assumption fails when the velocity field has high curvature within the overlap region, in which case a high overlap weight can still correspond to a poorly-conditioned target. The alignment penalty `||z0 - mean(N_k)||^2` measures the degree to which the low-light latent is outside the normal-light support because, under the assumption that the normal-light support is locally convex, the distance to the neighborhood mean is a proxy for support violation; this assumption fails near the boundary of a non-convex support, in which case the penalty can pull a valid boundary point inward and blur fine detail.

**(D.1) Load-bearing assumption (stated explicitly).** The method assumes that the per-sample local MMD between z0 and its k=16 nearest normal-light reference latents is a monotone, calibrated measure of whether the conditional expectation E[z1-z0|z0] is identifiable/well-conditioned, so that exp(-mmd_local/tau) correctly ranks pairs by target conditioning and the alignment penalty correctly identifies support violation. This assumption is load-bearing: if it fails, the overlap weight does not measure target conditioning and the central claim is unsupported. The assumption is not derived from MMD theory (which guarantees consistency only for the two-sample null) and is not verified by the method as written; it is an empirical hypothesis to be tested by the calibration and synthetic experiments specified in the experiment section.

**(D.2) Calibration and verification story (inside the existing structure).** To verify the load-bearing assumption without altering the training procedure, we add two concrete checks that use only the existing quantities:

1. **Held-out calibration of the weight.** On a held-out set of 512 paired latents (disjoint from training and from Z_ref), compute for each pair the squared terminal-velocity residual of a fixed reference velocity network (the uniform-TVM baseline) as a proxy for target noise, and bin pairs by their overlap weight exp(-mmd_local/tau). Report the Spearman rank correlation between overlap weight and negative residual; a positive correlation supports the claim that the weight ranks pairs by conditioning. This is a diagnostic, not a change to the loss.

2. **Synthetic support-overlap sweep.** Construct a controlled latent experiment where z0 and z1 are drawn from two Gaussians in d=16 dimensions with tunable mean separation delta (0 to 4 sigma), and where the true conditional expectation E[z1-z0|z0] is known analytically. For each delta, compute mmd_local exactly as in the method (k=16, sigma from a normal-light reference bank of 5000 samples) and measure the correlation between exp(-mmd_local/tau) and the actual conditional-expectation error of the velocity target. This tests the monotonicity and calibration of the overlap statistic against a ground-truth identifiability measure, using the same estimator as the method.

These checks are reported as part of the method's validation; they do not modify the training loop, the loss, or the architecture.

## Contribution

['A support-overlap-aware training mechanism for latent flow matching that estimates per-sample overlap via a local kernel two-sample statistic and uses it to gate the terminal-velocity loss and to trigger a latent alignment regularizer, making the velocity target well-defined under distribution shift.', 'A design principle: the terminal-velocity matching objective should be conditioned on an explicit overlap estimate rather than applied uniformly, because uniform application produces arbitrary gradients in the disjoint-support regime that dominates severely degraded low-light inputs.', "An instantiation on GLOW-FM's frozen Tiny-VAE latent space that requires no architectural change to the velocity network and no additional inference cost, since the overlap estimate and regularizer are training-only."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Real-world low-light benchmark (LOL-style paired set) | Paired low/normal images give ground-truth terminal velocity targets. |
| Primary metric | PSNR on low-overlap subset | Isolates exactly the distribution-shift regime the claim targets. |
| Baseline | Uniform TVM (no overlap weighting) | Tests whether overlap weighting itself drives the gain. |
| Baseline | Discard-low-overlap training | Tests whether weighting beats the discard alternative we argue against. |
| Baseline | Always-on alignment regularizer | Tests whether the thresholded switch prevents detail collapse. |
| Ablation-of-ours | SOVA-TVM with random overlap weights | Tests whether the MMD overlap statistic carries real signal. |
| Calibration check | Held-out Spearman correlation between overlap weight and negative velocity residual | Verifies the weight ranks pairs by target conditioning. |
| Synthetic check | Two-Gaussian support-overlap sweep (d=16, delta 0–4 sigma) | Tests monotonicity of mmd_local against known conditional-expectation error. |

### Why this setup validates the claim

The central claim is that a per-sample overlap estimate improves the conditioning of the terminal-velocity target precisely where low-light latents fall outside the normal-light support. To falsify or confirm this, we need a baseline that applies the same velocity loss uniformly (uniform TVM) — if our gain over it is concentrated on low-overlap pairs and vanishes on high-overlap pairs, the overlap-weighting mechanism is doing the work; if the gain is uniform, the statistic is decorative. The discard-low-overlap baseline tests our specific argument that dropping hard pairs starves the model of the degraded inputs that matter, so we expect it to underperform on the very subset it discards. The always-on alignment baseline tests whether the rho threshold is load-bearing for detail preservation. The random-weight ablation is the crux: if replacing the MMD overlap with random weights of the same marginal distribution reproduces our gain, the kernel statistic is not measuring what we claim. PSNR restricted to the low-overlap subset is the right primary metric because it is the only regime where the predicted mechanism operates; averaging over all pairs would dilute the predicted effect toward parity. The two added checks (held-out calibration and synthetic sweep) directly test the load-bearing assumption that the overlap weight ranks pairs by target conditioning: the calibration check uses the existing overlap statistic and a fixed reference network to measure rank correlation with residual, and the synthetic sweep provides a ground-truth conditional-expectation error against which the same estimator is compared. If either check shows no monotone relationship, the assumption is falsified even if PSNR gains appear.

### Expected outcome and causal chain

**vs. Uniform TVM (no overlap weighting)** — On a case where the low-light latent sits far outside the normal-light support (e.g. a severely underexposed region whose latent is an outlier), uniform TVM fits a velocity target that is poorly conditioned because the conditional mean E[z1−z0|z0] is extrapolated from sparse support, producing a blurred or color-shifted reconstruction in that region. Our method instead down-weights that pair's velocity residual and pulls z0 toward its normal-light neighborhood, so the terminal velocity is estimated within a better-conditioned region. We expect a noticeable PSNR gap concentrated on the low-overlap subset and near-parity on the high-overlap subset.

**vs. Discard-low-overlap training** — On a case where the low-light image is globally degraded (most patches are low-overlap), the discard baseline never sees these inputs during training and therefore at inference produces a velocity field that has never been calibrated for this shift, yielding artifacts or a regression to the mean. Our method keeps these pairs with weak but non-zero gradient and applies alignment, so the network retains exposure to the hard regime. We expect the discard baseline to trail us specifically on the globally-degraded subset, while matching us on mildly-degraded inputs it was trained on.

**vs. Always-on alignment regularizer** — On a case where the low-light latent is valid but near a fine-texture boundary (high overlap, non-convex support edge), always-on alignment pulls z0 toward the neighborhood mean because it applies the convex-hull penalty unconditionally, blurring fine detail and depressing high-frequency fidelity. Our method gates alignment off when overlap exceeds rho, so valid boundary points are left untouched. We expect the always-on baseline to show a localized PSNR drop on textured regions while matching us on smooth low-overlap regions, and our random-weight ablation to fail to reproduce this pattern.

**vs. SOVA-TVM with random overlap weights** — On a case where the MMD overlap statistic correctly identifies a low-overlap pair (e.g. a severely underexposed region), the random-weight ablation assigns a random weight of the same marginal distribution, so the velocity residual for that pair is weighted independently of its true conditioning. If the MMD statistic carries real signal, the random-weight ablation should fail to concentrate gains on the low-overlap subset and should show lower PSNR there than SOVA-TVM, while matching on high-overlap pairs where weights are near 1. We expect the random-weight ablation to underperform SOVA-TVM specifically on the low-overlap subset, confirming that the kernel statistic, not the marginal weight distribution, drives the gain.

**Calibration check (held-out Spearman correlation)** — On the held-out set of 512 pairs, we compute the overlap weight for each pair and the squared terminal-velocity residual of a fixed uniform-TVM reference network. If the load-bearing assumption holds, the overlap weight should be positively rank-correlated with negative residual (i.e., higher overlap → lower residual). We expect a Spearman correlation of at least 0.3 (moderate positive) with a 95% bootstrap confidence interval excluding zero. If the correlation is near zero or negative, the weight does not rank pairs by conditioning and the assumption is falsified.

**Synthetic check (two-Gaussian sweep)** — For mean separations delta from 0 to 4 sigma, we compute mmd_local exactly as in the method and compare exp(-mmd_local/tau) to the known conditional-expectation error of the velocity target. If the assumption holds, the overlap weight should decrease monotonically with delta and should be strongly rank-correlated (Spearman ≥ 0.7) with the negative conditional-expectation error across the sweep. If the correlation is weak or non-monotone (e.g., due to k-NN concentration in d=16), the local MMD is not a calibrated measure of identifiability, and the method's central claim is unsupported even if PSNR gains appear on LOL.

### What would falsify this idea

If our gain over uniform TVM is uniform across the low-overlap and high-overlap subsets rather than concentrated on low-overlap pairs, or if the random-weight ablation reproduces our gain, or if the held-out calibration check shows no positive rank correlation between overlap weight and negative residual, or if the synthetic sweep shows non-monotone or weak correlation between overlap weight and true conditional-expectation error, then the MMD overlap statistic is not measuring the conditioning of the terminal-velocity target and the central claim is wrong.

## References

1. GLOW-FM: Efficient Generative Low-Light Image Enhancement with One-Step Latent Flow Matching
2. Building Normalizing Flows with Stochastic Interpolants
