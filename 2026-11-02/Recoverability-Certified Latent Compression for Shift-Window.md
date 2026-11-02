# Recoverability-Certified Latent Compression for Shift-Window Video Denoising

## Motivation

Across the interactive-video-generation tree (Yan's real-time framework with compressed 3D-VAE plus KV-cache shift-window denoising, and autoregressive-distillation pipelines like the Slow-Bidirectional-to-Fast work), latent compression is tuned by rate-distortion proxies (PSNR/LPIPS) and denoiser distillation is tuned separately, with no formal sufficiency relation between the two. The structural gap is that no branch can certify, at encode time, that a compressed latent is sufficient for the shift-window denoiser to reproduce the uncompressed denoising trajectory within a bounded gap — so Yan's 3D-VAE and the distilled denoiser are optimized as independent knobs and the interaction between compression error and denoiser sensitivity is invisible until failure at inference.

## Key Insight

Because the shift-window denoiser's attention is restricted to a bounded temporal window, its Jacobian with respect to any single latent is a local contraction whose modulus is estimable at encode time without touching denoiser weights, so the latent's downstream trajectory error is dominated by this modulus times the latent perturbation — making a per-latent recoverability certificate a computable sufficient statistic for trajectory fidelity rather than a proxy.

## Method

**What it is.** We introduce **ReCLa** (Recoverability-Certified Latent Compression), which augments a 3D-VAE encoder with a per-latent *denoising-recoverability certificate* — the local Lipschitz modulus of the frozen KV-cache shift-window denoiser restricted to that latent's neighborhood — and trains the encoder to emit latents whose certificate falls below a fidelity budget, so that distillation on the compressed latent tracks the uncompressed trajectory within a bounded gap. The load-bearing assumption, stated plainly: the finite-difference local Lipschitz modulus c(z) of the frozen shift-window denoiser, estimated with K=8 Gaussian probes at sigma=0.05, is a sufficient statistic that upper-bounds the downstream trajectory gap after distillation, so that enforcing c(z) <= eps at encode time guarantees the compressed-latent denoising trajectory tracks the uncompressed one within a bounded gap. This assumption is verified empirically in Phase 1b below by regressing measured trajectory gap against c(z)*||delta|| on a held-out set; the certificate is treated as a validated bound only within the regime where that regression holds.

**How it works.**
```pseudocode
Inputs: frozen 3D-VAE encoder E, frozen shift-window denoiser D_theta,
        fidelity budget eps, window W=16 frames, probe count K=8,
        perturbation scale sigma=0.05, calibration set of 512 clips

# Phase 1: offline certificate estimator (no denoiser retraining)
for each latent z = E(x) of a calibration clip:
    sample K=8 random perturbations {delta_k} ~ N(0, sigma^2 I) in latent space
    for each k: compute tau_k = || D_theta(z + delta_k) - D_theta(z) || / ||delta_k||
                # D_theta restricted to the W=16-frame window containing z
    certificate c(z) = max_k tau_k          # local Lipschitz modulus estimate
    # also record the per-probe values {tau_k} for variance reporting

# Phase 1b: empirical calibration of eps (validates the load-bearing assumption)
# On a held-out calibration split of 128 clips, for each latent z:
#   - run the full W-step denoising rollout from z and from z+delta_k
#   - measure true trajectory gap g_k = || rollout(z+delta_k) - rollout(z) ||
#   - regress g_k against c(z)*||delta_k||; report R^2 and residual
#   - set eps = the 90th percentile of c(z) over the calibration set such that
#     the regression-predicted gap on held-out clips stays below the target
#     trajectory-gap tolerance (pre-registered as 0.02 in LPIPS units)
#   - report c(z) variance across the K=8 probes and across sigma in {0.01,0.05,0.1}
#   - gate acceptance on the upper 95% confidence bound of c(z) (mean + 1.96*std/sqrt(K))
#     rather than the point max, so a latent is accepted only if the bound <= eps

# Phase 2: encoder fine-tuning with certificate penalty
for each training video x:
    z = E(x);  c(z) = certificate_estimator(z)   # K=8, sigma=0.05, W=16 frames
    L_rd  = reconstruction_loss(x) + beta * KL(q(z|x) || p(z))
    L_cert = max(0, c_ub(z) - eps)               # hinge on upper confidence bound, eps calibrated in Phase 1b
    L = L_rd + lambda * L_cert                   # lambda = 1.0
    update E only (D_theta frozen)

# Phase 3: distillation on certified latents
student S initialized along teacher ODE trajectory; train with DMD loss
on latents z that satisfy c_ub(z) <= eps; reject-and-resample z with c_ub(z) > eps

# Inference: encode clip -> check c_ub(z) -> if c_ub(z) > eps, apply one
#   residual-recovery step (add back the top principal direction of the
#   estimated delta that maximizes tau) before feeding to S.
```

**Why this design.** We chose to *freeze* the denoiser and estimate the certificate via finite-difference probes rather than backpropagating through it, because Yan's real-time constraint means the denoiser is already deployed and retraining it would break the KV-cache shift-window contract, accepting the cost that the certificate is a sampled estimate rather than an exact Jacobian norm. We chose a *hinge* certificate penalty over an additive penalty because the fidelity budget eps is a hard operational requirement for real-time certification, accepting the cost that latents far below eps receive no gradient signal and the encoder may under-utilize capacity. We chose *reject-and-resample* during distillation over weighting the loss by c(z), because rejection gives a clean certified training set whose trajectory-gap behavior is homogeneous, accepting the cost of discarding some usable latents and a slightly higher sample complexity. We added Phase 1b because the load-bearing assumption — that a single-point finite-difference modulus bounds the multi-step trajectory gap — is not self-evident: the denoiser is applied iteratively and its Jacobian along the sampling path is a product of per-step Jacobians, so a per-latent modulus at the encode point does not by itself bound error growth over the sampling path. Phase 1b makes the bound empirical and regime-specific: eps is set from measured trajectory gaps on held-out clips, and the certificate is trusted only where the regression of gap against c(z)*||delta|| holds. This differs from the Slow-Bidirectional-to-Fast distillation pipeline, which supervises a causal student with a bidirectional teacher but never conditions the training latents on a denoiser-sensitivity quantity; a domain expert would not describe ReCLa as a variant of that method because the load-bearing object here is the encoder-side certificate, not the distillation schedule.

**Why this measures what we claim.** The finite-difference modulus c(z) measures *denoising-recoverability* (the motivation-level sufficiency of the compressed latent) because the shift-window denoiser's bounded attention makes its response to a latent perturbation locally linear with slope tau_k, so the trajectory gap after distillation is dominated by c(z) times the latent error; this assumption fails when the perturbation sigma is large enough to leave the linear regime of the windowed attention (e.g., a scene cut that changes which frames fall in the window), in which case c(z) reflects the worst-case local slope rather than the actual trajectory gap. Phase 1b tests this directly by regressing measured trajectory gap against c(z)*||delta|| and reporting R^2 and residuals, so the regime where the linear model breaks is identified rather than assumed. The reconstruction loss measures *rate-distortion fidelity* because PSNR-optimal reconstruction implies small latent error under the VAE prior; this assumption fails when the VAE posterior is multi-modal, in which case reconstruction loss reflects mode-averaged fidelity rather than the specific latent the denoiser will see. The hinge budget eps measures *operational certification* because it is the threshold below which the trajectory gap is provably small under the contraction assumption; this assumption fails when the denoiser is not a contraction (e.g., at high noise levels where attention saturates), in which case eps certifies only the local window and not the full trajectory. The reject-and-resample step measures *certified training-set purity* because accepted latents satisfy the same bound the certificate was derived from; this assumption fails when the calibration distribution differs from the training distribution, in which case the accepted set is certified only in-distribution.

## Contribution

(1) A per-latent denoising-recoverability certificate — the local Lipschitz modulus of a frozen KV-cache shift-window denoiser estimated by finite-difference probes — that is computable at encode time and serves as a sufficient statistic for downstream trajectory fidelity, replacing PSNR/LPIPS proxies. (2) A certificate-constrained 3D-VAE fine-tuning procedure with a hinge penalty on the recoverability modulus, plus a reject-and-resample distillation protocol that trains the student only on certified latents. (3) An analysis of the certificate's behavior across noise levels and scene cuts, identifying the linear-regime boundary where the contraction assumption holds and the failure mode where it does not.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | StreamEval interactive benchmark | Scripted cases with state and fidelity metrics. |
| Primary metric | Trajectory gap vs. uncompressed teacher | Directly tests certified distillation bound. |
| Secondary metric | Regression R^2 of gap vs. c(z)*||delta|| | Validates the load-bearing linear model. |
| Baseline | Slow-Bidirectional-to-Fast distillation | Tests whether certificate conditioning adds value. |
| Baseline | Yan real-time interactive generator | Tests whether ReCLa improves deployed system. |
| Baseline | HunyuanVideo (uncompressed 3D-VAE) | Tests whether compression harms fidelity. |
| Ablation-of-ours | ReCLa without certificate penalty | Isolates effect of recoverability certification. |
| Ablation-of-ours | ReCLa with sigma in {0.01, 0.05, 0.1} and W in {8, 16, 32} | Locates regime where linear model holds. |

### Why this setup validates the claim

This setup forms a falsifiable test because each baseline isolates a different sub-claim. Slow-Bidirectional-to-Fast tests whether conditioning distillation latents on a denoiser-sensitivity certificate improves trajectory tracking beyond a standard distillation schedule; if ReCLa matches it, the certificate is unnecessary. Yan tests whether our encoder-side certification improves a deployed real-time system; if no gain, the certificate does not transfer to the operational setting. HunyuanVideo tests whether compression itself is the bottleneck; if ReCLa matches uncompressed quality, compression is harmless. The no-certificate ablation isolates the hinge penalty: if removing it does not degrade trajectory gap, the certificate is not load-bearing. The sigma/W sweep tests the load-bearing assumption directly: if the regression of measured trajectory gap against c(z)*||delta|| has high R^2 only for small sigma and W=16, the certificate is a validated bound only in that regime, and we report the gap separately outside it. The primary metric, trajectory gap, is the right detector because the central claim is a bounded gap after distillation; task-specific metrics could improve for unrelated reasons, but trajectory gap directly measures the certified quantity. The regression R^2 is the right validator because it tests whether c(z)*||delta|| actually predicts the downstream gap, which is the load-bearing assumption stated in the method.

### Expected outcome and causal chain

**vs. Slow-Bidirectional-to-Fast distillation** — On a case with rapid motion where the compressed latent has high local denoiser sensitivity, the baseline produces drift because its distillation loss weights all latents equally and cannot reject high-sensitivity ones. Our method instead rejects latents with c_ub(z) > eps and resamples, because the hinge penalty drives the encoder to emit recoverable latents, so we expect a noticeable trajectory-gap reduction on high-motion subsets but parity on static subsets. We also expect the regression of measured gap against c(z)*||delta|| to show R^2 > 0.7 on the high-motion subset at sigma=0.05, W=16, confirming the linear model in the regime where the gain appears.

**vs. Yan real-time interactive generator** — On a case with a scene cut that shifts the KV-cache window, the baseline produces temporal inconsistency because its fixed compression ignores window-local denoiser sensitivity. Our method instead checks c_ub(z) at inference and applies a residual-recovery step when c_ub(z) > eps, because the certificate flags the shift-window neighborhood, so we expect a visible consistency improvement on cut-containing cases but parity on continuous shots. We expect the regression R^2 to drop on cut-containing cases, identifying the regime where the linear model fails and the certificate reflects worst-case local slope rather than actual trajectory gap.

**vs. HunyuanVideo (uncompressed 3D-VAE)** — On a case with fine texture and slow motion, the baseline produces high fidelity but at prohibitive latency, because it uses no compression. Our method instead achieves comparable fidelity with certified latents, because the hinge penalty preserves recoverability within eps, so we expect trajectory gap within a small margin of uncompressed while retaining real-time speed. We expect the sigma/W sweep to show that the Pareto frontier of latency vs. trajectory gap for ReCLa dominates the Slow-Bidirectional-to-Fast baseline at the calibrated eps, with the frontier degrading gracefully as sigma increases beyond 0.1 or W exceeds 32.

### What would falsify this idea

If ReCLa's trajectory-gap gain over Slow-Bidirectional-to-Fast is uniform across all subsets rather than concentrated on high-c(z) cases, then the certificate is not the causal mechanism. If the no-certificate ablation matches ReCLa on high-c(z) cases, the hinge penalty is not load-bearing. If the regression of measured trajectory gap against c(z)*||delta|| has low R^2 even at sigma=0.05, W=16 on held-out clips, then the load-bearing assumption that c(z) bounds the trajectory gap is false and the certificate is only a local slope proxy.

## References

1. Yan: Foundational Interactive Video Generation
2. HunyuanVideo: A Systematic Framework For Large Video Generative Models
3. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
4. StreamEval: Evaluating Interactive Streaming Video Generation Models
