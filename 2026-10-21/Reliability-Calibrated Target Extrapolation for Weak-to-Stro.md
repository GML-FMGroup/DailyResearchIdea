# Reliability-Calibrated Target Extrapolation for Weak-to-Strong Diffusion Training

## Motivation

SWIFT and its guidance ancestors (CFG, bad-version guidance, guidance-free methods) all leave the guidance/extrapolation strength as a hand-tuned hyperparameter, so the strength is never structurally tied to the reliability of the reference model; as a result, SWIFT cannot guarantee that a better weak checkpoint yields a better guided target, and checkpoint selection plus strength tuning remain ad-hoc. The root cause is that extrapolation strength is treated as a free scalar rather than a monotone function of a measurable property of the weak denoiser, so the method has no consistency guarantee between weak-model quality and guided-target quality.

## Key Insight

The optimal extrapolation strength is a monotone decreasing function of the weak denoiser's estimated score-matching error, so calibrating strength as a closed-form function of that error makes the guided target's expected denoising risk monotonically decrease as the weak model improves.

## Method

**What it is.** Reliability-Calibrated Extrapolation (RCE) is a training-time target rule for weak-to-strong diffusion: given a frozen weak denoiser $\epsilon_w$ and its estimated per-timestep score-matching error $\hat\sigma_w^2(t)$ on a held-out probe set, RCE sets the extrapolation strength $\lambda(t)$ by a closed-form calibration $\lambda(t)=\hat\sigma_w^2(t)/(\hat\sigma_w^2(t)+\tau^2(t))$ and uses it to form the regression target. Inputs: noisy batch $(x_t,t)$, weak prediction $\epsilon_w(x_t,t)$, strong model $\epsilon_\theta$. Output: calibrated target $y_t^{\mathrm{RCE}}$ and the standard denoising loss.

**Load-bearing assumption (stated explicitly).** RCE assumes the weak denoiser's score-matching error is a function of the noise level $t$ (or SNR bin), not a single scalar, and that within each of $K=20$ equal-width timestep bins the error is approximately constant. It further assumes the probe residuals are unbiased estimates of that per-bin error (probe set exchangeable with training distribution, weak model frozen) and that the strong model's noise floor $\tau^2(t)$ is stationary during training. If the per-bin residual variance is not approximately flat within a bin, the bin-level calibration averages over heterogeneity the method claims to exploit; if the probe set is small or shifted, $\hat\sigma_w^2(t)$ reflects probe variance rather than model error; if $\tau^2(t)$ changes rapidly early in training, $\lambda(t)$ reflects a stale noise floor.

**How it works.**

```pseudocode
# Phase 0: one-time calibration (before strong-model training)
Input: frozen weak denoiser eps_w, probe set D_probe of (x0, eps) pairs (|D_probe| = 2048)
K = 20 equal-width timestep bins over [0,1]
for each (x0, eps) in D_probe:
    sample t ~ U[0,1], xt = alpha_t x0 + sigma_t eps
    r = eps - eps_w(xt, t)                      # residual at weak model
    assign r to bin k = floor(t * K)
for each bin k:
    sigma_hat_sq[k] = mean over probe samples in bin k of ||r||^2
# closed-form calibration per bin (derived from bias-variance of extrapolated target)
# tau^2[k] is the strong-model noise floor per bin, estimated from a 500-step warmup
# (see Phase 0b below); lambda_max = 0.9, lambda_min = 0.0
for each bin k:
    lambda[k] = clip( sigma_hat_sq[k] / (sigma_hat_sq[k] + tau_sq[k]) , lambda_min, lambda_max )

# Phase 0b: warmup for tau^2[k] (500 steps, standard denoising loss, no extrapolation)
# After warmup, compute per-bin mean squared error of eps_theta on D_probe:
#   tau_sq[k] = mean over probe samples in bin k of ||eps - eps_theta(xt, t)||^2
# Freeze tau_sq[k] for the rest of training (stationarity assumption).

# Phase 1: strong-model training with calibrated target
for each training batch (x0, eps):
    sample t ~ U[0,1], xt = alpha_t x0 + sigma_t eps
    eps_w_pred = eps_w(xt, t)                    # frozen, no grad
    k = floor(t * K)
    y_RCE = eps + lambda[k] * (eps - eps_w_pred) # target extrapolated away from weak
    loss = || eps_theta(xt, t) - y_RCE ||^2
    update eps_theta by Adam on loss
# Phase 2: discard eps_w; sample with original sampler
```

**Why this design.** We chose a per-timestep (per-bin) closed-form calibration $\lambda(t)=\hat\sigma_w^2(t)/(\hat\sigma_w^2(t)+\tau^2(t))$ over a learned or grid-searched strength because the extrapolated target's bias-variance trade-off has a known optimum at this ratio (the weak model's error versus the strong model's noise floor) and because score-matching error in diffusion denoisers is strongly $t$-dependent, so a single global scalar would average over the exact heterogeneity the method claims to exploit; the cost is that $\tau^2(t)$ must be estimated from a short warmup and that the calibration is only as good as the probe set. We chose to freeze the weak denoiser and estimate $\hat\sigma_w^2(t)$ on a held-out probe set rather than reuse training batches, because training-batch residuals are contaminated by the strong model's own updates and would break the monotonicity claim; the cost is one extra forward pass per probe sample. We chose to clip $\lambda(t)$ to $[\lambda_{\min},\lambda_{\max}]$ with $\lambda_{\max}=0.9<1$ rather than allow unbounded extrapolation, because the spectral analysis in SWIFT shows the beneficial range is bounded; the cost is that a very poor weak model yields a conservative (small) strength rather than an aggressive one. Unlike classifier-free guidance (Ho & Salimans) or SWIFT, which treat strength as a free hyperparameter, RCE derives it from a measurable property of the reference model, so a domain expert would not describe RCE as a variant of SWIFT: SWIFT tunes $\lambda$, RCE computes it. Unlike a learned gating head (anti-pattern 1), RCE has no learned module asserting independence; the calibration is a deterministic function of an estimated error.

**Why it measures what we claim.** The quantity $\hat\sigma_w^2(t)$ measures the weak denoiser's reliability because, under the assumption that the probe residuals are unbiased estimates of the weak model's score-matching error (i.e., the probe set is exchangeable with the training distribution and the weak model is frozen), the mean squared residual per bin equals the score-matching error up to sampling noise; this assumption fails when the probe set is small or distribution-shifted, in which case $\hat\sigma_w^2(t)$ reflects probe-set variance rather than true weak-model error. The calibrated strength $\lambda(t)$ measures the optimal extrapolation strength because, under the assumption that the strong model's noise floor $\tau^2(t)$ is stationary during training, the bias-variance optimum of the extrapolated target is exactly $\sigma_w^2(t)/(\sigma_w^2(t)+\tau^2(t))$ per bin; this assumption fails when the strong model's error changes rapidly early in training, in which case $\lambda(t)$ reflects a stale noise floor rather than the current optimum. The monotonicity claim (better weak model $\Rightarrow$ better guided target) holds because $\lambda(t)$ is monotone decreasing in $\hat\sigma_w^2(t)$ and the guided target's expected risk is monotone decreasing in $\lambda(t)$ over the clipped range; this fails when $\hat\sigma_w^2(t)$ is outside the clip range, in which case the target saturates at $\lambda_{\min}$ or $\lambda_{\max}$ and the monotonicity is flat rather than strict.

## Contribution

(1) A reliability-calibrated target-extrapolation rule (RCE) that replaces SWIFT's free extrapolation strength with a closed-form monotone function of the weak denoiser's estimated score-matching error, eliminating strength tuning. (2) A calibration protocol that ties extrapolation strength to a measurable property of the reference model and yields a monotone consistency property: as the weak checkpoint improves, the guided target's expected denoising risk decreases. (3) An empirical design principle that checkpoint selection and strength setting can be unified into a single probe-based calibration step, removing two hyperparameters from the weak-to-strong diffusion pipeline.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CIFAR-10 and ImageNet-256 | Standard weak-to-strong diffusion benchmarks, comparable to SWIFT |
| Primary metric | FID | Sensitive to target bias, the quantity RCE controls |
| Baseline | Standard diffusion training (no guidance) | Tests whether any extrapolation helps at all |
| Baseline | SWIFT with tuned fixed λ | Tests whether computing λ beats tuning λ |
| Baseline | Classifier-free guidance | Tests against a free-hyperparameter guidance reference |
| Ablation-of-ours | RCE with λ fixed at grid-searched optimum | Isolates calibration from extrapolation itself |
| Ablation-of-ours | RCE with scalar λ (global, not per-bin) | Tests whether per-bin calibration is necessary |

### Why this setup validates the claim

The central claim is that the extrapolation strength should be computed from the weak model's measured error rather than tuned, and that this computed strength tracks the bias-variance optimum of the extrapolated target. Standard training tests the null that no extrapolation is needed; SWIFT with a tuned fixed λ tests the stronger alternative that extrapolation helps but its strength is a free hyperparameter; the fixed-λ ablation of RCE isolates whether any gain comes from the calibration rule or merely from target extrapolation; the scalar-λ ablation tests whether the per-bin calibration is necessary or whether a global scalar suffices. FID is the right detector because the predicted effect is a shift in the target's bias-variance balance, which surfaces directly as sample-quality degradation or improvement; a metric insensitive to target bias would not distinguish the mechanisms. If RCE's gain over SWIFT is concentrated exactly where the weak model's measured error is high, the calibration story is supported; if it is uniform, the calibration is doing nothing and the ablation should match.

### Expected outcome and causal chain

**vs. Standard diffusion training (no guidance)** — On a case where the strong model is trained from a weak checkpoint's neighborhood and early training targets are noisy, standard training fits the raw noise target and inherits the weak model's bias, producing blurred or mode-collapsed samples because it has no mechanism to push the target away from the weak predictor. Our method instead extrapolates the target away from the weak prediction by λ(t), because the residual direction is exactly the weak model's error direction, so we expect a clear FID gap early in training that narrows as the strong model's own error floor dominates later.

**vs. SWIFT with tuned fixed λ** — On a case where the weak model's error varies across timesteps and the tuned λ is a single global constant, SWIFT over-extrapolates at timesteps where the weak model is already accurate and under-extrapolates where it is poor, producing a target that is biased in the wrong direction on a subset of t. Our method instead scales λ(t) by the measured per-bin error, because the calibration ratio σ̂²(t)/(σ̂²(t)+τ²(t)) shrinks λ(t) precisely where the weak model is reliable, so we expect a noticeable FID gap concentrated on the high-weak-error timestep band and near-parity on the low-error band.

**vs. Classifier-free guidance** — On a case where guidance strength must be hand-tuned for a new dataset and the default scale is mismatched, CFG produces over-saturated or washed-out samples because its strength is a free hyperparameter disconnected from the reference model's actual reliability. Our method instead derives strength from a measurable property of the frozen weak denoiser, because the probe-set residual is an unbiased estimate of its score-matching error, so we expect RCE to match CFG at its best-tuned scale without per-dataset tuning, showing a smaller spread across datasets than CFG's scale sweep.

**vs. RCE with scalar λ (global, not per-bin)** — On a case where the weak model's error varies strongly across timesteps, a global scalar λ averages the high-error and low-error bands into a strength that is optimal for neither, producing a target that is over-extrapolated in the low-error band and under-extrapolated in the high-error band. Our method instead uses per-bin λ(t), because the calibration is computed separately for each of K=20 timestep bins, so we expect the per-bin version to outperform the scalar version specifically on the high-weak-error timestep band, with near-parity on the low-error band.

### What would falsify this idea

If RCE's gain over the fixed-λ ablation is uniform across timesteps and datasets rather than concentrated on the high-weak-error subset where the calibration ratio predicts it should act, the claim that computed strength tracks the bias-variance optimum is wrong. Equally, if RCE underperforms a well-tuned fixed λ on the high-error subset, or if the scalar-λ ablation matches per-bin RCE, the calibration rule is miscalibrated or the per-bin refinement is unnecessary.

## References

1. SWIFT: Self-guided Weak-to-strong dIFfusion models via Target extrapolation
