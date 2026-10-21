# Reverse-Time SDE Closure for Target Extrapolation in Diffusion Training

## Motivation

SWIFT's extrapolation target is constructed per noise level without enforcing that the resulting composite field is a valid score function along the reverse trajectory; its spectral analysis is population-based and local, so the extrapolated target can drift from the score of any density as training proceeds. This missing closure affects every guidance branch built on SWIFT, because the extrapolation strength must be tuned empirically rather than derived from a global consistency condition. The root cause is structural: SWIFT augments the regression target pointwise and never checks the Fokker–Planck residual that ties the target's time derivative to its spatial score along the reverse-time SDE.

## Key Insight

Enforcing a Fokker–Planck residual penalty on the extrapolated target forces the composite field to be the score of a valid density along the reverse trajectory, turning SWIFT's local pointwise extrapolation into a globally closed score function without tuning the extrapolation strength.

## Method

**What it is.** We introduce FP-SWIFT (Fokker–Planck-closed SWIFT), a training procedure that augments SWIFT's per-noise-level extrapolation target with a reverse-time SDE closure regularizer. Inputs are a frozen weak denoiser $D_w$, a trainable strong denoiser $D_\theta$, and a noise-level schedule $\{\sigma_t\}$; outputs are the trained parameters $\theta$ and a closed composite target $\hat{x}_0^{\text{FP}}(x_t,t)$ that is a valid score along the reverse trajectory.

**How it works.** The procedure has three phases per mini-batch.

```pseudocode
# Phase 1: SWIFT extrapolation target (per sample, per noise level)
x_t = alpha_t * x_0 + sigma_t * eps          # forward noising, eps ~ N(0,I)
target_swift = D_theta(x_t, t) + lambda * (D_theta(x_t, t) - D_w(x_t, t))
# lambda is the SWIFT extrapolation strength (fixed at 1.0 in our runs)

# Phase 2: reverse-time SDE closure residual
# Reverse SDE: dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dW_bar
# Score of the composite target: s_hat(x_t,t) = (alpha_t * target_swift - x_t) / sigma_t^2
# Fokker-Planck residual for the density p_hat induced by s_hat:
#   R(x_t,t) = d_t log p_hat(x_t,t) + div_x [ f(x_t,t) - g(t)^2 * s_hat(x_t,t) ]
#             + 0.5 * g(t)^2 * || s_hat(x_t,t) ||^2
# We evaluate R at (x_t, t) and at a perturbed time t+dt along the reverse trajectory:
#   x_{t+dt} = x_t + [f(x_t,t) - g(t)^2 * s_hat(x_t,t)] * dt + g(t) * sqrt(dt) * z,  z ~ N(0,I)
#   R_dt = ( log p_hat(x_{t+dt}, t+dt) - log p_hat(x_t, t) ) / dt
#          + div_x [ f - g^2 s_hat ](x_t,t) + 0.5 * g(t)^2 * || s_hat(x_t,t) ||^2
# Use Hutchinson trace estimator for div_x and log p_hat via the denoiser's score:
#   log p_hat(x,t) = -|| s_hat(x,t) ||^2 * sigma_t^2 / 2 + const   (Gaussian approximation)
# Closure loss:
L_closure = mean( R_dt^2 )

# Phase 3: total loss
L_total = L_denoise(target_swift, x_0) + mu * L_closure
# mu is the closure weight, set to 0.1 by default; dt = 1e-3; lambda = 1.0
```

**Why this design.** We chose a Fokker–Planck residual penalty over a direct score-matching regularizer because the FP residual is the exact PDE that any valid score must satisfy along the reverse trajectory, so it enforces global closure rather than local consistency; the cost is that the residual requires a time derivative and a divergence, which we estimate with a finite-difference along the reverse SDE and a Hutchinson trace estimator. We fixed the SWIFT extrapolation strength $\lambda=1.0$ instead of tuning it because the closure term absorbs the role of $\lambda$: the residual penalizes any extrapolation that breaks score validity, so the strength no longer needs empirical search; the trade-off is that $\mu$ now controls the balance, but $\mu$ has a clear interpretation as a PDE penalty weight. We evaluate the residual at a perturbed time step $t+dt$ along the reverse trajectory rather than at $t$ alone because the reverse-time SDE closure is a statement about the time derivative of the density, and a single-time residual would miss that derivative; the cost is an extra denoiser evaluation per sample, which we amortize by sharing the forward pass. A domain expert would not describe this as 'SWIFT + a regularizer' because the closure term changes the target itself: the composite target is projected onto the manifold of valid scores, which is a structural constraint that SWIFT's pointwise extrapolation does not impose.

**Why it measures what we claim.** The computational quantity $R_{dt}$ measures trajectory closure because the Fokker–Planck residual is zero if and only if the composite score $s_\hat{}$ is the score of a density that evolves under the reverse-time SDE; this assumption fails when the Gaussian approximation for $\log p_\hat{}$ is inaccurate (e.g., at very low noise levels where the density is multimodal), in which case $R_{dt}$ reflects the Gaussianization error rather than true closure. The quantity $\text{div}_x[f - g^2 s_\hat{}]$ measures the divergence of the reverse drift because the trace estimator is unbiased for the divergence under the assumption that the Hutchinson probe vectors are isotropic; this assumption fails when the score has strong anisotropic curvature, in which case the estimator has high variance and the residual is noisy. The finite-difference time derivative measures the rate of change of the log-density along the reverse trajectory because the Euler–Maruyama step approximates the SDE path under the assumption that $dt$ is small relative to the noise schedule's timescale; this assumption fails when $dt$ is too large, in which case the residual reflects discretization error rather than closure violation. The SWIFT extrapolation target $\text{target}_{swift}$ measures the weak-to-strong guidance because the frozen weak denoiser provides an early-stopped estimate of the score under the assumption that the weak model's errors are underfit components that extrapolation amplifies; this assumption fails when the weak model is overfit or when $\lambda$ is too large, in which case the target amplifies noise rather than signal.

**Load-bearing assumption (explicit).** The load-bearing assumption is that the Fokker–Planck residual $R_{dt}$, computed with a Gaussian-approximated log-density, a Hutchinson divergence estimator, and a single Euler–Maruyama finite-difference step, is a faithful and low-variance proxy for score validity such that penalizing it actually projects the composite target onto the manifold of valid scores rather than merely minimizing estimator noise. This assumption is stated in plain language here because the entire method depends on it: if $R_{dt}$ is dominated by estimator variance or Gaussianization error, then the closure term is a noisy regularizer and the central claim fails.

**Calibration and verification of the load-bearing assumption.** To make the assumption testable, we replace the self-referential Gaussian log-density with an explicit, verifiable closure condition and add a synthetic ground-truth check. Specifically, we enforce that $s_\hat{}$ is a gradient field (curl-free) via a Jacobian-symmetry penalty $L_{\text{sym}} = \| \nabla_x s_\hat{} - (\nabla_x s_\hat{})^\top \|_F^2$ and that it matches the denoiser-implied score via a proper score-matching identity against the true conditional score: $L_{\text{sm}} = \| s_\hat{}(x_t,t) - \nabla_{x_t} \log p(x_t|x_0) \|^2$, where $\nabla_{x_t} \log p(x_t|x_0) = -\epsilon/\sigma_t$ is available in closed form during training. The total loss becomes $L_{\text{total}} = L_{\text{denoise}} + \mu L_{\text{closure}} + \mu_{\text{sym}} L_{\text{sym}} + \mu_{\text{sm}} L_{\text{sm}}$, with $\mu_{\text{sym}} = 0.01$ and $\mu_{\text{sm}} = 0.1$ by default. We validate on a tractable 2D density (a Gaussian mixture with 8 components) where the exact Fokker–Planck residual is computable: we verify that $R_{dt}$ is near zero for the true score and grows monotonically with injected score error $\| s_\hat{} - s_{\text{true}} \|$, confirming that $R_{dt}$ tracks closure violation rather than approximation error. This calibration is a diagnostic, not a new mechanism; it specifies how the existing closure term is verified.

## Contribution

["A new training objective, FP-SWIFT, that augments SWIFT's per-noise-level extrapolation target with a reverse-time SDE Fokker–Planck residual penalty, yielding a composite target that is a valid score function along the reverse trajectory rather than a pointwise extrapolation.", 'A concrete design principle: the extrapolation strength in SWIFT can be fixed rather than tuned when a closure regularizer is present, because the PDE residual penalizes invalid extrapolations and removes the need for empirical search over $\\lambda$.', 'An implementation recipe using Hutchinson trace estimation and Euler–Maruyama finite differences to evaluate the Fokker–Planck residual with one extra denoiser evaluation per sample, making the closure penalty practical for standard diffusion training.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CIFAR-10 | Standard benchmark with known diffusion failure modes |
| Dataset | ImageNet 64x64 | Higher-resolution check that closure benefit is not an artifact of one underfit regime |
| Dataset | LSUN (bedroom, 64x64) | Second higher-resolution check for generality |
| Dataset | Synthetic 2D Gaussian mixture (8 components) | Tractable density where exact Fokker–Planck residual is computable for calibration |
| Primary metric | FID | Captures distribution-level score validity |
| Secondary metric | Recall at matched precision | Tests diversity collapse relative to CFG |
| Secondary metric | $R_{dt}$ vs. injected score error curve (2D) | Verifies that $R_{dt}$ tracks closure violation, not approximation error |
| Baseline | Standard diffusion training | Isolates benefit of any weak-to-strong guidance |
| Baseline | SWIFT (λ=1.0) | Isolates benefit of FP closure over extrapolation alone |
| Baseline | Classifier-free guidance | Strong external guidance reference point |
| Baseline | Plain score-matching regularizer (no FP residual) | Tests whether the FP residual mechanism is distinct from score matching |
| Ablation-of-ours | FP-SWIFT with μ=0 | Tests whether closure term, not λ, drives gains |
| Ablation-of-ours | FP-SWIFT with λ swept over {0.5, 1.0, 2.0, 4.0} | Reports FID-vs-λ sensitivity for both SWIFT and FP-SWIFT to substantiate 'no tuning needed' |
| Ablation-of-ours | FP-SWIFT with weak model early-stopped at {50k, 100k, 200k} steps | Tests closure benefit across multiple weak-model quality levels |
| Ablation-of-ours | FP-SWIFT with antithetic Hutchinson probes (2 probes) | Variance-reduction check for the divergence estimator |
| Ablation-of-ours | FP-SWIFT with dt ∈ {1e-3, 1e-2, 1e-1} | Checks discretization sensitivity of the finite-difference time derivative |

### Why this setup validates the claim

The central claim is that SWIFT's pointwise extrapolation target is not a valid score along the reverse trajectory, and that a Fokker–Planck closure residual projects it onto the manifold of valid scores, yielding better generation. Standard diffusion training tests whether guidance helps at all; SWIFT tests whether extrapolation alone helps; classifier-free guidance tests whether our gains merely match a strong external guidance scheme; the plain score-matching regularizer tests whether the FP residual is reducible to ordinary score matching. The μ=0 ablation is the decisive control: it is exactly SWIFT with our code path, so any gap between FP-SWIFT and μ=0 isolates the closure term rather than the extrapolation strength λ. FID is the right metric because the claim is about distribution-level score validity — a better-calibrated score along the reverse trajectory should reduce the mismatch between generated and data distributions, which FID directly measures, whereas per-sample losses would not distinguish a valid score from a locally consistent one. The synthetic 2D experiment with known true score directly tests the load-bearing assumption: if $R_{dt}$ does not grow with injected score error, the closure term is measuring estimator noise rather than closure violation. The λ-sweep and weak-model-quality sweep test whether the 'no tuning needed' claim holds across regimes rather than at a single operating point.

### Expected outcome and causal chain

**vs. Standard diffusion training** — On a case where the frozen weak denoiser is an early-stopped checkpoint with underfit high-frequency components, the baseline produces blurry samples and elevated FID because it has no mechanism to amplify the missing signal. Our method instead extrapolates away from the weak model and then penalizes extrapolations that break score validity, so we expect a clear FID improvement over the baseline concentrated on samples whose reverse trajectories pass through noise levels where the weak model is most underfit. We expect the improvement to persist on ImageNet 64x64 and LSUN, and to be largest for the weakest early-stopped checkpoint (50k steps) and shrink as the weak model improves (200k steps), consistent with the closure term doing more work when the extrapolation target is further from a valid score.

**vs. SWIFT (λ=1.0)** — On a case where pointwise extrapolation overshoots at intermediate noise levels, SWIFT produces a target that is locally sharp but not a valid score, yielding artifacts and mode distortion because it has no global closure constraint. Our method instead evaluates the FP residual along the reverse trajectory and penalizes drift-divergence mismatch, so we expect FP-SWIFT to beat SWIFT on FID with the gap largest at intermediate noise levels and near-parity at very low noise where the Gaussian approximation degrades. The λ-sweep should show SWIFT's FID is sensitive to λ (a U-shaped curve with a minimum at some λ*), while FP-SWIFT's FID is flatter across λ, substantiating the 'no tuning needed' claim.

**vs. Classifier-free guidance** — On a case where CFG's fixed guidance scale over-saturates easy classes, the baseline produces reduced diversity and FID degradation because its guidance is external and class-conditional rather than score-validity-driven. Our method instead regularizes the composite score toward PDE closure without an external scale, so we expect comparable or better FID without the diversity collapse CFG shows, observable as a smaller recall penalty at matched precision.

**vs. Plain score-matching regularizer** — On a case where the plain score-matching regularizer enforces pointwise agreement with the conditional score but does not constrain the composite field to be a valid score along the reverse trajectory, the baseline should match FP-SWIFT at low noise but underperform at intermediate noise where the FP residual's divergence and time-derivative terms carry information not present in pointwise score matching. We expect FP-SWIFT to beat the plain score-matching regularizer on FID, with the gap concentrated at intermediate noise levels, demonstrating that the FP residual is not reducible to score matching.

**vs. μ=0 ablation** — On a case where the closure term is active, the μ=0 ablation is exactly SWIFT with our code path, so any FID gap between FP-SWIFT and μ=0 isolates the closure term. We expect FP-SWIFT to beat μ=0 on FID, with the gap largest at intermediate noise levels and near-zero at very low noise where the Gaussian approximation degrades. If μ=0 matches FP-SWIFT within noise, the closure term is not doing the causal work.

**Synthetic 2D calibration** — On the 8-component Gaussian mixture where the true score is known, we expect $R_{dt}$ to be near zero for the true score and to grow monotonically with injected score error $\| s_\hat{} - s_{\text{true}} \|$, confirming that $R_{dt}$ tracks closure violation rather than approximation error. If $R_{dt}$ does not grow with injected error, the load-bearing assumption is false and the closure term is measuring estimator noise.

### What would falsify this idea

If FP-SWIFT's gain over the μ=0 ablation is uniform across noise-level subsets rather than concentrated at intermediate noise levels where the closure residual is most informative, or if the μ=0 ablation matches FP-SWIFT within noise, or if the synthetic 2D calibration shows $R_{dt}$ does not grow with injected score error, then the closure term is not doing the causal work and the central claim is wrong.

## References

1. SWIFT: Self-guided Weak-to-strong dIFfusion models via Target extrapolation
