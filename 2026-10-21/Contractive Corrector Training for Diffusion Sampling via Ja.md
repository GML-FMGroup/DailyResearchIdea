# Contractive Corrector Training for Diffusion Sampling via Jacobian-Norm Regularization

## Motivation

Correction Training for Diffusion Models trains its corrector with the same denoising loss as the base model, so the corrector is optimized only to match clean-sample targets and carries no constraint tying successive refinement steps together. Structurally, this means the composed corrector-sampler map has no certified error-contraction property: iterative refinement can amplify, not shrink, the integration error, and there is no mechanism guaranteeing that applying the corrector twice is better than applying it once. The field's trajectory toward learned post-hoc correction inherits this gap because correction is treated as a per-step regression problem rather than as a dynamical system whose stability must be controlled.

## Key Insight

If the corrector is Lipschitz-constrained and trained so that the local Jacobian norm of the composed corrector-sampler map is strictly below one on the error-relevant subspace, then the refinement iteration is a contraction in a Lyapunov function of the integration error, so each corrector application is guaranteed to shrink rather than amplify the residual regardless of the base model's error distribution.

## Method

**What it is.** We introduce CACO (Contraction-Aware Corrector), a corrector network for diffusion sampling that takes the base denoiser's clean-sample prediction x̂₀ and the current noise level σ as input and outputs a refined prediction, trained with an added Jacobian-norm regularizer so that the composed corrector-sampler map is a contraction in a Lyapunov function of the integration error. The load-bearing assumption, stated explicitly: the corrector's contraction of the x̂₀ prediction error at a fixed noise level σ is the object that controls accumulated sampling error, i.e., shrinking ||corrector(x̂₀,σ) − x₀|| at each DDIM step guarantees the composed multi-step sampler trajectory contracts rather than amplifies error. This holds only when the DDIM update's own gain (the ∂x_{t-1}/∂x_t coefficient and the frozen base denoiser's local Jacobian) does not amplify the residual between corrector applications; we therefore calibrate and verify this assumption empirically (see 'Calibration and verification' below) rather than assume it.

**How it works.**

```pseudocode
# Training (base denoiser frozen)
for each batch (x0, σ, ε):
    x_σ   = x0 + σ·ε
    x̂0    = base_denoiser(x_σ, σ)              # frozen
    y0    = corrector(x̂0, σ; θ)                # first refinement
    y1    = corrector(y0, σ; θ)                # second refinement (self-input)
    L_task = ||y1 - x0||²                       # denoising target
    # Lyapunov function: V(e) = e^T P e, P = I on error-relevant subspace
    e0 = y0 - x0 ; e1 = y1 - x0
    L_lyap = max(0, ||e1||²_P - κ·||e0||²_P)    # κ = 0.9 contraction margin
    # Local Jacobian norm of composed map, power-iteration estimate of top singular value
    # (replaces single-probe Hutchinson; see Calibration and verification)
    v  ~ N(0, I)
    for i = 1..5:                               # power iteration steps
        w = J^T v ; v = J w / ||J w||           # J = ∂corrector(y0,σ;θ)/∂y0 via JVP/VJP
    σ_max² = ||J v||²                           # estimate of top singular value squared
    L_jac = max(0, σ_max² - (1 - γ)²)           # γ = 0.1 slack
    L = L_task + λ_lyap·L_lyap + λ_jac·L_jac    # λ_lyap = 1.0, λ_jac = 0.5
    θ ← θ - η·∇L
# Architecture: Spectral-normalized U-Net corrector (power iteration per layer)

# Sampling
x_T ~ N(0, I)
for t = T..1:
    x̂0 = base_denoiser(x_t, σ_t)
    for k = 1..K:                               # K = 3 refinement steps
        x̂0 = corrector(x̂0, σ_t)
    x_{t-1} = DDIM_step(x_t, x̂0, σ_t)
```

**Calibration and verification of the load-bearing assumption.** To test whether contraction of the corrector's x̂₀ error at fixed σ controls the sampler's accumulated error, we run the following concrete procedure on a held-out set of 512 CIFAR-10 samples, using the trained base denoiser and a 50-step DDIM sampler:
1. For each sample and each of 10 evenly spaced noise levels σ_t, compute the base residual r_t = x̂₀(x_t,σ_t) − x₀ and the corrected residual r'_t = corrector^K(x̂₀,σ_t) − x₀.
2. Propagate both residuals through the actual DDIM update: δ_t = DDIM_step(x_t, x̂₀+r_t, σ_t) − DDIM_step(x_t, x̂₀+r'_t, σ_t), and record the gain g_t = ||δ_t|| / ||r_t − r'_t|| (the empirical operator norm of the sampler step restricted to the residual direction).
3. Report the distribution of g_t across t and samples. The assumption is verified if the median g_t ≤ 1 for all t; it is falsified if g_t > 1 systematically at low noise (where the DDIM coefficient is largest).
4. If g_t > 1 at some t, we do not change the training objective (which remains the corrector-map contraction); instead we report this as a measured limitation and use it to interpret any FID gap that appears only at high noise.

**Why this design.** We chose a spectral-normalized U-Net over an unconstrained corrector because the contraction property requires a bounded Lipschitz constant that the architecture itself must enforce, accepting the cost that spectral normalization slightly reduces the corrector's expressive capacity for sharp high-frequency corrections. We chose a two-step self-input training objective (y0 and y1 both supervised) over single-step training because the contraction condition is a property of the composed map, not of one application, accepting the cost of roughly doubled corrector forward passes per training step. We chose the Lyapunov penalty on the error norm ratio over a direct Jacobian penalty alone because the Jacobian norm is only a local proxy while the error-ratio penalty directly enforces the contraction inequality along the actual error trajectory, accepting the cost that the error-ratio term is non-stationary early in training and requires a warm-up of 5k steps before the Lyapunov penalty is activated. We chose a power-iteration estimate of the top singular value (5 iterations, JVP/VJP pairs) over a single-probe Hutchinson estimate to reduce the under-penalization of anisotropic Jacobians, accepting roughly 5× the backward-pass cost per training step; we retain λ_jac = 0.5 to stabilize the estimate. A domain expert would not describe this as a variant of Correction Training because that method's corrector has no cross-step coupling in its objective; CACO's objective is defined on the composed two-step map, which is exactly the object the contraction condition constrains.

**Why it measures what we claim.** The error-ratio penalty `||e1||²_P / ||e0||²_P` measures the contraction of the refinement iteration because the assumption that the error-relevant subspace (spanned by the base model's residual) is captured by the identity metric P makes the ratio equal to the squared operator norm of the composed map restricted to that subspace; this assumption fails when the base model's error has significant components orthogonal to the residual it produces at the sampled point, in which case the ratio reflects contraction only on the observed error direction and can overstate global contraction. To test this assumption directly, we add a held-out diagnostic: for 256 samples, we construct orthogonal error perturbations e_⊥ (random vectors orthogonal to the base residual r_t) and measure the ratio ||corrector^2(x̂₀+e_⊥) − x₀||² / ||e_⊥||²; if this ratio exceeds 1 while the along-residual ratio is below κ, the identity-metric assumption is violated and the reported contraction is local to the residual direction. The power-iteration Jacobian estimate `σ_max²` measures the local Lipschitz constant of the corrector because the assumption that 5 power iterations converge to the top singular direction makes the estimate concentrate near the spectral norm; this assumption fails when the Jacobian has near-degenerate top singular values, in which case the estimate reflects a mixture of the top few directions and may under-penalize the worst case. The spectral normalization on each layer measures the architectural Lipschitz bound because the assumption that the product of per-layer spectral norms upper-bounds the network's global Lipschitz constant holds for the composition of 1-Lipschitz activations used here; this assumption fails when skip connections create paths whose combined norm exceeds the product bound, in which case the architectural constraint is necessary but not sufficient and the Lyapunov penalty must carry the remaining enforcement.

## Contribution

(1) A contraction-aware corrector training framework (CACO) that augments the denoising objective with a two-step Lyapunov error-ratio penalty and a Hutchinson-estimated Jacobian-norm penalty under spectral-normalized architecture constraints. (2) A design principle showing that the corrector's training objective must be defined on the composed multi-step refinement map rather than on single-step clean-sample regression to obtain error-contraction behavior. (3) An analysis of the gap between architectural Lipschitz bounds and trajectory-level contraction when skip connections are present, motivating the dual architectural-plus-penalty enforcement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | CIFAR-10 | Standard, fast to iterate, well-understood failure modes |
| Dataset (scale check) | ImageNet 64×64 | Tests whether effect persists at higher resolution |
| Primary metric | FID | Sensitive to both mode coverage and sample sharpness |
| Secondary metric | Contraction ratio ||e1||²/||e0||² on held-out 512 samples | Directly measures the claimed contraction |
| Secondary metric | Sampler-gain g_t (median over 512 samples, 10 σ levels) | Tests the load-bearing assumption |
| Baseline | Base denoiser, no corrector | Isolates the contribution of the corrector itself |
| Baseline | Base denoiser + 10× sampling steps | Tests whether extra compute alone recovers quality |
| Baseline | Correction Training (single-step corrector) | Tests whether cross-step coupling is necessary |
| Baseline | Fixed-point/denoising-consistency corrector (corrector(y0)≈y0 at fixed point) | Direct alternative contraction enforcement |
| Ablation-of-ours | CACO without Lyapunov penalty | Isolates the contraction-enforcement mechanism |
| Ablation-of-ours | CACO with single-probe Hutchinson (original) vs. power iteration | Quantifies Jacobian-estimate variance |
| Compute-matched control | K=3 corrector passes vs. equivalent extra DDIM steps (FLOPs-matched) | Separates refinement from raw compute |

### Why this setup validates the claim

The central claim is that a corrector trained on the composed two-step map with a contraction-inducing Lyapunov penalty yields a refinement iteration whose error shrinks across applications, producing better sample quality than either the unrefined base model or a corrector trained without cross-step coupling. The base-denoiser baseline tests the sub-claim that any corrector is needed; the 10×-steps baseline and the FLOPs-matched control test whether the gain is merely from extra network evaluations rather than a qualitatively better refinement map; the single-step Correction Training baseline tests whether the two-step self-input objective is the causal ingredient rather than just having a corrector at all. The fixed-point/consistency baseline tests whether CACO's Jacobian regularization yields a tighter certified contraction bound than a direct fixed-point objective on a controlled synthetic error field (a 2D Gaussian mixture with known residual field). The no-Lyapunov ablation isolates the contraction mechanism: if the gain survives without the Lyapunov penalty, the mechanism story is wrong. The secondary metrics directly test the load-bearing assumption: the contraction ratio measures whether the corrector map contracts along the residual, and the sampler-gain g_t measures whether the DDIM update preserves that contraction. FID is the right metric because the predicted effect is a reduction in accumulated integration error that manifests as fewer distorted or mode-missing samples—exactly what FID penalizes—rather than a change in per-pixel likelihood or a perceptual metric that saturates before error accumulation becomes visible. The ImageNet 64×64 run tests whether the effect is an artifact of CIFAR-10's low resolution.

### Expected outcome and causal chain

**vs. Base denoiser, no corrector** — On a case where the base model's x̂₀ prediction has a systematic bias in high-noise regimes (e.g., an early denoising step where the residual is large and the network's approximation error compounds over subsequent DDIM steps), the baseline produces a trajectory that drifts toward low-density regions because each step inherits and amplifies the prior step's error. Our method instead applies three contraction-guaranteed refinement steps that shrink the error norm by the κ=0.9 margin per application, so we expect a noticeable FID gap concentrated in the high-noise portion of the trajectory, with parity or near-parity at low noise where the base error is already small. The sampler-gain diagnostic g_t will show whether the DDIM update preserves this contraction; if g_t > 1 at low noise, the FID gap may shrink there.

**vs. Base denoiser + 10× sampling steps** — On a case where the base model's error is a persistent bias rather than a discretization artifact, the baseline produces only marginal improvement because more steps reduce discretization error but cannot correct the network's systematic prediction bias—the extra evaluations re-sample the same biased denoiser. Our method instead directly contracts the prediction error through the corrector, independent of step count, so we expect CACO to outperform the 10×-step baseline on the biased subset while the two are comparable on the subset where discretization dominates, producing a crossover pattern rather than uniform improvement. The FLOPs-matched control (K=3 corrector passes vs. equivalent extra DDIM steps) will confirm that any gain is not merely from extra compute.

**vs. Correction Training (single-step corrector)** — On a case where the corrector's first application undershoots (e.g., a moderate residual that the single-step objective maps only partway to x₀), the baseline produces a refinement that does not compose—applying it again can oscillate or stagnate because it was never trained to be a contraction of the composed map. Our method instead trains on y₁ as the supervised target, so the second application is explicitly optimized to further reduce ||y₁ − x₀||, and the Lyapunov penalty enforces ||e₁||² ≤ 0.9||e₀||²; we expect CACO's FID to improve monotonically with K=1,2,3 refinements while the single-step baseline's FID plateaus or degrades after K=1. The measured contraction ratio on held-out samples will directly show whether CACO achieves the κ=0.9 margin while the baseline does not.

**vs. Fixed-point/denoising-consistency corrector** — On a controlled synthetic error field (2D Gaussian mixture with known residual), the fixed-point baseline enforces corrector(y0)≈y0 at the fixed point but does not constrain the Jacobian norm away from the fixed point. Our method's power-iteration Jacobian penalty directly bounds the top singular value, so we expect CACO to achieve a strictly smaller measured contraction ratio (closer to κ) than the fixed-point baseline on the synthetic field, and a correspondingly larger FID gain at K=3 on CIFAR-10.

**vs. CACO without Lyapunov penalty** — On a case where the Jacobian of the corrector has a dominant singular direction orthogonal to the power-iteration probe (or where the probe has not converged), the ablation produces a corrector that passes the task loss but whose composed map is not contractive along the actual error trajectory, so repeated refinement amplifies error components in that direction. Our method instead enforces the error-ratio inequality directly along the observed trajectory, so we expect the ablation to match CACO at K=1 but diverge at K=3, with the gap concentrated in samples whose base residual lies near the unpenalized Jacobian's top singular direction. The orthogonal-perturbation diagnostic will quantify how much of the gap is due to error components orthogonal to the residual.

**vs. CACO with single-probe Hutchinson** — On a case where the Jacobian is highly anisotropic, the single-probe estimate under-penalizes the top singular direction. Our method uses 5 power iterations, so we expect a smaller gap between estimated and true spectral norm (measured on 256 held-out batches) and a tighter contraction ratio, at the cost of ~5× backward-pass compute per training step.

### What would falsify this idea

If CACO's FID gain over the single-step Correction Training baseline is uniform across all noise levels and refinement counts rather than concentrated at K≥2 and in the high-noise regime where contraction is predicted to matter, the cross-step coupling mechanism is not the cause and the central claim is wrong. Likewise, if removing the Lyapunov penalty leaves FID unchanged at K=3, the contraction enforcement is doing no work and the design rationale fails. Additionally, if the sampler-gain diagnostic g_t is systematically >1 at low noise and the FID gap vanishes there, the load-bearing assumption that corrector contraction controls sampler error is falsified.

## References

1. Correction Training for Diffusion Models
