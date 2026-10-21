# Pullback-Consistent Target Extrapolation for Latent Diffusion

## Motivation

Latent diffusion trains the denoiser on targets that live in VAE latent space, and methods that manipulate those targets—most directly SWIFT's weak-to-strong extrapolation—operate as if the encoder-decoder map were a lossless, geometry-preserving interface. This assumption is structurally false: the VAE encoder is neither surjective nor an isometry, so an extrapolated latent target can leave the image of the encoder and its decoded image can drift arbitrarily far from the pixel-space extrapolation that SWIFT's spectral argument actually licenses. The same unexamined assumption underlies REPA-E's latent-space alignment, meaning every latent-extrapolation branch inherits a target that is not guaranteed to be realizable by any image.

## Key Insight

Requiring the extrapolated latent target to be the encoder image of a pixel-space extrapolation—i.e., forcing extrapolation to commute with the encoder-decoder map—constrains the target to the encoder's range and removes the component of extrapolation that is pure decoder-nullspace distortion.

## Method

**What it is.** Pullback-Consistent Extrapolation (PCE) is a drop-in replacement for SWIFT's latent target construction. Inputs: noisy latent $z_t$, frozen weak denoiser $\epsilon_w$, trainable strong denoiser $\epsilon_\theta$, frozen VAE $(E,D)$, extrapolation strength $\lambda$, equivariance weight $\beta$, finite-difference step $\varepsilon=10^{-3}$, round-trip gain threshold $g_{\min}=0.1$. Output: a training loss whose regression target is a latent that is (approximately) the encoder image of a pixel-space extrapolation, plus an equivariance penalty on the predicted clean latent.

**Load-bearing assumption (stated explicitly).** PCE assumes the VAE encoder-decoder pair acts as a near-left-inverse on the latent neighborhood visited during training, so that the round-trip gain $g = \|E(D(z)+\varepsilon d) - E(D(z))\| / (\varepsilon\|d\|)$ is not too small (we require $g \ge g_{\min}$) and the residual $\|E(D(z))-z\|^2$ is a valid measure of off-range distortion. When this assumption fails (e.g., a low-dimensional or heavily quantized VAE whose $E\circ D$ is a strong contraction), the pullback target collapses toward the weak estimate and the method degrades gracefully to SWIFT by gating $\lambda$ on $g$.

**How it works.**
```pseudocode
# per training step, batch of (x, z_t, t)
z0_w   = (z_t - sqrt(1-abar_t)*eps_w(z_t,t)) / sqrt(abar_t)   # weak clean-latent estimate
x0_w   = D(z0_w)                                               # decode to pixel space
d      = x0_w - D(stopgrad(z0_w))                              # pixel-space extrapolation direction
# residual-preserving pullback: project d through encoder local Jacobian via finite difference
g      = || E(D(z0_w) + eps*d) - E(D(z0_w)) || / (eps * ||d|| + 1e-8)   # round-trip gain
lam_eff = lambda * min(1.0, g / g_min)                         # gate lambda on gain
z0_tgt = z0_w + lam_eff * (E(D(z0_w) + eps*d) - E(D(z0_w))) / (eps + 1e-8)   # pullback direction
# main regression
z0_theta = (z_t - sqrt(1-abar_t)*eps_theta(z_t,t)) / sqrt(abar_t)
L_reg = || z0_theta - stopgrad(z0_tgt) ||^2
# encoder-equivariance penalty on the model's own clean-latent prediction
L_eq  = || E(D(z0_theta)) - z0_theta ||^2                      # commute residual
L     = L_reg + beta * L_eq
```
Hyperparameters: $\lambda$ annealed from 0 to $\lambda_{\max}$ over the first 20% of training (same schedule family as SWIFT); $\beta$ fixed at $10^{-3}$; $\varepsilon=10^{-3}$; $g_{\min}=0.1$; $E,D$ frozen; $\epsilon_w$ frozen early checkpoint, discarded after training.

**Why this design.** We chose to build the extrapolation target in pixel space and re-encode it, rather than extrapolating directly in latent space as SWIFT does, because the pixel-space target is guaranteed to lie in the domain where SWIFT's spectral argument is stated, accepting the cost of one extra decode-encode pass per step (≈15% wall-clock). We chose a soft equivariance penalty $\beta\|E(D(z_\theta))-z_\theta\|^2$ over a hard projection onto the encoder range because a hard projection requires solving a per-sample optimization at every step, accepting that residual off-range components are shrunk rather than eliminated. We chose to apply the penalty to the model's own clean-latent prediction rather than only to the target because it also regularizes the denoiser's output geometry, accepting a small bias toward decoder-reconstructible latents. A domain expert would not describe this as a variant of SWIFT: SWIFT's mechanism is target extrapolation with no reference to the autoencoder, whereas PCE's load-bearing mechanism is a commutativity constraint $E\circ D\circ(\text{extrapolate}) = (\text{extrapolate})\circ E\circ D$ that SWIFT never enforces; nor is it REPA-E, which aligns latents to external features rather than to the encoder's own range.

**Why it measures what we claim.** The commutativity residual $\|E(D(z_\theta))-z_\theta\|^2$ measures **encoder-equivariance of the predicted clean latent** because it equals zero exactly when $z_\theta$ lies in the range of $E$ and $D$ acts as a left inverse there; this assumption fails when the VAE decoder is not injective on the relevant latent neighborhood (decoder-nullspace directions survive), in which case the residual reflects only the nullspace-projected distortion rather than full equivariance. The re-encoded target $E(x0_{tgt})$ measures **pullback-consistency of the extrapolation** because it is by construction the encoder image of a pixel-space extrapolation, so its distance from the naive latent extrapolation $z0_w+\lambda(z0_w-z0_w')$ quantifies exactly the off-range component SWIFT silently injects; this assumption fails when the pixel-space extrapolation itself leaves the natural-image manifold, in which case $E(x0_{tgt})$ measures encoder sensitivity to off-manifold inputs rather than a meaningful target. The round-trip gain $g$ measures **the VAE's local left-inverse quality** because it is the finite-difference norm of $E\circ D$ along the extrapolation direction; this assumption fails when $E\circ D$ is a strong contraction (low-dimensional or quantized VAE), in which case $g$ is small and $\lambda$ is gated down, so the method degrades gracefully to SWIFT rather than producing a collapsed target.

## Contribution

(1) A pullback-consistent target-extrapolation mechanism (PCE) that replaces latent-space extrapolation with encoder-image extrapolation plus a soft encoder-equivariance penalty, applicable to any latent diffusion model with a frozen VAE. (2) The design principle that weak-to-strong target manipulation must commute with the encoder-decoder map, yielding a measurable commutativity residual that predicts when latent extrapolation degrades generation. (3) An analysis tool—the off-range component $\|E(x0_{tgt}) - (z0_w+\lambda\Delta)\|$—that diagnoses the failure mode of latent-space extrapolation as a function of extrapolation strength and VAE reconstruction quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CIFAR-10 and ImageNet-256 | Matches SWIFT's own evaluation surface for direct comparison |
| Primary metric | FID | Standard distributional distance; sensitive to off-manifold latent drift |
| Baseline | Standard diffusion training | Isolates the benefit of any extrapolation target |
| Baseline | SWIFT (latent-space target extrapolation) | Isolates the benefit of pixel-space pullback over latent extrapolation |
| Baseline | REPA-E-style latent alignment | Tests whether gains are just generic latent regularization |
| Ablation-of-ours | PCE without equivariance penalty (β=0) | Isolates the commutativity term's contribution |
| Ablation-of-ours | PCE with hard projection onto encoder range (per-sample min ||z - z0_tgt||^2 s.t. z in range(E) via 5 gradient steps) | Compares soft penalty vs hard projection to show commutativity is distinct from generic regularization |
| Ablation-of-ours | PCE with λ gated off (g_min = ∞, i.e., always use SWIFT target) | Tests the round-trip gain gating mechanism |
| Controlled perturbation | Inject known off-range perturbations into z0_w: z0_w + α * u where u is a random unit vector orthogonal to the encoder range (estimated via 10-step gradient ascent on ||E(D(z))-z||^2), α ∈ {0.1, 0.5, 1.0} | Directly tests causal link between off-range drift and FID degradation |

### Why this setup validates the claim

The central claim is that SWIFT's latent-space extrapolation silently injects off-range components that the denoiser must then fit, and that building the target in pixel space and re-encoding it (PCE) removes exactly those components while the equivariance penalty keeps the model's own predictions near the encoder range. Standard training tests whether any extrapolation helps at all; SWIFT tests whether the *latent-space* form of extrapolation is the right one; REPA-E tests whether PCE's gain is merely the generic effect of regularizing latents toward a structured subspace rather than the specific commutativity mechanism. The β=0 ablation is the sharpest test: if the equivariance penalty is load-bearing, removing it should degrade FID disproportionately on the samples where the weak model's latent estimate is most off-range, and if it is not load-bearing, PCE reduces to pixel-space SWIFT and the ablation should match. The hard-projection ablation tests whether the soft penalty is sufficient or whether the commutativity mechanism requires exact range enforcement. The λ-gated-off ablation tests whether the round-trip gain gating is necessary to avoid the VAE smoothing failure mode. The controlled perturbation experiment directly tests the causal link: if injecting known off-range perturbations into the weak estimate degrades FID for SWIFT but not for PCE, then the off-range component is the cause of degradation and PCE's removal of it is the mechanism. FID is the right primary metric because off-range latent targets manifest as distributional drift in generated samples, which FID captures directly and which per-sample reconstruction metrics would miss.

### Expected outcome and causal chain

**vs. Standard diffusion training** — On a case where the weak denoiser's clean-latent estimate is biased toward blurry, low-frequency modes (early training, high noise), standard training fits the model to the same biased target and inherits that bias, because it has no mechanism to push the target away from the weak model's mode. Our method extrapolates away from the weak estimate in pixel space and re-encodes, so the target is a sharper natural image that the strong model is pushed toward, and the equivariance penalty keeps its own predictions reconstructible. We expect a modest but consistent FID gap that grows with λ_max, with the largest gains on high-noise timesteps where the weak model is most biased.

**vs. SWIFT (latent-space target extrapolation)** — On a case where the weak clean-latent estimate has a component outside the VAE encoder's range (common when the weak model is undertrained), SWIFT's latent extrapolation amplifies that off-range component by (1+λ), and the strong model is trained to reproduce a latent that no image decodes to, because SWIFT never references the autoencoder. Our method builds the target as E(x0_w + λ(x0_w − D(z0_w))), which is by construction in the encoder range, so the amplified direction is projected back onto decodable latents. We expect PCE to beat SWIFT by a clear margin on FID, with the gap concentrated on samples whose weak estimate has the largest off-range residual, and near-parity on samples where the weak estimate is already in-range.

**vs. REPA-E-style latent alignment** — On a case where the denoiser's latent prediction drifts off the encoder range during training, REPA-E pulls it toward external feature statistics that need not lie in the encoder range, so the drift is redirected rather than eliminated. Our method penalizes the commutativity residual directly, so the prediction is pulled toward the encoder's own range with no external reference. We expect PCE to match REPA-E on overall FID but to show a distinctly lower commutativity residual, and to outperform REPA-E specifically on the subset where the weak model's estimate is most off-range.

**vs. PCE without equivariance penalty (β=0)** — On a case where the model's own clean-latent prediction drifts off the encoder range during training, the β=0 ablation has no mechanism to pull it back, so its predictions can accumulate off-range components that the regression target does not penalize. We expect full PCE to outperform β=0 by a margin that grows with the off-range residual of the weak estimate, and to show a lower commutativity residual on held-out samples.

**vs. PCE with hard projection** — On a case where the soft penalty leaves residual off-range components, the hard projection enforces exact range membership at the cost of per-sample optimization. We expect the hard projection to match or slightly exceed full PCE on FID, with the difference concentrated on samples where the soft penalty's residual is largest, demonstrating that the commutativity mechanism is distinct from generic regularization.

**vs. PCE with λ gated off** — On a case where the VAE round-trip gain g is small (e.g., low-dimensional VAE), the gated-off variant uses the SWIFT target and should match SWIFT's performance, while full PCE with gating should degrade gracefully to SWIFT rather than collapsing. We expect full PCE to match SWIFT when g < g_min and to outperform SWIFT when g ≥ g_min, confirming the gating mechanism.

**Controlled perturbation experiment** — On a case where we inject a known off-range perturbation α·u into the weak estimate, SWIFT's target amplifies it by (1+λ) and FID should degrade monotonically with α, while PCE's pullback should project it out and FID should remain stable across α. We expect a significant interaction: FID degradation with α for SWIFT but not for PCE, directly confirming that off-range components cause FID degradation and that PCE removes them.

### What would falsify this idea

If PCE's FID gain over SWIFT is uniform across samples rather than concentrated on the samples with the largest off-range residual (measured by ||E(D(z0_w)) − z0_w||), or if the β=0 ablation matches full PCE, or if the controlled perturbation experiment shows no differential effect of α on FID between SWIFT and PCE, then the commutativity mechanism is not what produces the gain and the central claim is wrong.

## References

1. SWIFT: Self-guided Weak-to-strong dIFfusion models via Target extrapolation
