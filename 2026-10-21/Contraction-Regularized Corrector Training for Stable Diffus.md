# Contraction-Regularized Corrector Training for Stable Diffusion Sampling

## Motivation

Existing corrector training for diffusion models (Correction Training for Diffusion Models) trains the corrector with a standard denoising loss on the base model's predictions, but provides no guarantee that repeated application of the corrector at a fixed sampling step forms a contraction in the sampler's state space. This structural gap means the iterative refinement can diverge or oscillate, especially when the base model's errors are large or the corrector is applied many times, which is precisely the regime where correction is most needed. The root cause is that the training objective never penalizes the composed map's deviation from a non-expansive operator, so stability is left to chance rather than enforced by design.

## Key Insight

Enforcing a contraction constraint on the corrector's Jacobian relative to the base sampler's step yields a composed map that is provably non-expansive, turning iterative refinement from a heuristic into a stable fixed-point iteration.

## Method

**What it is** — We introduce **StableCorrector**, a training procedure for a corrector network that refines the base diffusion model's clean-sample prediction while penalizing expansion of the composed map (base denoiser followed by the corrector) in the sampler's state space. Inputs are the base model's prediction at a given noise level and the corresponding noisy latent; outputs are the refined clean-sample prediction. The load-bearing assumption is stated explicitly: the spectral norm of the composed Jacobian J_corrector @ J_base, estimated by 5-step power iteration on the base denoiser's Jacobian at the noisy latent x_t, is a faithful and trainable proxy for the actual contraction behavior of the iterated map x -> corrector(base_denoiser(x,t), x, t) applied K times during sampling. This assumption is verified empirically via the calibration procedure described below.

**How it works** — The corrector is a U-Net sharing the base architecture but taking the base model's clean-sample prediction as an additional input channel. Training proceeds in two phases:

```pseudocode
# Phase 1: Standard correction training (warm-up)
for each batch (x0, xt, t):
    x0_hat = base_denoiser(xt, t)          # frozen base model
    x0_refined = corrector(x0_hat, xt, t)
    loss_denoise = ||x0_refined - x0||^2
    update corrector by ∇ loss_denoise

# Phase 2: Stability-aware fine-tuning
for each batch (x0, xt, t):
    x0_hat = base_denoiser(xt, t)
    x0_refined = corrector(x0_hat, xt, t)
    # Compute Jacobian of corrector w.r.t. its input x0_hat via JVP
    J = jacobian(corrector, x0_hat)        # shape [B, D, D]
    # Estimate contraction factor: spectral norm of composed map
    # Composed map: x -> corrector(base_denoiser(x, t), x, t)
    # We approximate its Jacobian as J @ J_base, where J_base is base denoiser Jacobian
    J_base = jacobian(base_denoiser, xt)   # frozen, computed once per batch
    composed_J = J @ J_base
    # Penalize deviation from non-expansiveness: ||composed_J||_2 <= 1 - epsilon
    # Use power iteration to estimate spectral norm
    sigma = power_iteration(composed_J, num_iters=5)
    loss_stability = relu(sigma - (1 - epsilon))^2   # epsilon = 0.01
    loss_total = loss_denoise + lambda * loss_stability   # lambda = 0.1
    update corrector by ∇ loss_total

# Calibration (every 500 steps, on a held-out batch of 64 samples):
#   - Compute true top singular value via 50 power iterations (or Lanczos)
#   - Compare to 5-iteration estimate; log relative error
#   - Apply corrector K=3 times to its own output; log per-step iterate distances
#     d_k = ||f^{k}(x) - f^{k-1}(x)|| for k=1..K
#   - If mean d_{k+1}/d_k > 1 - epsilon/2, increase lambda to 0.2 for next 500 steps
```

During sampling, at each denoising step, the corrector is applied K times (K=3 by default) to its own output. The stability loss ensures that each application reduces the distance to the fixed point. The calibration procedure directly tests the load-bearing assumption by measuring the true spectral norm and the empirical contraction ratio across K applications; if the 5-iteration estimate deviates from the true top singular value by more than 20% relative error, we increase power iteration to 10 iterations for the next 500 steps.

**Why this design** — We chose to penalize the spectral norm of the composed Jacobian rather than the corrector's Jacobian alone because the base sampler's step is part of the map, and a contraction of the corrector alone does not guarantee contraction of the composition; this requires estimating the base Jacobian, which we do efficiently via JVP and power iteration, accepting the cost of extra backward passes during training. We chose a warm-up phase with standard denoising loss before adding the stability penalty because early in training the corrector is far from optimal and the stability constraint would dominate, preventing useful refinement; this two-phase schedule adds a hyperparameter (warm-up duration) but stabilizes optimization. We chose a soft penalty (squared hinge) rather than a hard projection because projecting the Jacobian onto the contraction set is computationally intractable for high-dimensional outputs, accepting that the constraint is only softly enforced. Unlike the original Correction Training for Diffusion Models, which applies the corrector iteratively without any stability guarantee, our method explicitly trains the corrector to be a contraction relative to the base sampler, which a domain expert would not describe as a mere variant of the original because it introduces a fundamentally different training signal derived from the composed map's spectral properties.

**Why it measures what we claim** — The spectral norm of the composed Jacobian measures the contraction factor of the iterative refinement because the mean value theorem implies that for small perturbations, the distance between successive iterates is bounded by the spectral norm times the previous distance; this assumption fails when the corrector is highly nonlinear and perturbations are large, in which case the spectral norm only reflects local contraction and global convergence may not hold. The stability loss (hinge on spectral norm) measures the deviation from non-expansiveness because it is zero exactly when the spectral norm is at most 1-epsilon, which is the definition of a contraction with factor 1-epsilon; this assumption fails when the power iteration estimate is inaccurate (e.g., due to few iterations), in which case the loss may underestimate the true spectral norm and allow instability. The warm-up denoising loss measures the fidelity of the refined prediction because it is the standard L2 reconstruction error; this assumption fails when the base model's errors are not correctable (e.g., out-of-distribution noise levels), in which case the loss reflects the irreducible error of the base model rather than the corrector's contribution. The calibration procedure measures the gap between the 5-iteration power iteration estimate and the true top singular value (via 50 iterations) because the hinge loss relies on the estimate being accurate; this assumption fails when the composed Jacobian is ill-conditioned or non-normal, in which case the calibration logs the relative error and triggers an increase in power iteration count.

## Contribution

["A stability-aware training procedure for diffusion correctors that penalizes the spectral norm of the composed map's Jacobian, enforcing a contraction constraint relative to the base sampler's step.", 'An empirical demonstration that enforcing contraction during training leads to stable iterative refinement and improved sample quality compared to unconstrained corrector training, particularly when the number of refinement iterations is large.', 'A practical implementation using Jacobian-vector products and power iteration to estimate the spectral norm efficiently during training, making the stability penalty scalable to high-dimensional image data.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | CIFAR-10 | Standard benchmark for diffusion sample quality |
| Primary metric | FID | Captures sample fidelity and diversity jointly |
| Baseline | Base diffusion model | Tests whether correction helps at all |
| Baseline | Original Correction Training | Tests whether stability training improves over naive correction |
| Baseline | Extra sampling steps | Tests whether gains are just from more compute |
| Baseline | Spectral normalization on corrector's own Jacobian | Tests whether composed-map formulation is necessary |
| Baseline | Fixed-point/DEQ-style training of corrector | Tests whether contraction is just a reparameterization |
| Ablation-of-ours | StableCorrector without stability loss | Isolates effect of contraction penalty |
| Ablation-of-ours | StableCorrector with 10-iteration power iteration | Tests sensitivity to spectral norm estimate accuracy |
| Diagnostic | Per-step iterate distances d_k across K=3 | Directly tests contraction during sampling |

### Why this setup validates the claim

This design tests the central claim that enforcing contraction of the composed base-denoiser-plus-corrector map yields more reliable iterative refinement than naive correction. The base diffusion model baseline establishes the no-correction reference: if our method cannot beat it, the corrector adds nothing. The original Correction Training baseline is the critical comparison: it uses the same architecture and iterative application but lacks the stability penalty, so any FID gap between it and StableCorrector directly isolates the value of the composed-map contraction constraint. The extra-sampling-steps baseline controls for the obvious confound that our method uses more computation (K=3 corrector applications); if StableCorrector merely matches extra steps, the stability mechanism is not doing causal work. The spectral normalization on the corrector's own Jacobian baseline tests whether penalizing the composed map is necessary versus a simpler contraction trick on the corrector alone. The fixed-point/DEQ-style training baseline tests whether our two-phase training with a soft penalty is just a reparameterization of existing fixed-point training. The ablation without the stability loss tests whether the penalty itself, rather than the two-phase schedule or architecture, drives the improvement. The 10-iteration power iteration ablation tests sensitivity to the spectral norm estimate. The per-step iterate distance diagnostic directly measures whether the K-step refinement contracts, providing a mechanistic check beyond FID. FID is the right metric because the predicted failure mode of unstable correction is divergence or mode collapse during iterative refinement, which FID detects as degraded sample quality and reduced diversity; a metric that only measured single-step reconstruction error would miss this.

### Expected outcome and causal chain

**vs. Base diffusion model** — On a case where the base denoiser makes a moderately large error at an intermediate noise level (e.g., t around 0.3–0.5 of the schedule), the baseline produces a sample that inherits this error because it has no mechanism to correct its own clean-sample prediction. Our method instead refines the base prediction by applying the corrector, which was trained to reduce reconstruction error, so we expect a moderate FID improvement (roughly 5–15% relative) concentrated at noise levels where the base model's single-step error is largest, with near-parity at very low and very high noise levels.

**vs. Original Correction Training** — On a case where the corrector's output is iterated K=3 times, the original method can produce oscillating or diverging refinements because its training signal never penalizes expansion in the composed map, so successive applications may amplify errors. Our method instead explicitly penalizes the spectral norm of the composed Jacobian, so each application is trained to contract toward a fixed point, and we expect StableCorrector to show a larger FID gap over the original correction method specifically on samples where the original's iterative refinement diverges (detectable as increased variance across K iterations), with the gap shrinking when K=1. The per-step iterate distance diagnostic should show d_{k+1}/d_k < 1 for StableCorrector and potentially >1 for the original method on those samples.

**vs. Extra sampling steps** — On a case where the base model is run with additional denoising steps to match our method's compute budget, the baseline produces improved but still uncorrected samples because more steps reduce discretization error but do not correct the base model's systematic prediction bias. Our method instead uses the same compute to apply a learned contraction toward the true clean sample, so we expect StableCorrector to outperform extra-steps at equal compute, with the advantage largest on samples where the base model's bias (not discretization error) dominates, such as highly textured regions.

**vs. Spectral normalization on corrector's own Jacobian** — On a case where the corrector's own Jacobian has spectral norm < 1 but the composed map's Jacobian has spectral norm > 1 due to the base denoiser's amplification, the spectral normalization baseline fails to prevent divergence because it ignores the base step. Our method penalizes the composed map, so we expect StableCorrector to show lower FID and more stable iterate distances than this baseline, especially at intermediate noise levels where the base Jacobian's norm is largest.

**vs. Fixed-point/DEQ-style training of corrector** — On a case where the corrector is trained to be a fixed-point operator without explicit contraction penalty, the baseline may converge to a fixed point but not necessarily a contraction, so iterative refinement can oscillate. Our method adds the contraction penalty, so we expect StableCorrector to have lower FID and smaller iterate distances, with the gap largest when K=3 and shrinking when K=1.

### What would falsify this idea

If StableCorrector's FID gain over the original Correction Training is uniform across noise levels and sample subsets rather than concentrated where iterative refinement is unstable (e.g., intermediate noise, high-frequency regions), or if the ablation without the stability loss matches StableCorrector's performance, or if the per-step iterate distance diagnostic shows no contraction (d_{k+1}/d_k ≥ 1) for StableCorrector, then the contraction mechanism is not the cause of any improvement and the central claim is wrong.

## References

1. Correction Training for Diffusion Models
