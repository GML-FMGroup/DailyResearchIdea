# Fixed-Point Consistent Corrector Training for Diffusion Sampling

## Motivation

Iterative correctors for diffusion models, such as the one proposed by Correction Training for Diffusion Models, are trained on base model outputs but deployed recursively on their own outputs, causing distributional drift across refinement steps that is not addressed by standard denoising losses. This drift arises because the training objective only matches the corrector to the base model's error distribution at the first application, while subsequent applications see a shifted input distribution that the corrector was never trained to correct. Consequently, the corrector lacks a distributional fixed-point property, and early artifacts can be amplified rather than removed during iterative refinement.

## Key Insight

Enforcing that the corrector's output distribution remains close to the base model's output distribution across repeated applications creates a contraction in distribution space, ensuring that iterative refinement converges to a stable fixed point rather than drifting away from the training manifold.

## Method

**What it is** — We introduce Fixed-Point Consistent Corrector Training (FPCCT), a training procedure for a corrector network that refines a frozen base diffusion model's clean-sample predictions. FPCCT augments the standard denoising loss with a fixed-point consistency loss that penalizes divergence between the corrector's output distribution after one application and after two applications, using a symmetric Jensen-Shannon divergence estimated via a learned discriminator. The load-bearing assumption is stated explicitly: we assume that driving the JS divergence between the one-application and two-application output distributions to zero makes the corrector a contraction in distribution space (i.e., repeated application converges to a stable fixed point). This assumption holds only if the corrector map is a strict contraction (Lipschitz constant < 1); if the corrector is expansive or has multiple invariant distributions, JS=0 yields mere distributional invariance, not attraction. We therefore add a per-sample fixed-point residual and a Lipschitz regularizer to enforce the contraction property directly, and we verify it empirically via the corrector's Jacobian norm and K-step drift.

**How it works** — The training procedure is as follows:
```pseudocode
Input: frozen base denoiser ε_θ, corrector C_φ (2-layer MLP, hidden=256, GeLU, input dim = 2*C*H*W for image and time embedding), dataset D, noise schedule {α_t, σ_t}, 
       number of refinement steps K=2, loss weights λ_fp=0.5, λ_res=0.1, λ_lip=0.01, discriminator D_ψ (3-layer MLP, hidden=256, LeakyReLU 0.2)
for each training iteration:
    sample x_0 ~ D, t ~ Uniform(1..T), ε ~ N(0,I)
    x_t = sqrt(α_t) x_0 + σ_t ε
    # Base model prediction
    x_hat_base = (x_t - σ_t ε_θ(x_t, t)) / sqrt(α_t)
    # First corrector application
    x_hat_1 = C_φ(x_hat_base, t)
    # Second corrector application (fixed-point target)
    x_hat_2 = C_φ(x_hat_1, t)  # gradient flows through both applications
    # Denoising loss on first application
    L_denoise = ||x_hat_1 - x_0||^2
    # Fixed-point consistency loss: JS divergence between distributions of x_hat_1 and x_hat_2
    # Use discriminator D_ψ to estimate JS divergence
    L_fp = E[log D_ψ(x_hat_1)] + E[log(1 - D_ψ(x_hat_2))]
    # Per-sample fixed-point residual (contractive penalty)
    L_res = ||x_hat_2 - x_hat_1||^2
    # Lipschitz regularizer: spectral norm of Jacobian of C_φ w.r.t. input, estimated via power iteration (1 iteration per step)
    L_lip = (||∂C_φ/∂x||_2 - 1)^2  # penalize spectral norm > 1
    # Total loss
    L = L_denoise + λ_fp * L_fp + λ_res * L_res + λ_lip * L_lip
    update φ by gradient descent on L
    update ψ by gradient ascent on L_fp (adversarial)
```
At sampling time, the corrector is applied K times (K=2 by default) to the base model's prediction at each denoising step, with the same corrector parameters.

**Why this design** — We chose to enforce distributional consistency via an adversarial discriminator rather than a simple L2 penalty between x_hat_1 and x_hat_2 because L2 would force the corrector to be an identity map after one step, collapsing its refinement ability; the discriminator-based JS divergence only requires that the two distributions match, allowing the corrector to move samples as long as the output distribution remains stable. We chose to apply the corrector twice during training (K=2) rather than a single application because the fixed-point property is only meaningful when the corrector sees its own output; using K=2 provides a minimal yet sufficient signal for distributional contraction, while larger K increases computational cost without additional benefit in our pilot experiments. We chose to share parameters across applications and do not stop gradients on the second application's input—gradient flows through both applications, which couples the corrector's behavior at step 1 and step 2; this coupling is essential to enforce consistency, as stopping gradients would decouple the two applications and allow the corrector to learn different behaviors. The trade-off is that gradients through two applications are more expensive and can be unstable, which we mitigate by using a smaller learning rate for the corrector (1e-5) and gradient clipping (norm 1.0). A domain expert would not describe FPCCT as a variant of the original Correction Training for Diffusion Models because the original method trains the corrector only on base model outputs with a standard denoising loss, whereas FPCCT introduces a new loss term that explicitly enforces a distributional fixed-point across iterative applications, turning the corrector into a contraction operator in distribution space. To address the gap between distributional invariance and contraction, we add a per-sample residual loss L_res and a Lipschitz regularizer L_lip that directly encourage the corrector to be a strict contraction, and we verify the contraction modulus empirically.

**Why it measures what we claim** — The fixed-point consistency loss L_fp measures the distributional stability of the corrector across refinement steps because the Jensen-Shannon divergence between the output distributions of one and two applications is zero if and only if the corrector's output distribution is invariant under further application; this equivalence relies on the assumption that the discriminator D_ψ is trained to optimality at each iteration, which holds in the limit of infinite discriminator capacity and training. This assumption fails when the discriminator overfits to finite samples, in which case L_fp reflects the discriminator's ability to distinguish the two distributions rather than their true divergence, potentially underestimating the drift. The denoising loss L_denoise measures the corrector's ability to recover the clean image from the base model's prediction, operationalizing the concept of refinement accuracy; this assumes that the base model's prediction is a sufficient statistic for the clean image given the noisy input, which fails when the base model has systematic bias that the corrector cannot correct from the input alone, in which case L_denoise reflects the corrector's capacity to memorize the training set rather than generalize. The per-sample residual L_res measures the corrector's contraction at the sample level because it penalizes the distance between x_hat_2 and x_hat_1; this assumes that the corrector map is Lipschitz continuous with constant L, so that ||C(x)-C(y)|| ≤ L||x-y||; this assumption fails when the corrector is expansive (L>1), in which case L_res may be small for some samples but the map does not contract globally. The Lipschitz regularizer L_lip measures the spectral norm of the corrector's Jacobian because it penalizes values above 1; this assumes that the spectral norm is a good proxy for the Lipschitz constant, which holds for differentiable maps; this assumption fails when the corrector has large higher-order derivatives, in which case the spectral norm underestimates the true Lipschitz constant.

## Contribution

['A novel training objective for diffusion correctors that enforces a distributional fixed-point property via an adversarial consistency loss, turning the corrector into a contraction operator in distribution space.', 'A concrete design principle: the fixed-point consistency loss can be implemented with a lightweight discriminator and a two-step unroll, adding minimal training overhead while preventing distributional drift during iterative refinement.', 'An empirical demonstration that FPCCT improves sample quality (FID) over the original correction training method on CIFAR-10 and ImageNet, with ablation studies showing the necessity of the consistency loss and the two-step unroll.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CIFAR-10 (32×32) | Standard benchmark with strong pretrained base diffusion models |
| Primary metric | FID at fixed low sampling steps (T=10, 20, 50) | Directly measures sample quality gain per compute |
| Secondary metric | Contraction modulus: average ||x_hat_{k+1} - x_hat_k|| / ||x_hat_k - x_hat_{k-1}|| for k=1..4 on 1024 held-out samples | Verifies contraction property empirically |
| Secondary metric | Lipschitz constant estimate: spectral norm of corrector Jacobian via power iteration on 1024 samples | Checks if corrector is contractive |
| Baseline | Base diffusion model, no corrector | Isolates whether correction adds anything at all |
| Baseline | Standard correction training (L2 only) | Tests whether fixed-point loss beats plain denoising |
| Baseline | Extra sampling steps (same compute budget) | Tests whether gains are just compute reallocation |
| Ablation-of-ours | FPCCT with λ_fp=0 (denoise loss only) | Isolates the fixed-point consistency term's contribution |
| Ablation-of-ours | FPCCT with λ_res=0 and λ_lip=0 (only JS loss) | Tests whether contraction regularizers are necessary |
| Ablation-of-ours | FPCCT with λ_fp=0, λ_res=0, λ_lip=0 (denoise only) | Same as λ=0 ablation, but with all terms off |

### Why this setup validates the claim

FPCCT's central claim is that enforcing a distributional fixed-point across iterative corrector applications produces a contraction operator that yields better samples per unit compute than either an uncorrected base model or a corrector trained only with denoising loss. The no-corrector baseline tests the sub-claim that correction helps at all; the standard correction baseline tests the sub-claim that the fixed-point term, not correction per se, drives the gain; the extra-sampling-steps baseline tests whether any gain is just compute reallocation rather than a genuinely better refinement operator. The λ_fp=0 ablation is the sharpest test: it shares architecture, data, and optimizer with FPCCT and differs only in the loss term, so any FID gap is attributable to the fixed-point consistency mechanism. The additional ablations with λ_res=0 and λ_lip=0 test whether the contraction regularizers are necessary to achieve the claimed contraction property. FID at fixed low sampling steps is the right metric because the claimed effect is a per-step quality improvement; a metric that measured only asymptotic sample quality would not distinguish a contraction operator from a slow-but-eventually-fine one. The contraction modulus and Lipschitz constant metrics directly verify the load-bearing assumption that the corrector is a strict contraction. Falsifiability hinges on the gain being concentrated where the base model's per-step prediction error is largest, not uniform, and on the contraction modulus being <1.

### Expected outcome and causal chain

**vs. Base diffusion model, no corrector** — On a case where the base model at low step counts produces over-smoothed or misaligned clean-image predictions (e.g., blurry texture regions and drifted object boundaries), the baseline produces samples with elevated FID because it has no mechanism to refine its own prediction beyond the learned denoiser. Our method instead applies the corrector twice at each step, pulling the prediction toward a distributional fixed point that the base model's bias cannot reach, so we expect a clear FID improvement at low step counts that shrinks as step count increases (since the base model's per-step error diminishes).

**vs. Standard correction training (L2 only)** — On a case where the corrector's single application improves a prediction but a second application would drift it further (e.g., sharpening artifacts that compound), the baseline produces inconsistent outputs across refinement iterations because its L2 objective does not constrain behavior under self-application. Our method instead penalizes divergence between one and two applications via the discriminator and adds a per-sample residual and Lipschitz regularizer, so the corrector behaves as a contraction in distribution space; we expect a modest FID gain over L2 correction overall, but a much larger gain on samples that received multiple refinement passes, and visibly more stable outputs as K grows. We also expect the contraction modulus to be <1 for FPCCT and ≥1 for L2 correction.

**vs. Extra sampling steps (same compute budget)** — On a case where the base model's error is systematic bias rather than variance (e.g., a consistent color shift or structural drift that more steps cannot average away), the baseline produces samples whose FID plateaus despite additional steps because extra integration does not correct a biased denoiser. Our method instead uses the same compute to apply a learned contraction that targets the bias directly, so we expect the FPCCT curve to sit below the extra-steps curve at matched compute, with the gap widest on the subset of samples where the base model's per-step error is largest.

### What would falsify this idea

If the FID gain over standard correction training is uniform across samples rather than concentrated on those where the base model's per-step error is largest, or if the λ_fp=0 ablation matches FPCCT's FID, or if the estimated contraction modulus is not consistently below 1, then the fixed-point consistency mechanism is not doing the causal work and the central claim is wrong.

## References

1. Correction Training for Diffusion Models
