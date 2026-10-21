# Guidance-Consistent Corrector Training for Diffusion Sampling

## Motivation

Correction Training for Diffusion Models trains a corrector to refine the base denoiser's clean-sample prediction using an unconditional denoising loss, which ignores the guidance weight applied during sampling. Consequently, when classifier-free guidance is used, the corrector's iterative refinement can drift from the conditional mode because it was never trained to preserve the guided conditional score; the corrector's output is not invariant to guidance-weight variation, and this structural gap persists across guidance scales.

## Key Insight

The corrector's output can be made invariant to guidance-weight variation by training it with a consistency loss that matches its composed output to the conditional score evaluated at the same guidance weight, so iterative refinement preserves the guided mode rather than drifting from it.

## Method

**What it is** — We introduce GCC (Guidance-Consistent Corrector), a corrector network trained with a conditional-consistency loss. Its inputs are the base denoiser's clean-sample prediction, the current noise level, and the guidance weight w; its output is a refined clean-sample prediction that is consistent with the conditional score at weight w. The corrector is a 2-layer MLP with hidden dimension 256 and GeLU activation, taking as input the concatenation of x0_hat (3×256×256), a sinusoidal time embedding of t (dim 128), a class embedding of c (dim 128), and a scalar w embedded via a 2-layer MLP (dim 64); output is a 3×256×256 tensor. The base denoiser is the frozen EDM2-XXL model pretrained on ImageNet 256×256 (class-conditional).

**How it works** —
```pseudocode
# Training (base model frozen)
for each batch (x0, c) and noise level t ~ Uniform[0, T]:
    eps_hat = base_denoiser(x_t, t, c)              # unconditional base output
    x0_hat = (x_t - sqrt(1-alpha_t)*eps_hat)/sqrt(alpha_t)
    w ~ Uniform{1, 2, 4, 8}                         # discrete set, equal probability
    x0_corr = corrector(x0_hat, t, c, w)            # w-conditioned corrector
    # standard denoising loss on corrector output
    L_denoise = || x0_corr - x0 ||^2
    # conditional-consistency loss: composed corrector must match conditional score at w
    eps_corr = (x_t - sqrt(alpha_t)*x0_corr)/sqrt(1-alpha_t)
    eps_cond_w = (1+w)*eps_cond(x_t, t, c) - w*eps_uncond(x_t, t)  # conditional score at w
    L_cons = || eps_corr - eps_cond_w ||^2
    # w-smoothness regularizer (replaces w-invariance): penalize deviation from eps_cond_w at a perturbed w
    w2 = w + delta, delta ~ Uniform[-0.5, 0.5]       # neighborhood of w
    eps_cond_w2 = (1+w2)*eps_cond(x_t, t, c) - w2*eps_uncond(x_t, t)
    x0_corr2 = corrector(x0_hat, t, c, w2)
    eps_corr2 = (x_t - sqrt(alpha_t)*x0_corr2)/sqrt(1-alpha_t)
    L_reg = || eps_corr2 - eps_cond_w2 ||^2         # smoothness in w, not invariance
    L = L_denoise + lambda_cons * L_cons + lambda_reg * L_reg
    # hyperparameters: lambda_cons = 1.0, lambda_reg = 0.1

# Sampling
for t in schedule (DDIM, 50 steps):
    x0_hat = base_denoiser(x_t, t, c)
    for k in range(K=2):
        x0_hat = corrector(x0_hat, t, c, w)
    x_{t-1} = DDIM_step(x_t, x0_hat, t)
```

**Why this design** — We chose to train the corrector with a conditional-consistency loss (L_cons) rather than only a denoising loss because the denoising loss alone (as in Correction Training for Diffusion Models) does not constrain the corrector to preserve the guided score; this adds a direct penalty for drift from the conditional mode, accepting the cost that the corrector must now be conditioned on w and c. We chose to sample w from a discrete set {1,2,4,8} and apply a smoothness regularizer across weights (L_reg) rather than training at a single fixed w because the corrector must generalize across guidance scales used at sampling; this adds variance to training but enforces smooth tracking of the w-dependent guided target. We chose to use the base model's unconditional and conditional score estimates to compute the target eps_cond_w rather than a separate guided model because the base model is frozen and its scores are the ground truth for the corrector; this ties the corrector to the base model's behavior, accepting that any base-model error in the score is inherited. A domain expert would not describe this as a variant of Correction Training for Diffusion Models because that method trains the corrector unconditionally with a standard denoising loss, whereas our method adds a conditional-consistency term that explicitly enforces the corrector's output to match the guided conditional score, a signal absent in the prior work.

**Why this measures what we claim** — The conditional-consistency loss L_cons measures the corrector's fidelity to the guided conditional score because we assume that the base model's conditional and unconditional score estimates are accurate enough that their linear combination at weight w equals the true guided score; this assumption fails when the base model is poorly trained or when w is very large, in which case L_cons reflects the base model's score error rather than the corrector's drift. The smoothness regularizer L_reg measures the corrector's ability to track the w-dependent guided target smoothly because we assume that the corrector's output should vary smoothly with w (Lipschitz continuity) rather than being invariant; this assumption fails when the guided target changes abruptly with w (e.g., near mode collapse), in which case L_reg may over-smooth legitimate rapid changes. The denoising loss L_denoise measures sample quality because we assume the ground-truth clean image is the target; this assumption fails when the base model's prediction is far from the true clean image, in which case L_denoise may push the corrector toward the base model's error rather than the truth.

## Contribution

["A guidance-consistent corrector training framework that introduces a conditional-consistency loss and a cross-weight consistency regularizer, making the corrector's iterative refinement invariant to guidance-weight variation.", 'An empirical design principle: conditioning the corrector on the guidance weight and training with a consistency loss across multiple weights reduces drift from the conditional mode, as measured by improved sample quality and mode preservation under high guidance.', 'A training procedure that requires no modification to the base diffusion model and can be applied post-hoc to any pretrained guided diffusion sampler.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet 256×256 | Standard benchmark with diverse conditional modes. |
| Primary metric | FID | Captures distributional fidelity to conditional data. |
| Baseline | Base diffusion + CFG | Tests need for any corrector at all. |
| Baseline | Correction Training (unconditional corrector) | Tests value of conditional-consistency loss. |
| Baseline | CFG with 2× sampling steps | Tests whether gains are just extra compute. |
| Ablation-of-ours | GCC without L_cons | Isolates contribution of conditional-consistency term. |
| Ablation-of-ours | GCC with L_reg replaced by w-invariance penalty | Tests whether smoothness vs invariance matters. |
| Diagnostic | Base-model score error vs. FID gain | Correlates L_cons effect with base error. |

### Why this setup validates the claim

The central claim is that a corrector trained with a conditional-consistency loss (L_cons) will produce refined clean-sample predictions that stay on the guided conditional mode, yielding better FID than an unconditional corrector and than simply spending more sampling compute. The base diffusion + CFG baseline tests whether any correction is needed: if GCC does not beat it, the corrector adds nothing. The unconditional Correction Training baseline tests the specific sub-claim that L_cons is necessary: it uses the same corrector architecture and denoising loss but lacks the conditional-consistency signal, so any gap between it and GCC isolates the effect of L_cons. The 2× sampling-steps baseline tests whether GCC's gain is merely from extra refinement compute; if GCC beats it at equal or lower cost, the gain is due to the learned corrector, not step count. The ablation GCC without L_cons directly tests the causal role of the consistency term. The ablation GCC with L_reg replaced by w-invariance penalty tests whether the smoothness regularizer (Lipschitz continuity in w) is preferable to an invariance penalty, which would contradict w-conditioning. The diagnostic experiment measures base-model score error (via Tweedie's formula residual on a held-out set of 512 images at each w) and correlates it with the observed FID gain, grounding the assumption that L_cons corrects base-score error. FID is the right metric because the predicted failure mode is drift away from the conditional mode, which manifests as distributional mismatch; FID is sensitive to exactly that. A uniform FID gain across all guidance weights would not support the mechanism, but a gain concentrated at high w (where score drift is largest) would.

### Expected outcome and causal chain

**vs. Base diffusion + CFG** — On a case where w=8 and the conditional score is sharp, the base CFG sampler produces over-saturated, mode-collapsed samples because it moves along the guided score without correcting discretization and score-estimation errors. Our method instead refines the clean-sample prediction with a corrector trained to match the guided score at that w, so we expect a noticeable FID improvement at high w (e.g., 5–10% relative) but near parity at w=1 where the guided score equals the conditional score and drift is minimal.

**vs. Correction Training (unconditional corrector)** — On a case where the unconditional corrector is applied at high w, it produces samples that drift toward the unconditional mode because its denoising loss never sees the guidance weight and thus cannot enforce consistency with the guided score. Our method instead conditions the corrector on w and penalizes deviation from eps_cond_w, so we expect a clear FID gap between GCC and the unconditional corrector that grows with w (e.g., small at w=1, large at w=8), with GCC winning on the high-w subset.

**vs. CFG with 2× sampling steps** — On a case where extra sampling steps are used to reduce discretization error, the baseline still accumulates score-estimation error because more steps do not correct a biased score; it produces samples with residual artifacts and mode drift at high w. Our method instead learns a correction that targets the guided score directly, so we expect GCC to achieve lower FID than the 2×-step baseline at comparable or lower compute, with the advantage concentrated at high w where score bias dominates over discretization error.

### What would falsify this idea

If GCC's FID gain over the unconditional corrector is uniform across guidance weights rather than concentrated at high w, then the conditional-consistency loss is not doing what we claim. If GCC without L_cons matches GCC, the central mechanism is unnecessary. If the diagnostic shows no correlation between base-model score error and FID gain, the assumption that L_cons corrects base-score error is unsupported.

## References

1. Correction Training for Diffusion Models
