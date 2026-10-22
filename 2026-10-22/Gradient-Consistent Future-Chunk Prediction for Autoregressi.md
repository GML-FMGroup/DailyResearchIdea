# Gradient-Consistent Future-Chunk Prediction for Autoregressive Video Diffusion

## Motivation

Jointly optimizing future-chunk prediction (FCP) with teacher-forced denoising, as in Looking Ahead, lacks any guarantee that the auxiliary gradient does not degrade the primary denoising objective. Because both losses share the same residual stream and parameters, FCP gradients can have negative cosine similarity with denoising gradients, causing immediate generation fidelity to drop even when long-horizon evolution modeling improves. This structural conflict is invisible to loss-weight tuning and recurs across all branches of the shared residual gap.

## Key Insight

The auxiliary FCP gradient can be projected onto the non-conflicting half-space of the primary denoising gradient at every optimization step, guaranteeing monotone improvement of the primary loss while retaining the evolution-modeling signal.

## Method

# GC-FCP: Gradient-Consistent Future-Chunk Prediction
# Inputs: video chunks x_{1:T}, context chunk x_t, future chunk x_{t+k} (k>=1)
# Model: autoregressive video diffusion transformer f_theta
# Hyperparams: lambda (FCP weight, default 0.1), alpha (projection threshold, default 0.0), beta (trust-region cap, default 0.5), ema_decay (gradient EMA decay, default 0.9)

# Load-bearing assumption (explicit): The component of the FCP gradient orthogonal to the denoising gradient carries a useful, non-conflicting evolution-modeling signal. This holds when the FCP and denoising gradients have negligible shared low-rank subspace, so orthogonality implies task-relevant information. It fails when shared low-rank structure makes the orthogonal component mostly noise.

# Initialize EMA of gradients: g_denoise_ema = 0, g_fcp_ema = 0

for each training step:
    # 1. Primary teacher-forced denoising loss
    eps ~ N(0,I); x_t_noisy = sqrt(alpha_bar)*x_t + sqrt(1-alpha_bar)*eps
    L_denoise = || eps - f_theta(x_t_noisy, context=x_{<t}) ||^2
    g_denoise = grad(L_denoise, theta)

    # 2. Auxiliary FCP loss (same model, no intervening chunks)
    eps_f ~ N(0,I); x_{t+k}_noisy = sqrt(alpha_bar)*x_{t+k} + sqrt(1-alpha_bar)*eps_f
    L_fcp = || eps_f - f_theta(x_{t+k}_noisy, context=x_{<t}) ||^2
    g_fcp = grad(L_fcp, theta)

    # 3. Update gradient EMAs to reduce stochastic noise
    g_denoise_ema = ema_decay * g_denoise_ema + (1 - ema_decay) * g_denoise
    g_fcp_ema = ema_decay * g_fcp_ema + (1 - ema_decay) * g_fcp

    # 4. Gradient consistency projection using EMA gradients
    cos_sim = dot(g_fcp_ema, g_denoise_ema) / (||g_fcp_ema|| * ||g_denoise_ema|| + 1e-8)
    if cos_sim < alpha:
        g_fcp_proj = g_fcp - (dot(g_fcp, g_denoise_ema) / (||g_denoise_ema||^2 + 1e-8)) * g_denoise_ema
        # g_fcp_proj is orthogonal to g_denoise_ema, so cos_sim becomes 0 (>= alpha)
    else:
        g_fcp_proj = g_fcp

    # 5. Trust-region cap on FCP update magnitude
    fcp_update = lambda * g_fcp_proj
    denoise_norm = ||g_denoise|| + 1e-8
    if ||fcp_update|| > beta * denoise_norm:
        fcp_update = fcp_update * (beta * denoise_norm / (||fcp_update|| + 1e-8))

    # 6. Combined update
    theta = theta - lr * (g_denoise + fcp_update)

# Diagnostic (logged every 100 steps): estimate principal angle between gradient subspaces
# Collect top-5 singular vectors of g_denoise_ema and g_fcp_ema via power iteration on their outer products.
# Compute principal angles and log; also log correlation between orthogonal component and held-out future-chunk loss reduction.

**Why this design** — We chose projected gradient over a Lagrangian multiplier because projection gives a closed-form, per-step guarantee that the combined update has non-negative cosine similarity with the denoising gradient, whereas a Lagrangian penalty only asymptotically discourages conflict and requires tuning a dual variable. We chose to project only when cosine similarity falls below a threshold α (default 0) rather than always projecting, because unconditional projection would discard the component of FCP that is already consistent with denoising, wasting useful evolution signal. We chose to compute the projection in parameter space rather than in a shared latent space because the conflict manifests in the update direction that directly affects the denoising loss; parameter-space projection is exact and adds only one inner product and one vector subtraction per step, accepting the cost of an extra backward pass through the FCP loss (which is already computed in Looking Ahead). We chose a fixed λ rather than an adaptive schedule because the projection already removes the harmful component, so λ only scales the remaining orthogonal signal; this avoids introducing a second hyperparameter that would need tuning. We added a trust-region cap β to ensure the first-order guarantee degrades gracefully under non-convexity and gradient noise, and we use EMA gradients to reduce stochastic noise in cosine similarity estimation. A domain expert would not describe this as a variant of Diffusion Forcing or Masked Diffusion Transformer because those methods change the noise schedule or masking pattern, whereas GC-FCP leaves the forward process and architecture untouched and instead operates on the gradient geometry of two losses at the same parameters.

**Why it measures what we claim** — The cosine similarity between g_fcp and g_denoise measures gradient conflict because we assume that negative cosine similarity implies the FCP update would increase the denoising loss to first order; this assumption fails when the loss landscape is highly non-convex and first-order linearization is inaccurate over the step size, in which case the projection may not prevent a transient increase in L_denoise. The projected gradient g_fcp_proj measures the evolution-modeling signal that is orthogonal to denoising because we assume that the orthogonal component carries information about future-chunk prediction that does not interfere with immediate denoising; this assumption fails when the FCP task and denoising task share a common low-rank structure such that the orthogonal component is mostly noise, in which case the projection removes both conflict and useful signal. The threshold α measures the allowed tolerance for conflict because we assume that α=0 is sufficient to guarantee monotone improvement; this assumption fails when stochastic gradient noise makes the estimated cosine similarity unreliable, in which case a small positive α (e.g., 0.1) is needed to avoid over-projecting. The trust-region cap β measures the allowed magnitude of the FCP update relative to the denoising update because we assume that a bounded step preserves the first-order guarantee; this assumption fails when the FCP update is large and the linearization is inaccurate, in which case the cap limits potential harm.

## Contribution

(1) A gradient-consistency projection mechanism (GC-FCP) that enforces non-negative cosine similarity between the auxiliary FCP gradient and the primary denoising gradient at every step, providing a per-step monotone improvement guarantee for the denoising loss without changing the model architecture or inference procedure. (2) A design principle that auxiliary training objectives for autoregressive video generation should be optimized under a gradient conflict constraint rather than a fixed loss weight, which we show empirically prevents the fidelity degradation observed in Looking Ahead when FCP is naively added. (3) An analysis of the cosine similarity dynamics during training that identifies when FCP gradients conflict with denoising gradients, enabling practitioners to diagnose and correct auxiliary-objective interference in video diffusion models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Something-Something V2 (video prediction) | Long-horizon physical interactions stress future-chunk modeling |
| Primary metric | Physical plausibility score (human/CLIP) | Directly measures future evolution fidelity |
| Baseline | Standard teacher-forced AR diffusion | Tests whether FCP adds beyond standard training |
| Baseline | Diffusion Forcing | Tests whether gradient projection beats noise-schedule changes |
| Baseline | Looking Ahead (unprojected FCP) | Tests whether projection is necessary vs. naive auxiliary loss |
| Ablation-of-ours | GC-FCP with always-project (α=1) | Tests whether conditional projection preserves useful signal |
| Ablation-of-ours | GC-FCP without trust-region cap (β=∞) | Tests whether cap is necessary for stability |
| Ablation-of-ours | GC-FCP without EMA (ema_decay=0) | Tests whether EMA reduces noise in cosine estimation |
| Diagnostic | Principal angle between gradient subspaces | Verifies load-bearing assumption about orthogonal signal |

### Why this setup validates the claim

This design isolates the central claim that gradient projection of the FCP loss prevents interference with denoising while retaining useful future-evolution signal. The standard teacher-forced AR diffusion baseline tests the sub-claim that FCP contributes anything at all: if GC-FCP does not beat it, the auxiliary loss is useless. Diffusion Forcing tests the sub-claim that our parameter-space gradient projection is a better conflict-resolution mechanism than changing the forward process or noise schedule; if Diffusion Forcing matches or exceeds GC-FCP, our mechanism is unnecessary. The unprojected Looking Ahead baseline tests the sub-claim that naive FCP causes gradient conflict and degrades denoising: if unprojected FCP performs as well as GC-FCP, projection is not needed. The always-project ablation (α=1) tests the sub-claim that conditional projection preserves useful signal: if it matches GC-FCP, the threshold logic is pointless. The no-trust-region and no-EMA ablations test the sub-claims that the cap and EMA are necessary for stability and noise reduction. The principal angle diagnostic directly tests the load-bearing assumption: if the principal angle between gradient subspaces is small (high overlap), the orthogonal component is likely noise, and projection may remove useful signal. Physical plausibility is the right metric because the predicted failure mode of unprojected FCP is not blurry images but physically implausible futures (objects teleporting, violating contact), which standard FID or FVD may not capture. A falsifiable test requires that the gain be concentrated on physically demanding subsets (e.g., object permanence, collision) and absent on static or trivial subsets.

### Expected outcome and causal chain

**vs. Standard teacher-forced AR diffusion** — On a case where an object disappears behind an occluder and must reappear at a predictable future location, the baseline produces a physically implausible future (object vanishes or teleports) because its training signal never explicitly models multi-step future evolution beyond the immediate next chunk. Our method instead adds a future-chunk prediction loss that encourages the model to encode long-horizon dynamics, so we expect a noticeable improvement in physical plausibility on occlusion and contact subsets, with parity on static scenes.

**vs. Diffusion Forcing** — On a case requiring long-horizon rollout with variable-length context, Diffusion Forcing may produce stable but overly smooth futures that lack fine-grained physical detail because its noise schedule is tuned for full-sequence diffusion rather than autoregressive chunk prediction. Our method leaves the forward process untouched and instead resolves gradient conflict between denoising and future prediction, so we expect GC-FCP to show better physical detail (e.g., correct object trajectories) on long rollouts while matching Diffusion Forcing on short-horizon stability.

**vs. Looking Ahead (unprojected FCP)** — On a case where the future-chunk prediction gradient conflicts with the denoising gradient (e.g., early training when the model is fitting local textures), the unprojected baseline produces degraded denoising quality and blurry immediate next chunks because the conflicting gradient update increases the denoising loss. Our method projects out the conflicting component, so we expect unprojected FCP to underperform even the standard baseline on immediate-chunk fidelity, while GC-FCP matches or exceeds the standard baseline on immediate chunks and improves on future chunks.

**vs. GC-FCP with always-project (α=1)** — On a case where the FCP gradient is already consistent with denoising (e.g., late training on simple motion), always projecting would discard the useful aligned component, slowing learning of future dynamics. Our method projects only when cosine similarity is below α, so we expect the always-project ablation to show slower convergence and worse final physical plausibility on simple-motion subsets, while GC-FCP retains the full benefit.

**vs. GC-FCP without trust-region cap (β=∞)** — On a case where the FCP gradient is large and the linearization is inaccurate (e.g., early training with high learning rate), the uncapped update may cause a transient increase in L_denoise despite projection. Our method caps the FCP update magnitude, so we expect the uncapped variant to show higher variance in L_denoise and worse final performance, while GC-FCP remains stable.

**vs. GC-FCP without EMA (ema_decay=0)** — On a case where minibatch gradient noise makes cosine similarity unreliable, the no-EMA variant may over-project or under-project. Our method uses EMA to smooth the gradients, so we expect the no-EMA variant to have noisier cosine estimates and slightly worse performance, while GC-FCP benefits from reduced noise.

### What would falsify this idea

If GC-FCP's improvement over the unprojected Looking Ahead baseline is uniform across all subsets rather than concentrated on cases where gradient conflict is predicted (early training, complex physical interactions), then the projection mechanism is not the cause of any gain. Additionally, if the always-project ablation (α=1) matches GC-FCP on all subsets, the conditional threshold logic is unnecessary and the central claim about preserving useful signal is wrong. Furthermore, if the principal angle diagnostic shows high overlap between gradient subspaces (small angles) and the orthogonal component does not correlate with held-out future-chunk loss reduction, then the load-bearing assumption fails and the projection may be removing useful signal.

## References

1. Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation
2. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
3. Masked Diffusion Transformer is a Strong Image Synthesizer
4. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
5. Diffusion-LM Improves Controllable Text Generation
6. Continuous diffusion for categorical data
7. DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models
8. Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise
