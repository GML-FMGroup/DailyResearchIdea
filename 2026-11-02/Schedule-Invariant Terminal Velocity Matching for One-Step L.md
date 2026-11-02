# Schedule-Invariant Terminal Velocity Matching for One-Step Latent Low-Light Enhancement

## Motivation

GLOW-FM's Terminal Velocity Matching trains a one-step flow under a single fixed interpolant schedule and never verifies that the learned terminal velocity is independent of that schedule; when low-light and normal-light latents are far apart in the frozen Tiny-VAE latent space, the fixed-schedule velocity is only one of many admissible paths, and a single-function-evaluation model that has memorized one schedule's terminal direction can produce off-manifold latents at inference. This is the local instance of a broader structural gap: flow-matching objectives in the stochastic-interpolant family (Maoutsa & Opper's Building Normalizing Flows with Stochastic Interpolants) are known to depend on the choice of interpolant, yet one-step enhancement methods inherit that dependence silently and never enforce trajectory-independence of the terminal velocity field. The root cause is that schedule identity is an unconstrained nuisance variable that the objective never marginalizes over or holds invariant, so the flow is free to encode schedule-specific artifacts into a quantity that must generalize across the latent gap.

## Key Insight

If the terminal velocity is required to be identical at the same latent state across randomly sampled interpolant schedules, then schedule identity becomes an unidentifiable nuisance variable and the one-step map is forced to approximate the schedule-invariant component of the transport field, which is exactly the component that remains well-defined when endpoints are far apart.

## Method

**What it is.** SITVM (Schedule-Invariant Terminal Velocity Matching) is a training objective for GLOW-FM's one-step latent flow. It takes a low-light latent $z_0$, a paired normal-light latent $z_1$ from the frozen Tiny-VAE, and a schedule index $s$ sampled per example, and returns a velocity-network gradient that couples the terminal-velocity regression loss with a cross-schedule endpoint-consistency penalty. The load-bearing assumption, stated plainly: the one-step endpoint map $\hat{z}_1(s) = z_0 + v_\theta(z_0, t{=}1, z_0)$ must be identical across interpolant schedules $s$, and we enforce this by penalizing disagreement between the endpoints produced under two different schedules for the same $(z_0, z_1)$ pair. We do NOT claim that the pointwise velocity field is schedule-invariant; only the terminal map is required to be.

**How it works.**

```pseudocode
# Per training step, batch of (z0, z1) latent pairs from frozen Tiny-VAE
# VelocityNet: 4-layer MLP, hidden=512, GeLU, input dim = 2*d_z + 1 (z_t, z0, t)
# d_z = 4 (Tiny-VAE latent channels) * 32 * 32 = 4096; so input dim = 8193
for each example i in batch:
    s_i ~ Uniform{linear, cosine, variance-preserving}   # schedule family S, |S|=3
    t_i ~ Uniform(0, 1)                                    # interpolation time
    alpha, dalpha = schedule_coeffs(s_i, t_i)             # alpha_t, d alpha_t / dt
    z_t = alpha * z1 + (1 - alpha) * z0                    # interpolant state
    u_target = dalpha * (z1 - z0)                          # instantaneous velocity
    v_pred = VelocityNet(z_t, t_i, z0)                     # conditioned on low-light latent
    L_tvm += || v_pred - u_target ||^2                     # standard velocity regression

    # --- endpoint-consistency branch: same (z0, z1), a DIFFERENT schedule ---
    s'_i ~ Uniform(S \ {s_i})                              # forced schedule swap
    alpha', dalpha' = schedule_coeffs(s'_i, t_i)
    z_t' = alpha' * z1 + (1 - alpha') * z0                 # different interpolant state
    v_pred' = VelocityNet(z_t', t_i, z0)
    # one-step endpoint under each schedule (Euler step from t_i to 1)
    # using the instantaneous velocity scaled by (1 - t_i)
    z1_hat_s  = z_t  + (1 - t_i) * v_pred
    z1_hat_s' = z_t' + (1 - t_i) * v_pred'
    L_inv += || z1_hat_s - z1_hat_s' ||^2                  # endpoint-map agreement

L = L_tvm + lambda_inv * L_inv        # lambda_inv = 0.1, annealed 0 -> 0.1 over first 5k steps
```

The invariance branch compares the one-step endpoints produced by two different schedules, not the chord-projected scalars of the velocities at mismatched states. This is because the velocity field is defined relative to a chosen interpolant (Maoutsa & Opper; Albergo & Vanden-Eijnden), so pointwise velocity agreement across schedules is not a well-posed requirement; the terminal map, however, must be schedule-independent for a one-step method to be well-defined. The endpoint-consistency loss directly penalizes disagreement in the quantity that is actually used at inference. At inference the model is run once at $t=1$ with the linear schedule, exactly as GLOW-FM.

**Why this design.** We chose a discrete three-element schedule family over a continuous schedule-parameter distribution because the invariance penalty needs a well-defined "swap" partner per example and a finite family gives an unbiased, low-variance estimate of the cross-schedule disagreement with only one extra forward pass; the cost is that schedules outside the family are not explicitly regularized, which we accept because the three span the interpolation-curvature range used in practice. We chose endpoint-consistency over pointwise velocity matching because the velocity field is schedule-dependent by construction, and forcing pointwise agreement would either be ill-posed (comparing velocities at different states) or collapse the interpolant family; the cost is that the penalty requires a one-step Euler approximation to the endpoint, which introduces discretization error that we mitigate by using the exact instantaneous velocity scaled by the remaining time. We chose to condition the velocity network on $z_0$ rather than on schedule identity because feeding the schedule as an input would let the network satisfy the invariance loss trivially by ignoring the input, whereas leaving it out forces the shared weights to average over schedules; the cost is a slightly harder optimization landscape early in training, mitigated by the annealed $\lambda_{inv}$. This differs from Maoutsa & Opper's stochastic-interpolant velocity loss, which regresses velocity under one chosen interpolant and treats the interpolant as a fixed design choice rather than a nuisance variable to be marginalized; a domain expert would not describe SITVM as a variant of that loss because the endpoint-consistency term is a cross-schedule coupling that the single-interpolant objective structurally cannot express.

**Why it measures what we claim.** The endpoint-consistency loss $\| \hat{z}_1(s) - \hat{z}_1(s') \|^2$ measures schedule-invariance of the one-step terminal map because the assumption is that for a fixed $(z_0, z_1)$ pair, any admissible interpolant must yield the same endpoint when integrated to $t=1$; this assumption fails when the one-step Euler approximation is inaccurate (e.g., for large $|1 - t_i|$ or highly curved paths), in which case the penalty reflects discretization error rather than genuine schedule dependence. The per-example schedule swap measures the nuisance-variable marginalization the motivation calls for because the assumption is that $s$ is independent of $(z_0, z_1)$ given the batch, so swapping $s$ is a valid intervention on the nuisance; this assumption fails when a schedule systematically pairs with a particular latent-gap magnitude, in which case the penalty reflects a spurious schedule-gap correlation instead of invariance. The annealed weight $\lambda_{inv}$ measures the trade-off between fitting the terminal velocity and enforcing invariance because the assumption is that early training benefits from unconstrained velocity fitting before the invariance constraint is binding; this assumption fails when the base velocity is already schedule-dependent at initialization, in which case the anneal delays rather than prevents schedule leakage.

## Contribution

(1) A schedule-randomized terminal-velocity matching objective with a chord-projected cross-schedule invariance regularizer that makes the one-step latent flow's endpoint map independent of the interpolant schedule without adding inference cost. (2) The design principle that schedule identity should be treated as a nuisance variable to be marginalized in one-step flow-matching enhancement, with the chord projection identified as the minimal sufficient statistic for cross-schedule agreement. (3) A schedule-swap intervention protocol for auditing schedule dependence in one-step latent enhancement models, usable as a diagnostic independent of the proposed training objective.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LOL-v2 real + synthetic | Paired latents, real low-light distribution |
| Primary metric | PSNR on one-step output | Direct measure of terminal map accuracy |
| Baseline | GLOW-FM (single-schedule TVM) | Tests whether schedule marginalization helps |
| Baseline | Random-schedule TVM (no invariance term) | Tests whether invariance penalty adds beyond randomization |
| Baseline | Stochastic-interpolant velocity loss | Tests cross-schedule coupling vs single-interpolant |
| Baseline | Diffusion-based LLIE (e.g. DiffLL) | Tests one-step efficiency vs multi-step quality |
| Ablation-of-ours | SITVM without L_inv (λ=0) | Isolates invariance penalty contribution |
| Ablation-of-ours | SITVM with pointwise chord-projection penalty (original) | Tests endpoint-consistency vs chord-projection |

### Why this setup validates the claim

The central claim is that SITVM's cross-schedule endpoint-consistency penalty produces a one-step endpoint map that is more accurate than a single-schedule velocity loss, specifically by reducing schedule-dependent leakage in the terminal direction. LOL-v2 real and synthetic provide paired latents across a range of low-light degradations, so the schedule-gap correlation failure mode can actually manifest. The primary metric PSNR on the one-step output directly measures endpoint-map accuracy, which is exactly what the endpoint-consistency term is designed to improve; a perceptual metric would conflate tangential path-curvature differences that the method deliberately leaves unconstrained. GLOW-FM with a single schedule tests whether the schedule randomization and invariance coupling add anything beyond the base one-step latent flow. The random-schedule TVM baseline (schedule randomization but no invariance term) is critical to isolate the effect of the invariance penalty from mere schedule randomization, which is already implicit in the stochastic-interpolant family. The stochastic-interpolant velocity loss tests whether the cross-schedule coupling is necessary versus simply regressing velocity under a randomly chosen schedule. The diffusion baseline tests whether the one-step efficiency claim holds against a strong multi-step generative method. The λ=0 ablation is the critical control: if SITVM's gain is due to the invariance penalty, removing it should reproduce the single-schedule baseline's error pattern, not merely a uniformly worse version. The pointwise chord-projection ablation tests whether the endpoint-consistency formulation is necessary versus the original chord-projection penalty, which the adversarial alert identified as potentially ill-posed due to mismatched states.

### Expected outcome and causal chain

**vs. GLOW-FM (single-schedule TVM)** — On a case where a low-light latent has a large gap to its normal-light pair, the single-schedule baseline produces a terminal velocity whose chord projection is tuned to the linear interpolant's curvature, so the one-step endpoint overshoots or undershoots depending on the local gap magnitude. Our method instead averages the endpoint map over three schedules via the endpoint-consistency penalty, because the shared weights cannot exploit schedule identity, so we expect a noticeable PSNR gap concentrated on large-gap pairs and near-parity on small-gap pairs.

**vs. Random-schedule TVM (no invariance term)** — On a case where the randomly drawn schedule places z_t in a region where the velocity network has seen few examples of that schedule, the random-schedule baseline produces an endpoint that reflects schedule-specific extrapolation error rather than the true terminal direction. Our method instead forces the endpoints of two different schedules at matched (z_0, z_1, t) to agree, because the endpoint-consistency term penalizes exactly this disagreement, so we expect the gain to be largest on examples whose sampled schedule is rare and smallest on examples whose schedule is the linear default.

**vs. Stochastic-interpolant velocity loss** — On a case where the randomly drawn schedule places z_t in a region where the velocity network has seen few examples of that schedule, the single-interpolant baseline produces a chord projection that reflects schedule-specific extrapolation error rather than the true terminal direction. Our method instead forces the endpoints of two different schedules at matched (z_0, z_1, t) to agree, because the endpoint-consistency term penalizes exactly this disagreement, so we expect the gain to be largest on examples whose sampled schedule is rare and smallest on examples whose schedule is the linear default.

**vs. Diffusion-based LLIE** — On a case with severe low-light noise, the diffusion baseline produces a high-quality but multi-step endpoint, because its iterative refinement corrects early errors at the cost of inference time. Our method instead produces a one-step endpoint whose accuracy is bounded by the terminal-velocity regression, so we expect parity in PSNR on clean-pair cases but a small PSNR deficit on the noisiest subset, with a large wall-clock advantage that the PSNR-only comparison does not capture.

### What would falsify this idea

If SITVM's PSNR gain over the λ=0 ablation is uniform across latent-gap magnitudes and schedule-rarity strata rather than concentrated on large-gap and rare-schedule examples, then the invariance penalty is not acting through the claimed causal channel and the central claim is wrong. Additionally, if the endpoint-consistency penalty does not outperform the pointwise chord-projection penalty, then the adversarial alert's concern about mismatched states is validated and the method requires redesign.

## References

1. GLOW-FM: Efficient Generative Low-Light Image Enhancement with One-Step Latent Flow Matching
2. Building Normalizing Flows with Stochastic Interpolants
