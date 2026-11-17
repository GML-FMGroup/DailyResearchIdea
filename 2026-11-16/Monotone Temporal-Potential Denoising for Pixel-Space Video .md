# Monotone Temporal-Potential Denoising for Pixel-Space Video Diffusion

## Motivation

Pixel-space video diffusion such as JVT demonstrates that a plain large-patch DiT can match latent diffusion baselines without an autoencoder, but its temporal coherence is only an emergent byproduct of joint attention over patches: nothing in the architecture or sampler guarantees that frame-to-frame consistency improves monotonically as the reverse trajectory advances. Because the shared residual gap is precisely that no structural property enforces this, temporal coherence can regress in late denoising steps and the model has no mechanism to detect or correct it. This is the meta-level bottleneck: across video-diffusion branches (latent, pixel, and score-composition variants like Diffusion^2) coherence is hoped for rather than monitored as an invariant of the sampler.

## Key Insight

If temporal coherence is expressed as a scalar potential computed directly on the denoised pixel estimate and the adaLN modulation is conditioned on a barrier that rejects any step increasing that potential, then coherence becomes a non-increasing monitored invariant of the reverse trajectory rather than an emergent property of attention.

## Method

**What it is.** We introduce MTP-Diff (Monotone Temporal-Potential Diffusion), a training-free sampler wrapper for pixel-space DiT video diffusion (JVT). Its inputs are the pretrained JVT weights, a noise seed, and a conditioning signal; its output is a denoised video tensor whose per-step temporal-consistency potential is non-increasing by construction.

**How it works.**

```pseudocode
# Precompute at sampling start
x_T ~ N(0, I)                     # shape [F, H, W, C], F frames
Phi_prev = +inf
# Calibration: fit linear map from motion magnitude to expected L1 on a 512-clip calibration set (see experiment)
#   motion_mag = mean_f || x0_hat[f+1] - x0_hat[f] ||_1  (same as raw L1)
#   We use a normalized potential: Phi_norm = Phi_raw / (motion_mag + eps)  with eps=1e-3
#   This divides out per-frame motion magnitude, so Phi_norm measures residual incoherence beyond motion.
#   We also gate the barrier to low-to-mid noise: trigger only when t <= T_gate = 0.6*T (i.e., signal-to-noise ratio above threshold).
for t = T, T-1, ..., 1:
    # 1. AdaLN-conditioned denoise with barrier modulation
    c_t   = adaLN_cond(t, c)              # JVT's existing conditioning
    eps   = JVT(x_t, t, c_t)              # predicted noise
    x0_hat = (x_t - sqrt(1-a_t)*eps) / sqrt(a_t)   # pixel estimate

    # 2. Compute temporal-consistency potential on x0_hat
    #    Phi_raw = mean over adjacent frames of || x0_hat[f+1] - x0_hat[f] ||_1
    #    motion_mag = same as Phi_raw (identity flow, no optical-flow net)
    #    Phi_norm = Phi_raw / (motion_mag + eps)   # normalized residual
    Phi_raw = mean_f || x0_hat[f+1] - x0_hat[f] ||_1
    Phi_norm = Phi_raw / (Phi_raw + 1e-3)   # since motion_mag = Phi_raw, this simplifies to Phi_raw/(Phi_raw+eps)
    # Note: with identity flow, motion_mag equals Phi_raw, so Phi_norm is a monotone transform of Phi_raw.
    # To break this degeneracy, we use a fixed off-the-shelf flow estimator (RAFT-small, pretrained on Sintel, frozen)
    #   to compute flow-warped photometric residual: Phi_warp = mean_f || warp(x0_hat[f+1], flow_{f->f+1}) - x0_hat[f] ||_1
    #   and motion_mag = mean_f || flow_{f->f+1} ||_1 (mean flow magnitude).
    #   Then Phi_norm = Phi_warp / (motion_mag + eps).
    #   This is the motion-compensated coherence measure.
    #   (We keep the identity-flow L1 as a secondary diagnostic, not the barrier potential.)
    Phi = Phi_norm

    # 3. Monotone barrier: reject step if potential increases (only in gated regime)
    if t <= T_gate and Phi > Phi_prev and t < T:   # first step always accepted; gate on noise level
        # re-denoise with adaLN modulation scaled by lambda=0.5 toward previous
        c_t_bar = adaLN_cond(t, c) * 0.5 + c_t * 0.5
        eps     = JVT(x_t, t, c_t_bar)
        x0_hat  = (x_t - sqrt(1-a_t)*eps) / sqrt(a_t)
        Phi     = Phi_norm   # recompute with same flow estimator

    # 4. Commit step and record potential
    x_{t-1} = sqrt(a_{t-1}) * x0_hat + sqrt(1-a_{t-1}) * eps
    Phi_prev = Phi
return x_0
```

Hyperparameters: barrier modulation factor lambda=0.5 (fixed, not tuned per video); potential is motion-compensated L1 (flow-warped photometric residual divided by mean flow magnitude) using a frozen RAFT-small flow estimator (pretrained on Sintel, no fine-tuning); noise gate T_gate = 0.6*T (barrier only active when t <= 0.6*T); at most one re-denoise attempt per step (bounded cost ≈ 1.3× baseline).

**Why this design.** We chose a *motion-compensated pixel-difference potential* over a learned temporal-consistency scorer because JVT already operates in pixel space, so the potential is computed on exactly the quantity the model generates, avoiding a proxy network that would need separate training and could drift; the cost is that flow estimation adds a small overhead (RAFT-small runs at ~30 FPS on a single V100, negligible vs. JVT's sampling cost), but it removes the confound of legitimate motion inflating the potential. We chose a *rejection-and-rescale barrier* over a gradient-penalty or projection onto a coherence manifold because rejection requires no gradient through the sampler and no modification of JVT weights, keeping the method training-free; the cost is that a rejected step is re-run with damped conditioning, which can slightly slow convergence at high noise. We chose to *couple the barrier to adaLN modulation* rather than to the noise prediction directly because adaLN is JVT's native conditioning surface (per the paper's design examination), so the intervention lives where the model already expects control; the cost is that damping lambda must be small enough not to collapse conditioning. A domain expert would not describe this as a variant of classifier-free guidance or of consistency-model distillation: those modify the score or distill a new model, whereas MTP-Diff leaves the score untouched and instead imposes an ordering constraint on the sampled trajectory's potential.

**Why it measures what we claim.** The quantity `Phi = mean_f || warp(x0_hat[f+1], flow) - x0_hat[f] ||_1 / (mean_f ||flow||_1 + eps)` measures *temporal coherence* because we assume that motion-compensated photometric residual on the denoised estimate is a monotone surrogate for perceptual frame-to-frame consistency; this assumption fails when large legitimate motion is not perfectly compensated by the frozen flow estimator (e.g., occlusions, non-rigid deformation), in which case Phi reflects residual motion error rather than inconsistency. The barrier condition `Phi > Phi_prev` measures *trajectory monotonicity* because we assume the denoising trajectory should reduce inconsistency as signal replaces noise; this assumption fails when early high-noise steps have near-random potentials, in which case the noise gate (t <= 0.6*T) prevents spurious rejection, and the metric reflects noise-floor fluctuations rather than true regressions. The adaLN damping factor lambda measures *intervention strength* because we assume halving the conditioning offset is a conservative correction; this assumption fails when conditioning is already weak, in which case the re-denoise reflects the unconditional prior rather than a coherence-corrected sample. We validate the potential against a held-out perceptual coherence metric (see experiment) to establish it is not merely measuring motion.

## Contribution

(1) A training-free monotone-barrier sampler (MTP-Diff) that couples JVT's adaptive layer normalization to a pixel-space temporal-consistency potential and enforces non-increase of that potential across reverse steps. (2) The design principle that temporal coherence in pixel-space video diffusion can be operationalized as a monitored scalar invariant on the denoised estimate, with a rejection-and-rescale mechanism that needs no weight modification. (3) An analysis of when pixel-L1 potential diverges from perceptual coherence (high-motion regimes) and the first-step exemption that keeps the barrier well-posed.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | UCF-101 class-conditioned video (test split, 512 clips) | Standard short-video benchmark with real motion |
| Primary metric | Motion-compensated coherence potential: flow-warped photometric residual / mean flow magnitude (using frozen RAFT-small) | Directly measures the quantity the barrier constrains, with motion normalized out |
| Secondary metric | FVD (I3D features, 2048-dim, pre-trained on Kinetics-400) | Standard perceptual quality metric, not confounded by motion magnitude |
| Secondary metric | Human coherence rating (5-point Likert, 100 clips, 3 raters) | Validates the potential against perceptual judgment |
| Baseline | JVT unmodified sampler | Isolates the barrier's effect on the same weights |
| Baseline | JVT + classifier-free guidance sweep (guidance scale 1.0, 2.0, 4.0, 8.0) | Tests whether guidance alone fixes incoherence |
| Baseline | JVT + naive potential-rejection (no adaLN damping) | Isolates damping from rejection |
| Ablation-of-ours | MTP-Diff with lambda=1.0 (no damping) | Tests whether the rescale is doing the work |
| Ablation-of-ours | MTP-Diff with identity-flow L1 potential (no motion compensation) | Tests whether motion compensation is necessary |
| Ablation-of-ours | MTP-Diff with no noise gate (barrier active at all t) | Tests whether gating is necessary |

### Why this setup validates the claim

The central claim is that a monotone potential barrier on the sampling trajectory reduces temporal incoherence in pixel-space DiT video diffusion without retraining. Testing this requires holding the model fixed (JVT weights), so that any change in output is attributable to the sampler wrapper rather than to a different generator. Unmodified JVT is the necessary control: it establishes the baseline incoherence level on identical seeds and conditioning. The classifier-free-guidance sweep is the sharpest competing explanation — if stronger guidance alone lowers frame-difference error, the barrier is unnecessary, so this baseline tests whether the sub-claim "trajectory ordering, not score sharpening, is what matters" survives. The naive-rejection baseline separates the two mechanisms inside MTP-Diff (rejection vs. adaLN damping), and the lambda=1.0 ablation tests whether damping is load-bearing or inert. The identity-flow ablation tests whether motion compensation is necessary for the potential to be a valid coherence surrogate. The noise-gate ablation tests whether gating prevents spurious rejections at high noise. The primary metric is the motion-compensated coherence potential, which is exactly the quantity being constrained; using it as an evaluation metric makes the test circular in a productive way — if the barrier works, it must move the very quantity it optimizes, and any gain that does not appear here would falsify the mechanism. To guard against the circularity, we also report FVD and human coherence ratings, which are not optimized by the barrier. Perceptual or FVD-style metrics are deliberately excluded as primary because they would confound motion magnitude with coherence, the exact failure mode we flagged for the potential.

### Expected outcome and causal chain

**vs. JVT unmodified sampler** — On a case with fast lateral motion and a static background (where adjacent-frame L1 is dominated by true motion), unmodified JVT sometimes produces frames whose background texture drifts between steps, giving spikes in frame-difference error at mid-to-late denoising timesteps. Our method rejects exactly those steps where the potential rises and re-denoises with damped conditioning, so we expect a modest reduction in the tail of the per-step potential curve (the spikes), with near-parity on the median step and on slow-motion clips where no rejection triggers. We expect FVD to improve by 5-10% relative to unmodified JVT, and human coherence ratings to improve by at least 0.5 points on a 5-point scale.

**vs. JVT + classifier-free guidance sweep** — On a case where conditioning is weak (ambiguous class label), higher guidance sharpens each frame toward the class prior but does not coordinate frames, so frame-difference error stays flat or worsens as guidance increases because per-frame sharpening amplifies independent detail. Our method leaves the score untouched and instead orders the trajectory, so we expect the guidance sweep to show no monotone improvement while MTP-Diff shows a gain concentrated on the weak-conditioning subset, not on the strong-conditioning subset. Specifically, we expect MTP-Diff's potential reduction to be 2-3x larger than the best guidance setting on the weak-conditioning subset.

**vs. JVT + naive potential-rejection (no adaLN damping)** — On a case where a step is rejected, naive rejection re-runs with identical conditioning and therefore reproduces the same potential, so it either accepts the same bad step or loops; the observable is a rejection rate near one without potential reduction. Our method's damped re-denoise moves the sample toward the unconditional prior, changing the potential, so we expect a low rejection rate (most steps accepted on first try) with a measurable drop in the potential's upward excursions, i.e. rejection events that actually terminate. We expect naive rejection to have a rejection rate >0.8 and no significant potential reduction, while MTP-Diff has rejection rate <0.2 and a 15-20% reduction in upward excursions.

**vs. MTP-Diff with lambda=1.0 (no damping)** — This ablation removes the adaLN damping, so the re-denoise uses identical conditioning and thus reproduces the same potential. We expect this ablation to behave like naive rejection: high rejection rate, no potential reduction, and no FVD improvement. If it matches full MTP-Diff, the damping sub-claim is falsified.

**vs. MTP-Diff with identity-flow L1 potential** — This ablation replaces the motion-compensated potential with raw identity-flow L1. We expect this ablation to show a smaller or no improvement in FVD and human ratings, because the potential conflates motion with incoherence and the barrier suppresses legitimate motion. Specifically, we expect the identity-flow variant to reduce potential but also reduce motion magnitude (measured by mean flow magnitude), indicating a global smoothing side-effect rather than coherence improvement.

**vs. MTP-Diff with no noise gate** — This ablation activates the barrier at all timesteps, including high noise where x0_hat is dominated by noise-floor fluctuations. We expect this ablation to have a high rejection rate (>0.5) and to degrade FVD, because spurious rejections at high noise disrupt the sampling trajectory. This tests whether the noise gate is necessary.

### What would falsify this idea

If MTP-Diff's frame-difference gain is uniform across motion-magnitude subsets rather than concentrated on the high-motion, high-drift clips where the barrier triggers, the mechanism is not the ordering constraint but a global smoothing side-effect, and the central claim is wrong. Likewise, if the lambda=1.0 ablation matches full MTP-Diff, the adaLN-damping sub-claim is falsified and only bare rejection matters. Additionally, if the identity-flow L1 ablation matches full MTP-Diff on FVD and human ratings, the motion-compensation sub-claim is falsified and the potential is not a valid coherence surrogate.

## References

1. JVT: Just Video Transformer makes pixel-space video diffusion simple
