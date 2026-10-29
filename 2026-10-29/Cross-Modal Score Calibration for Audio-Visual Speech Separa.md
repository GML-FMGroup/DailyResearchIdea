# Cross-Modal Score Calibration for Audio-Visual Speech Separation

## Motivation

Existing diffusion-based audio-visual speech separation (e.g., the two-score posterior sampler of the Noise Prior paper) composes independently trained speech and noise score networks but inherits image-domain preconditioning coefficients and assumes both scores are calibrated to the same effective noise level. This assumption is structurally false: each network was trained with its own denoising score-matching objective on isolated data, so their effective noise levels drift apart, and summing their gradients in the posterior sampler produces an inconsistent score field that degrades separation.

## Key Insight

The two score networks can be tied to a shared effective noise level by a closed-form cross-modal score-consistency invariant derived from the posterior factorization, which replaces the inherited image-domain preconditioning schedule with a data-derived schedule that preserves the invariant at every sampling step.

## Method

**What it is** — We introduce CISC (Cross-modal Invariant Score Calibration), a sampling-time procedure that takes a pretrained audio-visual speech score network and a pretrained noise score network, computes a per-step effective noise level from their outputs, and enforces a cross-modal score-consistency invariant that replaces the image-domain preconditioning schedule with a data-derived one. **Load-bearing assumption (explicit):** CISC assumes that each independently pretrained score network is calibrated to a single, recoverable scalar effective noise level at every diffusion timestep, so that a shared sigma_shared exists and can be estimated from their outputs via the Tweedie-based invariant solve. This assumption is load-bearing: if it fails (e.g., because the two networks were trained on different data with different schedules and their timestep-to-noise mappings are not monotone scalar functions of t), the invariant solve fits a compromise between two different targets rather than a true shared level.

**How it works** —
```python
# Input: noisy mixture y, visual features v, pretrained speech score s_theta(x_t, t, v),
#        pretrained noise score s_phi(n_t, t), initial x_T ~ N(0, sigma_max^2 I)
# Hyperparameters: K=1000 sampling steps, invariant tolerance eps=1e-4, EMA beta=0.9
# Precompute (once, on 256 held-out batches of isolated speech and isolated noise):
#   For each t, estimate sigma_eff^speech(t) and sigma_eff^noise(t) via
#     sigma_eff^speech(t) = E_{x,eps}[ ||eps + sigma_t s_theta(x + sigma_t eps, t, v)||^2 ] / d
#     (analogously for noise network on isolated noise data), using 256 held-out batches.
#   Fit monotone splines S_speech(t) and S_noise(t) mapping t -> effective noise level.
#   Store as lookup tables sigma_eff^speech(t) and sigma_eff^noise(t) for all t.
for k in range(K):
    t = t_k  # decreasing schedule
    # 1. Evaluate both scores at current x_t and n_t = y - x_t
    s_speech = s_theta(x_t, t, v)
    s_noise  = s_phi(y - x_t, t)
    # 2. Compute cross-modal consistency residual
    r = s_speech + s_noise  # posterior score should be zero-mean under correct calibration
    # 3. Estimate shared effective noise level via invariant:
    #    sigma_shared = argmin_s || s_speech - (- (x_t - x_0_hat)/s^2 ) ||^2 + || s_noise - (- (n_t - n_0_hat)/s^2 ) ||^2
    #    where x_0_hat = (x_t + s_speech * sigma_t^2)/alpha_t (Tweedie), n_0_hat analogously.
    #    Solve numerically with 10 Newton iterations initialized at sigma_t.
    sigma_shared = solve_invariant(s_speech, s_noise, x_t, y, alpha_t, sigma_t)
    # 4. Data-derived preconditioning: replace image-domain c_skip(t), c_out(t) with
    #    c_skip*(t) = sigma_shared^2 / (sigma_shared^2 + sigma_t^2), c_out*(t) = sigma_shared*sigma_t / sqrt(sigma_shared^2 + sigma_t^2)
    # 5. Update x_t via reverse SDE step with calibrated scores:
    #    x_{t-1} = x_t - (t_k - t_{k-1}) * (0.5 * s_speech + 0.5 * s_noise) + sqrt(t_k - t_{k-1}) * sigma_shared * z
    # 6. Enforce invariant: project s_speech and s_noise onto the manifold r=0 with tolerance eps
    #    using 5 iterations of alternating projection.
    if ||r|| > eps: s_speech, s_noise = project_consistency(s_speech, s_noise, sigma_shared)
return x_0
```

**Why this design** — We chose a closed-form invariant (step 3) over a learned calibration head because the posterior factorization gives an exact relation between the two scores and the shared noise level, accepting the cost that the invariant must be solved numerically at each step. We chose to derive the preconditioning schedule from the effective noise level (step 4) rather than inheriting the image-domain schedule, because image-domain coefficients assume a fixed noise schedule that does not match the audio-visual speech/noise data distributions, accepting the cost of a precomputation step. We chose a projection step (step 6) over simply summing the scores, because unconstrained summation allows the two scores to drift apart, accepting the cost of an extra projection per step. A domain expert would not describe this as a variant of the Noise Prior paper's posterior sampler because that method assumes calibrated scores and fixed preconditioning; CISC explicitly estimates and enforces the shared noise level, which is a structural property absent from the prior method.

**Why it measures what we claim** — The cross-modal consistency residual r = s_speech + s_noise measures the calibration mismatch between the two score networks because under the assumption that both networks are trained on the same underlying noise level, the sum of the speech score and noise score should equal the score of the mixture distribution, which is zero-mean at the true posterior mean; this assumption fails when the noise is highly non-stationary or speech-like, in which case r reflects model bias rather than calibration error. The shared effective noise level sigma_shared measures the common noise level to which both scores are calibrated because the Tweedie denoising formula maps each score to an implicit noise level, and the invariant solve finds the level that minimizes the combined denoising error; this assumption fails when the two networks were trained on data with different noise scales, in which case sigma_shared reflects a compromise rather than a true shared level. The data-derived preconditioning coefficients measure the invariant-preserving schedule because they are computed from sigma_shared, which is tied to the invariant; this assumption fails when the invariant is not exactly satisfied, in which case the coefficients reflect an approximate schedule.

## Contribution

['A cross-modal score-consistency invariant that ties the audio-visual speech score and the noise score to a shared effective noise level, enforced at sampling time without retraining either network.', 'A data-derived preconditioning schedule that replaces the inherited image-domain coefficients and preserves the invariant across sampling steps.', 'An empirical demonstration that the invariant reduces score mismatch and improves separation quality on audio-visual noisy speech benchmarks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AVSpeech with added non-stationary noise | Tests calibration under realistic noise mismatch |
| Primary metric | SI-SNR improvement over mixture | Directly measures separation gain, not reconstruction fidelity |
| Baseline | Noise Prior posterior sampler | Tests whether fixed preconditioning suffices |
| Baseline | Unconstrained score summation | Tests whether the projection/invariant actually matters |
| Baseline | EDICT-style fixed image-domain schedule | Tests whether data-derived schedule is needed |
| Ablation-of-ours | CISC without invariant solve (fixed sigma) | Isolates the shared-noise-level estimation contribution |
| Ablation-of-ours | CISC with per-network calibration map (no shared sigma) | Tests the load-bearing assumption of a shared scalar |

### Why this setup validates the claim

The central claim is that a shared effective noise level, estimated at sampling time and enforced as a cross-modal invariant, is what makes two independently pretrained score networks separable on noisy audio-visual mixtures. The Noise Prior baseline tests the sub-claim that calibrated preconditioning is necessary: it uses a fixed image-domain schedule, so if CISC's gain over it is concentrated on non-stationary-noise subsets, that isolates the schedule-calibration mechanism rather than generic diffusion quality. The unconstrained-summation baseline tests the sub-claim that the projection step matters: it shares CISC's scores and schedule but drops step 6, so any gap between them is attributable to consistency enforcement alone. The fixed-schedule baseline tests the data-derived preconditioning claim. The ablation without the invariant solve (fixed sigma) is the clean negative control: it keeps the architecture and scores but removes the shared-level estimate, so it should collapse toward the Noise Prior behavior. The per-network calibration map ablation tests the load-bearing assumption: it replaces the shared scalar with two independent monotone splines (as precomputed in step 1) and defines the invariant as equality of the two calibrated sigmas, projecting only when they agree within tolerance and otherwise falling back to the pretrained schedule; if this ablation matches full CISC, the shared-scalar assumption is not doing the work we attribute to it. SI-SNR improvement is the right metric because it directly penalizes residual noise and speech distortion, the two failure signatures the invariant is supposed to suppress; a reconstruction-fidelity metric would not distinguish calibration error from perceptual smoothing.

### Expected outcome and causal chain

**vs. Noise Prior posterior sampler** — On a case where the added noise is non-stationary (e.g., babble that swells mid-utterance), the Noise Prior sampler produces over-smoothed speech with residual babble because its image-domain preconditioning assumes a fixed noise schedule that no longer matches the actual noise level. Our method instead re-estimates sigma_shared each step from the two scores, so the reverse-SDE drift tracks the true noise level, and we expect a noticeable SI-SNR gap concentrated on the non-stationary subset with parity on stationary-noise clips.

**vs. Unconstrained score summation** — On a case where the speech and noise scores disagree (e.g., a fricative that the noise network partially absorbs), unconstrained summation produces a drifted estimate with a speech-shaped noise artifact because the two scores are allowed to separate without a shared anchor. Our method instead projects both scores onto the r=0 manifold each step, so the disagreement is resolved before the update, and we expect a moderate gap that grows with the disagreement magnitude, visible as reduced artifact energy on fricative-heavy segments.

**vs. EDICT-style fixed image-domain schedule** — On a case where the audio-visual speech and noise distributions have effective noise scales far from the image-domain schedule's assumption, the fixed schedule produces a biased c_skip/c_out pair and thus a biased denoised estimate because the coefficients were never fit to these data. Our method instead derives c_skip* and c_out* from sigma_shared, so the preconditioning matches the data, and we expect a consistent but smaller gap than the Noise Prior comparison, largest on clips whose effective noise scale deviates most from the image-domain prior.

**vs. CISC without invariant solve (fixed sigma)** — On a case where the true effective noise level drifts during sampling (e.g., noise that is speech-like and non-stationary), the fixed-sigma ablation uses a constant sigma_shared (the average over t) and thus cannot track the drift, producing residual noise and speech distortion. Our method re-estimates sigma_shared each step, so we expect a clear SI-SNR gap that grows with the drift magnitude, confirming that the invariant solve is the active ingredient.

**vs. CISC with per-network calibration map (no shared sigma)** — On a case where the two networks' calibrated sigmas disagree (e.g., because the noise network was trained on a different noise scale), the per-network calibration map ablation falls back to the pretrained schedule when the sigmas differ beyond tolerance, effectively reverting to the Noise Prior behavior on those steps. Our method instead enforces a shared sigma_shared, so we expect a gap on clips where the disagreement is large, but parity on clips where the two calibrated sigmas already agree; this tests the load-bearing assumption that a shared scalar exists and is recoverable.

### What would falsify this idea

If CISC's gain over the Noise Prior baseline is uniform across stationary and non-stationary noise subsets rather than concentrated where the fixed-schedule assumption is most violated, the schedule-calibration claim is wrong. If the ablation without the invariant solve matches full CISC, the shared-noise-level estimation is not doing the work we attribute to it. If the per-network calibration map ablation matches full CISC, the load-bearing assumption of a shared scalar is not necessary and the contribution reduces to a calibration-mapping method.

## References

1. Diffusion-Based Unsupervised Audio-Visual Speech Separation in Noisy Environments with Noise Prior
2. Elucidating the Design Space of Diffusion-Based Generative Models
3. Diffusion Posterior Sampling for General Noisy Inverse Problems
