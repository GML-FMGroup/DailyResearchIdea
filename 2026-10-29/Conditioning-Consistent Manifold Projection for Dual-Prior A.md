# Conditioning-Consistent Manifold Projection for Dual-Prior Audio-Visual Speech Separation

## Motivation

Dual-prior diffusion separation (as in the audio-visual noise-prior score model of the referenced work) samples the mixture posterior by combining a speech score and a noise score, but this posterior is only identifiable when the speech and noise priors are independent; when noise is speech-like or non-stationary, the sampler can drift along directions that preserve the mixture likelihood while swapping speech and noise content. The audio-visual score model already contains a low-dimensional sufficient statistic (the visual embedding) that disambiguates speech from noise, yet in the dual-prior sampler this conditioning information is discarded rather than used to constrain the posterior trajectory, so identifiability is lost exactly where visual evidence would recover it. This is the local instance of a recurring structural problem: auxiliary conditioning that resolves an ill-posed inverse problem is used only as a prior input and never as a consistency constraint on the sampler, so its disambiguating power does not transfer to the posterior geometry.

## Key Insight

The audio-visual embedding is a function of the speech component alone, so enforcing that each reverse-diffusion posterior update lies on the level set where the sampled speech explains the visual evidence removes precisely the speech-noise-swap directions that the mixture likelihood leaves unconstrained, restoring identifiability without requiring independent priors.

## Method

**What it is.** We introduce CCP-Sep (Conditioning-Consistent Projected Separation), a dual-prior reverse-diffusion sampler for audio-visual speech separation whose every reverse step is projected onto the audio-visual conditioning manifold. Inputs: noisy mixture waveform, visual stream, a pretrained audio-visual speech score network, and a pretrained noise score network. Output: separated speech estimate (noise is the residual).

**Load-bearing assumption (stated explicitly).** The method assumes that the frozen audio-visual embedding f_AV is a sufficient statistic for speech content up to noise robustness, i.e. that the Jacobian J of f_AV has near-zero components along the noise subspace, so that the Gauss-Newton residual r = f_AV(x_t) - f_AV(x_speech_ref(v)) measures speech-content error and its projection removes only speech-noise-swap directions without corrupting the mixture-consistent posterior. This assumption is not assumed silently: it is calibrated and verified empirically as described in steps 2b and 3b below, and the method falls back to a soft penalty when the residual lies within the calibrated noise floor.

**How it works.**

```pseudocode
Input: mixture y, visual features v, speech score s_theta(x_t, t, v), noise score s_phi(n_t, t)
       audio-visual embedding f_AV(.) (frozen, pretrained on clean speech+video)
       guidance scale w, projection step eta_0, K Langevin inner steps, T diffusion steps
       calibration set D_cal of 512 held-out clean speech+video pairs
       per-dimension tolerance tau (from D_cal), trust-region radius delta, damping lambda

# 0. Calibration (offline, once)
#    Compute empirical mean mu_f and covariance Sigma_f of f_AV(x_clean) on D_cal.
#    Compute per-dimension tolerance tau = 2 * std(f_AV(x_clean)) on D_cal.
#    Compute whitening matrix W = Sigma_f^{-1/2} (regularized with 1e-4 I).

Initialize x_T ~ N(0, I); n_T = y - x_T
for t = T, ..., 1:
    # 1. Dual-prior posterior score (branch-B update)
    grad_speech = s_theta(x_t, t, v)
    grad_noise  = s_phi(n_t, t)
    score_post  = grad_speech + grad_noise          # mixture-consistent posterior score
    x_t <- x_t + w * score_post + noise_schedule(t)

    # 2. Conditioning-consistency residual (whitened, calibrated)
    r_raw = f_AV(x_t) - f_AV(x_speech_ref(v))       # f_AV(x_speech_ref) precomputed from v
    r = W @ r_raw                                    # whitened residual
    # 2b. Calibrated tolerance: only project components exceeding tau
    r = sign(r) * max(|r| - tau, 0)                 # soft-threshold at calibrated noise floor

    # 3. Manifold projection: remove posterior-gradient component violating conditioning identity
    J = jacobian of f_AV at x_t (via one JVP)
    g = J^T (J J^T + lambda I)^{-1} r               # tangent-space projection of residual
    # 3b. Trust region: bound step magnitude
    g = g * min(1, delta / (||g|| + 1e-8))
    x_t <- x_t - eta_t * g                          # project onto conditioning manifold

    # 4. Re-enforce mixture consistency
    n_t <- y - x_t
return x_1
```

The projection in step 3 is a Gauss-Newton step on the constraint ||f_AV(x_t) - f_AV(x_speech_ref(v))||^2 = 0, computed with a single Jacobian-vector product through the frozen embedding network and a small Levenberg-Marquardt damping lambda = 1e-3. The scale eta is annealed as eta_t = eta_0 * (1 - t/T) so late reverse steps (where fine structure is decided) are projected more aggressively than early steps; eta_0 = 0.1, delta = 0.05 (trust-region radius in whitened residual space), tau computed per dimension from D_cal as 2 standard deviations. The mixture-consistency re-projection in step 4 is a hard constraint (n_t = y - x_t) rather than a penalty, keeping the noise branch exactly consistent with the observed mixture. When the whitened residual r lies entirely within the calibrated tolerance (i.e. r = 0 after soft-thresholding), the projection step is skipped and the update reduces to the soft-penalty variant, providing a principled fallback.

**Why this design.** We chose a hard manifold projection over a soft conditioning-consistency penalty (e.g., adding ||f_AV(x_t)-f_AV(x_speech_ref)||^2 to the score) because a soft penalty merely biases the posterior and can be overridden by a strong noise score in speech-like-noise regimes, accepting the cost of one extra JVP per reverse step. We chose Gauss-Newton projection with Levenberg-Marquardt damping over exact pseudo-inverse projection because f_AV is nonlinear and exact projection is ill-conditioned near the manifold boundary, accepting a small residual r at convergence in exchange for stable updates. We chose to re-enforce mixture consistency exactly (n_t = y - x_t) rather than sample noise independently, because independent noise sampling would reintroduce the speech-noise-swap freedom the projection is meant to remove, accepting that the noise branch's own score is only used as a gradient direction rather than as a full generative prior. We added the calibrated whitening and trust region because the load-bearing assumption (that f_AV is a sufficient statistic for speech with near-zero Jacobian on the noise subspace) is not guaranteed by any published AV embedding; the calibration set D_cal of 512 held-out clean speech+video pairs provides an empirical noise floor, and the trust region bounds the step so it cannot alter speech content outside the visually-identifiable subspace. A domain expert comparing this to Diffusion Posterior Sampling (Chung et al.) would note that DPS uses manifold-constrained gradients toward a *measurement* likelihood, whereas here the constraint is toward an *audio-visual conditioning identity* that is invariant to the speech-noise decomposition — a different manifold with a different invariance — and comparing to the dual-prior separation work, the conditioning is used as a geometric constraint on the sampler rather than only as an input to the speech score.

**Why it measures what we claim.** The conditioning-consistency residual r = f_AV(x_t) - f_AV(x_speech_ref(v)) measures *identifiability of the speech component* because the named assumption is that f_AV is a sufficient statistic for speech content up to noise robustness, which makes r = 0 equivalent to 'the sampled speech explains the visual evidence'; this assumption fails when the visual embedding is corrupted (occlusion, lip-sync offset), in which case r reflects visual-encoder error rather than speech-noise confusion and the projection pulls toward a wrong manifold. The projection magnitude ||g|| measures *how far the unconstrained posterior has drifted along speech-noise-swap directions* because the assumption is that the Jacobian J of f_AV has near-zero components along the noise subspace, so any residual component in the range of J^T corresponds to speech-content error; this assumption fails when noise is visually correlated (e.g., a second speaker on screen), in which case ||g|| conflates speech error with genuine visual ambiguity. The annealed step size eta_t measures *confidence in the conditioning constraint over diffusion time* because the assumption is that early reverse steps are dominated by global structure where the visual embedding is less discriminative, which fails for very short utterances where early steps already commit to fine content. The calibrated tolerance tau measures *the noise floor of the visual embedding* because the assumption is that residuals within tau are due to encoder noise rather than speech-content error; this assumption fails when the calibration set D_cal is not representative of the test distribution (e.g., different recording conditions), in which case tau is mis-calibrated and the projection either over-corrects or under-corrects.

## Contribution

(1) A conditioning-consistent manifold-projection sampler (CCP-Sep) that recasts an audio-visual conditioning pathway as a geometric constraint on a dual-prior diffusion posterior, with a Gauss-Newton projection using a single Jacobian-vector product per reverse step and an annealed projection strength. (2) A design principle showing that identifiability of the mixture posterior under dependent speech and noise priors is recovered by constraining the sampler to the level set where the sampled speech explains the visual evidence, rather than by requiring prior independence. (3) An analysis of when the audio-visual embedding fails to be a sufficient statistic (visual corruption, visually correlated noise) and how the projection residual diagnoses this failure at inference time.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LRS3 clean speech + DEMAND noise mixtures | Visual speech with controllable acoustic noise regimes |
| Primary metric | SI-SNRi on separated speech | Directly measures speech-component recovery, not noise fit |
| Baseline | Dual-prior diffusion separation (noise-prior) | Tests whether soft conditioning alone suffices |
| Baseline | DPS-style measurement-projection separation | Tests whether measurement manifold substitutes conditioning manifold |
| Baseline | Discriminative AV separation (no diffusion) | Tests whether generative sampling is needed at all |
| Ablation-of-ours | CCP-Sep with soft penalty instead of hard projection | Isolates the geometric projection as the causal ingredient |
| Ablation-of-ours | CCP-Sep with random-embedding projection (same architecture, random weights) | Tests whether gain comes from learned AV identity manifold |
| Ablation-of-ours | CCP-Sep without calibrated tolerance (tau=0) | Tests whether calibration is necessary |
| Diagnostic | Jacobian orthogonality check: cosine similarity between J^T columns and ground-truth speech-minus-noise direction on held-out mixtures | Grounds the 'noise subspace near-orthogonal to range(J^T)' premise |
| Diagnostic | Failure-mode flag: visually correlated noise (second on-screen speaker) and lip-sync offset subsets | Flags when the load-bearing assumption breaks |

### Why this setup validates the claim

The central claim is that projecting every reverse step onto the audio-visual conditioning manifold removes the speech-noise-swap freedom that soft conditioning cannot. The dual-prior baseline isolates the sub-claim that a noise score plus soft visual input is insufficient; if CCP-Sep beats it only where noise is speech-like, that is exactly the swap regime the projection targets. The DPS-style baseline isolates the sub-claim that the relevant manifold is the conditioning identity, not a measurement likelihood; if projection toward measurement likelihood does not close the same gap, the conditioning manifold is doing distinct work. The discriminative baseline tests whether the gain is merely from diffusion at all. The soft-penalty ablation is the decisive control: same score network, same noise prior, same data, differing only in whether the constraint is a penalty or a hard projection, so any gap is attributable to the projection. The random-embedding ablation tests whether the gain comes from the learned AV identity manifold or from generic manifold regularization: if projection onto a randomly-initialized embedding's level set yields similar gains, the learned AV structure is not doing the work. The no-calibration ablation tests whether the calibrated tolerance tau is necessary or whether the raw residual suffices. The Jacobian orthogonality diagnostic directly measures the load-bearing assumption: we compute the empirical cosine similarity between J^T columns and the ground-truth speech-minus-noise direction on held-out mixtures and report the distribution; if this similarity is not near zero, the premise that the noise subspace is near-orthogonal to range(J^T) is false. The failure-mode flag subsets (visually correlated noise, lip-sync offset) test the predicted breakdown: if CCP-Sep's advantage disappears or reverses on these subsets, that confirms the diagnostic is detecting the failure mode rather than a generic degradation. SI-SNRi is the right metric because it penalizes residual noise and speech distortion together, which is precisely the trade the swap freedom induces; a metric that ignored residual noise would be blind to the predicted failure. The dataset choice gives us clean visual speech and controllable noise type, so we can construct the speech-like-noise subset where the causal chain predicts the largest effect.

### Expected outcome and causal chain

**vs. Dual-prior diffusion separation (noise-prior)** — On a mixture where the interfering noise is itself speech-like (babble or a competing talker), the baseline's soft visual conditioning is overridden by a strong noise score, so it assigns speech energy to the noise branch and produces a separated signal that is locally plausible but drops or swaps phonemes. Our method instead re-projects the residual f_AV(x_t) - f_AV(x_speech_ref) at each step, and because the noise subspace is near-orthogonal to the range of J^T, the projection removes the swap component directly. We expect a pronounced SI-SNRi gap concentrated on the speech-like-noise subset, with near-parity on stationary noise, since stationary noise offers no swap ambiguity to correct.

**vs. DPS-style measurement-projection separation** — On a case with heavy nonstationary noise, the DPS-style baseline projects toward measurement consistency, which is satisfied by many speech-noise decompositions, so it converges to a mixture-consistent but visually inconsistent estimate, showing visible phoneme substitution. Our method instead projects toward the audio-visual conditioning identity, which is invariant to the decomposition, so the constraint selects the visually supported decomposition rather than any consistent one. We expect the gap to appear specifically on high-noise, high-visual-information segments, and to shrink where visual evidence is weak, because there the conditioning manifold is uninformative and projection cannot help.

**vs. Discriminative AV separation (no diffusion)** — On a case where the noise spectrally overlaps speech, the discriminative baseline's single forward pass regresses to the conditional mean and produces over-smoothed speech with muffled consonants and residual noise bleed, since it has no mechanism to iterate on a constraint. Our method instead runs reverse diffusion with per-step projection, so fine structure is decided under an active conditioning constraint that sharpens late steps via annealed eta_t. We expect a moderate overall SI-SNRi advantage that widens on consonant-heavy and low-SNR segments, where the regression-to-mean failure is strongest, and narrows on easy high-SNR segments where the discriminative model is already near-optimal.

**vs. CCP-Sep with soft penalty instead of hard projection** — On the speech-like-noise subset, the soft-penalty variant adds ||f_AV(x_t)-f_AV(x_speech_ref)||^2 to the score, which biases the posterior but can be overridden by a strong noise score, so it should show intermediate performance: better than the dual-prior baseline but worse than hard projection. We expect the hard-projection gain over soft penalty to be concentrated on the speech-like-noise subset and to vanish on stationary noise, confirming that the geometric projection is the causal ingredient.

**vs. CCP-Sep with random-embedding projection** — If the gain comes from generic manifold regularization rather than the learned AV identity manifold, then projecting onto a randomly-initialized embedding's level set should yield similar SI-SNRi improvements. We expect the random-embedding variant to perform near the soft-penalty or dual-prior baseline, with a large gap to full CCP-Sep on the speech-like-noise subset, confirming that the learned AV structure is necessary.

**vs. CCP-Sep without calibrated tolerance (tau=0)** — Without the calibrated noise floor, the projection acts on encoder noise as well as speech-content error, potentially over-correcting and corrupting speech. We expect the no-calibration variant to underperform full CCP-Sep, especially on segments with visual noise (occlusion, compression artifacts), and to show higher variance across runs, confirming that calibration is necessary for stable projection.

**Jacobian orthogonality diagnostic** — We expect the empirical cosine similarity between J^T columns and the ground-truth speech-minus-noise direction to be near zero (mean absolute cosine < 0.2) on held-out mixtures, supporting the premise that the noise subspace is near-orthogonal to range(J^T). If the distribution is broad or centered away from zero, the load-bearing assumption is false and the projection may not isolate swap directions.

**Failure-mode flag (visually correlated noise, lip-sync offset)** — On subsets with a second on-screen speaker or lip-sync offset, we expect CCP-Sep's advantage over the dual-prior baseline to shrink or reverse, because the visual embedding is corrupted and the residual r reflects visual-encoder error rather than speech-noise confusion. This would confirm that the diagnostic flags the predicted failure mode.

### What would falsify this idea

If the hard-projection ablation shows no gain over the soft-penalty variant specifically on the speech-like-noise subset, or if CCP-Sep's advantage is uniform across noise types and visual-quality subsets rather than concentrated where the swap freedom is predicted to occur, then the geometric-projection claim is wrong and the gain (if any) comes from something other than conditioning-manifold projection. Additionally, if the Jacobian orthogonality diagnostic shows that J^T columns are not near-orthogonal to the speech-minus-noise direction, or if the random-embedding ablation matches full CCP-Sep, then the load-bearing assumption fails and the method's identifiability argument collapses.

## References

1. Diffusion-Based Unsupervised Audio-Visual Speech Separation in Noisy Environments with Noise Prior
2. Elucidating the Design Space of Diffusion-Based Generative Models
3. Diffusion Posterior Sampling for General Noisy Inverse Problems
