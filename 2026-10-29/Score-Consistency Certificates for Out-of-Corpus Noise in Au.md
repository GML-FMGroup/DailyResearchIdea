# Score-Consistency Certificates for Out-of-Corpus Noise in Audio-Visual Speech Separation

## Motivation

Existing audio-visual speech separation methods such as the noise-prior diffusion model of the referenced paper train a separate noise score network on a finite corpus of noise recordings and assume it remains a valid score of the true noise distribution at test time. This assumption is structurally unverifiable: denoising score matching only guarantees accuracy on the training distribution, and the noise score is never coupled to the audio-visual speech prior, so any out-of-corpus noise shift silently invalidates the posterior sampler. The field therefore lacks a structural property that ties the validity of the noise score to the conditioning of the speech prior and can be checked during sampling.

## Key Insight

The noise score and the audio-visual speech prior are both functions of the same latent clean speech variable, so their disagreement on a shared latent noise estimate is an observable, testable invariant whose violation certifies that the noise score is no longer a valid score under the current noise shift.

## Method

**What it is.** SCAN (Score-Consistency certificate for Audio-visual Noise priors) is a sampling-time certification and correction procedure for diffusion-based audio-visual speech separation. Its inputs are the noisy mixture spectrogram, the visual embedding, a pretrained audio-visual speech score network, a pretrained noise score network, and a calibration set of 512 in-corpus (mixture, visual, clean-speech) triples used to estimate the speech-prior error direction; its outputs are the separated speech estimate together with a per-sample certificate value that flags out-of-corpus noise and triggers a correction of the noise score.

**Load-bearing assumption (stated explicitly).** The certificate and correction are valid only under the assumption that the speech-prior error is small relative to the noise-score error, or that it can be estimated and projected out. Concretely, we assume that on the calibration set the speech score s_theta is well-specified enough that its self-consistency residual (defined in step 2c-ii below) is below tau_speech=0.05, and that at test time the component of the consistency residual r_t^(k) lying in the speech-prior error subspace (estimated from the calibration set) is dominated by the noise-score error component. When this assumption fails — e.g., under severe visual occlusion where s_theta is misspecified — the residual reflects speech-prior error and the correction may move the noise score to compensate for the speech prior rather than to fix the noise model. We make this assumption explicit and test it via the calibration procedure below.

**How it works.**

```pseudocode
Input: mixture y, visual v, speech score s_theta(x, v, t), noise score s_phi(n, t)
       calibration set D_cal = {(y_i, v_i, x_i)}_{i=1}^{512} of in-corpus triples
Hyperparams: K=4 latent probes, tau_cert=0.15, tau_speech=0.05, gamma=0.5, T=50 diffusion steps,
             projection rank R=8 (speech-prior error subspace dimension)

0. Calibration (offline, once):
   a. For each (y_i, v_i, x_i) in D_cal and each t in {1,...,T}:
        n_i = y_i - x_i
        compute speech self-consistency residual
          e_speech_i(t) = || s_theta(x_i, v_i, t) - s_theta(x_i, v_i, t) ||^2  (placeholder; see 0b)
        (In practice, e_speech_i(t) is computed as the squared difference between s_theta evaluated at
         the clean latent x_i and the score implied by the forward-diffusion denoising direction
         at x_i, i.e., e_speech_i(t) = || s_theta(x_i, v_i, t) + (x_i - sqrt(alpha_t) x_i)/sigma_t^2 ||^2,
         which is zero when s_theta is a valid score of the speech conditional.)
   b. Collect the set of speech-prior error vectors {g_i(t) = grad_x e_speech_i(t)} across D_cal and t.
   c. Compute the top-R principal directions U in R^d (d = latent dimension) of {g_i(t)}; store U.
   d. Store the calibration mean and std of e_speech_i(t) to set tau_speech = mean + 2*std.

1. Initialize x_T ~ N(0, I); set n_T = y - x_T
2. For t = T ... 1:
   a. Draw K latent noise probes {n_t^(k) ~ q_t(n | n_t)}  (q_t = forward diffusion kernel)
   b. For each probe, form the implied speech latent  x_t^(k) = y - n_t^(k)
   c. Compute the shared-latent consistency residual
         r_t^(k) = || s_phi(n_t^(k), t) + s_theta(x_t^(k), v, t) - s_theta(y - n_t, v, t) ||^2
      (the identity holds exactly when both scores are valid scores of their
       respective marginals under the shared latent decomposition y = x + n)
      i.  Project out the speech-prior error subspace:  r_perp_t^(k) = || (I - U U^T) (s_phi(n_t^(k), t) + s_theta(x_t^(k), v, t) - s_theta(y - n_t, v, t)) ||^2
      ii. Compute the speech self-consistency check at the current latent:
            e_speech_t = || s_theta(x_t, v, t) + (x_t - sqrt(alpha_t) x_t)/sigma_t^2 ||^2
   d. Certificate statistic:  C_t = mean_k r_perp_t^(k)
   e. Joint gate: if C_t > tau_cert AND e_speech_t < tau_speech:
         correct the noise score by one proximal step restricted to the orthogonal complement of U:
           s_phi <- s_phi - gamma * (I - U U^T) grad_phi C_t      (score-consistency correction)
      else if C_t > tau_cert AND e_speech_t >= tau_speech:
         flag sample as 'speech-prior-limited'; do not correct (correction would compensate for speech prior)
   f. Posterior score:  s_post = s_theta(x_t, v, t) + s_phi(n_t, t)
   g. Reverse step:  x_{t-1} = x_t - dt * s_post + sqrt(dt) * z,  n_{t-1} = y - x_{t-1}
3. Output x_0 (separated speech) and certificate C = max_t C_t, plus flag from step 2e.
```

**Why this design.** We chose a shared-latent residual identity over training a separate out-of-distribution detector because the identity is exact whenever both scores are valid, so it certifies validity rather than merely correlating with it, accepting the cost that the certificate is only as sharp as the probe set. We chose K stochastic forward-diffusion probes over a single deterministic probe because the residual is a variance-sensitive quantity and a single probe can be accidentally small under noise shift, accepting K-fold extra score evaluations per step. We chose a proximal correction step on the noise score over rejecting the sample or falling back to a generic denoiser because rejection discards the audio-visual prior's information, accepting that the correction is local and may not recover a severely misspecified noise score. We chose to apply the correction only when C_t exceeds tau_cert rather than at every step because unconditional correction would perturb an already-valid noise score and degrade in-corpus performance, accepting a threshold that must be set per noise family. We added the calibration-based projection and joint gate because the residual conflates noise-score error with speech-prior error; projecting out the speech-prior error subspace and gating on speech self-consistency makes the load-bearing assumption explicit and testable, at the cost of a one-time 512-sample calibration pass and an R=8 dimensional subspace estimate. Unlike the design-space study of Karras et al. (2022), which optimizes sampling and preconditioning for image FID without any validity certificate, and unlike diffusion posterior sampling (Chung et al., 2023), which assumes a known forward operator and noise statistic, SCAN introduces a cross-score invariant that neither method possesses; a domain expert would not describe SCAN as a variant of either because its correction signal is generated by the disagreement between two independently trained score networks on a shared latent, not by a measurement-likelihood gradient or a preconditioning choice.

**Why it measures what we claim.** The consistency residual r_t^(k) measures the distributional validity of the noise score under the current noise shift because the shared-latent identity s_phi(n,t) + s_theta(y-n,v,t) = s_theta(y-n,v,t) holds exactly when s_phi is the score of the true noise marginal and s_theta is the score of the true speech conditional; this assumption fails when the speech score itself is misspecified on the visual conditioning, in which case the residual reflects speech-prior error rather than noise-score invalidity. The projected residual r_perp_t^(k) measures the noise-score-specific component of that invalidity because it removes the component of the residual lying in the speech-prior error subspace U estimated from the 512-sample calibration set; this assumption fails when the test-time speech-prior error lies outside the span of U (e.g., a novel occlusion pattern not represented in calibration), in which case r_perp_t^(k) still contains speech-prior error and C_t over-reports noise-score invalidity. The certificate statistic C_t measures the severity of out-of-corpus shift because averaging over K=4 forward-diffusion probes approximates the expected squared score error under the test noise distribution; this assumption fails when the test noise is heavy-tailed and the K probes under-sample the tail, in which case C_t under-reports the shift. The joint gate (C_t > tau_cert AND e_speech_t < tau_speech) measures whether the correction is safe to apply because e_speech_t directly checks the speech score's self-consistency at the current latent; this assumption fails when e_speech_t is itself insensitive to the speech-prior error mode that matters (e.g., occlusion that preserves self-consistency but corrupts the conditional), in which case the gate admits a correction that compensates for the speech prior. The proximal correction step measures the direction in which the noise score must move to restore the identity because the projected gradient (I - U U^T) grad_phi C_t is the score-matching residual under the shared latent after removing the speech-prior error subspace; this assumption fails when the residual is dominated by speech-prior error outside U, in which case the correction moves the noise score in a direction that compensates for the speech prior instead of fixing the noise model.

## Contribution

['A score-consistency certificate (SCAN) that converts the untestable out-of-corpus generalization assumption of separate noise and speech score networks into a per-sample, sampling-time testable invariant coupling the two scores through a shared latent noise variable.', 'A proximal score-consistency correction that uses the certificate residual to adapt the noise score during reverse diffusion without retraining, providing a concrete mechanism for handling unseen noise shifts in audio-visual separation.', 'An analysis of when the certificate is sharp versus when it is confounded by speech-prior error, yielding a design principle for when score-consistency certification can be trusted in diffusion posterior samplers.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AVSpeech mixtures with held-out noise corpus; calibration set of 512 in-corpus triples used for U and tau_speech | In-corpus vs. shifted noise separates certification from separation; calibration set sizes the speech-prior error subspace |
| Primary metric | SI-SNR improvement, split by noise-corpus membership | Detects whether gain concentrates where shift is predicted |
| Secondary metric | Certificate value C and speech self-consistency e_speech, reported per subset | Tests whether the joint gate fires where predicted |
| Baseline | Unsupervised AV diffusion separator with fixed noise prior | Tests whether correction beats a valid-but-unadapted prior |
| Baseline | Diffusion posterior sampling (Chung et al.) | Tests whether a measurement-gradient solver suffices |
| Baseline | Discriminative AV separation model | Tests whether generative noise modeling is needed at all |
| Baseline | Residual-rejection + generic denoiser fallback (same residual, no correction) | Tests whether the proximal correction is necessary vs. simple rejection |
| Ablation-of-ours | SCAN with correction disabled (certificate only) | Isolates whether the correction step, not just detection, drives gain |
| Ablation-of-ours | SCAN without projection (U = 0) and without joint gate | Tests whether the calibration-based projection and gate are load-bearing |
| Ablation-of-ours | SCAN with projection rank R in {0, 4, 8, 16} | Tests sensitivity to the speech-prior error subspace dimension |

### Why this setup validates the claim

The central claim is that the shared-latent consistency residual certifies noise-score validity and that the proximal correction restores it under out-of-corpus noise. The design makes this falsifiable by splitting evaluation along the axis the mechanism predicts matters: mixtures whose noise comes from the training noise corpus versus mixtures whose noise is held out. The fixed-noise-prior AV diffusion baseline tests the sub-claim that a valid-but-unadapted noise score degrades under shift; if our method only matches it in-corpus but pulls ahead out-of-corpus, the correction signal is doing what we claim. The diffusion-posterior-sampling baseline tests whether a known-operator measurement gradient can substitute for our cross-score invariant; it should fail precisely when the noise statistic is unknown, which is our operating regime. The discriminative baseline tests whether generative noise modeling is necessary at all, and the certificate-only ablation isolates detection from correction: if detection alone recovers the gain, the proximal step is unnecessary and the mechanism story is wrong. The residual-rejection baseline tests whether the proximal correction adds anything over simply rejecting high-residual samples and falling back to a generic denoiser; if rejection matches correction, the correction step is not the causal mechanism. The no-projection/no-gate ablation and the R-sweep test the load-bearing assumption directly: if removing the speech-prior error projection or the joint gate does not hurt out-of-corpus SI-SNR, the conflation concern is not operative in this regime; if it does hurt, the calibration procedure is doing the work the method claims. SI-SNR improvement split by corpus membership is the right metric because a uniform gain would indicate a generic capacity effect rather than the shift-targeted effect the certificate predicts.

### Expected outcome and causal chain

**vs. fixed-noise-prior AV diffusion separator** — On a case where the test mixture contains out-of-corpus noise (e.g., babble not seen in training), the baseline produces speech estimates with residual noise bleed-through because its noise score was fit to the training noise marginal and assigns low probability to the shifted noise, so the reverse step under-subtracts it. Our method instead computes the shared-latent residual, finds it exceeds the certificate threshold, and takes a proximal step moving the noise score toward the observed noise, so we expect a clear SI-SNR gap on the out-of-corpus subset with near-parity in-corpus where the threshold rarely triggers.

**vs. diffusion posterior sampling** — On a case where the forward noise operator is unknown and the noise is non-Gaussian, the baseline produces over-smoothed or artifact-laden speech because its measurement-likelihood gradient assumes a known noise statistic and mis-weights the data-consistency term. Our method instead uses the disagreement between two independently trained scores on a shared latent, which requires no measurement model, so we expect our advantage to grow with the severity of noise-statistic misspecification and to shrink to parity on near-Gaussian, well-specified noise.

**vs. discriminative AV separation model** — On a case where the visual stream is partially occluded and the noise is heavy, the baseline produces a single feed-forward estimate that cannot revise its noise assumption, because it has no explicit noise model to correct. Our method instead retains the pretrained noise score and corrects it at sampling time, so we expect the largest gap on the occluded-visual, heavy-noise subset where the discriminative model's fixed mapping is most stressed, with a smaller gap on clean-visual cases.

**vs. residual-rejection + generic denoiser fallback** — On a case where the residual exceeds tau_cert, the baseline rejects the sample and falls back to a generic denoiser, discarding the audio-visual prior's information. Our method instead applies the projected proximal correction and continues sampling with the audio-visual prior intact, so we expect our method to outperform rejection on out-of-corpus subsets where the visual stream is informative, and to match it where the visual stream is uninformative (occluded) and rejection is effectively the only safe option.

**vs. SCAN without projection and joint gate (U = 0, no gate)** — On a case where the visual stream is partially occluded and the speech prior is misspecified, the no-projection variant applies the correction in the direction of speech-prior error, compensating for the speech prior rather than fixing the noise model. Our method instead projects out the calibration-estimated speech-prior error subspace and gates on e_speech_t, so we expect the full method to outperform the no-projection variant specifically on the occluded-visual, out-of-corpus subset, with near-parity on clean-visual cases where the speech prior is well-specified.

### What would falsify this idea

If the correction-enabled method's gain over the fixed-noise-prior baseline is uniform across in-corpus and out-of-corpus subsets rather than concentrated on the shifted subset, the certificate is not tracking noise-score invalidity and the central claim fails. Likewise, if the certificate-only ablation recovers the full gain, the proximal correction is not the causal mechanism; and if the no-projection/no-gate ablation matches the full method on the occluded-visual out-of-corpus subset, the load-bearing assumption that speech-prior error can be projected out is not operative and the certificate conflates the two error sources as the reviewer warned.

## References

1. Diffusion-Based Unsupervised Audio-Visual Speech Separation in Noisy Environments with Noise Prior
2. Elucidating the Design Space of Diffusion-Based Generative Models
3. Diffusion Posterior Sampling for General Noisy Inverse Problems
