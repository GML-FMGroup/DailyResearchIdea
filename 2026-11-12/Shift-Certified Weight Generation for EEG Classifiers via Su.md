# Shift-Certified Weight Generation for EEG Classifiers via Subject-Statistic Conditioning

## Motivation

Language-model weight-space programming (Mixed Signals) produces EEG classifiers that match gradient descent with few subjects, but it treats the generated weight tensor as valid whenever it happens to score well on the source subjects — there is no mechanism tying the decision boundary to any calibrated notion of subject-level covariate shift, so robustness to new subjects is only ever observed post hoc, never enforced at generation time. This is a meta-level failure that recurs across every 'generated-model' tree: the frontier model's broad capability is silently used as a proxy for domain coverage, and the why_not_resolved note confirms no calibration mechanism exists. The structural consequence is that a boundary can be arbitrarily far from any shift-stable boundary while still fitting the handful of source subjects, and nothing in the generation objective penalizes this.

## Key Insight

Because the hand-crafted pipeline's boundary is a fixed, subject-statistic-parameterized function, the distance from a generated boundary to it can be computed in closed form from first and second moments of each subject's EEG, so shift-robustness becomes a differentiable constraint on the generated weights rather than an evaluation-time property.

## Method

**What it is.** We introduce SCWG (Shift-Certified Weight Generation). Input: architecture, task label set, a small set of source subjects with EEG trials (n_s = 3–5 subjects, 40–60 trials each), and a target shift budget ε. Output: a weight tensor W for the EEG classifier such that, for every source subject s, the decision-boundary induced by W lies within a certified subject-level Wasserstein-2 distance ε of the hand-crafted pipeline's boundary on s. The language model proposes W; a closed-form shift certificate either accepts W or returns a correction direction that the model must satisfy.

**Load-bearing assumption (stated explicitly).** The certificate d_s(W) is a valid proxy for subject-level shift-robustness only under assumption A: the hand-crafted anchor boundary w_h_s is itself a valid robustness target, i.e., its error on shifted subjects is bounded and it does not systematically mis-rank subjects by shift magnitude. Assumption A fails when the hand-crafted boundary is biased on heavy-tailed or artifact-contaminated subjects (e.g., CSP-LDA overfits the majority covariance), in which case certified proximity to it inherits that bias and d_s(W) ≤ ε does not imply robustness to new subjects. We test A directly in the experiment (hand-crafted boundary error on high-shift subjects) and report it alongside all certified results.

**How it works.** Three phases, with the LLM in the loop only in phase 2.

```pseudocode
# Phase 1: subject statistics (no LLM)
for each source subject s:
    X_s = band-power features (alpha 8-12Hz, beta 13-30Hz, theta 4-8Hz), shape [n_trials, 3]
    mu_s, Sigma_s = mean, cov of X_s            # 3x3 covariance, per subject
    # hand-crafted boundary on s: linear band-power anchor w_h_s (3-dim), fitted on s
    # (CSP-LDA is retained as a separate baseline, not as the certificate anchor,
    #  because CSP-LDA is non-linear in X_s and its moments are not well-defined here)
    w_h_s = fit_linear_bandpower_lda(s)        # deterministic, fixed pipeline, 3-dim
    C_s   = sqrt(trace(Sigma_s))               # subject-level covariate-shift scale

# Phase 2: LLM weight proposal with certificate feedback
W = LLM(arch, task, {s}, prompt_with_shift_budget(epsilon))   # no gradient steps
for iter in 1..K (K=5):
    for each s:
        # closed-form decision-boundary distance under Gaussian covariate shift,
        # computed on the induced scalar decision-value distributions, not on weights:
        #   z_s(W)   = W^T X_s        (n_trials scalar scores)
        #   z_s(w_h) = w_h_s^T X_s    (n_trials scalar scores)
        # Gaussian-W2 between the two scalar score distributions (Dowson-Landau):
        #   d_s(W) = sqrt( (m_W - m_h)^2 + (v_W + v_h - 2*sqrt(v_W*v_h)) ) / C_s
        # where m_W = mean(z_s(W)), v_W = var(z_s(W)), m_h = mean(z_s(w_h)), v_h = var(z_s(w_h))
        d_s(W) = gaussian_w2_scalar(z_s(W), z_s(w_h)) / C_s
    if max_s d_s(W) <= epsilon: break
    # correction: project the violating direction onto the LLM's next prompt
    delta_s = (d_s(W) - epsilon) * (W - w_h_s) / ||W - w_h_s||   # per-subject slack
    prompt += "subject s boundary exceeds budget by delta_s; re-emit W with these slack vectors"
    W = LLM(prompt)

# Phase 2b: deterministic fallback (guarantees termination)
if max_s d_s(W) > epsilon after K iterations:
    # closed-form projection of W onto the certified ball around w_h_s in score space:
    # shrink the score-distribution gap by scaling the deviation from w_h_s
    for each violating s:
        scale_s = epsilon / d_s(W)
        W = w_h_s + scale_s * (W - w_h_s)
    record fallback_used[s] = True

# Phase 3: calibration of epsilon from subject statistics
epsilon = quantile_{0.9}( {d_s(W_grad)} )   # W_grad = gradient-descent reference on same subjects
return W, {d_s(W)}, epsilon, fallback_used
```

**Why this design.** We chose a closed-form Gaussian-W2 certificate on the induced scalar score distributions over a learned shift predictor because the hand-crafted anchor is a fixed linear functional whose score moments are computable exactly from X_s, accepting the cost that the certificate is only valid for Gaussian-approximated score distributions and will be loose for heavy-tailed EEG artifacts (we report the Gaussian-approximation gap against a sliced-Wasserstein estimate on the same scores). We chose to keep the LLM as the weight proposer and feed it only the per-subject slack vectors, rather than fine-tuning the LLM on shift-robust weights, because the slack vector is a low-dimensional, interpretable correction that the model can act on without gradient descent — accepting the cost that convergence within K=5 iterations is not guaranteed and may require more LLM calls; the deterministic fallback projection in phase 2b guarantees termination and we report the fraction of subjects requiring it. We chose to calibrate ε from the gradient-descent reference on the same subjects rather than fixing ε a priori, because a fixed budget is meaningless across datasets with different feature scales — accepting the cost that ε inherits any bias in the reference training. A domain expert would not describe this as Mixed Signals plus a regularizer: Mixed Signals never computes a boundary distance, and its LLM sees only architecture and task, whereas here the LLM's prompt is a per-subject slack vector derived from a moment-based certificate on score distributions, which is a different generation surface.

**Why it measures what we claim.** The score-space Gaussian-W2 distance d_s(W) measures subject-level distributional robustness because it equals the Wasserstein-2 distance between the generated classifier's scalar decision-value distribution and the hand-crafted classifier's scalar decision-value distribution on subject s under the assumption that each subject's scalar scores are Gaussian; this assumption fails when EEG band-power features are heavy-tailed or when artifact rejection changes the score variance, in which case d_s reflects a Gaussian-projected shift rather than the true Wasserstein distance (we report the gap to a sliced-Wasserstein estimate). The scale C_s = sqrt(trace(Sigma_s)) measures subject-level covariate-shift magnitude because it is the total variance of the subject's feature distribution under the assumption that shift is isotropic in feature space; this assumption fails when shift is concentrated in one band, in which case C_s over-estimates the effective shift and the certificate is conservative (we report a per-band anisotropic scale as a diagnostic). The calibrated ε measures the shift budget that the gradient-descent reference already satisfies because it is the 90th percentile of the reference's own d_s values under the assumption that the reference is a reasonable robustness target; this assumption fails when the reference overfits the source subjects, in which case ε is too tight and the LLM is forced toward the hand-crafted boundary rather than a genuinely robust one. Assumption A (hand-crafted anchor is a valid robustness target) is measured directly by the hand-crafted boundary's error on high-shift subjects; when A fails, d_s(W) ≤ ε certifies proximity to a biased anchor, not robustness.

## Contribution

(1) A shift-certified weight-generation framework (SCWG) that turns subject-level covariate-shift robustness into a closed-form, LLM-actionable constraint on generated EEG classifier weights, using per-subject moment statistics and a Mahalanobis/scale certificate. (2) A calibration procedure that sets the shift budget ε from a gradient-descent reference on the same subjects, making the budget dataset-relative rather than a fixed hyperparameter. (3) An analysis of when the Gaussian-moment certificate is tight versus conservative, tied to band-power feature tails, that tells practitioners which EEG feature sets admit closed-form shift certification.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-subject EEG motor-imagery benchmark (BCI Competition IV 2a; 9 subjects, 2 sessions, 4 classes, 250 Hz) | Many subjects expose subject-level shift directly. |
| Primary metric | Cross-subject accuracy vs. per-subject certified slack d_s(W) (score-space Gaussian-W2 / C_s) | Ties accuracy gain to the shift mechanism claimed. |
| Secondary metric | Gaussian-approximation gap: |d_s(W) − SW_2(z_s(W), z_s(w_h))| / C_s, where SW_2 is a sliced-Wasserstein-2 estimate with 50 random projections on the same scalar scores | Validates the Gaussian-W2 certificate against a non-parametric Wasserstein estimate. |
| Secondary metric | Hand-crafted anchor error on high-shift subjects (top-tertile C_s) | Tests assumption A directly. |
| Secondary metric | Fraction of subjects requiring phase-2b fallback projection | Feasibility diagnostic for K=5 LLM iterations. |
| Baseline | Gradient-descent trained EEG classifier (same architecture, Adam lr=1e-3, 100 epochs) | Standard learned reference, no shift certificate. |
| Baseline | Hand-crafted alpha-power + CSP-LDA pipeline | The fixed boundary our certificate anchors to (also the anchor for the linear band-power variant). |
| Baseline | Mixed Signals LLM weight-writer (no slack feedback) | Isolates the certificate-feedback contribution. |
| Ablation-of-ours | SCWG with slack feedback removed (LLM sees only architecture/task) | Tests whether per-subject slack is the active ingredient. |
| Ablation-of-ours | SCWG with isotropic C_s replaced by per-band anisotropic scale sqrt(diag(Sigma_s)) | Tests the reviewer-requested anisotropic shift scale. |

### Why this setup validates the claim

The central claim is that feeding the LLM a per-subject Gaussian-W2 slack derived from a moment-based certificate on score distributions produces boundaries that are both accurate and provably close to the hand-crafted boundary, and that this slack is the causal ingredient. The gradient-descent baseline tests whether the LLM-plus-certificate can match a learned reference at all; the hand-crafted pipeline tests whether accuracy is retained while staying within budget and, via its error on high-shift subjects, tests assumption A; the Mixed Signals-style writer tests whether the certificate feedback — not merely LLM weight-writing — drives any improvement. The ablation is decisive: if removing slack feedback leaves accuracy unchanged, the slack is inert and the mechanism is not doing what we claim. Cross-subject accuracy plotted against per-subject d_s(W) is the right metric because the claim predicts a specific joint pattern — gains concentrated on subjects where the certificate is tight and the baseline drifts — which a single aggregate accuracy number would wash out. The Gaussian-approximation gap and the fallback fraction are reported alongside every certified result so that the certificate's validity and the method's termination are auditable rather than assumed.

### Expected outcome and causal chain

**vs. Gradient-descent trained EEG classifier** — On subjects whose band-power covariance differs sharply from the pooled training distribution (top-tertile C_s), the gradient-descent baseline produces boundaries that overfit the majority covariance and misclassify the shifted subject because its loss never observes a per-subject shift constraint. Our method instead re-emits W until each subject's score-space Gaussian-W2 slack falls under ε because the certificate projects the violating direction into the prompt. We expect a noticeable accuracy gap on the high-shift subject subset but rough parity on low-shift subjects.

**vs. Hand-crafted alpha-power + CSP-LDA pipeline** — On subjects with heavy-tailed artifact-driven features, the hand-crafted pipeline produces a rigid boundary that is certified but suboptimal, because its fixed moments ignore trial-level structure the LLM can exploit; we report its error on high-shift subjects to test assumption A. Our method instead keeps d_s(W) within ε of the linear band-power anchor while letting the LLM move tangentially within the certified ball, because the slack budget permits movement that does not inflate boundary distance. We expect accuracy modestly above the hand-crafted pipeline with d_s(W) clustered just under ε, and we expect the hand-crafted anchor error on high-shift subjects to be non-trivial, which bounds how much robustness certified proximity can confer.

**vs. Mixed Signals LLM weight-writer (no slack feedback)** — On a subject where the LLM's first proposal drifts far from the hand-crafted boundary, the Mixed Signals-style writer produces an unconstrained W that exceeds the shift budget because its prompt carries no per-subject distance signal. Our method instead receives the slack vector and corrects within K iterations because the certificate returns a concrete direction to satisfy. We expect the no-feedback writer to show high variance in d_s(W) across subjects, while SCWG shows d_s(W) bounded and accuracy gains concentrated where the baseline exceeded ε.

**vs. SCWG without slack feedback (ablation)** — On mid-shift subjects, the ablated model produces boundaries similar to the unconstrained writer because removing slack removes the only subject-specific signal. Our method instead concentrates its accuracy advantage exactly on subjects whose initial d_s exceeded ε, because the slack vector is what directs correction. We expect the ablation to match SCWG on low-shift subjects and lose most of the gain on high-shift subjects.

**vs. SCWG with anisotropic C_s (ablation)** — On subjects whose shift is concentrated in one band (e.g., beta-only drift), the isotropic C_s over-estimates the effective shift and makes the certificate conservative, because it sums variance across bands. The anisotropic variant instead scales each band by its own standard deviation, so d_s(W) tracks the dominant shift direction. We expect the anisotropic variant to tighten d_s(W) on band-concentrated subjects and to leave accuracy roughly unchanged on isotropic-shift subjects.

### What would falsify this idea

If SCWG's accuracy gain over the unconstrained LLM writer is uniform across subjects rather than concentrated on the high-d_s subset the certificate flags, the slack-feedback mechanism is not the cause and the central claim fails. Likewise, if the ablation without slack feedback matches SCWG on the high-shift subset, the certificate is inert; and if the hand-crafted anchor's error on high-shift subjects is large (assumption A fails), then certified proximity to it cannot be read as shift-robustness.

## References

1. Mixed Signals: Weight-Space Programming for EEG and ECG
