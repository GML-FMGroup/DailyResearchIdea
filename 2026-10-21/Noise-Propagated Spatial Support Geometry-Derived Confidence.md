# Noise-Propagated Spatial Support: Geometry-Derived Confidence Intervals for VLM Spatial Claims

## Motivation

Existing spatial-reasoning pipelines treat geometric cues as point estimates: Geo3R extracts depth and surface normals from a single image and feeds them into structured 3D reasoning as if they were exact, while S-Agent 2.0 interprets tool readings in plain language without tracking how reading error changes the answer. Both are open-loop with respect to geometric noise — a single noisy depth value silently becomes a premise for a downstream 'left-of' or 'in-front-of' claim, and the only confidence attached to that claim is a post-hoc language confidence that has no causal relationship to the cue that established it. The missing mechanism is a propagation operator that maps cue-level estimator variance to claim-level support, so that reliability of the geometric premise is represented rather than assumed.

## Key Insight

Every spatial relation over 3D primitives is a closed-form algebraic predicate on cue values (e.g., 'left-of' is a sign test on a depth-difference), so the truth value of a spatial claim is a deterministic function of cue noise whose support interval can be computed analytically rather than learned or verbalized.

## Method

**What it is.** NPS (Noise-Propagated Support) is a training-free operator that sits between a geometric cue extractor and a VLM's spatial answer. Its inputs are (i) per-pixel depth and surface-normal estimates with per-cue variance from a variance-exposing estimator, (ii) a parsed spatial relation `R(a, b)` over two 3D primitives, and (iii) the relation's algebraic form. Its output is a claim-level support interval `[τ_lo, τ_hi]` on the relation's truth margin, which is rendered into the VLM prompt as a calibrated confidence and used to trigger a fallback. It is not a learned head and not a language confidence.

**Load-bearing assumption (stated explicitly).** NPS assumes that the per-pixel depth estimator's predictive variance `s_d^2(p)` is a calibrated, spatially-uncorrelated estimate of true depth error, so that the propagated centroid covariance `Σ_o = cov(P_o) + diag(s_d^2)` equals the actual centroid uncertainty and the standardized margin `z` has its nominal sampling distribution. This assumption is load-bearing: if `s_d^2` is mis-scaled or spatially correlated, `Σ_m` is wrong by an unknown factor, `z` is not standardized, the 95% interval has arbitrary coverage, and the 'undecided' trigger fires on the wrong cases. We therefore do not assume calibration; we verify and correct it with a held-out calibration set (step 0 below), and we report the interval as a conformal prediction interval on the margin rather than a Gaussian z-interval.

**How it works.**

```pseudocode
INPUT: image I; relation R(a,b) in {left-of, right-of, above, below, in-front-of, behind, near, far}
       depth estimator D with predictive variance; normal estimator N with ensemble variance
HYPER: sigma_floor = 1e-3 (relative depth units); K = 32 samples; alpha = 0.05
       calibration set C = 512 labeled (image, relation, true-margin) triples from MMSI-Bench train split
       kappa_init = 1.0; correlation_length L_init = 8 px

0. CALIBRATION (fit once on C, before main experiment)
   # Fit scalar calibration factor kappa and effective correlation length L by minimizing
   # negative log-likelihood of true margins under N(m_R, Sigma_m(kappa, L)):
   for each calibration triple (I, R(a,b), m_true):
       compute mu_d, s_d^2, mu_n, s_n^2 on I
       Sigma_o(kappa, L) = cov(P_o) + kappa * diag(s_d^2) * (1 + (L-1)*rho)  # rho = mean off-diag corr
       Sigma_m(kappa, L) = grad^T (Sigma_a + Sigma_b) grad + sigma_floor^2
       loss += -log N(m_true; m_R, Sigma_m(kappa, L))
   kappa*, L* = argmin loss   # grid search kappa in [0.1,10], L in {1,2,4,8,16,32}
   # Conformal quantile: on held-out 256 triples, compute nonconformity scores
   # s_i = |m_true_i - m_R_i| / sqrt(Sigma_m_i(kappa*, L*)); set q_hat = (1-alpha) empirical quantile
   # Report empirical coverage on a further 256 held-out triples as calibration diagnostic.

1. CUE EXTRACTION WITH NOISE
   for each pixel p: d(p) ~ N(mu_d(p), s_d^2(p))   # mu_d, s_d from D
                     n(p) ~ N(mu_n(p), s_n^2(p))   # mu_n, s_n from N ensemble

2. PRIMITIVE LIFTING
   for object o in {a, b}:
       P_o = backproject(seg_mask(o), mu_d)      # 3D point cloud
       c_o = centroid(P_o);  Sigma_o = cov(P_o) + kappa* * diag(s_d^2) * (1 + (L*-1)*rho)
       # covariance of centroid is propagated from per-pixel depth variance, corrected by kappa*, L*

3. RELATION ALGEBRA (closed form)
   For relation R, define margin m_R(a,b) = f_R(c_a, c_b) where
     f_left-of  = (c_b.x - c_a.x)          # sign test on x-difference
     f_in-front  = (c_a.z - c_b.z)         # sign test on depth-difference
     f_near      = (||c_a - c_b||) - r     # distance minus threshold r
   (each f_R is linear or quadratic in the c_o, hence differentiable in cues)

4. NOISE PROPAGATION (deterministic, not sampled)
   grad = d f_R / d(c_a, c_b)               # analytic Jacobian
   Sigma_m = grad^T (Sigma_a + Sigma_b) grad + sigma_floor^2
   # Sigma_m is the variance of the margin induced by cue noise, with kappa*, L* correction

5. SUPPORT INTERVAL (conformal, not Gaussian z)
   z = m_R(a,b) / sqrt(Sigma_m)             # standardized margin
   [tau_lo, tau_hi] = [z - q_hat, z + q_hat]   # conformal 95% interval, q_hat from step 0
   support = 1 if tau_lo > 0 else 0 if tau_hi < 0 else 0.5   # sign decided / undecided

6. VLM INTEGRATION
   prompt += f"Geometric support for '{R(a,b)}': margin={m_R:.3f}, 95% CI=[{tau_lo:.3f},{tau_hi:.3f}], verdict={support}"
   if support == 0.5:  # undecided -> fallback
       request second viewpoint / re-estimate depth on crop, then repeat 1-5
   return answer, support
```

**Why this design.** We chose a closed-form analytic Jacobian over Monte-Carlo sampling of cue noise because the relation algebra is differentiable and low-dimensional (centroid coordinates only), so the analytic propagation is exact for linear relations and cheap for quadratic ones, accepting the cost that it assumes locally-Gaussian cue noise and will under-cover for heavy-tailed depth errors near occlusion boundaries. We chose centroid-plus-covariance lifting over per-pixel relation evaluation because per-pixel evaluation would entangle segmentation error with cue error, accepting the cost that object-level relations with non-convex shapes (e.g., 'left-of' for an L-shaped object) are approximated by the centroid. We chose a support interval on the relation margin over a learned confidence head because the margin's distribution is derived from the cue noise model, not fit to answer labels, accepting the cost that the interval is only as calibrated as the estimator's predictive variance. This is not Geo3R with error bars: Geo3R consumes geometric evidence as exact premises and its confidence is whatever the language model emits; NPS makes the premise's reliability an explicit algebraic object that determines the answer's support, so a domain expert would describe it as a propagation operator, not as an augmented prompt. It is also not a retrieval or router scheme: the support value is computed from the relation's algebraic form, not matched against a library.

**Why it measures what we claim.** The standardized margin `z = m_R / sqrt(Sigma_m)` measures claim-level support because the assumption that cue errors are locally Gaussian and independent across pixels makes the margin's sampling distribution equal to the propagated Gaussian, so the interval's coverage is the nominal 1−alpha; this assumption fails at occlusion boundaries and specular surfaces where depth variance is heavy-tailed, in which case the interval reflects the estimator's stated variance rather than true error and can over-cover. The centroid covariance `Sigma_o` measures cue-level uncertainty because the assumption that per-pixel depth errors are independent makes the centroid variance equal to the mean per-pixel variance divided by mask area; this assumption fails for spatially correlated depth errors (e.g., smooth bias fields), in which case `Sigma_o` underestimates true centroid uncertainty. The `support` verdict measures the reliability of the geometric premise because the assumption that the relation's truth value is exactly the sign of the margin makes sign-decidability equivalent to premise reliability; this assumption fails for relations with thresholds (e.g., 'near') where the threshold itself is uncertain, in which case the verdict reflects margin magnitude relative to a fixed threshold rather than true reliability. The calibration factor `kappa*` and correlation length `L*` measure the degree to which the estimator's predictive variance is mis-scaled and spatially correlated because the assumption that `s_d^2` is calibrated and uncorrelated makes `kappa*=1` and `L*=1`; this assumption fails for monocular depth estimators trained on residuals, in which case `kappa*` and `L*` absorb the miscalibration and the conformal quantile `q_hat` restores nominal coverage rather than the Gaussian z-interval.

## Contribution

(1) A training-free noise-propagation operator that maps per-cue depth/normal estimator variance through the closed-form algebra of a spatial relation to a claim-level support interval, making the geometric premise's reliability an explicit object rather than a post-hoc language confidence. (2) A design principle — sign-decidability of the relation margin under propagated cue noise — that yields an undecided verdict for genuinely ambiguous relations and triggers a second-viewpoint fallback, separating cue-limited failures from reasoning-limited failures. (3) An analysis of which spatial relations admit exact closed-form propagation (linear sign tests) versus which require threshold-uncertainty treatment (distance predicates), with the corresponding interval-construction rules.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MMSI-Bench spatial-relation subset | Multi-view relations with known ground-truth geometry |
| Primary metric | Accuracy on sign-decided relations, stratified by occlusion | Isolates where propagation should and shouldn't help |
| Secondary metric | Empirical coverage of 95% CI vs. realized sign-error rate | Verifies calibration assumption, not just accuracy |
| Baseline | Geo3R | Exact-premise geometric prompting without uncertainty |
| Baseline | APC mental-imagery simulation | Perspective augmentation without cue-noise modeling |
| Baseline | S-Agent 2.0 perception toolkit | Tool-augmented agent without claim-level support intervals |
| Ablation-of-ours | NPS with fixed variance (no propagation) | Removes noise model, keeps prompt format |
| Ablation-of-ours | NPS with Gaussian z-interval (no conformal calibration) | Isolates effect of calibration vs. propagation |
| Calibration set | 512 labeled (image, relation, true-margin) triples from MMSI-Bench train split | Fits kappa*, L*, q_hat; held-out 256 for coverage check |

### Why this setup validates the claim

The central claim is that making a geometric premise's reliability an explicit algebraic object — a support interval on the relation margin — yields answers whose correctness tracks cue-noise decidability rather than language fluency. MMSI-Bench's multi-view spatial-relation items are the right testbed because ground-truth geometry exists, so we can label each instance by whether the relation is genuinely sign-decidable under the estimator's variance. Geo3R tests the sub-claim that exact-premise consumption is insufficient: it should answer confidently on cases where the premise is actually undecidable. APC tests the sub-claim that perspective augmentation without cue-noise propagation cannot distinguish decidable from undecidable relations. S-Agent 2.0 tests whether tool access alone, absent a support interval, recovers the same gains. The fixed-variance ablation is the critical control: it keeps NPS's prompt format and centroid lifting but removes the propagated variance, so any gain attributable to the interval's *content* rather than its *presence* is isolated. The Gaussian-z ablation isolates the effect of conformal calibration from the effect of propagation itself. The secondary coverage metric directly tests the load-bearing assumption: if the 95% CI's empirical coverage on held-out MMSI geometry deviates from 0.95, the calibration step is not doing its job and the support verdict is not trustworthy. Stratified accuracy on sign-decided relations is the right metric because the claim is about decidability-conditioned correctness, not raw accuracy that conflates abstention with error.

### Expected outcome and causal chain

**vs. Geo3R** — On a case where an object sits near an occlusion boundary and its depth estimate is heavy-tailed, Geo3R consumes the depth as an exact premise and emits a confident left-of/right-of verdict because its confidence is whatever the language model produces. Our method instead propagates the boundary-inflated depth variance into the centroid covariance, so the margin's standardized value straddles zero and the support verdict is undecided, triggering the second-viewpoint fallback. We expect a noticeable accuracy gap concentrated on the occlusion-boundary subset, with near-parity on clean, well-separated relations where both methods decide correctly. The conformal calibration step (kappa*, L*, q_hat) ensures the interval's coverage is verified on held-out MMSI geometry rather than assumed; we report empirical coverage alongside accuracy.

**vs. APC mental-imagery simulation** — On a case where two objects are close in depth and the viewpoint shift is small, APC simulates an alternative view but has no mechanism to quantify whether the simulated relation is robust to depth noise, so it commits to whichever relation the simulation renders and inherits estimator error silently. Our method instead computes the margin's variance from the cue-noise model and only decides the sign when the interval excludes zero, so on near-threshold depth-difference cases it abstains and re-estimates rather than committing. We expect APC to show confident errors on the near-threshold subset that our method converts into abstentions or corrections, visible as a gap in *confident-error rate* rather than overall accuracy.

**vs. S-Agent 2.0 perception toolkit** — On a case where the relation is genuinely undecidable from the given view, S-Agent's toolkit returns a geometric measurement but no reliability estimate, so the agent's language head rationalizes a verdict from the raw number. Our method instead attaches a support interval to the claim and routes undecided cases to a fallback, so the agent's answer is gated by decidability rather than by the plausibility of the measurement. We expect S-Agent to match us on decidable relations but to produce a cluster of high-confidence wrong answers on undecidable ones, giving us a gap on the undecidable subset with parity elsewhere.

**vs. NPS with fixed variance (no propagation)** — This ablation keeps NPS's prompt format and centroid lifting but replaces the propagated `Sigma_m` with a fixed constant variance, so the support interval is constant across instances. If full NPS matches this ablation, the propagation itself is inert and only the prompt format matters; we expect full NPS to outperform on occlusion-boundary and near-threshold subsets where the propagated variance varies, with parity on clean subsets.

**vs. NPS with Gaussian z-interval (no conformal calibration)** — This ablation uses the Gaussian z-interval `[z - z_{1-alpha/2}, z + z_{1-alpha/2}]` instead of the conformal interval with `q_hat`. If the Gaussian interval's empirical coverage on held-out MMSI geometry is close to 0.95, the calibration step is unnecessary; we expect the Gaussian interval to under-cover on occlusion-boundary subsets (where `s_d^2` is miscalibrated and correlated) and the conformal interval to restore coverage, visible as a gap in the coverage diagnostic and in accuracy on the undecidable subset.

### What would falsify this idea

If our accuracy gain over Geo3R and the fixed-variance ablation is uniform across occlusion-boundary and clean subsets rather than concentrated where the propagated variance predicts undecidability, then the support interval is not doing the causal work and the central claim fails. Equally falsifying: if the fixed-variance ablation matches full NPS, the propagation itself is inert and only the prompt format matters; or if the conformal interval's empirical coverage on held-out MMSI geometry deviates from 0.95 by more than 0.05, the calibration assumption is not met and the support verdict is not trustworthy.

## References

1. Geo3R: Mitigating Spatial Reasoning Hallucination in Multimodal Large Language Models
2. S-Agent 2.0: Spatial Reasoning with a General Coding Agent
3. Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation
