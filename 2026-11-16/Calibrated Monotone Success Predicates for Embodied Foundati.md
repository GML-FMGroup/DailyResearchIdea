# Calibrated Monotone Success Predicates for Embodied Foundation Model Verification

## Motivation

Across embodied foundation model research, surrogate success signals are used without any guarantee that they are monotone with true physical task completion under domain shift. FOUNDER (E1) learns a temporal-distance reward from a world model, and Self-Improving Embodied Foundation Models (E3) extract a steps-to-go success detector, but neither establishes that these surrogates remain ordered with respect to actual completion when the deployment distribution differs from training. The agentic robotics paradigm (E4) adds a verify stage that inherits these uncalibrated surrogates, so skill invocation can be certified on the basis of a score that may invert under shift. The root cause is structural: success predicates are trained as regression targets without any identifiability constraint or distribution-free coverage requirement, so there is no mechanism that forces the surrogate ordering to coincide with physical completion ordering outside the training envelope.

## Key Insight

A success predicate can be made monotone with true physical completion under a declared domain-shift envelope by jointly learning a ranking-constrained surrogate and a sequential conformal calibration set whose coverage guarantee is conditioned on a shift-detection statistic, so that the predicate's confidence set excludes failure exactly when the surrogate ordering is certified to be preserved.

## Method

**What it is.** We introduce **CAMP** (Calibrated Monotone Predicate), a wrapper layer that takes as input (i) a surrogate success score $s(x)$ produced by an existing embodied foundation model branch (FOUNDER's temporal-distance reward or Self-Improving's steps-to-go detector), (ii) a small held-out calibration stream of trajectories with binary physical-completion labels, and (iii) a domain-shift statistic $d(x)$ computed from the observation embedding relative to the calibration distribution. It outputs a certified success decision $\hat{y} \in \{0,1\}$ together with a confidence set $C(x) \subseteq \{0,1\}$ and a certificate flag $z(x) \in \{\text{certified}, \text{uncertified}\}$, which the agentic harness's verify stage consumes to gate skill invocation.

**Load-bearing assumption (stated explicitly).** CAMP's certificate is valid only under the assumption that the base surrogate's ranking of states by proximity to physical task completion is preserved under the declared domain-shift envelope, so that an isotonic projection fitted on calibration labels plus a shift-conditioned conformal threshold can certify ordering out-of-distribution. This assumption is not taken on faith: it is operationalized as a measured ordering-preservation statistic (Phase 1b below), and the certificate is emitted as `certified` only when that statistic exceeds a pre-registered threshold on held-out shift-stratified calibration blocks. When ordering is not empirically stable, CAMP defaults to `uncertified` and the harness abstains rather than certifying on an assumed premise.

**How it works.**

```pseudocode
# Phase 1: Monotone surrogate re-ranking (training)
Input: base surrogate s(x), calibration trajectories D_cal = {(x_i, y_i)},
       shift statistic d(x), envelope radius rho, target miscoverage alpha=0.1,
       ordering-stability threshold tau=0.7
1. Fit isotonic projection g: R -> R on D_cal minimizing sum (g(s(x_i)) - y_i)^2
   subject to g nondecreasing  # enforces ordering, not absolute calibration
   (pool-adjacent-violators; no regularization; ties broken by left-continuity)
2. Compute nonconformity scores a_i = |g(s(x_i)) - y_i| for all i in D_cal
3. Split D_cal into K=5 sequential blocks; for each block k, compute
   q_k = (1-alpha) quantile of {a_i : d(x_i) <= rho_k} where rho_k is the
   empirical (1-alpha) quantile of d over block k  # shift-conditioned threshold
4. Fit a monotone map h: R -> R+ mapping d(x) to q via isotonic regression
   on {(rho_k, q_k)}  # threshold grows with shift

# Phase 1b: Ordering-invariance check (load-bearing assumption test)
5. For each held-out shift-stratified block k in {1..K}, refit g_k on D_cal \ block k
6. Compute Spearman rank correlation rho_k = corr(g_k(s(x_i)), g(s(x_i)))
   over x_i in block k
7. Set ordering_stable = 1 if min_k rho_k >= tau else 0
   # tau=0.7 pre-registered; if ordering_stable=0, all certificates are uncertified

# Phase 2: Certified decision (inference, called by harness verify)
Input: observation x, base score s(x), shift statistic d(x)
8. s_tilde = g(s(x))
9. q_x = h(d(x))            # shift-conditioned nonconformity threshold
10. C(x) = { y in {0,1} : |s_tilde - y| <= q_x }
11. if ordering_stable == 0: return (y_hat=argmax_y s_tilde, C(x), uncertified)
    elif C(x) == {1}: return (y_hat=1, C(x), certified)
    elif C(x) == {0}: return (y_hat=0, C(x), certified)
    else: return (y_hat=argmax_y s_tilde, C(x), uncertified)

# Phase 3: Harness gating rule
12. Harness invokes skill only if certificate == certified AND y_hat == 1
13. On uncertified, harness requests additional evidence (extra observation
    or re-query) rather than invoking; if budget exhausted, abstains.
```

Hyperparameters: isotonic projection uses pool-adjacent-violators; $\alpha=0.1$; $K=5$ sequential blocks; envelope radius $\rho$ set to the 90th percentile of $d$ over a held-out reference set; $h$ is fit with isotonic regression on 5 points and extrapolated linearly beyond the largest $\rho_k$; ordering-stability threshold $\tau=0.7$ (Spearman rank correlation, pre-registered). Calibration stream size: 512 trajectories (approximately 100 per block). Shift statistic $d(x)$: cosine distance between the observation embedding and the calibration-set mean embedding, using the frozen vision encoder of the base surrogate (FOUNDER's world-model encoder or Self-Improving's policy encoder, whichever produced $s(x)$); no additional embedding layer is introduced.

**Why this design.** We chose an isotonic projection of the base surrogate over a learned calibration head because the base surrogate's *ordering* is the only property we can reasonably expect to transfer across shift, while its absolute scale is not identified; this accepts the cost that calibration cannot repair a surrogate whose ordering is already inverted, which is why Phase 1b measures ordering stability and defaults to abstention when it is not empirically supported. We chose a shift-conditioned sequential conformal threshold $h(d(x))$ over a fixed global conformal threshold because a fixed threshold would either be too conservative in-distribution or unsound out-of-distribution; this accepts the cost that the threshold becomes noisy when $d(x)$ exceeds the largest observed $\rho_k$. We chose to expose a three-valued certificate (certified-success, certified-failure, uncertified) rather than a binary gate because the harness in E4 already has a verify stage that can request more evidence; this accepts the cost of extra interaction budget on uncertified cases. A domain expert would not describe CAMP as a variant of standard conformal prediction or of Self-Improving's steps-to-go detector, because standard conformal prediction provides marginal coverage over a fixed distribution and does not enforce monotonicity of the underlying surrogate, while CAMP's certificate is conditioned on a shift statistic, gated by a measured ordering-stability check, and its decision rule is derived from a monotone projection that is explicitly checked against the calibration stream.

**Why it measures what we claim.** The isotonic projection $g$ measures *ordering preservation* of the base surrogate because the pool-adjacent-violators solution is the unique monotone function minimizing squared error on the calibration set under the assumption that the calibration labels are noiseless physical-completion indicators; this assumption fails when trajectory labels are annotated from video with temporal ambiguity, in which case $g$ reflects the annotator's boundary convention rather than physical completion. The ordering-stability statistic $\min_k \rho_k$ measures *empirical stability of that ordering across shift-stratified blocks* because it compares block-wise refits of $g$ under the assumption that the held-out blocks are representative of the deployment shift envelope; this assumption fails when a deployment subpopulation is absent from calibration, in which case $\rho_k$ reflects within-calibration stability rather than out-of-distribution preservation. The shift-conditioned threshold $q_x = h(d(x))$ measures *distribution-free coverage under a declared shift envelope* because $h$ is fit to empirical quantiles of nonconformity scores within shift-stratified blocks under the assumption that $d(x)$ is a sufficient statistic for the shift; this assumption fails when the shift is driven by an unobserved factor (e.g., lighting change that does not move the embedding), in which case $q_x$ reflects in-distribution nonconformity and the certificate overstates coverage. The certificate flag $z(x)$ measures *identifiability of the success predicate* because the confidence set $C(x)$ is constructed to contain the true label with probability $1-\alpha$ under the calibrated shift model and the measured ordering-stability gate; this assumption fails when the base surrogate's ordering is inverted for a subpopulation not represented in calibration, in which case $z(x)$ reflects calibration-set coverage and block-wise stability rather than true predicate identifiability.

## Contribution

["A calibrated monotone success-predicate layer (CAMP) that wraps an existing embodied foundation model surrogate with an isotonic ordering projection and a shift-conditioned sequential conformal threshold, producing a three-valued certificate consumed by an agentic harness's verify stage.", "A design principle for embodied verification: skill invocation should be gated on the calibrated predicate's confidence set excluding failure, rather than on the raw surrogate score, and this gating rule can be added to any FOUNDER-style or Self-Improving-style branch without retraining the base model.", 'An empirical protocol for measuring predicate monotonicity under controlled domain shift, using the shift statistic $d(x)$ to stratify evaluation and report certified-coverage versus actual completion rate per stratum.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-task offline visual control with shift-stratified splits (FOUNDER benchmarks) | Provides text/video-specified tasks and domain gaps |
| Primary metric | Selective success rate at fixed abstention budget (10% abstention) | Measures gated decisions, not raw scores |
| Secondary metric | Empirical coverage gap on held-out shifted blocks (target 1-alpha=0.9) | Tests sufficiency of d(x) and certificate soundness |
| Baseline | Raw FOUNDER temporal-distance reward (no certificate) | Tests whether calibration adds value over surrogate |
| Baseline | Global conformal threshold on base surrogate | Tests whether shift-conditioning matters |
| Baseline | Self-Improving steps-to-go detector as success gate | Tests cross-surrogate transfer of the wrapper |
| Ablation-of-ours | CAMP without monotone projection (identity g) | Isolates ordering-preservation contribution |
| Ablation-of-ours | CAMP without ordering-stability gate (always certified when C(x) singleton) | Isolates Phase 1b contribution |
| Ablation-of-ours | CAMP with label-noise perturbation (5% flipped calibration labels) | Tests sensitivity to annotation ambiguity |
| Ablation-of-ours | CAMP with synthetic shift perturbation (controlled lighting/viewpoint) | Tests d(x) sufficiency under known shift magnitude |

### Why this setup validates the claim

The central claim is that a monotone, shift-conditioned certificate layer converts an uncalibrated embodied surrogate into a decision the harness can trust, and that the gain comes specifically from ordering preservation plus shift-conditioned thresholds, gated by a measured ordering-stability check. The raw-surrogate baseline tests whether the certificate layer adds anything at all beyond thresholding the existing score; if CAMP only matches it, the wrapper is decorative. The global-conformal baseline tests the sub-claim that a fixed threshold is unsound under shift, so the predicted gap should concentrate on shifted subsets rather than in-distribution ones. The steps-to-go baseline tests whether the wrapper is surrogate-agnostic, since a certificate that only works for one surrogate is a re-tuning, not a layer. The identity-g ablation isolates the monotone projection: if removing it costs nothing, ordering preservation is not doing the work we claim. The no-ordering-gate ablation isolates Phase 1b: if removing the stability check does not increase false certifications on shifted subsets, the load-bearing assumption is not being tested. The label-noise and synthetic-shift ablations directly probe the two failure modes named in block D: annotation ambiguity and unobserved shift. Selective success rate at a fixed abstention budget (10%) is the right primary metric because CAMP's whole value proposition is trading interaction budget for trustworthy invocations; a metric that ignores abstention would reward the uncertified branch for guessing. The coverage-gap secondary metric quantifies whether the certificate's claimed coverage holds on held-out shifted blocks, making the sufficiency of d(x) an empirical question rather than an assumption. Together these form a falsifiable design: the predicted effect is a subset-concentrated gap, not a uniform lift, and the ablations isolate each mechanism.

### Expected outcome and causal chain

**vs. Raw FOUNDER temporal-distance reward** — On a case where the surrogate's absolute scale drifts with task horizon (long-horizon task scores compressed near zero), the baseline invokes skills on low scores because it thresholds the raw value, producing false successes on unfinished tasks. Our method instead re-ranks through the isotonic projection fitted on calibration labels because ordering, not scale, is what transfers, so we expect a noticeable gap on long-horizon subsets and rough parity on short-horizon ones.

**vs. Global conformal threshold** — On a case where observations shift (novel lighting or viewpoint) without changing task semantics, the baseline's fixed threshold was set in-distribution and either over-invokes or over-abstains, producing a coverage violation concentrated in the shifted block. Our method instead expands the nonconformity threshold via $h(d(x))$ because the shift statistic is in the conditioning, so we expect certified coverage near the target (0.9) on shifted subsets where the global baseline degrades, with the gap growing with shift magnitude.

**vs. Self-Improving steps-to-go detector** — On a case where the steps-to-go head is well-ordered in-distribution but its reward shaping was tuned for practice, not verification, the baseline's success gate fires early on tasks with deceptive intermediate progress, producing premature invocations. Our method instead treats the detector as an opaque surrogate and applies the same monotone projection plus shift-conditioned threshold, so we expect the wrapper to recover most of the gated-decision quality even though the underlying surrogate differs, signaling surrogate-agnosticism.

**vs. CAMP without ordering-stability gate** — On a case where the base surrogate's ordering inverts on a shifted subpopulation absent from calibration, the no-gate variant emits `certified` whenever $C(x)$ is a singleton, producing false certifications that the harness acts on. Our method instead requires $\min_k \rho_k \ge 0.7$ before any certificate is emitted, so we expect the no-gate variant to show higher false-certification rate on shifted subsets while full CAMP degrades to abstention, confirming that the load-bearing assumption is being measured rather than assumed.

**vs. CAMP with label-noise perturbation** — On a case where 5% of calibration labels are flipped to simulate annotation ambiguity, the isotonic projection $g$ absorbs the noise and the ordering-stability statistic $\rho_k$ drops. Our method's gate should therefore emit more `uncertified` decisions, reducing selective success rate but preserving coverage; we expect the label-noise variant to show a measurable drop in certified fraction and a smaller coverage gap than the no-gate variant, quantifying the annotation-ambiguity failure mode.

**vs. CAMP with synthetic shift perturbation** — On a controlled lighting/viewpoint perturbation where ground-truth shift magnitude is known, we expect the coverage gap between predicted and empirical coverage to grow with unobserved-shift magnitude, and the ordering-stability statistic to decline, providing a direct test of d(x) sufficiency.

### What would falsify this idea

If CAMP's gain over the raw surrogate and global-conformal baselines were uniform across in-distribution and shifted subsets rather than concentrated where the predicted failure modes occur, the shift-conditioning and monotone-projection mechanisms would not be doing the claimed work. Likewise, if the identity-g ablation matched full CAMP, ordering preservation would be refuted as the operative ingredient, and if the no-ordering-gate ablation did not increase false certifications on shifted subsets, the load-bearing assumption would not be empirically testable by our design.

## References

1. Envisioning Agentic Robotics: A New Era of Robotic Systems
