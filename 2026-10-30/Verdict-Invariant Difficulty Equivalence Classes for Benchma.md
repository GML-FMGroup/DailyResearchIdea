# Verdict-Invariant Difficulty Equivalence Classes for Benchmark Diagnostics

## Motivation

The measure-mismatch and coverage-gap diagnostics introduced by 'Your Benchmark Is an Empirical Measure Over Difficulty' — and consumed downstream by estimator families such as IRT (Fluid Benchmarking), Bradley-Terry (LLM compare), and LLM-simulated rankers (RIDE) — all treat a single scalar difficulty estimate per item as ground truth, so seed-level and estimator-level variance is never propagated into the diagnostic verdict. This is the recurring meta-gap across these paper-trees: each ancestor (and the frontier) assumes difficulty is summarized accurately enough to serve as ground truth, and the distribution-level diagnostics amplify rather than attenuate estimation error, leaving the diagnostic verdict with no identifiability guarantee under resampling or estimator choice.

## Key Insight

The diagnostic verdict is forced by the data exactly when it is constant over the resampling-stable equivalence class of difficulty vectors consistent with the observed measurement-error covariance — so reporting verdicts only when they are invariant across that class converts an unidentifiable point-estimate diagnostic into an estimator-agnostic one.

## Method

**What it is.** VIDEC (Verdict-Invariant Difficulty Equivalence Classes) takes as input a response matrix (items × models), a set of difficulty estimators, and the two diagnostic functionals (measure-mismatch divergence D and coverage-gap mass G), and outputs a *certified verdict* in {mismatch, no-mismatch, undetermined} × {gap, no-gap, undetermined}, together with the equivalence class of difficulty vectors that all yield that verdict.

**Load-bearing assumption (stated explicitly).** VIDEC assumes that after an explicit per-estimator affine calibration (scale-linking) to a common difficulty scale, the three estimators' calibrated difficulty vectors are all statistically consistent with a single shared true difficulty vector up to their respective sampling covariances, so that intersecting their chi-square Mahalanobis envelopes yields a class that represents sampling noise around one truth rather than a union of mutually incompatible, differently-biased parameterizations. This assumption is tested in the calibration experiment (synthetic ground-truth coverage) and is conditional on the stated equating map; if the equating map is misspecified, the class reflects equating error rather than sampling noise.

**How it works.**

```pseudocode
INPUT: responses R (n_items x n_models), estimators E = {IRT-2PL, Bradley-Terry, LLM-ranker},
       diagnostics D(.), G(.), bootstrap draws B=200, seed-batch size k=5, tolerance tau=0.02,
       anchor set A (n_anchor items with known difficulty, e.g., from a calibration split)

# Phase 0: per-estimator affine calibration (scale-linking) to a common scale
for e in E:
    fit affine map f_e(theta) = a_e * theta + b_e on anchor set A   # least-squares on anchors
    theta_e_cal = f_e(theta_e)                                      # calibrated difficulty
    Sigma_e_cal = a_e^2 * Sigma_e                                   # propagated covariance

# Phase 1: per-estimator difficulty + measurement-error covariance (on calibrated scale)
for e in E:
    theta_e, Sigma_e = fit_and_covariance(R, e)   # Sigma_e from Fisher info / bootstrap
    theta_e_cal, Sigma_e_cal = calibrate(theta_e, Sigma_e, f_e)

# Phase 2: seed-level resampling to inflate covariance honestly
for b in 1..B:
    R_b = resample_seeds(R, k)                    # resample k seeds per item
    theta_e^b = refit(R_b, e)
    theta_e_cal^b = f_e(theta_e^b)
Sigma_e_cal <- max(Sigma_e_cal, Cov_b(theta_e_cal^b))   # conservative envelope

# Phase 3: construct the resampling-stable equivalence class (on calibrated scale)
#   a difficulty vector theta' is in the class iff its Mahalanobis distance
#   to every estimator's calibrated estimate is within the chi-square envelope
Class = { theta' : for all e, (theta'-theta_e_cal)^T Sigma_e_cal^{-1} (theta'-theta_e_cal) <= q_chi2(0.95) }

# Phase 4: verdict evaluation over the class (corners + sampled interior)
for theta' in sample_from(Class, B=200) + corners(Class):
    v_D[theta'] = sign(D(theta') - tau)
    v_G[theta'] = sign(G(theta') - tau)

# Phase 5: invariance test
if all v_D equal:  verdict_D = that sign   else: verdict_D = UNDETERMINED
if all v_G equal:  verdict_G = that sign   else: verdict_G = UNDETERMINED
OUTPUT: (verdict_D, verdict_G), Class, per-estimator disagreement flags, per-estimator verdicts
```

**Why this design.** We chose a *set-valued equivalence class* over a single point estimate plus error bars because the diagnostic functionals D and G are nonlinear in the difficulty vector, so variance propagation through a scalar standard error is not faithful; accepting the cost that the class must be sampled (200 draws) rather than reported in closed form. We chose to *merge* estimator covariances into one conservative envelope rather than report three separate verdicts because the research goal is estimator-agnostic identifiability, accepting the cost that a single noisy estimator (the LLM ranker) can widen the class and push more verdicts to UNDETERMINED. We chose seed-level resampling (k=5 seeds) to inflate the covariance over a purely analytic Fisher-information estimate because seed variance is the dominant unmodeled term in the frontier paper's single-run setup, accepting the cost of k× more evaluation runs. We chose a sign-based invariance test over a magnitude test because the diagnostic verdict — not the numerical value — is what downstream users act on, accepting the cost that near-threshold cases are reported as undetermined rather than resolved. This differs from the closest prior method, the point-estimate diagnostics of 'Your Benchmark Is an Empirical Measure Over Difficulty', in that a domain expert would not describe VIDEC as that diagnostic with error bars: it reports a *verdict over a class*, not a number with a confidence interval.

**Why it measures what we claim.** The Mahalanobis envelope measures *estimator-agnostic identifiability* of the diagnostic because the assumption that all three estimators' calibrated difficulty vectors lie within a shared chi-square envelope of the truth makes membership in the class equivalent to being statistically indistinguishable from the observed data; this assumption fails when the LLM ranker is systematically biased (e.g., prefers longer problems) or when the affine calibration map is misspecified, in which case the class reflects ranker bias or equating error rather than sampling noise. The sign-invariance test measures *verdict robustness* because the assumption that the diagnostic verdict is a step function of the difficulty vector makes constancy of the sign over the class equivalent to the verdict being forced by the data; this assumption fails when D or G is discontinuous at the threshold tau, in which case a verdict can flip inside the class without any member being near the boundary. The seed-resampling covariance measures *resampling stability* because the assumption that seeds are exchangeable draws makes the bootstrap covariance an unbiased estimate of seed-level variance; this assumption fails when seeds are correlated (shared templates), in which case the covariance is underestimated and the class is too narrow.

## Contribution

(1) A framework, VIDEC, that replaces the point-estimate difficulty vector in benchmark diagnostics with a resampling-stable equivalence class and reports a diagnostic verdict only when it is invariant across that class, yielding an estimator-agnostic certified verdict. (2) A design principle: the equivalence class is constructed by taking the conservative envelope of per-estimator measurement-error covariances, so that an estimator-agnostic verdict is obtained without choosing a single difficulty estimator. (3) A diagnostic protocol that flags per-estimator disagreement and near-threshold cases as UNDETERMINED, making explicit when a benchmark's mismatch or coverage verdict is an artifact of the estimator rather than a property of the data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | 25 benchmarks from "Empirical Measure" (7 domains) | Matches prior diagnostic scope, enables direct comparison |
| Primary metric | Fraction of items with certified non-UNDETERMINED verdict | Measures identifiability, the core claim |
| Secondary metric | Empirical coverage of the equivalence class on synthetic data with known ground-truth difficulty (nominal 95%) | Validates the shared-envelope assumption |
| Baseline | Point-estimate diagnostics (no error bars) | Tests need for set-valued equivalence class |
| Baseline | Bootstrap CI per estimator, then majority vote | Tests need for merged conservative envelope |
| Baseline | Single-run diagnostics with analytic Fisher covariance | Tests need for seed-level resampling inflation |
| Baseline | Generic ellipsoid-intersection baseline (no calibration) | Tests need for per-estimator affine calibration |
| Ablation-of-ours | VIDEC without seed resampling (analytic covariance only) | Isolates contribution of resampling-based covariance inflation |
| Ablation-of-ours | VIDEC without per-estimator calibration (shared envelope on raw scales) | Isolates contribution of scale-linking |

### Why this setup validates the claim

The central claim is that VIDEC yields *certified* verdicts about measure-mismatch and coverage-gap that are robust to estimator choice and seed noise, whereas point-estimate diagnostics do not. The 25-benchmark dataset is the exact corpus on which the closest prior method was demonstrated, so any difference in certification rate is attributable to the method, not the data. The primary metric — the fraction of items receiving a non-UNDETERMINED verdict — directly operationalizes identifiability: if VIDEC's equivalence class is too wide, this fraction collapses; if the class is appropriately sized, the fraction is high. The secondary coverage metric on synthetic data with known ground-truth difficulty tests the load-bearing shared-envelope assumption: if the class covers the truth at the nominal 95% rate, the assumption that the calibrated estimators are consistent with a single truth is supported. The three baselines each test a distinct sub-claim: the point-estimate baseline tests whether error bars are necessary at all; the per-estimator majority-vote baseline tests whether merging covariances is better than voting; the analytic-covariance baseline tests whether seed resampling is necessary; the generic ellipsoid-intersection baseline tests whether per-estimator affine calibration is necessary. The ablations isolate the contributions of seed resampling and calibration from the rest of the pipeline. If the method works as claimed, we expect the point-estimate baseline to produce verdicts that flip under small perturbations (low stability), the majority-vote baseline to produce many UNDETERMINED verdicts because the LLM ranker disagrees, the analytic-covariance baseline to produce overconfident (too many non-UNDETERMINED) verdicts that are not stable under seed resampling, and the generic ellipsoid-intersection baseline to produce a class that is either too wide (many UNDETERMINED) or too narrow (unstable) because it conflates scale differences with sampling noise. The metric is the right one because it is a direct count of how often the method can make a decision it is willing to stand behind — exactly what the equivalence-class construction is designed to control.

### Expected outcome and causal chain

**vs. Point-estimate diagnostics (no error bars)** — On a benchmark where the IRT-2PL and Bradley-Terry difficulty estimates differ by a few hundredths near the diagnostic threshold τ, the point-estimate baseline produces a confident mismatch verdict because it treats the single point estimate as truth and ignores estimator disagreement. Our method instead samples the equivalence class, finds that both mismatch and no-mismatch signs occur within the class, and returns UNDETERMINED because the verdict is not invariant. So we expect a noticeable gap in certification rate on benchmarks with high estimator disagreement (e.g., mixed-domain benchmarks) but parity on benchmarks where all estimators agree tightly (e.g., single-domain math).

**vs. Per-estimator bootstrap CI with majority vote** — On a benchmark where the LLM ranker is systematically biased toward longer items, the majority-vote baseline produces a mismatch verdict because two of three estimators (IRT-2PL and Bradley-Terry) agree, but the verdict is not robust to the ranker's bias. Our method instead merges the ranker's covariance into a conservative envelope, widening the class until the ranker's bias is absorbed, and returns UNDETERMINED because the class spans both signs. So we expect the majority-vote baseline to show a higher non-UNDETERMINED rate but lower stability under seed perturbation, while our method shows a lower non-UNDETERMINED rate concentrated on benchmarks where the ranker is known to be biased (e.g., those with length-confounded items).

**vs. Single-run diagnostics with analytic Fisher covariance** — On a benchmark with high seed variance (e.g., items drawn from a small pool of templates), the analytic-covariance baseline produces a confident gap verdict because Fisher information underestimates seed-level variance, making the class too narrow. Our method instead inflates the covariance via seed-level resampling (k=5), widening the class until it covers the observed seed-to-seed variation, and returns UNDETERMINED because the gap sign flips across resamples. So we expect the analytic-covariance baseline to show an inflated non-UNDETERMINED rate that drops sharply when re-evaluated on held-out seeds, while our method's non-UNDETERMINED rate remains stable across seed splits.

**vs. Generic ellipsoid-intersection baseline (no calibration)** — On a benchmark where IRT-2PL and Bradley-Terry difficulties are on different scales (e.g., logit vs. log-strength), the generic ellipsoid-intersection baseline intersects raw-scale ellipsoids, producing a class that is either empty (forcing UNDETERMINED) or dominated by the estimator with the largest covariance. Our method instead applies per-estimator affine calibration on anchor items before intersecting, so the class reflects sampling noise conditional on the equating map. So we expect the generic baseline to show a much lower non-UNDETERMINED rate on benchmarks with scale mismatch, while our method's rate is higher and stable.

### What would falsify this idea

If our method's non-UNDETERMINED rate is uniformly high across all benchmarks — including those where the LLM ranker is known to be biased and those with high seed variance — rather than being selectively lower on exactly those hard subsets, or if the empirical coverage of the equivalence class on synthetic data with known ground-truth difficulty is far below the nominal 95%, then the equivalence class is not capturing the intended uncertainty and the central claim of certified identifiability is wrong.

## References

1. Your Benchmark Is an Empirical Measure Over Difficulty
2. Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation
3. Fluid Language Model Benchmarking
4. Estimating problem difficulty without ground truth using Large Language Model comparisons
5. RIDE: Difficulty Evolving Perturbation with Item Response Theory for Mathematical Reasoning
6. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations
7. metabench -- A Sparse Benchmark of Reasoning and Knowledge in Large Language Models
8. Improving Model Evaluation using SMART Filtering of Benchmark Datasets
