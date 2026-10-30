# Weighted Validity Domains for Interleaved-Generation Benchmarks

## Motivation

IntReal's evaluation suite reports seven expert-model dimension scores as point estimates calibrated only to four curated real-world categories, with no mechanism that guarantees those scores remain diagnostic when a generated model's interleaved outputs drift outside the curated distribution. The structural root cause is that IntReal treats expert scoring as a fixed measurement function, so the same score is reported regardless of whether the evaluated output lies in the region where the expert was ever validated; this is the benchmark-level instance of the broader problem that unified interleaved generators such as BAGEL (Emerging Properties in Unified Multimodal Pretraining) produce outputs whose distribution is not controlled by the benchmark designer, and no branch of prior work establishes score validity under that shift.

## Key Insight

A benchmark score is only diagnostic on the support where its expert was calibrated, so attaching a density-ratio-weighted effective sample size to each dimension score converts an uninterpretable point estimate into a score with an explicit, testable validity domain.

## Method

**What it is.** We introduce CALIB-IntReal, a calibration-and-invariance layer that wraps the IntReal expert-model suite. Its inputs are (i) the IntReal-100K category statistics and the frozen IntReal feature extractor phi, (ii) an evaluated model's interleaved outputs, and (iii) the seven expert-model dimension scores; its outputs are, per dimension, a calibrated score with a confidence interval and a binary diagnostic flag.

**Load-bearing assumption (stated explicitly).** CALIB-IntReal assumes that the weighting variable used in Phase 3 captures the shift-relevant features of each expert's calibration region. In the original design this variable was the coarse 4-category label c(x); the repair below replaces it with the frozen feature vector phi(x) so that the density ratio is estimated per output rather than per category. Sufficiency of the weighting variable is not assumed silently: Phase 2b tests it, and if the test fails the partition is refined (see Phase 2b).

**How it works.**

```pseudocode
# Phase 1: per-category expert calibration (offline, on IntReal-100K)
for expert e in 7 dimensions:
    for category c in 4 IntReal categories:
        fit score distribution D[e,c] on held-out IntReal-100K samples of c
        store mean mu[e,c], variance sigma2[e,c], n[e,c]

# Phase 2: evaluated-model distribution estimation
for each evaluated interleaved output x:
    assign category c(x) via the IntReal category classifier
    compute embedding phi(x) with the frozen IntReal feature extractor   # dim d = 768
estimate p_eval over categories as empirical frequencies

# Phase 2b: sufficiency test for the weighting variable (per dimension e)
# Regress expert score s_e(x) on phi(x) WITHIN each category c, on held-out IntReal-100K:
#   s_e(x) = a_e,c + b_e,c^T phi(x) + eps,   fit by ridge (lambda=1.0) on n[e,c] samples
# Report R2_within[e,c] = fraction of within-category score variance explained by phi(x).
# If max_c R2_within[e,c] > 0.10 for dimension e, the category partition is INSUFFICIENT
#   for e; refine by replacing c(x) with the k-means partition of phi(x) into
#   K = 16 clusters (fit on IntReal-100K, fixed seed), and recompute D[e,.] per cluster.
# Record sufficiency_status[e] in {SUFFICIENT, REFINED}.

# Phase 3: importance weighting against curated priors (per-output, in phi-space)
# Density ratio estimated by a logistic discriminator on phi(x):
#   train g(z) = sigmoid(w^T z + b) to separate curated IntReal-100K embeddings (label 1)
#   from evaluated-model embeddings (label 0); w in R^768, L2 reg 1e-3, 200 epochs Adam lr=1e-3.
#   ratio r(x) = g(phi(x)) / (1 - g(phi(x)))   (Sugiyama et al. density-ratio form)
#   clip r(x) to [0.01, 100] to bound variance.
for each dimension e:
    for each output x:
        w(x) = r(x)                               # per-output density-ratio weight
        s_e(x) = expert_e(x)                      # raw dimension score
    ESS_e = (sum_x w(x))^2 / sum_x w(x)^2         # effective sample size (per-output weights)
    score_e = sum_x w(x) s_e(x) / sum_x w(x)      # weighted score

# Phase 4: calibrated CI and closure rule
for each dimension e:
    CI_e = bootstrap_percentile(score_e, B=1000, weights=w)   # 95% interval
    if ESS_e < tau_ESS (tau_ESS = 30):
        flag_e = NON_DIAGNOSTIC
    else:
        flag_e = DIAGNOSTIC
return {(score_e, CI_e, flag_e, sufficiency_status[e]) for e in 1..7}
```

**Why this design.** We chose a density-ratio importance-weighting scheme over retraining a fresh expert for each new output distribution because the IntReal experts are already validated on the curated categories and retraining would discard that validation, accepting the cost that weights degenerate when the evaluated distribution has near-zero overlap with the curated categories. We estimate the ratio per output in the frozen feature space phi(x) rather than per category because the coarse 4-category partition is not guaranteed to be a sufficient statistic for each expert's calibration behavior, and Phase 2b makes that sufficiency testable rather than assumed. We chose a bootstrap percentile interval over a Gaussian interval because dimension scores are bounded and skewed and Gaussian intervals would misstate coverage, accepting the cost of 1000 resamples per dimension. We chose a hard ESS closure threshold over a soft warning because a diagnostic instrument must be able to declare a score uninterpretable rather than emit a wide interval that downstream users may still read as a number, accepting the cost that some borderline cases are flagged conservatively. A domain expert would not describe this as a variant of standard importance-sampling evaluation (e.g., the offline-policy-evaluation estimators used in RL) because those estimators assume the target policy's support is contained in the logging policy's support and report no per-dimension closure rule; CALIB-IntReal makes the support condition itself the reported object via ESS, which is the quantity that turns a benchmark score into a validity-domain statement.

**Why it measures what we claim.** The weighted score `score_e` measures the expert's dimension judgment on the curated distribution because the assumption that the per-output density ratio r(x) in phi-space captures the expert's calibration-relevant shift (verified by the Phase 2b within-category R2 test, with refinement to K=16 clusters when R2_within > 0.10) makes reweighting by r(x) equivalent to reweighting by the expert's actual input distribution; this assumption fails when two outputs with similar phi(x) differ in features the expert is sensitive to but phi does not encode, in which case the weighted score reflects the feature extractor's partition rather than the expert's true calibration region, and sufficiency_status[e] reports REFINED. The effective sample size `ESS_e` measures the diagnostic validity of the score because the assumption that weight concentration tracks distributional overlap with the calibration set makes low ESS equivalent to a score supported by too few effective curated-like samples; this assumption fails when the evaluated distribution has high overlap but extreme per-output weight variance within a region, in which case ESS reflects weight dispersion rather than genuine support loss. The bootstrap CI measures uncertainty in the weighted score because the assumption that resampling with the importance weights preserves the score's sampling distribution makes percentile coverage valid; this assumption fails when weights are heavy-tailed, in which case the interval reflects weight-tail variability rather than sampling variability.

## Contribution

(1) CALIB-IntReal, a calibration-and-invariance layer that attaches per-dimension calibrated confidence intervals and an explicit validity domain to the IntReal expert-model suite via category-level importance weighting and an effective-sample-size closure rule. (2) The design principle that a benchmark score should be reported jointly with the effective sample size of its calibration support, which makes distribution shift a first-class reported quantity rather than a silent assumption. (3) A per-dimension diagnostic flag that converts the IntReal suite from a point-estimate benchmark into an instrument that can declare its own scores non-diagnostic when the evaluated model leaves the calibrated region.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | IntReal-100K with held-out category subsets, plus pre-specified shifted subsets (see shift protocol) | Provides curated priors, category labels, and known support status |
| Primary metric | Diagnostic validity rate: fraction of outputs where flag matches the externally-defined support condition | Directly measures whether the closure rule tracks support loss |
| External criterion | Held-out human/expert re-scoring agreement on shifted subsets (fraction of flagged cases where re-scoring disagrees with reported score) | Provides non-circular ground truth for support loss |
| Baseline | Raw expert scores (no calibration) | Tests whether calibration and closure add value beyond raw scores |
| Baseline | Gaussian CI on weighted scores | Tests whether bootstrap percentile intervals are necessary for skewed bounded scores |
| Baseline | Soft ESS warning (no hard flag) | Tests whether a hard diagnostic closure rule is needed |
| Baseline | Category-level density ratio (original Phase 3) | Tests whether per-output phi-space weighting is necessary |
| Ablation-of-ours | CALIB-IntReal without importance weighting (uniform weights) | Isolates the effect of density-ratio reweighting |
| Ablation-of-ours | CALIB-IntReal with Phase 2b sufficiency test disabled (always use 4-category partition) | Isolates the contribution of the sufficiency test and refinement |

**Shift protocol (pre-specified, reproducible).** Shifted subsets are generated by a released script with fixed seed 0: (S1) balanced mixture of the 4 curated categories (known support); (S2) 90/10 concentration on one curated category (known partial support); (S3) controlled mixture of 70% curated categories plus 30% synthetic out-of-category interleaved formats (known support loss); (S4) 100% synthetic out-of-category formats (known no support). Support status per subset is recorded before any scoring, so the diagnostic-validity metric has a ground truth independent of ESS.

### Why this setup validates the claim

The central claim is that CALIB-IntReal turns expert-model scores into calibrated, support-aware diagnostics by combining per-output importance weighting in phi-space, bootstrap intervals, and a hard ESS closure rule, with the sufficiency of the weighting variable tested rather than assumed. The dataset choice is essential because IntReal-100K provides both curated priors and held-out samples where the evaluated distribution can be deliberately shifted away from the curated distribution, and the pre-specified shift protocol (S1–S4) creates the support-mismatch condition the method is designed to detect with a known ground-truth label. The primary metric, diagnostic validity rate, is the right one because it directly operationalizes the claim: a score should be flagged NON-DIAGNOSTIC precisely when its effective sample size falls below the threshold, and DIAGNOSTIC otherwise, and the external criterion (held-out human/expert re-scoring agreement) breaks the circularity by checking whether flagged cases are the ones where the reported score actually disagrees with an independent re-score. Raw expert scores test the sub-claim that calibration is necessary; Gaussian CI tests the sub-claim that bootstrap percentile intervals are necessary for bounded skewed scores; soft ESS warning tests the sub-claim that a hard closure rule is needed; the category-level density-ratio baseline tests the sub-claim that per-output phi-space weighting is necessary; the no-weighting and no-sufficiency-test ablations isolate the contributions of density-ratio reweighting and of the Phase 2b sufficiency check. Together, these conditions form a falsifiable test: if the method works, diagnostic validity rate should be high, the gap between DIAGNOSTIC and NON-DIAGNOSTIC cases should be large, and flagged cases should concentrate on S3/S4 where the external criterion shows re-scoring disagreement; if the method fails, the flag will not track support loss and the external criterion will not separate flagged from unflagged cases.

### Expected outcome and causal chain

**vs. Raw expert scores (no calibration)** — On S2 (90/10 concentration on one IntReal category) while the curated prior is balanced, raw expert scores produce systematically biased dimension estimates because they ignore the category frequency shift. Our method instead reweights each output by the per-output density ratio r(x) estimated in phi-space, correcting the shift before aggregating, so we expect raw scores to show a noticeable gap from curated-reference scores on S2 but parity on S1, while CALIB-IntReal maintains stable scores across both.

**vs. Gaussian CI on weighted scores** — On a dimension score bounded near 1 and heavily left-skewed due to many near-perfect expert judgments, the Gaussian interval produces symmetric bounds that extend beyond the score's valid range and misstate coverage because it assumes normality. Our method instead uses bootstrap percentile intervals, which respect the bounded skewed distribution and yield asymmetric bounds that stay within range, so we expect the Gaussian interval to show poor coverage (true score outside the nominal 95% interval noticeably more than 5% of the time) on skewed dimensions while CALIB-IntReal's interval achieves near-nominal coverage.

**vs. Soft ESS warning (no hard flag)** — On S4 (100% synthetic out-of-category interleaved formats), the soft ESS warning emits a wide interval but still returns a numeric score that downstream users may interpret as valid, because it lacks a closure rule. Our method instead sets flag_e = NON-DIAGNOSTIC when ESS_e < 30, explicitly declaring the score uninterpretable, so we expect the soft-warning baseline to produce a substantial fraction of low-ESS cases that are still reported as numeric scores (e.g., 30-50% of low-ESS cases unflagged), while CALIB-IntReal flags all such cases, yielding a higher diagnostic validity rate against the S1–S4 ground truth.

**vs. Category-level density ratio (original Phase 3)** — On S3 (70% curated plus 30% synthetic out-of-category formats), the category-level ratio is constant within each category and cannot detect within-category drift in phi-space, so its ESS reports high support precisely when within-category drift is worst. Our method instead estimates r(x) per output in phi-space and computes ESS on those per-output weights, so we expect the category-level baseline to show high ESS on S3 (failing to flag) while CALIB-IntReal's per-output ESS falls below tau_ESS and flags S3, with the external re-scoring criterion confirming that S3 scores are the ones that disagree with independent re-scoring.

**vs. Ablation without importance weighting (uniform weights)** — On S2, uniform weights reproduce the raw-score bias because they ignore the category frequency shift, so we expect the no-weighting ablation to match raw scores on S2 and diverge from CALIB-IntReal, isolating the contribution of density-ratio reweighting.

**vs. Ablation with sufficiency test disabled (always 4-category partition)** — On S3, disabling Phase 2b forces the coarse partition even when within-category R2_within exceeds 0.10, so we expect this ablation to reproduce the category-level baseline's failure to flag S3, isolating the contribution of the sufficiency test and refinement.

### What would falsify this idea

If the diagnostic validity rate of CALIB-IntReal is no better than the soft ESS warning or category-level density-ratio baselines against the S1–S4 ground truth, or if the flag fails to concentrate on S3/S4 where the external re-scoring criterion shows disagreement (e.g., high flag rate on S1 but low flag rate on S3/S4), then the central claim that ESS-based closure turns benchmark scores into validity-domain statements is wrong. Additionally, if the bootstrap percentile interval shows no coverage advantage over the Gaussian interval on skewed bounded dimensions, the claim that bootstrap is necessary would be falsified.

## References

1. IntReal: A Real-World Dataset and Benchmark for Multimodal Interleaved Generation
2. Emerging Properties in Unified Multimodal Pretraining
3. Causal Diffusion Transformers for Generative Modeling
