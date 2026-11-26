# Cross-Benchmark Gap Calibration for Sim-to-Real Transfer

## Motivation

Sim-to-real gap estimators are internally defined by each method—SPiDR uses a pessimism-based uncertainty set, co-training uses an OT distance, DEAL uses a discriminator score, and inductive SBI uses a posterior shift—so a gap estimate of, say, 0.3 from one method on one benchmark is not comparable to a gap estimate of 0.3 from another method on another benchmark. Real2SimLab evaluates each method with benchmark-native scorers but does not expose a shared reference quantity that would let these heterogeneous proxies be placed on a common scale, leaving the field unable to say which method actually transfers better across benchmarks.

## Key Insight

Benchmark-native scores provide an external, method-agnostic anchor for the true sim-to-real gap; calibrating each method's internal proxy against this anchor yields comparable gap estimates even though the proxies themselves are structurally different.

## Method

**What it is** — CGC (Cross-benchmark Gap Calibration) is a calibration layer added to Real2SimLab that takes any method's internal gap proxy and a set of benchmark-native scores, and outputs a calibrated gap estimate on a shared reference scale. Inputs: a method's proxy values on a set of (simulation, real) pairs, and the benchmark's native success scores for the same pairs. Output: a scalar calibrated gap per pair and a cross-benchmark comparable ranking of methods.

**Stated measurement model (load-bearing assumption, made explicit)** — CGC assumes there exists a single latent true sim-to-real gap `g*[m,t]` for method m on task t that is shared across benchmarks, and that each benchmark's native relative gap is a monotone, benchmark-offset distortion of it: `native[m,b,t] = g_b(g*[m,t]) + offset_b + ε`, where `g_b` is monotone (not necessarily linear) and `offset_b` is a benchmark-specific additive difficulty/scorer offset identified up to the anchor. Under this model, per-benchmark isotonic fits recover `g_b` up to the offset, and the anchor shift removes `offset_b`. This assumption fails when (i) the benchmark-native scorer rewards simulation-specific artifacts so `native` reflects scorer bias rather than transfer gap, (ii) the proxy is non-monotone (e.g., DEAL's discriminator under mode collapse), or (iii) benchmark difficulty enters multiplicatively rather than additively on the native scale. CGC reports a per-benchmark monotonicity diagnostic (Spearman ρ between proxy and native within benchmark) and flags any proxy with ρ below 0.3 as failing the monotonicity assumption rather than silently rank-transforming it.

**How it works** —
```pseudocode
# Phase 0: Data-availability audit (run before any fitting)
for each benchmark b in Real2SimLab's 6 benchmarks:
    record whether paired (sim_score, real_score) exist for >= 20 (method, task) pairs
    record whether each method's internal proxy is obtainable for those pairs
qualifying_benchmarks = {b : both conditions hold}
if |qualifying_benchmarks| < 3:
    augment with synthetic benchmark generator (known ground-truth gap) to reach >= 20 pairs per benchmark

# Phase 1: Collect proxy and native scores
for each method m, benchmark b, task t:
    proxy[m,b,t]   = method m's internal gap proxy (SPiDR uncertainty, OT distance, DEAL discriminator score, SBI posterior shift)
    native[m,b,t]  = 1 - (real_score / sim_score)   # benchmark-native relative gap
    # guard: if sim_score < 0.05 or sim_score > 0.95, mark native as non-identifiable and exclude from fit

# Phase 2: Per-benchmark monotone calibration + monotonicity diagnostic
for each benchmark b in qualifying_benchmarks:
    rho_b = Spearman(proxy[m,b,t], native[m,b,t]) over all (m,t)
    if rho_b < 0.3: flag proxy as non-monotone; report rank-order artifact, do not claim calibrated gap
    fit isotonic regression f_b: proxy[m,b,t] -> native[m,b,t]
    # isotonic regression chosen for monotonicity without assuming linearity
    # hyperparameter: bin count = 20 for isotonic fit; minimum 20 pairs per benchmark

# Phase 3: Cross-benchmark alignment via paired anchor set
# anchor set A = methods evaluated on >= 2 qualifying benchmarks (paired, not a single benchmark's mean)
for each benchmark b:
    shift_b = mean over (m,t) in A of [ f_b(proxy[m,b,t]) - mean_over_b'(f_b'(proxy[m,b',t])) ]
    calibrated[m,b,t] = f_b(proxy[m,b,t]) - shift_b
# reference scale R = mean(native[m,b,t]) over the paired anchor set A

# Phase 4: Rank methods
for each method m:
    CGC_score[m] = mean over b,t of calibrated[m,b,t]
    rank methods by CGC_score
```

**Why this design** — We chose isotonic regression over linear regression because the relationship between internal proxies and native scores is monotone but not necessarily linear (e.g., DEAL's discriminator score saturates), accepting the cost that isotonic regression needs more data points per benchmark (at least 20 pairs) to avoid overfitting. We chose a paired anchor set of methods evaluated on at least two benchmarks for the reference scale rather than a single anchor benchmark's mean because a single benchmark's mean is itself defined by a benchmark-specific scorer, so the shift would absorb scorer bias rather than transfer difficulty; accepting the cost that the anchor set must contain methods with multi-benchmark coverage. We chose to calibrate each method's proxy separately rather than fitting a joint model across methods because proxies live on different scales and joint fitting would require a shared parametric form that none of the methods satisfy, accepting the cost that cross-method comparisons rely on the calibration quality of each individual fit. We added the Phase-0 data-availability audit and the per-benchmark Spearman diagnostic because the isotonic fit and the monotonicity claim are only valid when paired data exist and the proxy is monotone; accepting the cost that non-monotone proxies (DEAL under mode collapse) are flagged rather than calibrated. A domain expert would not describe this as a variant of Platt scaling or temperature scaling because those methods calibrate classifier probabilities against a single ground-truth label, whereas CGC calibrates a structural gap proxy against a benchmark-native relative gap that is itself derived from paired simulation and real-world scores, and the cross-benchmark shift step has no analogue in standard calibration.

**Why it measures what we claim** — The computational quantity `native[m,b,t] = 1 - (real_score / sim_score)` measures the true sim-to-real gap because we assume that the benchmark-native success score is a sufficient statistic for task performance and that the simulation score is an unbiased estimate of in-simulation performance; this assumption fails when the benchmark's native scorer is itself sensitive to simulation artifacts (e.g., a scorer that rewards simulation-specific visual features) or when sim_score is near 0 or 1 (ratio non-identifiable), in which case `native` reflects scorer bias or division blow-up rather than transfer gap, and we exclude such pairs via the Phase-1 guard. The isotonic calibration `f_b` measures the monotone mapping from proxy to true gap because we assume that the proxy is monotonically related to the true gap within a benchmark (stated measurement model, monotone `g_b`); this assumption fails when the proxy is non-monotone (e.g., DEAL's discriminator score can be high for both very good and very bad simulations due to mode collapse), in which case the calibrated value reflects the proxy's rank order rather than the true gap, and the Phase-2 Spearman diagnostic (ρ < 0.3) flags this rather than silently rank-transforming. The cross-benchmark shift `shift_b` measures benchmark difficulty offset because we assume that the paired anchor set's mean native gap is representative of the average transfer difficulty across benchmarks; this assumption fails when the anchor set is unrepresentative (e.g., contains only methods that are strong on one benchmark family), in which case the calibrated scores are shifted by a constant that does not reflect true cross-benchmark differences.

## Contribution

['A calibration procedure (CGC) that maps heterogeneous internal sim-to-real gap proxies onto a shared benchmark-native reference scale, enabling cross-benchmark comparison of gap estimates for the first time.', 'A design principle: benchmark-native scores can serve as external anchors for calibrating structural gap proxies, with isotonic regression providing monotone calibration without assuming linearity.', 'An extension to Real2SimLab that exposes calibrated gap estimates alongside native scores, enabling method ranking across benchmarks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Real2SimLab's 6 benchmarks, 47 coding agents + 5 non-agent methods, restricted to qualifying benchmarks from Phase-0 audit (≥20 paired (sim,real) points per benchmark) | Only source with paired sim/real scores across methods |
| Dataset (augmentation) | Synthetic benchmark generator with known ground-truth gap, used if fewer than 3 benchmarks qualify | Supplies ≥20 pairs per benchmark for isotonic fit |
| Primary metric | Spearman rank correlation between CGC_score and held-out native gap | Rank agreement is the actual claim being made |
| Secondary metric | Per-benchmark Spearman ρ between proxy and native (monotonicity diagnostic) | Flags non-monotone proxies (e.g., DEAL) before calibration |
| Baseline | Uncalibrated raw proxy ranking | Tests whether calibration adds anything over raw proxies |
| Baseline | Global linear rescaling of proxies | Tests whether monotone nonlinearity matters |
| Baseline | Per-benchmark z-score without cross-benchmark shift | Tests whether the anchor-shift step matters |
| Ablation-of-ours | CGC with linear fit instead of isotonic | Isolates the monotonicity assumption's contribution |
| Ablation-of-ours | CGC with single-anchor-benchmark shift instead of paired anchor set | Isolates the paired-anchor repair's contribution |

### Why this setup validates the claim

The central claim is that raw internal gap proxies are not comparable across benchmarks, and that a monotone per-benchmark calibration plus a paired-anchor shift produces a cross-benchmark comparable ranking that tracks the benchmark-native relative gap better than any uncalibrated alternative. The uncalibrated raw proxy baseline directly tests the necessity of calibration: if raw proxies already rank methods in the same order as native gaps, CGC is doing nothing. The global linear rescaling baseline tests the specific monotonicity-without-linearity claim — if a single linear map suffices, our isotonic choice is unjustified complexity. The per-benchmark z-score baseline isolates the anchor-shift step: it has per-benchmark normalization but no shared reference scale, so if it matches CGC, the shift step is inert. The linear-fit ablation is the sharpest test of the design rationale, since it keeps everything except the isotonic form. The single-anchor-benchmark ablation tests the load-bearing assumption repair: if a single benchmark's mean shift matches the paired-anchor shift, the paired-anchor repair is unnecessary. The Phase-0 data-availability audit and the per-benchmark Spearman diagnostic make the measurement-model assumptions testable rather than assumed: benchmarks with fewer than 20 paired points or proxies with ρ < 0.3 are reported as failing the model rather than silently calibrated. Spearman rank correlation on held-out pairs is the right metric because the deliverable is a ranking of methods, not a point estimate of gap magnitude; a method that gets magnitudes right but order wrong has failed the stated purpose. Held-out evaluation prevents the calibration fit from trivially reproducing the ranking it was trained on.

### Expected outcome and causal chain

**vs. Uncalibrated raw proxy ranking** — On a case where two methods have proxies on different scales (e.g., DEAL's discriminator score bounded near 1 versus SBI's posterior shift unbounded above), the raw-proxy baseline produces an ordering driven by scale artifacts rather than transfer quality because it never places proxies on a common axis. Our method instead fits each proxy to its benchmark's native gap and shifts to a shared paired-anchor scale, so we expect raw-proxy ranking to scramble method order across benchmarks while CGC's Spearman correlation with native gaps stays clearly higher, with the gap concentrated on cross-benchmark comparisons and near-parity within a single benchmark.

**vs. Global linear rescaling of proxies** — On a case where the proxy saturates (DEAL's discriminator score compresses near its ceiling for both good and mediocre simulations), the linear baseline produces a calibrated value that keeps rising past the saturation point and misranks the saturated region because a single slope cannot bend. Our method's isotonic fit flattens in the saturated region, so we expect the linear baseline to lose rank correlation specifically on the high-proxy tail while matching CGC in the linear middle range — a tail-localized gap rather than a uniform one. If the Phase-2 Spearman diagnostic flags DEAL's proxy as non-monotone (ρ < 0.3), we report that the isotonic fit is a rank-order artifact for that proxy and exclude it from the calibrated ranking claim.

**vs. Per-benchmark z-score without anchor shift** — On a case where one benchmark is systematically easier (native gaps clustered low) than another, the z-score baseline produces per-benchmark scores centered on each benchmark's own mean, so a method that is mediocre everywhere can outrank a method that is strong on the hard benchmark because the easy benchmark's low-variance scores inflate its standardized values. Our method instead subtracts a paired-anchor-derived shift, so we expect z-score to produce rank inversions between easy and hard benchmarks that CGC does not, visible as higher cross-benchmark rank disagreement for z-score than for CGC.

**vs. CGC with single-anchor-benchmark shift (ablation)** — On a case where the single anchor benchmark is systematically easier than the others, the single-anchor shift absorbs that benchmark's scorer bias into the reference scale, so methods strong on the hard benchmarks are mis-shifted. Our method's paired-anchor set averages over methods evaluated on ≥2 benchmarks, so we expect the single-anchor ablation to lose rank correlation specifically on cross-benchmark comparisons involving the non-anchor benchmarks, while matching CGC on the anchor benchmark itself.

### What would falsify this idea

If the uncalibrated raw-proxy ranking matches CGC's Spearman correlation with held-out native gaps as closely as any calibrated variant, then the calibration layer adds nothing and the central claim is wrong. Likewise, if CGC's advantage over the linear-fit ablation is uniform across all proxy ranges rather than concentrated in the predicted saturation region, the monotonicity rationale is not the operative mechanism; and if the Phase-0 audit finds fewer than 3 qualifying benchmarks and the synthetic augmentation does not recover the known ground-truth scale, the measurement model is unverified.

## References

1. Real2SimLab: A Unified Platform for Real-to-Simulation Transfer
