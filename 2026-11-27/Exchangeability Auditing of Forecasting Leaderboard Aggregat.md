# Exchangeability Auditing of Forecasting Leaderboard Aggregates via Configuration Resampling

## Motivation

Leaderboard resolution reporting treats a board as a measuring instrument but only quantifies pairwise noise between adjacent entries; it never validates the aggregate score itself as an estimator of expected transfer error. The root cause is structural: rank aggregation over a fixed configuration set silently imports an exchangeability assumption over datasets and horizons, so the aggregate is reported as a point estimate even though the resolution analysis itself shows that dataset-level clustering and correlated configurations dominate the sampling variability (the 'What a Forecasting Leaderboard Can Establish' report finds resolution exceeds the median neighbour gap on every board built from 15–39 public sources, and that pool-ranking rules reversed 43 untouched top-ten pairs). Because no diagnostic separates 'the aggregate converged' from 'the configuration frame is too small to converge', the displayed order can rest on an untested sampling assumption rather than on a stable target.

## Key Insight

The aggregate score of a leaderboard is a sample mean over a clustered configuration frame, so its stability is identifiable from the shape of its resampling curve — if the bootstrap standard error fails to shrink toward a plateau as configurations are added, the aggregate is not estimating a fixed expected-transfer-error target and the reported order is an artifact of frame composition rather than a property of the models.

## Method

**(A) What it is.** CONVERGE (Configuration-resampling Order-stability and Variance Estimator) takes a board's published per-configuration score matrix (models × configurations, with each configuration tagged by its source dataset, sampling frequency, and horizon) and returns, per model and per pair of models, a frame-sensitivity diagnostic: the bootstrap standard error of the aggregate as a function of the number of datasets resampled, reported against an explicit exchangeable-null reference band, plus a plateau flag that compares the observed curve to that band rather than to a fixed exponent threshold.

**Stated load-bearing assumption (explicit).** CONVERGE treats the board's datasets as an exchangeable draw from a super-population of transfer datasets stratified by (domain, sampling frequency, horizon). Under this assumption, drawing whole datasets with replacement imitates the sampling design the board claims to represent, and the SE-vs-k curve's shape reflects convergence toward a fixed expected-transfer-error target. When the datasets are a convenience sample (as the source report argues), the bootstrap distribution describes within-frame heterogeneity, not transfer uncertainty; CONVERGE therefore reports the diagnostic as a frame-sensitivity audit and calibrates every verdict against an exchangeable null with the same within-dataset covariance, so the contribution survives even when exchangeability is false.

**(B) How it works.**

```pseudocode
INPUT: S[m, c]  # score of model m on configuration c (lower=better)
       cluster[c] = (dataset_id, freq, horizon)  # two-level nesting: configs in datasets
       B = 2000 bootstrap replicates; grid K = [8,16,24,...,97]; alpha = 0.05
       strata[d] = (domain, freq, horizon) for each dataset d

# Phase 0: exchangeability check (variance decomposition)
for each stratum s:
    compute between-dataset variance / total variance of S[m, c] within s
    report ICC_s[m]  # intraclass correlation; high ICC => clustering matters

# Phase 1: cluster-respecting resampling
for k in K:
    for b in 1..B:
        draw k/8 datasets with replacement, stratified by strata[d]  # 8 configs per dataset in GIFT-Eval
        take all configs of each drawn dataset      # preserves within-dataset correlation
        A_b[m,k] = mean over drawn configs of S[m,c]
    SE[m,k] = std_b( A_b[m,k] )                     # bootstrap SE of aggregate

# Phase 2: exchangeable-null reference band
for k in K:
    for b in 1..B_null (B_null = 2000):
        # parametric bootstrap: draw datasets from an exchangeable super-population
        # fitted as a stratified Gaussian with mean and within-dataset covariance estimated from S
        draw k/8 synthetic datasets from the fitted super-population
        A_null_b[m,k] = mean over synthetic configs
    SE_null[m,k] = std_b( A_null_b[m,k] )
    band[m,k] = [quantile_0.025(SE_null), quantile_0.975(SE_null)]

# Phase 3: plateau flag (frame-relative, not absolute)
for m:
    # observed curve vs null band at the full frame k=97
    plateau[m] = (SE[m,97] <= band[m,97].upper) and (SE[m,97] < tau)
    # tau = 0.1 * median neighbour gap (board's own resolution scale)
    # report raw SE curve and null band alongside the flag

# Phase 4: pair-level stability
for (m, m'):
    d_b[k] = A_b[m,k] - A_b[m',k]
    P_stable[m,m'] = fraction of b with sign(d_b[97]) == sign(d_b[8])
    flag pair if P_stable < 1 - alpha

OUTPUT: SE curves, null bands, plateau flags, per-pair stability, and a board-level verdict
        (stable / frame-limited / order-unresolved)
```

**(C) Why this design.** We chose cluster-respecting (dataset-level) resampling over naive configuration-level bootstrap because configurations within a dataset share sampling frequency and horizon and are therefore correlated, so configuration-level resampling would understate SE and manufacture false convergence, accepting the cost that the effective sample size is the number of datasets (28 for GIFT-Eval) rather than 97 configurations. We chose a frame-relative null band over a fixed power-law exponent threshold because the source report establishes that the 28 datasets are a convenience sample, so an absolute gamma >= 0.5 rule cannot distinguish 'non-convergence' from 'the 28-cluster frame is too small to estimate a tail'; the null band is generated by a parametric bootstrap from an explicitly specified exchangeable super-population stratified by (domain, freq, horizon), so the plateau flag compares the observed curve to what an exchangeable frame of the same size and within-dataset covariance would produce. We chose a plateau criterion tied to the board's own median neighbour gap (tau) over an absolute SE cutoff because the resolution report already establishes that gap as the board's meaningful scale, accepting the cost that the diagnostic inherits the gap estimate's own uncertainty. We report the raw SE curve and null band alongside every flag so the parametric tail assumption is auditable. This differs from the resolution report's pairwise test, which estimates the smallest detectable difference between two displayed entries; CONVERGE instead asks whether the aggregate each entry is ranked by has stopped moving as the frame grows, a question the pairwise test is structurally blind to.

**(D) Why it measures what we claim.** The bootstrap standard error SE[m,k] measures aggregate sampling instability because drawing whole datasets with replacement is equivalent to sampling from the population of transfer datasets the board claims to represent; this assumption fails when the 28 datasets are a convenience sample rather than an exchangeable draw, in which case SE reflects within-frame heterogeneity instead of transfer uncertainty. The exchangeable-null band SE_null[m,k] measures the SE an exchangeable frame of the same size and within-dataset covariance would produce because it is generated by a parametric bootstrap from a stratified Gaussian super-population fitted to S; this assumption fails when the fitted Gaussian misrepresents the score distribution's tails, in which case the band reflects the parametric fit rather than exchangeability. The plateau flag measures frame sufficiency relative to the board's resolution because it requires SE[m,97] to fall inside the null band and below tau; this assumption fails when the neighbour-gap distribution is heavy-tailed, in which case the flag reflects the median gap's insensitivity rather than true sufficiency. The pair stability P_stable measures order robustness because sign agreement across resample sizes is the operational form of 'the ranking does not depend on which configurations were drawn'; this assumption fails when both models improve identically with k, in which case stability reflects shared trend rather than genuine separation. The intraclass correlation ICC_s[m] measures whether datasets within a stratum are exchangeable because high between-dataset variance relative to total variance indicates that dataset identity, not sampling noise, drives scores; this assumption fails when strata are too coarse, in which case ICC reflects stratum composition rather than exchangeability.

## Contribution

(1) A resampling-based convergence diagnostic (CONVERGE) that turns a leaderboard's published score matrix into per-model SE-vs-configuration curves, a power-law plateau test, and per-pair order-stability flags, operationalizing aggregate consistency as a measurable property rather than an assumed one. (2) A design principle for leaderboard reporting: aggregate ranks should be accompanied by a convergence verdict that distinguishes 'stable target' from 'frame-limited', so that a displayed order is only interpreted when its aggregate has plateaued relative to the board's own resolution. (3) An open diagnostic script that consumes any board's per-configuration scores and emits the convergence report, applicable beyond GIFT-Eval to BOOM and other shared boards.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GIFT-Eval board score matrix (28 datasets, 97 configs) | Real published board with per-config scores and dataset tags |
| Dataset (fallback) | fev-bench board score matrix (public per-config scores) | Pre-specified fallback if GIFT-Eval ingestion stalls |
| Primary metric | Convergence verdict agreement with held-out-dataset reranking | Directly tests whether aggregate has stabilized |
| Secondary metric | ICC_s[m] per stratum (between/total variance) | Quantifies exchangeability within strata |
| Baseline | Percentile bootstrap over configurations | Naive resampling ignores within-dataset correlation |
| Baseline | Resolution-report pairwise test | Detects pairwise differences, blind to aggregate drift |
| Baseline | Fixed-SE threshold rule | Cannot separate convergence from frame homogeneity |
| Ablation-of-ours | CONVERGE without cluster-respecting resampling | Isolates the value of dataset-level resampling |
| Ablation-of-ours | CONVERGE without exchangeable-null band (fixed gamma >= 0.5) | Isolates the value of frame-relative calibration |

### Why this setup validates the claim

The central claim is that a board's aggregate ranking can be diagnosed as stable, frame-limited, or order-unresolved by comparing its bootstrap standard error curve against an exchangeable-null reference band as configurations accumulate. GIFT-Eval is the right dataset because it is the only board with enough entries (128) and public per-configuration scores to reconstruct the score matrix and to hold out datasets as a pseudo-ground-truth; fev-bench is pre-specified as a fallback with a scripted ingestion path so the experiment does not stall on data access. The percentile-bootstrap baseline tests the sub-claim that correlation within datasets matters: if naive configuration-level resampling yields systematically narrower SE curves and earlier plateau flags than our cluster-respecting version, the correlation argument is confirmed. The resolution-report pairwise test tests the sub-claim that aggregate drift is a distinct question from pairwise detectability: if our plateau flags disagree with the pairwise test exactly on boards where the resolution exceeds the median gap, the two diagnostics are measuring different things. The fixed-SE threshold baseline tests the sub-claim that a frame-relative null band distinguishes genuine convergence from homogeneity. The ablation without the null band isolates the value of frame-relative calibration: if the fixed-gamma rule and the null-band rule disagree on the low-variance subset, the null band is doing work. The ICC_s[m] metric directly checks the exchangeability assumption: if ICC is high within strata, the datasets are not exchangeable draws and the diagnostic is reported as a frame-sensitivity audit. The primary metric — agreement between our verdict and the ranking stability observed when datasets are held out — is the right one because it directly operationalizes 'the aggregate stopped moving' against an external criterion rather than against our own SE curve.

### Expected outcome and causal chain

**vs. Percentile bootstrap over configurations** — On a board like GIFT-Eval where configurations cluster into 28 datasets with shared frequency and horizon, the baseline draws individual configurations, so correlated configurations enter and leave together by chance, producing an SE curve that shrinks too fast and crosses the plateau threshold early. Our method draws whole datasets, so within-dataset correlation is preserved and the SE curve stays high longer before decaying. We expect the ablation (ours minus clustering) to reproduce the baseline's premature plateau on most models, while full CONVERGE flags frame-limited on the same models — a clear gap on boards with few datasets and near-parity on BOOM, where each series is its own source.

**vs. Resolution-report pairwise test** — On a case where two adjacent entries differ by less than the board's median neighbour gap but the whole top-ten aggregate is still drifting with added datasets, the baseline reports the pair as unresolvable and stops, because its mechanism is a pairwise detectability test that never asks whether the aggregate itself has settled. Our method instead tracks the SE of each model's aggregate across resample sizes and flags order-unresolved when the plateau test fails, catching drift the pairwise test structurally cannot see. We expect the two diagnostics to agree on clearly separated pairs but to diverge on the tight top-ten cluster, where CONVERGE flags instability while the pairwise test reports a clean order.

**vs. Fixed-SE threshold rule** — On a homogeneous frame where all models score similarly because the datasets are easy and correlated, the baseline sees small SE and declares convergence, because a threshold cannot tell 'small because converged' from 'small because the frame is homogeneous'. Our method compares the observed SE curve to an exchangeable-null band generated from a stratified super-population with the same within-dataset covariance, so a flat SE curve under homogeneity falls inside the null band and yields a non-plateau verdict. We expect the threshold baseline to declare plateau on the homogeneous subset while CONVERGE withholds it, producing a visible disagreement concentrated exactly on the low-variance subset and parity elsewhere.

**vs. CONVERGE without exchangeable-null band (fixed gamma >= 0.5)** — On a frame with only 28 clusters, the fixed-gamma rule has ~12 grid points and enormous sampling error in gamma, so it may flag non-convergence even when the aggregate is stable. Our method's null band accounts for finite-frame sampling error by construction, so it should agree with held-out-dataset reranking more often than the fixed-gamma rule, especially on the low-variance subset where gamma is noisy.

### What would falsify this idea

If CONVERGE's plateau and stability flags agree with the naive configuration-level bootstrap across all models and boards — that is, if the cluster-respecting correction never changes a verdict — then the correlation argument is wrong and the diagnostic adds nothing. Likewise, if the divergence from the pairwise test is uniform across boards rather than concentrated where the resolution exceeds the median gap, the claimed distinct question is not actually distinct.

## References

1. What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models
