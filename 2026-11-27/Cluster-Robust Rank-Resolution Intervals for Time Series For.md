# Cluster-Robust Rank-Resolution Intervals for Time Series Forecasting Leaderboards

## Motivation

Leaderboard resolution reports (arxiv:2611.31619) treat each dataset as an independent draw and assume that a single nominal-level test — whose false-difference rate is asserted at 5.2% for 28 datasets — yields valid rank separations for every board. This assumption is structurally wrong for boards built from 15–39 public datasets, because configurations within a dataset (e.g., the eight GIFT-Eval configurations from four datasets) share the same underlying series, so residuals are correlated across configurations and the effective number of independent units is far smaller than the number of displayed entries. The consequence is that the reported resolution and the top-ten separations are computed with a variance that ignores this clustering, so nominal 95% intervals can under-cover precisely when the board is small — the regime where the 15–39-source boards live. The same problem recurs across every benchmark tree that aggregates per-configuration scores (GIFT-Eval, fev-bench, TS-Arena, BOOM), making it a meta-level calibration gap rather than a single-board artifact.

## Key Insight

The rank statistic on a leaderboard is a smooth function of per-configuration score means; its sampling variance is therefore dominated by the cluster-robust sandwich of those means, and a CR3-type leverage correction on the cluster score residuals restores nominal coverage at the 15–39-cluster scale where the uncorrected bootstrap under-covers.

## Method

**What it is.** CRRI (Cluster-Robust Rank Intervals) takes a leaderboard's published per-configuration scores, a dataset-to-cluster map, and a target pair (i, j) of displayed entries, and returns a calibrated confidence interval for the rank separation Δ_ij = rank(i) − rank(j), together with a board-level resolution estimate at nominal level 1−α. It replaces the pool-ranking bootstrap of the resolution report with a cluster-robust sandwich variance on the score means and a CR3 leverage correction adapted to the non-linear rank statistic. The load-bearing assumption is stated explicitly: configurations within a dataset share a common series-level shock, so the cluster score sum is the sufficient statistic for sampling variability; when this assumption fails (configurations within a dataset are independent), the variance is over-stated and intervals become conservative rather than anti-conservative.

**How it works.**
```pseudocode
Input: score matrix S[entry e, configuration c] (published), 
       cluster map g(c) in {1..G} (dataset identity, published as a CSV with columns configuration_id, dataset_id),
       target pair (i,j), nominal level 1-alpha (default 0.95)
Output: CI for Delta_ij, board resolution R_alpha

0. Preliminary diagnostics (reported before any interval):
     G = number of distinct datasets in g(c)
     n_g = #configurations in cluster g, for g=1..G
     ICC = intra-cluster correlation of residuals r[e,c] = S[e,c] - mean_c S[e,c]
           estimated as (MS_between - MS_within) / (MS_between + (n_bar-1)*MS_within)
           where MS_between and MS_within are from a one-way ANOVA of r[e,c] by g(c),
           n_bar = harmonic mean of n_g. Report ICC with a 95% bootstrap CI over clusters.

1. For each entry e and cluster g:
     mu_e     = mean_c in g of S[e,c]          # cluster mean
     n_g       = #configurations in cluster g
     s_e,g     = sum_c in g (S[e,c] - mu_e)   # cluster score residual sum
     h_e,g     = leverage of cluster g for entry e under the
                  configuration design (diagonal of X_g (X'X)^-1 X_g')
     s~_e,g    = s_e,g / (1 - h_e,g)           # CR3-type correction

2. Cluster-robust covariance of the entry-mean vector:
     V_e,e'   = sum_g s~_e,g * s~_e',g / (G - 1) / (N/G)   # N = total configs

3. Rank-separation variance via the delta method on the rank statistic:
     rank(e)  = sum_k 1[mu_k < mu_e] + 0.5 * 1[mu_k == mu_e]
     grad_e(rank) = vector of indicator weights from pairwise comparisons
     Var(Delta_ij) = (grad_i - grad_j)' V (grad_i - grad_j)   # CR3-adjusted

4. CI for Delta_ij:
     z = Phi^{-1}(1 - alpha/2)
     CI = [Delta_ij - z * sqrt(Var), Delta_ij + z * sqrt(Var)]

5. Board resolution:
     R_alpha = smallest delta such that the CI for every adjacent pair
               with |Delta| >= delta excludes 0 at level alpha
Return CI, R_alpha
```

**Why this design.** We chose a cluster-robust sandwich with a CR3 leverage correction over the percentile bootstrap used in the resolution report because the bootstrap resamples configurations as if they were independent, which is structurally false when four datasets contribute eight GIFT-Eval configurations; the CR3 correction is the finite-sample analogue that keeps the variance from collapsing when G is small, accepting the cost that it requires the cluster map g(c) to be published (a mild release requirement). We chose the delta method on the rank statistic over a non-parametric rank bootstrap because the rank is a sum of pairwise indicators whose gradient is available in closed form, so the variance propagates analytically without resampling and without the Monte-Carlo error that would obscure a 5.2% vs 6–10% comparison, accepting the cost that the delta method is a local approximation near ties. We chose to report resolution as the smallest gap whose adjacent-pair CI excludes zero rather than as a single standard-error multiple because the resolution report's own definition is a detection threshold, and the CI-exclusion rule is the direct operationalization of that threshold, accepting the cost that resolution depends on alpha and must be reported per level. A domain expert would not describe this as a variant of the percentile bootstrap of arxiv:2611.31619: the bootstrap resamples configurations, whereas CRRI resamples clusters and applies a leverage correction that the bootstrap does not have, so the two differ in the unit of resampling and in the finite-sample correction, not merely in hyperparameters.

**Why it measures what we claim.** The cluster-robust covariance V measures the sampling variability of displayed entry means because the assumption that configurations within a dataset share a common series-level shock makes the cluster score sum the sufficient statistic for that variability; this assumption fails when configurations within a dataset are genuinely independent (e.g., different sampling frequencies with no shared series), in which case V over-states the variance and the intervals become conservative rather than anti-conservative. The CR3 leverage factor 1/(1−h_e,g) measures finite-sample bias correction because the assumption that cluster leverage is bounded away from one makes the residual sum an unbiased estimator of the cluster variance; this assumption fails when a single cluster dominates the configuration design (h_e,g near 1), in which case the correction inflates and the interval reflects a conservative bound rather than an exact nominal level. The delta-method gradient measures the sensitivity of the rank separation to entry means because the assumption that no two entries are exactly tied makes the rank differentiable almost everywhere; this assumption fails at exact ties, in which case the gradient uses the 0.5 tie weight and the interval reflects a tie-averaged separation rather than a strict order. The resolution R_alpha measures the board's smallest detectable difference because the assumption that adjacent-pair CIs are the right detection rule makes CI exclusion equivalent to the resolution report's threshold; this assumption fails when the board's entries are pool-dependent, in which case R_alpha reflects the resolution of the displayed pool rather than of the underlying model population. Empirical coverage of the CRRI interval measures nominal calibration of the rank-separation estimator because the assumption that the repeated-split reference rank converges to the population rank separation as the number of resamples grows; this assumption fails when the reference is itself estimated from the same clustered scores, in which case coverage is measured against a noisy target and can appear nominal even when the interval is mis-sized.

## Contribution

['CRRI, a cluster-robust rank-separation interval estimator with a CR3 leverage correction adapted to the non-linear rank statistic, producing calibrated confidence intervals for top-ten rank separations and a level-indexed board resolution from published per-configuration scores.', 'A design principle for leaderboard reporting: resolution must be computed on the cluster (dataset) as the resampling unit, not on the configuration, and must be reported with a nominal level and a cluster count, because the configuration-level bootstrap under-covers at the 15–39-cluster scale.', "An open evaluation protocol that takes a leaderboard's published score matrix and cluster map and emits per-pair intervals plus a resolution report, enabling cross-board calibration audits without rerunning models."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GIFT-Eval published per-configuration scores (97 configs, dataset map published as CSV) | Public, 97 configs, dataset map recoverable |
| Dataset | BOOM published per-configuration scores (2,554 series as clusters) | Many independent series, tests opposite regime |
| Dataset | Synthetic score matrices with known population ranks (G=15, 28, 39; ICC=0.0, 0.3, 0.6) | Ground-truth ranks enable coverage measurement |
| Primary metric | Empirical coverage of rank-separation CIs against synthetic ground truth | Directly tests calibration claim with known target |
| Secondary metric | Number of displayed top-ten pairs that become unresolvable under CRRI vs baseline | Quantifies downstream leaderboard verdicts |
| Baseline | Percentile bootstrap (resolution report) | The mechanism CRRI replaces |
| Baseline | Paired bootstrap intervals (fev-bench) | Established uncertainty convention on boards |
| Baseline | Naive iid delta method (no clustering) | Isolates value of cluster structure |
| Ablation-of-ours | CRRI without CR3 leverage correction | Isolates finite-sample correction |

### Why this setup validates the claim

The central claim is that treating configurations as independent understates rank-separation uncertainty when configurations nest inside datasets, and that a cluster-robust variance with a CR3 leverage correction recovers calibrated intervals where the percentile bootstrap does not. GIFT-Eval is the right stress case because its 97 configurations are nested in a small number of datasets, so cluster dependence is strong and G is small — exactly the regime where the bootstrap's iid resampling should collapse variance and where CR3 should matter most. BOOM is the contrast: with thousands of series each counting as its own source, cluster dependence is weak, so CRRI should converge to the naive result and any large divergence there would signal over-correction. The synthetic score matrices provide the only defensible ground truth: because the population ranks are known by construction, empirical coverage is well-defined and not measured against a noisy reference estimated from the same clustered scores. The primary metric, empirical coverage of the rank-separation interval against the synthetic ground truth, is chosen because it is the only quantity that directly adjudicates calibration; a method can be sharper and still wrong, so sharpness alone cannot confirm the claim. The naive iid delta method baseline isolates whether clustering itself (not just delta-method propagation) drives the effect, and the no-CR3 ablation isolates whether the finite-sample correction or the cluster structure is doing the work. Together these form a falsifiable design: the claim predicts coverage near nominal on GIFT-Eval for CRRI but undercoverage for the bootstrap, with near-parity on BOOM.

### Expected outcome and causal chain

**vs. Percentile bootstrap (resolution report)** — On a case where four datasets contribute eight GIFT-Eval configurations, the baseline resamples configurations as independent units, so its rank-separation interval is too narrow because within-dataset series-level shocks are treated as fresh information. Our method instead sums residuals within each dataset cluster and applies the CR3 leverage factor before propagating through the rank gradient, so the shared shock is counted once rather than eight times. We expect the bootstrap to undercover noticeably on GIFT-Eval-like boards with few clusters, while CRRI sits near nominal, and both to agree closely on BOOM where clusters are numerous.

**vs. Paired bootstrap intervals (fev-bench)** — On a case where an adjacent pair's displayed gap is small relative to within-dataset variability, the baseline's paired intervals are computed per task without a cluster-level variance floor, so it declares separation between entries whose rank order is driven by one dataset's configurations. Our method instead builds the covariance from cluster score sums and evaluates the adjacent-pair CI directly as the detection rule, so a pair separated only by one cluster's shock fails to exclude zero. We expect fev-bench-style intervals to flag more adjacent pairs as resolved than CRRI, with the excess concentrated on pairs whose gap is carried by a single dominant dataset.

**vs. Naive iid delta method (no clustering)** — On a case where configurations within a dataset are strongly correlated, the baseline's iid variance treats each configuration as an independent draw, so its interval width shrinks with total configuration count and it overstates resolution. Our method instead uses the cluster-robust sandwich, so effective sample size is governed by G rather than N and the interval widens accordingly. We expect the naive delta method to produce systematically narrower intervals and inflated resolution on GIFT-Eval, converging toward CRRI on BOOM where the cluster correction is nearly inert.

### What would falsify this idea

If CRRI's coverage advantage over the percentile bootstrap is uniform across boards rather than concentrated on the low-G, strongly nested GIFT-Eval regime and absent on BOOM, the cluster-dependence mechanism is not what drives the effect. Likewise, if removing the CR3 correction leaves coverage unchanged on small-G boards, the finite-sample correction claim is wrong.

## References

1. What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models
