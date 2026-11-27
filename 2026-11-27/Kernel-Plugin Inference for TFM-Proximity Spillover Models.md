# Kernel-Plugin Inference for TFM-Proximity Spillover Models

## Motivation

The S-APT literature treats the firm-proximity kernel as a fixed adjacency matrix, so the spillover coefficient on a learned proximity kernel is only ever reported as a sign. The closest concrete instance is the TFM proximity work (E1/E3), which reports that TFM-representation dissimilarity gains over direct SEC-disclosure distance become 'consistently positive' once broad filing data are included, but never attaches a sampling distribution to the kernel itself, leaving the magnitude of the kernel's marginal contribution unquantified and its out-of-sample stability untested. Structurally this recurs across the whole spatial-asset-pricing branch: whenever a first-stage representation defines the network, the second-stage risk-premium coefficient inherits first-stage sampling error that no existing estimator propagates, so effect sizes are sign-only by construction.

## Key Insight

Because the spillover coefficient is a smooth functional of the kernel's Gram matrix, first-stage TFM-representation sampling error can be propagated into the second stage exactly by a kernel-plugin influence function, making the kernel's marginal contribution a bias-corrected, confidence-bounded, and rolling-out-of-sample-stable estimand rather than a fixed-adjacency sign.

## Method

**What it is.** We introduce KPI-SAPT (Kernel-Plugin Inference for Spatial APT), which takes frozen TFM firm representations and SEC filing text as inputs and outputs a bias-corrected spillover coefficient $\hat\beta_W$ on the TFM-proximity kernel, its influence-function standard error, a confidence interval, and a rolling out-of-sample stability path of that coefficient.

**Load-bearing assumption (stated explicitly).** The influence-function standard error $\text{se\_plugin}$ is valid under the assumption that, conditional on controls $X_{i,t}$ and the realized spatial weight matrix $W$, the firm-level representation draws $h_i$ are exchangeable across firms, so that $\beta_W$ is an asymptotically linear smooth functional of the empirical Gram matrix and the firm-level bootstrap over representations yields a consistent variance for $\beta_W$. This assumption fails when firms are cross-sectionally dependent through the very spillovers being estimated (network autocorrelation): the spatial autoregressive model $R = \rho W R + X\gamma + \beta(W-D)R + \varepsilon$ implies dependence in returns and, through the S-APT mechanism, in firm fundamentals and disclosures, so $h_i$ are not i.i.d. across firms. In that regime a firm-level bootstrap that resamples firms independently destroys the dependence structure that generates the sampling error of $\rho$ and $\beta_W$, and $\text{se\_plugin}$ reflects representation noise only, not the full sampling error, so the CI is too narrow. The method below therefore uses a dependence-preserving resampling scheme (Phase 3) and reports a coverage diagnostic (Phase 5) that quantifies the under-coverage under induced dependence.

**How it works.**

```pseudocode
INPUT: frozen TFM encoder f (4 models), firm disclosures D_i, returns R_{i,t}, controls X_{i,t}
       bootstrap draws B = 500, rolling window W = 36 months, ridge penalty lambda = 1e-3
       dependence-preserving resampling: block bootstrap over W-graph connected components
       (components defined by thresholding W at its 90th percentile of nonzero entries)

PHASE 1  Kernel construction
  h_i = f(D_i)                                  # frozen representation, no fine-tuning
  K_ij = exp(-||h_i - h_j||^2 / (2 * median_pairwise_dist^2))   # RBF kernel, bandwidth = median heuristic
  W_ij = K_ij / sum_j K_ij                      # row-normalized spatial weight
  D_ij = direct disclosure distance (cosine on TF-IDF of same filing)

PHASE 2  Two-step estimation
  beta_W_hat = argmin_beta  ||R - X*gamma - rho * W * R - beta * (W - D) * R||^2 + lambda*||beta||^2
  # (W - D) isolates the TFM kernel's marginal contribution over direct distance
  beta_D_hat = same regression with D as the weight

PHASE 3  Kernel-plugin influence function (dependence-preserving)
  # Resample connected components of the W-graph, not individual firms, so that
  # cross-sectional dependence induced by the spatial autoregressive structure is preserved.
  for b in 1..B:
     resample connected components of the W-graph with replacement -> h^(b), recompute K^(b), W^(b)
     beta_W^(b) = PHASE 2 on W^(b)
  IF_beta = empirical influence function of beta_W w.r.t. component-level representation draws
  se_plugin = sqrt( (1/B) * sum_b (beta_W^(b) - beta_W_hat)^2 )
  beta_bc = beta_W_hat - bias_estimate(IF_beta)      # bias-corrected two-step

PHASE 4  Rolling out-of-sample stability
  for t in W..T:
     fit PHASE 2-3 on [t-W, t); record beta_W_hat(t), se_plugin(t)
  stability = std_t(beta_W_hat(t)) / mean_t(se_plugin(t))   # primary estimand

PHASE 5  Coverage diagnostic under induced dependence
  # Known-truth simulation: generate synthetic returns from R = rho W R + X gamma + beta (W-D) R + eps
  # with rho in {0.0, 0.1, 0.3}, beta in {0.0, 0.05, 0.1}, and W fixed to the estimated TFM kernel.
  # For each (rho, beta) cell, run PHASE 2-3 on S = 200 synthetic panels and record the fraction of
  # 95% CIs that contain the true beta. Report nominal coverage (target 0.95) and under-coverage
  # as a function of rho, using both the component bootstrap (ours) and the i.i.d. firm bootstrap
  # (baseline) as the resampling scheme.

OUTPUT: beta_bc, se_plugin, 95% CI, stability ratio, sign-consistency rate, coverage table
```

**Why this design.** We chose a kernel-plugin influence function over a naive plug-in of the point-estimate kernel because the latter ignores the first-stage sampling distribution and understates uncertainty, accepting the cost that the influence function requires the kernel to be differentiable in the representation draws (satisfied by the RBF form). We chose a bias-corrected two-step estimator over a fully joint MLE because joint estimation over millions of firm-pair parameters is computationally infeasible at US-equity scale, accepting a small residual bias that the correction addresses. We chose the marginal kernel $(W-D)$ as the regression object rather than $W$ alone because the research target is the kernel's *incremental* contribution over direct disclosure distance, accepting that the two weights are correlated and that the difference must be bootstrapped jointly. We chose rolling out-of-sample stability as the primary estimand over in-sample sign because a reviewer's concern is precisely that the sign is fragile; this accepts the loss of full-sample power in exchange for an honest stability statement. We chose a component-level block bootstrap over an i.i.d. firm bootstrap because the spatial autoregressive structure induces cross-sectional dependence that the i.i.d. bootstrap destroys, accepting the cost that the effective number of resampling units is the number of connected components rather than the number of firms. A domain expert would not describe this as a variant of the TFM-proximity paper (E1/E3): that paper fixes the kernel and reports a sign, whereas KPI-SAPT treats the kernel as an estimated random object and delivers a sampling-error-bounded magnitude — a different estimand, not a different dataset.

**Why it measures what we claim.** The computational quantity $\text{se\_plugin}$ measures the sampling uncertainty of the TFM kernel's marginal spillover contribution because the named assumption is that, conditional on controls and the realized spatial weight matrix, the firm-level representation $h_i$ is an exchangeable draw and that $\beta_W$ is a smooth functional of the empirical Gram matrix; under that assumption the influence function is asymptotically linear and the component-level bootstrap variance is consistent. This assumption fails when firms are cross-sectionally dependent through the very spillovers being estimated (network autocorrelation), in which case $\text{se\_plugin}$ reflects idiosyncratic representation noise rather than the full sampling error and the CI is too narrow; the Phase 5 coverage diagnostic quantifies this under-coverage as a function of $\rho$. The quantity $\text{stability} = \text{std}_t(\hat\beta_W(t)) / \text{mean}_t(\text{se\_plugin}(t))$ measures out-of-sample effect-size persistence because the named assumption is that the rolling window is long enough for the estimator to be in its asymptotic regime; this fails under structural breaks (e.g., regime shifts in filing content), in which case stability reflects regime drift rather than estimator fragility. The bias-corrected $\hat\beta_{bc}$ measures the kernel's true marginal contribution because the named assumption is that the first-order bias expansion is accurate at the observed representation dimension; this fails in very high dimension, in which case $\hat\beta_{bc}$ reflects residual first-stage bias.

## Contribution

["KPI-SAPT, a kernel-plugin inference layer that propagates frozen-TFM representation sampling error into the S-APT spillover coefficient via an influence function, yielding a bias-corrected effect size with a confidence interval for the kernel's marginal contribution over direct disclosure distance.", 'A rolling out-of-sample stability estimand (std of rolling coefficient over mean plugin SE) that reframes the TFM-proximity result from a sign claim into a quantified, stability-tested magnitude, with an explicit diagnosis of when network autocorrelation invalidates the plugin SE.', "An open diagnostic protocol and code path for auditing whether a learned proximity kernel's spillover gains survive sampling-error propagation, applicable to any frozen-representation network in asset pricing."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | US equities, SEC filings 2011–2025 | Matches E1/E3 scope and disclosure coverage |
| Primary metric | Rolling stability ratio std_t(β̂)/mean_t(se) | Directly measures estimand fragility, not sign |
| Secondary metric | CI coverage under known-truth simulation (Phase 5) | Quantifies under-coverage under induced dependence |
| Baseline | Fixed-kernel sign report (E1/E3 style) | Tests whether point-estimate sign is fragile |
| Baseline | Naive plug-in kernel standard error | Tests whether first-stage uncertainty is understated |
| Baseline | W-only regression (no marginal (W−D)) | Tests whether incremental kernel contribution is real |
| Baseline | i.i.d. firm bootstrap (no component resampling) | Tests whether dependence-preserving resampling matters |
| Ablation-of-ours | No bias correction, joint bootstrap only | Isolates contribution of bias-corrected two-step |

### Why this setup validates the claim

The central claim is that the TFM-proximity kernel's marginal spillover contribution is a random object whose magnitude can be bounded by sampling error, and that treating it as such changes the conclusion relative to a fixed-kernel sign report. The fixed-kernel baseline tests the sub-claim that a point-estimate sign is fragile: if the sign is stable and the naive SE is honest, our machinery adds nothing. The naive plug-in baseline tests the sub-claim that first-stage representation sampling matters: if plug-in and kernel-plugin SEs coincide, the influence-function correction is unnecessary. The W-only baseline tests whether the marginal (W−D) object is the right target: if W-alone and (W−D) give the same β̂, direct distance carries no incremental information and the estimand is mis-specified. The i.i.d. firm bootstrap baseline tests the load-bearing assumption directly: if component-level and i.i.d. bootstrap SEs coincide across all ρ regimes in the Phase 5 simulation, then network autocorrelation does not materially affect the variance and the dependence-preserving resampling is unnecessary. The ablation isolates the bias-correction term. The stability ratio is the right primary metric because a reviewer's objection is specifically fragility of sign, not average effect size; a ratio near one means the rolling variation is fully explained by sampling noise, which is the falsifiable prediction. The Phase 5 coverage table is the right secondary metric because it converts the load-bearing assumption into a measurable calibration check: nominal 95% coverage under induced dependence is the concrete verification that the component bootstrap propagates the dependence-induced variance.

### Expected outcome and causal chain

**vs. Fixed-kernel sign report (E1/E3 style)** — On a case where the filing panel shifts composition mid-sample (e.g., a wave of new filers entering the SEC corpus), the baseline reports a single sign from a full-sample fit and cannot distinguish a genuine spillover from a composition artifact, because its kernel is fixed and its SE ignores representation resampling. Our method recomputes W^(b) across component-level bootstrap draws and reports a stability ratio, so we expect a noticeably higher std_t(β̂)/mean_t(se) in sub-periods with filing-composition churn but parity with the baseline in stable sub-periods.

**vs. Naive plug-in kernel standard error** — On a case where two firms have near-identical TFMs but materially different filings, the baseline plugs the point-estimate kernel into the second stage and reports a narrow SE because it never propagates representation noise. Our method resamples firm representations at the connected-component level, recomputes the RBF Gram matrix per draw, and forms the empirical influence function, so we expect the kernel-plugin SE to exceed the plug-in SE most sharply for firm pairs near the median-bandwidth boundary, where the RBF derivative is largest.

**vs. W-only regression (no marginal (W−D))** — On a case where TFM proximity and direct disclosure distance are highly correlated (same-firm filings, near-duplicate text), the baseline attributes all spillover to W and inflates β̂ because D is omitted. Our method regresses on (W−D) jointly bootstrapped, so we expect β̂ to shrink toward zero on the high-correlation subset while remaining materially positive on the low-correlation subset, revealing where the incremental kernel contribution actually lives.

**vs. i.i.d. firm bootstrap (no component resampling)** — On the Phase 5 known-truth simulation with ρ = 0.3, the baseline resamples firms independently, which destroys the cross-sectional dependence induced by the spatial autoregressive structure, so its 95% CI under-covers the true β (expected coverage materially below 0.95, worsening as ρ increases). Our method resamples connected components of the W-graph, preserving the dependence structure, so we expect coverage near the nominal 0.95 across all ρ ∈ {0.0, 0.1, 0.3} and a coverage gap between the two schemes that widens monotonically in ρ.

### What would falsify this idea

If the kernel-plugin SE is indistinguishable from the naive plug-in SE across all firm-pair bandwidth regimes, and the stability ratio sits near one uniformly rather than concentrating in filing-composition-churn sub-periods, then the claim that the kernel is an estimated random object with material first-stage uncertainty is wrong. Equivalently, if the (W−D) coefficient matches the W-only coefficient on the high-correlation subset, the marginal-contribution estimand collapses to the fixed-kernel sign report and the design adds no inferential content. Additionally, if the Phase 5 coverage table shows the component bootstrap and the i.i.d. firm bootstrap achieve indistinguishable nominal coverage even at ρ = 0.3, then the dependence-preserving resampling adds no calibration value and the load-bearing assumption is not operative in this setting.

## References

1. Pretrained Tabular Features and the Measurement of Financial Proximity
