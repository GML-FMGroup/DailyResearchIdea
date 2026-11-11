# Factor-Loading Misspecification-Aware Self-Supervised Financial Representation Learning

## Motivation

Existing self-supervised financial representation methods, including MC-JEPA, rely on a latent factor model with a single market factor and assume the factor-loading matrix is correctly specified and shocks are independent. This structural assumption fails in real markets where shocks are correlated and loadings are estimated with error, so the channel-independent robustness argument that underpins graceful degradation is never extended to this multi-factor, shock-correlated setting. As a result, there is no guarantee that representations degrade gracefully when loadings are misspecified or shocks are correlated, and decorrelation along the market-factor direction is unaddressed across capacity–robustness and frequency-masked branches.

## Key Insight

Sensitivity of the market-conditioned representation to loading-matrix perturbations is a computable, differentiable quantity whose penalization during self-supervised training yields representations whose degradation under misspecification is bounded by the perturbation magnitude rather than by the raw shock correlation.

## Method

**What it is.** FALCON (Factor-Loading Certificate-Optimized Network) is a self-supervised financial time-series encoder trained with a misspecification-aware regularizer. Inputs: minute-level panel of asset returns and covariates. Outputs: per-asset latent embeddings used for downstream prediction, trained so that the embedding's sensitivity to perturbations of the factor-loading matrix is penalized jointly with the predictive self-supervised loss.

**How it works.**

```pseudocode
# Per asset i, window t
z_i = Encoder(x_i)                       # asset representation; Encoder = 2-layer MLP, hidden=256, GeLU
B_i = LoadingHead(z_i)                    # factor loadings, shape (K,); LoadingHead = linear(256→K), K∈{1,3}
f_t = LeaveOneOutMarket({z_j}_{j!=i})     # market factor from other assets' targets; mean of peer targets
y_hat_i = Predictor(z_i, B_i * f_t)       # market-conditioned prediction; Predictor = 2-layer MLP, hidden=256, GeLU
L_pred = ||y_hat_i - sg(z_i_target)||^2   # JEPA-style latent prediction loss

# Misspecification-aware regularizer (correlated shocks)
# Perturb loadings along correlated-shock directions
Sigma_e = EmpiricalShockCovariance()      # K x K shock covariance, estimated online via rolling 60-day window with Ledoit-Wolf shrinkage
Delta = epsilon * (Sigma_e^{1/2} u)       # u ~ N(0,I), epsilon = 0.05
B_i_pert = B_i + Delta
L_robust = ||Predictor(z_i, B_i_pert * f_t) - y_hat_i||^2 / (epsilon^2)

# Certificate term: bound degradation by perturbation magnitude
# kappa is estimated per batch as the empirical local Lipschitz constant of Predictor w.r.t. B_i
kappa_batch = max_{batch} ||Predictor(z_i, B_i + delta) - Predictor(z_i, B_i)|| / ||delta||  # delta ~ N(0,I), 10 samples per asset
L_cert = relu( L_robust - kappa_batch * ||Delta||^2 )   # kappa_batch replaces fixed kappa=1.0

L = L_pred + lambda_r * L_robust + lambda_c * L_cert   # lambda_r=0.1, lambda_c=0.05
```

**Why this design.** We chose to perturb loadings along the empirical shock-covariance square root rather than along isotropic Gaussian directions because correlated shocks are exactly the regime where the channel-independent robustness argument breaks; isotropic perturbation would leave the correlation structure unpenalized, accepting the cost that the regularizer requires an online covariance estimate that is noisy in early training. We chose a differentiable certificate term (a hinge on the ratio of degradation to perturbation magnitude) over a pure penalty on L_robust because the hinge directly targets the bound constant rather than shrinking the prediction uniformly, accepting the cost that the hinge threshold kappa introduces a sensitivity hyperparameter. We chose to derive loadings from the asset's own representation (as in MC-JEPA) rather than a separate loading head trained on returns, because this couples the robustness penalty to the same representation used downstream, accepting the cost that loading misspecification and representation misspecification are entangled and cannot be separately diagnosed. A domain expert would not describe this as MC-JEPA plus a noise regularizer: MC-JEPA fixes a single market factor and never differentiates through the loading matrix, whereas FALCON's certificate term is a function of the loading-Jacobian and the shock covariance, which MC-JEPA does not compute.

**Why it measures what we claim.** The quantity L_robust measures loading-misspecification sensitivity because the directional derivative of the predictor with respect to B_i along Sigma_e^{1/2}u is equivalent to the worst-case degradation under correlated shocks, under the assumption that the shock covariance is stationary over the training window; this assumption fails when a regime change alters the correlation structure, in which case L_robust reflects sensitivity to the stale covariance rather than to current shocks. The certificate hinge L_cert measures the gap between realized degradation and the perturbation-magnitude bound because kappa_batch*||Delta||^2 is the first-order bound on predictor change under a Lipschitz predictor, under the assumption that the predictor is locally Lipschitz with constant kappa_batch estimated per batch; this assumption fails when the predictor saturates, in which case L_cert reflects saturation rather than true misspecification robustness. The leave-one-out market factor f_t measures market-wide shock exposure because excluding the asset's own target prevents leakage, under the assumption that other assets' targets span the market factor; this assumption fails in thinly traded cross-sections, in which case f_t reflects idiosyncratic noise from peer assets.

**(D) Explicit assumptions and failure modes.** The equivalence between L_robust and worst-case loading-misspecification degradation holds under two assumptions: (i) Sigma_e is stationary over the training window, and (ii) the predictor is locally Lipschitz with constant kappa_batch. Failure mode (i): regime change makes Sigma_e stale, so L_robust reflects sensitivity to an outdated covariance rather than current shocks. Failure mode (ii): predictor saturation makes L_cert reflect saturation rather than true robustness. To verify assumption (ii) empirically, we compute kappa_batch per batch (10 perturbation samples per asset) and log its distribution; if kappa_batch varies by more than 20% across batches, the fixed-kappa certificate is invalid and we report the batch-wise values. To verify assumption (i), we compute the correlation between training-time L_robust and held-out rank-IC degradation across windows (see experiment calibration step).

## Contribution

(1) A self-supervised training objective, FALCON, that augments market-conditioned latent prediction with a differentiable loading-misspecification certificate computed along the empirical shock-covariance directions, yielding representations trained for graceful degradation under correlated shocks. (2) A design principle: penalizing the ratio of prediction degradation to perturbation magnitude (a hinge certificate) rather than the raw perturbation loss produces robustness that is targeted at the loading matrix instead of shrinking all predictions. (3) An online shock-covariance estimator coupled to the loading Jacobian that makes the robustness penalty track the correlation structure of the panel rather than assuming independence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Minute-level Chinese A-share panel (11 years) | Matches MC-JEPA regime; dense cross-section for leave-one-out factor |
| Primary metric | Downstream rank IC on held-out targets | Directly measures embedding's predictive utility, not proxy loss |
| Baseline | MC-JEPA | Tests whether loading-perturbation certificate beats fixed-factor JEPA |
| Baseline | Per-asset JEPA | Tests whether market-conditioning + robustness beats asset-isolated encoding |
| Baseline | Time-series predictive model | Tests whether self-supervised loading structure beats supervised forecasting |
| Ablation-of-ours | Isotropic-Gaussian perturbation (no shock covariance) | Isolates the contribution of correlated-shock directions |
| Ablation-of-ours | Input-noise perturbation along Sigma_e^{1/2}u (identical epsilon, kappa) | Isolates loading-Jacobian term from generic input-noise regularization |
| Ablation-of-ours | kappa sweep: fixed kappa ∈ {0.5, 1.0, 2.0} vs. estimated kappa_batch | Shows certificate term, not kappa value, drives gain |
| Calibration check | Regress held-out rank-IC degradation on training-time L_robust across windows; report Pearson correlation | Tests whether L_robust transfers to out-of-sample rank IC |

### Why this setup validates the claim

The central claim is that penalizing the embedding's sensitivity to factor-loading perturbations along correlated-shock directions yields representations that generalize better to downstream financial targets than encoders that ignore loading misspecification. The dataset is the natural testbed because it is the same panel on which MC-JEPA was established, so any gain over MC-JEPA is attributable to the certificate mechanism rather than to data or target choice. MC-JEPA is the critical baseline: it shares the market-conditioned JEPA backbone but fixes a single factor and never differentiates through loadings, so a win over it isolates the loading-Jacobian contribution. Per-asset JEPA tests the market-conditioning sub-claim (does cross-asset factor structure help at all), and the time-series model tests the self-supervised sub-claim (does the SSL objective beat direct supervised forecasting). The isotropic-Gaussian ablation is the falsification probe for the correlation-structure sub-claim: if FALCON's advantage survives when perturbations are isotropic, then the shock-covariance square root is doing no work and the design's core rationale collapses. The input-noise ablation isolates whether differentiating through B_i matters beyond any covariance-shaped noise: if FALCON's rank-IC gain over input-noise perturbation is near zero, the loading-Jacobian term is not the driver. The kappa sweep tests whether the certificate term itself, rather than the specific kappa value, drives the gain. The calibration check directly tests the load-bearing assumption that training-time L_robust transfers to out-of-sample rank-IC degradation: if the correlation is near zero, the certificate framing must be dropped and the method repositioned as a directional smoothness regularizer with no robustness guarantee. Rank IC is the right metric because the claim is about the ordering quality of per-asset embeddings across the cross-section — exactly what a downstream ranker consumes — and it is scale-free, so a uniform shrinkage of predictions (the failure mode of a pure L_robust penalty) would not masquerade as improvement.

### Expected outcome and causal chain

**vs. MC-JEPA** — On a case where a regime shift reorders which assets load on the market factor (e.g., a sector rotation), MC-JEPA produces stale loadings because its single fixed factor cannot express asset-specific loading change, so its predictions lag the new correlation structure. Our method instead perturbs each asset's loading along the empirical shock-covariance directions during training, forcing the predictor to remain accurate under loading drift, so we expect a noticeable rank-IC gap concentrated in regime-transition windows and near-parity in calm windows.

**vs. Per-asset JEPA** — On a case where an asset's own history is short or noisy (thinly traded names), the per-asset baseline produces an embedding dominated by idiosyncratic noise because it has no cross-sectional signal to borrow. Our method instead conditions on a leave-one-out market factor built from peer targets, injecting shared structure that stabilizes the representation, so we expect a large rank-IC gap on the low-liquidity subset and a smaller gap on the high-liquidity subset.

**vs. Time-series predictive model** — On a case where the target is a risk or market-exposure quantity rather than raw return, the supervised forecaster produces overfit predictions because it optimizes the target directly with no auxiliary structure. Our method instead learns embeddings that encode factor-loading sensitivity, which transfers across all nine targets, so we expect a broad rank-IC advantage that is widest on exposure and risk targets and narrowest on raw-return targets.

**vs. Input-noise perturbation along Sigma_e^{1/2}u** — On a case where loading misspecification is the dominant error source (e.g., regime-transition windows), the input-noise variant perturbs the encoder input but does not differentiate through B_i, so it cannot shape the loading-Jacobian. Our method instead penalizes sensitivity of the predictor to B_i perturbations, so we expect a rank-IC gap over the input-noise variant concentrated in regime-transition windows, confirming the loading-Jacobian term is the driver.

**vs. kappa sweep (fixed kappa ∈ {0.5, 1.0, 2.0} vs. estimated kappa_batch)** — If the certificate term drives the gain, rank IC should be relatively flat across kappa values and the estimated kappa_batch should match or exceed the best fixed kappa. If rank IC varies strongly with kappa, the gain is attributable to the kappa value rather than the certificate mechanism, falsifying the claim.

**Calibration check** — We expect a positive Pearson correlation (target ≥ 0.3) between training-time L_robust and held-out rank-IC degradation across windows. If the correlation is near zero, the load-bearing assumption that L_robust transfers to out-of-sample rank IC is false, and the method must be repositioned as a directional smoothness regularizer with no robustness guarantee.

### What would falsify this idea

If FALCON's gain over MC-JEPA is uniform across regime-transition and calm windows rather than concentrated where loading misspecification is predicted to bite, or if the isotropic-Gaussian ablation matches FALCON's rank IC, or if the input-noise ablation matches FALCON's rank IC, or if the calibration correlation between L_robust and held-out rank-IC degradation is near zero, then the correlated-shock certificate is not the causal driver and the central claim is wrong.

## References

1. Market-Conditioned JEPA: Learning from Unpredictable Market Shocks
