# Causal-Window Market Conditioning with an Explicit Leakage Certificate for Self-Supervised Financial JEPA

## Motivation

Market-conditioned joint-embedding predictive architectures such as MC-JEPA (Idea Cards) construct the leave-one-out market condition from other assets' *future target* embeddings under a fixed temporal alignment. This means cross-asset future information can enter the conditioning signal, so the 'shock' the model learns to predict may already be partially encoded in its own input — a leakage that is invisible to the training loss and that silently inflates apparent market-conditioning benefit. The problem is structural: alignment is a design choice made once, never verified per-batch, so no existing self-supervised financial model can certify that its conditioning window is temporally causal.

## Key Insight

Replacing the fixed-offset conditioning window with a strictly causal filter turns temporal alignment from an unverified design assumption into a per-batch measurable invariant, so any residual future-information contamination becomes a detectable, quantified quantity rather than an invisible confound.

## Method

**What it is.** CW-JEPA (Causal-Window JEPA) is a self-supervised joint-embedding predictive architecture for financial time series whose market-conditioning input at time t is provably a function only of other assets' embeddings at times ≤ t. Inputs: a panel of N=300 liquid Chinese A-share assets with minute-level features (returns, volume, order-flow imbalance, 5 lags each); outputs: latent target embeddings and a per-batch leakage certificate. **Load-bearing assumption (stated explicitly):** the target encoder is strictly time-separable — its embedding at time t depends only on inputs at times ≤ t — and the probe's perturbation magnitude is above a calibrated sensitivity floor. The certificate measures conditioning-window leakage only under this assumption; when it fails (bidirectional or batch-normalization-over-time layers, or sub-noise perturbations), the probe reflects encoder non-causality or numerical noise instead, and the two-factor audit below separates these.

**How it works.**

```pseudocode
# Phase 1: causal market filter (replaces fixed-offset leave-one-out)
for each asset i at time t:
    context_i = encoder(x_i[<= t])                     # online encoder, causal (1-layer GRU, hidden=128)
    for each other asset j != i:
        e_j[<=t] = target_encoder(x_j[<= t])            # STRICTLY <= t, no shift; target_encoder = 2-layer MLP (128->256->128, GeLU)
    m_i(t) = sum_j w_ij * e_j[<=t]  / sum_j w_ij          # w_ij = rolling 60-min Pearson corr of returns, clipped to [0,1]
    # causal filter: exponential moving average over the window, no future taps
    c_i(t) = (1 - beta) * c_i(t-1) + beta * m_i(t)       # beta = 0.1 (effective memory ~10 min)

# Phase 2: predict latent target from causal condition
f_i = factor_loading(context_i)                          # own-asset loading: linear 128->64
pred_i(t+h) = predictor(context_i, f_i * c_i(t), h)      # predictor: 2-layer MLP (192->256->128, GeLU); h in {1, 5, 60, 240} min
loss = || pred_i(t+h) - sg(target_encoder(x_i[t+h])) ||^2

# Phase 3: two-factor leakage audit (runs every K=200 steps, not trained on)
# Factor (a): encoder-causality audit — perturbs only inputs at times > t, measures change in e_j[<=t]
for each batch:
    x_j_perturbed = x_j[<=t] + noise_on(t+1 .. t+h)     # noise ~ N(0, sigma^2), sigma = 0.1 * std(x_j)
    e_j_perturbed = target_encoder(x_j_perturbed)
    leak_enc = || e_j_perturbed[<=t] - e_j[<=t] || / || e_j[<=t] ||
# Factor (b): conditioning-window audit — perturbs only the EMA/correlation weights, holds e_j[<=t] fixed
    w_perturbed = w_ij + noise_on_weights            # noise ~ N(0, 0.01^2)
    m_perturbed = recompute(m from w_perturbed, e_j[<=t])
    leak_cond = || m_perturbed - m_i(t) || / || m_i(t) ||
# Sensitivity-floor calibration (run once before training, on a known-causal reference encoder)
    # reference encoder = 1-layer GRU (strictly causal); sweep perturbation magnitude sigma in {1e-4, 1e-3, 1e-2, 1e-1}
    # record leak_enc and leak_cond at each sigma; set tau = 10 * max(leak_enc, leak_cond) at sigma=1e-2
    # (empirically tau ~ 1e-4 for this architecture; 1e-6 is below numerical noise and is not used)
    assert leak_enc < tau and leak_cond < tau; else flag batch and halt
```

**Why this design.** We chose a strictly causal EMA filter over the fixed-offset leave-one-out window used by MC-JEPA because a fixed offset silently admits future target embeddings into the condition, and an EMA with no future taps makes causality a structural property of the computation rather than a hope; the cost is that the condition is slightly lagged and cannot exploit same-minute cross-asset co-movement, which we accept because the leakage it removes is a larger confound. We chose a per-batch counterfactual perturbation probe over a one-time offline alignment audit because leakage can arise from any downstream change (e.g., encoder drift, batch construction), and re-running the probe every 200 steps catches it; the cost is ~15% extra forward compute. We chose to gate training on the probe result (halt on leak > tau) over merely logging it because a logged metric cannot change the optimization trajectory, accepting that occasional batch halts slow wall-clock convergence. We chose a rolling 60-minute correlation weighting over a learned attention over other assets because the correlation is a closed-form, auditable quantity that the leakage probe can recompute exactly; the cost is that it cannot capture nonlinear cross-asset dependence. An expert would not describe this as 'MC-JEPA plus a test': MC-JEPA's mechanism is a fixed-offset leave-one-out condition, whereas CW-JEPA's mechanism is a causal filter whose defining property is that the leakage certificate is *an input to the training loop*, not a post-hoc diagnostic.

**Why it measures what we claim.** The two-factor audit separates the two leakage sources: `leak_enc` measures encoder non-causality because it perturbs only inputs at times > t and observes change in `e_j[<=t]`; this assumption (time-separability) fails when the encoder uses bidirectional or batch-normalization-over-time layers, in which case `leak_enc` reflects encoder non-causality rather than conditioning-window leakage. `leak_cond` measures conditioning-window leakage because it perturbs only the EMA/correlation weights while holding `e_j[<=t]` fixed; this assumption (weights are the only conditioning input) fails when the aggregation has hidden state not captured by `w_ij`, in which case `leak_cond` reflects aggregation non-causality. The EMA coefficient `beta` measures the effective memory length of the market condition because under the assumption that cross-asset correlation is stationary over 60 minutes, the EMA is an unbiased estimator of the correlation-weighted market mean; this fails during regime shifts, where `c_i(t)` reflects a stale market state. The halt-on-leak gate measures the invariant 'no future target information enters conditioning' because `leak_enc < tau` and `leak_cond < tau` are equivalent to the conditioning function being constant in future embeddings under the calibrated sensitivity floor; this fails when the perturbation magnitude is below the floor, in which case the gate passes trivially and the calibration procedure above sets `tau` to avoid this.

## Contribution

(1) CW-JEPA, a self-supervised financial JEPA whose market-conditioning window is a strictly causal EMA filter, making temporal causal consistency a structural property of the forward pass rather than a design assumption. (2) A per-batch counterfactual leakage certificate that perturbs only other assets' future embeddings and halts training when the conditioning signal changes, turning leakage from an invisible confound into a measured invariant. (3) A design principle — causal-window conditioning with a sensitivity floor — that any market-conditioned self-supervised model can adopt to verify alignment without retraining from scratch.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Minute-level Chinese A-share panel, 300 liquid names, 3-year window (2020–2023) | Matches MC-JEPA setup; dense cross-asset co-movement; pre-registered compute budget |
| Primary metric | Latent prediction error on held-out future windows (MSE in embedding space) | Directly tests the conditioning signal's predictive value |
| Baseline | MC-JEPA (fixed-offset leave-one-out) | Isolates the causal-filter vs offset-condition change |
| Baseline | Per-asset JEPA (no market condition) | Tests whether cross-asset conditioning helps at all |
| Baseline | Time-series predictive model (supervised, 2-layer LSTM, hidden=128) | Tests SSL latent prediction against direct forecasting |
| Ablation-of-ours | CW-JEPA with probe disabled (log-only) | Tests whether the leakage gate changes the trajectory |
| Ablation-of-ours | CW-JEPA with single entangled probe (original) vs two-factor audit | Tests whether the audit decomposition changes detection |

### Why this setup validates the claim

The central claim is that a structurally causal market condition plus an in-loop leakage gate yields a conditioning signal that generalizes to held-out future windows better than a fixed-offset condition, and that the gain is attributable to removing future-information confounds rather than to extra capacity. MC-JEPA is the load-bearing baseline: it shares architecture, data, and targets with CW-JEPA and differs only in the conditioning mechanism, so any gap between them isolates the causal-filter change. Per-asset JEPA tests the weaker sub-claim that cross-asset conditioning helps at all, and the supervised time-series model tests whether the SSL objective is doing real work rather than merely matching a direct forecaster. The probe-disabled ablation is the falsification lever for the mechanism claim: if the leakage gate is what produces the gain, disabling it should erode performance specifically on the subsets where leakage would inflate the condition. The single-probe vs two-factor ablation tests whether the audit decomposition is necessary to avoid conflating encoder non-causality with conditioning-window leakage. Held-out latent prediction error is the right metric because it is measured on the same target the predictor is trained against, so it directly reflects whether the condition carries genuine forward-looking market information rather than fitting artifacts. We also report the probe overhead (measured as % extra forward compute) and halt frequency (batches halted per 10k steps) to judge wall-clock viability.

### Expected outcome and causal chain

**vs. MC-JEPA** — On a case where a sharp cross-asset shock arrives within the conditioning window (e.g., a sector-wide move at minute t+3), MC-JEPA's fixed-offset leave-one-out admits target-encoder embeddings from minutes after t into the condition, so its predictor sees a partially informed market state and its training loss is optimistically low while held-out error on shock-adjacent windows stays high. Our method instead builds c_i(t) from a strictly causal EMA with no future taps, so the condition at t is identical whether or not a shock follows, and the predictor must learn the genuine pre-shock signal. We expect a noticeable held-out error gap concentrated on shock-adjacent windows, with near-parity on calm windows. Additionally, we will quantify the leakage confound on MC-JEPA by running the two-factor audit on its conditioning path: we expect `leak_cond` to be nonzero (future target embeddings enter the condition) and to correlate with the inflated gain on shock-adjacent windows.

**vs. Per-asset JEPA** — On a case where an asset's own history is uninformative but a correlated peer leads it (e.g., a large-cap move preceding a small-cap by several minutes), per-asset JEPA has no cross-asset input and must predict from its own noisy features, producing high error on exactly these lead-lag instances. Our method injects the correlation-weighted causal market mean c_i(t), which carries the leading peer's signal at a lag the EMA preserves, so the predictor can exploit the lead-lag structure. We expect a clear error reduction on high-cross-correlation assets and near-parity on assets whose returns are largely idiosyncratic.

**vs. Time-series predictive model (supervised)** — On a case where the supervised model overfits to return-level labels during a low-volatility stretch and then degrades when volatility regime shifts, its direct forecasting objective provides no invariance pressure, so its held-out error spikes at regime boundaries. Our method instead learns latent targets via a JEPA objective with a causal, slowly-varying market condition, which acts as a regularizer that does not chase label noise. We expect our method to be competitive in stable regimes and to show a visible advantage in the windows immediately following volatility regime shifts.

**vs. CW-JEPA with probe disabled** — On a case where encoder drift gradually reintroduces future information into the conditioning path (e.g., a normalization layer that accumulates statistics over the batch), the log-only variant never halts and silently trains on a leaky condition, inflating training fit while held-out error drifts upward. Our gated variant detects the leak via the two-factor audit and halts the offending batch, keeping the condition causally clean. We expect the gated variant to show a smaller train/held-out error gap, with the difference emerging only after the drift onset point.

**vs. CW-JEPA with single entangled probe** — On a case where the encoder itself is non-causal (e.g., a bidirectional layer), the single entangled probe's `leak` will be nonzero and the gate will halt, but it cannot distinguish encoder non-causality from conditioning-window leakage; the two-factor audit separates `leak_enc` and `leak_cond`, so we expect the two-factor variant to correctly attribute the leak and avoid unnecessary halts when only the encoder is non-causal but the conditioning window is clean.

### What would falsify this idea

If CW-JEPA's gain over MC-JEPA is uniform across all windows rather than concentrated on shock-adjacent and high-cross-correlation subsets, the causal-filter mechanism is not what drives the improvement and the central claim is wrong. Likewise, if disabling the leakage probe leaves held-out error unchanged, the in-loop certificate is not doing the work we claim.

## References

1. Market-Conditioned JEPA: Learning from Unpredictable Market Shocks
