# Regret-Certified Flow-Matching Portfolio Allocation with a Universal Safety Layer

## Motivation

Across the deep portfolio-optimization lineage, the mapping from a predictive or generative signal to an executed allocation is optimized against a differentiable portfolio-level reward with no distribution-free performance certificate. AllocFlow concretely instantiates this: its cross-asset flow Transformer and group-relative RL refinement maximize portfolio-level financial rewards, but the abstract reports no worst-case guarantee, so an adversarial or regime-shifted return stream can silently degrade the learned policy below a trivial constant-rebalanced portfolio. Structurally, the surrogate reward is treated as sufficient for the true objective, which leaves the executed allocation unanchored to any hindsight-competitive benchmark.

## Key Insight

Composing the flow Transformer's generated weights with a rebalanced-arithmetic-mean safety layer makes the executed allocation a learning-augmented online portfolio whose wealth is bounded below by a constant fraction of hindsight-optimal wealth under perfect forecasts and by the geometric mean under adversarial ones, independently of how good the generative model is.

## Method

**What it is.** We introduce SAFE-FLOW (Safety-Augmented Flow for Portfolio Allocation), which takes as input a raw multi-asset market history window and outputs executed signed portfolio weights. Internally it wraps a flow-matching allocation generator in a RAM-style rebalanced-arithmetic-mean safety layer, and trains the generator's RL refinement only on policies whose empirical regret against the hindsight-optimal constant-rebalanced portfolio is certified within a distribution-free bound derived from the RAM construction.

**How it works.**

```pseudocode
Input: market history X_{1:t} (multi-asset returns), horizon T, safety mixing schedule eta_t, RAM window W, regret slack delta

# Phase 1: generative allocation proposal
z ~ N(0, I)                                  # flow-matching base sample
w_hat_t = FlowTransformer_theta(z, phi(X_{1:t}))   # cross-asset flow Transformer, temporal encoder phi
w_hat_t = clip_to_simplex(w_hat_t, eps=1e-3)       # project to bounded simplex subset

# Phase 2: RAM safety layer (universal-portfolio-style composition)
for each asset i:
    b_t[i] = arithmetic_mean(returns_{t-W:t}[i])          # RAM component
    g_t[i] = geometric_mean(returns_{1:t}[i])             # universal-portfolio floor
w_safe_t = eta_t * w_hat_t + (1 - eta_t) * b_t            # convex composition
w_exec_t = normalize_to_simplex_and_capital(w_safe_t)     # shared capital constraint

# Phase 3: certified RL refinement
# Load-bearing assumption: The RAM wealth-ratio guarantee transfers to w_exec_t only if w_hat_t is bounded in the simplex and the empirical regret on the realized path is a valid proxy for worst-case regret. We make this explicit: the certificate is valid under the assumption that the realized return path is representative of the return-sequence class (exchangeability).
trajectory tau = (X_{1:t}, w_exec_t, realized_return_t)
regret_hat = empirical_regret(tau, hindsight_optimal_CRP)
# Compute a conservative bound that includes a penalty for generator deviation:
R_hat_bound = RAM_bound(eta_t, W, delta) + eta_t * L * ||w_hat_t - b_t||_1   # L is a Lipschitz constant of the wealth ratio w.r.t. weights, estimated as max over assets of |return| in window
if regret_hat <= R_hat_bound:
    update theta via group-relative policy gradient on portfolio reward, frozen reference policy pi_ref
else:
    reject update; eta_t <- eta_t + step (increase safety weight)

Output: w_exec_t
```

Hyperparameters: flow-matching solver steps (fixed at 16), RAM window W (chosen by validation from {20, 60, 120} trading days), safety schedule eta_t initialized at 0.5 and annealed toward 0.1 only while the empirical regret certificate holds (annealing step 0.01 per epoch), group size for relative RL (8), frozen reference refresh interval (every 5 epochs), simplex clipping epsilon=1e-3, Lipschitz constant L estimated per window as max_i |return_i| over the window.

**Why this design.** We chose a convex RAM-style composition over a learned gating network because the RAM construction supplies a distribution-free wealth ratio that a learned gate cannot certify, accepting the cost that the safety layer may dilute high-conviction generative bets. We chose to constrain the RL refinement by an empirical regret certificate rather than to add a regret penalty to the reward because a penalty is a soft surrogate that an adversarial return stream can exploit, accepting the cost that some high-reward updates are rejected and sample efficiency drops. We chose to anneal eta_t only while the certificate holds rather than to fix it, because a fixed mixing weight leaves performance on the table when the generator is strong, accepting the cost of a more complex training schedule. A domain expert would not describe SAFE-FLOW as AllocFlow plus RAM, because the certificate gates the RL update itself — the generative policy is trained inside the certified region rather than merely post-processed by a safety layer.

**Why it measures what we claim.** The empirical regret `regret_hat` measures the worst-case competitiveness of the executed allocation because the RAM wealth-ratio inequality makes empirical regret on the realized return path an upper bound on the distribution-free regret whenever the realized path is representative of the return sequence class; this assumption fails under non-stationary regime shifts where the realized path is not representative, in which case `regret_hat` reflects in-sample competitiveness rather than a worst-case certificate. The safety mixing weight `eta_t` measures the trust placed in the generative proposal because the RAM bound is monotone in the mixing weight under the assumption that the generator's output is bounded in the simplex; this assumption fails when the flow Transformer emits weights outside the simplex before normalization, in which case `eta_t` reflects post-normalization trust rather than raw generative confidence. The certificate threshold `RAM_bound(eta_t, W, delta)` measures the admissible regret region because the RAM construction's constant-fraction guarantee is distribution-free over arbitrary return sequences under the assumption that the arithmetic-mean component uses a fixed window W; this assumption fails when W is tuned to the evaluation regime, in which case the bound reflects in-window adaptivity rather than a distribution-free guarantee.

## Contribution

['SAFE-FLOW, a framework that composes a flow-matching allocation generator with a RAM-style rebalanced-arithmetic-mean safety layer and gates the group-relative RL refinement by an empirical regret certificate against the hindsight-optimal constant-rebalanced portfolio.', 'A design principle: the generative policy is trained inside the certified region rather than post-processed, so the certificate shapes the learned allocation distribution instead of merely clipping its output.', 'An empirical protocol that reports the realized wealth ratio against the hindsight-optimal CRP alongside Sharpe ratio and turnover across stationary and adversarial return streams.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-asset equity return histories with regime shifts (e.g., sector ETFs, 20-year daily data) | Tests certificate under non-stationary paths |
| Primary metric | Certified-regret violation rate vs. realized Sharpe | Detects whether safety gate holds and still pays |
| Baseline | AllocFlow (end-to-end flow-matching allocator) | Tests whether certificate gating adds robustness |
| Baseline | Markowitz / Black-Litterman posterior portfolio | Tests classical optimization under estimation error |
| Baseline | RAM universal portfolio with predictions | Tests whether generative proposal beats pure RAM |
| Ablation-of-ours | SAFE-FLOW with fixed eta, no certificate gate | Isolates the gating mechanism's contribution |

### Why this setup validates the claim

The central claim is that gating RL refinement on a RAM-derived regret certificate yields allocations that stay competitive with hindsight-optimal constant-rebalanced wealth under arbitrary return sequences, while still converting generative predictive power into gains. The AllocFlow baseline tests the first sub-claim by asking whether an unconstrained end-to-end flow allocator, trained purely on portfolio reward, drifts into regions where realized regret exceeds the distribution-free bound during regime shifts; if our gated version holds the bound and AllocFlow does not, the gating mechanism is doing real work. The Markowitz/BL baseline tests the second sub-claim from the opposite direction: classical optimizers can be competitive in stable regimes but blow up under estimation error, so parity-or-better there plus superiority in shifted regimes isolates the safety layer's value. RAM-with-predictions is the hardest baseline because it already has worst-case guarantees; beating it requires the generative proposal to add value inside the certified region, which is exactly what the annealing schedule claims to do. The fixed-eta, no-gate ablation is the diagnostic: it holds the architecture and data fixed and only removes the certificate, so any gap between it and full SAFE-FLOW is attributable to the gating logic rather than to the flow Transformer. Certified-regret violation rate is the right primary metric because it directly operationalizes the claim — a method can have high Sharpe by luck while violating the bound, and only the violation rate distinguishes certified competitiveness from fortunate in-sample fit.

### Expected outcome and causal chain

**vs. AllocFlow** — On a case where a regime shift inverts cross-asset correlations mid-window, the baseline produces concentrated signed positions because its flow Transformer was refined purely on portfolio-level reward and has no mechanism to detect that its proposal has drifted outside the hindsight-competitive region. Our method instead rejects those RL updates because the empirical regret exceeds the RAM bound, and raises eta_t to lean on the arithmetic-mean component, so we expect a visible gap in certified-regret violation rate concentrated in post-shift windows, with near-parity in stable regimes.

**vs. Markowitz / Black-Litterman posterior** — On a case where sample covariance is estimated from a short noisy window, the baseline produces extreme long-short weights because mean-variance optimization amplifies estimation error, and its realized regret against the hindsight CRP spikes. Our method instead composes the generative proposal with a geometric-mean floor and an arithmetic-mean component that carries no covariance inversion, so we expect the baseline's violation rate to spike exactly where estimation noise is highest while ours stays flat, with the largest Sharpe gap in high-noise subsets.

**vs. RAM universal portfolio with predictions** — On a case where the prediction stream is informative but the return path is adversarial within the class, the baseline produces allocations that capture the constant-fraction guarantee but leave generative signal unused, so its Sharpe saturates at the universal-portfolio level. Our method instead anneals eta_t toward the generative proposal only while the certificate holds, so we expect a modest but consistent Sharpe advantage over RAM in regimes where predictions are genuinely informative, collapsing to RAM-parity when predictions are adversarial — a crossover pattern, not a uniform gain.

**vs. SAFE-FLOW with fixed eta, no gate** — On a case where the generator is initially weak, the ablation produces high-variance allocations because it trusts the flow Transformer at a constant mixing weight regardless of its current regret. Our method instead increases eta_t when the certificate fails, so we expect the ablation to show a higher violation rate early in training and a slower recovery after regime shifts, with the gap shrinking only after the generator has matured.

### What would falsify this idea

If SAFE-FLOW's certified-regret violation rate is statistically indistinguishable from the ungated ablation across all regimes, the certificate is not doing causal work and the central claim fails. Equally fatal: if the Sharpe advantage over RAM is uniform across informative and adversarial prediction streams rather than concentrated where predictions are informative, the annealing mechanism is not converting predictive power and the design rationale is wrong.

## References

1. AllocFlow: End-to-End Portfolio Trading Action Generation from Raw Market Data
