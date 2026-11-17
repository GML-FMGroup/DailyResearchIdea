# Regime-Conditional Coherent Flow Matching for Portfolio Allocation

## Motivation

End-to-end allocation generators such as AllocFlow (arxiv:2611.19937) shape a candidate portfolio distribution via strategy-supervised imitation and then refine it with group-relative RL anchored to a frozen policy reference, but nothing in that scheme establishes that the generated joint weights remain identifiable or coherent when the temporal market representation drifts; the frozen reference is a single stationary anchor that treats the imitation prior as a stable regularizer rather than a regime-dependent object. Structurally, the flow map is conditioned only on a temporal representation that entangles regime identity with asset-level signal, so a shift in regime silently moves the generated weight distribution without any signal that the map has left the region where its outputs are coherent. The same gap recurs across the tree — Bayesian BL work (venue:icml:2025:59095ddf9d62) learns latent views but has no shift test, and RAM (venue:neurips:2025:882c4942295c) guarantees worst-case wealth but not distributional stability of allocations — so the field lacks a mechanism that makes regime-conditional coherence of generated weights an explicit, testable property rather than an implicit hope.

## Key Insight

If the regime variable is inferred from the same temporal representation that conditions the flow, then the flow's velocity field becomes a smooth function of a low-dimensional regime coordinate, so coherence under regime shift is equivalent to Lipschitz continuity of the velocity field in that coordinate — a property that can be trained by a distributionally-robust objective and audited out-of-distribution without any labels.

## Method

**What it is.** ReCo-Flow (Regime-Conditional Coherent Flow) is an end-to-end allocation generator that maps raw multi-asset market histories to joint signed portfolio weights, but conditions the cross-asset flow Transformer on an explicit latent regime variable z inferred from the temporal representation, and trains it with a distributionally-robust (DR) objective over regime-conditional candidate portfolio distributions. Inputs: multi-asset return/feature window; outputs: joint weight vector w and regime posterior q(z).

**How it works.**

```pseudocode
# Phase 1: regime inference
h_t   = TemporalEncoder(market_window)            # per-asset temporal repr; 2-layer Transformer, d_model=128, 4 heads
z_mu, z_ls = RegimeHead(h_t)                       # latent regime mean, log-scale; 2-layer MLP, hidden=128, GeLU
z     = z_mu + eps * exp(z_ls),  eps ~ N(0, I)     # reparameterized sample, dim d_z=4
q(z)  = softmax(RegimeClassifier(h_t))             # discrete regime posterior, K=6; 2-layer MLP, hidden=128, GeLU

# Phase 2: regime-conditional flow matching
for step s in 1..S (S=20 Euler steps):
    v_s = CrossAssetFlowTransformer(w_s, s, h_t, z)   # velocity, conditioned on z; 6-layer Transformer, d_model=128, 4 heads
    w_{s+1} = w_{s} + (1/S) * v_s
w_0 ~ N(0, I);  output w = w_S  (project to simplex / leverage cap)

# Phase 3: DR training objective over regime-conditional candidate distributions
L_imitation = E_{z~q} E_{w~pi_strategy(.|z)} ||v_theta - v_target||^2   # strategy-supervised
L_dr = sup_{P: W(P, P_hat) <= rho} E_{z~P} [ L_coherence(w(z)) ]
     where L_coherence(w) = ||J_w(z)||_2^2   # squared spectral norm of Jacobian of w w.r.t. z, estimated by power iteration (5 iterations) on the velocity field
L_rl = E[ -R(w) ] with anchor A(z) = imitation policy frozen per-regime, KL(pi || A(z)) <= delta
L = L_imitation + lambda_dr * L_dr + lambda_rl * L_rl   (lambda_dr=0.5, lambda_rl=0.3, rho=0.1, delta=0.05)

# Phase 4: OOD coherence audit
for held-out regime r*: compute Delta(r*) = ||w(z_{r*}) - A(z_{r*})|| / ||z_{r*} - z_bar||
flag if Delta(r*) > tau (tau = 95th pct of in-distribution Delta)
```

**Why this design.** We chose an explicit low-dimensional latent regime variable (d_z=4) over feeding the full temporal representation to the flow because the coherence claim is only meaningful if there is a coordinate along which continuity can be measured; the cost is that a 4-d bottleneck may under-represent regime nuance, which we accept because the audit is about the map's behavior in z, not about z's expressive power. We chose a distributionally-robust penalty on the velocity field's Lipschitz constant in z over a frozen single anchor (the AllocFlow design) because a frozen reference cannot represent regime-conditional behavior — it is by construction regime-agnostic — and the DR ball (Wasserstein radius rho=0.1) forces the penalty to hold for the worst regime perturbation in the ball rather than the empirical one; the cost is a more conservative policy that may under-trade in benign regimes. We chose per-regime imitation anchors inside the RL KL constraint over a single global anchor because the KL must be evaluated where the policy actually operates; the cost is K=6 separate anchor copies and more memory. A domain expert would not describe this as AllocFlow-plus-a-regime-head: AllocFlow's frozen reference is a stationary regularizer, whereas here the anchor is a function of z and the DR term is a constraint on the flow map's derivative in z, which AllocFlow does not compute at all.

**Why it measures what we claim.** The Lipschitz penalty L_coherence measures regime-conditional coherence of generated weights because it assumes the latent z is a sufficient statistic for regime identity — i.e., that two market histories with the same z posterior must receive the same allocation up to the residual asset-level signal; this assumption fails when the temporal encoder leaks asset-specific information into z (e.g., a single-asset shock masquerading as a regime), in which case L_coherence reflects encoder leakage rather than genuine regime instability. The DR supremum over the Wasserstein ball measures robustness to distribution shift because it assumes the shift is a bounded perturbation of the regime-conditional distribution; this fails under a structural break (new asset universe, regime never seen), in which case the ball is mis-centered and the penalty reflects distance to the wrong reference. The OOD audit statistic Delta(r*) measures out-of-distribution coherence because it assumes the held-out regime's z is in the support of the DR ball; when it is not, Delta reflects extrapolation distance rather than coherence failure.

**Assumption A (sufficiency of z).** The coherence claim is conditional on z being a sufficient statistic for regime identity: the temporal encoder must not route asset-specific shocks into z. We make this assumption explicit and test it in the experiment via a leakage probe (predicting asset-level residual returns from z) and a synthetic single-asset-shock stress test; if the probe R^2 exceeds 0.1, the coherence claim is reported as conditional on the probe result and the audit statistic is interpreted with the leakage caveat.

## Contribution

["ReCo-Flow: a regime-conditional flow-matching allocation framework that infers an explicit latent regime variable from the temporal representation and conditions the cross-asset flow Transformer on it, with a distributionally-robust training objective on the velocity field's Lipschitz constant in the regime coordinate.", 'A regime-conditional anchor replacing the frozen policy reference in the RL stage, together with an OOD coherence audit statistic that flags when generated weights leave the region where the map is regime-stable.', "An empirical design principle: coherence under regime shift is governed by the flow map's derivative in the regime coordinate, not by the strength of the imitation prior, which reframes how the two-stage training should be regularized."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Regime-labeled multi-asset equity panel (Dow-Jones + sector ETFs, with volatility/trend regime annotations) | Contains distinct, annotated regimes needed to test coherence and OOD behavior. |
| Primary metric | Regime-conditional Sharpe ratio, plus turnover on regime-transition subsets | Sharpe captures risk-adjusted gain; turnover exposes regime-instability cost. |
| Baseline | AllocFlow (end-to-end flow, frozen single anchor) | Tests whether regime conditioning beats a regime-agnostic flow. |
| Baseline | Markowitz / Black-Litterman (Bayesian, view-free) | Tests whether learned joint allocation beats classical mean-variance construction. |
| Baseline | RAM (online selection with ML predictions) | Tests whether end-to-end allocation beats prediction-then-allocation pipelines. |
| Ablation-of-ours | ReCo-Flow without DR penalty (lambda_dr=0) | Isolates whether the Wasserstein-robust coherence term drives the gain. |
| Diagnostic | Leakage probe: ridge regression predicting asset-level residual returns from z, reporting R^2 | Tests Assumption A (z is a regime statistic, not an asset statistic). |
| Stress test | Synthetic single-asset-shock injection into the market window, checking whether Delta(r*) falsely flags a regime transition | Tests whether the audit statistic is confounded by encoder leakage. |

### Why this setup validates the claim

The central claim is that conditioning the cross-asset flow on an explicit latent regime variable, and regularizing its Lipschitz behavior in that variable under a distributionally-robust ball, yields allocations that stay coherent across regime shifts and therefore earn better risk-adjusted returns with less churn. The dataset must contain genuine, annotated regime episodes so that a regime-transition subset can be isolated; otherwise the coherence claim is untestable. AllocFlow tests the sub-claim that regime conditioning matters at all: if a regime-agnostic frozen anchor were sufficient, conditioning would add nothing and the two would tie. The classical Markowitz/BL baseline tests whether the learned generator beats analytic construction, while RAM tests whether end-to-end allocation beats a decoupled forecast-then-allocate pipeline. The ablation is the load-bearing test: removing the DR penalty while keeping the regime head isolates whether the robustness term, rather than the regime head alone, produces the predicted pattern. Regime-conditional Sharpe is the right primary metric because the claim is about risk-adjusted performance *within* regimes, not raw return; turnover on transition subsets is the diagnostic that separates coherence from mere aggressiveness. The leakage probe and synthetic shock stress test directly test Assumption A: if z carries asset-level signal, the probe R^2 will be high and the shock will falsely trigger Delta(r*), indicating that the coherence penalty is measuring encoder leakage rather than regime instability.

### Expected outcome and causal chain

**vs. AllocFlow** — On a case where the market shifts from a low-volatility trend regime to a high-volatility mean-reverting regime, AllocFlow produces a large, abrupt reallocation because its frozen, regime-agnostic anchor has no coordinate to express the shift and its flow must absorb the change through asset-level velocity alone. Our method instead conditions the flow on z, so the same regime change moves the latent coordinate and the anchor A(z) moves with it, keeping the weight change proportional to the regime distance. We expect a noticeable Sharpe gap concentrated on regime-transition windows, with near-parity in stable regimes, and materially lower turnover on transition subsets.

**vs. Markowitz / Black-Litterman** — On a case where historical covariance is estimated from a regime that has just ended, the classical estimator produces concentrated weights driven by stale second moments, and the view-free Bayesian variant still inherits the same covariance sensitivity. Our method instead learns the allocation map from joint multi-asset histories conditioned on the inferred regime, so the covariance regime enters through z rather than through a single pooled estimate. We expect a modest but persistent Sharpe advantage that widens in the post-transition window, with the largest gains on the highest-volatility regime subset.

**vs. RAM** — On a case where the ML prediction stream is accurate but the allocation step is myopic, RAM captures the forecast but its rebalancing rule does not coordinate positions across assets under a shared capital constraint, so it under-allocates to correlated opportunities. Our method instead generates all signed weights jointly through the cross-asset flow, so shared-capital coordination is internal to the generator. We expect our advantage to appear mainly where cross-asset correlation is high and to shrink where assets are near-independent, which is the signature of coordinated versus decoupled allocation.

### What would falsify this idea

If the gap versus AllocFlow is uniform across stable and transition subsets rather than concentrated on regime transitions, then the regime-conditioning claim is not what drives performance. Likewise, if removing the DR penalty (lambda_dr=0) leaves regime-conditional Sharpe and transition turnover unchanged, the robustness mechanism is inert and the central claim is wrong. Additionally, if the leakage probe R^2 exceeds 0.1 and the synthetic single-asset-shock stress test falsely triggers Delta(r*), then Assumption A fails and the coherence claim is not supported.

## References

1. AllocFlow: End-to-End Portfolio Trading Action Generation from Raw Market Data
