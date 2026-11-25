# Calibration-Constrained Route Gating for Payment Foundation Models under Structural Shift

## Motivation

The Antom 3-in-1 Transformer (E1) learns task-route gating statistics exclusively from Ant International's internal payment graph, so its risk scores carry no structural guarantee of calibration or monotone consistency once the deployment graph's homophily profile shifts — a failure mode that graph anomaly work attributes to homophilic shortcut reliance under structural distribution shift (E9, E8). Because no branch in the Antom architecture establishes invariance of route-level risk scores to structural shift, the model's 37.88% online loss reduction is validated only on the ecosystem it was pretrained on, and cross-institution calibration is left unverified.

## Key Insight

Monotone calibration of a route-mixture risk score is preserved under graph shift exactly when the gate's per-route weights are invariant to the homophily profile of the input graph, so enforcing gate invariance across homophily-mixed synthetic environments is the structural condition that makes held-out calibration certificates attainable.

## Method

**What it is.** CIRG (Calibration-Invariant Route Gating) is a post-hoc wrapper on a frozen Antom-style 3-in-1 transformer: it takes a payment graph G with node features and a frozen per-route risk head, and outputs a calibrated, monotone-consistent risk score together with a calibration certificate (ECE, monotonicity violation rate, and a distribution-shift bound) computed on held-out shifted graphs.

**Load-bearing assumption (stated explicitly).** CIRG's certificate is valid under the assumption that (A) the synthetic homophily-mixed graphs span the deployment shift manifold — i.e., homophily is the dominant shift axis between source and target institutions — and (B) the frozen per-route heads {h_r} remain individually calibrated on the target distribution once routing is held fixed. Under (A) and (B), gate weights that are invariant across homophily-mixed graphs are sufficient to make the frozen route-mixture risk score's calibration transfer to real target institutions. The certificate is therefore restricted to targets where a per-route calibration audit (Phase 2b) bounds component-head drift; when (A) or (B) fails, the certificate reports the bound as violated rather than claiming transfer.

**How it works.**

```pseudocode
INPUT: frozen encoder f, route heads {h_r}_{r=1..R}, gate g_phi(G) -> w in simplex^R
       source graph G_src (Ant internal), unlabeled target graphs {G_tgt}
       homophily-mixup strength alpha in [0,1], invariance weight lambda=0.5
       monotone temperature grid T = {0.5,0.75,1.0,1.25,1.5}
       per-route audit tolerance tau_route = 0.02 (ECE units)

PHASE 1  Environment synthesis (homophily-aware mixup, HEM-style)
  for k = 1..K=8:
      sample edge-weight perturbations eps ~ Dirichlet(alpha)
      build G_k = mix(G_src, eps)   # rewires edges to sweep homophily h in [0.1,0.9]
      record homophily h_k = fraction of same-label edges in G_k

PHASE 2  Gate invariance training (only g_phi updated; f, h_r frozen)
  for each batch of nodes v across {G_k}:
      w_k(v) = g_phi(G_k, v)
      p_k(v) = sum_r w_k(v)_r * h_r(f(G_k, v))
      L_cal   = sum_k  | mean_{v in bin b} p_k(v) - frac_pos(b) |   # bin-wise ECE proxy, 15 bins
      L_inv   = sum_{k,k'} || w_k(v) - w_{k'}(v) ||_1              # gate invariance across h_k
      L_mono  = sum_k  relu( p_k(v_i) - p_k(v_j) )  for pairs (i,j) with y_i=1, y_j=0, score_i<score_j
      loss = L_cal + lambda * L_inv + 0.1 * L_mono
      update phi by Adam, lr=1e-4, 200 steps

PHASE 2b  Per-route calibration audit (bounds assumption B)
  for each route r and each held-out G_tgt:
      ECE_r(G_tgt) = bin-wise ECE of h_r(f(G_tgt, v)) against labels on G_tgt
      drift_r = | ECE_r(G_tgt) - ECE_r(G_src) |
  route_bound(G_tgt) = max_r drift_r
  if route_bound(G_tgt) > tau_route: mark certificate as BOUND_VIOLATED for G_tgt

PHASE 3  Monotone projection of the route mixture
  for each T in T (selected on a source-held-out split):
      p_T(v) = sigmoid( logit(p(v)) / T )
  pick T* = argmin_T  ECE(p_T on source-held-out)
  enforce monotonicity: sort scores, apply isotonic regression (PAVA) on source-held-out
      -> monotone map m; final score s(v) = m(p_{T*}(v))

PHASE 4  Calibration certificate on held-out shifted graphs
  for each held-out G_tgt with homophily h_tgt:
      compute ECE(s), monotonicity violation rate MVR(s), and
      Delta_cert = | ECE(s on G_tgt) - ECE(s on G_src) |
      report route_bound(G_tgt) from Phase 2b alongside Delta_cert
      emit certificate (ECE, MVR, Delta_cert, h_tgt, route_bound, status) per target graph
OUTPUT: calibrated score s(v), gate weights w(v), certificate table
```

**Why this design.** We chose to freeze the encoder and route heads and train only the gate, rather than fine-tuning the 7B-parameter backbone, because the abstract-level evidence for Antom (E1) does not expose architectural details needed for safe fine-tuning, accepting the cost that gate-only adaptation cannot repair representation-level miscalibration — which is why Phase 2b audits per-route drift and restricts the certificate to targets where that drift is bounded by tau_route=0.02. We chose homophily-aware environment mixup (E8/E9) over generic Gaussian feature noise because the failure mode named in E9 is structural (low-homophily shortcut reliance), accepting the cost that synthesized graphs may not match real external institutions' degree distributions. We chose an L1 gate-invariance penalty over a KL-divergence penalty because L1 produces sparse route selection that is easier to certify per-route, accepting the cost of non-smooth gradients near zero. We chose isotonic projection (PAVA) over a learned monotone head because PAVA yields an exact monotone map with no extra parameters, accepting the cost that it cannot extrapolate beyond the source score range. A domain expert would not describe this as 'HEM + Antom': HEM (E8) is a training-time representation regularizer for GAD encoders, whereas CIRG leaves the encoder untouched and certifies a post-hoc gate whose invariance is the load-bearing property — the composition does not arise from either paper's mechanism.

**Why it measures what we claim.** The gate-invariance penalty L_inv measures invariance of route selection to the homophily profile because we assume (A) that synthetic homophily-mixed graphs span the deployment shift manifold, so gate weights that agree across h_k are equivalent to shift-invariant routing; this assumption fails when a target institution's shift is driven by feature drift rather than homophily drift, in which case L_inv reflects robustness only to the simulated structural axis. The ECE proxy L_cal measures calibration because we assume bin-wise empirical frequency on mixed graphs is an unbiased estimate of the deployment calibration error, an assumption that fails under label shift, in which case L_cal reflects source-conditional calibration only. The monotonicity penalty L_mono measures monotone consistency because we assume the isotonic projection on source-held-out pairs transfers to shifted pairs with the same score ordering, which fails when shifted scores invert ordering, in which case MVR reflects source-ordering fidelity. Delta_cert measures cross-ecosystem calibration stability because we assume held-out shifted graphs are representative draws from the target ecosystem, which fails for adversarial targets, in which case Delta_cert reflects only the sampled shift set. The per-route audit route_bound measures assumption (B) — that frozen component heads stay individually calibrated — because we assume ECE_r on the target is estimable from the target's labels; this assumption fails when the target is unlabeled, in which case route_bound is unavailable and the certificate is marked BOUND_UNVERIFIED rather than BOUND_VIOLATED.

## Contribution

["A calibration-invariant route gating wrapper (CIRG) that freezes a payment foundation model's backbone and enforces homophily-invariance of the mixture gate via an L1 penalty over HEM-synthesized environments, producing a monotone-projected risk score.", 'A per-graph calibration certificate (ECE, monotonicity violation rate, and cross-graph ECE delta) that operationalizes cross-ecosystem calibration stability as a reported quantity rather than an unverified claim.', 'A design principle: gate-level invariance, not backbone fine-tuning, is the minimal intervention sufficient to preserve route-mixture calibration under structural homophily shift.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Antom internal payment graphs + held-out merchant domains | Source and shifted targets share the claimed shift axis |
| Dataset (reproducible surrogate) | Small open 3-in-1-style transformer (<100M params: sequence route = 4-layer Transformer, tabular route = 3-layer MLP hidden=256, graph route = 2-layer GNN hidden=128) trained on a public payment-fraud dataset; Antom numbers reported as external validation if access granted | Makes the full CIRG pipeline and certificate table testable without proprietary data |
| Dataset (public fraud graph) | G-Internet benchmark from E7 plus one public GAD dataset | Tests whether Delta_cert is a model-agnostic certificate, not an Antom artifact |
| Primary metric | ECE gap between source and shifted graphs (Delta_cert) | Directly measures the calibration-stability claim |
| Secondary metric | route_bound(G_tgt) = max_r |ECE_r(target) - ECE_r(source)|, with tolerance tau_route=0.02 | Operationalizes assumption (B): certificate is only claimed where per-route drift is bounded |
| Secondary metric | MVR (monotonicity violation rate) | Tests the monotone-consistency sub-claim |
| Baseline | Frozen Antom 3-in-1 + XGB-Online scores | Tests whether freezing alone already calibrates |
| Baseline | Uncalibrated frozen gate (no L_inv, no PAVA) | Tests whether invariance and projection carry the gain |
| Baseline | HEM-style fine-tuned encoder | Tests whether post-hoc gating beats representation retraining |
| Ablation-of-ours | CIRG without L_inv (L_cal + L_mono only) | Isolates the gate-invariance contribution |
| Ablation-of-ours | CIRG without Phase 2b audit (certificate reported unconditionally) | Tests whether the per-route bound is what makes the certificate honest |
| Stress target | Feature-drift-only target (homophily held at source level, features perturbed) | Measures Delta_cert degradation when assumption (A) is violated |

### Why this setup validates the claim

The central claim is that a frozen Antom-style backbone plus a post-hoc, homophily-invariant gate yields calibrated, monotone-consistent risk scores whose calibration error stays stable across structurally shifted graphs, conditional on the stated assumptions (A) homophily spans the shift and (B) per-route heads stay calibrated. The primary metric, the source-to-target ECE gap (Delta_cert), is the right detector because the claim is precisely about stability of calibration under shift, not raw ranking quality; a method that only improved AP could still fail the claim. The uncalibrated frozen gate baseline tests the sub-claim that calibration machinery (L_cal, temperature, PAVA) is necessary rather than incidental. The HEM-style fine-tuned encoder baseline tests the sub-claim that post-hoc gate invariance is sufficient compared to representation-level retraining, which is the strongest competing explanation. The no-L_inv ablation is the load-bearing test: if removing the invariance penalty does not degrade Delta_cert, then invariance is not what produces calibration stability, and the mechanism story collapses. The Phase 2b audit and its ablation test assumption (B): if the certificate is honest only when route_bound is reported, then the audit is load-bearing rather than cosmetic. The feature-drift-only stress target tests assumption (A): if Delta_cert degrades sharply there, the certificate is correctly scoped to homophily-driven shift rather than overclaimed. The reproducible surrogate and public G-Internet benchmark test whether the certificate protocol generalizes beyond one proprietary backbone. Together these form a falsifiable test because each baseline and stress target isolates one causal link in the claimed chain from environment synthesis to invariant gating to certified calibration.

### Expected outcome and causal chain

**vs. Frozen Antom 3-in-1 + XGB-Online scores** — On a target merchant domain where homophily drops from roughly 0.7 to 0.3, the baseline produces inflated confidence on fraudulent nodes that rely on low-homophily shortcuts because its per-route heads were fit under the source homophily profile and its scores are never recalibrated. Our method instead reweights route selection across synthesized homophily-mixed environments and applies a source-fit isotonic map, so we expect a visibly smaller Delta_cert on the low-homophily target while roughly matching the baseline on high-homophily targets, with the gap concentrated in the low-homophily subset and the certificate reporting route_bound below tau_route=0.02 on those targets.

**vs. Uncalibrated frozen gate (no L_inv, no PAVA)** — On a graph where the gate is trained only on source-like environments, the baseline produces route weights that drift with the local homophily profile, so its calibration error swings between targets because the mixture of route heads changes even when the underlying risk is unchanged. Our method instead penalizes L1 disagreement of gate weights across homophily-mixed graphs and projects scores through an exact monotone map, so we expect the baseline to show a larger spread of per-target ECE and a higher monotonicity violation rate, while our method shows a flatter ECE profile across targets and near-zero MVR.

**vs. HEM-style fine-tuned encoder** — On a target with structural shift but limited labels, the baseline produces calibration that improves over scratch training but still varies with the target's homophily because its invariance is learned implicitly in the encoder and is entangled with feature learning. Our method instead leaves the encoder frozen and certifies invariance explicitly at the gate, so we expect comparable or better Delta_cert with far fewer adapted parameters, and the diagnostic signal is that our certificate table shows stable ECE across targets while the fine-tuned encoder's ECE tracks each target's homophily level.

**vs. CIRG without Phase 2b audit** — On a target where per-route drift exceeds tau_route=0.02, the unaudited variant reports a Delta_cert that looks stable while the component heads are individually miscalibrated, so the certificate is misleading. Our method reports route_bound alongside Delta_cert and marks the target BOUND_VIOLATED, so we expect the unaudited variant to show lower apparent Delta_cert on exactly those targets where route_bound is high, and our method to show the same Delta_cert but with the violation flagged — demonstrating that the audit, not the gate alone, is what makes the certificate honest.

**vs. Feature-drift-only stress target** — On a target where homophily is held at the source level but features are perturbed, assumption (A) is violated because the shift is not on the homophily axis. We expect Delta_cert to degrade relative to the homophily-shifted targets, and route_bound to rise, confirming that the certificate is scoped to homophily-driven shift and that the method does not claim transfer under feature drift.

### What would falsify this idea

If removing the L_inv penalty leaves Delta_cert essentially unchanged across shifted targets, or if our calibration gain is uniform across high- and low-homophily subsets rather than concentrated where the predicted structural shortcut failure occurs, then gate invariance is not the causal driver and the central claim is wrong rather than merely underperforming. Additionally, if Delta_cert on the feature-drift-only stress target is as small as on homophily-shifted targets, then assumption (A) is not load-bearing and the certificate's scoping claim is wrong.

## References

1. Antom 3-in-1 Transformer: A Payment Foundation Model
