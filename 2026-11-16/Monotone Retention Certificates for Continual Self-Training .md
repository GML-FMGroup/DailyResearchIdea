# Monotone Retention Certificates for Continual Self-Training of Language Agents

## Motivation

Continual-learning pipelines for language agents currently treat retention as a post-hoc admission test: G-CAST promotes a candidate checkpoint only if it passes a cumulative evaluation gate, and the Freezing remedy in Spurious Forgetting fixes bottom layers on the assumption that preserved task alignment implies preserved knowledge. Both therefore leave open the same structural failure — a checkpoint that has merely re-aligned to the new drift distribution can pass the gate while retained capability has silently degraded, because nothing in either pipeline constrains the trajectory of retention loss across transitions. The gap is meta-level: retention is checked at discrete admission points rather than enforced as a property of the update path, so the very quantity the gate claims to protect is never measured as a monotone invariant.

## Key Insight

Retention becomes enforceable rather than auditable once the dual-head decomposition is used to define a per-transition retention-loss functional that the optimizer is constrained to keep non-increasing, so re-alignment cannot buy admission at the price of silent capability loss.

## Method

**What it is.** MRC (Monotone Retention Certificate) is a training-time constraint layer wrapped around a continual self-training loop. Inputs: a stream of drift-distribution batches from daily production evidence, a candidate update direction, and a dual-head probe set (a retained-capability head `h_ret` and a task-alignment head `h_align`) evaluated on a coverage-aware anchor set `A` (see below). Outputs: an admitted checkpoint together with a signed retention certificate recording the measured change in retained-capability loss at that transition.

**Load-bearing assumption (stated explicitly).** The certificate is valid only if the anchor set `A` is distributionally representative of previously relied-upon behavior, so that a bounded increase in anchor loss `L_ret(theta; A)` is equivalent to bounded capability loss. This assumption fails when drift introduces new intents that `A` never covered; in that case the delta reflects anchor-fit rather than true retention and can be non-positive while real capability degrades. To make the assumption testable and to bound its failure, `A` is not a single frozen buffer but a **coverage-aware, expanding anchor set** maintained as follows: `A` is partitioned into `S = 8` strata by intent cluster (k-means on sentence embeddings of the anchor examples, `k=8`, fixed seed). At each admitted checkpoint, new examples from the drift batch `D_t` are assigned to the nearest stratum centroid; if the minimum cosine distance to any centroid exceeds `tau_cov = 0.35`, a new stratum is created (capacity per stratum `N_s = 256`, total capacity `N = 2048`; when full, reservoir sampling within each stratum). The certificate is computed as a **worst-case over strata** with an explicit coverage/uncertainty term: `L_ret(theta; A) = max_s [ mean_{x in A_s} CE(h_ret(x), y) + beta * sqrt(log(2/alpha) / (2 |A_s|)) ]` with `beta = 1.0`, `alpha = 0.05` (Hoeffding-style finite-sample bound per stratum). A checkpoint is admitted only when the **upper confidence bound** on retention loss is non-increasing relative to `theta_ref`. This is the concrete calibration story for the load-bearing assumption: the coverage term makes the certificate conservative exactly when a stratum is thin, and the expanding set ensures new intents enter `A` at admission time rather than being silently excluded.

**How it works.**

```pseudocode
# per training round t, drift batch D_t, coverage-aware anchor set A (strata A_1..A_S, S<=8, N_s=256, N=2048)
# dual heads: h_ret (retained capability), h_align (task alignment)
# L_ret(theta) = max_s [ mean CE of h_ret on A_s + beta*sqrt(log(2/alpha)/(2|A_s|)) ], beta=1.0, alpha=0.05
# L_align(theta) = mean CE of h_align on D_t
# constraint margin eps = 0.002 ; trust-region radius delta = 0.05 (fixed across all regimes; sensitivity reported for delta in {0.02,0.05,0.1}) ; steps K = 200

1. theta_ref <- last admitted checkpoint
2. L_ret_ref <- L_ret(theta_ref)            # certificate baseline (worst-case over strata, with coverage term)
3. delta_ret <- 0
4. for step k in 1..K:
5.     g_task  <- grad_theta L_align(theta; D_t)          # drift-fitting gradient
6.     g_ret   <- grad_theta L_ret(theta; A)             # retention gradient (worst-case stratum)
7.     if L_ret(theta) - L_ret_ref > eps:                # invariant would be violated
8.         g_task <- project_out(g_task, g_ret)          # remove retention-increasing component
9.         delta_ret <- L_ret(theta) - L_ret_ref         # record violation magnitude
10.    theta <- theta - lr * (g_task + lambda * g_ret)    # lambda = 0.5
11.    if ||theta - theta_ref|| > delta: theta <- theta_ref + delta * unit(theta - theta_ref)
12. certificate <- (sign = -1 if L_ret(theta*) - L_ret_ref <= eps else +1,
13.                 magnitude = |L_ret(theta*) - L_ret_ref|,
14.                 coverage = min_s |A_s| / N_s)      # explicit coverage term reported alongside sign
15. admit theta* only if certificate.sign == -1 ; else roll back to theta_ref
16. on admission: assign D_t examples to strata; create new stratum if min cosine dist > tau_cov=0.35; reservoir-sample into A
17. update h_ret by distillation on A only (never on D_t) ; h_align on D_t
```

The projection in line 8 is the load-bearing operation: it removes from the drift-fitting gradient any component that increases retained-capability loss, so the update path is confined to a region where the certificate can stay non-positive. Line 17 keeps the measurement head from absorbing drift signal, which is what makes the certificate comparable across rounds. Line 16 is the concrete realization of the coverage-aware anchor set: it is the mechanism by which the load-bearing assumption is maintained (new intents enter `A`) and by which its failure is made visible (the `coverage` field in the certificate).

**Why this design.** We chose a gradient-projection constraint over G-CAST's post-hoc cumulative gate because a gate can only reject an already-drifted checkpoint, whereas projection prevents the drift from being purchased at retention cost in the first place, accepting the cost that some drift-fitting directions are unavailable and convergence on the new distribution is slower. We chose to freeze the retained-capability head after each admission (distilling it only on the anchor buffer) over jointly training both heads, because a head trained on the drift distribution would redefine the quantity it is supposed to certify, accepting the cost that the certificate is only as good as the anchor buffer's coverage — which is why the anchor set is coverage-aware and expanding rather than a single frozen buffer. We chose a hard trust-region radius over an unconstrained penalty term because a penalty trades retention against drift with a tunable weight that reintroduces the gate-passing loophole, accepting the cost that the radius must be tuned per drift regime — we fix `delta = 0.05` across all regimes and report sensitivity in `{0.02, 0.05, 0.1}` to avoid per-regime tuning. A domain expert would not describe MRC as a variant of G-CAST: G-CAST governs *which* checkpoints are promoted, while MRC constrains *how* the update path may move, and the certificate is a property of the transition rather than of the endpoint.

**Why it measures what we claim.** The signed retention delta `L_ret(theta*) - L_ret_ref` measures monotone non-degradation of retained capability because the assumption that the coverage-aware anchor set `A` (with per-stratum uncertainty term) is distributionally representative of previously relied-upon behavior makes a bounded increase in the worst-case anchor loss equivalent to bounded capability loss; this assumption fails when drift introduces new intents that no stratum covers, in which case the delta reflects anchor-fit rather than true retention and can be non-positive while real capability degrades — the `coverage` field in the certificate makes this failure visible (low `min_s |A_s|/N_s` flags thin strata). The projection residual `||g_task - project_out(g_task, g_ret)||` measures the amount of drift-fitting capacity sacrificed for retention because the assumption that the retention gradient is locally orthogonal to the drift gradient makes the projected-away component a clean estimate of retention-increasing drift signal; this assumption fails when the two gradients are strongly correlated (early in a new task), in which case the residual over-attributes drift progress to retention cost. The certificate sign measures admission eligibility because the assumption that the anchor set is refreshed only at admitted checkpoints makes the baseline `L_ret_ref` a stable reference; this assumption fails if the buffer is contaminated by drift batches, in which case the sign reflects buffer drift rather than checkpoint quality — line 16's stratum-assignment and reservoir-sampling rule is the concrete guard against this.

## Contribution

(1) A training-time monotone retention constraint, MRC, that replaces post-hoc checkpoint gating with a per-transition certificate over a dual-head retained-capability functional. (2) A design principle — freezing the retained-capability head and refreshing the anchor buffer only at admitted checkpoints — that makes the retention certificate comparable across drift rounds. (3) An anchor-buffer protocol and certificate-logging tool for auditing retention trajectories in continual self-training of language agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | E-commerce demand-drift stream (G-CAST setting) | Matches evolving-demand agents with harness evolution |
| Primary metric | Retained-capability loss on held-out anchor probes (worst-case over 8 intent strata, with coverage term) | Directly tests monotone non-degradation claim |
| Secondary metric | Tool-use success rate under harness evolution (downstream agent task) | Ties the reframing to agent behavior the community cares about, not only a proxy |
| Baseline | G-CAST cumulative-evaluation gate | Tests post-hoc rejection vs path constraint |
| Baseline | Unconstrained continual self-training | Tests whether constraint is load-bearing |
| Baseline | Penalty-term retention (tunable lambda) | Tests hard trust region vs soft tradeoff |
| Baseline | GEM/OGD-style gradient-episodic-memory projection | Isolates what the dual-head certificate adds beyond known projection |
| Ablation-of-ours | MRC with jointly-trained retained head | Tests whether frozen head protects the certificate |
| Ablation-of-ours | MRC with single frozen buffer (no coverage-aware expansion) | Tests whether the coverage-aware anchor set is load-bearing for the certificate |
| Diagnostic | Anchor-loss vs true held-out retained capability gap, constructed by drift intents absent from the anchor buffer | Quantifies when the anchor-loss equivalence breaks |

### Why this setup validates the claim

The central claim is that constraining the update path, rather than gating the endpoint, keeps retained capability monotone non-degrading while still fitting drift. The G-CAST-style cumulative gate tests exactly the sub-claim that post-hoc rejection is insufficient: it can only refuse an already-drifted checkpoint, so if our projection prevents drift from being purchased at retention cost, we should see fewer rollbacks and lower retained-loss at comparable drift fit. The unconstrained baseline tests whether the constraint is load-bearing at all — if it matches us, the projection is inert. The penalty baseline tests the hard radius against a soft tradeoff that reintroduces the gate-passing loophole. The GEM/OGD-style projection baseline is the key novelty control: it uses the same projection mechanism but a single episodic memory and no per-transition signed certificate, so any advantage we show over it isolates the dual-head certificate and coverage-aware anchor set rather than the projection itself. The jointly-trained-head ablation isolates the frozen-head design: if joint training does not degrade certificate comparability, freezing is unnecessary. The single-frozen-buffer ablation isolates the coverage-aware expansion: if it matches us, the load-bearing assumption is not actually load-bearing in this stream. The diagnostic directly measures the gap between anchor-probe loss and true held-out retained capability on constructed drift intents absent from the anchor buffer, quantifying when the anchor-loss equivalence breaks rather than only stating it as a caveat. Retained-capability loss on held-out anchor probes is the right primary metric because it is the quantity the certificate claims to certify; the tool-use success rate ties it to downstream agent behavior; a metric measuring only drift fit would not detect the failure the method is built to prevent.

### Expected outcome and causal chain

**vs. G-CAST cumulative-evaluation gate** — On a case where a new product category arrives and the drift batch pulls strongly on retention-sensitive layers, the gate admits a checkpoint whose retained loss has already risen, only detecting the regression at cumulative evaluation and rolling back, wasting the round because its mechanism is endpoint rejection. Our method instead projects the retention-increasing gradient component away before the step, so the update path never enters the degraded region, and we expect a modest but consistent retained-loss advantage concentrated on drift batches with high retention-gradient correlation, with parity where drift is benign.

**vs. Unconstrained continual self-training** — On a case where harness evolution introduces new tools requiring behavior the anchor buffer never covered, the unconstrained model fits drift fast but overwrites relied-upon behavior because nothing bounds the retention-increasing component. Our method sacrifices those drift directions, so we expect slower early drift fit but a clear separation in retained-capability loss that widens with each round, with the gap largest on rounds following large demand shifts.

**vs. Penalty-term retention** — On a case where the tunable weight is set for a mild drift regime and a sharp drift regime follows, the penalty under-constrains and retention loss rises because the soft tradeoff lets the optimizer buy drift at retention cost. Our method's hard radius (`delta = 0.05`, fixed across regimes) bounds the step regardless of regime, so we expect the penalty baseline to show regime-dependent retention spikes while our retained loss stays flat across regimes, at the cost of slower convergence on the sharp-drift rounds.

**vs. GEM/OGD-style gradient-episodic-memory projection** — On a case where the episodic memory is a single fixed buffer, GEM/OGD projects against the memory gradient but provides no per-transition signed certificate and no coverage-aware strata, so it cannot distinguish a checkpoint that has re-aligned to drift from one that has retained capability. Our dual-head certificate with coverage-aware anchor set reports a signed delta and a coverage term per transition, so we expect GEM/OGD to match us on retained loss when the buffer happens to cover the drift intents but to degrade on rounds where new intents fall outside the memory, while our certificate's coverage field flags those rounds and our expansion rule admits them into `A`.

**vs. MRC with jointly-trained retained head** — On a case where drift batches share surface features with anchor probes, the jointly-trained head absorbs drift signal and its reported delta shrinks even as true retention degrades, because the measurement head is redefined by the distribution it certifies. Our frozen-head version keeps the baseline stable, so we expect the joint-head ablation to show non-positive certificates that fail to track held-out retention, while our certificates track it closely.

**vs. MRC with single frozen buffer (no coverage-aware expansion)** — On a case where drift introduces an intent absent from the initial buffer, the single-buffer certificate stays non-positive because the new intent is never measured, while true retained capability on that intent degrades. Our coverage-aware expansion admits the new intent into a new stratum at the next admission, so we expect the single-buffer ablation to show a growing gap between certificate sign and held-out retained capability on the constructed absent-intent diagnostic, while our certificate's coverage field and worst-case-over-strata delta track the held-out degradation.

### What would falsify this idea

If our retained-capability advantage over the cumulative gate is uniform across drift batches rather than concentrated where retention-gradient correlation is high, the projection is not doing the causal work claimed. If the jointly-trained-head ablation matches our certificate's tracking of held-out retention, the frozen-head design is unnecessary and the central mechanism story is wrong. If the coverage-aware expansion ablation matches the single frozen buffer on the absent-intent diagnostic, the load-bearing assumption is not load-bearing in this stream and the coverage-aware anchor set is unnecessary.

## References

1. G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands
