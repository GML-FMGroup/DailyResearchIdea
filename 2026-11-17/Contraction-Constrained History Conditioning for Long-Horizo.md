# Contraction-Constrained History Conditioning for Long-Horizon Autoregressive Video

## Motivation

Long-horizon autoregressive video generators accumulate distribution shift because each self-generated frame becomes conditioning for the next, and existing remedies treat this compounding error as an irreducible property to be compensated for rather than a property of the conditioning map to be constrained. MovieRoll's Window-wise Teacher Forcing with HieraKV compression reduces cache memory but does not structurally bound how far the self-generated history distribution drifts from the training-window distribution at rollout depth; Learning World Models for Interactive Video Generation explicitly names compounding error as irreducible and responds with retrieval-augmented conditioning (VRAG) rather than an error-contraction mechanism. The shared structural gap is that no existing streaming-rollout method constrains the Lipschitz constant of the history-conditioning operator, so divergence between training and rollout distributions is uncontrolled at arbitrary depth.

## Key Insight

If the history-conditioning operator is trained as a contraction in a bounded-divergence summary space, then the divergence between the self-generated history distribution and the training-window distribution is bounded at every rollout depth by a geometric series whose ratio is the contraction coefficient, so compounding error becomes a controlled quantity rather than an irreducible one.

## Method

**What it is.** We introduce CROK (Contraction-Regularized Operator Kernel), a history-conditioning module that replaces HieraKV's sink-recent retention inside MovieRoll's WinTF rollout with a divergence-bounded summary produced by an operator trained to satisfy a fixed-point contraction constraint. CROK takes as input the stream of past window summaries and the current noisy chunk, and outputs a conditioning vector plus a scalar contraction certificate used at training time.

**Load-bearing assumption (stated explicitly).** The certificate is enforced on the composed SummaryNet+Phi operator in summary space (dim d_s=256), and we assume this summary-space contraction transfers to pixel-space rollout divergence: that is, the summary s_t is a sufficient statistic for the drift that causes late-rollout FVD decay, and the decoder is Lipschitz-bounded so that summary drift upper-bounds decoded-chunk drift. This assumption is not proven; it is tested directly by the certificate-validation diagnostic in the experiment (empirical W2 between self-generated and training summary distributions vs. the predicted kappa^k bound, plus a decoder-Lipschitz diagnostic). If the decoder amplifies summary error beyond the bound, the certificate will be conservative and the transfer claim fails.

**How it works.**

```pseudocode
# Phase 1: Window summary encoding (replaces HieraKV retention)
for each completed window w_t:
    h_t = Encoder(x_t)                      # per-window latent, dim d=1024
    s_t = SummaryNet(h_t, s_{t-1})          # recurrent summary, dim d_s=256 << d
    # SummaryNet = 1-layer GRU, hidden=256, layer-norm on input and hidden
    # s_t is the ONLY history passed forward

# Phase 2: Contraction-constrained conditioning operator
c_t = Phi(s_t, z_t^noisy)                   # cross-attention: query=z_t^noisy, key/value=s_t
# Phi = 4-head cross-attention, head dim 64, 2 residual MLP blocks (hidden=512, GeLU)
# Phi is constrained: for all s, s' with ||s - s'|| <= eps,
#   W2(Phi(s,.) , Phi(s',.)) <= kappa * W2(s, s')   with kappa = 0.85 < 1

# Phase 3: Contraction certificate loss (training only)
L_ctr = relu( d_W2(Phi(s_t), Phi(s_t')) - kappa * d_W2(s_t, s_t') + margin )
# s_t' = SummaryNet(h_t, s_{t-1} + delta), delta ~ N(0, sigma^2 I), sigma=0.05
# d_W2 approximated by sliced Wasserstein with 64 random projections
# margin = 0.01

# Phase 4: Rollout with bounded divergence
for rollout step k = 1..K:                  # K = 64 windows (4x training horizon of 16)
    s_k = SummaryNet(Decode(c_{k-1}), s_{k-1})
    c_k = Phi(s_k, z_k^noisy)
    # certificate: D_KL(p_k || p_train) <= kappa^k * D_KL(p_0 || p_train) + L_ctr_residual/(1-kappa)

# Phase 5: Training objective
L = L_WinTF + lambda_ctr * L_ctr + lambda_dmd * L_CuLT-DMD
# lambda_ctr = 0.1, lambda_dmd = 1.0, kappa = 0.85, margin = 0.01

# Phase 6: DMD buffer gating (training-time coupling)
# admit window w into CuLT-DMD distillation buffer iff L_ctr_residual(w) < tau, tau = 0.02
```

**Why this design.** We chose a recurrent summary with an explicit contraction penalty over HieraKV's layer-wise sink-recent retention because HieraKV's compression is a heuristic that preserves attention patterns but makes no claim about how the summary distribution evolves under self-rollout, accepting the cost that our summary is lower-capacity and may lose fine-grained history that HieraKV's full-shallow-layer retention preserves. We chose a sliced-Wasserstein contraction penalty over a KL-divergence penalty because sliced Wasserstein is well-defined for distributions with non-overlapping supports (which occurs early in rollout when self-generated history diverges from training windows), accepting the cost of a 64-projection Monte-Carlo approximation error. We chose to perturb the summary input with a fixed Gaussian noise rather than adversarially maximizing the contraction violation because the fixed-noise version targets the local Lipschitz constant around the training manifold, which is where rollout trajectories live, accepting the cost that the certificate does not cover off-manifold summaries. We chose to keep CuLT-DMD's gradient averaging rather than replace it with full backprop-through-rollout because the contraction penalty already bounds the rollout divergence, so the expensive full-graph computation is redundant, accepting the cost that the certificate is enforced only approximately through the finite-sample penalty. A domain expert would not describe this as 'MovieRoll + a Lipschitz regularizer' because the contraction is enforced on the composition of SummaryNet and Phi as a single operator whose fixed point is the training-window summary distribution, and the certificate is used at training time to gate which windows enter the DMD distillation buffer — a coupling between the contraction residual and the distillation data selection that neither MovieRoll nor standard Lipschitz-regularized training provides.

**Why it measures what we claim.** The sliced-Wasserstein contraction residual L_ctr measures the Lipschitz constant of the history-conditioning operator Phi composed with SummaryNet because the sliced-Wasserstein distance between output distributions under input perturbations of size sigma upper-bounds the operator norm of the Jacobian in expectation over the projection directions, under the assumption that the summary manifold is locally Euclidean with bounded curvature; this assumption fails when the summary distribution has multi-modal structure (e.g., scene cuts), in which case L_ctr reflects the distance between modes rather than the local Lipschitz constant and the certificate is conservative. The perturbation magnitude sigma measures the local neighborhood over which contraction is enforced because sigma is chosen to match the per-window summary drift observed during teacher-forced training, under the assumption that rollout drift is of the same order as teacher-forced drift; this assumption fails when rollout drift exceeds teacher-forced drift by more than an order of magnitude, in which case the certificate covers a neighborhood smaller than the actual rollout trajectory and the bound is loose. The certificate-derived quantity kappa^k * D_KL(p_0 || p_train) measures the divergence between self-generated history and training-window history at depth k because the contraction inequality composes across rollout steps under the assumption that Phi is applied to summaries drawn from the same distribution at each step; this assumption fails when the decoder introduces distribution shift not captured by the summary, in which case the certificate reflects summary-space divergence but not pixel-space divergence. The DMD buffer gating by contraction residual measures the reliability of distillation targets because windows with low residual are those whose summaries are close to the training manifold, under the assumption that contraction residual correlates with target reliability; this assumption fails when a window has low residual but high pixel-space error, in which case the buffer admits unreliable targets. The transfer from summary-space contraction to pixel-space FVD decay additionally assumes the decoder is Lipschitz-bounded (decoder-Lipschitz assumption A); this assumption fails when decoder nonlinearity amplifies summary error beyond the bound (failure mode F), in which case the certificate is conservative and late-rollout FVD may rise even with L_ctr small.

## Contribution

["A contraction-regularized history-conditioning operator (CROK) that replaces HieraKV's heuristic retention with a recurrent summary trained under a sliced-Wasserstein Lipschitz penalty, providing a training-time certificate of bounded summary-space divergence at rollout depth.", 'A coupling between the contraction residual and the CuLT-DMD distillation buffer that gates which self-generated windows enter long-rollout distillation, turning the certificate into a data-selection signal rather than a passive regularizer.', 'An empirical characterization of the trade-off between summary capacity and contraction coefficient kappa across rollout depths, identifying the regime where contraction-constrained conditioning outperforms retrieval-augmented conditioning for long-horizon video.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-form multi-scene video corpus, self-rollout to K=64 windows (4x training horizon of 16) | Stresses compounding error past training window |
| Primary metric | FVD on late-rollout segments (windows 33-64, beyond training horizon) | Isolates divergence growth, not early-frame fidelity |
| Secondary metric | Empirical W2 between self-generated and training-window summary distributions at each depth k, plotted against predicted kappa^k bound; fraction of depths where empirical > certificate | Directly falsifies the contraction certificate |
| Diagnostic | Decoder-Lipschitz proxy: ratio of decoded-chunk latent W2 to summary W2 across rollout depth; correlation of summary-space divergence with pixel-space FVD | Tests assumption A (decoder Lipschitz-bounded) |
| Baseline | MovieRoll (WinTF + HieraKV + CuLT-DMD) | Direct host framework we modify |
| Baseline | MovieRoll with extended context window (2x) | Tests whether memory capacity, not contraction, is the fix |
| Baseline | NOVA-style frame-by-frame AR without quantization | Tests contraction benefit against a different AR design |
| Ablation-of-ours | CROK without L_ctr (unconstrained SummaryNet+Phi) | Isolates the contraction penalty's causal contribution |
| Ablation-of-ours | CROK with L_ctr but gating disabled (all windows admitted to DMD buffer) | Isolates buffer-gating coupling |
| Ablation-of-ours | CROK with gating by non-contraction proxy (pixel FVD of window) instead of L_ctr_residual | Tests whether contraction residual specifically drives target reliability |
| Ablation-of-ours | CROK with kappa=1.0 (no contraction, but same architecture) | Separates architecture from contraction coefficient |

### Why this setup validates the claim

The central claim is that enforcing a contraction constraint on the composed history-conditioning operator bounds rollout divergence, so quality decay beyond the training horizon is slowed relative to heuristic retention. This is falsifiable only if the metric is read on late-rollout segments, where divergence has had depth to accumulate: early-frame FVD would be near-parity for everyone and hide the effect. MovieRoll is the right host baseline because CROK swaps into its exact rollout, so any gain is attributable to the retention mechanism, not the pipeline. The extended-context MovieRoll baseline tests the competing hypothesis that HieraKV's problem is capacity rather than uncontrolled divergence; if context extension alone closes the gap, contraction is redundant. NOVA tests generality across AR designs. The no-contraction ablation is the load-bearing control: it holds SummaryNet, Phi, and CuLT-DMD fixed and removes only L_ctr, so if the gain survives its removal, the contraction story is wrong and the gain is just the recurrent summary. The certificate-validation diagnostic makes the bound falsifiable: at each rollout depth k we measure the actual W2 between self-generated and training summary distributions and compare to kappa^k * D_KL(p_0 || p_train); if the empirical divergence exceeds the certificate at more than 10% of depths, the bound is invalid. The decoder-Lipschitz diagnostic tests assumption A: if the ratio of decoded-chunk W2 to summary W2 grows with depth, the decoder amplifies summary error and the transfer claim fails. The gating ablations test whether the contraction residual specifically, rather than any reliability proxy, drives distillation-target quality.

### Expected outcome and causal chain

**vs. MovieRoll (WinTF + HieraKV + CuLT-DMD)** — On a long rollout crossing several scene transitions, HieraKV retains shallow-layer full history and sink-recent tokens, so self-generated history drifts unchecked and its distribution walks away from training windows; late-rollout FVD climbs steeply and content drifts. Our method passes only a contraction-bounded summary forward, so per-step divergence is damped by kappa<1 and the certificate gates unreliable windows out of the DMD buffer. We expect a growing FVD gap that widens with rollout depth — near-parity early, clearly separated late — concentrated in multi-scene segments. We also expect the empirical summary W2 to track the kappa^k bound within a factor of 2 for at least 90% of depths, and the decoder-Lipschitz ratio to remain roughly constant (not growing) across depth.

**vs. MovieRoll with extended context window** — On a case where drift accumulates over many windows, extending the context window gives the model more drifted history to attend to, and prior evidence indicates naive extended-context AR is less effective because in-context learning is limited; the baseline therefore amplifies rather than corrects drift. Our method bounds the summary's divergence regardless of horizon length, so added depth costs little. We expect the extended-context baseline to gain little or regress on late segments while ours holds, showing the fix is contraction, not capacity.

**vs. CROK without L_ctr (unconstrained operator)** — On a case where the summary distribution is perturbed by self-generated decode error, the unconstrained SummaryNet+Phi has no bound on its Lipschitz constant, so small summary perturbations amplify across rollout steps and late FVD rises. With L_ctr active, the sliced-Wasserstein penalty pulls the operator toward kappa-contraction, damping amplification. We expect the ablation to show a late-rollout FVD gap relative to full CROK that is small at shallow depth and grows with depth, and that the buffer-gating residual correlates with per-window error only in the full model.

**vs. CROK with L_ctr but gating disabled** — If the gain is driven by the contraction penalty alone, disabling gating should leave late-rollout FVD nearly unchanged; if gating is a first-class mechanism, disabling it should degrade late FVD by admitting high-residual windows into the DMD buffer. We expect a measurable degradation, confirming the coupling.

**vs. CROK with gating by pixel FVD proxy** — If the contraction residual is the causal driver of target reliability, gating by pixel FVD should be less effective than gating by L_ctr_residual, because pixel FVD is noisier and not aligned with summary-space divergence. We expect the contraction-residual gate to outperform the pixel-FVD gate on late-rollout FVD.

**vs. CROK with kappa=1.0** — This ablation isolates the effect of the contraction coefficient from the architecture: with kappa=1.0 the penalty becomes a no-op (relu of non-positive term), so any difference from full CROK is attributable to kappa<1. We expect kappa=1.0 to match the no-L_ctr ablation, confirming that the contraction coefficient, not the penalty term's presence, produces the gain.

### What would falsify this idea

If full CROK's advantage over the no-contraction ablation is uniform across rollout depths and scene subsets rather than concentrated at late depth and multi-scene transitions — or if extended-context MovieRoll matches it — then the contraction bound is not what produces the gain and the central claim is wrong. Additionally, if the empirical summary W2 exceeds the kappa^k certificate at more than 10% of depths, or if the decoder-Lipschitz ratio grows with depth (indicating the decoder amplifies summary error), then the transfer from summary-space contraction to pixel-space FVD decay is falsified.

## References

1. MovieRoll: A Systematic Framework for Rolling Out Long-Form Videos
