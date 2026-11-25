# Spectral Growth Invariance of Kernel-Memory Residual Streams in Rewired Transformers

## Motivation

Both branches of the rewired/state-space Transformer line report strong compute-efficiency gains but leave the scale behavior of their expressive mechanism uncharacterized: LoomFormer (Rewirable Circuits with Loom Attention) demonstrates downstream gains only up to 2.8B parameters and 300B tokens, and DeltaFormer's kernel-function recurrence is argued to escape the TC0 expressivity limit without any statement of how its effective expressive rank grows with width or token budget. The root cause is structural: neither branch defines a measurable quantity that couples the matrix-memory / kernel state to model scale, so gains cannot be predicted or compared across the two mechanisms, and a practitioner cannot tell whether the reported efficiency is a fixed-size artifact or a scaling property. This leaves the field unable to extrapolate either mechanism past its reported budget.

## Key Insight

The per-token matrix memory of LoomFormer and the kernel-function recurrence of DeltaFormer are both low-rank state accumulators whose normalized spectral entropy is invariant to width and token budget under bounded second-residual gradient flow, so the effective expressive rank — not the raw parameter count — is the scale-invariant quantity that governs downstream gain.

## Method

**What it is.** We introduce SGIM (Spectral Growth Invariance Metric), a training-time instrumentation and reparameterization for dual-stream kernel-memory Transformers. SGIM takes as input the per-token matrix memory $M_t \in \mathbb{R}^{d_h \times d_h}$ of a LoomFormer-style vertical-attention head and the kernel recurrence state of a DeltaFormer-style block, and outputs (i) a per-layer deflated effective-rank trajectory $r_{\text{eff}}^{\text{def}}(t)$ and (ii) a mechanism-specific normalized spectral-growth coefficient $\kappa_{\text{Loom}}$ or $\kappa_{\text{Delta}}$ that is compared against model width $d$ and cumulative tokens $N$.

**Load-bearing assumption (stated explicitly).** The claim is that the *deflated* effective rank of the kernel memory — spectral entropy after projecting out the top attention-sink direction(s) — is a scale-invariant quantity governed by a stable exponent $\kappa$ that is shared across LoomFormer's matrix memory and DeltaFormer's kernel recurrence, such that $\kappa$ fitted at small width/token budgets predicts downstream gain at unseen scales. This is an assumption, not a derived result: the two mechanisms are structurally non-equivalent (DeltaFormer's Delta-Rule state space is fixed-size by construction; LoomFormer's matrix memory is bounded by head count and context), so the bridge claim is tested as an equality of deflated exponents rather than assumed. The invariance is only claimed under bounded second-residual gradient flow, verified by an absolute gradient-share criterion (below).

**How it works.**

```pseudocode
# Per training step, per kernel-memory layer l
for each layer l with matrix memory M_t (d_h x d_h):
    # 1. Spectral snapshot (every K=50 steps, no grad)
    s = singular_values(M_t)                  # descending, length d_h
    # 1a. Sink deflation: project out top-r_sink singular directions
    #     r_sink = 1 if top singular value > 0.5 * sum(s) else 0
    #     (diagnostic: report r_eff before and after deflation)
    s_def = s[r_sink:]                        # deflated spectrum
    p = s_def^2 / sum(s_def^2)                # normalized spectral energy
    H = -sum(p * log(p + eps))                # spectral entropy
    r_eff_def = exp(H)                        # deflated effective rank in [1, d_h]

    # 2. Mechanism-specific normalized spectral growth:
    #    regress log r_eff_def on log d and log N, with structural bound term
    #    For LoomFormer: bound = log(min(d_h, n_heads * context_len))
    #    For DeltaFormer: bound = log(fixed_state_size)
    kappa_l = slope( log r_eff_def ~ log d + log N + log(bound) )   # fitted over checkpoints

    # 3. Absolute gradient-share gate on second residual stream
    g_share = ||dL/dM_t||_F / (||dL/dM_t||_F + ||dL/dx_main||_F + eps)
    gate_l = 1[ g_share > tau_share ]         # tau_share = 0.05 (5% of total gradient)

    # 4. Invariance regularizer (only active when gate_l = 1)
    L_sgim += lambda * ( log r_eff_def - alpha * log d - beta * log N )^2
              # alpha=1.0, beta=0.5, lambda=1e-3
```

At evaluation, SGIM reports $\kappa_{\text{Loom}} = \frac{1}{L}\sum_l \kappa_l$ for LoomFormer layers and $\kappa_{\text{Delta}}$ for DeltaFormer layers separately, and a predicted downstream gain $\hat{G}(d,N) = c\, (r_{\text{eff}}^{\text{def}})^{\kappa}$ fitted on the smallest two scales, then tested at the next scale. The reparameterization is a rank-normalization of $M_t$ before the vertical-attention read, rescaling singular values by $s_i / (\sum_j s_j^2)^{1/2}$ so that the spectral energy is comparable across widths.

**Why this design.** We chose a deflated spectral-entropy effective rank over raw parameter count or Frobenius norm because parameter count is fixed by the architecture and cannot express the kernel's *used* capacity, and because the geometric attention-sink result (E2/E7) shows that a few reference-frame directions dominate $M_t$ in early training, so an undeflated entropy would measure sink concentration rather than usable capacity; we accept the cost that effective rank is noisy at small $d_h$ and requires $K$-step snapshotting rather than per-step logging. We chose a log-log regression of $r_{\text{eff}}^{\text{def}}$ on $d$ and $N$ with an explicit structural bound term over a per-step ratio because the invariance claim is a scaling claim, and a regression coefficient is directly falsifiable at a held-out scale; the bound term makes the mechanism-specific cap (fixed state size for DeltaFormer, head/context bound for LoomFormer) explicit rather than absorbed into the slope, accepting that it needs at least three scale checkpoints to be identifiable. We chose the absolute gradient-share gate $g_{\text{share}} > \tau_{\text{share}}$ over a relative-gradient norm because a relative norm passes trivially when another stream dominates the loss, whereas a share criterion verifies that the matrix stream actually carries a non-negligible fraction of the gradient; we accept that layers below the 5% share threshold are excluded from the $\kappa$ estimate and reported separately. We chose to apply SGIM to *both* LoomFormer's matrix memory and DeltaFormer's kernel state rather than one, because the bridge claim is that the two share the same scaling law; a single-mechanism instrument could not test that, and fitting separate exponents $\kappa_{\text{Loom}}$ and $\kappa_{\text{Delta}}$ makes the bridge a testable equality rather than an assumption. Unlike LoomFormer, which reports gains only as an empirical multiplier at fixed 2.8B/300B, SGIM exposes the exponent that predicts the multiplier at unseen scales.

**Why it measures what we claim.** The deflated effective rank $r_{\text{eff}}^{\text{def}} = \exp(H)$ measures *expressive rank* because the spectral entropy of the matrix memory after removing the top sink direction(s) is, under the assumption that the vertical-attention read is a full-rank linear projection of $M_t$'s row space and that $M_t$ is not sink-dominated, a monotone proxy for the dimension of the subspace the head can address; this assumption fails when the read projection is rank-deficient or when a single sink direction collapses $r_{\text{eff}}$ to ~1 while usable capacity is unchanged, in which case $r_{\text{eff}}^{\text{def}}$ reflects residual sink concentration rather than usable capacity (we report the before/after deflation diagnostic to detect this). The log-log slope $\kappa_l$ measures *scale invariance* because a slope of zero on $\log d$ and $\log N$ (after controlling for the structural bound) is exactly the statement that deflated effective rank does not grow with scale; this assumption fails when the training-token budget is too small for the memory to saturate, in which case $\kappa_l$ reflects under-training rather than a scaling property. The absolute gradient-share gate $g_{\text{share}} > \tau_{\text{share}}$ measures *bounded second-residual flow* because a non-negligible share of the total gradient flowing through $M_t$ indicates the matrix stream is in a stable accumulation regime rather than being dominated by another stream; this assumption fails when the loss is dominated by a different stream, in which case the gate excludes the layer and the invariance is not asserted. The predicted gain $\hat{G}(d,N)$ measures *downstream scaling* because the fit of $c\, (r_{\text{eff}}^{\text{def}})^{\kappa}$ on two scales and test on a third is a genuine extrapolation; this assumption fails when downstream tasks are saturated at small scale, in which case $\hat{G}$ reflects task headroom rather than kernel expressivity.

## Contribution

(1) SGIM, a spectral-growth instrumentation and rank-normalization for dual-stream kernel-memory Transformers that yields a per-layer effective-rank trajectory and a normalized scaling coefficient $\kappa$ comparable across LoomFormer-style and DeltaFormer-style mechanisms. (2) A design principle: the effective expressive rank of a kernel-memory residual stream, not its parameter count, is the quantity that should be matched when comparing rewired architectures, together with the bounded-gradient-flow condition under which it is invariant. (3) A cross-mechanism measurement protocol that fits $\hat{G}(d,N)$ on two scales and tests it at a third, enabling falsifiable extrapolation of downstream gains beyond the 2.8B/300B regime.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Pile language modeling + downstream suite | Matches LoomFormer/DeltaFormer training distribution |
| Primary metric | Held-out-scale predicted-gain error | Directly tests extrapolation of κ |
| Baseline | LoomFormer (2.8B/300B) | Tests whether κ predicts its fixed-scale multiplier |
| Baseline | DeltaFormer | Tests shared scaling law across kernel-memory mechanism |
| Baseline | Peri-LN Transformer | Tests whether spectral growth is LN-placement artifact |
| Ablation-of-ours | SGIM without invariance regularizer | Isolates measurement from reparameterization effect |
| Ablation-of-ours | SGIM without sink deflation | Tests whether deflation is necessary for stable κ |

### Why this setup validates the claim

The central claim is that the *deflated* effective rank of the kernel memory grows with a stable, mechanism-specific exponent κ across width and tokens, and that κ predicts downstream gain at unseen scales. Using the Pile plus downstream suite keeps the substrate identical to the LoomFormer and DeltaFormer reports, so any κ we fit is comparable to their published multipliers rather than to a different task regime. LoomFormer tests sub-claim one: that a single exponent predicts its fixed 2.8B/300B empirical gain when extrapolated from smaller scales. DeltaFormer tests sub-claim two: that the same exponent family holds for a different kernel-memory recurrence, which is the bridge claim; we fit κ_Loom and κ_Delta separately and test their equality rather than assuming it. Peri-LN tests sub-claim three: that spectral growth is a property of the matrix stream, not of normalization placement, so a stable κ under Peri-LN would rule out a trivial architectural confound. The ablation separates the instrument from the intervention: if the regularizer-free SGIM still fits κ but loses the held-out gain prediction, the reparameterization is doing the work, not the metric. The sink-deflation ablation tests whether the deflated rank is necessary to avoid the attention-sink artifact identified in E2/E7. Held-out-scale predicted-gain error is the right primary metric because it is the only quantity that can be wrong in the direction the claim forbids — a correct κ must extrapolate, whereas a fixed-scale multiplier cannot be falsified by construction.

### Expected outcome and causal chain

**vs. LoomFormer (2.8B/300B)** — On a case where downstream gain is reported only as a single multiplier at 2.8B/300B, LoomFormer produces a number that cannot be checked at 1B or 7B because the multiplier is measured, not derived. Our method instead fits κ_Loom on the smallest two scales and predicts the third, because the log-log slope of deflated effective rank is the exponent that generates the multiplier. We expect the predicted gain to land within a small band of the observed 2.8B gain, with the largest deviation concentrated at the smallest scale where saturation is incomplete.

**vs. DeltaFormer** — On a case where the kernel state is a fixed-size Delta-Rule memory, DeltaFormer produces strong state-tracking behavior but its capacity is capped by that fixed size, so its deflated effective rank should plateau early and its κ_Delta should drift toward zero on the width axis. Our method instead measures the same deflated spectral-growth quantity on DeltaFormer's recurrence and tests whether κ_Delta equals κ_Loom, because the bridge claim requires a shared exponent. We expect a close match on the token axis and a visible divergence on the width axis, which is exactly the diagnostic that separates a shared law from a shared task.

**vs. Peri-LN Transformer** — On a case where normalization is placed peripherally around sublayers, Peri-LN produces more balanced variance growth, which could trivially inflate or deflate apparent effective rank by rescaling activations rather than by changing usable capacity. Our method instead applies the rank-normalization before the vertical-attention read, so spectral energy is comparable across placements. We expect κ to be stable across Pre-LN, Post-LN, and Peri-LN, with any residual gap attributable to the sink direction rather than to LN placement.

### What would falsify this idea

If the held-out-scale predicted-gain error is no smaller than the error of a constant-multiplier baseline, or if κ_Loom and κ_Delta are equal across width and token axes rather than showing the predicted divergence between LoomFormer and DeltaFormer, then the invariance claim is wrong and deflated effective rank is not the scaling variable. A second falsifier: if the ablation without the regularizer matches the full method on prediction error, the reparameterization is inert and the metric alone does not carry the claim; a third: if the sink-deflation ablation shows no change in κ, then the deflation is unnecessary and the original undeflated rank was not sink-dominated.

## References

1. Rewirable Circuits in Transformers with Loom Attention
