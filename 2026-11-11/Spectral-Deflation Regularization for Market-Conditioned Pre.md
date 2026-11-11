# Spectral-Deflation Regularization for Market-Conditioned Predictive Embeddings

## Motivation

Market-conditioned JEPA (MC-JEPA) conditions each asset's latent prediction on a leave-one-out market factor, but during market shocks the factor direction dominates embedding variance, so the market-conditioned representation collapses onto that single direction and the asset-specific residual subspace is squeezed out. This is a structural failure: the training objective has no term that reserves covariance mass for directions orthogonal to the factor loading, so the residual branch that is supposed to carry idiosyncratic signal is unidentifiable exactly when conditioning matters most. Frequency-masked embedding inference mitigates collapse in the temporal domain but leaves the cross-sectional factor-direction collapse untouched, confirming that decorrelation along the market-factor direction is the unaddressed gap.

## Key Insight

Collapse onto the market factor is a spectral concentration phenomenon, so constraining the embedding covariance spectrum to retain a prescribed fraction of its trace outside the factor-loading direction is a direct, identifiable condition on the representation rather than a proxy for one.

## Method

**What it is.** SDR-JEPA (Spectral-Deflation-Regularized JEPA) is a market-conditioned joint-embedding predictive architecture whose embedding covariance is explicitly deflated along the market-factor direction during training. Inputs: per-asset windows of minute-level features; outputs: a market-conditioned latent embedding z_i for each asset and a trained predictor used for nine downstream targets.

**How it works.**

```pseudocode
for each batch of assets i = 1..N (window length T, embedding dim d=64):
    # 1. Encode with frequency masking (as in frequency-masked embedding inference)
    x_i_mask = random_frequency_mask(x_i, keep_ratio=0.7)      # mask contiguous FFT bands
    h_i      = encoder(x_i_mask)                               # d-dim embedding, 2-layer MLP (hidden=256, GeLU)
    z_i      = project(h_i)                                    # d-dim projected embedding, linear

    # 2. Market condition (leave-one-out, as in MC-JEPA)
    m_i      = mean_{j != i} target_embedding(x_j)             # leave-one-out market factor
    beta_i   = factor_head(z_i)                                # scalar loading, from own repr., linear head
    z_hat_i  = predictor(z_i, beta_i * m_i)                    # conditioned prediction, 2-layer MLP (hidden=256, GeLU)

    # 3. Spectral deflation regularizer
    Z        = stack(z_1..z_N)                                 # N x d
    Zc       = Z - mean(Z, axis=0)                             # centered
    C        = (Zc^T Zc) / (N - 1)                             # d x d covariance
    u        = m_i / ||m_i||                                   # unit market-factor direction
    # Orthogonalize u against top eigenvectors of C to target only the component not spanned by other common directions
    [V, D]   = eig(C)                                          # eigenvectors V, eigenvalues D sorted descending
    k        = 3                                               # number of top common factors to deflate (market + up to 2 others)
    U_common = V[:, :k]                                        # top-k principal subspace
    u_perp   = u - U_common (U_common^T u)                     # component of u orthogonal to top-k subspace
    u_perp   = u_perp / ||u_perp||                             # renormalize
    lambda_u = u_perp^T C u_perp                               # variance along deflated factor dir
    C_perp   = C - lambda_u * (u_perp u_perp^T)                # deflated covariance (single direction)
    trace_perp = trace(C_perp)
    trace_all  = trace(C)
    L_spec   = relu(rho * trace_all - trace_perp)             # rho = 0.5, hinge penalty

    # 4. Standard JEPA prediction loss + regularizer
    L_pred   = || z_hat_i - target_embedding(x_i) ||^2
    L        = L_pred + gamma * L_spec                         # gamma = 0.1
    update encoder, predictor, factor_head by Adam on L (lr=1e-3, batch size N=64 assets)
```

**Why this design.** We chose a hinge penalty on the deflated trace over a soft orthogonality penalty between z_i and u because the failure mode is loss of *mass* outside the factor direction, not misalignment of individual vectors; the hinge activates only when the perpendicular share drops below rho = 0.5, accepting that the model may still concentrate up to half its variance on the factor. We chose to compute u from the leave-one-out market factor (the same quantity MC-JEPA already constructs) rather than a learned decorrelation head, because the factor direction is identified by construction and does not require a second network to estimate; the cost is that the regularizer is only as good as the leave-one-out factor estimate, which is noisy for small N. We chose frequency masking at keep_ratio = 0.7 on the encoder input rather than on the embedding, because masking the input forces the encoder to spread information across frequencies instead of letting one band dominate, accepting a small loss of raw predictive signal. We chose to deflate the batch covariance rather than a running estimate, because batch statistics align with the shock regime where collapse occurs; the cost is sensitivity to batch composition. A domain expert would not describe this as a variant of BYOL/VICReg-style variance-covariance regularization because those penalties are direction-agnostic, whereas SDR-JEPA penalizes variance *along a specific identified direction* — the market factor — which no prior self-supervised regularizer does.

**Why it measures what we claim.** The quantity `trace_perp / trace_all` measures the fraction of embedding variance that is asset-specific rather than market-wide because it assumes the leave-one-out market factor u is a faithful estimate of the dominant common direction; this assumption fails when N is small or when a second common factor (e.g., sector) dominates, in which case trace_perp reflects residual variance along both idiosyncratic and unmodeled common directions. The hinge `relu(rho * trace_all - trace_perp)` measures the *severity* of factor-direction collapse because it assumes collapse is monotone in the perpendicular variance share; this assumption fails when the encoder achieves high trace_perp by inflating noise dimensions, in which case the penalty reflects noise variance rather than genuine asset-specific signal — which is why frequency masking is applied jointly to suppress noise-band inflation. The frequency-mask keep_ratio measures the encoder's reliance on any single spectral band because it assumes masking a band removes its informational contribution; this assumption fails when the encoder has memorized the masked band's statistics, in which case the mask reflects robustness to masking rather than spectral spread. To make the single-factor assumption explicit: we assume that after orthogonalizing u against the top-k principal subspace (k=3), the remaining component u_perp captures the market-wide direction not already spanned by other common factors, so that deflating along u_perp removes exactly the market-wide component and leaves trace_perp as a valid measure of asset-specific variance; this assumption fails when the top-k subspace does not contain all pervasive common factors (e.g., a fourth sector factor), in which case trace_perp still includes unmodeled common variance, and we test this by comparing against a sector-factor-augmented variant (k=5) and by correlating trace_perp/trace_all with per-asset rank IC.

## Contribution

(1) A spectral-deflation regularizer that constrains the market-conditioned embedding covariance to retain a prescribed fraction of its trace outside the identified market-factor direction, turning the collapse-onto-factor problem into an explicit hinge penalty on a deflated trace. (2) A design principle: conditioning a JEPA on a leave-one-out market factor requires a direction-specific decorrelation term, since direction-agnostic variance regularizers (BYOL/VICReg) do not prevent concentration along the conditioning direction. (3) A training recipe combining input-side frequency masking with cross-sectional spectral deflation that jointly suppresses noise-band inflation and factor-direction collapse.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 11 years minute-level Chinese A-shares (2010–2021), split into calm and shock subsets by top-decile market volatility days | Same regime as target shock setting; shock/calm stratification tests the mechanism |
| Primary metric | Per-target rank IC on held-out assets (9 targets: return, risk, liquidity, etc.) | Detects asset-specific signal vs market beta |
| Baseline | Per-asset JEPA (same encoder/predictor, no market conditioning) | Tests whether market conditioning matters |
| Baseline | MC-JEPA (same as ours but gamma=0, no spectral deflation) | Tests whether deflation adds over conditioning |
| Baseline | Time-series predictive model (supervised LSTM, same input windows) | Tests whether SSL structure matters at all |
| Baseline | VICReg-style variance/covariance penalty (direction-agnostic) | Tests whether identified-direction deflation is reproducible by generic variance regularization |
| Baseline | Learned decorrelation head (second network estimating u) | Tests whether fixed leave-one-out u is sufficient |
| Ablation-of-ours | SDR-JEPA w/o spectral deflation hinge (gamma=0) | Isolates the deflation regularizer |
| Ablation-of-ours | SDR-JEPA with k=1 (no orthogonalization against top-k) | Tests the single-factor assumption |
| Ablation-of-ours | SDR-JEPA with k=5 (sector-factor-augmented deflation) | Tests whether additional common factors matter |
| Diagnostic | Plot trace_perp/trace_all of MC-JEPA embeddings across calm vs. shock subsets and training epochs | Documents the collapse failure mode empirically |
| Diagnostic | Correlation between trace_perp/trace_all and per-asset rank IC across assets and shock/calm subsets | Tests whether trace_perp tracks genuine asset-specific predictive skill |

All models trained with Adam (lr=1e-3), batch size N=64 assets, 3 random seeds; reduced decisive matrix (MC-JEPA vs. SDR-JEPA vs. no-hinge ablation on return/risk/liquidity targets, 3 seeds, shock/calm stratified splits) pre-registered to complete first.

### Why this setup validates the claim

The central claim is narrow: deflating batch covariance along the leave-one-out market-factor direction preserves asset-specific variance that otherwise collapses into the market factor, and this preserved variance is what drives downstream per-asset predictive skill. The design must therefore let us separate three things — the value of market conditioning (per-asset JEPA vs. MC-JEPA), the value of the deflation regularizer specifically (MC-JEPA vs. SDR-JEPA, and SDR-JEPA vs. its no-hinge ablation), and the value of the self-supervised signal at all (time-series baseline). Per-target rank IC is the right primary metric because the predicted effect is a *redistribution* of predictive mass from market-wide to asset-specific targets: a method that merely learned the market factor better would show gains concentrated on market-exposure targets, whereas a method that genuinely deflates would show gains on return, risk, and liquidity targets where cross-sectional asset differences matter. The ablation is the load-bearing comparison: if removing the hinge penalty does not degrade performance specifically on the asset-specific targets, the mechanism story is wrong regardless of how SDR-JEPA compares to baselines. The added VICReg-style and learned-decorrelation baselines test whether the identified-direction deflation is necessary over generic variance regularization, and the k=1 vs. k=3 vs. k=5 ablations test the single-factor assumption directly. The diagnostic plots and correlation analysis test whether trace_perp/trace_all actually tracks asset-specific skill rather than noise or unmodeled common factors.

### Expected outcome and causal chain

**vs. Per-asset JEPA** — On a case where a synchronized market shock drives most minute-level variance into a common direction, per-asset JEPA sees no cross-asset structure and its embeddings drift toward whatever dominates each asset's own window, so it cannot distinguish an asset that co-moves with the market from one that has genuine idiosyncratic dynamics; its per-asset targets end up predicted largely from own-history autocorrelation. Our method instead conditions on the leave-one-out market factor and explicitly caps its share of embedding variance, so idiosyncratic structure survives training. We expect a clear IC gap on return and liquidity targets, with near-parity on pure market-exposure targets where the baseline's implicit market loading is already adequate.

**vs. MC-JEPA** — On a case where the market factor dominates a batch (a shock regime), MC-JEPA's conditioning path lets the predictor explain most of the target through the market factor, so the encoder is free to collapse asset embeddings onto that direction and still minimize prediction loss; its embeddings become near-collinear across assets. Our method adds the hinge on the deflated trace, which activates exactly when perpendicular variance share falls below rho, forcing mass off the factor direction. We expect MC-JEPA to match us on market-exposure targets but to lose noticeably on risk and cross-sectional return targets, with the gap widening on shock-heavy subsets relative to calm subsets. The diagnostic plot of trace_perp/trace_all for MC-JEPA should show a sharp drop during shock periods and a recovery in calm periods, confirming the collapse.

**vs. Time-series predictive model** — On a case where the predictive signal is spread across frequency bands and one band dominates the raw loss, a supervised-style time-series model fits that band and generalizes poorly when the dominant band's statistics shift. Our method's frequency masking at keep_ratio 0.7 forces the encoder to spread reliance across bands, and deflation keeps that spread asset-specific rather than market-wide. We expect the largest gap on targets whose signal is concentrated in non-dominant bands — risk and liquidity — and a smaller, possibly marginal gap on return targets, which is the diagnostic signature of the masking-plus-deflation mechanism rather than generic capacity.

**vs. VICReg-style variance/covariance penalty** — On a case where the market factor dominates, a direction-agnostic variance penalty encourages variance in all directions equally, including the market direction, so it does not specifically prevent collapse onto the market factor; it may even increase market-direction variance. Our method deflates along the identified market direction, so we expect VICReg-style to show lower trace_perp/trace_all and lower IC on asset-specific targets than SDR-JEPA, with the gap concentrated on shock subsets.

**vs. Learned decorrelation head** — On a case where the leave-one-out market factor is noisy (small N), a learned head might estimate u more accurately, but it adds a second network and may overfit. We expect similar performance to SDR-JEPA on large N, but SDR-JEPA may be more stable; the comparison tests whether the fixed u is sufficient.

**vs. SDR-JEPA w/o spectral deflation hinge (gamma=0)** — This is the load-bearing ablation: if removing the hinge leaves asset-specific target performance unchanged, the central claim is wrong. We expect a significant drop in IC on return, risk, and liquidity targets, with no drop on market-exposure targets.

**vs. SDR-JEPA with k=1 (no orthogonalization)** — If the single-factor assumption holds, k=1 should perform similarly to k=3; if a second common factor (sector) carries mass, k=1 will show lower trace_perp/trace_all and lower IC on asset-specific targets, especially in shock regimes where sector factors may dominate.

**vs. SDR-JEPA with k=5 (sector-factor-augmented)** — If the top-3 subspace misses a fourth common factor, k=5 will improve trace_perp/trace_all and IC on asset-specific targets; if not, k=5 will be similar to k=3, supporting the sufficiency of k=3.

### What would falsify this idea

If SDR-JEPA's gain over MC-JEPA is uniform across target types and across shock versus calm subsets, rather than concentrated on asset-specific targets and shock regimes where factor-direction collapse is predicted to occur, then the deflation mechanism is not doing what we claim. Equivalently, if removing the spectral-deflation hinge leaves asset-specific target performance unchanged, or if trace_perp/trace_all does not correlate with per-asset rank IC across assets, the central claim is wrong.

## References

1. Market-Conditioned JEPA: Learning from Unpredictable Market Shocks
