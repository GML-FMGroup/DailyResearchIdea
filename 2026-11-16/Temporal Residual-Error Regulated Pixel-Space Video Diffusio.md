# Temporal Residual-Error Regulated Pixel-Space Video Diffusion

## Motivation

Pixel-space video diffusion (JVT, arxiv:2611.18549) relocated all detail synthesis into a DiT with no autoencoder bottleneck, but the latent bottleneck it removed was implicitly regularizing per-step residual error; without it, high-frequency pixel residuals compound across the frame sequence because joint attention lets later frames inherit uncorrected early-frame error. This is the meta-level failure of pixel-space generation generally: removing the compression stage removes the field's only built-in error-damping mechanism, so temporal error accumulation becomes an unconstrained byproduct rather than a monitored quantity, as also visible in AR-Diffusion's (cvpr:2025:4b6d1c37903f) explicit need for corruption-timestep constraints to stop error accumulation in asynchronous generation.

## Key Insight

Per-frame high-frequency residual energy is a directly observable, causally attributable error signal in pixel space — unlike in latent space where it is entangled with the bottleneck — so penalizing its temporal growth during training installs the error-damping role the removed autoencoder used to play.

## Method

**What it is.** TERRA (Temporal Error-Regularized Residual Attention) is a training-time regularization plus a multi-scale residual correction head for JVT-style pixel-space DiT video diffusion. Inputs: a noisy pixel video batch and its denoising-step index; outputs: the standard velocity/epsilon prediction plus a coarse-to-fine residual correction, trained under a temporal residual-growth penalty. The load-bearing assumption, stated plainly: per-frame high-pass (Laplacian) residual energy in pixel space is dominated by uncorrected synthesis error rather than by genuine scene motion and texture, so that the inter-frame growth of this energy is a valid, causally attributable error signal that can be penalized during training to damp temporal error accumulation. This assumption is made explicit and is tested by the motion-compensated residual and the diagnostic experiment below; it fails under fast motion or occlusion, where high-pass energy reflects true motion rather than error, and the penalty would over-smooth moving regions.

**How it works.**

```pseudocode
# Training step for TERRA on a large-patch pixel DiT
for batch (x0, cond) in loader:
    t ~ Uniform(0, T)                      # diffusion timestep
    eps ~ N(0, I)
    xt = sqrt(abar[t]) * x0 + sqrt(1-abar[t]) * eps
    # Standard DiT trunk over large patches
    h = DiT_trunk(patchify(xt), t, cond)   # h: per-patch hidden states
    eps_hat = head_eps(h)                  # base prediction
    # Multi-scale residual correction head (coarse->fine)
    r_coarse = head_c(h)                   # patch-level residual, scale 1
    r_mid    = head_m(h, upsample(r_coarse))   # scale 2
    r_fine   = head_f(h, upsample(r_mid))      # scale 4 (pixel)
    eps_final = eps_hat + r_coarse + r_mid + r_fine
    # Predicted clean frame sequence
    x0_hat = (xt - sqrt(1-abar[t]) * eps_final) / sqrt(abar[t])
    # Motion-compensated temporal high-frequency residual error per frame
    flow_k = RAFT(x0_hat[k], x0_hat[k+1])   # optical flow, frozen RAFT (e.g. raft-things)
    x0_hat_warp[k+1] = warp(x0_hat[k], flow_k)
    e_k = HP_filter(x0_hat_warp[k+1] - x0[k+1])  # high-pass (3x3 Laplacian)
    # Normalize by local motion energy to target error orthogonal to true motion
    m_k = mean(|flow_k|)                    # motion magnitude per frame pair
    e_k_norm = e_k / (m_k + 1e-6)
    L_growth = mean_k( ||e_{k+1}_norm||^2 - ||e_k_norm||^2 )_+   # penalize growth, lambda_g=0.1
    L_diff   = MSE(eps_final, eps)
    L = L_diff + lambda_g * L_growth
```

Hyperparameters: patch size 16 (JVT-style large patches), correction scales {1,2,4}, lambda_g=0.1, high-pass filter = 3x3 Laplacian, growth penalty applied over K=16 frame windows, optical flow from frozen RAFT (raft-things weights), motion normalization epsilon=1e-6.

**Why this design.** We chose a *training-time* residual-growth penalty over a test-time post-hoc error monitor because the growth term must shape the denoiser's own dynamics, accepting the cost that it adds a training loop over frame pairs. We chose *multi-scale* residual correction heads (coarse patch → mid → pixel) over a single pixel-resolution head because high-frequency error lives at fine scales while temporal drift is visible at coarse scales, accepting extra parameters and a small per-step compute increase. We chose an explicit high-pass Laplacian filter over a learned frequency decomposition because a fixed filter gives a reproducible, hyperparameter-free error signal, accepting that it is not learned to the data distribution. Crucially, this is not the 'X + Y in Z' composition of a residual head with a diffusion model (cf. standard residual-refinement diffusion): the residual heads are *coupled* to a temporal error-growth penalty so the correction is explicitly trained to flatten inter-frame residual growth, a structural property a plain residual head or a plain consistency loss (e.g., AR-Diffusion's timestep constraint) does not produce because those operate on latent/velocity space rather than on measured pixel high-frequency residuals.

**Why it measures what we claim.** The quantity `||e_{k+1}_norm||^2 - ||e_k_norm||^2` (temporal growth of motion-compensated, motion-normalized high-pass residual energy) measures *per-step temporal error accumulation* because the assumption is that after warping by estimated optical flow and normalizing by local motion magnitude, the remaining high-pass pixel residual energy is dominated by uncorrected synthesis error rather than by genuine scene motion; this assumption fails when the video contains fast motion or occlusion that the flow estimate cannot capture, in which case the metric reflects true high-frequency motion energy instead of error, and the penalty would over-smooth moving regions. The multi-scale correction heads measure *coarse-to-fine error attribution* because the assumption is that coarse-scale residual captures temporal drift and fine-scale captures detail loss; this fails when the two scales are correlated (e.g., global brightness shift), in which case the heads redundantly correct the same error. The growth penalty weight lambda_g measures *how strongly error damping is enforced* because the assumption is that a fixed penalty is calibrated across content; it fails on low-motion clips where the penalty is inactive, in which case it reflects only the diffusion loss.

## Contribution

['A temporal high-frequency residual-growth regularizer for pixel-space video DiT training that explicitly penalizes per-frame growth of residual error, replacing the implicit error-damping role of the removed autoencoder bottleneck.', 'A multi-scale (coarse-to-fine) residual correction head that re-injects pixel residuals at each denoising step, coupled to the growth penalty so correction targets temporal error accumulation rather than generic detail.', 'An empirical design principle: pixel-space diffusion quality is bounded not by patch capacity but by unbounded temporal residual growth, which a training-time penalty can monitor and bound.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | UCF-101 class-conditioned clips (16 frames, 128x128 resolution) | Standard pixel-space video benchmark; motion-heavy regime stresses the motion-compensation assumption |
| Primary metric | Temporal high-pass residual growth rate (motion-compensated, motion-normalized) | Directly measures the quantity we penalize |
| Secondary metric | FVD (Fréchet Video Distance) and warping error (optical-flow-based) | Independent, non-circular temporal-consistency outcomes |
| Baseline | JVT pixel-space DiT | Tests gain over the base architecture |
| Baseline | Latent diffusion video baseline | Tests whether pixel-space + TERRA closes gap |
| Baseline | AR-Diffusion timestep-constraint model | Tests residual-space vs velocity-space regularization |
| Ablation-of-ours | TERRA without growth penalty | Isolates penalty from residual heads |
| Ablation-of-ours | TERRA with raw (non-motion-compensated) Laplacian residual | Tests whether motion compensation is necessary for the error signal |
| Ablation-of-ours | TERRA with learned frequency-band decomposition instead of fixed Laplacian | Tests whether the fixed filter is an incidental choice |
| Ablation-of-ours | TERRA with latent-space residual-growth penalty | Tests whether pixel-space residual signal is the necessary ingredient |
| Diagnostic | GT high-pass energy vs. denoiser residual on same clips | Quantifies motion-dominance assumption before building penalty |

### Why this setup validates the claim

The central claim is that coupling multi-scale residual correction heads to a temporal high-pass residual-growth penalty flattens per-step error accumulation in pixel-space DiT video diffusion. JVT is the right base baseline because TERRA is a modification of it: any gain over JVT isolates the contribution of the residual heads plus penalty rather than architecture. The latent diffusion baseline tests the secondary claim that pixel-space diffusion with explicit residual regularization can match or exceed latent methods on temporal consistency, which is where pixel-space methods are presumed weak. AR-Diffusion tests the specific claim that our penalty differs from a timestep-space consistency constraint: if AR-Diffusion already flattens residual growth, our mechanism is redundant. The no-penalty ablation is the decisive control: it holds the residual heads fixed and removes only the growth term, so any remaining gain must come from the penalty, not from extra parameters. The raw-residual and learned-decomposition ablations test whether the motion-compensated, fixed-Laplacian form is necessary rather than incidental. The latent-space penalty ablation tests whether operating on pixel residuals is the necessary ingredient. The diagnostic experiment directly measures the ratio of GT high-pass energy to denoiser residual on the same clips, providing empirical justification for the motion-dominance assumption before the penalty is trusted. Temporal high-pass residual growth is the right primary metric because it is exactly the quantity the penalty optimizes; if the causal story is correct, the metric should move where the penalty is active (motion-heavy clips) and stay flat where it is inactive (static clips), which no generic perceptual metric would reveal. FVD and warping error are included as independent, non-circular outcomes so that the proxy is validated against standard temporal-consistency measures.

### Expected outcome and causal chain

**vs. JVT pixel-space DiT** — On a case with sustained moderate motion, JVT's large-patch trunk accumulates high-frequency synthesis error frame over frame because nothing in its loss constrains inter-frame residual growth, so late frames drift and blur. TERRA's multi-scale heads supply coarse-to-fine corrections while the growth penalty explicitly flattens the residual-energy trajectory, so we expect a visible reduction in late-frame residual growth concentrated in the second half of 16-frame windows, with near-parity on static clips. We further expect TERRA to show lower FVD and warping error than JVT, confirming that the residual-growth reduction translates to independent temporal-consistency gains.

**vs. Latent diffusion video baseline** — On a case with fine texture and rapid local motion, the latent baseline loses high-frequency detail through the autoencoder bottleneck, producing temporally flickering texture because decoder reconstruction error is unconstrained across frames. TERRA operates in pixel space with an explicit Laplacian residual signal, so its correction heads act directly on the frequencies the baseline discards; we expect TERRA to show lower residual growth on texture-rich clips while the two are comparable on smooth low-detail clips. On FVD, we expect TERRA to be competitive or better, with the largest gains on texture-rich subsets.

**vs. AR-Diffusion timestep-constraint model** — On a case requiring long coherent motion, AR-Diffusion enforces a non-decreasing corruption schedule in velocity space, which constrains frame ordering but not measured pixel residual energy, so high-frequency error can still grow within the allowed schedule. TERRA's penalty is applied to the actual high-pass pixel residual, so it targets the error the schedule leaves free; we expect a moderate but consistent gap on motion-heavy subsets and parity on short, low-motion clips. Warping error should favor TERRA on motion-heavy subsets if the penalty is acting through the claimed error-accumulation pathway.

### What would falsify this idea

If the no-penalty ablation matches full TERRA on temporal residual growth, the growth penalty is not the causal driver and the claim collapses to a plain residual-head story. If TERRA's gain over JVT is uniform across motion-heavy and static subsets rather than concentrated where residual growth is predicted to occur, the mechanism is not acting through the claimed error-accumulation pathway. If the diagnostic experiment shows that GT high-pass energy dominates denoiser residual on motion-heavy clips, the motion-dominance assumption is false and the penalty is over-smoothing rather than damping error.

## References

1. JVT: Just Video Transformer makes pixel-space video diffusion simple
