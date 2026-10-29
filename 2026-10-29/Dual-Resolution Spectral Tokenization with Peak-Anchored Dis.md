# Dual-Resolution Spectral Tokenization with Peak-Anchored Discrete Tokens for Cross-Domain Spectral Transfer

## Motivation

OmniSpectra and similar continuous-coordinate masked autoencoders assume a single smooth latent geometry can span all spectra, encoding position through continuous rotary embeddings and reconstructing via a band-limited Fourier decoder. This assumption is structurally wrong for discrete line-structured spectra (e.g., molecular rovibrational bands), where energy concentrates on near-delta spectral lines that a band-limited continuous representation necessarily blurs — the Fourier decoder's anti-aliasing explicitly suppresses exactly the high-frequency content that carries the line identity. The field's trajectory toward universal spectral foundation models therefore hits a representational bottleneck: extending to more surveys (eight in OmniSpectra) cannot fix a mismatch that is geometric rather than data-volume-driven.

## Key Insight

Sharp spectral lines are sparse, near-Dirac events whose identity is carried by their positions and relative intensities rather than by smooth amplitude envelopes, so a representation that allocates discrete token slots to detected line coordinates is invariant to the band-limiting that continuous-coordinate encoders impose, letting one latent space hold both smooth continua and discrete line structure without blurring either.

## Method

**What it is.** Peak-Anchored Dual-Resolution Spectral Tokenization (PADST) is a masked-autoencoder tokenizer that emits, for each spectrum, a set of continuous-coordinate tokens (as in OmniSpectra) plus a set of discrete line tokens produced by a multi-scale peak detector; both token streams are fused by a coordinate-conditioned cross-attention block and decoded by a hybrid Fourier-plus-line-synthesis decoder. Inputs: a spectrum sampled on physical wavelengths plus its instrument resolution curve. Outputs: a fused latent sequence and a reconstructed spectrum.

**Load-bearing assumption (stated explicitly).** The 4-tuple [lambda_p, log(A_p), width_p, snr_p] extracted by the multi-scale LoG detector is a sufficient statistic for each line's identity, so that additive Voigt synthesis from these tuples reconstructs line-dominated regions more faithfully than a band-limited Fourier decoder. This assumption is valid only for isolated, high-SNR lines; it fails for blended or low-SNR lines, where the detector's tuple is a biased centroid fit of a multi-component profile. The method below is specified so that this assumption is testable and its failure mode is handled by the per-peak mixture token and the deblending-aware auxiliary loss.

**How it works (pseudocode).**
```pseudocode
# Stage 1: continuous branch (OmniSpectra-style)
z_cont = ContinuousMAEEncoder( log_coords(x), x )        # log-coord rotary embeddings, d=256

# Stage 2: discrete line branch
for sigma in [1, 2, 4, 8] px:                              # multi-scale kernels
    s_sigma = GaussianSmooth(x, sigma)
    d_sigma = -sigma^2 * Laplacian(s_sigma)                # scale-normalized DoG response
    P_sigma = local_maxima(d_sigma, threshold=3*MAD(d_sigma), min_sep=sigma)
P = merge_peaks(P_sigma across scales, non_max_radius=2 px) # union with dedup
for each peak p in P:
    # per-peak mixture token: K-component Voigt with learned K and component weights
    K_p = 3  # fixed maximum number of components per peak token
    tok_p = LineTokenMLP([lambda_p, log(A_p), width_p, snr_p, {w_k, mu_k, sigma_k, gamma_k}_{k=1..K_p}])  # 4 + 4*K_p dims -> d=256
z_line = TransformerEncoder(P_tokens, n_layers=2)          # line-context encoder

# Stage 3: fusion
z_fused = CrossAttention(query=z_cont, key=z_line, value=z_line)  # coordinate-conditioned

# Stage 4: hybrid decoder
x_hat = FourierINR(z_fused, coords) + sum_p sum_{k=1..K_p} w_{p,k} * Voigt(coords; mu_{p,k}, sigma_{p,k}, gamma_{p,k})
loss = L1(x_hat, x) + lambda_aux * L1(peak_amplitude_head(z_line), A_P)  # lambda_aux=0.1
# deblending-aware auxiliary loss: supervise token against multi-component fit
loss_aux_deblend = L1(peak_amplitude_head(z_line), A_P_multi)  # A_P_multi from independent multi-Voigt fit
loss = loss + lambda_deblend * loss_aux_deblend  # lambda_deblend=0.1
```

**Why this design.** We chose a scale-normalized Laplacian-of-Gaussian peak detector over a learned line-proposal network because LoG responses are provably invariant to line amplitude and respond only to local curvature, so the token set depends on line *positions* rather than on overall flux normalization — accepting the cost that very broad lines (width > 8 px) are not tokenized and fall back to the continuous branch. We chose to encode each peak as an explicit 4-tuple (wavelength, log-amplitude, width, SNR) rather than a patch embedding because the tuple is the minimal sufficient statistic for a Voigt profile, letting the decoder synthesize lines analytically — accepting that overlapping blends are approximated as additive Voigts. We chose coordinate-conditioned cross-attention (continuous tokens as queries, line tokens as keys/values) over concatenation because concatenation forces the continuous branch to re-learn line positions from scratch, whereas cross-attention injects line coordinates into the continuous latent at the exact query positions — accepting a modest parameter increase (≈1.1M). An expert would not describe this as OmniSpectra-plus-a-peak-finder: OmniSpectra's Fourier decoder is band-limited by construction and cannot represent near-delta events, whereas PADST routes line energy through an analytic synthesis path that bypasses the band limit entirely; the discrete tokens are not a preprocessing feature but a second representational substrate fused into the latent.

**Why it measures what we claim.** The scale-normalized LoG response `d_sigma` measures *line presence* because a local-maximum condition on the scale-normalized response is equivalent to a line being narrower than the kernel scale under the assumption that lines are locally Voigt-shaped; this assumption fails when two lines are separated by less than `sigma`, in which case the detector reports a single merged peak and the token reflects a blended centroid rather than either line. The 4-tuple `[lambda_p, log(A_p), width_p, snr_p]` measures *discrete line identity* because log-amplitude and width are the sufficient statistics of a Voigt profile under the assumption that lines do not overlap; this fails for blended lines, where the tuple reflects a best-fit single profile absorbing the blend. The cross-attention fusion measures *structural compatibility between continuous and discrete representations* because attention weights concentrate at query coordinates near detected peaks under the assumption that the continuous latent already encodes approximate line locations; this fails on spectra with no detectable peaks, where attention degenerates to uniform and the fused latent reduces to the continuous branch alone. The auxiliary amplitude head loss measures *token faithfulness to detected lines* because it supervises the token embedding to predict the detector's own amplitude under the assumption that the detector is correct; this fails on false-positive detections, where the loss teaches the token to encode noise peaks, so we gate it with `snr_p > 5`. The deblending-aware auxiliary loss measures *token faithfulness to true line identity* because it supervises the token against an independently fit multi-component Voigt amplitude under the assumption that the multi-component fit is closer to ground truth than the single-peak detector; this fails when the multi-component fit is itself ill-conditioned (e.g., low SNR), in which case the loss falls back to the single-peak amplitude.

## Contribution

(1) A dual-substrate spectral tokenizer that pairs continuous-coordinate tokens with discrete peak-anchored tokens produced by a scale-normalized multi-scale LoG detector, fused by coordinate-conditioned cross-attention. (2) A hybrid Fourier-plus-Voigt decoder that reconstructs band-limited continua and near-delta line structure through separate analytic paths, removing the band-limit bottleneck of continuous-coordinate MAEs. (3) An empirical design principle: discrete line tokens transfer across instruments with different resolution curves because they encode positions and shapes rather than pixel patterns, which we test by cross-instrument transfer from astronomical to molecular spectra.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Survey spectra with sharp emission/absorption lines (e.g., SDSS-class rest-frame) | Contains near-delta features Fourier decoders cannot represent |
| Primary metric | Reconstruction error on line-dominated spectral regions | Directly probes the predicted band-limit failure mode |
| Baseline | OmniSpectra (Fourier INR decoder) | Tests whether analytic line synthesis beats band-limited decoding |
| Baseline | OmniSpectra + peak-detector preprocessing (concatenation fusion) | Tests whether fusion mechanism, not just peaks, drives gain |
| Baseline | Continuous-only PADST (no line branch) | Isolates contribution of discrete tokens |
| Ablation-of-ours | PADST with concatenation fusion instead of cross-attention | Tests the coordinate-conditioned attention sub-claim |
| Ablation-of-ours | PADST with single-Voigt token (no mixture) and no deblending loss | Tests the load-bearing assumption on blends |

### Why this setup validates the claim

The central claim is that routing line energy through an analytic synthesis path bypasses the band limit of a Fourier decoder, and that coordinate-conditioned cross-attention is the right way to inject discrete line identity into the continuous latent. This design makes each sub-claim separately falsifiable. The OmniSpectra baseline tests the band-limit claim: if analytic synthesis matters, our gain must concentrate on line-dominated regions and vanish on smooth continuum regions — a uniform gain would refute the mechanism. The concatenation-fusion baseline (both the external "OmniSpectra + peak preprocessing" and the internal ablation) tests the fusion sub-claim: if cross-attention is doing real work rather than just adding peak features, it must outperform concatenation specifically where line coordinates must be aligned to continuous query positions. The continuous-only ablation tests whether discrete tokens contribute at all. The single-Voigt ablation tests the load-bearing assumption: if the mixture token and deblending loss are necessary, the single-Voigt variant must underperform specifically on blended-line subsets, confirming that the 4-tuple sufficiency assumption fails exactly where predicted. Reconstruction error restricted to line-dominated regions is the correct metric because it is the only place where the predicted mechanism (analytic line synthesis) can manifest; averaging over the whole spectrum would dilute exactly the signal the claim predicts.

### Expected outcome and causal chain

**vs. OmniSpectra (Fourier INR decoder)** — On a case with a sharp, narrow emission line, the baseline produces a smoothed, broadened reconstruction because its Fourier decoder is band-limited and cannot represent near-delta events. Our method instead synthesizes that line analytically via the Voigt path, bypassing the band limit entirely, so we expect a large gap on line-dominated regions and near-parity on smooth continuum regions — the gap should track line sharpness.

**vs. OmniSpectra + peak-detector preprocessing (concatenation fusion)** — On a case where a line sits at a coordinate the continuous branch has only approximately localized, the baseline produces misaligned line energy because concatenation forces the continuous branch to re-learn positions from scratch and cannot bind peaks to queries. Our method instead injects line coordinates at the exact query positions via cross-attention, so we expect a gap that grows with the number of closely spaced lines and shrinks to parity on isolated single lines.

**vs. Continuous-only PADST (no line branch)** — On a case with a blend of two close lines, the baseline produces a merged, over-smoothed profile because it has no discrete substrate to carry line identity. Our method instead routes the blend through additive Voigt tokens, so we expect a moderate gap concentrated on blended-line subsets, with parity on clean single lines — confirming the discrete branch's contribution.

**vs. PADST with single-Voigt token (no mixture) and no deblending loss** — On a case with a blended multiplet (e.g., [OII] doublet), the baseline produces a biased centroid and amplitude because its single-Voigt token cannot represent the blend, and the auxiliary loss supervises it toward the detector's blended amplitude. Our method instead uses a per-peak mixture token and a deblending-aware loss against an independent multi-component fit, so we expect a gap concentrated on blended-line subsets, with parity on isolated lines — confirming that the load-bearing assumption fails exactly where predicted and that the mixture token repairs it.

### What would falsify this idea

If our gain over OmniSpectra were uniform across line-dominated and smooth-continuum regions rather than concentrated where the band-limit failure is predicted, the analytic-synthesis mechanism would be refuted. Likewise, if cross-attention showed no advantage over concatenation specifically on closely spaced lines, the fusion sub-claim would be wrong. If the mixture token and deblending loss showed no advantage over the single-Voigt variant on blended-line subsets, the load-bearing assumption would be refuted.

## References

1. OmniSpectra: A Unified Foundation Model for Universal Spectra Representation Learning
