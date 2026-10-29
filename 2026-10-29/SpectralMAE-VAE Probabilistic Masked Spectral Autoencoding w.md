# SpectralMAE-VAE: Probabilistic Masked Spectral Autoencoding with Calibrated Stellar Parameter Posteriors

## Motivation

Existing spectral representation learning, exemplified by OmniSpectra's continuous-coordinate masked autoencoder, treats the latent code as a deterministic point estimate and the reconstruction objective as a pixel-wise (or coordinate-wise) L2/MAE loss, which structurally cannot represent the multi-modal, heteroscedastic posteriors that probabilistic catalogs such as PROVABGS produce for stellar mass and SFR. Because the encoder is optimized only to minimize reconstruction error, any uncertainty information present in the input spectrum or in the catalog labels is discarded at the bottleneck, so downstream tasks that require calibrated error bars (e.g., mass-function estimation, SFR--mass diagram population studies) inherit uncalibrated, systematically overconfident representations. The root cause is the deterministic reconstruction objective itself, not the architecture: the decoder is never asked to produce a distribution, so the latent space has no incentive to encode uncertainty.

## Key Insight

A variational information bottleneck whose decoder is a conditional generative model of both masked spectral bins and catalog-parameter posterior samples forces the latent code to be a sufficient statistic of the joint (spectrum, parameter-posterior) distribution, so calibrated catalog uncertainty is preserved by construction rather than learned as an afterthought.

## Method

## SpectralMAE-VAE: Probabilistic Masked Spectral Autoencoding

**(A) What it is.** SpectralMAE-VAE is a conditional variational autoencoder that takes a partially masked spectrum on continuous wavelength coordinates plus a mask pattern, and outputs (i) a predictive distribution over the masked spectral bins and (ii) a posterior distribution over stellar parameters (stellar mass, SFR) that is trained to match the PROVABGS-like catalog posterior samples. The latent code is a diagonal Gaussian whose parameters are predicted by the encoder; the decoder is a resolution-gated Fourier implicit neural network that emits per-bin means and variances, and a parameter head that emits a Gaussian mixture over (log M*, log SFR).

**Load-bearing assumption (stated explicitly).** The method's calibration claim rests on the assumption that the PROVABGS posterior samples used to train the parameter head are unbiased draws from the true stellar-parameter posterior, so that minimizing the empirical NLL L_param transfers calibration to q_φ(p|z). This assumption is known to fail in low-S/N and quiescent regimes where SED-fitting posteriors are biased by prior-volume effects and template-set choice. We therefore do not claim calibration 'by construction'; instead, we treat the catalog posterior as a reference target and add an explicit post-hoc calibration step (Phase 5) evaluated against an independent truth set, and we report the catalog's own coverage against that truth set so that our calibration is measured relative to, not inherited from, PROVABGS bias.

**(B) How it works.**

```pseudocode
Input: spectrum s = {(λ_i, f_i, σ_i)}_{i=1..N} on continuous λ, mask m ∈ {0,1}^N, catalog posterior samples P = {p_k = (logM*_k, logSFR_k)}_{k=1..K} (K=64 from PROVABGS)

Phase 1 — Coordinate encoding (reuse OmniSpectra):
  x_i = log(λ_i)                                  # logarithmic wavelength basis
  e_i = ContinuousRotaryEmbed(x_i)                # shift-invariant relative position
  tokens = MLP([e_i ; f_i ; σ_i ; m_i])           # visible tokens only (mask ratio ρ=0.6)

Phase 2 — Probabilistic encoder:
  h = TransformerEncoder(tokens)                  # 8 layers, d=256, 8 heads
  μ_z, logσ²_z = Linear(h_pool)                   # diagonal Gaussian q(z|s_vis, m), dim(z)=32
  z ~ q(z|s_vis, m) = N(μ_z, diag(exp(logσ²_z)))  # reparameterized sample

Phase 3 — Conditional decoder:
  # (3a) spectral reconstruction with heteroscedastic variance
  for each masked coordinate λ_j:
     μ_f(λ_j), logσ²_f(λ_j) = FourierINR(z, λ_j)  # resolution-gated, anti-aliased
  L_spec = Σ_j [ (f_j − μ_f(λ_j))² / (2 exp(logσ²_f(λ_j))) + 0.5·logσ²_f(λ_j) ]

  # (3b) parameter posterior reconstruction via Gaussian mixture
  {π_c, μ_c, Σ_c}_{c=1..C} = ParamHead(z)          # C=4 components, full 2×2 Σ_c
  q_φ(p|z) = Σ_c π_c · N(p ; μ_c, Σ_c)
  L_param = −(1/K) Σ_k log q_φ(p_k|z)             # fit to catalog posterior samples

Phase 4 — Joint training objective:
  L = L_spec + λ_param · L_param + β · KL(q(z|s_vis,m) ‖ N(0,I))
  # λ_param = 1.0, β annealed 0 → 0.1 over first 20k steps (β-VAE warmup)

Phase 5 — Post-hoc calibration (applied after training, before reporting coverage):
  # Fit a calibration map on a held-out calibration set of 512 spectra with independent truth parameters
  # (simulation-truth or cross-catalog estimates). Calibration map: per-parameter temperature scaling
  # T_M, T_SFR fitted by minimizing NLL of q_φ(p|z) on the calibration set; optionally a 2×2 affine
  # transform on (logM*, logSFR) if temperature alone leaves residual miscalibration.
  q_cal(p|z) = TemperatureScale(q_φ(p|z); T_M, T_SFR)
  # Report coverage of q_cal(p|z) against the truth set, and also report coverage of the raw
  # PROVABGS posterior against the same truth set as a reference baseline.

Output at inference: z (latent), q_φ(p|z) (raw parameter posterior), q_cal(p|z) (calibrated posterior), predictive spectrum distribution
```

**(C) Why this design.** We chose a conditional VAE with a heteroscedastic Gaussian spectral likelihood over a deterministic L2 reconstruction (as in OmniSpectra) because the deterministic objective forces the latent code to encode only the conditional mean of the spectrum, discarding heteroscedastic noise that is itself informative about parameter uncertainty; the cost is that the reconstruction metric is noisier and requires variance calibration, which we accept because downstream tasks need calibrated error bars. We chose to fit the parameter head to PROVABGS posterior *samples* via a Gaussian mixture rather than regressing point-estimate labels because the catalog posterior is multi-modal in the SFR--mass plane (star-forming vs. quiescent branches), and a unimodal head would collapse the modes; the cost is C=4 mixture components and a heavier head, which we accept because it preserves branch structure. We chose KL annealing (β: 0→0.1) over a fixed β because early training with a strong KL term collapses z to the prior before the decoder learns the joint structure; the cost is 20k extra warmup steps, accepted for stability. We chose to keep the resolution-gated Fourier decoder from OmniSpectra rather than a pixel-index decoder because continuous coordinates are required for redshift-invariant encoding, and the anti-aliasing gate prevents the decoder from hallucinating high-frequency structure that would inflate the learned variance. **Unlike OmniSpectra, which is a deterministic masked autoencoder, and unlike standard VAE-based spectral models that treat the latent as a nuisance for generation, SpectralMAE-VAE couples the spectral likelihood and the parameter-posterior likelihood through a shared latent code, so the latent is trained to be a sufficient statistic of both — a coupling that neither a deterministic MAE nor a standalone regression head produces.** The post-hoc calibration step (Phase 5) is included because the catalog posterior is a biased reference target in low-S/N regimes; without it, coverage would reflect fidelity to PROVABGS bias rather than to the true posterior.

**(D) Why it measures what we claim.** The latent code z is claimed to encode *calibrated catalog uncertainty*, and this is operationalized by two computational quantities. First, the KL term KL(q(z|s_vis,m) ‖ N(0,I)) measures *uncertainty compression* because the information-bottleneck assumption states that the minimal KL consistent with the reconstruction likelihood is the sufficient statistic of the input; this assumption fails when the mask ratio ρ is so high that the visible spectrum is uninformative about the masked bins, in which case the KL reflects prior collapse rather than calibrated compression. Second, the parameter head's negative log-likelihood L_param measures *posterior fidelity to the catalog* because the assumption that PROVABGS posterior samples are draws from the true parameter posterior makes the empirical NLL an unbiased estimator of the forward KL between q_φ(p|z) and the catalog posterior; this assumption fails when the catalog posterior is itself biased (e.g., prior-volume effects in low-S/N regimes), in which case L_param reflects fidelity to the catalog's bias rather than to the true posterior. The heteroscedastic spectral variance exp(logσ²_f) measures *aleatoric spectral noise* because the Gaussian likelihood assumption states that residuals are drawn from the predicted variance; this assumption fails for instrumental artifacts (sky-line residuals, flux-calibration systematics) that are heteroscedastic but not Gaussian, in which case the variance reflects artifact magnitude rather than physical noise. The post-hoc calibration map (Phase 5) measures *calibration relative to an independent truth set* because the assumption that the truth set is unbiased makes the coverage of q_cal(p|z) an unbiased estimate of true-posterior coverage; this assumption fails when the truth set is itself biased (e.g., simulation-truth mismatch with real data), in which case the calibration map reflects the truth-set bias rather than true calibration.

## Contribution

(1) A probabilistic masked spectral autoencoder (SpectralMAE-VAE) whose latent code is a variational distribution jointly trained on a heteroscedastic spectral reconstruction likelihood and a Gaussian-mixture parameter-posterior likelihood, making the latent a sufficient statistic of the joint (spectrum, catalog-posterior) distribution. (2) A design principle: coupling the spectral reconstruction and parameter-posterior heads through a shared variational bottleneck produces latent representations whose downstream uncertainty estimates track catalog posterior width, whereas decoupled deterministic MAE + regression heads do not. (3) A public training recipe and evaluation protocol for probabilistic spectral representation learning on PROVABGS-like catalogs, including the KL-annealing schedule and mask-ratio regime where the information bottleneck remains informative.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PROVABGS spectra + catalog posteriors | Provides both spectra and multimodal parameter posteriors |
| Calibration set | 512 held-out spectra with independent truth parameters (simulation-truth or cross-catalog estimates) | Enables post-hoc calibration and truth-referenced coverage |
| Primary metric | Calibration of parameter posterior (coverage + NLL) against truth set, plus coverage of raw PROVABGS posterior against same truth set | Directly tests calibrated uncertainty claim relative to catalog bias |
| Baseline | OmniSpectra (deterministic MAE) | Tests value of probabilistic latent coupling |
| Baseline | Standard VAE + regression head | Tests value of posterior-sample fitting vs point labels |
| Baseline | Point-estimate MLP regressor | Tests value of uncertainty modeling at all |
| Baseline | Raw PROVABGS posterior (catalog reference) | Measures catalog's own coverage against truth set |
| Ablation-of-ours | SpectralMAE-VAE w/o heteroscedastic variance | Isolates contribution of learned aleatoric noise |
| Ablation-of-ours | SpectralMAE-VAE w/o post-hoc calibration | Isolates contribution of calibration map |

### Why this setup validates the claim

The central claim is that coupling a heteroscedastic spectral likelihood with a posterior-sample-fitting parameter head through a shared latent produces a latent code that is a calibrated sufficient statistic of catalog uncertainty, subject to the explicit assumption that the catalog posterior is an unbiased reference target. PROVABGS is the right dataset because it supplies both the spectra and the multimodal catalog posterior samples needed to train and evaluate that claim, so the calibration metric measures the thing we assert rather than a proxy. The held-out calibration set of 512 spectra with independent truth parameters is essential because it allows us to measure coverage against truth rather than against the catalog, and to report the catalog's own coverage as a reference baseline; without it, we could not distinguish our method's calibration from inherited PROVABGS bias. Each baseline isolates one sub-claim: OmniSpectra's deterministic MAE tests whether probabilistic coupling buys calibration over a strong deterministic encoder; the standard VAE with a regression head tests whether fitting posterior samples beats regressing point labels; the point-estimate MLP tests whether uncertainty modeling matters at all; the raw PROVABGS posterior provides the catalog reference. The ablations remove heteroscedastic variance and post-hoc calibration to test whether the learned aleatoric term and the calibration map are what carry calibration. Coverage-and-NLL against truth is the correct primary metric because a calibrated posterior is exactly what the claim asserts, and miscalibration is the failure mode the design predicts; a point-accuracy metric would be blind to it.

### Expected outcome and causal chain

**vs. OmniSpectra (deterministic MAE)** — On a low-S/N spectrum near the star-forming/quiescent boundary, OmniSpectra produces a single confident parameter estimate because its L2 objective collapses the latent to the conditional mean and discards heteroscedastic noise. Our method instead samples z from a Gaussian posterior and emits a mixture over parameters because the heteroscedastic likelihood and posterior-sample NLL force the latent to retain spread, so we expect clearly better coverage and lower NLL on the ambiguous subset with parity on high-S/N spectra, after post-hoc calibration.

**vs. Standard VAE + regression head** — On a bimodal SFR--mass case where the catalog posterior splits between branches, the regression-head VAE produces a unimodal prediction near the branch midpoint because a Gaussian point-regression loss cannot represent two modes. Our method instead fits a four-component Gaussian mixture to the posterior samples because L_param is the empirical NLL over K samples, preserving branch structure, so we expect a visible gap in mixture-aware calibration metrics on bimodal cases but similar performance on unimodal ones.

**vs. Point-estimate MLP regressor** — On any spectrum where catalog uncertainty is large, the MLP regressor outputs a fixed-width prediction with no uncertainty because it is trained only on point labels and has no variance channel. Our method instead outputs a full posterior whose width tracks the input because the latent KL and heteroscedastic variance encode input-dependent uncertainty, so we expect the MLP to show severe under-coverage everywhere while our coverage tracks the nominal level, with the gap widening as S/N drops.

**vs. Raw PROVABGS posterior (catalog reference)** — On low-S/N spectra where prior-volume effects bias the catalog, the raw PROVABGS posterior may itself be over- or under-confident against the truth set. Our method's post-hoc calibration map is fitted on the held-out calibration set to correct residual miscalibration, so we expect our calibrated coverage to be closer to nominal than the raw catalog's coverage, with the improvement concentrated in the low-S/N regime where the catalog bias is largest.

### What would falsify this idea

If our calibration gain over the deterministic and regression baselines is uniform across S/N and bimodality subsets rather than concentrated exactly where the predicted failure modes (low S/N, multimodal posteriors) occur, then the coupling is not doing the claimed work and the central claim is wrong. Equally, if removing the heteroscedastic variance or the post-hoc calibration step leaves calibration unchanged, the aleatoric-noise or calibration-map mechanism we credit is not the cause.

## References

1. OmniSpectra: A Unified Foundation Model for Universal Spectra Representation Learning
