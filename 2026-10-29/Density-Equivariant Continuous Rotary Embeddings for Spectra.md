# Density-Equivariant Continuous Rotary Embeddings for Spectra

## Motivation

OmniSpectra assumes spectra are sampled on a continuous physical coordinate with a fixed monotone mapping, so its continuous rotary embeddings encode relative positions as if sampling were uniform; when the acquisition pattern is non-uniform (e.g., variable resolution, masked or irregular sampling), the latent code changes with the sampling density rather than only with the underlying spectrum. This is a structural gap: shift-invariance was achieved by canceling a shared redshift shift, but density-invariance was never modeled, so the representation remains tied to the acquisition pattern rather than to the physical signal.

## Key Insight

Sampling density is an acquisition nuisance that can be separated from the physical spectrum by making the kernel bandwidth and the pooling weights explicit functions of local density, so the latent code is equivariant to any resampling of the same underlying continuous spectrum.

## Method

**What it is.** We introduce DE-CRO (Density-Equivariant Continuous Rotary Embedding), a drop-in replacement for the continuous rotary position embedding and decoder pooling in OmniSpectra. Inputs: a set of tokens with physical coordinates $x_i$ and values $v_i$, plus a local density estimate $\rho_i$. Outputs: a latent code $z$ that is invariant to non-uniform resampling of the same underlying spectrum, and a reconstructed spectrum on any target grid.

**Load-bearing assumption (stated explicitly).** The method assumes that the density-induced distortion of the latent code is fully captured by (i) a scalar local sampling density $\rho_i$ that multiplicatively rescales the rotary phase via $\theta_i = x_i \omega_0 / \text{bandwidth}_i$, and (ii) inverse-density pooling weights $w_i \propto 1/\rho_i$, so that a single learned scalar gate per token can cancel the density effect and make $z$ invariant to any non-uniform resampling of the same underlying continuous spectrum. This assumption holds when the sampling is locally Poisson (so the k-NN distance scales as $\rho^{-1/2}$) and when the local sample spacing does not exceed the Nyquist limit of the decoder's Fourier bands. It fails when the sampling is strongly clustered or lattice-like (so $\rho_i$ reflects lattice spacing rather than smooth density) or when sparse-region spacing exceeds the decoder's Nyquist limit (so aliasing is irrecoverable by phase rescaling or reweighting).

**How it works.**

```python
# Phase 1: density estimation
# For each token i, estimate local sampling density from k-NN distances in physical coordinate
for i in tokens:
    d_i = distance to k-th nearest neighbor in x-space   # k=8
    rho_i = (k / (pi * d_i^2))                           # local density estimate
    rho_i = rho_i / median(rho)                          # normalize by median density

# Phase 2: density-modulated continuous rotary embedding
# Base rotary frequency for coordinate x: theta(x) = x * omega_0
# Modulate the kernel bandwidth by a learned function of log-density
for i in tokens:
    g_i = MLP(log(rho_i))                                # 2-layer MLP, hidden=32, GeLU, outputs scalar gate in (0,1) via sigmoid
    bandwidth_i = bandwidth_0 * exp(g_i)                 # bandwidth_0 = 1.0
    # Continuous rotary embedding with density-aware kernel:
    # R_i = [cos(theta(x_i)), sin(theta(x_i))] with theta scaled by bandwidth_i
    theta_i = x_i * omega_0 / bandwidth_i
    R_i = [cos(theta_i), sin(theta_i)]

# Phase 3: density-invariant pooling
# Aggregate token latents h_i (from transformer encoder) using density-corrected weights
for i in tokens:
    w_i = 1 / (rho_i + eps)                              # inverse-density weight, eps=1e-6
    w_i = w_i / sum_j(w_j)                               # normalize
    w_i = min(w_i, percentile(w, 95))                    # clip at 95th percentile
    w_i = w_i / sum_j(w_j)                               # renormalize after clipping
z = sum_i w_i * h_i                                       # pooled latent code

# Phase 4: resolution-gated Fourier decoder (unchanged from OmniSpectra)
# Reconstruct spectrum on target grid x' using z and density-aware Fourier features
for x' in target_grid:
    rho_x' = interpolate_density(x')                     # interpolate density to target coordinate
    phi(x') = [cos(2*pi*f_j*x' / bandwidth(x')), sin(...)] for j in Fourier bands
    s(x') = Decoder(z, phi(x'))
```

**Why this design.** We chose to modulate the rotary embedding bandwidth by a learned function of log-density rather than to simply concatenate density as an extra feature, because concatenation leaves the relative-position encoding unchanged and therefore cannot cancel density-induced phase shifts; the cost is that the bandwidth gate must be trained jointly with the encoder, which we handle by initializing the MLP to output near-zero so pretraining starts from the original OmniSpectra behavior. We chose inverse-density pooling weights over learned attention pooling because inverse-density weighting has a closed-form unbiasedness property for Monte-Carlo integration over non-uniform samples, whereas learned pooling would need to rediscover this weighting from data and could overfit to survey-specific sampling patterns; the cost is that inverse-density weights can be noisy when density estimates are poor, which we mitigate by normalizing by the median density and clipping weights at the 95th percentile. We chose to keep the resolution-gated Fourier decoder unchanged rather than to make the decoder density-aware, because the decoder already operates on continuous target coordinates and the density correction is fully absorbed into the latent code $z$; the cost is that if the decoder's Fourier features themselves alias under extreme non-uniformity, the reconstruction may still show artifacts, which we accept as a boundary condition. A domain expert would not describe this as a variant of OmniSpectra because the core mechanism—density-modulated rotary bandwidth plus inverse-density pooling—introduces a new equivariance (density equivariance) that OmniSpectra's fixed monotone mapping cannot express, and the learned bandwidth gate is a structural modification to the embedding rather than a new head on top.

**Why it measures what we claim.** The computational quantity $\rho_i$ (local k-NN density) measures the local sampling density because the k-th nearest-neighbor distance in a Poisson process is inversely proportional to the square root of density; this assumption fails when the sampling pattern is strongly clustered or lattice-like, in which case $\rho_i$ reflects the lattice spacing rather than a smooth density. The bandwidth gate $g_i = \text{MLP}(\log \rho_i)$ measures the density-induced phase distortion because the rotary phase $\theta(x_i) = x_i \omega_0 / \text{bandwidth}_i$ is designed so that a change in density rescales the effective frequency; this assumption fails when the MLP collapses to a constant, in which case the embedding reverts to OmniSpectra and density equivariance is lost. The inverse-density pooling weight $w_i = 1/\rho_i$ measures the unbiased Monte-Carlo contribution of each sample to the underlying continuous spectrum because inverse-density weighting is the standard importance-sampling correction for non-uniform samples; this assumption fails when density is estimated with high variance, in which case the pooled latent code reflects estimation noise rather than the true spectrum. Together, these three quantities operationalize density equivariance: if the same underlying spectrum is resampled with a different density profile, $\rho_i$ changes, the bandwidth gate compensates the phase, and the inverse-density weights reweight the pool so that $z$ remains approximately constant—exactly the invariance the motivation claims.

**Diagnostic for assumption A (local Poisson sampling).** To report when the k-NN density estimator is valid, we compute for each token the variance of the distances to its $k=8$ nearest neighbors, normalized by the mean distance squared. A value near 1 indicates Poisson-like sampling; values substantially above 1 indicate clustering or lattice-like structure. We report the fraction of tokens with normalized variance > 2 as the violation rate of assumption A.

## Contribution

['A density-equivariant continuous rotary embedding (DE-CRO) that modulates the embedding bandwidth by a learned function of local sampling density, making the latent code invariant to non-uniform resampling of the same underlying spectrum.', 'A density-invariant pooling layer with inverse-density importance weights that provides an unbiased aggregation over non-uniform samples, replacing the uniform-mean pooling in OmniSpectra.', 'An empirical demonstration on multi-survey spectral data that density equivariance improves downstream representation quality under variable resolution and irregular sampling compared to the fixed-monotone-mapping baseline.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Synthetic spectra with known ground-truth continuous spectrum, sampled under controlled density profiles (uniform, dense-core/sparse-wing, clustered, lattice-like) | Provides exact ground truth to isolate density-equivariance from spectral content |
| Dataset | Non-uniformly resampled spectroscopic survey spectra (held-out density profile) | Tests transfer to real acquisition patterns |
| Primary metric | Downstream reconstruction error (MSE) on a held-out target grid under resampling shift, reported separately for sparse-wing and dense-core subsets | Detects density-equivariance, not raw fit |
| Secondary metric | Latent-code drift: L2 distance between pooled latent codes $z$ for the same underlying spectrum under two different density profiles | Directly measures the claimed invariance |
| Baseline | OmniSpectra (fixed rotary, mean pooling) | Tests whether density modulation is needed |
| Baseline | OmniSpectra + density concatenation | Tests whether feature concat suffices |
| Baseline | OmniSpectra + learned attention pooling | Tests whether closed-form weighting matters |
| Ablation-of-ours | DE-CRO with bandwidth gate frozen to constant (g_i = 0) | Isolates rotary bandwidth contribution |
| Ablation-of-ours | DE-CRO with uniform pooling (w_i = 1/N) | Isolates inverse-density pooling contribution |
| Diagnostic | Normalized nearest-neighbor distance variance per token; fraction of tokens with variance > 2 | Reports when assumption A (local Poisson sampling) is violated |

### Why this setup validates the claim

The central claim is that density-modulated rotary bandwidth plus inverse-density pooling yields a latent code invariant to non-uniform resampling, which fixed OmniSpectra cannot express. The synthetic dataset is the crux: it contains the same underlying continuous spectrum observed under different density profiles, so that resampling shift is the controlled variable rather than spectral content, and the exact ground truth allows us to verify phase cancellation numerically rather than only via downstream error. The primary metric — reconstruction error on a held-out target grid after resampling, split by sparse-wing and dense-core subsets — is the right detector because density equivariance predicts that error stays flat as density profile changes, whereas non-equivariant methods degrade. The secondary metric — latent-code drift between density profiles — directly measures the claimed invariance without confounding from decoder capacity. The three baselines each probe a distinct sub-claim: vanilla OmniSpectra tests whether any density handling is needed; density concatenation tests whether the phase-shift cancellation requires structural modulation rather than extra features; learned attention pooling tests whether the closed-form inverse-density unbiasedness matters versus rediscovering weights from data. The frozen-gate ablation is the sharpest test: if the bandwidth gate is doing the equivariance work, freezing it to a constant should collapse performance back toward vanilla OmniSpectra, isolating the mechanism from the pooling change. The uniform-pooling ablation isolates the pooling contribution. The diagnostic reports when the k-NN density estimator's Poisson assumption is violated, flagging where the method's guarantees are not expected to hold.

### Expected outcome and causal chain

**vs. OmniSpectra (fixed rotary, mean pooling)** — On a case where the same spectrum is resampled with a strongly non-uniform density profile (dense core, sparse wings), the baseline produces a shifted latent code and elevated reconstruction error in the sparse region because its fixed monotone rotary phase and uniform mean pooling treat every sample as equally informative, so the dense core dominates. Our method instead rescales rotary phase via the density gate and down-weights dense-region tokens via inverse-density pooling, so the pooled code tracks the underlying continuous spectrum. We expect a noticeable error gap concentrated on the sparse-wing subset, with near-parity on uniformly sampled subsets.

**vs. OmniSpectra + density concatenation** — On a case where density varies enough to induce a phase shift in the rotary embedding, the baseline produces a latent code that still carries density-induced phase distortion because concatenating density as a feature leaves the relative-position encoding untouched, so the phase cannot be cancelled. Our method instead modulates the rotary bandwidth itself, so the phase shift is structurally absorbed. We expect the concatenation baseline to partially improve over vanilla but retain a residual gap on the high-density-variation subset, while our method closes it.

**vs. OmniSpectra + learned attention pooling** — On a case where the survey's sampling pattern is idiosyncratic and training data for that pattern is scarce, the baseline produces pooling weights that overfit the survey-specific density profile because attention pooling must rediscover the importance weighting from data. Our method instead uses closed-form inverse-density weights, which are unbiased for Monte-Carlo integration regardless of pattern. We expect the learned-pooling baseline to match or slightly beat ours on the in-distribution survey but to degrade on the held-out density profile, while our method stays stable — a gap that appears specifically under distribution shift.

**vs. DE-CRO with bandwidth gate frozen to constant** — On the synthetic dataset with controlled density profiles, the frozen-gate variant cannot cancel the density-induced phase shift because its rotary phase is fixed, so its latent-code drift and sparse-wing reconstruction error should match vanilla OmniSpectra. Our full DE-CRO should show substantially lower drift and error on the same cases, isolating the bandwidth gate as the cause.

**vs. DE-CRO with uniform pooling** — On the same synthetic dataset, the uniform-pooling variant retains the density-aware rotary phase but weights all tokens equally, so dense-region tokens dominate the pool. We expect it to improve over vanilla but retain a residual gap on the sparse-wing subset relative to full DE-CRO, isolating the inverse-density pooling contribution.

### What would falsify this idea

If the gain over vanilla OmniSpectra is uniform across all density-variation subsets rather than concentrated on the high-density-variation and sparse-region subsets where the predicted phase-distortion failure occurs, the density-equivariance mechanism is not the cause of any improvement. Equivalently, if the frozen-gate ablation matches full DE-CRO, the bandwidth gate is not doing the claimed work.

## References

1. OmniSpectra: A Unified Foundation Model for Universal Spectra Representation Learning
