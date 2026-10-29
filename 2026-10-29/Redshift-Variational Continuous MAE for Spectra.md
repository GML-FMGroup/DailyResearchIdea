# Redshift-Variational Continuous MAE for Spectra

## Motivation

OmniSpectra's continuous-coordinate MAE achieves shift-invariance by mapping wavelengths to a logarithmic basis so that a shared redshift cancels, but this cancellation is exact only for a pure multiplicative scaling on a fixed coordinate and leaves the latent representation undefined under unknown redshift compositions; the model cannot represent spectra whose redshift distribution differs from pretraining, because redshift is treated as a nuisance to be cancelled rather than a quantity to be inferred. The structural gap is that closure under arbitrary redshift transformations requires the representation to be equivariant, not merely invariant, and no existing spectral foundation model—including OmniSpectra—provides a generative mechanism that propagates redshift uncertainty into the latent code.

## Key Insight

Treating the redshift scale factor as a latent variable in a variational autoencoder makes the spectral representation closed under arbitrary redshift composition, because the posterior over the latent code is conditioned on the scale factor and the variational lower bound jointly infers redshift and spectral parameters, so the representation is equivariant by construction rather than invariant by cancellation.

## Method

**What it is.** We introduce RV-MAE (Redshift-Variational Masked Autoencoder), which takes a masked spectrum on continuous wavelength coordinates and outputs (i) a posterior distribution over a low-dimensional latent spectral code z, (ii) a posterior distribution over the redshift scale factor s = 1 + z_redshift, and (iii) a reconstructed spectrum. The model is a variational autoencoder whose encoder is conditioned on a sampled scale factor and whose decoder is a resolution-gated Fourier implicit neural network. The load-bearing assumption is that conditioning the encoder on s and penalizing q_φ(z|x,s) with a KL term to a fixed prior p(z) forces z to be redshift-invariant; we make this assumption explicit and, because a KL to a fixed prior constrains only the aggregate posterior and not the conditional independence of z and s, we add an explicit HSIC(z, s) penalty to the training objective and report HSIC(z, s) per redshift bin as a diagnostic. We also enforce a hard architectural scaling identity in the decoder so that equivariance is structural rather than merely regularized.

**How it works.**

```pseudocode
Input: spectrum x on continuous wavelengths λ (masked at 75% of tokens), prior p(s) = LogNormal(μ_s, σ_s) with μ_s = 0, σ_s = 0.5
Encoder E_φ(λ, x_masked, s_sample) -> (μ_z, σ_z)   # s_sample ~ q_ψ(s | x_masked); E_φ is a 6-layer transformer with d_model=256, 8 heads, GeLU, sinusoidal positional encoding of λ
Redshift posterior q_ψ(s | x_masked) = LogNormal(μ_ψ(x), σ_ψ(x))   # μ_ψ, σ_ψ are 2-layer MLPs (hidden=128, GeLU) on the mean-pooled encoder features
Sample z ~ N(μ_z, σ_z^2)   # z ∈ R^64
Decoder D_θ(λ, z, s_sample) -> x̂   # Fourier INR with anti-aliasing, frequencies gated by resolution; parameterized so that D_θ(λ, z, s) = D_θ(s·λ, z, 1) exactly (scale enters only through frequency rescaling)
Loss = E_{z,s}[ -log p_θ(x | z, s) ] + KL( q_ψ(s|x) || p(s) ) + KL( q_φ(z|x,s) || p(z) ) + λ_HSIC * HSIC(z, s)
  # KL(q_ψ(s|x) || p(s)) estimated by Monte-Carlo with 1 reparameterized sample per minibatch (variance reduction: same sample used for encoder conditioning and loss)
  # HSIC(z, s) computed on each minibatch of size 256 using a Gaussian kernel with median-heuristic bandwidth; λ_HSIC = 1.0
```

At inference, we compute the posterior mean of s and z and use z as the spectral representation; the redshift estimate is the posterior mean of s − 1. We also report HSIC(z, s) on the test set per redshift bin as a direct measure of residual redshift information in z.

**Why this design.** We chose a variational lower bound over a deterministic autoencoder because the reconstruction loss alone cannot propagate redshift uncertainty into the latent code; the KL term on q_ψ(s|x) forces the redshift posterior to be calibrated against the prior, accepting the cost that the bound is looser than a deterministic loss. We conditioned the encoder on the sampled scale factor rather than marginalizing it out because conditioning makes the latent code explicitly equivariant to redshift—shifting s shifts the code in a predictable way—accepting the cost that the encoder must learn a family of codes rather than a single invariant code. Because a KL to a fixed prior does not by itself minimize the mutual information between z and s, we add an explicit HSIC(z, s) penalty on each minibatch to directly target conditional independence; the cost is an additional hyperparameter λ_HSIC and O(batch^2) kernel computation per step. We used a resolution-gated Fourier decoder rather than a standard MLP because the Fourier basis is closed under scaling: a redshifted spectrum is exactly representable by rescaling the Fourier frequencies, so the decoder can reconstruct arbitrary redshift compositions without aliasing; we enforce the scaling identity D_θ(λ, z, s) = D_θ(s·λ, z, 1) architecturally by making s enter only through the frequency argument, and we verify it numerically on synthetic spectra (residual < 1e-5 relative error). The cost is that the decoder is slower than a convolutional decoder. We chose a LogNormal prior on s rather than a Gaussian because redshift is positive and multiplicative; the cost is that the KL divergence is not in closed form and requires a reparameterized sample, which we estimate with a single-sample Monte-Carlo estimator per minibatch.

**Why this measures what we claim.** The posterior variance of q_ψ(s|x) measures the model's uncertainty about the redshift because the variational lower bound penalizes mismatch between q_ψ(s|x) and the prior p(s) only through the KL term, which is minimized when the posterior is calibrated; this assumption fails when the prior is misspecified (e.g., the true redshift distribution is bimodal), in which case the posterior variance reflects prior mismatch rather than aleatoric uncertainty. The latent code z measures the redshift-invariant spectral shape because the encoder is conditioned on s and the training objective includes an explicit HSIC(z, s) penalty that directly minimizes dependence between z and s; this assumption fails when the HSIC penalty is too weak or the decoder is under-expressive, in which case z retains redshift information that the decoder cannot otherwise use, and we detect this via the per-bin HSIC diagnostic. The reconstruction loss measures the fidelity of the joint (z, s) representation because the Fourier decoder is closed under scaling, so a correct (z, s) pair can reconstruct the spectrum exactly; this assumption fails when the spectrum contains instrumental artifacts that are not multiplicative scaling, in which case the reconstruction error reflects those artifacts rather than representation error.

## Contribution

['A variational latent-variable extension of the continuous-coordinate MAE that treats the redshift scale factor as a latent variable and conditions the spectral code on it, yielding a representation that is equivariant under arbitrary redshift composition.', 'A resolution-gated Fourier decoder whose frequency basis is closed under scaling, enabling exact reconstruction of redshifted spectra without aliasing.', 'An empirical demonstration that the variational lower bound jointly infers redshift and spectral parameters, with the posterior variance of the scale factor providing a calibrated uncertainty estimate.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DESI spectra with held-out redshift range (training: 0.5 < z_redshift < 1.5; test: 1.5 < z_redshift < 2.0) | Tests extrapolation beyond training redshift support |
| Primary metric | Redshift-invariant retrieval accuracy of z (top-1 recall of rest-frame shape match across redshift bins, with balanced query/gallery redshift distribution) | Directly probes the equivariance claim |
| Diagnostic metric | HSIC(z, s) per redshift bin (Gaussian kernel, median heuristic bandwidth, computed on 1024 test samples per bin) | Directly measures residual redshift information in z |
| Baseline | OmniSpectra | Large foundation model with no explicit redshift factor |
| Baseline | Deterministic masked autoencoder | Isolates value of variational redshift posterior |
| Baseline | Gaussian-prior VAE on s | Tests LogNormal vs Gaussian prior choice |
| Ablation-of-ours | RV-MAE without resolution-gated Fourier decoder (standard MLP decoder) | Tests whether Fourier closure drives reconstruction |
| Ablation-of-ours | RV-MAE without HSIC penalty (λ_HSIC = 0) | Tests whether explicit conditional-independence penalty is necessary for invariance |
| Ablation-of-ours | RV-MAE with soft KL only (no hard scaling identity in decoder) | Tests whether architectural equivariance vs. regularized equivariance matters |

### Why this setup validates the claim

The central claim is that conditioning the encoder on a sampled scale factor, combined with a LogNormal posterior on s, an explicit HSIC(z, s) penalty, and a resolution-gated Fourier decoder with a hard scaling identity, yields a latent code that is genuinely redshift-invariant and a redshift estimate whose posterior variance is calibrated. Each baseline isolates one sub-claim. OmniSpectra, a large foundation model without an explicit multiplicative scaling variable, tests whether the equivariance structure—not raw capacity—is what produces invariant codes; if a much larger model without our factorization matches us, the equivariance story is unnecessary. The deterministic masked autoencoder tests whether the variational redshift posterior contributes anything beyond a point estimate, since it has the same masking and reconstruction objective but no q_ψ(s|x). The Gaussian-prior VAE isolates the LogNormal choice, since redshift is multiplicative and a Gaussian prior should mis-calibrate the posterior at high s. The ablation without the Fourier decoder removes the closure property, so if our reconstruction advantage survives without it, the closure argument is wrong. The ablation without HSIC tests whether the explicit conditional-independence penalty is necessary, since a KL to a fixed prior does not target I(z; s). The ablation with soft KL only (no hard scaling identity) tests whether architectural equivariance is stronger than regularized equivariance. The primary metric—retrieval accuracy of z across redshift bins—is the right detector because it directly measures whether z carries redshift information: a non-invariant code will retrieve poorly when the query and gallery differ in redshift, which is exactly the predicted failure mode. We balance the query and gallery redshift distributions so that retrieval failures are attributable to z's s-dependence rather than sampling; we note the failure mode that class imbalance or noise-level correlation with redshift could confound retrieval, and we report HSIC(z, s) per bin as a direct check that does not rely on retrieval.

### Expected outcome and causal chain

**vs. OmniSpectra** — On a case where a test spectrum sits at a redshift outside the training support, OmniSpectra produces a latent code that drifts with redshift because it has no explicit multiplicative scaling variable and must absorb redshift into an entangled representation. Our method instead conditions the encoder on s, penalizes residual s-dependence of z via the HSIC term, and enforces the scaling identity in the decoder, so the code is forced to be equivariant. We expect a noticeable retrieval gap on the held-out redshift range but near-parity on in-support spectra, with the gap widening as the redshift offset grows, and we expect HSIC(z, s) to be lower for our method in the held-out bins.

**vs. Deterministic masked autoencoder** — On a case where two spectra share the same rest-frame shape but differ in redshift and noise level, the deterministic autoencoder produces a code that conflates the two because its reconstruction loss alone cannot separate redshift uncertainty from shape. Our method instead samples s from a calibrated posterior and conditions the encoder on it, so shape and scale are disentangled. We expect our retrieval accuracy to exceed the deterministic baseline specifically on the high-noise, high-redshift subset, with parity on clean low-redshift spectra, and we expect HSIC(z, s) to be lower for our method on that subset.

**vs. Gaussian-prior VAE on s** — On a case where the true redshift is large and the posterior should be skewed, the Gaussian-prior model produces a posterior mean biased low and a variance that does not shrink with signal-to-noise because a symmetric prior cannot represent multiplicative scale. Our method instead uses a LogNormal prior matching the multiplicative structure, so the posterior tracks the true scale. We expect a systematic redshift bias reduction at high s and posterior variances that contract with SNR, unlike the Gaussian baseline whose variances stay flat.

**vs. RV-MAE without HSIC penalty** — On a case where the encoder can lower reconstruction error by using s-correlated features in z, the model without HSIC has no explicit incentive to remove that dependence, so z retains redshift information. Our method with HSIC directly minimizes I(z; s) on each minibatch. We expect the HSIC penalty to reduce HSIC(z, s) and improve retrieval accuracy on the held-out redshift range, with the largest gains in bins where the reconstruction loss is most sensitive to s.

**vs. RV-MAE with soft KL only (no hard scaling identity)** — On a case where the decoder must represent a redshifted spectrum, the soft-KL model relies on regularization to approximate equivariance, while our hard scaling identity enforces D_θ(λ, z, s) = D_θ(s·λ, z, 1) exactly. We expect the hard identity to yield lower reconstruction error on redshift-composed synthetic spectra (residual < 1e-5 relative error) and to produce more invariant codes, as measured by HSIC(z, s).

### What would falsify this idea

If our retrieval-accuracy gain over OmniSpectra is uniform across redshift bins rather than concentrated on the held-out high-redshift range where the equivariance failure is predicted, the central claim that conditioning on s produces redshift invariance is wrong. Likewise, if the Fourier-decoder ablation shows no reconstruction deficit on redshift-composed spectra, the closure argument does not explain our performance; and if the HSIC ablation shows no increase in HSIC(z, s) or no retrieval drop, the explicit conditional-independence penalty is unnecessary.

## References

1. OmniSpectra: A Unified Foundation Model for Universal Spectra Representation Learning
