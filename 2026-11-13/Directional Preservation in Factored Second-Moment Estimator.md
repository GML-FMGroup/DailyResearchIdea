# Directional Preservation in Factored Second-Moment Estimators: A Whitening-Spectrum Analysis and Rank-One Correction for FOAM

## Motivation

Factored second-moment optimizers such as FOAM (Factored One-sided Adam-Moment for Practical and Scalable SOAP) replace SOAP's dense rotated second moment with a row-by-column estimator, justified under a K-FAC Fisher assumption with diagonal output curvature. The structural gap is that FOAM's input-only preconditioning has never been shown to preserve the specific curvature directions that matrix-whitening optimizers (SOAP, Shampoo) rely on for convergence; FOAM's own empirical report of 15-20% K-FAC violation is measured in full-matrix space and says nothing about which directions the factored estimator distorts. Because the estimator is built from row and column marginals, it is structurally blind to the interaction (off-marginal) component of the curvature, so the field cannot tell whether FOAM's efficiency gains come at the cost of losing convergence-critical directions.

## Key Insight

The row-by-column moment estimator is exactly the diagonal-plus-rank-one projection of the whitened curvature onto its row and column marginal spectra, so every direction it loses lies in the orthogonal complement of the span of the row and column Gram eigenvectors, and that complement has a fixed, computable dimension that shrinks relative to the whitened spectrum as the input-side whitening approaches the true curvature — meaning the lost directions are precisely the ones whose whitened curvature is already near-isotropic.

## Method

## FARA: Factored Anisotropy-Residual Augmentation

**(A) What it is.** FARA is a drop-in modification of FOAM's factored second-moment estimator. Inputs: per-layer activations' row and column moment statistics collected by FOAM, the current input-side whitening matrix $W$ (from FOAM's input-only preconditioner), and a damping $\lambda$. Outputs: a corrected factored second-moment estimate $\hat{S}_{\text{FARA}}$ used exactly where FOAM uses its estimator, plus a scalar per-layer *direction-preservation diagnostic* $\rho_\ell$ that reports the fraction of whitened curvature energy lying in the preserved subspace.

**(B) How it works.**

```pseudocode
# Per layer, per optimizer step.  All quantities are in the whitened basis
# U = W (input-side whitening eigenvectors), so S_hat is expressed as a
# matrix in the whitened coordinate system.

Input: row statistics r in R^m, column statistics c in R^n,
       whitening basis U (m x m), damping lambda, residual rank k=1

1.  Build the factored marginal estimate exactly as FOAM:
        S_hat_foam = diag(r) (x) I_n  +  I_m (x) diag(c)      # Kronecker-sum form
    Express it in the whitened basis:
        S_tilde = U^T S_hat_foam U

2.  Compute the preserved-subspace projector:
        P_par = P_row + P_col - P_row P_col
    where P_row = U^T diag(r) U / tr(diag(r)) and
          P_col = U^T diag(c) U / tr(diag(c)) are the (normalized)
          marginal spectral projectors.  P_par spans the row/column
          marginal eigenspaces that FOAM's estimator can represent.

3.  Estimate the lost (orthogonal-complement) curvature energy from
    an exponential moving average (EMA, decay beta=0.9) of the
    empirical Fisher residual over minibatches:
        R_b = (1/B) sum_b g_b g_b^T   -   S_hat_foam        # whitened residual, batch b
        R_ema = beta * R_ema + (1 - beta) * R_b
        E_lost = || (I - P_par) U^T R_ema U (I - P_par) ||_F

4.  Rank-k correction targeted at the most energetic lost directions,
    with k set adaptively from the complement spectrum:
        C = (I - P_par) U^T R_ema U (I - P_par)
        eigvals, eigvecs = eigh(C)
        k = number of eigenvalues >= 0.1 * max(eigvals)   # energy-capture threshold
        V = top-k eigenvectors of C
        S_hat_FARA = S_hat_foam + (E_lost / (E_lost + lambda)) * V V^T

5.  Diagnostic (no effect on the update):
        rho_l = tr(P_par S_tilde) / tr(S_tilde)

Return S_hat_FARA, rho_l
```

Hyperparameters: damping $\lambda$ (default 1e-8, matching FOAM's epsilon); residual rank $k$ set adaptively from the complement spectrum via an energy-capture threshold of 0.1 (number of eigenvalues $\ge 0.1 \times$ the top eigenvalue); EMA decay $\beta=0.9$ for the residual; minibatch size $B$ for the residual (same as FOAM's statistic batch).

**(C) Why this design.** We chose to correct only the orthogonal complement of the marginal projectors rather than re-estimating the full whitened second moment, because re-estimating it is exactly SOAP's dense operation and would forfeit FOAM's memory and eigendecomposition savings — accepting the cost that FARA cannot recover energy in directions that are only *partially* outside the marginal span. We chose an adaptive rank-$k$ correction over a fixed rank-one correction because the residual spectrum in the complement is not guaranteed to be rank-one concentrated: under near-isotropy the top eigenvector captures only ~1/d of the complement energy, so a fixed $k=1$ would under-cover the lost directions, whereas setting $k$ from the complement spectrum itself (eigenvalues $\ge 0.1 \times$ the top eigenvalue) captures the dominant lost energy without a tuned hyperparameter — accepting the cost that in strongly anisotropic regimes the threshold may select many directions and increase the correction's rank. We chose to estimate the complement residual from an EMA of minibatch residuals rather than a single batch because a single-batch residual's top complement eigenvector is dominated by minibatch gradient noise (the standard stochastic Hessian/Fisher estimation variance problem), so the EMA makes the top complement eigenvector reflect curvature rather than noise — accepting the cost that the EMA introduces a lag in tracking rapidly changing curvature. We chose to gate the correction magnitude by $E_\text{lost}/(E_\text{lost}+\lambda)$ rather than applying the full residual, because the empirical Fisher residual contains minibatch noise that should not be injected into the preconditioner when it is small relative to damping — accepting the cost that a genuinely large but noisy residual is partially attenuated. A domain expert would not describe FARA as 'FOAM plus a residual term' in the sense of a generic add-on, because the correction is *constructed* to live in the exact orthogonal complement of FOAM's marginal projectors — the one subspace that FOAM provably cannot represent — which is a structural placement decision that no generic residual or momentum term replicates.

**(D) Why it measures what we claim.** The diagnostic $\rho_\ell$ measures *directional preservation* of the factored estimator because $P_\text{par}$ is, by construction, the orthogonal projector onto the span of the row and column marginal eigenspaces, and under the K-FAC-with-diagonal-output assumption FOAM's estimator is exactly the restriction of the whitened curvature to that span; this assumption fails when the output-side curvature is non-diagonal, in which case $\rho_\ell$ reflects the marginal overlap rather than true preservation and should be read as an upper bound. The residual energy $E_\text{lost}$ measures the *convergence-critical lost curvature* because the whitened residual $U^T R_\text{ema} U$ restricted to $(I-P_\text{par})$ is the component of the empirical Fisher that FOAM's update cannot see, and the matrix-whitening optimizers' convergence argument depends on the preconditioner matching the curvature on precisely the directions the update traverses; this assumption fails when the minibatch residual is dominated by gradient noise rather than curvature, in which case $E_\text{lost}$ reflects noise energy and the damping gate suppresses the correction. The adaptive rank-$k$ choice measures *concentration of lost energy* because $k$ is set from the complement spectrum itself via an energy-capture threshold of 0.1, so the correction covers the dominant lost directions whenever the complement spectrum has a few eigenvalues above the threshold; this assumption fails under strong isotropy, in which case $k$ grows to cover many directions and the correction approaches a full complement estimate, or under strong anisotropy, in which case $k$ shrinks and the correction under-covers the rest. The EMA of minibatch residuals measures *curvature rather than noise* because averaging over batches reduces the variance of the empirical Fisher estimate; this assumption fails when the curvature changes faster than the EMA decay $\beta=0.9$, in which case the EMA lags the true curvature and the correction reflects stale directions.

## Contribution

["A structural decomposition of FOAM's row-by-column moment estimator as the diagonal-plus-rank-one projection of whitened curvature onto its row and column marginal spectra, yielding a computable per-layer direction-preservation diagnostic $\\rho_\\ell$ that identifies exactly which whitened curvature directions the factored estimator cannot represent.", "FARA, a rank-one anisotropy-residual augmentation of FOAM's estimator that places its correction in the orthogonal complement of the marginal projectors, retaining FOAM's memory and eigendecomposition savings while recovering the dominant lost curvature direction.", "An empirical protocol that compares $\\rho_\\ell$ and the corrected update against full-matrix SOAP on transformer training, isolating whether FOAM's efficiency gains coincide with systematic loss of specific whitened directions."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Transformer LM pretraining (mid-scale) | Where SOAP/FOAM curvature gaps actually bite |
| Primary metric | Tokens-to-target-loss | Directly reflects preconditioner quality, not throughput |
| Baseline | FOAM | Tests whether residual correction beats factored marginal alone |
| Baseline | SOAP | Tests whether we recover dense-quality preconditioning cheaply |
| Baseline | AdamW | Tests whether second-order structure matters at all |
| Ablation-of-ours | FARA with rank-k residual disabled (gate→0) | Isolates the complement-correction from the diagnostic |
| Ablation-of-ours | FARA with fixed rank-1 residual (k=1) | Tests whether adaptive rank-k is necessary |
| Ablation-of-ours | FARA with single-batch residual (no EMA) | Tests whether EMA reduces noise in the complement |
| Diagnostic validation | Per-layer $\rho_\ell$ vs. FOAM-vs-SOAP gap correlation | Tests whether $\rho_\ell$ predicts where factored estimators fail |
| Synthetic control | Known whitened curvature with controlled anisotropy | Measures complement spectrum decay and rank-one concentration ratio |

### Why this setup validates the claim

The central claim is that FOAM's factored marginal estimator provably cannot represent curvature in the orthogonal complement of its marginal projectors, and that an adaptive rank-$k$ correction placed exactly in that complement recovers convergence-critical energy that FOAM misses. This is a *structural* claim, so the design must isolate exactly that subspace. FOAM is the indispensable baseline: it shares FARA's whitening, memory footprint, and eigendecomposition path, so any gap between FARA and FOAM is attributable to the complement correction and nothing else. SOAP is the ceiling test: if FARA approaches SOAP's tokens-to-target-loss while retaining FOAM's cost profile, the claim that the complement is where the lost energy lives is supported; if FARA never closes toward SOAP, the complement correction is not doing the work we claim. AdamW confirms the second-order structure is load-bearing at all. The gate-zero ablation (FOAM-equivalent estimator) is the direct falsifier of the residual term: it must sit on top of FOAM and below FARA, otherwise the correction is inert. The fixed rank-1 ablation tests whether the adaptive rank-$k$ selection is necessary, and the single-batch ablation tests whether the EMA reduces noise in the complement. The diagnostic validation correlates per-layer $\rho_\ell$ with the FOAM-vs-SOAP gap to test whether $\rho_\ell$ has standalone value as a predictor of where factored estimators fail. The synthetic control with known whitened curvature measures the actual spectrum of the complement (eigenvalue decay, rank-one concentration ratio) as a function of anisotropy, grounding the near-isotropy and rank-one assumptions empirically rather than asserting them. Tokens-to-target-loss is the right metric because a preconditioner that misses curvature directions shows up as *slower progress per token*, not as a different final loss floor; wall-clock or memory would confound the mechanism with implementation.

### Expected outcome and causal chain

**vs. FOAM** — On a case where the whitened Fisher has a dominant eigenvector lying mostly outside the span of the row/column marginal projectors — common in transformer MLP and attention output projections where output-side curvature is non-diagonal — FOAM produces a preconditioner that is blind to that direction, so its update under-steps along it and tokens-to-target-loss lags. Our method instead estimates the lost energy from the EMA of whitened residuals and injects an adaptive rank-$k$ correction along the top complement eigenvectors, because that is precisely the subspace FOAM's projectors annihilate. We expect FARA to reach the target in noticeably fewer tokens than FOAM, with the gap concentrated in layers where the diagnostic $\rho_\ell$ is low, and near-parity where $\rho_\ell$ is high.

**vs. SOAP** — On a case where the true curvature is dense and strongly non-diagonal, SOAP's full dense second moment captures every direction, so it produces the fastest per-token progress but at dense memory and eigendecomposition cost. Our method instead recovers the dominant lost directions in the complement via adaptive rank-$k$, because the complement spectrum has a few eigenvalues above the energy-capture threshold in the regime FOAM targets. We expect FARA to sit close to SOAP on tokens-to-target-loss — closing most but not all of the FOAM–SOAP gap — while matching FOAM's memory and step cost; a large residual gap to SOAP would mean the complement is not captured by the adaptive rank-$k$ threshold.

**vs. AdamW** — On a case where curvature is ill-conditioned across parameter directions, AdamW produces per-coordinate scaling that ignores cross-direction coupling, so it makes slow progress along the coupled low-curvature subspace and needs many more tokens. Our method instead applies a whitened, curvature-aware preconditioner with the complement correction, because the marginal plus residual estimate approximates the coupled curvature that diagonal scaling cannot represent. We expect a clear, roughly monotone tokens-to-target-loss advantage over AdamW that widens on the harder-conditioned layers, confirming the second-order structure — not just the residual term — is doing work.

### What would falsify this idea

If FARA's gain over FOAM is uniform across layers rather than concentrated in layers with low $\rho_\ell$ (and absent where $\rho_\ell$ is high), the correction is not acting through the predicted complement mechanism and the central claim is wrong. Likewise, if the gate-zero ablation matches FARA, the residual term is inert and the structural placement argument fails; if the fixed rank-1 ablation matches adaptive rank-$k$, the rank-one concentration assumption holds and the adaptive selection is unnecessary; if the single-batch ablation matches the EMA version, the EMA does not reduce noise and the curvature-dominance assumption is unsupported.

## References

1. Factored One-sided Adam-Moment for Practical and Scalable SOAP
