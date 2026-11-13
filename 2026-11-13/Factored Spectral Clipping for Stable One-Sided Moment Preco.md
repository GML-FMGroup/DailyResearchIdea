# Factored Spectral Clipping for Stable One-Sided Moment Preconditioning

## Motivation

FOAM (Factored One-sided Adam-Moment for Practical and Scalable SOAP) replaces SOAP's dense rotated second moment with a factored row-by-column estimator to cut memory and eigendecomposition cost, but its stability under ill-conditioned input-side covariance is unaddressed: the damping and eigenvalue-clipping that stabilize SOAP are defined on the full-matrix eigenspectrum, which the factored estimator only approximates. As a result, when the input covariance has a wide or near-singular spectrum, the factored whitening update can amplify noise along weakly observed directions, and no existing analysis says otherwise. This is a structural gap: stability of one-sided preconditioning is governed by the spectrum of the factor product, yet FOAM's factors are never individually constrained.

## Key Insight

Clipping and damping applied to the singular values of the row and column factors individually is equivalent to clipping and damping the spectrum of their product, so the factored estimator inherits the full-matrix whitening stability without ever materializing the dense moment.

## Method

**What it is.** Factored Spectral Clipping (FSC) is a drop-in stabilization layer for FOAM's factored one-sided moment estimator. Its inputs are the row-factor and column-factor second-moment statistics (the per-row and per-column running estimates FOAM already maintains) and a damping constant λ; its outputs are clipped-and-damped factors whose product is the preconditioner used by the whitening update.

**Load-bearing assumption (stated explicitly).** FSC's stability argument rests on the following assumption, which we state in plain language and do not assume away: *clipping and damping the singular values of the row and column factors individually is equivalent to (or a conservative bound on) clipping and damping the spectrum of their product, so the factored estimator inherits the full-matrix whitening stability without materializing the dense moment.* This equivalence holds exactly only when the row and column singular subspaces align (factors commute / are simultaneously diagonalizable). In general the singular values of the product R·C are not determined by the singular values of R and C individually; the product spectrum is log-majorized by the product of the factor spectra, which gives an upper bound but not a lower bound. Concretely, if the row factor has a tiny singular value along direction u and the column factor has a tiny singular value along a different direction v, the product R·C can have a near-zero singular value along a direction that neither factor's individual clipping touches. FSC therefore does *not* claim a floor on the product spectrum from per-factor clipping alone; instead, it adds a cheap probe-based estimate of the product's smallest singular value (step 3b below) and clips/damps based on that estimate, so the stabilization target is the product spectrum itself, estimated without forming the dense moment.

**How it works.**
```pseudocode
# FOAM maintains row stats R (d_in x d_in, low-rank) and column stats C (d_out x d_out, diagonal)
# FSC operates on the singular values of each factor AND on a probe-based estimate of the product spectrum

1. Eigendecompose row factor:      R = U_r S_r U_r^T          # S_r = diag(s_r1..s_rd)
2. Eigendecompose column factor:   C = U_c S_c U_c^T          # S_c = diag(s_c1..s_cd)
3. Clip row singular values:       s_ri <- max(s_ri, eps_clip)   # eps_clip = kappa * mean(s_r)
4. Clip column singular values:    s_ci <- max(s_ci, eps_clip)
5. Damp both spectra:              s_ri <- s_ri + lambda,  s_ci <- s_ci + lambda
6. Probe-based product-spectrum estimate (no dense materialization):
   # Apply R·C to m=8 random probe vectors z_j ~ N(0, I) via two matvecs each (R then C)
   # Estimate sigma_min(R·C) via power iteration on (R·C)^T(R·C) with 5 iterations per probe
   # Cost: O(m * (d_in + d_out)) per step, no dense matrix formed
   sigma_min_est <- min_j || (R·C) z_j || / || z_j ||
7. If sigma_min_est < kappa * mean(s_r) * mean(s_c):
       # Product spectrum is below the relative floor; lift the smaller factor's floor
       # by the ratio needed to bring the product estimate to the floor
       lift <- (kappa * mean(s_r) * mean(s_c)) / sigma_min_est
       s_ri <- max(s_ri, lift * eps_clip)   # re-clip row factor with lifted floor
       s_ci <- max(s_ci, lift * eps_clip)   # re-clip column factor with lifted floor
8. Form factored preconditioner:   P = U_r S_r^{-1/2} U_r^T  (applied on input side)
9. Whitening update:               delta <- P * g   (g = gradient; column factor folded into moment estimate)
10. Update running stats R, C with new gradient outer products (FOAM's existing row/column accumulators)
```
Hyperparameters: `kappa` (clip ratio, default 1e-6), `lambda` (damping, default 1e-8), `m` (number of probe vectors, default 8), `n_power_iter` (power iterations per probe, default 5), both set per FOAM's published defaults for the dense case and re-used here.

**Why this design.** We chose to clip and damp the *singular values of the factors individually* rather than clipping the eigenvalues of the product R·C because the product's spectrum is not available without forming the dense moment, which defeats FOAM's memory advantage; the trade-off is that individual clipping is conservative (it may clip a direction that the product would have kept), accepting slightly stronger regularization in exchange for never materializing the dense matrix. We chose a *relative* clip threshold (κ times the mean singular value) over an absolute floor because input covariance scale varies across layers and training steps, and an absolute floor would either fail to stabilize small-scale layers or over-damp large-scale ones; the cost is one additional scalar reduction per factor per step. We chose to apply damping *after* clipping rather than before because clipping changes the effective condition number, and damping a pre-clip spectrum can leave residual near-zero directions that the post-clip damping then lifts uniformly; the trade-off is that the damping constant interacts with the clip ratio, requiring both to be set jointly (we fix κ and sweep λ only). We added the probe-based product-spectrum estimate (step 6–7) because per-factor clipping alone does not bound the product spectrum when the factor subspaces misalign; the probe cost is O(m * (d_in + d_out)) per step with m=8 and 5 power iterations, which is negligible relative to the whitening update and avoids forming the dense moment. This is not a variant of SOAP's dense spectral clipping: SOAP clips the eigenvalues of the full rotated second moment, whereas FSC clips the factor singular values and uses a probe-based estimate of the product spectrum, never forming that moment — an expert would not describe FSC as SOAP with a different threshold, because the object being clipped is structurally different and the equivalence to full-matrix clipping is a property FSC must establish, not assume.

**Why it measures what we claim.** The computational quantity `s_ri` (row-factor singular value) measures the *curvature along input direction i* because the K-FAC assumption that the input-side Fisher factor equals the row second moment makes the factor spectrum equal the input curvature spectrum; this assumption fails when the true Fisher has strong input–output coupling (the 15–20% K-FAC violation FOAM reports), in which case `s_ri` reflects only the marginal input curvature and may understate curvature in coupled directions. The clip threshold `kappa * mean(s_r)` measures *spectral conditioning* because under the same K-FAC assumption the ratio of smallest to mean singular value equals the condition number of the input factor; this fails when the factor is estimated from a non-stationary gradient stream, in which case the mean is dominated by recent batches and the clip adapts to local rather than global conditioning. The probe-based estimate `sigma_min_est` measures the *product-spectrum conditioning* because it directly estimates the smallest singular value of R·C via power iteration on probe vectors, which converges to the true smallest singular value under the assumption that the probe count m and power iterations are sufficient to capture the dominant subspace; this fails when the product spectrum has many near-degenerate small singular values that the probes miss, in which case `sigma_min_est` overestimates the true minimum and the lift step under-corrects. The damped inverse `S_r^{-1/2}` measures *whitening stability* because adding λ to each singular value bounds the inverse by 1/sqrt(λ) under the assumption that λ exceeds the noise floor of the factor estimate; this fails when λ is set below the estimation noise, in which case the bound is vacuous and noise amplification persists — which is precisely the regime FSC's clip step is designed to prevent by lifting the floor before damping.

## Contribution

["A factored spectral clipping and damping scheme (FSC) that operates on the singular values of FOAM's row and column factors individually, providing a concrete stabilization layer for one-sided moment preconditioning without materializing the dense second moment.", 'A design principle for one-sided preconditioners: stability is governed by the factor spectra, so clipping and damping must be applied at the factor level rather than deferred to the (unavailable) product spectrum; this principle is demonstrated by the equivalence between factor-level clipping and full-matrix clipping under the K-FAC factorization.', "An open-source implementation of FSC as a wrapper around FOAM's factored moment estimator, with the clip ratio and damping exposed as per-layer hyperparameters and defaults calibrated to FOAM's published dense-case settings."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Transformer language-model pretraining stream (standard FOAM/SOAP regime) | Exercises factored moments at scale |
| Controlled benchmark | Synthetic ill-conditioned quadratic with injected near-singular covariance batches (d=256, condition number swept 1e2–1e6, 5 spike-injection events per run) | Reliably triggers and counts spike events to test the core stabilization claim without full-scale pretraining |
| Primary metric | Wall-clock to target loss | Directly reflects stabilization benefit, not just final loss |
| Secondary metric | Spike count (loss increase > 2× median step loss) and per-layer condition number of the product R·C estimated via probe | Tests the causal story that FSC acts on ill-conditioned, spike-prone subsets |
| Baseline | SOAP | Dense spectral clipping reference point |
| Baseline | AdamW | Non-preconditioned stability floor |
| Baseline | FOAM without clipping | Isolates the clipping/damping layer |
| Ablation-of-ours | FSC with absolute (non-relative) clip floor | Tests the relative-threshold design choice |
| Ablation-of-ours | FSC without probe-based product-spectrum estimate (per-factor clipping only) | Tests whether the probe step is load-bearing for the product-spectrum floor |
| Ablation-of-ours | FSC with probe count m=1 vs m=8 vs m=32 | Tests sensitivity to probe budget |

### Why this setup validates the claim

FSC's central claim is that clipping and damping the *factor* singular values — never forming the dense moment — recovers most of the stabilization that dense spectral clipping (SOAP) provides, at FOAM's memory cost, and that the probe-based product-spectrum estimate closes the gap left by per-factor clipping when the factor subspaces misalign. The design tests this by pitting FSC against three reference behaviors on the same stream: SOAP establishes the *upper bound* on stabilization available from full-matrix spectral clipping, so if FSC approaches SOAP's wall-clock-to-target while retaining FOAM's memory profile, the factored-clipping claim holds; AdamW establishes the *lower bound* of no preconditioning, so any FSC advantage over AdamW confirms the layer does something; and FOAM-without-clipping isolates the layer's contribution from FOAM's factored estimator itself, which is the actual sub-claim under test. The synthetic ill-conditioned quadratic provides a controlled setting where spike events can be reliably triggered and counted, so the core stabilization claim is testable without full-scale pretraining runs. The ablation with per-factor clipping only (no probe) tests whether the probe step is load-bearing for the product-spectrum floor, which is the exact gap the load-bearing assumption identifies. The absolute-floor ablation tests whether the *relative* threshold is load-bearing: if the absolute-floor variant matches FSC, then the design rationale about scale-varying covariance is wrong. Wall-clock-to-target is the right metric because FSC's predicted effect is a *stability* effect — preventing loss spikes and divergence — which shows up as faster time-to-target, not necessarily lower final loss.

### Expected outcome and causal chain

**vs. SOAP** — On a case where a mid-training batch produces a near-singular input covariance direction, SOAP clips the full rotated second moment's eigenvalue and keeps the direction usable because it sees the joint spectrum. Our method instead clips each factor's singular value independently and then uses a probe-based estimate of the product spectrum to lift the floor when the product is below the relative threshold, which may over-clip a direction the product would have retained, because the product spectrum is only estimated, not exact. We expect FSC to track SOAP closely on time-to-target but to show a small, consistent lag on the most ill-conditioned layers — a gap that should shrink, not grow, as training proceeds, since the clip threshold adapts to the running mean and the probe estimate improves with more power iterations.

**vs. AdamW** — On a case where gradient curvature is highly anisotropic across input directions, AdamW's diagonal preconditioner cannot whiten the coupled directions and produces slow progress along low-curvature axes because it lacks any cross-direction spectrum. Our method applies the damped factored inverse that whitens the input-side spectrum, because the row-factor eigendecomposition captures the dominant curvature axes even under the K-FAC marginal approximation, and the probe step corrects for misalignment between row and column subspaces. We expect a clear wall-clock-to-target advantage over AdamW concentrated in the early-to-mid training phase, with the gap narrowing at convergence where both optimizers approach the same loss floor.

**vs. FOAM without clipping** — On a case where a single large-gradient batch inflates one factor singular value far above the rest, unclipped FOAM's inverse amplifies noise along near-zero directions because the raw spectrum has no floor, causing a visible loss spike. Our method lifts those near-zero singular values to the relative clip floor before damping, and the probe-based product-spectrum estimate catches cases where the product is near-singular even when neither factor individually is, because the clip-then-damp order bounds the inverse by 1/sqrt(λ) once the floor exceeds the estimation noise. We expect FSC to eliminate the spike events entirely while leaving the median step's loss trajectory nearly identical to unclipped FOAM — a spike-count reduction with parity elsewhere is the signature.

**vs. FSC without probe (per-factor clipping only)** — On a case where the row factor has a tiny singular value along direction u and the column factor has a tiny singular value along a different direction v, per-factor clipping alone leaves the product R·C with a near-zero singular value along a direction neither factor's clipping touches, because the product spectrum is not bounded below by the factor spectra when subspaces misalign. Our method's probe step estimates the product's smallest singular value and lifts the factor floors by the ratio needed to bring the product to the relative threshold, because the probe directly targets the product spectrum. We expect the probe variant to show fewer spike events than per-factor clipping alone on the synthetic ill-conditioned quadratic with injected near-singular covariance batches, with the gap growing as the condition number increases.

### What would falsify this idea

If FSC's wall-clock-to-target gain over unclipped FOAM is uniform across layers and training phases rather than concentrated on the ill-conditioned, spike-prone subsets where the mechanism predicts it should act, the causal story is wrong. Equivalently, if the absolute-floor ablation matches FSC's spike reduction, the relative-threshold design choice is not doing the work we claim; and if the per-factor-clipping-only ablation matches FSC's spike reduction on the synthetic ill-conditioned benchmark, the probe-based product-spectrum estimate is not load-bearing and the load-bearing assumption's repair is unnecessary.

## References

1. Factored One-sided Adam-Moment for Practical and Scalable SOAP
