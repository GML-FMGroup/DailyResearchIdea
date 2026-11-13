# Low-Rank Output-Side Correction for Exact Factored Adam Moments

## Motivation

FOAM (Factored One-sided Adam-Moment for Practical and Scalable SOAP) drops the output-side Kronecker factor and replaces SOAP's dense rotated second moment with a factored row-by-column estimator, which is exact only under the K-FAC assumption of diagonal output-side curvature; FOAM's own measurements report 15-20% K-FAC violation, so the factored moment structurally cannot represent cross-output correlations and therefore cannot reproduce full-matrix Adam when output curvature is non-diagonal. This is the local instance of a recurring meta-problem in second-order optimization: low-rank/Kronecker factorizations buy memory and compute by asserting a separability (here row-by-column) that the true curvature does not satisfy, so the factored preconditioner silently diverges from the dense one precisely where cross-coordinate coupling is strongest.

## Key Insight

A rank-r correction to the output-side factor absorbs exactly the cross-output covariance directions that the separable row-by-column estimator cannot represent, so the factored moment equals the dense Adam moment whenever the output-side Fisher has rank at most r above its diagonal — an equivalence that is a property of the residual's rank, not of the curvature being diagonal.

## Method

**What it is.** LOC-FOAM (Low-rank Output-Corrected FOAM) is a factored Adam-moment estimator that augments FOAM's diagonal output-side factor with a rank-r correction. Inputs: per-layer gradient mini-batch G ∈ R^{m×n} (m outputs, n inputs), running row statistic and running residual sketch. Outputs: a factored second-moment estimate M̂ that is used to precondition the update, plus the maintained low-rank output basis U_r ∈ R^{m×r} and correction eigenvalues Λ_r ∈ R^{r×r}.

**Load-bearing assumption (stated explicitly).** The output-side second moment that Adam/SOAP must reproduce is well-approximated by a diagonal-plus-rank-r matrix with small fixed r (default 8), i.e., the off-diagonal output covariance spectrum decays fast enough that the top-r subspace captures the coupling that matters for preconditioning. This assumption is verified per layer by the diagnostic in step 9; if the tail eigenvalue mass exceeds tolerance, the layer is flagged and r is increased for that layer up to r_max=32.

**How it works.**
```pseudocode
# per layer, per step; hyperparams: beta2=0.999, r (rank, default 8, r_max=32), eps=1e-8, refresh=50, tol=0.01
1. G = gradient of loss wrt layer weight (m x n)
2. a = row_energy(G)            # a_i = mean_j G_ij^2,  vector in R^m  (FOAM row stat)
   b = col_energy(G)            # b_j = mean_i G_ij^2,  vector in R^n  (FOAM col stat)
3. D_out = diag(beta2*a + (1-beta2)*a_prev_running)   # diagonal output factor
4. R = G - diag(sqrt(a)) @ (normalized row projection)  # residual after diagonal model
   # residual captures cross-output structure the diagonal cannot
5. if step % refresh == 0:
       U_r, S_r = top_r_svd(EMA_of_R^T R restricted to output covariance)
       U_r = orthonormalize(U_r);   Lambda_r = S_r
   else:  Lambda_r = beta2*Lambda_r + (1-beta2)*diag(U_r^T R R^T U_r)
6. M_hat = kron(D_out, D_in) + (U_r Lambda_r U_r^T) ⊗ I_n
   # D_in = diag(beta2*b + (1-beta2)*b_running)
7. preconditioned_update = M_hat^{-1/2} G   via Woodbury on the rank-r block
8. return preconditioned_update, U_r, Lambda_r
9. diagnostic (per layer, logged every refresh):
       tail_mass = sum_{i>r} S_r[i] / sum_i S_r[i]   # fraction of output covariance energy outside U_r
       subspace_drift = ||U_r^T U_r_prev - I||_F   # drift since last refresh
       if tail_mass > tol and r < r_max: r = min(r*2, r_max)  # adaptive rank increase
```
The Woodbury step in (7) makes the rank-r correction cost O(m n r) rather than O(m^3), so memory is O(mr + n) beyond FOAM.

**Why this design.** We chose a rank-r correction over a full output-side matrix because the full matrix restores SOAP's O(m^2) memory and eigendecomposition cost, which is exactly what FOAM was built to avoid; the trade-off is that we must choose r and accept a residual approximation error that decays as the tail eigenvalues of the output covariance. We chose a periodic SVD refresh (every 50 steps) over per-step eigendecomposition because the output covariance is slowly varying in transformer training, accepting a lag of up to 50 steps in tracking the dominant cross-output directions. We chose to add the correction on top of the diagonal D_out rather than replacing it because the diagonal captures the bulk per-output scale cheaply and the low-rank term only needs to model the off-diagonal residual, accepting that the two terms must be kept separately rather than folded into one factorization. We chose Woodbury inversion over materializing M̂ because M̂ is only dense in the m×m output block and the rank-r structure gives an exact inverse at O(m n r). A domain expert would not describe this as a variant of SOAP or K-FAC: SOAP keeps a dense output factor (we keep diagonal plus rank-r), and K-FAC asserts a Kronecker-separable Fisher (we explicitly model the non-separable residual), so LOC-FOAM occupies the memory regime between FOAM and SOAP that neither covers.

**Why this measures what we claim.** The residual R in step (4) measures cross-output curvature coupling because subtracting the diagonal row-energy model is equivalent to projecting out the diagonal Fisher component, leaving exactly the off-diagonal output covariance under the assumption that the per-output gradient variance is captured by a_i; this assumption fails when a single output's gradient is heavy-tailed, in which case R reflects that output's tail rather than genuine cross-output coupling. The rank-r truncation in step (5) measures the dominant cross-output correlation subspace because the top-r eigenvectors of the output covariance span the directions of maximal non-diagonal curvature under the assumption that the covariance spectrum decays; this assumption fails when the output covariance is nearly isotropic, in which case U_r captures an arbitrary r-subspace and the equivalence to dense Adam holds only approximately. The maintained Λ_r in step (5) measures the running second moment along that subspace because the EMA with beta2 is the same estimator Adam uses per-coordinate, under the assumption that curvature is slowly varying relative to beta2; this assumption fails under abrupt loss-landscape shifts, in which case Λ_r lags the true subspace moment. The diagnostic in step (9) measures the validity of the load-bearing assumption because tail_mass directly quantifies the unmodeled off-diagonal energy and subspace_drift quantifies the staleness of U_r; this assumption fails when tail_mass > tol or subspace_drift is large, in which case the layer is flagged and r is increased adaptively.

## Contribution

(1) LOC-FOAM, a factored Adam-moment estimator that augments FOAM's diagonal output-side factor with a rank-r residual correction and uses a Woodbury solve to keep the correction at O(mnr) cost, giving a memory regime strictly between FOAM and SOAP. (2) A design principle: the gap between factored and dense Adam moments is governed by the rank of the output-side residual above its diagonal, so a rank-r correction closes the gap whenever that residual rank is at most r, which we verify empirically by measuring residual spectrum decay on transformer layers. (3) A practical recipe for choosing r and the SVD refresh interval from the observed residual spectrum, validated against the K-FAC violation rate FOAM reports.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Transformer LM pretraining (small/medium) | Exposes slowly-varying per-layer output covariance |
| Primary metric | Validation loss at matched wall-clock | Tests convergence-per-cost, not raw steps |
| Baseline | AdamW | Tests whether cross-output coupling matters at all |
| Baseline | FOAM | Tests whether rank-r residual beats diagonal-only |
| Baseline | SOAP | Tests memory-efficient parity with dense output factor |
| Ablation-of-ours | LOC-FOAM with r=0 | Isolates contribution of the low-rank correction |
| Ablation-of-ours | LOC-FOAM with fixed r=8 vs adaptive r (tol=0.01, r_max=32) | Tests the adaptive-rank criterion |
| Diagnostic | Per-layer tail_mass and subspace_drift (step 9) | Verifies the load-bearing assumption |

### Why this setup validates the claim

The central claim is that a low-rank correction to FOAM's diagonal output factor recovers most of SOAP's benefit at FOAM's memory, and that the benefit comes specifically from modeling off-diagonal output coupling. AdamW is the floor: it has no second-moment structure at all, so it establishes that any gain is attributable to preconditioning rather than tuning. FOAM is the direct test of the residual: if LOC-FOAM and FOAM converge identically, the rank-r term is doing nothing and the diagonal already captures all usable structure. SOAP is the ceiling on quality: it materializes the dense output factor, so LOC-FOAM should approach SOAP's loss-per-wall-clock while staying near FOAM's memory, which is the specific gap the design targets. The r=0 ablation is the cleanest control because it is LOC-FOAM with the correction switched off, so any difference between it and full LOC-FOAM is causally the rank-r term, not a code or hyperparameter difference. The fixed-r vs adaptive-r ablation tests whether the load-bearing assumption (fast-decaying off-diagonal spectrum) holds per layer: if adaptive r increases on many layers, the fixed-r default is insufficient. The per-layer diagnostic (tail_mass, subspace_drift) directly measures the validity of the assumption: layers with tail_mass > 0.01 are where the rank-r correction is expected to matter most. Validation loss at matched wall-clock is the right metric because the claim is about the memory-compute-quality frontier: a method that matches SOAP's loss but takes twice the time has not occupied the claimed regime, and a method that matches FOAM's speed but not its loss has not earned the correction.

### Expected outcome and causal chain

**vs. AdamW** — On layers where outputs are strongly correlated (attention output projections, MLP down-projections with redundant features), AdamW produces noisy per-coordinate updates because its diagonal second moment cannot see that several outputs move together, so it over-corrects along shared directions and under-corrects along the few that dominate. Our method instead captures those shared directions in U_r and rescales them jointly via the Woodbury inverse, so we expect a clear validation-loss gap concentrated in the middle-to-late training phase, where correlated output structure has formed, and near-parity in the first few hundred steps before that structure emerges.

**vs. FOAM** — On layers where the output covariance has a few dominant off-diagonal directions (e.g., residual-stream-aligned projections), FOAM produces updates that are correctly scaled per output but wrong in the shared subspace, because its diagonal D_out assigns each output an independent scale and cannot rotate the preconditioner. Our method instead adds exactly the rank-r output block that spans those directions, so we expect LOC-FOAM to beat FOAM by a modest but consistent margin in final validation loss, with the gap widening as r grows from 4 to 8 and saturating, and with the per-step update norm along the top output eigenvector visibly better conditioned than FOAM's. The per-layer diagnostic should show that the gain is concentrated in layers with high tail_mass (>0.01) and low subspace_drift.

**vs. SOAP** — On the same correlated layers, SOAP produces the best-conditioned updates because it keeps the full dense output factor, but it pays O(m^2) memory and per-step eigendecomposition cost, so at matched wall-clock on memory-constrained hardware it runs fewer steps or smaller batches. Our method instead approximates the same dense factor with diagonal-plus-rank-r and inverts it in O(m n r), so we expect LOC-FOAM to sit between FOAM and SOAP in loss-per-step but to match or slightly beat SOAP in loss-per-wall-clock and to stay within a small constant factor of FOAM's memory, with the advantage largest when the output covariance spectrum decays quickly (low tail_mass).

### What would falsify this idea

If LOC-FOAM's gain over FOAM is uniform across layers and training phases rather than concentrated in layers with high off-diagonal output covariance (high tail_mass), the residual is not measuring cross-output coupling and the mechanism claim is wrong. Equivalently, if the r=0 ablation matches full LOC-FOAM at r=8, the rank-r correction is inert and the design collapses to FOAM.

## References

1. Factored One-sided Adam-Moment for Practical and Scalable SOAP
