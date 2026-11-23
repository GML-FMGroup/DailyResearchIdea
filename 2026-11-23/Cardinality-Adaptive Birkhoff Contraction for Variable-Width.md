# Cardinality-Adaptive Birkhoff Contraction for Variable-Width Residual Streams

## Motivation

Multi-stream residual formulations inherit the mixing guarantee of cmHC (E9) only under a fixed stream count: cmHC contracts the Birkhoff polytope toward its interior to escape under-mixing, but the doubly-stochastic constraint and its stream-difference bounds are defined on a single n×n mixing matrix, so any architecture whose residual width changes with depth (e.g., progressive stream growth/merging, or width-changing blocks) falls outside the guarantee and reverts to either degenerate independent streams or an uncontrolled mixing. This is a structural gap shared across the Hyper-Connections→cmHC branch rather than a local tuning issue: the contraction operator has no well-defined action when the domain and codomain polytopes differ in dimension.

## Key Insight

A stream-count change between layers is exactly a rectangular transportation map, and composing it with the square contraction preserves total residual mass and doubly-stochastic marginals, so the contraction's mixing guarantee transfers to variable-cardinality streams because the augmented operator remains a convex combination of permutation-like couplings rather than an unconstrained rectangular matrix.

## Method

**What it is.** CABIC (Cardinality-Adaptive Birkhoff Contraction) is a residual-stream mixer that accepts n_in streams at layer ℓ and emits n_out streams at layer ℓ+1, where n_in ≠ n_out is allowed. Its output is a set of n_out residual streams produced by a rectangular doubly-stochastic coupling whose interior-ness is controlled by a contraction depth τ ∈ (0,1). Inputs: the n_in residual streams X ∈ R^{n_in×d}. Outputs: n_out residual streams X' ∈ R^{n_out×d}.

**Load-bearing assumption (stated explicitly).** The method assumes that composing a square Birkhoff contraction with a lift/marginalize map and a Sinkhorn renormalization yields a rectangular coupling C whose interior-ness (distance from the degenerate vertex regime) is still governed by τ, so that cmHC's stream-difference and mixing bounds transfer to the variable-cardinality case. This assumption holds only when (i) C is full-rank and has exact row sums 1 and column sums n_in/n_out, and (ii) the lift/marginalize map is not rank-deficient. When (i) or (ii) fails, C reflects the lift's partition geometry rather than the learned mixer, and equal row sums no longer imply bounded per-stream gradient difference.

**How it works.**

```pseudocode
# per layer, given n_in, n_out, contraction depth tau, temperature-free
# 1. Build the square reference mixer M on the larger of the two counts
m = max(n_in, n_out)
P = softmax_free_permutation_basis(m)      # m x m, rows/cols sum to 1, entries in [0,1]
M = (1 - tau) * P + tau * (1/m) * ones(m,m)   # contraction toward barycenter

# 2. Lift or marginalize to the rectangular coupling
if n_out > n_in:                          # lifting: replicate then mix
    L = lift_matrix(n_in, n_out)          # n_out x n_in, each row one-hot, row sums 1
    C = M[0:n_out, 0:n_in] @ L            # n_out x n_in, column sums <= 1
else:                                     # marginalization: project down
    G = marginal_matrix(n_in, n_out)      # n_in x n_out, each col one-hot, col sums 1
    C = G.T @ M[0:n_in, 0:n_out]          # n_out x n_in

# 3. Renormalize to exact rectangular doubly-stochastic (Sinkhorn, fixed 5 iters)
C = sinkhorn_rect(C, row_target=1, col_target=n_in/n_out, iters=5)

# 4. Apply to streams
X' = C @ X                                # n_out x d, total mass preserved
return X'
```

Hyperparameters inline: contraction depth τ (default 0.1), Sinkhorn iterations fixed at 5, lift/marginalize maps built from the identity and uniform partition respectively. The lift map L is constructed as follows: for each output stream i ∈ {1..n_out}, L[i, j] = 1 if j = i mod n_in (with 1-indexed modulo), else 0; this ensures each input stream is replicated at least ⌊n_out/n_in⌋ times. The marginalize map G is constructed as: for each input stream j ∈ {1..n_in}, G[j, i] = 1 if i = j mod n_out, else 0; this ensures each output stream receives at least ⌊n_in/n_out⌋ inputs. These maps are fixed (not learned) and are full-rank when n_out and n_in are coprime; when they are not coprime, the maps have rank min(n_in, n_out) and the null space is non-trivial, which is the failure mode under which the gradient-difference bound breaks.

**Why this design.** We chose a square reference mixer on the larger count followed by an explicit lift/marginalize map over directly parameterizing a rectangular matrix because the square Birkhoff polytope is where cmHC's contraction and stream-difference bounds are defined (E9), so building on it lets the guarantee be inherited rather than re-derived; the cost is that the lift/marginalize step introduces a rank deficiency we correct with Sinkhorn. We chose fixed 5-step Sinkhorn over an exact projection because exact rectangular doubly-stochastic projection is iterative anyway and 5 steps keeps the operator differentiable and cheap, accepting a small marginal residual that we absorb in the renormalization target. We chose a single scalar τ shared across lift and marginalize branches over per-branch τ because a shared τ keeps the contraction depth comparable across depth-varying widths, accepting that it cannot specialize to growth vs. shrinkage regimes. A domain expert would not describe CABIC as "cmHC plus a resize layer" because the resize is not applied after mixing — it is folded into the coupling so that the contracted square operator and the cardinality change are the same linear map, which is what preserves the marginal structure.

**Why it measures what we claim.** The Sinkhorn-renormalized coupling C measures effective residual mixing because the assumption that a rectangular doubly-stochastic matrix with column target n_in/n_out is the unique maximum-entropy coupling between the two stream counts makes its interior-ness equivalent to mixing strength; this assumption fails when n_out and n_in are coprime and the lift map is highly non-uniform, in which case C reflects the lift's partition geometry rather than the learned mixer. The contraction depth τ measures the distance from the degenerate vertex regime because the assumption that the barycenter is the unique maximally mixed point of the polytope makes τ a monotone coordinate for under-mixing; this fails when the reference permutation basis P is itself close to the barycenter, in which case τ no longer indexes mixing. Total residual mass preservation (row sums of C) measures stream-difference control because the assumption that equal row sums imply bounded per-stream gradient difference holds only when C is doubly stochastic and full-rank, and fails under rank-deficient lifts where gradient difference is governed by the null space rather than the marginals; in that case, the row sums reflect the lift's replication pattern rather than the learned mixing.

## Contribution

(1) A cardinality-adaptive Birkhoff contraction operator that composes a square contraction with lift/marginalization maps and Sinkhorn renormalization, giving a single differentiable mixer that accepts unequal input and output stream counts while preserving residual mass and marginal structure. (2) A design principle that the stream-count change should be folded into the mixing coupling rather than applied as a separate resize, which keeps the contraction depth a shared coordinate across depth-varying widths. (3) A reference implementation and protocol for evaluating variable-width residual streams on standard vision and language backbones.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CIFAR-100 with depth-varying-width ResNetV2 (widths: 64, 128, 256, 512 at stages; stream counts n_in→n_out = 2→4, 4→8, 8→16 at stage transitions) | Tests mixing under genuine cardinality change |
| Dataset | ImageNet-1k with depth-varying-width ResNetV2 (same width schedule) | Confirms scaling to larger images |
| Primary metric | Top-1 accuracy vs. depth-matched baseline | Detects mixing quality, not raw capacity |
| Secondary metric | Per-stream gradient norm spread (std/mean) at each stage | Directly probes stream-difference control |
| Baseline | cmHC (fixed-cardinality Birkhoff contraction, n fixed to max(n_in,n_out) with zero-padding) | Isolates benefit of cardinality adaptivity |
| Baseline | Hyper-Connections (unconstrained stream mixer) | Tests whether contraction matters at all |
| Baseline | Orthogonal Residual Update (single-stream) | Anchors against strong single-stream residual variant |
| Ablation-of-ours | CABIC with τ=0 (no contraction, Sinkhorn only) | Separates contraction from resize machinery |
| Ablation-of-ours | CABIC with exact Sinkhorn (iters=20) vs. 5 iters | Tests sensitivity to marginal residual |

### Why this setup validates the claim

The central claim is that folding a cardinality change into a contracted Birkhoff coupling preserves residual mixing quality where a fixed-cardinality mixer cannot. A depth-varying-width ResNetV2 on CIFAR-100 gives a natural instance where n_in ≠ n_out at stage transitions, so the claim is testable without contriving a toy setting; the ImageNet-1k run checks that the effect persists at larger scale. cmHC is the decisive baseline: it inherits the same Birkhoff contraction that CABIC builds on, so any gain over cmHC must come from the cardinality-adaptive coupling rather than from contraction per se. Hyper-Connections tests whether contraction matters at all — if CABIC only matches it, the contraction story is unsupported. Orthogonal Residual Update anchors the result against a strong single-stream variant, guarding against the possibility that any multi-stream mixing is simply worse than a well-designed single stream. The τ=0 ablation is the mechanism probe: it keeps the Sinkhorn resize but removes contraction, so if the gain survives τ=0, the gain is attributable to the resize and not to the contraction claim. The exact-Sinkhorn ablation checks whether the fixed 5-iteration truncation leaves a marginal residual large enough to explain any gap. Top-1 accuracy is the right metric because the predicted effect is a representational-quality effect at fixed parameter count, and accuracy is the observable that discriminates mixing quality from capacity; per-stream gradient norm spread is the direct observable for the stream-difference bound that the load-bearing assumption concerns.

### Expected outcome and causal chain

**vs. cmHC** — On a case where a stage transition requires n_out > n_in, cmHC must either pad streams or fix cardinality, so its contracted coupling operates on a mismatched stream count and the extra streams carry near-duplicate directions. Our method instead lifts the contracted square operator through an explicit lift map and renormalizes with Sinkhorn, so the new streams inherit distinct, contracted directions. We expect a noticeable accuracy gap concentrated at stage transitions with the largest cardinality jump (e.g., 8→16), and near-parity on stages where n_in = n_out. We also expect CABIC to show a smaller per-stream gradient norm spread at those transitions, consistent with the stream-difference bound holding under the full-rank condition.

**vs. Hyper-Connections** — On a case where the learned stream-mixing matrix drifts toward a vertex of the Birkhoff polytope, Hyper-Connections produces nearly independent streams with little cross-stream gradient sharing, so deep stages train unevenly. Our method instead contracts toward the barycenter by τ, keeping the coupling interior and the streams mutually informative. We expect a modest but consistent accuracy advantage that grows with depth, and a smaller spread in per-stream gradient norms.

**vs. Orthogonal Residual Update** — On a case where the module output is dominated by the existing stream direction, the orthogonal update discards most of the module output, so capacity is underused at the layers where new features matter most. Our method instead preserves total residual mass across streams while contracting, so the module output redistributes across streams rather than being projected away. We expect parity on shallow stages and a gap in favor of CABIC on the deepest stages where feature novelty is needed.

### What would falsify this idea

If CABIC's accuracy gain over cmHC is uniform across stages rather than concentrated at large-cardinality-jump transitions, the claim that the folded resize is the causal mechanism is wrong. If the τ=0 ablation matches full CABIC, then contraction contributes nothing and the central claim is unsupported. If the per-stream gradient norm spread under CABIC is not smaller than under cmHC at transitions where the lift map is full-rank, the load-bearing assumption that equal marginals imply bounded gradient difference fails in practice.

## References

1. cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction
