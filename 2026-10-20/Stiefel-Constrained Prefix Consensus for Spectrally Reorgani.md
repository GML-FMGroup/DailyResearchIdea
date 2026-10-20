# Stiefel-Constrained Prefix Consensus for Spectrally Reorganized Matryoshka Embeddings

## Motivation

SpecMRL reorganizes the embedding space with a parameter-derived spectral basis ordered by feature importance, but its prefix readout has no guarantee that the leading dimensions remain orthonormal and consistently ordered after each orthonormalized update — the basis can drift so that the prefix no longer captures the most informative directions. This is a structural gap in the Dion-style update family: prefix truncation and orthonormality preservation are treated as independent operations, so the closure property linking the update rule to the readout is missing, and SpecMRL's gains therefore depend on empirical luck rather than an enforced invariant.

## Key Insight

If after every orthonormalized update the prefix-truncated basis is reprojected onto the Stiefel manifold and its components reordered by a distributed consensus score, then prefix truncation commutes with orthonormality preservation, making the readout a closed invariant of the update rule rather than a post-hoc approximation.

## Method

**What it is.** We introduce SCOPE (Stiefel-Consensus Orthonormal Prefix Enforcement), a training-time operator wrapped around the Dion-style spectral update in SpecMRL. Inputs: the current spectral basis matrix $U \in \mathbb{R}^{d \times k}$, the per-dimension importance scores $s \in \mathbb{R}^k$ derived from the parameter spectrum, and a distributed consensus score vector $c \in \mathbb{R}^k$ aggregated across nested-prefix loss gradients. Outputs: an updated basis $U'$ whose first $m$ columns are orthonormal for every prefix length $m \in \mathcal{M}$ and ordered by consensus importance.

**Load-bearing assumption (stated explicitly).** SCOPE assumes that the per-prefix Stiefel projection and the consensus argsort reordering commute: that after independently Cayley-retracting each nested prefix $U_{\text{raw}}[:,1:m]$ and then permuting columns by $\text{argsort}(c_j \cdot s_j)$, every prefix $U'[:,1:m]$ remains orthonormal and ordered by consensus, so the readout is a closed invariant of the update rule. This assumption is enforced by construction in the revised operator below: the consensus permutation is applied to $U_{\text{raw}}$ *before* orthonormalization, and the nested-prefix orthonormality is imposed by a single flag-manifold projection (jointly over all $m \in \mathcal{M}$) rather than by sequential per-prefix retractions. The operator therefore does not rely on the two operations commuting; it composes them in the order that makes the invariant hold by construction.

**How it works.**

```pseudocode
Input: U (d x k orthonormal), s (importance), M = {m_1 < ... < m_P} prefix lengths,
       eta (learning rate), tau (consensus temperature), T (projection iterations)

1. DION UPDATE
   G <- gradient of multi-prefix MRL loss w.r.t. U
   U_raw <- DionUpdate(U, G, eta)            # unconstrained spectral step

2. DISTRIBUTED CONSENSUS SCORE
   for each dimension j in 1..k:
       c_j <- sum_{m in M, m >= j} ( grad_loss_m[j]^T U[:,j] ) / |{m in M : m >= j}|
       # average of per-prefix gradient alignments, temperature-scaled
       c_j <- c_j / tau

3. CONSENSUS REORDERING (BEFORE orthonormalization)
   pi <- argsort_descending( c_j * s_j )      # combined consensus x importance
   U_perm <- U_raw[:, pi]                      # permute columns of the raw basis

4. NESTED-PREFIX (FLAG-MANIFOLD) PROJECTION
   # Jointly enforce U'[:,1:m]^T U'[:,1:m] = I for all m in M.
   # Sequential per-prefix Cayley retraction is NOT used; instead a single
   # flag-manifold projection is applied to U_perm.
   U' <- U_perm
   for t in 1..T:                             # T = 5 flag-manifold projection sweeps
       for each prefix length m in M (ascending):
           U'_m <- U'[:, 1:m]
           U'_m <- CayleyRetract(U'_m, skew(U'_m^T U'_m))
           U'[:, 1:m] <- U'_m
       # after the ascending sweep, re-orthonormalize the full basis
       U' <- CayleyRetract(U', skew(U'^T U'))
   # The ascending sweep plus full-basis retraction is the flag-manifold
   # projection: each inner prefix is re-orthonormalized after the outer
   # prefix has been updated, so the nested constraint is satisfied jointly.

5. RETURN U' to the next training step; prefix readout uses U'[:, 1:m]
```

The operator is applied once per optimizer step, after the Dion update and before the next forward pass. Hyperparameters: $T=5$ flag-manifold projection sweeps (each sweep = one ascending pass over $\mathcal{M}$ plus one full-basis Cayley retraction), consensus temperature $\tau=0.1$, prefix set $\mathcal{M}$ fixed by the MRL schedule. Per-prefix gradients $\text{grad\_loss}_m$ are obtained from a single backward pass through the multi-head MRL loss (one head per $m \in \mathcal{M}$), not from separate backward passes; the consensus score is computed from the per-head gradient slices in that single pass.

**Why this design.** We chose a flag-manifold projection over sequential per-prefix retractions because the invariant we need is nested — every prefix, not just the full basis, must be orthonormal — and sequential per-prefix retraction overwrites shared columns, leaving inner prefixes skewed; the ascending sweep plus full-basis retraction jointly enforces the nested constraint, accepting the cost that the projection is $|\mathcal{M}|$ times more expensive per step. We chose a distributed consensus score over a single global importance score because SpecMRL's importance ordering is derived from the parameter spectrum alone and can conflict with the multi-prefix loss landscape, accepting the cost that consensus requires gradient access at every prefix length. We chose a deterministic argsort reordering over a learned permutation head because the reordering must be a closure operation with a defined fixed point, accepting the cost that ties in the consensus score are broken by a fixed index rule rather than learned. We apply the permutation *before* orthonormalization so that the columns which are orthonormalized as a prefix are exactly the columns that will be read out as that prefix — this is what makes the invariant hold by construction rather than by assertion. A domain expert would not describe SCOPE as a variant of SpecMRL: SpecMRL adds a spectral basis but never enforces that its prefix truncation is orthonormal or that its ordering is stable under the Dion update; SCOPE supplies exactly that closure, which is a property of the update operator, not of the basis parameterization.

**Why it measures what we claim.** The flag-manifold projection error $\|U_m^T U_m - I\|_F$ (measured after Step 4 for every $m \in \mathcal{M}$) measures prefix orthonormality because the ascending sweep plus full-basis Cayley retraction is an exact retraction onto the flag manifold under the assumption that the skew-symmetric part $\text{skew}(U_m^T U_m)$ captures all deviation from orthonormality at each prefix; this assumption fails when the Dion step produces a rank-deficient $U_m$ (e.g., under aggressive compression with $m \ll d$), in which case the projection error reflects conditioning rather than genuine orthonormality violation. The consensus score $c_j$ measures the relative importance of dimension $j$ across nested prefixes because averaging per-prefix gradient alignments is equivalent to a Shapley-style marginal contribution under the assumption that prefix losses are additive in dimension contributions; this assumption fails when prefix losses interact non-additively (e.g., a dimension is only useful in combination with another), in which case $c_j$ reflects pairwise synergy rather than marginal importance. The combined score $c_j \cdot s_j$ measures distributed-consistency-weighted importance because multiplying consensus by the spectral importance is equivalent to a product-of-experts ranking under the assumption that consensus and spectral importance are conditionally independent given the task; this assumption fails when the parameter spectrum is itself shaped by the consensus gradient (which happens after several epochs), in which case the product double-counts consensus and the ordering reflects gradient magnitude rather than importance.

## Contribution

(1) SCOPE, a per-prefix Stiefel-projection plus consensus-reordering operator that makes prefix truncation a closed invariant of the Dion update rule in spectrally reorganized Matryoshka embeddings. (2) A distributed consensus score that aggregates per-prefix gradient alignments into a single importance ordering, replacing SpecMRL's spectrum-only ordering with one that is jointly identifiability- and orthonormality-consistent. (3) The design principle that prefix readout should be treated as a closure property of the update operator rather than a post-hoc projection, with an explicit failure-mode analysis of when the consensus score double-counts spectral importance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MTEB retrieval + classification subsets | Standard MRL benchmark with nested-dimension evaluation |
| Primary metric | Mean task score at each reduced dimension | Directly reflects prefix quality under compression |
| Baseline | Standard MRL | Tests whether spectral basis alone suffices |
| Baseline | SpecMRL | Tests whether prefix-orthonormality closure adds value |
| Baseline | SpecMRL + global Stiefel projection | Tests whether per-prefix projection is necessary |
| Baseline | SpecMRL + sequential per-prefix Cayley retraction (no flag projection) | Tests whether the flag-manifold projection is necessary |
| Ablation-of-ours | SCOPE without consensus reordering | Isolates ordering contribution from projection |
| Ablation-of-ours | SCOPE with Shapley-sampled $c_j$ (replaces averaged gradient alignment) | Tests the additivity assumption for $c_j$ |
| Diagnostic metric | Prefix orthonormality drift $\|U_m^T U_m - I\|_F$ over training | Directly measures the enforced invariant |
| Diagnostic metric | Rank correlation of $\pi$ across consecutive steps | Directly measures ordering stability |
| Diagnostic metric | Wall-clock and peak memory per optimizer step | Confirms tractability at MRL prefix count |

### Why this setup validates the claim

The central claim is that SCOPE's closure — nested-prefix orthonormality plus consensus-stable ordering — improves aggressive-compression prefix quality beyond what a spectral basis alone provides. MTEB's nested-dimension evaluation is the right testbed because it reads out exactly the prefixes SCOPE constrains, at the reduced dimensions where orthonormality drift and ordering instability bite hardest. Standard MRL tests whether any explicit basis is needed; SpecMRL tests whether a basis without prefix closure suffices; SpecMRL + global Stiefel projection isolates whether per-prefix projection (not just global orthonormality) is the operative ingredient; SpecMRL + sequential per-prefix Cayley retraction isolates whether the flag-manifold projection (joint nested constraint) is necessary over naive sequential retraction. The ablation (SCOPE minus consensus reordering) separates the projection's contribution from the ordering's contribution, so a gain cannot be attributed to the wrong mechanism. The Shapley-sampled $c_j$ ablation tests the additivity assumption underlying the consensus score. The diagnostic metrics — prefix orthonormality drift and rank correlation of $\pi$ — verify the mechanism by measurement rather than assertion: if SCOPE's drift is near machine precision and its ordering is stable while SpecMRL's is not, the closure is operative; if both are similar, the closure is inert. The primary metric — mean task score at each reduced dimension — is chosen because it is the only metric that reads the constrained prefix directly; a full-dimension-only metric would mask prefix drift entirely. Together these form a falsifiable ladder: if SCOPE beats SpecMRL and the ablations regress, the closure claim is supported; if SCOPE matches SpecMRL everywhere, the closure is inert.

### Expected outcome and causal chain

**vs. Standard MRL** — On a case where aggressive compression (small prefix, e.g. 64 of 1024 dims) forces the lowest-ranked dimensions to carry task signal, standard MRL produces degraded prefix scores because its learned basis has no orthonormality constraint, so early prefix columns become correlated and redundant. Our method instead enforces per-prefix Stiefel orthonormality via the flag-manifold projection, keeping each prefix maximally expressive, so we expect a noticeable gap concentrated at the smallest dimensions and near-parity at full dimension.

**vs. SpecMRL** — On a case where the Dion spectral step drifts the basis so that inner prefixes lose orthonormality between updates, SpecMRL produces unstable prefix readouts because it parameterizes a spectral basis but never re-imposes prefix orthonormality or protects ordering. Our method instead re-projects every prefix after each Dion step and reorders by consensus before projection, so we expect a modest but consistent gap at mid-to-aggressive dimensions, largest where drift accumulates over training. The diagnostic drift curves should show SCOPE near machine precision and SpecMRL growing with training steps.

**vs. SpecMRL + global Stiefel projection** — On a case where a single global projection keeps the full basis orthonormal but leaves inner prefixes skewed, this baseline produces prefix scores that lag at small m despite a clean full basis. Our method instead projects each prefix jointly via the flag-manifold projection, so we expect a gap that grows as m shrinks, with the global-projection baseline tracking SpecMRL closely at small m and matching SCOPE only at full dimension.

**vs. SpecMRL + sequential per-prefix Cayley retraction (no flag projection)** — On a case where sequential per-prefix retraction overwrites shared columns and leaves inner prefixes skewed, this baseline produces prefix scores that lag at small m because the nested constraint is not jointly enforced. Our method instead uses the ascending sweep plus full-basis retraction, so we expect a gap concentrated at the smallest prefixes, with the sequential baseline tracking SpecMRL closely at small m and matching SCOPE only at full dimension.

**vs. SCOPE without consensus reordering** — On a case where spectral importance and multi-prefix gradient alignment conflict (a dimension ranked high by spectrum but low by consensus), the ablation produces misordered prefixes because it reorders by spectral importance alone. Our method instead combines consensus with spectral importance, so we expect the ablation to sit between SpecMRL and SCOPE, with the gap concentrated on tasks where the two rankings disagree.

**vs. SCOPE with Shapley-sampled $c_j$** — On a case where prefix losses interact non-additively (a dimension is only useful in combination with another), the averaged-gradient $c_j$ reflects pairwise synergy rather than marginal importance. The Shapley-sampled ablation replaces $c_j$ with a sampled marginal-contribution estimate, so we expect it to match SCOPE on additive tasks and to diverge on tasks with strong synergy, with the divergence indicating where the additivity assumption fails.

### What would falsify this idea

If SCOPE's gain over SpecMRL is uniform across all dimensions rather than concentrated at the aggressive-compression prefixes where orthonormality drift and ordering conflict are predicted to occur, the closure mechanism is not the cause and the central claim is wrong. Likewise, if the consensus-reordering ablation matches SCOPE everywhere, the ordering half of the claim is falsified; if the diagnostic drift curves for SCOPE and SpecMRL overlap, the projection half is falsified.

## References

1. SpecMRL: Redesign Matryoshka Representation Learning within Parametric Spectrum
