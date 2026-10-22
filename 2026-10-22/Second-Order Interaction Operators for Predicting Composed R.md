# Second-Order Interaction Operators for Predicting Composed RAG Performance

## Motivation

Existing RAG evaluation, as exemplified by LASAGNA's 450-configuration sweep, only measures composition effects post hoc and assumes that isolated technique gains are informative about composed performance—an additive or monotone assumption that the LASAGNA study itself falsifies (predicted +47% vs. measured +17%). This structural failure recurs across the field: Self-RAG, Dense X Retrieval, and M3-Embedding all report isolated gains that do not transfer to compositions, because no method models the pairwise interactions between pipeline phases that make composition non-additive.

## Key Insight

When RAG phases interact through a low-rank operator—because each phase's effect is mediated by a small number of shared latent factors such as retrieval precision and context sufficiency—the composed performance is a second-order Taylor expansion whose interaction matrix has rank at most the number of latent factors, making it identifiable from O(k) pairwise phase interventions rather than O(k^2) full compositions.

## Method

**What it is.** We propose **SCOPE** (Second-Order Composition Predictor), which takes as input a set of k RAG techniques, their measured isolated gains, and a small budget of pairwise composition measurements, and outputs a predicted performance score for any subset composition (e.g., any of the 450 LASAGNA configurations).

**How it works.** SCOPE operates in three phases:

```python
# Phase 1: First-order term estimation (isolated gains)
# Run each of k techniques alone on a calibration set; record gain delta_i.
# Cost: k runs.

# Phase 2: Second-order interaction probing
# Select a set of pairwise probes P (|P| = O(k * r)) via a deterministic
# low-rank design: choose anchors A (|A| = r) and probe each technique
# against each anchor, where r is the assumed interaction rank (r=3 default).
# For each (i, j) in P, measure composed gain delta_{ij}.
# Compute interaction term: I_{ij} = delta_{ij} - delta_i - delta_j.
# Cost: O(k * r) runs.

# Phase 3: Low-rank factorization and prediction
# Arrange I into a k x k matrix (observed entries only).
# Solve for low-rank factors U (k x r), V (k x r) via alternating least squares
# with nuclear-norm regularization (lambda=0.1).
# For any subset S of techniques, predict:
#   score(S) = sum_{i in S} delta_i + sum_{i<j in S} (U_i . V_j)
# Return score(S) for all 2^k subsets (or the 450 LASAGNA configs).
```

**Why this design.** We chose a low-rank interaction operator over a full pairwise matrix because the number of pairwise probes scales as O(k·r) instead of O(k²), accepting the cost that the low-rank assumption may be violated for techniques with high-order interactions (e.g., three-way synergies). We chose alternating least squares with nuclear-norm regularization over a neural tensor network because the former has a closed-form update per factor and is identifiable under the phase-separability condition, accepting the cost that it may underfit non-linear interactions that a neural model could capture with more data. We chose deterministic anchor-based probing over random pairwise sampling because anchors ensure that every technique is probed against at least r diverse partners, accepting the cost that the choice of anchors introduces a design bias if the anchor set is not representative of the technique space. A domain expert would not describe SCOPE as a variant of standard RAG or Toolformer-style retrieval because those methods select techniques by nearest-neighbor similarity in an embedding space, whereas SCOPE explicitly estimates a low-rank interaction operator from counterfactual pairwise interventions—a quantity that retrieval cannot compute because it requires intervening on the composition itself.

**Why it measures what we claim.** The interaction term I_{ij} = delta_{ij} - delta_i - delta_j measures the non-additive synergy between techniques i and j because it is the discrete second-order Taylor coefficient under the assumption that composition effects are separable into first-order (isolated) and second-order (pairwise) terms; this assumption fails when three or more techniques interact synergistically (e.g., reranking amplifies compression only when query augmentation is also present), in which case I_{ij} reflects an average pairwise effect that conflates higher-order terms. The low-rank factorization of I measures the latent interaction structure because it assumes that pairwise interactions are mediated by r shared factors (e.g., retrieval precision, context sufficiency), an assumption that fails when interactions are technique-specific and not low-rank, in which case the factorization underfits and predictions regress toward the additive baseline. The anchor-based probe design measures the full interaction matrix because it assumes that the anchor set spans the row space of I, an assumption that fails when anchors are correlated with each other, in which case the estimated factors are biased toward the anchor subspace.

**Load-bearing assumption (stated explicitly).** SCOPE assumes that the k×k matrix of pairwise interaction terms I_{ij} = delta_{ij} - delta_i - delta_j is approximately low-rank with rank r=3, so that O(k·r) anchor probes suffice to recover the full interaction structure and predict arbitrary subset compositions. This assumption is load-bearing: if I is not low-rank, the ALS factorization underfits and predictions regress toward the additive baseline. We verify this assumption empirically on the LASAGNA interaction matrix by (i) computing the singular-value spectrum of the measured I and reporting the fraction of composition variance explained by the top-r factors, (ii) reporting a rank-sensitivity curve (r = 1, 2, 3, 5, 10) of rank correlation vs. r, and (iii) reporting the condition number of the anchor submatrix to check anchor incoherence. If the measured spectrum is not dominated by the top-3 singular values, the low-rank premise fails and SCOPE's predictions are not trustworthy.

## Contribution

['A second-order interaction operator framework (SCOPE) that predicts composed RAG performance from O(k·r) pairwise phase interventions rather than O(k²) full compositions, with a concrete alternating-least-squares fitting procedure and anchor-based probe design.', 'A phase-separability condition under which the interaction operator is low-rank and identifiable from anchor probes, operationalized as a rank-selection criterion (nuclear-norm regularization with cross-validated r) that determines when the predictor can be trusted.', 'An empirical validation protocol on the LASAGNA 450-configuration sweep that quantifies prediction error as a function of probe budget and interaction rank, identifying which RAG phase pairs exhibit low-rank structure and which violate it.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LASAGNA 450-config suite, 8 benchmarks, 5K–200K docs | Compositional ground truth across corpus scales. |
| Primary metric | Rank correlation + top-config regret vs. measured score | Detects whether predicted ordering matches real composition. |
| Baseline | Additive isolated-gain predictor | Tests whether interactions are truly non-additive. |
| Baseline | Full pairwise matrix (O(k²) probes) | Tests whether low-rank probing loses accuracy. |
| Baseline | Random pairwise sampling with same probe budget | Tests whether anchor design beats unstructured probing. |
| Ablation-of-ours | SCOPE with r=1 (no interaction rank) | Isolates contribution of low-rank interaction term. |
| Diagnostic | Singular-value spectrum of measured I; variance explained by top-r factors | Tests the load-bearing low-rank assumption directly. |
| Diagnostic | Rank-sensitivity curve (r = 1, 2, 3, 5, 10) | Shows whether r=3 is the right operating point. |
| Diagnostic | Condition number of anchor submatrix | Tests anchor incoherence for low-rank completion. |

### Why this setup validates the claim

The central claim is that composition effects in RAG are dominated by a low-rank pairwise interaction structure that can be recovered from O(k·r) targeted probes rather than O(k²) exhaustive ones. LASAGNA provides exactly the ground-truth composition landscape needed to test this: 450 measured configurations let us ask whether SCOPE's predicted ordering of subsets matches the measured ordering, which is the operational form of the claim. The additive baseline tests the sub-claim that interactions are non-negligible (if additive wins, SCOPE is unnecessary). The full pairwise matrix tests the sub-claim that low-rank probing is sufficient (if full-pairwise dominates, the low-rank assumption is too lossy). The random-sampling baseline tests the sub-claim that anchor-based design is the right probe allocation. The r=1 ablation tests whether the interaction term itself carries signal beyond first-order gains. The singular-value spectrum and rank-sensitivity curve directly test the load-bearing assumption that I is approximately rank-3, and the anchor condition number tests whether the deterministic anchor design satisfies the incoherence conditions required for low-rank completion. Rank correlation and top-config regret are the right metric because the downstream use of SCOPE is selecting a good composition, not calibrating absolute scores; a method that ranks well but miscalibrates is still useful, whereas one that calibrates but mis-ranks is not.

### Expected outcome and causal chain

**vs. Additive isolated-gain predictor** — On a case where reranking is combined with query augmentation and compression, the additive predictor sums three isolated gains and overestimates the composition because it treats each technique's benefit as independent of context sufficiency. Our method instead adds a learned pairwise interaction term that captures the observed synergy (reranking amplifies compression only when augmentation is present), because the interaction matrix encodes exactly these non-additive residuals. We expect a large rank-correlation gap on subsets containing ≥3 interacting techniques, with near-parity on singleton and pair subsets — i.e., the gap is concentrated where the additive assumption breaks.

**vs. Full pairwise matrix (O(k²) probes)** — On a case where k is large and the probe budget is fixed, the full pairwise matrix spends its budget thinly across all pairs, so many interaction estimates are noisy and the resulting prediction is high-variance. Our method instead concentrates probes on anchors that span the interaction row space, so each technique is measured against r diverse partners and the low-rank factorization denoises the rest. We expect SCOPE to match or slightly trail full-pairwise on rank correlation when budget is abundant, but to overtake it noticeably when budget is tight — a crossover pattern that directly tests the low-rank sufficiency claim.

**vs. Random pairwise sampling (same budget)** — On a case where a few techniques have strong, structured interactions with many partners, random sampling under-samples exactly those high-degree rows, so their interaction terms are poorly estimated and predictions for subsets containing them are off. Our anchor design guarantees every technique is probed against r structurally diverse partners, so high-degree rows are covered. We expect a visible gap on subsets containing the high-interaction techniques, with parity on subsets of low-interaction techniques — the gap should track interaction degree, which is the diagnostic signature of anchor coverage working.

**vs. Low-rank assumption diagnostics** — On the measured LASAGNA interaction matrix, we expect the top-3 singular values to explain a majority of the composition variance if the load-bearing assumption holds; if they do not, SCOPE's predictions will regress toward the additive baseline and the rank-sensitivity curve will show rank correlation improving with r beyond 3. The anchor condition number should be moderate (not ill-conditioned) for the deterministic anchor design to satisfy the incoherence conditions required for low-rank completion.

### What would falsify this idea

If SCOPE's rank-correlation gain over the additive baseline is uniform across subsets rather than concentrated on multi-technique compositions with predicted synergy, or if the r=1 ablation matches full SCOPE, then the low-rank interaction term is not carrying the claimed signal and the central claim is wrong. Likewise, if random sampling matches anchor-based probing at equal budget, the anchor design premise fails, and if the singular-value spectrum of the measured interaction matrix is not dominated by the top-3 factors, the load-bearing low-rank assumption is falsified.

## References

1. Retrieval Augmented Generation Compositionality
2. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
3. Dense X Retrieval: What Retrieval Granularity Should We Use?
4. Few-shot Learning with Retrieval Augmented Language Models
5. Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models
6. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories
