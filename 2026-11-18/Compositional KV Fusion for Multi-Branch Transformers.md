# Compositional KV Fusion for Multi-Branch Transformers

## Motivation

Multi-Branch Transformers (MBT) introduce Joint-KV as a heuristic shared cache built from branch representations, but the abstract provides no evidence that fused-cache attention reproduces branch-wise attention outputs; this reflects a broader structural gap in efficient transformer design where aggregation operators (token merging, cache sharing, quantization) are assumed lossless without an equivalence condition. In MBT specifically, branches are treated as interchangeable feature producers, so Joint-KV cannot guarantee that arbitrary branch-count scaling preserves each branch's attention semantics, undermining the long-context inference claim that motivates the architecture.

## Key Insight

A per-branch affine transform applied to keys and values, whose inverse is folded into the query, makes the fused KV cache an exact sufficient statistic for every branch's attention output, so cache-compositionality holds by construction rather than by approximation.

## Method

**What it is.** We introduce **Compositional Joint-KV (CJKV)**, a drop-in replacement for MBT's Joint-KV that takes per-branch key/value tensors at each depth and produces a single fused KV cache together with per-branch query-side transforms, such that attention over the fused cache with the transformed query equals each branch's own attention output up to a fixed per-branch scale. Inputs: per-branch K_b, V_b ∈ R^{L×d} for b=1..B, and per-branch queries Q_b ∈ R^{L×d}. Outputs: fused K*, V* ∈ R^{L×d}, and per-branch (A_b, s_b) where A_b ∈ R^{d×d} is an invertible affine map and s_b a scalar.

**Load-bearing assumption (stated explicitly).** CJKV assumes that a single fused cache K*, V* formed by averaging per-branch keys/values in each branch's own whitened coordinate frame, combined with a per-branch query-side inverse transform, is an exact sufficient statistic that reproduces each branch's own softmax attention output up to a fixed scalar. This holds only when the branch key distributions are identical Gaussians (so cross-branch contributions cancel in the softmax denominator) and when calibration and inference distributions match. When these conditions fail, the fused cache is a bounded approximation, not an equivalence; the residual is quantified by the per-branch residual term R_b defined below.

**How it works.**
```pseudocode
# Phase 1: per-branch calibration (once per depth, offline on a calibration batch of 512 sequences)
for b in 1..B:
    mu_K_b, mu_V_b = mean(K_b, axis=0), mean(V_b, axis=0)
    Sigma_K_b = cov(K_b) + eps*I          # eps = 1e-5, ridge for invertibility
    A_b = cholesky(Sigma_K_b)             # lower-triangular, invertible by construction
    c_b = mu_K_b                          # translation
    s_b = mean(||V_b - mu_V_b||_2) / mean(||K_b - mu_K_b||_2)   # scale matching
    # residual term: per-branch low-rank correction to shared base cache
    R_b = (K_b - c_b) @ A_b^{-1} - mean((K_b - c_b) @ A_b^{-1}, axis=0)  # centered whitened keys
    R_b = R_b[:, :r]                      # keep top-r principal directions, r=16
# Phase 2: fused cache construction (per token, online)
K_star = sum_b (K_b - c_b) @ A_b^{-1} / B          # average in whitened space
V_star = sum_b (V_b - mu_V_b) / (B * s_b)          # average in centered space
# Phase 3: branch-wise attention over fused cache
for b in 1..B:
    Q_b_tilde = (Q_b - c_b) @ A_b^{-T}              # query-side inverse transform
    scores_b = Q_b_tilde @ K_star^T / sqrt(d)
    out_b = softmax(scores_b) @ V_star * s_b        # rescale to branch units
    # add residual correction: attention over per-branch residual keys
    scores_res_b = Q_b_tilde @ R_b^T / sqrt(d)
    out_b = out_b + softmax(scores_res_b) @ (V_b - mu_V_b) / s_b  # residual term
return K_star, V_star, {(A_b, s_b, c_b, R_b)}_b
```

**Why this design.** We chose a per-branch Cholesky whitening transform over a learned gating MLP because the Cholesky factor is invertible by construction, which is exactly the property needed for the fused cache to be a sufficient statistic; the cost is that the transform is linear and cannot capture branch-specific nonlinear key structure. We chose to average in whitened space rather than in raw space because raw-space averaging is what Joint-KV already does and it breaks compositionality whenever branch covariances differ; the cost is one d×d matrix inverse per branch per depth, which we precompute offline. We chose to fold the inverse transform into the query rather than into the keys because folding into keys would require per-branch caches (defeating the purpose of sharing), accepting that each branch pays a d×d query-side matmul at inference. The residual term R_b is added to bound the compositionality gap when the Gaussian assumption fails; it is a low-rank correction (r=16) that captures the dominant per-branch deviations from the shared whitened mean. A domain expert would not describe CJKV as a variant of Joint-KV: Joint-KV concatenates or averages branch KV tensors with no equivalence condition, whereas CJKV's fused cache is defined as the image of a per-branch invertible map whose inverse is carried by the query — the cache is a coordinate change, not an aggregation.

**Why it measures what we claim.** The quantity `K_star` measures *cache-compositionality* because the named assumption is that each branch's key distribution is well-approximated by a Gaussian with covariance Sigma_K_b, under which whitening makes branches exchangeable and the average in whitened space is the sufficient statistic for the mixture; this assumption fails when a branch's key distribution is heavy-tailed or multimodal (e.g., a branch specializing in rare tokens), in which case `K_star` reflects the whitened mean of the dominant mode and the per-branch attention output deviates from the branch-wise output by the residual covariance mismatch, which is captured by the residual term R_b. The quantity `s_b` measures *semantic scale preservation* because the assumption is that value magnitude is branch-invariant after centering; this fails when a branch's value distribution has a different norm profile, in which case `s_b` rescales the output but cannot recover the lost per-token value direction. The quantity `Q_b_tilde` measures *query-side equivalence* because the assumption is that the affine map A_b is the same one used to whiten K_b; this fails if calibration and inference distributions drift, in which case the transformed query no longer lands in the fused cache's coordinate frame and the attention logits are systematically biased; we report the calibration-vs-inference covariance distance per depth as a drift diagnostic.

## Contribution

['A compositional KV fusion operator (CJKV) for Multi-Branch Transformers that replaces heuristic Joint-KV with a per-branch invertible affine transform and query-side inverse, making the shared cache a sufficient statistic for branch-wise attention by construction.', 'A cache-compositionality invariant — the fused-cache attention output equals branch-wise attention up to a fixed per-branch scale — instantiated as a testable per-token residual metric that can be measured at every depth without retraining.', 'An empirical characterization of when branch KV distributions are exchangeable enough for shared-cache fusion to be lossless, via a calibration-drift probe across branch counts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-context language modeling corpus (e.g., PG-19, 512-token calibration set) | Stresses KV-cache sharing where branch divergence is largest |
| Primary metric | Perplexity gap vs. unshared per-branch attention | Directly measures compositionality error, not just speed |
| Baseline | Sequential Transformer (matched params/compute) | Tests whether branching plus CJKV beats depth scaling |
| Baseline | MBT with Joint-KV | Tests whether whitened fusion beats raw averaging |
| Baseline | MBT with concatenated per-branch KV | Tests whether sharing loses anything versus no sharing |
| Ablation-of-ours | CJKV with identity whitening (A_b = I) | Isolates the contribution of covariance-aware fusion |
| Ablation-of-ours | CJKV without residual term (R_b = 0) | Isolates the contribution of the residual correction |
| Diagnostic | Calibration-vs-inference covariance distance per depth | Measures distribution shift that breaks sufficiency |

### Why this setup validates the claim

The central claim is that a per-branch invertible whitening map turns the fused KV cache into a sufficient statistic, so attention over one shared cache reproduces each branch's own attention output up to a fixed scalar, with a bounded residual when the Gaussian assumption fails. That claim is falsifiable only if we can separate *compositionality error* from *capacity loss* from *speed*. The concatenated per-branch KV baseline is the upper bound on fidelity — it shares nothing, so any perplexity gap between it and CJKV is pure compositionality error, which is exactly the quantity the method claims to drive near zero. The MBT-with-Joint-KV baseline isolates whether whitening matters: both share one cache, so the difference is attributable to the coordinate change rather than to sharing itself. The sequential Transformer baseline tests the orthogonal scaling sub-claim that branching is a worthwhile axis at matched budget. The identity-whitening ablation is the decisive control: if CJKV's gain survives with A_b = I, then the covariance matching is doing no work and the mechanism story is wrong. The no-residual ablation isolates the contribution of the low-rank correction that bounds the compositionality gap. Perplexity gap versus unshared attention is the right primary metric because it is monotone in exactly the residual covariance mismatch the method is designed to eliminate; a throughput-only metric would be blind to whether the fused cache is a faithful substitute. The covariance distance diagnostic directly measures the distribution-shift failure mode that the sufficiency claim depends on.

### Expected outcome and causal chain

**vs. Sequential Transformer (matched params/compute)** — On a case where long-context tokens require integrating evidence across many positions, the sequential baseline must spend its depth budget on serial layers, so it under-utilizes parallel capacity and shows higher perplexity at matched compute. Our method instead runs independently parameterized branches in parallel and fuses them through one shared cache, so the same compute buys more effective width. We expect a modest but consistent perplexity advantage concentrated on long-context evaluation slices, with near-parity on short-context slices where depth suffices.

**vs. MBT with Joint-KV** — On a case where two branches have learned markedly different key covariances (one branch specializing in rare, heavy-tailed tokens), raw-space averaging in Joint-KV mixes incompatible coordinate frames, so the dominant branch's keys swamp the other and the fused cache misrepresents the minority branch, producing a perplexity penalty on rare-token subsets. Our method instead whitens each branch before averaging, so each branch contributes in its own standardized frame, and the query-side inverse transform restores branch units. We expect a visible gap on rare-token and minority-branch subsets with parity on high-frequency subsets.

**vs. MBT with concatenated per-branch KV** — On a case where branches genuinely disagree about key structure, concatenation preserves every branch exactly, so it is the fidelity ceiling but pays full cache memory and attention cost. Our method instead compresses to one cache, so any gap here is the price of sharing. We expect CJKV to sit close to this ceiling on most subsets and to diverge only where the Gaussian key assumption fails — heavy-tailed or multimodal branch distributions — signaling where the sufficient-statistic assumption breaks; the residual term R_b should reduce this gap relative to the no-residual ablation.

**vs. CJKV with identity whitening (A_b = I)** — If the identity-whitening ablation matches full CJKV on the primary metric, then covariance-aware fusion is doing no work and the claimed mechanism is wrong. We expect full CJKV to outperform identity whitening on rare-token and minority-branch subsets, confirming that the whitening transform is the active ingredient.

**vs. CJKV without residual term (R_b = 0)** — The residual term is designed to bound the compositionality gap when the Gaussian assumption fails. We expect the no-residual variant to show a larger perplexity gap versus the concatenated baseline on heavy-tailed or multimodal subsets, and the full CJKV to close part of that gap, confirming the residual's role.

### What would falsify this idea

1. If the identity-whitening ablation (A_b = I) matches full CJKV on the primary metric, then covariance-aware fusion is doing no work and the claimed mechanism is wrong. 2. If CJKV's perplexity gap versus the concatenated baseline is uniform across subsets rather than concentrated on heavy-tailed or minority-branch subsets, then the predicted failure mode is not the operative one and the compositionality claim is unsupported. 3. If the calibration-vs-inference covariance distance is large and correlates with perplexity degradation, the sufficiency claim is invalid at inference.

## References

1. Multi-Branch Transformer
