# Monotone Sub-Attribute Decomposition for Discrimination-Aware Multimodal Retrieval Embeddings

## Motivation

Universal multimodal embedding models such as DiLE (and contrastive vector models like VLM2Vec) compress relevance into a single monolithic scalar dot product, so their similarity scores are not guaranteed to respect graded relevance refinement: a candidate that is at least as good as another on every relevance sub-attribute can still receive a lower score. DiLE improves discrimination only through better training signals (on-policy hard-negative distillation), which reshapes the scalar but leaves its internal structure unconstrained; GENIUS and MM-Embed similarly optimize ranking behavior without any structural invariant tying sub-attribute dominance to score ordering. The root cause is structural: without decomposing similarity into interpretable, ordered sub-attribute contributions, monotone consistency under relevance refinement cannot be enforced by any amount of negative mining.

## Key Insight

If similarity is a monotone combination of signed contributions along learned sub-attribute axes, then sub-attribute dominance becomes a sufficient condition for higher score, making graded-relevance monotonicity an enforceable structural invariant rather than a hoped-for byproduct of contrastive training.

## Method

**What it is.** MoSA (Monotone Sub-Attribute scoring) wraps DiLE's discrimination-aware latent queries with a graded relevance decomposition head: the query and candidate latent representations are jointly mapped to K=8 signed sub-attribute contributions, and the final similarity is their monotone sum. Inputs are a query embedding and a candidate embedding (both produced by DiLE's frozen latent-query encoder); output is a scalar relevance score plus a K=8-dimensional contribution vector.

**How it works.**

```pseudocode
# Phase 0: inherit DiLE latent queries
z_q = DiLE.encode(query)            # compact latent query representation, dim d=4096 (Qwen2VL hidden)
z_c = DiLE.encode(candidate)

# Phase 1: sub-attribute projection head (K=8 axes)
# Each axis k has W_k in R^{1 x 3d}, b_k in R, u_k in R
for k in 1..K:
    a_k = tanh( W_k [z_q ; z_c ; z_q * z_c] + b_k )   # signed contribution in (-1,1)
    s_k = softplus( u_k^T a_k )                       # non-negative axis score, u_k learned
score(q,c) = sum_k s_k                                # monotone sum over axes

# Phase 1b: axis-decorrelation constraint (enforces non-degenerate dominance predicate)
# Over each mini-batch B of (q,c) pairs, compute S in R^{|B| x K} with S[i,k] = s_k(q_i,c_i)
# Center each column: S_centered = S - mean_over_batch(S)
# Decorrelation loss: L_decor = || (S_centered^T S_centered) / |B| - I_K ||_F^2
# This penalizes off-diagonal axis correlations and prevents axis collapse onto the dominant direction.

# Phase 2: training with dominance-margin + monotonicity regularizer
for each anchor q, positive c+, negative c- in batch:
    L_rank   = max(0, m - score(q,c+) + score(q,c-))   # m = 0.2
    # Soft dominance predicate: c+ dominates c- if s_k(q,c+) >= s_k(q,c-) - tau for all k
    # tau = 0.05 (tolerance calibrated to axis noise; see Phase 2b)
    if s_k(q,c+) >= s_k(q,c-) - tau for all k:
        L_mono = max(0, score(q,c-) - score(q,c+) + delta)  # delta = 0.05
    L = L_rank + lambda * L_mono + gamma * L_decor      # lambda = 0.5, gamma = 0.1

# Phase 2b: axis-correlation calibration (run once after 1 epoch of warmup)
# Compute Pearson correlation matrix R in R^{K x K} of s_k values over a calibration set of 512 training pairs.
# Set tau = 0.05 * (1 + max_{k!=l} |R_{kl}|)  # tolerance grows with observed axis correlation
# If max_{k!=l} |R_{kl}| > 0.7, increase gamma to 0.5 and re-warmup for 1 epoch.

# Phase 3: on-policy dominance-violating hard-negative mining
candidate_pool = top-N by current score (N = 64) using FAISS over frozen DiLE embeddings
for c in candidate_pool:
    if exists c+ with s_k(q,c+) >= s_k(q,c) - tau for all k AND score(q,c) >= score(q,c+):
        add c to violation buffer  # these violate the monotone invariant
retrain head on violation buffer with L_mono emphasized (lambda = 1.0, gamma = 0.1)
```

**Why this design.** We chose a signed per-axis decomposition summed monotonically over a single shared scalar dot product because a monolithic score cannot express which sub-attribute drove the ranking, accepting the cost that K=8 axes must be tuned and may not align with human-interpretable attributes. We chose a softplus non-negative axis score over raw signed contributions because it guarantees that improving any axis can only raise the total score (the monotonicity invariant), accepting the cost that the head loses the ability to express trade-offs where one axis must compensate for another. We chose on-policy dominance-violating hard-negative mining over uniform in-batch negatives because only candidates that actually break the invariant provide gradient signal for the monotonicity regularizer, accepting the cost of a second retrieval pass per query (mitigated by FAISS approximate search over frozen embeddings). We added the axis-decorrelation loss L_decor and the soft dominance tolerance tau because the dominance predicate is only meaningful if the K=8 axes are not collapsed onto a shared confound; without this constraint, correlated axes make the predicate either vacuous (always true) or never-true, and L_mono degenerates to a constant. Unlike DiLE's on-policy hard-negative distillation, which aligns a scalar similarity distribution to a teacher's relevance distribution, MoSA constrains the internal additive structure of the score itself, so a domain expert would not describe it as a variant of DiLE's distillation — the mechanism is a structural decomposition with an explicit ordering invariant, not a distillation objective.

**Why it measures what we claim.** The per-axis contribution s_k(q,c) measures sub-attribute relevance because the tanh-projected interaction term W_k[z_q;z_c;z_q*z_c] is assumed to encode a separable relevance factor that is monotone in the axis direction, and the decorrelation loss L_decor (gamma=0.1) plus the calibration step (tau scaled by max off-diagonal correlation) are designed to enforce that the K=8 axes are statistically independent over the training batch; this assumption fails when sub-attributes are correlated in the training data beyond what L_decor can remove (max |R_{kl}| > 0.7 after warmup), in which case s_k reflects a shared confound rather than an independent factor and the dominance predicate fires almost always or almost never. The soft dominance predicate (s_k(q,c+) >= s_k(q,c-) - tau for all k, with tau = 0.05 * (1 + max_{k!=l} |R_{kl}|)) measures graded-relevance dominance because the monotone sum makes axis-wise dominance sufficient for score ordering up to the tolerance tau; this assumption fails when the softplus saturation compresses two axes to near-equal values, in which case dominance is detected but contributes negligibly to the score gap — we report the fraction of dominance cases where the score gap is below 0.01 as a diagnostic. The violation buffer measures the frequency of monotonicity failures because it collects exactly the candidates where the invariant is breached; this assumption fails when the top-N=64 pool is too small to surface violation cases, in which case the buffer underestimates true violation rate — we report buffer size per query and flag queries with zero violations.

## Contribution

(1) A monotone sub-attribute decomposition head (MoSA) that replaces DiLE's monolithic similarity with K signed axis contributions combined by a monotone sum, making graded-relevance monotonicity an explicit structural constraint. (2) A dominance-margin monotonicity regularizer paired with on-policy dominance-violating hard-negative mining that targets the specific candidates breaking the invariant, rather than uniformly mining confusable negatives. (3) A design principle showing that decomposable scoring exposes per-axis relevance signals that a scalar dot product cannot, enabling diagnosis of which sub-attribute drives retrieval errors.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | M-BEIR / MMEB retrieval suites | Multi-task, multi-modality, confusable negatives stress discrimination |
| Primary metric | Recall@1 on hard-negative subsets | Directly detects ranking of confusable candidates |
| Secondary metric | Axis-correlation matrix max |R_{kl}| and dominance-to-score-gap fraction | Verifies non-degenerate dominance predicate |
| Baseline | DiLE (frozen latent-query encoder) | Tests whether decomposition adds over inherited scorer |
| Baseline | CLIP-style dual encoder | Tests cross-modal alignment gap on composed queries |
| Baseline | MM-Embed (modality-aware hard negatives) | Tests whether axis structure beats scalar hard-negative mining |
| Baseline | Additive attribute model (K=8 linear sum of tanh projections, no monotone softplus) | Tests whether monotone invariant beats plain additive decomposition |
| Ablation-of-ours | MoSA without L_mono (λ=0) | Isolates monotonicity invariant from decomposition head |
| Ablation-of-ours | MoSA without L_decor (γ=0) | Tests whether decorrelation constraint is necessary for non-degenerate dominance |
| Ablation-of-ours | MoSA with hard dominance (τ=0) | Tests whether soft tolerance is needed for stable training |

### Why this setup validates the claim

The central claim is that constraining the score's internal additive structure — K=8 monotone sub-attribute axes plus an explicit dominance invariant — improves discrimination on confusable candidates beyond what a monolithic similarity can express. The M-BEIR/MMEB suites are the right testbed because they pair multi-task retrieval with genuinely hard, modality-mixed negatives, so a scorer that merely re-ranks easy cases cannot pass. DiLE is the decisive baseline: it shares our frozen encoder, so any gain must come from the decomposition and invariant, not from better representations. CLIP tests the cross-modal alignment sub-claim; MM-Embed tests whether axis-wise structure beats scalar modality-aware negative mining. The additive attribute model (K=8 linear sum of tanh projections without softplus monotonicity) tests whether the monotone invariant itself, rather than the decomposition head, drives the gain. Recall@1 on hard-negative subsets is the correct metric because our predicted effect is concentrated exactly where near-miss candidates must be separated; a gain there, with parity on easy subsets, is the falsifiable signature. The λ=0 ablation isolates the monotonicity regularizer from the head itself, so a gain that survives only with L_mono on confirms the invariant is doing the work rather than extra parameters. The γ=0 ablation tests whether the decorrelation constraint is necessary to keep the dominance predicate non-degenerate; if γ=0 matches full MoSA, the axis-collapse concern is empirically unfounded. The τ=0 ablation tests whether the soft tolerance is needed for stable training. The secondary metrics (max off-diagonal axis correlation and dominance-to-score-gap fraction) directly test the load-bearing assumption that the K=8 axes are separable and that dominance is non-vacuous.

### Expected outcome and causal chain

**vs. DiLE (frozen latent-query encoder)** — On a case where a candidate matches the query's dominant attribute but is wrong on a secondary attribute, the baseline produces a high scalar similarity because its monolithic score collapses all evidence into one direction with no axis to penalize the mismatch. Our method instead lowers that axis's softplus contribution and, via the dominance invariant, cannot let the total score exceed a true positive that dominates it on every axis, so we expect a concentrated Recall@1 gain on hard-negative subsets and near-parity on easy ones. We also expect the axis-correlation matrix to show max |R_{kl}| < 0.5 after decorrelation training, confirming non-degenerate axes.

**vs. CLIP-style dual encoder** — On a case where the query composes text and image and the relevant item is confusable with a modality-mismatched distractor, the baseline produces a ranking driven by independent encoding without task guidance, because it has no mechanism to weigh sub-attributes of a composed query. Our method instead projects the joint interaction term into signed axes that encode which sub-attribute the composed query demands, so we expect a visible gap on composed-query and cross-modal subsets, with the largest separation where distractors share surface modality.

**vs. MM-Embed (modality-aware hard negatives)** — On a case where a hard negative is hard for a single scalar reason (modality bias), the baseline produces corrected ranking but still cannot articulate why a near-miss loses, because its hard-negative mining only reshapes one similarity direction. Our method instead surfaces the specific violating axis and enforces dominance, so we expect a modest but consistent gain on modality-mixed confusables and a larger gain on cases where multiple axes disagree — precisely the subset where scalar mining is blind.

**vs. Additive attribute model (K=8 linear sum of tanh projections, no monotone softplus)** — On a case where a candidate is better on one axis but worse on another, the baseline can produce a higher score for the worse candidate because its signed contributions can trade off without a monotonicity constraint. Our method's softplus non-negative axis scores guarantee that improving any axis can only raise the total score, so we expect a gain on cases where axis-wise dominance should determine ranking, and we expect the additive baseline to show a higher dominance-violation rate in the violation buffer.

### What would falsify this idea

If the gain over DiLE is uniform across easy and hard-negative subsets rather than concentrated on confusable, dominance-violating cases, the decomposition is acting as generic capacity rather than the claimed structural constraint. Likewise, if the λ=0 ablation matches the full model, the monotonicity invariant is not doing the work and the central claim fails. Additionally, if the γ=0 ablation matches full MoSA and the axis-correlation matrix shows max |R_{kl}| > 0.7, the dominance predicate is degenerate and the load-bearing assumption is empirically false.

## References

1. DiLE: Discrimination-Aware Latent-Space Learning for Universal Multimodal Embeddings
