# Rate-Distortion Certified Memory Encoding for KV Cache Reconstruction in MemOnce

## Motivation

MemOnce compresses long-horizon KV history into a fixed-size set of final-layer hidden states and decodes them back into per-layer KV caches, but its compression ratio is a free hyperparameter with no stated distortion metric tying memory size to information loss (E5, E10). The same structural gap recurs across the efficient-transformer literature: token-redundancy methods such as the 3D point cloud token-merging work report 90–95% token reduction while only asserting 'competitive performance' with no bound on how much attention output changes (E1, E6), and linear-time attention operators like ToST swap the attention form for linear complexity without any certificate on the deviation from the original attention distribution (E4, E8). The root cause is that compression budgets in these architectures are set by heuristics—subset selection, eviction, or fixed ratios—rather than by a quantity that provably controls the downstream attention-output error, so a practitioner cannot know, for a given memory size, how far the decoded attention distribution can drift from the uncompressed one.

## Key Insight

Because the decoded KV cache is a deterministic function of the retained final-layer memory, the KL divergence between original and decoded per-layer attention distributions is controlled by a single per-token reconstruction residual, so selecting memory entries to minimize that residual—rather than by a heuristic subset—turns memory size into an explicit, computable error budget.

## Method

## Rate-Distortion Certified Memory Encoder-Decoder (RD-Mem)

**(A) What it is.** RD-Mem replaces MemOnce's fixed-ratio final-layer memory selection with a distortion-minimizing encoder and a certified decoder. Input: the stream of final-layer hidden states $\{h_t\}_{t=1}^{T}$ produced during prefill, a memory budget $M$ (number of retained states), and the per-layer KV projection matrices $\{W_K^{\ell}, W_V^{\ell}\}_{\ell=1}^{L}$. Output: a retained memory set $\mathcal{S}$ with $|\mathcal{S}|=M$, a decoded per-layer KV cache $\{\hat{K}^{\ell}, \hat{V}^{\ell}\}$, and a scalar certificate $\varepsilon(M)$ that upper-bounds, at each decoding step, the KL divergence between the original attention distribution $p^{\ell}_t$ and the decoded one $\hat{p}^{\ell}_t$ for every layer $\ell$.

**(B) How it works.**

```pseudocode
INPUT: final-layer hidden states H = {h_t}_{t=1..T}, budget M,
       KV projections {W_K^l, W_V^l}_{l=1..L}, ridge lambda=1e-3,
       residual slack delta=1e-2, temperature tau (model's softmax temp),
       calibration set C of 512 held-out queries per layer (sampled from a
       held-out split of the same long-context corpus, disjoint from test)

# Phase 1: per-token reconstruction residual under a low-rank memory code
1. Fit a rank-r linear decoder D in closed form by ridge regression:
       D = argmin_D  sum_t || h_t - D D^T h_t ||^2 + lambda ||D||_F^2
   with r = M (memory size equals code rank, so |S| = r).
   (Warm-start: when M increases by 1, update D via incremental SVD of the
    centered H matrix; the search grid is M in {16, 32, 64, 128, 256}.)
2. For each token t compute residual
       e_t = || h_t - D D^T h_t ||_2
   and per-token KV reconstruction error vector
       d_t^l = ( || W_K^l (h_t - D D^T h_t) ||_2 , || W_V^l (h_t - D D^T h_t) ||_2 )

# Phase 2: distortion-minimizing memory selection (NOT a heuristic subset)
3. Solve the cardinality-constrained selection by greedy residual-cover:
       S <- empty
       repeat M times:
           t* <- argmax_{t not in S}  e_t * (1 + sum_{s in S} <h_t, h_s>_+ / ||h_s||^2)
           S <- S union {t*}
   (the factor rewards tokens whose residual direction is not yet spanned
    by already-selected states; ties broken by lower index)
4. Store memory M_S = {h_t : t in S} together with the low-rank decoder D.

# Phase 3: certified decoding of KV caches
5. For every layer l and every token t not in S:
       h_hat_t = D D^T h_t                      # decoded final-layer state
       K_hat_t^l = W_K^l h_hat_t ;  V_hat_t^l = W_V^l h_hat_t
   For t in S: K_hat_t^l = W_K^l h_t (exact).
6. Compute the certificate at decode time for query q_t^l = W_Q^l h_t:
       eps_l = (1/tau) * max_t  || W_K^l (h_t - h_hat_t) ||_2 * ||q_t^l||_2
               + (1/tau) * max_t  || W_V^l (h_t - h_hat_t) ||_2
       eps(M) = max_l eps_l
   (Assumption A, stated explicitly: the worst-case single-token logit
    perturbation upper-bounds the aggregate softmax-normalized KL, i.e.,
    perturbations do not accumulate constructively across the softmax
    denominator. Failure mode F: heavy-tailed logits causing softmax
    saturation and denominator shifts that violate the per-token bound.)
7. Emit eps(M) alongside the decoded cache; if eps(M) > delta, the caller
   increases M by one (next grid point) and repeats from Phase 1 (monotone
   budget search).
8. Empirical calibration of the certificate: on the calibration set C,
   compute the measured KL divergence between original and decoded
   attention distributions for each layer and each query, and record the
   empirical maximum KL_C(M). Report both eps(M) and KL_C(M) as the
   certified and empirical distortion curves.
```

**(C) Why this design.** We chose a closed-form low-rank decoder $D$ over a learned neural decoder because the certificate in Phase 6 is only valid if the reconstruction map is fixed and differentiable in closed form, accepting the cost that $D$ cannot exploit nonlinear token structure the way MemOnce's learned encoder-decoder could. We chose greedy residual-cover selection over MemOnce's heuristic subset because the objective in Phase 3 is exactly the quantity that enters the certificate (the per-token residual $e_t$), accepting the cost that greedy selection is $O(M \cdot T)$ rather than a single pass. We chose to tie memory size $M$ directly to the code rank $r$ over treating them as independent hyperparameters because a rank-$M$ code makes the certificate a function of a single scalar $M$, accepting the cost that the code family cannot have rank larger than the memory budget. A domain expert would not describe RD-Mem as a variant of MemOnce: MemOnce's encoder-decoder is a learned module with a free compression ratio and no distortion metric, whereas RD-Mem's selection rule and certificate are both derived from the same residual $e_t$, so the memory budget and the error bound are the same object viewed twice. Nor is it a variant of token-merging methods such as the 3D point cloud token-merging work (E1, E6): those merge tokens and report task accuracy, while RD-Mem retains tokens and certifies a KL bound on the attention distribution.

**(D) Why it measures what we claim.** The per-token residual $e_t = \|h_t - DD^\top h_t\|_2$ measures the reconstruction distortion of the final-layer memory because we assume the KV projections $W_K^\ell, W_V^\ell$ are Lipschitz with constant equal to their spectral norm, which makes $\|W_K^\ell(h_t - \hat h_t)\|_2 \le \|W_K^\ell\|_2 \, e_t$; this assumption fails when the projection is rank-deficient in the direction of the residual, in which case $e_t$ overstates the actual KV error and the certificate is conservative rather than tight. The certificate $\varepsilon(M)$ measures the KL divergence between original and decoded attention distributions because we assume (Assumption A) that the worst-case single-token logit perturbation upper-bounds the aggregate softmax-normalized KL, i.e., perturbations do not accumulate constructively across the softmax denominator, and that the softmax logits are bounded so that a logit perturbation of magnitude $\Delta$ changes the KL divergence by at most $\Delta/\tau$ (Pinsker-type bound combined with the softmax Lipschitz property); this assumption fails (failure mode F) when the attention logits are heavy-tailed and the softmax saturates, in which case $\varepsilon(M)$ reflects a worst-case logit perturbation rather than the typical KL. The greedy selection score measures which memory entries most reduce the certified distortion because we assume the residual directions of unselected tokens are approximately orthogonal to selected ones, so covering a new direction reduces the max residual; this assumption fails when residuals are highly correlated across tokens, in which case the greedy score reflects coverage of the residual span rather than of the individual errors. The empirical calibration step (Phase 8) measures the tightness of $\varepsilon(M)$ against the measured KL on held-out queries because we assume the calibration set is representative of the test distribution; this assumption fails when the test queries exhibit heavier-tailed attention than the calibration set, in which case the reported $\varepsilon(M)$ remains a worst-case bound while the empirical KL underestimates the test-time distortion.

## Contribution

['A rate-distortion memory encoder for MemOnce-style architectures in which memory size M and low-rank code rank are the same scalar, and memory entries are selected by a greedy residual-cover rule derived from the same per-token residual that enters the certificate.', 'A certified decoding procedure that emits a scalar ε(M) bounding the per-layer KL divergence between original and decoded attention distributions as an explicit function of the memory budget, with a monotone budget-search loop that increases M until ε(M) falls below a user-specified slack δ.', 'A design principle for efficient transformer memory compression: the compression budget should be defined as the rank of a closed-form reconstruction code rather than as a free ratio, so that the budget and the error bound are the same quantity.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongBench (16 tasks, 4k–32k context) with a held-out calibration split of 512 queries per layer | Standard long-context benchmark with repeated entities and diverse attention patterns |
| Primary metric | KL divergence between original and decoded attention distributions, measured on test queries | Directly matches the certified quantity |
| Secondary metric | Certificate tightness: ratio ε(M) / measured KL, reported across M ∈ {16,32,64,128,256} | Quantifies conservativeness of the bound |
| Baseline | MemOnce | Tests whether learned encoder-decoder beats certified selection |
| Baseline | Radar decoding | Tests whether token-importance selection matches residual-cover |
| Baseline | Token Statistics Transformer (ToST) | Tests linear-time attention against certified memory |
| Baseline | Leverage-score selection (greedy PCA/leverage scores) at equal M | Tests whether residual-cover is reducible to standard selection |
| Ablation-of-ours | RD-Mem without certificate-driven budget (fixed M=64) | Isolates the adaptive monotone budget search |
| Ablation-of-ours | RD-Mem with random subset selection at equal M | Tests whether residual-cover selection matters beyond budget |

### Why this setup validates the claim

The central claim is that tying memory selection and the error certificate to the same per-token residual yields a memory encoder-decoder whose attention-distortion is both low and bounded, unlike learned or heuristic compressors. The dataset choice matters because LongBench's long-context tasks with repeated entity mentions create long-range dependencies where the final-layer hidden states of a token and its later references share residual directions; this is exactly the regime where a residual-cover selection should outperform heuristic subsetting. The primary metric, KL divergence between original and decoded attention distributions, is the quantity the certificate upper-bounds, so measuring it directly tests whether the bound is meaningful rather than vacuous. The secondary metric, certificate tightness ε(M)/KL, quantifies how conservative the bound is across budgets; a ratio near 1 indicates a tight bound, while orders-of-magnitude gaps would confirm the bound is vacuous. MemOnce tests the sub-claim that a learned decoder with a free compression ratio cannot match a certified closed-form decoder at equal budget; Radar tests whether dynamic token-importance search, which is training-free but not residual-aware, achieves comparable distortion; ToST tests whether abandoning pairwise attention entirely is a fair alternative. The leverage-score baseline tests whether the residual-cover objective is reducible to standard greedy PCA/leverage-score selection; if leverage scores match residual-cover on the certificate, the selection rule adds no value. The fixed-M ablation isolates the adaptive budget search: if fixed-M RD-Mem already matches adaptive, the certificate is doing no work beyond selection, which would weaken the design's rationale. The random-subset ablation tests whether any selection at equal M suffices, isolating the contribution of the residual-cover score itself.

### Expected outcome and causal chain

**vs. MemOnce** — On a case where a rare entity is mentioned early and referenced after a long intervening span, MemOnce's learned encoder-decoder compresses history into a fixed-size state and loses the fine-grained KV detail needed to reconstruct that entity's keys, because its compression ratio is free and its decoder is trained for average reconstruction rather than worst-case residual. Our method instead retains exactly the tokens whose residuals are not spanned by already-selected states, because greedy residual-cover targets the certificate's per-token error term. We expect a noticeable KL gap on the rare-entity subset but near-parity on common-token subsets where MemOnce's average-case training suffices.

**vs. Radar decoding** — On a case where the most important context tokens for a query are not the ones with the largest residual norm but the ones whose residual directions are orthogonal to the retained span, Radar's importance search selects high-attention tokens and misses orthogonal-direction tokens, because its criterion is attention magnitude, not reconstruction coverage. Our method instead selects by residual-cover score, which rewards unspanned directions. We expect RD-Mem to show lower worst-case KL on queries whose relevant tokens are directionally diverse, with parity on queries dominated by a single repeated token.

**vs. Token Statistics Transformer (ToST)** — On a case where fine-grained token identity matters, such as distinguishing two similar entities, ToST's linear-time attention discards pairwise token similarity and aggregates token statistics, because its operator computes statistics rather than pairwise logits. Our method instead retains exact KV for selected tokens and decodes the rest with a bounded residual, because it preserves pairwise attention structure. We expect RD-Mem to show lower KL on identity-sensitive subsets and higher memory cost, revealing the efficiency-distortion tradeoff rather than a uniform win.

**vs. Leverage-score selection** — On a case where the residual directions of unselected tokens are highly correlated (e.g., repeated entity mentions), leverage-score selection, which ranks tokens by their variance along principal components, may select tokens that span the dominant variance but leave orthogonal residual directions uncovered, because leverage scores do not account for the residual-cover objective. Our method instead selects tokens whose residuals are not yet spanned by already-selected states, because the greedy score explicitly rewards unspanned directions. We expect RD-Mem to show lower worst-case KL on the correlated-residual subset, with parity on subsets where residuals are already orthogonal.

**vs. RD-Mem fixed M=64** — On a case where the certificate ε(M) exceeds the slack δ at M=64, the fixed-M variant cannot adapt and incurs higher KL, because it lacks the monotone budget search that increases M until ε(M) ≤ δ. Our method instead increases M by one grid point and repeats from Phase 1, because the certificate directly drives the budget. We expect the adaptive variant to achieve lower KL on queries requiring larger memory, with similar average M on easy queries.

**vs. RD-Mem random subset** — On a case where the retained memory must cover diverse residual directions, random subset selection leaves many directions unspanned, because it ignores the residual-cover score. Our method instead selects by residual-cover, which targets the certificate's per-token error term. We expect RD-Mem to show lower KL across all subsets, confirming that the selection rule, not just the budget, drives the improvement.

### What would falsify this idea

If the KL gap between RD-Mem and MemOnce is uniform across all subsets rather than concentrated on the rare-entity and orthogonal-direction subsets where the residual-cover mechanism predicts its advantage, then the certificate is not tracking the failure mode it claims to bound. If the emitted certificate ε(M) is consistently orders of magnitude above the measured KL so that the budget search never triggers, the bound is vacuous and the central claim of a certified memory encoder-decoder fails.

## References

1. MemOnce: A Scalable Memory Architecture for Large Language Models
