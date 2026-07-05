# Provably Sufficient Token Selection for KV Cache Compression via Determinantal Point Processes

## Motivation

Prior KV cache compression methods like ReFreeKV and SnapKV rely on heuristic importance scores (e.g., attention weights) that lack theoretical guarantees of sufficiency. These methods assume token importance is independent or can be inferred locally, ignoring the pairwise coupling in the attention kernel, leading to reconstruction errors on tasks requiring long-range dependencies. This limitation is structural: without a provable condition for sufficiency, compression is inherently lossy in a task-dependent way.

## Key Insight

The determinant of the attention kernel submatrix measures the volume spanned by the selected tokens' attention interactions, and if this determinant dominates the full kernel, the reconstruction error from pruning is bounded.

## Method

(A) **What it is**: DPP-KV is a provably sufficient token selection method for KV cache compression that models the set of cached tokens as a greedy maximizer of the log-determinant of the attention Gram matrix. Input: attention matrix A ∈ ℝ^{n×d} (key-value hidden states) and budget k. Output: set S of k token indices to cache.

(B) **How it works**:
```python
# Input: Attention matrix A (n x d), budget k, optional threshold delta
# Output: Set S of k token indices

# Step 1: Compute Gram matrix G = A @ A.T  (n x n, positive semidefinite)
G = A @ A.T

# Step 2: Initialize empty set S and iterate greedily
S = []
for t in range(k):
    best_i = None
    best_det = -inf
    for i in range(n):
        if i in S: continue
        # Compute determinant of G_{S ∪ {i}} using rank-1 update formula
        candidate_set = S + [i]
        # Since log-det is submodular, greedy increment: logdet(G_{candidate}) - logdet(G_S)
        # Efficiently compute via Cholesky update: we maintain L_S (lower triangular of G_S)
        # For simplicity, direct computation (O(|S|^2) per candidate) but we can do better.
        # Using the identity: det(G_{S∪i}) = det(G_S) * (G_ii - G_{i,S} @ G_S^{-1} @ G_{S,i})
        # We maintain G_S^{-1} via Sherman-Morrison updates.
        # Here we outline the core step:
        if t == 0:
            det_val = G[i,i]
        else:
            # G_S is the submatrix for current S
            G_S = G[np.ix_(S, S)]  # shape (t, t)
            G_iS = G[i, S]          # shape (t,)
            G_Si = G_iS.T
            # Schur complement: s = G[i,i] - G_iS @ inv(G_S) @ G_Si
            inv_G_S = np.linalg.inv(G_S)  # cached and updated
            s = G[i,i] - G_iS @ inv_G_S @ G_iS
            det_val = det_S * s  # det_S is the determinant of G_S (maintained)
        if det_val > best_det:
            best_det = det_val
            best_i = i
    S.append(best_i)
    # Update det_S and inverse for next iteration
    # ... (details omitted for brevity but implementable)
return S
```
Hyperparameter: k (budget) is user-specified; threshold δ (optional) can be used to adapt k if the determinant ratio det(G_S)/det(G) does not exceed 1-δ.

(C) **Why this design**: We chose greedy determinant maximization over exact MAP inference because the latter requires a full eigendecomposition of the kernel (O(n^3)), which is prohibitive for long sequences; greedy runs in O(nk^2) and provides a (1-1/e) approximation guarantee for submodular log-determinant. We accept that greedy may not find the truly optimal set, but the approximation bound ensures near-optimal volume coverage in practice. We use the Gram matrix G = A A^T rather than raw attention scores because G captures pairwise attention interactions as inner products of attention vectors, aligning with the geometric interpretation of volume; the trade-off is the O(n^2 d) cost to compute G, which is acceptable since n is typically < 10k. We selected a deterministic greedy selection over stochastic DPP sampling because we need a reproducible and interpretable selection to check the sufficiency condition; stochastic sampling could produce different sets each inference, complicating verification. The cost is that we sacrifice the natural diversity of DPP samples, but volume maximization inherently encourages diversity.

(D) **Why it measures what we claim**: The determinant of submatrix G_S measures the volume spanned by the attention vectors of selected tokens in the feature space induced by the attention kernel. The ratio det(G_S)/det(G) measures the fraction of total interaction volume retained; our key insight is that if this ratio is near 1, the reconstruction error from pruning the remaining tokens is bounded, because the attention output can be approximated as a linear combination of selected tokens' values via least-squares projection. This equivalence relies on the assumption that the attention kernel is positive semidefinite and that selected tokens' attention vectors linearly span the space of all attention vectors; this assumption fails when the attention kernel has rank lower than the number of tokens, in which case the determinant ratio may be 1 with fewer tokens, but the bound still holds. The computational quantity det(G_S) measures 'sufficiency' of the selected set because it quantifies coverage of attention interaction volume; the assumption that makes this equivalent to reconstruction error bound is that the attention output is a linear function of the key-value inner products, which holds for standard dot-product attention. This assumption fails when non-linear activations are applied after attention, in which case the determinant ratio reflects only the linear component of the interaction.

## Contribution

(1) A novel KV cache compression method, DPP-KV, that models token selection as greedy log-determinant maximization of the attention Gram matrix, providing a provably sufficient condition for lossless compression via the determinant ratio. (2) A theoretical bound linking the determinant ratio of the selected submatrix to the reconstruction error of the pruned attention output, establishing the first theoretical foundation for importance scoring in KV cache compression. (3) An efficient O(nk^2) greedy algorithm with update formulas that maintains the inverse and determinant incrementally, enabling practical deployment.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LongBench | Long context, diverse tasks |
| Primary metric | Perplexity degradation | Directly measures output fidelity |
| Baseline | H2O | Threshold-based eviction |
| Baseline | SnapKV | Accumulated attention selection |
| Baseline | StreamingLLM | Initial+recent token retention |
| Ablation-of-ours | DPP-KV (random subset) | Tests necessity of determinant selection |

### Why this setup validates the claim
LongBench provides multiple long-context tasks (e.g., document QA, summarization) that stress the KV cache. Perplexity degradation directly measures how well compressed KV cache preserves language modeling quality, aligning with our reconstruction error bound. Baselines cover the main families of pruning methods: threshold-based (H2O), selection-based on average attention (SnapKV), and heuristic retention (StreamingLLM). The ablation isolates the greedy determinant maximization from the overall framework. If DPP-KV outperforms all baselines, especially on tasks requiring diverse token interactions, the claim that volume-based selection is sufficient is supported. Conversely, comparable performance to random selection would falsify the central insight.

### Expected outcome and causal chain

**vs. H2O** — On a case where attention is uniformly distributed across all tokens (e.g., summarization with no clear salient tokens), H2O’s threshold may evict many tokens with low individual scores but collectively important interactions, leading to high perplexity degradation. Our method selects tokens that maximize the volume of attention vectors, preserving diversity even when individual scores are low. We expect DPP-KV to show 0.5–1.0 lower perplexity degradation on such tasks, while parity on tasks with peaked attention.

**vs. SnapKV** — On a case where token importance changes dynamically (e.g., multi-turn dialogue with shifting focus), SnapKV accumulates attention over many generations, giving high weight to outdated tokens. This biases selection toward past context, missing new critical tokens. Our method recalculates the Gram matrix each step, so it adapts to current attention patterns. We expect DPP-KV to have lower perplexity degradation on dialogue tasks (gap ∼0.3–0.7) and similar on static QA.

**vs. StreamingLLM** — On a case requiring retrieval of an arbitrary distant token (e.g., needle-in-a-haystack), StreamingLLM retains only initial and recent tokens, losing the needle. Our method re-evaluates all tokens and can select the needle token if its attention vector is distinctive (i.e., spans a new direction). We expect DPP-KV to achieve near-perfect recall on such synthetic tasks, while StreamingLLM degrades sharply with context length.

### What would falsify this idea
If DPP-KV’s perplexity degradation is uniform across all LongBench subsets, or if its performance is statistically indistinguishable from the random ablation on tasks requiring long-range dependencies, then the volume-maximization claim is falsified.

## References

1. ReFreeKV: Towards Threshold-Free KV Cache Compression
2. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
3. SnapKV: LLM Knows What You are Looking for Before Generation
4. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
5. Lost in the Middle: How Language Models Use Long Contexts
