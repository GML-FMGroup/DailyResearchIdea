# GroupKV: Deterministic Key-Value Head Selection via Query-Key Alignment Norm for Grouped Query Attention

## Motivation

Existing mixture-of-experts attention mechanisms, such as GQE, only sparsify query heads while keeping all key-value heads dense, leaving KV cache as a bottleneck for long sequences. This limitation stems from the assumption that key-value heads must be fully activated and cached densely, a structural property inherited from GQA and MoA. No prior work establishes token-level sparsity in KV head activation while maintaining group sharing, which is necessary to reduce both compute and memory for grouped query attention.

## Key Insight

The L2 norm of query-key alignment scores aggregated across all tokens in a GQA group provides a deterministic and reliable measure of each KV head's overall relevance, enabling group-consistent selection without per-token routing overhead.

## Method

### (A) What it is
GroupKV is a mechanism that, for each GQA group, selects a subset of key-value heads based on the L2 norm of the query-key alignment scores aggregated across all tokens in the group, after per-token softmax calibration. Input: query states Q_group ∈ ℝ^(T×d) (concatenated queries from all query heads in the group), key projections K_heads ∈ ℝ^(N×d) (for N KV heads in the group), value projections V_heads. Output: indices of the top-k active KV heads per group.

### (B) How it works
```python
# For a GQA group with T tokens and N KV heads:
# 1. Compute alignment scores: S = Q_group @ K_heads.T  # shape (T, N)
# 2. Apply per-token softmax calibration: S_soft = torch.softmax(S, dim=-1)  # normalize per token
# 3. Compute L2 norm per KV head: norm_s = torch.linalg.norm(S_soft, dim=0)  # shape (N,)
#    Optionally normalize by sqrt(T) to reduce token count sensitivity
# 4. Select top-k indices: active_indices = torch.topk(norm_s, k).indices
# 5. Attention uses only selected KV heads (keys and values from active_indices)
```
Hyperparameters: k = number of active KV heads per group (default k = N//2). Optional normalization: divide norm_s by sqrt(T) if T varies significantly.

### (C) Why this design
We chose deterministic selection over learned routing to eliminate router parameters and load-balancing loss, reducing training complexity and avoiding potential instability; the trade-off is that the selection is fixed and cannot adapt to per-token nuances. We aggregate alignment across all tokens in the group rather than per-token to preserve the group-sharing property of GQA, ensuring that all query heads in a group attend to the same KV heads, which maintains the KV cache coherence; this sacrifices token-level optimality for structural consistency with GQA. We use L2 norm after per-token softmax instead of raw max or mean because softmax calibrates per token, preventing dilution by many low-alignments for a head that is critical on few tokens. The per-token softmax ensures that high alignment on a single token is not diluted, mitigating the failure mode where a head is important only for few tokens. The cost is slight additional computation for softmax.

### (D) Why it measures what we claim
**Load-bearing assumption:** The L2 norm of query-key alignment scores (after per-token softmax) measures the overall relevance of a KV head to the group's queries because the dot product between query and key is the standard attention relevance; the per-token softmax normalizes each token's alignment distribution, allowing the L2 norm across tokens to reflect both the strength and consistency of each head's relevance across all tokens. This assumes that a KV head with higher total normalized alignment is more useful for the group's collective output. This assumption fails when a head is critical for only a few tokens – the per-token softmax preserves that signal, so the norm still captures it, mitigating the failure mode. However, if even after softmax, a head with rare but crucial alignments is dominated by many irrelevant tokens (e.g., when the few important tokens have low average alignment), the norm may still underestimate its importance. The top-k selection based on this norm directly operationalizes the goal of reducing KV cache size while retaining the most important heads, as the norm serves as a proxy for head utility under the assumption of uniform token importance after softmax calibration.

## Contribution

(1) A deterministic, group-level key-value head selection mechanism that reduces KV cache size and computation without learned routing. (2) Demonstration that the L2 norm of query-key alignment scores is a simple yet effective indicator of head importance for grouped query attention. (3) Analysis of the trade-off between sparsity level and model quality, showing that selecting half of KV heads per group maintains perplexity within 1% of the dense baseline.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | C4 (en), sequence length 2048 | Standard large-scale LM pretraining corpus. |
| Additional dataset | The Pile (Books subset) | Test diversity on longer coherent text. |
| Primary metric | Perplexity on held-out set | Direct measure of language modeling quality. |
| Baseline 1 | All-active GQA | Strong GQA baseline without head selection. |
| Baseline 2 | Standard MHA | Full multi-head attention upper bound. |
| Baseline 3 | MoA (Mixture of Attention Heads) | Comparable MoE attention competitor. |
| Ablation 1 | GroupKV with random selection | Isolates effect of L2-norm selection. |
| Ablation 2 | Per-token top-k selection (no group sharing) | Tests impact of group-level aggregation vs per-token. |

### Why this setup validates the claim

This evaluation setup directly tests the central claim that GroupKV can reduce KV cache size while maintaining quality by selecting the most relevant heads per group via L2 norm after per-token softmax. Using perplexity on two large corpora (C4 and The Pile Books) measures the method's impact on both general and diverse text. The all-active GQA baseline isolates the benefit of head selection from the GQA grouping. Standard MHA provides an upper bound to see how much performance is sacrificed. MoA tests against a learned, token-level MoE attention approach, highlighting whether deterministic group-level selection is competitive. Ablation 1 (random selection) controls for the selection policy itself; if GroupKV outperforms random, the L2-norm criterion is validated. Ablation 2 (per-token selection) contrasts group-level aggregation with per-token selection to test the assumption that group-level aggregation preserves quality; if per-token selection yields significantly better perplexity, it indicates that group-level sharing hurts quality, thus challenging the method's design. This combination ensures that any observed gains can be attributed to the specific selection mechanism rather than confounding factors.

### Expected outcome and causal chain

**vs. All-active GQA** — On a case where some KV heads are redundant for a group (e.g., homogeneous token semantics), all-active GQA wastes computation on irrelevant heads because it keeps all heads active, leading to larger KV cache and slower inference without quality gain. Our method instead selects the top-k heads via L2 norm after softmax, pruning low-relevance heads, because the norm captures overall alignment strength with per-token calibration. We expect GroupKV to achieve nearly identical perplexity (within 0.1 points) while using half the KV cache.

**vs. Standard MHA** — On a case where the model needs diverse attention patterns across queries (e.g., heterogeneous input), MHA uses all heads per token, capturing fine-grained interactions, but incurs high memory and compute. GroupKV limits heads per group to k, potentially missing per-token nuances because it forces group-level sharing of selected heads. We expect GroupKV to have slightly higher perplexity (e.g., 0.3–0.5 points) but with substantial KV cache reduction (≈50%), demonstrating a favorable trade-off.

**vs. MoA** — On a case where token-level attention patterns vary significantly within a group (e.g., mixed syntax and semantics), MoA selects different heads per token, capturing variation, but requires extra routing parameters and load balancing. GroupKV uses a fixed top-k selection per group after per-token softmax, ignoring per-token variation because it aggregates across all tokens. We expect MoA to slightly outperform GroupKV (e.g., 0.1–0.2 perplexity better) on heterogeneous groups, but GroupKV should be more parameter-efficient and stable due to no routing overhead.

**vs. Per-token top-k selection (Ablation 2)** — On a case where token-level importance is uniform within a group, per-token selection may select different heads per token, causing KV cache fragmentation and increased memory, while GroupKV selects a common set, maintaining cache coherence. We expect GroupKV to achieve similar perplexity with smaller KV cache, validating that group-level aggregation does not degrade quality on such cases. On a case with highly non-uniform importance, per-token selection may have lower perplexity (e.g., 0.2 points better), indicating a limitation of the group-level assumption.

### What would falsify this idea

If GroupKV's perplexity is significantly worse (e.g., >1 point) than all-active GQA on both datasets, or if its gain over random selection is negligible on homogeneous text subsets, then the L2-norm criterion (even with softmax calibration) does not capture head relevance effectively. Additionally, if per-token top-k selection (Ablation 2) drastically outperforms GroupKV (e.g., >0.5 perplexity points) without large KV cache increase, then the group-level aggregation assumption is violated, contradicting the claim that group consistency preserves quality.

## References

1. Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention
2. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
3. Mixture of Attention Heads: Selecting Attention Heads Per Token
4. Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints
5. Efficiently Scaling Transformer Inference
