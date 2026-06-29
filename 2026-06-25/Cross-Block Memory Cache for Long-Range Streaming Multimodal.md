# Cross-Block Memory Cache for Long-Range Streaming Multimodal Foundation Models

## Motivation

Wan-Streamer's block-causal attention enables real-time streaming but loses long-range dependencies because each block only attends to past tokens within the same block and a limited cross-block context (e.g., only the previous block's final hidden state). This structural limitation prevents the model from capturing dependencies spanning many blocks, which is critical for coherent multimodal interaction such as maintaining a conversation topic over seconds. Existing methods like Qwen2.5-VL and MIDAS do not address streaming constraints simultaneously with long-range modeling.

## Key Insight

A fixed-size, incrementally updated memory cache that stores compressed representations of all previous blocks allows the current block to attend to the full sequence history without increasing per-block computation beyond O(block_size × cache_size), because cache queries are fused into the standard attention mechanism.

## Method

### (A) What it is:
Cross-Block Memory Cache (CBMC) is an attention mechanism extension that equips each transformer layer with a memory cache storing key-value pairs from previous blocks. Its inputs are the current block's queries and the cache keys/values; outputs are attention-weighted representations.

### (B) How it works:
```python
# Pseudocode for one transformer layer with CBMC
cache = {}  # dictionary mapping block index to (K_cache, V_cache)
learned_query = nn.Parameter(torch.randn(d_model))  # learnable query for attention pooling
for block_idx, block_tokens in enumerate(stream):
    # Self-attention within block (causal)
    local_attn = causal_attention(block_tokens, block_tokens)  # K,V from block
    # Cross-attention to cache: attend to summaries from previous blocks
    cache_keys, cache_values = aggregate_cache(cache, block_idx)  # e.g., concatenate all cached summaries
    cross_attn = cross_attention(block_tokens, cache_keys, cache_values)  # queries from block, K,V from cache
    combined_attn = local_attn + cross_attn  # concatenation, weighted sum (hyperparameter: λ=0.5)
    # After processing block, compute a summary via attention pooling (single-head, query=learned_query)
    attn_weights = softmax(block_tokens @ learned_query / sqrt(d_model))
    block_summary = attn_weights @ block_tokens  # (d_model,)
    # Project to key-value pair (2*d_head*num_heads) — using linear projection for each head
    K_proj = linear_projection(block_summary, out_dim=d_head*num_heads)
    V_proj = linear_projection(block_summary, out_dim=d_head*num_heads)
    cache[block_idx] = (K_proj, V_proj)
    # Optional: prune old entries to keep cache size <= C (e.g., C=64)
    if len(cache) > C:
        oldest_idx = min(cache.keys())
        del cache[oldest_idx]
```

### (C) Why this design:
We chose a fixed-size cache over storing all past token key-values to bound memory and latency, accepting that compression may lose fine-grained details. We chose attention-based pooling to compute block summaries, as it can adaptively weigh tokens within a block, preserving important information while still compressing. This adds a small parameter cost (one learnable query of size d_model) but avoids the overhead of a fully learned projection for each token. We chose to attend to all cached summaries (not just the most recent) to enable long-range dependencies, but this requires cross-attention computation O(block_size * C); to keep latency low, we set C small (e.g., 64). The λ hyperparameter balances local and global context; we initialize λ=0.5 and tune via grid search, as the optimal trade-off depends on the task (e.g., dialogue vs. video). This design builds on memory networks (e.g., Transformer-XL) but differs by operating on block-level summaries rather than token-level recurrence, which aligns with the block-causal paradigm. A domain expert would not describe this as a variant of Transformer-XL because Transformer-XL uses token-level recurrence across segments, while CBMC uses explicit block-level caching with compression and fixed-size memory, preserving the streaming property of minimal latency per block.

### (D) Why it measures what we claim:
The cache size C measures the model's capacity for long-range context because it directly controls how many past block summaries are available for cross-attention; this assumes that block summary is a sufficient representation for cross-block dependencies. This assumption fails when fine-grained token-level alignment across blocks is needed (e.g., precise object tracking across cuts), in which case the cache reflects approximate temporal context. The attention-pooling function measures the aggregate information of a block under the assumption that a single learned query can capture the block's essential content; this assumption fails for blocks containing abrupt scene changes, where the query may overemphasize a subset of tokens. To validate, we compute the probe metric 'attention weight on far blocks' (average attention weight from current block on cache entries older than 10 blocks; high values indicate successful long-range capture). The λ weight measures the relative importance of local vs. global context under the assumption that the two attention maps are complementary; this assumption fails when the cross-attention attends to irrelevant summaries (noisy cache), leading to a degraded combined representation. The cache update mechanism (projecting block summary to key-value pairs) measures the fidelity of encoding past block content under the assumption that a linear projection suffices; this assumption fails when the summary dimension is too low to capture multimodal variations (e.g., concurrent audio and visual events), causing information loss.

## Contribution

(1) We introduce Cross-Block Memory Cache (CBMC), a plug-in attention extension that augments block-causal attention with fixed-size memory for long-range dependencies in streaming multimodal models. (2) We demonstrate that block-level summary caching preserves streaming latency while improving coherence in tasks requiring cross-block context, such as maintaining conversational flow and tracking objects across video cuts. (3) We provide an open-source implementation and hyperparameter sensitivity analysis on multimodal streaming benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | LongVideoQA benchmark | Tests long-range multimodal dependencies |
| Primary metric | Accuracy on long-range questions | Captures ability to use distant context |
| Baseline1 | Vanilla Transformer (no cache) | No cross-block context, tests necessity |
| Baseline2 | Transformer-XL (token-level recurrence) | Token-level recurrence with O(L) memory |
| Baseline3 | Full token cache (unbounded memory) | Unbounded memory, tests compression trade-off |
| Ablation1 | CBMC with mean pooling | Compare summary functions |
| Ablation2 | CBMC with learned attention pooling (ours default) | Central method |
| Probe metric | Average attention weight on far blocks | Validates cache use for long-range context |

### Why this setup validates the claim
This experimental design forms a falsifiable test of CBMC's central claim—that block-level compressed caching enables effective long-range context without unbounded memory. The LongVideoQA dataset requires reasoning across video segments, demanding cross-block integration. Accuracy on long-range questions directly measures the model's ability to use distant information. The baselines isolate key sub-claims: Vanilla Transformer tests whether any cross-block memory is necessary; Transformer-XL tests whether token-level recurrence outperforms block-level compression; Full token cache tests the cost of compression. The ablations compare summary functions to justify our choice of attention pooling. The probe metric 'attention weight on far blocks' directly measures whether the model actually uses distant cache entries, providing an additional signal beyond final accuracy. If CBMC outperforms the vanilla baseline and approaches the full cache while surpassing Transformer-XL, the claim that block-level compression is sufficient for long-range context is supported. Conversely, if CBMC fails to beat the vanilla baseline or random cache matches its performance, the idea is falsified.

### Expected outcome and causal chain

**vs. Vanilla Transformer** — On a long video question where answer appears in an early block (e.g., "What color was the jacket in the first scene?"), Vanilla Transformer produces incorrect answer because its self-attention is confined within each block and cannot access earlier tokens. Our method instead retrieves the cached summary from the first block via cross-attention, so we expect a significant accuracy gap (e.g., +15-20%) on such questions, with parity on short-range questions.

**vs. Transformer-XL** — On a video with many short cuts (e.g., 100 blocks), Transformer-XL suffers from context dilution because its token-level recurrence accumulates irrelevant tokens, degrading attention precision. Our method compresses each block into a single summary, preserving key semantics without noise. We expect CBMC to outperform Transformer-XL on long videos (e.g., >50 blocks) by 5-10% accuracy, while on short videos performance is comparable.

**vs. Full token cache** — On a task requiring fine-grained token alignment (e.g., tracking a small object across cuts), Full token cache retains all past tokens and achieves high accuracy. Our method loses token-level details due to compression, leading to slight degradation. We expect CBMC to reach within 2-3% of Full token cache accuracy on most benchmarks, but to show a larger gap on tasks requiring precise cross-block fine-grained reasoning (e.g., object tracking).

**Ablation: mean vs. attention pooling** — Mean pooling dilutes important tokens; attention pooling should yield higher accuracy on long-range questions. We expect attention pooling to outperform mean pooling by 2-3% on LongVideoQA, with a U test showing significance (p<0.05).

**Probe metric** — The average attention weight on far blocks for CBMC should be significantly higher than for a random cache baseline (e.g., 0.05 vs. 0.02), confirming that the model leverages distant context.

### What would falsify this idea
If CBMC performs no better than the Vanilla Transformer (or if random cache achieves similar performance), then the central claim that compressed block caching provides useful long-range context is invalid. Also, if Transformer-XL consistently outperforms CBMC by a large margin, the idea that block-level compression is sufficient fails. If the probe metric shows no difference between CBMC and random cache, the cache mechanism is not being used effectively.

## References

1. Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models
2. Qwen2.5-VL Technical Report
3. MIDAS: Multimodal Interactive Digital-humAn Synthesis via Real-time Autoregressive Video Generation
4. Qwen2.5 Technical Report
