# Bounded-Attention Compression: Theoretical Error Guarantees for Dynamic KV Cache Pruning

## Motivation

Existing dynamic compression methods like ReFreeKV achieve practical efficiency but provide no formal guarantees on the error introduced by token pruning, which is critical for high-stakes applications such as medical diagnosis or autonomous driving. The root structural cause is that these methods rely on heuristics (e.g., cumulative attention thresholds) without exploiting the Lipschitz continuity of the softmax attention mechanism, which could yield input-independent error bounds tied to the mass of pruned tokens.

## Key Insight

The output of softmax attention is Lipschitz continuous with respect to its attention weight vector, so the error from pruning KV pairs is bounded by the Lipschitz constant times the total attention mass of pruned tokens, enabling provable guarantees without runtime heuristics.

## Method

**(A) What it is:**  
Bounded-Attention Compression (BAC) is a KV cache pruning method that, given a user-specified error tolerance

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongBench (long-context QA) | Tests diverse context lengths and tasks. |
| Primary metric | Task accuracy at equal memory budget | Directly measures quality under compression. |
| Baseline 1 | Full cache (no compression) | Upper bound on accuracy. |
| Baseline 2 | H2O (threshold-based) | Representative pruning with fixed threshold. |
| Baseline 3 | StreamingLLM (attention sink) | Baselines on recent+attention sink caching. |
| Ablation-of-ours | BAC with fixed threshold (no adaptive bound) | Isolates benefit of adaptive error bound. |

### Why this setup validates the claim

This experimental design creates a falsifiable test of BAC's central claim: that a user-specified error tolerance can guide KV cache compression to preserve task accuracy. By comparing against Full cache, we establish the accuracy ceiling and memory usage upper bound. H2O tests whether a fixed threshold can match the trade-off; if BAC outperforms H2O, it shows that adaptive bounding preserves more relevant context. StreamingLLM tests whether simple recent+attention sink caching suffices; BAC should beat it on tasks requiring long-range memory. The ablation (BAC without adaptive bound) isolates the mechanism: if fixed-threshold BAC performs worse, the adaptive tolerance is crucial. Accuracy at equal memory budget directly reflects how well each method respects the error constraint, making the metric diagnostic of the core idea.

### Expected outcome and causal chain

**vs. Full cache** — On a long-context QA instance with a 100K token context, Full cache requires massive memory, often causing out-of-memory or extreme latency. Our BAC compresses the cache under a user tolerance of 1% accuracy loss, retaining only critical tokens. Thus, we expect BAC to achieve near-identical accuracy (within the tolerance) while reducing memory by 80-90%, with the observable gap being a small accuracy drop (≤1%) at a fraction of memory.

**vs. H2O** — On a narrative comprehension task where the answer requires recalling a minor character mentioned in the first 10% of a long story, H2O's cumulative attention threshold may discard that early mention because its attention later is small. BAC, guided by error tolerance, recognizes that pruning that token would exceed the allowed deviation from full cache output, so it retains it. Hence, we expect BAC to show a noticeable accuracy advantage (e.g., 5-10%) on tasks with long-range dependencies, while performing similarly on short-context tasks.

**vs. StreamingLLM** — On a multi-document summarization where key information is scattered across early and middle passages, StreamingLLM only caches the latest tokens and attention sinks, losing critical early content. BAC adaptively selects tokens whose pruning would cause the least increase in error, preserving a broader set. The expected outcome is a clear accuracy gap (e.g., 10-15%) on tasks requiring sequential cross-document reference, with parity on tasks with strong recency bias.

### What would falsify this idea

If BAC's accuracy is systematically lower than H2O's on tasks with long-range dependencies, or if the empirical error consistently exceeds the user-specified tolerance (e.g., by more than 2x), then the central claim of bounded-attention compression is invalid.

## References

1. ReFreeKV: Towards Threshold-Free KV Cache Compression
2. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
3. SnapKV: LLM Knows What You are Looking for Before Generation
4. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
5. Lost in the Middle: How Language Models Use Long Contexts
