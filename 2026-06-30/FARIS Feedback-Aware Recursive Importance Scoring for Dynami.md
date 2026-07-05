# FARIS: Feedback-Aware Recursive Importance Scoring for Dynamic KV Cache Compression

## Motivation

ReFreeKV achieves threshold-free dynamic KV cache allocation but relies on static attention-based importance scores computed from early layers. As the cache evolves during generation, these scores become stale because a token's importance for future tokens depends on later-layer attention patterns that are not reflected in the initial scoring. This structural mismatch—static scoring versus dynamic cache state—is the fundamental bottleneck limiting compression accuracy in adaptive methods.

## Key Insight

Later-layer query-to-key attention provides a sufficient statistic for the causal downstream influence of a key-value pair, because tokens that are consistently attended by high-level queries are exactly those that propagate information through the network to future tokens.

## Method

FARIS is a lightweight feedback module that updates per-token importance scores during KV cache compression using attention signals from a small set of later layers. Its inputs are the current KV cache and its current importance scores; its outputs are updated scores that reflect the evolving cache state.

**Load-Bearing Assumption**: Attention from the last 2 layers (specifically, the query-to-key attention of the current token at those layers) provides a sufficient and reliable signal for a token's causal downstream importance. This assumption may be violated if later layers exhibit attention sinks (Xiao et al., 2024) or positional biases; we verify the assumption via a calibration check described in (C).

### (A) What it is
FARIS is a lightweight feedback module that updates per-token importance scores during KV cache compression using attention signals from a small set of later layers. Its inputs are the current KV cache and its current importance scores; its outputs are updated scores that reflect the evolving cache state.

### (B) How it works
```pseudocode
Input: KV cache with tokens, initial importance scores S (from ReFreeKV or uniform), 
       attention probe layers L_probe = last 2 layers (e.g., layers 30 and 31 for 32-layer LLaMA), 
       mixing weight alpha=0.3, top-k fraction k=0.1 (fraction of tokens to update)
Process at each cache update (after appending new tokens):
1. For each probe layer l in L_probe:
   a. Use the query vectors of the current (last) token at layer l to compute attention
      over all key positions in the cache: attn = softmax(q_l @ K^T)
   b. Store attn as the reflection score for each token from layer l.
2. Aggregate feedback: for each token i, compute max attention across all probe layers:
      fb_i = max_{l in L_probe} attn_l[i]
3. Compute update mask: select tokens with current score below median (top-k * 0.5) but 
   high feedback (above median of feedback scores). This targets tokens whose importance 
   may be underestimated.
4. Update scores: S_new[i] = alpha * S[i] + (1 - alpha) * fb_i for mask-selected tokens;
   others unchanged.
5. Normalize scores to sum to cache budget.
```
Hyperparameters: L_probe = last 2 layers (e.g., layers 30 and 31 for a 32-layer model); alpha = 0.3; top-k fraction = 0.1.

### (C) Why this design
We chose to sample probe layers from the final few layers rather than all layers to minimize computational overhead—full-layer feedback would require O(L) attention computations per update—accepting that we may miss early-layer signals that could also be relevant. We selected the maximum feedback across probe layers instead of the mean because maximum captures the strongest signal, which is more robust to noisy attention in individual layers; this comes at the cost of sensitivity to outlier spikes. We update only a sparse set of tokens (those with low initial score but high feedback) rather than all tokens to reduce disruption and maintain stability, avoiding the overhead of full cache re-scoring while risking that some truly important tokens are missed. Finally, we use a fixed mixing weight alpha rather than learning it, favoring simplicity and reproducibility over potential adaptation to different models or tasks.

**Calibration verification**: To ensure the load-bearing assumption holds for the target model, we run a calibration step on a held-out set of 100 sequences from the training set of LongBench (or a similar corpus). For each token, we compute (1) the feedback score max_{l in L_probe} attn_l[i] and (2) an oracle importance score obtained by running a full model pass with a single token ablated and measuring the change in loss. We compute the Spearman rank correlation between feedback and oracle importance. If the correlation is below 0.5, we fall back to using all layers (L_probe = all layers) for feedback, effectively adopting the smallest repair from the adversarial alert. This calibration adds negligible overhead as it is done once before deployment.

### (D) Why it measures what we claim
The computational quantity max_{l in L_probe} attn_l[i] measures the concept of causal downstream importance because it quantifies how much a token's key-value pair is attended to by later-layer queries that directly contribute to the output distribution; the assumption is that later-layer attention divergence correlates with the token's influence on future logits. This assumption fails when later layers attend to tokens based on positional regularity rather than semantic content (e.g., attention sinks), in which case the feedback reflects salience rather than causal necessity. The mixing step with initial score S[i] operationalizes the concept of correcting stale estimates: S[i] captures the importance at the time of insertion, and the weighted average combines the old and new evidence; the assumption that initial scores remain partially informative fails when the cache has undergone many updates, and then the feedback dominates. The update mask (low initial score + high feedback) operationalizes the concept of detection of undervalued tokens: selecting tokens where the discrepancy is large ensures we correct the most likely misclassifications; the assumption that discrepancy indicates error fails when both signals are low (token is truly unimportant) or both are high (already correctly scored).

## Contribution

(1) A feedback-aware importance scoring mechanism (FARIS) that dynamically updates token importance using later-layer attention, correcting stale scores from static methods like ReFreeKV. (2) A design principle showing that cross-layer attention consistency—specifically maximum attention from later layers—is a reliable proxy for a token's causal future influence in the KV cache. (3) An efficient sparse update strategy that mixes old and new scores with minimal overhead, enabling integration into existing threshold-free compression pipelines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | LongBench (including subsets with sequences >32k tokens) | Diverse long-context tasks; tests scalability. |
| Primary metric | Accuracy | Measures correctness under compression. |
| Baseline 1 | Full KV cache | Upper bound without compression. |
| Baseline 2 | H2O | Threshold-based pruning baseline. |
| Baseline 3 | SnapKV | Recent compression method. |
| Ablation of ours | FARIS w/o feedback | Isolates effect of feedback module. |

### Why this setup validates the claim

This experimental design creates a falsifiable test of FARIS's central claim: that per-token importance scores can be corrected using attention signals from later layers, improving compression quality. LongBench provides diverse long-context scenarios where attention patterns vary, enabling detection of where feedback helps or fails. Including subsets with sequences >32k tokens specifically tests scalability and the effectiveness of feedback over very long contexts. Comparing against full cache establishes an upper bound; H2O and SnapKV represent two distinct families of compression methods — threshold-based and observation-based — testing different failure modes. The ablation removes feedback to isolate its contribution. Accuracy is chosen because it directly reflects whether the compressed cache retains task-relevant information; if feedback corrects underestimates, accuracy should improve specifically on examples where initial scores are poor. A uniform improvement across all cases would not support the claim, which predicts gains concentrated where feedback identifies undervalued tokens.

### Expected outcome and causal chain

**vs. Full KV cache** — On a case where important tokens get evicted early due to initial score errors (e.g., a crucial entity in early context), the full cache performs perfectly because it retains all tokens. Our method corrects the score for that token via high feedback from later layers, preventing its eviction. We expect FARIS accuracy to approach full cache within 2-3% on most tasks, with larger gaps only on tasks requiring extremely precise information recall (e.g., needle-in-a-haystack).

**vs. H2O** — On a case where H2O's fixed threshold incorrectly prunes tokens that appear sporadically (e.g., a critical number mentioned once in a long document), H2O removes them because their cumulative attention score is below threshold. Our method catches such tokens because the later-layer feedback gives them a temporary importance spike. We expect a noticeable gap on subsets with sparse critical tokens (e.g., 5-10% accuracy improvement), but similar performance on dense coherent text.

**vs. SnapKV** — On a case where SnapKV's observation window (e.g., last few tokens) misses a long-range dependency (e.g., a reference to early context repeated later), SnapKV may retain only the later mention and prune the earlier one. Our method's feedback from later layers attends to both copies, preserving the earlier token because its mask selects it. We expect FARIS to outperform SnapKV on long-range dependency tasks (e.g., 3-5% better on narrative understanding), with parity on shorter contexts.

### What would falsify this idea

If FARIS shows equal or worse accuracy than the ablation (no feedback) across all LongBench tasks, the feedback module fails to improve importance scoring. If gains are uniform across subsets rather than concentrated on cases with sparse critical tokens (vs. H2O) or long-range dependencies (vs. SnapKV), the claim that feedback corrects specific underestimates is not supported.

## References

1. ReFreeKV: Towards Threshold-Free KV Cache Compression
2. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
3. SnapKV: LLM Knows What You are Looking for Before Generation
4. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
5. Lost in the Middle: How Language Models Use Long Contexts
