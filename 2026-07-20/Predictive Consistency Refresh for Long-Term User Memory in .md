# Predictive Consistency Refresh for Long-Term User Memory in LLM-Based Recommenders

## Motivation

The Memory Hub in RecGPT-V3 stores condensed user representations to avoid full-history reprocessing, but over long-term usage, memory fidelity degrades due to irrelevant accumulation and behavior shifts. Without a principled mechanism to decide when and how to refresh, the system either reprocesses too often (losing efficiency) or not often enough (losing accuracy). This scalability-fidelity tension is the key bottleneck for persistent memory in production recommenders.

## Key Insight

The divergence between a memory's predicted user embedding and the actual observed embedding—computed from a small recent-window buffer—is a sufficient statistic for memory staleness, enabling selective refresh without full-history encoding.

## Method

## (A) What It Is
**Predictive Consistency Refresh (PCR)** is a hybrid episodic-semantic memory system that maintains an online predictor to gauge memory freshness. Its input is a stream of user interactions; its output is an up-to-date semantic memory vector that can be reused across sessions without storing all history.

**Load-bearing assumption:** The lightweight predictor MLP produces embeddings that are sufficiently correlated with observed embeddings so that cosine distance reliably indicates memory staleness. We verify this assumption by measuring the Pearson correlation between predicted and observed embeddings on a held-out calibration set of 512 interactions from the same domain; if the correlation > 0.5, the predictor is deemed adequate.

## (B) How It Works
```pseudocode
Algorithm: PCR
Input: interaction stream (x_1, x_2, ...), window size K=50, initial threshold τ_0=0.2
Parameters: g (2-layer MLP, hidden=256, ReLU activation), f (frozen 6-layer Transformer, hidden=512, pre-trained on RecGPT-V3), α=0.9 (forgetting factor for threshold update)

Initialize semantic memory s_0 ← f(empty buffer)
For each new interaction x_t:
  1. Append x_t to episodic buffer B (if |B| > K, pop oldest)
  2. Compute predicted embedding: p_t ← g(s_{t-1})
  3. Compute observed embedding: o_t ← mean_pool(B)   // average of item embeddings in buffer
  4. Compute consistency score: c_t ← cosine_distance(p_t, o_t) = 1 - cosine_similarity(p_t, o_t)
  5. If c_t > threshold τ:
       - Update semantic memory: s_t ← f(B)
       - Reset threshold: τ ← α * τ + (1-α) * c_t
     Else:
       - Keep s_t ← s_{t-1}
       - Decay threshold: τ ← α * τ
  6. Return s_t for downstream recommendation
```

## (C) Why This Design
We chose a lightweight predictor g (a 2-layer MLP, hidden=256, ReLU) instead of a full neural network to avoid compounding inference cost, accepting that the predictor may be less accurate than a larger model but still sufficient for detecting large divergences. We used cosine distance over KL divergence because (i) it is bounded and easier to threshold, and (ii) it measures angular mismatch which aligns with semantic drift in embedding spaces; the trade-off is sensitivity to magnitude (e.g., concept scaling) that KL could capture, but we mitigate this by normalizing embeddings. We used an adaptive threshold with a forgetting factor α rather than a fixed value because user behavior volatility changes over time: α=0.9 gives slow adaptation to trend shifts but allows rapid tightening after unnecessary refreshes; the cost is that α itself must be tuned per-domain. We re-encode only the episodic buffer (window K=50) instead of the entire past to bound computation, gambling that the window captures sufficient signal for meaningful refresh—this can miss long-term trends that a larger window would capture, but the consistency signal already indicates when the window is insufficient. We deliberately freeze f (the semantic encoder, a 6-layer Transformer with hidden=512 pretrained on RecGPT-V3) after pretraining to avoid catastrophic forgetting in the encoder itself, accepting that f may not adapt to new item semantics over time; in practice, the encoder can be updated periodically in a background job.

## (D) Why It Measures What We Claim
The computational quantity `cosine_distance(p_t, o_t)` measures **memory fidelity** because we assume that a faithful semantic memory s should predict the aggregated user behavior in the recent window; when the memory is stale, the predicted embedding deviates from the observed one. This assumption fails when the predictor g is systematically biased (e.g., always outputting a neutral embedding) — in that case, a low distance could indicate a lazy predictor rather than high fidelity. The quantity `mean_pool(B)` measures **current user behavior** under the assumption that the episodic buffer is a uniformly weighted sample of recent interests; if the user's behavior is bursty (e.g., one anomalous click), the mean may over-represent noise, causing false low consistency. The threshold adaptation via α ensures that the refresh trigger is **self-calibrating** to typical variability, but this holds only if the forgetting factor is set appropriately for the user's volatility; too slow adaptation causes unnecessary refreshes on stable users. The re-encoding operation `f(B)` operationally defines the **refresh** as replacing the semantic memory with a representation of the recent window, which concretely resets the memory to current observations — this is valid under the assumption that the window contains sufficient signal to capture the user's true interests; if the window is dominated by noise, the refresh could introduce a worse memory. Together, these components form a causal chain: consistency score triggers refresh, refresh updates memory, updated memory should reduce future consistency scores, thereby keeping fidelity high with minimal computation. The failure mode for the whole system is when the episodic window is statistically unrepresentative (e.g., all interactions are noise) — then consistency scores are meaningless and refreshes may oscillate; we assume this is rare in production settings with continuous interaction streams.

## Contribution

(1) A novel hybrid episodic-semantic memory architecture for LLM-based recommenders that uses a predictive consistency signal to decide when to refresh the semantic memory, avoiding full-history reprocessing. (2) A design principle showing that a lightweight predictor paired with a small recent-window buffer is sufficient for detecting memory staleness, backed by a theoretical argument linking cosine divergence to memory fidelity under the assumption of a well-calibrated predictor. (3) An adaptive threshold mechanism that self-tunes to user volatility, reducing manual tuning overhead.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MovieLens-20M (timestamped) | Sequential interactions for memory drift |
| Primary metric | NDCG@10 | Measures top-K recommendation accuracy |
| Baseline 1 | PopRec | Simple non-personalized, tests memory necessity |
| Baseline 2 | GRU4Rec | Strong sequential baseline, tests benefit of episodic memory |
| Baseline 3 | Frozen semantic memory (no refresh) | Direct ablation of refresh mechanism |
| Baseline 4 | Periodic refresh (refresh every 20 steps) | Tests if adaptive timing matters |
| Ablation 1 (ours) | PCR with fixed threshold | Tests importance of adaptive threshold |
| Ablation 2 (ours) | PCR with BERT encoder (frozen, base) | Tests model-agnostic generality |
| Ablation 3 (ours) | PCR with GPT-2 encoder (frozen, small) | Tests model-agnostic generality |

### Why this setup validates the claim

MovieLens-20M provides long user interaction sequences where semantic drift naturally occurs, making it ideal for testing memory freshness. NDCG@10 directly measures recommendation quality under drift. PopRec tests whether personalization is needed at all; GRU4Rec tests whether a pure episodic model suffices; Frozen semantic memory isolates the refresh effect. The periodic refresh baseline (every 20 steps) ensures that any gains from PCR are due to adaptive timing rather than mere periodic updates. The encoder ablations (BERT, GPT) show that PCR is not tied to the specific Transformer from RecGPT-V3, broadening the applicability. The ablation with fixed threshold determines whether adaptive thresholding is crucial for handling varying drift rates. This combination forms a falsifiable test: if the method works, it should outperform GRU4Rec on users with sudden interest shifts and beat the frozen baseline overall, while the fixed-threshold ablation should perform worse on high-volatility segments.

### Expected outcome and causal chain

**vs. PopRec** — On a user who suddenly switches from Action to Drama movies, PopRec continues recommending popular Action titles because it ignores personal history. Our method detects the shift via consistency drop and refreshes memory to Drama embeddings, causing recommendations to adapt. We expect PCR to achieve significantly higher NDCG@10 for such users (e.g., +0.05 relative) while matching PopRec on perfectly stable users.

**vs. GRU4Rec** — On a user with a long history but a temporary anomaly (e.g., one-click to a docu on whales), GRU4Rec may overfit the anomaly due to its fixed-length window, recommending similar items. Our method uses the consistency check to avoid refresh when the anomaly is dampened by the mean pool, thus preserving earlier interests. We expect PCR to outperform GRU4Rec on users with noisy sequences (e.g., +0.02 NDCG) but parity on clean sequences.

**vs. Frozen semantic memory** — On a user whose interests gradually shift from romance to thriller over 100 interactions, the frozen memory remains stuck on romance while our method refreshes periodically. The frozen baseline will have declining NDCG over time; PCR maintains stable accuracy. We expect PCR to show a noticeable gap in later timestamps (NDCG >0.05 higher) and similar performance on static users.

**vs. Periodic refresh (every 20 steps)** — On a user with highly variable drift, fixed-period refresh may refresh too often (wasting computation) or not often enough (missing shifts). PCR with adaptive threshold should match or exceed periodic refresh while using fewer refreshes. On stable users, periodic refresh may cause unnecessary refreshes, degrading accuracy; PCR avoids this. We expect PCR to have higher NDCG and lower refresh rate on stable and medium-volatility users.

**vs. PCR with BERT/GPT encoders** — If PCR's effectiveness depends on the specific RecGPT-V3 encoder, then replacing it with BERT or GPT should degrade performance. We expect PCR to maintain its advantage over frozen memory and periodic refresh across encoders, demonstrating model-agnostic generality.

### What would falsify this idea

If PCR's NDCG@10 improvement over the frozen baseline is uniform across all users (rather than concentrated on those with detected drift), or if the fixed-threshold ablation matches adaptive threshold on high-volatility users, then the central claim that adaptive refresh driven by consistency is necessary would be falsified. Additionally, if PCR with periodic refresh matches or beats PCR on most users, then the adaptive timing is not the source of improvement.

## References

1. RecGPT-V3 Technical Report
2. OpenOneRec Technical Report
3. FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets
4. Qwen3 Technical Report
5. RecGPT Technical Report
6. Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation
7. Item-Language Model for Conversational Recommendation
8. Learnable Item Tokenization for Generative Recommendation
