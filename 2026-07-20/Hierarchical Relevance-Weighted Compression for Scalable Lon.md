# Hierarchical Relevance-Weighted Compression for Scalable Long-User-History Memory in LLM Recommenders

## Motivation

RecGPT-V3's Memory Hub assumes user history can be condensed into a fixed-size memory without loss, but over extremely long time horizons (spanning years), this bounded representation degrades predictive accuracy because it cannot distinguish between ephemeral and persistent patterns. This structural limitation recurs across multiple works that rely on fixed-capacity memory, from RecGPT-V3 to Transformer-based sequential recommenders, causing systematic underperformance on users with extensive histories.

## Key Insight

By organizing memory as a hierarchy of compressed representations where each level captures a different temporal scale and a learnable relevance weight controls retention, we decouple memory capacity from history length, allowing the model to preserve both recent nuances and long-term patterns without saturating a fixed budget.

## Method

### (A) What it is
**Hierarchical Relevance-Weighted Compression (HRWC)** is a memory mechanism that takes an arbitrarily long user interaction history and produces a compact hierarchical representation. It inputs a sequence of timestamped item embeddings and outputs a multi-scale memory tensor used by a downstream LLM recommender for next-item prediction.

### (B) How it works
```pseudocode
Input: user history H = [(i_1, t_1), ..., (i_n, t_n)] where n can be up to 10^5.
Hyperparameters: segment sizes S = [s1=10, s2=100, s3=1000], compression dimensions D = [d1=128, d2=64, d3=32], decay base alpha=0.1

1. Partition H into segments per scale:
   For scale=1 (finest): split H into chunks of size s1 (e.g., 10 interactions each).
   For scale=2: split H into chunks of size s2 (e.g., 100 interactions).
   For scale=3 (coarsest): split H into chunks of size s3 (e.g., 1000 interactions).

2. For each segment at each scale:
   a. Encode segment via a lightweight Transformer (2 layers, 4 heads) to get segment embedding e_seg.
   b. Compute relevance weight w = sigma(MLP([e_seg; u_state])), where u_state is the current user embedding (averaged over recent 10 interactions).
   c. Compress e_seg using a learned linear projection with dimension d_s: h_seg = W_s @ e_seg + b_s.
   d. Multiply h_seg by w: h_seg_weighted = w * h_seg.
   e. Apply learnable decay: d_seg = exp(-alpha * (t_now - t_avg_seg)) where t_avg_seg is the mean timestamp of the segment. Then h_seg_final = d_seg * h_seg_weighted.

3. Build hierarchical memory:
   For each scale, stack all h_seg_final for that scale into a matrix M_s of shape (number_of_segments_at_scale, d_s).
   Optionally, attend over M_s using u_state to produce a single vector per scale: v_s = Attend(u_state, M_s, M_s).
   Final memory = concatenation of [v_1, v_2, v_3].

4. Use final memory as additional input to the LLM recommender (e.g., prepend as a prefix sequence).
```

### (C) Why this design
We chose a multi-scale segmentation over fixed-window sliding attention because it explicitly captures patterns at different temporal granularities—micro, meso, macro—while keeping computational cost linear in history length rather than quadratic. The trade-off is that coarse segments may blur short-lived but important bursts, which we mitigate by allowing segments to overlap slightly at boundaries. We chose learnable relevance weights (via an MLP conditioned on current user state) over simple recency-based decay because recency alone cannot distinguish between a transient interest and a persistent one (e.g., a one-time holiday vs. a weekly routine); the cost is additional parameters and reliance on accurate user state. We chose linear compression (a learned projection) over a VAE because it is faster and easier to train end-to-end with the recommender, but it cannot model uncertainty about what information to retain; we accept that less important details may be discarded deterministically. Finally, we chose to keep the memory size as a function of history length (by having a fixed number of segments per scale) rather than a fixed-total-size approach because it guarantees that no information is dropped due to capacity overflow; the downside is that memory grows logarithmically with history length, but in practice for 10^5 items the total dimension (≤3000) is manageable.

### (D) Why it measures what we claim
The segment embedding e_seg measures the aggregate pattern of interactions within that time window, under the assumption that grouping interactions by temporal proximity preserves coherent behavioral motifs; this assumption fails when a user's behavior is highly erratic within a window, in which case e_seg simply averages over noisy data without capturing the variance. The relevance weight w measures the predictive utility of that segment for the current user state, under the assumption that the MLP can learn a stable mapping from segment content and state to importance; this assumption fails when the user state is poorly estimated (e.g., cold-start or infrequent user), in which case w becomes uninformative (near 0.5) and the memory becomes uniform. The compression step h_seg = W_s @ e_seg measures a lossy summary of e_seg with a fixed information bottleneck, under the assumption that the most discriminative directions in the item embedding space align with recommendation-relevant features; this assumption fails when items have multiple independent attributes relevant for different users, in which case the compression discards orthogonal but useful dimensions. The decay weight d_seg measures the time-based obsolescence of a segment, under the assumption that predictive relevance decays exponentially with age; this assumption fails when long-standing habits re-emerge after years (e.g., a user returns to a forgotten hobby), in which case decay prematurely reduces the weight of such segments. Together, these quantities operationalize the motivation-level concept of 'scalable memory with adaptive retention' by ensuring that each segment's contribution is a product of its temporal scale, age, content relevance, and compression fidelity, and the hierarchical structure guarantees that no segment is omitted even for arbitrarily long histories.

## Contribution

(1) A hierarchical memory architecture (HRWC) that compresses user history into multi-scale representations with relevance-weighted retention, scaling linearly with history length. (2) A learnable decay function that combines recency and predictive relevance to decide which past information to preserve, avoiding the fixed-capacity bottleneck of prior memory hubs. (3) The principle that memory capacity should grow with history length rather than being bounded, enabling LLM recommenders to handle arbitrarily long user histories without degradation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Amazon Books (long histories) | Rich temporal dynamics, long sequences |
| Primary metric | NDCG@10 | Captures ranking quality of top recommendations |
| Baseline 1 | Fixed-window attention (window=200) | Tests need for multi-scale over fixed window |
| Baseline 2 | Pure recency-weighted sum | Tests need for learnable relevance over decay-only |
| Baseline 3 | RecGPT (token-level history) | Strong LLM baseline without hierarchical compression |
| Ablation (ours) | HRWC w/o multi-scale (only finest) | Isolates benefit of hierarchical structure |

### Why this setup validates the claim

The combination of a long-history dataset (Amazon Books, where users often have thousands of interactions), ranking metric NDCG@10, and three baselines forms a direct falsification test of HRWC's central claim: that explicit multi-scale, relevance-weighted compression improves next-item prediction over both fixed-window attention and naive recency. The fixed-window baseline tests whether coarse scales capture long-range dependencies absent in a window. The pure recency baseline tests whether learnable relevance outweighs simple decay. RecGPT tests whether hierarchical compression beats raw token input at a comparable parameter count. The ablation isolates the contribution of the hierarchical design. NDCG@10 is sensitive to ranking improvements at the top, where HRWC's ability to retain salient long-term patterns should shine. If HRWC fails to beat these baselines on long sequences, the core idea is invalid.

### Expected outcome and causal chain

**vs. Fixed-window attention (window=200)** — On a case where a user has periodic seasonal interests (e.g., holiday decorations purchased yearly), the fixed-window model misses patterns beyond its 200-item horizon, treating each season as isolated. Our HRWC's coarse scale (s3=1000) aggregates entire year-long patterns, and the relevance weight (conditioned on current user state) amplifies the seasonal signal. We expect HRWC to outperform by a noticeable margin on users with long-term periodicity, but parity on short-session users.

**vs. Pure recency-weighted sum** — On a case where a user briefly explored a hobby (e.g., hiking gear) weeks ago but then ignored it, a recency model decays that segment too slowly if the user has recent unrelated interactions. HRWC's learnable relevance, conditioned on current user embedding (averaged over recent 10 items), downweights the outdated hobby segment because the state vector no longer matches. We expect HRWC to show significant gains on users with transient interest shifts, while both methods perform similarly on stable preference patterns.

**vs. RecGPT (token-level history)** — On a case where a user's history exceeds 10^5 items, RecGPT's token limit truncates earlier interactions, losing long-term preferences. HRWC compresses all history into hierarchical segments, preserving coarse patterns even from remote past. The compression loss may hurt fine-grained recency, but the hierarchical memory captures macro trends. We expect HRWC to outperform RecGPT on users with very long histories (>10^4) by a moderate margin, but possibly underperform on short histories where raw tokens are more expressive.

### What would falsify this idea

If HRWC shows no improvement on long-history users compared to the fixed-window baseline, or if gains are uniform across all history lengths (i.e., not concentrated where long-term patterns exist), then the multi-scale design fails to capture the claimed temporal granularities.

## References

1. RecGPT-V3 Technical Report
2. OpenOneRec Technical Report
3. FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets
4. Qwen3 Technical Report
5. RecGPT Technical Report
6. Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation
7. Item-Language Model for Conversational Recommendation
8. Learnable Item Tokenization for Generative Recommendation
