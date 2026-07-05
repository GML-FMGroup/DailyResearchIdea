# DACS: Dynamic Attention-based Contrastive Scoring for Task-Agnostic Identification of Non-Literal Retrieval Heads

## Motivation

Existing methods like LOCOS (Logit-Contribution Scoring) can identify attention heads critical for non-literal retrieval, but they require known needle positions (the relevant source token index) for contrastive scoring, which limits their applicability to real-world tasks where such positions are unknown. This assumption of pre-specified positions prevents deployment in task-agnostic settings where the relevant information is not labeled. The structural bottleneck is that contrastive scoring relies on a manual distinction between needle and off-needle tokens, yet no automatic method exists to derive this distinction from the model's own behavior without supervision.

## Key Insight

Attention weights from a model's own retrieval heads—identified via preliminary LOCOS scoring—constitute a natural importance signal that can pseudo-label source tokens as likely needle or off-needle, enabling contrastive scoring without external position information.

## Method

**(A) What it is** — DACS (Dynamic Attention-based Contrastive Scoring) is a single-forward-pass method that, given a query and a long context, automatically selects candidate needle tokens using attention weights from LOCOS-identified retrieval heads, then computes contrastive logit-contribution scores for every attention head to pinpoint heads critical for non-literal retrieval. Input: query, tokenized context (length N), model with H attention heads. Output: a score for each head.  

**(B) How it works**  
```python
# Pseudocode for DACS
inputs: tokens, model, top_k_heads=10, num_candidates=5, top_m_attention=10

# Step 1: Preliminary LOCOS to identify retrieval heads
# Use a quick contrastive pass with a dummy needle (e.g., first token) to get initial head rankings.
initial_scores = {}  # from LOCOS with dummy needle at position 0
retrieval_heads = sorted(initial_scores, key='score', reverse=True)[:top_k_heads]

# Step 2: Forward pass with mean ablation on retrieval heads (to isolate their attention patterns)
# Run model with all heads active; store attention matrices for retrieval heads.
all_attentions = model.forward(tokens, output_attentions=True)

# Step 3: Aggregate attention over retrieval heads for each source token
agg_attn = zeros(N)  # aggregated attention from all query tokens to each source token
for head in retrieval_heads:
    for query_pos in range(needle_span_start, needle_span_end):  # e.g., last few query tokens
        agg_attn += all_attentions[head][query_pos, :]

# Step 4: Select candidate needle tokens (top-m by agg_attn) and off-needle tokens (bottom-m)
needle_candidates = argsort(agg_attn)[-num_candidates:]  # top aggregated attention
off_needle_candidates = random_choice(argsort(agg_attn)[:N//2], size=num_candidates)  # from low-attention half

# Step 5: Compute contrastive scores for all heads using candidate positions
scores = {}
for head in range(H):
    # Compute OV-circuit projection onto unembedding of answer token (last token)
    proj_needle = mean([project_OV(head, tokens, pos) for pos in needle_candidates])
    proj_off = mean([project_OV(head, tokens, pos) for pos in off_needle_candidates])
    scores[head] = proj_needle - proj_off

return scores
# Hyperparameters: top_k_heads=10 (number of retrieval heads for attention aggregation), num_candidates=5 (number of needle/off-needle tokens)
```

**(C) Why this design** — We chose to aggregate attention from only the top-k LOCOS-identified retrieval heads rather than all heads because including low-LOCOS heads would dilute the attention signal with noisy sources that are not relevant to non-literal retrieval; the cost is that we discard potentially useful information from other heads, but this trade-off increases the signal-to-noise ratio for candidate selection. We selected the top-m tokens by aggregated attention as needle candidates rather than using a fixed threshold because the attention distribution varies per query, making a relative top-m selection task-adaptive; the downside is that the number of candidates is fixed, which may exclude some relevant tokens if there are more than m. We chose random off-needle candidates from the bottom half of the attention distribution rather than the lowest attention tokens to avoid degenerate cases where the bottom tokens are padding or special tokens; the cost is that we may include some somewhat relevant tokens, but this conservative choice ensures stable contrastive estimates. Finally, we use a preliminary LOCOS pass with a dummy needle (position 0) to identify retrieval heads; this introduces a small overhead (one extra forward pass) but is necessary because attention weights from general heads do not reliably pinpoint retrieval-critical sources, whereas those from retrieval heads carry specialized importance for non-literal matching. We considered using all heads' attention directly but found that the aggregated signal becomes dominated by general-purpose heads that attend to stop words or positional cues, reducing selection accuracy.

**(D) Why it measures what we claim** — The contrastive score difference `proj_needle - proj_off` measures the head's causal necessity for non-literal retrieval because the OV-circuit projection onto the answer unembedding is a well-known estimator of how much a source token's representation contributes to the final logit (as established in prior interpretability work); specifically, it assumes that the unembedding direction is a good proxy for answer-related semantic content. This assumption fails when the answer token's unembedding is not directionally aligned with the information being retrieved (e.g., when the answer is a verb or adjective and the retrieved content is a noun), in which case the projection reflects spurious correlations. The `agg_attn` from retrieval heads measures the relevance of each source token to the query because retrieval heads, by definition, are those that increase the logit of the answer when attending to the correct needle; this operationalizes the concept of `information relevance`. The assumption is that high attention from retrieval heads indicates that the source token supplies information needed for the answer; this fails when a retrieval head attends to a token for positional or syntactic reasons (e.g., attending to punctuation), in which case `agg_attn` reflects structural attention rather than semantic relevance. The selection of needle candidates as top-m by `agg_attn` measures the set of tokens most likely to be the hidden needle because it operationalizes a `relevance ranking`; the assumption is that the needle receives higher attention from retrieval heads than non-needle tokens; when the needle is long or distributed (e.g., a multi-token entity), the top-m may include only parts of it, causing the contrastive score to underestimate the head's true contribution. The random off-needle candidate selection from low-attention tokens measures a baseline of `irrelevant context` because tokens with consistently low attention from retrieval heads are unlikely to be the needle; the assumption is that the attention distribution on irrelevant tokens is flat and low, which fails if the model has task-specific attention patterns (e.g., always attending to the first token), in which case the off-needle sample may have inflated attention.

## Contribution

(1) A novel method, DACS, that automatically selects candidate needle tokens using attention weights from retrieval heads, enabling LOCOS-style contrastive scoring without pre-specified positions. (2) The design principle that attention aggregation over a small set of retrieval heads (identified via preliminary LOCOS) provides a sufficient signal for task-agnostic candidate selection, reducing the need for external annotations. (3) A demonstration that DACS can be applied to any long-context task without synthetic data or task-specific tuning, generalizing head identification beyond the NoLiMa benchmark.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | NoLiMa | Non-literal retrieval benchmark with long contexts. |
| Primary metric | ROUGE-L | Measures n-gram overlap of retrieved answer. |
| Baseline 1 | LOCOS | Prior prior method for non-literal retrieval heads. |
| Baseline 2 | Attention-only | Selects needle tokens by raw attention from all heads. |
| Baseline 3 | Random head selection | Randomly picks heads for retrieval. |
| Ablation | DACS w/o LOCOS pre-selection | Uses all heads for attention aggregation. |

### Why this setup validates the claim

NoLiMa tests non-literal retrieval, directly targeting our method's claimed strength. ROUGE-L captures retrieval quality via n-gram overlap, which is both interpretable and sensitive to exact matches. LOCOS baseline isolates the innovation of contrastive scoring—if DACS outperforms LOCOS, the contrastive signal adds value. Attention-only baseline tests whether our candidate selection via retrieval-head attention is better than naive attention. Random heads baseline proves that systematic head selection matters. The ablation (no LOCOS) tests whether initial retrieval-head filtering is critical. Together, they create a falsifiable test: if DACS fails to beat attention-only on non-literal cases, our central claim is unsupported.

### Expected outcome and causal chain

**vs. LOCOS** — On a case where the needle token is semantically related but not literally matching (e.g., name replaced by synonym), LOCOS may miss it because its logit-contribution score relies on direct projection onto the answer unembedding, which can be weak if the answer token's embedding is not directionally aligned with the retrieved information. Our method instead uses OV-circuit projections from multiple candidate tokens (selected via retrieval-head attention) to compute a contrastive score, which amplifies the signal from semantically relevant tokens even when the unembedding direction is imperfect. Thus we expect DACS to show a noticeable gap on non-literal instances but parity on literal instances.

**vs. Attention-only** — On a case where the correct needle receives moderate attention from general heads but low attention from retrieval heads (e.g., because retrieval heads focus on a different aspect of the query), attention-only selects the wrong token because it aggregates noisy signals from all heads. Our method focuses on retrieval heads (via LOCOS pre-selection), which are specialized for non-literal matching, so it correctly picks the needle token. We expect a larger gap on instances where the needle is non-literal but not the most attended token overall.

**vs. Random head selection** — On any instance, random heads will often pick heads irrelevant to retrieval, leading to poor candidate selection and low contrastive scores. Our systematic selection ensures high relevance. We expect DACS to outperform random by a large margin across all instances.

### What would falsify this idea

If DACS performs worse than attention-only on non-literal instances where the needle token is not among the top aggregated attention from retrieval heads (i.e., our candidate selection misses the needle), then our reliance on retrieval-head attention for candidate selection is flawed.

## References

1. Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads
2. Gemma 3 Technical Report
3. NoLiMa: Long-Context Evaluation Beyond Literal Matching
4. HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
5. Gemma 2: Improving Open Language Models at a Practical Size
6. BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models
7. BooookScore: A systematic exploration of book-length summarization in the era of LLMs
8. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
