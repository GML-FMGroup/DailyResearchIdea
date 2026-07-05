# RANDOM-CONTRAST: Task-Agnostic Attention Head Importance via Random Pair Projection

## Motivation

LOCOS (Logit-Contribution Scoring) identifies attention heads crucial for non-literal retrieval by contrasting OV-circuit projections between known needle and off-needle source positions. However, this requires explicit specification of relevant source positions, which are unavailable in open-ended tasks or when the needle is not predefined. This limitation roots from the contrastive nature of the score that necessitates a known positive-negative pair. We address this by extending LOCOS to a task-agnostic metric that leverages random source token pairs, removing the dependency on known positions.

## Key Insight

The average absolute difference of OV-circuit projections over random token pairs converges to a head-specific quantity that is monotonic with the head's contribution to non-literal retrieval, because the variance of projections across random pairs captures the discriminative power of the head irrespective of specific source positions.

## Method

(A) **What it is**: RANDOM-CONTRAST is a scoring method that estimates the importance of each attention head for non-literal retrieval by computing the mean absolute difference of the OV-circuit output projections over many randomly sampled source token pairs, without requiring any known task-specific positions. Input: a single forward pass through the model on the input sequence. Output: an importance score per head.

(B) **How it works** (pseudocode):
```python
# Given: model with L layers, H heads per layer; input sequence of length N
# Hyperparameters: num_pairs = 500 (number of random token pairs)
# Efficient implementation: precompute projections in a single forward pass, then vectorize pair sampling

scores = zeros(L, H)
for layer in 1..L:
    for head in 1..H:
        # Precompute OV-circuit output for each token position (cached after first layer)
        projections = []  # list of projection values for each token position
        for pos in 1..N:
            h = model.get_ov_output(layer, head, pos)  # OV-circuit output at this position
            v = model.get_unembedding_of_last_token()   # answer token unembedding (use last token as proxy)
            proj = dot(h, v)                            # projection scalar
            projections.append(proj)
        # Vectorized random pair sampling: generate two arrays of indices
        indices = random.randint(0, N-1, size=(2, num_pairs))
        i_indices = indices[0]
        j_indices = indices[1]
        proj_array = array(projections)
        diffs = abs(proj_array[i_indices] - proj_array[j_indices])
        scores[layer, head] = mean(diffs)               # importance score
```

(C) **Why this design**: We chose to average absolute differences rather than signed differences because we seek a magnitude of contrast regardless of direction; signed differences would cancel if head outputs are symmetric across positions. Using random uniform sampling over all token positions (not just a fixed subset) ensures coverage of both relevant and irrelevant positions while requiring no prior knowledge; the trade-off is that many pairs may involve two irrelevant positions, diluting signal, which we accept because the rare high-difference pairs (relevant vs irrelevant) dominate the mean when num_pairs is sufficiently large. We use the last token's unembedding as proxy for the answer token (since in open-ended generation the answer is unknown) based on the assumption that the final token encodes task intent; this may misalign if the last token is not the target, but we accept the resulting noise as a cost for task-agnosticism. The hyperparameter num_pairs=500 balances computational cost (linear in N) and estimation accuracy; higher values reduce variance but increase time. We chose mean over median as aggregator because it is more sensitive to outliers (the few high-difference pairs that signal important heads). Efficient implementation via caching OV outputs and vectorized pair sampling reduces runtime from O(L*H*N*num_pairs) to O(L*H*N + L*H*num_pairs).

(D) **Why it measures what we claim**: The quantity `abs(proj_i - proj_j)` measures the **contrastive discriminability** of head between two source positions because under the assumption that projection magnitude correlates with relevance (as argued in LOCOS), a large absolute difference implies one position is significantly more relevant than the other; this assumption fails when both positions are equally relevant (e.g., two synonymous needles) or equally irrelevant, in which case the difference is small and contributes little to the average, but such pairs are common and dilute the score. However, the mean over many random pairs is dominated by the small fraction of pairs where one position is relevant and the other irrelevant, because those produce large differences; thus the score effectively measures the expected contrast between a random position and a random other position, which is high only for heads that strongly differentiate relevant from irrelevant tokens. The use of the last token's unembedding as proxy for the answer token assumes that in long-context tasks, the final token's representation is heavily influenced by the retrieval objective; this assumption fails in multi-task settings where the last token may not be a retrieval target, in which case the projection may reflect other factors and the score becomes noisier. **Load-bearing assumption**: The last token's unembedding serves as a valid proxy for the answer token in all open-ended tasks; this assumption is fatal if the last token does not encode the retrieval goal (e.g., in multi-task or when the answer is not the final token). Calibration: In practice, we verify by checking that the score correlates with head importance on a small validation set (e.g., 100 examples) where ground truth relevant positions are known; if correlation is weak, the proxy is invalid for that task.

## Contribution

(1) Introduction of RANDOM-CONTRAST, a task-agnostic importance metric that identifies non-literal retrieval heads from a single forward pass without requiring known source positions, by averaging absolute OV-circuit projection differences over random token pairs. (2) A design principle that contrastive head importance can be estimated via random pair sampling, removing the bottleneck of task-specific annotations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | NoLiMa (non-literal retrieval) | Tests target task directly. |
| Primary metric | ROUGE-L drop after top-k head ablation | Measures functional importance. |
| Baseline 1 | Random head ablation | Isolates chance-level importance. |
| Baseline 2 | LOCOS (logit-based) | State-of-the-art for this task. |
| Baseline 3 | Attention-score-based | Simple literal-focused baseline. |
| Ablation-of-ours | RANDOM-CONTRAST w/ median | Tests mean vs median aggregation. |

### Why this setup validates the claim

This setup forms a falsifiable test of the claim that RANDOM-CONTRAST identifies heads critical for non-literal retrieval. Comparing to random ablation verifies that our selected heads are more important than random chance. Comparing to LOCOS tests whether our contrastive random sampling outperforms logit-based importance, which may miss heads whose effect is not directly reflected in the final logits. The attention-score baseline checks whether simple attention suffices; if not, it reinforces the need for our contrastive approach. The ablation with median tests whether the mean's sensitivity to outliers is essential. The metric ROUGE-L drop directly quantifies the functional impact of ablating the identified heads, providing a clear signal of their importance for generation quality. Additionally, we use an efficient implementation (cached OV outputs and vectorized pair sampling) to ensure the method is computationally feasible on standard hardware.

### Expected outcome and causal chain

**vs. Random head ablation** — On a case where the model must retrieve a non-literal fact (e.g., a synthesized relationship), random ablation degrades performance uniformly across heads. The baseline expects no head to be particularly critical. Our method instead identifies a concentrated set of heads that strongly differentiate relevant from irrelevant tokens via random contrast, so ablating our top-k heads should cause a significantly larger ROUGE-L drop than random ablation, especially on non-literal subsets.

**vs. LOCOS (logit-based)** — On a case where the correct answer token is indirectly predicted (e.g., through multiple layers), LOCOS may assign low importance to heads that are crucial because it measures logit contribution to the final token. Our method uses OV-circuit output projections onto the last token's unembedding, capturing contributions regardless of direct logit influence. Thus we expect our ablation to produce a larger ROUGE-L drop than LOCOS on non-literal tasks where the answer is not directly read off.

**vs. Attention-score-based** — On a case where the relevant token has high attention to many distractors (e.g., long context), attention scores dilute. The baseline rewards heads that attend to the correct token even if also attending to noise. Our method computes contrast between random pairs, so heads with uniform attention get low scores. We therefore expect our top-k heads to be more specific, causing a larger performance drop when ablated, particularly on long-context non-literal retrieval.

### What would falsify this idea

If the performance drop from ablating our top-k heads is not larger than random ablation on non-literal subsets, or if our method underperforms LOCOS on the tasks where LOCOS was designed to excel, then the central claim fails. Furthermore, if the load-bearing assumption that the last token's unembedding serves as a valid proxy is violated (e.g., in multi-task settings), the method may fail to identify relevant heads, and this should be detectable via poor correlation with ground-truth head importance on a calibration set.

## References

1. Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads
2. Gemma 3 Technical Report
3. NoLiMa: Long-Context Evaluation Beyond Literal Matching
4. HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
5. Gemma 2: Improving Open Language Models at a Practical Size
6. BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models
7. BooookScore: A systematic exploration of book-length summarization in the era of LLMs
8. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
