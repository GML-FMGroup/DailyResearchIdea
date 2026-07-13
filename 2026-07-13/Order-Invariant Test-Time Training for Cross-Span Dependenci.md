# Order-Invariant Test-Time Training for Cross-Span Dependencies in Long-Context Language Models

## Motivation

Current test-time training methods, such as Self-Guided Test-Time Training (S-TTT), select evidence spans but treat them independently, failing to capture crucial dependencies that arise from their joint reasoning context. This structural limitation occurs because the adaptation loss (standard LM loss) does not enforce any interaction between selected spans, leading to representations that ignore cross-span semantics. Consequently, the model cannot integrate multiple evidence pieces coherently, limiting performance on multi-hop reasoning tasks.

## Key Insight

Enforcing permutation invariance on the hidden state sequences of selected spans during test-time training forces the model to learn representations that capture the joint semantics of the span set, because the only way to produce consistent states across all orderings is to encode the collective evidence independent of order.

## Method

We propose Order-Invariant Test-Time Training (OI-TTT). (A) **What it is**: OI-TTT augments S-TTT's self-guided span selection with a permutation consistency objective that forces the model to produce invariant hidden state sequences across random orderings of selected spans. The central assumption is that if the model's hidden state sequence is invariant to the order of spans (as measured by Earth Mover's Distance), then the representation must have integrated cross-span semantics, because independent per-span encodings would vary with permutation. Inputs: a test query and associated long context; outputs: adapted model parameters that capture cross-span dependencies. (B) **How it works**:

```python
# OI-TTT: Order-Invariant Test-Time Training
# Given: model M, test query q, long context C, selection module S (attention-based)
# Hyperparameters: permutation count K=5, consistency weight λ=0.1, Sinkhorn iterations=20

# Step 1: Self-guided span selection (as in S-TTT)
spans = S(C, q)  # list of L spans (text segments)

# Step 2: Generate K random permutations of span order
permuted_orders = [random_permutation(range(L)) for _ in range(K)]

# Step 3: For each permutation, compute hidden state sequences
hidden_states = []
for perm in permuted_orders:
    reordered_text = concatenate([spans[i] for i in perm], separator=' ')
    h = M.encode(reordered_text)  # last-layer hidden states, shape (seq_len, d_model)
    # Aggregate to span-level: average hidden states per span (using original span boundaries)
    hidden_states.append(aggregate_per_span(h, perm))  # list of L vectors

# Step 4: Compute consistency loss using Earth Mover's Distance (EMD) with Sinkhorn approximation
avg_repr = element-wise average of all hidden_states lists (over spans)  # shape (L, d_model)
consistency_loss = sum(EMD_sinkhorn(avg_repr, h_i, iterations=20) for h_i in hidden_states) / K

# Step 5: Standard LM loss on selected spans (original order)
lm_loss = cross_entropy(M, concatenate(spans, separator=' '))

# Step 6: Total loss and gradient update
total_loss = lm_loss + λ * consistency_loss
update(M, total_loss)
```

(C) **Why this design**: We chose EMD over KL divergence because EMD operates on the geometry of the representation space, penalizing not just distributional differences but also the cost of moving mass between bins, which better reflects semantic distance between span-level states. We chose span-level aggregation (averaging hidden states per span) over token-level consistency to reduce computational cost and to focus on span semantics rather than exact token positions. We chose to compute consistency against the average across permutations rather than all pairwise distances to keep the loss O(K) instead of O(K^2). The trade-off for EMD is that it requires solving optimal transport, which is slower than KL; we mitigate this by using the Sinkhorn approximation with a small number of iterations (20). The trade-off for span-level aggregation is that we lose intra-span ordering information, which is acceptable because the key information is the joint meaning of the span set. The trade-off for using the average as target is that we lose variance information, but it enforces a single invariant representation.

(D) **Why it measures what we claim**: The consistency loss (EMD between average representation and each permutation's representation) measures **cross-span dependency capture** because if the model treats spans independently, its hidden states for different orderings will diverge (since position and context differ). By enforcing low EMD, we force the model to encode a representation that is invariant to permutation order; this invariance is only possible if the representation captures the joint semantics of all spans together, not individual span meanings. The computational quantity `avg_repr` serves as a consensus target that must be close to all permutations; if the model relies on positional cues, then permuting changes those cues, causing high EMD. This assumption fails when the selected spans are already order-invariant (e.g., if they are all independent facts that do not interact), in which case the consistency loss may still be low without capturing dependencies, but then the LM loss already handles the task. In multi-hop reasoning where spans must be combined, the consistency loss explicitly drives the model to build integrative representations. 

**Failure mode F**: The model may learn trivial invariant representations (e.g., all hidden states set to zero) that achieve low EMD but contain no semantic information. To detect this, we monitor the L2 norm of span-level hidden states during adaptation and discard updates if the norm drops below a threshold (e.g., 0.01). Additionally, we verify that low EMD correlates with dependency strength via a diagnostic synthetic dataset (see experiment).

## Contribution

(1) We introduce Order-Invariant Test-Time Training (OI-TTT), a novel method that augments self-guided test-time adaptation with a permutation consistency objective using Earth Mover's Distance, enabling models to capture cross-span dependencies during test-time adaptation. (2) We demonstrate that enforcing permutation invariance on span-level hidden states is an effective design principle for improving integration of multiple evidence pieces, as it forces the model to encode joint semantics independent of order. (3) We provide a computationally tractable formulation of the consistency loss via average representation as target and Sinkhorn-approximated EMD, suitable for long-context LLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongBench-v2 (including legal subset) | Multi-hop reasoning over long contexts; legal subset for case study. |
| Primary metric | Accuracy | Measures correct reasoning output. |
| Baseline 1 | S-TTT | Self-guided span selection baseline. |
| Baseline 2 | Random Span TTT | Tests importance of span selection. |
| Baseline 3 | Oracle Span TTT | Upper bound for span selection. |
| Baseline 4 | Deep Set TTT | Set-invariance baseline using averaging of span repr without permutation. |
| Ablation-of-ours | OI-TTT w/o consistency | Isolates effect of consistency loss. |
| Diagnostic | Synthetic dataset with controlled dependency | Verifies that low EMD implies joint encoding. |

### Why this setup validates the claim
LongBench-v2 features multi-hop questions that require synthesizing information across multiple distant spans, directly testing cross-span dependency capture. Comparing OI-TTT to S-TTT isolates the added value of the permutation consistency objective. Random and oracle span TTT bracket the benefit of self-guided selection, while the ablation (OI-TTT without consistency) teases apart the contribution of the consistency loss from the selection module. Deep Set TTT (averaging span representations without permutation) tests whether simple set-invariance suffices. Accuracy is the appropriate metric because it directly reflects whether the model produces correct answers that depend on integrating multiple spans; a method that truly captures cross-span dependencies should achieve higher accuracy on multi-hop questions. The diagnostic synthetic dataset (e.g., pairs of spans with known dependency: one where spans must be combined and one where they are independent) will verify that OI-TTT yields lower EMD on dependent pairs, supporting the linking assumption. This combination forms a falsifiable test: if OI-TTT improves accuracy only where cross-span dependencies exist (i.e., multi-hop subtasks), then the claim is supported.

### Expected outcome and causal chain

**vs. S-TTT** — On a multi-hop query that requires linking a person to an event via an intermediate entity (e.g., "Where did the CEO of CorpX graduate?"), S-TTT first selects spans about the CEO and then about the graduation. Because it processes spans in the original sequential order, the hidden states may over-rely on positional cues and treat spans independently, failing to bind the CEO identity to the graduation location. Our method instead permutes spans randomly and enforces invariant hidden states via EMD, forcing the model to encode joint semantics that decouple from order. We expect OI-TTT to outperform S-TTT by a noticeable margin (e.g., 5–10% absolute accuracy) on multi-hop subsets, while achieving parity on single-hop questions where order is irrelevant.

**vs. Random Span TTT** — On a long document where pertinent spans are scattered, Random Span TTT selects arbitrary text segments, often missing critical facts. The model then attends to irrelevant content, producing incorrect answers. Our method first selects spans via an attention-based module (S-TTT style), ensuring relevant segments are chosen. Then, the consistency loss further strengthens their integration. We expect OI-TTT to significantly exceed Random Span TTT across all tasks (e.g., 15–25% improvement), particularly on questions requiring precise retrieval from multiple locations.

**vs. Oracle Span TTT** — Oracle Span TTT has perfect span selection, serving as an upper bound. On a complex question where spans are ambiguous (e.g., multiple entities share names), OI-TTT may select suboptimal spans, leading to lower accuracy than oracle. However, because OI-TTT's consistency loss encourages the model to reason about the relationships among the chosen spans, it can still recover some reasoning even with imperfect selection. We expect OI-TTT to be close to oracle (within 3–5%) on multi-hop questions, but lag more on trivia-style single-span tasks where selection is paramount.

**vs. Deep Set TTT** — Deep Set TTT averages span representations without permutation consistency. It may achieve invariance but does not force cross-span interaction. On multi-hop questions, Deep Set TTT will fail to integrate spans (e.g., linking CEO to graduation), leading to lower accuracy than OI-TTT. We expect OI-TTT to outperform Deep Set TTT by 5–10% on multi-hop subsets.

### What would falsify this idea
If OI-TTT shows uniform accuracy gains across both multi-hop and single-hop subsets (or fails to improve on any), then the central claim that consistency loss specifically enhances cross-span dependency capture is falsified. The diagnostic pattern is a gain concentrated on multi-hop questions vs. S-TTT; absence of such concentration would indicate the method is not operating as hypothesized. Additionally, if on the synthetic diagnostic dataset, low EMD does not correlate with high dependency (i.e., low EMD occurs even for independent spans), then the assumption that permutation invariance forces joint encoding is false.

## References

1. Self-Guided Test-Time Training for Long-Context LLMs
2. Test-Time Training Done Right
3. Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking
4. Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers
5. Understanding Synthetic Context Extension via Retrieval Heads
6. Titans: Learning to Memorize at Test Time
7. Attention Sorting Combats Recency Bias In Long Context Language Models
