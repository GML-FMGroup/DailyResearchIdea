# Forgetting-Robust Test-Time Training via Orthogonal Gradient Projection

## Motivation

Existing test-time training methods, such as Self-Guided Test-Time Training (S-TTT), assume that performing gradient descent on selected spans is safe. However, this overlooks the fact that updating parameters on selected spans can increase loss on non-selected tokens, leading to catastrophic forgetting of context. The root cause is the lack of an explicit constraint preventing updates that harm non-selected tokens.

## Key Insight

Projecting the gradient update onto the orthogonal complement of the gradient of non-selected token loss guarantees that the update does not increase loss on non-selected tokens to first order, effectively isolating the adaptation to selected spans.

## Method

# Forgetting-Robust Test-Time Training via Orthogonal Gradient Projection

## Method

### **What it is**
Forgetting-Robust Orthogonal Test-Time Training (FROTTT) is a parameter update rule that takes model parameters θ, selected spans S, and non-selected tokens N, and produces updated parameters θ' that optimize loss on S without increasing loss on N to first order.

### **How it works**
```python
def frottt_update(θ, S, N, α=1e-4, ε=1e-8, backtrack=False, tol=1e-6):
    g_S = ∇_θ L(θ, S)          # gradient on selected spans
    g_N = ∇_θ L(θ, N)          # gradient on non-selected tokens
    # Project g_S onto orthogonal complement of g_N
    coeff = dot(g_S, g_N) / (dot(g_N, g_N) + ε)
    proj = g_S - coeff * g_N
    θ' = θ - α * proj
    if backtrack:
        # Verify first-order assumption: check actual non-selected loss change
        L_N_old = L(θ, N)
        L_N_new = L(θ', N)
        if L_N_new > L_N_old + tol:
            θ' = θ  # revert update
    return θ'
```
Hyperparameters: α (learning rate, default 1e-4), ε (small constant for numerical stability, default 1e-8), backtrack (boolean flag to enable verification, default False), tol (tolerance for loss increase, default 1e-6).

### **Why this design**
We chose orthogonal projection over simply adding a regularization term (e.g., elastic weight consolidation) because projection guarantees that non-selected loss does not increase to first order, whereas regularization only penalizes large changes without a hard constraint. This ensures forgetting robustness at the cost of potentially slowing adaptation if g_S and g_N are aligned. We also chose to compute g_N on all non-selected tokens rather than a subset to avoid approximation errors, accepting higher computational cost. Finally, we use a single projection step per update instead of iterative refinement to keep training efficient, trading off precision for speed. **Load-bearing assumption**: The first-order orthogonal projection of the selected-span gradient onto the nullspace of the non-selected token gradient guarantees that the non-selected token loss does not increase after the update, even with finite step sizes and nonlinear loss landscapes. This assumption holds only when the loss landscape is locally linear and gradients are accurate; it may fail due to curvature or noise. To calibrate, we optionally include a backtracking check that reverts the update if the actual non-selected loss increases beyond a tolerance.

### **Why it measures what we claim**
The quantity `proj` measures the component of the selected-span gradient that is orthogonal to the non-selected gradient. This operationalizes "forgetting-robust integration" because it guarantees that the update does not increase loss on non-selected tokens to first order, assuming that the loss landscape is smooth and gradient information is locally linear; this assumption fails when the Hessian is ill-conditioned or when gradients are noisy, in which case `proj` might still allow some increase in loss due to higher-order effects. The dot product scaling factor `coeff = dot(g_S, g_N)/dot(g_N,g_N)` measures the alignment between the two gradients; a high alignment indicates that updating on S would harm N, so the projection removes that component. This component is causally linked to forgetting because increasing loss on non-selected tokens directly corresponds to forgetting of that context. The backtracking flag, when enabled, verifies whether the first-order approximation holds by measuring the actual change in non-selected loss after the update.

## Contribution

(1) A novel algorithm, Forgetting-Robust Orthogonal Test-Time Training (FROTTT), that constrains gradient updates on selected spans to prevent catastrophic forgetting by projecting onto the orthogonal complement of the non-selected gradient. (2) A design principle that test-time adaptation can be made forgetting-robust without external data or complex regularization, simply by ensuring first-order invariance on non-selected tokens. (3) An analysis of the trade-offs between computational cost and forgetting robustness, highlighting that the projection step can be computed efficiently with a dot product and vector scaling.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongBench-v2 (2000 examples, balanced Chinese/English) | Long-context QA benchmark with diverse tasks |
| Primary metric | Accuracy | Measures task performance directly |
| Secondary metric | Non-selected loss change | Verifies first-order assumption empirically |
| Baseline | Random span TTT | Forgets due to misaligned gradients |
| Baseline | Oracle span TTT | Ideal but unrealistic upper bound |
| Baseline | Standard full-context TTT | Computationally heavy alternative |
| Ablation | FROTTT without projection | Tests necessity of orthogonal update |
| Ablation | FROTTT with backtracking (tol=1e-6) | Tests impact of verifying first-order assumption |

### Why this setup validates the claim

LongBench-v2 requires reasoning over long contexts, making forgetting a critical issue. Random span TTT serves as a baseline that inevitably forgets because its gradient on selected spans may increase loss on non-selected tokens, causing performance drop. Oracle span TTT uses perfect span selection (impossible in practice) to show the ceiling of improvement. Standard full-context TTT represents a compute-intensive alternative that avoids forgetting but is impractical for long contexts. The ablation (FROTTT without projection) isolates the effect of the orthogonal update: if FROTTT outperforms it, the projection is indeed responsible for forgetting robustness. Accuracy is the appropriate metric because it directly reflects whether the model retains relevant information from the entire context while adapting to selected spans. The secondary metric of non-selected loss change provides direct evidence of whether the first-order assumption holds, with a small change indicating successful forgetting prevention.

### Expected outcome and causal chain

**vs. Random span TTT** — On a case where selected spans are partially relevant but non-selected tokens contain crucial context, the baseline updates parameters to minimize loss on selected spans, inadvertently increasing loss on non-selected tokens (e.g., forgetting a key entity). This leads to incorrect answers on queries requiring that entity. Our method projects out the harmful gradient component, so the update does not increase loss on non-selected tokens, preserving that entity. We expect FROTTT to outperform random span TTT by a noticeable margin on questions that depend on non-selected context, with similar performance on questions that depend only on selected spans.

**vs. Oracle span TTT** — On a case where the optimal spans are unknown, oracle span TTT assumes perfect selection and updates only on truly relevant tokens, yielding the best possible adaptation without forgetting. Our method does not have oracle knowledge but avoids forgetting by construction. We expect oracle span TTT to achieve higher accuracy overall, but FROTTT should approach it on tasks where gradient alignment is low (i.e., selected and non-selected gradients are orthogonal), and fall short when alignment is high because projection reduces the adaptation signal.

**vs. Standard full-context TTT** — On a long sequence (e.g., 100K tokens), standard TTT updates on all tokens equally, which is computationally expensive and may overfit to irrelevant tokens. Our method updates only on selected spans, making it faster and more targeted, but risks losing information from non-selected tokens if projection is not perfect. We expect FROTTT to achieve competitive accuracy with much lower computational cost, especially on very long contexts where standard TTT is prohibitive.

### What would falsify this idea

If FROTTT performs no better than random span TTT on questions where non-selected tokens are critical, or if its gains are uniform across all question types rather than concentrated on those requiring preserved non-selected context, then the claim that projection prevents forgetting is false. Additionally, if the non-selected loss change metric shows a significant increase after projection updates, the first-order assumption would be violated, indicating the need for backtracking.

## References

1. Self-Guided Test-Time Training for Long-Context LLMs
2. Test-Time Training Done Right
3. Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking
4. Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers
5. Understanding Synthetic Context Extension via Retrieval Heads
6. Titans: Learning to Memorize at Test Time
7. Attention Sorting Combats Recency Bias In Long Context Language Models
