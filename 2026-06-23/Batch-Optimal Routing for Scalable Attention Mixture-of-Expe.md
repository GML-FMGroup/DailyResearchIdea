# Batch-Optimal Routing for Scalable Attention Mixture-of-Experts

## Motivation

Per-token top-k routing in attention MoE, as in Grouped Query Experts (GQE), suffers from two structural problems as models scale: (1) communication overhead because each token's router decision must be communicated independently, and (2) routing collapse where experts become unbalanced despite load-balancing losses. These problems arise because the per-token router ignores global batch statistics, making scaling beyond 250M parameters unreliable.

## Key Insight

By formulating expert assignment as an optimal transport problem over the entire batch, we guarantee load balance and minimize communication by construction, eliminating the need for auxiliary losses and scaling gracefully.

## Method

### (A) What it is
BATCH-MoA replaces per-token top-k expert selection in attention MoE with a batch-level optimal assignment solved via a regularized optimal transport (Sinkhorn) algorithm. Inputs: batch of N tokens with query vectors, E experts with centroids, and per-token top-K. Outputs: soft assignment matrix P (N×E) routing each token to K experts with load balance constraints.

### (B) How it works
```pseudocode
Input: queries q_i (i=1..N), expert centroids c_j (j=1..E), K experts per token
Hyperparameters: τ (Sinkhorn temperature, e.g., 0.1), ε (convergence threshold, e.g., 1e-3)

1. Compute relevance: R[i,j] = q_i^T c_j         // dot product similarity
2. Initialize log-weights: W[i,j] = R[i,j] / τ
3. Set row target: r_target[i] = K
   Set column target: c_target[j] = N*K / E
4. Repeat until convergence (||P - prev|| < ε):
   a. Normalize rows: P[i,:] = softmax(W[i,:]) * r_target[i]   // row-wise softmax scaled
   b. Normalize columns: scale each column so sum over rows equals c_target[j]
   c. Update W = log(P) (with numerical stability)
5. Output soft assignment matrix P

# During inference, round to hard assignment while respecting column capacities:
6. Initialize expert loads L[j] = 0
7. For each token i in descending order of max(P[i,:]):
   a. Select top-K experts j where L[j] < c_target[j], sorted by P[i,j]
   b. Allocate token to those experts, update L[j] += 1
8. Hard assignment: only selected experts compute attention; outputs weighted by original P.
```

### (C) Why this design
We chose optimal transport over per-token top-K because it directly enforces global load balance without an auxiliary loss, removing the collapse tendency that arises when the per-token router ignores peer tokens' choices. We chose Sinkhorn over the Hungarian algorithm because Hungarian is O(N^3) while Sinkhorn scales to large batches with O(NE) per iteration and is differentiable, enabling end-to-end training. We chose entropy regularization to smooth decisions and make the assignment differentiable, accepting that soft assignments may allow tokens to split attention suboptimally during training (e.g., when two experts have nearly equal relevance). We enforced equal column sums as a hard constraint rather than adding a penalty for imbalance, because equal load is a necessary condition for scaling to many experts without overloading memory or communication. This design prioritizes scalability and simplicity over exact per-token optimality, which is acceptable because the global optimum under load constraints is a good proxy for the overall sparse attention quality.

### (D) Why it measures what we claim
The column sum constraint in Sinkhorn operationalizes load balance (the motivation-level concept) because it ensures each expert receives exactly N*K/E tokens in expectation; this equivalence holds under the assumption that the algorithm converges to the optimal transport solution, which fails when the cost matrix is degenerate (e.g., all tokens have identical relevance), in which case P becomes uniform and the assignment reflects uniform load but not expert specialization. The relevance score R[i,j] measures attention relevance (the concept of using the most relevant experts per token) because it is the dot product between token query and expert centroid, which approximates the attention score in a softmax-free manner; this assumption fails if queries and centroids do not capture the attention distribution (e.g., if the model learns poor representations), in which case R becomes random and the routing reduces to arbitrary assignment. Thus, BATCH-MoA's effectiveness hinges on the quality of the relevance computation; if relevance is well-estimated, the optimal transport guarantees load balance and communication efficiency, else it may distribute tokens evenly but without any semantic benefit.

## Contribution

(1) Introduces BATCH-MoA, a batch-level optimal routing for attention Mixture-of-Experts that replaces per-token top-K with a Sinkhorn-based optimal transport assignment, ensuring load balance without auxiliary losses. (2) Demonstrates that this approach removes routing collapse and communication overhead by construction, enabling scaling to larger models and longer sequences compared to methods like GQE. (3) Provides a formal mechanism for jointly optimizing attention relevance and load balance in a differentiable manner, with analysis of the trade-offs between soft and hard assignments.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | C4 language modeling | Standard large-scale text corpus |
| Primary metric | Perplexity | Measures model quality directly |
| Baseline | Standard Transformer | No MoE, baseline for comparison |
| Baseline | Per-token top-k MoA | Common MoA without global balance |
| Baseline | Load-balanced MoA with auxiliary loss | Classic solution for load balance |
| Ablation | BATCH-MoA without column constraint | Isolates effect of load balance |

### Why this setup validates the claim

Using language modeling on C4 with perplexity as the metric provides a direct test of the model's ability to efficiently route attention. Perplexity reflects the model's predictive quality, and improvements should come from better expert selection. The standard Transformer baseline establishes the baseline quality without MoE. The per-token top-k baseline tests the collapse failure mode that our method fixes. The load-balanced auxiliary loss baseline tests an alternative approach to balance, allowing comparison of constraint vs. penalty. The ablation removes the column constraint, isolating the specific effect of load balance. If BATCH-MoA outperforms both per-token top-k and the auxiliary loss baseline, it validates that Sinkhorn-based assignment provides superior load balance without quality degradation. The ablation would show the necessity of the column constraint.

### Expected outcome and causal chain

**vs. Standard Transformer** — On a case where the model has many heads but few are useful per token, the standard Transformer uses all heads, wasting computation and potentially adding noise. Our method selects only relevant experts via optimal transport, so we expect lower perplexity (e.g., 0.2–0.5 points) with comparable total parameters.

**vs. Per-token top-k MoA** — On a case where multiple tokens favor the same expert (e.g., common syntactic patterns), per-token top-k overloads that expert, causing load imbalance and eventually collapsing to using a few experts. Our method enforces uniform load via column constraints, preventing collapse. We expect BATCH-MoA to maintain higher perplexity improvement over the standard baseline, whereas per-token top-k may degrade after initial gains.

**vs. Load-balanced MoA with auxiliary loss** — On a case where the auxiliary loss coefficient is set too high, it overwhelms the primary objective, hurting quality; too low, it fails to balance. Our hard constraint avoids this trade-off. We expect BATCH-MoA to match the best-tuned auxiliary loss model and surpass it in stability, with lower perplexity variance across seeds.

### What would falsify this idea

If BATCH-MoA fails to outperform per-token top-k on perplexity, or if its advantage disappears on tasks with inherently balanced expert usage, then the central claim that optimal transport routing improves attention MoE is false. Specifically, if the gain is uniform across all token types rather than concentrated on subsets prone to overload, the load balance mechanism is not driving improvements.

## References

1. Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention
2. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
3. Mixture of Attention Heads: Selecting Attention Heads Per Token
4. Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints
5. Efficiently Scaling Transformer Inference
