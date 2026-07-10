# Spectral Regularized Sparse Linear RNNs for Provable Memory Capacity and Retrieval

## Motivation

Sparse linear RNNs like Sparse Delta Memory achieve strong empirical retrieval but lack formal guarantees on memory capacity and retrieval error, inheriting a meta-level problem across linear attention and Mamba-2: they assume standard language modeling loss alone suffices to learn well-conditioned memory states. Without controlling the condition number of the memory matrix, retrieval quality degrades under arbitrary sequence lengths. Sparse Delta Memory's empirical success highlights the need, but its training objective provides no bound on capacity or recall fidelity.

## Key Insight

The condition number of the memory state Gram matrix directly controls both storage capacity (via subspace dimension) and retrieval error (via noise amplification), and can be explicitly bounded by a differentiable spectral regularizer.

## Method

**(A) What it is** — SRS-RNN (Spectral Regularized Sparse RNN) is a sparse linear RNN whose training objective includes a regularizer that penalizes the log condition number of the memory state Gram matrix. Inputs: sequence of tokens; outputs: next-token predictions and readout from a memory matrix. **(B) How it works** — Pseudocode:
```pseudocode
# SRS-RNN Training Step
Input: Sequence x_1,...,x_T, memory size M, sparsity k, regularization strength λ
Initialize recurrent weight W (M x M) and input projection U, output projection C.

For each batch:
  1. Compute memory states: h_t = σ(W h_{t-1} + U x_t)  # sparse activation via top-k
  2. Compute Gram matrix G = H H^T where H = [h_1, ..., h_T] (M x T)
  3. Compute spectral regularizer: R = log(cond(G)) = log(σ_max(G)/σ_min(G))
  4. Compute retrieval loss: L_retrieval = MSE( C h_read, target ) # linear readout
  5. Total loss: L = L_lm + λ * R   # L_lm is language modeling cross-entropy
  6. Backpropagate through all operations, including differentiable SVD (e.g., via PyTorch's SVD).
  7. Update parameters with Adam.
```
Hyperparameters: M=4096, k=32, λ=0.1 (tuned via validation). **(C) Why this design** — We chose to regularize the condition number of the Gram matrix over other measures like nuclear norm because condition number directly bounds the retrieval error via the Hoffman-Wielandt inequality, whereas nuclear norm only controls total energy without guaranteeing well-separated memory slots. We accepted the computational cost of differentiable SVD for each batch (O(M^3)) because it is amortized over batch size and the benefit of provable guarantees outweighs the overhead; for large M (e.g., >10k), we can use randomized SVD approximations. We chose log(cond) over raw cond to make the regularizer scale-invariant (log cond is invariant to constant scaling of G) and to avoid dominance by large eigenvalues; this prevents the regularizer from encouraging degenerate scaling of memory states. We used a sparse activation (top-k) to maintain computational efficiency, accepting that sparsity may limit expressiveness compared to dense states; however, the regularizer ensures that the sparse subspace remains well-conditioned. We opted for a linear readout (C h_read) to align with the theoretical connection between condition number and retrieval error; this sacrifices the potential expressiveness of nonlinear readouts but guarantees the bound's validity. **(D) Why it measures what we claim** — The condition number cond(G) measures worst-case noise amplification in memory recall because the retrieval error is bounded by cond(G) * ||noise|| / ||signal|| under the assumption that the memory read operation is a linear projection from memory states; this assumption fails when using a nonlinear readout, in which case cond(G) reflects only the linear sensitivity of the states themselves. The log(cond) regularizer measures well-conditionedness because it is a monotonic transformation of cond(G); this equivalence holds under the assumption that cond(G) > 1 (true for non-orthogonal matrices), and fails only when cond(G)=1 (optimal, no gradient). The Gram matrix G measures pairwise similarity of memory states because its entries are dot products; this is used to compute cond(G) based on the assumption that memory states span a subspace, and the failure mode of linear dependence is exactly captured by cond(G) approaching infinity. Thus, the regularizer directly operationalizes the meta-level concept of 'provable retrieval guarantee' by controlling the structural quantity that upper-bounds recall error.

## Contribution

(1) A novel spectral regularizer for sparse linear RNNs that provably bounds the condition number of the memory state Gram matrix, enabling guaranteed retrieval error bounds. (2) A training algorithm that integrates differentiable spectral decomposition with sparse activation, providing the first method to explicitly control memory conditioning in linear RNNs. (3) The insight that condition number, rather than sparsity or capacity alone, is the key structural property governing reliable long-context retrieval.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Synthetic Associative Recall | Tests memory retrieval under noise. |
| Primary metric | Recall@1 accuracy | Directly measures retrieval precision. |
| Baseline 1 | Gated DeltaNet (dense key-value) | Dense memory with high interference. |
| Baseline 2 | Mamba-2 (linear attention) | Strong linear RNN baseline. |
| Baseline 3 | Softmax Transformer | Full attention, no memory bottleneck. |
| Ablation-of-ours | SRS-RNN w/o spectral regularizer | Isolates effect of condition number penalty. |

### Why this setup validates the claim

Synthetic associative recall isolates the memory retrieval mechanism by presenting a controlled sequence of key-value pairs followed by a query; noise is injected to degrade the memory. The condition number regularizer aims to make the memory states well-conditioned, which directly bounds retrieval error. By comparing against baselines with different memory architectures (dense key-value, selective state-space, full attention), we test whether our regularizer specifically improves retrieval robustness in high-interference regimes. The ablation removes the regularizer to verify that any gain is not due to other architectural choices. Recall@1 is the natural metric because it reflects exact retrieval quality, which the regularizer is designed to protect.

### Expected outcome and causal chain

**vs. Gated DeltaNet** — On a case with many stored keys (e.g., 1024) and noisy input, Gated DeltaNet produces interference among similar keys because its dense key-value store lacks a mechanism to separate overlapping representations, leading to high condition number. Our method enforces a well-conditioned Gram matrix via spectral regularizer, ensuring that memory states remain nearly orthogonal, thus reducing retrieval confusion. We expect a noticeable gap (e.g., 15–20% higher recall) on high-noise/high-capacity instances, but parity on low-noise/small-capacity ones.

**vs. Mamba-2** — On a sequence with abrupt topic shifts, Mamba-2's gating may overcompress earlier memories when new inputs dominate, because its state update is adaptive but not explicitly conditioned on memory orthogonality. Our method's regularizer maintains a stable eigenstructure of the memory Gram matrix, preserving distinct memory slots even after shifts. We expect a moderate gain (e.g., 5–10%) on long sequences with multiple segments, but similar performance on uniform short sequences.

**vs. Softmax Transformer** — On a long sequence (>8K tokens), Softmax Transformer can attend to all positions directly, but its quadratic cost limits practical length; moreover, it is not designed for compressed memory recall. Our method uses a fixed-size memory with spectral regularization, so on tasks requiring retrieval from many stored items, we anticipate competitive or better recall on long sequences (e.g., 4K+ keys) while using fewer FLOPs. However, on short sequences with few keys, transformer may slightly outperform due to unbounded attention flexibility.

### What would falsify this idea

If SRS-RNN's recall advantage is uniform across all noise levels and key counts (rather than concentrated in high-interference regimes), or if the regularizer actually degrades performance on low-capacity tasks, then the claim that condition number directly controls retrieval error would be invalidated.

## References

1. Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity
2. Short window attention enables long-term memorization
3. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
4. Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling
5. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
6. Learning to (Learn at Test Time)
7. Meta-Learning Fast Weight Language Models
8. What learning algorithm is in-context learning? Investigations with linear models
