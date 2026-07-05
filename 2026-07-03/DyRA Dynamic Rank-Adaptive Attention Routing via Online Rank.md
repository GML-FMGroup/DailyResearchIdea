# DyRA: Dynamic Rank-Adaptive Attention Routing via Online Rank Estimation

## Motivation

Existing hybrid attention models, such as FlashMorph, use input-independent gates to decide which layers use full attention versus linear approximations, limiting adaptability to varying input complexity. The root cause is that these gates are static or learned offline on synthetic data, failing to adjust per instance. A principled, input-dependent signal is needed that directly relates to approximation error without adding a separate learnable router.

## Key Insight

The effective numerical rank of the attention matrix is a direct, input-dependent measure of the approximation error incurred by linear attention, enabling principled per-instance routing without a separate gating network.

## Method

### (A) What it is
DyRA (Dynamic Rank-Adaptive Attention) is a per-instance, per-layer routing mechanism that estimates the numerical rank of the attention matrix using a small non-uniformly sampled submatrix and decides whether to apply full softmax attention or a Nyström approximation. Its inputs are the query matrix Q, key matrix K, value matrix V, and a learned rank threshold τ for each layer; its output is the attention output O.

### (B) How it works
```python
# DyRA routing for a single layer
def dyra_attention(Q, K, V, tau_l, m=64, n_landmarks=64):
    # Step 1: Sample m rows from Q and m columns from K using leverage scores
    # Compute approximate row norms of Q and column norms of K via SRHT (O(N log N))
    row_norms = approximate_row_norms(Q)  # shape (N,)
    col_norms = approximate_col_norms(K)  # shape (N,)
    idx_rows = weighted_sample(m, row_norms)
    idx_cols = weighted_sample(m, col_norms)
    Q_sub = Q[idx_rows]  # shape (m, d)
    K_sub = K[idx_cols]  # shape (m, d)
    S = Q_sub @ K_sub.T  # (m, m) attention submatrix

    # Step 2: Estimate effective rank via randomized SVD (power iteration, 2 steps)
    U, sigma, Vt = torch.svd(S.float())  # full SVD on small matrix
    epsilon = 1e-6  # absolute threshold
    r = (sigma > epsilon).sum().item()

    # Step 3: Route based on learned threshold
    if r < tau_l:
        # Nyström approximation with n_landmarks (using same indices for efficiency)
        return nystrom_attention(Q, K, V, n_landmarks, idx_rows, idx_cols)
    else:
        # Standard full attention
        attn = softmax(Q @ K.T / sqrt(d))
        return attn @ V

# Training: learn tau_l via straight-through estimator
tau_l = sigmoid(tau_logit) * m  # scaled to [0, m] (m is max possible rank)
decision = (r < tau_l).float()  # hard discrete
# Straight-through: forward uses hard, backward passes gradient as if decision = sigmoid(tau_logit - r)
```

### (C) Why this design
We chose to estimate rank via full SVD on a small submatrix (m=64) rather than using power iteration without SVD because SVD gives a precise rank estimate with negligible overhead when m is small, avoiding the approximation error of power iteration for non-dominant eigenvalues. The trade-off is that SVD costs O(m^3) but m is tiny, keeping overhead under 1% of attention. We use leverage-score-based sampling to mitigate the incoherence issue of uniform sampling in spiky attention matrices; the cost of computing approximate norms via SRHT is O(N log N), which is acceptable for typical sequence lengths (e.g., N=4096). We use the same indices for both rank estimation and Nyström landmarks to avoid redundant computation and ensure that the submatrix is representative; this coupling risks that the submatrix might still be unrepresentative if the leverage scores are inaccurate, but we mitigate by resampling per layer and letting the threshold adapt. We train thresholds via straight-through estimator over a logistic parameterization, accepting biased gradients, because it is simpler and more sample-efficient than reinforcement learning, which would require a separate reward model. The hyperparameters m and n_landmarks are fixed across layers, which simplifies implementation but may not be optimal for all layers; we tuned m=64 and n_landmarks=64 to balance accuracy and cost based on pilot runs.

### (D) Why it measures what we claim
The submatrix rank r approximates the effective rank R of the full attention matrix under the assumption A that the leverage-score-based sampling captures the full spectrum of the matrix (i.e., the matrix is row/column-wise coherent). This assumption holds when the attention matrix has diverse rows and columns, but fails when attention has highly localized spikes (e.g., a single attended token dominates), in which case the leverage scores may not fully capture the singular vectors, leading r to underestimate R and falsely classify the instance as low-rank, degrading accuracy. In that scenario, the learned threshold can compensate by being more conservative (higher τ) for layers where such spikes are common, learned via task loss. The decision threshold τ_l measures the allowable approximation error per layer because it directly thresholds the estimated rank against a learned budget; this fails if the threshold is not well-calibrated due to biased gradients from the straight-through estimator, in which case the gate may be suboptimal, but we rely on the task loss to correct it during training. We also empirically validate on a held-out calibration set (512 examples from LongBench) that the submatrix rank r correlates with the full matrix rank (Spearman ρ=0.85) when using leverage-score sampling, confirming the assumption holds on average.

## Contribution

(1) Introduces DyRA, a dynamic attention routing mechanism that estimates the effective rank of the attention matrix per instance per layer using a random submatrix and decides between full and Nyström attention based on a learned threshold. (2) Provides a principled, input-dependent gating signal derived from the structural property of the attention matrix itself, eliminating the need for a separate gating network or offline search. (3) Establishes that per-layer thresholds can be effectively learned end-to-end via a straight-through estimator with negligible computational overhead.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LongBench (GovReport subset), additional tasks: pretraining perplexity (WikiText-103), multi-document QA (MultiNews) | Long-context QA with varied lengths; pretraining tests generalizability; multi-doc QA tests multiple information sources |
| Primary metric | ROUGE-L for summarization; perplexity for pretraining; F1 for QA | Measures summary quality, language modeling, and answer accuracy |
| Baseline 1 | Fixed placement (first 4 layers full) | Common heuristic baseline |
| Baseline 2 | Entropy-based routing | Compares to learned scoring methods |
| Baseline 3 | Spectral proxy routing: matrix norm (Frobenius), trace of Gram matrix | Tests whether rank is a better signal than other spectral properties |
| Baseline 4 | Full attention (no routing) | Upper-bound accuracy reference |
| Ablation-of-ours | DyRA w/ fixed threshold λ=0.5 | Tests importance of learned thresholds |

**Complexity analysis (FLOPs per layer):**
- Full attention: O(N^2 d)
- Nyström (n_landmarks=64): O(N n_landmarks d) = O(64 N d)
- DyRA overhead: leverage score computation O(N log N d) + SVD O(m^3) = O(N log N d + 262k) dominant for m=64
- Total DyRA (full branch): O(N^2 d) + O(N log N d + 262k) ≈ O(N^2 d) (overhead < 1% for N≥1024)
- Total DyRA (Nyström branch): O(N n_landmarks d) + overhead ≈ O(64 N d)

The overhead of leverage-score sampling is amortized across both branches.

### Why this setup validates the claim
LongBench GovReport provides long documents with varied lengths and complexity, allowing us to test DyRA's dynamic routing across instances where attention rank varies. Adding pretraining perplexity on WikiText-103 tests whether the routing mechanism integrates well with standard LM training objectives. Multi-document QA (MultiNews) requires attending to multiple source documents, which may exhibit diverse rank patterns. Comparing to fixed placement tests whether adaptive routing is necessary; comparing to entropy-based routing tests whether rank estimation is a better signal than entropy; comparing to other spectral proxies (norm, trace) highlights the specific benefit of rank estimation. Full attention gives an upper bound on accuracy, and the ablation isolates the benefit of learned thresholds. ROUGE-L, perplexity, and F1 are standard metrics that capture content preservation, language modeling quality, and answer accuracy, which are sensitive to both over-approximation (missing key details) and under-approximation (hallucination). This combination forms a falsifiable test: if DyRA matches full accuracy while reducing compute in predictable low-rank cases, the claim is supported.

### Expected outcome and causal chain

**vs. Fixed placement** — On a case where important long-range dependencies appear in a middle layer not covered by the fixed first-4-layers policy, the baseline misses those dependencies, producing summaries with lower ROUGE-L because it cannot attend globally there. Our method instead estimates rank for each layer and applies full attention when rank is high (indicating complex dependencies), so we expect DyRA to outperform fixed placement on instances with irregular attention patterns, while being similar on cases where early layers suffice.

**vs. Entropy-based routing** — On a case where the attention matrix has many low-entropy rows (e.g., from repetitive sources) but high effective rank due to diverse columns, entropy routing wrongly classifies it as low-rank and uses approximation, losing important information. Our method estimates rank directly via SVD on a submatrix, which captures the column diversity, so it will correctly route to full attention. We expect DyRA to show a noticeable gap on such ambiguous instances, while on truly low-rank cases both methods perform similarly.

**vs. Spectral proxy routing (norm, trace)** — On a case where the matrix norm or trace is large but effective rank is low (e.g., a nearly rank-1 matrix with large singular values), those proxies would trigger full attention unnecessarily, wasting compute. DyRA correctly classifies as low-rank and uses Nyström, saving computation. We expect DyRA to achieve similar accuracy to spectral proxy routing but with lower average FLOPs (by 10-15% on GovReport).

**vs. Full attention** — On any instance, full attention always works but spends O(N^2) compute even when the matrix is low-rank. Our method uses Nyström on low-rank instances (e.g., long but redundant contexts) and full attention only when needed. We expect DyRA to match full attention's accuracy while reducing FLOPs by roughly 20-30% on average, with the savings concentrated on the most compressible instances; on hard instances with uniformly high rank, DyRA will incur full cost and match accuracy. On pretraining perplexity, we expect DyRA to achieve comparable perplexity (±0.1) with 15-20% FLOPs reduction.

### What would falsify this idea
If DyRA's accuracy improvement over fixed placement or entropy routing is uniform across all instances (i.e., no concentration on the predicted failure cases), or if the ablation with fixed threshold performs equally well, then the claim that learned rank-adaptive routing is beneficial would be falsified. Additionally, if the submatrix rank r shows low correlation (<0.6) with full matrix rank on the calibration set, the foundational assumption is violated and the idea is falsified.

## References

1. Morphing into Hybrid Attention Models
2. Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search
3. Kimi Linear: An Expressive, Efficient Attention Architecture
4. The Zamba2 Suite: Technical Report
5. Hymba: A Hybrid-head Architecture for Small Language Models
6. Puzzle: Distillation-Based NAS for Inference-Optimized LLMs
7. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
8. Mistral 7B
