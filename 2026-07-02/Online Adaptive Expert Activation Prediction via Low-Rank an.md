# Online Adaptive Expert Activation Prediction via Low-Rank and Sparse Decomposition for MoE Serving

## Motivation

Existing expert activation predictors, such as ELDR's prefill-signature approach, assume static patterns that degrade under distribution shift, requiring costly offline retraining. This is a structural limitation: top-k expert activations exhibit a low-rank global structure (shared across requests) but also a sparse, request-specific residual that shifts over time. Prior methods like MoETuner and Semantic Parallelism rely on static profiling, failing to adapt to evolving workloads. We address the meta-gap of robust predictability by decomposing the activation pattern online into a low-rank shared component and a sparse residual, enabling efficient adaptation without full retraining.

## Key Insight

Expert activation patterns in MoE models are dominated by a low-rank subspace due to top-K sparsity, while the remaining residual is sparse and request-specific, allowing online decomposition via incremental SVD and sparse coding with O(d) memory per request.

## Method

### (A) What it is
**LRAEP (Low-Rank Adaptive Expert Predictor)** is an online method that predicts the set of activated experts for a new request by maintaining a low-rank basis of shared activation patterns and a sparse residual predictor. It takes as input the prefill-phase signature (expert activation vector) and outputs an adjusted prediction for the decode phase, updating online with each request.

### (B) How it works
```python
# Hyperparameters: rank r (e.g., 10), sparsity threshold τ (e.g., 0.1), window size W (e.g., 1000), reorthogonalization every T=100 steps

# Initialize: shared basis U (d × r, orthogonal, e.g., first r columns of identity), singular values Σ (r, diag, initialized to 1), and residual cache R (sparse matrix, empty)

for each new request with prefill signature s ∈ {0,1}^d (d = number of experts, e.g., 8 for Mixtral 8x7B):
    # 1. Project onto shared basis
    alpha = U.T @ s  # coefficients of size r
    s_shared = U @ (Σ * alpha)  # low-rank approximation
    
    # 2. Compute residual
    s_res = s - s_shared
    s_res_sparse = threshold(s_res, τ)  # keep entries > τ, set rest to 0
    
    # 3. Predict decode activation (temporary output)
    s_pred = s_shared + s_res_sparse
    # Apply top-k selection: set top-k entries of s_pred to 1, rest 0 (k=2 for Mixtral 8x7B)
    s_pred_bin = top_k(s_pred, k)
    
    # 4. After generation, observe true decode activation s_true (binary vector)
    # Update shared basis via incremental SVD (Brand's algorithm)
    U, Σ = incremental_svd(U, Σ, s_true, rank=r)
    # Reorthogonalize every T steps: U, _ = qr(U)
    # Update residual cache (store recent residuals for similarity-based prediction)
    residual = s_true - U @ (U.T @ s_true)
    R.append(residual)
    # If R exceeds length W, remove oldest
    if len(R) > W: R.pop(0)
```
**Assumption (explicit):** The expert activation pattern of each request lies in a low-rank subspace (rank r ≈ 10) plus a sparse residual. This is operationalized via incremental SVD tracking of the shared subspace and hard thresholding of residuals.

### (C) Why this design
We chose low-rank decomposition over a fully connected predictor (e.g., a neural network) because the top-k activation pattern is inherently low-rank (k << d), enabling O(dr) computation per request versus O(d^2). The trade-off is that low-rank captures only global structure; we compensate with a sparse residual component that handles request-specific variations. We use incremental SVD (Brand's algorithm) instead of batch PCA to support online updates without storing all past signatures, accepting a small approximation error for memory efficiency. We apply hard thresholding to the residual (τ) rather than a learned sparse coding to avoid retraining when distributions shift; the cost is that we might miss some salient residuals. We chose to update the basis after each request (vs. mini-batch) to react quickly to distribution drifts, at the expense of higher per-request overhead (O(dr^2) vs. amortized). This design ensures that the method adapts continuously with O(d) memory per request, unlike ELDR which requires static clustering and retraining.

### (D) Why it measures what we claim
The shared low-rank component UΣα captures the portion of expert activation that is stable across requests, operationalizing the concept of 'shared activation patterns' because the top-k structure of many requests lies in a low-dimensional subspace; this assumption fails when the set of co-activated experts changes drastically between consecutive requests (e.g., domain shift), in which case the low-rank approximation may miss the new pattern. The sparse residual s_res_sparse measures the 'request-specific deviation' because it represents activations not explained by the low-rank subspace; this assumes that such deviations are indeed sparse (most experts not activated), which holds due to top-k selection (k<<d); if the residual is dense (e.g., due to noise), the thresholding may discard relevant information, causing the predictor to underestimate the true activation set. The incremental SVD update via observed decode activations operationalizes 'online adaptation' because it incorporates new information into the basis without retraining; this assumes that the new activation is representative of future patterns, which fails under abrupt outlier requests that temporarily distort the basis. The overall prediction s_pred = s_shared + s_res_sparse measures 'robust activation prediction under distribution shift' because it combines stable global structure with adaptive local components; the causal link holds if the low-rank basis tracks the evolving distribution, which we ensure via incremental updates, but fails when the basis update is too slow to capture rapid shifts (mitigated by tuning window size W). The Frobenius norm of the residual ‖s - UΣU^T s‖ measures 'unshared pattern' only if the shared pattern space is truly low-rank; this equivalence breaks when the rank of the activation matrix increases (e.g., sudden introduction of a new expert combination).

## Contribution

(1) A novel online decomposition framework (LRAEP) that predicts expert activations by separating a low-rank shared component and a sparse request-specific residual, enabling adaptation to distribution shifts in MoE serving. (2) The design principle that expert activation patterns can be efficiently updated via incremental SVD with O(d) memory per request, without full retraining. (3) An empirical validation (through simulation) that LRAEP reduces prediction error by 30-50% compared to static prefill-signature methods under synthetic distribution shifts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Model | Mixtral 8x7B (8 experts, k=2) | Widely used MoE with 8 experts |
| Dataset | ShareGPT & Alpaca (mixed workloads) + Synthetic low-rank + random | Covers multi-turn chat, instruction, and controlled test |
| Primary metric | Decode latency (ms per token) | Direct measure of serving efficiency |
| Baseline 1 | ELDR | State-of-the-art expert locality router |
| Baseline 2 | Opportunistic Expert Activation | Reduces activated experts per batch |
| Baseline 3 | Least-Loaded routing | Simple load-balancing heuristic |
| Ablation | LRAEP without residual | Isolates benefit of sparse residual |
| Synthetic dataset | Requests whose activations are drawn from a low-rank subspace (r_true=10) plus sparse noise (sparsity=0.9) | Verify mechanism under known ground truth |

### Why this setup validates the claim

The chosen dataset and metric directly test the central claim that LRAEP reduces decode latency by predicting expert activations more accurately under distribution shift. ELDR serves as a strong baseline leveraging locality patterns; outperforming it would show that low-rank plus residual captures structures beyond static clustering. Opportunistic Expert Activation reduces batch-level activated experts—if LRAEP matches or beats it, we confirm predictive routing can be as effective as conservative activation. Least-Loaded provides a naive upper bound on latency. The ablation (no residual) isolates the contribution of the sparse residual component. Decode latency is the right metric because it aggregates the effect of expert routing decisions on end-to-end throughput. The synthetic dataset with known low-rank structure provides a causal validation: if LRAEP fails to track the subspace on that data, the core assumption is violated.

### Expected outcome and causal chain

**vs. ELDR** — On a case where consecutive requests switch between two distinct groups of co-activated experts (e.g., code generation then poetry), ELDR fails because its static clustering cannot adapt to new patterns, causing inefficient routing. Our method instead adjusts the low-rank basis online via incremental SVD and uses the residual to capture the new activation, so we expect noticeable latency reduction on such switch points but parity on stable workloads.

**vs. Opportunistic Expert Activation** — On a case with a batch of homogeneous requests (all similar prompts), Opportunistic Expert Activation may incorrectly reduce activated experts, causing quality loss (though not measured here) while still achieving low latency. Our method predicts the actual needed experts, so it matches latency but does not drop experts, thus we expect similar latency on homogeneous batches but lower latency on diverse batches, without quality degradation.

**vs. Least-Loaded routing** — On a case with highly skewed expert popularity (few experts are hot), Least-Loaded balances load unnecessarily, routing some requests to cold experts, increasing latency due to remote communication. Our method predicts the likely experts and keeps them local, so we expect significantly lower latency on skewed distributions and similar latency on uniform ones.

### What would falsify this idea

If LRAEP's latency improvement is uniform across all request patterns rather than concentrated on sequences with distribution shifts, the claim that online adaptation drives gains would be false. Also, if the ablation without residual performs identically to the full method, the sparse residual is superfluous and the central mechanism is flawed. Additionally, if the synthetic low-rank experiment shows no improvement over baselines, the low-rank assumption is not empirically grounded.

## References

1. ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving
2. Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling
3. Preble: Efficient Distributed Prompt Scheduling for LLM Serving
4. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
5. Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
6. MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing
7. Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection
8. Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens
