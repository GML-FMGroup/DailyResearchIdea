# DART: Dynamic Adaptive Rank Tracking for Online Low-Rank Subspace Identification in RL Fine-Tuning

## Motivation

Existing methods like SAR extract a reasoning subspace post-hoc but assume stationarity, failing in non-stationary RL fine-tuning where policy drift and reward shifts cause the subspace to evolve. Meanwhile, On Predictability shows RL updates are low-rank and linearly predictable but requires full SVD at each step, which is computationally prohibitive. The root cause is the lack of an online algorithm that tracks the dynamic low-rank structure without full decomposition, leading to either stale subspaces or expensive recomputation.

## Key Insight

The reasoning subspace in RL fine-tuning evolves smoothly due to gradual policy updates, enabling a streaming approach with temporal smoothness regularization to maintain an accurate low-rank approximation incrementally via Riemannian SGD.

## Method

(A) **What it is:** DART (Dynamic Adaptive Rank Tracking) is a streaming algorithm that maintains a factored low-rank approximation (U, Σ, V) of the gradient subspace during RL fine-tuning, updating it incrementally via Riemannian SGD on a regularized objective with a convex temporal smoothness penalty. Input: gradients at each RL step; Output: an orthonormal basis U (d×k), singular values Σ (k×k), and right singular vectors V (d'×k) representing the reasoning subspace.

(B) **How it works:**
```python
# Initialize: U = random orthogonal d×k, Σ = I_k, V = random orthogonal d'×k
# Hyperparameters: λ (temporal smoothness), η (learning rate), γ (rank adjustment threshold)
for step t in 1..T:
    g_t = gradient from RL step t  # shape d×d' (parameter update matrix)
    # Compute reconstruction error
    E_t = g_t - U @ Σ @ V.T
    # Update U via Riemannian SGD on Grassmann manifold (tangent step + retraction)
    grad_U = -2 * (E_t @ V @ Σ)  # Euclidean gradient
    U_new = retraction(U - η * (I - U U.T) @ grad_U)  # project to tangent space
    # Update V symmetrically
    grad_V = -2 * (E_t.T @ U @ Σ)
    V_new = retraction(V - η * (I - V V.T) @ grad_V)
    # Update Σ via gradient descent with L2 regularization for smoothness
    grad_Σ = -2 * U.T @ E_t @ V + 2 * λ * (Σ - Σ_prev)  # temporal smoothness penalty
    Σ_new = Σ - η * grad_Σ
    # Rank adjustment: compute spectral gap of Σ_new (e.g., |σ_i - σ_{i+1}|)
    if gap below threshold γ at rank k: increase k by 1; re-initialize new singular vectors
    if singular value σ_k < ε: decrease k by 1; truncate
    # Store Σ_prev = Σ_new for next step's smoothness
return U_T, Σ_T, V_T
```

(C) **Why this design:** We chose Riemannian SGD on the Grassmann manifold instead of Euclidean updates because the orthogonality constraint on U and V must be preserved for a valid subspace representation; the Riemannian retraction ensures constraints are satisfied exactly, avoiding costly re-orthogonalization. The temporal smoothness penalty on Σ (L2 difference from previous step) is convex and encourages gradual evolution, reflecting the empirical observation that RL updates change slowly (On Predictability). We use a spectral gap test for rank adaptation because it provides a principled criterion based on eigenvalue decay, avoiding manual rank selection. The trade-off is that the spectral gap threshold γ requires tuning: too low may miss rank increases, too high causes overfitting. We accept this cost because adaptive rank is essential for computational efficiency. Finally, we update U and V symmetrically rather than only updating U and recomputing V, which would break the factorized form; symmetric updates preserve the bilinear structure at the cost of double computation per step.

(D) **Why it measures what we claim:** The reconstruction error E_t measures *subspace alignment* because the low-rank approximation UΣV^T captures the dominant spectral components of the gradient; the assumption is that the reasoning-relevant updates lie in a low-rank subspace (SAR, On Predictability). This assumption fails when gradients are full-rank (e.g., early random drift), in which case E_t reflects generic reconstruction error rather than misalignment. The temporal smoothness penalty on Σ measures *temporal coherence* of singular values under the assumption that the subspace evolves slowly; when jumps occur (e.g., reward shift), the penalty biases the estimate toward a smoother sequence than the data, causing lag. The spectral gap test measures *rank sufficiency* under the assumption that a significant gap separates signal from noise; if the gap is ambiguous (e.g., uniform spectrum), the test may mis-estimate rank, leading to over- or under-fitting. Each component operationalizes a concept from the motivation: (i) reconstruction error → alignment with reasoning subspace, (ii) temporal penalty → smooth evolution, (iii) spectral gap → rank identification. Without these explicit assumption-bounded links, the method would risk claiming validity under unstated conditions.

## Contribution

(1) DART, an online streaming algorithm that tracks the low-rank subspace of RL gradient updates via Riemannian SGD with a convex temporal smoothness penalty, enabling efficient non-stationary subspace identification without full SVD. (2) A principled rank adaptation mechanism based on spectral gap testing, eliminating the need for manual rank selection and automatically adjusting to changes in subspace dimensionality. (3) Demonstration that the method can be integrated into existing RL fine-tuning pipelines with minimal overhead, preserving reasoning performance while reducing computational cost.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MATH500 | Standard reasoning benchmark, varied difficulty |
| Primary metric | Pass@1 accuracy | Direct measure of reasoning quality |
| Baseline 1 | Full RL training | Test benefit of low-rank subspace tracking |
| Baseline 2 | SAR (static subspace) | Test benefit of dynamic rank adaptation |
| Baseline 3 | LoRA (fixed rank) | Test benefit of adaptive rank vs. fixed |
| Ablation of ours | DART without rank adaptation | Isolate impact of rank adjustment |

### Why this setup validates the claim
This design directly tests the central claim that a dynamically tracked, low-rank gradient subspace improves RL fine-tuning efficiency. Full RL (no subspace) establishes the baseline cost of full-parameter updates; if DART outperforms it, the subspace assumption is validated. SAR (static subspace) isolates the benefit of rank adaptation over a fixed post-hoc extraction. LoRA (fixed low rank) tests whether adaptive rank is necessary over a static low-rank bottleneck. The ablation without rank adaptation controls for the rank adjustment mechanism. The primary metric, pass@1 on MATH500, is sensitive to subtle reasoning improvements and avoids confounding from multi-step evaluation. This combination forms a falsifiable test: if DART only matches baselines uniformly, the claim of dynamic adaptation is unsupported.

### Expected outcome and causal chain

**vs. Full RL** — On a case where gradients become large and erratic (e.g., reward spike), full RL overfits to noisy updates because it has no subspace constraint, causing performance drop. Our method instead maintains a smooth low-rank subspace via temporal penalty, so expected signal: DART shows a noticeable gain on volatile training steps but parity on stable phases.

**vs. SAR** — On a case where the reasoning distribution shifts (e.g., new problem category), SAR’s static subspace fails to capture new gradient directions because it was frozen after RL. Our method adapts rank and basis dynamically, so expected signal: DART improves on the shifted subset while SAR degrades, with overall gap on mixed benchmarks.

**vs. LoRA** — On a case where the required rank exceeds LoRA’s fixed budget (e.g., complex multi-step reasoning), LoRA underfits because its capacity is insufficient. Our method grows rank via spectral gap, so expected signal: DART outperforms LoRA on high-complexity problems but matches on simple ones.

### What would falsify this idea
If DART’s improvement is uniform across all problem subsets (easy and hard) or on all baselines, rather than concentrated where gradient subspace drift or rank insufficiency occurs, then the central claim of dynamic adaptive tracking being the cause is false.

## References

1. Spectral Rewiring for Exploration, Purification, and Model Merging
2. Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
3. On Predictability of Reinforcement Learning Dynamics for Large Language Models
4. Training language models to follow instructions with human feedback
5. Locating and Editing Factual Associations in GPT
6. Qwen2.5-Coder Technical Report
7. Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems
8. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement
