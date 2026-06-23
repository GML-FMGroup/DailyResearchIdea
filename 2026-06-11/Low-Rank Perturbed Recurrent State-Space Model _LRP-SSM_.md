# Low-Rank Perturbed Recurrent State-Space Model (LRP-SSM)

## Motivation

Existing diagonal state-space models (e.g., LrcSSM from 'Parallelization of Non-linear State-Space Models') achieve parallel training via a diagonal Jacobian but restrict cross-dimensional interactions, limiting expressivity for tasks with coupled dynamics. This reflects a broader meta-gap: the fixed parametric form of diagonal or physical models cannot generalize to the full space of nonlinear recurrent dynamics, as seen across hybrid architectures from synthetic tasks and electrical network models. A low-rank off-diagonal perturbation can bridge this gap while preserving parallelization via the Woodbury identity.

## Key Insight

A low-rank update to the diagonal Jacobian preserves O(N) parallel computation through the Woodbury identity because the inversion of the perturbed system reduces to inverting a small r×r matrix, enabling efficient capture of cross-dimensional nonlinear couplings.

## Method

LRP-SSM is a recurrent state-space model where the state transition Jacobian is parameterized as a diagonal matrix plus a low-rank factor: J = Λ + UVᵀ, with Λ ∈ ℝ^{N×N} diagonal, U,V ∈ ℝ^{N×r} and r << N. The forward pass computes h_t = J h_{t-1} + B x_t in parallel using a closed-form solution via the Woodbury identity, outputting y_t = C h_t + D x_t.  

**Load-bearing assumption**: The low-rank perturbation enables full parallelization of the diagonal parts, but the correction step is sequential due to the coupling; however, with r << N, the sequential cost O(T r^2) is much smaller than the parallel diagonal scan O(T N). The parameterization assumes that off-diagonal couplings in the true dynamics are low-rank.  

**Concrete forward pass**:  
1. Diagonal scan: Compute ĥ_t = Λ ĥ_{t-1} + B x_t via a parallel prefix sum (associative scan) over time, with ĥ_0 = h_0. Complexity O(T N).  
2. Low-rank correction: For each time step t (sequential), compute c_t ∈ ℝ^r from:  
   c_t = (I_r - V^T (I_N - Λ)^{-1} U)^{-1} [ V^T (I_N - Λ)^{-1} (B x_t + Λ ĥ_{t-1} - ĥ_t?) + V^T Λ^t h_0 ]?  
   Actually, using the Woodbury identity, the full state can be written as: h_t = ĥ_t + U (I_r + V^T U)^{-1} V^T (ĥ_t - Λ ĥ_{t-1} - B x_t?). This simplifies to:  
   Let k_t = V^T h_t, then the recurrence for k_t is:  
   k_t = V^T Λ ĥ_{t-1} + (V^T U) k_{t-1} + V^T B x_t, with ĥ_{t-1} known from the diagonal scan. This is a linear recurrence of order 1 in ℝ^r. Solve it sequentially in O(T r^2). Then h_t = ĥ_t + U k_t.  
   Complexity: O(T r^2) sequential, O(T N) parallel (diagonal scan). Since N >> r, overall is O(T N).  
3. Output: y_t = C h_t + D x_t.  

**Initialization**: Λ initialized from LrcSSM (diagonal eigenvalues from liquid networks, e.g., real parts negative, imaginary parts random). U, V initialized with small random values (e.g., uniform ±0.01). B, C, D initialized as in standard SSMs (e.g., random orthogonal for B, small random for C).  

(A) What it is: ... (B) How it works: [as above] (C) Why this design: We chose a low-rank off-diagonal correction over a full dense matrix because the Woodbury identity allows exact inversion in O(N + r^3) per time step rather than O(N^3), ensuring the core parallelization benefit of diagonal models is preserved. We chose to compute the correction sequentially rather than via another parallel scan because the low-rank structure makes the sequential cost O(T r^2) negligible compared to O(T N) of the diagonal scan, and a purely parallel correction would require solving a block-tridiagonal system of size T r, which is more complex and memory-intensive. We chose to initialize Λ from the LrcSSM initialization (diagonal eigenvalues from liquid networks) and U,V randomly small to encourage off-diagonal couplings to be learned from data, avoiding the need for a specific physical prior. The trade-off is that very high-rank couplings cannot be captured, but we accept this because real-world tasks often exhibit low-dimensional interactions. (D) Why it measures what we claim: The diagonal scan produces ĥ_t, which measures the uncoupled dynamics because it assumes state dimensions evolve independently under Λ. The low-rank correction term U k_t measures cross-dimensional interactions because it adds back the effect of UVᵀ; this assumes that off-diagonal couplings are exactly low-rank, which fails if the true coupling matrix is full-rank, in which case the correction only captures the top r singular modes. The computation of k_t via the recurrence measures the projection of the state onto the row space of V, which captures the most correlated directions under the low-rank assumption. If the assumption fails, the remaining unmodeled interactions degrade performance, but the method still benefits from the dominant modes.

## Contribution

(1) LRP-SSM, a recurrent state-space model with a diagonal-plus-low-rank Jacobian that enables parallel training via the Woodbury identity while learning cross-dimensional interactions. (2) A design principle that low-rank perturbations can bridge the expressivity gap of diagonal SSMs without sacrificing parallel computation efficiency. (3) Empirical validation (in proposed experiments) that LRP-SSM outperforms LrcSSM and S5 on long-range tasks requiring coupled dynamics, such as chaotic time series forecasting.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Long Range Arena (LRA) + synthetic sine mixing | LRA measures long-range; synthetic tests low-rank assumption |
| Primary metric | Classification accuracy on LRA; RMSE on synthetic | Accuracy for LRA tasks; RMSE for recovery of coupled dynamics |
| Baseline 1 | LRU | Pure diagonal; tests need for off-diagonal |
| Baseline 2 | S5 | Structured diagonal; tests advantage over it |
| Baseline 3 | Mamba | Selective SSM; tests efficiency vs selectivity |
| Baseline 4 | Low-rank RNN (LRNN: full N×N matrix factorized as UVᵀ, r=5) | Directly compares against low-rank recurrent model |
| Ablation-of-ours | LRP-SSM with r=0 | Removes low-rank; isolates its benefit |

### Why this setup validates the claim
Long Range Arena (LRA) comprises tasks requiring long-range context, where off-diagonal interactions are crucial (e.g., ListOps aligning parentheses). Comparing against LRU (pure diagonal) directly tests whether the low-rank correction improves expressivity. S5 (structured diagonal with complex eigenvalues) tests if low-rank captures more than complex diagonal dynamics. Mamba (selective state) tests if low-rank offers a favorable accuracy–efficiency trade-off versus input-dependent gating. The low-rank RNN baseline (LRNN) tests whether our parameterization is more efficient than a standard low-rank recurrent model (which also uses UVᵀ but without diagonal scan). The ablation LRP-SSM (r=0) isolates the low-rank contribution. Additionally, we include a synthetic sine wave mixing task where the true Jacobian has a known rank r0; this directly tests the assumption that off-diagonal couplings are low-rank. Accuracy is the right metric for LRA tasks; RMSE on synthetic measures how well the model recovers the dynamics. If LRP-SSM outperforms baselines on tasks requiring cross-dimensional interactions while performing similarly on independent tasks, the claim about low-rank capturing couplings is supported. Conversely, uniform improvement would not refute the method but would not validate the specific mechanism.

### Expected outcome and causal chain

**vs. LRU** — On ListOps, tracking multiple parentheses types requires state dimensions to communicate (e.g., to match opening and closing). LRU's diagonal dynamics treat each dimension independently, so it fails to align parentheses far apart. Our method's low-rank correction allows dimensions to couple via U and V, enabling the state to propagate cross-information. We expect LRP-SSM achieves >10% higher accuracy on ListOps, with smaller gains on Text or IMDb where cross-dim interactions are less critical.

**vs. S5** — On Pathfinder (long-range pixel correspondence), spatial relationships require cross-dim mixing. S5 uses complex diagonal recurrence; dimensions do not directly interact, so it struggles to selectively route information between pixels. Our low-rank correction provides adaptive mixing via learned low-rank factors. Expect LRP-SSM to improve Pathfinder accuracy by ~5%.

**vs. Mamba** — Mamba's selectivity adjusts per-token, but recurrence remains diagonal (input-dependent decay). On tasks requiring sustained cross-dim interaction (e.g., ListOps), Mamba must learn to route through input gating, which can be unstable. Our model maintains fixed low-rank coupling, providing stable cross-dim mixing. Expect comparable accuracy on most tasks, but LRP-SSM uses fewer FLOPs due to fixed recurrence.

**vs. LRNN** — LRNN uses a full low-rank recurrence without a diagonal baseline, so it suffers from slow training (no parallel scan) and vanishing/exploding gradients. Our diagonal scan provides a stable base, and the low-rank correction adds expressivity with minimal overhead. Expect LRP-SSM to train 5× faster and achieve higher accuracy on long-range tasks.

### What would falsify this idea
If LRP-SSM (r>0) fails to outperform LRP-SSM (r=0) on any LRA task, or if its advantage is uniform across tasks (suggesting the low-rank is not the cause of improvement), then the claim that low-rank captures off-diagonal couplings is unsupported. Additionally, if on synthetic sine mixing with known rank r0, LRP-SSM with r≥r0 performs no better than with r<r0, then the assumption that off-diagonal couplings are low-rank is invalid for that task.

## References

1. Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling
2. Oscillatory State-Space Models
3. Mechanistic Design and Scaling of Hybrid Architectures
4. Simplified State Space Layers for Sequence Modeling
5. Neural Oscillators are Universal
6. It's Raw! Audio Generation with State-Space Models
7. Resurrecting Recurrent Neural Networks for Long Sequences
8. Mistral 7B
