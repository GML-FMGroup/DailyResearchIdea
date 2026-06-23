# ADMM-Router: Differentiable Exact Optimal Routing for Mixture-of-Experts

## Motivation

Existing MoE routing methods, such as SoftTopk in MaxScore, rely on approximate differentiable top-k operators that solve a convex relaxation of the min-cost flow problem, offering no guarantee of optimality for the discrete token-to-expert assignment. This approximation causes suboptimal capacity utilization and degrades performance under tight capacity constraints. The root cause is that the exact discrete assignment is non-differentiable and computationally hard; prior work avoids it by smoothing the selection, leaving routing as a heuristic.

## Key Insight

The min-cost flow formulation for MoE routing decomposes into independent per-token top-k and per-expert capacity projections via ADMM, each admitting a closed-form solution through sorting and thresholding, enabling exact optimal assignments that are differentiable through implicit differentiation of the fixed-point iteration.

## Method

# (C) Why this design
We chose ADMM over alternative exact solvers (e.g., linear programming or Lagrangian relaxation) because ADMM decomposes the problem into two per-variable projections that are both closed-form (sorting + thresholding) and trivially parallelizable on GPU, avoiding expensive interior-point methods or combinatorial search. We chose implicit differentiation over unrolled iterations to avoid storing intermediate states across many iterations, saving memory at the cost of solving a linear system via conjugate gradient at the fixed point (adds O(NE) cost per backward pass). We chose a fixed iteration count instead of a convergence criterion to maintain a deterministic computational graph, trading asymptotic convergence guarantee for reproducibility and gradient stability. The dual variable Λ explicitly tracks the misalignment between token preferences and expert capacities, enabling the algorithm to negotiate conflicting constraints exactly and converge to the global optimum of the convex relaxation.

# (D) Why it measures what we claim
The per-token top-k projection (Z_tok) operationalizes global optimality of the assignment because ADMM converges to the optimal solution of the convex cost linear programming problem (Boyd et al., 2011) under the assumption that the cost matrix S is fixed and the problem is convex; this assumption fails when S is parameterized by a neural network and changes during training, in which case the optimality guarantee applies only to the current S, not to the overall model objective. The per-expert capacity projection (Z_exp) measures feasibility because it directly enforces the hard capacity constraint by truncating excess assignments; if the capacity C is too small to accommodate all tokens, the projection forces a feasible assignment that may be suboptimal in terms of total cost, reflecting the inevitable trade-off between capacity and routing quality. The dual variable Λ measures the price of misalignment; when the primal residuals converge to zero, it indicates that the token and expert constraints are simultaneously satisfied, and its magnitude reflects the marginal cost of relaxing constraints.

**Load-bearing assumption:** The fixed-point mapping from S to A is differentiable because the optimization problem is strongly convex (with epsilon quadratic penalty) and has a unique solution that is continuously differentiable by the implicit function theorem (see Amos & Kolter, 2017). To ensure this, we add a small quadratic penalty ε||A||₂² to the linear objective, where ε is annealed from 0.1 to 0 at inference. This makes the problem strongly convex for finite ε, guaranteeing differentiability and enabling correct gradient computation via implicit differentiation.

**Failure mode:** Minimizing per-step routing cost (S·A) may not directly lead to lower final loss because the overall model objective is non-convex and a low-cost assignment at one step may not align with global optimization. We empirically monitor the correlation between routing cost and perplexity during training to detect this.

## Contribution

(1) We introduce ADMM-Router, the first differentiable routing mechanism for mixture-of-experts that exactly solves the token-to-expert min-cost flow problem with formal optimality guarantees for the convex relaxation. (2) We show that the ADMM subproblems reduce to closed-form sorting and thresholding operations, making the algorithm efficient (O(K log K) per iteration) and simple to implement on GPU. (3) We provide a practical implicit differentiation scheme that enables end-to-end training through the ADMM fixed point without memory-intensive unrolling.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WikiText-103 | Standard for MoE language model evaluation |
| Primary metric | Perplexity | Measures overall routing quality and model fit |
| Baseline 1 | Constrained Top-k MoE | Fixed capacity factor; tests capacity enforcement |
| Baseline 2 | Unconstrained Top-k MoE | No capacity limit; tests routing flexibility |
| Baseline 3 | Dykstra-Router | Differentiable LP solver via alternating projections; tests ADMM's efficiency |
| Baseline 4 | Unconstrained Top-k (large capacity) | Capacity set to N; tests if capacity constraints hurt performance |
| Ablation-of-ours | ADMM-Router (unrolled) | Tests necessity of implicit differentiation |
| Ablation-of-ours | ADMM-Router (large-scale, 128 experts) | Tests scalability to many experts |

### Why this setup validates the claim
This setup isolates the effect of exact, differentiable constraint satisfaction. Comparing against constrained MoE (which enforces capacity with a soft limit) tests whether ADMM's hard constraints yield better assignments. Against unconstrained MoE (which ignores capacity), it tests whether capacity enforcement hurts performance when capacity is sufficient. The Dykstra baseline controls for the choice of differentiable LP solver—Dykstra uses alternating Bregman projections and can be slower to converge. The large-scale ablation verifies that ADMM's closed-form projections remain efficient even with 128+ experts. The unrolled ablation isolates the benefit of implicit differentiation for gradient quality. Perplexity is sensitive to routing quality because poor assignments waste model capacity or overload experts, increasing loss.

### Expected outcome and causal chain

**vs. Constrained Top-k MoE** — On a case where token loads are uneven across experts, the constrained baseline uses a soft penalty that allows occasional capacity violations or forces random drops, leading to suboptimal token-expert matches. Our ADMM-Router negotiates exact capacity through dual updates, achieving feasible yet high-quality assignments. We expect ADMM-Router to show lower perplexity, especially on tasks with high variance in token difficulty.

**vs. Unconstrained Top-k MoE** — On a case where capacity is limited and routing decisions are critical, the unconstrained baseline ignores expert capacity, causing some experts to be overloaded and others underused, which saturates expert capacity and increases gradient noise. Our method respects capacity exactly, balancing load. We expect ADMM-Router to maintain stable perplexity, while unconstrained sees degradation as capacity tightens.

**vs. Dykstra-Router** — Dykstra typically requires more iterations to converge to the same precision. Under a fixed iteration budget (e.g., 10 iterations), ADMM achieves lower primal residuals and thus better assignments, leading to lower perplexity. We expect ADMM-Router to outperform Dykstra in both convergence speed and final perplexity.

**vs. Unconstrained Top-k (large capacity)** — When capacity is abundant (C >= N), ADMM-Router should produce assignments identical to unconstrained top-k (each token gets its top-k experts). Thus perplexity should match, confirming that capacity constraints do not degrade performance when capacity is sufficient.

**Ablation: Unrolled vs. implicit** — Unrolled ADMM uses backprop through iterations, which may suffer from gradient bias due to truncation. Implicit differentiation yields exact gradients at the fixed point. We expect implicit to yield better convergence and lower perplexity in the early stages of training.

**Large-scale (128 experts)** — With many experts, the sorting operations scale as O(NE log E) and O(EN log N). ADMM's per-iteration cost remains acceptable (e.g., <100ms for N=1024, E=128). We expect ADMM-Router to maintain its perplexity advantage over constrained baselines, demonstrating practical scalability.

### What would falsify this idea
If ADMM-Router's perplexity is indistinguishable from the constrained baseline across all capacity settings, or if it underperforms unconstrained on ample capacity, the claim that exact hard constraints improve routing would be falsified. Also, if ADMM-Router fails to converge to feasible assignments (e.g., primal residuals large) under tight capacity, then the algorithmic claim fails.

## References

1. Maximum Score Routing For Mixture-of-Experts
