# Global Optimization for Non-convex Semi-infinite Programs in Microgrid Optimal Control via Spatial Branch-and-Bound Integrated with Local Reduction

## Motivation

Existing local reduction methods for microgrid optimal control under uncertainty (e.g., An Efficient Method for the Optimal Control of Microgrids Under Uncertainties using Local Reduction) solve non-convex semi-infinite programs (SIPs) but only guarantee local optimality due to reliance on local NLP solvers. This can lead to suboptimal control policies, especially under worst-case uncertainties, because the local reduction algorithm may converge to a local optimum of the reduced problem or miss the true worst-case scenario. The root cause is that both the master problem and worst-case detection steps use local optimization, which provably fails to certify global optimality for non-convex SIPs.

## Key Insight

By systematically partitioning the variable space with spatial branch-and-bound within the local reduction framework, we can prune suboptimal regions and certify global optimality while leveraging dimension reduction to keep the problem tractable.

## Method

### (A) What it is
**GLOBAL-SIP** is an algorithm for solving non-convex semi-infinite programs arising in microgrid optimal control. It takes as input the SIP formulation with decision variables `x` (control) and uncertain parameters `ξ` (continuous set `Ξ`), and outputs a globally optimal solution `x*` and a finite set of critical scenarios `S` that certify optimality.

### (B) How it works
```pseudocode
Algorithm: GLOBAL-SIP
Input: Non-convex SIP: min_x f(x) s.t. g(x,ξ) ≤ 0 ∀ ξ ∈ Ξ.
Output: Globally optimal x* and scenario set S.

Initialize: S = {ξ0} (initial scenario, e.g., nominal uncertainty).
Iterate:
  1. Solve Master Problem: 
     min_x f(x)  s.t.  g(x,ξ) ≤ 0 ∀ ξ ∈ S,
     using spatial branch-and-bound global solver (e.g., BARON) to obtain (x*, f*).
     Hyperparameter: relative optimality tolerance ε = 1e-4.
  2. Solve Worst-Case Detection:
     For each constraint i, solve: max_{ξ ∈ Ξ} g_i(x*, ξ) using spatial B&B (same solver) to obtain violation v_i* and worst-case ξ_i*.
     Hyperparameter: feasibility tolerance δ = 1e-6.
  3. If max_i v_i* ≤ 0: terminate with x* globally optimal.
  4. Else: add ξ* = argmax v_i* to S, go to Step 1.
```

### (C) Why this design
We chose to use spatial branch-and-bound for both master and worst-case subproblems because it provides guaranteed global optimality, unlike local solvers used in prior local reduction methods (e.g., [First paper]). This design accepts higher computational cost per iteration but ensures that the resulting solution is truly globally optimal for the non-convex SIP. We deliberately use the same global solver for both subproblems to maintain consistency and avoid approximation errors. An alternative would be to use local solvers in the worst-case detection and rely on heuristics, but that would reintroduce the local optimality gap. We accept the trade-off that the algorithm may require more iterations if the global solver is slow, but we mitigate by exploiting the structure: the master problem size grows slowly (number of scenarios), and the worst-case problem is low-dimensional (uncertainty space). We also adapt branching priority based on sensitivity from the master solution to prune faster. Another design decision: we initialize with a single nominal scenario to quickly reach feasibility, accepting that early iterations may add many worst-case scenarios, but this reduces the risk of starting with a sparse set that causes slow convergence.

### (D) Why it measures what we claim
The spatial branch-and-bound solver in the master problem computes a global optimum of the reduced problem, ensuring that the objective value `f*` is a rigorous lower bound for the true SIP optimum (since the reduced problem relaxes constraints by using only a subset `S`). The global worst-case detection solves `max_{ξ} g_i(x*,ξ)` to global optimality; if the maximum violation `v_i*` is ≤ 0, then `x*` is universally feasible, and `f*` equals the objective of `x*`, proving global optimality of `x*` for the full SIP. Thus, the computational quantities "global optimality certificate from B&B" and "worst-case violation computed globally" measure "global optimality for the SIP" because the equivalence rests on the assumption that worst-case detection covers all constraints and all uncertainty. This assumption fails if the worst-case subproblem is not solved to global optimality (e.g., due to premature termination with loose tolerances), in which case the algorithm may incorrectly declare feasibility and output a suboptimal point. To guard against this, we use rigorous B&B with tight tolerances and verify that the termination condition holds for all constraints simultaneously.

## Contribution

(1) A novel algorithm GLOBAL-SIP that integrates spatial branch-and-bound global optimization with the local reduction framework to solve non-convex semi-infinite programs to global optimality. (2) Demonstration that the worst-case detection step can be performed via spatial B&B on the low-dimensional uncertainty space, enabling certification of feasibility and optimality with rigorous guarantees. (3) The first approach to provide global optimality guarantees for non-convex SIPs with existence constraints arising from microgrid optimal control, addressing a key limitation of prior local reduction methods.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Microgrid optimal control with continuous uncertainty | Realistic non-convex SIP test |
| Primary metric | Objective value gap to best known global optimum | Directly measures global optimality |
| Baseline 1 | Scenario approach with fixed large sample | Tests necessity of adaptive scenario selection |
| Baseline 2 | Full discretization of uncertainty | Tests cost vs accuracy tradeoff |
| Baseline 3 | Local reduction with local worst-case solver | Tests importance of global worst-case detection |
| Ablation of ours | GLOBAL-SIP with local worst-case detection | Isolates effect of global worst-case solver |

### Why this setup validates the claim

This experimental design directly tests the central claim that GLOBAL-SIP provides a globally optimal solution for non-convex SIPs arising in microgrid optimal control. The baseline of scenario approach with fixed samples evaluates whether adaptive scenario generation is necessary for achieving global optimality; if the scenario approach fails to find the true optimum, it indicates that a fixed set misses critical constraints. The full discretization baseline assesses the trade-off between computational cost and optimality; if GLOBAL-SIP achieves similar objective value with much fewer scenarios, it validates its efficiency. The local reduction with local worst-case solver baseline isolates the impact of using global optimization for worst-case detection; if it incorrectly declares feasibility, it proves that local solvers are insufficient. The ablation with local worst-case detection within our framework directly measures the contribution of the global worst-case solver. The primary metric, gap to best known global optimum, directly quantifies global optimality—the key claim. Together, these comparisons form a falsifiable test: if GLOBAL-SIP does not outperform the baselines on gap while maintaining reasonable runtime, the claim is refuted.

### Expected outcome and causal chain

**vs. Scenario approach with fixed large sample** — On a case where the uncertainty set contains a narrow region causing constraint violation not captured by the fixed sample, the scenario approach produces a solution that appears feasible but actually violates the constraint, because it only enforces constraints at sampled points. Our method instead detects the worst-case scenario via global optimization and adds it to the master problem, so we expect a noticeable gap in objective value (e.g., our solution yields lower cost because it avoids overly conservative constraints) while the scenario approach remains infeasible or overly conservative.

**vs. Full discretization of uncertainty** — On a case with high-dimensional or continuous uncertainty, full discretization solves an enormous finite optimization that is computationally prohibitive or yields an intractable model, because it enumerates all vertices or a dense grid. Our method instead iteratively adds only critical scenarios, drastically reducing problem size. We expect GLOBAL-SIP to achieve an objective value within 0.1% of the true global optimum (as verified by a verification step) while the full discretization either cannot solve or yields a similar optimum at orders-of-magnitude higher runtime.

**vs. Local reduction with local worst-case solver** — On a case where the worst-case violation occurs at a local optimum of the constraint over uncertainty, the local worst-case solver gets stuck in a local maximum and underestimates the true violation, so the algorithm terminates prematurely with a solution that is not actually feasible. Our method uses global branch-and-bound to find the true worst-case, so it correctly identifies violations and continues. We expect GLOBAL-SIP to converge to the true globally optimal objective value (gap < 1e-4), whereas the local baseline outputs a lower (infeasible) objective value that appears better but violates constraints, leading to a measurable infeasibility in post-hoc Monte Carlo checks.

### What would falsify this idea

If GLOBAL-SIP's objective value gap to the best known global optimum is not significantly smaller than that of the local reduction baseline, or if it fails to certify feasibility on instances where a known global optimum exists, then the claim that global worst-case detection guarantees global optimality is falsified.

## References

1. An Efficient Method for the Optimal Control of Microgrids Under Uncertainties using Local Reduction
2. Semi-Infinite Programs for Robust Control and Optimization: Efficient Solutions and Extensions to Existence Constraints
3. Automatic scenario generation for efficient solution of robust optimal control problems
