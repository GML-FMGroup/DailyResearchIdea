# Distributionally Robust Local Reduction for Microgrid Optimal Control under Distributional Shift

## Motivation

Existing local reduction methods for microgrid optimal control, such as the approach in 'An Efficient Method for the Optimal Control of Microgrids Under Uncertainties using Local Reduction', assume a fixed parametric uncertainty distribution and thus lack robustness to distributional shift. This structural limitation arises because the scenario generation process is tied to a single nominal distribution, causing solutions that are optimal under that distribution to become infeasible or suboptimal when the true distribution deviates. We address this by extending the local reduction framework with distributionally robust optimization (DRO).

## Key Insight

The worst-case scenarios under a distributional ambiguity set can be identified by solving a tractable inner optimization problem that exploits the convexity of the ambiguity set, allowing the local reduction algorithm to iteratively add scenarios that guarantee robustness over all distributions in the set.

## Method

### (A) What it is
**Distributionally Robust Local Reduction (DRLR)** is an iterative algorithm that extends the local reduction method for microgrid optimal control by constructing a Wasserstein ambiguity set around the nominal parametric distribution. Its inputs are the optimal control problem, nominal distribution, ambiguity radius ε, and an initial scenario set. Its output is a control policy robust to all distributions within the ambiguity set.

### (B) How it works
```python
# DRLR Algorithm
Input: Nominal distribution P0, ambiguity radius ε, initial scenario set S0 (e.g., samples from P0), tolerance δ, linearization tolerance δ_lin=1e-3 for approximating non-convex constraints
converged = False
S = S0
while not converged:
    # Master problem: solve deterministic equivalent with current scenarios S
    x = solve_master(S)  # MILP or NLP as in base paper
    new_scenarios = []
    for each constraint j in problem:
        # Linearize constraint around nominal mean scenario ξ0 (e.g., mean of P0)
        # g_j(x, ξ) ≈ g_j(x, ξ0) + ∇_ξ g_j(x, ξ0)^T (ξ - ξ0)
        # Worst-case subproblem: maximize linearized violation over distributions Q within Wasserstein ball W_ε(P0)
        # Formulation: max_{Q in W_ε(P0)} E_Q[ max(0, g_j(x, ξ) - 0) ] (where 0 is constraint bound)
        # Using dual of Wasserstein DRO, this reduces to solving a convex projection problem with the linearized constraint.
        violation, worst_scenario = solve_worst_case_linearized(P0, ε, x, j, ξ0)
        if violation > δ:
            new_scenarios.append(worst_scenario)
    if len(new_scenarios) == 0:
        converged = True
    else:
        S = S ∪ new_scenarios
return x
```
The function `solve_worst_case_linearized` uses the dual formulation of Wasserstein DRO with linearized constraints. For a given nominal distribution P0 (represented as a discrete set of scenarios with probabilities), the Wasserstein ball of radius ε is defined via the L1 metric. The worst-case violation for constraint j is computed by solving a finite-dimensional optimization over the scenarios and an additional variable representing distribution shift, which becomes a linear program due to linearization. The new scenario is the point in the uncertainty space that achieves the maximum violation under the worst-case distribution. This design assumes the constraint functions are convex or can be locally linearized such that the worst-case violation under the Wasserstein ball is tractable. If constraints are highly non-convex, the linearization may lead to optimistic robustness; in practice, we validate on a held-out set.

### (C) Why this design
We chose DRO over a simple robust worst-case method (e.g., robust optimization over a bounded set) because DRO maintains probabilistic feasibility under distributional shift while being less conservative, accepting the trade-off that the bi-level structure increases computational overhead per iteration. We selected the Wasserstein ambiguity set over KL-divergence because Wasserstein applies to discrete scenarios and naturally handles out-of-sample support (e.g., extreme renewable generation), whereas KL requires absolute continuity and can miss events with zero nominal probability. We decided to add scenarios iteratively (as in classic local reduction) rather than solving a monolithic DRO problem upfront, because the monolithic problem grows prohibitively with scenario size; the iterative approach exploits the fact that only a few critical scenarios drive robustness. Unlike the original local reduction which finds worst-case scenarios under a fixed distribution, our method identifies worst-case scenarios under the entire ambiguity set, requiring a fundamentally different subproblem that accounts for distributional perturbation via a penalty term derived from the Wasserstein distance. This design ensures that the scenario set captures distributional shift rather than just nominal extremes.

### (D) Why it measures what we claim
The worst-case violation computed in `solve_worst_case_linearized` (by maximizing over the Wasserstein ball) measures **distributional robustness** because it guarantees feasibility for every distribution Q within distance ε of P0; this equivalence relies on the assumption that the linearized constraint accurately represents the true constraint violation under the worst-case distribution. This assumption fails when the linearization error is large (e.g., highly non-convex constraints), in which case the solution may be over-conservative or still violate constraints. The master problem's optimal objective value with the accumulated scenario set measures **worst-case expected cost** over distributions represented by the scenarios; this measures robustness because the iterative process ensures no scenario outside the set yields higher violation—this relies on the convergence of the local reduction method, which assumes that the worst-case subproblem is solved globally (we accept local optimality in practice). The tolerance δ controls the trade-off between scenario count and violation tightness; it measures **approximate robustness** because smaller δ yields stricter feasibility at the cost of more scenarios. A failure mode arises if the linearized subproblem underestimates the true worst-case violation, leading to optimistic robustness guarantees. To mitigate, we validate on a held-out set of distributions.

## Contribution

(1) A novel algorithm DRLR that integrates distributionally robust optimization with local reduction for optimal control under distributional shift, specifically targeting microgrids with parametric uncertainty. (2) Empirical demonstration on a microgrid case study that DRLR achieves robust feasibility with significantly fewer scenarios than a full DRO discretization, and maintains performance under distributional shift where standard local reduction yields constraint violations. (3) Characterization of the trade-off between the Wasserstein ambiguity radius ε and solution conservatism, providing guidance for selecting ε based on available historical data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | CIGRE low-voltage microgrid with solar & load; distribution shift simulated by using summer data for training (nominal) and winter data for testing (shifted) | Captures renewable variability and load uncertainty; shift due to seasonal weather patterns |
| Primary metric | Worst-case constraint violation probability under distributional shift, computed via 1000 Monte Carlo samples from the shifted distribution | Directly measures distributional robustness |
| Baseline 1 | Nominal MPC (no robustness) | Baseline without any robustness |
| Baseline 2 | Robust optimization (fixed interval bounds: ±30% of nominal) | Classical robust method, potentially over-conservative |
| Baseline 3 | Standard local reduction (LR) using nominal distribution only | Ablates distributional robustness component |
| Baseline 4 | Scenario-based robust optimization (worst-case over a fixed set of 500 scenarios uniformly sampled from the Wasserstein ball of radius ε) | Compares against a different robust approach that does not use iterative scenario addition |
| Ablation 1 of ours | DRLR with ε=0 (no ambiguity set) | Isolates effect of Wasserstein ambiguity set |
| Ablation 2 of ours | DRLR with full non-linear constraint solving in subproblem (using local NLP solver rather than linearization) | Isolates the effect of linearization on robustness guarantees |

Hyperparameter tuning: ε is selected via 5-fold cross-validation on a validation set of 1000 scenarios from a held-out season (spring), using grid search over {0.01, 0.05, 0.1, 0.2} times the empirical Wasserstein distance between nominal and validation sets; δ is set to 0.001. Baselines use tuned parameters where applicable.

### Why this setup validates the claim
The CIGRE microgrid benchmark with solar and load uncertainty provides realistic scenarios for distributional shift (e.g., cloud cover reducing solar yield). Using seasonal shift (summer to winter) ensures a natural, non-synthetic distribution perturbation. The primary metric—worst-case violation probability under a perturbed distribution—directly tests the central claim of distributional robustness. Comparing against Nominal MPC reveals the cost of ignoring uncertainty; Robust Optimization (fixed intervals) tests conservatism; Standard LR tests value of adding distributional robustness; Scenario-based robust optimization tests whether the iterative selection of scenarios in DRLR is beneficial over a static robust set. Ablation 1 (ε=0) shows whether performance gain stems from the ambiguity set or other algorithmic features; Ablation 2 shows whether linearization impacts the robustness guarantee. Together, these form a falsifiable test: if DRLR outperforms LR only on ambiguous shifts, the claim is supported.

### Expected outcome and causal chain

**vs. Nominal MPC** — On a case where actual solar generation is consistently lower than nominal (e.g., winter overcast), Nominal MPC violates power balance constraints because it relies on the nominal scenario. Our method instead anticipates such shifts via the Wasserstein ball, so we expect DRLR to incur negligible violation (<0.1%) while Nominal MPC shows >5% violation rate under distributional shift.

**vs. Robust Optimization** — On a case with moderate but not extreme uncertainty (e.g., 20% solar variance), Robust Optimization uses fixed wide bounds, leading to overly conservative control and high operational cost (e.g., 15% higher cost). Our method uses DRO to tighten the worst-case distribution, so we expect DRLR to achieve 10-20% lower cost while maintaining feasibility (violation <0.1%).

**vs. Standard Local Reduction** — On a case where the worst-case scenario under the nominal distribution (e.g., low solar) is not the worst under distributional shift (e.g., high correlation between solar and load), LR misses critical scenarios. Our method identifies scenarios from the ambiguity set via the worst-case subproblem, so we expect DRLR to reduce violation probability by half (from 2% to <1%) under shifted distributions.

**vs. Scenario-based robust optimization** — On a case with distributional shift, scenario-based robust optimization with a fixed set of 500 scenarios will either be over-conservative (if the set is too wide) or miss critical shifts (if too narrow). DRLR selects scenarios iteratively, thus achieving lower cost (e.g., 10% lower) and comparable violation (<0.5%) than scenario-based robust optimization with the same ε.

### What would falsify this idea
If DRLR does not significantly reduce violation probability under distributional shift compared to Standard LR (e.g., both have similar violation rates >2%), or if the improvement from Ablation 2 shows that linearization causes a large increase in violation (>1%) compared to the full non-linear subproblem, then the central claim that the Wasserstein ambiguity set provides meaningful robustness via tractable linearization is falsified.

## References

1. An Efficient Method for the Optimal Control of Microgrids Under Uncertainties using Local Reduction
2. Semi-Infinite Programs for Robust Control and Optimization: Efficient Solutions and Extensions to Existence Constraints
3. Automatic scenario generation for efficient solution of robust optimal control problems
