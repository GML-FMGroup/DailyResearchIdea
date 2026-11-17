# Feasibility-Certified Flow Matching for Budget-Constrained Portfolio Generation

## Motivation

End-to-end generative portfolio methods, exemplified by AllocFlow, removed the explicit optimization layer that classically enforced budget and sum-to-one constraints, replacing exact feasibility with reward penalties that do not guarantee the generated allocation distribution is closed under shared-capital constraints. This structural gap recurs across generative allocation pipelines: the sampler can drift off the budget-feasible manifold at any integration step, and no mechanism-level certificate exists to detect or correct it. The root cause is that flow-matching training optimizes trajectory likelihood and portfolio rewards, neither of which is a constraint-satisfaction signal over the terminal allocation.

## Key Insight

Composing a differentiable simplex projection into every flow-matching integration step makes the terminal allocation lie on the budget-feasible manifold by construction, so feasibility becomes an invariant of the sampler rather than a property hoped for from reward shaping.

## Method

**What it is.** We introduce **CFM-Port** (Constrained Flow Matching for Portfolios), a flow-matching sampler whose velocity field is composed with a differentiable projection onto the shared-capital simplex at every integration step, producing signed position weights that satisfy sum-to-one and budget closure by construction. Inputs are raw multi-asset market histories; outputs are joint allocation trajectories whose terminal states are certified feasible.

**Load-bearing assumption (stated explicitly).** The projection operator is an exact differentiable Euclidean projection onto the budget-constrained simplex (Duchi et al. 2008; Wang & Carreira-Perpinan 2013), applied to a signed, budget-shifted variable, so that the projected state satisfies sum_i z_{k+1,i} = b exactly up to floating-point round-off (≤ 1e-12 in float64). The certificate c_k is defined against the pre-projection velocity sum, c_k = || sum_i z_{k+1/2,i} - b ||_1, so it measures the drift the projection had to correct rather than the projection's own normalization residual. Because the projection is exact, the terminal allocation lies on the budget-feasible manifold by algebraic construction, and the certificate is non-tautological: it is zero only when the unconstrained velocity step already lands on the budget hyperplane.

**How it works.**

```pseudocode
Input: market history x_{1:T} (N assets), capital budget b, step count K=20,
       certificate tolerance eps=1e-4, box bounds [l_i, u_i] for signed weights

1. Encode temporal market representation h = CrossAssetTransformer(x_{1:T})
2. Initialize z_0 ~ N(0, I^{N})  # signed weight prior
3. For k = 0..K-1:
     a. v_k = FlowVelocity(z_k, h; theta)          # unconstrained velocity
     b. z_{k+1/2} = z_k + (1/K) * v_k               # Euler step
     c. z_{k+1} = Proj_{Simplex∩Box}(z_{k+1/2}; b, l, u)  # exact differentiable projection
     d. record certificate c_k = || sum_i z_{k+1/2,i} - b ||_1
4. Terminal allocation w = z_K
5. Feasibility certificate C = max_k c_k ; accept iff C < eps

Training:
  Stage 1 (imitation): minimize L_imit = E[ || w - w_strategy ||^2 ]
     over strategy-supervised portfolio targets, with exact projection applied
     at every forward pass.
  Stage 2 (RL): group-relative policy gradient with reward
     R = R_financial(w) + lambda * (1 - C / b),  lambda = 0.5
     where R_financial is realized Sharpe over the episode and
     the frozen reference policy is the Stage-1 checkpoint.
```

The projection `Proj_{Simplex∩Box}` is the exact Euclidean projection onto the intersection of the budget hyperplane {z : sum_i z_i = b} and box constraints [l_i, u_i], computed by the sorted-threshold algorithm of Duchi et al. (2008) extended to signed variables via the budget shift y = z - b/N, followed by a box-clipping pass and a re-normalization to enforce sum_i z_i = b exactly. The projection is piecewise-linear and differentiable almost everywhere; gradients flow through the active-set selection, and the zero-gradient issue on the boundary is handled by the box constraints keeping the solution interior to the feasible set during training.

**Why this design.** We chose to embed the exact projection inside the integration loop rather than apply it only at the terminal step because drift accumulated across Euler steps can push intermediate states arbitrarily far from the feasible manifold, and a terminal-only projection would mask infeasibility that the velocity field was never trained to avoid; the trade-off is that projection at every step adds K extra sorting passes, increasing wall-clock cost roughly linearly in K. We chose an exact differentiable Euclidean projection over a softmax-temperature surrogate because a softmax maps to the positive probability simplex, cannot produce signed weights, and its output sums to b by construction, making the certificate tautological; the exact projection preserves signed weights and yields a non-tautological certificate measured against the pre-projection velocity sum. We chose to augment the RL reward with an explicit feasibility certificate rather than rely on portfolio-level financial rewards alone because financial rewards are insensitive to small budget violations that compound under leverage; the cost is that lambda must be tuned to avoid the certificate term dominating the Sharpe signal during early training. This differs from AllocFlow, which uses a reward penalty without a projection operator and therefore cannot guarantee closure; a domain expert would not describe CFM-Port as a variant of AllocFlow because the feasibility guarantee comes from the sampler's algebraic structure, not from reward shaping.

**Why it measures what we claim.** The certificate `c_k = || sum_i z_{k+1/2,i} - b ||_1` measures capital drift because the L1 norm of the deviation of the pre-projection velocity step from the budget is exactly the budget violation that the projection must correct; this assumption fails when the velocity field produces extreme signed weights whose sum is far from b and the projection's active-set changes between steps, in which case c_k reflects large velocity drift rather than projection residual. The terminal acceptance criterion `C < eps` measures distributional feasibility because the maximum over all integration steps bounds the worst-case drift along any sampled trajectory under the assumption that the Euler discretization error is dominated by the velocity drift; this assumption fails when K is too small and discretization error exceeds eps, in which case C reflects solver error rather than sampler drift. The reward term `lambda * (1 - C/b)` measures the degree to which the policy has learned to stay on the feasible manifold because C is monotone in the distance of the pre-projection velocity sum from the budget hyperplane under the assumption that the velocity network's outputs are bounded; this assumption fails under adversarial market inputs that produce extreme velocities, in which case the reward term reflects input magnitude rather than policy feasibility.

## Contribution

['A constrained flow-matching sampler (CFM-Port) that composes a differentiable simplex projection into every integration step, making budget closure an algebraic invariant of the generative trajectory rather than a reward-shaped preference.', 'A feasibility-certificate-augmented RL objective that couples the projection residual to the training signal, yielding a design principle: generative allocators should expose their constraint residuals as first-class training quantities rather than hiding them in penalty terms.', 'An evaluation protocol that measures capital closure across all sampled trajectories (max-over-steps certificate distribution) rather than only terminal weights, providing a diagnostic for feasibility drift in end-to-end portfolio generators.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-asset equity histories with leverage regimes (Dow-Jones 30 constituents, 30 years; sector ETFs, 20 years) | Exercises budget-violation triggers and regime shifts |
| Primary metric | Budget-closure violation rate under leverage, measured as fraction of episodes with C ≥ eps=1e-4 | Directly tests the feasibility guarantee claim |
| Secondary metric | Realized Sharpe ratio and maximum drawdown over the evaluation period | Tests whether closure translates into risk-adjusted advantage |
| Baseline | AllocFlow (RL, no projection) | Tests whether algebraic closure beats reward shaping |
| Baseline | Markowitz / Black-Litterman optimizer | Tests closure and stability versus classical construction |
| Baseline | Softmax-output allocator (terminal-only projection) | Tests whether in-loop projection matters |
| Ablation-of-ours | CFM-Port without in-loop projection | Isolates the contribution of per-step projection |
| Ablation-of-ours | CFM-Port with certificate measured on post-projection state (tautological variant) | Demonstrates the non-tautological certificate is necessary |

### Why this setup validates the claim

The central claim is that composing the velocity field with an exact differentiable simplex projection at every integration step yields allocations that are feasible by construction, and that this structural guarantee is what converts into better risk-adjusted outcomes under leverage. To falsify this, we need a metric that isolates feasibility from raw performance: budget-closure violation rate under leverage measures exactly the quantity the design claims to control, and it can be measured on any baseline without reward-shaping confounds. AllocFlow tests the sub-claim that reward penalties without a projection operator cannot guarantee closure — if AllocFlow matches our closure rate, the algebraic-structure argument collapses. The Markowitz/Black-Litterman baseline tests whether classical construction, which also enforces budget constraints but through convex solvers, achieves comparable closure with less turnover; if it does, our sampler adds no feasibility value. The terminal-only softmax baseline tests the specific sub-claim that in-loop projection is necessary rather than terminal projection sufficient. The no-projection ablation removes our mechanism entirely, so any gap between it and full CFM-Port is attributable to the per-step projection rather than to the flow-matching backbone. The tautological-certificate ablation confirms that measuring c_k on the pre-projection velocity sum is necessary for the certificate to reflect drift rather than normalization residual. Together these baselines bracket the claim from both directions: one side tests whether structure beats reward shaping, the other tests whether per-step projection beats terminal projection, and the ablations confirm the causal attribution.

### Expected outcome and causal chain

**vs. AllocFlow** — On a leveraged episode where the market representation shifts abruptly mid-episode, AllocFlow produces signed weights whose sum drifts from the budget because its reward penalty only weakly discourages violations and its policy has no algebraic closure. Our method instead projects onto the simplex at every Euler step, so intermediate states cannot drift far from the feasible manifold. We expect a visible gap in violation rate concentrated in high-volatility, leverage-heavy episodes, with near-parity in calm regimes.

**vs. Markowitz / Black-Litterman optimizer** — On a case where estimated covariances are noisy and the optimizer concentrates weight on a few assets near the simplex boundary, the classical solver produces feasible but unstable weights that flip between rebalances because the hard projection has no gradient signal to regularize the boundary. Our method instead uses an exact projection with box constraints that keeps gradients alive at the boundary, yielding stable trajectories. We expect comparable closure but noticeably lower turnover and steadier weight trajectories across rebalances.

**vs. Softmax-output allocator (terminal-only projection)** — On a case where the velocity field accumulates drift over K steps, the terminal-only baseline masks intermediate infeasibility and its velocity network never learns to avoid the boundary, so terminal projection must absorb a large correction. Our method instead trains the velocity field to stay near-feasible at every step, so the terminal correction is small. We expect a clear gap in max-over-steps certificate values, with the largest divergence at larger K and stronger drift.

**vs. CFM-Port with tautological certificate** — On a case where the velocity field produces large pre-projection drift, the tautological variant reports near-zero c_k because the post-projection state sums to b by construction, masking the drift. Our method instead measures c_k on the pre-projection velocity sum, so the certificate reflects the drift the projection had to correct. We expect the tautological variant to show artificially low violation rates that do not correlate with realized budget violations, while our method's certificate tracks the true drift.

### What would falsify this idea

If CFM-Port's closure advantage over AllocFlow is uniform across calm and leveraged regimes rather than concentrated where reward-penalty violations should occur, the structural-guarantee argument is wrong. If the no-projection ablation matches full CFM-Port on closure and turnover, the per-step projection contributes nothing and the central mechanism claim fails. If the tautological-certificate ablation matches full CFM-Port on closure and turnover, the pre-projection certificate adds no information and the non-tautological claim fails.

## References

1. AllocFlow: End-to-End Portfolio Trading Action Generation from Raw Market Data
