# Self-Certifying Value Sequences via Monotone Bellman Residual Reduction with LLM Priors

## Motivation

Existing LLM-based planning methods (e.g., MC-DML, LLM-MCTS) produce plausible but unverified value estimates, lacking formal error bounds. This absence of certification prevents deployment in safety-critical domains. The root cause is that LLMs generate value approximations without a verifiable guarantee on their quality relative to the true optimal value function.

## Key Insight

The Bellman residual can be computed exactly on the set of visited states, enabling a self-verification loop that monotonically reduces the residual to a user-specified threshold by leveraging the LLM as a heuristic proposer.

## Method

### (A) What it is
**CV-LLM** (Certified Value Iteration with LLM) takes an LLM, an environment simulator, and a target Bellman residual bound ε, and outputs a value function estimate V̂ with guaranteed Bellman residual ≤ ε on a set of sampled states.

### (B) How it works
```pseudocode
Input: LLM, simulator, state space S (or subset), ε > 0, step size α ∈ (0,1), max iterations T, batch size N=64
Initialize: V̂0(s) = 0 for all s ∈ S (or LLM-predicted initial values)
Maintain a replay buffer of visited states (size M=500).
For t = 1 to T:
  # Step 1: Sample states and compute current residual
  Sample a batch B of size N:
    - 50% from states with highest δ_t(s) (top-32 from buffer and uniform pool)
    - 30% from replay buffer (random)
    - 20% uniformly from the state space S (if S is finite, else from a known subset)
  For each s ∈ B:
    a* = argmax_a [ R(s,a) + γ * E_{s'} V̂_{t-1}(s') ]  (using simulator for expectations)
    T̂ V̂_{t-1}(s) = R(s,a*) + γ * E_{s'} V̂_{t-1}(s')   # exact Bellman operator on sampled next states
    δ_t(s) = |T̂ V̂_{t-1}(s) - V̂_{t-1}(s)|   # estimated absolute residual
  # Step 2: Propose new value via LLM
  For each s ∈ B with δ_t(s) > ε:
    Prompt = "Given state s, current value V̂_{t-1}(s) (value: {V}), and target value T̂ V̂_{t-1}(s) (value: {T}), propose a new value V̂_new(s) that is closer to the target. Output a number between 0 and V_max."
    V̂_proposed(s) = LLM(Prompt)  # LLM outputs a numeric value in [0, V_max]
  # Step 3: Accept/reject with monotone reduction
  For each s ∈ B:
    V̂_t(s) = V̂_{t-1}(s) if δ_t(s) ≤ ε else (1-α)*V̂_{t-1}(s) + α*V̂_proposed(s)  # convex combination ensures smoothing
  Add all sampled states to replay buffer.
  # (Optional) Update full V̂ by also applying the update to unvisited states via LLM extrapolation
Output: V̂_T
```
Hyperparameters: α = 0.1, ε = 0.05, T = 100, N=64, replay buffer size M=500.

**Explicit assumption:** The certification claim (Bellman residual ≤ ε) holds on the set of states that have been sampled during training, provided that the sampling distribution covers the state space sufficiently. We enforce coverage by including uniform samples (20% of batch) and a replay buffer. However, no formal guarantee extends to all unseen states; we measure generalization empirically via held-out states.

### (C) Why this design
We chose a convex combination update (accept/reject) over a hard replacement (directly setting V̂_t(s)=V̂_proposed(s)) because replacement can destabilize the residual if the LLM proposes an extreme value; the convex combination preserves monotonic decay of the residual by ensuring the new value lies between the old and the target, at the cost of slower convergence. We chose to compute the Bellman operator T̂ exactly (via simulation) rather than approximate via another LLM because exact computation is necessary for a certified bound; the trade-off is higher computational cost per iteration but a verifiable guarantee. We chose to use an LLM to propose updates rather than classical gradient descent because the LLM can leverage semantic knowledge of the state to suggest intelligent updates that reduce the number of iterations needed, accepting the possibility of occasional misleading proposals that are dampened by the convex combination. Finally, we sample states with high residual (error-driven sampling) combined with uniform and replay buffer sampling to focus computation where it is most needed while maintaining coverage, risking bias toward overfitting on high-residual states but accelerating overall convergence.

### (D) Why it measures what we claim
The computational quantity δ_t(s) = |T̂ V̂_{t-1}(s) - V̂_{t-1}(s)| measures the Bellman residual, which directly quantifies how far V̂ is from satisfying the Bellman equation. The assumption linking δ_t(s) to approximation error (distance to the optimal value V*) is that a small Bellman residual implies a small L∞ error (by standard error bounds, e.g., V* - V̂ ≤ (2γ/(1-γ)) max_s δ(s)). This assumption fails when the Bellman residual is computed on a subset of states that are not representative of the entire state space, in which case the bound holds only on the sampled states and may not generalize. The convex combination update V̂_t(s) = (1-α)V̂_{t-1}(s) + α V̂_proposed(s) measures a monotonic residual reduction rate: when V̂_proposed(s) is closer to T̂ V̂_{t-1}(s) than V̂_{t-1}(s) is, the residual decreases, and due to the convex combination the new value is guaranteed to reduce the residual (by at least α times the improvement). The assumption here is that the LLM proposal moves toward the target; this assumption fails when the LLM proposes a value moving away from the target, in which case the update still ensures a reduction because the convex combination pulls the value toward the target by factor α. The termination condition δ_t(s) ≤ ε for all sampled s measures achievement of the target residual bound on the sampled states; we additionally measure generalization on held-out states. This assumption fails when high-residual states are systematically missed, which we mitigate by including uniform samples.

## Contribution

(1) A novel algorithm CV-LLM that interleaves LLM-based value proposals with exact Bellman residual verification to produce a value function with a certified residual bound. (2) A design principle for ensuring monotonic residual reduction by combining LLM proposals with a convex combination accept/reject scheme. (3) A demonstration that the LLM's semantic priors can accelerate convergence compared to naive iterative updates.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Text-based games (Jericho) | Challenging large state space with simulator |
| Primary metric | Max Bellman residual on held-out states (1000 states uniformly sampled) | Directly measures certified bound achievement |
| Baseline 1 | Monte Carlo Planning + LLM (MCP+LLM) | Planning without certified guarantee |
| Baseline 2 | LLM-based PSRL (Exploration) | Exploration-focused method without certification |
| Ablation-of-ours | CV-LLM without LLM proposals (VI only) | Isolates effect of LLM on convergence |

### Why this setup validates the claim

The central claim is that CV-LLM achieves a certified Bellman residual bound on a set of sampled states. Using text-based games from Jericho provides an environment with a large, structured state space where exact value iteration is intractable but a simulator enables exact Bellman updates on sampled states. The primary metric directly measures the residual on held-out states (1000 uniformly sampled, never used for training), testing generalization of the bound. We include a diagnostic: if the held-out residual exceeds ε, we report certification only on training states. The baselines cover alternative approaches: MCP+LLM representative of planning-based methods without certification, and LLM-based PSRL representing exploration-driven methods with approximate value estimates. The ablation removes LLM proposals, testing whether the LLM accelerates convergence beyond standard value iteration. This combination forms a falsifiable test: if CV-LLM's residual is not lower or does not generalize as expected, the central claim is weakened.

### Expected outcome and causal chain

**vs. Monte Carlo Planning + LLM** — On a case where a state has high long-term reward but weak immediate signal, MCP+LLM's shallow rollouts yield a noisy value estimate, causing high residual. Our method instead computes the exact Bellman target via simulation and uses the LLM to propose a refined value, then softly updates, ensuring monotonic residual reduction. We expect CV-LLM to achieve significantly lower max residual, especially on such sparse-reward states, with a gap of at least 0.2 in residual magnitude on held-out states.

**vs. LLM-based PSRL** — On a case where state semantics are ambiguous (e.g., multiple plausible interpretations), PSRL's posterior may mis-specify dynamics, leading to a value estimate far from the fixed point. Our method directly minimizes the Bellman residual by querying the LLM to correct the value toward the exact target, avoiding posterior misspecification. We expect CV-LLM to have a smaller maximum residual (by >0.15) on states where PSRL's posterior is inaccurate, while parity on well-specified states.

**vs. CV-LLM without LLM proposals (VI only)** — On a large state space, standard value iteration updates only visited states and converges slowly; many states remain with initial zero value, causing high residual. With LLM proposals, the LLM extrapolates from visited states to unvisited ones, accelerating residual reduction. We expect CV-LLM to show faster decrease in average residual (by at least 2× per iteration) and reach the target bound in fewer iterations.

### What would falsify this idea

If CV-LLM's residual bound on held-out states is not significantly smaller than that of MCP+LLM or LLM-based PSRL, or if the improvement is uniform across all states rather than concentrated where previous methods fail (e.g., sparse-reward or ambiguous states), then the claimed certification advantage is not supported. Additionally, if the held-out residual systematically exceeds ε, the certification claim must be restricted to training states.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
3. Multi-skill Mobile Manipulation for Object Rearrangement
4. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
5. Inner Monologue: Embodied Reasoning through Planning with Language Models
6. Toward Efficient Exploration by Large Language Model Agents
7. Isoperimetry is All We Need: Langevin Posterior Sampling for RL with Sublinear Regret
8. Efficient Reinforcement Learning with Large Language Model Priors
