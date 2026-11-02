# Intervention-Identifiable Credit Decomposition from Team-Level Human Feedback

## Motivation

Across multiple paper-trees, team- or episode-level feedback is treated as a sufficient training signal without establishing which per-agent credit decompositions are recoverable from it: R2BC replaces per-robot teleoperation with round-robin team feedback, and SWEET-RL replaces a privileged step-level critic with an episode-level signal, yet neither characterizes the equivalence class of per-agent rewards that produce the identical team signal. The structural root cause is that the credit-aggregation operator (mapping per-agent rewards to the team scalar) is never perturbed, so observationally indistinguishable decompositions cannot be separated and learned per-agent credit is fundamentally underdetermined.

## Key Insight

Applying controlled, reversible interventions to the credit-aggregation operator breaks the observational symmetry among per-agent reward decompositions, so the recoverable equivalence class is exactly the kernel of the intervention-response matrix rather than the full decomposition space.

## Method

**What it is.** CREDIT-ID is a decentralized multi-robot training procedure that first identifies which per-agent credit decompositions are distinguishable under team-level human feedback, then trains a credit model whose loss is invariant only within that identified equivalence class. Inputs: trajectories of joint robot behavior, team-level scalar human feedback (preferences or ratings), and a controllable credit-aggregation operator g_θ (a differentiable map from per-agent reward vector r to team scalar, parameterized by θ). Outputs: per-agent reward model r̂_i and decentralized policies trained on r̂.

**How it works.**

```pseudocode
# Phase 1: Intervention design on the aggregation operator
Input: trajectory buffer D, aggregation family {g_θ}, intervention budget K
for k in 1..K:
    sample joint trajectory τ ~ D
    sample per-agent reward perturbation δ ~ {+ε, -ε}^N  (reversible, zero-mean)
    query team feedback y_base = h(τ)   # human scalar
    query team feedback y_int  = h(τ, g_θ(r + δ))  # same trajectory, perturbed aggregation
    record response s_k = (y_int - y_base) / ε       # finite-difference sensitivity
build response matrix S ∈ R^{K×N}, S[k,i] = ∂y/∂r_i under intervention k

# Phase 2: Identifiability certificate
compute rank ρ = rank(S) and null space N(S)   # kernel = indistinguishable directions
if ρ < N:  flag agents/directions in N(S) as unidentifiable under current g_θ
design next intervention batch (greedy max |det S_sub| over candidate perturbations)

# Phase 3: Invariant training objective
for each training step:
    predict r̂ = f_φ(τ)                       # per-agent credit model
    team_pred = g_θ(r̂)
    L_team = ||team_pred - y||^2             # fit team signal
    L_inv  = Σ_{v ∈ N(S)} || v^T r̂ ||^2       # penalize only unidentifiable directions
    L = L_team + λ * L_inv + μ * ||r̂||^2
    update φ, θ via gradient descent
```

Hyperparameters: perturbation magnitude ε (set to 0.1× empirical reward std), intervention budget K (200–500 queries), λ=1.0, μ=1e-3, N = number of robots (5–20).

**(D) Assumptions and failure modes.** The response matrix S measures the causal sensitivity of team feedback to each agent's credit because the reversibility of δ makes the finite difference equal the partial derivative under the assumption that the human feedback function h is locally linear in the aggregation input g_θ(r) over the ε-ball; this assumption fails when feedback is thresholded or ordinal (e.g., binary preferences), in which case S reflects a smoothed surrogate of a step function rather than a true gradient. The null-space penalty L_inv measures the unidentifiability of a credit direction because projecting r̂ onto N(S) captures exactly the components that leave team feedback unchanged under all interventions, under the assumption that the intervention family spans the relevant perturbation space and S is full-rank-estimated; this assumption fails when the budget K is too small to cover high-order interactions, in which case L_inv reflects sampling noise rather than genuine ambiguity. The rank ρ measures the number of recoverable credit degrees of freedom because rank equals the dimension of the identifiable subspace under the same linearity assumption; it fails when agents' rewards are functionally coupled in g_θ, in which case ρ undercounts recoverable structure.

**Why this design.** We chose reversible zero-mean perturbations over random reward resampling because reversibility lets us estimate the local Jacobian ∂y/∂r_i via finite differences while keeping the trajectory fixed, isolating the aggregation operator as the only varying factor; the cost is that we only recover local (first-order) identifiability, not global structure. We chose to derive the null space N(S) explicitly and penalize only within it, rather than regularizing all directions uniformly, because uniform regularization shrinks identifiable credit and destroys the very signal we want; the cost is sensitivity to rank estimation error when S is near-singular. We chose a separate differentiable g_θ over a fixed known aggregation, because human feedback is often given under an unknown (possibly biased) aggregation; the cost is that θ must be co-estimated, which we handle by alternating updates. A domain expert would not describe this as a variant of R2BC or SWEET-RL because neither perturbs the aggregation operator nor computes an intervention-response kernel: those methods fix the coarse-graining and hope fine credit survives, whereas CREDIT-ID measures where it survives and constrains training accordingly.

**Why it measures what we claim.** The response matrix S measures the causal sensitivity of team feedback to each agent's credit because the reversibility of δ makes the finite difference equal the partial derivative under the assumption that the human feedback function is locally linear in the aggregation input; this assumption fails when feedback is thresholded or ordinal (e.g., binary preferences), in which case S reflects a smoothed surrogate of a step function rather than a true gradient. The null-space penalty L_inv measures the unidentifiability of a credit direction because projecting r̂ onto N(S) captures exactly the components that leave team feedback unchanged under all interventions, under the assumption that the intervention family spans the relevant perturbation space; this assumption fails when the budget K is too small to cover high-order interactions, in which case L_inv reflects sampling noise rather than genuine ambiguity. The rank ρ measures the number of recoverable credit degrees of freedom because rank equals the dimension of the identifiable subspace under the same linearity assumption; it fails when agents' rewards are functionally coupled in g_θ, in which case ρ undercounts recoverable structure.

## Contribution

(1) CREDIT-ID, a training framework that couples intervention-based identification of the credit-aggregation operator with an equivalence-class-invariant credit loss for decentralized multi-robot learning from team-level human feedback. (2) A design principle: the recoverable per-agent credit is exactly the row space of the intervention-response matrix, so intervention design (which perturbations to query) directly controls what credit is learnable, yielding a greedy determinant-maximizing intervention schedule. (3) An open diagnostic tool that reports, per robot, whether its credit is identifiable under a given feedback budget, converting the hidden sufficiency assumption of R2BC and SWEET-RL into a measurable quantity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Simulated multi-robot foraging with synthetic team feedback | Ground-truth per-agent credit enables identifiability checks |
| Primary metric | Per-agent credit recovery error in identifiable subspace | Directly tests whether invariant training preserves recoverable credit |
| Baseline | R2BC-style fixed-aggregation credit fitting | Tests claim that ignoring aggregation kernel loses credit |
| Baseline | Uniform-regularization credit model | Tests claim that blanket shrinkage destroys identifiable signal |
| Baseline | Oracle-aggregation training (known g) | Tests whether co-estimating g_θ costs much |
| Ablation-of-ours | CREDIT-ID without null-space penalty | Isolates contribution of identifiability certificate |

### Why this setup validates the claim

The central claim is that credit is only recoverable along directions the team-level feedback actually distinguishes, and that penalizing exactly the null space (rather than all directions) preserves the recoverable part. The simulated foraging dataset is essential because it supplies ground-truth per-agent rewards, so we can measure credit recovery error *restricted to the identifiable subspace* rather than against an unattainable full-vector target. The R2BC-style baseline tests the sub-claim that fixing the aggregation and hoping fine credit survives fails when the true aggregation is biased; the uniform-regularization baseline tests the sub-claim that blanket shrinkage destroys identifiable signal; the oracle-aggregation baseline bounds how much co-estimating g_θ costs. The no-null-space ablation isolates whether the identifiability certificate itself, rather than just having a flexible g_θ, produces the gain. The subspace-restricted error metric is the right detector because the predicted effect is a *concentration* of improvement where the response matrix has high rank—a uniform-metric comparison would wash that out.

### Expected outcome and causal chain

**vs. R2BC-style fixed-aggregation credit fitting** — On a case where the human aggregates credit with a mild bias toward high-activity robots, the baseline fits per-agent rewards through a fixed known operator, so it absorbs the bias into individual credit and misattributes team reward to the active robot even when a quiet robot caused the outcome. Our method instead estimates the aggregation response matrix and learns which credit directions the biased feedback cannot see, so it does not chase the biased component. We expect a clear gap on the biased-aggregation subset and near-parity when the aggregation is unbiased.

**vs. Uniform-regularization credit model** — On a case where several agents contribute weakly but distinguishably, the uniform-regularization baseline shrinks all per-agent credit toward zero because it penalizes every direction equally, erasing the weak-but-recoverable signal. Our method penalizes only the null-space directions, leaving the identifiable weak components untouched. We expect the uniform baseline to show inflated error concentrated on low-magnitude-credit agents while our method stays close to oracle on those same agents, with both comparable on high-magnitude agents.

**vs. Oracle-aggregation training (known g)** — On a case where the true aggregation is known, the oracle baseline has no θ estimation error, so it should be the strongest competitor and our method can only match it. Our method co-estimates g_θ, which introduces some variance, so we expect our method to track the oracle closely when the intervention budget is adequate and to degrade gracefully (not catastrophically) when K is reduced—demonstrating that the identifiability machinery, not perfect aggregation knowledge, carries the gain.

### What would falsify this idea

If the improvement over the R2BC-style and uniform-regularization baselines is uniform across biased- and unbiased-aggregation subsets and across high- and low-rank response-matrix conditions, rather than concentrated where the response matrix is near-singular, then the null-space mechanism is not what produces the gain and the central claim is wrong. Equivalently, if the no-null-space ablation matches our method everywhere, the identifiability certificate adds nothing.

## References

1. Multi-Robot Learning from Human Feedback
2. Discovery and Deployment of Emergent Robot Swarm Behaviors via Representation Learning and Real2Sim2Real Transfer
3. Zespol: A Lightweight Environment for Training Swarming Agents
4. Optimising autonomous robot swarm parameters for stable formation design
5. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
6. Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks
7. Training Language Models to Self-Correct via Reinforcement Learning
8. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
