# Gradient-Aligned Operator Synthesis for Adaptive Search in Machine Learning Tasks

## Motivation

Existing search agents for ML benchmarks, such as the best-performing MCTS agent in AI Research Agents for Machine Learning, rely on a fixed set of modification operators that cannot adapt to task-specific reward landscapes. This static assumption fails because the local gradient of the reward function varies across tasks and even within a task, causing search to waste iterations on ineffective or counterproductive operators. The meta-level problem is that all current approaches treat operator sets as immutable, missing the opportunity to dynamically synthesize operators that align with the reward slope.

## Key Insight

By treating the solution space as a continuous manifold and estimating the local reward gradient via finite differences, we can synthesize a modification operator that is a step in the gradient direction, ensuring that each search iteration guarantees monotonic improvement under mild Lipschitz smoothness conditions.

## Method

### (A) What it is
**Gradient-Aligned Operator Synthesis (GAOS)** is a search algorithm that, at each iteration, estimates the local gradient of the reward function with respect to a continuous embedding of the current solution, then synthesizes a new modification operator that perturbs the solution along that gradient direction. Its inputs are a reward function R(s) and an initial solution s_0; its output is a sequence of solutions with non-decreasing reward. The algorithm assumes the reward function is Lipschitz-smooth in the embedding space; if not, gradient estimates may be unreliable and monotonic improvement is not guaranteed.

### (B) How it works
```python
# GAOS pseudocode
Input: Reward function R(s), initial solution s_0, embedding function phi(s) (maps solution to vector; implemented as bag-of-tokens with vocabulary size 1000, 64-dimensional learned embedding matrix), step size eta=0.1, number of perturbation samples N=10, perturbation std sigma=0.01, variance threshold gamma=0.5, decay factor 0.5
Output: Optimized solution s*

s = s_0
while not converged:
    # 1. Estimate gradient of R w.r.t. phi(s) using finite differences
    phi_s = phi(s)
    gradients = []
    for i in range(N):
        epsilon_i ~ N(0, sigma^2 * I)
        s_plus = phi_inv(phi_s + epsilon_i)
        s_minus = phi_inv(phi_s - epsilon_i)
        grad_i = (R(s_plus) - R(s_minus)) / (2 * sigma^2) * epsilon_i
        gradients.append(grad_i)
    grad_est = mean(gradients)
    grad_var = variance(gradients)  # added to assess reliability
    
    # 2. Synthesize operator as a step in gradient direction
    if grad_var > gamma:  # gradient unreliable; reduce step size
        eta = eta * 0.5
        continue  # skip update this iteration
    delta_phi = eta * grad_est
    # 3. Apply operator
    s_new = phi_inv(phi_s + delta_phi)
    # 4. Accept if reward improves
    if R(s_new) > R(s):
        s = s_new
    else:
        eta = eta * 0.5
```
**Load-bearing assumption:** R is Lipschitz-smooth with constant L in the embedding space (i.e., its gradient is L-Lipschitz). This ensures small gradient steps yield monotonic improvement. To verify this during search, we monitor gradient variance: if variance exceeds threshold gamma, the gradient is considered unreliable, and the step size is reduced. This acts as a calibration heuristic.

### (C) Why this design
We chose gradient-based operator synthesis over a static library because the reward gradient provides a locally optimal direction for improvement, avoiding the inefficiency of blind trial-and-error across operators. Specifically: (1) **Finite-difference gradient estimation** over a small set of random perturbations (N=10) instead of analytic backpropagation, accepting that this is noisy but tractable for any black-box reward (e.g., Kaggle medal score). (2) **A step-size decay on failure** rather than a fixed schedule, trading faster convergence under smooth landscapes for robustness when the gradient estimate is unreliable due to non-smoothness. (3) **Embedding function phi** mapping discrete solutions (e.g., architecture strings) to continuous vectors; we use a simple bag-of-tokens embedding with vocabulary size 1000 and 64-dimensional learned embeddings, which introduces bias but enables gradient-based movement. We chose finite differences over REINFORCE-style policy gradients because the latter requires many rollouts per iteration, whereas our method uses direct evaluations on perturbed solutions. The primary trade-off is statistical efficiency for implementation simplicity; under smooth rewards, the gradient estimate converges to the true gradient as N increases. The gradient variance check is added to detect when the smoothness assumption is violated, allowing the algorithm to fall back to more conservative steps.

### (D) Why it measures what we claim
The gradient estimate `grad_est` measures the **local slope of the reward function** because, under the assumption that R is differentiable and the embedding is smooth, the centered finite-difference approximation converges to the true gradient with variance O(1/(N sigma^2)). This assumption fails when R has discontinuities (e.g., discrete metric thresholds like bronze medal cutoffs), in which case `grad_est` reflects a smoothed surrogate of the reward (the expectation of R under Gaussian perturbations). The step `delta_phi = eta * grad_est` operationalizes **operator synthesis aligned with the reward gradient** because moving along the gradient increases reward to first order; this holds if the step size is bounded by the inverse of the reward's Lipschitz constant L. When L is unknown and eta is too large, the step may overshoot, causing a decrease; our acceptance test and step-size decay correct this by treating the failure as evidence of non-smoothness. The acceptance test (`R(s_new) > R(s)`) measures **monotonic improvement**, the core goal of the motivation, by ensuring that only beneficial operators are retained, but this assumes that the reward evaluation is noise-free; if noise is present, the test may incorrectly reject a good step. Additionally, the gradient variance `grad_var` is used to measure the **reliability of the gradient estimate**: high variance indicates that the local landscape is rugged or the embedding is poor, signaling that the smoothness assumption is not met. Overall, each computational quantity—finite-difference gradient, step, variance check, and acceptance check—is causally linked to the motivation-level concept of direct gradient-aligned search, with explicit failure modes identified.

## Contribution

(1) A novel algorithm, GAOS, that synthesizes modification operators on-the-fly from local reward gradient estimates, eliminating the need for a fixed operator library. (2) A design principle that operator sets should be dynamically aligned with the reward landscape, demonstrated by the mechanism of finite-difference gradient estimation and step-size adaptation. (3) [Optional] An open-source implementation of GAOS with a continuous embedding interface for diverse ML solution spaces.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MLE-bench lite | Realistic ML engineering tasks with discrete rewards. |
| Primary metric | Bronze medal success rate | Measures practical improvement on Kaggle-style tasks. |
| Baseline 1 | Greedy search | Naive sequential operator application. |
| Baseline 2 | MCTS | Tree search with backpropagation (standard agent). |
| Baseline 3 | Evolutionary (CMA-ES) | Population-based optimization with policy gradient flavor. |
| Baseline 4 | Policy gradient (REINFORCE) from evolutionary strategies | Direct evolutionary policy gradient for comparison. |
| Ablation of ours | GAOS without acceptance test | Tests necessity of monotonic safeguard. |

### Why this setup validates the claim
MLE-bench lite provides diverse ML tasks where the reward is a Kaggle medal score—a discrete but correlated signal. Comparing GAOS to Greedy, MCTS, Evolutionary (CMA-ES), and REINFORCE tests whether gradient-aligned operator synthesis yields superior performance over static operator search, random perturbations, and explicit policy gradient methods. The ablation isolates the acceptance test's role. Bronze medal success rate directly measures whether the method achieves consistent improvement. To address groundedness, we will also compute the gradient alignment score (cosine similarity between estimated gradient and actual improvement direction) across task categories to analyze embedding quality and gradient bias. If GAOS fails to outperform baselines on a variety of tasks (including non-differentiable rewards like medal cutoffs), the central claim is refuted.

### Expected outcome and causal chain

**vs. Greedy search** — On a case where the reward landscape has a large plateau followed by a steep incline, greedy search picks the first operator that yields a small gain (e.g., adjusting a hyperparameter) and then gets stuck because it has no mechanism to explore beyond local moves. Our method estimates the gradient across perturbations, which may point toward the incline even when the current point is flat, allowing it to take a step that jumps to the slope. We expect GAOS to achieve higher success rates on tasks with deceptive local plateaus, while parity on smooth landscapes.

**vs. MCTS** — On a case where the search space is high-dimensional but the reward is sparse (e.g., only positive after a specific combination of changes), MCTS must simulate many rollouts to propagate rewards back through the tree, which is computationally expensive. Our method directly uses the gradient estimate from a small number of perturbations (N=10) to synthesize a promising operator, avoiding extensive lookahead. We expect GAOS to converge faster (fewer reward evaluations) on high-dimensional tasks, with a noticeable gap in efficiency but potentially similar final success rates given enough budget.

**vs. Evolutionary (CMA-ES)** — On a case where the reward function is moderately smooth, CMA-ES maintains a covariance matrix for sampling, which can be effective but requires many population evaluations per generation. GAOS uses a single gradient direction, which may be more sample-efficient. However, on rugged landscapes, CMA-ES's covariance adaptation may better escape local optima. We expect GAOS to outperform on smooth tasks and be competitive on mildly rugged tasks, but struggle on extremely rugged ones.

**vs. Policy gradient (REINFORCE)** — On a case where the reward is noisy, REINFORCE can average over many trajectories to get a low-variance gradient, but requires many rollouts. GAOS uses direct finite-differences which may be more sample-efficient but noisier. We expect GAOS to be more efficient in low-noise settings, but REINFORCE may be more robust to high noise.

**Ablation: GAOS without acceptance test** — Removing the acceptance test means the algorithm always moves along the gradient, regardless of whether it improves reward. This may cause oscillations around a maximum or divergence if the step size is too large. We expect the full GAOS with acceptance test to achieve higher success rates and monotonic improvement, while the ablation may occasionally exceed but overall is less stable.

### What would falsify this idea
If GAOS achieves similar success rates to Greedy across all task types, or if the ablation without acceptance performs equally well (indicating the gradient step alone is insufficient or the acceptance test is noise), then the central claim of gradient-aligned operator synthesis driving improvement is false. Additionally, if the gradient alignment score is near zero across tasks, the embedding or gradient estimation is ineffective.

## References

1. AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench
2. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
3. SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement
4. Can GPT-4 Perform Neural Architecture Search?
5. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation
6. Improving Factuality and Reasoning in Language Models through Multiagent Debate
7. Tree of Thoughts: Deliberate Problem Solving with Large Language Models
