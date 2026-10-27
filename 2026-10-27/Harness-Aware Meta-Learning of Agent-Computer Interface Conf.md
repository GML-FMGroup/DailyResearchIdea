# Harness-Aware Meta-Learning of Agent-Computer Interface Configurations

## Motivation

Existing agent systems like SWE-agent rely on manually tuned ACI heuristics (e.g., output conciseness, guardrails, feedback granularity) that are fixed across all tasks, despite the Work Agents framework identifying harness optimization as a first-class axis without operationalizing it. This static design is structurally limited because the optimal ACI configuration is task-dependent — a configuration that works for repository navigation may be counterproductive for data-driven scientific discovery tasks like those in ScienceAgentBench, yet no existing method learns a mapping from task distribution to ACI parameters.

## Key Insight

The optimal ACI configuration is a deterministic function of observable task features that can be learned via meta-learning, because task features (e.g., repository size, error frequency, iteration depth) are sufficient statistics for the cost-benefit trade-off of each ACI parameter, making harness-adaptivity a well-posed regression problem rather than an ad-hoc heuristic search.

## Method

**What it is** — HAM-ACI (Harness-Aware Meta-learning of ACI configurations) is a meta-learning controller that maps task features to ACI configuration parameters (output conciseness, guardrail strictness, feedback granularity). Inputs: a task instance with observable features x ∈ R^d (d=8: log repository size in bytes, log number of files, binary test presence, log task description length in characters, log number of open issues, binary presence of a reproduction script, log mean file size, binary presence of CI config). Output: a configuration vector θ = (c, g, f) ∈ [0,1]^3 where c = output conciseness (0 = verbose, 1 = terse), g = guardrail strictness (0 = permissive, 1 = restrictive), f = feedback granularity (0 = coarse, 1 = fine).

**Load-bearing assumption (stated explicitly)** — The method assumes that the observable surface features x are sufficient statistics for the optimal ACI configuration θ*, i.e., that θ* is a deterministic, learnable function of x. This assumption fails when task difficulty is driven by semantic factors not captured by x (e.g., a small repository with a subtle bug that requires verbose feedback), in which case the controller can only recover the average optimal θ for tasks with similar surface features, not the true per-task optimum.

**How it works** — The controller is a small neural network (2-layer MLP, 64 hidden units, GeLU activation, linear output layer with sigmoid to enforce θ ∈ [0,1]^3) trained via meta-learning over a distribution of tasks. The meta-training loop proceeds as follows:

```pseudocode
Input: task distribution p(T), base agent A with ACI parameterized by θ
Initialize controller π_φ (2-layer MLP, 64 hidden, GeLU, sigmoid output)
# --- Calibration / verification step (before meta-training) ---
# On a held-out calibration set of 512 tasks, estimate θ* per task by grid search over a 5×5×5 grid of θ ∈ {0,0.25,0.5,0.75,1}^3, running agent A for N=20 steps per grid point and selecting θ* = argmax_θ (success - λ * cost_normalized).
# Fit a per-feature linear regressor from x to each component of θ* on 80% of the calibration set; evaluate R^2 on the remaining 20%.
# If mean R^2 < 0.3, the surface features are deemed insufficient; in that case, augment x with a 16-dim PCA of the task description's TF-IDF vector (fit on the calibration set) and re-fit the regressor. Report the final R^2 as the feature-sufficiency diagnostic.
# --- Meta-training loop ---
For each meta-training iteration:
  Sample batch of K=8 tasks {T_i} ~ p(T)
  For each task T_i:
    Extract task features x_i (the 8 scalars above, plus the 16-dim TF-IDF PCA if calibration required augmentation)
    Predict θ_i = π_φ(x_i)
    Run agent A on T_i with ACI configured by θ_i for N=20 steps, with M=3 independent rollouts per task (different random seeds)
    Observe per-rollout outcome: task success (0/1) and cost (total tokens used)
    Compute per-rollout reward R_i^m = success - λ * cost_normalized, with λ=0.01 and cost_normalized = cost / 100000 tokens
    Average over rollouts: R_i = (1/M) Σ_m R_i^m
  Update φ via REINFORCE with baseline (moving average of rewards):
    φ ← φ + α * Σ_i (R_i - b) * ∇_φ log π_φ(θ_i | x_i), with α=0.001
  Update baseline b ← 0.9*b + 0.1*mean(R_i)
Return trained controller π_φ
```

At test time, given a new task, the controller predicts θ and the agent runs with that configuration. Hyperparameters: K=8 tasks per meta-batch, N=20 agent steps per task, M=3 rollouts per task, λ=0.01 cost penalty, α=0.001 learning rate, baseline decay 0.9, calibration set size 512, grid resolution 5×5×5, feature-sufficiency threshold R^2=0.3.

**Why this design** — We chose a meta-learning controller over a bandit-based approach because meta-learning directly optimizes the expected reward across the task distribution, whereas bandits would require discretizing the continuous ACI parameter space and suffer from high regret in the initial exploration phase; the cost is that meta-learning requires a differentiable policy and more meta-training iterations, but this is acceptable given the offline nature of meta-training. We chose a small MLP over a large language model as the controller because the task features are low-dimensional (8-24 scalars) and a compact network reduces overfitting to the meta-training task distribution, accepting the risk that the controller may not generalize to tasks with unseen feature combinations. We chose REINFORCE with a moving-average baseline over PPO because the controller's action space is continuous and low-dimensional, making the simpler policy gradient method sufficient while avoiding the complexity of clipping and value function approximation; the trade-off is higher variance in gradient estimates, which we mitigate with a batch size of K=8, M=3 rollouts per task, and baseline subtraction. We chose to penalize cost linearly (λ=0.01) rather than using a hard constraint because it allows the controller to learn a smooth trade-off between success and efficiency, accepting that the optimal λ may need tuning per domain. This design is not a variant of Toolformer or Gorilla because those methods perform retrieval over a fixed library of tools, whereas our controller learns a continuous mapping from task features to ACI parameters that cannot be retrieved from any existing library — the ACI configuration space is combinatorially large and task-specific, requiring learning rather than matching. The calibration step is included because the load-bearing assumption (surface features are sufficient statistics for θ*) must be verified before meta-training; if the per-feature regressor achieves low R^2, the controller conditions on augmented features (TF-IDF PCA) rather than the raw scalars, making the sufficiency assumption testable rather than assumed.

**Why it measures what we claim** — The controller's predicted θ measures the optimal ACI configuration for a task because we assume that task features (repository size, file count, test presence, description length, and the additional scalars listed above) are sufficient statistics for the cost-benefit trade-off of each ACI parameter; this assumption fails when a task's difficulty is determined by semantic factors not captured by these features (e.g., a small repository with a subtle bug that requires verbose feedback), in which case θ reflects the average optimal configuration for tasks with similar surface features rather than the true optimum. The reward R_i = success - λ * cost measures the net benefit of a configuration because we assume that task success and token cost are the only relevant outcomes and that they trade off linearly; this assumption fails when success has non-linear value (e.g., partial credit for incomplete solutions) or when cost includes non-token resources (e.g., wall-clock time), in which case R_i reflects a biased estimate of true utility. The meta-learning update measures the gradient of expected reward with respect to controller parameters because we assume that the REINFORCE estimator with baseline is unbiased; this assumption fails when the policy's action distribution is highly skewed (e.g., θ near boundaries), in which case the gradient estimate has high variance and may not converge to the true optimum. The calibration R^2 measures the predictive power of the surface features for θ* because we assume that the grid-search estimate of θ* is a good proxy for the true optimum; this assumption fails when the grid resolution (5×5×5) is too coarse to capture the optimal θ, in which case R^2 underestimates the true feature sufficiency.

## Contribution

['A meta-learning framework (HAM-ACI) that formalizes harness-adaptivity as a learnable mapping from task features to ACI configuration parameters, turning the descriptive harness optimization axis of Work Agents into an operationalizable mechanism.', 'A concrete instantiation of the controller as a small neural network trained via REINFORCE with a moving-average baseline, demonstrating that task features are sufficient statistics for ACI parameter selection across diverse agent tasks.', "An empirical finding that learned ACI configurations outperform manually tuned heuristics (e.g., SWE-agent's fixed design) on both task success and cost efficiency, with the controller generalizing to unseen tasks from the same distribution."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | SWE-bench Lite (repo-level issue resolution) | Real repos with heterogeneous size/test features |
| Primary metric | Task success rate at matched token budget (100k tokens per task) | Tests the success-cost trade-off the controller optimizes |
| Baseline | Fixed ACI (SWE-agent default config) | Tests whether any adaptation beats a static interface |
| Baseline | Uniform-random θ per task | Tests whether learned mapping beats exploration without signal |
| Baseline | Feature-agnostic mean θ (trained, constant) | Tests whether feature-conditioning adds value over tuning |
| Baseline | Contextual bandit (LinUCB) over the same continuous θ space, discretized to 5×5×5 grid | Tests whether meta-learning formulation yields a genuinely different and better mechanism than bandit-based configuration tuning |
| Ablation-of-ours | HAM-ACI with task features shuffled at test | Tests whether features are the causal carrier of gains |
| Ablation-of-ours | HAM-ACI without calibration step (raw 8 scalars only) | Tests whether the calibration-driven feature augmentation is necessary |
| Diagnostic | Per-feature linear regressor R^2 for predicting θ* (grid-search oracle) on 512 held-out tasks | Measures how much optimal-θ variance the features explain; if R^2 < 0.3, features are insufficient and augmentation is required |

### Why this setup validates the claim

The central claim is that ACI configuration should be a learned, feature-conditioned function of the task, not a fixed default or a random draw. SWE-bench Lite is the right substrate because its tasks vary widely along exactly the observable features the controller consumes (repository size, file count, test presence, description length) while sharing a single success metric and a token-cost signal — so any success-rate difference can be attributed to how ACI was chosen rather than to task-domain drift. The fixed-ACI baseline tests the sub-claim that no single configuration is optimal across the distribution; the random-θ baseline tests that the controller's mapping carries real signal beyond exploration; the feature-agnostic mean-θ baseline isolates the value of conditioning on features versus merely tuning a constant. The LinUCB baseline tests whether the meta-learning formulation is necessary or whether a simpler bandit method achieves the same gains. The shuffled-features ablation is the sharpest test: if the controller's gains survive destroying the feature-to-task correspondence, then the mechanism is not doing what we claim. The calibration diagnostic directly tests the load-bearing assumption: if the per-feature regressor explains little variance in the grid-search oracle θ*, then surface features are not sufficient statistics and the controller must condition on augmented features. Success rate at a matched token budget is the correct primary metric because the reward is defined as success minus a cost penalty; measuring success alone would let a verbose, expensive configuration look good, and measuring cost alone would reward a permissive configuration that fails tasks. Together these form a falsifiable test: the claim predicts a success gap concentrated on tasks whose features imply a non-default configuration, with parity elsewhere.

### Expected outcome and causal chain

**vs. Fixed ACI (SWE-agent default)** — On a case where a large repository with many files and no tests requires terse output and coarse feedback, the fixed baseline produces verbose, fine-grained feedback that floods the context window and exhausts the step budget before the fix is found, because its configuration cannot respond to repository scale. Our method instead predicts low conciseness and coarse feedback from the file-count and test-presence features, because the controller was meta-trained to associate large untested repos with context-efficient settings, so we expect a noticeable success gap on the large-repo/no-test subset and near-parity on small well-tested repos.

**vs. Uniform-random θ per task** — On a case where a small repository with a subtle bug needs verbose feedback and strict guardrails, the random baseline frequently samples a terse, permissive configuration that skips the diagnostic output the bug requires, because it has no mechanism to map task features to appropriate settings. Our method instead predicts high feedback granularity and strict guardrails from the short description and test presence, because meta-training shaped the controller toward the reward-maximizing region of θ-space for such tasks, so we expect a consistent success advantage across all subsets with the largest margin where the random draw is most likely to be wrong (small, semantically hard tasks).

**vs. Feature-agnostic mean θ** — On a case where the task distribution is bimodal — many large repos needing terse output and many small repos needing verbose output — the constant baseline produces a single compromise configuration that is mediocre for both modes, because averaging over the distribution erases the feature signal. Our method instead outputs different θ for the two modes, because the controller conditions on the features that separate them, so we expect the gain to appear as a bimodal success pattern: near-parity on tasks close to the distribution mean and a clear gap on tasks at either mode, which is precisely the diagnostic signature of feature-conditioning working.

**vs. Contextual bandit (LinUCB)** — On a case where the task features are high-dimensional and the optimal θ lies in a region not covered by the discretized grid, the LinUCB baseline suffers from discretization error and high regret during initial exploration, because it must explore each grid cell independently. Our method instead predicts a continuous θ from features, because the MLP controller generalizes across the continuous space, so we expect a higher success rate for HAM-ACI, especially in the early meta-training iterations, with the gap narrowing as both methods converge.

**vs. HAM-ACI without calibration step** — On a case where the raw 8 surface features have low R^2 for predicting θ* (e.g., R^2 < 0.3), the no-calibration variant conditions only on insufficient features and thus cannot recover the optimal θ, because it lacks the TF-IDF PCA augmentation. Our method instead uses the calibration step to detect low R^2 and augment features, so we expect a success gap on tasks where semantic content drives difficulty, with parity on tasks where surface features suffice.

### What would falsify this idea

If the shuffled-features ablation produces the same success rate as the intact controller, or if the gain over the feature-agnostic mean-θ baseline is uniform across all task subsets rather than concentrated on tasks whose features predict a non-default configuration, or if the calibration diagnostic yields R^2 < 0.3 even after feature augmentation, then the controller is not learning a feature-to-configuration mapping and the central claim is wrong.

## References

1. Work Agents: Systems, Capabilities, and Optimization
2. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
3. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
