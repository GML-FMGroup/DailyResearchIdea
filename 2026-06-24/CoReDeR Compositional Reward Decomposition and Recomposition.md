# CoReDeR: Compositional Reward Decomposition and Recomposition for Multi-Task Agent Training

## Motivation

Current agent training methods, such as those used in OpenThoughts-Agent, assume a single monolithic reward signal or rely on implicit composition via data curation, failing to generalize across heterogeneous sub-tasks. This structural limitation arises because reward compositionality is not explicitly modeled, leading to interference between sub-task gradients and poor compositional generalization. OpenThoughts-Agent's data recipes implicitly assume that diverse data alone suffices, but this neglects the need for dynamic reward weighting during training.

## Key Insight

Explicitly decomposing and recomposing reward signals via a learned layer that weights sub-task rewards by their mutual information and difficulty enables compositional generalization because it captures the structural dependencies and credit assignment across sub-tasks that implicit compositions miss.

## Method

## CoReDeR (Compositional Reward Decomposition and Recomposition)

(A) **What it is**: A trainable reward composition layer inserted between the environment's reward function and the reinforcement learning (RL) update. It accepts raw environment rewards and outputs a scalar reward for each training step, dynamically weighting sub-task contributions to enable compositional generalization.

(B) **How it works - Pseudocode for forward pass during training:**
```python
def coreder_forward(reward_raw, task_embedding, sub_task_bounds):
    # reward_raw: scalar reward from environment per step
    # task_embedding: vector encoding of the current task (e.g., from instruction)
    # sub_task_bounds: list of (start_step, end_step) defining sub-task intervals
    # Sub-task boundaries are precomputed from task descriptions using a frozen LLM (GPT-4)
    # to generate (start_step, end_step) tuples, with manual verification on a calibration set of 100 episodes.
    
    # Phase 1: Decompose raw reward into sub-task rewards
    sub_rewards = []
    for (s, e) in sub_task_bounds:
        raw_segment = reward_raw[s:e]  # reward sum or max over segment
        # Learned decomposition head predicts sub-task reward weight
        # MLP: 2-layer, hidden=256, ReLU, input=raw_segment sum (scalar), output=weight via sigmoid scaling to [0,1]
        decomposition_weight = decomposition_mlp(raw_segment.sum())  # learned, >0 via sigmoid scaling
        sub_reward = decomposition_weight * raw_segment.sum()
        sub_rewards.append(sub_reward)
    
    # Phase 2: Compute sub-task importance via mutual information gating
    # Estimate mutual information between sub-task reward and final success (from replay buffer)
    mi_weights = []
    for r in sub_rewards:
        # Learned MI estimator: 3-layer MLP with hidden sizes 128,64,32, Donsker-Varadhan representation
        # Input: sub_reward scalar and success_flag scalar (1 for success, 0 for failure)
        # Output: estimated MI (unbounded)
        mi = mutual_information_estimator(r, success_flag)  # success_flag from episode outcome
        mi_weights.append(torch.sigmoid(mi))  # scale to [0,1]
    
    # Phase 3: Dynamically recompose reward
    # Difficulty-based reweighting: lower sub-task accuracy → higher weight
    accuracy = smoothed_sub_task_accuracy  # from recent rollouts, exponential moving average with alpha=0.1
    difficulty_weight = 1.0 / (accuracy + 1e-6)  # inverse proportional
    combined_weights = torch.softmax(torch.log(mi_weights) + torch.log(difficulty_weight), dim=0)
    
    # Final reward = weighted sum of sub-rewards, scaled by overall performance
    total_reward = torch.sum(combined_weights * torch.stack(sub_rewards))
    return total_reward
```
**Calibration verification**: After each training epoch, compute the Pearson correlation between the MI weights and the empirical sub-task completion rates on a held-out set of 200 episodes. If correlation < 0.3, replace MI weights with uniform weights for the next epoch.

(C) **Why this design**: We chose a learned decomposition head over a fixed hand-crafted decomposition because sub-task boundaries may be only partially known; learning allows the model to adapt weights within segments, accepting the cost of additional parameters and potential overfitting if segments are short. We selected mutual information as the gating signal (over simpler methods like direct reward magnitude) because MI captures non-linear dependencies between sub-task reward and eventual success, which is critical when sub-tasks have delayed effects; the trade-off is higher computational cost and the need for a reliable success signal. Using inverse accuracy as a difficulty weight (instead of a separate learned module) avoids introducing another learned component that could compete with the decomposition head, but risks biasing toward sub-tasks that are simply noisy rather than difficult. The softmax combination of MI and difficulty weights ensures that both aspects contribute deterministically without additional learned degrees of freedom.

(D) **Why it measures what we claim**: The decomposition weight `decomposition_weight` measures the fraction of raw reward attributable to each sub-task because we assume sub-task intervals are properly aligned to true task boundaries via manual annotation; this assumption fails when intervals overlap or miss transitions, in which case the weight may capture spurious correlations. The mutual information `mi` measures the causal necessity of a sub-task for eventual success under the assumption that success_flag is a deterministic function of sub-task completion and that the replay buffer contains sufficient diverse episodes to accurately estimate MI; this assumption fails when success is influenced by unmodeled factors (e.g., stochasticity, exploration noise), in which case MI may reflect spurious correlations. To mitigate, we incorporate the calibration verification step (see (B)). The difficulty weight `1/accuracy` measures sub-task difficulty because we assume lower historical accuracy reliably indicates harder sub-tasks (via the proxy of policy performance); this assumption fails when accuracy is dominated by exploration noise or stale data, in which case the weight reflects outdated difficulty. Together, these quantities operationalize compositional generalization by dynamically reweighting sub-tasks according to their importance and difficulty, overcoming the single-task reward assumption of prior work like OpenThoughts-Agent.

## Contribution

(1) A novel reward composition layer, CoReDeR, that explicitly decomposes raw environment rewards into sub-task components and recomposes them via learned dynamic weighting. (2) A principled weighting mechanism combining mutual information (for causal importance) and difficulty (for learning focus) that enables compositional generalization without hand-crafted reward shaping. (3) An analysis framework linking sub-task reward decomposition, mutual information gating, and difficulty reweighting to the open problem of multi-task reward composition in agentic models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | AgentBench suite | Diverse agentic tasks with sub-task structure. |
| Synthetic benchmark | Sub-task boundary perturbation (up to 20% offset) | Measures robustness of decomposition head. |
| Primary metric | Task success rate | Directly measures compositional generalization. |
| Secondary metric | Decomposition weight correlation with ground-truth (synthetic) | Validates alignment with true sub-task contribution. |
| Baseline 1 | OpenThoughts-Agent | State-of-the-art agentic training baseline. |
| Baseline 2 | Vanilla PPO (raw rewards) | No reward decomposition, tests necessity. |
| Baseline 3 | Fixed decomposition (equal weights) | Ablates adaptivity, tests dynamic weighting. |
| Ablation of ours | CoReDeR without MI gating | Isolates contribution of mutual information. |

### Why this setup validates the claim

This setup directly tests the central claim that dynamic compositional reward decomposition improves compositional generalization. AgentBench provides a diverse set of agentic tasks with clear sub-task structures, enabling evaluation of generalization across varying task complexities. The synthetic benchmark with misaligned sub-task boundaries tests robustness of the decomposition head to annotation errors, a key assumption. Comparing against OpenThoughts-Agent tests whether our method surpasses a strong data-curation baseline. Vanilla PPO isolates the effect of reward decomposition from standard RL, while fixed decomposition abates the adaptive weighting mechanism. The ablation of MI gating pinpoints the source of improvement. Task success rate is the appropriate metric because it reflects the ultimate goal of compositional reasoning, and it is sensitive to the reweighting that our method performs. A failure to outperform fixed decomposition on tasks with varying sub-task difficulties would directly falsify the need for dynamic weighting. The synthetic metric provides additional mechanistic insight.

### Expected outcome and causal chain

**vs. OpenThoughts-Agent** — On tasks where the distribution of sub-tasks shifts from the training data (e.g., novel combinations of instructions), OpenThoughts-Agent suffers because its data recipes are static and may not cover the new distribution. Our method dynamically recomposes rewards based on online sub-task importance and difficulty, adapting to the new distribution. Thus, we expect a noticeable gap favoring CoReDeR on held-out compositional tasks, while performance may be similar on in-distribution tasks.

**vs. Vanilla PPO (raw rewards)** — On a case like a long-horizon task with delayed rewards (e.g., multi-step navigation), Vanilla PPO sees only a sparse final reward and fails to credit sub-tasks, leading to random exploration. Our method decomposes the reward per sub-task and uses mutual information to weight them, enabling fine-grained credit assignment. We expect CoReDeR to achieve significantly higher success rates on such tasks, while performance on short-horizon tasks may be comparable.

**vs. Fixed decomposition (equal weights)** — On a task with imbalanced sub-task difficulties (e.g., one sub-task requires precise actions while others are trivial), fixed weighting over-weights easy sub-tasks and under-weights hard ones, causing the policy to neglect critical skills. Our method adapts weights via inverse accuracy, focusing learning on harder sub-tasks. Consequently, we expect CoReDeR to show a clear advantage on tasks with skewed difficulty, while parity on balanced tasks.

**vs. CoReDeR without MI gating** — On tasks where sub-task importance is not captured by reward magnitude alone (e.g., a sub-task with small reward but high importance for success), removing MI gating should degrade performance. We expect the full method to outperform the ablation on such tasks.

### What would falsify this idea

If CoReDeR’s improvement over fixed decomposition is uniform across all task types rather than concentrated on tasks with delayed rewards or imbalanced difficulties, then the dynamic weighting is not addressing the intended failure modes, falsifying the core claim. Additionally, if the synthetic boundary perturbation experiment shows that the decomposition head is highly sensitive (e.g., correlation with ground-truth drops below 0.5 for even small perturbations), then the assumption of well-aligned boundaries is violated, undermining the method's validity.

## References

1. OpenThoughts-Agent: Data Recipes for Agentic Models
