# Test-Time Online Gradient Adaptation for Open-Ended Text Game Agents

## Motivation

AgentOdyssey (2026) evaluates agents in open-ended text games where tasks evolve without train/test splits, yet existing agents rely on static pre-trained models and cannot update parameters during test time. This structural limitation prevents adaptation to shifting world dynamics and long-horizon dependencies, as demonstrated by agents' poor performance on episodic memory and world knowledge tests that require continual update.

## Key Insight

By performing gradient descent on a sliding window of recent experiences using a self-supervised auxiliary loss, the agent can continually update its internal representations to match the evolving data distribution without catastrophic forgetting.

## Method

### (A) What it is
**Test-Time Online Gradient Adaptation (TOGA)** updates a neural agent's parameters at each interaction step by minimizing an auxiliary loss over a sliding window of the most recent experiences. Input: current model parameters θ, sliding window buffer B (size W=100), auxiliary loss function L. Output: updated parameters θ'.

**Load-bearing assumption:** Minimizing next-observation prediction error on a sliding window of the most recent 100 experiences improves the agent's world knowledge quiz accuracy. To verify this, during training we periodically compute the Pearson correlation between the average auxiliary loss on a held-out buffer of recent 200 experiences and the agent's accuracy on a world knowledge quiz. If the correlation is below 0.3, we flag the assumption as violated and consider redesigning the auxiliary task.

### (B) How it works
```python
def toga_update(agent, B, step):
    # B: list of (obs, action, reward, next_obs) of size W
    # Observations are encoded as 512-dim embeddings via a frozen BERT model.
    # Agent: LSTM (hidden=256) with a world model head (linear layer outputting 512-dim).
    loss = 0
    for (obs, action, _, next_obs) in B:
        pred_next_obs = agent.forward(obs, action)  # world model head prediction
        loss += mse(pred_next_obs, next_obs) / W
    # Gradient descent with learning rate alpha=1e-3 that decays as step increases
    grad = autograd(loss, agent.parameters)
    # Optimizer: SGD without momentum.
    grad = clip_by_norm(grad, 1.0)  # norm_max = 1.0
    agent.parameters -= (1e-3 / (1 + step * 1e-3)) * grad
    return agent
```
Hyperparameters: window size W=100, initial learning rate alpha=1e-3, decay rate beta=1e-3, gradient norm_max=1.0.

### (C) Why this design
We chose a sliding window over a growing replay buffer because it bounds memory cost and naturally forgets outdated experiences, preventing staleness at the cost of potential forgetting of rare events. We selected next-observation prediction as the auxiliary task because it is self-supervised and requires learning world dynamics, unlike action-value regression which demands reward signals. Gradient clipping is used to avoid destabilizing updates from noisy gradients, accepting slower convergence in favor of stability. The learning rate decays over time to reduce large updates as the model converges, balancing adaptation speed and stability.

### (D) Why it measures what we claim
The gradient update magnitude `alpha / (1 + step * beta) * grad` measures **adaptation rate** because it controls how much parameters shift per step; this operational assumption equates larger parameter changes to faster adaptation, but fails when the gradient direction is noisy, in which case it instead measures instability. The window size W measures **memory horizon** because it determines how many past experiences contribute to the loss; this assumes recent experiences are most relevant, which fails when a rare but critical event falls outside the window. The auxiliary loss L measures **world model alignment** because minimizing prediction error forces the model to capture dynamics; this assumes prediction error correlates with task performance, which fails if the agent learns spurious correlations (e.g., always predicting the same next observation).

## Contribution

(1) Introduces TOGA, a gradient-based online weight adaptation method for open-ended text game agents that updates parameters via self-supervised learning on a sliding window. (2) Provides a principled integration of test-time training into the AgentOdyssey framework, directly addressing the adaptation gap identified by its evaluation metrics. (3) Characterizes the trade-off between adaptation speed and forgetting through window size and learning rate decay, offering a design space for continual learning agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Procedurally generated text games (TextWorld with entity-based state) | Open-ended, long-horizon, tests continual learning. |
| Primary metric | World knowledge quiz accuracy | Directly measures learned dynamics alignment. |
| Baseline | Static policy (same architecture, no parameter updates) | Isolates adaptation necessity. |
| Baseline | Episodic memory buffer (stores last 1000 experiences, uniformly replayed) | Tests memory recency vs. full window. |
| Baseline | Full online gradient descent (all past experiences, bounded to 10k) | Baselines forgetting from unbounded memory. |
| Ablation | TOGA w/o gradient clipping | Isolates stability effect of clipping. |
| Analysis metric | Pearson correlation between avg auxiliary loss (held-out 200 steps) and quiz accuracy | Validates causal link; if <0.3, assumption violated. |

### Why this setup validates the claim
This setup tests the central claim that TOGA's sliding window and gradient adaptation enable effective continual learning from recent experiences. The text-game dataset requires agents to build and update world knowledge over many steps, making it ideal for evaluating adaptation. Comparing against a static policy shows whether any learning helps; against an episodic memory buffer reveals if the sliding window's forgetfulness is beneficial; and against full online gradient descent tests whether bounded memory prevents catastrophic forgetting. The world knowledge quiz accuracy directly measures whether the auxiliary next-observation prediction task yields usable dynamics knowledge. The ablation of gradient clipping isolates its role in stabilizing updates. Additionally, the correlation analysis directly tests the load-bearing assumption that minimizing auxiliary loss leads to improved quiz accuracy, providing an explicit verification mechanism. Together, these components create a falsifiable test: if TOGA's advantage stems from its specific design choices, then the pattern of results across baselines and ablation must align with the predicted failure modes of each alternative.

### Expected outcome and causal chain

**vs. Static policy** — On a case where the game introduces a new room with unseen objects, the static policy fails to update its world knowledge, resulting in incorrect quiz answers about object locations because it cannot adapt. Our method instead learns from recent observations in the sliding window, updating its dynamics model to predict new object positions, so we expect a large gap on quiz questions about recently encountered objects, but parity on static world knowledge.

**vs. Episodic memory buffer** — On a case where the agent must quickly adapt after a sudden environmental shift (e.g., a looted room), the episodic memory buffer retains outdated examples from before the shift, causing conflicting gradients and slower adaptation. Our method's sliding window automatically discards old experiences, allowing rapid convergence to the new dynamics, so we expect a noticeable gap in the first few steps after the shift, with both methods eventually converging.

**vs. Full online gradient descent** — On a case where the agent accumulates many interactions (e.g., 10k steps), full online gradient descent uses all past experiences, leading to gradient interference from stale memories and eventual forgetting of recent important events. Our method's bounded window prevents this, maintaining a fresh dynamics model, so we expect a growing gap as the episode length increases, with full online gradient descent's accuracy declining after a horizon.

### What would falsify this idea
If TOGA shows no significant improvement over the static policy on recently introduced world knowledge, or if its gains are uniformly distributed across all quiz questions (rather than concentrated on recent experiences), then the central claim that sliding-window adaptation drives performance would be falsified. Additionally, if the Pearson correlation between auxiliary loss and quiz accuracy is below 0.3 (indicating that minimizing auxiliary loss does not correspond to better world knowledge), the assumption underlying TOGA is violated and the design must be revised.

## References

1. AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents
