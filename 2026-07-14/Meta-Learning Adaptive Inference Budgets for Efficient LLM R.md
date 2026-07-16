# Meta-Learning Adaptive Inference Budgets for Efficient LLM Reasoning

## Motivation

Existing methods like PUST assume a fixed reasoning budget during inference, leading to inefficiency on tasks of varying difficulty. The root cause is that the optimization objective does not penalize over-thinking or under-thinking, and no mechanism exists to adjust depth per task. Weak-to-strong generalization approaches also ignore budget adaptation, leaving a gap in efficiency. This problem is structural: without a learned budget policy, models waste compute on easy tasks and underperform on hard ones.

## Key Insight

The optimal reasoning budget for a task can be inferred from the model's intermediate representations via a learned stop policy, and this policy can be efficiently transferred from a lightweight proxy model to a larger primary model through distillation and reinforcement learning.

## Method

```pseudocode
# BART (Budget-Adaptive Reasoning Transformer) Training Pipeline

Input: Task distribution D, proxy model P (small), primary model M (large)

# Stage 1: Proxy Budget Estimation
for each task t in D:
    sample budget b from uniform(1, B_max)
    reward R(t, b) = correctness(M_b(t)) - lambda * b   # M_b = M with max b steps
    train P to predict optimal budget P(t) = argmax_b R(t, b) via supervised regression

# Stage 2: Primary Model Fine-tuning
for each task t in D:
    b_target = P(t)   # proxy's predicted budget
    # Standard RL: generate reasoning steps z_1,...,z_T with stop head
    # Stop head: MLP(hidden_state) -> logit, stop probability = sigmoid(logit)
    # Action: at each step, sample stop or continue
    # Reward: +1 if correct and stop before or at step b_target, -0.5 if incorrect, -0.1 per step after b_target
    update M via PPO to maximize expected reward
    # Additionally, distillation loss: L_distill = KL(stop_head(M, t) || stop_target) where stop_target = 0 for steps < b_target, 1 at b_target
    combine losses: L = L_RL + alpha * L_distill
```

**Why this design:** We chose a two-stage training pipeline over end-to-end joint training because (1) separating budget estimation into a proxy model provides a stable learning signal for the primary model's stop head, preventing collapse to always stopping early; (2) using a lightweight proxy reduces computational cost during training (can explore many budgets cheaply); (3) the distillation loss provides a curriculum that anchors the stop head near the proxy's estimate, while RL fine-tuning allows the primary model to deviate when the proxy is suboptimal. We accept the cost that the proxy may be inaccurate for out-of-distribution tasks, potentially biasing the primary model, but the RL component can correct this with sufficient exploration. We chose PPO over REINFORCE for stability, and a sigmoid stop head over a softmax stop/continue head because we only need a binary decision per step. We chose a linear budget penalty in the reward to match the proxy's objective, avoiding quadratic penalties that could overly penalize over-thinking.

**Why it measures what we claim:** The stop head's output (stop probability) measures the model's learned belief that the current partial answer is correct and efficient; we assume that a high stop probability correlates with both correctness and sufficient reasoning. This assumption fails when the model is confidently wrong (e.g., adversarial tasks), in which case the stop head reflects overconfidence, not correctness. The proxy's predicted budget measures task difficulty; we assume that the optimal number of reasoning steps is a function of task features (e.g., length, complexity). This assumption fails when the proxy is poorly calibrated due to limited training data, in which case the budget reflects the proxy's bias rather than true difficulty. However, the distillation loss combined with RL fine-tuning mitigates this by allowing the primary model to learn from reward signals even when the proxy is inaccurate. The linear budget penalty in the reward measures efficiency cost; we assume each extra step incurs equal marginal cost, which may not hold for very long reasoning chains where later steps have diminishing returns.

## Contribution

(1) We introduce BART, a single transformer architecture with a learnable stop head that enables adaptive inference-time budget control without a separate controller module. (2) We demonstrate that proxy-guided budget estimation and distillation can efficiently transfer an efficiency-aware reasoning policy to a larger primary model, reducing wasteful computation. (3) We provide a design principle: decoupling budget estimation from policy execution via a lightweight proxy enables stable training and task-adaptive reasoning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GSM8K | Standard multistep math reasoning benchmark. |
| Primary metric | Mean reward | Reflects both correctness and efficiency. |
| Baseline 1 | Standard PPO (fixed budget) | Tests benefit of adaptive budgeting. |
| Baseline 2 | Supervised fine-tuning (full chains) | Tests benefit of RL over imitation. |
| Ablation | Ours without distillation | Tests importance of distillation signal. |

### Why this setup validates the claim

The chosen dataset (GSM8K) features multi-step math problems where reasoning length correlates with correctness, making it ideal to test budget adaptation. Comparing against a standard PPO baseline with fixed budget isolates the benefit of adaptive stopping; if our method outperforms, it confirms that adapting budget per task improves efficiency. Supervised fine-tuning on full reasoning chains serves as a non-RL baseline; outperforming it shows that RL fine-tuning with budget penalty yields better trade-offs. The ablation (removing distillation) tests whether the proxy-guided curriculum is essential; if performance drops, it confirms that distillation stabilizes learning. The primary metric (mean reward) directly captures the objective, ensuring that observed gains reflect true improvements in both correctness and efficiency.

### Expected outcome and causal chain

**vs. Standard PPO (fixed budget)** — On a problem requiring only 2 steps but fixed max steps=10, the baseline generates 10 steps and incurs high penalty, so its reward is low. Our method adapts to stop at 2 steps, gaining reward. We expect a noticeable gap in mean reward, especially on problems with low true steps, where our method's budget adaptation yields higher reward.

**vs. Supervised fine-tuning (full chains)** — On a problem where human demonstrations have varying step lengths, SFT learns a fixed distribution, so it may overthink or underthink on average. Our method uses RL to optimize reward directly, so it learns to match the task-specific optimal budget. We expect our method to achieve higher mean reward, particularly on tasks where the optimal budget deviates from the average demonstration length.

**vs. Our method without distillation** — On a problem with sparse reward signal (e.g., incorrect until last step), the RL-only variant may struggle to learn when to stop because the stop head gradient is noisy. Distillation provides a curriculum that anchors the stop head near the proxy's estimate, facilitating learning. We expect the full method to show faster convergence and higher final reward, especially on tasks where the proxy is accurate.

### What would falsify this idea

If our method's gain over standard PPO is uniform across all problem difficulties (e.g., all problems show similar improvement), then the budget adaptation is not the cause; instead, it might be due to hyperparameter tuning. More specifically, if the ablation without distillation performs equally well, then the distillation loss is unnecessary and the core claim about proxy stabilization is wrong.

## References

1. Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals
2. Incentivizing Strong Reasoning from Weak Supervision
3. Weak-to-Strong Generalization beyond Accuracy: a Pilot Study in Safety, Toxicity, and Legal Reasoning
4. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models
5. Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision
6. Discovering Language Model Behaviors with Model-Written Evaluations
7. Constitutional AI: Harmlessness from AI Feedback
8. Measuring Progress on Scalable Oversight for Large Language Models
