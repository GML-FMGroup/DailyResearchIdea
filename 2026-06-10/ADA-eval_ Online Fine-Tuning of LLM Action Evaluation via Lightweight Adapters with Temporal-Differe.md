# ADA-eval: Online Fine-Tuning of LLM Action Evaluation via Lightweight Adapters with Temporal-Difference Learning in Text-Based Games

## Motivation

MC-DML (Monte Carlo Planning with Large Language Model for Text-Based Game Agents) freezes the pre-trained LLM's action evaluation, preventing it from adapting to domain shifts or improving over time. This structural limitation—the reliance on a static LLM for action scoring—blocks adaptation across all methods in this tree, as the LLM's evaluations are never refined by online experience.

## Key Insight

A lightweight linear adapter on the LLM's frozen internal representations can be trained with temporal-difference learning from sparse game rewards, enabling online adaptation without catastrophic forgetting, because the adapter learns a task-specific additive bias that corrects the LLM's generic evaluations while preserving its knowledge.

## Method

```python
# ADA-eval: Online Fine-Tuning of LLM Action Evaluation

# Hyperparameters: learning_rate η=1e-3, discount factor γ=0.9, exploration rate ε=0.1, MLP hidden_dim=256
# Initialization: adapter network: MLP (input d_model → hidden_dim → |A|) with ReLU activation
# LLM is frozen
# Load-bearing assumption: The MLP adapter on the frozen LLM hidden state can learn task-specific action value biases via TD learning because the hidden state provides a representable basis for the optimal Q-function via a shallow MLP; this is unverified and we test it via probing experiments.

def select_action(state, adapter, llm):
    # Get LLM hidden state for the state (last layer before output)
    h = llm.get_hidden_state(state)  # shape (d_model,)
    # Compute base logits from LLM
    base_logits = llm.evaluate_actions(state)  # shape (|A|,)
    # Compute adapter adjustments via MLP
    adjustments = adapter(h)  # shape (|A|,)
    # Combined Q-values
    q_values = base_logits + adjustments
    # Epsilon-greedy action selection
    if random() < ε:
        return random_action()
    else:
        return argmax(q_values)

def update(state, action, reward, next_state, done, adapter, llm, optimizer):
    h = llm.get_hidden_state(state)
    next_h = llm.get_hidden_state(next_state) if not done else None
    # Compute Q-values for current state
    base_logits = llm.evaluate_actions(state)
    adjustments = adapter(h)
    q_vals = base_logits + adjustments
    q_val = q_vals[action]
    # Compute target
    if done:
        target = reward
    else:
        next_base_logits = llm.evaluate_actions(next_state)
        next_adjustments = adapter(next_h)
        next_q_vals = next_base_logits + next_adjustments
        target = reward + γ * max(next_q_vals)
    # Loss
    loss = (q_val - target)**2
    # Update adapter
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Why this design**: We chose a small MLP adapter (2 layers, hidden=256, ReLU) over full fine-tuning of the LLM to avoid catastrophic forgetting while enabling task-specific adaptation with increased representational capacity; the trade-off is that the adapter may still be limited if deep non-linear composition is required. We selected TD(0) learning over Monte Carlo returns to enable online updates after each step, accepting higher variance but faster adaptation. We used epsilon-greedy exploration (ε=0.1) instead of Boltzmann exploration to simplify hyperparameter tuning, acknowledging that it may be less sample-efficient. The adapter inputs the LLM's frozen hidden state rather than the raw observation, leveraging the LLM's rich representations without requiring retraining, but this assumes that the hidden state contains sufficient information for the task, which may fail if the LLM's pre-training distribution differs significantly from the game domain.

**Why it measures what we claim**: The computational quantity `adjustments = adapter(h)` measures the task-specific bias in action evaluation because we assume that the LLM's frozen hidden state `h` captures general semantic features of the state, and the MLP adapter learns to map these features to a correction that aligns with the true task reward; this assumption fails when the hidden state does not contain features relevant to the task, in which case the bias only reflects spurious correlations from the pre-training. The TD target `r + γ * max_a Q(s',a')` measures the true discounted return by bootstrapping from the current Q-function, but this is only equivalent to the optimal value when the Q-function is accurate and the environment is deterministic Markovian; in stochastic or partially observable games, the TD target estimates a biased expectation biased by the current policy and approximation error. (D2) Specifically, the TD target assumes the environment is Markovian and the current Q-function is sufficiently accurate; if the environment is non-Markovian (e.g., partially observable), the TD target uses a biased estimate of the next state's value, leading to systematic error. Our probing experiment on the LLM hidden state's linear separability of optimal Q-values will quantify this assumption's validity.

## Contribution

(1) A novel framework for online fine-tuning of LLM-based action evaluation using lightweight adapters trained with temporal-difference learning from sparse task rewards. (2) The identification that a linear adapter on frozen LLM hidden representations suffices for task-specific adaptation in text-based games, enabling sample-efficient learning without catastrophic forgetting. (3) An algorithmic instantiation (ADA-eval) with concrete design choices (linear adapter, TD(0), epsilon-greedy) that addresses the structural limitation of static LLM evaluations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Jericho text-based games + WebShop (web navigation) | Test generalization across domains |
| Primary metric | Average normalized score | Measures task success directly |
| Baseline 1 | Pure LLM (no adaptation) | Tests LLM alone ability |
| Baseline 2 | DQN (state features) | Tests RL without LLM |
| Baseline 3 | Full LLM fine-tuning | Tests full adaptation vs adapter |
| Baseline 4 | LoRA (rank=8, scaling=32) | Tests parameter-efficient fine-tuning baseline |
| Ablation-of-ours | ADA-eval with random hidden features | Isolates effect of LLM hidden state |

### Why this setup validates the claim

The Jericho benchmark and WebShop are specifically designed to require both language understanding and sequential decision-making, directly challenging the claim that a small MLP adapter on LLM hidden states can learn task-specific biases. The Pure LLM baseline isolates the need for adaptation; DQN tests whether LLM representations are superior to raw state features; Full fine-tuning tests whether the adapter's parameter efficiency avoids catastrophic forgetting; LoRA tests whether our specific adapter architecture offers advantages over other parameter-efficient methods. The ablation replaces the hidden state with random features, controlling for the source of the adapter's input. The primary metric (average normalized score) captures overall task performance and is sensitive to the quality of action evaluation. Additionally, we conduct a probing experiment on a held-out set of 512 episodes from each domain, training a linear probe on the LLM hidden state to predict Monte Carlo returns; if the probe's R^2 is below 0.5, the assumption that hidden states encode task-relevant Q-values is undermined. Together, these components form a falsifiable test: if the adapter's advantage disappears in any baseline comparison or the ablation matches its performance, the central claim is undermined.

### Expected outcome and causal chain

**vs. Pure LLM (no adaptation)** — On a case where the LLM initially overestimates an action due to pre-training biases (e.g., "take the shiny object" when it's a trap), the pure LLM repeats that mistake because it has no feedback mechanism. Our method instead learns a negative bias for that action via the TD error, so we expect a noticeable gap on games with deceptive initial preferences but parity on games where LLM logits are already accurate.

**vs. DQN (state features)** — On a game requiring language inference (e.g., "the key is hidden under the mat" where the mat is never described), DQN cannot extract the necessary latent state because it only sees tokenized features, causing it to randomly explore. Our method leverages the LLM's hidden state which encodes semantics, so we expect a large gap on language-heavy tasks and a small gap (or even DQN better) on purely statistical environments.

**vs. Full LLM fine-tuning** — On a game with limited training episodes (e.g., 100 steps), full fine-tuning overfits to early rewards and forgets pretrained knowledge, resulting in degraded generalization. Our adapter only modifies a small MLP, preserving the LLM's general knowledge, so we expect our method to match or exceed full fine-tuning on early performance and show slower degradation as training continues.

**vs. LoRA** — LoRA adds trainable rank-8 matrices to the LLM's attention layers, altering the internal representations. In domains requiring preservation of pretrained knowledge (e.g., WebShop), LoRA may cause catastrophic forgetting of general language understanding, whereas our adapter leaves the LLM untouched. Thus, we expect our method to outperform LoRA on tasks where full LLM fine-tuning also fails, but to be comparable or slightly worse on tasks that benefit from representation adaptation.

**Ablation (random hidden features)** — If the adapter's input is replaced by random noise, it cannot learn meaningful biases, so performance should drop to near Pure LLM levels. If not, the method is not using the hidden state effectively.

### What would falsify this idea

If ADA-eval does not significantly outperform Pure LLM on tasks where the LLM is known to make systematic errors (e.g., games with delayed rewards), or if the random-hidden-state ablation matches the full method's performance, then the central claim that the adapter learns task-relevant biases from LLM hidden states is false. Additionally, if the probing experiment shows a linear probe R^2 < 0.5, the assumption that hidden states encode linearly separable Q-values is undermined, suggesting the need for a deeper adapter or a different approach.

**Resource estimates**: All experiments will be conducted on a single NVIDIA V100 GPU. Expected total GPU hours: 50 hours for Jericho (10 games × 5 seeds × 1 hour each) and 30 hours for WebShop (3 tasks × 5 seeds × 2 hours each). We will open-source the code and provide a Docker image for reproducibility.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. LgTS: Dynamic Task Sampling using LLM-generated sub-goals for Reinforcement Learning Agents
3. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
4. Chain of Thought Prompting Elicits Reasoning in Large Language Models
5. LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models
6. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
