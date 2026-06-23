# Next-State Perplexity Minimization: Self-Supervised Reward for LLM Agents

## Motivation

Existing LLM agent methods such as Memory-R1 rely on explicit outcome-driven rewards for policy learning, which limits their applicability to environments where task rewards are sparse or absent. The root cause is that these approaches treat reward as an external engineering artifact rather than leveraging the inherent structure of the LLM's pretrained knowledge, which already encodes plausible state progressions. This design forces practitioners to manually design reward functions even when the underlying dynamics are well-understood by the LLM.

## Key Insight

The LLM's next-token prediction loss on observed state transitions provides a universal intrinsic reward because it quantifies the agent's ability to generate actions that align with the LLM's pretrained world model, eliminating the need for task-specific reward engineering.

## Method

We propose **Next-State Perplexity Minimization (NSPM)**, a self-supervised reinforcement learning framework for LLM-based agents.

(A) **What it is**: NSPM is an RL algorithm that uses a frozen pretrained LLM as a dynamics model to compute an intrinsic reward equal to the negative perplexity of the next state given the current state and action. The agent's policy is updated via PPO to minimize this perplexity.

(B) **How it works** (pseudocode):
```python
# Hyperparameters
LLM = load_frozen_model("distilgpt2")   # 82M parameters, smaller causal LM
policy = MLP(state_dim=768, action_dim=A, hidden_dims=[256, 256], activation=nn.GELU)
lr = 3e-4                                # Adam learning rate
clip_epsilon = 0.2                       # PPO clip range
gae_lambda = 0.95
batch_size = 64
num_epochs = 10
max_steps = 50
# Compute: ~4 GPU hours on a single NVIDIA T4 for 100k episodes

for episode in range(num_episodes):
    s = env.reset()                        # s is string
    s_embed = LLM.embed(s)                 # use LLM's last hidden layer
    trajectory = []
    for t in range(max_steps):
        a = policy(s_embed).sample()
        s_next = env.step(a)
        prompt = f"State: {s} Action: {a} Next state:"
        loss = 0
        for token in LLM.tokenize(s_next):
            logits = LLM(prompt)[-1]
            loss += cross_entropy(logits, token)
        reward = -loss / len(s_next_tokens)
        trajectory.append((s_embed, a, reward, LLM.embed(s_next)))
        s = s_next
        s_embed = LLM.embed(s)
    advantages = compute_gae(trajectory, reward_to_go=False, gamma=1.0, lam=gae_lambda)
    for epoch in range(num_epochs):
        for batch in minibatch(trajectory, batch_size):
            states, actions, old_log_probs = batch
            new_log_probs = policy.log_prob(states, actions)
            ratio = exp(new_log_probs - old_log_probs)
            clipped_ratio = clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
            loss = -min(ratio * advantages, clipped_ratio * advantages)
            update(policy.parameters(), loss, lr)
```

(C) **Why this design**: We chose a frozen pretrained LLM rather than learning a dynamics model from scratch because the LLM already encodes broad semantic regularities, avoiding costly environment-specific training; the trade-off is that the LLM may be biased toward text-like dynamics and ignore environment-specific constraints. We selected PPO over simpler policy gradient methods (e.g., REINFORCE) because PPO's clipped objective prevents destructive policy updates when rewards are noisy (as intrinsic rewards often are); the cost is additional hyperparameter tuning (e.g., clip range). We compute reward as negative next-state perplexity (average per-token loss) rather than raw log-likelihood because perplexity normalizes by sequence length and is less sensitive to verbosity; however, this equally weights all tokens, potentially missing critical sub-sequences. We embed states using the LLM's hidden representation rather than a separate encoder to leverage the LLM's contextual understanding; the downside is that the policy may rely on spurious correlations in the embedding space.

Key differences from prior intrinsic motivation methods:
1. Unlike Random Network Distillation (RND) which uses a random network as target, NSPM leverages a pretrained LLM as a frozen dynamics model, requiring no additional training of a target network.
2. Unlike Intrinsic Curiosity Module (ICM) which uses forward prediction error of learned dynamics, NSPM uses the LLM's next-token perplexity, which is computed without any environment interaction and grounded in language-commonsense knowledge.
3. NSPM does not train any auxiliary network alongside the policy; the intrinsic reward is computed directly from the frozen LLM's forward pass, simplifying training.

(D) **Why it measures what we claim**: The negative next-state perplexity measures the agent's capability to produce actions that lead to state transitions consistent with the LLM's pretrained knowledge; this assumes that the LLM's next-token predictions accurately reflect plausible real-world dynamics for the target environment. When this assumption holds, high negative perplexity corresponds to actions that the LLM deems likely given the context, which correlates with goal-directed behavior in tasks that follow language-commonsense dynamics. This assumption fails when the environment dynamics diverge from the LLM's training distribution (e.g., in domains with specialized physics), in which case the reward may penalize correct but atypical transitions or reward trivial but predictable ones. The computational quantity `loss` measures state transition plausibility because the LLM was trained to approximate the distribution of natural and technical text sequences; the equivalence relies on the premise that the agent's environment can be faithfully serialized into text that the LLM can process. When state serialization loses critical features (e.g., spatial coordinates approximated as text), the loss may reflect text fluency rather than actual transition correctness.

**Assumption Verification**: Before applying NSPM to a new environment, we verify the load-bearing assumption by collecting 100 random trajectories and computing the Spearman rank correlation between the intrinsic reward (negative perplexity) and a simple extrinsic success indicator (e.g., whether the goal was eventually achieved). A positive correlation (ρ > 0.2) indicates that the LLM's perplexity aligns with goal progress; otherwise, the environment likely violates the language-commonsense assumption, and NSPM may fail. This calibration step ensures the method is only deployed where the assumption holds.

## Contribution

(1) A self-supervised reinforcement learning framework (NSPM) that derives an intrinsic reward from a frozen LLM's next-state prediction perplexity, enabling policy learning without external task rewards. (2) Empirical demonstration that NSPM achieves non-trivial task performance in a text-based game (e.g., from the Jericho benchmark) without any task-specific reward shaping, outperforming random exploration and a curiosity-driven baseline that uses prediction error on learned dynamics. (3) An analysis of the sensitivity to LLM model scale and state serialization format, identifying conditions under which the intrinsic reward aligns with task success.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | TextWorld | Tests language-commonsense dynamics |
| Violation environment | PHYSICA (simple physics grid world) | Tests assumption breakdown |
| Primary metric | Episode success rate | Reflects goal achievement |
| Baseline 1 | Random policy | Lower bound performance |
| Baseline 2 | PPO with sparse extrinsic reward | Tests intrinsic reward benefit |
| Baseline 3 | LLM planner (ReAct) | Tests RL over planning |
| Ablation-of-ours | NSPM with random LLM dynamics | Tests importance of frozen LLM |

### Why this setup validates the claim

TextWorld is a text-based game where state transitions follow natural language descriptions, aligning with the frozen LLM's pretrained knowledge. Comparing against a random policy establishes a floor; against PPO with only sparse extrinsic reward isolates the effect of the intrinsic perplexity reward; against an LLM planner (ReAct) tests whether RL with NSPM outperforms direct planning. The ablation with a random LLM verifies that the frozen LLM's knowledge is critical. Episode success rate directly measures the agent's ability to achieve goals, providing a clear falsifiable signal. The inclusion of PHYSICA, a physics grid world where state transitions are governed by physical laws rather than language commonsense, tests the assumption breakdown: here, the LLM's perplexity is expected to be uncorrelated with goal progress, leading to poor performance and confirming the limitation.

### Expected outcome and causal chain

**vs. Random policy** — On a task requiring multi-step reasoning (e.g., "cook a meal"), the random baseline selects actions uniformly, leading to failure because no coherent sequence is learned. Our method receives dense rewards for transitions that the LLM deems plausible (e.g., "take pan" then "turn on stove"), so it learns to compose such sequences. We expect a large success gap (e.g., 70% vs 10%) on tasks with clear commonsense dynamics.

**vs. PPO with sparse extrinsic reward** — On a long-horizon task with delayed reward (e.g., "escape the dungeon"), sparse PPO struggles to explore the correct path due to rare positive feedback. Our method provides intrinsic reward at every step for plausible state changes, guiding exploration toward meaningful subgoals. We expect our method to converge faster and achieve higher success (e.g., 60% vs 30%) on tasks where extrinsic reward is sparse.

**vs. LLM planner (ReAct)** — On a task requiring backtracking from dead ends (e.g., "solve a mystery"), ReAct's single-shot reasoning may lock into a wrong plan. Our method uses RL to iteratively refine the policy via environment interaction, allowing correction from negative intrinsic rewards (high perplexity transitions). We expect our method to have higher success on tasks that require trial-and-error (e.g., 50% vs 25%).

**Ablation: NSPM with random LLM dynamics** — Random LLM provides noisy intrinsic rewards uncorrelated with transition plausibility. The agent learns to follow spurious patterns, performing no better than random baseline. This confirms that the frozen LLM's pretrained knowledge is essential for the reward shaping to be beneficial.

**On violation environment (PHYSICA)** — In the physics grid world, state transitions obey physics (e.g., pushing objects) not language commonsense. The LLM's perplexity does not reflect correct actions; both NSPM and random baselines achieve low success (<10%), demonstrating that the assumption fails and NSPM is not universally applicable. This falsification confirms the limitation claimed in the method.

### What would falsify this idea
If NSPM fails to outperform PPO with sparse reward on tasks where state transitions reflect commonsense (e.g., TextWorld), or if the advantage is uniform across all tasks (not concentrated on those requiring language-consistent dynamics), then the central claim that next-state perplexity from a frozen LLM provides useful intrinsic reward is invalid. Additionally, if NSPM performs well on PHYSICA (the violation environment), it would indicate that the perplexity reward is capturing something other than language-commonsense dynamics, contradicting our explanation.

## References

1. Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning
2. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
3. Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks
4. Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. Chain of Thought Prompting Elicits Reasoning in Large Language Models
7. Prompt a Robot to Walk with Large Language Models
8. SayTap: Language to Quadrupedal Locomotion
