# Nested Optimization with Associative Memory for Continual Learning in Open-Ended Games

## Motivation

AgentOdyssey agents exhibit critical limits in exploration and long-horizon planning, with performance scaling only weakly with base model strength. The root cause is that existing methods lack a hierarchical optimization structure where the agent's inner optimizer acts as a content-addressable memory, preventing efficient online adaptation without catastrophic forgetting. This structural gap is identified as the shared residual across multiple branches in recent continual learning research.

## Key Insight

Gradient-based meta-learning enables the inner optimizer's state to function as a content-addressable memory because the update direction encodes task-specific information, allowing recall and adaptation through optimization dynamics rather than explicit storage.

## Method

### (A) What it is
We propose NOAM (Nested Optimization with Associative Memory), a meta-learning framework where an agent's inner optimizer serves as a content-addressable memory that is updated via an outer meta-learning loop. The inner optimizer's parameters encode task-specific adaptations, and the outer loop learns to update these parameters quickly for new tasks. The key assumption guiding our design is that the hidden state of the learned RNN-based inner optimizer can store and retrieve task-specific adaptation information in a content-addressable manner, enabling continual learning without explicit memory.

### (B) How it works
```python
# NOAM pseudocode with concrete specifications
# Outer loop: meta-training across multiple game instances (16 instances, β=0.001)
for epoch in range(meta_epochs):
    sample game instance D from AgentOdyssey generator
    # Inner loop: adapt agent to current game instance
    initialize agent parameters θ (Neural Network with 2 hidden layers, 256 units each, ReLU) and inner optimizer state φ (hidden state of LSTM, initial zero)
    for step in range(inner_steps=5):
        # Agent interacts with environment, collects trajectory τ (up to 100 steps)
        τ = agent.act(θ, env)
        # Compute inner loss L_inner = task_success_reward + 0.01 * exploration_bonus (entropy of action distribution)
        g = ∇_θ L_inner(τ)
        # Update θ using inner optimizer: LSTM with hidden size 128, takes g and previous hidden state, outputs scaling factor s_t and new hidden state
        s_t, φ = inner_optimizer(g, φ)  # inner_optimizer is a 2-layer LSTM, hidden=128, output dim = |θ|, with tanh activation
        θ = θ - 0.01 * s_t ⊙ g
    # After inner loop, compute meta-loss L_meta = 0.5 * held_out_task_success + 0.3 * exploration_coverage (fraction of states visited) + 0.2 * memory_recall (accuracy on probe task)
    # Update outer parameters ψ (LSTM weights, 148,480 parameters) via gradients
    ψ = ψ - 0.001 * ∇_ψ L_meta
    # Optionally update agent's initial θ (meta-initialization) as well
```
Hyperparameters: α=0.01 (inner learning rate), β=0.001 (outer learning rate), inner_steps=5, inner_optimizer is a 2-layer LSTM with hidden size 128 (total 148,480 parameters), trained with meta-batch size 1 and 100 meta-epochs.

### (C) Why this design
We chose an RNN-based inner optimizer over a fixed optimizer (e.g., SGD) because the learned optimizer can adapt its update rule to different game dynamics, enabling associative memory through its hidden state. The trade-off is increased computational cost during meta-training (RNN forward/backward). We chose to update the inner optimizer's parameters (ψ) via gradient-based meta-learning rather than evolutionary methods because gradients provide precise credit assignment, though they require second-order derivatives. We chose to train the outer loop on 16 game instances from AgentOdyssey rather than a single game to encourage generalization, accepting that meta-training is more expensive (approx. 2 GPU days on a V100). We chose to include exploration bonus (coefficient 0.01) in the inner loss to address the exploration limitation noted in AgentOdyssey, but this introduces a tuning parameter. The 5 inner steps balance between allowing enough adaptation and preventing overfitting. We will verify the load-bearing assumption by probing hidden state similarity across tasks.

### (D) Why it measures what we claim
The inner loop's adaptation speed (measured by performance improvement over inner_steps) measures continual learning ability because it quantifies how quickly the agent can incorporate new information without forgetting; this assumption fails when the inner optimizer overfits to a single game instance, in which case the metric reflects memorization rather than adaptation. The outer loop's meta-loss (composite of performance, memory recall, exploration) measures generalization across game distributions because it aggregates over multiple episodes; this assumption fails when the sampled games are too similar, in which case the metric reflects within-distribution performance. The inner optimizer's hidden state (memory) measures content-addressable recall because it is updated by gradient-based meta-learning to encode task-specific updates; we probe this by computing cosine similarity between hidden states after adaptation to different tasks and correlating with task reward structure similarity; this assumption fails when the hidden state collapses to a trivial representation, in which case similarity is uniformly high and uncorrelated with task similarity. The exploration bonus in inner loss measures exploration because it penalizes low diversity actions; this assumption fails when the bonus dominates the task reward, in which case the metric reflects random behavior.

## Contribution

(1) A novel framework, NOAM, that implements nested optimization with associative memory by using a learned inner optimizer as a content-addressable memory for continual learning. (2) Empirical demonstration that NOAM agents outperform baselines on AgentOdyssey's diagnostic metrics including world knowledge acquisition and exploration coverage, while mitigating catastrophic forgetting. (3) Open-source implementation of NOAM integrated with AgentOdyssey's procedural generation engine.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | AgentOdyssey (16 instances for meta-train, 16 for meta-test) | Procedural text games for continual learning |
| Primary metric | Held-out task success rate | Measures generalization across tasks |
| Baseline 1 | Static policy | No adaptation, tests need for learning |
| Baseline 2 | Inference-only | No memory update, tests memory importance |
| Baseline 3 | Episodic buffer (replay buffer of 1000 experiences) | Simple memory, tests our associative memory |
| Baseline 4 | MAML + replay buffer (MAML with inner loop SGD and external replay) | Meta-learning with explicit memory |
| Ablation-of-ours | NOAM with fixed optimizer (SGD, learning rate 0.01) | Tests learned optimizer contribution |

### Why this setup validates the claim

This experimental design forms a falsifiable test of NOAM's central claim: that a meta-learned inner optimizer acting as associative memory enables rapid adaptation and generalization across long-horizon tasks. The AgentOdyssey dataset provides diverse, procedurally generated game instances, ensuring that the agent must continually adapt rather than memorize a fixed distribution. The static policy baseline isolates the need for any learning; inference-only isolates the need for parameter updating; episodic buffer tests whether a simple replay memory can match our content-addressable approach; MAML + replay buffer tests a strong meta-learning baseline with explicit memory. The ablation of the learned optimizer (replaced with SGD) directly tests the contribution of the meta-learned update rule. The held-out task success rate on 16 held-out instances is the appropriate primary metric because it captures both adaptation speed (via performance on new tasks) and generalization (across instances). Additionally, we will conduct a probing experiment: after each inner loop on a held-out task, we record the hidden state of the inner optimizer and compute pairwise cosine similarities between tasks. We then correlate these similarities with task similarity (measured by reward function distance). If the correlation is positive and significant (Spearman's ρ > 0.5), it supports the associative memory claim.

### Expected outcome and causal chain

**vs. Static policy** — On a new game instance with an unfamiliar reward structure, the static policy repeats the same actions, achieving near-zero success (expected ~5%) because it cannot adjust. Our method's inner loop, guided by the learned optimizer, quickly updates parameters to align with the new reward, enabling rapid success (expected ~60%). Thus, we expect a large gap on first-instance tasks.

**vs. Inference-only** — On a multi-task sequence where early knowledge is required later, inference-only (which never modifies weights) forgets earlier context or fails to integrate it, leading to plateauing performance (e.g., flat at 20% after task 2). Our method's outer loop updates the inner optimizer's memory state, allowing recall of relevant past adaptations, so success rate should increase (e.g., 40% after task 2, 60% after task 4). Expected signal: NOAM shows a positive trend across tasks, while inference-only does not.

**vs. Episodic buffer** — On a compositional task requiring combining two previously seen sub-skills, the episodic buffer replays random past experiences, causing interference and slow integration; success might reach 30%. NOAM's associative memory retrieves exactly the relevant update rules via gradient history, enabling seamless composition; we expect success ~70% on such tasks. Observable signal: NOAM outperforms buffer significantly on compositional tasks but parity on simple tasks.

**vs. MAML + replay buffer** — MAML with a replay buffer uses explicit storage and replay of past gradients, which is memory-intensive and may cause interference on long sequences. On a task requiring recall of a specific adaptation from 10 tasks ago, MAML+replay may suffer from retrieval noise (expected ~40% success), while NOAM's implicit memory through optimizer state can retrieve precisely (expected ~65% success).

**Probing experiment**: We expect cosine similarity of hidden states to correlate with task similarity (Spearman ρ > 0.5). If not, the associative memory claim is challenged.

### What would falsify this idea
If NOAM's advantage over all baselines is uniform across task types (simple vs. compositional) or if the ablation with SGD performs equally well, then the central claim of associative memory and learned optimizer is unsupported. Specifically, observing that the episodic buffer or MAML+replay matches NOAM on compositional tasks would invalidate the memory claim. If the probing experiment shows no correlation between hidden state similarity and task similarity, the assumption of content-addressable memory is falsified.

## References

1. AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents
