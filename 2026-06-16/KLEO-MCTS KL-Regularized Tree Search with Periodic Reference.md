# KLEO-MCTS: KL-Regularized Tree Search with Periodic Reference Reset for Exploration in Text-Based Games

## Motivation

In LLM-guided Monte Carlo tree search (MCTS) for text-based games, dynamic memory modules (as in MC-DML) introduce a persistent bias: the tree policy increasingly favors actions suggested by past experiences, causing premature convergence to suboptimal paths and reducing exploration. Existing memory-guided MCTS methods lack explicit mechanisms to counteract this drift. We observe that ProRL uses KL divergence with periodic reference resetting to prevent policy collapse during reinforcement learning. We transfer this structural property to the MCTS setting, hypothesizing that a similar regularizer can prevent memory-induced bias and sustain exploration across planning trials.

## Key Insight

Periodically resetting the reference distribution to the LLM's memory-free zero-shot output re-establishes an unbiased anchor, ensuring that the KL-penalized tree policy remains close to an explorer that is not corrupted by past experience, thereby maintaining exploration in the presence of accumulating memory.

## Method

### (A) What it is
KLEO-MCTS (KL-constrained Exploration with Online reference reset) is an MCTS planning algorithm for text-based games. It takes as input a large language model (LLM), a dynamic memory module (in-trial and cross-trial memory as in MC-DML) and outputs action sequences. At the core, it regularizes the tree policy during expansion with a KL divergence penalty against a periodically reset reference policy (the LLM's zero-shot action distribution without memory).

### (B) How it works
Pseudocode:
```python
# Initialize
π_ref = LLM_zero_shot_prompt()  # reference from memory-free prompt
reset_counter = 0
reset_interval = K  # hyperparameter, e.g., 10 trials
# Calibration: verify π_ref is not systematically biased (see (D))

for each planning trial t:
    tree = new Tree()
    for each simulation (UCB-based selection):
        # Selection and expansion as in standard MCTS
        if node is leaf and not terminal:
            # Get tree policy at node n: π_tree(n) = softmax(LLM_with_memory(state_n))
            π_tree = get_tree_policy(node, LLM, memory)
            # Compute KL divergence penalty
            kl_pen = β * KL(π_ref || π_tree)  # β ∈ {0.05, 0.1, 0.2}
            # Combine with UCB score: 
            # UCB = Q(n) + C * sqrt(log(N_parent) / N_n) - kl_pen
            # C ∈ {0.5, 1.0, 2.0}
            # Expand node and initialize statistics
            ...
        # Backpropagation as usual
    # After trial, update memory (in-trial and cross-trial)
    memory.update(tree, outcome)
    reset_counter += 1
    if reset_counter >= reset_interval:
        π_ref = LLM_zero_shot_prompt()  # reset reference
        reset_counter = 0

# At each expansion: compute π_tree as LLM probabilities via in-context learning with memory.
# KL divergence computed as -Σ π_ref(a) log(π_tree(a)) + constant.
# Hyperparameter search: grid over β, K, C; 5 runs per setting; 100 trials per game.
```

### (C) Why this design
Three key decisions: (i) We choose to apply the KL penalty during expansion rather than on the final action selection because the expansion step determines which nodes are explored; penalizing divergence at this stage directly shapes the tree's growth. (ii) We use a symmetric KL (KL(π_ref || π_tree)) instead of reverse KL because we want to penalize tree policy actions that are too confident where the reference is uncertain, which is more effective for exploration—at the cost of being slightly more conservative. (iii) We reset the reference after every K trials rather than every simulation to balance stability and computational overhead; a reset every trial would be costly and might destabilize the tree, while a very long interval allows bias to accumulate. The reset frequency K is a key trade-off: smaller K yields more exploration but less memory guidance, larger K vice versa. This design is not a trivial combination of ProRL and MCTS because the tree policy is a per-node distribution affected by memory, not a trained policy; the KL penalty integrates with the UCB formula in a way that prior work (e.g., MC-DML) does not support. Unlike using a simple count-based exploration bonus, our KL penalty directly ties exploration to the LLM's zero-shot distribution, which provides a task-agnostic signal.

### (D) Why it measures what we claim
The KL divergence penalty KL(π_ref || π_tree) measures the deviation of the memory-informed tree policy from the memory-free reference. This deviation is an operationalization of **memory-based bias** because π_tree is computed using memory that may over-represent past successful paths; by penalizing high KL, we enforce that the tree policy stays close to an unbiased explorer (π_ref). **Assumption A:** π_ref itself is an unbiased baseline—i.e., the zero-shot LLM without memory does not systematically favor suboptimal actions. **Failure mode F:** when π_ref is biased (e.g., due to pre-training data skew), the KL penalty may discourage sensible memory-guided actions, and the metric reflects conservatism rather than bias avoidance. To verify assumption A, we calibrate each game: sample 100 random states and compute average KL between π_ref and a uniform distribution over valid actions; if this exceeds 0.5 nats, we flag the game as potentially problematic and interpret results with caution. Periodic resetting of π_ref ensures that the reference remains anchored to the current zero-shot distribution, which evolves as the environment state changes? Actually, the zero-shot distribution is constant per state definition; but across trials, the state may vary, so resetting aligns the reference to the current state's zero-shot output. The computational quantity (KL divergence) combined with periodic resetting operationalizes **exploration maintenance** because the tree policy is regularized toward a distribution that is independent of past memory, thus preventing collapse to a narrow set of actions. The failure mode is when the environment requires heavy reliance on memory (e.g., long-term dependencies), where the KL penalty may harm performance; in that case, the metric reflects over-regularization.

## Contribution

(1) A novel algorithm, KLEO-MCTS, that integrates a KL divergence penalty with periodic reference resetting into the expansion phase of LLM-guided MCTS for text-based games, directly addressing memory-induced exploration collapse. (2) A design principle that the same structural mechanism (KL regularization + periodic reset) used in RL to prevent policy collapse can be adapted to online planning with dynamic memory, without requiring additional training. (3) Identification of the trade-off between reset frequency and exploration-exploitation balance, formalized via the β and K hyperparameters.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Jericho text-based games (e.g., Zork1, The Hitchhiker's Guide) | Standard benchmark for long-horizon reasoning, including exploration-heavy games. |
| Primary metric | Success rate (%) | Direct measure of task completion. |
| Baseline 1 | ProRL (RL-trained LLM) | Tests if planning beats RL training. |
| Baseline 2 | MC-DML (MCTS+memory, no KL) | Tests contribution of KL constraint. |
| Baseline 3 | LLM zero-shot (no memory, no MCTS) | Baseline for minimal reasoning. |
| Ablation of ours | KLEO-MCTS without KL penalty | Isolates effect of KL regularization. |
| Additional ablation | KLEO-MCTS with uniform reference (instead of zero-shot) | Tests sensitivity to π_ref bias. |

### Why this setup validates the claim

This setup directly tests the central claim that the KL penalty prevents memory-based bias and maintains exploration in MCTS planning. Comparing against MC-DML (memory without KL) isolates the KL contribution; outperforming it on exploration-heavy games would confirm the bias-reduction hypothesis. The uniform-reference ablation probes the assumption that π_ref is unbiased; if performance drops significantly, it indicates zero-shot bias is problematic. ProRL represents a strong RL alternative; if our method succeeds where ProRL fails due to sparse rewards, it demonstrates the advantage of structured planning. The zero-shot baseline establishes the necessity of memory and planning. The ablation removes only the KL term, so any performance drop relative to the full method must be attributed to that component. Success rate is the appropriate metric because it directly measures task completion, which is the ultimate goal. The choice of Jericho games provides diverse challenges—some requiring exploration—enabling a falsifiable test: if our method does not outperform MC-DML on those subsets, the claim is contradicted.

### Expected outcome and causal chain

**vs. ProRL** — On a game with sparse rewards (e.g., Zork1's trap door requiring a specific key), ProRL's RL training may never encounter the reward signal, resulting in near-zero success. Our method uses MCTS to simulate many trajectories, leveraging memory and KL regularization to explore diverse actions and exploit partial credit. We expect our success rate to be noticeably higher (e.g., >20% vs. <5%) on such games, demonstrating planning's advantage over pure RL in sparse-reward settings.

**vs. MC-DML** — On a game requiring exploration after repeated failures (e.g., The Hitchhiker's Guide's Babel Fish sequence with many dead ends), MC-DML's memory biases the tree toward previously successful branches, missing novel solutions. Our method's KL penalty pushes the tree policy back toward the zero-shot distribution, encouraging exploration of new actions. We expect a significant gap (e.g., 30% vs. 10% success) on such exploration-heavy games, but parity on linear games where memory alone suffices.

**vs. LLM zero-shot** — On any multi-step game, the zero-shot LLM lacks memory and cannot maintain coherent plans, resulting in near-zero success. Our method integrates memory and planning, so we expect success rates significantly above zero (e.g., 20% vs. 0%), confirming that memory and MCTS are essential for reasoning.

**vs. uniform reference ablation** — If π_ref is unbiased, performance should be similar to the full method; if π_ref is biased, the uniform reference may improve exploration, leading to higher success on exploration-heavy games. We expect the full method to match or slightly exceed the uniform ablation on most games, confirming that zero-shot is a reasonable anchor.

### What would falsify this idea

If KLEO-MCTS without KL penalty matches or exceeds the full method on exploration-demanding games, the KL constraint is unnecessary and the claim is false. Alternatively, if our method underperforms MC-DML on tasks requiring exploration, the KL penalty harms rather than helps. If the uniform-reference ablation significantly outperforms the full method, the assumption of unbiased π_ref is violated and the method's theoretical grounding weakens.

## References

1. ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
2. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
3. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
4. Multi-skill Mobile Manipulation for Object Rearrangement
5. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
6. Inner Monologue: Embodied Reasoning through Planning with Language Models
7. HybridFlow: A Flexible and Efficient RLHF Framework
8. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
