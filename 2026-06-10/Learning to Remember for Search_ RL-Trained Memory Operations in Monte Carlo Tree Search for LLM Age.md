# Learning to Remember for Search: RL-Trained Memory Operations in Monte Carlo Tree Search for LLM Agents

## Motivation

MC-DML uses heuristic trial-outcome-based memory updates that fail to capture task-specific dependencies and cannot adapt dynamically, leading to suboptimal planning. Memory-R1 learns optimal memory operations from rewards but only in a single-step retrieval setting, not integrated with search. The root cause is that memory and search are treated as separate processes; there is no mechanism for memory updates to be guided by the search state's importance estimates. We propose to condition memory operations on MCTS node statistics, enabling the memory policy to learn which information to store based on its projected utility for future search.

## Key Insight

MCTS node statistics (value estimates and visit counts) provide a sufficient statistic for the expected utility of storing information about that node, allowing the memory policy to learn an optimal memory update strategy from task rewards.

## Method

We present Memory-Augmented MCTS with Learned Memory Management (MAM-MCTS).

(A) **What it is**: MAM-MCTS augments standard MCTS for LLM agents in text-based games with a learned memory manager that, at each search step, observes the current search tree state and outputs a discrete memory operation (ADD, UPDATE, DELETE, NOOP) to an external memory bank. The memory manager is a small MLP trained with REINFORCE using game-level reward.

**Load-bearing assumption**: MCTS node statistics (value estimates and visit counts) provide a sufficient statistic for the expected utility of storing information about that node. This assumption is explicitly stated and verified empirically via calibration analysis (see experiment).

(B) **How it works** (pseudocode):
```
Input: LLM, environment, memory manager policy π_θ (2-layer MLP, hidden 64, tanh)
Initialize empty memory M (list of text strings)
Hyperparameters: N=50 simulations per step, D_memory=2 (depth for memory ops),
learning rate α=1e-3, baseline decay β=0.99, reward discount γ=1.0

for each episode:
    state = env.reset()
    while not terminal:
        root = Node(state)
        for sim in range(N):
            node = root
            # Selection using UCB with LLM score:
            while not node.is_leaf():
                node = argmax_{child} V(child) + c * sqrt(log(N_visits(parent))/N_visits(child))
            # Expansion: LLM generates top-3 actions
            actions = LLM_generate_actions(node.state, memory=M)
            for a in actions:
                child = Node(state=transition(node.state, a))
                node.add_child(child)
            # Simulation: LLM predicts value v (0 to 1)
            v = LLM_rollout(node.state)
            # Backpropagation
            backpropagate(node, v)
            # Memory update at leaf nodes of depth D_memory:
            if node.depth == D_memory and node.is_leaf:
                feats = extract_features(node)  # [V(node), visits(node), avg_sibling_V, avg_sibling_visits, depth]
                op_probs = π_θ(feats)  # softmax over 4 ops
                op = sample(op_probs) if training else argmax(op_probs)
                if op == ADD:
                    entry = f"{node.state} -> {node.parent.action} -> success prob {v}"
                    M.append(entry)
                elif op == UPDATE:
                    key = retrieve_similar(embed(node.state), M, threshold=0.8)
                    if key: M.replace(key, updated_entry)
                elif op == DELETE:
                    key = retrieve_similar(embed(node.state), M, threshold=0.8)
                    if key: M.remove(key)
                # NOOP: do nothing
        # Select best action from root (most visited)
        action = argmax_{child} visits(child)
        state, reward = env.step(action)
    # After episode, compute cumulative reward R
    # For each memory operation decision (feats_t, op_t) log π_θ(op_t|feats_t)
    # Update policy using REINFORCE with baseline b (running average reward, updated as b ← β*b + (1-β)*R):
    # θ += α * (R - b) * ∇_θ log π_θ(op_t|feats_t)
```

(C) **Why this design**: We chose to condition on MCTS node statistics rather than raw state or history because these statistics summarize the search tree's evaluation of state importance, which is directly relevant for deciding what to remember. We opted for REINFORCE over PPO because memory operations are rare per episode (~10-20), making importance-sampling unstable with small sample sizes; the trade-off is higher variance but simpler implementation. We set D_memory=2 to intervene early in the search tree, balancing computational cost and effectiveness—deeper updates would be more informative but increase memory overhead. We used a small MLP rather than an LLM for the policy because features are low-dimensional (5 floats), avoiding unnecessary computation; the cost is that the policy cannot leverage natural language context directly. For UPDATE and DELETE, we use a pre-trained Sentence-BERT to embed state descriptions and retrieve via cosine similarity with threshold 0.8, accepting that retrieval may fail if state descriptions are too dissimilar. This design explicitly builds on Memory-R1's RL training of memory operations but adapts it to the search setting by using search statistics as input, rather than query-based features. Unlike a simple combination of Memory-R1 and MC-DML, our method couples memory and search at the level of node statistics, not by heuristic trial outcomes. The load-bearing assumption that MCTS node statistics are a sufficient statistic for memory utility is verified empirically via a calibration check (see experiment).

(D) **Why it measures what we claim**: The input features (value estimates and visit counts) operationalize the concept of "projected utility of memory" because value estimates measure state promise, and visit counts measure exploration confidence; the assumption is that their combination captures the expected benefit of storing information about that state. This assumption fails when the LLM's value estimates are poorly calibrated (e.g., systematically overconfident), in which case the features reflect noise rather than utility. The discrete operation choice (ADD/UPDATE/DELETE/NOOP) measures the policy's ability to discriminate between beneficial and harmful memory updates; the REINFORCE gradient relies on the episode-level reward as the ground truth, assuming that a memory operation's effect on future search is monotonically related to final reward. When multiple memory operations interact non-monotonically (e.g., early ADD helps later UPDATE), credit assignment becomes noisy, and the policy may converge to a locally optimal strategy. The memory retrieval for UPDATE/DELETE via cosine similarity measures relevance of stored information, assuming that embedding similarity corresponds to task-relevant similarity; this assumption fails when subtle distinctions matter (e.g., "red key" vs "blue key" embedded closely), causing incorrect retrievals.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Jericho text-based games | Standard benchmark for text-based games |
| Primary metric | Success rate | Directly measures task completion |
| Secondary metric | Expected Calibration Error (ECE) on 512 held-out states | Checks assumption A: node statistics predict memory utility |
| Baseline 1 | Memory-R1 | Learns memory ops from queries |
| Baseline 2 | MC-DML | MCTS with LLM, no explicit memory |
| Baseline 3 | Fixed heuristic: ADD if V>0.5 and visits>5, else NOOP | Isolates necessity of learning memory ops |
| Ablation of ours | MAM-MCTS w/o memory ops (always NOOP) | Isolates effect of memory management |

### Why this setup validates the claim
This setup provides a direct test of our central claim: that learned memory management conditioned on search tree statistics improves MCTS performance. The Jericho benchmark includes games that require both planning and memory across multiple steps. Comparing against Memory-R1 tests if using search statistics as features outperforms query-based features; comparing against MC-DML tests if adding explicit memory to MCTS helps; the fixed heuristic baseline tests whether learning is necessary; the ablation isolates the memory component. Success rate is the appropriate metric because it reflects the ultimate task goal. Additionally, we validate the load-bearing assumption by computing Expected Calibration Error (ECE) on a held-out set of 512 game states: for each decision, we compute the predicted probability of beneficial memory update (derived from π_θ's NOOP probability as a proxy) vs. observed future utility (reward impact). If ECE is low (<0.15), the assumption holds; otherwise, the statistics are poorly correlated with utility, indicating a need for recalibration. Expected outcome: our method should outperform all baselines on games requiring memory, with the largest gap on long-horizon tasks, and ECE should be below 0.15 on games where success rate improves.

### Expected outcome and causal chain

**vs. Memory-R1** — On a case where the agent must remember a sequence of object interactions across multiple rooms (e.g., pick up key, unlock door), Memory-R1’s memory operations rely on query similarity to past states, which may fail when states are semantically similar but strategically different (e.g., two doors with different keys). Our method conditions memory decisions on search tree statistics (value and visits), enabling it to prioritize storing information about states that are deemed promising by the search. We expect our method to achieve noticeably higher success rate on multi-step goal tasks (e.g., 15–20% improvement) but parity on single-step tasks.

**vs. MC-DML** — On a case where the optimal action depends on information gathered several steps earlier (e.g., "use the rusty key" after exploring a drawer), MC-DML’s pure MCTS must rely on the LLM’s context window to retain that information; when the context exceeds the window, it fails. Our method explicitly stores such information in an external memory, which can be retrieved during subsequent searches. Thus, we expect our method to maintain performance on long episodes (e.g., >10 steps) while MC-DML degrades after a few steps, leading to a substantial gap (e.g., 30–50% relative improvement on long episodes).

**vs. Fixed heuristic** — The fixed heuristic (ADD if value>0.5 and visits>5) will underperform our learned policy because it cannot adapt to task-specific patterns; e.g., it may store irrelevant states (high value but redundant) and fail to update/deallocate. We expect our method to outperform by at least 10% success rate on all multi-step games, confirming that learning is necessary.

### What would falsify this idea
If the ablation (MAM-MCTS without memory ops) performs similarly to our full method, or if our method’s gains are uniform across all game types (not concentrated on tasks requiring memory across steps), then the central claim that learned memory management conditioned on search statistics is beneficial would be falsified. Additionally, if the calibration analysis shows ECE > 0.2 (indicating node statistics poorly predict memory utility), the load-bearing assumption is violated and the method's design rationale is undermined.

## References

1. Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning
2. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
3. LgTS: Dynamic Task Sampling using LLM-generated sub-goals for Reinforcement Learning Agents
4. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
5. Chain of Thought Prompting Elicits Reasoning in Large Language Models
6. LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models
7. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
8. MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory
