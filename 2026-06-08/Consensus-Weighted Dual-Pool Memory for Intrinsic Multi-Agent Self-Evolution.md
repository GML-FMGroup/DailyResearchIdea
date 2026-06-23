# Consensus-Weighted Dual-Pool Memory for Intrinsic Multi-Agent Self-Evolution

## Motivation

Existing self-evolving multi-agent systems like DecentMem rely on an external LLM judge to reweight exploitation and exploration memory pools, introducing cost, latency, and a single point of failure. The root cause is that the reweighting signal is not generated within the agent population itself. Replacing this external dependency with an intrinsic, decentralized signal is necessary for scalable and autonomous agent evolution.

## Key Insight

Cross-agent trajectory adoption entropy provides an intrinsic proxy for trajectory quality because, in homogeneous task settings, widespread adoption correlates with high utility, enabling decentralized consensus to replace external evaluation.

## Method

(A) **What it is**: We propose **Consensus-Weighted Dual-Pool (CDP)**, a method that intrinsically reweights each agent's exploitation and exploration memory pools using the entropy of cross-agent trajectory adoption, eliminating the need for an external LLM judge. Input: per-agent dual-pool trajectories. Output: updated pool weights. (B) **How it works** (pseudocode):

```python
# Per agent i, after each task episode
# Agent i broadcasts a hash of its chosen trajectory
# All agents receive hash set H from all agents

# Compute global trajectory frequency
freq = {}  # trajectory_hash -> count
for hash in H:
    freq[hash] = freq.get(hash, 0) + 1
N = len(H)

# Compute total entropy over trajectories
total_entropy = 0.0
for hash, count in freq.items():
    p = count / N
    if p > 0:
        total_entropy -= p * log(p)

# Normalize entropy to [0,1]
if N > 1:
    entropy_norm = total_entropy / log(N)
else:
    entropy_norm = 0.0

# Compute intrinsic reward for trajectory t_i used by agent i
# consensus_score = 1 - entropy_norm
# Intrinsic reward = consensus_score - baseline (average consensus over recent episodes)
intrinsic_reward = (1 - entropy_norm) - moving_avg_baseline

# Update exploitation pool weight for trajectory t_i
w_e[t_i] += learning_rate * intrinsic_reward
# Ensure non-negative
w_e[t_i] = max(0, w_e[t_i])

# If reward > threshold, promote to exploitation pool with strong weight
if intrinsic_reward > promotion_thresh and t_i in exploration_pool:
    move_to_exploitation(t_i, weight=start_weight)

# Adjust global exploration probability beta
# If total_entropy is high (disagreement), increase exploration
beta = sigmoid(scale * (total_entropy - 0.5 * log(N)))
```

Hyperparameters: `learning_rate=0.1`, `promotion_thresh=0.3`, `scale=2.0`. (C) **Why this design**: We chose entropy as the intrinsic signal over alternatives like prediction error or information gain because entropy directly quantifies consensus, which is a natural emergent property in cooperative multi-agent systems. The trade-off is that entropy may converge to local optima if agents prematurely agree on suboptimal trajectories; to mitigate this, we adjust the exploration probability β to be high when entropy is high, encouraging diversity. We chose to broadcast trajectory hashes rather than full trajectories to minimize communication overhead, accepting some loss of semantic granularity—this is acceptable because hashes still capture adoption patterns, and fine-grained differences are unlikely to affect the consensus signal at scale. We use an additive update with a moving average baseline to reduce variance and avoid weight explosion; a multiplicative update (e.g., Bayesian) would provide stronger priors but require hand-tuned priors for each domain. The promotion threshold prevents exploration pool candidates from entering exploitation too easily, balancing the rate of consolidation against the risk of premature locking. (D) **Why it measures what we claim**: The computational quantity `total_entropy` measures **decentralized consensus** because it computes the information-theoretic uncertainty in trajectory adoption across agents, under the assumption that agents are homogeneous in task objectives and environment; this assumption fails when agents have heterogeneous capabilities or local observations, in which case entropy reflects coordination rather than utility. The normalized `consensus_score = 1 - entropy_norm` measures **intrinsic trajectory quality** as perceived by the collective, assuming that widespread adoption correlates with high task utility in a shared reward setting; this assumption fails when agents have complementary strengths and low adoption still yields high individual utility, causing consensus to underestimate quality. The moving average baseline measures **expected consensus** to center the reward, assuming that deviations from average signal meaningful differences; this assumption fails under non-stationary task distributions where the baseline lags, potentially penalizing transiently novel trajectories. Together, these components operationalize intrinsic reweighting without an external judge.

## Contribution

(1) A novel decentralized memory reweighting mechanism that uses cross-agent trajectory adoption entropy as an intrinsic reward signal, replacing the external LLM judge in self-evolving multi-agent systems. (2) The design principle that consensus entropy can serve as a proxy for trajectory quality in homogeneous multi-agent settings, enabling fully autonomous memory evolution. (3) A concrete algorithm (CDP) with hyperparameter choices and trade-offs, ready for implementation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Overcooked-AI (collaborative cooking) | Cooperative trajectory adoption required |
| Primary metric | Average task success rate | Measures collective utility |
| Baseline 1 | No memory (independent agents) | Tests if memory reuse improves performance |
| Baseline 2 | Fixed uniform pool | Tests if adaptive weighting is beneficial |
| Baseline 3 | Centralized LLM judge | Tests if decentralized consensus is sufficient |
| Ablation | CDP without entropy-driven exploration | Tests if adaptive exploration is key |

### Why this setup validates the claim

This combination creates a direct falsification test for the claim that entropy-weighted dual-pool reweighting replaces an external judge. Overcooked-AI requires agents to coordinate on subtasks; success depends on adopting effective trajectories. The "no memory" baseline isolates the benefit of any shared experience, while "fixed pool" tests the contribution of adaptive weighting versus static sharing. Crucially, comparing to a centralized LLM judge tests whether the decentralized entropy signal achieves similar quality without an external oracle. The ablation of removing entropy-driven exploration isolates whether the exploration modulation is necessary for the improvement. The primary metric, average task success rate, directly captures whether the collective improves, as the method aims to boost collective utility through intrinsic reweighting.

### Expected outcome and causal chain

**vs. No memory** — On a case where agents must coordinate a complex recipe, the no-memory baseline repeatedly fails due to independent exploration without reuse. Our method instead shares trajectory hashes, allowing agents to converge on successful strategies via consensus weighting, so we expect a large absolute gap (e.g., 20-30% success rate improvement).

**vs. Fixed uniform pool** — On a case where some trajectories are mildly suboptimal but widely adopted, fixed pool weights dilute signal, slowing convergence. Our method rewards consensus differentially and boosts exploration when disagreement is high, so we expect faster convergence and 5-10% higher success rate after the same number of episodes.

**vs. Centralized LLM judge** — On a case where the LLM judge has high inference cost and latency, our decentralized method avoids this bottleneck. The entropy signal may occasionally misjudge quality (e.g., when silence is actually beneficial), but overall we expect comparable success rate (within 5%) while being more scalable.

### What would falsify this idea

If the CDP method performs measurably worse than the fixed uniform pool baseline (i.e., adaptive weighting hurts), or if its advantage over the centralized judge is entirely due to the exploration modulation (ablation equals full method), then the core claim that consensus entropy can replace an external judge is falsified.

## References

1. Self-Evolving Multi-Agent Systems via Decentralized Memory
2. A-MEM: Agentic Memory for LLM Agents
3. From Commands to Prompts: LLM-based Semantic File System for AIOS
4. AIOS: LLM Agent Operating System
5. MemGPT: Towards LLMs as Operating Systems
6. LLM as OS, Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem
