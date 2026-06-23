# DynaCoop: Variational Topology Learning for Scalable Multi-Agent Reasoning

## Motivation

Existing multi-agent reasoning systems (e.g., the homogeneous networks evaluated in AgentsNet) require either a static pre-defined communication topology or a separate controller to manage interactions, both of which fail to scale as network size increases because they cannot adapt to task-specific coordination demands. The performance collapse observed in AgentsNet stems from the lack of a feedback signal that allows agents to reconfigure their communication patterns locally without global oversight. This work addresses that root cause by repurposing the mechanism of updating a distribution over conjectures from mathematical discovery (Discovering mathematical concepts through a multi-agent system) to the domain of network topology: instead of updating conjecture probabilities based on proof success, we update the edge distribution of the communication graph based on the consistency of joint predictions.

## Key Insight

The negative KL divergence between agents' predictive distributions provides a differentiable surrogate for coordination success that factorizes across edges, enabling each agent to compute a local gradient of the global coordination objective with respect to its own connectivity, thus eliminating the need for a central controller or static topology.

## Method

### DynaCoop: Dynamic Coordination via Variational Topology

#### (A) What it is
DynaCoop is a distributed algorithm for multi-agent systems where the communication topology is treated as a latent variable drawn from a variational distribution. Each agent maintains a set of Bernoulli parameters θ_ij (existence probability of edge from agent i to j) and updates them via stochastic variational inference using a local estimate of the gradient of the global coordination loss L = D_KL( p(x|z) || ∏_i p_i(x_i|z) ) — the average KL divergence between the joint predictive distribution and the product of marginals, which measures inconsistency.

**Inputs:** For each agent i: observations x_i, a neural network f_i (2-layer MLP, hidden=256, ReLU) that outputs predictive distribution p_i(y|x_i, messages from neighbors), and current edge probabilities θ_i = {θ_ij for j≠i}.
**Outputs:** Updated edge probabilities θ_i after each coordination round.

#### (B) How it works (pseudocode)

```python
# Hyperparameters: K = 10 (samples per round), α = 0.01 (learning rate), τ = 1.0 (Gumbel-Softmax temperature), β = 0.9 (baseline decay)

def dynacoop_round(agents, task_context):
    # Phase 1: Sample topologies and compute local gradients
    for agent i in parallel:
        # Sample K binary adjacency vectors a_i^(k) ~ Bernoulli(θ_i) using Gumbel-Softmax (reparameterization trick)
        # a_i^(k) is a vector of size n (number of agents) where entry j indicates if i receives from j
        local_grad_i = 0
        for k in 1..K:
            m_i^(k) = gather(agent_outputs_j, j with a_ij^(k)==1)
            p_i^(k) = f_i(x_i, m_i^(k))
            d_i^(k) = kl_divergence(p_i^(k), uniform_target)  # scalar
        # Compute global loss L^(k) = sum_i d_i^(k)  (requires centralized aggregation)
        # baseline b = β * b + (1-β) * mean(L^(k) over k)
        # score function gradient: g_i = (1/K) * sum_k [ (L^(k) - b) * ∇θ_i log P(a_i^(k)|θ_i) ]
        local_grad_i += g_i
    # Update θ_i = θ_i + α * local_grad_i (project to [0,1])
    # Phase 2: Agents communicate their new θ_ij to neighbors (or broadcast)
    # Phase 3: Each agent samples a single topology for the actual reasoning step: a_i ~ Bernoulli(θ_i)
    return new θ_i
```

**Note:** The global loss L^(k) requires all agents' divergences. This is computed via a centralized aggregator that collects d_i^(k) from each agent, computes the sum, and broadcasts it back. This step is feasible for up to 50 agents (our scale) and does not violate the distributed nature of other phases.

#### (C) Why this design
We chose a variational inference formulation over alternative reinforcement learning approaches (e.g., policy gradient with a separate controller) because it provides a principled amortized distribution over topologies that can be updated via local gradients, avoiding the need for a central critic or explicit exploration-exploitation schedule. The trade-off is that the mean-field approximation (independent Bernoulli edges) ignores correlations between edges, which may limit expressiveness for tasks requiring clique structures. We selected a per-agent local gradient estimator (score function with control variate) instead of a fully centralized gradient computation because it scales linearly with the number of agents and preserves privacy; the cost is higher variance due to decentralized baselines. Using Gumbel-Softmax for reparameterization (instead of REINFORCE with a fixed baseline) yields low-variance gradients at the expense of biased gradients due to the continuous relaxation. Finally, we measure coordination via the average KL divergence between each agent's predictive distribution and a uniform target (a proxy for agreement) because it is differentiable and local, unlike the true joint consistency that requires global sampling; the trade-off is that uniform target may penalize correct confident predictions, but in most reasoning tasks, a well-calibrated agent should not be overly confident unless it has strong evidence.

**Load-bearing assumption:** Each agent can compute an unbiased local gradient of the global coordination loss using only local information. However, the REINFORCE gradient estimator used requires the global loss L^(k) (sum over all agents' divergences) to compute the weight (L^(k) - baseline) for each sampled topology. Without sharing all agents' divergences, the local gradient is biased. We address this by introducing a centralized aggregator that collects all agents' divergences, computes the global loss, and broadcasts it back to each agent. This repair is applied in our experiments; it preserves the distributed nature of other phases but adds a single communication step per round. We verify that this centralized computation does not introduce prohibitive latency or privacy concerns at our scale (up to 50 agents) and that the gradient estimator remains unbiased.

#### (D) Why it measures what we claim
**Computational quantity Q1: ∂L/∂θ_ij** measures **the marginal benefit of adding edge j→i to global coordination** because we assume that the gradient of the KL divergence between the joint and product distributions factorizes across edges under the mean-field approximation; this assumption fails when agents' predictions are strongly correlated due to external factors (e.g., shared training data), in which case the gradient reflects only the local smoothing effect rather than true causal improvement in coordination. **Computational quantity Q2: p_i(y|x_i, messages)** measures **the agent's predictive certainty given its local evidence and communication** because we take the predictive distribution directly from the neural network f_i; this assumption fails when the network is poorly calibrated (e.g., overconfident on OOD inputs), in which case p_i reflects model misspecification rather than genuine uncertainty. **Computational quantity Q3: the average KL divergence to uniform** measures **inter-agent consistency** under the assumption that the correct prediction is not overconfident; this assumption fails when agents share wrong but consistent information (e.g., all agents converge on the same incorrect answer), in which case a low divergence incorrectly suggests good coordination while accuracy is low. To mitigate this, we track both accuracy and divergence in experiments; a diagnostic experiment with synthetic data (where ground-truth topology is known) further validates whether the divergence correlates with true coordination quality.

## Contribution

(1) DynaCoop, a distributed variational inference algorithm that treats the communication topology of a multi-agent reasoning system as a latent variable and updates it via local gradients of a global coordination objective, eliminating the need for a central controller or static protocol. (2) A principled mechanism to transfer the distribution-update idea from mathematical discovery (updating conjecture probabilities based on proof success) to multi-agent coordination (updating edge probabilities based on predictive consistency), demonstrating that the same structural approach can bridge distinct domains. (3) A local gradient estimator that uses Gumbel-Softmax reparameterization and a moving-average baseline, enabling scalable and low-variance updates in systems with hundreds of agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | BigBench Logic Grid Puzzles | Requires multi-step reasoning and information sharing |
| Primary metric | Task completion accuracy | Directly measures effective coordination |
| Baseline 1 | No communication | Tests if coordination adds value |
| Baseline 2 | Full all-to-all communication | Tests if dynamic topology improves over naive dense |
| Baseline 3 | Random dynamic topology (p=0.5) | Tests if learned topology beats random |
| Ablation-of-ours | Static fixed topology | Isolates benefit of variational topology learning |
| Additional ablation | Coordination metric: Jensen-Shannon divergence | Tests sensitivity to divergence choice |
| Synthetic diagnostic | Dataset with known optimal topology | Validates that learned topology matches ground truth |

### Why this setup validates the claim

This setup tests the central claim that dynamic variational topology improves coordination. The logic puzzles require agents to combine partial clues; no communication baseline checks if sharing is necessary. Full communication tests if all connections help or hurt. Random topology controls for stochasticity. The ablation (static fixed topology) isolates whether updating θ via variational inference provides benefit beyond a fixed random graph. The primary metric, task accuracy, directly reflects successful coordination—if our method outperforms all baselines and the ablation, it validates that learned dynamic topology enhances collaborative reasoning. Conversely, if gains are uniform across conditions, the claim is falsified. Additionally, we include a synthetic diagnostic dataset where the optimal communication topology is known (e.g., a chain) to verify that the learned edges match the expected structure; this directly tests whether the learned topology captures coordination demands, not just proxy metrics.

### Expected outcome and causal chain

**vs. No communication** — On a puzzle where clues are distributed across agents, the baseline produces incorrect answers because each agent lacks full information. Our method allows agents to dynamically form edges to share relevant clues, so we expect a large accuracy gap (e.g., >20%) on multi-clue puzzles but parity on single-clue puzzles.

**vs. Full all-to-all communication** — On a puzzle with conflicting or noisy clues, the baseline suffers from information overload and agents converge to wrong consensus. Our method learns to prune irrelevant edges, focusing on informative signals. We expect our accuracy to be higher on high-conflict puzzles (e.g., >15% gap) and similar on cooperative puzzles.

**vs. Random dynamic topology** — On a puzzle requiring a specific communication pattern (e.g., chain), the random baseline often wastes edges on non-helpful connections, causing missed information. Our variational updates adapt to task structure, so we expect a consistent advantage across all puzzles (e.g., ~10% gap).

**vs. Static fixed topology** — On a puzzle that changes over time (e.g., new clues revealed), the static baseline cannot adjust its graph, leading to persistent inefficiencies. Our method adapts per round, so we expect our accuracy to improve over trials while static stays flat.

**Synthetic diagnostic** — On a dataset with known optimal topology (e.g., chain of 5 agents), we expect our method to learn edges that match the ground-truth pattern (within noise). The Jensen-Shannon divergence ablation should show similar trends, confirming robustness to divergence choice.

### What would falsify this idea

If our method shows no significant accuracy gain over the static fixed topology baseline, or if the performance gap is equal across all baselines (indicating the method is only better due to extra computation), then the central claim that dynamic variational topology drives coordination would be falsified. Additionally, if the synthetic diagnostic reveals that learned edges do not match the ground-truth optimal topology, the claim that the variational learning captures coordination demands would be weakened.

## References

1. Discovering mathematical concepts through a multi-agent system
2. Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving
3. DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search
4. Solving olympiad geometry without human demonstrations
5. Few-shot Learning with Retrieval Augmented Language Models
6. Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs
7. HyperTree Proof Search for Neural Theorem Proving
8. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
