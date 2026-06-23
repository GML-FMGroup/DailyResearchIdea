# Value-of-Information-Guided Budgeting for Memory-Augmented MCTS Planning

## Motivation

Memory-augmented MCTS methods like MC-DML cache past evaluation results but still allocate simulation budget uniformly or via UCB, wasting computations on nodes whose value is already well-estimated. This wastes simulations and limits scaling to large branching factors or long horizons. For instance, MC-DML's in-trial memory captures immediate outcomes but does not prioritize exploration of nodes with high expected information gain, causing redundant rollouts on low-impact paths.

## Key Insight

The expected reduction in decision uncertainty, quantified as the difference between prior and posterior entropy of the value distribution, decomposes across nodes and can be computed online from the empirical value distribution stored in shared memory, making simulation budget allocation a tractable optimization problem.

## Method

### Current method

(A) **What it is:** Value-of-Information Guided Allocation (VIGA) is a budgeting mechanism for memory-augmented MCTS that, at each decision step, computes the VoI for each frontier node from the memory's population statistics and allocates a simulation budget proportional to the VoI, thereby minimizing the expected decision error per simulation.

(B) **How it works:**
```pseudocode
function VIGA(MCTS_node, memory, total_budget B):
    // B is total simulations for this decision step
    frontier = get_frontier_nodes(MCTS_node)
    for each node in frontier:
        node.memory_stats = retrieve(node.hash, memory)  // returns (mean, variance, count)
        node.value_uncertainty = variance / sqrt(count + 1)  // standard error
        // VoI estimate: reduction in variance after one simulation
        node.voi = (node.value_uncertainty^2) / (node.value_uncertainty^2 + 1)  // based on Kalman gain simplification
        // Correlation discount: adjust for shared descendant endpoints
        node.overlap_factor = compute_overlap(node.hash, frontier_hashes, memory)  // fraction of descendant endpoints shared
        node.voi_adjusted = node.voi / (1 + node.overlap_factor)
    total_voi = sum(node.voi_adjusted for node in frontier)
    for each node in frontier:
        node.budget = round(B * node.voi_adjusted / total_voi)  // allocate proportional to adjusted VoI
    // Perform simulations according to budget
    for (node, budget) in zip(frontier, budget_allocation):
        for _ in range(budget):
            result = simulate(node)
            update_memory(node.hash, result)
            update_node_statistics(node, result)
```
Hyperparameters: total_budget B (e.g., 50), uncertainty prior variance (set to 1.0 if no memory), overlap_factor computed via hash intersection of descendant endpoints (e.g., using a bloom filter of depth-3 endpoints).

(C) **Why this design:** We chose VoI computed from memory's empirical variance (node.value_uncertainty) over UCB's exploration bonus because UCB ignores the covariance between sibling nodes and does not directly estimate the information gain of a simulation; accepting that VoI calculation requires storing variance in memory, increasing memory overhead slightly. We chose a Kalman-gain approximation for VoI over a full Bayesian update because it is additive and lightweight; accepting that it assumes a Gaussian reward distribution, which may not hold exactly. We chose to allocate budget proportionally to VoI rather than greedily selecting the highest VoI node (sequential halving) because proportionality avoids starvation of nodes with moderate VoI and is simpler to implement; accepting that it may allocate simulations to nodes that are not the most informative in the absolute sense but contribute to overall uncertainty reduction. We added a correlation discount to adjust VoI for overlapping descendant endpoints, because ignoring correlations can waste simulations on redundant information; accepting that this adds computational overhead and relies on the memory's ability to track visited states.

(D) **Why it measures what we claim:** The value_uncertainty (standard error of mean) measures the confidence in the node's value estimate because smaller standard error implies tighter confidence interval; this assumes the memory's sample mean is an unbiased estimator of the true value, which fails when the simulation policy is biased (e.g., random rollouts in a domain with systematic model error), in which case value_uncertainty reflects only sampling variance. The VoI approximation (node.voi) measures the expected reduction in decision uncertainty because it is derived from the expected drop in variance after one additional observation under a Gaussian assumption; this assumption fails when the reward distribution is heavy-tailed or multimodal, in which case VoI may over- or under-estimate the true information gain. The budget allocation (proportional to VoI) operationalizes the principle of minimizing expected decision error per simulation under the assumption that reducing variance of value estimates minimizes expected decision error (under zero-one loss); this assumption fails when distributions are heavy-tailed or multimodal, in which case variance reduction may not correspond to decision error reduction. The correlation discount (node.voi_adjusted) corrects for the load-bearing assumption that nodes are independent; this requires that the memory can reliably identify overlapping descendant endpoints, which fails if the hash function has collisions or the descendant set is too large to fully store.

## Contribution

(1) A VoI-based budgeting mechanism that dynamically allocates MCTS simulation budget proportional to the expected information gain from each node, using memory-stored variance estimates.
(2) An analytical reduction of budget allocation to a tractable per-node computation, demonstrating that VoI can be approximated online without full Bayesian inference.
(3) A new principle for LLM planning: prioritize simulation effort on nodes where memory indicates high uncertainty, rather than exploration-exploitation trade-off.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Game of 24, Sokoban | Multi-step reasoning; replay memory beneficial. |
| Primary metric | Success rate per budget | Direct measure of decision accuracy. |
| Baseline: Uniform | Uniform MCTS | Equal simulations; no VoI prioritization. |
| Baseline: UCB | MCTS with UCB | Exploration bonus ignores memory statistics. |
| Baseline: Greedy | Greedy depth-first search | No simulation allocation; baseline without MCTS. |
| Ablation: NoMem | VIGA with uniform prior | Tests memory's role in VoI estimation. |
| Ablation: NoCorr | VIGA without correlation discount | Tests importance of correlation adjustment. |

### Why this setup validates the claim

Game of 24 and Sokoban require multi-step planning with many possible intermediate states. The replay memory stores visit statistics from previous episodes, enabling VoI estimation from variance. Comparing VIGA (with memory and correlation discount) to uniform MCTS tests whether VoI allocation improves sampling efficiency. The UCB baseline tests whether memory-informed VoI beats standard exploration bonuses that ignore cross-episode statistics. The greedy baseline establishes whether MCTS (even uniform) is necessary at all. The NoMem ablation isolates whether memory-derived variance provides additional benefit beyond a fixed prior. The NoCorr ablation directly tests the load-bearing assumption of independence; if NoCorr matches VIGA, then the correlation discount is unnecessary. Success rate under a fixed budget directly tests the claim that VIGA minimizes expected decision error per simulation.

### Expected outcome and causal chain

**vs. Uniform MCTS** — On a problem where one subtree has high outcome variance (e.g., multiple different correct paths), uniform MCTS allocates equal simulations to all nodes, wasting resources on already confident paths. VIGA allocates more simulations to high-variance nodes, reducing uncertainty faster. We expect a noticeable gap: VIGA solves ~20% more problems at low budgets (e.g., 50 simulations) but parity at high budgets (e.g., 200 simulations) as all methods eventually converge.

**vs. UCB MCTS** — On a problem with repeated subproblems across episodes (e.g., same intermediate target sum appears in different games), UCB's exploration bonus decays slowly and ignores prior visits from other episodes, over-exploring already confident nodes. VIGA uses memory to reduce variance estimates via prior statistics, focusing on truly uncertain nodes. We expect VIGA to outperform UCB by ~10-15% on problems with common subproblems, and parity on novel ones.

**vs. Greedy** — On a problem requiring backtracking (e.g., no single greedy path yields correct answer), greedy search fails early by committing to a wrong branch. VIGA explores multiple branches via MCTS and uses VoI to allocate simulations efficiently. We expect VIGA to solve >80% of such problems while greedy solves <20%.

**vs. NoCorr** — On domains where frontier nodes share many descendant endpoints (e.g., Sokoban with overlapping states), VIGA with correlation discount should outperform NoCorr by ~5-10% in success rate, as the discount avoids redundant simulations. On domains with low overlap (e.g., Game of 24 with distinct intermediate sums), performance should be similar.

### What would falsify this idea

If VIGA with memory and correlation discount performs similarly to VIGA without memory (NoMem) across diverse problems, then memory-derived VoI does not provide additional benefit. Alternatively, if NoCorr matches VIGA in Sokoban, then the correlation discount is unnecessary, contradicting the load-bearing assumption. If VIGA underperforms uniform MCTS on problems with low variance, then VoI allocation may be misdirected.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation
3. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory
4. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. Training language models to follow instructions with human feedback
7. Chain of Thought Prompting Elicits Reasoning in Large Language Models
8. Code as Policies: Language Model Programs for Embodied Control
