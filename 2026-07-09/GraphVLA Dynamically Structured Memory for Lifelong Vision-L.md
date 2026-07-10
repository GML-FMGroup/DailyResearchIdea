# GraphVLA: Dynamically Structured Memory for Lifelong Vision-Language-Action Models

## Motivation

Existing memory-augmented VLA models, such as the dual-vault design in Dual Latent Memory, rely on manual tuning of memory size and recency-based forgetting, which become brittle under interleaved tasks and unknown horizons. This structural limitation prevents scalability in lifelong settings where task boundaries are unknown and structurally important episodes may be temporally distant.

## Key Insight

Graph connectivity measures like PageRank naturally capture the structural importance of experiences in a transition graph, enabling automatic forgetting that prioritizes nodes critical for future actions regardless of their temporal recency.

## Method

**GraphVLA** is a memory-augmented VLA framework where experience is stored as a directed graph G = (V, E, w). Each node v has an embedding e_v ∈ ℝ^256 (from a ViT-B/16 observation encoder and a 2-layer MLP (hidden 128, ReLU) for action tokenization, concatenated and projected via a linear layer). Edges u→v have weight w_uv = (#successful transitions) / (#total transitions) along that edge, initialized to 1.0 for the first occurrence; if no transition occurs on that edge for 5 timesteps, weight decays by factor 0.9 per timestep; if weight < 0.1, edge is removed. Graph is updated online via insertion and pruning based on an incremental PageRank score (approximate personalized PageRank with Monte Carlo random walks of length 100, reset probability 0.15, updated every timestep; full PageRank recomputation via power iteration every 1000 steps). At each step, a subgraph is retrieved via random walk sampling: from the current context (the node for the previous step), a random walk of length k=5 is simulated, and the visited nodes (up to 50) are sampled as candidate set. Attention over these nodes (transformer with 4 layers, 8 heads, hidden 512) produces context vector c_t, which is combined with o_t embedding via a linear layer to predict action a_t. New node v_new is added after executing a_t, with embedding from (o_t, a_t, r_t), and edge v_prev → v_new with weight 1.0. Pruning: if |V| > max_nodes=10000, remove nodes with lowest PageRank score. Hyperparameters: max_nodes=10000, decay_rate=0.9, threshold=0.1, k=5, random_walk_length=100, reset_prob=0.15, transformer_layers=4, hidden=512, heads=8.

**Load-bearing assumption (explicit):** The directed transition graph with outcome-confidence edge weights accurately captures the causal structure of task dependencies, so PageRank scores reflect the structural importance of experiences for future actions. This assumption fails in highly stochastic environments where transitions are unreliable; PageRank may then amplify noise. To verify this, we conduct a synthetic gridworld experiment (see experiment section) where ground-truth importance is known, and measure correlation between PageRank and success-based oracle scores.

**Why this design:** We chose graph-based over vector-based memory because edges naturally capture causal structure, enabling forgetting based on connectivity rather than recency. We chose PageRank over simpler metrics (e.g., degree centrality) because PageRank accounts for both incoming and outgoing connections and is robust to noise; the cost is incremental O(log N) updates with approximate algorithms. We chose outcome-confidence edge weights over uniform weights to discount spurious transitions, improving robustness. We chose retrieval via random walk sampling over full-graph attention because it is computationally feasible at scale, though it may miss globally important distant nodes.

**Why it measures what we claim:** PageRank score measures structural importance of an experience node because it reflects how likely a random walk from recent nodes will visit that node; this assumes that the transition graph accurately captures task dependencies. This assumption fails when the environment is highly stochastic and transition edges are unreliable; in that case, PageRank may instead reflect noise patterns, and we rely on outcome-confidence weights and edge pruning to mitigate. The retrieval subgraph via random walk covers nodes that are reachable within k steps from the current context, which measures temporal proximity in the graph; this assumes that relevant experiences are near the current node in the graph, which fails when there are long-range dependencies; then the subgraph may miss crucial but distant nodes, but the attention mechanism can still attend to all sampled nodes. The pruning threshold max_nodes and decay rate are hyperparameters that indirectly control memory size, but the graph connectivity-based selection ensures that the most connected nodes remain, automatically adapting to task length. We diagnose the equivalence between global PageRank ranking and local retrieval distribution by computing Spearman correlation on a synthetic gridworld with known oracle importance.

**Pseudocode for incremental PageRank update:**
```
# per timestep, after adding new node
for i in range(random_walk_length):
    current = seed_node (recent node)
    if random() < reset_prob: current = seed_node
    else: current = random_outgoing_neighbor(current)
    PageRank[current] += 1 / random_walk_length
for v in V:
    PageRank[v] = (1 - reset_prob) * PageRank[v] + reset_prob * (1/|V|)
# Normalize? Approximate; full recomputation every 1000 steps using power iteration.
```

## Contribution

(1) A graph-structured memory for VLA models that dynamically stores and forgets experiences based on connectivity rather than recency, eliminating manual partition and retention tuning. (2) A pruning mechanism using PageRank scores that automatically retains structurally important episodes across varying task horizons. (3) An online retrieval method via random walk sampling that balances computational efficiency with relevance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | RLBench long-horizon tasks (e.g., pick_and_lift, stack_blocks, open_drawer) — each episode 50-100 steps | Requires memory over many steps. |
| Primary metric | Task success rate (averaged over 100 episodes) | Direct measure of task completion. |
| Baseline 1 | End-to-end VLA (no memory) — same encoder/decoder without graph | Tests need for memory. |
| Baseline 2 | Dual Latent Memory (DLM) — dual-vault with recency-based forgetting (vault size 1000) | Tests structured memory vs. ours. |
| Baseline 3 | MemER — flat episodic memory with k-NN retrieval (k=5) using feature similarity | Tests retrieval-based memory. |
| Baseline 4 | GraphVLA w/ degree centrality instead of PageRank (ablation) | Tests PageRank contribution. |
| Baseline 5 | GraphVLA w/ recency-only pruning (forgetting oldest nodes) instead of PageRank | Tests connectivity-based pruning. |
| Baseline 6 | Memory-Augmented Transformer (MAT) — with learnable read/write memory of size 1000 | Broader neural memory baseline. |
| Baseline 7 | Differentiable Neural Computer (DNC) — with memory size 1000 | Broader differentiable memory baseline. |
| Ablation | GraphVLA w/ uniform retrieval (random node sampling instead of PageRank) | Tests connectivity-based retrieval. |

### Why this setup validates the claim

This experimental design forms a falsifiable test of GraphVLA's central claim: that graph-structured memory with PageRank-based retrieval improves long-horizon manipulation. The dataset (RLBench long-horizon) inherently requires temporal reasoning across dozens of steps, making it a hard test of memory. The baselines isolate different sub-claims: end-to-end VLA tests whether any memory helps; Dual Latent Memory tests whether a dual-vector vault is sufficient; MemER tests whether simple experience retrieval works; MAT and DNC test whether learnable neural memory suffices; the degree centrality and recency-only ablation baselines isolate the specific benefit of PageRank over simpler connectivity metrics or forgetting rules. The uniform retrieval ablation tests the specific contribution of PageRank for retrieval. The success rate metric directly captures task completion, which depends on correct action sequences informed by memory. If GraphVLA truly captures causal structure, it should outperform all baselines on tasks with complex dependencies, while performing similarly on simple tasks. The ablation should degrade performance primarily on tasks requiring integration of distant experiences. Additionally, to verify the load-bearing assumption, we perform a synthetic gridworld experiment: create a 10x10 grid with deterministic transitions and known optimal paths, compute the oracle importance of each node (inverse of number of times visited in optimal trajectories), and measure Spearman correlation between PageRank and oracle importance over 1000 random episodes. We expect ρ > 0.7; if ρ < 0.3 under stochastic noise (transition noise 30%), the assumption fails and repair (outcome-confidence weights) is validated.

### Expected outcome and causal chain

**vs. End-to-end VLA (no memory)** — On a case where the robot must recall a precise gripper pose from 20 steps earlier, the baseline fails because it lacks any memory mechanism and relies only on current observation, leading to repeated errors or task failure. Our method instead retrieves that node via PageRank from the graph, because edges preserve the transition sequence and PageRank keeps important nodes accessible; we expect a large gap in success rate (e.g., >30% absolute) on long-horizon tasks, with parity on short tasks.

**vs. Dual Latent Memory** — On a case where the robot must chain multiple subgoals (e.g., pick A, place B, then pick C), Dual Latent Memory's dual-vector vault may conflate temporally distant experiences due to recency-biased forgetting, causing it to lose the first subgoal's context after the second. Our method, by maintaining a directed graph with PageRank, preserves the causal chain because edges explicitly link outcome sequences and pruning removes only low-connectivity nodes; we expect a moderate gap (e.g., 10-20% absolute) specifically on tasks requiring multistep chaining.

**vs. MemER** — On a case where the environment has stochastic transitions (e.g., object slippage causing varied outcomes), MemER's flat retrieval retrieves many irrelevant experiences due to feature similarity, diluting the context. Our method's PageRank, updated with personalized seeds from recent nodes, favors structurally important nodes even if features differ, because the graph's connectivity reflects reliable transitions; outcome-confidence weights further prune noisy edges. We expect our method to degrade less on stochastic tasks, with a gap especially on high-variance subsets.

**vs. GraphVLA w/ degree centrality or recency-only pruning** — On tasks requiring integration of distant but structurally important nodes, degree centrality may overemphasize hubs (nodes with many connections) while recency-only pruning may forget old but critical nodes. PageRank balances both, leading to better memory retention for temporally distal important experiences. Expect gap of 5-10% on long-horizon tasks.

**vs. MAT and DNC** — On tasks with clear causal structure, the explicit graph may outperform learned memory due to inductive bias, expect gap of 5-10%.

**Synthetic gridworld validation** — The Spearman correlation between PageRank and oracle importance should be >0.7 in deterministic settings; with 30% transition noise and uniform weights, correlation drops to <0.3, but with outcome-confidence weights it recovers to >0.6.

### What would falsify this idea

If GraphVLA shows uniform improvement over all baselines across all task subsets, rather than a pronounced advantage on tasks with long-range dependencies or complex causal structure, then the graph's connectivity mechanism is not the key driver. Alternatively, if the ablation with uniform retrieval performs similarly to full GraphVLA, then PageRank importance is irrelevant. If the synthetic gridworld correlation is low (<0.5) even with outcome-confidence weights, the structural importance assumption is flawed.

## References

1. Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation
2. MemER: Scaling Up Memory for Robot Control via Experience Retrieval
3. HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy
4. VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models
5. Commonsense Reasoning for Legged Robot Adaptation with Vision-Language Models
6. Language-Embedded Gaussian Splats (LEGS): Incrementally Building Room-Scale Representations with a Mobile Robot
7. SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM
8. Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding
