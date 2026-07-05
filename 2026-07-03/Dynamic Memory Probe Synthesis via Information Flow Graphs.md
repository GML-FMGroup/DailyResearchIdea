# Dynamic Memory Probe Synthesis via Information Flow Graphs

## Motivation

AgenticSTS relies on handcrafted scenarios within a single game (Slay the Spire 2), making cross-domain memory evaluation labor-intensive and domain-specific. The root cause is that manual scenario design cannot scale to arbitrary environments, preventing systematic isolation of memory capabilities (e.g., recall, tracking) across domains. Our method automates this by extracting the causal dependency structure of environment dynamics.

## Key Insight

The minimal memory length required for optimal performance in any task is determined by the causal dependency structure of the environment's transition dynamics; by extracting this structure, we can generate diagnostic tasks that isolate specific memory capabilities without human annotation.

## Method

### (A) What it is
**Dynamic Memory Probe Synthesis (DMPS)** takes an environment simulator with state space S, action space A, transition function T, and observation function O, and outputs a parametrized test suite where each task isolates a specific memory capability (e.g., recall distance, tracking precision). Each task is labeled with its required memory type and minimal memory demand (in steps).

### (B) How it works
```
Algorithm: DMPS
Input: Environment dynamics (S, A, T, O), horizon limit H, memory types M = {‘recall’, ‘tracking’}, symbolic variable set size limit L=50
Output: Task scenarios with memory labels

1.  Abstract state into symbolic variables V (e.g., discrete predicates) using hierarchical abstraction based on domain knowledge (e.g., grouping pixel coordinates into room labels) to keep |V| ≤ L.
2.  [New] Calibrate causal discovery on a held-out subset of environments with known ground-truth dependencies (e.g., Minigrid with explicit door states). Verify that estimated dependency windows match true memory length within ±1 step.
3.  For each variable v ∈ V, compute its *information dependency window*:
    - Sample N=1000 random trajectories of length H=50.
    - For each time t, compute the minimal k such that v(t) is predictable with accuracy > 95% using only observations from [t-k, t-1]. Use approximate conditional independence test based on mutual information (threshold 0.1, estimated via k-nearest neighbors with k=5) to determine predictability.
    - Store k as the *memory demand for v*.
4.  Construct Information Flow Graph G:
    - Nodes: (v, t) for each variable at each time.
    - Directed edges: v(t) → w(t’) if w(t’) depends on v(t) (via Granger causality or conditional independence test). Use the FCI algorithm (Spirtes et al., 2000) to handle latent confounders. Edge weight = |t - t’| (memory delay).
5.  For each memory type m ∈ M:
    - If m == ‘recall’: find paths in G where a variable v(t) influences a goal predicate g(t+Δ) with Δ ≥ d. Parameter d (memory length).
    - If m == ‘tracking’: find cycles or recurrent dependencies where summing or averaging past values is needed.
6.  Generate task scenario:
    - Set initial state such that the critical information (e.g., v(t)) is only available at a specific past time.
    - Set goal to achieve a state that requires that information after delay d.
    - Record (task, d, m) as test case.
```

**Assumption (load-bearing):** The approximate conditional independence tests (based on mutual information) and the FCI algorithm correctly identify causal dependencies even in the presence of latent confounders. We mitigate this by using FCI, which explicitly models unobserved confounders, and by validating on a controlled subset (step 2).

### (C) Why this design
We chose to extract the information flow graph via conditional independence tests rather than a pre-defined state abstraction (e.g., feature engineering) because it captures causal rather than correlational dependencies, accepting the computational cost of repeated trajectory sampling (O(|V|^2 × H × N)). We parametrize memory demand by a single integer d instead of generating tasks for all possible combinations of variables, which reduces generation time but may miss interactions where multiple memory types are required simultaneously. We label tasks by memory type based on graph topology (linear chains for recall, branching for tracking) because it ensures each task tests the intended capability, though some tasks may be solvable by alternate strategies that bypass memory (e.g., using shortcuts in the environment). The decision to use a 95% predictability threshold for dependency windows balances sensitivity (catching weak dependencies) and specificity (avoiding spurious correlations), at the risk of underestimating memory demand when noise is high. We conduct a sensitivity analysis over thresholds (90%, 95%, 99%) to verify robustness. Compared to prior retrieval-based baselines (e.g., Toolformer), our method generates tasks from dynamics rather than retrieving from a library, providing a formal guarantee that the generated tasks truly require memory given the assumption of optimal reward maximization.

### (D) Why it measures what we claim
**Equivalence**: Dependency window length d measures minimal memory capacity required for optimal performance. Assumption A: The optimal policy's reward-maximizing strategy must rely solely on the predictability of the variable from past observations. Failure mode F: If the environment provides redundant information at decision time (e.g., observable proxy variables), d overestimates actual memory need. In that case, d reflects the redundancy rather than true memory requirement. The computational quantity ‘minimal memory length’ d (from dependency windows) measures memory capacity because it quantifies the time span over which information must be retained from past observation to future decision; this assumes that the task cannot be solved by other heuristics that bypass memory (e.g., reactive planning using immediate sensory input). When the environment provides redundant information that allows bootstrapping without memory (e.g., the same information is available again later), the d value overestimates the actual memory requirement, in which case the test becomes a proxy for whether the agent exploits shortcuts rather than true memory. The information flow graph’s edge weights measure *causal necessity* of past observations for current decisions; this assumes that the conditional independence tests correctly identify causal links. If the environment has latent confounders that create spurious dependencies, the graph may include unnecessary edges, leading to tasks that require memory only when all confounders are unobserved. We mitigate this by using FCI, but failure is possible if the confounding structure is too complex. The graph topology (linear vs. branching) measures the type of memory (recall vs. tracking) because recall requires a single past fact while tracking requires integrating multiple pieces of information over time; this assumes that the agent’s optimal policy must use the corresponding computational strategy. If the environment can be solved by a simpler strategy (e.g., lookup table), the topology label becomes meaningless, and the test may fail to discriminate memory types.

## Contribution

["(1) A method to automatically generate diagnostic memory-probing tasks from any environment's transition dynamics, eliminating the need for manual scenario design.", '(2) A framework for decomposing memory capabilities into constituent demands (recall, tracking) based on the topology of an information flow graph extracted from environment dynamics.', '(3) A cross-domain memory benchmark suite that requires no human annotation, enabling systematic evaluation of long-horizon agent memory.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AgenticSTS, Minigrid, DSGBench environments | Diverse long-horizon dynamics. Include a subset (e.g., 3 Minigrid rooms) with known ground-truth dependencies for calibration. |
| Primary metric | Task validity (%) | Measures label-action mismatch. |
| Secondary metric | Shortcut detection rate (%) | Fraction of tasks solvable without intended memory. |
| Baseline 1 | Random task generation | No memory-based generation. |
| Baseline 2 | Heuristic rule-based generation | Handcrafted memory tasks. |
| Baseline 3 | Toolformer retrieval generation | Retrieval-based library. |
| Baseline 4 | PC algorithm-based generation | Causal structure learning without latent confounder handling. |
| Ablation | DMPS without FCI (standard conditional independence tests) | Removes latent confounder handling. |
| Ablation | DMPS with threshold sensitivity (90%, 99%) | Tests robustness of 95% threshold. |

### Why this setup validates the claim
This design tests whether DMPS tasks genuinely isolate memory demands. Using diverse environments (card game, grid navigation, strategic games) ensures generality. The primary metric—task validity (fraction of tasks where actual memory need matches labeled d)—directly measures the method's core promise. The secondary metric—shortcut detection rate—identifies tasks where agents cheat via shortcuts, demonstrating DMPS's ability to pinpoint failure modes of existing benchmarks. Baselines contrast absence of structure (Random), oversimplified heuristics (Heuristic), data-driven but non-causal retrieval (Toolformer), and causal discovery without latent confounder handling (PC algorithm). The ablation removes the information flow graph's causal handling, isolating its contribution. The sensitivity analysis over thresholds ensures the method is not brittle. Together, they form a crisp test: if DMPS outperforms baselines on validity and reveals shortcut failures, it proves that causal dependency analysis produces more accurate memory probes.

### Expected outcome and causal chain

**vs. Random** — On a case where a key variable is only observable once (e.g., enemy intent in Slay the Spire), Random may label a task with low d but actually require long memory, causing false negatives. DMPS computes dependency windows from trajectories, so it correctly assigns high d to such tasks. We expect DMPS to achieve >80% validity while Random stays near chance (50%).

**vs. Heuristic** — On a case requiring tracking a moving average (e.g., cumulative reward in Minigrid), Heuristic might create a too-simple task missing integration of multiple time steps. DMPS detects cyclic dependencies in the information flow graph, labeling it as tracking with appropriate d. We expect DMPS to show >20% higher validity on tracking tasks, with Heuristic failing on complex temporal dependencies.

**vs. Toolformer** — On a novel environment not in the retrieval library, Toolformer returns irrelevant tasks. DMPS generates tasks from the environment's own dynamics, so it always produces valid cases. We expect DMPS validity to remain high (>75%) while Toolformer drops below 30% for unseen environments.

**vs. PC algorithm** — On a case with latent confounders (e.g., unobserved random seed affecting multiple variables), PC algorithm may infer incorrect dependencies, leading to invalid tasks. DMPS using FCI handles confounders, so it achieves higher validity (>80% vs. <60%).

### What would falsify this idea
If DMPS achieves validity no higher than Random on tasks requiring recall > 10 steps, or if its advantage disappears on tracking tasks despite graph cycles, then the causal dependency analysis fails to capture true memory demand, refuting the claim that information flow graphs isolate memory capabilities. Additionally, if the shortcut detection rate is not significantly higher than in random tasks, then DMPS does not identify failure modes.

## References

1. AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents
2. DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning
3. DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments
4. UrzaGPT: LoRA-Tuned Large Language Models for Card Selection in Collectible Card Games
5. The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs
6. ExpeL: LLM Agents Are Experiential Learners
7. Large Language Models can Learn Rules
8. GameBench: Evaluating Strategic Reasoning Abilities of LLM Agents
