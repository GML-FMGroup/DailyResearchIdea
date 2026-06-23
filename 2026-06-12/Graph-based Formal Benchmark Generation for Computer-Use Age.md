# Graph-based Formal Benchmark Generation for Computer-Use Agents

## Motivation

Existing benchmark construction methods for computer-use agents, such as BeTaL, rely on LLMs or human judges to evaluate task correctness, perpetuating oracle dependence and introducing biases. The root cause is the lack of a formal, decidable notion of ground truth; every benchmark generation tree assumes a judge is necessary for quality assurance. This structural limitation prevents scalable, unbiased benchmark creation and validation.

## Key Insight

Graph reachability provides a formal, decidable ground truth for benchmark correctness, eliminating the need for any external judge because task completion reduces to checking whether a path exists in the state graph.

## Method

### (A) What it is
**GFB** (Graph-based Formal Benchmark) autonomously constructs computer-use benchmarks by modeling the environment as a deterministic finite automaton (state graph) and generating tasks whose correctness is algorithmically decidable via path existence.

**Inputs:** A specification of the environment (e.g., UI state types, actions with preconditions/postconditions) or a set of observed traces.
**Outputs:** A benchmark of tasks, each defined by an initial state $s_0$ and a goal condition $g$ (a set of target states), where correctness is $\exists$ path $s_0 \leadsto s_g \in g$.

### (B) How it works

```python
# Pseudocode for GFB task generation
Input: Environment specification E (states S, actions A, transition function T)
      Difficulty parameters: L_min, L_max, coverage_ratio
Output: Benchmark B = {(s0, g)}

1. Build state graph G = (V=S, E={(s, s') | s' in T(s, a) for some a in A})
2. Prune unreachable states (BFS from initial state set) -> reachable subset R
3. For each target difficulty level d:
   a. Sample initial states uniformly from R
   b. For each initial state s0, sample goal state sg from nodes reachable
      with path length in [L_min_d, L_max_d] (using BFS depth layer)
   c. Define goal condition g = {sg} (single state) or a set for generality
   d. If path length > threshold, add intermediate waypoint constraints
4. Compute coverage: ensure each action type appears in at least one task
5. Output tasks with metadata: path length, branching factor
```

### (C) Why this design
We chose graph reachability over LLM-based verification because the latter introduces judge bias and cannot guarantee correctness—a fundamental problem identified in the forest-level motivation. This decision trades off the need for a complete state space specification (which may require upfront engineering) against the benefit of oracle-free evaluation. Second, we use path length as the primary difficulty controller rather than, e.g., number of distractors, because path length has a direct formal interpretation (step count) and is independent of LLM randomness; however, it may not capture semantic complexity such as reasoning about tool outputs. Third, we sample goals from reachable states only, ensuring every task is feasible, which avoids the pitfall of generating impossible tasks that would waste agent evaluation, but it may miss tasks that require the agent to discover novel states not in the initial graph—a risk we accept because the graph can be iteratively expanded from observed traces. Fourth, we apply coverage constraints on action types to avoid degenerate benchmarks that only test a subset of agent capabilities, trading off some diversity in state coverage for behavioral breadth.

### (D) Why it measures what we claim
The state graph G represents the set of all possible action sequences in the environment. The existence of a path from s0 to sg measures **formal task correctness** because the graph encodes all deterministic transitions; a path is both necessary and sufficient for task completion under the assumed determinism of the environment. This assumption fails when the environment is nondeterministic or partially observable, in which case path existence would only indicate possibility, not guaranteed success. The decision to use graph search (e.g., BFS) as the judge measures **oracle independence** because the correctness check is a purely algorithmic procedure with no learned parameters or human input; it is fully deterministic and reproducible. This assumption fails if the graph abstraction is incomplete (missing states or actions due to abstraction error), in which case the benchmark would measure correctness only with respect to the model, not the real environment. The difficulty parameter (path length) measures **task difficulty** because longer paths require more planning steps and hence more complex agent behavior, under the assumption that step count correlates with problem-solving effort; this assumption fails when a long path is trivially navigable via learned shortcuts, in which case path length overestimates difficulty. By covering all action types, we measure **behavioral coverage** of the agent, assuming that each action type tests a distinct capability; this fails if some actions are substitutable, leading to redundant coverage.

## Contribution

(1) A framework for autonomously constructing computer-use agent benchmarks with formal ground-truth independence via graph-based state abstraction. (2) A method for generating tasks with controlled difficulty using graph-theoretic properties (path length, branching factor) that is verifiable without any human or LLM judge. (3) A design principle that formal verifiability eliminates oracle dependence, applicable to any environment that can be modeled as a deterministic finite automaton.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | τ-bench airline + hotel domains | Realistic tool-use scenarios with known ground truth |
| Primary metric | Formal task accuracy (path existence in ground-truth graph) | Directly measures correctness without judge bias |
| Baseline 1 | τ-bench (dynamic user simulation) | Strong baseline for real-world agent evaluation |
| Baseline 2 | BeTaL (LLM-verified benchmark generation) | Automated benchmark with LLM judge, suffers bias |
| Baseline 3 | Random goal sampling (no graph guarantee) | Tests value of formal verification |
| Ablation | GFB without action coverage | Isolates effect of ensuring behavioral breadth |

### Why this setup validates the claim
This design tests the central claim that GFB provides oracle-free, formally correct evaluation. The τ-bench baseline measures how close GFB gets to a gold-standard, human-designed benchmark with high ecological validity. BeTaL represents the state-of-the-art in automated benchmark generation but relies on an LLM judge, which introduces bias and cannot guarantee correctness. Random sampling tests whether any formal structure is needed. The ablation removes the action-coverage constraint to see if benchmark diversity affects measured agent performance. The primary metric—formal task accuracy—directly encodes the correctness property, and if GFB outperforms baselines on this metric, it confirms that graph-based verification yields more reliable evaluations. The metric is also deterministic, avoiding LLM randomness. An observed advantage would imply that formal guarantees are crucial for trustworthy benchmarking.

### Expected outcome and causal chain

**vs. τ-bench** — On a task requiring precise multi-step tool calls (e.g., booking a flight with specific dates and price range), τ-bench evaluates via a simulated user comparing database states, which is accurate but costly and not scalable. GFB instead uses a precomputed state graph to verify correctness via path existence, achieving similar accuracy with zero human effort. We expect GFB's accuracy to match τ-bench’s on simple tasks but potentially exceed it on complex tasks where human simulation errors occur. However, the gap should be small (within 5%) because both use ground truth transitions.

**vs. BeTaL** — On a task with ambiguous goal wording (e.g., "find a cheap flight"), BeTaL’s LLM judge may mark correct plans as wrong due to interpretation differences, or mark wrong plans as correct if the LLM hallucinates. GFB avoids this by checking if the agent’s action sequence reaches a goal state defined in the graph, which is unambiguous. We expect GFB to have significantly higher formal accuracy (e.g., >90% vs. BeTaL’s ~70%) on tasks requiring precise reasoning, especially when LLM judge is unreliable.

**vs. Random goal sampling** — On a task with no coverage constraints, random sampling may generate many tasks that only test trivial actions (e.g., clicking a button), missing complex multi-step behaviors. GFB’s action coverage ensures each action type appears in some task, forcing agents to demonstrate diverse skills. We expect GFB to reveal broader agent strengths/weaknesses, with lower variance in per-action success rates (e.g., across action types, GFB’s success rate standard deviation <0.15 vs. random’s >0.3).

**vs. GFB without action coverage** — On a domain with imbalanced action frequencies (e.g., many simple clicks, rare complex API calls), the ablation will generate benchmarks that overrepresent easy tasks, inflating agent accuracy. GFB with coverage corrects this. We expect the ablation to show higher average accuracy but lower discrimination (i.e., small differences between agent models) compared to full GFB, confirming that coverage forces harder tasks.

### What would falsify this idea
If GFB’s formal accuracy is not significantly higher than BeTaL’s on tasks with long reasoning chains (where LLM bias is expected to be largest), or if the advantage is uniform across all task difficulties, then the claim that graph-based verification reduces bias is unsupported. Also, if the ablation with action coverage does not produce more balanced per-action success rates, the coverage mechanism is ineffective.

## References

1. Automating Benchmark Design
2. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
3. Automatic Generation of Benchmarks and Reliable LLM Judgment for Code Tasks
4. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion
5. MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use
6. WebArena: A Realistic Web Environment for Building Autonomous Agents
7. AgentBench: Evaluating LLMs as Agents
8. Identifying the Risks of LM Agents with an LM-Emulated Sandbox
