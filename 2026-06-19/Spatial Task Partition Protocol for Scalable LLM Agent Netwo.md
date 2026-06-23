# Spatial Task Partition Protocol for Scalable LLM Agent Networks

## Motivation

Prior multi-agent LLM benchmarks like AgentsNet assume small networks where agents can maintain full task context, but scaling to many tasks exceeds context windows, causing performance degradation. This bottleneck arises from the structural assumption that each agent must be aware of all tasks, preventing scalability. To address this, we need a protocol that provably bounds per-agent context while ensuring global task coverage.

## Key Insight

By partitioning the task space into contiguous regions assigned to individual agents and adjusting region boundaries via local workload consensus, the protocol ensures each agent's task count is bounded by its context window, and communication is limited to boundary shifts, decoupling scalability from total task count.

## Method

**(A) What it is:** Spatial Task Partition Protocol (STPP) — input: set of agents, task space with coordinates (assumption: task space T is embedded in a spatial coordinate system such that tasks in close spatial proximity share similar characteristics; verified by measuring pairwise task embedding cosine distance within vs. across initial grid cells); output: assignment of tasks to agents and dynamic region boundaries. **(B) How it works:** ```pseudocode // Hyperparameters: M = floor((context_window - prompt_overhead) / L_max) (max tasks per agent), theta (workload imbalance threshold, default 0.2), L_max (max token length per task, default 512) Initialize: Partition task space T into regions R_i for each agent a_i (e.g., grid cells). Each agent maintains local task queue Q_i. Global directory stores region boundaries (lightweight consensus). function AgentLoop(agent a_i): while True: // Phase 1: Local task execution if Q_i not empty: task = pop(Q_i) execute task via LLM (within context window) // Phase 2: Workload monitoring and boundary shift request if |Q_i| > M * (1 + theta): send boundary_shrink_request to neighbors if |Q_i| < M * (1 - theta): accept boundary_expand_request from neighbors // Phase 3: Consensus on boundary changes if request received: evaluate based on local workload; if accept, update global directory via Paxos over leader set (4 leaders) reassign tasks in shifted region to new owner ``` **(C) Why this design:** We chose contiguous spatial regions over random distribution because spatial locality reduces communication overhead (agents only interact with neighbors). We used a threshold-based workload trigger rather than periodic rebalancing to minimize unnecessary boundary shifts, accepting that some transient imbalance may occur. We implemented a lightweight consensus (Paxos over a small set of leaders) instead of full distributed consensus to reduce communication cost, acknowledging a single point of failure risk. We opted for explicit task reassignment upon boundary change rather than lazy reassignment to ensure each task is covered by exactly one agent, accepting the overhead of atomic migration. **(D) Why it measures what we claim:** The bound M on tasks per agent directly operationalizes "bounded context" because it ensures no agent's task queue exceeds the LLM's maximum input length, assuming each task description fits within L_max tokens (M = floor((C - P)/L_max), where C is context window, P is prompt overhead). This measures bounded context when task token lengths are uniform ≤ L_max; failure mode: tasks with length > L_max cause context overflow even if count ≤ M, so M must be reduced. The region boundary adjustment protocol measures "scalability" because it keeps per-agent task count independent of total number of tasks, provided the task space can be partitioned into regions with density variations bounded by theta. This assumption fails when tasks are so densely concentrated that even the smallest region exceeds M, requiring hierarchical partitioning.

## Contribution

(1) Introduces Spatial Task Partition Protocol (STPP), a decentralized protocol for LLM agent networks that provably bounds per-agent context by partitioning task space and dynamically adjusting region boundaries based on local workload. (2) Derives the design principle that contiguous spatial partitioning combined with threshold-triggered boundary shifts achieves communication-efficient scalability. (3) Provides a lightweight consensus mechanism for updating global region boundaries, requiring only O(log N) messages per boundary shift.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Spatial Grid Tasks (SGT-1K) | Spatial tasks with varying density; includes sparse and dense clusters; task token lengths are uniformly ≤ 512 tokens. |
| Primary metric | Task success rate | Directly measures task completion quality |
| Secondary metric | Context overflow rate (fraction of tasks executed while agent queue exceeded M after accounting for token length) | Measures how often per-agent context bound is violated despite count bound |
| Baseline: Homogeneous Agents | All agents same, no spatial structure | Tasks assigned randomly; no region concept |
| Baseline: Fully Connected | All-to-all communication network | No spatial regions; global communication |
| Baseline: Centralized | Single LLM handles all tasks | Tests bounded context constraint |
| Ablation: Static Partition | Fixed grid regions, no boundary shift | Tests necessity of dynamic rebalancing |

### Why this setup validates the claim
This experimental design tests the central claim that STPP provides bounded context and scalability via spatial partitioning and dynamic workload balancing. The dataset includes both sparse and dense task clusters to stress-test the region adjustment mechanism. The homogeneous baseline isolates the effect of spatial locality: if STPP outperforms it, then partitioning reduces context overflow. The fully connected baseline tests whether local neighbor communication suffices compared to global messaging. The centralized baseline directly probes the bounded-context benefit: if STPP handles larger task sets than a single LLM, then distribution works. The ablation of static partitions reveals whether dynamic boundary shifts are necessary for balancing workload under density variations. The primary metric, task success rate, captures overall solution quality and implicitly reflects context failures and communication overhead. The secondary metric, context overflow rate, explicitly measures violations of the per-agent context bound due to task token length variability. Together, these components form a falsifiable test that can attribute performance differences to specific design choices.

### Expected outcome and causal chain

**vs. Homogeneous Agents** — On a dense task cluster, homogeneous agents randomly receive tasks, quickly exceeding the per-agent context limit M and failing due to truncated prompts. Our method assigns each agent a contiguous spatial region, so tasks are spread evenly; when one region becomes dense, the protocol shrinks its boundary to shift tasks to neighbors, keeping each agent below M. We therefore expect a noticeable gap on dense subsets but parity on sparse subsets.

**vs. Fully Connected** — On a task requiring coordination between two distant agents (e.g., moving an object across regions), the fully connected network has all agents exchange messages about the object, causing high overhead and potential miscoordination. Our method restricts communication to neighbors and only consensuses on boundary changes, so coordination is handled implicitly via region assignment. We expect STPP to show lower communication overhead and faster completion times on tasks spanning multiple regions.

**vs. Centralized** — On a large-scale task set exceeding the LLM's context window, the single agent cannot process all tasks and fails entirely. Our method partitions tasks across agents, each keeping its queue below M, so all tasks can be executed. We expect STPP to achieve non-zero success on large task sets where the centralized baseline fails completely, with the gap widening as task count increases.

### What would falsify this idea
If STPP performs worse than homogeneous agents on all task distributions—especially those where spatial locality is irrelevant—then the claim that spatial partitioning reduces overhead is false. Also, if dynamic boundary adjustment does not improve over static partitions in terms of success rate, then the workload balancing mechanism is unnecessary.

## References

1. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
2. A Scalable Communication Protocol for Networks of Large Language Models
3. Problem-Solving in Language Model Networks
4. War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars
5. S3: Social-network Simulation System with Large Language Model-Empowered Agents
6. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate
7. PaLM: Scaling Language Modeling with Pathways
8. GLM-130B: An Open Bilingual Pre-trained Model
