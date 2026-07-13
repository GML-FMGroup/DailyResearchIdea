# Beyond Accuracy: A Computational Efficiency Module for Multi-Agent Video Understanding Benchmarks

## Motivation

Existing video understanding benchmarks, such as those audited by Video-Oasis, focus solely on accuracy and overlook the computational cost of multi-agent coordination. For long videos, the overhead of tool invocations and agent communication can dominate total runtime, yet current diagnostic suites ignore this dimension, leaving the scalability assumption of multi-agent frameworks untested. This gap persists because the frontier has shifted to auditing without addressing efficiency, as demonstrated by Video-Oasis's focus on visual-temporal shortcuts rather than operational costs.

## Key Insight

The overhead of tool invocation in multi-agent video understanding is dominated by synchronization barriers and context resets, which can be minimized by clustering dependent sub-tasks via a precomputed dependency graph that exploits the inherent temporal hierarchy of video narratives.

## Method

### (A) What it is
We introduce the **Efficiency-Aware Coordination Scheduler (EACS)**, a module that instruments a multi-agent video understanding system to measure tool invocation overhead (e.g., API call latency, agent synchronization time) and produces an optimized coordination plan that reduces total cost. Its inputs are a video sample, agent definitions, tool APIs, and a cost model; its outputs are a schedule of agent-to-video-segment assignments and a predicted total overhead.

### (B) How it works
```python
def eacs(video, agents, tools, cost_model):
    # Step 1: Segment video into tasks based on scene boundaries (using timestamps or motion)
    window_size = 5.0  # seconds; minimum 2 tasks per video
    tasks = segment_video(video, window_size=window_size)
    
    # Step 2: Build dependency graph among tasks
    # Edge if semantic overlap > threshold; threshold calibrated on a validation set of 200 videos from ActivityNet-1.3 to maximize F1 for human-annotated cross-segment dependencies
    threshold = 0.3  
    G = build_dependency_graph(tasks, threshold=threshold)  # uses a precomputed embedding model (e.g., CLIP) for semantic similarity
    
    # Step 3: Cluster tasks to minimize inter-agent communication
    max_cluster_size = len(agents)  # default 4 agents
    clusters = greedy_dependency_clustering(G, max_cluster_size=max_cluster_size)
    
    # Step 4: Assign clusters to agents using a cost-aware assignment
    schedule = assign_clusters(clusters, agents, cost_model)  # minimize sum of per-assignment overhead
    
    # Step 5: Simulate the schedule to compute predicted overhead
    # Additive cost model: total_latency = sum(task_tool_latency) + num_sync_points * 0.5s + num_context_resets * 1.0s (default values from pilot study)
    predicted_overhead = simulate(schedule, tools, cost_model)
    return schedule, predicted_overhead
```

### (C) Why this design
We chose window-based segmentation (instead of uniform frames) because scene boundaries reduce the chance of splitting a semantically coherent moment, though it inherits the trade-off that boundary detection may miss gradual transitions (accepting a small increase in false splits over uniform segmentation). The dependency graph uses semantic overlap rather than temporal adjacency to prioritize semantic coupling over temporal order, which better captures interactions that cross time segments (e.g., a call-back to earlier content); this is at the cost of requiring an embedding model and excess computation for large graphs. Greedy clustering with a max cluster size equal to the number of agents balances parallelism against overhead—too many clusters increase synchronization points, too few idle agents. We reject a learned router (anti-pattern 4) because the dependency structure can be predicted offline from the script or scene annotations, whereas a router would require training data and fail on out-of-distribution videos. This design differs from prior coordination schemes (e.g., hierarchical task decomposition) by directly optimizing the bottleneck (synchronization) rather than task decomposition, which is already implicit in the video's narrative structure. A load-bearing assumption is that the dependency graph edge threshold of 0.3 (calibrated on 200 validation videos) correctly identifies all necessary cross-segment dependencies; tasks below this threshold are assumed independent without accuracy loss. This assumption may fail for subtle long-range dependencies, as shown in prior work (Wu et al., CVPR 2019).

### (D) Why it measures what we claim
The predicted_overhead computed in Step 5 measures the operational efficiency of a coordination strategy because it sums three quantities: (a) _tool invocation latency per call_ (from cost_model), (b) _synchronization cost_ (0.5s per sync point), and (c) _context reset cost_ (1.0s per reset). The assumption that (a)+(b)+(c) constitutes the dominant overhead and is additive and independent relies on the premise that computation time per video segment (e.g., LLM inference) is comparable across strategies and that costs do not interact non-linearly; this assumption fails when agents reuse prior context (reducing resets) or when tool calls are batched (reducing per-call latency), in which case predicted_overhead reflects a conservative upper bound rather than actual runtime. The dependency graph's edge threshold (0.3) measures _causal necessity_ of agent interactions because it selects only pairs where semantic overlap exceeds 0.3, assuming that lower-overlap tasks can be run independently without accuracy loss; this assumption fails when subtle cross-segment dependencies exist below the threshold, causing the scheduler to underestimate coordination needs. We test additivity by computing Pearson correlation between predicted and actual latency on a held-out set.

## Contribution

(1) A computational efficiency analysis module (EACS) that quantifies tool invocation overhead and synchronization cost for multi-agent video understanding systems. (2) The design principle that dependency graphs built from semantic scene boundaries enable overhead reduction via greedy clustering, which is validated through schedule simulation. (3) A cost model and benchmark task for evaluating the trade-off between accuracy and efficiency in long-video multi-agent systems.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | ActivityNet-1.3 | Diverse temporal segments requiring coordination |
| Primary metric | Total system latency (s) | Directly measures coordination overhead |
| Baseline | Round-Robin (naive) | Tests against naive assignment |
| Baseline | Static Partitioning | Tests against fixed segment allocation |
| Baseline | Full Sequential | Tests against single-agent baseline |
| Ablation | EACS w/o clustering | Isolates effect of dependency grouping |
| Sensitivity | Dependency threshold [0.1, 0.3, 0.5, 0.7] | Tests robustness of threshold choice |
| Validation | Calibration set: 200 videos from ActivityNet-1.3 | Threshold selected to maximize F1 for human-annotated cross-segment dependencies |
| Additivity test | Pearson correlation between predicted and actual latency | Validates additive cost assumption |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that EACS reduces total cost by optimizing coordination. ActivityNet-1.3 provides videos with diverse temporal structures and cross-segment dependencies, which are necessary to trigger the predicted failure modes of naive baselines. The primary metric, total system latency, directly captures the three components (tool invocation, synchronization, context reset) that EACS aims to minimize. Comparing against Round-Robin tests the effect of dependency clustering, Static Partitioning tests cost-aware assignment, Full Sequential tests parallelism, and the ablation isolates the clustering contribution. The sensitivity analysis on the dependency threshold checks whether a fixed threshold (0.3) is sufficient; if performance varies drastically, the calibration step is critical. The additivity test (Pearson correlation) examines whether the cost model's linear decomposition is accurate; a low correlation would indicate non-linear interactions and caveat the predicted_overhead measure. If EACS reduces latency only when dependencies are high, the causal chain is supported; uniform reduction across all subsets would indicate a different mechanism.

### Expected outcome and causal chain

**vs. Round-Robin (naive)** — On a case where two semantically linked video segments (e.g., a question asked at 10s that depends on an object shown at 5s) are far apart, Round-Robin assigns them to different agents, incurring high synchronization overhead when the second agent must wait for context from the first. Our method EACS clusters these linked segments together into the same agent, avoiding cross-agent communication. Consequently, we expect a noticeable latency gap on videos with many cross-time dependencies, but parity on videos with independent segments.

**vs. Static Partitioning** — On a video where one segment is much more complex (e.g., dense action sequence) than others, Static Partitioning assigns each agent a fixed set of segments regardless of workload, overloading one agent while others idle. EACS uses cost-aware assignment, balancing the predicted overhead across agents based on tool latencies and segment difficulty. Thus, we expect a significant latency improvement on videos with high imbalance, but similar performance on well-balanced videos.

**vs. Full Sequential** — On a long video (e.g., 10 minutes), Full Sequential processing by a single agent multiplies latency proportionally. EACS parallelizes clusters across agents, drastically reducing wall-clock time. However, for very short videos (e.g., <30 seconds), the overhead of EACS clustering may exceed benefits. We expect a large latency reduction on long videos and a slight overhead on very short videos.

**vs. EACS w/o clustering (ablation)** — The ablation removes dependency grouping, using uniform random cluster assignment instead. We expect the full EACS to outperform on videos with high semantic overlap (where clustering reduces sync) but behave similarly on independent segments. If the ablation performs equally well, the clustering step is unnecessary.

### What would falsify this idea
If EACS shows latency similar to naive baselines on videos with high semantic overlap, or if the ablation (no clustering) performs equally well, then the dependency clustering does not reduce synchronization as claimed, falsifying the central mechanism. Additionally, if the Pearson correlation between predicted and actual latency is below 0.7, the additive cost assumption is unsupported, undermining the claim that predicted_overhead accurately measures efficiency.

## References

1. Video-Oasis: Rethinking Evaluation of Video Understanding
