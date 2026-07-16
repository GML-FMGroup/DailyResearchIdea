# Coordination Nodes: Integrating Multi-Agent Interaction Transparency into Hierarchical Behavior Trees for Agent Harnesses

## Motivation

The Harness Handbook (Idea Card 1) provides a static hierarchical behavior tree for single-agent behaviors, but it lacks representation of inter-agent coordination. OpenHands (Idea Card 3) achieves transparency through explicit event-driven communication, but this is not integrated into a navigable tree structure. Consequently, developers cannot inspect or navigate multi-agent coordination within the handbook, making debugging multi-agent harnesses tedious and error-prone.

## Key Insight

By treating inter-agent message exchanges as first-class transitions in the behavior tree, we preserve the single-parent invariant and hierarchical navigability while exposing coordination patterns.

## Method

(A) **What it is**: Cross-Agent Coordination Node Extraction (CANCE) is a static analysis algorithm that extends the Harness Handbook's behavior tree with coordination nodes. Input: multi-agent harness codebase. Output: extended behavior tree where each coordination node encapsulates a cross-agent interaction as a single atomic transition.

(B) **How it works** (pseudocode):
```python
def CANCE(agent_harnesses):
    # Phase 1: Extract per-agent behavior trees via static analysis (as in Harness Handbook)
    trees = [static_analysis(harness) for harness in agent_harnesses]
    # Phase 2: Identify coordination candidates
    shared_vars = intersect([agent.state_accesses for agent in trees])
    comm_pairs = []  # list of (agentA, nodeA, agentB, nodeB) tuples
    for var in shared_vars:
        writers = [agent.nodes_writing(var) for agent in trees]
        readers = [agent.nodes_reading(var) for agent in trees]
        for w in writers:
            for r in readers:
                if w.agent != r.agent:
                    comm_pairs.append((w.agent, w.node, r.agent, r.node))
    # Also detect explicit message passing (e.g., send/receive calls)
    for agent in trees:
        for node in agent.nodes:
            if node.type == 'send':
                # match with corresponding receive in another agent
                match = find_receive(node, trees)
                if match:
                    comm_pairs.append((agent, node, match.agent, match.node))
    # Phase 3: Insert coordination nodes
    # Merge trees under a common root
    all_trees = trees
    for (agentA, nodeA, agentB, nodeB) in comm_pairs:
        lca = lowest_common_ancestor(nodeA, nodeB, all_trees)
        coord_node = CoordinationNode(label=f"{agentA}↔{agentB}: {nodeA.action}⇄{nodeB.action}")
        insert_child(lca, coord_node)
    return all_trees  # now with coordination nodes
```
Hyperparameters: `shared_vars` computed by scanning all variable accesses; communication pairs threshold at 1 match.

(C) **Why this design**: We chose static over dynamic analysis because it captures all possible coordination patterns without execution traces, accepting the cost of false positives due to unreachable code. We used lowest-common-ancestor (LCA) insertion to maintain hierarchy rather than flattening coordination into a separate layer, preserving navigability at the cost of potential deep nesting. We treat message exchanges as atomic transitions rather than expanding into sub-trees, keeping tree size manageable but losing internal message ordering. We opted for a deterministic rule-based matching instead of learned models to ensure reproducibility and avoid data dependence, at the expense of missing implicit coordination patterns (e.g., temporal dependencies without shared state). These trade-offs prioritize transparency and interpretability over completeness.

(D) **Why it measures what we claim**: The count of coordination nodes measures multi-agent coordination transparency because it operationalizes the mapping from inter-agent interactions to visible tree nodes; this assumes all relevant interactions are captured by static analysis of shared state and communication patterns. This assumption fails when coordination relies on implicit temporal ordering (e.g., timeouts) not captured by state accesses or explicit message passing, in which case the metric undercounts coordination complexity. The depth of coordination nodes in the tree measures hierarchical integration: shallower nodes indicate coordination at higher abstraction levels, assuming that LCA reflects the natural modularity of the system; this fails when agents are not modularly decomposed, making LCA an arbitrary common ancestor that conflates independent behaviors.

## Contribution

(1) A static analysis algorithm (CANCE) that extends hierarchical behavior trees with coordination nodes explicitly representing inter-agent message exchanges as first-class transitions. (2) The design principle that coordination transparency can be achieved within a single-parent tree by leveraging cross-agent static analysis of shared state and communication patterns, without breaking navigability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Multi-agent harness codebases (SWE-bench) | Provides varied coordination patterns from real agent systems |
| Primary metric | F1 score of detected coordination nodes | Directly measures detection accuracy against annotated ground truth |
| Baseline 1 | Random coordination node insertion | Tests if extraction is better than random guessing |
| Baseline 2 | Dynamic analysis coordination extraction | Compares static vs trace-based coordination discovery |
| Ablation of ours | CANCE with flat coordination insertion | Isolates effect of hierarchical preservation via LCA |

### Why this setup validates the claim
This evaluation provides a direct, falsifiable test of the central claim that CANCE extracts meaningful coordination nodes via static analysis. The dataset of real multi-agent harnesses ensures realistic coordination patterns. The primary metric F1 directly measures how well our method detects ground-truth coordination nodes. Baseline 1 (random insertion) tests whether extraction outperforms chance, isolating the value of analysis. Baseline 2 (dynamic analysis) tests whether static analysis is necessary—if dynamic achieves similar F1, the claim that static captures all possible coordination is weakened. The ablation (flat insertion) tests whether the LCA hierarchy specifically improves node placement, validating that design choice. Together, these comparisons isolate the causal mechanism: static analysis of shared state and messages leads to accurate detection, and hierarchical placement improves interpretability. If our method fails to outperform these baselines on relevant subsets, the claim is falsified.

### Expected outcome and causal chain

**vs. Random coordination node insertion** — On a harness where coordination relies on shared variables (e.g., multiple agents accessing the same state), random insertion is blind to variable dependencies, placing nodes arbitrarily; it both misses true coordination and creates false positives. Our CANCE method analyzes variable accesses to pinpoint actual coordination points. Consequently, we expect a large gap in F1: CANCE near 0.8 while random near 0.0 on such cases, because random cannot exploit the structural cues of shared state.

**vs. Dynamic analysis coordination extraction** — On a harness where coordination only occurs in rare execution paths (e.g., error recovery), dynamic analysis misses nodes not covered by traces. CANCE's static analysis examines all control flow, so it detects those nodes regardless. We expect CANCE's recall to be higher (0.9 vs 0.5) on these paths, with precision slightly lower due to unreachable code false positives, yielding overall higher F1.

**vs. CANCE with flat coordination insertion** — On a harness with hierarchically nested coordination (a high-level coordination containing sub-coordinations), flat insertion places all nodes as siblings under root, breaking hierarchy. CANCE with LCA places nodes at natural ancestors. While detection F1 may be similar, navigability metrics (e.g., depth consistency) will show CANCE superior, isolating the benefit of hierarchical preservation.

### What would falsify this idea
If our F1 gain over random is uniform across coordination types (e.g., no larger on shared-state than message-passing), then the claim that static analysis of shared state drives improvement is falsified.

## References

1. Harness Handbook: Making Evolving Agent Harnesses Readable,Navigable, and Editable
2. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
3. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
4. OpenAgents: An Open Platform for Language Agents in the Wild
