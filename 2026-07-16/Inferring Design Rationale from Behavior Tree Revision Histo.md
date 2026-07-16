# Inferring Design Rationale from Behavior Tree Revision Histories via Causal Discovery

## Motivation

Existing methods like the Harness Handbook rely on static analysis and LLM-assisted structuring to map behaviors to source locations, but they cannot capture the designer's intent behind structural modifications because they treat commits as opaque updates. This limitation stems from the lack of a causal model linking revision operations to the underlying rationale, leading to incomplete behavior understanding. We address this by exploiting the hierarchical partial order in behavior trees to discover causal dependencies among modifications.

## Key Insight

The hierarchical structure of behavior trees induces a partial order that mirrors causal dependencies among design decisions, enabling causal discovery algorithms to separate refinements from corrections without explicit annotations.

## Method

**Rationale Causal Miner (RCM)**

(A) **What it is** — RCM is a method that takes a chronological sequence of behavior tree commits (each as a tree diff) and outputs each modification annotated as *refinement* (extending intent) or *correction* (fixing prior error), by learning a causal graph over tree operations with a soft hierarchy prior.

(B) **How it works** — Pseudocode:
```python
# Input: commits = [BT_0, BT_1, ..., BT_n] (behavior trees as hierarchical action graphs)
# Output: annotated operations = [(op_i, type_i)] where type_i ∈ {refinement, correction}

1. # Phase 1: Extract operations and hierarchy
   ops = []
   for i from 1 to n:
       diff = tree_diff(BT_{i-1}, BT_i)
       for each op in diff:  # op = (type, node_path, new_subtree) where type ∈ {add, delete, modify}
           op.commit_index = i
           op.parent_path = node_path[:-1]
           ops.append(op)

2. # Phase 2: Construct temporal adjacency matrix (fully connected initial)
   # Let H be the union of all parent-child relations across all BT states.
   # Initialize with a complete undirected graph over ops.
   edges = []
   for i in range(len(ops)):
       for j in range(i+1, len(ops)):
           edges.append((i, j))  # start fully connected

3. # Phase 3: Learn causal graph via PC algorithm (with Fisher-Z test, alpha=0.05)
   # Start with complete graph on ops node set, then:
   # - Remove edges by conditional independence tests using commit index as time.
   # - Orient edges using collider rules, with a soft hierarchy prior:
   #   Penalize orientations that violate hierarchical influence (descendant->ancestor) by a factor of 0.5
   #   when applying Meek rules (i.e., only orient descendant->ancestor if strongly supported).
   causal_graph = pc_algorithm_hierarchy_soft(nodes=range(len(ops)), edges=edges, independence_test=fisher_z, alpha=0.05, hierarchy_prior='soft', hierarchy_penalty=0.5)

4. # Phase 4: Classify each operation
   for i, op in enumerate(ops):
       parents = causal_graph.parents(i)
       if not parents:
           op.type = 'rationale_introduction'  # new intent root
       else:
           # Check if majority of parent operations lie on the path from root to op.node_path
           ancestors_in_tree = {p for p in parents if is_ancestor(ops[p].node_path, op.node_path)}
           if len(ancestors_in_tree) >= len(parents) * 0.5:  # threshold
               op.type = 'refinement'
           else:
               op.type = 'correction'
   return ops
```

(C) **Why this design** — We chose the PC algorithm over GES because PC's conditional independence tests scale better with the number of operations (O(k^2) vs O(2^k) for GES) and we have a soft hierarchy prior that guides orientation. We start with a fully connected graph rather than a hierarchy-constrained one to allow discovery of bidirectional influences, addressing the known limitation that low-level changes can trigger high-level refactors. The soft hierarchy prior (penalty factor 0.5 on descendant-to-ancestor edges) reduces false orientations while remaining open to evidence of reverse causation. We set the classification threshold at 50% to be robust to spurious edges from the PC algorithm; a higher threshold would increase precision but leave more operations unclassified as neither refinement nor correction. We chose Fisher's Z test for independence because the commit index is a continuous time variable and we assume linear-Gaussian structural equations, though this assumption is violated when operations have non-linear effects; we accept the risk that some dependencies may be missed, but the soft hierarchy prior partially compensates by guiding orientations.

(D) **Why it measures what we claim** — The computational quantity *edge onset direction from ancestor to descendant operations* measures *refinement* because the assumption that design intent propagates from higher-level (ancestor) to lower-level (descendant) modifications in the tree hierarchy; this assumption fails when a modification at a deep node induces a high-level refactor (a correction scenario), in which case the edge direction will be reversed or missing and the operation will be classified as correction. The quantity *proportion of parents that are tree ancestors* measures *intent coherence* because the assumption that refinements are predominantly driven by higher-level design decisions; this assumption fails when multiple independent intents are introduced simultaneously at different levels, in which case the proportion may be low and the operation might be misclassified as correction. The *PC algorithm's conditional independence tests* measure *causal necessity* because the assumption that removing an edge from a to b implies that a's change is not necessary for b's occurrence conditioned on other operations; this assumption fails when the test lacks statistical power (few commits), in which case edges may remain spuriously, leading to overestimation of refinement.

## Contribution

(1) A novel method (Rationale Causal Miner) that infers implicit design rationale from behavior tree revision histories by applying causal discovery constrained by the tree hierarchy. (2) A concrete design principle: hierarchical partial order in behavior trees can separate intent-refining modifications from error-correcting ones without requiring explicit annotations. (3) A tool for augmenting behavior tree maintainers with rationale traces, improving readability and editability of evolving harnesses.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | 50 synthetic BT histories (10 commits each) + 50 real curated histories | Synthetic provides ground truth for validation; real for generalization |
| Primary metric | F1 for refinement class | Balances precision and recall |
| Secondary metric | Structural Hamming Distance (SHD) of learned causal graph vs. ground truth (synthetic only) | Measures causal graph recovery accuracy |
| Baseline-1 | Random classifier | Lower bound on performance |
| Baseline-2 | Hierarchy-depth heuristic (classify by depth: deeper = refinement) | Tests need for causal reasoning beyond depth |
| Baseline-3 | Temporal heuristic (later = correction) | Tests need for hierarchical info |
| Baseline-4 | Hierarchy-only classifier (no causal discovery: classify by ancestor proportion using true hierarchy) | Isolates benefit of causal discovery |
| Ablation | RCM without hierarchy prior (fully unconstrained causal graph) | Isolates hierarchy contribution |
| Synthetic data generation | Start with random BT (depth 3, 10 nodes); simulate 10 operations per history: 70% refinements (add child to random ancestor) and 30% corrections (modify leaf node). Record ground truth rationale type. | |

### Why this setup validates the claim
This setup tests whether RCM's soft hierarchy prior is necessary to distinguish refinements from corrections. The random baseline establishes chance performance, while depth and temporal heuristics represent simple alternative explanations. The hierarchy-only baseline uses only hierarchical relations without causal discovery to isolate the benefit of learning causal structure. The ablation removes the hierarchy prior entirely to measure its specific contribution. SHD on synthetic data directly evaluates whether the learned causal graph matches the true dependencies. F1 on refinement class measures the quality of the primary output. If RCM significantly outperforms all baselines and the ablation, the claim that hierarchical causal reasoning improves annotation is supported. Conversely, near-parity with heuristics or ablation would falsify the core assumption.

### Expected outcome and causal chain

**vs. Random classifier** — On a case where a modification is actually a refinement but appears similar to corrections, the random baseline will guess randomly, often misclassifying. Our method uses causal edges from ancestors to descendants, correctly attributing the modification as refinement. We expect our method's F1 to be >0.7 while random is ~0.5.

**vs. Hierarchy-depth heuristic** — On a case where a deep modification is actually a correction (e.g., fixing a bug in a leaf node), the depth heuristic would incorrectly label it as refinement (since deeper means refinement). Our method sees that its parent operations are not ancestors (or the causal graph reveals a reverse direction), so it classifies as correction. We expect our method to have higher recall on corrections, while depth heuristic has high precision but low recall, leading to lower F1.

**vs. Temporal heuristic** — On a case where an early modification is a correction (e.g., reverting a previous change), temporal heuristic (later = correction) would misclassify early corrections. Our method uses causal links to see that the correction is caused by earlier ancestors, thus correctly labels it. We expect our method to outperform especially on non-monotonic sequences, with F1 gap > 0.15.

**vs. Hierarchy-only classifier** — On a case where a modification is a refinement but its parent operations are not solely ancestors (e.g., multiple unrelated ancestors), the hierarchy-only classifier (which uses true ancestor proportion without causal graph) may misclassify due to lack of causal filtering. Our method learns which parent operations are truly causal, thus correctly classifies. We expect our method to have higher F1 by at least 0.1 on synthetic data where ground truth causal structure is known.

**vs. Ablation (no hierarchy prior)** — On a case where the causal graph has spurious edges due to lack of hierarchical constraints, the ablation may misclassify refinements as corrections because it fails to orient edges correctly. Our method uses soft hierarchy prior to guide orientation, reducing errors. We expect SHD of our method to be at least 5 lower than ablation on synthetic data.

### What would falsify this idea
If our method's F1 is within 0.05 of the best heuristic baseline, or if SHD on synthetic data is not significantly lower than the ablation, then the causal hierarchy assumption is not providing additional value, falsifying the claim that hierarchy-guided causal reasoning is necessary.

## References

1. Harness Handbook: Making Evolving Agent Harnesses Readable,Navigable, and Editable
2. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
3. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
4. OpenAgents: An Open Platform for Language Agents in the Wild
