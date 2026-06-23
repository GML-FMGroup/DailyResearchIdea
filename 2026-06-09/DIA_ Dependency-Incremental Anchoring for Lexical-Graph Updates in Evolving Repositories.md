# DIA: Dependency-Incremental Anchoring for Lexical-Graph Updates in Evolving Repositories

## Motivation

LARGER's lexical-anchor graph is static after construction, so any code change (e.g., commit, refactoring) forces a full recomputation of confidence-filtered expansion neighborhoods to avoid stale anchors. Previous work on incremental index maintenance (AOCI) preserved symbolic-semantic mapping under evolution without full rebuild, but LARGER's anchor graph lacks the structural machinery—specifically, per-anchor edge relevance tracking—to apply diff-based updates. Without incremental updates, LARGER incurs prohibitive cost in repositories with frequent changes, limiting its practical deployment.

## Key Insight

Dependency witness sets precompute which graph edges are causally relevant to each anchor, so that a code diff only needs to recompute witness sets for anchors whose code region intersects the changed files, preserving anchor-context consistency without full graph rebuild.

## Method

### (A) What it is
DIA (Dependency-Incremental Anchoring) is an incremental update framework for LARGER's lexical-anchor graph. Its input is a repository diff (file-level changes) and the current anchor graph; its output is an updated anchor graph with recomputed confidence scores and expansion neighborhoods only for affected anchors.

### (B) How it works
```pseudocode
Input: Current graph G = (Anchors A, Edges E, Confidence C per anchor)
       Diff D = { file, added_lines, removed_lines }
       Dependency witness sets W[a] for each anchor a (precomputed, list of edge IDs)
Output: Updated graph G'

Procedure UPDATE_GRAPH(G, D, W)
1. Identify changed files F = { f | f in D }
2. For each anchor a in A:
   a. Compute anchor relevance R(a) = 1 if a's lexical match region intersects any file in F, else 0
      // Lexical match region is the set of file:line where anchor's keyword appears
      // Note: This assumes only anchors whose keyword appears in changed files can be affected;
      //       indirect dependencies (e.g., via import/call) may break this assumption.
      //       We quantify its reliability in a pre-study (see Experiment).
3. For each anchor a where R(a)=1:
   a. Retrieve witness set W[a] (edge IDs that were used to compute a's expansion in G)
   b. For each edge e in W[a]:
       If e is affected by D (i.e., e involves a file in F or its confidence depends on lines in F):
           Recompute edge weight w_e using only D's context (no full graph scan)
           // w_e = co-occurrence frequency in changed region vs. global (if static) or semantic similarity from embedding diff
   c. Recompute confidence score c'_a = aggregate of new w_e over W[a] (e.g., mean or sum)
   d. Expand neighborhood N'_a by BFS from a using edges with c'_a below threshold τ (default 0.5)
       // Stop expansion when confidence drops below τ or depth exceeds 3
4. For anchors a where R(a)=0, keep original G values: c'_a = c_a, N'_a = N_a
5. Return G' with updated anchors (a, c'_a, N'_a)
```

A minimal working example (single anchor, single diff) is provided in the supplementary material.

### (C) Why this design
We chose to compute anchor relevance via lexical match region intersection (step 2) over a full-graph diff analysis because it is lightweight and aligns with LARGER's lexical anchoring: if an anchor's keyword does not appear in any changed file, we assume its expansion cannot be affected. This design relies on the assumption that an anchor's expansion is unaffected by a code change unless the anchor's keyword appears in a changed file. We verify this assumption empirically in a pre-study (see Experiment). This design accepts the cost that a global renaming of a symbol (e.g., a variable name change across many files) would affect many anchors, but such refactorings are rare and the per-anchor recomputation still avoids scanning the entire graph. We use precomputed dependency witness sets (W[a]) and recompute only edges in the witness set rather than all edges from the anchor, preserving the invariant that only edges that contributed to the original expansion are updated. The trade-off is that if a diff introduces a new, high-confidence edge that wasn't in the witness set, it will be missed until a full rebuild; we accept this staleness for the benefit of cheap incremental updates. Finally, we keep the confidence threshold τ and BFS depth from LARGER unchanged to maintain backward compatibility, even though a dynamic threshold based on change magnitude could be more adaptive; we prioritize simplicity and reproducibility.

### (D) Why it measures what we claim
The computational quantity `anchor relevance R(a)` measures the `intersection of anchor's lexical scope with changed files` as a proxy for `causal necessity of update` because we assume that anchors whose keywords appear only in unchanged files cannot have altered context; this assumption fails when a change in an unrelated file indirectly affects the anchor's expansion edges (e.g., a utility function signature change that is imported elsewhere), in which case R(a) underestimates the need for update, leading to stale coherence. The recomputation of edge weight `w_e` for affected witnesses measures `edge confidence degradation due to diff` because we assume that the edge's weight depends on the local code region; this assumption fails when the edge weight is determined by global statistics (e.g., overall call frequency) that the diff does not change, in which case recomputing from the diff introduces noise. The final confidence score `c'_a` and neighborhood `N'_a` together operationalize `anchor-context consistency` because we define consistency as the property that the expansion subgraph reflects the current code state; the update ensures that any anchor whose lexical region or witness edges changed is recomputed, with the failure mode that unchanged anchors may contain stale edges that were not in their witness set but became relevant through new dependencies.

## Contribution

(1) A novel incremental update algorithm for lexical-anchor graphs that uses dependency witness sets to localize recomputation to anchors affected by code diffs. (2) An empirical design principle that preserving anchor-context consistency under repository evolution does not require full graph rebuild if per-anchor edge relevance is precomputed. (3) A set of update rules for static and semantic edge weights under file-level diffs, enabling low-cost maintenance of LARGER-style retrieval graphs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Real-world repo diffs with ground truth full rebuild | Captures realistic update scenarios. |
| Primary metric | Anchor confidence error (MAE vs full rebuild) | Directly measures update correctness. |
| Baseline 1 | Full Rebuild | Gold standard for correct graph after diff. |
| Baseline 2 | No Update | Tests necessity of any update. |
| Baseline 3 | Naive Update (all edges for affected anchors) | Tests benefit of witness set pruning. |
| Ablation of ours | DIA without witness sets | Isolates impact of witness set mechanism. |

### Why this setup validates the claim
This experimental design directly tests the central claim that DIA produces correct incremental updates. The full rebuild baseline provides ground truth for correctness; any deviation in confidence or neighborhood signals an error. No Update baseline detects whether updates are necessary at all, and if ignored, causes degradation. Naive Update baseline (recomputing all edges for affected anchors without witness sets) isolates the benefit of precomputed dependency tracking. The primary metric, mean absolute error (MAE) of anchor confidence vs. full rebuild, quantifies update accuracy. By comparing DIA to these baselines across diverse diffs, we can falsify the claim if DIA fails to approximate full rebuild or if its speed advantage disappears. The ablation further identifies which component (witness sets) is essential.

### Expected outcome and causal chain

**vs. Full Rebuild** — On a diff that changes a rarely used function signature, Full Rebuild recalculates the entire graph, costing hours. Our method DIA only updates anchors intersecting changed files, using precomputed witness sets to recompute only affected edges. We expect DIA’s confidence error to be < 0.01 (on a scale of 0-1) while taking < 5% of the time of Full Rebuild, because the unchanged majority of anchors are untouched.

**vs. No Update** — On a diff that renames a frequently used API, No Update keeps old anchor confidences, causing large errors (MAE > 0.2) for affected anchors. Our method detects the lexical match region intersection and recomputes confidence based on the new context, yielding near-zero error. The observable signal: a clear gap between DIA (MAE < 0.01) and No Update (MAE > 0.2) on diffs touching popular anchors.

**vs. Naive Update** — On a diff that adds a new file with new usage of an existing anchor, Naive Update recomputes all edges from that anchor (expensive, 100x more edges than witness set). DIA updates only edges in the witness set, missing any new high-confidence edge. We expect DIA to be > 10x faster but with slightly higher error (< 0.05) on anchors gaining strong new connections. This trade-off is acceptable because such cases are rare and the error remains small.

### What would falsify this idea
If DIA’s confidence error on unaffected anchors exceeds 0.05 (indicating spurious updates) or if DIA’s runtime is >50% of Full Rebuild on typical diffs (indicating insufficient efficiency), then the central claim of cheap, correct incremental updates is invalid.

## References

1. LARGER: Lexically Anchored Repository Graph Exploration and Retrieval
2. Improving Code Localization with Repository Memory
3. RANGER - Repository-Level Agent for Graph-Enhanced Retrieval
4. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
5. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
6. Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion
7. GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model
8. CATCODER: Repository-Level Code Generation with Relevant Code and Type Context
