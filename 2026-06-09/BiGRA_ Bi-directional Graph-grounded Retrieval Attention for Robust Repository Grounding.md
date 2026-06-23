# BiGRA: Bi-directional Graph-grounded Retrieval Attention for Robust Repository Grounding

## Motivation

LARGER's reliance on lexical anchors for graph exploration creates a structural bottleneck: when the initial lexical match is poor or ambiguous, expansion is misdirected, and the graph's structural context is only used after retrieval. This failure is rooted in the decoupled design where graph grounding is conditioned on lexical success, rather than integrated as a parallel signal. To maintain precision when lexical overlap is weak, the model needs a mechanism that uses graph structure as the primary grounding, with lexical matches as soft biases, enabling redundancy across modalities.

## Key Insight

Bi-directional attention between query tokens and graph nodes allows structural context to compensate for poor lexical alignment because graph edges encode invariant dependencies (e.g., call graphs, data flow) that are independent of surface term overlap, providing a complementary signal that lexical-only methods lack.

## Method

(A) **What it is**: BiGRA is a retrieval framework that takes a natural language query and a repository graph (nodes with precomputed embeddings, edges representing dependencies) and outputs a ranked list of relevant nodes. It uses bi-directional cross-attention where lexical matches from BM25 are injected as soft additive biases to attention scores, not hard anchors. (B) **How it works**:

```python
# Pseudocode for BiGRA retrieval
# Input: query string Q, repository graph G = (V, E) with node features X_v (from CodeBERT) and edge weights
# Hyperparameters: alpha (lexical bias strength, default=0.3), top_k (number of retrieved nodes, default=20)

# 1. Encode query
query_tokens = tokenize(Q)  # list of token ids
query_emb = LLM_embedding(query_tokens)  # [L, d] from frozen CodeBERT

# 2. Encode graph nodes via GNN (e.g., 2-layer GCN)
node_emb = GNN(X_v, E)  # [N, d]

# 3. Compute lexical match scores
lex_scores = BM25(Q, each node's text)  # [N], normalized to [0,1]

# 4. Bi-directional attention
#   a. Query-to-node attention: each query token attends to all nodes
Q2N_logits = query_emb @ node_emb.T / sqrt(d)  # [L, N]
Q2N_logits += alpha * lex_scores.unsqueeze(0)  # add lexical bias
Q2N_attn = softmax(Q2N_logits, dim=-1)  # [L, N]

#   b. Node-to-query attention: each node attends to all query tokens
N2Q_logits = node_emb @ query_emb.T / sqrt(d)  # [N, L]
N2Q_logits += alpha * lex_scores.unsqueeze(1)  # broadcast lexical bias (same bias per node)
N2Q_attn = softmax(N2Q_logits, dim=-1)  # [N, L]

# 5. Compute node relevance scores
node_relevance = sum(Q2N_attn, dim=0) * sum(N2Q_attn, dim=1)  # element-wise product of marginal attention masses
#   Alternative: node_relevance = diag(Q2N_attn.T @ N2Q_attn)  # attention cycle consistency

# 6. Retrieve top-k nodes
retrieved_nodes = argsort(node_relevance, descending=True)[:top_k]

# Optionally: expand graph using confidence-filtered BFS from retrieved nodes (as in LARGER)
```

(C) **Why this design**: We chose bi-directional attention (both Q→N and N→Q) over unidirectional Q→N only because N→Q allows graph nodes to reshape their representations based on query context, which is critical when a node's surface text is irrelevant but its structural role matches the query. We added lexical scores as soft bias (additive, not multiplicative) to preserve the model's ability to override poor lexical matches, accepting the cost that strong lexically matching but structurally irrelevant nodes may still get high attention. We used frozen CodeBERT embeddings for query and GNN for nodes rather than end-to-end training because the graph structure is static and large, making joint training computationally prohibitive; the trade-off is that the graph node embeddings may not be perfectly aligned with query semantics, but the bi-directional attention partially compensates. We chose the product of marginal attentions as the relevance score (over max or sum) because it rewards nodes that are both attended by many query tokens and attend to many query tokens simultaneously, capturing mutual alignment; the cost is that nodes with only one-sided attention (e.g., lexically matching but structurally isolated) get lower scores.

(D) **Why it measures what we claim**: The bi-directional attention weights (Q2N_attn and N2Q_attn) operationalize the key concept of `structural joint alignment`: a node that is structurally central to the query will have high attention both ways. Specifically, `sum(Q2N_attn, dim=0)` measures how much the query's tokens collectively attend to a node (alignment from query to graph), while `sum(N2Q_attn, dim=1)` measures how much the node's representation aligns with the query's tokens (alignment from graph to query). The product `node_relevance` measures the strength of bi-directional agreement; this is equivalent to the assumption that a node is relevant iff it is both a target of query attention and a source of query-attending attention. This assumption fails when the query is very short (e.g., one token) because the marginal attentions become degenerate (both sums reduce to the same vector), in which case the product merely squares the unidirectional attention and loses the benefit of bi-directionality. The additive lexical bias `alpha * lex_scores` ensures that even when structural alignment is low due to embedding mismatch, the lexical signal provides a soft anchor; this operationalizes `lexical soft bias` under the assumption that lexical overlap is a weak indicator of relevance. The assumption fails when the query uses synonyms or paraphrases that have zero lexical overlap with relevant nodes, in which case the bias may incorrectly downrank relevant nodes; the bi-directional attention must overcome this via structural alignment alone.

## Contribution

(1) A novel bi-directional attention mechanism between query and repository graph that treats lexical matches as soft biases rather than hard anchors, enabling robust grounding when lexical overlap is poor. (2) An empirical demonstration that integrating graph structure directly into the attention process improves retrieval precision on repository-level tasks with low lexical overlap, compared to prior lexical-anchor methods like LARGER. (3) A design principle that separates structural and lexical signals into parallel pathways, providing a general template for multi-modal grounding in code retrieval.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | SWE-bench-live | Realistic repo-level bug fixing tasks |
| Primary metric | Top-1 accuracy | Measures exact file retrieval success |
| Baseline | BM25 | Lexical baseline; tests need for semantics+structure |
| Baseline | LARGER | Graph baseline with lexical anchors; tests bi-directional attention |
| Baseline | LocAgent+Memory | SOTA agent baseline; tests retrieval-only vs agent pipeline |
| Ablation | BiGRA w/o lexical bias | Isolates contribution of lexical soft bias |

### Why this setup validates the claim
SWE-bench-live tasks require pinpointing the exact file to modify, often with indirect textual clues but strong structural dependencies (e.g., a bug in a utility function called from many places). Top-1 accuracy directly measures whether the correct node is retrieved first, which is the core claim of structural joint alignment. BM25 isolates the value of lexical matching alone; LARGER tests whether lexical anchors in a graph suffice; LocAgent+Memory tests whether a full agent with memory can beat a pure retrieval method. The ablation (no lexical bias) quantifies the contribution of soft lexical injection. Any failure of BiGRA to outperform LARGER on structurally complex queries would falsify the central claim that bi-directional attention adds value beyond one-sided lexical anchoring.

### Expected outcome and causal chain

**vs. BM25** — On a case where the bug report mentions a function name that does not appear in the target file (e.g., a configuration error propagated via call chain), BM25 ranks the function's calling site high but misses the target file because lexical overlap is zero. Our method instead attends to the calling site via lexical bias but then allows the target file to gain attention through graph edges (structural alignment), so we expect a large gap (e.g., 30+% absolute improvement on queries with <50% lexical overlap).

**vs. LARGER** — On a case where a node's surface text is irrelevant but its structural role is central (e.g., a constant definition used by many buggy functions), LARGER's hard lexical anchors dominate and downrank that node because BM25 score is low. Our method's soft bias allows the structural alignment from node-to-query attention to override the weak lexical signal, so we expect BiGRA to outperform LARGER on files with low lexical overlap but high structural importance (e.g., 15-20% gain on such subset).

**vs. LocAgent+Memory** — On a case where the bug report is short and ambiguous (e.g., "fix null pointer in logging"), LocAgent+Memory relies on iterative reasoning that may get distracted by irrelevant files, and its memory may not capture the full graph structure. Our method directly computes bi-directional attention across the whole graph, giving a holistic relevance score without iterative exploration, so we expect BiGRA to match or exceed LocAgent+Memory on queries with clear structural signals but possibly lag on queries that require multi-step reasoning (e.g., 5% lower overall but faster).

### What would falsify this idea
If BiGRA's advantage over LARGER is uniform across all query types (high and low lexical overlap) rather than concentrated on low-overlap but structurally important cases, then the bi-directional attention is not providing the hypothesized structural benefit—only the lexical bias is responsible.

## References

1. LARGER: Lexically Anchored Repository Graph Exploration and Retrieval
2. Improving Code Localization with Repository Memory
3. RANGER - Repository-Level Agent for Graph-Enhanced Retrieval
4. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
5. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
6. Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion
7. GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model
8. CATCODER: Repository-Level Code Generation with Relevant Code and Type Context
