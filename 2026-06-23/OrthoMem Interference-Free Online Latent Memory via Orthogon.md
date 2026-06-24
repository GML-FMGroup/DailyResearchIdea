# OrthoMem: Interference-Free Online Latent Memory via Orthogonal Gradient Projection

## Motivation

EvoEmbedding's online memory update, which processes segments sequentially without offline consolidation, leads to memory saturation and topic interference because unconstrained gradient updates overwrite or blur distinct topical representations. LightMem's offline sleep-time consolidation temporarily avoids this but is incompatible with fully online agents requiring immediate memory availability. The root cause is that standard gradient descent on the latent memory does not respect the subspace of previously consolidated information.

## Key Insight

By projecting each memory update onto the orthogonal complement of the subspace spanned by all prior updates, new information is stored without altering the inner product structure of existing memory directions, guaranteeing that previously encoded topics remain unchanged.

## Method

### (A) What it is
OrthoMem is an online latent memory update mechanism that constrains each gradient update to be orthogonal to the subspace of previously consolidated memory directions. It takes as input a sequence of segments, maintains a latent memory vector m and an orthonormal basis B of previous update directions, and outputs an updated memory m' that stores new information without interfering with past content.

### (B) How it works
```python
# Hyperparameters: learning rate η, update threshold τ, max basis size K
B = []  # list of orthonormal basis vectors (initially empty)
m = initialize_memory()  # e.g., zero vector

for segment in segments:
    # compute gradient of memory w.r.t. retrieval loss on segment
    g = compute_gradient(m, segment)
    
    # project g onto orthogonal complement of span(B)
    if len(B) > 0:
        proj = sum([(g @ v) * v for v in B])
        g_orth = g - proj
    else:
        g_orth = g
    
    # update memory
    m = m + η * g_orth
    
    # update basis if g_orth is significant
    if np.linalg.norm(g_orth) > τ and len(B) < K:
        v_new = g_orth / np.linalg.norm(g_orth)
        B.append(v_new)
```

### (C) Why this design
We chose orthogonal gradient projection over naive gradient descent (which causes interference) because it guarantees that each new update does not affect the components of memory aligned with previous updates, accepting the cost that memory capacity is bounded by the basis size K (a user-specified trade-off between memory capacity and interference). We maintain an explicit orthonormal basis B rather than recomputing a subspace on each update (e.g., via SVD) because incremental Gram-Schmidt is O(K^2) per step versus O(d^3) for full SVD, accepting the approximation that early bases remain representative. We use a threshold τ for adding new basis vectors rather than adding every update because this prevents noise from saturating the basis, trading off perfect preservation of very small gradients against computational efficiency. We limit K to avoid unbounded memory growth; when K is reached, the oldest basis vector is replaced, which trades off long-term separation of topics for finite resource use.

### (D) Why it measures what we claim
The computed quantity `g_orth` measures *non-interference* of the new update on existing memory directions because we assume that the span of B captures all previously consolidated topic subspaces; the projection removes any component that would alter those directions. This assumption fails when two distinct topics share a common underlying direction (e.g., both involve the entity "bank"), in which case `g_orth` may be zero and the new information is discarded, leading to missing rather than interfering memory. The basis update threshold τ measures *significance* of a new direction: we assume that gradient norms above τ correspond to genuinely new topic structure; this fails when noisy but large gradients are misidentified as new topics, adding spurious basis vectors that waste capacity. The memory vector m stores the cumulative sum of all orthogonal updates, which we claim measures *complete episodic information* under the assumption that the orthogonal decomposition is sufficient; this fails when interactions between topics (e.g., a relation that involves both) require non-orthogonal components, in which case m omits those interactions. Overall, the mechanism operationalizes interference prevention via geometric constraints, with explicit failure modes that bound its scope.

## Contribution

['(1) OrthoMem, a novel online memory update algorithm that uses orthogonal gradient projection to prevent topic interference in latent memories without requiring offline consolidation or explicit topic labels.', '(2) The design principle that constraining updates to the nullspace of past updates suffices to preserve distinct topic representations in online agentic memory, demonstrating that interference can be eliminated geometrically rather than via consolidation schedules.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LongMemEval | Tests long-context multi-topic retrieval |
| Primary metric | Recall@1 | Measures correct memory retrieval |
| Baseline 1 | EvoEmbedding | Strong specialized embedding model |
| Baseline 2 | LightMem | Existing memory-augmented system |
| Ablation-of-ours | OrthoMem w/o orthogonal proj | Isolates orthogonal update effect |

### Why this setup validates the claim
The central claim is that orthogonal gradient updates prevent interference in latent memory. LongMemEval contains sequences with multiple topics and contradictions, ideal for testing interference. EvoEmbedding serves as a non-memory baseline that processes all context at once; if orthogonal updates reduce interference, OrthoMem should outperform it on sequences where topic shifts cause confusion. LightMem represents a prior memory system with standard gradient updates; comparison isolates the benefit of the orthogonal constraint. The ablation removes the projection, directly testing whether orthogonal updates are responsible for any improvement. Recall@1 is appropriate because it reflects exact retrieval of the correct memory, which interference would degrade.

### Expected outcome and causal chain

**vs. EvoEmbedding** — On a case where the same entity (e.g., "bank") appears first as a financial institution and later as a river bank, EvoEmbedding’s monolithic representation may retrieve the wrong sense because it averages conflicting information. Our method instead updates memory orthogonally, preserving both senses in separate subspaces, so we expect a noticeable gap on such ambiguous sequences, with OrthoMem achieving higher recall, while parity on unambiguous sequences.

**vs. LightMem** — On a long sequence where many updates occur, LightMem’s standard gradient descent can cause catastrophic forgetting; for instance, after memorizing customer A’s preferences, later updates about customer B may overwrite A’s memory. OrthoMem’s orthogonal projection ensures that each update only affects a new orthogonal direction, so A’s memory remains intact. We expect OrthoMem to retain stable recall for early segments even after many updates, whereas LightMem’s recall on those segments declines.

### What would falsify this idea
If OrthoMem’s gain over the ablation is uniform across all sequence lengths or topic shifts, rather than concentrated on sequences with high potential for interference, then the orthogonal update claim is not supported. Alternatively, if OrthoMem underperforms the ablation on interference-heavy subsets, the mechanism fails.

## References

1. EvoEmbedding: Evolvable Representations for Long-Context Retrieval and Agentic Memory
2. LightMem: Lightweight and Efficient Memory-Augmented Generation
3. Memory in the Age of AI Agents
4. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
5. From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs
6. Towards LifeSpan Cognitive Systems
7. Titans: Learning to Memorize at Test Time
8. Needle in the Haystack for Memory Based Large Language Models
