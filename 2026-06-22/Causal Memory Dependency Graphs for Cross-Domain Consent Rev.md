# Causal Memory Dependency Graphs for Cross-Domain Consent Revocation in Multi-Agent Systems

## Motivation

Existing multi-agent memory architectures, such as those evaluated in the GateMem benchmark, assume static roles and fixed domains, making consent revocation impossible once data is shared across boundaries. The structural root cause is that no mechanism tracks causal derivation chains across memory entries, so revoking a source leaves downstream memories undetected and undeletable, violating privacy regulations like GDPR.

## Key Insight

By maintaining a directed acyclic graph where each edge records the causal operation that produced a memory entry, consent revocation can be propagated via forward reachability, automatically affecting all derivations without retraining or re-embedding.

## Method

### (A) What it is:
CausalDAG-Mem is a graph-based memory architecture that tracks causal dependencies between memory entries across agents and domains, enabling consent revocation to cascade based on domain-specific policies. Input: multi-agent interactions with explicit source references and operation type declarations per write; output: a dynamic graph that supports revocation queries.

### (B) How it works:
```python
class CausalDAGMem:
    def __init__(self, policy_engine):
        self.nodes = {}  # node_id -> MemoryEntry
        self.edges = []  # (src_id, tgt_id, op_type, src_agent, src_domain)
        self.policy_engine = policy_engine
    
    def write_entry(self, source_ids, content, agent, domain, op_type):
        new_id = generate_id()
        self.nodes[new_id] = MemoryEntry(id=new_id, content=content, domain=domain, agent=agent, accessible=True)
        for src in source_ids:
            # operation type is declared by agent: 'copied', 'transformed', 'inferred'
            self.edges.append((src, new_id, op_type, agent, domain))
        return new_id
    
    def revoke_consent(self, node_id):
        # BFS from node_id; check each edge against policy
        visited = set()
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for src, tgt, op_type, agent, domain in self.edges:
                if src == current:
                    # policy: (src_domain, domain, op_type) -> propagate?
                    if self.policy_engine.propagate(self.nodes[src].domain, domain, op_type):
                        queue.append(tgt)
        # mark all visited as inaccessible
        for v in visited:
            self.nodes[v].accessible = False
```

### (C) Why this design:
We chose a DAG over a simple key-value store because a DAG explicitly encodes the derivation chain needed for cascade revocation, accepting the overhead of O(E) storage and O(V+E) traversal per revocation. We chose hard flagging (setting `accessible=False`) instead of physical deletion to preserve referential integrity for other agents that may depend indirectly on *other* paths, accepting that stale placeholder entries may persist. We placed the policy engine on edges (checking (src_domain, tgt_domain, op_type)) rather than on nodes because domain boundaries are crossed at the moment of derivation; this finer granularity allows domain-specific rules (e.g., no propagation from healthcare to marketing) but increases rule complexity. The fixed hyperparameter ε for `infer_operation()` (e.g., Jaccard similarity >0.9 => 'copied', else 'transformed') is removed; instead, agents explicitly declare the operation type. This avoids ambiguous classification but relies on agents to truthfully declare dependencies. We assume that the system enforces truthful declarations via auditing mechanisms (e.g., cross-checking with content similarity, but not implemented here). If agents misdeclare, revocation may under- or over-propagate.

### (D) Why it measures what we claim:
The forward reachability algorithm operationalizes 'cascade effect' because each edge labeled with operation type and source domain captures the causal dependency that a downstream memory exists only due to its source; this assumes that the graph faithfully records all derivations, which fails when agents perform external memory operations outside the API (e.g., copying content manually), in which case the revocation may miss untracked copies. The policy check on edges operationalizes 'domain-specific consent' because it maps (src_domain, tgt_domain, op_type) to a propagation decision derived from privacy rules; this assumes that policies are static and known at design time, failing when policies change dynamically or are unknown for unforeseen domain pairs, in which case the method defaults to non-propagation, potentially under-revoking. The explicit declaration of operation type (instead of heuristic inference) operationalizes 'derivation type' because it relies on agent honesty; this assumes agents accurately declare dependencies, failing when agents misremember or intentionally mislabel, leading to over- or under-propagation.

## Contribution

(1) Introduction of CausalDAG-Mem, a graph-based multi-agent memory architecture that tracks explicit causal derivation chains across domains. (2) A cascade revocation algorithm that uses forward reachability on the graph to propagate consent revocation according to domain-specific policies. (3) Design of an edge-level policy engine that integrates with the graph to enforce cross-domain privacy rules.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | GateMem benchmark | Multi-domain multi-agent memory interactions |
| Primary metric | Revocation Recall | Measures cascade revocation correctness |
| Baseline | Long-context prompting (GPT-4) | Lacks structured memory for cascade tracking |
| Baseline | RAG with chunk retrieval | Retrieves but cannot trace derivations |
| Baseline | Mem0 (graph memory) | Prior graph memory without policy enforcement |
| Ablation | CausalDAG-Mem w/o policy engine | Tests necessity of domain-specific policies |

### Why this setup validates the claim

GateMem provides multi-domain episodes with hidden dependencies and consent revocations, making it ideal to test cascade revocation. Revocation Recall directly measures the proportion of downstream memories correctly made inaccessible after a source revocation, which is the core claim. Long-context prompting tests if implicit memory can handle revocation without explicit structure; RAG tests if retrieval-based access suffices; Mem0 tests if graph structure alone (without domain policy) is enough. The ablation isolates the contribution of the policy engine. This combination forms a falsifiable test: if our method outperforms on cross-domain revocations where propagation should be blocked, it confirms that both causal DAG and policy engine are necessary. Crucially, our method assumes that operation types are explicitly declared by agents; in experiments, we use ground-truth operation labels from the GateMem benchmark (which include metadata on derivation type). This ensures that our method is evaluated under ideal conditions for dependency tracking, while baselines must infer dependencies from content or ignore them.

### Expected outcome and causal chain

**vs. Long-context prompting (GPT-4)** — On a case where a medical memory is copied to marketing domain and later revoked, GPT-4 lacks explicit dependency tracking so it cannot recall the downstream copy; it may hallucinate or leave it accessible. Our method captures the edge and policy (no propagation from healthcare to marketing) so it revokes only within domain. We expect a large gap on cross-domain revocation subsets, but parity on same-domain.

**vs. RAG with chunk retrieval** — On a transformed entry (e.g., paraphrased medical advice stored in education domain), RAG retrieves based on semantic similarity but cannot know the origin; revocation of source does not affect retrieved copy because no causal link exists. Our method's 'transformed' edge propagates if policy allows, revoking it. We expect a gap on transformed derivations where similarity is high but derivation unknown to RAG.

**vs. Mem0 (graph memory)** — On a cross-domain derivation (marketing entry derived from healthcare source), Mem0 stores the graph but lacks edge-level policy, so it would propagate revocation unconditionally, over-revoking. Our method checks policy and stops propagation. We expect our method to have higher Revocation Recall on cross-domain subsets (avoiding false revocations) while recall on same-domain subsets is comparable.

### What would falsify this idea
If our method's Revocation Recall is not significantly higher than Mem0 on cross-domain derivations, or if the ablation without policy matches our full method, then the policy engine is unnecessary and the causal DAG alone is insufficient. Also if our method underperforms on same-domain revocations due to overhead, then the DAG complexity is not justified.

## References

1. GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents
2. Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions
3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
4. HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
5. Needle in the Haystack for Memory Based Large Language Models
6. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
7. LONG²RAG: Evaluating Long-Context & Long-Form Retrieval-Augmented Generation with Key Point Recall
8. LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks
