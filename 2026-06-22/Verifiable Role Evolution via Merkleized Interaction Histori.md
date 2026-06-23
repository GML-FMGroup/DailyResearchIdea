# Verifiable Role Evolution via Merkleized Interaction Histories in Multi-Agent Shared Memory

## Motivation

Existing benchmarks like GateMem assume static, pre-assigned roles for agents, failing to capture realistic multi-turn interactions where permissions should adapt based on history. This structural limitation prevents evaluation of dynamic access control. We propose that roles should be derived from a Merkle tree of memory access events, making transitions causally verifiable and eliminating reliance on external role assignments.

## Key Insight

Because each role transition is linked to a cryptographic hash of the preceding interaction sequence, the system can provably enforce that permission updates are consistent with the entire shared history, eliminating the need for external authority.

## Method

(A) **What it is**: MerkleRole is a memory governance system for multi-agent LLM agents where each agent's role and access permissions are derived from a Merkleized interaction history. Inputs: a sequence of memory access events (read/write) with agent IDs. Outputs: for each agent, a role label and associated permission set at each timestep.

(B) **How it works** (pseudocode):
```pseudocode
# Initialize
role_tree = empty MerkleTree()
for each agent A in {1..n}:
    set_role(A, "viewer", permissions=["read"])

# Main loop
for each timestep t:
    for each agent A in agents:
        action = propose_action(A, current_role)
        if permission_check(action, current_role_permissions):
            # Execute and record event
            event = (t, A, action_type, key, value_if_write, result)
            role_tree = merkle_append(role_tree, event)
            root = merkle_root(role_tree)
            # Compute new role (hyperparameters: thresholds)
            summary = summarize(events_by_agent[A], max_recency=10)
            new_role = rule_based_role(current_role, root, summary,
                                        write_threshold=5, read_threshold=10,
                                        violation_threshold=2, demotion_decay=0.9)
            set_role(A, new_role)
        else:
            log_denial(A, action)
```
Key components: Merkle tree of events, rule-based role derivation function, per-agent event summary. Hyperparameters: write_threshold, read_threshold, violation_threshold, demotion_decay.

(C) **Why this design**: We chose a Merkle tree over a simple hash chain because it allows efficient verification of any subset of events without reprocessing all history, accepting the cost of O(log n) update time. We chose a rule-based role derivation over a learned model because it ensures deterministic, verifiable transitions; the trade-off is less adaptability to complex or nuanced role changes. We chose to summarize per-agent events rather than full history to compute roles, because it remains compact while capturing individual contribution; this may miss cross-agent interaction patterns that could inform role evolution. The design prioritizes verifiability and determinism, which is critical for auditability in high-stakes multi-agent settings. We avoid a learned controller or router because the rule-based function directly operationalizes the motivation-level concept of causally verifiable role transitions; a learned alternative would obscure the causal link and undermine verifiability.

(D) **Why it measures what we claim**: The Merkle root measures global consistency of the interaction history because any tampering changes the root and agents can verify against their local events; this assumption fails when agents collude to manipulate the root off-chain, in which case the system reverts to trusting the root publisher. The rule-based role derivation measures role appropriateness as a deterministic function of the history; the assumption is that predefined thresholds capture the semantics of role evolution (e.g., frequent writers deserve higher privileges); this fails when nuanced or context-sensitive role changes are needed, leading to suboptimal permissions. The per-agent summary measures an agent's contribution level, assuming that frequency and recency of actions are sufficient indicators; this fails when an agent performs critical but rare actions, causing the summary to underestimate their contribution. Thus, each computational quantity operationalizes a motivation-level concept (consistency, role evolution, contribution) with explicit assumptions and failure modes.

## Contribution

(1) A novel framework, MerkleRole, that dynamically derives agent roles and permissions from a verifiable Merkle tree of memory access events, eliminating the static role assumption. (2) A design principle that role transitions should be causally verifiable via cryptographic hashes, enabling auditability in multi-agent memory systems. (3) A concrete rule-based role evolution function suitable for deployment in environments with clear interaction patterns.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GateMem | Multi-principal shared-memory with hidden checkpoints |
| Primary metric | Unauthorized access rate | Measures correctness of permission enforcement |
| Baseline 1 | Long-context GPT-4 | Standard baseline without memory governance |
| Baseline 2 | RAG (chunk size 512) | Retrieval-augmented memory baseline |
| Baseline 3 | MemGPT | External memory management baseline |
| Ablation | MerkleRole w/o Merkle tree | Tests importance of Merkleized history |

### Why this setup validates the claim

This setup tests the central claim that MerkleRole enforces causally verifiable role transitions in multi-agent systems. The GateMem dataset provides multi-principal episodes with hidden checkpoints, forcing agents to reason about evolving permissions. The unauthorized access rate directly measures whether roles are correctly assigned and enforced. Long-context GPT-4 tests the necessity of explicit governance, RAG tests whether retrieval alone suffices, and MemGPT tests a structured memory baseline. The ablation (replacing the Merkle tree with a linear hash chain) isolates the benefit of efficient subset verification. Together, these comparisons form a falsifiable test: if MerkleRole reduces unauthorized access more where baselines fail due to memory inconsistency or role confusion, the claim is supported.

### Expected outcome and causal chain

**vs. Long-context GPT-4** — On a case where an agent must gradually earn write permissions through repeated benign reads, GPT-4 lacks a mechanism to track past actions across long contexts and may either grant permissions prematurely or deny them incorrectly due to context truncation. Our method builds a Merkle tree of events and applies rule-based thresholds, so it correctly promotes agents after sufficient authenticated reads. We expect a large gap on episodes requiring incremental role changes, but parity on simple single-task episodes.

**vs. RAG (chunk size 512)** — On a case where multiple agents concurrently access overlapping memory, RAG retrieves chunks based on similarity but cannot verify the order or integrity of events, leading to stale or conflicting permissions. Our Merkle tree preserves a tamper-evident event sequence, so the role derivation always sees the correct history. We expect RAG to show higher unauthorized access rates on episodes with interleaved agent actions, while our method remains robust.

**vs. MemGPT** — On a case where an agent performs a critical but rare action (e.g., updating a safety parameter), MemGPT’s external memory may underweight that event if it falls outside the recent window, causing premature demotion. Our per-agent summary with recency weighting captures rare actions as long as they are within the max_recency window, and the rule-based function cannot arbitrarily ignore them. We expect a gap on episodes where rare but important actions define role boundaries; both methods may perform similarly on frequent-action episodes.

### What would falsify this idea

If MerkleRole does not show a disproportionate reduction in unauthorized access relative to baselines on episodes with high interaction complexity (e.g., many agents, intertwined reads/writes), but instead shows uniform improvement across all episode types, then the central claim that the Merkle tree’s verifiability drives the gain would be falsified.

## References

1. GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents
2. Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions
3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
4. HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
5. Needle in the Haystack for Memory Based Large Language Models
6. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
7. LONG²RAG: Evaluating Long-Context & Long-Form Retrieval-Augmented Generation with Key Point Recall
8. LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks
