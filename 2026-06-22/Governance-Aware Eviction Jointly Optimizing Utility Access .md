# Governance-Aware Eviction: Jointly Optimizing Utility, Access Control, and Forgetting in Multi-Agent Memory

## Motivation

Existing multi-agent memory systems like Mem0 optimize retrieval utility independently, treating access control and forgetting as afterthoughts. As shown in GateMem, no evaluated method simultaneously achieves strong utility, robust access control, and reliable forgetting. The root cause is separated optimization: utility drives memory retention without governance constraints, leading to information leakage and inability to enforce deletion.

## Key Insight

By training a learnable eviction policy under a multi-objective loss that couples utility, access control, and forgetting targets, the memory inherently balances competing demands because the same mechanism decides what to evict based on all three criteria.

## Method

(A) **What it is** — GAE is a differentiable key-value memory with a learnable eviction policy that outputs eviction probabilities. Input: sequence of write operations (key, value, principal_id, forget_flag), query operations. Output: retrieved values, updated memory state.

(B) **How it works**
```python
# GAE: Governance-Aware Eviction
# Memory: matrix M of size N x (d_key + d_value + d_gov)
# where gov vector = (utility_gain, access_mask, forget_gate)
# d_gov = 3 (fixed)

def write(key, value, principal_id, forget_flag):
    key_emb = embed(key)  # embedding function
    # initial governance vector
    utility_gain = 1.0
    access_mask = encode_principal(principal_id)  # one-hot or multi-hot
    forget_gate = float(forget_flag)  # 0 or 1
    gov = [utility_gain, access_mask, forget_gate]
    slot = find_slot(key_emb)  # returns index if key exists, else None
    if slot is None:
        # evict via eviction policy (MLP with Gumbel-Softmax, tau=1.0)
        evict_logits = eviction_policy(M)  # shape (N, 1)
        evict_probs = softmax(evict_logits)  # Gumbel-Softmax over positions
        slot = one_hot_sample(evict_probs, hard=True)  # straight-through
        M[slot] = 0  # erase
    M[slot] = concatenate(key_emb, value, gov)  # write
    return slot

def query(key, principal_id):
    key_emb = embed(key)
    # attention weights
    attn = softmax(similarity(key_emb, M[:, :d_key]))  # cosine similarity
    # access control mask
    allowed = (M[:, d_key+d_value:d_key+d_value+1] & encode_principal(principal_id)) > 0
    attn_masked = attn * allowed.float()
    if attn_masked.sum() == 0:
        return None
    attn_masked = attn_masked / attn_masked.sum()
    value = (attn_masked.unsqueeze(-1) * M[:, d_key:d_key+d_value]).sum(dim=0)
    return value

# Training loss (hyperparameters: α=1.0, β=0.5, γ=0.5)
def compute_loss(batch):
    L_util = MSE(predicted_value, true_value)
    L_access = mean( (attn > 0) * (1 - allowed.float()) )  # penalize unauthorized attention
    L_forget = mean( (query_after_forget_attn > 0) )  # penalize retrieval after forget
    return α * L_util + β * L_access + γ * L_forget
```

(C) **Why this design** — We chose a learned eviction policy (small MLP with Gumbel-Softmax) over fixed heuristics (e.g., LRU, LFU) because governance constraints are context-dependent and cannot be captured by simple statistics; the cost is higher training complexity and potential instability from discrete sampling. We encode governance as a separate vector (utility_gain, access_mask, forget_gate) rather than a single scalar because it allows fine-grained control: access_mask supports multiple principals, and forget_gate enables targeted forgetting. We train with a weighted multi-objective loss instead of a single reward because governance and utility are incommensurate; this introduces a tuning burden but enables explicit trade-off control. We use differentiable attention for querying (allowing gradient flow) but hard eviction via straight-through estimator; the trade-off is that the eviction gradient may be biased, but it allows end-to-end training without relaxing the constraint that eviction is binary.

(D) **Why it measures what we claim** — The utility loss (MSE on query outputs) measures retrieval utility because it directly penalizes deviation from correct values; this equivalence assumes the correct value is known, which holds in supervised training but fails in open-ended scenarios where there is no ground truth—then utility reflects proxy consistency. The access control loss (mean unauthorized attention) measures information leakage prevention because it quantifies the amount of attention on slots whose access_mask does not include the querying principal; this assumption fails when the memory is accessed via indirect keys or cross-principal inference, in which case leakage may occur without observable attention. The forgetting loss (mean retrieval after forget) measures reliable forgetting because it enforces that after a forget write (forget_flag=1), subsequent queries return zero attention; this assumes the forget write operation effectively overwrites governance—if the memory retains residual information through the value field, forgetting may be incomplete.

## Contribution

(1) A differentiable multi-agent memory architecture that jointly optimizes retrieval utility, access control, and forgetting via a learnable eviction policy trained under a multi-objective loss. (2) A training framework that operationalizes governance constraints as differentiable loss terms, enabling end-to-end learning of eviction decisions. (3) An analysis (via comparison to GateMem benchmarks) demonstrating that coupling all three objectives in a single mechanism avoids the separated optimization failure identified in prior work.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | GateMem | Multi-principal memory governance benchmark |
| Primary metric | Governance Compliance Score | Combines utility, access, forget metrics |
| Baseline 1 | Long-context prompting (GPT-4) | Tests no explicit memory governance |
| Baseline 2 | RAG with FAISS | Tests retrieval without governance |
| Baseline 3 | MemGPT (LRU eviction) | Tests fixed eviction policy |
| Ablation | GAE w/o governance | Tests importance of governance vector |

### Why this setup validates the claim

The evaluation setup directly tests the central claim that governance-aware eviction improves multi-agent memory. GateMem is designed to stress memory governance with multi-party episodes, including access restrictions and targeted forgetting. Long-context prompting serves as a sanity check: if a trivial context fits, no memory is needed; otherwise it fails due to limited context window. RAG tests retrieval but lacks access control and forgetting mechanisms, leading to information leakage across principals. MemGPT uses fixed eviction (LRU) which ignores governance signals, causing it to evict high-utility memories prematurely or retain forgotten ones. The ablation isolates the effect of the governance vector, controlling for the learned eviction policy. The primary metric, Governance Compliance Score, captures both retrieval accuracy and adherence to access and forget constraints, making it a direct measure of the method's intended benefit. Together, they provide a falsifiable test: if our method does not outperform baselines on the compliance metric, the claim is unsupported.

### Expected outcome and causal chain

**vs. Long-context prompting (GPT-4)** — On a case where an agent must recall a fact written by a different principal after many turns, GPT-4's context window may truncate or the fact may be buried among irrelevant information. It produces wrong retrieval because it lacks explicit memory and relies on a fixed-length prompt. Our method instead retrieves from the key-value memory using governance-aware attention, retrieving the fact if the accessing principal is authorized. We expect a large gap (>20%) on long episodes but parity on very short ones where context fits.

**vs. RAG with FAISS** — On a case where a fact is intentionally forgotten via a forget write, RAG still retrieves it from the external store because it has no forget mechanism, only index-based retrieval. It produces unauthorized retrieval due to lack of governance. Our method's forget gate sets the governance vector to disallow retrieval, and the attention mask enforces this. We expect our method to have zero retrieval after forget while RAG leaks, leading to a >30% gap on forgotten items.

**vs. MemGPT (LRU eviction)** — On a case where a principal writes a fact with high utility but low recency (e.g., an important rule mentioned once early), LRU evicts it to make space for newer but less important facts. MemGPT loses the fact because it ignores utility. Our method's learned eviction policy considers utility and governance, preserving high-utility memories even if they are older. We expect our method to retain more important facts, leading to a >15% gap on storage-constrained settings.

### What would falsify this idea

If our method's Governance Compliance Score is not significantly higher than the ablation without governance, or if performance gains are uniform across all subsets rather than concentrated on scenarios with access control or forgetting constraints, then the governance vector is not providing the claimed benefit and the idea is falsified.

## References

1. GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents
2. Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions
3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
4. HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
5. Needle in the Haystack for Memory Based Large Language Models
6. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
7. LONG²RAG: Evaluating Long-Context & Long-Form Retrieval-Augmented Generation with Key Point Recall
8. LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks
