# Layer-Aware Memory Budgeting for Heterogeneous LLM Post-Training

## Motivation

Existing long-context reinforcement learning systems like LongStraw assume homogeneous computation across layers, applying a uniform chunk size for gradient checkpointing. This fails to account for the fundamentally different memory footprints of attention layers (quadratic in sequence length) versus MoE layers (linear), leading to suboptimal GPU utilization: uniform chunking either over-allocates memory to MoE layers or under-allocates to attention layers, causing either memory waste or overflow. This structural mismatch becomes a bottleneck when scaling context length beyond 2M tokens.

## Key Insight

Peak memory during backpropagation is determined by the layer with the largest activation footprint, not the sum of all layers, so allocating smaller chunks to attention layers reduces peak memory without increasing total recomputation time.

## Method

### (A) What it is
LAMB (Layer-Aware Memory Budgeting) takes a model graph and a target GPU memory budget as input and outputs per-layer chunk sizes (number of tokens per gradient checkpoint) and a recomputation strategy (full vs. selective) that minimizes total computation time subject to the memory constraint. It exploits the heterogeneity of activation memory across attention and MoE layers.

### (B) How it works
```python
# Profiling phase (offline, one forward pass)
profile = {}
for layer in model.layers:
    # Measure activation memory for a single chunk of size C0
    profile[layer] = measure_activation_memory(layer, chunk_size=C0, seq_len=S)

# Budget allocation phase
# Hyperparameters: C_attention_min, C_moe_min (minimum chunk sizes to avoid excessive recomputation)
budget_remaining = total_memory_budget
chunk_sizes = {}
for layer in sorted(model.layers, key=lambda l: profile[l]['per_token_memory'], descending=True):
    # Attention layers: allocate minimum chunk to reduce peak memory
    if layer.type == 'attention':
        chunk_sizes[layer] = C_attention_min
    # MoE layers: allocate as many tokens as possible, up to budget
    else:
        # Compute max chunk size that fits within remaining budget
        max_chunk = min(S, floor(budget_remaining / profile[layer]['per_token_memory']))
        chunk_sizes[layer] = max(C_moe_min, max_chunk)
    budget_remaining -= profile[layer]['per_token_memory'] * chunk_sizes[layer]

# For layers where chunk_size < S, enable gradient recomputation (selective recomputation for attention, full for MoE)
recompute_strategy = {
    layer: 'selective' if layer.type == 'attention' else 'full'
    for layer, cs in chunk_sizes.items() if cs < S
}
```

### (C) Why this design
We chose **offline profiling** over online adaptive tuning because it introduces no overhead during training and the memory characteristics of transformer layers are deterministic given the architecture; the trade-off is that it cannot adapt to runtime memory fluctuations. We used **per-token memory** as the allocation metric rather than total activation memory or FLOPs, because the memory bottleneck is per-token buffering in the backward pass, and FLOPs correlate poorly with memory for attention. We chose **static allocation** (chunk sizes fixed per layer) instead of dynamic scheduling because the heterogeneous memory budget problem is a simple knap-sack that is convex per-layer (memory scales linearly with chunk size for MLP/MoE, quadratically for attention but we cap at minimum chunk); dynamic scheduling would introduce overhead without benefit since the per-layer ratios are stable. The decision to **allocate minimum chunk to attention layers first** prioritizes the main memory bottleneck (attention), accepting that MoE layers may recompute more often; this is optimal because attention memory grows quadratically, while MoE recomputation cost is linear. Finally, we chose **selective recomputation for attention** (only recompute attention scores) and **full recomputation for MoE** (recompute entire forward pass) because attention's activation memory is dominated by scores, which are cheap to recompute (O(n^2)), whereas MoE's gating and expert computation have similar cost to forward; this trade-off saves memory without significant extra time.

### (D) Why it measures what we claim
The claim is that LAMB reduces peak memory under a fixed budget by exploiting layer heterogeneity. The **chunk size allocation** (computational quantity X) measures **memory efficiency** (motivation concept Y) under the assumption that each chunk's activation memory is independent and additive; this assumption fails when layers share residual stream buffers (tied in gradient checkpointing), in which case the metric overestimates memory savings. The **prioritization of attention layers** measures **bottleneck targeting** under the assumption that attention memory is the dominant peak; this assumption fails for extremely deep MoE models where expert memory per layer becomes large, in which case the metric may under-allocate to the actual peak layer. The **selective versus full recomputation choice** measures **compute-memory trade-off optimality** under the assumption that recomputation cost is proportional to memory saved; this assumption fails when the model uses fused kernels that cannot be partially recomputed, in which case selective recomputation incurs overhead without saving memory. Each component is causally linked to the motivation-level concept through these explicit assumptions, preventing the method from relying on unstated proxies.

## Contribution

(1) Introduces LAMB, a layer-aware memory budgeting algorithm that assigns heterogeneous chunk sizes and recomputation strategies to attention and MoE layers, enabling efficient long-context RL post-training under fixed GPU memory. (2) Reveals that attention layers are the primary memory bottleneck for long sequences but MoE layers dominate recomputation cost, leading to a design principle that minimizes peak memory by aggressively chunking attention layers while allocating remaining budget to MoE layers. (3) Provides an open-source profiling toolkit and integration wrapper for the verl framework, enabling reproducible evaluation of the trade-offs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AIME 2024 math reasoning | Standard benchmark for RL post-training. |
| Primary metric | Throughput (tokens/sec) under 80GB budget | Directly measures compute time under memory constraint. |
| Baseline 1 | Uniform chunking | All layers same chunk size. |
| Baseline 2 | Heuristic: allocate more to MoE layers | Inverse of our attention-first approach. |
| Baseline 3 | Standard gradient checkpointing | Full recomputation for all layers. |
| Ablation | LAMB with all recomputation = full | Remove selective recomputation advantage. |

### Why this setup validates the claim
This design forms a falsifiable test of LAMB's central claim—that exploiting layer heterogeneity reduces compute time under memory constraints. The dataset (AIME) stresses reasoning, but the memory bottleneck is architecture-agnostic, isolating the method's effect. Baseline 1 (uniform chunking) tests whether layer-aware allocation improves over naively equal treatment; if LAMB wins only on heterogeneous models, the claim holds. Baseline 2 (prioritizing MoE) tests the attention-first ordering; if LAMB's ordering is optimal, it should outperform this heuristic. Baseline 3 (standard checkpointing) is the default, showing total improvement. The ablation isolates the recomputation strategy's contribution. The primary metric—throughput under fixed budget—directly captures the claimed objective (minimizing time given memory). A diagnostic pattern would be LAMB showing larger gains on models with high layer heterogeneity (e.g., mixture of attention and large MoE layers) versus uniform models.

### Expected outcome and causal chain

**vs. Uniform chunking** — On a model where attention layers consume quadratically more memory per token, uniform chunking either sets a small chunk for all layers (wasting MoE capacity) or a large chunk causing OOM in attention. LAMB allocates only a minimal chunk to attention, freeing budget for MoE layers, reducing recomputation overhead. We expect a noticeable throughput gap (e.g., 20% higher) on models with many attention layers, but parity on uniform architectures.

**vs. Heuristic (MoE-first)** — On a model where attention layers dominate peak memory, the heuristic allocates large chunks to MoE first, leaving little budget for attention, causing attention to recompute frequently. LAMB prioritizes attention, allocating just enough to avoid its quadratic explosion, resulting in smoother memory usage. We expect LAMB to show 10-15% higher throughput on attention-heavy models, while the heuristic may win on MoE-heavy models.

**vs. Standard gradient checkpointing** — On a long-sequence training run (e.g., 8K tokens), standard checkpointing recomputes all layers per chunk, incurring high overhead. LAMB's selective recomputation for attention (cheap to recompute scores) and full for MoE reduces overall recomputation time. We expect LAMB to achieve 30-40% higher throughput than full recomputation, especially on deep models with many attention layers.

### What would falsify this idea
If LAMB's throughput gain is uniform across all layer configurations (i.e., equal improvement on attention-heavy and MoE-heavy models), the claim that exploiting heterogeneity drives the gain is false.

## References

1. LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget
2. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
3. Qwen2.5 Technical Report
4. HybridFlow: A Flexible and Efficient RLHF Framework
5. ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models
