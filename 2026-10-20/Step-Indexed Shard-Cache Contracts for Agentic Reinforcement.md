# Step-Indexed Shard-Cache Contracts for Agentic Reinforcement Learning

## Motivation

Agentic RL optimizes variable-length step trajectories (Agent-R1), but the execution substrate still assumes a fixed homogeneous data-parallel graph and fixed-size KV blocks. This structural mismatch means that step boundaries are not aligned with shard boundaries, so model-state shards and KV-cache blocks silently diverge across steps, introducing hidden gradient/context inconsistencies. Prior systems like HybridFlow and GEAR inherit this homogeneity assumption and therefore cannot guarantee that any step boundary is a legal re-sharding point.

## Key Insight

A step-indexed shard-cache contract that is closed under trajectory concatenation, truncation, and heterogeneous length mixing makes every step boundary a legal re-sharding point, eliminating the hidden divergence between model-state shards and KV-cache blocks without changing the optimizer.

## Method

# Algorithm: Step-Indexed Shard-Cache Contract (SISCC)
# Input: trajectory T = [s_1, s_2, ..., s_L] (variable length steps)
#        model-state shards M = {M_1, ..., M_P} (data-parallel grid, P=8 on 1 node)
#        KV-cache blocks K = {K_1, ..., K_B} (fixed-size blocks, block_size=16 tokens)
# Output: shard assignment A: step -> (subset of M, subset of K)
#         and migration plan that preserves contract closure

# Load-bearing assumption (stated explicitly):
# The closure set {concat, truncate, mix} exactly captures every trajectory
# manipulation the optimizer performs, so that a per-step union/restrict of
# model-shard and KV-block sets is a sufficient invariant for gradient
# consistency at every step boundary. This is a combinatorial invariant over
# shard/block identifiers; it does NOT by itself guarantee numerical
# consistency (same parameter version, same token prefix). To make the
# invariant numerical rather than purely combinatorial, each contract carries
# a parameter-version tag and a token-prefix hash per KV block, and closure
# operations are defined as commuting maps on (version, prefix-hash) pairs.
# The legality assertion checks version equality and prefix-hash continuity
# across the boundary, so it survives step permutation and asynchronous
# staleness.

# Phase 1: Step partition
partition = []
for step in T:
    # Each step gets a unique shard-cache contract
    contract = Contract(
        step_id = step.id,
        model_shards = assign_model_shards(step, M, strategy='step_affinity'),
        kv_blocks = assign_kv_blocks(step, K, strategy='step_affinity'),
        param_version = current_param_version(),   # integer tag, incremented on optimizer step
        kv_prefix_hash = {b: hash(step.token_prefix(b)) for b in kv_blocks},  # per-block token-prefix hash
        closure_ops = {'concat', 'truncate', 'mix'}
    )
    partition.append(contract)

# assign_model_shards(step, M, strategy='step_affinity'):
#   affinity score for shard m = 1.0 if m in previous step's model_shards else 0.0
#   plus 0.5 * (1 - load(m)/max_load) where load(m) is current step count on m
#   select top-|M_step| shards by score, where |M_step| = ceil(len(step.tokens) / tokens_per_shard)
#   tokens_per_shard = 2048 (fixed for 8-GPU single-node setup)

# assign_kv_blocks(step, K, strategy='step_affinity'):
#   affinity score for block k = 1.0 if k in previous step's kv_blocks else 0.0
#   select blocks covering step's token prefix in order, block_size=16 tokens

# Phase 2: Contract closure under operations
def concat(c1, c2):
    # Merge two step contracts into one that covers both
    # Version check: require c1.param_version == c2.param_version; if not, raise StaleContractError
    # Prefix-hash check: for any block b in both kv_blocks, require c1.kv_prefix_hash[b] == c2.kv_prefix_hash[b]
    return Contract(
        step_id = (c1.step_id, c2.step_id),
        model_shards = union(c1.model_shards, c2.model_shards),
        kv_blocks = union(c1.kv_blocks, c2.kv_blocks),
        param_version = c1.param_version,  # must equal c2.param_version
        kv_prefix_hash = merge_hashes(c1.kv_prefix_hash, c2.kv_prefix_hash),
        closure_ops = c1.closure_ops | c2.closure_ops
    )

def truncate(c, k):
    # Remove last k steps from contract
    return Contract(
        step_id = c.step_id[:-k],
        model_shards = restrict(c.model_shards, c.step_id[:-k]),
        kv_blocks = restrict(c.kv_blocks, c.step_id[:-k]),
        param_version = c.param_version,
        kv_prefix_hash = {b: h for b, h in c.kv_prefix_hash.items() if b in restrict(c.kv_blocks, c.step_id[:-k])},
        closure_ops = c.closure_ops
    )

def mix(c_list):
    # Heterogeneous length mixing: interleave steps from multiple trajectories
    # Version check: all c.param_version must be equal; else raise StaleContractError
    return Contract(
        step_id = interleave([c.step_id for c in c_list]),
        model_shards = union([c.model_shards for c in c_list]),
        kv_blocks = union([c.kv_blocks for c in c_list]),
        param_version = c_list[0].param_version,
        kv_prefix_hash = merge_hashes([c.kv_prefix_hash for c in c_list]),
        closure_ops = union([c.closure_ops for c in c_list])
    )

# Phase 3: Migration execution
for step in T:
    contract = partition[step.id]
    # Migrate model shards and KV blocks to match contract
    migrate_model_shards(contract.model_shards)
    migrate_kv_blocks(contract.kv_blocks)
    # Ensure all closure operations are valid at this step boundary
    assert is_legal_resharding_point(step, contract)

# is_legal_resharding_point(step, contract) checks:
#   1. contract.param_version == current_param_version()
#   2. for every KV block b in contract.kv_blocks, contract.kv_prefix_hash[b] == hash(step.token_prefix(b))
#   3. every model shard in contract.model_shards is resident on the rank assigned to step
#   4. no shard appears in two contracts with different param_version at the same boundary

**What it is**: SISCC (Step-Indexed Shard-Cache Contract) is a substrate-level mechanism that assigns model-state shards and KV-cache blocks to each step of a trajectory via a contract that is closed under concatenation, truncation, and heterogeneous length mixing. Its inputs are a variable-length step trajectory, the current model-shard grid, and the KV-cache block pool; its output is a step-indexed assignment and migration plan that guarantees any step boundary is a legal re-sharding point. Each contract carries a parameter-version tag and a per-block token-prefix hash, so the invariant is numerical (version equality, prefix-hash continuity) rather than purely combinatorial.

**How it works**: The pseudocode above shows three phases. Phase 1 partitions the trajectory into steps and assigns each step a contract that binds a subset of model shards and KV blocks using a step-affinity strategy (affinity score = 1.0 if shard/block was active in the previous step, plus 0.5*(1 - load/max_load); select top-k by score where k = ceil(len(step.tokens)/2048) for shards and blocks cover the token prefix at block_size=16). Phase 2 defines closure operations: concatenation merges two contracts by taking the union of their shard/block sets and merging their (version, prefix-hash) maps, requiring version equality and prefix-hash agreement on overlapping blocks; truncation restricts a contract to a prefix of steps and drops prefix hashes for removed blocks; mixing interleaves steps from multiple trajectories and unions their resources under the same version/hash checks. Phase 3 executes migrations at each step boundary, asserting that the boundary is a legal re-sharding point under the contract by checking version equality, prefix-hash continuity, shard residency, and absence of cross-version shard duplication.

**Why this design**: We chose a step-indexed contract over a fixed data-parallel grid because step boundaries are the natural optimization unit in agentic RL (Agent-R1), and a fixed grid forces silent divergence between model-state shards and KV-cache blocks; accepting the cost that contracts must be recomputed and migrations may incur communication overhead. We chose closure under concatenation, truncation, and mixing over ad-hoc per-step assignment because these operations are exactly the trajectory manipulations used in step-level credit assignment and heterogeneous batch construction; accepting the cost that the contract representation must track step provenance. We chose a union-based merge for concatenation over a re-sharding heuristic because union preserves the invariant that every step's resources are a subset of the merged contract, which is necessary for gradient consistency; accepting the cost that merged contracts may over-allocate resources. We chose an explicit migration phase over lazy allocation because lazy allocation can violate the legal re-sharding point assertion at step boundaries; accepting the cost of proactive communication. A domain expert would not describe this as a variant of HybridFlow or GEAR because those systems assume a fixed homogeneous graph and do not maintain a per-step contract that is closed under trajectory operations.

**Why this measures what we claim**: The computational quantity `is_legal_resharding_point(step, contract)` measures the motivation-level concept of "step boundaries are legal re-sharding points" because we assume that the contract's closure operations exactly capture the trajectory manipulations allowed by the optimizer (concatenation, truncation, mixing) and that version equality plus prefix-hash continuity are sufficient numerical proxies for gradient consistency; this assumption fails when the optimizer performs an operation not in the closure set (e.g., arbitrary step permutation) or when asynchronous staleness changes parameter versions between steps, in which case the assertion may pass spuriously and the metric reflects only local legality, not global consistency. The quantity `assign_model_shards(step, M, strategy='step_affinity')` measures the concept of "model-state shards are aligned with step boundaries" because we assume that step-affinity assignment (preferring shards active in the previous step) minimizes divergence between shards and KV blocks; this assumption fails when step lengths are highly heterogeneous and affinity causes load imbalance, in which case the assignment reflects affinity preference rather than optimal alignment. The quantity `union(c1.model_shards, c2.model_shards)` in concat measures the concept of "contract closure under concatenation" because we assume that the union of shard sets covers all steps in both contracts and that version/hash checks prevent double-counting; this assumption fails when shard sets overlap with different parameter versions or prefix hashes, in which case the union reflects a superset rather than an exact cover and the concat raises StaleContractError. The quantity `gradient-consistency error at step boundaries` measures the concept of "global gradient consistency" because we assume (A) that boundary-localized divergence dominates aggregate gradient error, i.e., there are no compensating errors across boundaries; this assumption fails (F) when errors cancel across steps or accumulate away from boundaries, in which case the metric reflects only boundary-localized error, not global gradient consistency.

## Contribution

['A step-indexed shard-cache contract (SISCC) that defines closure operations for trajectory concatenation, truncation, and heterogeneous length mixing, guaranteeing that any step boundary is a legal re-sharding point.', 'A migration mechanism that aligns model-state shards and KV-cache blocks with step boundaries, eliminating the hidden divergence present in fixed data-parallel grids.', 'An open-source implementation integrated with Agent-R1, demonstrating that step-indexed contracts reduce gradient/context inconsistency without modifying the optimizer.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Agentic-RL multi-turn trajectories (variable-length), generated by a synthetic generator with length-variance strata: uniform (all lengths within ±10%), high-variance (lengths spanning 1–50 steps), and truncation-heavy (30% of trajectories truncated by k∈{1,2,4,8} steps) | Exercises concat/truncate/mix operations directly and stratifies by length variance |
| Primary metric | Gradient-consistency error at step boundaries: L2 norm between the gradient computed with the contract's shard/cache assignment and the gradient computed with a reference full-synchronization assignment, measured only at step boundaries | Directly detects silent shard/cache divergence |
| Secondary metric | Wall-clock training throughput (steps/sec) and final task reward on a held-out set of 200 trajectories | Demonstrates downstream learning outcome |
| Baseline | Fixed data-parallel grid (HybridFlow-style, P=8, static sharding) | Tests whether per-step contracts beat static sharding |
| Baseline | Lazy KV allocation (no explicit migration; deallocation deferred until allocation pressure) | Tests whether proactive migration is necessary |
| Baseline | Per-step ad-hoc assignment (no closure; each step assigned independently) | Tests whether closure invariant matters |
| Ablation-of-ours | SISCC without closure tracking (keeps step-affinity assignment and migration, drops version/hash checks and closure ops) | Isolates the contribution of closure operations |
| Setup | Single-node 8-GPU, small open model (e.g., 1.5B parameters), synthetic variable-length trajectory generator; full distributed migration deferred to follow-up | Scopes first experiment to testable core claim within timeline |

### Why this setup validates the claim

The central claim is that a step-indexed contract closed under concatenation, truncation, and mixing keeps model-state shards and KV-cache blocks aligned at every step boundary, and that this alignment is what prevents silent divergence during trajectory manipulation. The fixed-grid baseline tests the sub-claim that static data-parallel sharding is insufficient when steps are variable-length and manipulated; if our method only matches it, the step-indexing premise is unmotivated. The lazy-allocation baseline tests the sub-claim that proactive migration is required to keep boundaries legal; if lazy allocation matches us, the migration phase is dead weight. The ad-hoc-assignment baseline tests the sub-claim that closure under the three operations is necessary rather than incidental. The no-closure ablation isolates the closure invariant from the rest of the mechanism: it keeps step-affinity assignment and migration but drops closure tracking, so any gap between it and full SISCC is attributable to closure alone. Gradient-consistency error at step boundaries is the right metric because the claim is specifically about alignment at boundaries, not aggregate throughput; a metric that averaged over trajectories would wash out the boundary-localized effect the design predicts. We explicitly state the equivalence assumption (A) that boundary-localized divergence dominates aggregate gradient error (no compensating errors across boundaries); the failure mode (F) is that errors cancel across steps or accumulate away from boundaries, in which case the metric reflects only boundary-localized error, not global gradient consistency. The secondary metrics (throughput and final reward) test whether the alignment translates to downstream learning outcomes.

### Expected outcome and causal chain

**vs. Fixed data-parallel grid (HybridFlow-style)** — On a case where a heterogeneous batch mixes a long trajectory with many short ones, the fixed grid produces silent divergence because shards are pinned by data-parallel rank while KV blocks are allocated by sequence position, so step boundaries land mid-shard. Our method instead recomputes a per-step contract and migrates shards and blocks to match, because step-affinity plus explicit migration makes every boundary a legal re-sharding point. We expect a noticeable gradient-consistency gap concentrated on high-length-variance batches and near-parity on uniform-length batches. We also expect higher wall-clock throughput on high-variance batches (due to fewer stalls from divergence) and higher final task reward on the held-out set.

**vs. Lazy KV allocation** — On a case where truncation removes the last k steps of a trajectory, lazy allocation produces stale KV blocks that remain bound to now-removed steps because deallocation is deferred until the next allocation pressure, so the truncation boundary is not a legal re-sharding point. Our method instead restricts the contract to the retained prefix during truncate and migrates immediately, because the closure operation carries provenance of which blocks belong to which steps. We expect a visible error spike at truncation boundaries for lazy allocation that our method does not exhibit, with the gap scaling with k.

**vs. Per-step ad-hoc assignment (no closure)** — On a case where two trajectories are concatenated for step-level credit assignment, ad-hoc assignment produces double-counted or orphaned shards because each step was assigned independently with no invariant tying the merged contract to the union of the originals. Our method instead merges via union under the concat closure, because the contract representation tracks step provenance and version/hash checks prevent double-counting, so the union is an exact cover of both step sets. We expect a gap that grows with the number of concatenated trajectories, and near-parity on single-trajectory cases where concat is never invoked.

### What would falsify this idea

If the gap over the fixed-grid baseline is uniform across length-variance strata rather than concentrated on high-length-variance and truncation-heavy batches, then the step-boundary alignment mechanism is not what produces the gain and the central claim is wrong. Likewise, if the no-closure ablation matches full SISCC on concatenation-heavy batches, closure tracking is not doing the work we attribute to it.

## References

1. Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning
2. HybridFlow: A Flexible and Efficient RLHF Framework
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
4. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
5. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text
6. Improving Generalization of Alignment with Human Preferences through Group Invariant Learning
7. Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates
8. GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models
