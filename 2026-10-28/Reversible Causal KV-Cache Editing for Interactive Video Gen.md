# Reversible Causal KV-Cache Editing for Interactive Video Generation

## Motivation

Existing interactive video generation models such as Hy-Nomas rely on an append-only causal KV cache that is never revised, so any physical or geometric inconsistency discovered after generation cannot be corrected without full recomputation. This limitation is structural: the history substrate is monotonic and non-revisable, a property shared across multiple paper-trees (Hy-Nomas, Yume-1.5, Decoupled DMD) where generation is a forward-only process over a fixed or compressed conditioning context. Consequently, even with reward signals from preference optimization (e.g., PhyGDPO) to detect violations, no mechanism exists to act on them retroactively.

## Key Insight

Tagging each KV entry with a provenance trace that records its exact generative lineage enables exact inverse edits, so a violated segment can be rolled back and repaired without recomputing the entire history, because the cache's contents are a deterministic function of the recorded provenance and can be restored to any prior state.

## Method

**What it is** — Reversible Causal KV-Cache Editing (RCKE) is a mechanism for interactive video generation that maintains a causal KV cache as a checkpointed substrate: each segment's KV entries are tagged with a provenance trace (segment ID, prompt, camera parameters, random seed, and a block index into a stored checkpoint buffer), and the cache stores an explicit per-segment KV snapshot in a bounded ring buffer of the last K=8 segment checkpoints. Given a detected violation (e.g., physical inconsistency), RCKE rolls back the cache to the stored checkpoint before the violated segment, and re-generates a repaired continuation using the same model, without recomputing earlier segments. The load-bearing assumption is stated explicitly: rollback restores the exact pre-violation prefix by loading a stored KV tensor, not by re-deriving it from provenance; re-generation of the repaired suffix uses the same model with a fixed sampler (DDIM, 50 steps, η=0) and deterministic kernels (torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8), and we measure reconstruction error when determinism is relaxed.

**How it works** — The core operations are:

```pseudocode
# Cache entry: (key, value, provenance) where provenance = (seg_id, prompt, cam_params, seed, checkpoint_idx)
# Global history: list of segments with their provenance and checkpoint indices
# Checkpoint buffer: ring buffer of K=8 stored KV tensors, one per segment, indexed by checkpoint_idx

function generate_segment(model, cache, prompt, cam_params):
    seed = random_seed()
    checkpoint_idx = store_checkpoint(cache)  # store current cache tensor in ring buffer
    new_kv = model.forward(prompt, cam_params, cache)  # causal attention over cache; fixed sampler DDIM 50 steps, η=0
    provenance = (next_seg_id(), prompt, cam_params, seed, checkpoint_idx)
    cache.append(new_kv, provenance)
    history.append(provenance)
    return new_kv

function detect_violation(segment, reward_model):
    # reward_model: PhyGDPO-style preference model trained to score physical/geometric consistency
    score = reward_model.score(segment)
    return score < threshold  # threshold = 0.5 (calibrated on validation set of 512 labeled segments to maximize F1)

function rollback_and_repair(cache, history, violated_seg_id, new_prompt, new_cam_params):
    # Find the provenance of the violated segment
    prov = history[violated_seg_id]
    # Restore cache to state before violated segment by loading stored checkpoint
    cache = load_checkpoint(prov.checkpoint_idx)  # O(1) tensor load from ring buffer
    # Remove violated segment and all subsequent segments from history
    history = history[:violated_seg_id]
    # Re-generate repaired continuation with new prompt/cam_params
    new_seg = generate_segment(model, cache, new_prompt, new_cam_params)
    return cache, history, new_seg

function store_checkpoint(cache):
    # Store a copy of the current cache tensor in the next slot of the ring buffer (K=8)
    # Returns the index of the stored checkpoint
    idx = next_ring_index()
    checkpoint_buffer[idx] = cache.clone()
    return idx

function load_checkpoint(idx):
    # Return the stored KV tensor from the ring buffer at index idx
    return checkpoint_buffer[idx].clone()
```

**Why this design** — We chose a provenance-tagged append-only cache with explicit per-segment checkpoints over a mutable cache with in-place editing because in-place editing would break the causal attention structure (future segments depend on past keys/values), whereas truncation via checkpoint load preserves causality by removing only the suffix. We chose stored checkpoints over checksum-based truncation because diffusion/flow-based video generators are stochastic samplers whose outputs depend on the full noise schedule, sampler state, and floating-point reduction order, not just a seed; GPU non-determinism (atomic adds, cuDNN/cuBLAS kernel selection, TF32) means identical inputs do not yield bit-identical KV tensors across runs or hardware, so a cumulative checksum over KV blocks will not match on re-execution and a binary-search truncation loop would never terminate or would truncate to the wrong prefix. Storing explicit checkpoints avoids this by restoring the exact tensor that was present before the violated segment, at the cost of K=8 extra cache copies (memory overhead = 8 × cache size, bounded by the ring buffer). We chose to re-generate the repaired segment from scratch using the same model rather than attempting to patch the existing KV entries because patching would require a learned editor that could introduce new inconsistencies, accepting the cost of one extra forward pass per repair. This design is not a variant of retrieval-based editing (e.g., k-NN cache editing) because the provenance trace enables exact inverse operations on the generative process via stored checkpoints, not approximate nearest-neighbor lookup; nor is it a two-module controller (Anti-pattern 4) because rollback and repair are unified in a single cache operation that directly manipulates the history substrate.

**Why it measures what we claim** — The checkpoint index measures the exact generative lineage of each KV entry because we assume that the cache state is stored as an explicit tensor and restored by loading that tensor, not by re-deriving it from provenance; this assumption fails when the checkpoint buffer overflows (K=8) and the required checkpoint has been evicted, in which case rollback falls back to full regeneration from the nearest available checkpoint and the restored state is an approximation. The violation score from the reward model measures physical/geometric consistency because we assume the reward model is calibrated on held-out violations (threshold set to maximize F1 on a validation set of 512 labeled segments); this assumption fails when the violation type is out-of-distribution for the reward model, in which case the score reflects the model's bias rather than true consistency. The rollback operation's correctness (restoring the exact pre-violation state) measures the reversibility of the cache because we assume the stored checkpoint tensor is bit-identical to the cache state at the time of storage; this assumption fails when the cache has been compressed (e.g., via linear attention in Yume-1.5), in which case the stored checkpoint is a lossy summary and the restored state is an approximation, and we measure reconstruction error (L2 distance between restored and original KV tensors) to bound this effect.

## Contribution

['A reversible causal KV-cache editing mechanism (RCKE) that tags each KV entry with a provenance trace and supports exact rollback-and-repair of violated segments without recomputing the entire history.', 'A concrete design principle: provenance-tagged append-only caches with checksum-based truncation preserve causal attention structure while enabling non-monotonic history revision, trading O(log n) truncation cost for exact reversibility.', 'An open-source implementation of RCKE integrated with a diffusion backbone (Hy-Nomas-style) and a PhyGDPO-style reward model for violation detection, demonstrating repair of physical inconsistencies in interactive video generation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Synthetic 2D/3D scene generator with programmatic violation injection (teleportation, wall penetration, camera jump) | Provides exact ground-truth labels and violation locations, reproducible |
| Primary metric | Repair success rate (fraction of violations fixed) | Directly measures whether rollback+regeneration restores consistency |
| Baseline | Full regeneration from scratch | Tests whether cache reuse preserves quality vs. recomputation |
| Baseline | In-place KV patching (learned editor) | Tests whether checkpoint rollback avoids editor-induced new inconsistencies |
| Baseline | No rollback (continue from violated state) | Tests whether causal rollback is necessary for repair |
| Baseline | Periodic full KV snapshots (every 10 segments) | Tests whether per-segment checkpoints improve repair success vs. coarse snapshots |
| Baseline | Diffusion-forcing-style suffix resampling | Tests whether RCKE's checkpoint rollback outperforms resampling without stored checkpoints |
| Ablation-of-ours | RCKE without per-segment checkpoints (checksum-based truncation) | Isolates cost/benefit of stored checkpoints vs. checksum truncation |

### Why this setup validates the claim

The central claim is that reversible causal KV-cache editing enables efficient, exact repair of interactive video generation without recomputing earlier segments. The synthetic scene generator supplies programmatic physical violations with known locations, so repair success rate directly measures whether the method fixes the targeted inconsistency. Full regeneration from scratch tests the efficiency sub-claim: if RCKE matches its quality at lower cost, cache reuse is validated. In-place KV patching tests the causality sub-claim: if patching introduces new inconsistencies while RCKE does not, checkpoint-based rollback is shown to preserve causal structure. No-rollback tests the necessity of rollback: if continuing from the violated state fails to repair, the rollback step is causally required. The periodic full KV snapshots baseline tests whether per-segment checkpoints (K=8) provide finer-grained rollback than coarse snapshots, isolating the benefit of checkpoint granularity. The diffusion-forcing-style suffix resampling baseline tests whether RCKE's stored checkpoints outperform resampling without explicit checkpoints, isolating the value of the checkpoint buffer. The ablation isolates the checkpoint mechanism: if checksum-based truncation fails to terminate or truncates incorrectly due to non-determinism, while stored checkpoints succeed, the checkpoint design is justified. Repair success rate is the right metric because it directly operationalizes the predicted effect—successful repair—and would be flat if the mechanism were inert. Additionally, we measure wall-clock and memory scaling of repair latency versus prefix length (10/50/200 segments) to show a regime where full regeneration is infeasible, and we report calibration curves for the reward-model threshold on a validation set of 512 labeled segments, plus a human-annotated consistency check on a held-out subset of 100 segments to validate the proxy.

### Expected outcome and causal chain

**vs. Full regeneration from scratch** — On a case where a mid-video physical violation (e.g., object teleportation) requires repair, full regeneration recomputes all segments from the start, which may fix the violation but at high latency and with potential drift in earlier segments due to stochastic sampling. Our method instead loads the stored checkpoint before the violation and regenerates the suffix, preserving earlier segments exactly. We expect RCKE to match full regeneration on repair success rate but with a noticeable latency advantage (e.g., 3× faster for 50-segment prefixes), and parity on consistency for earlier segments. At prefix length 200, full regeneration becomes infeasible (exceeds memory or time budget), while RCKE remains feasible.

**vs. In-place KV patching (learned editor)** — On a case where a geometric inconsistency (e.g., camera pose jump) occurs, in-place patching edits the violated segment's KV entries via a learned editor, which can introduce new inconsistencies because the editor is not constrained to preserve causal dependencies. Our method instead loads the stored checkpoint before the violation and regenerates the segment from scratch, ensuring the new KV entries are causally consistent with the prefix. We expect RCKE to achieve higher repair success rate on geometric violations, with the gap concentrated on violations requiring long-range causal consistency.

**vs. No rollback (continue from violated state)** — On a case where a physical violation (e.g., object passing through a wall) is detected, continuing generation from the violated state propagates the inconsistency because future segments attend to corrupted KV entries. Our method instead rolls back to before the violation and regenerates with a corrected prompt, removing the corrupted suffix entirely. We expect RCKE to show a large repair success rate advantage on violations that cascade, with parity on isolated violations that do not affect future segments.

**vs. Periodic full KV snapshots (every 10 segments)** — On a case where a violation occurs at a segment not aligned with the snapshot interval, periodic snapshots force rollback to the nearest earlier snapshot, discarding up to 9 valid segments and requiring their regeneration. Our method stores a checkpoint per segment (K=8 ring buffer), enabling rollback to the exact pre-violation state. We expect RCKE to achieve higher repair success rate (by avoiding unnecessary regeneration of valid segments) and lower latency, with the gap increasing for violations occurring far from snapshot boundaries.

**vs. Diffusion-forcing-style suffix resampling** — On a case where a violation is detected, diffusion-forcing-style suffix resampling regenerates the suffix without explicit stored checkpoints, relying on the model's stochastic sampler to produce a consistent continuation. Our method loads the exact stored checkpoint before the violation, ensuring the prefix is bit-identical to the original. We expect RCKE to achieve higher repair success rate on violations requiring exact prefix preservation, with the gap concentrated on cases where resampling introduces drift in earlier segments.

### What would falsify this idea

If RCKE's repair success rate is uniform across violation types and baselines rather than concentrated on violations where causal rollback is predicted to matter (cascading or geometric inconsistencies), then the central claim that reversible causal editing is the key mechanism would be wrong. Additionally, if the checksum-based truncation ablation matches RCKE on both success rate and memory usage (i.e., checksum truncation terminates correctly and restores the exact prefix), the checkpoint design would be unnecessary.

## References

1. Hy-Nomas: Towards Free-form Interactive World Generation
2. Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield
3. Yume-1.5: A Text-Controlled Interactive World Generation Model
