# Versioned MemCube Transactions: Cross-Type Memory Consistency with Priority-Merge Conflict Resolution

## Motivation

MemOS introduces the MemCube as a unified abstraction over parametric, plaintext, and activation memory, but it provides no invariants that keep these heterogeneous entries consistent when they are updated concurrently, and no conflict-resolution protocol for simultaneous edits. The root cause is structural: the MemCube abstraction was defined as a representation-and-governance container without a concurrency or dependency model, so a weight edit, its plaintext justification, and its cached activation can silently diverge — a failure that recurs across the GraphRAG-to-MemOS lineage where memory updates are treated as independent writes rather than transactions over a shared dependency graph.

## Key Insight

Cross-type memory consistency can be enforced without a global lock because every MemCube update carries a versioned read-set over its dependency graph, and any two updates whose read-sets intersect are serialized by a deterministic priority-merge rule that is total and stable, so concurrent writers can be validated and merged locally while preserving the invariant that parametric, plaintext, and activation entries derived from the same source are observationally equivalent.

## Method

**What it is.** VMT (Versioned MemCube Transactions) is a transaction layer for MemOS MemCubes: it takes a batch of concurrent MemCube update requests (each tagged with a target memory type, a payload, and a read-set of dependency IDs) and outputs either a committed set of versioned MemCube entries or an abort with the conflicting dependency. It enforces a cross-type invariant — the *Derivation Invariant* — requiring that any parametric, plaintext, and activation entries sharing a provenance root be mutually consistent at every committed version. **Load-bearing assumption (stated explicitly):** the `DERIVATION_EQUIV` predicate is a complete and sound record of derivation, so that passing it at commit guarantees the parametric, plaintext, and activation entries sharing a provenance root are observationally equivalent. This assumption is load-bearing because the entire phase-3 commit gate inherits it; if it fails, VMT can commit entries that are bookkeeping-consistent but behaviorally divergent.

**How it works.**

```pseudocode
# Input: MemCube store S (each entry e has id, type, payload, version vector VV, provenance root r)
# Input: batch of update requests U = {u_1..u_n}; each u has:
#   target_id, mem_type, payload, read_set R(u) (dependency ids + versions), priority p(u), provenance_root r(u)
# Hyperparameters: max_retries = 3, merge_policy = PRIORITY_MERGE,
#   probe_set_size = 64 (held-out probes per provenance root),
#   tolerance_tau = 0.05 (max allowed output divergence for behavioral equivalence),
#   activation_recompute_budget = 1 (full recompute per commit, no sampling)

phase 1  # Snapshot and read-set validation
for u in U (in arrival order):
    for (dep_id, dep_ver) in R(u):
        if S[dep_id].version != dep_ver: mark u as STALE
    if STALE: u.status = ABORT; record conflict_cause = (dep_id, S[dep_id].version)

phase 2  # Conflict detection on overlapping read-sets
build conflict graph G over non-aborted U: edge (u,v) iff R(u) ∩ R(v) != ∅ or target_id(u)==target_id(v)
for each connected component C of G:
    order C by PRIORITY_MERGE key = (p(u) desc, u.arrival_ts asc, u.hash asc)   # total, deterministic

phase 3  # Serial commit with cross-type invariant check
for u in ordered C:
    apply u to S: create new entry version S[target_id].version += 1; VV[target_id] = VV[target_id] ⊔ VV[u]
    for each sibling type t in {parametric, plaintext, activation} sharing r(u):
        if exists sibling s of type t:
            if not DERIVATION_EQUIV(s, u): u.status = ABORT; rollback target_id; break
    if committed: propagate VV to all siblings sharing r(u)   # cross-type version bump

phase 4  # Retry and report
for u with status ABORT and retry_count < max_retries:
    refresh R(u) from S; retry_count += 1; resubmit u
return committed set, abort log with conflict_cause and merge order
```

`DERIVATION_EQUIV(s,u)` is a type-specific equivalence check with two components: (i) a *bookkeeping precondition* that the plaintext provenance hash is listed in the parametric entry's edit ledger (for plaintext↔parametric) or that the activation's cached context hash equals the plaintext entry's content hash (for activation↔plaintext); and (ii) a *behavioral equivalence test* on the actual payloads. For plaintext↔parametric, the behavioral test draws `probe_set_size = 64` held-out probes from the plaintext entry's content, runs them through the parametric entry, and compares outputs against the plaintext-derived reference; equivalence holds iff the mean absolute difference is ≤ `tolerance_tau = 0.05`. For activation↔plaintext, the behavioral test recomputes the activation from the plaintext entry (full recompute, `activation_recompute_budget = 1`) and checks that the recomputed activation matches the stored activation within `tolerance_tau`. Ledger/hash agreement is treated only as a necessary precondition, not as the invariant itself. `VV` is a standard version vector; `⊔` is pointwise max.

**Why this design.** We chose optimistic read-set validation (phase 1–2) over pessimistic locking because MemOS MemCubes are read-mostly and lock acquisition across three memory types would serialize the activation cache, accepting the cost that stale updates abort and must retry. We chose a priority-merge rule keyed on (priority, arrival timestamp, hash) over last-writer-wins because last-writer-wins silently discards a parametric edit whose plaintext justification was written concurrently, accepting the cost that low-priority writers may be starved and must retry up to `max_retries`. We chose a per-provenance-root version vector propagated to all sibling types over independent per-entry versions because the Derivation Invariant is a cross-type predicate and per-entry versions cannot detect a parametric edit that orphans its plaintext sibling, accepting the cost of O(siblings) version bumps per commit. We chose a behavioral equivalence test inside `DERIVATION_EQUIV` over a hash/ledger-only check because the literature on model editing (ROME, Hase et al., Cohen et al.) shows that a matching edit-ledger hash does not guarantee the parametric weights encode the plaintext's content, and activation caches are lossy functions of context, so hash equality certifies input identity, not representational equivalence; accepting the cost of `probe_set_size` forward passes and one activation recompute per commit. A domain expert would not describe VMT as a variant of database MVCC applied to MemOS because the load-bearing mechanism is the cross-type `DERIVATION_EQUIV` check on a shared provenance root — MVCC provides no notion of a derivation invariant spanning parametric, plaintext, and activation payloads, and MemOS itself provides no concurrency model at all.

**Why this measures what we claim.** The read-set intersection graph `G` measures *conflict potential* because the named assumption is that two updates conflict iff they read a common dependency at the same version — this assumption fails when two updates write disjoint entries that share a provenance root but no read dependency, in which case `G` reflects only syntactic read overlap and misses a semantic derivation conflict caught later by `DERIVATION_EQUIV`. The `DERIVATION_EQUIV` predicate measures *cross-type consistency* because the named assumption is that provenance hashes and edit ledgers are complete records of derivation — this assumption fails when a parametric edit is applied outside the VMT layer (e.g., direct weight write), in which case the predicate reflects only ledger-recorded derivations and reports a false consistent. The priority-merge key measures *deterministic conflict resolution* because the named assumption is that the (priority, timestamp, hash) tuple is total and stable across replicas — this assumption fails under clock skew, in which case the key reflects arrival-order bias rather than writer priority. The propagated version vector measures *invariant preservation at every committed version* because the named assumption is that all sibling types are updated through VMT — this assumption fails for activation caches evicted outside the layer, in which case the version vector reflects only VMT-managed siblings.

## Contribution

['A cross-type memory consistency protocol (VMT) for MemOS that introduces versioned read-set transactions over MemCube provenance roots and a deterministic priority-merge conflict-resolution rule enforcing a Derivation Invariant across parametric, plaintext, and activation entries.', 'A concrete design principle: conflict detection in heterogeneous memory must be keyed on provenance-root read-sets rather than per-entry versions, because per-entry versioning cannot detect cross-type orphaning of a plaintext justification by a concurrent parametric edit.', 'An open-source reference implementation of the VMT layer plus a concurrency stress harness that generates interleaved MemCube update traces across the three memory types for reproducibility.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Synthetic concurrent multi-type MemCube update traces generated with parameterized shared-provenance-root rate (ρ ∈ {0.1, 0.3, 0.5}) and read-set overlap rate (σ ∈ {0.1, 0.3, 0.5}); MemOS-derived traces deferred to a second milestone | Exercises read-set overlap and shared provenance roots; the generator parameters allow isolating the disjoint-write-shared-root subset for the falsification test |
| Primary metric | Derivation-Invariant violation rate at commit, measured by the behavioral equivalence test (probe_set_size=64, tolerance_tau=0.05) | Directly tests the cross-type consistency claim using the same behavioral check as the method |
| Secondary metric | False-consistent rate: fraction of commits where `DERIVATION_EQUIV` passes but the behavioral test fails, measured by injecting out-of-band weight writes (bypassing VMT) at rate β ∈ {0.05, 0.1, 0.2} | Quantifies the load-bearing assumption's failure mode and bounds the fraction of the invariant VMT can enforce |
| Secondary metric | Merge-order divergence rate across replicas under injected clock skew (skew ∈ {0, 10ms, 100ms, 1s}) | Tests the priority-merge key's stability assumption |
| Baseline | Last-writer-wins (no versioning) | Tests whether versioning prevents orphaned siblings |
| Baseline | Pessimistic per-type locking | Tests whether cross-type invariant needs global coordination |
| Baseline | Read-set validation only (no DERIVATION_EQUIV) | Isolates the load-bearing invariant check |
| Ablation-of-ours | VMT with per-entry versions, no VV propagation | Tests whether cross-type version bump matters |
| Ablation-of-ours | VMT with hash/ledger-only DERIVATION_EQUIV (no behavioral test) | Isolates the contribution of the behavioral equivalence component |

### Why this setup validates the claim

The central claim is that a cross-type Derivation Invariant, enforced by `DERIVATION_EQUIV` plus a provenance-root-propagated version vector, is what keeps parametric, plaintext, and activation entries mutually consistent under concurrent updates. The dataset must therefore contain concurrent updates that share provenance roots but may or may not share read-sets, since that is exactly the region where syntactic conflict detection and the invariant check diverge; the synthetic generator's parameters ρ and σ let us sweep this region and isolate the disjoint-write-shared-root subset. The primary metric — violation rate of the Derivation Invariant at commit, measured by the behavioral equivalence test — is the right one because it is the predicate the method exists to preserve; throughput or latency would not distinguish a method that commits fast but inconsistently. Last-writer-wins tests the sub-claim that versioning is needed at all; pessimistic locking tests whether cross-type coordination can be achieved without the invariant check; read-set-only validation isolates the contribution of `DERIVATION_EQUIV` by removing it while keeping everything else; the per-entry-version ablation isolates the contribution of the propagated version vector; the hash/ledger-only ablation isolates the contribution of the behavioral test inside `DERIVATION_EQUIV`. The false-consistent-rate experiment with injected out-of-band weight writes directly measures the load-bearing assumption's failure mode, bounding the fraction of the invariant VMT can actually enforce. Together these form a falsifiable test: if the gain is not concentrated where the predicted failure modes occur, the mechanism story is wrong.

### Expected outcome and causal chain

**vs. Last-writer-wins (no versioning)** — On a case where a parametric edit and its justifying plaintext edit are submitted concurrently, last-writer-wins commits both independently and silently discards the loser's justification, leaving a parametric entry whose edit ledger no longer matches any plaintext sibling. Our method instead detects the shared provenance root, serializes the two via priority-merge, and runs `DERIVATION_EQUIV` (including the behavioral test) before commit, aborting the inconsistent order. We expect a large violation-rate gap on the shared-provenance concurrent subset and near-parity on the disjoint-entry subset.

**vs. Pessimistic per-type locking** — On a case where an activation cache update and a plaintext edit share a provenance root but touch different memory types, per-type locking acquires disjoint locks and commits both, orphaning the activation's cached context hash from the plaintext content. Our method instead bumps the version vector across all siblings sharing the root and checks equivalence, so the orphan is caught. We expect the baseline to show violations concentrated on cross-type concurrent cases while our method stays near zero, at the cost of somewhat lower throughput on those cases.

**vs. Read-set validation only (no DERIVATION_EQUIV)** — On a case where two updates write disjoint entries that share a provenance root but have no read dependency, read-set validation sees no conflict and commits both, missing the semantic derivation conflict. Our method instead catches it in phase 3 via the invariant check. We expect the baseline's violations to cluster exactly on the disjoint-write-shared-root subset, where our method shows a clear gap, with parity on genuinely independent updates.

**vs. VMT with hash/ledger-only DERIVATION_EQUIV** — On a case where a parametric edit is applied outside VMT (direct weight write) but the edit ledger is updated to list the plaintext provenance hash, the hash/ledger-only check passes while the parametric weights do not encode the plaintext's content. Our method's behavioral test catches this divergence via the held-out probes, so we expect the hash/ledger-only ablation to show a higher false-consistent rate that grows with the out-of-band injection rate β, while full VMT's false-consistent rate stays near zero up to the probe-set coverage limit.

**vs. VMT with per-entry versions, no VV propagation** — On a case where a parametric edit and an activation update share a provenance root but are committed in different components of the conflict graph, per-entry versions do not bump the sibling's version, so a later read of the sibling sees a stale version. Our method propagates VV to all siblings, so the stale read is detected. We expect the ablation to show violations on cross-type concurrent cases that full VMT avoids.

### What would falsify this idea

If our violation-rate advantage over read-set-only validation is uniform across subsets rather than concentrated on disjoint-write-shared-root cases, or if the per-entry-version ablation matches full VMT on cross-type concurrent cases, or if the false-consistent rate under out-of-band weight injection is not reduced by the behavioral test relative to the hash/ledger-only ablation, then the cross-type invariant, the propagated version vector, and the behavioral equivalence component are not the load-bearing mechanisms and the central claim is wrong.

## References

1. MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models
