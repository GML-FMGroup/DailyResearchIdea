# Causal Staleness-Bounded Verification for Asynchronous Subagent Orchestration

## Motivation

Asynchronous multi-agent systems like Sage-A1.5's orchestration layer compute verification signals against the latest available subagent state, which concurrent writes can supersede, making verdicts temporally inconsistent. The root cause is inherited from AReaL-style async rollouts: verification is treated as a read of the freshest snapshot rather than a read of a causally consistent one, so a verifier may approve or reject progress based on evidence that no longer holds. This structural flaw recurs across agentic verification pipelines, where no explicit synchronization bound exists between subagent writes and verifier reads.

## Key Insight

Verification is sound iff the state snapshot it reads dominates every write that causally precedes the subagent's current progress frontier in the dependency graph; a versioned store can compute this bound locally and either delay or condition the verdict, eliminating temporal inconsistency without serializing execution.

## Method

## Causal Staleness-Bounded Verification (CSBV)

**(A) What it is.** CSBV is a verification protocol for asynchronous subagent orchestration. Inputs: a dependency DAG over subtasks, a versioned state store tracking per-subagent write frontiers, a verifier function, and the verifier's declared read set R(t) (the set of state keys the verifier is instrumented to read, including shared scratchpads and non-ancestor state). Output: a verdict tagged either `causally-sufficient` (computed on a snapshot whose version dominates all writes in R(t) that causally precede the subtask's progress frontier) or `staleness-conditional` (with an explicit bound and the set of writes that may invalidate it). The load-bearing assumption is stated explicitly: the verifier's decision depends only on state written by the subtask's DAG ancestors, so that vector-clock dominance over Anc(t) write frontiers is both necessary and sufficient for the verdict to be causally sound. This assumption is checked at runtime by comparing R(t) to Anc(t); when R(t) contains keys written by non-ancestor subagents, the method falls back to computing `required` over R(t) rather than Anc(t), and the verdict is tagged `staleness-conditional` with the non-ancestor writes enumerated in `missing`.

**(B) How it works.**

```pseudocode
# State store: each subagent s has write frontier W[s] = (counter, vector clock)
# Dependency DAG: for subtask t, Anc(t) = transitive causal ancestors
# Verifier read set: R(t) = declared/instrumented keys the verifier reads

function verify(subagent t, verifier V, max_wait_iters=K):
    # explicit assumption check: does the verifier read only ancestor state?
    non_ancestor_keys = R(t) \ keys_written_by(Anc(t))
    if non_ancestor_keys is empty:
        required = union_{a in Anc(t)} W[a]        # causal write set (ancestor-only)
    else:
        required = union_{s in writers_of(R(t))} W[s]  # actual read set
    snapshot = store.snapshot()                     # latest available
    bound = snapshot.version
    if not dominates(bound, required):
        # try to advance snapshot up to K times
        for i in 1..K:
            snapshot = store.wait_advance(required, timeout=delta_i)
            if dominates(snapshot.version, required): break
    if dominates(snapshot.version, required):
        return V(snapshot), tag="causally-sufficient", bound=snapshot.version
    else:
        # emit conditional verdict with explicit staleness bound
        v = V(snapshot)
        missing = required \ snapshot.version
        return v, tag="staleness-conditional", missing=missing, bound=snapshot.version
```

`dominates(v, w)` is vector-clock dominance. `wait_advance` blocks until the store's frontier covers `required` or timeout `delta_i` elapses (exponential backoff, base 10ms, K=5). The orchestrator consumes `staleness-conditional` verdicts by scheduling a re-verification once `missing` writes land, rather than acting on the verdict immediately. The store maintains per-subagent vector clocks (O(|subagents|) metadata per write) and a frontier-heartbeat lease: each subagent's frontier is considered advertised only if a heartbeat was received within the last 500ms; a stalled writer's frontier is flagged as unadvertised and excluded from `required` until the lease is renewed, preventing silent trust of crashed writers.

**(C) Why this design.** We chose a vector-clock dominance bound over a wall-clock timeout because the dependency DAG already encodes which writes can causally affect a subtask, so a causal bound is both tighter and sound, accepting the cost that the store must maintain per-subagent vector clocks (O(|subagents|) metadata per write). We chose conditional verdicts over blocking until sufficiency because blocking serializes the verifier behind slow subagents, accepting the cost that the orchestrator must handle two verdict types and schedule re-checks. We chose exponential-backoff waiting (K=5, base 10ms) over unbounded waiting because most causal gaps close quickly in practice, accepting the cost that pathological dependency chains degrade to conditional verdicts. Unlike AReaL-style async rollout verification (which reads the latest snapshot unconditionally) or Sage-A1.5's progress checks (which assume verifier reads are consistent), CSBV makes the read's causal sufficiency an explicit, checked precondition; a domain expert would not describe it as a variant of either because neither computes or enforces a causal bound on the verifier's snapshot. The explicit assumption check (R(t) vs Anc(t)) and the frontier-heartbeat lease are included to make the causal bound robust to shared-state reads and crashed writers, respectively, without changing the core dominance mechanism.

**(D) Why it measures what we claim.** The vector-clock dominance check `dominates(snapshot.version, required)` measures causal sufficiency of the verification snapshot because the assumption that vector-clock dominance over per-subagent write frontiers is equivalent to "no causally prior write is missing from the snapshot" holds when each subagent's writes are totally ordered locally and the DAG's ancestor relation is transitively closed; this assumption fails when a subagent writes to the store without updating its frontier (e.g., a crashed writer), in which case the check reflects the last advertised frontier rather than actual causal coverage. The `staleness-conditional` tag measures the verifier's temporal inconsistency exposure because the assumption that the `missing` set enumerates exactly the writes that could invalidate the verdict holds when the verifier's decision depends only on state written by `Anc(t)`; this assumption fails when the verifier reads state from non-ancestor subagents (e.g., shared scratchpads), in which case the tag reflects only DAG-ancestor staleness. The `bound` field measures the freshness guarantee because the assumption that snapshot version equals the maximum write frontier across all subagents holds under the store's linearizable snapshot semantics; this assumption fails under eventual-consistency replication, in which case `bound` reflects the local replica's view. The explicit assumption check (R(t) vs Anc(t)) measures whether the verifier's actual read set is covered by the causal bound because the assumption that R(t) is correctly instrumented holds when the verifier's reads are declared or traced; this assumption fails when the verifier reads state through an uninstrumented path (e.g., a side channel), in which case the check reflects only the declared read set.

## Contribution

["A staleness-bounded verification protocol (CSBV) that computes a causal-consistency bound from the subtask dependency DAG and the versioned state store's per-subagent write frontiers, emitting either causally-sufficient or staleness-conditional verdicts.", 'A versioned state store design with vector-clock write frontiers and a `wait_advance` primitive that lets verifiers request causally sufficient snapshots with bounded waiting, without serializing subagent execution.', "A design principle: verification soundness in asynchronous orchestration is a property of the snapshot's causal dominance relation, not of wall-clock freshness, enabling conditional verdicts that the orchestrator can schedule around."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Async subagent DAG suite with injected slow writers (DAG generator: 10–100 subtasks, fan-out 2–5, 5–20% slow writers with 100–1000ms publish delay, 10% shared-scratchpad reads) | Induces causal gaps that stress verification reads |
| Primary metric | Fraction of verdicts acted on that are causally sufficient, verified by a ground-truth soundness oracle that checks verdict correctness against the full causal write set including shared state | Directly measures soundness of acted-on verdicts |
| Baseline | AReaL-style unconditional latest-snapshot verification | Represents async rollout reading latest state blindly |
| Baseline | Sage-A1.5-style progress-check verification | Represents verifier assuming consistent reads |
| Baseline | Blocking-until-sufficiency verifier | Represents the serializing alternative we reject |
| Baseline | COPS-style causal snapshot store (generic causal consistency) | Represents a standard causal-consistency primitive to compare bound tightness and metadata/latency |
| Ablation-of-ours | CSBV with wall-clock timeout instead of vector-clock bound | Isolates the causal bound as the causal ingredient |
| Ablation-of-ours | CSBV without explicit R(t) assumption check (ancestor-only bound) | Isolates the effect of handling non-ancestor reads |

### Why this setup validates the claim

The central claim is that making a read's causal sufficiency an explicit, checked precondition yields more soundly-acted-on verdicts than reading the latest snapshot unconditionally, and does so without serializing the verifier. The dataset must therefore contain dependency DAGs where slow writers create real causal gaps between what a subtask needs and what the snapshot holds — otherwise no method differs. The primary metric, the fraction of acted-on verdicts that are causally sufficient as judged by a ground-truth soundness oracle that checks verdict correctness against the full causal write set including shared state, is the right detector because it directly operationalizes the claim: a method that acts on stale reads will show a lower fraction precisely on the gap-inducing subset. AReaL-style verification tests the sub-claim that unconditional latest-snapshot reads are unsound; Sage-A1.5-style progress checks test the sub-claim that assuming consistent reads is insufficient; the blocking baseline tests that soundness need not cost serialization; the COPS-style causal snapshot store tests that CSBV's DAG-ancestor-restricted bound is strictly tighter than generic causal consistency, with quantified metadata and latency savings on the same DAG suite. The wall-clock-timeout ablation is the decisive control: if the gain comes from the causal bound rather than from merely waiting, swapping the vector-clock bound for a timeout should erode the gain. The ablation without the explicit R(t) assumption check isolates the effect of handling non-ancestor reads. Together these form a falsifiable test because each baseline isolates one mechanism gap and the metric is sensitive to exactly that gap.

### Expected outcome and causal chain

**vs. AReaL-style unconditional latest-snapshot verification** — On a case where a slow ancestor subagent has not yet published a write that a subtask's verifier depends on, the baseline reads the latest snapshot and returns a verdict that silently omits that causal write, so the orchestrator acts on an unsound verdict. Our method instead checks `dominates(snapshot.version, required)` against the ancestor write set (or the actual read set when non-ancestor reads are detected) and, when it fails, tags the verdict `staleness-conditional` with the missing set, so the orchestrator defers rather than acts. We expect a noticeable gap in acted-on causal sufficiency concentrated on the gap-inducing subset, with near-parity on DAGs where writers are fast and gaps rarely open.

**vs. Sage-A1.5-style progress-check verification** — On a case where a verifier reads a shared scratchpad or a non-ancestor write while the DAG's ancestor frontier appears satisfied, the baseline's progress check passes and it treats the read as consistent, producing a verdict that is stale with respect to state it actually consumed. Our method detects the non-ancestor read via the explicit R(t) check, computes `required` over the actual read set, and tags sufficiency correctly; its explicit `missing` set makes the residual non-ancestor exposure visible rather than hidden. We expect the baseline to show elevated unsound acted-on verdicts on cases with shared-scratchpad reads, and our method to match it on pure-ancestor cases, so the gap is confined to the shared-state subset.

**vs. Blocking-until-sufficiency verifier** — On a case with a long dependency chain where a slow ancestor takes many iterations to publish, the blocking baseline waits unboundedly, serializing the verifier behind that subagent and stalling throughput on independent subtasks. Our method waits with exponential backoff up to K iterations and then emits a `staleness-conditional` verdict, letting the orchestrator schedule re-verification and proceed on independent work. We expect causal sufficiency of acted-on verdicts to be comparable to the blocking baseline, but with markedly higher orchestration throughput on long-chain DAGs — the signal is parity on soundness with a throughput advantage that grows with chain length.

**vs. COPS-style causal snapshot store** — On the same DAG suite, the COPS-style store enforces generic causal consistency over all writes, which is sound but not restricted to the verifier's actual read set, so it may wait on irrelevant writes and incur higher metadata and latency. Our method's DAG-ancestor-restricted bound (or read-set-restricted bound when non-ancestor reads are detected) is strictly tighter, yielding lower wait times and metadata overhead while maintaining the same acted-on causal sufficiency. We expect CSBV to show equal or better soundness with lower latency and metadata cost on the gap-inducing subset.

### What would falsify this idea

If the gain in acted-on causal sufficiency were uniform across all subsets rather than concentrated on the gap-inducing and shared-scratchpad subsets, or if the wall-clock-timeout ablation matched CSBV's soundness, or if the ablation without the explicit R(t) assumption check matched CSBV's soundness on shared-scratchpad cases, then the causal bound and the read-set check are not doing the work and the central claim is wrong.

## References

1. Sage-A1.5: From Long-Horizon Execution to Scientific Exploration with Capability Composition
2. Generalizing Verifiable Instruction Following
3. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
