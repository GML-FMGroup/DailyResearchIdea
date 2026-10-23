# Prefix-Anchored Reconciliation: A Monotone Consistency Invariant for Speculative Agent Records

## Motivation

Heartbeat Thinking (the Idea Card's speculative realtime agent) lets periodic Thinker invocations write reasoning and tool results produced from partial transcripts into a shared record, but it has no mechanism to retract or repair those entries once the full transcript contradicts the prefix they were derived from. Structurally, the shared record is append-only and prefix-blind: an entry carries no provenance tying it to the transcript prefix that justified it, so later invocations consume stale or contradictory speculation as if it were established fact. This is the meta-level failure of speculative agent records generally — speculation is admitted without a bookkeeping discipline that keeps the record monotone in transcript refinement — and it recurs wherever partial-input speculation is accumulated across an interactive session.

## Key Insight

If every shared-record entry carries the exact transcript prefix that justified it, then the record can be kept monotone-consistent by a single rule: an entry survives a transcript refinement iff its supporting prefix remains a prefix of the refined transcript, so invalidation becomes a prefix test rather than a re-derivation of the whole record.

## Method

**What it is.** PAR (Prefix-Anchored Reconciliation) maintains the shared speculative record of a Heartbeat Thinking agent as a dependency graph whose nodes are entries (reasoning steps, resolved arguments, tool results) and whose edges encode 'entry B was derived using entry A'. Each node stores its *support prefix*: the exact partial-transcript string that was current when the entry was produced, plus a *read-set*: the set of context spans (transcript substrings, cached tool-result IDs, and prior-turn reference IDs) that the Thinker actually consumed during derivation, captured by a wrapper that logs every record read and every context span passed to the model. Inputs: a stream of transcript refinements and new heartbeat entries. Output: a shared record guaranteed monotone-consistent — no entry whose support prefix or read-set is contradicted by the current full transcript survives, and dependent entries are repaired or dropped.

**Load-bearing assumption (stated explicitly).** The prefix-containment test `T.startswith(v.support_prefix)` is a sound and complete proxy for evidential support *only if* the Thinker is prefix-Markovian with respect to its input: its output depends on the transcript solely through the substring it read. This assumption is load-bearing and is not guaranteed by the model; it is enforced operationally by recording the full read-set and by the calibration study below, which measures how often it fails.

**How it works.**
```pseudocode
state: G = (V, E)            # dependency graph of record entries
       T = ""                # current full transcript
       k = 3                 # max re-derivation passes per refinement

on_refinement(T_new):
    T = T_new
    # Phase 1: classify by support-prefix and read-set test
    for v in V ordered by depth descending:
        if not T.startswith(v.support_prefix):
            mark v INVALID
        else if not read_set_contained(v.read_set, T):
            mark v INVALID
    # Phase 2: cascade invalidation along dependency edges
    for v in V in reverse topological order of G:
        if any(u INVALID for u in parents(v)):
            mark v INVALID
    # Phase 3: bounded re-derivation
    for pass in 1..k:
        for v in V marked INVALID ordered by depth ascending:
            if all(u VALID for u in parents(v)):
                v' = rederive(v, T, valid_parents(v))   # re-run Thinker on T
                v'.support_prefix = T
                v'.read_set = capture_read_set(rederive_call)
                replace v with v' in G
    # Phase 4: quarantine survivors
    for v in V still INVALID: move v to quarantine (never read by Thinker)

on_heartbeat_entry(e, T):
    e.support_prefix = T
    e.read_set = capture_read_set(derivation_call)   # spans + cached tool-result IDs + prior-turn IDs
    e.parents = entries referenced during derivation of e
    add e to G with edges to e.parents
```

**Why this design.** We chose an explicit per-entry support prefix over a global 'record version number' because a single version cannot distinguish entries derived from different partial prefixes of the same utterance, accepting the cost of storing one string per entry (linear in transcript length, bounded by the heartbeat interval). We chose a dependency graph with reverse-topological cascade over independent per-entry re-validation because a stale argument resolution must also invalidate the tool call that consumed it, accepting the cost of maintaining edges that must be recorded at derivation time. We chose bounded re-derivation (k=3 passes) over unbounded fixpoint iteration because heartbeat latency budgets are hard realtime, accepting that a deep dependency chain may end in quarantine rather than a repaired value. We chose quarantine over deletion so that audit and debugging can inspect what speculation was discarded, accepting a small memory overhead. We added the read-set alongside the support prefix because the prefix test alone is unsound when the Thinker reads outside the recorded prefix; the read-set makes the test conservative (any change to a referenced span invalidates) at the cost of logging every context read. A domain expert would not describe PAR as a variant of cache invalidation in speculative execution (e.g., Knight/Soundarajan-style memoized speculation) because those systems invalidate on a *misprediction event* signaled by a commit point, whereas here there is no commit signal — invalidation is driven by a *prefix-containment and read-set containment test* against an ever-growing transcript, which is a strictly weaker and always-available signal that requires provenance the cache literature does not store.

**Why it measures what we claim.** The prefix-containment test `T.startswith(v.support_prefix)` measures *support validity* because of the assumption that a reasoning step's correctness depends on the transcript only through the substring it read — i.e., that the Thinker is prefix-Markovian with respect to its input; this assumption fails when an entry's derivation implicitly relied on conversational context outside its recorded support prefix (e.g., a tool result cached from an earlier turn), in which case the test reflects only local textual containment and not true evidential support. The read-set containment test `read_set_contained(v.read_set, T)` measures *evidential support* because of the assumption that the wrapper's logged read-set captures every context span the Thinker consumed; this assumption fails when the model attends to spans not passed through the wrapper (e.g., implicit system-prompt content), in which case the test reflects only the logged reads and not the full input. The dependency-edge cascade measures *contamination propagation* because of the assumption that an entry is only as valid as its parents — a conservative over-approximation; it fails when an entry used a parent only as a rejected hypothesis, in which case the cascade over-invalidates valid entries. The bounded re-derivation count k measures *repair completeness* because of the assumption that dependency chains are shallower than k in practice; it fails on deep chains, where quarantine count (not repair count) is what the metric then reflects. Quarantine rate is the operationalization of *stale-record persistence*: it measures how much speculation the monotone invariant had to discard because of assumption failure, not the invariant's failure itself. The correctness audit of surviving entries (fraction of non-quarantined entries whose re-derivation under the full transcript yields the same result) measures *semantic correctness* because of the assumption that surviving entries are not merely prefix-consistent but true; this assumption fails when an entry passes the prefix and read-set tests but is wrong because it relied on out-of-prefix context not captured by the wrapper, in which case the audit reflects the residual gap between containment and truth.

## Contribution

(1) A prefix-anchored dependency-graph record with a reconciliation algorithm that enforces a monotone consistency invariant over speculative agent speculation, including cascade invalidation and bounded re-derivation. (2) The design principle that partial-input speculation should be indexed by its support prefix rather than by time or version, making invalidation a containment test and exposing over-speculation as a measurable quarantine rate. (3) An instrumented shared-record implementation for realtime voice agents that logs provenance, invalidation cascades, and quarantine for analysis of speculative-record quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | FDBench v3 multi-turn tool dialogs (full set; subset analysis by utterance shape and dependency depth) | Contains mid-utterance refinements that stale speculation |
| Primary metric | Quarantine-free repair rate (fraction of entries that survive refinement in a non-quarantined state) | Directly measures monotone-consistency survival under refinement |
| Secondary metric | Correctness audit rate (fraction of surviving non-quarantined entries whose re-derivation under the full transcript yields the same result) | Bounds the gap between prefix-consistency and semantic correctness |
| Diagnostic metric | Prefix-Markovian violation rate (fraction of entries whose re-derivation under the recorded support prefix alone yields a different result than the original derivation) | Quantifies the fidelity of the prefix-containment proxy |
| Baseline | SHANKS-style windowed speculation | Tests whether prefix-anchoring beats chunk-bounded invalidation |
| Baseline | Global record-version invalidation | Tests whether per-entry support prefixes beat one version |
| Baseline | No-cascade independent re-validation | Tests whether dependency edges matter for contamination |
| Baseline | Knight/Soundarajan-style memoized speculation with commit-point invalidation | Tests whether prefix-containment (no commit signal) beats misprediction-event invalidation on the same traces |
| Ablation-of-ours | PAR with prefix test replaced by version match | Isolates the contribution of support-prefix containment |
| Ablation-of-ours | PAR with read-set test removed (prefix only) | Isolates the contribution of the read-set to soundness |

### Why this setup validates the claim

The central claim is that per-entry support prefixes plus dependency-edge cascading preserve a monotone-consistent record under mid-utterance transcript refinement better than mechanisms that lack either provenance or cascade. FDBench v3 is the right dataset because its multi-turn tool dialogs contain exactly the partial-prefix refinements (resolved arguments, corrected entities) that make a global version number insufficient. The SHANKS-style windowed baseline tests the sub-claim that chunk-bounded or interrupt-on-update speculation discards or keeps the wrong entries when a refinement lands mid-chunk; the global-version baseline tests the sub-claim that a single version cannot distinguish entries derived from different partial prefixes; the no-cascade baseline tests the sub-claim that a stale argument resolution must invalidate the tool call consuming it; the Knight/Soundarajan-style baseline tests the sub-claim that prefix-containment is a weaker but always-available signal than commit-point misprediction. The ablation swaps only the invalidation signal, so any gap between it and PAR is attributable to prefix-containment rather than to graph maintenance; the read-set-removed ablation isolates the soundness contribution of the read-set. Quarantine-free repair rate is the right primary metric because it counts entries that survive refinement in a usable (non-quarantined) state, which is precisely the monotone-consistency property claimed; final task quality alone would conflate repair with lucky re-derivation. The correctness audit and the prefix-Markovian violation rate are reported alongside so that the proxy's fidelity is quantified rather than assumed: if the violation rate is high, the quarantine-free repair rate reflects textual containment rather than evidential support, and the audit bounds that gap.

### Expected outcome and causal chain

**vs. SHANKS-style windowed speculation** — On a case where the speaker corrects a resolved slot mid-utterance (e.g., "book it for Tuesday—no, Wednesday"), the baseline produces a tool call with the stale Tuesday argument or discards the whole chunk and re-generates, because its commit point is the chunk boundary rather than a support-prefix test. Our method instead marks only the Tuesday argument entry INVALID, cascades to the dependent booking call, and re-derives both against the corrected transcript, so we expect a noticeable quarantine-free repair advantage concentrated on mid-utterance-correction subsets and rough parity on clean single-shot requests.

**vs. Global record-version invalidation** — On a case where two entries were derived from different partial prefixes of the same still-growing utterance, the baseline produces either over-invalidation (everything before the latest version is dropped) or under-invalidation (an entry whose prefix was superseded survives because the version bumped elsewhere), because one version cannot localize which entries actually read the contradicted text. Our method instead tests each entry's own support prefix against the current transcript, so we expect a gap that widens with the number of distinct partial prefixes per utterance and shrinks toward parity on short utterances.

**vs. No-cascade independent re-validation** — On a case where a stale argument resolution feeds a tool call that consumed it, the baseline produces a repaired argument alongside a still-stale tool result, because per-entry re-validation never propagates invalidity across the dependency edge. Our method instead reverse-topologically cascades invalidity so the consuming call is re-derived against valid parents, so we expect the advantage to concentrate on chains of depth ≥2 and vanish on depth-1 entries, where the two mechanisms coincide.

**vs. Knight/Soundarajan-style memoized speculation** — On a case where a refinement lands without a commit-point misprediction signal (e.g., a gradual entity correction spread across heartbeats), the baseline has no invalidation event and keeps the stale entry, because its invalidation is triggered only by a commit-point misprediction. Our method instead invalidates on the always-available prefix-containment and read-set tests, so we expect a quarantine-free repair advantage concentrated on refinements that lack a discrete commit signal and parity on cases with a clear commit point.

### What would falsify this idea

If PAR's quarantine-free repair advantage over the global-version and no-cascade baselines is uniform across subset depths and utterance shapes rather than concentrated on mid-utterance corrections and depth-≥2 chains, the causal story (prefix-containment localization plus edge cascade) is wrong. Equally falsifying: if the ablation with version-match invalidation matches PAR on those concentrated subsets, then support prefixes add nothing over a global version and the central claim collapses; and if the prefix-Markovian violation rate is high and the correctness audit shows surviving entries are frequently wrong, then the prefix-containment proxy is unsound and the monotone-consistency claim is not supported.

## References

1. Heartbeat Thinking: Speculative Reasoning and Tool Execution While Listening in Realtime Voice Agents
