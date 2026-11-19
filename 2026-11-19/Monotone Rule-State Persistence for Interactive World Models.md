# Monotone Rule-State Persistence for Interactive World Models via Trace-Equivalence Invariant Checking

## Motivation

Interactive world models that expose user-editable rules (e.g., WorldRuler's executable 3D rule graph) promise customizable gameplay, but their save/load path is unverified: the abstract reports joint saving and restoration of rules and world states without any mechanism that guarantees a reloaded world executes the same rule trace or preserves object-attribute semantics. This is structural rather than empirical — the rule graph, dynamic object binding, and stored-state-constrained generation form a closed loop whose state is distributed across rule definitions, dependency edges, and per-object attribute values, so a serializer that persists each component independently can silently break dependency ordering or object binding while every individual field round-trips correctly. The same residual appears in the exploration branch (E1), where a world model adapts to new environments by re-collecting interaction data rather than by verifying that a persisted model state reproduces prior behavior, confirming that monotone persistence is a cross-tree gap no single branch closes.

## Key Insight

If rule execution is a deterministic function of (rule graph, dependency edges, object attribute states), then save/load correctness is exactly the property that the post-load execution trace is identical to the pre-save trace under the same action sequence, which is checkable by replaying a canonical probe trace and comparing trace fingerprints rather than by trusting per-field serialization.

## Method

**What it is.** MRSP (Monotone Rule-State Persistence) is a persistence layer that wraps WorldRuler's executable 3D rule graph. Input: a live world state W = (R, D, S) — rule definitions R, dependency edges D, object attribute states S — plus a rule-graph-derived probe action sequence A. Output: a versioned state object V with an explicit serialization, a coverage-bounded trace-equivalence certificate, and a graded save/load verdict (ACCEPT, REPAIR, REJECT, or UNVERIFIED) with a per-region coverage report.

**Load-bearing assumption (stated explicitly).** The probe sequence A is *action-sufficient over the rule-graph-derived action closure*: any divergence in rule execution or object-attribute semantics between a pre-save and post-load world state that lies within the closure of rules, dependency edges, object-binding sites, and rule-induced attribute transitions enumerated from the rule graph will manifest within A. This is a coverage-bounded claim, not a claim of full behavioral equivalence over the unbounded user action space; rule regions not covered by A are reported as UNVERIFIED rather than silently ACCEPTed.

**How it works.**

```pseudocode
# Phase 0: derive probe set A from the rule graph (coverage-derived, not fixed)
function DERIVE_PROBES(R, D, S):
    A = []
    for r in R:                      # one probe per rule
        A.append(PROBE_FOR_RULE(r))
    for e in D:                      # one probe per dependency edge
        A.append(PROBE_FOR_EDGE(e))
    for b in BINDING_SITES(R, S):    # one probe per object-binding site,
        A.append(PROBE_FOR_BINDING(b))   # including unseen-object bindings
    for t in ATTRIBUTE_TRANSITIONS(R, S):  # one probe per rule-induced
        A.append(PROBE_FOR_TRANSITION(t))  # attribute transition
    return A                         # |A| = |R| + |D| + |BINDING_SITES| + |ATTRIBUTE_TRANSITIONS|

# Phase 1: fingerprint the live world over the derived probe set
function FINGERPRINT(W, A):
    trace = []
    for a in A:
        obs, S = EXECUTE_RULE_GRAPH(W.R, W.D, W.S, a)
        trace.append(hash(obs, canonicalize(S)))
    return merkle_root(trace)        # order-sensitive trace fingerprint

# Phase 2: serialize as one versioned object with coverage metadata
function SAVE(W, A):
    V = {rules: R, deps: D, states: S,
         fingerprint: FINGERPRINT(W, A),
         version: parent_version + 1,
         dep_closure: TOPO_CLOSURE(D),
         probe_set: A,               # stored so LOAD replays the same probes
         coverage: COVERAGE_MAP(A, R, D, S)}  # rule/edge/binding/transition -> covered?
    return V

# Phase 3: restore + coverage-bounded invariant check on every load
function LOAD(V, A):
    W' = DESERIALIZE(V)
    fp' = FINGERPRINT(W', V.probe_set)
    if fp' == V.fingerprint:
        verdict = ACCEPT if COVERAGE_COMPLETE(V.coverage) else UNVERIFIED
        return verdict(W', certificate=(V.fingerprint, fp', V.coverage))
    # non-monotone: attempt repair by re-binding objects to stored states
    W'' = REBIND(W', V.states, V.dep_closure)   # dynamic object binding replay
    fp'' = FINGERPRINT(W'', V.probe_set)
    if fp'' == V.fingerprint:
        return REPAIR(W'', certificate=(V.fingerprint, fp'', V.coverage))
    return REJECT(V, diff=TRACE_DIFF(V.fingerprint, fp''))
```

The probe set A is derived from the rule graph itself (Phase 0): one probe per rule in R, one per dependency edge in D, one per object-binding site (including bindings to objects unseen at save time), and one per rule-induced attribute transition. `COVERAGE_MAP` records, for each rule, edge, binding site, and transition, whether at least one probe in A exercises it; `COVERAGE_COMPLETE` returns true iff every enumerated region is covered. `REBIND` re-applies stored object attribute states to any object whose binding drifted during serialization, constrained by the saved dependency closure; it is the only repair path and is bounded to one pass.

**Why this design.** We chose a Merkle-rooted trace fingerprint over per-field equality checks because rule execution is order-sensitive — two states with identical fields can diverge if dependency edges are reordered — accepting the cost that fingerprinting requires executing the probe trace at every save and load (≈|A| rule-graph steps, cheap relative to video generation). We chose a single versioned state object over three independent serializers because WorldRuler's closed loop couples rules, dependencies, and object states through dynamic object binding, so independent serialization admits partial-load states that are individually valid but jointly inconsistent; the cost is a coarser version granularity that prevents partial rollback. We chose one bounded repair pass (REBIND) over unbounded search because a repair that requires more than re-binding indicates a structural serialization bug that should surface as REJECT rather than be papered over; the cost is that some recoverable states are rejected. We chose a coverage-derived probe set over a fixed 64-action script because the load-bearing assumption is coverage-bounded, so the certificate must report which rule-graph regions were exercised and mark unexercised regions UNVERIFIED rather than ACCEPT. This differs from VRAG-style retrieval-augmented generation (E3), which conditions generation on retrieved global state but does not define a save/load invariant; a domain expert would not describe MRSP as a variant of retrieval because the mechanism is trace replay and fingerprint comparison, not nearest-neighbor conditioning.

**Why it measures what we claim.** `merkle_root(trace)` measures rule-state semantic equivalence over the rule-graph-derived action closure between pre-save and post-load worlds because the named assumption is that the probe sequence A is action-sufficient over that closure — i.e., any divergence in rule execution or object-attribute semantics within the enumerated rules, dependency edges, binding sites, and attribute transitions manifests within A; this assumption fails when a divergence requires an action outside A's coverage (e.g., a rule triggered only by a rare multi-object interaction not enumerated as a binding site or transition), in which case the fingerprint reflects agreement on the probed sub-manifold rather than full semantic equivalence, and the coverage map reports the unprobed region as UNVERIFIED. `dep_closure` measures dependency-edge preservation because the assumption is that topological closure is a complete encoding of execution ordering; this fails when a rule's ordering depends on runtime state not captured in the static edge set, in which case the closure reflects static structure only. The `REPAIR` verdict measures recoverability because the assumption is that binding drift is the dominant serialization failure mode; this fails when drift originates in rule-definition corruption, in which case REPAIR masks a definitional defect as a binding issue. The ACCEPT/REPAIR/REJECT/UNVERIFIED verdict measures monotonicity of persistence because the assumption is that trace equality under A is necessary for any downstream behavioral equivalence within the covered closure; this fails when two worlds agree on A but diverge on user actions outside the closure, in which case the verdict reflects probe-relative monotonicity and the UNVERIFIED label flags the uncovered region.

## Contribution

(1) A monotone rule-state persistence protocol (MRSP) that treats rule definitions, dependency edges, and object attribute states as a single versioned state object with a Merkle-rooted trace fingerprint as its serialization invariant. (2) A three-way save/load verdict (accept/repair/reject) driven by trace replay against a canonical probe sequence, with a bounded re-binding repair path constrained by the saved dependency closure. (3) A trace-equivalence certificate format that makes persistence correctness auditable per save/load transition rather than assumed from field round-tripping.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | WorldRuler rule-graph worlds with save/load states, plus a fault-injection harness that synthesizes save/load corruptions (edge reordering, stale defaults, instance-ID drift) on a re-implemented open rule-graph | Native setting with executable rules and bindings; injector provides ground-truth verdict labels without an external oracle |
| Primary metric | Save/load verdict correctness (ACCEPT/REPAIR/REJECT/UNVERIFIED vs. injector ground truth) | Directly tests monotone persistence claim |
| Secondary metric | Probe coverage curve: fraction of injected rule-graph divergences detectable as a function of probe-set size and composition (rules-only, +edges, +bindings, +transitions) | Empirically grounds the action-sufficiency assumption |
| Secondary metric | Repair-pass distribution: number of REBIND passes needed on injected binding-drift faults | Justifies the one-pass bound |
| Baseline | Independent per-field serializers | Tests joint-consistency sub-claim |
| Baseline | Retrieval-conditioned persistence baseline that attempts save/load via VRAG-style global-state conditioning (E3) | Tests trace-replay vs. retrieval distinction non-strawman |
| Baseline | Naive full-state dump without fingerprint | Tests necessity of trace fingerprint |
| Ablation-of-ours | MRSP without REBIND repair pass | Isolates repair contribution |
| Ablation-of-ours | MRSP with fixed 64-action probe script instead of coverage-derived A | Isolates coverage-derivation contribution |

### Why this setup validates the claim

The central claim is that a Merkle-rooted trace fingerprint over a coverage-derived probe set, combined with a bounded REBIND repair, yields a save/load verdict that is monotone with respect to rule-state semantic equivalence over the rule-graph-derived action closure. To falsify this, we need baselines that each isolate one sub-claim. Independent per-field serializers test whether joint consistency matters: if they match MRSP, then coupling rules/deps/states in one object buys nothing. The retrieval-conditioned persistence baseline tests whether trace replay is genuinely different from retrieval-conditioned generation when both attempt save/load: if it achieves the same verdict correctness, our mechanism distinction collapses. The naive full-state dump tests whether the fingerprint itself is load-bearing. The no-REBIND ablation isolates whether repair recovers states that would otherwise be rejected. The fixed-64-action ablation isolates whether coverage-derivation is load-bearing. Verdict correctness is the right primary metric because it directly encodes the ACCEPT/REPAIR/REJECT/UNVERIFIED trichotomy against ground-truth labels obtained from the fault injector, which synthesizes corruptions with known labels; a metric that only measured fingerprint equality would be circular. The coverage curve and repair-pass distribution ground the action-sufficiency and one-pass assumptions empirically rather than asserting them.

### Expected outcome and causal chain

**vs. Independent per-field serializers** — On a case where dynamic object binding drifts during serialization (e.g., an object re-bound to a new instance ID while its attribute state stays valid), the baseline produces ACCEPT because each field individually round-trips, yet the restored world diverges on any dependency-chain action because the binding no longer points at the stored instance. Our method instead detects the divergence via the probe trace, then REBINDs the drifted object using the saved dependency closure. We expect a concentrated gap on binding-drift subsets and near-parity on clean round-trips.

**vs. Retrieval-conditioned persistence baseline (E3)** — On a case where a rule's effect depends on a dependency edge that is reordered across save/load, the baseline produces plausible-looking but wrong generation because retrieval conditions on a global state vector that does not encode edge ordering, so the ordering defect is invisible. Our method instead fingerprints the ordered trace and rejects or repairs the state because the Merkle root is order-sensitive. We expect a noticeable gap on dependency-ordering subsets and parity on states with no edge permutations.

**vs. Naive full-state dump without fingerprint** — On a case where rule-definition corruption occurs silently (e.g., a rule's condition is serialized with a stale default), the baseline produces ACCEPT because it compares fields for equality and the corruption is internally consistent. Our method instead executes the probe trace and observes a divergent observation sequence, producing REJECT with a trace diff. We expect a gap concentrated on definition-corruption subsets, with the no-REBIND ablation showing a smaller but still positive gap on binding-drift subsets only.

**vs. MRSP with fixed 64-action probe script** — On a case where a divergence requires an action outside the fixed script's coverage (e.g., a rule triggered only by a rare multi-object interaction not enumerated in the 16 cross-object probes), the fixed-script variant produces ACCEPT because the probe trace agrees, while the coverage-derived variant either detects the divergence via a derived probe or reports UNVERIFIED for the uncovered region. We expect the coverage curve to show detection rate rising with probe-set size and composition (rules-only < +edges < +bindings < +transitions), and the fixed-script ablation to underperform on rare-trigger subsets.

### What would falsify this idea

If MRSP's verdict-correctness gain over the naive full-state dump is uniform across all subsets rather than concentrated on binding-drift and dependency-ordering subsets, then the trace-fingerprint mechanism is not doing the work we claim and the central monotonicity argument is wrong. Likewise, if the no-REBIND ablation matches full MRSP on binding-drift cases, then REBIND is not the dominant repair path and the repair-recoverability claim fails; and if the coverage curve saturates at a small probe-set size with no benefit from coverage-derivation, then the action-sufficiency assumption is not load-bearing and the fixed-script design would suffice.

## References

1. WorldRuler: Interactive World Modeling under Configurable Rules
