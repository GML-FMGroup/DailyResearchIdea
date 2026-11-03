# Dependency-Closure Oracles for Cross-Repository Coding Benchmarks

## Motivation

CrossSWE (the benchmark of 171 cross-repository instances) verifies task necessity by checking that single-repository edits fail its tests, but its oracle remains the union of inherited per-repository test suites. This means an agent can pass by making locally-correct edits in each repository while leaving a hidden transitive dependency (A→B→C) violated, which is precisely the cross-repository inconsistency failure mode that accounts for 44.2% of observed failures. The structural root cause is that per-repository tests are insensitive to the closure property of an edit set: they observe behavior only at test-reachable call sites, not at the dependency edges that connect the coordinated edits.

## Key Insight

The set of repositories an edit set must touch is exactly the least fixed point of the transitive closure of the inter-repository dependency graph restricted to the symbols the instance's specification mentions, so closure satisfaction is a graph-reachability invariant that per-repository test outcomes cannot observe but that a manifest/import/call-graph oracle can decide independently of the tests.

## Method

**What it is.** DCO (Dependency-Closure Oracle) takes a CrossSWE-style instance (natural-language spec, the repository set, and the agent's multi-repo patch) and outputs a three-valued closure verdict plus a per-edge closure-satisfaction score. It builds an explicit inter-repository dependency graph G from build manifests, import statements, and call sites, then checks whether the agent's edit set is closed under the transitive cross-repo dependencies that the spec's changed symbols induce.

**Load-bearing assumption (stated explicitly).** DCO assumes that the static inter-repository dependency graph built from manifests, imports, and call sites is a sound and sufficiently complete proxy for the true build-time/runtime dependency relation, such that an unedited successor symbol y in the closure is genuinely a required-but-missing edit rather than a symbol already satisfied by unchanged code. This assumption is load-bearing because the OPEN/CLOSED verdict and the sigma score both hinge on it: if y is already correct, then the witness edge (x→y) in W is a false witness and sigma is pessimistic. The assumption fails in two known ways: (i) unchanged-but-correct downstream symbols, where y's behavior is already consistent with the edit even though y was not touched; and (ii) reflection/dynamic-dispatch edges, where the true dependency is invisible to static call-graph construction (Livshits et al., APLAS 2005; Rountev et al.), so the graph under-approximates the true edge set while simultaneously over-approximating via spurious long-range import aliases. The 0.6 weight threshold is a hand-set constant calibrated on 20 held-out instances and has no monotonicity guarantee; the two error types (under- and over-approximation) are not separable by that threshold alone.

**How it works.**

```pseudocode
INPUT: spec S, repos R, agent patch P = {(repo, file, hunk)}
OUTPUT: verdict v in {CLOSED, OPEN, UNRESOLVABLE}, score sigma, witness edges W

# Phase 1: build the inter-repo dependency graph
G = empty directed graph over (repo, symbol) nodes
for each repo r in R:
    manifest_edges  = parse_build_manifests(r)      # Maven/Gradle/npm/Cargo/pyproject deps
    import_edges    = parse_imports(r)              # resolved against manifest versions
    call_edges      = static_call_sites(r)          # resolved via import_edges only
    add_edges(G, manifest_edges, weight=1.0)        # manifest edges are authoritative
    add_edges(G, import_edges,   weight=0.8)        # import edges may be transitive aliases
    add_edges(G, call_edges,     weight=0.6)        # call edges are heuristic

# Phase 2: seed set from the spec and from the patch
seeds = symbols_mentioned_in(S) union symbols_touched_by(P)

# Phase 3: closure computation (least fixed point)
closure = seeds
frontier = seeds
while frontier not empty:
    x = pop(frontier)
    for y in G.successors(x):                # x depends on y across repos
        if y not in closure:
            closure.add(y); push(frontier, y)

# Phase 4: check edit-set closure
edited = symbols_touched_by(P)
W = [(x, y) in G : x in edited and y in closure and y not in edited]
if W is empty:
    v = CLOSED
elif any edge in W has weight < 0.6:        # only weak evidence
    v = UNRESOLVABLE
else:
    v = OPEN

# Phase 5: graded score (soft closure)
sigma = 1 - sum(weight(e) for e in W) / sum(weight(e) for e in closure_edges)
return v, sigma, W
```

Hyperparameters: edge weights 1.0/0.8/0.6 for manifest/import/call evidence (set by a one-time calibration on 20 held-out instances, see block C); the 0.6 UNRESOLVABLE threshold is the weight below which an unedited dependency is treated as too weak to penalize. The calibration procedure is: for each of the 20 held-out instances, revert each edge in G one at a time and observe whether the build or test suite breaks; this yields a ground-truth label (breaks / does not break) for each edge. We then compute precision and recall of each edge type (manifest, import, call) against these labels, and sweep the UNRESOLVABLE threshold over five values {0.4, 0.5, 0.6, 0.7, 0.8} to confirm the 0.6 choice is stable (i.e., the OPEN/UNRESOLVABLE split does not change by more than 5% across the sweep). The first version is restricted to the two languages covering the majority of CrossSWE instances (Java and Python); unresolved imports in other languages are marked UNRESOLVABLE rather than requiring full multi-language call-graph resolution.

**Why this design.** We chose an explicit typed dependency graph over a learned or LLM-judged dependency predictor because the closure property must be decided by a fixed, auditable relation, accepting the cost that the graph misses dynamic dispatch and reflection-based dependencies. We chose a least-fixed-point closure over a one-hop dependency check because CrossSWE's failure mode is precisely transitive (A→B→C), accepting the cost that spurious long-range edges inflate the closure and must be pruned by the weight threshold. We chose graded edge weights over a binary graph because manifest edges are authoritative while call-graph edges are heuristic, accepting the cost of a calibration step; this differs from retrieval-style dependency inference (e.g., the import-graph heuristics in CodeQL-style taint tools) in that DCO never predicts which repository an edit should go to — it only checks whether the agent's own edit set is closed, so the mechanism is a verification invariant, not a nearest-neighbor lookup. We chose the three-valued verdict over a binary pass/fail because UNRESOLVABLE edges are genuinely ambiguous evidence and folding them into OPEN would penalize agents for dependencies the graph cannot see.

**Why it measures what we claim.** The edited-symbol set `edited` measures the agent's coordinated edit coverage because we assume the patch's touched symbols are exactly the symbols whose behavior the agent intended to change; this assumption fails when an agent edits a helper symbol as a side effect, in which case `edited` over-counts and sigma inflates. The closure set `closure` measures the spec-implied dependency requirement because we assume symbols_mentioned_in(S) plus symbols_touched_by(P) is a superset of the spec's true dependency seeds; this assumption fails when the spec refers to behavior by an alias not present in the graph, in which case `closure` under-approximates and the oracle is too lenient (yielding UNRESOLVABLE rather than a false CLOSED). The witness set `W` measures the specific unmet transitive dependencies because we assume that any edge (x→y) with x edited and y in closure but y not edited is a genuine required-but-missing edit; this assumption fails when y is satisfied by an unchanged but already-correct symbol, in which case W contains a false witness and sigma is pessimistic. The graded score sigma measures the fraction of required dependency-edge weight the agent's edits cover because we assume edge weight is monotone in the probability that the edge is a real build-time dependency; this assumption fails for reflection-based edges, in which case sigma reflects static-analysis coverage rather than true closure.

## Contribution

(1) A dependency-closure oracle (DCO) that turns the informal 'coordinated edit set' requirement of cross-repository benchmarks into a decidable least-fixed-point check over an explicit typed inter-repository dependency graph built from build manifests, imports, and call sites. (2) A three-valued (CLOSED/OPEN/UNRESOLVABLE) verdict with a graded closure-satisfaction score and witness edges, which separates agents that pass per-repository tests by luck from agents that actually propagate changes across hidden transitive dependencies. (3) An analysis of how many CrossSWE-style instances have closure-satisfying edits that nonetheless fail or pass per-repository tests, quantifying the gap between test-union oracles and closure oracles.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CrossSWE held-out instances (Java and Python only) | Native multi-repo transitive failure mode |
| Primary metric | Pass@1 on closure-required instances | Directly measures coordinated-edit success |
| Baseline | GPT-6.1 Sol (reported) | Strongest published CrossSWE agent |
| Baseline | Closure-blind self-refine agent | Tests whether closure check adds signal |
| Baseline | One-hop dependency checker | Tests whether transitivity matters |
| Ablation-of-ours | DCO with binary edges, no weights | Isolates graded-score contribution |
| Ablation-of-ours | DCO with semantic-dependency filter (prune W to edges whose downstream symbol's behavior actually changes under the patch) | Tests whether structural closure alone suffices |

### Why this setup validates the claim

The central claim is that CrossSWE failures are dominated by unmet *transitive* cross-repo dependencies, and that an explicit closure invariant catches them where agents and shallow checkers cannot. Each baseline probes one sub-claim. GPT-6.1 Sol tests whether a strong general agent, unaided, already internalizes closure; if DCO's verdicts correlate with its residual failures, the claim that closure is a distinct capability gap is supported. The closure-blind self-refine baseline tests whether *any* verification signal suffices — if it matches DCO, the closure structure is doing no work. The one-hop checker tests transitivity specifically: it should catch A→B misses but leak A→B→C ones. The binary-edge ablation tests whether graded weights matter for the UNRESOLVABLE/OPEN distinction. The semantic-dependency filter ablation tests whether structural closure alone suffices or whether pruning false witnesses (unchanged-but-correct downstream symbols) is necessary to avoid pessimistic sigma. Pass@1 on closure-required instances is the right metric because it conditions on exactly the population where the mechanism should fire; a uniform gain across all instances would suggest the effect is generic agent quality, not closure reasoning. The evaluation is restricted to Java and Python instances (the two languages covering the majority of CrossSWE) to bound implementation risk; unresolved imports in other languages are marked UNRESOLVABLE.

### Expected outcome and causal chain

**vs. GPT-6.1 Sol (reported)** — On a case where the spec changes a symbol in repo A whose call site in repo C depends transitively through repo B's re-export, the baseline edits A and B but leaves C's call site unpatched because its context window sees the local import, not the transitive closure. Our method flags the A→B→C witness edge and returns OPEN before merge, so we expect a concentrated Pass@1 gap on instances whose gold patch spans ≥3 repos, with near-parity on 2-repo instances.

**vs. Closure-blind self-refine agent** — On a case where the agent's self-review confirms its own patch is consistent, the baseline accepts an open edit set because self-critique re-reads the same patch without an external dependency relation. Our method supplies an auditable witness set W that is independent of the agent's reasoning, so we expect DCO to convert a subset of self-refine false-accepts into correct rejections, visible as higher precision on closure-required instances with no loss on already-closed ones.

**vs. One-hop dependency checker** — On a case where repo A depends on repo B, which depends on repo C, and the spec's changed symbol in A requires a signature update in C, the baseline sees no direct A→C edge and passes the patch. Our least-fixed-point closure traverses B to reach C, so we expect the one-hop checker to match DCO on 2-repo instances but diverge sharply on ≥3-repo chains, where its recall drops while DCO holds.

**vs. DCO with semantic-dependency filter (ablation)** — On a case where the closure contains an unedited successor y that is already correct (unchanged-but-correct downstream symbol), the unfiltered DCO includes (x→y) in W and returns OPEN with pessimistic sigma, while the filtered DCO prunes that edge because y's behavior does not change under the patch. We expect the filtered DCO to have higher precision on closure-required instances (fewer false OPEN verdicts) but identical recall on genuine missing edits, confirming that structural closure alone over-penalizes and that the semantic filter is necessary to avoid the load-bearing assumption's failure mode.

### What would falsify this idea

If DCO's gain over the closure-blind baseline is uniform across 2-repo and ≥3-repo instances rather than concentrated on the transitive chains the mechanism targets, the closure invariant is not the causal driver. Likewise, if the binary-edge ablation matches the graded version on the OPEN/UNRESOLVABLE split, the weighting scheme is decorative and the central design choice is unsupported; and if the semantic-dependency filter ablation shows no precision gain, the load-bearing assumption's failure mode is not material.

## References

1. CrossSWE: Benchmarking Cross-Repository Joint Development for Coding Agents
