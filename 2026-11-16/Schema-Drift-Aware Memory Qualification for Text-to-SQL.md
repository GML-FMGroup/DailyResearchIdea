# Schema-Drift-Aware Memory Qualification for Text-to-SQL

## Motivation

Memory-based Text-to-SQL systems such as MEM-SQL gate retrieval layers with static match thresholds fixed at construction time, so a template that matched a historical query can still be retrieved after the underlying schema has drifted (a referenced column is dropped, a join key's referential integrity no longer holds, or an aggregation's grouping compatibility is broken), producing syntactically valid but semantically stale SQL. ROUTE's schema linking is likewise frozen after supervised fine-tuning, so neither branch re-validates stored memory against the live schema at retrieval time. The structural gap is that qualification is not monotone under schema evolution: a layer that qualified before a schema change may silently remain qualified afterward, and no existing mechanism makes qualification depend on the current schema state.

## Key Insight

If every stored template or block carries the exact schema-dependency invariants it relies on (column existence, join-key referential integrity, aggregation grouping compatibility) and qualification requires both the match threshold and all invariants to hold against the live catalog, then qualification becomes monotone-consistent under schema evolution: any schema change that invalidates a dependency deterministically demotes the layer, so a layer can only qualify when its stored evidence is still executable against the current schema.

## Method

**What it is.** We introduce DRIFT-GATE (Schema-Drift-Aware Memory Qualification), a retrieval-time qualification mechanism that augments each MEM-SQL memory layer with a set of runtime schema-dependency invariants and admits a layer only when both the original match threshold and every invariant hold against the live database catalog. Inputs: a natural-language request, the layered memory store (whole-query templates, reusable SQL blocks, concept-to-schema mappings), and the live schema catalog. Output: either a qualified layer's evidence forwarded to the generator, or a fall-through to On-demand Schema Linking.

**How it works.**

```pseudocode
# Offline: invariant extraction at memory construction
for each memory item m in layer L:
    deps = extract_schema_dependencies(m.sql)   # parse SQL AST
    m.invariants = {
        EXIST(col)        for each column reference in deps,
        REFINT(k1, k2)    for each join predicate in deps,
        GROUPCOMP(agg, gb) for each aggregation/group-by pair in deps,
    }
    store m with invariants

# Online: drift-aware qualification for a new request q
for each layer L in [templates, blocks, concept_mappings]:
    match_ok = every requirement r in L.assigned(q) has match(r) >= L.threshold
    if not match_ok: continue
    invariant_ok = True
    for each candidate item m in L matching q:
        for each inv in m.invariants:
            if inv.type == EXIST and not catalog.has_column(inv.col): invariant_ok = False
            if inv.type == REFINT and not catalog.referential_integrity_holds(inv.k1, inv.k2): invariant_ok = False
            if inv.type == GROUPCOMP and not catalog.grouping_compatible(inv.agg, inv.gb): invariant_ok = False
        if invariant_ok: emit m as qualified evidence; break
    if not invariant_ok: continue
if no layer qualified: invoke On-demand Schema Linking (MEM-SQL OSL)
```

Hyperparameters: match threshold per layer is inherited from MEM-SQL (templates highest, blocks middle, concept mappings lowest); invariant checks are boolean with no tunable threshold; a configurable `strict_mode` flag (default true) requires all three invariant families to pass rather than a majority.

**Why this design.** We chose to attach invariants at memory-construction time by parsing the SQL AST rather than re-deriving dependencies at retrieval time, because AST parsing is deterministic and cheap offline while retrieval-time re-parsing would add latency to every request, accepting the cost that invariants must be regenerated whenever a memory item is edited. We chose boolean invariant checks over a learned drift classifier because the three invariant families are decidable properties of the catalog (column existence, referential integrity, grouping compatibility) and a learned head would reintroduce the calibration problem that MEM-SQL's static thresholds already suffer from, accepting the cost that we cannot capture soft schema drift such as renamed-but-equivalent columns. We chose to gate at the layer level with fall-through to On-demand Schema Linking rather than per-item filtering, because MEM-SQL's generator expects a coherent layer's evidence and per-item filtering would produce a patchwork of stale and fresh items, accepting the cost that one bad item demotes an otherwise valid layer. A domain expert would not describe this as a variant of MEM-SQL because MEM-SQL treats memory qualification as a static similarity decision, whereas DRIFT-GATE makes qualification a function of the live schema state through decidable dependency invariants that MEM-SQL does not represent at all.

**Why it measures what we claim.** The computational quantity `invariant_ok` measures schema-dependency validity of a memory item because the assumption that a SQL item's parsed AST dependencies (column references, join predicates, aggregation/group-by pairs) are exactly the schema elements its execution requires; this assumption fails when a schema change preserves the referenced column name but alters its semantics (e.g., a unit change or a type widening that keeps the name), in which case `invariant_ok` reflects syntactic schema conformance rather than semantic executability. The layer-level qualification decision `match_ok AND invariant_ok` measures monotone consistency under schema evolution because the assumption that a schema change invalidating any dependency necessarily flips at least one invariant to false; this assumption fails when a drift removes a column that the item references only through a view or CTE not captured by AST parsing, in which case the metric reflects partial dependency coverage rather than full monotone consistency. The fall-through to On-demand Schema Linking measures the recovery of qualification coverage because the assumption that OSL produces correct schema evidence when no layer qualifies is inherited from MEM-SQL; this assumption fails when OSL itself is given a stale catalog snapshot, in which case the metric reflects the freshness of the catalog rather than the drift-awareness of the gate.

## Contribution

['A retrieval-time qualification mechanism (DRIFT-GATE) that augments each memory item with decidable schema-dependency invariants (column existence, join-key referential integrity, aggregation grouping compatibility) and admits a layer only when both the match threshold and all invariants hold against the live catalog, making memory qualification monotone-consistent under schema evolution.', 'A design principle that memory-layer qualification in Text-to-SQL should be a function of live schema state rather than a static similarity score, with a concrete invariant-extraction procedure from SQL ASTs that is decidable and requires no learned calibration.', 'An ablation-ready decomposition separating the contribution of match-threshold gating from invariant gating, enabling measurement of how much stale-template reuse each invariant family prevents.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HARE with schema-drift variants | History-aware requests plus injected catalog mutations expose stale memory |
| Primary metric | Execution accuracy on drifted subsets | Directly tests whether qualified evidence still executes correctly |
| Baseline | MEM-SQL (static thresholds) | Tests whether static similarity alone survives schema drift |
| Baseline | Direct Generation without history | Tests whether memory layers add value at all under drift |
| Baseline | MEM-SQL + On-demand Schema Linking only | Tests whether fall-through alone recovers coverage |
| Ablation-of-ours | DRIFT-GATE without invariant gating | Isolates the invariant checks from layer-level gating |

### Why this setup validates the claim

The central claim is that memory qualification must be a function of live schema state, not static similarity. HARE is the right substrate because MEM-SQL already established a 37.1%→50.0% gap there, so any additional gain must come from drift handling rather than from memory itself. Injecting controlled catalog mutations (dropped columns, broken joins, incompatible groupings) creates the exact conditions where static thresholds admit stale items. MEM-SQL with static thresholds tests the sub-claim that similarity alone is insufficient; Direct Generation tests whether memory helps at all; OSL-only tests whether fall-through recovers coverage without invariants. The ablation removes invariant checks while keeping layer gating, isolating the causal contribution of the decidable checks. Execution accuracy is the right metric because it is the only quantity that reflects whether qualified evidence actually runs against the live catalog, not merely whether it matches.

### Expected outcome and causal chain

**vs. MEM-SQL (static thresholds)** — On a case where a memory template references a column that was dropped or a join whose referential integrity broke, MEM-SQL admits the template because its match threshold is satisfied against the stored embedding, producing a syntactically plausible SQL that fails at execution. Our method instead evaluates EXIST and REFINT invariants against the live catalog and demotes the layer, falling through to OSL, so we expect a noticeable execution-accuracy gap concentrated on drifted subsets with parity on undrifted ones.

**vs. Direct Generation without history** — On a case where a new request reuses a join or calculation from an earlier verified query but the schema has drifted since, Direct Generation has no memory to consult and must re-derive the join from scratch, often guessing a wrong key. Our method qualifies the reusable block only when REFINT holds, and otherwise falls through to OSL with fresh schema evidence, so we expect DRIFT-GATE to exceed Direct Generation by a margin at least as large as MEM-SQL's 37.1%→50.0% gap, and larger on drift-heavy subsets.

**vs. MEM-SQL + OSL only** — On a case where a stale template passes MEM-SQL's threshold and OSL is only invoked when no layer qualifies, MEM-SQL+OSL never triggers fall-through and emits the stale template's SQL. Our method's invariant check forces the layer to fail qualification, invoking OSL precisely when drift invalidates the memory, so we expect the gap to appear only on requests whose matching memory items have broken invariants, confirming that gating rather than OSL alone drives the gain.

### What would falsify this idea

If DRIFT-GATE's gain over MEM-SQL is uniform across drifted and undrifted subsets rather than concentrated where invariants actually flip to false, then the invariants are not the causal mechanism and the central claim is wrong. Likewise, if the ablation without invariant gating matches DRIFT-GATE's accuracy, the decidable checks add nothing beyond layer-level gating.

## References

1. MEM-SQL: Memory-Empowered Text-to-SQL
