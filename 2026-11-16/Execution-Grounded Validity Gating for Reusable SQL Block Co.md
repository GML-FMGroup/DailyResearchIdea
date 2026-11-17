# Execution-Grounded Validity Gating for Reusable SQL Block Composition

## Motivation

Memory-based Text-to-SQL systems like MEM-SQL decide whether to reuse a stored SQL block using a static per-layer similarity threshold over requirement-to-block matches, which is a proxy score computed before any evidence about the block's behavior on the target database exists. This proxy can pass semantically incompatible blocks because the threshold only checks whether each requirement has a match above a fixed cutoff, not whether the block, when executed with its original parameter bindings against the target database, still returns the rows the block was meant to return. The same gap recurs across the field: SQLens detects semantic errors only after full SQL generation, so it cannot serve as a pre-composition oracle, and decomposition-based methods such as SGU-SQL recursively build SQL without ever validating intermediate components against execution.

## Key Insight

A stored SQL block's semantic validity for composition is identifiable from its execution result against the target database under its original parameter bindings, because a block whose isolated execution matches a cached, schema-anchored result signature is guaranteed to preserve its intended denotation when composed, whereas a similarity threshold only preserves lexical or structural overlap.

## Method

**What it is.** We introduce EXV-Gate (Execution-Validity Gate), a pre-composition oracle that sits between MEM-SQL's layer-threshold retrieval and the generator's composition step. Its inputs are a retrieved candidate block `b` with its stored parameter bindings `θ_b`, a cached semantic signature `σ_b = (result_schema(b, θ_b), result_hash(b, θ_b), row_count_bucket(b, θ_b))`, and the target database `D`; its output is a binary `compose`/`reject` decision plus, when rejected, a structured rejection record that the generator uses to fall back to schema linking.

**Load-bearing assumption (stated explicitly).** EXV-Gate assumes that a stored SQL block's cached result signature, computed once against its source database under original bindings, remains a valid proxy for the block's denotation on the target database at composition time, so that an exact signature match guarantees the block preserves its intended contribution when composed. This is a database-state stability assumption, not a semantic-validity assumption. It fails when benign divergence (row inserts/deletes, schema migration, differing data distributions between source and target) produces a false reject, or when a block whose semantics are genuinely incompatible but whose result happens to collide (same projected schema, same row-count bucket, hash collision on a small result) produces a false accept. We make this assumption explicit and calibrate it empirically (see experiment).

**How it works.** The gate is a deterministic two-stage rule; the threshold stage is inherited from MEM-SQL, and the execution stage is new.

```pseudocode
EXV-Gate(candidate_block b, bindings θ_b, target_db D, cached_signature σ_b):
  # Stage 1: inherited MEM-SQL layer-threshold match (unchanged)
  if not mem_sql_layer_threshold_pass(b, θ_b):
      return REJECT(reason="threshold")

  # Stage 2: execution-validity oracle (new)
  try:
      R = execute(b, θ_b, D)                 # isolated execution, original bindings
  except ExecutionError as e:
      return REJECT(reason="exec_error", detail=e)

  σ_now = (schema(R), hash_rows(R), bucket(row_count(R)))
  if σ_now == σ_b:                            # exact schema-anchored match
      return COMPOSE(b, θ_b)
  else:
      return REJECT(reason="signature_mismatch",
                    diff=diff(σ_now, σ_b))
```

A rejection at Stage 2 is not a dead end: the generator receives the `diff` record (which column, which hash bucket, which row-count bucket diverged) and is instructed to invoke MEM-SQL's On-demand Schema Linking rather than compose the block. The signature is cached at block-storage time by executing the block once against its source database, so no extra execution is needed at retrieval time beyond the one validation query per candidate.

**Why this design.** We chose isolated execution with original bindings over re-executing with the new request's bindings because the new bindings are not yet known at gate time (they depend on the composition we are gating), accepting that the gate validates block denotation rather than block-parameter compatibility; parameter compatibility is handled downstream by the generator. We chose an exact schema-anchored signature match over a fuzzy or learned similarity score because a fuzzy score reintroduces the same proxy problem that MEM-SQL's threshold already has, accepting that the gate will reject blocks whose result set has legitimately changed due to benign database updates — a false-reject cost we mitigate by bucketing row counts and hashing only the projected columns the block declares, not the full table. We chose to cache signatures at storage time rather than compute them lazily at retrieval because lazy computation would require executing every candidate block on every request, accepting a storage cost of one signature per block in exchange for a single validation query per retrieved candidate. A domain expert would not describe EXV-Gate as a variant of MEM-SQL's threshold or of SQLens: MEM-SQL never executes a block before composition, and SQLens executes only the fully generated query after generation, so neither has a pre-composition execution oracle.

**Why it measures what we claim.** The computational quantity `σ_now == σ_b` measures the motivation-level concept *pre-composition semantic validity of the block* because the named assumption is that a block's denotation on the target database under its original bindings is a sufficient statistic for whether composing it preserves the block's intended contribution to the final query; this assumption fails when the block's result changes due to a schema migration or data drift on the target database that does not affect the block's intended semantics, in which case the gate reflects *database-state stability* rather than *block semantic validity*. The quantity `bucket(row_count(R))` measures *denotational stability* rather than *exact row identity* because the named assumption is that row-count bucket membership is invariant to benign row insertions/deletions that do not change the block's meaning; this assumption fails when a block's semantics depend on exact cardinality (e.g., a `LIMIT`-sensitive block), in which case the bucket reflects *approximate stability* rather than *exact denotation*. The quantity `diff(σ_now, σ_b)` measures *actionable rejection signal* because the named assumption is that schema/hash/bucket divergence localizes the semantic mismatch to a specific clause of the block; this assumption fails when divergence is caused by an upstream table change unrelated to the block's own clauses, in which case the diff reflects *schema drift location* rather than *block clause error*.

## Contribution

["EXV-Gate, a pre-composition execution-validity oracle that augments MEM-SQL's static layer-threshold retrieval with an isolated-execution signature check, turning block reuse from a proxy-scored retrieval decision into a two-stage rule where execution, not similarity, is the final validity criterion.", 'A schema-anchored semantic signature (result schema + row hash + row-count bucket) cached at block-storage time, which makes pre-composition execution validation cost one query per retrieved candidate rather than one query per block per request.', 'A structured rejection-diff record that routes failed candidates to On-demand Schema Linking instead of silent composition, providing a concrete fallback path that prior memory-based Text-to-SQL systems lack.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HARE (history-aware Text-to-SQL benchmark) | Matches MEM-SQL's reuse setting with verified query pool |
| Primary metric | Mean execution accuracy | Directly measures whether composed SQL returns correct results |
| Baseline | MEM-SQL (full) | Tests whether gate adds value over threshold-only retrieval |
| Baseline | SQLens | Tests post-generation correction vs pre-composition gating |
| Baseline | Direct Generation without history | Lower bound; isolates value of history reuse |
| Ablation-of-ours | EXV-Gate w/o Stage-2 execution oracle | Isolates contribution of the new execution-validity stage |
| Ablation-of-ours | EXV-Gate with bucketed hashing (row_count_bucket) vs. exact row-count equality | Measures false-reject rate of exact hashing vs. bucketed hashing |

### Why this setup validates the claim

The central claim is that pre-composition execution-validity checking catches semantically stale or mismatched reusable blocks before they are composed, and that this produces a measurable execution-accuracy gain concentrated on requests where history reuse is attempted. HARE is the right dataset because it is the only benchmark in the evidence pool that explicitly annotates which components a request preserves and which it edits, so we can partition results by reuse type and test whether the gain is concentrated exactly where the gate is supposed to act. MEM-SQL full is the correct primary baseline because it shares the retrieval substrate; any gain over it must come from the gate, not from the memory layers. SQLens tests the orthogonal sub-claim that catching the error before composition is better than correcting a fully generated query after the fact — if SQLens matches us, the pre-composition placement is not doing work. Direct Generation without history anchors the lower bound and confirms the reuse machinery is active. The ablation (removing Stage 2) is diagnostic: it must reproduce MEM-SQL's behavior, so any difference between it and full EXV-Gate is attributable solely to the execution oracle. Mean execution accuracy is the right metric because the gate's whole purpose is denotational correctness, not surface-form similarity to a gold SQL string; a block that is syntactically composed but denotationally wrong is exactly the failure the gate claims to prevent. To ground the load-bearing assumption, we also measure the false-reject rate of exact hashing versus bucketed hashing on HARE, and we stratify results by HARE's edit annotations to confirm the gain is concentrated on edit-required requests.

### Expected outcome and causal chain

**vs. MEM-SQL (full)** — On a request that reuses a stored block whose underlying table gained a column or whose rows shifted, MEM-SQL's layer-threshold match passes because the block's textual/structural features still align, and the generator composes the stale block, yielding a query that executes but returns a wrong projection or wrong cardinality. EXV-Gate executes the block under its original bindings, compares the cached schema-anchored signature, and rejects on schema or bucket divergence, routing the generator to On-demand Schema Linking. We expect a noticeable accuracy gap concentrated on requests annotated as requiring edits to reused components, with near-parity on whole-query adaptations where the block is genuinely unchanged. We will report the false-reject rate of exact hashing versus bucketed hashing to quantify the brittleness of the signature assumption.

**vs. SQLens** — On a request where a reused block's semantics silently drift, SQLens only sees the fully generated query and must detect the error post hoc from execution signals and LLM self-evaluation, which is a weaker signal than a direct signature comparison and can miss errors that are syntactically valid and locally plausible. EXV-Gate intercepts the block before composition, so the error never enters the candidate query. We expect our gain over SQLens to be largest on multi-block requests where post-hoc localization is hardest, and to shrink toward parity on single-block requests where SQLens's whole-query check is nearly as informative.

**vs. Direct Generation without history** — On any request that would benefit from reuse, Direct Generation re-derives the SQL from scratch and pays the full generation cost and error rate, with no memory to consult. Our method inherits MEM-SQL's reuse and adds the gate, so it should dominate on reuse-eligible requests while matching Direct Generation on requests with no usable history. We expect a large mean gap on the reuse-annotated subset and a small, non-negative gap elsewhere, confirming the gate is a filter on reuse rather than a global rewriter.

### What would falsify this idea

If the accuracy gain over MEM-SQL is uniform across reuse types rather than concentrated on requests annotated as requiring edits to reused components, the gate is not acting through the pre-composition validity mechanism and the central claim is wrong. Equivalently, if removing Stage 2 (the ablation) leaves execution accuracy unchanged, the execution-validity oracle contributes nothing beyond the inherited threshold.

## References

1. MEM-SQL: Memory-Empowered Text-to-SQL
