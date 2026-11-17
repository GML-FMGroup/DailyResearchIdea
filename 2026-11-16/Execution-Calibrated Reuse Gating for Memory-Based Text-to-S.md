# Execution-Calibrated Reuse Gating for Memory-Based Text-to-SQL

## Motivation

Memory-augmented text-to-SQL systems such as MEM-SQL gate reuse of historical SQL templates with a per-layer match-score threshold, but that threshold is a heuristic proxy that is never validated against whether the reused component actually executes correctly on the current database. The same uncalibrated-proxy problem recurs in Alpha-SQL, whose self-supervised reward scores candidate queries without grounding the score in execution outcomes. Because neither signal is mapped to an empirical probability of execution-grounded correctness, a high match score can admit a template that is syntactically reusable but semantically wrong for the target database, and the field lacks a mechanism that converts reuse evidence into a calibrated correctness probability before composition.

## Key Insight

A per-layer match score becomes a valid reuse decision only after it is monotonically recalibrated against held-out execution outcomes, so that the score's numeric value equals the empirical probability that the reused template executes correctly on the target database.

## Method

## Execution-Calibrated Reuse Gating (ECRG)

**(A) What it is.** ECRG is a calibration layer that sits between MEM-SQL's per-layer match scores and its reuse decision. Its input is a tuple (layer match score s, layer identity, request features) for each memory layer; its output is a calibrated probability p̂ that reusing that layer's evidence yields a SQL query that executes correctly on the target database, and a binary reuse decision derived from p̂ rather than from a fixed threshold. The load-bearing assumption, stated plainly: **ECRG assumes that within a layer, the match score s is a sufficient statistic for execution correctness — i.e., no residual dependence on request features beyond s — so that a monotone map fitted on held-out (s, y) pairs yields a calibrated probability that transfers to unseen requests and databases.** This assumption fails when a requirement is matched lexically but not semantically (e.g., a column-name collision), in which case equal s values yield different correctness outcomes and isotonic regression's monotonicity constraint is a misspecification rather than a calibration. ECRG therefore augments s with execution-grounded signals available at inference and fits the calibrator on the augmented feature vector, and it constructs the calibration set cross-database so that p̂ is a transfer estimate rather than an in-pool frequency.

**(B) How it works.**

```pseudocode
# Phase 0: build cross-database calibration set (offline, held-out)
# Leave-one-dataset-out over HARE's four datasets: for each held-out dataset D_hold,
# calibration pairs come from the other three datasets; test requests come from D_hold.
for each (request r, layer L) in held_out_calibration_set:
    s = MEM_SQL.match_score(r, L)              # raw per-layer score
    x_exec = execution_features(r, L, db)      # see feature list below
    sql = MEM_SQL.generate(r, use_layer=L)     # generate with only L's evidence
    y = execute_and_compare(sql, gold_sql, db) # 1 if execution matches gold, else 0
    append (s, x_exec, L, y) to calibration_pairs

# Execution-grounded features x_exec (all available at inference, no gold SQL):
#   f1: dry-run/EXPLAIN validity of the candidate SQL (binary)
#   f2: schema-constraint check pass rate (fraction of referenced columns/tables that exist with compatible types)
#   f3: column-type compatibility score between matched requirements and schema columns (fraction compatible)
#   f4: join-cardinality estimate from EXPLAIN (log1p of estimated rows, normalized)
#   f5: schema-drift distance between source DB of the template and target DB (Jaccard distance over column sets of matched tables)
#   f6: generator's own execution-verified candidate flag (binary; 1 if the generator produced a candidate that passed dry-run)

# Phase 1: fit per-layer isotonic calibrator on augmented feature vector
for each layer L in {whole_query_template, reusable_block, concept_schema_map}:
    # feature vector z = [s, f1, f2, f3, f4, f5, f6]
    # fit a monotone model f_L : z -> p̂ using isotonic regression on the projection
    #   u = w_L · z  (w_L fitted by logistic regression on calibration pairs, then u is the 1-D score)
    #   fit isotonic regression f_L : u -> p̂  on {(u_i, y_i) : pairs with layer L}
    # isotonic regression is monotone non-decreasing by construction;
    # ties broken by pooling adjacent violators (PAV)
    # fallback: if layer L has fewer than N_min = 200 calibration pairs, use Platt scaling
    #   (logistic sigmoid on u) instead of isotonic, to avoid PAV overfitting.

# Phase 2: inference-time gated reuse
for each request r:
    for each layer L (in order: whole-query, block, concept-map):
        s = MEM_SQL.match_score(r, L)
        x_exec = execution_features(r, L, db)
        u = w_L · [s, x_exec]
        p̂ = f_L(u)                              # calibrated probability
        if p̂ >= tau_L:                          # tau_L = per-layer operating point
            admit L's evidence into the generator context
        else:
            skip L
    if no layer admitted OR generator flags insufficient evidence:
        invoke On-demand Schema Linking (MEM-SQL OSL)
    generate SQL from admitted evidence + OSL output
```

Hyperparameters: `tau_L` is set per layer as the smallest calibrated probability at which the empirical precision of reuse on the calibration set exceeds 0.9; the isotonic fit uses PAV with no smoothing; `N_min = 200` calibration pairs per layer triggers the Platt-scaling fallback; the logistic weights `w_L` are fitted with L2 regularization (C=1.0) on the same calibration pairs. The calibration set is a held-out 20% split of HARE requests, stratified by layer, constructed leave-one-dataset-out over HARE's four datasets so that every test request comes from a dataset unseen during calibration.

**(C) Why this design.** We chose per-layer isotonic regression over a single global calibrator because the three memory layers have structurally different score distributions (whole-query templates concentrate near 1.0, concept-schema maps spread broadly), so a shared mapping would conflate layers and violate the monotonicity assumption; the cost is that each layer needs enough calibration pairs, which limits applicability to layers with sparse history. We chose isotonic over Platt scaling because isotonic makes no parametric assumption about the score-correctness relationship and directly enforces monotonicity, accepting the cost of higher variance on small calibration sets; the Platt fallback below `N_min = 200` pairs bounds that variance. We chose to gate on the calibrated probability rather than MEM-SQL's raw threshold because the raw threshold is not comparable across layers or requests, accepting the cost of an offline calibration pass. We augment the scalar s with execution-grounded features because the paper's own column-collision counterexample shows s is not a sufficient statistic for correctness; the augmented vector restores the monotonicity assumption by conditioning on the execution signals that distinguish high-score-wrong from high-score-right cases. A domain expert would not describe ECRG as a variant of MEM-SQL or of standard retrieval gating: MEM-SQL's threshold is a fixed hyperparameter on an uncalibrated score, whereas ECRG replaces that hyperparameter with a learned monotone map whose output is an empirical execution-correctness probability, and the decision rule is a per-layer operating point on that probability.

**(D) Why it measures what we claim.** The calibrated probability p̂ = f_L(u) measures the empirical probability that reusing layer L yields an execution-correct SQL query because isotonic regression fitted on held-out (augmented-score, execution-outcome) pairs produces a monotone map whose output equals the observed execution-correctness frequency at each score level; this assumption fails when the held-out calibration distribution drifts from the test distribution (e.g., a new database schema with different join complexity), in which case p̂ reflects the calibration-set correctness frequency rather than the target-database correctness probability — the leave-one-dataset-out construction makes this a measured transfer estimate rather than an in-pool frequency. The per-layer operating point tau_L measures the reuse decision's precision because it is chosen as the smallest calibrated probability at which empirical precision exceeds 0.9 on the calibration set; this assumption fails when the calibration set is too small to estimate precision reliably (fewer than `N_min = 200` pairs), in which case tau_L reflects noise in the precision estimate rather than a stable operating point and the Platt fallback is used. The augmented score u measures the semantic reusability of layer L's evidence because it combines MEM-SQL's structural match score s with execution-grounded signals (dry-run validity, schema-constraint pass rate, column-type compatibility, join-cardinality estimate, schema-drift distance, generator execution-verified flag); this assumption fails when a requirement is matched lexically but not semantically and none of the execution features detect the mismatch (e.g., a column-name collision where types are compatible and dry-run passes), in which case u reflects surface overlap plus undetected execution plausibility rather than semantic reusability.

## Contribution

["A per-layer isotonic calibration procedure that maps MEM-SQL's raw match scores to empirical execution-correctness probabilities using held-out (match-score, execution-outcome) pairs, replacing the fixed reuse threshold with a calibrated decision rule.", 'A design principle for memory-augmented text-to-SQL: reuse gating should operate on a calibrated probability of execution-grounded correctness, with per-layer operating points set by a precision target, rather than on a single uncalibrated score threshold.', 'An analysis of when the calibration assumption (held-out distribution matches test distribution) is binding, with the per-layer precision-target operating point as a concrete mitigation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HARE (MEM-SQL benchmark), leave-one-dataset-out over its four datasets | Same history-aware setting; cross-database calibration tests transfer |
| Primary metric | Mean execution accuracy | Directly measures execution-correct reuse outcome |
| Secondary metric | Expected Calibration Error (ECE) of p̂ per layer, with reliability diagram | Tests whether p̂ is a calibrated probability |
| Baseline | MEM-SQL (raw threshold gating) | Isolates effect of calibration vs fixed threshold |
| Baseline | Direct Generation without history | Lower-bound reference MEM-SQL already beats |
| Baseline | Global single isotonic calibrator on s only | Tests whether per-layer and execution-conditioned calibration is necessary |
| Baseline | Per-layer isotonic calibrator on s only (no execution features) | Isolates contribution of execution-grounded features |
| Ablation-of-ours | ECRG without per-layer tau (global tau) | Isolates per-layer operating-point contribution |
| Ablation-of-ours | ECRG with Platt scaling for all layers | Tests isotonic vs parametric calibration choice |

### Why this setup validates the claim

The central claim is that replacing MEM-SQL's fixed uncalibrated threshold with a per-layer execution-calibrated probability and a precision-targeted operating point yields more correct reuse decisions. HARE is the right dataset because it is exactly the benchmark where MEM-SQL's threshold operates, so any gain is attributable to the gating change rather than to a different task or history regime. The leave-one-dataset-out construction over HARE's four datasets is essential because it makes the reported p̂ a transfer estimate: a 20% in-pool split would draw calibration and test pairs from the same 823 verified queries and could not test the load-bearing assumption that calibration transfers across databases. Mean execution accuracy is the correct primary metric because the calibrated probability p̂ is defined against execution correctness; a metric like match-score F1 or template-reuse rate could improve while execution accuracy stagnates, which would not test the claim. ECE and reliability diagrams per layer are the correct secondary metric because the claim is specifically that p̂ equals the empirical execution-correctness probability; execution accuracy alone could improve while p̂ remains miscalibrated. MEM-SQL with its raw threshold is the decisive baseline: it holds the memory layers, generator, and schema-linking fallback fixed, so the only difference is calibrated gating versus thresholded raw score. The per-layer-isotonic-on-s-only baseline isolates the contribution of the execution-grounded features: if ECRG beats it, the augmented feature vector is doing work beyond isotonic regression on s. The global-calibrator baseline and the global-tau ablation test the two design sub-claims separately: per-layer calibration is needed because score distributions differ across layers, and per-layer operating points are needed because the precision target is not uniform. If ECRG beats MEM-SQL only when per-layer calibration, execution features, and per-layer tau are all present, the mechanism story is supported; if the ablation recovers the full gain, the per-layer claim is falsified.

### Expected outcome and causal chain

**vs. MEM-SQL (raw threshold gating)** — On a request where the whole-query template layer scores 0.72 but the reusable-block layer scores 0.68, MEM-SQL admits the template because its fixed threshold sits below both scores, yet the template's surface match hides a column-name collision that makes the reused join wrong. Our method instead maps each augmented score through its own isotonic calibrator, and because the execution features (schema-drift distance f5 and column-type compatibility f3) flag the collision, the calibrated template probability falls below tau while the block probability clears it. We expect a noticeable execution-accuracy gap concentrated on requests where layer scores are close together, and near-parity where one layer scores very high.

**vs. Direct Generation without history** — On a request that reuses a prior join across two tables, Direct Generation has no memory of the earlier query and must reconstruct the join from schema linking alone, so it produces a plausible but mis-joined SQL that fails execution. Our method admits the reusable-block evidence when its calibrated probability clears tau, giving the generator the verified join, and falls back to schema linking only when no layer qualifies. We expect a large gap over Direct Generation on history-reusable requests and small differences on requests with no reusable history, mirroring MEM-SQL's reported 37.1% to 50.0% trajectory but with a further gain on the close-score subset.

**vs. Global single isotonic calibrator on s only** — On a request where the concept-schema-map layer scores 0.55, a single global calibrator fitted across all layers pools the high template scores with the broad concept-map scores, so it maps 0.55 to an inflated probability and admits a concept map that is only lexically matched. Our method instead uses the concept-map calibrator, whose monotone fit on that layer's own augmented pairs places 0.55 below tau, so the map is skipped and schema linking supplies correct evidence. We expect the global calibrator to trail on requests dominated by concept-map matches while matching ECRG on template-dominated requests, revealing the layer-conflation failure.

**vs. Per-layer isotonic calibrator on s only (no execution features)** — On a request where a column-name collision yields high s but wrong semantics, the s-only calibrator has no signal to separate this case from a genuinely correct high-s match, so PAV pools the violators into a flat step and admits the wrong template. Our method's execution features (f3 column-type compatibility, f5 schema-drift distance) separate the collision case, so the calibrated probability falls below tau. We expect ECRG to beat the s-only calibrator specifically on the column-collision subset, and to match it elsewhere, which is the direct test of the sufficient-statistic assumption.

### What would falsify this idea

If ECRG's gain over MEM-SQL is uniform across all request subsets rather than concentrated on requests where layer scores are close or where layers disagree, the calibration story is wrong and the gain is a generic artifact. If the global-tau ablation recovers the full gain of ECRG, then per-layer operating points are unnecessary and the central design claim collapses. If ECRG matches the per-layer-isotonic-on-s-only baseline on the column-collision subset, then the execution-grounded features add nothing and the sufficient-statistic assumption is not repaired.

## References

1. MEM-SQL: Memory-Empowered Text-to-SQL
