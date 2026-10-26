# Lineage-Constrained Acceptance for Composable Agent Recipe Evolution

## Motivation

Composable agent-evolution frameworks such as RSIHub decouple evolution policy from execution and lineage, but they leave acceptance as an unconstrained per-recipe decision: each mutation is judged only against the current candidate, so a composition of individually accepted mutations can silently break monotonicity relative to the recorded evaluation history. This is the recurring structural gap across evolution trees (RSIHub's recipes, Darwin Gödel Machine's archive of self-modifications, ACE's incremental playbook curation): lineage is a passive log rather than an active constraint, so composability is orthogonal to any improvement guarantee. The root cause is that acceptance is a function of the candidate alone, never of the sequence that produced it.

## Key Insight

If acceptance is defined as a partial order over candidates induced by the recorded evaluation history — requiring each accepted mutation to dominate every ancestor it composes with on a shared evaluation surface — then monotone consistency of the accepted sequence becomes a structural invariant of the recipe layer rather than an emergent hope, because any non-dominating composition is rejected by construction before it can enter the lineage.

## Method

**What it is.** We introduce LCAP (Lineage-Constrained Acceptance Protocol), an acceptance operator for RSIHub-style recipe mutation that takes as input a candidate recipe `c`, its parent lineage `L = [(r_0, E_0), ..., (r_k, E_k)]` (recipes and their recorded evaluation vectors), and a candidate evaluation vector `E_c`, and outputs an accept/reject decision together with a dominance witness. It replaces the per-recipe acceptance rule with a lineage-indexed partial order.

**Load-bearing assumption (stated explicitly).** LCAP assumes that the evaluation vector `E_c` is a sufficient statistic for a recipe's behavior on each dimension, so that componentwise comparison of recorded evaluation vectors across different candidates and different points in the lineage is a valid, stationary measure of whether a mutation regresses relative to its ancestors. This assumption fails when the evaluator is non-stationary across candidates (e.g., task difficulty drifts between evaluations), in which case the test reflects evaluator drift rather than recipe quality. To make the assumption verifiable rather than assumed, LCAP replaces raw recorded vectors with a calibrated, paired comparison: each mutable-resource ancestor and the candidate are re-evaluated on the same fixed, seeded task sample within the same evaluation batch, and dominance is defined on the paired per-task difference with a confidence bound rather than a fixed slack. The slack `alpha` is derived from measured variance (see step 2) rather than set to a default constant.

**How it works.**

```pseudocode
# Inputs: candidate c, lineage L (recipes + eval vectors), candidate eval E_c
# Hyperparameters: epsilon (tie tolerance, default 0.0), alpha (dominance slack, DERIVED per dimension from calibration, see step 2b)
# Calibration: fixed seeded task sample S (|S| = 200 tasks per benchmark), bootstrap resamples B = 1000, confidence level 1 - delta = 0.95

1. COMPOSE-ANCESTRY(c, L):
     A <- { (r_i, E_i) in L : r_i is a mutable-resource ancestor of c }
     # only ancestors whose mutable surface c actually touches; others are irrelevant
     return A

2. DOMINANCE-TEST(E_c, A, alpha):
     # 2a. PAIRED RE-EVALUATION: for each (r_i, E_i) in A, re-evaluate r_i and c on the same
     #     fixed seeded task sample S within the same evaluation batch, yielding paired per-task
     #     outcomes (y_i[s], y_c[s]) for s in S.
     # 2b. CALIBRATE alpha: for each dimension d, compute the paired per-task difference
     #     delta_d[s] = y_c[s,d] - y_i[s,d]; alpha_d = the (1 - delta) quantile of the bootstrap
     #     distribution of mean(delta_d) under the null of no regression (B = 1000 resamples,
     #     delta = 0.05). This replaces the fixed alpha = 0.05 default.
     for each (r_i, E_i) in A:
         # componentwise: c must not regress on any evaluation dimension beyond the calibrated bound
         if exists dim d with mean_s(delta_d[s]) < -alpha_d:
             return REJECT with witness (i, d, mean_s(delta_d[s]))
     # and c must strictly improve on at least the most recent ancestor on >=1 dim
     if E_c[d] <= E_{k}[d] + epsilon for all d:
         return REJECT with witness (k, 'no-strict-gain')
     return ACCEPT

3. LINEAGE-UPDATE(c, E_c, A, decision):
     if ACCEPT: append (c, E_c) to L with parent pointers to A
     else:      append (c, E_c) to L as a REJECTED node (retained for future witnesses)
     return updated L

4. RECIPE-LAYER-HOOK:
     register DOMINANCE-TEST as the acceptance operator in the RSIHub recipe protocol,
     so every mutation passes through step 2 before execution is committed.
```

The operator is invoked once per candidate mutation; `alpha_d` is derived per dimension from the bootstrap calibration in step 2b (confidence level 1 - delta = 0.95, B = 1000 resamples, fixed seeded task sample |S| = 200 per benchmark), and `epsilon` controls whether a tie counts as improvement (set to 0.0, requiring strict gain). Rejected candidates are retained as lineage nodes so later mutations can be tested against them too, preventing the search from re-proposing dominated recipes.

**Why this design.** We chose a componentwise dominance test over a scalarized score because scalarization hides per-dimension regressions that composition can amplify — a mutation that improves pass@1 while degrading recovery behavior would pass a scalar rule but fail dominance, accepting the cost that some genuinely useful trade-off mutations are rejected. We chose to restrict the ancestry set to mutable-resource ancestors rather than the full lineage because unrelated branches impose constraints that have no compositional meaning, accepting the cost that cross-branch monotonicity is not enforced. We chose to retain rejected candidates as lineage nodes rather than discard them because they serve as future dominance witnesses, accepting the cost of a larger lineage store. We chose paired re-evaluation on a fixed seeded task sample with a bootstrap-derived `alpha_d` rather than raw recorded vectors with a fixed `alpha = 0.05` because a fixed slack is a hand-set noise absorber with no calibration procedure, so the accept/reject boundary would be arbitrary relative to the actual noise scale of the benchmark; the paired comparison makes the dominance relation a statement about recipes rather than about evaluator drift, accepting the cost of re-evaluating each mutable-resource ancestor per candidate. A domain expert would not describe this as a variant of Darwin Gödel Machine's archive or ACE's curation step: DGM archives successful self-modifications and ACE integrates reflections, but neither defines acceptance as a partial order over the composition of mutable resources, so neither can reject a mutation that is individually fine but non-dominating relative to a composed ancestor.

**Why it measures what we claim.** The componentwise dominance test measures *monotone consistency of the accepted sequence* because the assumption that the evaluation vector is a sufficient statistic for a recipe's behavior on each dimension makes `mean_s(delta_d[s]) >= -alpha_d` equivalent to 'c does not regress on dimension d relative to ancestor i'; this assumption fails when the evaluator is non-stationary across candidates (e.g., task difficulty drifts between evaluations), in which case the test reflects evaluator drift rather than recipe quality — the paired re-evaluation on a fixed seeded task sample within the same batch and the bootstrap-derived `alpha_d` are the concrete steps that make this assumption verifiable rather than assumed. The ancestry restriction measures *compositional relevance* because the assumption that a mutation only affects the mutable resources it declares makes non-ancestor evaluations irrelevant to the candidate; this assumption fails when a mutation has side effects on shared resources not declared as mutable, in which case the test under-constrains and a regression can slip through. The strict-gain requirement measures *progress* because the assumption that `epsilon = 0` reflects a noise-free evaluator makes 'no dimension improves' equivalent to 'no progress'; this assumption fails under evaluator noise, in which case a noisy tie is misread as stagnation and a valid mutation is rejected.

## Contribution

(1) LCAP, a lineage-constrained acceptance operator that turns the recorded evaluation history into an active partial-order constraint on the recipe layer, with a concrete dominance-test algorithm and RSIHub protocol hook. (2) A design principle: restricting dominance constraints to mutable-resource ancestors is what makes lineage constraints compositional rather than globally blocking, isolating the constraint to the surface a mutation actually touches. (3) An open lineage-witness mechanism that retains rejected candidates as future dominance witnesses, changing the search dynamics from per-candidate filtering to sequence-level filtering.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Terminal Bench 2 + τ^3 Banking (RSIHub), replayed from logged runs | Compositional tasks where mutable-resource overlaps are frequent |
| Primary metric | Per-dimension regression rate on accepted mutations, computed on paired re-evaluations | Directly detects monotone-consistency claim, not aggregate score |
| Outcome metric | Final task success rate and rollback frequency on held-out tasks | Tests whether the diagnostic predicts downstream monotone consistency |
| Baseline | AHE in RSIHub | Scalar acceptance hides per-dimension regressions |
| Baseline | GEPA in RSIHub | Reflection curation does not enforce compositional order |
| Baseline | Hyperagents in RSIHub | No lineage-indexed dominance across mutable resources |
| Ablation-of-ours | LCAP with full lineage (no ancestry restriction) | Tests whether compositional relevance restriction matters |
| Ablation-of-ours | LCAP with fixed alpha = 0.05 (no calibration) | Tests whether bootstrap-derived alpha_d matters |

### Why this setup validates the claim

The central claim is that a lineage-indexed componentwise dominance operator rejects mutations that are individually acceptable but non-dominating relative to a composed ancestor, and that this yields a monotone-consistent accepted sequence. To falsify or confirm this, we need a setting where mutations compose mutable resources and where per-dimension regressions can be amplified by composition — Terminal Bench 2 and τ^3 Banking inside RSIHub provide exactly that, since recipes mutate shared agent surfaces (prompt scaffolds, tool policies, recovery logic) that later mutations re-touch. The primary metric — per-dimension regression rate on accepted mutations, computed on paired re-evaluations on a fixed seeded task sample (|S| = 200 per benchmark) — is chosen because it is the quantity the dominance operator claims to control; a scalarized score would blur the effect and cannot distinguish monotone consistency from aggregate luck. The outcome metric (final task success and rollback frequency on held-out tasks) is added so the diagnostic is not self-fulfilling: if LCAP reduces the regression-rate diagnostic but does not improve final task success or reduce rollbacks, the mechanism claim is unsupported. Each baseline tests a distinct sub-claim: AHE tests whether scalar acceptance lets per-dimension regressions through; GEPA tests whether reflection curation substitutes for a partial order; Hyperagents tests whether archive-style success tracking alone enforces compositional monotonicity. The full-lineage ablation isolates the ancestry restriction, so a gain that survives full-lineage LCAP but vanishes under ancestry-restricted LCAP would show the restriction is doing the work rather than dominance alone. The fixed-alpha ablation isolates the calibration step, so a gain that survives fixed-alpha LCAP but vanishes under calibrated LCAP would show the paired re-evaluation and bootstrap-derived alpha_d are doing the work rather than the dominance test alone. All methods are evaluated offline on identical candidate streams via a replay harness over logged RSIHub runs, so differences are attributable to the acceptance operator rather than to re-running full evolution. Together these form a falsifiable test: if LCAP's advantage appears only in aggregate score and not in the regression-rate diagnostic, the mechanism claim is unsupported.

### Expected outcome and causal chain

**vs. AHE in RSIHub** — On a case where a mutation improves pass@1 on Terminal Bench 2 while degrading recovery behavior on a shared tool-policy resource, AHE's scalar acceptance rule sees a net gain and accepts, because scalarization collapses the recovery regression into the aggregate. Our method instead runs the componentwise dominance test against the mutable-resource ancestor that owns the tool policy, sees the recovery dimension regress beyond the calibrated alpha_d, and rejects with a witness. We expect a visible gap in per-dimension regression rate on the subset of mutations that touch shared tool policies, with near-parity on single-resource mutations where scalarization and dominance coincide, and a corresponding reduction in rollback frequency on held-out tasks.

**vs. GEPA in RSIHub** — On a case where a mutation is individually sound but non-dominating relative to a composed ancestor two steps back, GEPA's reflection curation integrates the mutation because reflection is local to the recent trajectory and does not encode a partial order over composed resources. Our method composes the ancestry set, detects the non-domination against the earlier ancestor, and rejects, retaining the candidate as a future witness. We expect GEPA to show a higher rate of accepted mutations that later require rollback on τ^3 Banking multi-step tasks, while LCAP shows fewer rollbacks concentrated on deep-composition subsets.

**vs. Hyperagents in RSIHub** — On a case where two branches mutate disjoint resources and a cross-branch mutation later claims a shared resource, Hyperagents' archive of successful self-modifications has no notion of composed ancestry and accepts the cross-branch candidate on its own merits. Our method's ancestry restriction excludes unrelated branches but includes the shared-resource ancestor, so it rejects only when the shared resource actually regresses. We expect Hyperagents to accept a higher fraction of cross-branch mutations that later regress on the shared dimension, with LCAP showing a measurable reduction in that specific cross-branch regression rate and parity elsewhere.

**vs. LCAP with fixed alpha = 0.05 (ablation)** — On a case where evaluator noise on a dimension exceeds 0.05, the fixed-alpha variant misreads a noisy tie as a regression and rejects a valid mutation, or misreads a noisy regression as a tie and accepts a non-dominating one. Our calibrated variant derives alpha_d from the bootstrap distribution on the paired per-task difference, so the accept/reject boundary tracks the actual noise scale. We expect the fixed-alpha variant to show a higher false-rejection rate on high-variance dimensions and a higher false-acceptance rate on low-variance dimensions, with LCAP showing a regression rate closer to the oracle paired comparison.

### What would falsify this idea

If LCAP's gain in per-dimension regression rate is uniform across all mutation subsets rather than concentrated on shared-resource and deep-composition mutations where the dominance test is predicted to bite, then the mechanism is not doing the work and the effect is attributable to something else. Likewise, if the full-lineage ablation matches ancestry-restricted LCAP exactly, the compositional-relevance claim is unsupported; and if the fixed-alpha ablation matches calibrated LCAP exactly, the calibration step is not doing the work and the load-bearing assumption remains unverified.

## References

1. RSIHub: A Composable Framework for Agent-Evolution Methods
2. The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents
3. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
4. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
