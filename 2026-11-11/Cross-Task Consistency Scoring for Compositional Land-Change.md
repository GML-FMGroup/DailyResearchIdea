# Cross-Task Consistency Scoring for Compositional Land-Change Reasoning

## Motivation

EarthRecord evaluates ten land-change analysis tasks independently, so an agent can score well on classification, change detection, and temporal reasoning separately while never having to reconcile them into a single coherent account of a site's history. This mirrors a broader structural failure across multimodal agent benchmarks: task-level decomposition (as in CHOICE and EarthRecord) assumes sub-tasks can be scored in isolation, which blocks the field from testing whether evidence composed across sub-tasks is internally consistent. The root cause is that per-task accuracy rewards task-local pattern matching rather than the joint invariant that the same parcel of land must support one consistent land-change narrative across all analyses.

## Key Insight

A single grounded answer that must satisfy cross-task constraints exposes contradictions between an agent's own classification, change-detection, and temporal-reasoning outputs that independent per-task scoring structurally cannot detect, because the constraints are only jointly satisfiable if the agent's evidence composition is coherent rather than task-locally optimized.

## Method

**What it is.** We introduce C3 (Cross-Task Consistency Check), an evaluation protocol that replaces EarthRecord's ten independent per-task scores with a single compositional consistency score. Inputs are the agent's grounded answer to a composed query (a land-change narrative over a site and time window, citing pixel evidence), the EarthRecord withheld references, and a set of cross-task constraints derived from the task taxonomy. Output is a constraint-satisfaction report and a joint score.

**Load-bearing assumption (stated explicitly).** C3's score is a function of the parsed claim graph G_A, so it assumes the LLM-based parser faithfully recovers, for each free-form claim, (i) which parcel the claim is about, (ii) which time interval it refers to, and (iii) which evidence-pixel region grounds it. If the parser mis-binds any of these, a constraint can fire on a parsing artifact that is indistinguishable from a genuine agent contradiction. We therefore make parser grounding uncertainty first-class: the parser emits a per-claim binding (parcel, interval, evidence-region) with a confidence in [0,1], a constraint fires only when both claims it couples are bound with confidence >= 0.8, and claims below that threshold are routed to a separate 'unbound/underspecified' bucket that is excluded from S rather than counted as a violation. We report the parser's own parcel/interval recovery precision and recall on a held-out set of 200 hand-labeled answers with known ground-truth bindings, and report S both as a point estimate and as an interval [S_low, S_high] obtained by re-running constraint evaluation with the unbound bucket counted as all-violated (S_low) and all-satisfied (S_high), so the consistency score is interpretable net of parser error.

**How it works.**

```pseudocode
Input: agent answer A to composed query Q(site, t0, t1)
       EarthRecord references R = {r_class, r_change, r_temporal, ...}
       Task taxonomy T = {classification, change_detection, temporal_reasoning, ...}
       Binding confidence threshold theta = 0.8
Output: joint consistency score S (with interval [S_low, S_high]),
        violated constraint set V, unbound-claim bucket U

1. PARSE A into a structured claim graph G_A:
     nodes = {parcel, land_cover_label, change_event(t, type), evidence_pixel_region}
     edges = {covers, causes, precedes, contradicts}
     each node carries a binding confidence conf in [0,1] for its
     (parcel, interval, evidence-region) triple
   (LLM-based parser, fixed prompt, temperature 0; parser validated on 200
    hand-labeled answers; parser precision/recall on parcel and interval
    binding reported separately)

2. For each ordered pair (tau_i, tau_j) in T x T, instantiate constraints C_ij from a
   pre-registered constraint library (e.g., classification(t1) != classification(t0)
   iff change_event exists in (t0,t1]; change_event type must be consistent with the
   land-cover transition; temporal ordering of events must be monotone in t).
   Library size: 34 constraints over 10 tasks (reported in appendix).

3. For each constraint c in union C_ij:
     let the two claims c couples be a, b
     if min(conf_a, conf_b) < theta:
         route c to unbound bucket U (excluded from S)
     else:
         evaluate c against G_A and against R
         if c violated in G_A but satisfied in R: record as agent-internal contradiction (weight 1.0)
         if c violated in both: record as reference disagreement (weight 0.5)
         if c satisfied in G_A but violated in R: record as reference disagreement (weight 0.5)

4. Compute S = 1 - (sum of weighted violations) / (number of bound constraints)
   Compute S_low by treating all U as violations; S_high by treating all U as satisfied.
   Report V = violated constraints with the claim-graph path that witnesses each,
   and U = unbound constraints with the low-confidence binding that caused exclusion.
```

**Why this design.** We chose a pre-registered constraint library over a learned consistency scorer because the constraints must be auditable and fixed before seeing agent outputs, accepting the cost that the library cannot cover every conceivable cross-task relation and will under-detect exotic contradictions. We chose a structured claim graph parsed from the free-form answer over direct string matching against references because compositionality requires knowing which parcel and which time interval a claim is about, accepting the cost that parser errors propagate into constraint evaluation (mitigated by a fixed-temperature parser validated on hand-labeled answers, a confidence threshold theta=0.8 that routes low-confidence bindings to an excluded bucket, and a reported parser precision/recall so S can be read net of parser error). We chose to downweight reference disagreements relative to agent-internal contradictions because EarthRecord's references are themselves derived from global products with known disagreements, accepting the cost that a genuinely wrong agent answer that happens to agree with a noisy reference is under-penalized. This differs from the per-task scoring of EarthRecord and from CHOICE's decomposition assumption: neither prior method instantiates constraints that couple outputs across tasks, so neither can witness an internal contradiction — a domain expert would not describe C3 as a variant of per-task scoring because the scored object is a constraint set over a claim graph, not a set of independent answers. It is also not retrieval or a learned router: there is no library lookup of nearest answers and no controller choosing among modules.

**Why it measures what we claim.** The weighted violation count measures compositional consistency because the constraint library encodes the logical implications that must hold if the agent's classification, change-detection, and temporal-reasoning claims all refer to the same underlying land-change process; this equivalence assumes the parser faithfully recovers the parcel and time interval each claim is about and that the constraint library is complete for the relations that matter, and it fails when the agent's answer is too vague for the parser to ground a claim (in which case the metric reflects answer underspecification rather than genuine contradiction, which is why such claims are routed to U and reported as an interval [S_low, S_high] rather than silently counted) or when agents produce contradictions along relations absent from the 34-constraint library (in which case the metric reflects only the tested subset). The agent-internal-contradiction versus reference-disagreement split measures whether inconsistency originates in the agent's evidence composition rather than in noisy references, because the reference is held fixed across the pair and only the agent's claim graph varies; this assumes EarthRecord's references are accurate enough that a constraint satisfied in R but violated in G_A indicates an agent error, and it fails when the reference itself is wrong, in which case the metric reflects reference noise. The constraint coverage fraction measures the breadth of compositional testing because each constraint instantiates a specific cross-task implication from the taxonomy; this assumes the pre-registered library is representative of the cross-task relations that matter, and it fails when agents produce contradictions along relations absent from the library, in which case the metric reflects only the tested subset.

## Contribution

["C3, a cross-task consistency scoring protocol that parses a single grounded land-change answer into a structured claim graph and scores it against a pre-registered library of cross-task logical constraints, replacing EarthRecord's independent per-task accuracy with a joint invariant.", 'A constraint taxonomy and claim-graph schema for land-change reasoning that separates agent-internal contradictions from reference disagreements, providing a diagnostic that per-task scoring cannot produce.', 'A finding about which cross-task constraint families multimodal agents violate most, derived from re-scoring EarthRecord agent outputs under C3.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | EarthRecord 238-question, 56-site benchmark | Withheld references enable constraint checking |
| Primary metric | C3 joint consistency score (reported with [S_low, S_high] interval) | Couples cross-task claims into one auditable number |
| Baseline | EarthRecord per-task scoring | Tests whether coupling beats independent scores |
| Baseline | CHOICE-style decomposition agent | Tests whether decomposition hides contradictions |
| Baseline | Leading multimodal agent, unmodified | Tests whether consistency is already captured |
| Baseline | Rule-based parser + C3 constraints | Tests whether LLM parser is the source of violations |
| Ablation-of-ours | C3 with per-task-only constraints | Isolates value of cross-task coupling |
| Ablation-of-ours | C3 with constraint library reduced by leave-one-out | Tests sensitivity to library completeness |
| Ablation-of-ours | C3 with theta in {0.5, 0.8, 0.95} | Tests sensitivity to binding-confidence threshold |

### Why this setup validates the claim

The central claim is that compositional consistency across tasks is a distinct, measurable property that per-task scoring cannot witness. EarthRecord is the right dataset because its withheld references let us evaluate each constraint against ground truth rather than against the agent's own output, and its ten-task taxonomy supplies the cross-task implications the library instantiates. The per-task baseline tests the sub-claim that independent scoring misses internal contradictions: if C3 and per-task scoring rank agents identically, coupling adds nothing. The CHOICE-style decomposition baseline tests the sub-claim that decomposing a composed query into independent sub-answers conceals contradictions that a claim graph exposes. The unmodified leading agent tests whether frontier models already produce consistent answers without any consistency pressure. The rule-based-parser baseline tests the load-bearing assumption directly: if a deterministic parser that does not use an LLM produces the same violation set, the LLM parser is not manufacturing contradictions; if it produces a different set, the difference bounds parser-induced noise. The per-task-only ablation, which keeps the claim graph and parser but instantiates only within-task constraints, isolates the contribution of cross-task coupling from the contribution of structured parsing. The leave-one-out library ablation and the theta sweep test the completeness and binding-confidence assumptions respectively, and the [S_low, S_high] interval reports S net of the unbound bucket. C3 is the right metric because it is the only one that assigns a violation to a specific claim-graph path, so a predicted effect (contradictions concentrated on composed queries) can be located, not just counted.

### Expected outcome and causal chain

**vs. EarthRecord per-task scoring** — On a composed query where the agent labels a parcel "stable cropland" at t1 while also reporting a change event in the same window, per-task scoring assigns high marks to both answers because each is scored against its own reference in isolation. Our method instead parses both claims into one graph and fires the classification-change constraint, recording an agent-internal contradiction with the witnessing path. We expect a noticeable C3 penalty on composed queries with a large per-task/C3 rank inversion for agents that answer sub-tasks well but compose poorly, and near-parity on single-task queries.

**vs. CHOICE-style decomposition agent** — On a case where a temporal-ordering question and a change-type question are decomposed and answered by separate modules, the decomposition baseline returns a change type inconsistent with the temporal order it separately asserted, because no module sees both claims. Our method couples them through the precedes and causes edges and flags the violation. We expect the decomposition baseline to show a higher agent-internal contradiction rate than the unmodified agent, with the gap concentrated on queries spanning temporal reasoning and change detection.

**vs. Leading multimodal agent, unmodified** — On a case where the agent cites pixel evidence in a region that does not cover the parcel it names, the unmodified agent still scores well per-task because the label matches the reference, even though its grounding is internally inconsistent. Our method's covers edge exposes the mismatch and records it as an agent-internal contradiction. We expect a moderate C3 gap on evidence-grounded queries but parity on queries without pixel citations, confirming the metric is detecting grounding inconsistency rather than general quality.

**vs. Rule-based parser + C3 constraints** — On the same answers, a deterministic parser (regex + gazetteer for parcel names, fixed date grammar for intervals) binds only the claims whose surface form matches its templates and routes the rest to U. We expect the rule-based parser to have lower recall but higher precision on parcel/interval binding than the LLM parser, and we expect the violation sets to agree on the high-confidence subset; disagreement on that subset bounds parser-induced noise, and if the LLM parser's extra violations are concentrated in low-confidence bindings, the theta=0.8 gate is doing its job.

**vs. C3 with per-task-only constraints (ablation)** — This ablation keeps the claim graph and parser but instantiates only within-task constraints (e.g., two classification claims about the same parcel must not contradict). We expect it to recover most of the parser's structured-parsing benefit but to miss the cross-task contradictions that the full library fires, so the full-minus-ablation gap on composed queries isolates the coupling contribution.

**vs. C3 with leave-one-out constraint library (ablation)** — Removing each of the 34 constraints in turn and re-scoring measures how much S depends on any single relation. We expect S to change by less than 0.05 per removal on average, with the largest drops for the classification-change and temporal-monotonicity constraints, which would indicate the score is not dominated by one constraint.

**vs. C3 with theta sweep (ablation)** — Varying the binding-confidence threshold in {0.5, 0.8, 0.95} trades coverage against parser noise. We expect S to be stable in [0.5, 0.8] and to rise at 0.95 as more claims are routed to U, with the [S_low, S_high] interval widening; if S moves monotonically with theta on composed queries, the score is sensitive to parser confidence and the interval must be reported alongside the point estimate.

### What would falsify this idea

If C3's penalty over per-task scoring is uniform across all query types rather than concentrated on composed, cross-task queries, then the metric is not measuring compositional consistency but is simply a noisier rescaling of per-task accuracy. Likewise, if the cross-task ablation matches full C3, the coupling claim is wrong; and if the rule-based parser produces a disjoint violation set on high-confidence bindings, the LLM parser is manufacturing contradictions rather than detecting them.

## References

1. EarthRecord: From Pixels to Land Change with Multimodal Agents over Four Decades of Satellite Imagery
