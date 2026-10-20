# Entailment-Certified QA Generation via Relation-Graph Consistency for Egocentric World Models

## Motivation

Automated world-record and QA construction pipelines such as HOLIWORLD-Foundry generate records and question-answer pairs without ever checking them against the segmentation and object-relation ground truth they were derived from, so perception errors and hallucinated relations propagate unchecked into the training set; the VISOR benchmark supplies exactly the object-relation annotations that could serve as this check but treats them as static supervision rather than as an invariant the generator must satisfy. The root cause is structural: generation and verification live in separate, unidirectionally-coupled stages, so no signal from the annotation graph can reject an inconsistent generated pair before it contaminates training.

## Key Insight

Because every generated QA pair is a projection of a relation triple that must already exist as an edge in the source segmentation-relation graph, entailment against that graph is a decidable, annotation-grounded certificate rather than a learned judgment, so rejection sampling on graph entailment removes inconsistent pairs without requiring any calibrated model confidence.

## Method

```pseudocode
# Inputs: world records W (from HoliWorld-Foundry), VISOR-style segmentation-relation graph G = (O, R)
#   O = object instances with masks; R = typed relation edges (subject, predicate, object, t_start, t_end)
#   Coverage estimate cov(s,o) = fraction of frames in the annotated interval where any relation edge exists for object pair (s,o)
# Outputs: certified QA set Q_cert; relation-consistency statistics

# EXPLICIT LOAD-BEARING ASSUMPTION (stated in plain language):
#   The VISOR-style relation graph G is treated as a SOUND and COMPLETE encoding of the scene's object-relation facts
#   for the annotated interval. Under this assumption, absence of an edge (s,p,o) with temporal overlap is a valid
#   certificate of non-entailment. This assumption is known to be false in general (egocentric relation annotations are
#   sparse and incomplete, especially for occluded hand-object contact), so Phase 2 uses a THREE-VALUED check that
#   distinguishes CONTRADICTED (an edge exists with conflicting predicate/direction/time, or an explicit negative
#   annotation exists) from UNANNOTATED (no edge and no coverage evidence). Only CONTRADICTED triples are REJECTed;
#   UNANNOTATED triples are routed to the soft path (Phase 3). A per-predicate, per-scene annotation-coverage estimate
#   cov(s,o) is required so that 'absent edge' is never silently treated as 'false relation'.

Phase 1 — Relation-triple parsing of generated QA
for each candidate QA pair (q, a) generated from world record w:
    triple = LLM_PARSE(q, a)            # returns (s, p, o, t) with s,o object mentions, p in predicate vocabulary P, t time interval
    map s, o to object IDs via mask-overlap + track-ID lookup in w   # nearest mask IoU >= 0.5, else UNRESOLVED
    if UNRESOLVED: route to Phase 3 (soft check)

Phase 2 — Three-valued entailment check against relation graph G
for each resolved triple (s, p, o, t):
    if (s, p, o) in R and time-overlap(t, R[s,p,o].interval) > 0:
        ACCEPT into Q_cert
    else if (s, p, o) in R but time-overlap == 0:
        REJECT (temporal inconsistency)   # CONTRADICTED
    else if (s, INVERSE(p), o) in R:
        REJECT (predicate-direction inconsistency)   # CONTRADICTED
    else if explicit_negative_annotation(s, p, o, t) exists in G:
        REJECT (explicit negative)   # CONTRADICTED
    else:
        # UNANNOTATED: no edge, no explicit negative, and coverage may be low
        route to Phase 3 (soft check) with coverage flag cov(s,o) < 0.5

Phase 3 — Soft consistency loss over the relation graph (for unresolved / unannotated triples)
embed each object o as e_o = mask-pooled visual feature from w
score f(s,p,o) = MLP([e_s ; e_o ; onehot(p)])   # 2-layer MLP, hidden=256, GeLU activation
loss L_rel = -sum_{(s,p,o) in R} log sigma(f(s,p,o)) - sum_{(s,p,o) not in R} log(1 - sigma(f(s,p,o)))
# trained on G itself; used only to rank borderline pairs, never to ACCEPT
# NOTE: f(s,p,o) measures relation plausibility because the assumption is that graph edges and non-edges are
#   class-balanced representative samples of true and false relations; this assumption fails under severe predicate-
#   frequency imbalance, in which case the score reflects edge frequency rather than plausibility.

Phase 4 — Rejection sampling with certified set
Q_cert = ACCEPT set from Phase 2
for pairs in Phase 3 borderline set: keep if f(s,p,o) > tau (tau = 0.7) AND human spot-check sample (5%) passes
# Human spot-check replaced by pre-registered, fixed-size stratified annotation protocol:
#   300 pairs stratified by predicate frequency (10 predicates x 30 pairs), with inter-annotator agreement reported.
return Q_cert, L_rel, rejection-rate report
```

(A) **What it is.** EAGLE-QA (Entailment-Against-Graph for Label-verified Egocentric QA) takes world records and the VISOR-style segmentation-relation graph as inputs and returns a certified QA training set in which every pair's relation triple is entailed by the annotation graph or explicitly flagged as soft.

(B) **How it works.** See pseudocode above: Phase 1 parses each generated QA into a typed relation triple and grounds its arguments to tracked object IDs; Phase 2 performs a decidable three-valued graph lookup with temporal and inverse-predicate consistency checks, distinguishing CONTRADICTED from UNANNOTATED; Phase 3 trains a relation-consistency head (2-layer MLP, hidden=256, GeLU) on the graph itself to rank the residual borderline cases; Phase 4 admits only hard-entailed pairs into the certified set and applies a conservative threshold (tau=0.7) plus a pre-registered stratified human spot-check (300 pairs, 10 predicates x 30) to the soft set.

(C) **Why this design.** We chose a decidable graph lookup over a learned entailment classifier because the VISOR relation graph is a finite, annotated structure and membership in it is a fact, not a prediction — accepting the cost that pairs whose arguments cannot be grounded to a tracked mask fall through to the soft path. We chose to parse QA into relation triples rather than compare free-text answers because triple parsing exposes the exact predicate and time interval that the graph can check, accepting the cost of parse errors that we mitigate with the 0.5 IoU grounding threshold. We chose to train the soft consistency head on the graph itself rather than on model-generated pairs because graph-trained supervision is annotation-grounded and cannot inherit generator hallucinations, accepting the cost that the head is only as good as the graph's coverage. Unlike HoliWorld-Foundry's unidirectional generate-then-train pipeline, and unlike retrieval-style nearest-neighbor relation matchers, EAGLE-QA computes a membership certificate whose truth value is independent of the generator, so a domain expert would not describe it as a variant of retrieval or of HoliWorld-Foundry — the generator cannot influence the oracle.

(D) **Why it measures what we claim.** The three-valued graph-membership predicate (ACCEPT if (s,p,o) in R with time-overlap > 0; REJECT if CONTRADICTED; else UNANNOTATED) measures answer entailment because the named assumption is that the VISOR relation annotation is a sound and complete encoding of the object-relation facts in the scene for the annotated interval; this assumption fails when the graph omits a true relation (e.g., a partially occluded hand-object contact), in which case the predicate reflects annotation coverage rather than semantic entailment and the pair is conservatively routed to the soft path rather than rejected. The mask-IoU grounding threshold (0.5) measures object-identity faithfulness because the assumption is that tracked instance masks are the unique referents of object mentions; it fails when two instances overlap heavily, in which case grounding reflects mask ambiguity. The soft score f(s,p,o) measures relation plausibility because the assumption is that graph edges and non-edges are class-balanced representative samples of true and false relations; it fails under severe predicate-frequency imbalance, in which case the score reflects edge frequency rather than plausibility.

## Contribution

['A closed-loop entailment-verification stage, EAGLE-QA, that parses each generated egocentric QA pair into a typed relation triple and certifies it against the VISOR-style segmentation-relation graph via decidable membership, temporal-overlap, and inverse-predicate checks before the pair enters the training set.', 'A relation-consistency head trained directly on the annotation graph that ranks residual borderline pairs, together with a rejection-sampling protocol that separates hard-certified from soft-certified pairs so that no model-confidence signal is used as an acceptance gate.', 'An empirical characterization of how graph-entailment rejection rates vary with predicate type, occlusion level, and temporal interval length in egocentric recordings, yielding a design principle for which relation predicates can be certified automatically.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HoliWorld-Bench held-out recordings | Independent human annotation as external oracle |
| Primary metric | Certified-set precision (human-verified) | Directly measures label correctness of admitted QA |
| Baseline | HoliWorld-Foundry generate-then-train | Tests whether unverified generation admits hallucinated relations |
| Baseline | Retrieval nearest-neighbor relation matcher | Tests whether soft similarity substitutes for graph membership |
| Baseline | LLM-judge entailment classifier | Tests whether learned entailment matches decidable graph lookup |
| Ablation-of-ours | EAGLE-QA without Phase 2 hard check | Isolates contribution of decidable graph membership |
| Ablation-of-ours | EAGLE-QA with binary (closed-world) Phase 2 check | Tests whether three-valued check prevents false rejections |
| Calibration | Sweep mask-IoU threshold (0.3–0.7) and tau (0.5–0.9) | Grounds operating points empirically |
| Human annotation | Pre-registered stratified sample: 300 pairs (10 predicates × 30) | Budgetable, reproducible, reports inter-annotator agreement |

### Why this setup validates the claim

The central claim is that a QA pair is trustworthy exactly when its parsed relation triple is a member of the annotated relation graph, so correctness should track graph membership rather than generator fluency or embedding similarity. HoliWorld-Bench provides an annotation oracle that is independent of both the generator and our graph, so certified-set precision measured against it is a genuine external check rather than a self-consistency score. The HoliWorld-Foundry baseline tests the sub-claim that unverified generation admits hallucinated relations; the retrieval matcher tests the sub-claim that approximate similarity cannot substitute for exact membership; the LLM-judge baseline tests whether a learned entailment model can match the decidable lookup. The ablation removes the hard check while keeping parsing, grounding, and soft ranking, so any precision drop isolates the graph-membership certificate as the causal ingredient. The binary-check ablation tests whether the three-valued check prevents false rejections due to annotation sparsity. Precision on the admitted set is the right metric because the method's whole purpose is to refuse uncertain pairs, so coverage is deliberately sacrificed and precision is the invariant to defend. The calibration sweep ensures the chosen thresholds (IoU=0.5, tau=0.7) are empirically grounded, and the pre-registered human annotation protocol makes the human component budgetable and reproducible.

### Expected outcome and causal chain

**vs. HoliWorld-Foundry generate-then-train** — On a case where the generator invents a plausible but false relation such as "hand holds cup" when the cup is merely adjacent, the baseline admits the pair because its only filter is generation likelihood, which rewards fluency over fact. Our method parses the triple, grounds both arguments to tracked masks, and finds no `(hand, holds, cup)` edge with temporal overlap, so it rejects. We expect a concentrated precision gap on the hallucination-prone subset (occluded and adjacent-object scenes) with near-parity on unambiguous scenes, plus a visible rejection-rate report showing the baseline accepts far more unverifiable pairs.

**vs. Retrieval nearest-neighbor relation matcher** — On a case where the correct predicate is `touching` but the nearest graph edge is `holding` because their embeddings are close, the retrieval baseline accepts the wrong predicate since it ranks by similarity rather than exact typed membership. Our method checks the typed triple directly and rejects on predicate mismatch or inverse-direction inconsistency. We expect the gap to concentrate on near-synonym predicate pairs and on inverse-relation confusions, with parity on predicates that have no close neighbors, so the diagnostic is a predicate-confusion-specific precision advantage.

**vs. LLM-judge entailment classifier** — On a case where the QA answer is long and paraphrastic, the judge baseline drifts, accepting pairs whose surface text seems supported but whose underlying relation and time interval are wrong, because it never grounds arguments to instance IDs. Our method extracts the exact predicate and interval and checks them against the graph, so paraphrastic but relationally false pairs are rejected. We expect the judge to match us on short literal questions and to fall behind on paraphrastic and temporally shifted questions, giving a gap that grows with answer length and temporal-reference complexity.

**vs. EAGLE-QA without Phase 2 hard check** — On a case where the soft head scores a plausible but absent relation above threshold because the graph is sparse for that predicate, the ablated variant admits the pair since it has no membership certificate. Our method still rejects because the graph lookup is decidable regardless of head confidence. We expect the ablation to lose precision specifically on sparsely annotated predicates while matching us on dense ones, confirming the hard check rather than the soft ranker carries the guarantee.

**vs. EAGLE-QA with binary (closed-world) Phase 2 check** — On a case where a true relation is omitted from the graph due to occlusion (e.g., hand-object contact under occlusion), the binary check would REJECT the pair because it treats absence as falsity. Our three-valued check routes it to the soft path, avoiding false rejection. We expect the binary variant to have lower coverage and potentially lower precision on the admitted set if the soft path correctly recovers some true pairs, or higher precision but much lower coverage; the three-valued check should achieve a better precision-coverage trade-off.

**Calibration study** — We sweep mask-IoU threshold from 0.3 to 0.7 and tau from 0.5 to 0.9, reporting certified-set precision and coverage curves. We expect precision to increase with higher IoU and tau, with diminishing returns beyond 0.5 and 0.7 respectively, justifying the chosen operating points.

**Human annotation protocol** — We pre-register a stratified sample of 300 pairs (10 predicates × 30 pairs) for human verification, reporting inter-annotator agreement (Cohen's kappa). This replaces the open-ended 5% spot-check with a fixed-size, budgetable protocol.

### What would falsify this idea

If certified-set precision against the independent human oracle were uniform across the hallucination-prone, predicate-confusion, and sparse-annotation subsets rather than concentrated where the predicted failure modes occur, the graph-membership certificate would not be the causal ingredient and the central claim would be wrong. Equally fatal would be an ablation that matches full EAGLE-QA precision, since that would show the decidable hard check contributes nothing.

## References

1. Holi-World: Building Holistic World Intelligence from Human-Ego Experience
