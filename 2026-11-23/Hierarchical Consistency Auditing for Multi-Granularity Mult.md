# Hierarchical Consistency Auditing for Multi-Granularity Multimodal Identity Matching Benchmarks

## Motivation

Item-independent MCQ aggregation is a structural assumption shared across multimodal benchmark families: PURE scores base-level and variant-level identity-matching items separately, and VideoAutoArena similarly aggregates per-battle judgments independently, so no benchmark enforces the logical dependency that a variant-level match presupposes a base-level match. Because PURE's 1,457 items are scored in isolation, a model can achieve high per-item accuracy while simultaneously predicting variant-match and base-non-match on the same pair — a logically impossible judgment that the benchmark never surfaces. The root cause is that aggregation is defined over item scores, not over the joint distribution of predictions on logically linked items.

## Key Insight

If variant-match logically implies base-match for every identity pair, then the empirical violation rate of that implication under a model's own predictions is a benchmark-level property computable without any new annotation, and it decomposes per-item accuracy into logically coherent and incoherent components.

## Method

**What it is.** We introduce HICAP (Hierarchical Consistency Audit Protocol), a scoring layer that takes as input PURE's per-item MCQ predictions and its pair-to-item linkage map, and outputs for each model (i) standard accuracy, (ii) a pair-level violation rate for the implication variant-match ⇒ base-match, and (iii) a consistency-adjusted accuracy that penalizes incoherent pairs. HICAP is a deterministic audit, not a trained module.

**Load-bearing assumption (stated explicitly).** HICAP's violation indicator, AdjAcc penalty, and the entire 'logical incoherence' interpretation are defined on the premise that PURE's item construction guarantees that every pair labeled variant-match is also labeled base-match by the two domain experts — i.e., the benchmark's own label ontology enforces variant-match ⇒ base-match. This premise is not assumed silently: it is empirically checked on PURE's expert gold labels before any model audit (Phase 0 below), and the audit is reported as a normalized excess over the measured ground-truth violation rate, so the contribution survives even if the nesting assumption is false.

**How it works.**

```pseudocode
INPUT: model M, PURE items I = {i_k}, pair-linkage L mapping pair p -> (b_p, v_p)
       where b_p = base-level item, v_p = variant-level item
OUTPUT: Acc(M), ViolRate(M), AdjAcc(M), per-dimension breakdown,
        GT_ViolRate (benchmark construction-validity ceiling), coverage

# Phase 0: data-validity gate (benchmark property, model-independent)
#   Computed once from PURE's expert gold labels, before any model is audited.
for each pair p in L:
    gold_b = label(gold[b_p])            # in {match, no-match}
    gold_v = label(gold[v_p])
    gt_violation[p] = 1 if (gold_v == match and gold_b == no-match) else 0
GT_ViolRate = sum_p gt_violation[p] / |L|
L_nested = { p in L : gt_violation[p] == 0 }   # verified-nested subset
#   If GT_ViolRate > 0, the nesting premise is false for those pairs;
#   they are excluded from the violation denominator by construction.

# Phase 1: collect predictions
for each item i_k in I:
    yhat[k] = argmax_M(i_k)              # predicted option; options carry a match/no-match label

# Phase 2: pair-level consistency check (only pairs in L_nested where both items answered)
for each pair p in L_nested with both b_p and v_p answered:
    pred_b = label(yhat[b_p])            # in {match, no-match}
    pred_v = label(yhat[v_p])
    violation[p] = 1 if (pred_v == match and pred_b == no-match) else 0

# Phase 3: aggregate (pair-weighted, not item-weighted)
ViolRate(M) = sum_{p in L_nested, answered} violation[p] / |L_nested_answered|
Acc(M)       = mean_k 1[yhat[k] == gold[k]]
AdjAcc(M)    = mean_k 1[yhat[k] == gold[k] and not member_of_violating_pair(k)]

# Phase 3b: per-pair decomposition (attribution of AdjAcc penalty)
#   For each answered pair in L_nested, record the joint outcome:
#     n_base_correct_variant_wrong  = # pairs with pred_b == gold_b and pred_v != gold_v
#     n_base_wrong_variant_correct  = # pairs with pred_b != gold_b and pred_v == gold_v
#     n_both_correct, n_both_wrong
#   AdjAcc's penalty is attributable to the first count (base-correct/variant-wrong),
#   which is the failure mode F under which the coherence equivalence A breaks.

# Phase 4: report alongside PURE's four capability dimensions
return Acc, ViolRate, AdjAcc, GT_ViolRate, coverage, breakdown_by_dimension, per_pair_decomposition
```

Hyperparameters: a single threshold tau = 0.5 on the pair-level violation indicator (no tuning; the indicator is binary by construction). Pairs where one item is unanswered are excluded and their count is reported as coverage; no imputation is performed. The Phase 0 gate uses no threshold: GT_ViolRate is a measured fraction, and L_nested is defined by exact gold-label agreement (gt_violation[p] == 0).

**Why this design.** We chose pair-weighted violation rate over item-weighted violation rate because the logical implication is defined at the pair level, accepting the cost that models with many unanswered variant items get a smaller denominator and noisier estimates (reported as coverage). We chose a deterministic audit over a learned consistency head because the implication is a logical constraint, not a statistical regularity to be estimated, accepting the cost that HICAP cannot discover violations beyond the pre-specified implication set. We chose to report AdjAcc alongside raw Acc rather than replacing Acc, because PURE's diagnostic richness across Perception, Understanding, Reasoning, and Judgment must remain comparable to prior reported numbers, accepting the cost that two numbers must be read together. We chose to derive the linkage map from PURE's existing pair construction rather than re-annotate, because each PURE item is already derived from a real-world matching pair, accepting the cost that pairs whose two items were not both constructed are dropped from the violation denominator. We chose to add the Phase 0 data-validity gate because the nesting premise is asserted, not verified, in PURE's abstract; measuring GT_ViolRate on gold labels makes the audit a property of the benchmark-model pair that survives even if the nesting assumption is false, accepting the cost that the violation denominator shrinks to the verified-nested subset. Critically, a domain expert would not describe HICAP as a variant of a learned consistency head or of ensemble agreement methods, because HICAP adds no parameters and no training signal: it is an audit of an existing benchmark's aggregation function, and its output is a property of the benchmark-model pair, not a new predictor.

**Why it measures what we claim.** The pair-level violation indicator measures logical incoherence of a model's judgment system because the assumption is that variant-match entails base-match in the benchmark's own label ontology — i.e., the PURE item construction guarantees that any pair labeled variant-match is also labeled base-match by the two domain experts; this assumption is checked in Phase 0 (GT_ViolRate) and fails when a pair's base and variant items were constructed from non-nested record pairs, in which case the indicator reflects a benchmark construction artifact rather than model incoherence, and such pairs are excluded from L_nested. The consistency-adjusted accuracy measures the fraction of correct predictions that survive logical scrutiny because the assumption (A) is that a correct prediction inside a violating pair is correct by luck rather than by coherent reasoning; this assumption fails (failure mode F) when a model has a locally correct base prediction but a systematically biased variant prediction, in which case AdjAcc under-credits the base-level capability — the per-pair decomposition (Phase 3b) reports base-correct/variant-wrong and base-wrong/variant-correct counts separately so the penalty is attributable. The violation rate measures the benchmark's item-independence aggregation gap because the assumption is that the only dependency between base and variant items is the hierarchical implication, with no other cross-item correlations; this assumption fails when items within a pair share surface artifacts (e.g., the same product image), in which case the violation rate partly reflects shared-input shortcutting rather than pure logical inconsistency.

## Contribution

["HICAP, a parameter-free audit layer that converts PURE's item-independent MCQ scoring into a pair-constrained judgment system by computing a hierarchical implication violation rate and a consistency-adjusted accuracy, applicable to any benchmark with a pre-specified item-implication graph.", 'A design principle for multimodal benchmark aggregation: logical dependencies between items should be reported as first-class metrics rather than left implicit in item-level accuracy, with the violation rate serving as a diagnostic that separates coherent from incoherent model competence.', "A reproducible protocol and linkage-map specification that lets PURE's four capability dimensions be re-analyzed under hierarchical constraints without re-annotation."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PURE (1,457 MCQ, 14 tasks) | Provides pair-linkage and four dimensions |
| Data-validity gate | Phase 0 GT_ViolRate on PURE gold labels | Measures the benchmark's construction-validity ceiling before any model audit; defines L_nested |
| Primary metric | ViolRate (pair-weighted, over L_nested) | Directly quantifies logical incoherence, normalized against GT_ViolRate |
| Baseline | Raw PURE accuracy | Item-independent aggregation, no pair logic |
| Baseline | Item-weighted violation rate | Ignores pair-level implication structure |
| Baseline | Learned consistency head | Tests whether training beats deterministic audit |
| Ablation-of-ours | HICAP without AdjAcc penalty | Isolates contribution of consistency adjustment |
| Ablation-of-ours | HICAP without Phase 0 gate (full L denominator) | Isolates contribution of the data-validity gate |
| Diagnostic | Per-pair decomposition (base-correct/variant-wrong vs. base-wrong/variant-correct) | Attributes AdjAcc penalty to failure mode F |

### Why this setup validates the claim

The central claim is that PURE's item-independent accuracy hides logical incoherence between hierarchically linked base and variant items, and that a deterministic pair-level audit surfaces this gap. Raw PURE accuracy is the natural baseline because it is exactly the aggregation HICAP audits: if HICAP's violation rate and adjusted accuracy reveal structure that raw accuracy cannot, the claim holds. The Phase 0 gate is the first milestone: it reports GT_ViolRate on PURE's gold labels and the size of L_nested, so the audit is not contingent on an unverified nesting assumption. The item-weighted violation baseline tests the sub-claim that the implication lives at the pair level, not the item level; if item-weighting produces the same diagnostic pattern, the pair-weighted choice is unmotivated. The learned consistency head tests the sub-claim that a parameter-free audit suffices; if a trained head finds substantially more violations, HICAP's deterministic scope is too narrow. The ablation without the Phase 0 gate isolates whether excluding non-nested pairs changes the diagnostic pattern. The ablation without the AdjAcc penalty isolates whether the consistency adjustment adds information beyond the raw violation count. The per-pair decomposition attributes the AdjAcc penalty to failure mode F (base-correct/variant-wrong) rather than to base-wrong/variant-correct. ViolRate is the right primary metric because it is the direct operationalization of the falsifiable prediction: violations should concentrate on the Judgment dimension and on variant-level items, not be uniform across PURE's four capability dimensions. A uniform violation rate would mean HICAP is measuring generic model error, not hierarchical incoherence.

### Expected outcome and causal chain

**vs. Raw PURE accuracy** — On a case where a model scores well overall but answers a variant-level item as match while answering its linked base-level item as no-match, raw accuracy credits both items independently and reports a high aggregate score because it never inspects the pair. Our method instead flags the pair as a violation and removes the surviving correct prediction from AdjAcc because the implication variant-match ⇒ base-match is violated. We expect a noticeable gap between Acc and AdjAcc concentrated on the Judgment dimension and on models in the mid-to-high accuracy band, with near-parity on Perception where items are largely independent. The per-pair decomposition should show the gap is driven by base-correct/variant-wrong pairs, not by base-wrong/variant-correct pairs.

**vs. Item-weighted violation rate** — On a case where a model leaves many variant items unanswered, the item-weighted rate dilutes violations across all answered items and understates the pair-level incoherence because it treats each item as an independent trial. Our method instead computes the rate over answered pairs in L_nested only, so a model that systematically skips hard variant items shows a sharper violation signal on its answered pairs, with coverage reported separately. We expect the two rates to diverge most for models with low variant-item coverage, and to converge for models that answer nearly all pairs.

**vs. Learned consistency head** — On a case where a model's violation arises from a shared product-image shortcut rather than from the hierarchical implication, a learned head trained on pair labels can absorb the shortcut as a statistical regularity and predict violations that are not logical incoherence, because it estimates correlation rather than enforcing a constraint. Our method instead flags only the pre-specified implication on L_nested and reports the residual as a benchmark-construction caveat, so its violation set is a strict subset that is interpretable by construction. We expect the learned head to report a higher violation count with lower human-verified precision, while HICAP's violations align with the expert-annotated nested-pair subset.

**vs. HICAP without Phase 0 gate** — On a case where PURE contains pairs whose gold labels violate the nesting premise (gold_v = match, gold_b = no-match), the ungated audit counts a model's prediction on such a pair as a violation even though the model is faithfully reproducing the benchmark's own labels. Our method instead excludes those pairs from L_nested by construction, so ViolRate is a normalized excess over GT_ViolRate. We expect the ungated ViolRate to exceed the gated ViolRate by approximately GT_ViolRate, with the gap largest on the Judgment dimension where non-nested pairs are expected to concentrate.

**vs. HICAP without AdjAcc penalty** — On a case where a model has a high raw violation count but most violations fall on pairs where the base prediction is also wrong, the penalty-free audit reports the same ViolRate as the full audit but cannot distinguish incoherent from merely erroneous pairs. Our method instead reports AdjAcc and the per-pair decomposition, so the penalty is attributable to base-correct/variant-wrong pairs. We expect the penalty-free ablation to show the same ViolRate but a different per-dimension pattern, with the full audit's AdjAcc gap concentrated where base-correct/variant-wrong counts are highest.

### What would falsify this idea

If HICAP's violation rate is uniform across PURE's four capability dimensions and across models rather than concentrated on Judgment and on variant-level items, the pair-level implication is not capturing hierarchical incoherence and the central claim fails. Equally, if AdjAcc diverges from Acc by the same margin on pairs with and without the predicted nesting structure, or if the per-pair decomposition shows the penalty is driven by base-wrong/variant-correct pairs rather than base-correct/variant-wrong pairs, the adjustment is measuring generic error rather than logical scrutiny.

## References

1. PURE: A Diagnostic Benchmark for Multi-Granularity Multimodal Identity Matching
