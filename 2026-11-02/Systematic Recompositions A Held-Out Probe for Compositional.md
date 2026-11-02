# Systematic Recompositions: A Held-Out Probe for Compositional Spatial Generalization in Frozen Visual Representations

## Motivation

The probing study of spatial state representations in visual foundation models (Idea Card: Probing Spatial State Representations in Visual Foundation Models) measures readout accuracy on individual spatial predicates and matched controls, but it never tests whether the frozen representation supports systematic recomposition of those predicates into novel combinations. This is a structural limitation: because the probe trains and tests on the same set of relation pairs, high accuracy can be achieved by memorizing pairwise conjunctions rather than by composing spatial primitives. The field therefore lacks a probe that can distinguish compositional generalization from relational memorization, and no existing benchmark holds out recompositions at the predicate level while controlling for visual and linguistic confounds.

## Key Insight

A frozen representation supports zero-shot recomposition if and only if the probe's decision surface for a novel predicate combination is determined by the same linear functionals that decode the individual predicates, so a held-out split over systematic recompositions directly tests whether the representation's spatial code is compositional rather than conjunctive.

## Method

**What it is.** We introduce SCRAP (Systematic Compositional Readout of spatial Predicates), a frozen-feature probing framework that trains lightweight linear probes on individual spatial predicates and evaluates them on held-out systematic recompositions of those predicates. Inputs are frozen visual features from a visual foundation model over egocentric image sequences; outputs are per-predicate readout accuracies on seen pairs and on novel recompositions, plus a compositionality gap metric.

**Load-bearing assumption (stated explicitly).** The AND rule is valid as a test of compositional generalization only if the individual linear probes w_v, trained on single-predicate scenes, are unbiased estimators of the same predicate functionals that the AND rule composes. Because the AND rule is evaluated on joint scenes containing object triples (A,B,C) that instantiate both v1 and v2, while each w_v is trained on scenes where only that predicate is instantiated, the joint-scene feature distribution differs from the single-predicate training distribution (different object configurations, occlusion, co-occurrence statistics). This is a covariate-shift failure of composing independently trained linear classifiers. To make the assumption testable rather than assumed, we add a disjoint calibration split and a shared-basis variant, specified below, so that both the AND arm and the direct conjunction probe see identical joint-distribution exposure and the gap isolates separability rather than shift.

**How it works.**

```pseudocode
# Phase 1: Predicate vocabulary and scene generation
V = {left-of, right-of, in-front-of, behind, above, below, near, far}
Generate scenes S with object triples (A, B, C) such that each pair (A,B) instantiates exactly one predicate v in V.
# Pre-registered generator: 3D scene sampler with fixed seed; each scene rendered at 224x224; object identities drawn from a fixed pool of 50 shapes; spatial layout sampled uniformly over the 8-predicate configuration space.
# Pre-registered pair sampling: for each ordered pair (v1, v2) in V x V, guarantee at least N_MIN = 200 TRAIN scenes containing both v1 and v2 (stratified by conjunction frequency into bins: 200-499, 500-999, 1000+).

# Phase 2: Held-out recomposition split
For each ordered pair (v1, v2) in V x V, define a recomposition task: 'Is A v1 B and B v2 C?'
Randomly assign 20% of ordered pairs to HELD-OUT; ensure no predicate appears in HELD-OUT with the same partner predicate as in TRAIN.
# Disjoint calibration split: from the TRAIN scenes, hold out CAL = 512 joint scenes that do NOT contain any HELD-OUT pair. CAL is used only to recalibrate probe thresholds and (optionally) directions, never to fit w_v or w_{v1,v2}.

# Phase 3: Frozen feature extraction
For each scene s, extract frozen features f(s) from layer L of model M (layer chosen by validation on TRAIN).
# M in {CLIP ViT-L/14, DINOv2 ViT-L/14, VideoMAE ViT-L/16}; L chosen from {last, penultimate, middle} by TRAIN accuracy; features L2-normalized to unit norm.

# Phase 4: Train individual predicate probes
For each v in V, train a linear probe w_v on TRAIN scenes labeled with v (binary).
# Logistic regression, L2 regularization lambda=1e-3, trained to convergence (max 1000 epochs, early stop on TRAIN val).

# Phase 5: Compositional readout on held-out recompositions
For a held-out pair (v1, v2), predict 'A v1 B and B v2 C' by ANDing the individual probe outputs: y_hat = 1[w_v1^T f(s) > 0.5] AND 1[w_v2^T f(s) > 0.5].
# Calibrated AND: replace hard threshold 0.5 with per-probe threshold tau_v chosen on CAL to maximize balanced accuracy on joint scenes; report both hard-AND and calibrated-AND.
# Shared-basis variant: fit a low-rank basis W (rank r=4) jointly across all 8 predicates on TRAIN single-predicate scenes, then compose w_v1 and w_v2 in the shared basis; report as a separate arm.
# Direct conjunction probe: train w_{v1,v2} on TRAIN scenes that contain both v1 and v2, using the same data budget as the AND arm (i.e., matched number of joint scenes per pair, drawn from the same stratified bins).

# Phase 6: Metrics
Compositionality gap = accuracy(individual-probe AND) - accuracy(direct conjunction probe) on HELD-OUT.
Zero-shot recomposition accuracy = accuracy(individual-probe AND) on HELD-OUT.
Control: shuffle object identities to break compositional structure; recompute gap.
# Data-matched control: for each held-out pair, subsample TRAIN conjunction scenes to exactly match the number used by the AND arm's calibration, so the direct probe is not disadvantaged by data scarcity.
```

**Why this design.** We chose a linear probe over a nonlinear MLP because linearity makes the composition operation explicit: the AND of two linear readouts is exactly a compositional rule, whereas an MLP could learn the conjunction jointly and obscure whether composition or memorization occurred, accepting the cost that linear probes may underfit complex spatial predicates. We chose a held-out split over ordered predicate pairs rather than over scenes because scene-level splits allow the same predicate pair to appear in both train and test, which would not test recomposition, accepting the cost that some predicate pairs may be visually rare and produce noisy estimates. We chose to compare individual-probe AND against a direct conjunction probe because the direct probe is the strongest memorization baseline: if the direct probe matches the AND probe on held-out pairs, the representation supports conjunctive encoding but not compositional generalization, accepting the cost that the direct probe requires training data for each conjunction, which we obtain by including a small number of TRAIN scenes with both predicates. We chose to control for object identity shuffling because spatial predicates are object-relative, and a representation could encode object identity rather than spatial relations; shuffling breaks this confound, accepting the cost that shuffled scenes may be visually unnatural. We added a disjoint calibration split and a shared-basis variant because the AND rule is only a valid test of compositionality if the individual probes are not off-distribution at test time; recalibrating thresholds on joint scenes that exclude the held-out pair, and fitting a shared low-rank basis across predicates, makes the load-bearing assumption explicit and testable rather than assumed. This design is not a variant of standard probing (e.g., Alain & Bengio 2016) because standard probing evaluates readout on the same distribution as training, whereas SCRAP evaluates on a systematically held-out recomposition distribution and uses the gap between compositional and conjunctive readouts as the signal.

**Why this measures what we claim.** The compositional gap (accuracy of individual-probe AND minus accuracy of direct conjunction probe on held-out recompositions) measures compositional generalization because it assumes that if the representation encodes predicates as separable linear functionals, then the AND of those functionals will generalize to novel pairs; this assumption fails when the representation encodes predicates conjunctively (e.g., 'left-of-and-behind' as a single direction), in which case the gap reflects the degree of conjunctive encoding rather than compositionality. Zero-shot recomposition accuracy measures the representation's ability to support novel predicate combinations without additional training because it assumes that the individual probes trained on single predicates are unbiased estimators of the underlying predicate functionals; this assumption fails when the training scenes for a predicate are confounded with a particular object configuration, in which case the accuracy reflects object-configuration memorization rather than predicate composition. The object-identity shuffle control measures the degree to which the compositional gap is driven by spatial structure rather than object identity because it assumes that shuffling object identities preserves spatial relations while breaking object-specific associations; this assumption fails when the shuffle inadvertently changes the spatial relations (e.g., by moving objects), in which case the control reflects spatial distortion rather than identity confound removal. The calibrated-AND and shared-basis arms measure whether the gap survives when the individual probes are recalibrated on the joint-scene distribution, because they assume that threshold recalibration and a shared low-rank basis remove the covariate shift between single-predicate training and joint-scene test; this assumption fails when the shift is not a simple threshold or rank effect, in which case the gap reflects residual distribution shift rather than separability.

## Contribution

['A compositional probing framework (SCRAP) that introduces a held-out split over systematic recompositions of spatial predicates and a compositionality gap metric that separates compositional generalization from conjunctive memorization in frozen visual representations.', 'An empirical design principle: the compositionality gap is largest for representations that encode spatial predicates as separable linear directions and smallest for those that encode them conjunctively, providing a diagnostic for representation selection in spatial reasoning tasks.', 'A controlled benchmark of systematic recompositions for egocentric indoor scenes, including object-identity shuffle controls, that can be applied to any frozen visual foundation model.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Egocentric object-triple scenes with held-out predicate-pair split; pre-registered generator with N_MIN=200 TRAIN conjunctions per held-out pair, stratified into 200-499, 500-999, 1000+ bins | Directly instantiates novel recompositions the claim targets; controls conjunction-frequency confound |
| Primary metric | Compositionality gap on held-out recompositions (calibrated-AND minus data-matched direct conjunction probe) | Isolates compositional vs conjunctive encoding under matched joint-distribution exposure |
| Baseline | Direct conjunction probe, data-matched to the AND arm's calibration budget | Strongest memorization competitor, not disadvantaged by data scarcity |
| Baseline | Scene-level random split probe | Tests whether pair-level holdout is doing the work |
| Baseline | Object-identity shuffle control | Tests spatial-structure vs object-identity confound |
| Ablation-of-ours | Nonlinear MLP readout (2-layer, hidden=256, GeLU) replacing linear probes | Checks whether linearity is essential to the gap signal |
| Ablation-of-ours | Hard-AND vs calibrated-AND vs shared-basis (rank r=4) | Tests whether covariate shift explains the gap |

### Why this setup validates the claim

The central claim is that frozen visual features support separable linear predicate functionals whose AND generalizes to unseen predicate pairs. Each baseline probes a distinct threat to that claim: the data-matched direct conjunction probe tests whether compositional readout beats memorized conjunctions on the exact held-out pairs without being disadvantaged by fewer training examples, the scene-level split tests whether the pair-level holdout is what creates the effect (if a scene split shows the same gap, the holdout is not the causal lever), and the shuffle control tests whether the gap is driven by spatial structure rather than object identity. The calibrated-AND and shared-basis arms test the load-bearing assumption that the individual probes are not off-distribution at test time: if the gap survives threshold recalibration on a disjoint calibration split and a shared low-rank basis, it reflects separability rather than covariate shift. The MLP ablation tests whether linearity is the mechanism, since a nonlinear readout could learn conjunctions jointly and erase the gap. The compositionality gap metric is the right detector because it is a difference of two accuracies on identical held-out pairs under matched data budgets, so any representation-level advantage of separable encoding shows up directly rather than being masked by absolute accuracy differences or conjunction-frequency confounds.

### Expected outcome and causal chain

**vs. Direct conjunction probe (data-matched)** — On a held-out pair such as (left-of, behind) where TRAIN contains only a handful of scenes with both predicates, the direct conjunction probe overfits those few conjunctions and collapses on the held-out combination because it has no mechanism to transfer structure from the individual predicates. Our method ANDs the two individually trained linear probes, which were each fit on abundant single-predicate scenes, so it carries that structure into the unseen pair. We expect a visible gap concentrated on rare-conjunction held-out pairs (200-499 bin), with near-parity on pairs that had many TRAIN conjunction examples (1000+ bin).

**vs. Scene-level random split probe** — On a scene-level split, the same ordered predicate pair appears in both train and test, so a probe can memorize the pair-specific configuration and score high on test without any recomposition. That baseline therefore shows a small or absent gap because its test distribution is not compositional. Our pair-level holdout removes every seen pairing from test, so the individual-probe AND must rely on separable functionals. We expect the scene-split baseline to show a noticeably compressed gap relative to our pair-level split, with the difference concentrated on pairs whose conjunction appears only in test.

**vs. Object-identity shuffle control** — On shuffled scenes where object identities are permuted but spatial layout is preserved, a representation encoding object identity rather than relations will still predict well, so the shuffle baseline retains accuracy and the gap persists spuriously. If our gap is genuinely spatial, shuffling should not rescue the direct conjunction probe but should degrade any identity-driven signal. We expect the gap to shrink markedly under shuffling only if the original gap was identity-driven; a stable gap under shuffle is the predicted spatial signal.

**vs. Nonlinear MLP readout ablation** — On a held-out pair, an MLP trained jointly on conjunction labels can fit the conjunction as a single nonlinear region, so it matches the direct probe and shows no compositional advantage. Our linear AND is forced to factor the conjunction into two separable functionals, so its behavior on unseen pairs is determined by the individual predicates. We expect the MLP to close the gap relative to linear probes on pairs where the representation is conjunctive, and to match linear probes where it is separable, producing a gap pattern that tracks separability rather than overall accuracy.

**vs. Hard-AND vs calibrated-AND vs shared-basis** — If the gap is driven by covariate shift, recalibrating thresholds on the disjoint CAL split and fitting a shared low-rank basis should shrink the gap toward zero. If the gap persists under both, the individual probes are not merely off-distribution and the gap reflects separability. We expect calibrated-AND to reduce the gap modestly and shared-basis to reduce it further on pairs where single-predicate training is confounded with object configuration, with the residual gap concentrated on pairs predicted to be separable.

### What would falsify this idea

If the compositional gap is uniform across held-out pairs rather than concentrated on pairs predicted to be separable, or if the scene-level split baseline shows the same gap as our pair-level split, then the effect is not compositional generalization and the central claim is wrong. Likewise, if the gap vanishes entirely under object-identity shuffling, the signal was object identity rather than spatial structure; and if the gap vanishes under calibrated-AND or shared-basis, the signal was covariate shift rather than separability.

## References

1. Probing Spatial State Representations in Visual Foundation Models
