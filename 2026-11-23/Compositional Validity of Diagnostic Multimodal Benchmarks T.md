# Compositional Validity of Diagnostic Multimodal Benchmarks: Testing Whether Sub-Score Vectors Reconstruct Holistic Matching Accuracy

## Motivation

Multimodal benchmark suites increasingly report fine-grained diagnostic sub-scores (PURE's Perception/Understanding/Reasoning/Judgment, MEGA-Bench's multi-dimensional capability report, General-Level's synergy levels) while separately reporting a holistic task outcome (direct matching accuracy, ELO ranking, aggregate score), and both PURE and VideoAutoArena treat the overall number as an unconstrained aggregate that is assumed independent of the sub-score vector. This shared structural gap means a benchmark can report a 'diagnostic decomposition' that is compositionally invalid — the four sub-scores carry no information about the holistic decision the benchmark purports to explain — and no existing work tests this property, so PURE's reported overall scores (47.1%–86.0%) and its per-dimension means (Understanding highest, Judgment lowest) are never checked against the direct-matching accuracy they are supposed to compose into. The root cause is that sub-task MCQ scoring and holistic matching are collected on disjoint item pools with no shared conditioning structure, so the mapping from sub-score vector to holistic outcome is never identified, let alone validated.

## Key Insight

If a diagnostic benchmark's sub-scores are genuinely decomposing the holistic capability, then a fixed, cross-validated mapping from the four sub-score vectors to direct-matching accuracy must predict held-out models' holistic performance better than a permutation-invariant constant, and the residual of that mapping is a measurable, model-specific quantity — compositional validity — that existing benchmarks leave untested because they never co-measure sub-scores and holistic outcomes on a shared model set.

## Method

```pseudocode
# CVT: Compositional Validity Test (applied to PURE)
# Inputs: M models, each with 4-dim sub-score vector s_m in [0,1]^4
#         (Perception, Understanding, Reasoning, Judgment) from PURE's 14 MCQ tasks
#         and holistic direct-matching accuracy h_m in [0,1] from PURE's direct-matching probe
#         and item-level data: each of the 1,457 direct-matching items is tagged with the subset
#         of the 4 dimensions it exercises (PURE's task-to-dimension mapping, pre-registered)

# LOAD-BEARING ASSUMPTION (stated explicitly):
#   The four sub-scores and the holistic direct-matching accuracy are measured on a shared model set,
#   and the true composition rule from sub-score vector to holistic accuracy is additive and monotone
#   in logit space. Under this assumption, the sub-score vector is a sufficient statistic for holistic
#   accuracy across models. If the true rule is non-additive (e.g., Judgment matters only when Reasoning
#   is high), CVT_score reflects only the additive component and understates true compositional validity.

Phase 1  CO-MEASURE (fix the disjoint-pool problem)
  for each model m in M:
      s_m <- mean accuracy over PURE MCQ items in each of the 4 dimensions
      h_m <- accuracy on the 1,457-item direct-matching pool, 5-fold item bootstrap
  require |M| >= 25  # pre-registered: 10 PURE models + 15 additional open MLLMs re-run on PURE's item pool
  # Power analysis (pre-registered): with |M|=25, 4 predictors, alpha=0.05, the minimum detectable
  # true CVT_score is 0.30 at 80% power (computed by simulation from the additive-logit generative model
  # with weights w_d ~ Uniform(0.1, 0.5) and residual sd = 0.15 in logit space).

Phase 2  SPECIFY THE MAPPING FAMILY (committed, not searched)
  # Monotone additive logit map: the compositional-validity hypothesis is that
  # each sub-score contributes monotonically and separably to holistic logit.
  logit(h_m) ~ b0 + sum_d w_d * logit(s_m[d]),  w_d >= 0
  # Fit by constrained least squares (nonnegative w_d, scipy.optimize.nnls on logit-transformed data),
  # no interaction terms. w_d >= 0 encodes 'more of a capability cannot hurt the holistic decision'.

Phase 3  CROSS-VALIDATED RECONSTRUCTION
  # 3a. Item-level fit (identifies the composition rule from thousands of items):
  #     For each direct-matching item i tagged with dimension subset D_i, fit a logistic regression
  #     of item correctness on the model's sub-scores restricted to D_i:
  #         P(correct_{m,i}) = sigmoid(a0 + sum_{d in D_i} a_d * logit(s_m[d]))
  #     with a_d >= 0, fit by maximum likelihood on all (m, i) pairs (|M| x 1,457 rows).
  #     This yields item-level weights a_d estimated from ~36,000 observations, not ~25 models.
  # 3b. Cross-model generalization check (holds out models only for final validation):
  for each held-out model m*:
      fit (b0, w) on M \ {m*} using the item-level weights a_d as initialization
      h_hat_m* <- sigmoid(b0 + sum_d w_d * logit(s_m*[d]))
  CVT_score <- 1 - MAE(h_hat, h) / MAE(h_bar_train, h)   # skill over train-mean baseline

Phase 4  NULL / PERMUTATION CONTROL
  for r in 1..1000:
      shuffle the dimension labels of s_m independently per model
      recompute CVT_score_r
  p_value <- fraction of CVT_score_r >= CVT_score

Phase 5  RESIDUAL DIAGNOSIS
  r_m <- h_m - h_hat_m   # per-model compositional residual
  report corr(r_m, each dimension's s_m[d]) to localize which dimension
  the holistic outcome fails to compose from.

Phase 6  SYNTHETIC GROUND-TRUTH CALIBRATION (pre-registered)
  # Generate synthetic model sets with known composition rules:
  #   Additive: logit(h_m) = b0 + sum_d w_d * logit(s_m[d]) + eps,  eps ~ N(0, 0.15^2)
  #   Non-additive: logit(h_m) = b0 + sum_d w_d * logit(s_m[d]) + w_JR * logit(s_J) * logit(s_R) + eps
  # For each rule, generate 200 synthetic datasets of |M|=25 models with s_m ~ Uniform(0,1)^4.
  # Run Phases 2-4 on each; report mean CVT_score and 95% interval for additive (should be >0.3)
  # and non-additive (should be lower, quantifying understatement).
```

**Why this design.** We chose a *constrained monotone additive logit map* over a flexible learned regressor (e.g., gradient-boosted trees or a small MLP) because with only ~25 models a flexible regressor can fit the training models perfectly and yield a CVT_score that reflects overfitting rather than compositional validity, accepting the cost that we cannot detect non-additive compositional structure (e.g., a Judgment×Reasoning interaction) and will under-report validity if such structure exists. We chose *leave-one-model-out* cross-validation over a random item split because the quantity of interest is generalization across models, not across items — an item split would let the same model's sub-scores leak into its own holistic prediction, accepting the cost that CVT_score has high variance when |M| is small. We chose a *nonnegative-weight constraint* over unconstrained least squares because the compositional-validity claim is directional (a capability sub-score cannot anti-compose with the holistic outcome), accepting the cost that a genuinely negative weight (a dimension that is diagnostic of failure) is forced to zero and shows up as residual rather than as a fitted coefficient. We chose a *label-permutation null* over a parametric null because with 4 dimensions and ~25 models the parametric distribution of CVT_score is unknown, accepting the cost of 1000 permutations per run. We added *item-level fitting (Phase 3a)* because the 5-parameter cross-model map is underpowered with |M|~25; fitting on item-level data where each holistic item is tagged by which dimensions it exercises makes the functional form identifiable from thousands of items, and the cross-model check (Phase 3b) then tests generalization. We added *synthetic ground-truth calibration (Phase 6)* to verify that CVT_score recovers known compositional validity and to quantify its understatement under non-additive rules. A domain expert would not describe this as a variant of VideoAutoArena's ELO aggregation (E1, E9): ELO aggregates *pairwise battle outcomes* into a single scalar and never relates a sub-score vector to a holistic outcome; CVT instead tests a *fixed functional form* from a 4-vector to an independently measured holistic accuracy, and its output is a validity score plus a per-model residual, not a ranking.

**Why it measures what we claim.** `CVT_score` measures *compositional validity* — the degree to which PURE's four sub-scores carry information about the holistic direct-matching outcome — because the leave-one-model-out monotone logit map is assumed to be the correct functional form linking diagnostic decomposition to holistic decision, so that a positive CVT_score is equivalent to the sub-score vector being a sufficient statistic for holistic accuracy across models; this assumption fails when the true compositional structure is non-additive (e.g., Judgment only matters when Reasoning is high), in which case CVT_score reflects only the additive component of validity and understates true compositional validity. `r_m` (the per-model residual) measures *model-specific compositional failure* because the fitted map is assumed to capture the population-level composition rule, so a large |r_m| is equivalent to model m's holistic behavior deviating from what its sub-scores predict; this assumption fails when model m is an outlier in sub-score space (e.g., near-ceiling on all four dimensions), in which case r_m reflects extrapolation error rather than genuine compositional failure. The permutation `p_value` measures *whether the observed validity exceeds chance composition* because shuffling dimension labels is assumed to destroy any genuine sub-score-to-holistic link while preserving marginal distributions; this assumption fails when two dimensions are highly correlated across models, in which case permutation under-estimates the null and the p-value is anti-conservative. The item-level weights `a_d` (Phase 3a) measure *which dimensions drive item-level holistic correctness* because each item is tagged with the dimensions it exercises, so a positive `a_d` is equivalent to that dimension contributing to correctness on items that exercise it; this assumption fails when the task-to-dimension mapping is noisy, in which case `a_d` reflects tag error rather than genuine dimension contribution.

## Contribution

(1) A compositional validity test (CVT) — a cross-validated, monotone logit map from a diagnostic benchmark's sub-score vector to its holistic outcome, with a label-permutation null and a per-model compositional-residual diagnostic — that turns 'is the diagnostic decomposition meaningful?' into a measurable, falsifiable quantity. (2) The design principle that diagnostic benchmarks should co-measure sub-scores and holistic outcomes on a shared model set with a specified composition function, rather than reporting them as independent aggregates. (3) A reusable evaluation protocol and residual-diagnosis tool applicable to any benchmark that reports both a capability decomposition and a holistic task score (PURE, MEGA-Bench, General-Bench).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PURE (1,457 MCQ, 4 dimensions) + 15 additional open MLLMs re-run on PURE's item pool | Shared model set with both sub-scores and holistic probe; |M|≥25 for power |
| Primary metric | CVT_score (LOO skill over train-mean) | Measures model-level compositional generalization, not item fit |
| Baseline | Unconstrained OLS logit map | Tests whether nonnegativity constraint is doing work |
| Baseline | Gradient-boosted trees (LOO) | Tests whether flexibility inflates apparent validity |
| Baseline | Dimension-shuffled null (1000 perms) | Tests validity against chance composition |
| Ablation-of-ours | Drop Judgment dimension from map | Tests whether the weakest dimension carries signal |
| Calibration | Synthetic ground-truth simulator (additive and non-additive rules, 200 datasets each, |M|=25) | Verifies CVT_score recovers known validity and quantifies understatement |

### Why this setup validates the claim

The central claim is that PURE's four diagnostic sub-scores compose monotonically and separably into the holistic direct-matching outcome across models. This is a claim about *cross-model generalization of a fixed functional form*, so the design must punish any method that merely fits the models it sees. Leave-one-model-out with a skill score over the train-mean baseline is the right metric because a flexible regressor can drive training residuals to zero while carrying no information to unseen models; the train-mean baseline makes that overfitting visible as a CVT_score near zero rather than a misleadingly high R². The unconstrained-OLS baseline tests whether the nonnegativity constraint matters: if unconstrained weights are already nonnegative and CVT_score matches ours, the constraint is inert and the directional claim is untested; if unconstrained weights go negative with worse LOO skill, the constraint is load-bearing and the directional hypothesis is supported. The gradient-boosted-tree baseline tests the flexibility sub-claim directly: with ~25 models it should fit training models near-perfectly and lose to our constrained map on held-out models, which is exactly the overfitting signature we predict. The dimension-shuffled null anchors CVT_score against chance: a positive score only counts if it exceeds the permutation distribution. The drop-Judgment ablation tests whether the dimension PURE reports as weakest and most failure-prone still contributes to holistic composition; if removing it leaves CVT_score unchanged, the compositional claim is carried entirely by the other three dimensions and the diagnostic decomposition is partly decorative. The synthetic ground-truth calibration (Phase 6) tests whether CVT_score recovers a known compositional validity: under the additive rule it should be high (>0.3), and under the non-additive rule it should be lower, quantifying the understatement from the additive assumption. The item-level fit (Phase 3a) makes the functional form identifiable from thousands of items, so the cross-model check (Phase 3b) is not underpowered.

### Expected outcome and causal chain

**vs. Unconstrained OLS logit map** — On a case where one model is near-ceiling on Perception and Reasoning but mid-range on Judgment, unconstrained OLS is free to assign a negative Judgment weight to fit that single model's holistic accuracy, because it has no directional prior and only ~25 training points. Our method instead forces w_Judgment ≥ 0, because the compositional-validity claim is that more of any capability cannot anti-compose with the holistic decision, so the negative signal is pushed into the residual r_m rather than the coefficient. We expect our CVT_score to exceed unconstrained OLS by a modest but consistent margin, with the largest per-model residuals concentrated on models whose sub-score profiles are extreme, and near-parity on models near the centroid of sub-score space.

**vs. Gradient-boosted trees (LOO)** — On a case where a model has a distinctive sub-score profile (e.g., high Understanding, low Reasoning), the tree ensemble memorizes that profile's holistic value during training and then, at test time, has no comparable leaf for the held-out model, because with ~25 models it splits on model-specific idiosyncrasies rather than the separable dimension contributions. Our method instead fits four monotone coefficients shared across all training models, because the additive logit map has only five free parameters and cannot memorize per-model quirks. We expect the tree baseline to show near-zero or negative LOO CVT_score (train-mean parity or worse) while our map stays clearly positive, with the gap largest on the held-out models furthest from the training centroid.

**vs. Dimension-shuffled null (1000 permutations)** — On a case where the true dimension labels are permuted independently per model, the shuffled sub-score vectors retain their marginal distributions but lose the alignment between a model's actual capability profile and its holistic accuracy, because the permutation destroys exactly the sub-score-to-holistic link the claim asserts. Our method instead fits on the true labels, so the fitted weights reflect the genuine compositional rule. We expect the observed CVT_score to sit well above the permutation distribution (p below the conventional threshold), with the null distribution centered near zero and the observed score separated by a clear margin — unless two dimensions are highly correlated across models, in which case we expect the null to be wider and the p-value correspondingly less extreme, which is itself a diagnostic we would report.

**vs. Synthetic ground-truth calibration (additive rule)** — On synthetic data generated from the additive logit rule with known weights, CVT_score should recover a high value (>0.3) because the fitted map matches the true generative form. We expect the mean CVT_score across 200 synthetic datasets to be above 0.3 with a 95% interval that excludes zero, confirming that the score measures compositional validity when the additive assumption holds.

**vs. Synthetic ground-truth calibration (non-additive rule)** — On synthetic data generated with a Judgment×Reasoning interaction term, CVT_score should be lower than under the additive rule because the additive map cannot represent the interaction and absorbs it into residual. We expect the mean CVT_score to drop by a measurable amount (e.g., from >0.3 to <0.2), quantifying the understatement and showing that a null CVT_score on real data is ambiguous between no validity and non-additive validity.

### What would falsify this idea

If the constrained monotone map's LOO CVT_score is statistically indistinguishable from the train-mean baseline and from the dimension-shuffled null — i.e., the four sub-scores carry no cross-model information about holistic direct-matching accuracy — then the compositional-validity claim is wrong, not merely underpowered. A secondary falsifier: if dropping any single dimension (especially Judgment) leaves CVT_score unchanged, then the claimed four-dimension decomposition is not the compositional structure that drives the holistic outcome.

## References

1. PURE: A Diagnostic Benchmark for Multi-Granularity Multimodal Identity Matching
