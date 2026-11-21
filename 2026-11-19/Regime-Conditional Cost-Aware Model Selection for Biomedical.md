# Regime-Conditional Cost-Aware Model Selection for Biomedical Tabular Data via Meta-Learned Decision Rules

## Motivation

TabBioMed (E5) benchmarks 23 models across 94 biomedical tabular datasets and concludes that family-level rankings are 'strongly context-dependent,' with foundation models leading on aggregate but trees and tuned MLPs dominating under inference-time or training-cost constraints. The benchmark reports only family-level aggregate findings and does not instantiate a per-dataset decision rule, so a practitioner facing a new EHR or omics table still cannot know which family wins under their own cost budget without exhaustive benchmarking. The structural root cause is that the benchmark measures average family performance rather than the conditional mapping from dataset meta-features and cost constraints to the optimal family, leaving the 'context-dependent' conclusion descriptive rather than prescriptive.

## Key Insight

The optimal model family is a deterministic function of a low-dimensional sufficient statistic of dataset meta-features (dimensionality, missingness, imbalance, sample size) plus the cost constraint, so a meta-learner can estimate this function directly from leave-one-dataset-out folds and generalize to unseen biomedical tables without any per-dataset model training at deployment.

## Method

**What it is.** We introduce CARS (Cost-Aware Regime Selection), a meta-learner that maps a biomedical tabular dataset's meta-feature vector and an explicit cost budget to a recommended model family. Inputs: meta-features (log sample size, log dimensionality, feature-to-sample ratio, missingness rate, minority-class fraction, task type, target count) and a cost constraint (inference-time budget or training-cost budget). Output: a probability distribution over model families (classical, gradient-boosted trees, neural tabular, foundation) plus the argmax family, trained entirely on TabBioMed's 94 datasets with leave-one-dataset-out (LODO) validation.

**Load-bearing assumption (stated explicitly).** CARS assumes that the cost-constrained oracle family label y_{d,c} is a deterministic, learnable function of the seven cheap meta-features, so that datasets with near-identical meta-features share the same winning family and a LightGBM meta-learner trained on 94 LODO folds can recover that mapping for unseen biomedical tables. This assumption can fail when two datasets share nearly identical meta-features but have different oracle families due to content-level structure (feature semantics, label noise) not captured by m_d — a meta-feature collision. CARS therefore does not output a bare argmax; it outputs a calibrated confidence q and falls back to the global best-family constant when q is low, as specified in Phase 2b and Phase 3.

**How it works.**

```pseudocode
# Phase 1: meta-feature extraction (per dataset d in TabBioMed's 94)
for d in datasets:
    m_d = [log(n_samples), log(n_features), n_features/n_samples,
           missing_rate, minority_fraction, is_multitarget, n_targets]
    # cost-conditioned soft label: for each cost budget c in {inference, training}
    for c in budgets:
        # family-metric vector over 4 families, cost-feasible entries only
        s_{d,c}[f] = mean_metric(family=f, d)  if cost(f,d) <= c  else  -inf
        # soft target: softmax over cost-feasible families with temperature tau=0.1
        p_{d,c} = softmax(s_{d,c} / tau)
        # margin-thresholded hard label for stability check
        top1, top2 = top_two(s_{d,c})
        y_{d,c} = argmax_f s_{d,c}[f]  if (s_{d,c}[top1] - s_{d,c}[top2]) > margin=0.01
                  else  'ambiguous'   # excluded from hard-label training, kept for soft training

# Phase 2: regime-conditional meta-learner (gradient-boosted, LightGBM)
# separate GBM per cost regime; multiclass over 4 families
for c in budgets:
    GBM_c = LightGBM(objective='multiclass', num_class=4, n_estimators=300,
                     max_depth=4, learning_rate=0.05,
                     min_child_samples=5)   # small-n regularizer
    for held_out d* in datasets:                      # LODO
        GBM_c.fit({(m_d, y_{d,c}) : d != d*, y_{d,c} != 'ambiguous'})
        pred[d*,c] = GBM_c.predict_proba(m_{d*})

# Phase 2b: collision / uncertainty term
# For each held-out d*, find its k=5 nearest training datasets in m-space (Euclidean on z-scored m).
# Compute empirical disagreement among their oracle families:
#   disagree(d*) = 1 - (max_f count(y_{d,c}=f in kNN) / k)
# Calibrated confidence q(d*,c) = pred[d*,c].max() * (1 - disagree(d*))

# Phase 3: cost-constrained decision rule + calibration
recommend(d, c) = argmax_family GBM_c.predict_proba(m_d)  if q(d,c) >= q_threshold=0.5
                  else  global_best_family(c)   # constant fallback
# report per-fold top-1 family accuracy and regret vs oracle family
# report collision rate: fraction of held-out d* with disagree(d*) > 0.2
# report Bayes-optimal accuracy ceiling implied by meta-feature partition:
#   ceiling = mean over d* of (max_f count(y_{d,c}=f in kNN) / k)
```

**Why this design.** We chose a single gradient-boosted multiclass meta-learner over a hierarchical Bayesian model because with only 94 datasets the Bayesian posterior over family-selection coefficients would be dominated by the prior, whereas LightGBM's shallow trees with min_child_samples=5 regularize on small n while still capturing threshold effects (e.g., 'missingness > 0.3 favors trees') that a linear Bayesian link would miss; the cost is that we forfeit interpretable posterior credible intervals, which we recover post hoc via permutation importance rather than native uncertainty. We chose separate GBMs per cost regime over one joint model with cost as an input feature because the two regimes have structurally different label definitions (inference-time vs training-cost constraints select different families), and pooling them would let the dominant regime's labels swamp the minority regime; the cost is a doubled parameter count, mitigated by sharing the meta-feature extractor. We chose LODO over random k-fold because random folds leak dataset-level distributional information (e.g., two omics datasets from the same source) into training, inflating apparent accuracy; the cost is higher variance in per-fold estimates, which we report as mean ± std across 94 folds. We replaced the hard argmax label with a soft, cost-weighted target and a margin-thresholded hard label because near-ties in family performance can flip the argmax and make top-1 accuracy reward noise; the soft target and margin=0.01 filter stabilize the label, and the kNN disagreement term makes meta-feature collisions explicit rather than hidden. A domain expert would not describe this as 'TabBioMed plus a classifier,' because the label is not the benchmark's reported family ranking but a cost-constrained oracle family recomputed per dataset, and the meta-features are chosen to be the regime variables the benchmark names as decisive.

**Why it measures what we claim.** The meta-learner's predicted probability for a family measures the conditional optimality of that family given the dataset's regime because we assume the oracle family label y_{d,c} (the cost-feasible argmax of mean metric on dataset d) is a deterministic function of the meta-feature vector m_d; this assumption fails when two datasets share nearly identical meta-features but have different oracle families due to content-level structure (e.g., feature semantics, label noise) not captured by m_d, in which case the predicted probability reflects an average over an unresolvable meta-feature collision rather than a genuine per-dataset recommendation — the kNN disagreement term and the reported Bayes-optimal ceiling quantify exactly this failure. The LODO top-1 accuracy measures generalization to unseen datasets because we assume held-out datasets are exchangeable draws from the same biomedical tabular population as the training 94; this assumption fails for out-of-distribution tables (e.g., a modality absent from TabBioMed), in which case accuracy reflects in-distribution transfer rather than universal selection validity. The cost-feasibility filter measures the cost constraint because we assume the benchmark's reported inference/training cost proxies are monotone in the practitioner's actual budget; this fails when a deployment's cost profile differs sharply (e.g., GPU-only serving), in which case the recommendation respects a proxy budget rather than the true one. The calibrated confidence q(d,c) measures the reliability of the recommendation because we assume kNN disagreement in meta-feature space is a valid proxy for label ambiguity; this fails when the kNN neighbourhood is too small or unrepresentative, in which case q reflects local density rather than true uncertainty.

## Contribution

["(1) CARS, a cost-conditioned gradient-boosted meta-learner that converts TabBioMed's descriptive 'context-dependent' finding into a per-dataset, cost-constrained model-family decision rule validated with leave-one-dataset-out folds.", '(2) A design principle: dataset meta-features chosen to match the regime variables the benchmark names (dimensionality, missingness, imbalance, sample size) are sufficient to predict the cost-feasible optimal family, with per-regime regret reported against the oracle family.', '(3) A reusable meta-feature extraction and LODO evaluation protocol over the 94 TabBioMed datasets that any future benchmark can adopt to report prescriptive rather than descriptive rankings.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TabBioMed 94 datasets, LODO folds | Matches benchmark's regime diversity; tests unseen-dataset transfer |
| Dataset (external) | One held-out MIMIC-derived EHR table and one TCGA omics table, not in TabBioMed | Tests deployment value on unseen biomedical tables |
| Primary metric | Top-1 family accuracy + regret vs cost-constrained oracle | Directly measures selection quality, not raw prediction |
| Secondary metric | Collision rate (fraction of held-out d* with kNN disagreement > 0.2) and Bayes-optimal accuracy ceiling implied by meta-feature partition | Empirically bounds meta-feature sufficiency |
| Secondary metric | Compute cost saved vs exhaustive benchmarking (GPU-hours) | Quantifies practical payoff |
| Baseline | Global best-family constant predictor | Tests whether meta-features beat a regime-blind default |
| Baseline | TabBioMed reported family-level ranking (foundation-first) | Tests whether cost-conditioning beats unconditional ranking |
| Baseline | Meta-learner without cost feature (pooled regimes) | Tests whether separate cost regimes matter |
| Baseline | SUNNY (algorithm-selection via k-NN meta-features) | Tests against established algorithm-selection baseline |
| Baseline | Cost-aware utility model predicting continuous regret (ridge regression on m_d) | Tests discrete argmax vs continuous utility formulation |
| Ablation-of-ours | CARS with meta-features shuffled | Tests whether regime variables carry the signal |
| Ablation-of-ours | CARS with hard argmax label (no soft target, no margin filter) | Tests label-noise stability |
| Ablation-of-ours | CARS without kNN disagreement fallback (always argmax) | Tests whether collision handling matters |

### Why this setup validates the claim

The central claim is that cost-constrained family selection is predictable from cheap dataset meta-features, and that predicting it beats ignoring regime. Each baseline isolates one sub-claim. The constant predictor asks whether meta-features carry any selection signal at all: if a single family always wins, CARS cannot beat it and the regime story collapses. The TabBioMed family-level ranking baseline asks whether the benchmark's aggregate foundation-model advantage survives cost-conditioning: if foundation models win unconditionally and also under every budget, CARS reduces to a constant and its conditional machinery is decorative. The pooled-regime meta-learner asks whether the two cost regimes genuinely have different label structure; if pooling matches per-regime performance, the 'structurally different labels' argument is false. The SUNNY baseline asks whether CARS's LightGBM meta-learner beats an established k-NN algorithm selector, isolating whether the cost-conditional formulation adds value beyond standard meta-feature selection. The continuous-regret utility model asks whether the discrete argmax family label is the right target or whether predicting continuous regret is more effective. The shuffled-meta-feature ablation is the load-bearing falsifier: it holds the learner and labels fixed while destroying the regime signal, so any accuracy above shuffle is attributable to meta-features rather than to label imbalance or LightGBM priors. The hard-argmax and no-fallback ablations isolate the contributions of the soft/margin label and the collision handling. Top-1 accuracy and regret are the right metrics because the claim is about choosing the right family, not about predictive performance on the underlying tasks; regret against the cost-constrained oracle directly quantifies the cost of a wrong recommendation, which is the quantity a practitioner actually pays. The collision rate and Bayes-optimal ceiling empirically bound the meta-feature sufficiency assumption rather than assuming it.

### Expected outcome and causal chain

**vs. Global best-family constant predictor** — On a case where the dataset population is split between high-missingness omics tables and low-dimensional EHR tables, the constant predictor produces a single family for all 94 folds because it has no mechanism to condition on dataset structure. Our method instead routes high-missingness, high-dimensionality meta-features to trees and low-dimensional, large-sample meta-features to neural or foundation families because the LightGBM meta-learner learns threshold splits on exactly those regime variables. We expect a clear accuracy gap on the mixed-regime folds and near-parity on folds where one family dominates the whole population, with regret concentrated in the minority-regime datasets.

**vs. TabBioMed family-level ranking (foundation-first)** — On a case where inference-time budget is tight and a foundation model's cost exceeds it, the ranking baseline still recommends the foundation family because it optimizes unconditional mean metric and treats cost as an afterthought. Our method instead filters the label through the cost-feasibility constraint before training, so the recommended family is the cost-feasible argmax, and we expect the gap to appear specifically on cost-constrained folds while the two agree on unconstrained folds — the signature that cost-conditioning, not general competence, drives the gain.

**vs. Pooled-regime meta-learner (no cost feature)** — On a case where a training-cost budget selects trees but an inference-time budget selects a neural family for the same dataset, the pooled learner produces a blurred label dominated by whichever regime is more common and misroutes the minority-regime dataset. Our method instead trains separate GBMs per regime, preserving each regime's label definition, so we expect the pooled learner to match us on the majority regime and lose noticeably on the minority regime, with the gap widening as regime imbalance grows.

**vs. SUNNY (k-NN algorithm selection)** — On a case where meta-feature collisions are frequent (high collision rate), SUNNY's k-NN vote over raw meta-features is sensitive to local label noise, while CARS's LightGBM with soft targets and margin filtering is more robust. We expect CARS to outperform SUNNY on folds with high collision rate and to be comparable elsewhere.

**vs. Continuous-regret utility model (ridge regression)** — On a case where family performance gaps are large and stable, the continuous-regret model may predict regret well but its argmin family may not match the cost-feasible argmax due to regression to the mean. We expect CARS to have higher top-1 accuracy but the utility model to have lower regret on near-tie folds; the comparison isolates whether discrete or continuous targets are better for this task.

**vs. CARS with hard argmax label (no soft target, no margin filter)** — On a case where two families have nearly equal mean metric (margin < 0.01), the hard label flips arbitrarily between folds, adding noise. Our soft target and margin filter stabilize the label, so we expect higher top-1 accuracy and lower variance across LODO folds, especially on datasets with small performance gaps.

**vs. CARS without kNN disagreement fallback** — On a case where a held-out dataset has high meta-feature collision (disagreement > 0.2), the no-fallback version always outputs an argmax with potentially low confidence, leading to high regret. Our fallback to the global best-family constant when q < 0.5 avoids catastrophic errors, so we expect lower regret on high-collision folds at the cost of slightly lower top-1 accuracy on low-collision folds.

### What would falsify this idea

If CARS's top-1 accuracy is statistically indistinguishable from the shuffled-meta-feature ablation, or if its gain over the constant predictor is uniform across all dataset subsets rather than concentrated on mixed-regime and cost-constrained folds, or if the collision rate is so high that the Bayes-optimal accuracy ceiling is near the constant predictor's accuracy, then the meta-features carry no regime signal and the central claim is wrong.

## References

1. TabBioMed: A Large-Scale Benchmark for Biomedical Tabular Learning
