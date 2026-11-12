# Individual-Difference Calibration of Language Models on Cognitive Tasks

## Motivation

CogBattery evaluates 54 language models against condition-level aggregates of human data, showing that models match human effect directions but differ in error patterns and confidence separation; however, because the human reference is an aggregate, it cannot establish predictive equivalence at the individual-difference level. This structural limitation recurs across cognitive evaluation frameworks: condition-level aggregation discards the very variance that defines individual differences, so a model that fits the aggregate can still fail to reproduce the covariance structure among humans. Without an individual-differences model, claims about model 'cognitive profiles' remain untestable against the human variation they purport to capture.

## Key Insight

If model-to-model variation in trial-level cognitive parameters (false-alarm rate, confidence separation) reproduces the covariance structure of human individual differences, then model profiles are predictively equivalent to human profiles at the level where cognitive theories actually make differential predictions.

## Method

**What it is.** ID-Cog (Individual-Difference Cognitive Profiling) takes as input trial-level responses from a large human sample and from a set of language models on the same 13 text-adapted cognitive tasks, and outputs per-agent latent cognitive profiles plus a formal test of whether model-profile variation predicts human-profile variation.

**How it works.**

```text
Phase 1: Human trial-level collection
  N_humans >= 500 participants x 13 tasks x ~60 trials
  Each trial: (task_id, condition, target/non-target, response, confidence 1-6, wager)

Phase 2: Per-agent latent profile estimation (hierarchical Bayesian signal-detection model)
  For agent i, task t, condition c:
    d'_i,t,c ~ Normal(mu_d,t,c, sigma_d)          # discriminability
    c_i,t,c  ~ Normal(mu_c,t,c, sigma_c)          # criterion
    FA_rate_i,t,c = Phi(-c_i,t,c)                  # false-alarm rate
    ConfSep_i,t = d'_i,t,high - d'_i,t,low         # confidence separation
  Priors: mu_d,t,c ~ Normal(0,1); sigma_d,c ~ HalfNormal(0.5)
  Fit via HMC (4 chains, 2000 draws, 1000 warmup)
  Output: profile vector p_i = [d'_i, c_i, ConfSep_i] over tasks (39 dims)
  Reliability: posterior SD for each element of p_i; define reliable humans as those with
    max posterior SD across the 39 dims <= 0.5 (on the latent scale).

Phase 3: Individual-differences structure
  Human covariance: Sigma_H = Cov(p_h) over humans
  Model covariance: Sigma_M = Cov(p_m) over models
  Effective rank: report rank(Sigma_M) and condition number kappa(Sigma_M) using a
    threshold of 1e-3 on normalized eigenvalues.
  Latent-profile alignment: fit a 2-level latent profile model
    p_i ~ sum_k pi_k Normal(mu_k, Sigma_k)   # K=3 profiles selected by BIC
  Compare profile membership distributions and within-profile covariance

Phase 4: Predictive equivalence test
  Held-out human split: train on half, test on half; restrict test set to reliable humans
    (posterior SD <= 0.5).
  Mapping: replace free ridge W with a rank-constrained mapping W = U V^T where U (39 x r)
    and V (r x 39) are fit by reduced-rank regression with r selected from {1,2,3,5,10} by
    cross-validated R^2 on the training split; ridge penalty lambda=1.0 applied to V.
  Prediction: p_h_hat = W * p_m + b
  Metric: individual-level R^2 on held-out reliable humans and profile-membership accuracy.
  Null calibration: permutation-based null that preserves predictor covariance by applying a
    random orthogonal rotation R to model profiles before fitting W (i.e., p_m -> R p_m),
    repeated 1000 times, yielding a null distribution of held-out R^2.
  Baseline: condition-level aggregate prediction (CogBattery-style)
  Test: Delta R^2 > 0 with bootstrap 95% CI, and observed R^2 above the 95th percentile of
    the rotation null.
```

**Why this design.** We chose a hierarchical Bayesian signal-detection model over a simple per-task averaging because it separates discriminability from criterion while pooling information across conditions, accepting the cost that HMC fitting is slower and requires careful prior specification. We chose a latent-profile model over factor analysis because cognitive theories posit discrete profiles (e.g., high-FA vs low-FA subtypes) rather than continuous dimensions, accepting the cost that profile number K must be selected and may vary by sample. We chose held-out individual-level prediction over in-sample covariance comparison because covariance matching can be achieved by any model that reproduces the aggregate, accepting the cost that prediction requires a mapping W from model to human space, which we fit by reduced-rank regression with rank selected by cross-validation to avoid overfitting with 39 dimensions and ~500 humans. A domain expert would not describe this as a variant of CogBattery because CogBattery compares condition means, whereas ID-Cog estimates per-agent latent parameters and tests whether their covariance structure predicts held-out human profiles — a different estimand and a different test.

**Why this measures what we claim.** The per-agent latent profile p_i measures individual cognitive profile because the hierarchical signal-detection model assumes that trial-level responses are generated by agent-specific discriminability and criterion parameters, an assumption that fails when agents use response strategies not captured by signal detection (e.g., always-confident responding), in which case p_i reflects response bias rather than cognitive profile. The covariance Sigma_M measures individual-difference structure because it assumes that model-to-model variation is generated by the same latent profile mixture as human variation, an assumption that fails when models are fine-tuned on different data distributions, in which case Sigma_M reflects training-data diversity rather than cognitive structure. The held-out R^2 measures predictive equivalence because it assumes that the mapping W from model to human profiles is stable across human samples and that the model-profile manifold linearly spans the human-profile covariance (Sigma_H ⊆ span(Sigma_M)); this assumption fails when models are fine-tuned on disjoint data so Sigma_M is rank-deficient relative to Sigma_H, in which case R^2 reflects ridge shrinkage rather than alignment. We report rank(Sigma_M) and condition number kappa(Sigma_M) alongside R^2, and we calibrate the null by rotating model profiles with a random orthogonal matrix (preserving predictor covariance) rather than by row shuffling, so that a high R^2 from a low-rank predictor space is flagged as non-identifying.

## Contribution

['ID-Cog, a hierarchical Bayesian individual-differences framework that estimates per-agent latent cognitive profiles from trial-level data and formally tests predictive equivalence between model and human profile variation.', 'A design principle: predictive equivalence at the individual-difference level requires held-out individual prediction, not covariance matching, because covariance can be reproduced by aggregate-fitting models.', 'A trial-level human dataset (N≥500) and model response dataset (54 models × 13 tasks) enabling individual-difference analysis of language model cognition.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ID-Cog human sample (N≥500, 13 tasks, ~60 trials) | Trial-level data needed for latent parameter estimation |
| Primary metric | Held-out individual-level R² on reliable humans (posterior SD ≤ 0.5) | Tests prediction of person, not group mean |
| Baseline | CogBattery condition-mean aggregate | Tests whether condition means suffice |
| Baseline | Per-task averaging (no hierarchical pooling) | Tests whether latent separation matters |
| Baseline | Continuous factor model (no profile mixture) | Tests whether discrete profiles matter |
| Ablation-of-ours | ID-Cog with rotation-based null (orthogonal rotation of model profiles) | Tests whether alignment is real, not an artifact of predictor covariance |
| Diagnostic | Effective rank and condition number of Sigma_M | Flags non-identifying high R² from low-rank predictor space |

### Why this setup validates the claim

The claim is that per-agent latent cognitive profiles estimated from trial-level data carry individual-difference structure that predicts held-out human profiles better than aggregate condition means. The CogBattery condition-mean baseline tests the sub-claim that condition-level aggregation is insufficient: if condition means predicted held-out individuals as well as latent profiles, the central estimand (per-agent d′, c, ConfSep) would be unnecessary. The per-task averaging baseline tests whether hierarchical pooling and discriminability/criterion separation matter, isolating the value of the signal-detection structure. The continuous factor-model baseline tests whether the discrete-profile assumption is doing work, since the paper's mechanism depends on profile membership being a meaningful unit. The rotation-based null tests whether any predictive gain is an artifact of dimensionality or ridge regression rather than genuine model-human alignment: by applying a random orthogonal rotation to model profiles (which preserves the predictor covariance structure), we ensure that a shuffled-pairing baseline cannot yield non-trivial R² simply because the predictor space has low effective rank. We also report the effective rank and condition number of Sigma_M so that a high R² from a low-rank predictor space is flagged as non-identifying. Held-out individual-level R² is the right metric because it directly operationalizes "predicts human-profile variation": aggregate covariance matching could be passed by any model reproducing group means, but predicting a specific held-out person from model profiles cannot be achieved without the structure the method claims to capture. Together these form a falsifiable test: the method must beat condition means specifically on individuals, the gain must vanish under rotation-based null, and the predictor space must have sufficient effective rank.

### Expected outcome and causal chain

**vs. CogBattery condition-mean aggregate** — On a case where two humans share identical condition-level accuracy but differ in false-alarm rate and confidence separation (a high-criterion cautious responder vs. a low-criterion impulsive one), the baseline produces the same predicted profile for both because it collapses trials to condition means and discards criterion information. Our method instead estimates per-agent d′ and c separately from trial-level hits and false alarms, so the two humans land in different latent profiles, and reduced-rank regression from model profiles can distinguish them. We expect a noticeable R² gap on the criterion-separated subset (high-FA vs low-FA humans) but near-parity on the subset where humans are homogeneous in criterion, which is exactly the diagnostic pattern.

**vs. Per-task averaging (no hierarchical pooling)** — On a case where a single task has few trials per condition and noisy individual estimates, the baseline produces unstable per-task d′ estimates that overfit trial noise and yield poor held-out prediction because each human's parameters are estimated in isolation. Our method instead pools information across conditions through the hierarchical prior, shrinking noisy estimates toward task-level means while preserving reliable individual deviations, so held-out profiles are better calibrated. We expect a larger R² gain for humans with high trial-level noise (short or error-prone sessions) and smaller gain for clean responders, with the overall gain concentrated in the noisy tail rather than uniform.

**vs. Continuous factor model (no profile mixture)** — On a case where humans fall into discrete high-FA and low-FA subtypes that differ in within-profile covariance, the baseline produces a single continuous factor solution that averages the two covariance structures and misassigns borderline individuals. Our method instead fits a K=3 latent-profile mixture, so membership is a categorical predictor and within-profile covariance is modeled separately, improving membership accuracy for borderline cases. We expect profile-membership accuracy to exceed the factor baseline by a clear margin on borderline humans while both models tie on unambiguous cases, and the rotation-based null should drop R² to near zero, confirming the gain comes from real alignment.

### What would falsify this idea

If the held-out R² gain over the condition-mean baseline is uniform across criterion-separated and criterion-homogeneous subsets rather than concentrated where latent separation should matter, the per-agent latent profile is not capturing individual-difference structure. Likewise, if the rotation-based null yields the same R² as true pairing, or if the effective rank of Sigma_M is too low to span Sigma_H, the predictive equivalence claim is an artifact of the ridge mapping rather than genuine alignment.

## References

1. CogBattery: Human-Referenced Evaluation of Cognitive Task Behavior and Structure in Language Models
