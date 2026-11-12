# MTMM-CogBattery: Multitrait-Multimethod Construct Validation of Cognitive Batteries for Language Models

## Motivation

CogBattery (Idea Card) compares 54 models against human condition-level references on 13 cognitive tasks, but its own limitations note that predefined cognitive groupings show no predictive advantage over random groupings — meaning the battery never establishes that its task scores actually measure distinct cognitive constructs rather than a single undifferentiated performance factor. The root cause is structural: CogBattery treats each task as a monolithic score and validates only against human effect directions, which conflates trait variance (the construct a task is supposed to measure) with method variance (the paradigm through which it is measured), so convergent and discriminant validity are never separately identifiable. This is the meta-level construct-validity gap that recurs across model-evaluation paper-trees: batteries report scores without a formal measurement model that decomposes trait from method.

## Key Insight

Trait and method variance become separately identifiable only when every cognitive construct is measured by at least two distinct task paradigms and every paradigm is used for at least two constructs, so a correlated-trait-correlated-method (CTCM) factor model fitted jointly to model and human correlation matrices can attribute shared variance to constructs versus paradigms and anchor the method effects to a human reference.

## Method

INPUTS:
  M models (open-weight LLMs), H human reference participants from source studies
  Constructs T = {working memory, monitoring, confidence calibration}
  Paradigms  P = {recognition (old/new), span (serial recall), confidence judgment (post-decision rating)}
  Each (T,P) cell has >=1 observable score; each T seen under >=2 P; each P used for >=2 T

PHASE 1 -- Crossed design construction (ensure MTMM identifiability):
  Build task set so the T x P incidence matrix has no row or column with a single entry.
  For each model m and human h, compute per-(T,P) scores s_{m,t,p} (e.g., d-prime, span length, confidence separation)
  Standardize within (t,p) cell across the M+H units.

PHASE 2 -- Correlation matrices:
  R_models = corr over M units of standardized scores  (M x M-score vector)
  R_humans = corr over H units
  Pool with Fisher z, weighting by n; retain separate R for the human anchor.

PHASE 3 -- CTCM factor model (fitted by maximum likelihood, lavaan-style):
  For unit u and cell (t,p):  s_{u,t,p} = lambda_{t,p} * Trait_{u,t} + Method_{u,p} * gamma_{t,p} + e_{u,t,p}
  Constraints:
    - Trait factors correlated (Phi_T free) -> estimates convergent validity
    - Method factors correlated (Phi_P free) -> captures paradigm-shared variance
    - Trait-Method cross-loadings FIXED to zero (the MTMM identifying restriction)
    - Human anchor: Method factor means/variances estimated in humans, then held fixed when fitting models
      => method effects are defined relative to the human reference, not free in models
  Fit to R_models and R_humans separately, then to the pooled matrix with a group constraint
    that Trait loadings lambda are equal across groups (metric invariance test).

  [EXPLICIT LOAD-BEARING ASSUMPTION]
  The human reference data, after condition-matching to the model task adaptations, carry the same
  paradigm-induced method structure as the model data. This assumption is required for fixing
  method-factor means/variances to the human estimates to yield a valid, non-arbitrary origin for
  method effects and for the model-vs-human metric-invariance test to be interpretable.

  [CALIBRATION / VERIFICATION STORY -- replaces hard anchoring with partial-invariance scheme]
  Step 3a -- Method-factor invariance testing (configural -> metric -> scalar):
    Fit three nested models on the method factors only:
      Configural: method-factor means and variances free in both groups.
      Metric: method-factor loadings gamma_{t,p} constrained equal across groups.
      Scalar: additionally method-factor means constrained equal across groups.
    Compare with chi-square difference tests (alpha = 0.05, Bonferroni across the 3 method factors).
    The human group sets the scale via a marker indicator (first indicator per method factor fixed to 1.0)
    rather than by fixing method means.
  Step 3b -- Direct manipulation check:
    Before any anchoring is applied, test whether the text-adapted paradigm reproduces the human
    method-factor loading pattern: compute the congruence coefficient (Tucker's phi) between the
    human and model method-factor loading vectors for each paradigm; require phi >= 0.90 for anchoring
    to be applied. If phi < 0.90, report the paradigm as method-non-invariant and exclude it from the
    anchored invariance test.
  Step 3c -- Conditional invariance test:
    Report the model-vs-human metric-invariance test on trait loadings lambda ONLY conditional on
    method-factor invariance (Step 3a) holding. If method-factor invariance fails, the trait invariance
    test is reported as confounded and not interpreted as construct non-invariance.

PHASE 4 -- Validity tests (chi-square difference + bootstrapped CIs, 1000 resamples):
  Convergent: lambda_{t,p} > 0 and Trait correlations Phi_T significantly > 0
  Discriminant: for each trait pair, Phi_T < 1 and < the square root of the AVE of each trait
    AVE_t = (sum_{p} lambda_{t,p}^2) / (sum_{p} lambda_{t,p}^2 + sum_{p} Var(e_{t,p}))
    where the denominator includes only the fixed-zero-constrained cells (i.e., the trait-method
    cross-loading variance is set to zero by construction, so AVE is computed on the trait and error
    variance components only).
  Method dominance: compare variance explained by Trait vs Method factors per cell
  Model-vs-human: test whether Phi_T and Phi_P differ between groups (measurement non-invariance),
    conditional on method-factor invariance from Step 3a.

  [POWER / IDENTIFIABILITY ANALYSIS]
  Minimum M for stable ML estimation of the CTCM with 3 traits x 3 paradigms:
    The model has 9 observed variables, 3 trait factors, 3 method factors.
    Free parameters: 9 trait loadings + 9 method loadings + 3 trait variances + 3 method variances
    + 3 trait correlations + 3 method correlations = 30 parameters.
    With 9 observed variables, the covariance matrix has 9*10/2 = 45 unique elements.
    Degrees of freedom = 45 - 30 = 15. For stable ML estimation, require M >= 5 * (number of free parameters)
    / (number of observed variables) = 5 * 30 / 9 ≈ 17. We therefore require M >= 20 models.
    If M < 20, use a Bayesian estimator with weakly informative priors:
      Phi_T ~ LKJ(2), Phi_P ~ LKJ(2), lambda_{t,p} ~ Normal(0, 1), gamma_{t,p} ~ Normal(0, 1),
      error variances ~ Half-Normal(0, 1).

  [SENSITIVITY ANALYSIS FOR ADAPTATION MISMATCH]
  Quantify how much adaptation mismatch in the human anchor shifts the estimated method-factor variance:
    For each paradigm, compute the difference between human and model method-factor loadings (from Step 3b).
    Re-fit the CTCM with method-factor means/variances perturbed by +/- 1 SD of the human estimate and
    record the change in estimated method-factor variance. Report the elasticity (percent change in
    method variance per percent change in anchor mismatch).

OUTPUT: per-construct validity verdict, method-effect magnitudes, model-human invariance report

**Why this design.** We chose a crossed T x P design with at least two paradigms per construct over CogBattery's one-task-per-construct layout because a single paradigm per construct makes trait and method variance mathematically inseparable, accepting the cost that we must re-administer overlapping paradigms and cannot reuse all 13 original tasks unchanged. We fixed the trait-method cross-loadings to zero rather than estimating them freely because free cross-loadings let the model absorb any correlation into a method factor and destroy discriminant-validity interpretation, accepting the cost that a misspecified zero constraint will show up as poor fit and force model revision. We anchored method-factor means to the human reference rather than estimating them independently in models because an unanchored method factor is only identified up to an arbitrary origin, accepting the cost that human source-study data must be condition-matched to the model task adaptations. We chose a confirmatory CTCM over an exploratory PCA (which CogBattery used) because PCA cannot attribute variance to named constructs, accepting the cost that we must pre-register the construct-paradigm mapping. A domain expert would not describe this as a variant of standard factor analysis applied to a new domain: the mechanism is the human-anchored method-effect constraint plus the crossed incidence requirement, which together make trait and method variance separately estimable -- a property that neither CogBattery's random-grouping comparison nor an unconstrained EFA provides.

**Why it measures what we claim.** The trait-factor loading lambda_{t,p} measures convergent validity of construct t because the assumption that tasks sharing a construct share only trait variance (method factors orthogonal to traits by the fixed zero cross-loading) makes a positive loading equivalent to construct convergence; this assumption fails when two paradigms for the same construct also share surface format, in which case the loading reflects format similarity rather than the construct. The trait correlation Phi_T measures discriminant validity because the assumption that distinct constructs are not perfectly collinear makes Phi_T < 1 evidence of separability; this assumption fails when the battery genuinely taps one general factor, in which case Phi_T near 1 reflects a real unitary ability rather than a measurement artifact. The human-anchored method-factor variance measures paradigm-induced method effect because the assumption that human data carry the same paradigm structure as model data makes the human estimate a valid origin; this assumption fails when text adaptation changes the paradigm's demand profile, in which case the anchored method effect reflects adaptation mismatch. The group-invariance test on lambda measures whether models and humans share a measurement model because the assumption of metric invariance makes equal loadings evidence of comparable construct structure; this assumption fails when models solve tasks by a different route, in which case equal loadings reflect coincidental score alignment.

  [ADDITIONAL EQUIVALENCE -- zero trait-method cross-loading]
  The claim that trait loading lambda measures convergent validity assumes the zero trait-method cross-loading is correctly specified. The assumption A that closes it is that no genuine trait-method interaction exists (paradigms do not differentially engage constructs). The failure mode F is that a real interaction is forced into the method factor, inflating method variance and deflating lambda. When F occurs, lambda reflects the trait variance net of the forced interaction, not the construct convergence.

Hyperparameters (inline in Phase 3-4): estimator = maximum likelihood with robust SEs; missing cells = full-information ML; bootstrap = 1000 resamples; invariance alpha = 0.05 with Bonferroni across trait pairs; method-factor invariance alpha = 0.05 with Bonferroni across 3 method factors; congruence threshold phi >= 0.90 for anchoring; minimum M = 20 for ML, else Bayesian with LKJ(2) priors on Phi_T and Phi_P.

## Contribution

(1) A crossed trait-by-paradigm MTMM validation protocol (MTMM-CogBattery) with a human-anchored CTCM factor model that makes trait and method variance separately identifiable for LLM cognitive batteries. (2) A measurement-invariance test that determines whether model and human task scores share a common factor structure, yielding a per-construct verdict on which CogBattery constructs survive formal discriminant validation. (3) An open construct-paradigm incidence design and scoring pipeline that any cognitive battery can adopt to report convergent/discriminant validity instead of raw task scores.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Crossed T×P battery (3 traits × 3+ paradigms, models + human anchor) | Enables trait/method variance separation via incidence balance |
| Primary metric | CTCM trait-loading magnitude & invariance Δχ² | Directly tests convergent/discriminant validity, not raw accuracy |
| Baseline 1 | CogBattery one-task-per-construct + EFA | Standard practice; cannot separate trait from method variance |
| Baseline 2 | Unconstrained EFA on crossed battery | Same data, no zero cross-loading or human anchor |
| Baseline 3 | CTCM with free trait-method cross-loadings | Tests whether zero-constraint is what buys identifiability |
| Ablation-of-ours | CTCM with free method-factor means in models (no human anchor) | Isolates contribution of human anchoring to invariance test |
| Additional baseline | CTCM with method-factor scaling via marker variables (no human anchor) | Tests whether human anchor is necessary vs. alternative identification |
| Additional baseline | Bayesian CTCM with LKJ(2) priors on Phi_T and Phi_P (if M < 20) | Provides stable estimation when M is small |

### Why this setup validates the claim

The central claim is that human-anchored method-factor constraints plus a crossed trait×paradigm incidence matrix make trait and method variance separately estimable, yielding interpretable construct-validity verdicts for LLMs. The primary metric — CTCM trait loadings and the metric-invariance Δχ² between model and human groups — is the right detector because it reads off the exact quantity the claim asserts is identifiable: a positive, stable trait loading net of paradigm-shared variance. Baseline 1 (CogBattery's one-task-per-construct + EFA) tests the sub-claim that single-paradigm designs confound trait and method: if our crossed design is necessary, Baseline 1 should show trait loadings that collapse or invert once method variance is partialled. Baseline 2 (unconstrained EFA on the same crossed data) tests whether the zero cross-loading restriction is what buys interpretability, not the data collection alone. Baseline 3 (free cross-loadings) tests whether the identifying restriction is load-bearing. The ablation (free method means, no human anchor) tests whether anchoring — not merely fitting a CTCM — is what makes the invariance test meaningful. The additional baseline with marker-variable scaling tests whether the human anchor is necessary or whether any scale-setting strategy suffices. The Bayesian CTCM baseline provides a stable estimator when M < 20, as determined by the power analysis (minimum M = 20 for ML). Together these isolate each design choice's contribution, so a positive result cannot be attributed to "factor analysis on a new domain" generically.

### Expected outcome and causal chain

**vs. CogBattery one-task-per-construct + EFA** — On a case where the working-memory span task and the recognition task both load on a shared surface format (e.g., both require holding a list), the baseline produces a single dominant factor because its one-task-per-construct layout gives no within-construct, cross-paradigm replication to separate trait from format variance. Our method instead estimates a trait loading net of the paradigm method factor because the crossed incidence matrix provides the replication needed to identify both, so we expect a noticeable drop in apparent trait loading for constructs whose paradigms share format, with trait loadings that survive only where paradigms are format-diverse.

**vs. Unconstrained EFA on crossed battery** — On a case where two constructs' paradigms correlate through a shared method factor, the unconstrained EFA produces a factor that mixes trait and method variance because it has no zero cross-loading restriction to forbid trait-method absorption. Our method instead fixes trait-method cross-loadings to zero, forcing method-shared variance into a separate factor, so we expect cleaner discriminant validity (trait correlations Phi_T further from 1) and a fit penalty concentrated exactly on cells where trait and method are genuinely confounded.

**vs. CTCM with free trait-method cross-loadings** — On a case where a method factor can absorb any residual correlation, the free-cross-loading baseline produces near-zero discriminant evidence because the model can route trait-method covariance into a method factor. Our method instead imposes the identifying zero constraint, so we expect higher trait loadings and Phi_T values that are estimable and bounded away from 1, with the fit cost of the constraint appearing as localized misfit rather than global collapse.

**vs. CTCM with method-factor scaling via marker variables (no human anchor)** — On a case where the human anchor is not necessary, this baseline should produce trait loadings and invariance Δχ² similar to our method. If our method shows higher trait loadings or better discriminant validity, the human anchor is doing the work; if not, the anchor is not load-bearing.

**vs. Bayesian CTCM with LKJ(2) priors (if M < 20)** — On a case where M is small, the ML estimator may fail to converge or yield unstable estimates. The Bayesian CTCM should provide stable estimates with credible intervals; we expect similar point estimates but wider intervals, confirming that the power analysis correctly identifies the minimum M.

### What would falsify this idea

If trait loadings and invariance Δχ² are essentially identical across our anchored crossed CTCM, the unconstrained EFA, and the free-cross-loading CTCM — i.e., the gains are uniform rather than concentrated where method variance is predicted to confound — then the human-anchored crossed design is not doing the work the claim attributes to it, and the central claim is wrong. Additionally, if the method-factor invariance test (Step 3a) fails (i.e., method-factor loadings or means are non-invariant across groups), then the human anchor is invalid and the invariance Δχ² conflates construct and method non-invariance, falsifying the interpretability claim.

## References

1. CogBattery: Human-Referenced Evaluation of Cognitive Task Behavior and Structure in Language Models
