# Latent Bias–Indeterminacy Decomposition for LLM-Judge Validation

## Motivation

Existing validation of LLM-as-a-judge systems reduces the entire question of judge reliability to a scalar agreement statistic — AltRA-Test compares paired all-human and mixed-panel IAA and declares non-inferiority, while indeterminacy-aware validation (E5/E11) aggregates forced-choice human ratings into a single gold label before measuring agreement. Both structurally conflate two distinct sources of disagreement: judge-specific bias (self-preferencing, positional bias, per E3) and panel-composition/rating-indeterminacy noise (E5, E11), because a scalar IAA has no degrees of freedom to separate them. The field therefore cannot answer whether a judge is biased or merely evaluated against an indeterminate panel — a question that recurs across the AltRA-Test and indeterminacy-validation branches and that no scalar agreement test can resolve.

## Key Insight

When human and LLM annotator sets are disjoint, the cross-panel covariance structure of ratings carries enough constraint (via a low-rank bias factor anchored to the LLM's own outputs and an indeterminacy factor anchored to human rating-set multiplicity) that judge-specific bias and panel-composition noise become separately identifiable up to a scale rotation, which a single scalar IAA statistic discards by construction.

## Method

**What it is.** LADDER (Latent Annotator Disjoint Decomposition for EvaluatoR validation) takes as input (i) a corpus of items each rated by a disjoint set of human annotators and a disjoint set of LLM judges, (ii) each LLM judge's own generated outputs for the same items, (iii) multi-label "response set" ratings per E5/E11, and (iv) a small overlapping calibration subset of items rated by at least one human and at least one LLM judge (size N_cal = 256 items by default; see Phase 2b). It outputs a per-judge bias vector, a per-item indeterminacy variance, a panel-composition noise estimate, and a decomposed agreement score.

**Load-bearing assumption (stated explicitly).** The rank-2 bias factor b_a and the indeterminacy term g_i are separately identifiable from cross-panel covariance only when the rotation between them is pinned by an overlapping calibration subset. With purely disjoint panels, g_i is rater-type-dependent (E5/E11) and collinear with the rater-type contrast that b_a captures; the unit-norm and orthogonality constraints on f_i and b_a are normalization choices, not identifying restrictions. LADDER therefore does NOT claim disjoint-panel identifiability from covariance alone. Instead, the calibration subset provides the anchor that fixes the rotation, and the method reports a sensitivity check on N_cal (Phase 2b).

**How it works.**

```pseudocode
INPUT: items i=1..N; human raters h in H_i (|H_i|=k); LLM judges j in J_i (disjoint from H_i);
       judge outputs y_ji; multi-label allowed-rating sets S_hi, S_ji; rank R=2 (bias dim) + 1 (indeterminacy);
       overlapping calibration subset C of size N_cal=256 (items rated by >=1 human and >=1 LLM judge)

PHASE 1 — Build the rating tensor
  For each (i, rater) encode rating as a vector over the ordinal scale (one-hot for forced choice;
  multi-hot normalized for response-set ratings).
  Assemble R_{i,a} for a in H_i ∪ J_i.

PHASE 2 — Specify the generative factor model (fixed, not learned end-to-end)
  R_{i,a} = mu_i + b_a^T f_i + g_i * eps_{i,a} + delta_a * 1[a in J] * s_{i,a}
    mu_i        : item difficulty (fixed effect)
    b_a, f_i    : rank-R bias factor loadings (rater) and item bias coordinates
    g_i         : per-item indeterminacy std (log-parameterized, g_i = exp(eta_i))
    delta_a     : judge-specific self-preference coefficient on own-output indicator s_{i,a}
  Identifiability constraints: ||f_i||=1 per item; b_a orthogonal to the all-ones vector;
  delta_a identified ONLY from judges who also rated items where their own output is absent (anchor set).

PHASE 2b — Calibration anchor for the bias/indeterminacy rotation
  On the overlapping calibration subset C (N_cal=256 items, each rated by >=1 human and >=1 LLM judge),
  estimate the rotation matrix Q (2x2) that aligns the bias-factor coordinates f_i estimated from
  human-only items with those estimated from LLM-only items. Q is estimated by Procrustes alignment
  of the per-item factor scores on C, then held fixed when scoring held-out items.
  Sensitivity: repeat with N_cal in {64, 128, 256, 512, 1024} and report the stability of delta_j
  and ||b_j|| across these sizes (target: coefficient of variation < 0.15 across the three largest).

PHASE 3 — Estimate by penalized marginal likelihood
  Optimize (mu, b, f, eta, delta) with L2 penalty lambda=1e-2 on b and delta via L-BFGS;
  marginalize eps analytically (Gaussian); indeterminacy g_i enters as heteroskedastic noise.
  Use 5-fold item cross-fitting so f_i for held-out items is predicted from b only.
  The rotation Q from Phase 2b is applied to f_i before computing the likelihood on held-out items.

PHASE 4 — Decompose observed agreement
  For each judge j: bias_j = ||b_j||, selfpref_j = delta_j.
  Panel noise = mean_i g_i^2 over items rated by both panels.
  Decomposed agreement = 1 - (bias_j^2 + panel_noise) / total_variance.
  Variance-additivity assumption: total_variance = bias_j^2 + panel_noise + residual_variance,
  where residual_variance is the mean of eps_{i,a}^2; this holds by construction of the Gaussian
  heteroskedastic model in Phase 2, and fails when residual variance is itself rater-type-dependent
  (in which case decomposed_agreement absorbs that dependence into panel_noise).

PHASE 5 — Calibration check (NOT terminal statistic)
  Run AltRA-Test bootstrap non-inferiority on the SAME panels; record whether its scalar verdict
  is consistent with the sign of (bias_j - panel_noise). Report concordance rate only.

OUTPUT: (bias_j, selfpref_j, g_i, panel_noise, decomposed_agreement, calibration_concordance,
         rotation_stability_cv)
```

**Why this design.** We chose a fixed low-rank generative factor model over a learned neural disentanglement head because the whole point is identifiability, not flexibility: a learned head can always fit the data while asserting an independence the data does not contain (the classic anti-pattern), whereas a rank-2 factor model with explicit orthogonality and anchor-set constraints makes the separation a property of the parameterization. We chose multi-label response-set ratings over forced-choice elicitation because E5/E11 show forced-choice aggregation injects indeterminacy noise directly into the gold label, which would confound exactly the term we are trying to isolate, accepting the cost that response-set data is harder to collect. We chose to run AltRA-Test only as a calibration concordance check rather than as the validation statistic because AltRA-Test's bootstrap IAA comparison (the named prior method) is precisely the scalar statistic whose inability to separate composition from bias motivates this work; a domain expert would not describe LADDER as a variant of AltRA-Test, since LADDER never compares two IAA scalars but instead estimates a covariance decomposition and uses AltRA-Test only to check that the two verdicts agree. We chose analytic marginalization of the residual over MCMC because the Gaussian heteroskedastic form admits closed-form marginal likelihood, accepting the cost that non-Gaussian rating scales require the one-hot encoding in Phase 1. We added the overlapping calibration subset (Phase 2b) rather than claiming disjoint-panel identifiability from covariance alone, because E5/E11 show indeterminacy is rater-type-dependent and E1 bounds what any debiasing can recover at the frontier; the calibration subset is the smallest concrete anchor that pins the bias/indeterminacy rotation, and its size is varied as a sensitivity check rather than assumed sufficient.

**Why it measures what we claim.** The bias magnitude ||b_j|| measures judge-specific bias because the orthogonality constraint on b_a and the item-level unit-norm constraint on f_i make the bias factor the only component that can covary with rater identity across items while remaining orthogonal to item difficulty; this assumption fails when a judge's bias is perfectly correlated with item difficulty (e.g., a judge that only errs on hard items), in which case ||b_j|| reflects difficulty sensitivity rather than bias. The self-preference coefficient delta_j measures self-preferencing because it is identified only from the anchor set of items where the judge's own output is absent, so the own-output indicator is not collinear with the judge loading; this assumption fails when the anchor set is too small or unrepresentative, in which case delta_j absorbs general leniency. The per-item indeterminacy variance g_i^2 measures panel-composition/indeterminacy noise because multi-label response sets directly encode the multiplicity of reasonable ratings, so residual variance beyond the bias factor is attributable to indeterminacy; this assumption fails when raters systematically misinterpret the scale, in which case g_i^2 reflects scale confusion rather than genuine indeterminacy. The rotation matrix Q from Phase 2b measures the bias/indeterminacy rotation because it is estimated by Procrustes alignment on the overlapping calibration subset where both rater types are present; this assumption fails when the calibration subset is not exchangeable with the held-out items (e.g., calibration items are systematically easier), in which case Q reflects calibration-set selection rather than a stable rotation, and the sensitivity check on N_cal is designed to expose this.

## Contribution

(1) LADDER, a latent-variable judge-validation framework that replaces scalar IAA comparison with an identifiability-constrained rank-2 factor decomposition separating judge-specific bias from panel-composition/indeterminacy noise under disjoint annotator sets. (2) A design principle: multi-label response-set ratings plus an own-output anchor set are jointly necessary for the bias and self-preference terms to be identified, whereas forced-choice elicitation makes them collinear with indeterminacy. (3) A calibration-concordance diagnostic that quantifies how often AltRA-Test's scalar non-inferiority verdict agrees with the sign of the decomposed bias-minus-noise contrast, exposing when the scalar test masks bias.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-label response-set rating corpora with disjoint human/LLM panels (E5/E11) plus a 256-item overlapping calibration subset | Contains indeterminacy the decomposition must isolate and anchors the rotation |
| Primary metric | Decomposed agreement vs. observed agreement gap | Directly tests whether bias is separated from indeterminacy |
| Baseline | AltRA-Test IAA non-inferiority bootstrap | Scalar panel test our decomposition must outperform diagnostically |
| Baseline | Forced-choice agreement aggregation | Standard validation that E5/E11 show conflates indeterminacy |
| Baseline | Rank-1 bias-only factor model | Isolates whether rank-2 + anchor identification matters |
| Baseline | Off-the-shelf multi-facet latent model (two-way random-effects with rater and item facets) | Tests whether anchor-set constraint is the novel identifiable ingredient |
| Ablation-of-ours | LADDER without anchor-set constraint on delta | Tests whether self-preference identification is load-bearing |
| Ablation-of-ours | LADDER without calibration subset (pure disjoint-panel rotation) | Tests whether the calibration anchor is load-bearing |
| Ablation-of-ours | LADDER with N_cal in {64, 128, 256, 512, 1024} | Tests stability of delta_j and ||b_j|| across anchor size |

### Why this setup validates the claim

The central claim is that a fixed low-rank factor model with explicit orthogonality, anchor-set constraints, and a calibration-subset rotation can separate judge bias, self-preference, and indeterminacy noise where scalar agreement statistics cannot. Each baseline targets one sub-claim. AltRA-Test tests whether a scalar IAA verdict is genuinely insufficient: if LADDER's decomposed agreement tracks a known bias injection while AltRA-Test's verdict stays flat or misattributes it, the motivation is confirmed rather than assumed. Forced-choice aggregation tests the E5/E11 indeterminacy claim: since forced-choice gold labels absorb indeterminacy into the target, a method relying on them should show inflated panel noise and depressed decomposed agreement precisely on high-indeterminacy items. The rank-1 model tests whether the second bias dimension plus anchor identification is necessary; if rank-1 matches LADDER, our parameterization is over-engineered and the identifiability argument fails. The off-the-shelf multi-facet latent model tests whether the anchor-set constraint is the novel identifiable ingredient: if a generic two-way random-effects model with rater and item facets matches LADDER, then the disjoint-panel covariance constraints are not doing the work we claim. The anchor ablation tests whether delta_j is identified by the anchor set or merely absorbs leniency. The calibration-subset ablation tests whether the rotation anchor from Phase 2b is load-bearing: if removing it leaves delta_j and ||b_j|| unchanged, the load-bearing assumption is not actually load-bearing. The N_cal sensitivity sweep tests whether the rotation is stable across anchor sizes. The primary metric is the gap between decomposed and observed agreement, because that gap is exactly the quantity the claim says is recoverable — a method that merely re-labels observed agreement cannot manufacture it.

### Expected outcome and causal chain

**vs. AltRA-Test IAA bootstrap** — On a case where a judge panel is composed of one self-preferring judge and several neutral judges, AltRA-Test produces a single non-inferiority verdict that cannot localize the composition problem, because its mechanism is a scalar IAA comparison with no per-judge decomposition. Our method instead estimates per-judge bias and self-preference loadings from the anchor set, so we expect a clear per-judge bias spread on the mixed panel with near-parity on homogeneous panels, and calibration concordance that drops exactly when the scalar verdict and the bias-minus-noise sign disagree.

**vs. Forced-choice agreement aggregation** — On a case where items admit multiple reasonable ratings, forced-choice aggregation produces a gold label that collapses the response set into one rating, so indeterminacy variance leaks into the residual and inflates panel noise, because its mechanism has no representation for rating multiplicity. Our method instead encodes multi-label response sets in Phase 1, so g_i captures genuine indeterminacy separately from bias, and we expect the decomposed-agreement gap to widen on high-indeterminacy items while staying flat on low-indeterminacy items.

**vs. Rank-1 bias-only factor model** — On a case where a judge both self-prefers and has a systematic leniency tilt, the rank-1 model produces a single conflated bias magnitude because it has no anchor-identified self-preference term, so the two effects collapse into one loading. Our method instead separates delta_j from b_j via the anchor set, so we expect the rank-1 model to show a bias estimate that is uniformly inflated relative to LADDER, with the discrepancy concentrated on judges whose anchor sets are large and whose own-output rate is high.

**vs. Off-the-shelf multi-facet latent model** — On a case where the judge panel is disjoint from the human panel, a generic two-way random-effects model with rater and item facets has no constraint linking the rater-type contrast to the item-difficulty contrast, so it cannot separate bias from indeterminacy and its per-rater estimates are not comparable across panels. Our method instead imposes the orthogonality and unit-norm constraints plus the calibration-subset rotation, so we expect the multi-facet model to show higher variance in per-judge bias estimates and no stable delta_j, while LADDER's estimates remain stable across the N_cal sweep.

**vs. LADDER without anchor-set constraint on delta** — On a case where a judge's own-output rate is high, removing the anchor-set constraint makes delta_j collinear with b_j, so the model cannot tell self-preference from general bias. Our method instead identifies delta_j only from items where the judge's own output is absent, so we expect the ablated model to show delta_j estimates that are inflated and correlated with b_j, while LADDER's delta_j remains stable.

**vs. LADDER without calibration subset** — On a case where the human and LLM panels are fully disjoint, removing the calibration subset leaves the bias/indeterminacy rotation unidentified, so the model's f_i estimates are only defined up to an arbitrary rotation and ||b_j|| and g_i are not separately interpretable. Our method instead pins the rotation via Procrustes alignment on the 256-item calibration subset, so we expect the ablated model to show high variance in ||b_j|| and g_i across random seeds, while LADDER's estimates are stable.

### What would falsify this idea

If the decomposed-agreement gap is uniform across high- and low-indeterminacy items, if the anchor ablation changes delta_j by no more than estimation noise, and if the calibration-subset ablation leaves ||b_j|| and g_i unchanged, then the claimed separation is not happening and LADDER is merely re-expressing observed agreement. Likewise, if rank-1 matches LADDER on all subsets or if the off-the-shelf multi-facet model matches LADDER on per-judge bias stability, the rank-2 identifiability argument and the anchor-set novelty claim are wrong.

## References

1. AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators
