# Shift-Aware Operator Certification for LLM AutoML Agents

## Motivation

LLM AutoML agents select search operators and refine solutions using validation scores that implicitly assume exchangeability between validation and deployment folds; MLE-STAR (E6) refines via ablation studies on held-out splits, and AI Research Agents (E5) rank search policies by MLE-bench medal rates, but neither certifies that these rankings survive distribution shift. The PHM deployment benchmark (E7) shows that unit-held-out folds can invert rankings and that a single evaluation convention can halve reported error, so the proxy used to select operators and validate ablations is structurally unsound whenever deployment units differ from validation units. The root cause is that operator/ablation rankings are estimated as point estimates under an exchangeability assumption that the deployment regime violates.

## Key Insight

Operator and ablation rankings are only meaningful if their order is stable across shift-induced resampling of the evaluation units; therefore, a rank-stability certificate computed over unit-held-out folds is a necessary condition for any search-policy or refinement gain to transfer, and this condition is estimable without access to the hidden deployment units.

## Method

```pseudocode
Input: task T, base LLM agent A, operator library O = {o_1..o_k},
       evaluation unit partition U = {u_1..u_m} (e.g., engines, patients, sites),
       fold budget B_folds (default 5, capped at m), stability threshold tau (default 0.8),
       coverage threshold gamma (default 0.5)
Output: certified operator ranking R* (or UNCERTIFIED), per-operator stability scores S,
        refinement gate G, coverage diagnostic C

# (D) Proxy-to-concept equivalences and their closing assumptions:
#   D1. Copeland pairwise win-rate p_ij measures deployment operator dominance
#       because unit-held-out folds are assumed to be i.i.d. draws from the deployment
#       shift distribution with full coverage of hidden-unit shift space (assumption A1).
#       A1 fails when the unit partition is confounded with a nuisance variable (e.g.,
#       unit size correlating with label noise) or hidden units lie outside U's shift
#       support; then p_ij reflects fold-composition artifacts, not operator quality.
#   D2. Bootstrap rank variance S[o] measures rank stability because fold resampling
#       is assumed to approximate the deployment-unit sampling process (assumption A2).
#       A2 fails when folds are not independent (overlapping units); then S[o] reflects
#       within-fold correlation, not shift sensitivity.
#   D3. Sign-consistency gate measures ablation robustness because a refinement whose
#       effect sign is invariant across unit-held-out folds is assumed to preserve that
#       sign on hidden units (assumption A3). A3 fails when hidden units occupy a region
#       of shift space not represented in U; then the gate certifies local, not global,
#       robustness.

1. Build shift-aware fold set F:
   for b in 1..B_folds:
       F_b = { (train = T \ u_i, val = u_i) for u_i in U }   # unit-held-out folds
   # each fold holds out a whole evaluation unit, not a random row split

2. Coverage diagnostic C (conditions certification on A1):
   # C estimates whether U spans the deployment shift space.
   # (a) Unit-embedding dispersion: compute a fixed 64-dim summary vector v(u_i)
   #     per unit (mean of standardized per-unit feature statistics: mean, std,
   #     skew, missing-rate over the unit's rows).
   # (b) Meta-fold simulation: hold out one unit u_j from U, fit a k-NN (k=3)
   #     distance from v(u_j) to the remaining units; record d_j.
   # (c) C = fraction of units whose d_j <= median(d) * 2.0  (bracketing check).
   # If C < gamma, emit UNCERTIFIED and skip steps 3-5; log coverage warning.

3. For each operator o in O and each fold F_b:
       run A with operator set {o} on F_b.train, evaluate on F_b.val
       record score s[o][b]   # e.g., normalized medal rate / AUC

4. Rank-stability estimation (conditional on C >= gamma):
   for each pair (o_i, o_j):
       p_ij = fraction of folds where s[o_i][b] > s[o_j][b]
       # pairwise win-rate across folds
   build pairwise dominance matrix P = [p_ij]
   R* = Copeland ranking of P     # aggregate pairwise wins; ties broken by mean score
   S[o] = 1 - (variance of rank(o) across 1000 bootstrap resamples of folds)
   # stability = inverse rank variance under fold resampling

5. Refinement gate for ablation-guided refinement (MLE-STAR-style), conditional on C >= gamma:
   for each candidate refinement r proposed by A:
       compute ablation delta d[r][b] = score(with r) - score(without r) on fold F_b
       if sign(d[r][b]) is consistent across >= tau fraction of folds:
           accept r
       else:
           reject r and log proxy-inversion warning

6. Return R*, S, accepted refinements, coverage diagnostic C, and a per-operator
   certificate cert[o] = (rank, S[o], n_folds_consistent, C)
```

**Why this design.** We chose unit-held-out folds over random row splits because the PHM benchmark (E7) demonstrates that unit-level shift is the regime where rankings invert, accepting the cost that fold count is limited by the number of available units. We chose a Copeland ranking over a mean-score ranking because pairwise win-rates are invariant to score-scale shifts across folds, accepting the cost that Copeland can produce ties that require a secondary tie-break. We chose bootstrap rank-variance as the stability score over a single held-out estimate because variance across fold resamples directly operationalizes the exchangeability violation, accepting the cost that stability estimates are noisy when m is small. We chose a sign-consistency gate for ablations rather than a magnitude threshold because sign flips are the specific failure mode that E7 reports (a single convention halving reported error), accepting the cost that small-but-consistent refinements pass even when their effect size is modest. We added the coverage diagnostic C because the certificate is only meaningful when U spans the deployment shift space (assumption A1); C makes that precondition explicit and testable, accepting the cost that C is a heuristic bracketing check rather than a guarantee. A domain expert would not describe this as a variant of MLE-STAR: MLE-STAR refines against a fixed ablation split, whereas this method treats the ablation split itself as the object under test and certifies it against shift.

**Why it measures what we claim.** The Copeland pairwise win-rate p_ij measures operator dominance under shift because it assumes that unit-held-out folds are exchangeable samples of the deployment shift distribution (A1); this assumption fails when the unit partition is confounded with a nuisance variable (e.g., unit size correlating with label noise) or when hidden units lie outside U's shift support, in which case p_ij reflects fold-composition artifacts rather than operator quality. The bootstrap rank variance S[o] measures rank stability because it assumes that fold resampling approximates the deployment-unit sampling process (A2); this assumption fails when folds are not independent (overlapping units), in which case S[o] reflects within-fold correlation rather than shift sensitivity. The sign-consistency gate measures ablation robustness because it assumes that a refinement whose effect sign is invariant across unit-held-out folds will preserve that sign on hidden units (A3); this assumption fails when hidden units occupy a region of shift space not represented in U, in which case the gate certifies local rather than global robustness. The coverage diagnostic C measures whether U spans the deployment shift space because it assumes that unit-embedding dispersion and meta-fold distance bracket the hidden-unit regime; this assumption fails when the 64-dim summary vector omits the shift dimension that matters for deployment, in which case C overstates coverage and certification proceeds on an unrepresentative U.

## Contribution

['A shift-aware evaluation protocol for LLM AutoML agents that replaces point-estimate operator/ablation rankings with unit-held-out Copeland rankings and bootstrap rank-stability certificates.', 'A refinement gate that rejects ablation-guided refinements whose effect sign is not consistent across unit-held-out folds, directly targeting the proxy-inversion failure mode reported in deployment-style benchmarks.', 'A per-operator certificate format that couples search-policy selection and refinement validation to the deployment regime rather than to an exchangeable validation split.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | NextUnit-Bench (24 datasets, 58 unit-held-out folds) | Unit-level shift regime where rankings invert |
| Primary metric | Copeland rank agreement with hidden-unit operator ranking (Kendall tau) | Directly tests ranking stability under shift |
| Secondary metric | Coverage diagnostic C vs. hidden-unit rank agreement (correlation) | Tests whether C predicts transfer |
| Baseline | Mean-score operator ranking | Tests whether pairwise dominance beats averaging |
| Baseline | MLE-STAR fixed-split ablation gate | Tests whether shift-aware certification beats fixed-split refinement |
| Baseline | AutoML (per E7) | Tests whether agent operators clear the automated-search floor |
| Ablation-of-ours | Random-row-split folds instead of unit-held-out | Isolates the shift-aware fold construction |
| Ablation-of-ours | Certification without coverage gate (C always >= gamma) | Isolates the coverage diagnostic |

### Why this setup validates the claim

The central claim is that certifying operator rankings against unit-held-out shift, rather than against a single fixed split, produces rankings that transfer to hidden deployment units. NextUnit-Bench is the right substrate because E7 shows this is precisely the regime where published conventions and fixed-split designs invert: rankings computed on one split do not survive unit holdout. The primary metric — Kendall tau agreement between our certified Copeland ranking and the ranking observed on hidden units — is the direct operationalization of the claim, not a proxy. Mean-score ranking tests whether pairwise win-rates are actually necessary, or whether simple averaging would suffice; if mean-score agreement matches ours, the Copeland machinery is unjustified. The MLE-STAR fixed-split gate tests whether treating the ablation split as the object under test adds anything over refining against a fixed split. AutoML sets the floor: if our certified operators do not clear it, the operator library is not competitive regardless of ranking quality. The random-row-split ablation isolates the shift-aware fold construction as the causal ingredient: if unit-held-out folds and random-row folds produce the same certified ranking, then shift-awareness is doing no work and the design collapses to a standard cross-validation wrapper. The coverage-gate ablation isolates the coverage diagnostic: if removing C does not change hidden-unit rank agreement, then C is not doing the preconditioning work claimed.

### Expected outcome and causal chain

**vs. Mean-score operator ranking** — On a case where one operator wins on most units but loses badly on a few high-variance units, mean-score ranking places the volatile operator first because averaging lets a few extreme folds dominate. Our method instead aggregates pairwise win-rates via Copeland, so the consistently-dominant operator accumulates wins across most folds and outranks the volatile one. We expect a noticeable gap in hidden-unit rank agreement (Kendall tau) concentrated on datasets with heterogeneous unit difficulty, with near-parity on homogeneous datasets.

**vs. MLE-STAR fixed-split ablation gate** — On a case where a refinement's ablation delta is positive on the fixed split but sign-flips on a held-out unit family, the fixed-split gate accepts the refinement because it never observes the flip. Our method instead requires sign consistency across unit-held-out folds and rejects the refinement, logging a proxy-inversion warning. We expect our accepted refinements to show fewer hidden-unit regressions than MLE-STAR's, with the gap concentrated on refinements whose ablation deltas are small in magnitude and thus vulnerable to sign flips.

**vs. AutoML** — On a case where the operator library contains a strong shift-robust operator that AutoML's default search does not surface, AutoML returns a model near half the best agent's score, as E7 reports. Our method instead ranks that operator highly through unit-held-out pairwise dominance and surfaces it in R*, so the certified pipeline inherits its robustness. We expect our certified pipeline to clear the AutoML floor on most folds, with the largest margins on datasets where unit-level shift is strongest.

**vs. Certification without coverage gate** — On a case where U is narrow (few units, low dispersion) and hidden units lie outside U's shift support, the unconditional certificate would confidently rank operators that fail on hidden units. Our method instead emits UNCERTIFIED when C < gamma, so the certified subset excludes these cases. We expect the coverage-gated certificate to have higher hidden-unit rank agreement on the certified subset than the ungated certificate, with the gap concentrated on datasets where C is low.

### What would falsify this idea

If our certified ranking's agreement with hidden-unit rankings is statistically indistinguishable from mean-score ranking's agreement, and the unit-held-out ablation shows no advantage over random-row-split folds, then the shift-aware machinery is not doing the causal work claimed. A second falsifier: if gains over MLE-STAR's fixed-split gate are uniform across all datasets rather than concentrated on datasets with strong unit-level shift, the mechanism is not shift-awareness but something else. A third falsifier: if the coverage diagnostic C does not correlate with hidden-unit rank agreement, then the preconditioning story is unsupported.

## References

1. Moving Generality from Model to Modeler: LLM Agents for PHM on a Deployment-Style Benchmark
