# Rank-Calibrated Sensorimotor Alignment: A Monotone Proxy for Policy Ordering under Noisy Preferences

## Motivation

Sensorimotor Alignment (S²) provides a training-free scalar proxy for embodied policy quality, but it is validated only by empirical correlation with success and lacks monotone consistency under preference aggregation — the exact property RoboArena requires to derive valid rankings from noisy pairwise comparisons. This is a structural gap: S²'s score is an uncalibrated alignment magnitude, so its induced ordering can invert when preferences are aggregated across heterogeneous tasks, making it unsafe as a ranking device despite its correlation. The field's trajectory toward distributed, preference-based evaluation (RoboArena) and toward cheap pre-screening (S²) thus exposes a new bottleneck: no proxy has been calibrated to be order-preserving under the aggregation operator that the evaluation protocol actually uses.

## Key Insight

Isotonic calibration of the S² alignment score against aggregated pairwise preferences yields a monotone map whose induced policy ordering is, by construction, the maximum-likelihood ordering under the Bradley–Terry preference model — so ranking validity follows from the calibration constraint rather than from correlation.

## Method

**What it is.** Rank-Calibrated Sensorimotor Alignment (RC-S²) takes as input (i) raw S² scores for a set of policies and (ii) a set of noisy pairwise preference comparisons aggregated over tasks/environments (as in RoboArena), and outputs a calibrated scalar score per policy together with a total ranking. The calibrated score is a monotone transform of S², so it preserves S²'s cheap offline computation while acquiring order-consistency under preference aggregation.

**Load-bearing assumption (stated explicitly).** RC-S² assumes that the Bradley–Terry strengths theta_i fitted on the calibration comparisons are a sufficiently noise-free target that isotonic regression of S² onto them yields a map g whose induced ordering generalizes to held-out tasks — i.e., that g(s_i) is order-consistent with held-out BT strengths rather than merely reproducing the calibration-set MLE. This assumption is load-bearing because theta_i has variance inversely proportional to comparison degree; PAV will systematically pool and reorder exactly the low-degree policies the method claims to help, so the fitted g can be a projection of noise rather than of quality. The method below is specified so that this assumption is testable (Phase 3 audit) and so that the shrinkage claim is what the estimator actually implements (Phase 2 fits theta jointly with a monotone S² term inside the BT likelihood, rather than fitting g to a fixed theta).

**How it works.**

```pseudocode
Input: policies P = {p_1..p_n}; raw S² scores s_i for each p_i;
       pairwise comparisons C = {(a,b, y_ab)} where y_ab in {a wins, b wins};
       held-out task split T_ho; isotonic link family (nondecreasing, piecewise-constant).

Phase 1 — Preference aggregation model (baseline BT).
  Fit Bradley–Terry strengths theta_i by maximizing
      L(theta) = sum_{(a,b) in C} log sigma( theta_a - theta_b )
  with sigma the logistic link; ridge penalty 1e-3 on theta.
  This is the same aggregation operator RoboArena uses; theta_i is the
  maximum-likelihood ordering under noisy preferences. Used as the
  evaluation target in Phase 3, NOT as the fixed target of Phase 2.

Phase 2 — Joint monotone-constrained BT with S² shrinkage.
  Fit theta_i and a nondecreasing map g: R -> R jointly by penalized
  maximum likelihood over the comparison graph:
      maximize_{theta, g}  sum_{(a,b) in C} log sigma( theta_a - theta_b )
                           - (1/(2*tau^2)) * sum_i ( theta_i - g(s_i) )^2
                           - (ridge 1e-3) * ||theta||^2
  subject to g nondecreasing (piecewise-constant, knots at observed s_i).
  Implementation: alternate (i) PAV update of g given theta (pool-adjacent-
  violators on pairs (s_i, theta_i) sorted by s_i) and (ii) gradient ascent
  on theta given g (logistic BT gradient plus shrinkage term), 200
  iterations, step size 0.1, tau^2 = 1.0 (fixed; no tuning).
  g is piecewise-constant; its breakpoints are the calibration knots.
  Number of PAV blocks is determined by the data (no tuning); smoothing
  parameter lambda=0 (pure isotonic) is used within each PAV update.
  This makes S² shrink low-degree theta estimates toward the S² ordering
  (the shrinkage claim the estimator implements), rather than fitting g
  to a fixed noisy theta.

Phase 3 — Ranking and monotonicity audit.
  Calibrated score: r_i = g(s_i) from the Phase-2 fit.
  Ranking: sort by r_i. Because g is nondecreasing, r_i >= r_j iff s_i >= s_j
  whenever g is strictly increasing on the relevant block; ties are broken by theta_i.
  Monotonicity audit on held-out tasks T_ho:
    - Refit BT strengths theta_i^ho on T_ho WITHOUT S² (Phase-1 procedure).
    - Spearman rho between r_i and theta_i^ho.
    - Sufficiency test: does adding s_i to theta_i^ho improve held-out preference
      log-likelihood? (nested likelihood-ratio test with 1 df).
    - Inversion count: number of pairs where r_i > r_j but theta_i^ho < theta_j^ho.
    - Low-degree subset: repeat rho and inversion count restricted to policies
      in the bottom quartile of comparison degree.

Output: calibrated scores r_i, ranking, and audit statistics.
```

**Why this design.** We chose isotonic regression over a parametric (e.g., linear or logistic) calibration because isotonic regression is the *minimal* monotone assumption — it does not impose a functional form that could itself be misspecified, accepting the cost that the calibrated score is piecewise-constant and coarse when n is small. We chose to fit theta jointly with a monotone S² term inside the BT likelihood (rather than calibrating g to a fixed BT MLE) because the BT MLE is itself estimated with error that is correlated with comparison degree, and fitting g to it would project noise rather than shrink it; the cost is that the joint fit requires alternating optimization rather than a single closed-form PAV pass. We chose PAV (exact isotonic solver) as the inner update over a learned monotone network because PAV gives a deterministic, reproducible map with no optimization hyperparameters, accepting the cost that it cannot extrapolate beyond the observed S² range. A domain expert would not describe this as "S² + calibration" in the sense of Platt scaling on a classifier, because the target of calibration is not a probability but the *ordering* induced by a preference-aggregation MLE; the matched prior method is standard probability calibration (Platt/isotonic on classifier scores), and our method differs in that the calibration constraint is placed on the *ranking functional* rather than on a per-item probability.

**Why it measures what we claim.** The calibrated score r_i = g(s_i) measures *policy quality under aggregated preferences* because the joint fit forces g to be nondecreasing and shrinks theta_i toward g(s_i) inside the BT likelihood, so r_i reflects the S²-anchored estimate of BT strength; this equivalence holds under the assumption that preferences are generated by a BT model with policy-specific strengths and that S² is positively associated with those strengths. This assumption fails when evaluators' preferences are driven by factors orthogonal to S² (e.g., visual aesthetics unrelated to task success), in which case r_i reflects the projection of S² onto the preference-identifiable subspace rather than true task quality. The held-out sufficiency test operationalizes the claim that r_i is a *sufficient statistic* for ranking: it measures whether S² adds information beyond the BT strengths on held-out tasks, under the assumption that held-out tasks are exchangeable with calibration tasks; this assumption fails under distribution shift across embodiments, in which case the sufficiency test reflects in-distribution sufficiency only.

## Contribution

['A calibration layer, RC-S², that maps the training-free S² alignment score to a monotone calibrated score via isotonic regression against Bradley–Terry preference strengths, yielding a ranking that is order-consistent with aggregated noisy preferences by construction.', 'A held-out monotonicity-and-sufficiency audit protocol (inversion count, Spearman correlation with held-out BT strengths, nested likelihood-ratio test) that operationalizes ranking validity for any scalar proxy, not just S².', 'An empirical demonstration on RoboArena-style pairwise preference data that the calibrated proxy preserves policy ordering where raw S² inverts, and that S² is a sufficient statistic for preference-based ranking only within the calibration task distribution.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | RoboArena pairwise evals + S² scores on DROID; plus a synthetic BT-generated benchmark with known ground-truth strengths (n=50 policies, comparison graph with degree distribution matched to RoboArena, 10 seeds) | Matches the aggregation operator and modality the method targets; synthetic benchmark verifies audit statistics (inversion count, LRT) behave correctly before committing to noisier real data |
| Primary metric | Held-out Spearman rho between r_i and held-out BT strengths theta_i^ho (refit on T_ho without S²) | Directly measures order-consistency the calibration claims to add; refitting theta^ho without S² removes circularity |
| Secondary metric | Inversion count (pairs where r_i > r_j but theta_i^ho < theta_j^ho), reported overall and on the low-degree subset (bottom quartile of comparison degree) | Detects the errors-in-variables failure mode the load-bearing assumption warns about |
| Secondary metric | Sufficiency LRT statistic (1 df) for adding s_i to theta_i^ho on held-out preference log-likelihood | Operationalizes the sufficiency claim |
| Baseline | Raw S² ranking | Tests whether calibration adds order information at all |
| Baseline | Bradley–Terry strengths alone (Phase-1 theta_i) | Tests whether S² is sufficient/orthogonal to preferences |
| Baseline | Platt-scaled S² to win rates | Tests the ranking-functional vs probability-calibration distinction |
| Ablation-of-ours | RC-S² with linear (non-isotonic) link in Phase 2 | Tests whether the monotone-only assumption matters vs parametric form |
| Ablation-of-ours | RC-S² with point-target isotonic (original Phase-2: fit g to fixed Phase-1 theta_i) | Tests whether the joint shrinkage formulation is what produces the gain, vs fitting to a noisy target |
| Power analysis | On synthetic benchmark, compute minimum detectable rho difference (alpha=0.05, power=0.8) for n=50 policies and the near-tied subset size observed in RoboArena | Ensures the experiment is not underpowered on the near-tied subset |

### Why this setup validates the claim

The central claim is that a monotone calibration of S² onto Bradley–Terry strengths yields a score that is order-consistent with aggregated noisy preferences while retaining S²'s offline cheapness. Raw-S² ranking is the baseline that tests the sub-claim that calibration is necessary: if raw S² already matches held-out BT ordering, the calibration is doing nothing. BT-strengths-alone tests the sufficiency sub-claim: if BT alone already achieves the same held-out rho, then S² carries no orthogonal ranking signal and the whole premise of projecting S² onto the preference subspace collapses. Platt-scaling-to-win-rates tests the design-choice sub-claim: it calibrates a per-item probability rather than the ranking functional, so if it matches RC-S², our distinction is cosmetic. The linear-link ablation isolates whether the *minimal* monotone assumption is what produces the gain, or whether any parametric fit would do. The point-target-isotonic ablation isolates whether the joint shrinkage formulation (Phase 2) is what produces the gain, or whether fitting g to a fixed noisy theta would suffice — directly testing the load-bearing assumption. Spearman rho against held-out BT strengths refit without S² is the right metric because the claim is about *ordering*, not absolute score values, and refitting removes the circularity of calibrating onto the same MLE used as the evaluation target; an inversion count or sufficiency LRT would also detect it, but rho is the single number that captures rank agreement and degrades smoothly with the inversion pattern we predict. The synthetic BT benchmark with known ground-truth strengths verifies that the audit statistics (inversion count, LRT) behave correctly before committing to noisier real data, and the power analysis ensures the near-tied subset is not underpowered.

### Expected outcome and causal chain

**vs. Raw S² ranking** — On a case where two policies have nearly identical S² but one consistently wins pairwise comparisons because S²'s proxy misses a preference-relevant factor, raw S² produces a tie or the wrong order because it has no mechanism to absorb preference information. Our method instead fits a nondecreasing map g jointly with theta inside the BT likelihood, which shifts the tied block's scores according to the BT strengths, because the shrinkage term (1/(2*tau^2)) * sum_i (theta_i - g(s_i))^2 pulls theta toward the S² ordering and PAV redistributes mass within blocks to match theta ordering. We expect a modest but consistent rho gain concentrated on the near-tied S² subset, with parity on well-separated policies.

**vs. Bradley–Terry strengths alone** — On a case where the preference graph is sparse and unbalanced (few comparisons for some policies), BT alone produces a noisy strength estimate with high variance because the MLE has little data to pin down those policies. Our method instead anchors those noisy theta estimates to the S² ordering via the joint shrinkage term, because S² provides an independent, cheap signal that shrinks the estimated ordering toward a stable one. We expect the largest rho advantage on low-degree policies (bottom quartile of comparison degree), with near-parity on heavily-compared policies where BT is already well-determined.

**vs. Platt-scaled S² to win rates** — On a case where win rates are non-monotone in S² within a subset (a mid-S² policy beats both higher and lower ones), Platt scaling produces a sigmoidal fit that cannot represent the local inversion and forces a monotone probability curve that misranks the middle policy, because it calibrates a per-item probability with a fixed functional form. Our method instead lets PAV place a piecewise-constant block boundary exactly at the inversion, because the isotonic constraint is on the ranking functional and can absorb the local non-monotonicity as a block tie broken by theta. We expect a visible rho gap on the subset containing local inversions and parity elsewhere.

**vs. point-target isotonic (ablation)** — On a case where low-degree policies have high-variance theta_i, point-target isotonic fits g to those noisy estimates and systematically pools and reorders exactly the low-degree policies, because PAV treats theta_i as noise-free. Our joint shrinkage formulation instead pulls theta_i toward g(s_i) inside the likelihood, so the low-degree estimates are shrunk toward the S² ordering rather than the map being fitted to their noise. We expect the joint formulation to show a larger rho advantage on the low-degree subset and a lower inversion count there, with near-parity on high-degree policies.

### What would falsify this idea

If RC-S²'s rho gain over raw S² is uniform across all subsets rather than concentrated on the near-tied, low-degree, and locally-inverted subsets where the mechanism predicts it should act, then the calibration is not doing what we claim and the gain (if any) comes from something other than preference-subspace projection. Equivalently, if the linear-link ablation or the point-target-isotonic ablation matches the joint shrinkage formulation, the minimal-monotone-assumption and shrinkage arguments are wrong.

## References

1. Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success
2. RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies
3. Robot Learning as an Empirical Science: Best Practices for Policy Evaluation
4. π0: A Vision-Language-Action Flow Model for General Robot Control
5. BridgeData V2: A Dataset for Robot Learning at Scale
6. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
7. RT-1: Robotics Transformer for Real-World Control at Scale
