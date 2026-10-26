# Anchor-Linked Calibration for Commensurable Agent and World-Model Evaluation

## Motivation

Dual Exam reports an Agent Index and a WM Overall that are computed on disjoint task suites (1,088 agent items across 15 categories vs. 400 world-model cases across eight scenarios) with different aggregation metrics, so the two scores live on incomparable scales and no calibration-invariance property can hold; the 150-case coupled pilot's zero task successes is exactly the symptom that component scores on separate scales cannot predict joint competence. The root cause is structural: Dual Exam inherited WorldPrediction's single-protocol measurement model and layered a second track on top without ever establishing a shared latent scale, so the field has no principled way to say whether an agent at Agent Index 31.3 is 'as competent' as a world model at WM Overall 21.7, nor how either maps to coupled success.

## Key Insight

If a small set of anchor tasks is jointly solvable by both agents and world models, then the monotone linking function estimated from those shared anchors is the unique transformation that places both tracks on a common latent competence scale, making cross-track and cross-system comparisons invariant to the arbitrary raw-score metrics of each track.

## Method

**What it is.** ALC (Anchor-Linked Calibration) is a post-hoc measurement layer that takes as input the per-item correctness matrices of the Dual Exam Agent track and WM track plus a set of jointly-solvable anchor tasks, and outputs a linked competence estimate θ̂ for every system on a single interval scale, together with a commensurability certificate (residual misfit, anchor-invariance diagnostics, and DIF-rejection fraction) that states whether the two tracks are certifiably on the same scale.

**Load-bearing assumption (stated explicitly).** ALC assumes that the jointly-solvable anchor tasks measure the same latent competence construct in both the agent and world-model protocols, i.e., that anchor items are free of differential item functioning (DIF) across the two protocols, so that equipercentile linking on anchor scores yields a valid common scale rather than a scale built on protocol-specific shortcut variance. If this assumption fails — for example, a WM anchor solvable by an agent via surface-token pattern-matching — the linking residual reflects protocol-specific shortcut variance rather than true scale divergence, and a shortcut that is consistent across anchors can produce low residuals that look like good linking. To make this assumption testable rather than circular, ALC screens anchors for DIF before linking (Phase 0) and reports the fraction of anchors rejected for DIF as part of the commensurability certificate; if no overlapping calibration sample can be constructed to run the DIF screen, the linking is unidentified and ALC should not be applied.

**How it works.**

```text
INPUTS:
  A = agent item-response matrix  (systems x 1,088 items, 0/1)
  W = WM item-response matrix     (systems x 400 cases, 0/1)
  Anchor set K = {k_1..k_m} jointly solvable by both tracks (m ~ 60-120), each
      administered in BOTH protocols with matched content but protocol-specific
      surface form; K is partitioned by difficulty stratum (5 strata).
  Overlapping calibration sample C: a small set of systems (n_C ~ 20-30) that
      take BOTH the agent and WM protocols on the anchor items only, used for
      DIF screening (Phase 0). If |C| < 20, the linking is unidentified and ALC
      is not applied.

PHASE 0  DIF screening of anchors (new, makes the load-bearing assumption testable)
  For each anchor k, using the overlapping calibration sample C:
    - Fit a 2PL model to anchor responses separately for the agent and WM
      protocols (same EM settings as Phase 1).
    - Compute Lord's chi-square statistic for the difference in item parameters
      (a_k, b_k) between protocols; df = 2.
    - Compute Raju's area measure between the two item characteristic curves.
    - Flag anchor k as DIF if Lord's chi-square p < 0.05 (Bonferroni-corrected
      for m anchors) OR Raju's area exceeds 1.0.
  Retain only DIF-free anchors for linking; let m' be the retained count.
  Report DIF-rejection fraction = (m - m') / m.
  If m' < 40, the linking is unstable and ALC reports a NON-COMMENSURABLE
  certificate with reason 'insufficient DIF-free anchors'.

PHASE 1  Fit per-track 2PL models (separate, no linking yet)
  For track t in {A, W}:  P(correct | θ, b_i, a_i) = sigma(a_i (θ - b_i))
  Estimate item discriminations a_i and difficulties b_i by marginal MLE
  (EM, 200 iterations, convergence on log-likelihood delta < 1e-4).

PHASE 2  Equipercentile linking on DIF-free anchors
  For each track, compute the anchor score distribution S_t(k) over the m'
  retained anchors.
  Build the equipercentile map  phi_{A->W}:  raw_A -> raw_W  by matching
  percentile ranks of S_A and S_W (linear interpolation between anchor points,
  log-linear smoothing of the score distributions with 3 passes).
  Convert to latent-scale map:  L(θ_A) = theta_W_scale( phi_{A->W}( raw_A(θ_A) ) ).

PHASE 3  Cross-track consistency residual
  For each retained anchor k, compute delta_k = | theta_A(k) - L^{-1}(theta_W(k)) |.
  Report RMS(delta_k) as the linking residual and per-stratum residuals.
  Flag a stratum as NON-COMMENSURABLE if its residual exceeds 2x the pooled SD.
  Additionally, flag anchors with high discrimination (a_k > 1.5 in either
  protocol) but low cross-protocol agreement (|delta_k| > 2x pooled SD) as
  'shortcut-suspect' and report their count separately.

PHASE 4  Coupled-success calibration check
  Regress pilot coupled success (150 cases) on (theta_A, theta_W) jointly;
  report whether the linked scale yields a monotone, better-fitting predictor
  than the two raw indices (Spearman rho, AIC comparison).

OUTPUT: linked theta-hat per system, linking residual, DIF-rejection fraction,
commensurability certificate.
```

**Why this design.** We chose equipercentile linking with a 2PL latent backbone over a plain linear rescaling of the raw indices because the two tracks' raw metrics are case-weighted averages with different item difficulty distributions, so a linear map would only be valid if the two score distributions were identical in shape — a condition the Dual Exam data visibly violate; the cost is that equipercentile linking requires the anchor set to be large enough (m ≥ ~60) for stable percentile estimation, which we accept by drawing anchors from five difficulty strata. We chose a *jointly-solvable* anchor set over reusing the existing 150-case coupled pilot as the link because the pilot contains a single agent–world-model pairing and zero successes, giving no usable score variance for linking; the cost is that anchors must be constructed and administered, which we accept as a one-time benchmark-authoring expense. We chose to report a per-stratum residual certificate rather than a single scalar commensurability score because a single scalar hides the failure mode we most want to detect (one difficulty stratum where the tracks diverge); the cost is a more complex reporting surface. We added a DIF-screening phase (Phase 0) using a small overlapping calibration sample because the load-bearing assumption — that anchors measure the same construct in both protocols — is otherwise untestable and the residual certificate alone cannot detect shortcut-consistent anchors; the cost is that a small overlapping sample (n_C ~ 20-30) must be collected, which we accept as a bounded addition to the one-time authoring expense. A domain expert would not describe ALC as a variant of standard test-equating because standard equating assumes a *common population* taking both forms, whereas here the two tracks are taken by disjoint system populations and only the anchor items are shared — the linking function must therefore be estimated from anchor-only overlap, screened for DIF, and validated by the residual certificate, which is the structural property that distinguishes ALC from off-the-shelf equipercentile equating.

**Why it measures what we claim.** The linking residual RMS(delta_k) measures cross-track commensurability because it assumes the anchor tasks are drawn from the same latent competence construct as the full tracks and are DIF-free across protocols (construct-validity assumption, now screened in Phase 0); this assumption fails when an anchor is solvable by a shortcut unique to one protocol (e.g., a world-model anchor that an agent can solve by pattern-matching surface tokens), in which case the residual reflects protocol-specific shortcut variance rather than true scale divergence — which is why Phase 0 reports the DIF-rejection fraction and Phase 3 flags shortcut-suspect anchors separately. The linked θ̂ measures a system's position on a shared latent competence scale because it assumes the 2PL monotonicity and local-independence conditions hold within each track (IRT assumption); this fails when items within a reporting category are near-duplicates, in which case θ̂ reflects category-specific memorization rather than general competence. The coupled-success regression coefficient measures whether the linked scale predicts joint success because it assumes the 150 pilot cases are representative of the anchor difficulty range; this fails when the pilot is concentrated at one difficulty level, in which case the regression reflects range restriction rather than genuine predictive validity.

## Contribution

(1) ALC, an anchor-based linking-and-certification layer that maps Dual Exam's Agent Index and WM Overall onto a shared latent competence scale via equipercentile linking over jointly-solvable anchor tasks, with a per-stratum commensurability certificate that flags where the two tracks are not certifiably on the same scale. (2) A design principle: cross-track comparability in dual-track benchmarks is an *identifiability* property that requires a shared anchor set, not a reporting convention — demonstrated by showing that raw-index comparisons are not invariant under anchor-set resampling while linked comparisons are. (3) An open anchor-task specification and linking-residual reporting protocol that any dual-track benchmark can adopt to certify commensurability of its component scores.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Dual Exam Agent (1,088 items) + WM (400 cases) + 60–120 jointly-solvable anchors + overlapping calibration sample of 20–30 systems on anchor items only | Two disjoint-population tracks with shared anchor overlap; calibration sample enables DIF screening |
| Primary metric | Linking residual RMS(delta_k) per difficulty stratum, plus DIF-rejection fraction | Directly measures cross-track commensurability and tests the load-bearing DIF-free assumption |
| Baseline | Raw-index linear rescaling | Tests whether nonlinear equipercentile map is necessary |
| Baseline | Single-scalar commensurability score | Tests whether per-stratum certificate detects divergence |
| Baseline | Unlinked separate theta estimates | Tests whether linking adds predictive value at all |
| Baseline | Mean/sigma and Haebara/Stocking-Lord IRT linking (standard disjoint-population linking designs) | Tests whether anchor-only equipercentile linking is required where standard linking fails |
| Ablation-of-ours | ALC without anchor stratification (pooled anchors) | Tests whether stratum partition drives residual detection |
| Ablation-of-ours | ALC without DIF screening (all anchors retained) | Tests whether DIF screening is necessary to avoid shortcut-consistent low residuals |

### Why this setup validates the claim

The central claim is that the two tracks can be certifiably placed on one interval scale via anchor-only overlap, and that a per-stratum residual plus DIF screening is what makes that certification trustworthy. Each baseline targets a distinct sub-claim. Raw-index linear rescaling tests whether the score-distribution shape mismatch actually matters — if a linear map already yields low residuals, equipercentile linking is unmotivated. The single-scalar commensurability score tests whether per-stratum reporting is necessary — if a pooled scalar flags the same divergences, the certificate's complexity is unjustified. Unlinked separate thetas test whether linking buys any predictive gain on the 150-case coupled pilot, isolating the value of the shared scale from the value of the 2PL backbone. The mean/sigma and Haebara/Stocking-Lord baselines test whether standard disjoint-population linking designs, which assume a common population or known anchor parameters, fail under the Dual Exam's disjoint-population, protocol-specific-surface-form condition, thereby motivating anchor-only equipercentile linking. The ablation (pooled anchors, no stratification) isolates whether the stratum partition is the mechanism that surfaces a localized divergence rather than averaging it away. The DIF-screening ablation isolates whether the load-bearing assumption must be tested rather than assumed: if removing DIF screening produces low residuals on anchors that are later shown to be shortcut-suspect, the certificate without DIF screening is circular. The primary metric — per-stratum residual RMS plus DIF-rejection fraction — is the right detector because the predicted failure mode is a *localized* stratum divergence or a shortcut-consistent anchor set; a global accuracy or correlation metric would average that signal into noise, whereas residual RMS per stratum and DIF-rejection fraction keep it visible and falsifiable.

### Expected outcome and causal chain

**vs. Raw-index linear rescaling** — On a case where the WM track's difficulty distribution is skewed toward harder items than the Agent track, the linear baseline produces systematically compressed linked thetas for mid-range systems because a linear map cannot absorb the shape difference in the two score distributions. Our method instead fits an equipercentile map through matched anchor percentiles because it matches ranks rather than assuming equal shape, so we expect a visibly lower linking residual concentrated in the middle difficulty strata, with parity at the extremes.

**vs. Single-scalar commensurability score** — On a case where one difficulty stratum (say the hardest) diverges because a WM anchor is solvable via surface-token shortcut, the single-scalar baseline produces a pooled residual that stays below its flag threshold because the divergent stratum is diluted across five strata. Our method instead reports per-stratum residuals because it partitions anchors by difficulty, so we expect the single-scalar approach to declare commensurability while ALC flags exactly one stratum as non-commensurable — a diagnostic disagreement, not just a lower number.

**vs. Unlinked separate thetas** — On a case where a system is strong on Agent items but weak on WM items, the unlinked baseline produces two incomparable indices that cannot be jointly regressed meaningfully because the scales are not aligned. Our method instead places both on one linked scale via the anchor map, so we expect a monotone, better-fitting joint predictor of coupled success (higher Spearman rho, lower AIC) than the two raw indices — with the gain largest for systems whose two raw indices disagree.

**vs. Mean/sigma and Haebara/Stocking-Lord IRT linking** — On a case where the two tracks are taken by disjoint system populations and anchors have protocol-specific surface form, the mean/sigma baseline assumes a common population and produces biased linked thetas because the population means and SDs are not comparable; the Haebara/Stocking-Lord baselines assume known anchor parameters and fail when anchor parameters are estimated from disjoint populations. Our method instead estimates the linking function from anchor-only overlap after DIF screening, so we expect lower linking residual and higher coupled-success prediction than these standard designs, with the gap largest when the DIF-rejection fraction is non-zero.

**vs. ALC without DIF screening (ablation)** — On a case where a subset of anchors is shortcut-suspect (high discrimination but low cross-protocol agreement), the no-DIF-screening ablation retains those anchors and produces low residuals that falsely certify commensurability because the shortcut is consistent across anchors. Our method instead screens anchors in Phase 0 and reports the DIF-rejection fraction, so we expect the ablation to show lower residual but higher coupled-success prediction error, while ALC flags the shortcut-suspect anchors and yields a more honest certificate.

### What would falsify this idea

If the per-stratum residual certificate shows uniform residuals across all five strata (no stratum-specific divergence) while the single-scalar baseline and the stratified ALC agree on every system, then the per-stratum machinery adds nothing and the central commensurability claim collapses to a global scalar. Equivalently, if the linked scale fails to beat unlinked separate thetas on coupled-success prediction, or if DIF screening rejects a large fraction of anchors (m' < 40) such that the linking is unstable, the linking itself — not just its reporting granularity — is unsupported.

## References

1. Dual Exam: Controlled Evaluation of Multimodal Agents and World Models
2. WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning
