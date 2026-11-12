# ShiftBench: A Distribution-Shift Stress-Test Suite for Calibrated Ensemble-Routing Surrogates

## Motivation

Existing ensemble-routing surrogates, such as the tractable ensemble-success surrogate in PISA-E, are validated only on four structurally similar LLM routing benchmarks, so their claimed benchmark-agnostic generalization is never actually tested against adversarial, out-of-distribution, or synthetic distribution shifts. This failure is structural: the surrogate is fit to a fixed cluster/capability representation whose calibration is never audited under shift, so a surrogate can score well on held-out splits of the same benchmarks while silently degrading when the prompt distribution moves — exactly the regime a router must survive in deployment.

## Key Insight

A surrogate's generalization is only meaningful when its predicted ensemble-success ranking is calibrated against realized success under an explicit, measurable distribution shift, because ranking calibration — not raw accuracy — is the invariant that transfers across shift families.

## Method

**What it is.** ShiftBench is a benchmark-construction and calibration protocol for ensemble-success surrogates. Inputs: a routing benchmark with prompts, candidate LLMs, and per-prompt success labels; a shift generator family. Outputs: a shift-stratified evaluation suite and a calibrated surrogate whose predicted ensemble-success ranks are aligned with realized success under each shift family.

**How it works.**
```pseudocode
# Phase 1: shift family construction
for each base prompt p in benchmark:
    S_adv(p)   = adversarial paraphrase + distractor injection (k=5 variants)
    S_ood(p)   = cross-domain transplant (domain classifier >= 0.9 confidence)
    S_syn(p)   = synthetic shift: length x2, language swap, format scramble
suite = {ID, S_adv, S_ood, S_syn}

# Phase 2: surrogate stress-test
for each shift family f, each prompt p in suite[f]:
    y_hat[p] = surrogate_ensemble_success(p, candidate_ensembles)
    y[p]     = realized success of top-ranked ensemble on p
    rank_corr[f] = Spearman(y_hat, y) over prompts
    ECE[f]       = expected calibration error of y_hat vs y

# Phase 3: calibration mechanism (temperature-scaled isotonic recalibration)
# Assumption A (explicit): the distortion induced by shift family f on y_hat is a
# monotone (order-preserving) function of the surrogate's logits, and this distortion
# is stationary from the held-out ID fitting slice to the shifted prompts of family f.
# We verify A per family with a monotonicity diagnostic before applying the map.
for each shift family f:
    # Monotonicity diagnostic: bin y_hat into 10 equal-frequency bins on held-out ID slice;
    # compute mean realized success per bin; fit isotonic regression to bin means;
    # report monotonicity index = 1 - (SSE_isotonic / SSE_constant), in [0,1].
    # If index < 0.5, flag family as non-monotone (assumption A fails).
    fit isotonic regression on held-out ID slice -> g_f
    fit temperature T_f on held-out ID slice -> monotone map
    calibrated y_hat_cal = g_f( sigmoid(logit(y_hat)/T_f) )
    re-evaluate rank_corr and ECE on suite[f]
    # Non-monotone fallback: if monotonicity index < 0.5, replace g_f with a
    # pairwise ranking loss (RankNet) trained on the same held-out ID slice,
    # which can represent non-monotone distortions; report both variants.
```

**Why this design.** We chose a *stratified* shift suite (adversarial / OOD / synthetic) over a single pooled shift set because pooling hides which shift family breaks the surrogate, accepting the cost of a larger evaluation matrix. We chose *rank correlation plus ECE* over raw accuracy because routing only consumes the surrogate's ordering of candidate ensembles, so ranking calibration is the quantity that actually transfers, accepting that accuracy alone would overstate generalization. We chose *per-family isotonic + temperature recalibration* over a single global calibrator because each shift family induces a different monotone distortion of the surrogate's logits, accepting the cost of fitting separate calibrators on held-out ID data. This differs from PISA-E, whose surrogate is fit once and never recalibrated under shift; a domain expert would not describe ShiftBench as a variant of PISA-E because the contribution is a shift-stratified calibration protocol, not a new router.

**Why it measures what we claim.** Spearman rank correlation between predicted and realized ensemble success measures *benchmark-agnostic ranking generalization* because it assumes the surrogate's ordering is the only property routing consumes; this assumption fails when two ensembles have near-tied predicted scores but very different realized costs, in which case the metric reflects ordering noise rather than true ranking fidelity. Expected calibration error measures *surrogate confidence reliability under shift* because it assumes realized success is a well-defined Bernoulli draw per prompt; this assumption fails when prompts in a synthetic shift are near-duplicates, in which case ECE reflects prompt-clustering artifacts rather than miscalibration. Per-family isotonic recalibration measures *shift-specific monotone distortion* because it assumes the distortion is monotone in the surrogate's score and stationary from the ID fitting slice to the shifted prompts; this assumption fails under adversarial shifts that invert the ranking or under non-stationary distortion, in which case recalibration reflects only a monotone correction and the residual rank drop is the true adversarial damage. The monotonicity index (1 - SSE_isotonic/SSE_constant over 10 equal-frequency bins) measures *whether the monotone-distortion premise holds per family* because it assumes binned realized success is a reliable estimate of the conditional mean; this assumption fails when bins contain fewer than 30 prompts, in which case the index reflects binning noise rather than true monotonicity.

## Contribution

['ShiftBench, a distribution-shift benchmark suite with three explicit shift families (adversarial, out-of-distribution, synthetic) that stress-tests ensemble-success surrogates beyond the four structurally similar routing benchmarks used by PISA-E.', 'A per-shift-family isotonic-plus-temperature calibration protocol that restores rank correlation and reduces expected calibration error of the ensemble-success surrogate under each shift family, establishing ranking calibration as the transferable invariant for benchmark-agnostic routing surrogates.', 'An empirical analysis identifying which shift family most degrades surrogate ranking fidelity, providing a diagnostic that prior routing benchmarks cannot produce.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ShiftBench suite (ID, S_adv, S_ood, S_syn), capped at 2k prompts per family | Stratified shifts expose which family breaks ranking |
| Primary metric | Spearman rank correlation + ECE per shift family | Routing consumes ordering, not raw accuracy |
| Secondary metric | Monotonicity index per family (1 - SSE_isotonic/SSE_constant over 10 equal-frequency bins) | Directly tests the monotone-distortion premise |
| Baseline | PISA-E (uncalibrated surrogate) | Tests whether recalibration under shift is necessary |
| Baseline | Global single-calibrator surrogate | Tests whether per-family calibration beats pooling |
| Baseline | Fixed ensemble / single-model router | Tests whether ensemble-surrogate ranking matters at all |
| Ablation-of-ours | ShiftBench minus per-family recalibration | Isolates contribution of stratified calibration |
| Ablation-of-ours | ShiftBench with non-monotone fallback (RankNet) on families flagged by monotonicity index < 0.5 | Tests whether fallback recovers rank fidelity where isotonic cannot |

### Why this setup validates the claim

The central claim is that a surrogate's predicted ensemble-success ordering degrades unevenly across shift families, and that per-family isotonic-plus-temperature recalibration recovers ranking fidelity where the distortion is monotone and stationary. The design makes this falsifiable by pairing each baseline with a specific sub-claim. PISA-E tests the sub-claim that an uncalibrated surrogate is insufficient under shift: if PISA-E already ranks as well as ShiftBench on S_adv and S_ood, recalibration adds nothing. The global-calibrator baseline tests the sub-claim that distortion is family-specific: if one pooled calibrator matches per-family calibration, stratification is wasted effort. The fixed-ensemble baseline anchors whether the surrogate's ordering matters at all relative to ignoring routing. The ablation removes recalibration while keeping the stratified suite, isolating whether gains come from the calibration mechanism or merely from measuring on harder prompts. The monotonicity index is the diagnostic that grounds the load-bearing assumption: it is computed per family on held-out ID data before any recalibration, so a family with index < 0.5 is explicitly flagged as violating the monotone-stationary premise, and the non-monotone fallback ablation tests whether a pairwise ranking loss (RankNet) can recover rank fidelity there. Spearman rank correlation is the right primary metric because routing only consumes the surrogate's ordering of candidate ensembles; ECE is secondary because confidence reliability matters only insofar as it distorts that ordering. Together these form a test where the predicted effect (concentrated rank-correlation gains on the shift families whose distortion is monotone, and fallback gains on those flagged non-monotone) is distinguishable from a uniform improvement that would indicate the suite itself, not calibration, is doing the work.

### Expected outcome and causal chain

**vs. PISA-E (uncalibrated surrogate)** — On a case where an OOD prompt is transplanted from a different domain, PISA-E's surrogate, fit once on in-distribution data, produces confidently misordered ensemble scores because its logits were never exposed to the domain shift's monotone distortion. Our method instead applies the S_ood-family isotonic-plus-temperature map fit on held-out ID data, because the distortion is monotone in the surrogate's score and stationary from the ID fitting slice to the shifted prompts, so we expect a clear rank-correlation gap on S_ood and S_syn but near-parity on ID and on adversarially inverted prompts (where the monotonicity index is expected to fall below 0.5 and the non-monotone fallback is invoked).

**vs. Global single-calibrator surrogate** — On a case where a synthetic-shift prompt is length-doubled and format-scrambled, the global calibrator produces a correction tuned to the pooled shift distribution, because it cannot represent family-specific monotone distortions and averages them away. Our method instead fits a separate calibrator per shift family, because each family induces a distinct monotone distortion of the logits, so we expect per-family calibration to beat the global calibrator on S_syn and S_ood while tying on ID, with the gap widening as families diverge.

**vs. Fixed ensemble / single-model router** — On a case where the optimal ensemble changes between an ID prompt and its adversarial paraphrase, the fixed baseline produces the same routing decision for both because it never consults a per-prompt success surrogate. Our method instead re-ranks candidate ensembles per prompt via the calibrated surrogate, because it consumes the predicted ordering that the baseline ignores, so we expect a large rank-correlation and realized-success advantage on S_adv, with the fixed baseline collapsing to a constant prediction that our metric flags as zero-variance.

### What would falsify this idea

If the rank-correlation gain from per-family recalibration is uniform across ID, S_adv, S_ood, and S_syn rather than concentrated on the families whose distortion is monotone (monotonicity index ≥ 0.5), the claim that stratified calibration targets family-specific distortion is wrong. Likewise, if the ablation without recalibration matches the full method on every family, the calibration mechanism—not the stratified suite—is not doing the work.

## References

1. Ensemble Routing for Prompt-Adaptive Cost-Aware LLM Inference
