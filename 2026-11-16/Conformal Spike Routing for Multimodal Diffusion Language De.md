# Conformal Spike Routing for Multimodal Diffusion Language Decoding

## Motivation

StepRefine (E2) routes denoising steps between discrete commitment and latent refinement using an entropy-deviation threshold whose value is an uncalibrated statistic of the specific MDLM trajectory, so the discrete-vs-latent boundary carries no distribution-free risk interpretation and shifts whenever the joint image-text likelihood training reshapes the entropy profile. This is a forest-level problem: the same uncalibrated-trajectory-statistic failure appears whenever a diffusion decoder's internal uncertainty signal is used as a decision gate, and neither Dual Diffusion's cross-modal likelihood estimator (E3) nor StepRefine's refinement controller establishes identifiability of that signal under joint training, leaving every such gate tied to the training distribution it was tuned on.

## Key Insight

Exchangeability of spike magnitudes across held-out denoising trajectories and cross-modal likelihood conditions makes split-conformal quantiles a distribution-free, finite-sample-valid upper bound on the routing error rate, so the discrete-vs-latent boundary becomes a user-chosen risk level rather than a trajectory-specific threshold.

## Method

**What it is.** CASCADE (Conformal Adaptive Spike Routing) is a calibration layer wrapped around an MDLM decoder. Inputs: per-step entropy-deviation spike magnitude s_t, a modality-conditioning descriptor c (text-conditioned vs image-conditioned denoising), and a target risk level α. Output: a binary routing decision r_t ∈ {discrete, latent} whose realized error rate on exchangeable trajectories is ≤ α with probability ≥ 1−α.

**Load-bearing assumption (stated explicitly).** The per-step entropy-deviation spike magnitude s_t is a valid conformal score whose label y_t (hallucinated token) is exchangeable with test-time labels within each (modality, prompt-stratum) group, so that a group-conditional conformal quantile computed on calibration trajectories yields a finite-sample-valid bound on the routing error rate at test time. This assumption requires: (i) the score s_t and label y_t are computed on calibration trajectories drawn from the same prompt distribution as test trajectories; (ii) the label y_t is defined by an independent judge (not by the model's own commitment outcome); (iii) the group partition is defined on a quantity independent of s_t (prompt-level or modality-level strata fixed before score computation); and (iv) within each group, the (s_t, y_t) pairs are exchangeable between calibration and test. When (ii) or (iii) fails, the conformal quantile is self-referential and the coverage guarantee is vacuous; when (iv) fails under distribution shift, the quantile reflects the calibration distribution rather than the test one.

**How it works.**

```pseudocode
# Phase 0: define groups independent of the score
#   c_t = modality_descriptor(step_t)  # 0=text-cond, 1=image-cond
#   p_t = prompt_stratum(prompt)       # fixed before score computation:
#         prompt length bin (short/medium/long by tertiles of calibration prompt lengths)
#         × prompt difficulty bin (easy/hard by tertiles of calibration trajectory-level
#           hallucination rate under the base decoder)
#   group g_t = (c_t, p_t)  # 2 modalities × 3 length bins × 2 difficulty bins = 12 groups

# Phase 1: score collection on held-out calibration split
for each calibration trajectory tau in D_cal (held-out prompts, both modalities):
    for each denoising step t in tau:
        s_t   = entropy_deviation(step_t, window=w)        # same statistic as StepRefine
        y_t   = judge_label(step_t)                        # independent judge, see below
        g_t   = (modality_descriptor(step_t), prompt_stratum(prompt(tau)))

# Phase 2: group-conditional conformal quantiles with explicit abstention
for each group g:
    n_err_g = count( {s_t : y_t=1, g_t=g} )
    if n_err_g >= ceil(1/α):
        q_g = empirical_quantile( {s_t : y_t=1, g_t=g}, level=1-α )
        covered_g = True
    else:
        q_g = -inf   # route all steps in this group to latent
        covered_g = False

# Phase 3: test-time routing
for each denoising step t in test trajectory:
    if not covered_{g_t}:
        r_t = latent   # abstention fallback: always-latent for uncovered groups
    else:
        r_t = latent   if s_t >= q_g   else discrete
    # latent branch: probability-weighted embedding + R=3 continuous refinement rounds (StepRefine machinery)
    # refinement_controller monitors distributional stability and early-stops after R rounds
```

Hyperparameters: window w (default 5 steps), α ∈ {0.05, 0.10, 0.20}, R=3 refinement rounds, prompt length tertiles and difficulty tertiles computed on the calibration split (fixed before test), judge model = GPT-4o with token-level teacher-forced reference decoding (see experiment).

**Why this design.** We chose split-conformal group-conditional quantiles over a single global threshold because the exchangeability guarantee is only valid within a group whose score distribution is homogeneous; pooling modalities would violate exchangeability across the text-conditioned and image-conditioned denoising regimes, accepting the cost that each group requires enough calibration errors to estimate a stable quantile. We chose a fixed prompt-stratum grouping (modality × length tertile × difficulty tertile) over a quantile_bin(s_t) grouping because binning on the score itself makes the group-conditional quantile self-referential: the group is defined by the very statistic being thresholded, so the conditional coverage claim is not a genuine conditional guarantee (Barber et al. 2021). We chose an explicit abstention/fallback rule over a hard q_min floor because degenerate groups with fewer than ceil(1/α) calibration errors would otherwise route everything to latent refinement while still claiming risk ≤ α; the abstention rule reports such groups as 'uncovered' and routes them to always-latent, accepting a small inflation of the realized risk above α in exchange for bounded compute and an honest coverage report. A domain expert would NOT describe CASCADE as StepRefine-with-a-tuned-threshold: StepRefine's threshold is a free parameter fit to one trajectory's statistics, whereas CASCADE's boundary is a conformal quantile whose coverage is guaranteed by exchangeability, not by tuning — the calibration layer is a statistical object, not a hyperparameter search.

**Why it measures what we claim.** The computational quantity s_t (entropy deviation from the temporal window) measures hallucination risk at step t because the named assumption is that high entropy deviation from neighboring steps is monotonically associated with token-commitment errors; this assumption fails when a legitimate semantic shift (e.g., a rare but correct visual attribute) also produces a spike, in which case s_t reflects distributional surprise rather than error. The group descriptor g_t = (c_t, p_t) measures modality- and prompt-conditional exchangeability because the assumption is that trajectories sharing a modality descriptor and prompt stratum are exchangeable with the calibration set; this fails when test prompts are out-of-distribution relative to calibration prompts within a stratum, in which case the conformal quantile reflects the calibration distribution rather than the test one. The conformal quantile q_g measures the calibrated risk level α because the assumption is that calibration and test errors are exchangeable within group g and that the label y_t is an independent judge's error indicator; this fails under distribution shift or label noise, in which case q_g reflects the calibration error rate rather than the test error rate. The routing decision r_t measures the discrete-vs-latent boundary at risk α because the assumption is that the latent branch's refinement corrects any error that the discrete branch would have made; this fails when refinement itself hallucinates, in which case r_t reflects a trade-off between two error sources rather than a guaranteed correction. The abstention flag covered_g measures whether the group-conditional guarantee is non-vacuous because the assumption is that a group with at least ceil(1/α) calibration errors can estimate a stable 1-α quantile; this fails when the group has fewer errors, in which case covered_g=False and the group is reported as uncovered rather than claiming risk ≤ α.

## Contribution

["A group-conditional split-conformal calibration layer (CASCADE) that converts StepRefine's entropy-deviation spike magnitude into a distribution-free routing boundary with a user-specified risk level, jointly calibrated across text-conditioned and image-conditioned denoising groups.", 'A design principle: the discrete-vs-latent routing boundary in multimodal diffusion language models can be made exchangeability-invariant to the joint image-text likelihood training that reshapes entropy profiles, by calibrating on held-out trajectories rather than tuning on the training trajectory.', 'A calibration protocol and benchmark split (held-out prompts stratified by modality conditioning) for evaluating routing risk in MDLM decoding.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MDLM hallucination probe set (text-cond + image-cond splits), with prompts stratified into 3 length tertiles × 2 difficulty tertiles (easy/hard by tertiles of calibration trajectory-level hallucination rate under the base decoder) | Enables group-conditional calibration on prompt strata fixed before score computation, and per-regime error measurement |
| Primary metric | Hallucinated-token rate at matched compute budget (measured by an independent judge: GPT-4o with token-level teacher-forced reference decoding) | Directly measures the risk α the routing claims to control, with a label y_t not defined by the model's own commitment outcome |
| Secondary metric | Per-group coverage (fraction of groups with covered_g=True) and per-group realized error rate vs. nominal α | Reports the fraction of groups where the conformal guarantee is non-vacuous, as required by the abstention rule |
| Baseline | StepRefine fixed threshold | Tests whether conformal calibration beats a tuned global boundary |
| Baseline | Global conformal (pooled modalities) | Tests whether group-conditioning is necessary for coverage |
| Baseline | Always-latent refinement | Tests whether routing selectivity matters versus brute-force refinement |
| Ablation-of-ours | CASCADE without abstention (q_min=0.1 floor instead of abstention) | Tests whether the abstention rule's risk-compute trade-off is load-bearing |
| Ablation-of-ours | CASCADE with quantile_bin(s_t) grouping (self-referential grouping) | Tests whether the score-independent grouping is necessary for non-vacuous coverage |
| Diagnostic | ROC-AUC of s_t against y_t per group, and fraction of calibration steps where a legitimate spike (correct rare attribute) is labeled y_t=1 | Quantifies the monotonicity assumption rather than asserting it |

### Why this setup validates the claim

The central claim is that a group-conditional conformal boundary yields realized hallucination risk at or below α while spending latent refinement only where spikes predict errors. The fixed-threshold StepRefine baseline isolates the calibration sub-claim: if conformal coverage, not tuning, is what delivers risk control, CASCADE should hold risk near α across α settings while the fixed threshold drifts with the trajectory statistics. The pooled-conformal baseline isolates the group-conditioning sub-claim: pooling text- and image-conditioned steps breaks within-group exchangeability, so its coverage should degrade on the minority regime even though its average looks fine. The always-latent baseline isolates the selectivity sub-claim: it should match CASCADE on error rate but pay a large compute penalty, revealing that routing is what buys efficiency without sacrificing the guarantee. The no-abstention ablation isolates whether the abstention rule matters for honest coverage reporting: without it, degenerate groups claim risk ≤ α while routing everything to latent. The self-referential-grouping ablation isolates whether score-independent grouping is necessary: if quantile_bin(s_t) grouping yields vacuous coverage, the diagnostic should show it. The hallucinated-token rate at matched compute is the right metric because it is exactly the quantity α is supposed to bound; a method that lowers it only by refining everything is not controlling risk, it is buying it, and matched compute exposes that. The ROC-AUC diagnostic quantifies the monotonicity assumption: if s_t is not monotone with y_t even in-distribution, the conformal score is invalid and the coverage claim is vacuous.

### Expected outcome and causal chain

**vs. StepRefine fixed threshold** — On a case where a rare correct visual attribute triggers a legitimate entropy spike, the fixed-threshold baseline routes to latent refinement or commits discretely based on one globally tuned cutoff, so its realized error rate drifts away from the nominal α as prompt difficulty shifts because its boundary carries no coverage guarantee. Our method instead sets the boundary from the group-conditional conformal quantile, so the realized error rate tracks α across difficulty strata, and we expect a visible gap on the hard-spike subset with near-parity on the easy subset.

**vs. Global conformal (pooled modalities)** — On image-conditioned steps where the spike distribution is systematically shifted relative to text-conditioned steps, the pooled baseline estimates one quantile from a mixture, so its coverage is violated on whichever regime is the minority because exchangeability across the two regimes does not hold. Our method instead conditions the quantile on the modality descriptor and prompt stratum, so each regime gets its own calibrated boundary, and we expect the pooled baseline's coverage to look adequate in aggregate but fail sharply on the minority-regime split while CASCADE holds α on both.

**vs. Always-latent refinement** — On a case where most steps have low spike magnitude and the discrete branch would have been correct, the always-latent baseline spends refinement rounds on every step, so it matches or beats CASCADE on error rate only by inflating compute, because it never exploits the cheap correct path. Our method instead routes to latent only above the conformal quantile, so we expect comparable hallucination rates at a substantially lower refinement budget, with the gap concentrated on low-spike steps where the baseline wastes refinement.

**vs. CASCADE without abstention (q_min=0.1 floor)** — On groups with fewer than ceil(1/α) calibration errors, the no-abstention ablation applies a hard floor q_min=0.1 and claims risk ≤ α, so its realized error rate exceeds α in those groups because the floor is an admission that the conformal guarantee is vacuous exactly where it is needed. Our method instead reports covered_g=False and routes the group to always-latent, so we expect the no-abstention ablation to show inflated risk in degenerate groups while CASCADE's per-group coverage report honestly flags them as uncovered.

**vs. CASCADE with quantile_bin(s_t) grouping** — On groups defined by quantile_bin(s_t), the group is defined by the very statistic being thresholded, so the conditional coverage claim is self-referential and the realized error rate deviates from α because the group-conditional quantile is not a genuine conditional guarantee. Our method instead uses prompt-stratum grouping fixed before score computation, so we expect the self-referential-grouping ablation to show coverage violations while CASCADE holds α within covered groups.

### What would falsify this idea

If CASCADE's risk-control advantage over the fixed-threshold baseline is uniform across spike-magnitude strata rather than concentrated where conformal coverage should bite, or if the pooled-conformal baseline matches CASCADE's per-regime coverage, then group-conditional calibration is not doing the causal work claimed. A second falsifier: if the no-abstention ablation shows identical risk and compute to CASCADE, the abstention rule is inert and the degenerate-group story is wrong; a third falsifier: if the ROC-AUC diagnostic shows s_t is not monotone with y_t even in-distribution, the conformal score is invalid and the coverage claim is vacuous.

## References

1. Rethinking in Spikes: Mitigating Hallucinations in MDLMs with Step-Aware Decoding
