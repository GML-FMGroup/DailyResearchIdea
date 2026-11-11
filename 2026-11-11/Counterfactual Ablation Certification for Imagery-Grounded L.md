# Counterfactual Ablation Certification for Imagery-Grounded Land-Change Agents

## Motivation

EarthRecord's withheld-reference scoring and closed-book control cannot distinguish an answer derived from pixels from one recovered from memorized geographic priors, because a correct answer and a prior-consistent answer are observationally equivalent under withheld-reference scoring; the instrument-pair control alters sensor data but does not intervene on the specific claimed region, so an agent can hedge across sensors or exploit correlated priors. This is the shared residual gap across VLM evaluation critiques and ScienceAgent-Bench: no existing protocol certifies that a land-change claim is causally dependent on the imagery it names, so benchmark scores conflate pixel-grounded reasoning with geographic recall.

## Key Insight

A land-change answer is imagery-derived if and only if its confidence is causally necessary to the specific pixels it claims — an intervention that masks exactly the claimed change region (or temporally shuffles it) must produce a large, region-specific confidence drop, and this necessity is testable per-answer without any reference labels.

## Method

**What it is.** We introduce **CAC (Counterfactual Ablation Certification)**, a per-answer evaluation protocol that wraps any multimodal agent evaluated on EarthRecord. Given an agent, a question, the full satellite record, and the agent's answer with a claimed change region R and a decision readout, CAC produces a certification verdict (CERTIFIED / UNCERTIFIED) and a certified score. It requires no reference labels, only the agent's own decision readout under two targeted interventions.

**How it works.**

```pseudocode
Input: agent A, question q, record X, answer a with claimed region R, decision readout d0 = A(q, X)
Hyperparameters: tau_drop = 0.30 (absolute decision-margin drop threshold),
                tau_specificity = 2.0 (ratio of targeted to control drop),
                K = 3 (number of temporal-shuffle draws),
                M = 10 (number of random-region control draws),
                calibration split = 20% of EarthRecord questions (48 questions),
                calibration statistic = 90th percentile of closed-book decision-margin drops

Phase 1 — Claim extraction and region localization
  Parse a to obtain claimed change region R (bounding polygon or pixel mask)
  and claimed change type T. If R cannot be localized, return UNCERTIFIED_NO_REGION.

Phase 2 — Targeted counterfactual interventions
  X_mask <- X with all pixels in R masked (cloud/NaN fill) across all dates
  X_shuf <- for k in 1..K: X with the temporal order of R's pixels shuffled
            (dates permuted within R only; pixels outside R unchanged)
  d_mask <- A(q, X_mask).decision_margin for claimed type T
  d_shuf <- mean_k A(q, X_shuf^k).decision_margin for claimed type T
  d_target <- min(d_mask, d_shuf)   # worst-case targeted drop

Phase 3 — Control interventions (specificity)
  X_rand <- X with a randomly placed region R' of equal area and sensor
            masked (M=10 draws, averaged)
  X_global <- X with all pixels masked (global ablation)
  d_rand <- mean A(q, X_rand).decision_margin for claimed type T
  d_global <- A(q, X_global).decision_margin for claimed type T
  delta_target <- d0 - d_target
  delta_rand   <- d0 - d_rand

Phase 4 — Certification rule
  if delta_target >= tau_drop AND delta_target >= tau_specificity * delta_rand:
      verdict <- CERTIFIED
  else:
      verdict <- UNCERTIFIED
  certified_score <- verdict * (1 if answer matches withheld reference else 0)
  return verdict, certified_score, (d0, d_mask, d_shuf, d_rand, d_global)
```

The protocol is applied per answer; the benchmark's headline metric becomes the **certified accuracy**: the fraction of questions whose answer is both correct against the withheld reference and CERTIFIED. Uncertified correct answers are reported separately as "prior-consistent" and are not counted as imagery-derived.

**Decision readout.** The agent's scalar self-reported confidence is replaced by a decision-level readout: for the claimed change type T, the agent is prompted to output a forced-choice likelihood over the set of change types (e.g., "urban expansion", "reservoir shrinkage", "deforestation", "no change"), and the decision margin is defined as the log-probability of T minus the log-probability of the highest-probability alternative type. This readout is label-free and does not rely on calibrated verbalized confidence. The same readout is used for all interventions.

**Threshold calibration.** tau_drop and tau_specificity are set per agent family on a 20% calibration split (48 questions) using the closed-book control: tau_drop is the 90th percentile of decision-margin drops observed when the closed-book agent (which never sees the record) is subjected to the same targeted masking and shuffling of its claimed region; tau_specificity is fixed at 2.0 as a conservative ratio. The calibration procedure is pre-registered: split by site (not by question) to avoid leakage, use the same M=10 random-region draws, and report sensitivity over a grid of tau_drop ∈ {0.2, 0.3, 0.4} and tau_specificity ∈ {1.5, 2.0, 3.0}.

**Why this design.** We chose a *targeted region-specific* ablation over the existing instrument-pair control because instrument pairs alter a whole sensor and let an agent hedge across correlated bands, whereas masking exactly the claimed region R makes the counterfactual logically tied to the answer's own claim, accepting the cost that an agent which mislocalizes R will be certified against the wrong region (addressed by requiring R to be parsed from the answer, not supplied). We chose a *specificity ratio* against a random-region control over a single absolute drop threshold because a global confidence collapse under any perturbation would otherwise pass as imagery-derived, accepting the cost that agents with uniformly fragile confidence are penalized even when partially grounded. We chose *temporal shuffling within R* as a second intervention alongside masking because masking can be detected as an out-of-distribution input and trigger a generic refusal, whereas temporal shuffling preserves pixel statistics and isolates the temporal-change signal, accepting the cost of K=3 extra forward passes. We chose a *hard certification rule* over a learned gate because the rule is auditable and label-free, accepting the cost that tau_drop and tau_specificity must be set per agent family on a small calibration split (we fix them from the closed-book control's decision-margin distribution). This differs from the instrument-pair control in EarthRecord (which varies sensors, not the claimed region) and from standard input-attribution methods such as occlusion saliency, which attribute to input features but do not certify a *claimed region* against a *specificity control*; an expert would not describe CAC as occlusion because the unit of intervention is the answer's own claim, and the verdict is a per-answer certificate, not a saliency map.

**Why it measures what we claim.** The quantity `delta_target = d0 - min(d_mask, d_shuf)` measures *causal necessity of the claimed region to the answer* because the intervention changes exactly the pixels the answer names while holding the question and all other pixels fixed, under the assumption that the agent's decision margin is a monotone function of the evidence it uses; this assumption fails when the agent's decision margin is dominated by question phrasing or prior geographic knowledge, in which case delta_target reflects phrasing sensitivity rather than pixel grounding. The quantity `delta_rand` measures *generic perturbation sensitivity* because it masks an equal-area random region, under the assumption that a grounded agent's decision margin is insensitive to regions it did not claim; this assumption fails when the agent attends diffusely to the whole scene, in which case delta_rand inflates and the specificity ratio under-certifies a genuinely grounded answer. The ratio `delta_target / delta_rand` measures *region specificity of grounding* because it contrasts the claimed-region drop against the equal-area control drop, under the assumption that area and sensor are matched between the two interventions; this assumption fails when the claimed region R is much larger than the random control, in which case the ratio reflects region size rather than specificity (we therefore match R' area to R). The final verdict measures *imagery-derivedness* because it requires both a large targeted drop and specificity, under the assumption that a prior-knowledge shortcut is insensitive to the claimed region; this assumption fails when the prior itself is region-indexed (e.g., the agent memorized that this site changed), in which case the temporal-shuffle intervention is the residual check and a CERTIFIED verdict on a shuffled record indicates the prior is date-invariant.

**(D) Equivalence and failure mode.** The quantity `delta_target` measures causal necessity of the claimed region under assumption A: the agent's decision margin is a monotone, phrasing-invariant function of pixel evidence. The failure mode F is a region-indexed prior or phrasing-driven decision margin that produces a large delta_target on a temporally shuffled record. To flag F, we add a diagnostic: any answer that is CERTIFIED on the temporally shuffled record (i.e., delta_target computed from X_shuf alone exceeds tau_drop) is marked PRIOR_CONTAMINATED and excluded from certified accuracy. This diagnostic is computed from the same interventions already performed in Phase 2.

## Contribution

(1) A label-free per-answer certification protocol, CAC, that intervenes on the agent's own claimed change region and uses a specificity control to certify imagery-derivedness, converting EarthRecord's benchmark score into a certified accuracy that separates pixel-grounded from prior-consistent answers. (2) A design principle — region-targeted counterfactual ablation with an equal-area random control and within-region temporal shuffling — that closes the observational-equivalence gap left by withheld-reference scoring and instrument-pair controls, and that can be applied to any multimodal agent benchmark with localized claims. (3) A reusable certification harness and reported certified-vs-prior-consistent breakdown for the twelve EarthRecord agents, exposing which analysis tasks are solved by imagery versus prior knowledge.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | EarthRecord (56 sites, 238 questions) | Multimodal land-change QA with withheld references and controls |
| Primary metric | Certified accuracy | Counts only correct-and-CERTIFIED answers, isolating imagery-derived reasoning |
| Baseline | EarthRecord closed-book control | Tests whether CAC separates prior-only from imagery-grounded answers |
| Baseline | EarthRecord instrument-pair control | Tests whether region-targeted ablation beats whole-sensor ablation |
| Baseline | Occlusion saliency attribution (threshold swept over full range) | Tests whether per-answer certification beats feature-attribution scores; we sweep the saliency threshold over its full range and report the best achievable certified accuracy |
| Ablation-of-ours | CAC without specificity ratio (absolute drop only) | Isolates the contribution of the random-region control |
| Ablation-of-ours | CAC with scalar confidence instead of decision margin | Tests whether the decision readout is necessary for the specificity ratio to separate grounded from prior-only answers |
| Diagnostic | Monotonicity calibration | Measures fraction of answers where decision margin is non-monotone in fraction-of-claimed-region masked; report and exclude flagged answers |

### Why this setup validates the claim

The central claim is that a per-answer, region-targeted counterfactual with a specificity control can distinguish imagery-derived answers from prior-consistent ones, and that this distinction changes the ranking of multimodal agents on EarthRecord. The closed-book control tests the first sub-claim: if CAC is measuring imagery-derivedness, then closed-book answers (which never see the record) should be systematically UNCERTIFIED while their decision margin stays high, so certified accuracy for the closed-book control should be near zero even where raw accuracy is non-trivial. The instrument-pair control tests the second sub-claim: whole-sensor ablation lets an agent hedge across correlated bands, so it should over-certify relative to CAC's region-targeted ablation, producing a visible certified-vs-raw gap on questions where the agent's claimed region is small and localized. Occlusion saliency tests the third sub-claim: it produces a saliency map, not a verdict, so it cannot be turned into a certified accuracy without an arbitrary threshold; we sweep the threshold over its full range and report the best achievable certified accuracy, and if CAC's certified accuracy is matched by some threshold on the saliency score, the novelty claim collapses and must be reframed around the specificity control alone. The ablation (absolute drop only, no specificity) is the key diagnostic: it should over-certify agents whose decision margin collapses under any perturbation, so its certified accuracy should exceed CAC's on the diffuse-attention subset while matching on the localized subset. The scalar-confidence ablation tests whether the decision readout is necessary: if it matches CAC, the decision readout adds nothing and the method reduces to a confidence-based check. The monotonicity calibration directly tests the load-bearing assumption: for a held-out set of 48 questions, we mask fractions f ∈ {0.1, 0.25, 0.5, 0.75, 1.0} of the claimed region and plot decision margin against f; we report the fraction of answers where the curve is non-monotone (i.e., margin increases at any step) and exclude those answers from certified accuracy, flagging them as NON_MONOTONE. Certified accuracy is the right metric because it is the only one that simultaneously requires correctness and a causal, region-specific dependence on the record, which is exactly the quantity the claim asserts is being measured.

### Expected outcome and causal chain

**vs. EarthRecord closed-book control** — On a case where the question asks about a specific change at a named site the agent has memorized (e.g., a well-known reservoir expansion), the closed-book control produces a confident correct answer with no access to the record, because its mechanism is prior geographic knowledge rather than pixel evidence. Our method instead masks the claimed region and shuffles its temporal order, so the prior-driven decision margin is unchanged and delta_target stays near zero, yielding UNCERTIFIED. We expect the closed-book control to show high raw accuracy but near-zero certified accuracy, with a large gap concentrated on questions about well-documented sites.

**vs. EarthRecord instrument-pair control** — On a case where the agent claims a small localized change (e.g., a new building cluster) and relies on a single band that is correlated with a neighboring band, the instrument-pair control produces a confident answer because removing one sensor leaves the correlated band intact and the agent hedges, so its mechanism gap is that it perturbs sensors rather than the claimed pixels. Our method instead masks exactly the claimed polygon, removing the evidence the answer names, so delta_target is large while delta_rand stays small. We expect CAC to certify fewer answers than the instrument-pair control on localized-change questions, with the gap shrinking on questions whose claimed region is large or diffuse.

**vs. Occlusion saliency attribution (threshold swept)** — On a case where the agent's answer is driven by a small region but its attention is spread across the scene, occlusion saliency produces a diffuse saliency map with no clear peak, because its mechanism attributes to individual features without a specificity control and without tying attribution to the answer's claim. Our method instead compares the claimed-region drop against an equal-area random-region drop, so a diffuse-attention agent shows delta_rand inflating and fails the specificity ratio, yielding UNCERTIFIED. We expect CAC to produce a more stable ranking of agents across question subsets than thresholded saliency scores, with the largest divergence on the diffuse-attention subset; if some saliency threshold matches CAC's certified accuracy, we will report that and reframe the novelty around the specificity control alone.

**vs. CAC without specificity ratio** — On a case where the agent's decision margin collapses under any perturbation (e.g., masking any region triggers a generic refusal), the absolute-drop-only ablation produces CERTIFIED because delta_target exceeds tau_drop, since its mechanism gap is that it never checks whether the drop is specific to the claimed region. Our method instead requires delta_target to exceed tau_specificity times delta_rand, so the generic-collapse agent fails certification. We expect the ablation to over-certify on the diffuse-attention subset while matching CAC on the localized subset, producing a characteristic subset-dependent gap.

**vs. CAC with scalar confidence** — On a case where the agent's verbalized confidence is miscalibrated and shifts under meaning-preserving perturbations, the scalar-confidence version produces large delta_target and delta_rand that are not tied to the claimed pixels, because its mechanism relies on a poorly calibrated self-report. Our method instead uses a forced-choice decision margin over change types, which is less sensitive to phrasing. We expect the scalar-confidence version to show lower certified accuracy and less stable rankings across subsets than the decision-margin version.

**vs. Monotonicity calibration** — On a case where the agent's decision margin is non-monotone in the fraction of claimed region masked (e.g., margin increases when 25% is masked), the monotonicity assumption fails and delta_target is not a valid necessity measure. Our method flags such answers as NON_MONOTONE and excludes them. We expect the fraction of non-monotone answers to be below 10% for grounded agents and higher for prior-only agents, and we will report this fraction as a diagnostic.

### What would falsify this idea

If the closed-book control achieves certified accuracy comparable to the full-record agents, or if CAC's certified accuracy is uniform across the localized, diffuse-attention, and prior-memorized subsets rather than concentrated as predicted, then the specificity ratio is not isolating imagery-derivedness and the central claim is wrong. A second falsifier: if the absolute-drop-only ablation matches CAC on every subset, the specificity control contributes nothing and the method reduces to a generic perturbation check; additionally, if the fraction of non-monotone answers exceeds 20% for grounded agents, the decision-margin readout is not a reliable necessity signal and the method must be revised.

## References

1. EarthRecord: From Pixels to Land Change with Multimodal Agents over Four Decades of Satellite Imagery
