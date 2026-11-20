# Grounding-Consistent Step-by-Step Reasoning for Omni-Modal Video Understanding

## Motivation

Existing omni-modal video LLMs such as WeMM-Video produce step-by-step event descriptions without any structural guarantee that each claimed event corresponds to a verifiable audio-visual interval; the hybrid-thinking paradigm inherits the assumption that training on temporally annotated or automatically curated data (as in TriSense-2M and AVU) yields grounded outputs. This assumption fails because the reasoning chain is optimized only for answer likelihood, so event boundaries in the generated text drift from the actual audio-visual change points, and no mechanism enforces alignment between the claimed event and its evidence interval. The same residual gap — a temporal grounding consistency invariant — recurs across PreFM, TriSense, ViSAGe, and WeMM-Video, making it a high-value bridge target.

## Key Insight

Enforcing a boundary-alignment invariant between generated event intervals and detected audio-visual change points, via a differentiable consistency penalty applied during reasoning-mode decoding, makes each claimed event's evidence interval a verifiable certificate rather than a post-hoc rationalization.

## Method

**What it is.** We introduce GRACE (Grounded Reasoning with Audio-visual Change-point Enforcement), a decoding-time and fine-tuning mechanism for WeMM-Video's step-by-step reasoning mode. Its inputs are the video's audio-visual feature stream, the model's reasoning trace with candidate event intervals, and detected audio-visual change points; its output is a reasoning trace in which every claimed event is annotated with a verifiable evidence interval whose boundaries align with observed change points.

**Load-bearing assumption (stated explicitly).** GRACE assumes that audio-visual change points detected by the 90th-percentile L2 novelty score on 2 Hz frame features are the canonical, semantically valid event boundaries, so an interval aligned to them is a verifiable grounding certificate rather than proximity to an arbitrary statistic. This assumption fails for gradual scene changes and static events, and the fixed-percentile threshold guarantees ~10% of frames are labeled change points regardless of whether any semantic event boundary exists. We therefore calibrate the detector against human event-boundary annotations and gate the certificate by event type, as specified in steps 1b and 3b below.

**How it works.**

```pseudocode
Input: video V with audio stream A and visual stream F; WeMM-Video reasoning model M
Output: grounded reasoning trace T with per-event evidence intervals and per-event certificate scores

1. Detect change points:
   - Compute frame-level audio features a_t (from A) and visual features v_t (from F) at 2 Hz
   - Compute audio-visual novelty score n_t = ||a_t - a_{t-1}||_2 + ||v_t - v_{t-1}||_2
   - Detect change points C = {c_k} where n_t exceeds a percentile threshold (90th percentile of n_t within the clip)

1b. Calibrate detector against human boundaries:
   - On a calibration set of 512 clips from AVU validation split with human event-boundary annotations, compute precision/recall of C as a function of event type (sharp, gradual, static) at the 90th-percentile threshold
   - Fit a per-event-type reliability score r(type) = P(interval near c_k is a true event boundary | type), estimated as the fraction of detected change points within ±0.5s of a human boundary on the calibration set
   - Store r(sharp), r(gradual), r(static) as fixed scalars; if r(type) < 0.5, the certificate for that event type is marked invalid and the mask is not applied (abstain)

2. Reasoning-mode generation with interval emission:
   - M generates reasoning steps s_1 ... s_N in thinking mode
   - For each event claim e_i in a step, M emits an interval [l_i, r_i] in seconds and an event-type tag t_i ∈ {sharp, gradual, static} (predicted by M from the step context)

3. Consistency check (per event):
   - For each claimed interval [l_i, r_i], find the nearest detected change point c_k
   - Boundary alignment loss: L_align = sum_i [ |l_i - c_k(l_i)| + |r_i - c_k(r_i)| ]
   - Change-point coverage: penalize events whose interval contains no change point (L_cover)

3b. Certificate score per event:
   - cert_i = r(t_i) * 1[interval contains ≥1 change point] * 1[both boundaries within ±0.5s of a change point]
   - Events with cert_i = 0 are flagged as ungrounded in the output trace

4. Fine-tuning with consistency penalty:
   - Total loss L = L_lm + lambda_align * L_align + lambda_cover * L_cover
   - lambda_align = 0.5, lambda_cover = 0.2 (tuned on held-out AVU validation split)
   - Train on AVU and TriSense-2M instruction data with interval annotations derived from change points

5. Inference:
   - At decode time, apply a constrained decoding mask only when r(t_i) ≥ 0.5 for the predicted event type: interval boundaries must lie within +/- 0.5s of a detected change point
   - If no change point exists in a 2s window, the event is flagged as ungrounded and the model must revise the claim
```

**Why this design.** We chose a change-point-anchored consistency penalty over a learned grounding head because a learned head would assert independence between event text and evidence intervals that the data does not contain (Anti-Pattern 1); instead we exploit the structural fact that audio-visual change points are observable, task-independent statistics of the input, so the invariant is checkable without a learned module. We chose constrained decoding over post-hoc filtering because post-hoc filtering cannot correct a reasoning trace whose intermediate steps already committed to wrong boundaries, accepting the cost that constrained decoding may reduce fluency in exchange for verifiability. We chose to fine-tune with the consistency penalty rather than only apply it at inference, because WeMM-Video's multi-stage post-training (specialist RL, MrDPO) already shapes the reasoning distribution, and a decoding-only constraint would fight that distribution; the trade-off is additional training compute. A domain expert would not describe this as a variant of TriSense's Query-Based Connector or PreFM's predictive future modeling, because those methods reweight or predict modalities without producing a per-event verifiable interval whose boundaries are certified against observable change points.

**Why it measures what we claim.** The boundary alignment loss L_align measures temporal grounding consistency because it assumes that audio-visual change points are the canonical event boundaries in video, so an event interval that aligns with a change point is grounded; this assumption fails when an event boundary is semantically real but produces no audio-visual novelty (e.g., a gradual scene change), in which case L_align reflects proximity to a spurious change point rather than true grounding. The change-point coverage penalty L_cover measures evidence sufficiency because it assumes every grounded event must contain at least one observable change point; this assumption fails for static events (e.g., a prolonged silence with no visual change), in which case L_cover penalizes a valid event. The constrained decoding mask measures verifiability because it assumes that any interval not near a change point is unverifiable by a human annotator; this assumption fails when change points are missed by the 90th-percentile threshold, in which case the mask rejects a valid interval. The per-event certificate score cert_i measures the probability that a claimed interval is a true event boundary because it multiplies the detector's calibrated reliability r(t_i) by the alignment and coverage indicators; this assumption fails when the calibration set is not representative of the test distribution (e.g., different event-type mix), in which case cert_i reflects calibration-set bias rather than per-event grounding validity.

## Contribution

["GRACE, a consistency-enforcement mechanism that couples WeMM-Video's step-by-step reasoning trace to detected audio-visual change points via a boundary-alignment loss and constrained decoding, producing per-event verifiable evidence intervals.", 'A design principle showing that anchoring event intervals to observable audio-visual change points, rather than to learned grounding heads, yields verifiable reasoning without requiring temporally annotated training data beyond change-point statistics.', 'An evaluation protocol that measures grounding consistency as the fraction of claimed events whose intervals align with detected change points, enabling falsification of the grounding claim.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AVU + TriSense-2M held-out splits | Both carry audio-visual event intervals for grounding |
| Primary metric | Change-point-aligned interval IoU on claimed events, stratified by event type (sharp/gradual/static) | Directly measures boundary grounding, not just answer accuracy; stratification tests the causal claim that gains concentrate where change points are detectable |
| Secondary metric | Per-event certificate score cert_i (mean and fraction with cert_i = 0) | Measures the calibrated verifiability of each claimed interval |
| Baseline | WeMM-Video thinking mode (unmodified) | Tests whether grounding supervision adds anything beyond base reasoning |
| Baseline | Post-hoc interval filtering on WeMM-Video traces | Tests whether decode-time enforcement beats after-the-fact cleanup |
| Baseline | TriSense Query-Based Connector transfer | Tests whether query-reweighting already yields verifiable intervals |
| Ablation-of-ours | GRACE with L_align only (no L_cover, no mask) | Isolates whether coverage and constrained decoding each contribute |
| Ablation-of-ours | GRACE with uncalibrated detector (fixed 90th percentile, no r(type) gating) | Tests whether calibration and event-type gating are necessary for the certificate to be valid |

### Why this setup validates the claim

The central claim is that anchoring event intervals to observable audio-visual change points yields reasoning traces whose boundaries are verifiable, and that this must be enforced during decoding and fine-tuning rather than recovered afterward. The unmodified WeMM-Video thinking mode tests the null hypothesis: that a strong omni-modal reasoner already emits grounded intervals. The post-hoc filtering baseline tests the design choice that motivated constrained decoding — if filtering a committed trace suffices, our decode-time machinery is unnecessary. The TriSense transfer baseline tests whether generic query-based modality reweighting already produces per-event verifiable intervals, which the proposal explicitly denies. The L_align-only ablation tests whether the coverage penalty and mask are doing real work or whether boundary alignment alone explains any gain. The uncalibrated-detector ablation tests the load-bearing assumption directly: if calibration and event-type gating do not change the certificate's validity, then the fixed-percentile detector is sufficient and the calibration step is inert. Change-point-aligned interval IoU, stratified by event type, is the right primary metric because it directly operationalizes "verifiable boundary": an interval either lands near an observable change point or it does not, and the stratification reveals whether the effect is concentrated on sharp-transition events where change points are detectable, as the causal claim requires.

### Expected outcome and causal chain

**vs. WeMM-Video thinking mode (unmodified)** — On a case where an event's true boundary is a sharp audio onset plus a visual cut (e.g., applause starting as the speaker stops), the baseline emits a plausible-sounding interval that is loosely placed because nothing in its post-training ties interval tokens to observable statistics. Our method instead constrains boundaries to within ±0.5s of detected change points and penalizes uncovered intervals during fine-tuning, so we expect a clear IoU gain concentrated on sharp-transition events, with near-parity on gradual or static events where no change point exists.

**vs. Post-hoc interval filtering on WeMM-Video traces** — On a case where the reasoning trace commits early to a wrong boundary and then builds subsequent steps around it, post-hoc filtering can only delete or snap that interval, leaving the surrounding reasoning inconsistent because the trace already assumed the wrong span. Our method instead masks boundary tokens during decoding so the trace never commits to an off-change-point interval, so we expect our gain to appear specifically on multi-step traces with chained event claims, where filtering degrades to baseline parity.

**vs. TriSense Query-Based Connector transfer** — On a case where the query emphasizes speech over visual cues, TriSense reweights modality contributions but emits no per-event interval certified against observable statistics, so its temporal localization remains soft and unverifiable. Our method instead certifies each interval against change points, so we expect a gap on interval-IoU that is large even where TriSense's answer accuracy is comparable, showing the grounding gain is separable from modality fusion.

**vs. GRACE with uncalibrated detector** — On a gradual scene change where the 90th-percentile detector fires on a spurious novelty spike, the uncalibrated variant applies the mask and certifies an interval that is not a true event boundary, inflating IoU on gradual events while degrading precision. Our calibrated variant uses r(gradual) < 0.5 to abstain from masking on gradual events, so we expect the calibrated variant to match or exceed uncalibrated IoU on sharp events while reducing false certificates on gradual and static events, confirming that the calibration step is necessary for the certificate's validity.

### What would falsify this idea

If the interval-IoU gain over unmodified WeMM-Video is uniform across sharp-transition, gradual, and static-event subsets rather than concentrated where change points are actually detectable, then the mechanism is not the cause and the central claim fails. Equally, if the L_align-only ablation matches full GRACE, the coverage and constrained-decoding components are inert and the design rationale collapses; and if the uncalibrated-detector ablation matches calibrated GRACE on certificate validity, the load-bearing assumption is not repaired by calibration.

## References

1. WeMM-Video: Pushing the Frontiers of Open-Source Omni-Modal Video Understanding
