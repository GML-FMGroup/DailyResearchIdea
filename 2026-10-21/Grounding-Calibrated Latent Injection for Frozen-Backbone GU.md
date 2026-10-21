# Grounding-Calibrated Latent Injection for Frozen-Backbone GUI Agents

## Motivation

FocusGUI-style adapters (Focused Omni-Modal GUI Agents) estimate per-step relevance and inject a compressed latent into a frozen decoder, but their relevance estimate is silently conditioned on the frozen backbone's own visual grounding quality — if the backbone mislocalizes the element a referring expression names, the adapter's 'decision reference' is computed against the wrong pixels and the injected latent steers attention toward irrelevant evidence. The field has no established property of robustness of relevance estimation to backbone grounding failures: FocusGUI reports gains on four backbones but never measures whether its relevance estimate degrades when the backbone's grounding degrades, and Qwen-VL/CogAgent-style backbones differ widely in pixel-level grounding accuracy (Qwen-VL aligns image-caption-box tuples; CogAgent relies on a high-res encoder), so the adapter's behavior is backbone-idiosyncratic rather than a property of the method. This is a meta-level gap: any frozen-backbone latent-injection scheme inherits an unmeasured, backbone-specific grounding error into its relevance estimate, and no existing work provides a calibration signal that is independent of the backbone's own confidence.

## Key Insight

Grounding quality can be measured without a held-out oracle by re-grounding the agent's own successful trajectory steps — a step whose referring expression the frozen backbone can re-localize to the element it actually clicked is a step where the relevance estimate is trustworthy, so this re-localization rate is a backbone-specific calibration signal that gates latent injection rather than a confidence score.

## Method

**What it is.** GCLI (Grounding-Calibrated Latent Injection) wraps a FocusGUI-style relevance adapter with a calibration probe: before each episode it computes a backbone-specific grounding score g from the agent's own past successful trajectories, and uses g to reweight the adapter's latent injection, falling back to a history-attention prior when g is low. The load-bearing assumption is stated explicitly: **A single backbone-level scalar g, estimated once from ~200 re-localization probes, is a valid and sufficient predictor of per-step relevance-estimate trustworthiness, i.e., backbone grounding error is the dominant, step-invariant cause of misdirected latent injection.** This assumption is verified by the per-step diagnostic in Phase 1 (logging relevance error vs. grounding error) and by the stratified-g ablation in the experiment.

**How it works.**

```pseudocode
# Phase 0: calibration probe (once per backbone, ~200 real screenshots)
# Sample stratified by difficulty stratum d_t in {small, medium, large} by click_box area tertiles;
# include both successful and failed/hard steps to avoid selection on the dependent variable.
for each trajectory step (s_t, a_t) in stratified sample of agent's replay buffer:
    expr_t = auto_referring_expr(a_t)        # template from clicked element's OCR/accessibility text
    pred   = frozen_backbone.ground(s_t, expr_t)   # returns pixel box
    hit_t  = IoU(pred, a_t.click_box) > 0.5
    # record difficulty stratum d_t in {small, medium, large} by click_box area
    # record per-step diagnostic: relevance_error_t = ||r_t - r_t*|| (see Phase 1)
g = mean(hit_t)  stratified over d_t          # backbone grounding score, in [0,1]
g_d = mean(hit_t | d_t) for d in {small, medium, large}  # stratum-specific scores

# Phase 1: per-step gated injection
for each step t:
    r_t = adapter.relevance(s_t, channels)    # FocusGUI relevance scores over modalities
    z_t = adapter.compress(r_t)              # compact latent
    # gate: reweight injection by calibration, floored so it never fully vanishes
    w_t = clip(g_{d_t}, 0.15, 1.0)           # stratum-conditioned weight from Phase 0
    if w_t >= 0.5:
        inject(z_t * w_t)                    # scaled latent into decoder embeddings
    else:
        inject(history_prior_attention(s_{<t}))  # fallback: attend to prior-step evidence
    # per-step diagnostic (logged, not used for gating):
    #   grounding_error_t = 1 - IoU(frozen_backbone.ground(s_t, expr_t), a_t.click_box)
    #   relevance_error_t = ||r_t - r_t*|| where r_t* is the oracle relevance from human-annotated click box
    #   log (grounding_error_t, relevance_error_t) to test step-invariance assumption A
```

Hyperparameters: IoU threshold 0.5; floor 0.15; gate threshold 0.5; probe size ~200 steps (stratified across difficulty tertiles, including failed/hard steps); difficulty strata by click-box area tertiles. Sensitivity analysis over probe size {50, 100, 200, 500} and IoU threshold {0.3, 0.5, 0.7}.

**Why this design.** We chose a re-localization probe over the backbone's own log-probability confidence because log-probs are not calibrated for grounding correctness (Kadavath-style negative evidence), and a probe gives a signal that is monotone in actual localization error by construction; the cost is that the probe requires ~200 real screenshots and a template that turns a click into a referring expression, which we accept because those come free from the agent's own successful trajectories. We chose a single scalar gate g over a per-step learned gate because a per-step gate would be trained on the same action supervision as the adapter and would collapse onto the adapter's own relevance score — reintroducing the coupling we are trying to break; the cost is coarser granularity, which we accept because backbone grounding quality is a backbone-level property, not a step-level one. We chose a history-attention fallback over simply zeroing the latent because zeroing discards the adapter's information entirely and reduces to the frozen backbone (the FocusGUI baseline), whereas history attention still carries action-sufficient evidence from prior steps; the cost is the fallback is weaker than a correct latent injection, which we accept because it only triggers when the latent would be misdirected. We deliberately do not use a learned router between latent and history (which would match the two-subsystem-controller anti-pattern): the gate is a closed-form function of the probe statistic, not a trained module. We use a stratum-conditioned weight w_t = clip(g_{d_t}, ...) rather than a single global g to partially address within-backbone variance in grounding error across element sizes, while keeping the gate closed-form and not learned.

**Why this measures what we claim.** The re-localization hit rate `g` measures backbone grounding quality because the assumption that a template-derived referring expression naming the clicked element's text is a valid probe of whether the backbone can localize that element; this assumption fails when the clicked element has no distinctive text (e.g., an icon-only button), in which case `g` reflects template coverage rather than grounding ability. The IoU>0.5 threshold measures localization correctness because the assumption that a 0.5 IoU box overlaps the intended target sufficiently for the adapter's decision reference to be computed against the right region; this assumption fails for very small elements where a 0.5-IoU box can still miss the element center, in which case the hit rate reflects box-size sensitivity rather than semantic grounding. The gate weight `w_t = clip(g_{d_t},...)` measures the trustworthiness of the adapter's latent injection because the assumption that backbone grounding error is the dominant cause of relevance-estimate error at a step; this assumption fails when the relevance error comes from channel-selection noise rather than grounding, in which case the gate under-corrects. The history-prior fallback measures the claim that prior-step evidence is action-sufficient when current-step grounding is unreliable because the assumption that GUI tasks have temporal evidence continuity (the target element was visible or referenced in a prior step); this assumption fails on single-step tasks with no history, in which case the fallback provides no signal and the method reduces to the frozen backbone. The per-step diagnostic (grounding_error_t, relevance_error_t) directly tests the step-invariance assumption A: if relevance_error_t is not monotone in grounding_error_t across steps within a backbone, the global/stratum gate is insufficient.

## Contribution

(1) A backbone-robustness calibration mechanism for frozen-backbone latent-injection adapters that measures grounding quality from the agent's own successful trajectories via template-derived re-localization, requiring no held-out oracle and no backbone fine-tuning. (2) A closed-form gating rule that reweights latent injection by the calibration statistic and falls back to a history-attention prior, establishing the design principle that relevance estimation should be gated by a grounding-quality measurement rather than by the backbone's own confidence. (3) An analysis of how backbone-specific grounding accuracy (contrasting Qwen-VL-style tuple-aligned grounding with CogAgent-style high-res encoding) propagates into adapter relevance error, and the conditions under which the gate recovers frozen-backbone performance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Omni-modal GUI agent benchmark (FocusGUI suite) | Matches the four-backbone evaluation setting of the baseline. |
| Primary metric | Task success rate | End-to-end measure of whether grounding misdirection is corrected. |
| Secondary metric | Per-step relevance error vs. grounding error correlation | Tests step-invariance assumption A directly. |
| Baseline | Frozen backbone (no FocusGUI) | Tests whether GCLI needs the adapter at all. |
| Baseline | FocusGUI adapter, ungated | Tests whether calibration gating adds value over raw latent injection. |
| Baseline | FocusGUI with log-prob confidence gate | Tests re-localization probe against the rejected confidence signal. |
| Baseline | GCLI with global scalar g (no strata) | Tests whether stratum-conditioning adds value over global g. |
| Ablation-of-ours | GCLI with constant gate w=1.0 | Isolates the contribution of the calibration-derived gate. |
| Ablation-of-ours | GCLI with probe size {50, 100, 200, 500} | Sensitivity to probe size. |
| Ablation-of-ours | GCLI with IoU threshold {0.3, 0.5, 0.7} | Sensitivity to IoU threshold. |

### Why this setup validates the claim

The central claim is that a backbone-specific grounding probe, not the adapter's own relevance score, tells us when latent injection will misdirect the decoder, and that gating on it improves task success. The frozen backbone baseline tests the sub-claim that the adapter is necessary: if GCLI cannot beat the frozen backbone, the latent injection carries no usable signal regardless of gating. The ungated FocusGUI baseline tests the sub-claim that gating itself matters: if GCLI equals ungated FocusGUI, the probe adds nothing. The log-prob-confidence-gated baseline tests the specific design choice that a re-localization probe is better calibrated than the backbone's own confidence; if the two gates perform identically, our stated motivation is wrong. The global-scalar-g baseline tests whether stratum-conditioning (g_d) adds value over a single g, directly probing the step-invariance assumption A. The constant-gate ablation isolates the value of the probe statistic from the mere presence of a gate. The per-step diagnostic (grounding_error_t vs. relevance_error_t) directly tests whether backbone grounding error is the dominant, step-invariant cause of relevance error; if relevance_error_t is not monotone in grounding_error_t, assumption A fails. Task success rate is the right primary metric because the failure mode we predict is misdirected injection causing wrong clicks, which task success directly captures; intermediate grounding metrics would not show whether the gate prevents downstream action errors. The sensitivity analyses over probe size and IoU threshold bound the annotation/replay cost and show the method is not brittle to these hyperparameters.

### Expected outcome and causal chain

**vs. Frozen backbone (no FocusGUI)** — On a case where the target element has clear distinctive text and the adapter's relevance estimate is accurate, the frozen backbone produces a wrong click because it lacks any channel-relevance compression, so it attends to irrelevant visual or textual channels. Our method instead injects the gated latent, and because the probe g is high on this backbone, w_t is near 1 and the injection is nearly full-strength, so we expect a clear success-rate gain on text-rich, multi-channel screens and near-parity on screens where the backbone already grounds well.

**vs. FocusGUI adapter, ungated** — On a case where the backbone's grounding is weak (e.g., small icon-only buttons where the re-localization probe yields low g), ungated FocusGUI produces a wrong click because the misdirected latent overrides correct prior-step evidence. Our method instead falls back to history attention when w_t < 0.5, preserving the prior-step target evidence, so we expect a noticeable gain concentrated on the small-element stratum and parity on the large-element stratum where g stays high.

**vs. FocusGUI with log-prob confidence gate** — On a case where the backbone is confidently wrong about localization (high log-prob, low IoU), the confidence-gated baseline produces a wrong click because its gate stays open on miscalibrated confidence. Our method instead computes g from actual re-localization hit rate, which is low for this backbone, so the gate closes and history attention takes over; we expect the gap between GCLI and the confidence-gated baseline to widen specifically on the small-element stratum, where log-prob miscalibration is most pronounced.

**vs. GCLI with global scalar g (no strata)** — On a case where the backbone grounds large elements well but small elements poorly, the global-scalar-g baseline uses a single g that is too permissive on small-element steps and too conservative on large-element steps, producing wrong clicks on small elements and under-using the latent on large elements. Our method instead uses stratum-conditioned g_{d_t}, so w_t is lower on small-element steps (triggering history fallback) and higher on large-element steps (full injection), so we expect a gain concentrated on the small-element stratum and parity on the large-element stratum.

**vs. GCLI with constant gate w=1.0** — On a case where the backbone's grounding is weak, the constant-gate ablation injects the misdirected latent at full strength, producing wrong clicks. Our method instead gates on the probe statistic, so we expect a gain concentrated on the small-element stratum where g is low.

### What would falsify this idea

If GCLI's gain over ungated FocusGUI is uniform across difficulty strata rather than concentrated on the small-element stratum where the probe predicts low g, the calibration gate is not doing what we claim. Equally, if the constant-gate ablation matches GCLI, the probe statistic carries no information beyond the presence of any gate. If the per-step diagnostic shows relevance_error_t is not monotone in grounding_error_t, assumption A fails and the gate under-corrects.

## References

1. Focused Omni-Modal GUI Agents
2. Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents
3. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
4. CogAgent: A Visual Language Model for GUI Agents
5. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
6. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
