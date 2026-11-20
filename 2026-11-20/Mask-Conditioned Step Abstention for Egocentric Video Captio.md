# Mask-Conditioned Step Abstention for Egocentric Video Captioning

## Motivation

Both branches of egocentric video captioning — MMEgo's memory-pointer prompting and EgoCaptioner's GRPO refinement — assume a fixed pipeline must always emit a complete narration, so under partial observability (brief actions, occluded hands, blurred intervals) they produce confident but unverifiable steps rather than signaling that the wearer's action cannot be confirmed from pixels. EgoCaptioner in particular trains only coverage and temporal-alignment rewards, giving it no training signal that distinguishes 'step visible and correctly timed' from 'step plausible but not visually supported'; the structural root cause is that abstention is never an action in the captioner's output space, so the policy has no way to trade a hallucinated step for a verifiable gap.

## Key Insight

Step-level verifiability is identifiable from a controlled masking intervention because the same underlying step, when its supporting frames are deliberately masked, must receive a lower verifiability score than when they are visible — a paired contrast that pins the score to visual support rather than to language prior or step plausibility.

## Method

**What it is.** We extend EgoCaptioner with a step-level verifiability head V_θ that, for each timed step (t_start, t_end, description), outputs a scalar s ∈ [0,1] indicating whether the step's narration is visually supported by the corresponding frame interval. At inference, steps with s below a calibrated threshold τ are replaced by a typed abstention token ("[unverifiable: <reason>]") instead of being narrated. Inputs: video frames, candidate step list from the base captioner. Outputs: per-step verifiability scores and an abstention-augmented caption.

**Load-bearing assumption (stated explicitly).** The paired score gap (s_pos − s_neg) is treated as a measure of visual support for a step only under assumption A: masking the step's own frame interval removes all pixel evidence for that step, so the gap reflects dependence on the step's own pixels rather than on contextual inference from unmasked frames, language prior, or step plausibility. This assumption fails (failure mode F) when the step is inferable from context outside the interval (e.g., a hand entering frame implies a grasp), in which case the gap reflects contextual inference rather than direct visual evidence. To make A testable rather than assumed, we use a leakage-controlled masking variant (mask a wider temporal window, specified below) and we validate the gap against independent visibility labels (experiment section).

**How it works.**

```pseudocode
# Phase 1: paired masking supervision (training)
# Masking modes (leakage-controlled):
#   mode=excise_wide: hard temporal excision of [t_k_start - W, t_k_end + W]
#                     with W = 1.0 s (30 frames at 30 fps), replaced by black frames
#   mode=context_only: keep only frames outside [t_k_start, t_k_end],
#                      i.e., context view with the interval itself excised
for each training example (video V, steps S_1..S_K):
    for each step S_k with interval [t_k_start, t_k_end]:
        # positive view: original frames
        h_pos = Encoder(V[t_k_start:t_k_end])
        s_pos = V_theta(h_pos, text_embed(S_k))
        # negative view 1: interval excised with widened window (leakage control)
        V_neg = apply_mask(V, interval=[t_k_start - W, t_k_end + W],
                           mode=excise_wide, W=1.0s)
        h_neg = Encoder(V_neg[t_k_start:t_k_end])
        s_neg = V_theta(h_neg, text_embed(S_k))
        # negative view 2: context-only control (interval removed, context kept)
        V_ctx = apply_mask(V, interval=[t_k_start, t_k_end],
                           mode=context_only)
        h_ctx = Encoder(V_ctx[t_k_start:t_k_end])
        s_ctx = V_theta(h_ctx, text_embed(S_k))
        # paired ranking loss + calibration loss
        L_rank += max(0, margin - (s_pos - s_neg))          # margin = 0.3
        L_rank += max(0, margin - (s_pos - s_ctx))          # context control
        L_cal  += BCE(s_pos, 1) + BCE(s_neg, 0) + BCE(s_ctx, 0)
    # steps whose interval is already occluded in source get label 0
    L_cal += BCE(V_theta(h_occluded, text_embed(S_k)), 0)
L_total = L_rank + lambda_cal * L_cal + lambda_grpo * L_GRPO(EgoCaptioner)
# lambda_cal = 1.0, lambda_grpo = 1.0, GRPO term inherited

# Phase 2: calibration on held-out split (per occlusion regime)
# Split held-out 10% of 50k examples into occlusion regimes by
# fraction of frames with detected hand/object occlusion (thresholds 0.1, 0.5).
for each occlusion regime r:
    fit isotonic regression f_r: s -> p(verifiable) on held-out examples in r
    produce reliability diagram; set tau_r = argmax_t (precision + recall)/2 on held-out

# Phase 3: inference with abstention
S = base_captioner(V)                       # EgoCaptioner steps
for S_k in S:
    s_k = V_theta(Encoder(V[t_k_start:t_k_end]), text_embed(S_k))
    r_k = occlusion_regime(V, [t_k_start, t_k_end])   # same thresholds as Phase 2
    p_k = f_{r_k}(s_k)
    if p_k < tau_{r_k}:
        emit "[unverifiable: step at t_k_start]"
    else:
        emit S_k
```

**Why this design.** We chose a paired masking intervention over a single-view confidence head because a single-view head can be fit by step plausibility alone (a common failure of log-prob gating), whereas the paired contrast forces the score to depend on the pixels in the step interval; the cost is that we must synthesize masked negatives, which may not match real occlusion statistics. We chose isotonic regression over temperature scaling because isotonic regression is non-parametric and monotone by construction, giving a reliability diagram that is honest even when the raw head is miscalibrated; the cost is that it needs a held-out split and can overfit with few bins. We chose a typed abstention token over silent omission because downstream consumers (QA, summarization) need to distinguish 'no step here' from 'step here but unverifiable'; the cost is a longer output vocabulary. Unlike EgoCaptioner's GRPO reward, which scores whole captions, our head scores each step independently, so abstention is localized rather than global. A domain expert would not describe this as 'GRPO + a confidence head' because the training signal is a paired counterfactual over the same step, not a scalar reward over the whole caption.

**Why it measures what we claim.** The paired score gap (s_pos − s_neg) measures visual support for a step because the only difference between the two views is whether the step's supporting frames are visible; this assumption (that masking is a valid intervention on visual support) fails when the step is inferable from context frames outside the interval (e.g., a hand entering frame implies a grasp), in which case the gap reflects contextual inference rather than direct visual evidence. The isotonic-calibrated score f(s) measures the probability that a step is verifiable because isotonic regression is monotone and fit on held-out labels; this assumption fails when the held-out split has a different occlusion distribution from test (e.g., Ego4D vs. EgoExo), in which case f(s) reflects the training distribution's base rate rather than the test video's. The abstention threshold τ measures the precision/recall trade-off on verifiability because τ is selected on the held-out reliability diagram; this assumption fails when the downstream cost of a false abstention differs sharply from a false narration, in which case τ should be re-tuned per application.

## Contribution

['A step-level verifiability head with a paired masking objective that turns abstention into an explicit output action for egocentric captioners, together with an isotonic calibration procedure that produces a reliability diagram over verifiability scores.', 'An empirical design principle: paired counterfactual masking yields verifiability scores that are monotone in visual support, whereas single-view confidence heads collapse onto step plausibility — demonstrated by the reliability-diagram separation between the two.', "A typed abstention output format and evaluation protocol (abstention precision/recall) for the five egocentric benchmarks used by EgoCaptioner, distinguishing 'unverifiable step' from 'no step'."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Ego4D held-out + EgoExo cross-domain split | In-domain fit, cross-domain calibration stress test |
| Primary metric | Step-level verifiability AUROC + abstention precision/recall | Measures ranking and thresholded decision jointly |
| Baseline | EgoCaptioner (no abstention) | Tests whether abstention adds value over base narration |
| Baseline | Log-prob confidence gating on EgoCaptioner steps | Tests paired contrast vs single-view plausibility |
| Baseline | Silent omission of low-confidence steps | Tests typed abstention vs silent dropping |
| Baseline | Conformal prediction on EgoCaptioner step log-probs | Strong selective-prediction alternative |
| Baseline | Monte-Carlo-dropout variance head on EgoCaptioner | Strong uncertainty alternative |
| Ablation-of-ours | Single-view head, no paired masking | Isolates contribution of counterfactual supervision |
| Ablation-of-ours | Paired masking with box-blur + keep_context=2 (original) | Tests leakage-controlled masking vs original |
| Ablation-of-ours | Paired masking with excise_wide only (no context control) | Isolates context-control contribution |
| Validation | Correlation of (s_pos − s_neg) with independent visibility labels | Tests whether gap measures visual support |
| Downstream | Off-the-shelf QA/summarization on (a) EgoCaptioner, (b) silent omission, (c) typed abstention | Measures event-recall and hallucination-rate deltas |

### Why this setup validates the claim

The central claim is that a step is verifiable only if its own interval's pixels support the narration, and that a paired masking contrast is what forces the score to depend on those pixels. The log-prob gating baseline directly tests this: it is the natural single-view alternative and should be fit by plausibility alone, so if our paired contrast adds nothing, the two should track each other on AUROC. The conformal-prediction and MC-dropout baselines test whether the paired contrast uniquely wins over strong selective-prediction and uncertainty alternatives; if it does not, the contribution is reframed around the typed-abstention output space rather than the head. The silent-omission baseline tests the second sub-claim, that typed abstention carries information downstream consumers need; if silent omission matches typed abstention on abstention precision/recall, the typing is decorative. The no-paired-masking ablation isolates the counterfactual signal from the calibration scaffolding, so any gain can be attributed to the contrast rather than isotonic regression. The leakage-controlled masking ablation (excise_wide vs. original box-blur + keep_context=2) tests assumption A directly: if the gap survives wider excision, the score is not merely reading context leakage. The visibility-label correlation validation tests whether (s_pos − s_neg) tracks an independent visibility label (hand/object annotations or pose-derived motion facts as in EgoCaptioner); a weak correlation falsifies the intervention. Step-level AUROC plus thresholded precision/recall is the right metric because the claim is about ranking steps by visual support and then committing to a localized decision; whole-caption scores would average away exactly the per-step localization we assert. The EgoExo split is included because our own stated failure mode is calibration drift under occlusion-distribution shift, so the design must be able to expose it rather than hide it. The downstream QA/summarization evaluation converts the significance claim from rhetorical to measured by reporting event-recall and hallucination-rate deltas across the three caption types.

### Expected outcome and causal chain

**vs. EgoCaptioner (no abstention)** — On a case where the wearer's hands leave frame and a brief grasp is narrated from prior context, the baseline emits the step as confident narration because it has no per-step support signal and its GRPO reward scores whole captions, so a single unsupported step is diluted. Our method instead scores that step's interval against its own pixels, the masked negative is indistinguishable from the positive here, so the gap collapses and the step falls below τ. We expect a noticeable abstention-rate increase concentrated on brief-action and out-of-frame subsets, with near-parity on well-framed manipulation steps.

**vs. Log-prob confidence gating** — On a case where a step's narration is fluent and plausible but the interval shows a different object, the log-prob gate assigns high confidence because token likelihood tracks linguistic plausibility, not pixel support; the gate is blind to the interval. Our method instead sees the masked-negative gap shrink when the described object is absent, so the paired score drops even though the text is fluent. We expect our AUROC to exceed the gate's mainly on the plausible-but-wrong subset, with the two converging on the subset where narration is both fluent and correct.

**vs. Silent omission** — On a case where a step is genuinely unverifiable, silent omission drops it and the downstream consumer cannot tell whether the wearer did nothing or the captioner failed, so summarization silently loses an event. Our method instead emits a typed token at the timestamp, preserving event presence while flagging unsupported content. We expect comparable abstention precision but a clear separation on downstream event-recall under omission, with typed abstention retaining events that silent omission erases.

**vs. Conformal prediction on EgoCaptioner step log-probs** — On a case where a step is plausible but visually unsupported, conformal prediction calibrates the log-prob score on a held-out set and thus inherits the same plausibility bias; its prediction sets will include the unsupported step at the target coverage because the score is high. Our method instead uses the paired gap, which drops when the interval is masked, so the unsupported step falls outside the prediction set. We expect our AUROC to exceed conformal prediction's on the plausible-but-wrong subset, with parity on well-supported steps.

**vs. Monte-Carlo-dropout variance head** — On a case where the step is ambiguous from pixels but context makes it likely, MC-dropout variance may be low because the model is confidently fitting the context prior, not the interval. Our method instead sees the gap shrink under interval excision, so the score drops even when dropout variance is low. We expect our AUROC to exceed MC-dropout's on the context-inferable subset, with the two converging on steps where pixels are unambiguous.

**vs. Single-view head, no paired masking** — On a case where a step is plausible but visually unsupported, the single-view head can assign high confidence from language prior alone because it never sees a masked negative. Our method instead forces the score to depend on the interval via the paired contrast, so the gap collapses when the interval is excised. We expect our AUROC to exceed the single-view head's on the plausible-but-wrong subset, with parity on well-supported steps.

**vs. Original box-blur + keep_context=2 masking** — On a case where the step is inferable from context outside the interval (hand entering frame implies grasp), the original masking leaves those cues intact, so s_neg stays high and the gap collapses on exactly the target subset. Our leakage-controlled excise_wide masking removes the wider window, so the gap reflects the interval's own pixels. We expect the leakage-controlled variant to show a larger gap on the brief-action/out-of-frame subset and a higher correlation with independent visibility labels.

**vs. Excise_wide only (no context control)** — On a case where context frames alone license the narration, excise_wide removes the interval but keeps context, so s_neg may still be high from context. The context-only control view removes the interval and keeps context, isolating the residual support attributable to the interval after context is partialled out. We expect the triplet (full / excised / context-only) to yield a larger gap on the target subset than excise_wide alone.

### What would falsify this idea

If the paired-masking ablation matches our full method on step-level AUROC, the counterfactual contrast is doing no work and the claim that masking forces pixel dependence is wrong. If our gains are uniform across all subsets rather than concentrated on the brief-action and out-of-frame subsets where the predicted failure mode lives, the mechanism is not the one we claim. If the correlation between (s_pos − s_neg) and independent visibility labels is weak, the intervention is not measuring what is claimed and the design must change.

## References

1. EgoCaptioner: Towards Comprehensive and Fine-Grained Egocentric Video Captioning
