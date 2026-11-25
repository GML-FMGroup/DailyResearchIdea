# Interventional Trace Identifiability for Pretext-Supervised Interleaved Multimodal Reasoning

## Motivation

ThinkMorph-SSL's two-stage pretext-to-instruction recipe derives image supervision from colorization and denoising, but its only acceptance criterion is downstream answer accuracy, so a generated intermediate image can be rewarded even when the textual rationale would have reached the same sub-goal without it. This is the meta-level failure of current interleaved-MCoT training: visual thoughts are optimized as data augmentation rather than as identifiable causes of specific reasoning steps, as also visible in ICoT's attention-inserted regions and VLM-R³'s region-conditioned policy, both of which reward visual traces without an interventional criterion linking each trace to the sub-goal it is supposed to serve. The structural root cause is that the training objective contains no counterfactual term in which the specific visual trace is removed while the textual sub-goal is held fixed.

## Key Insight

A visual thought is causally necessary for a textual sub-goal exactly when the model's conditional likelihood of that sub-goal, given the prefix and the visual trace, strictly exceeds its likelihood under an intervention that replaces the trace with a content-matched null while holding the prefix fixed, and this comparison can be optimized as a hinge on the log-likelihood gap without any reward model or answer-accuracy proxy.

## Method

```pseudocode
Input: two-stage training stream from ThinkMorph-SSL
  Stage-1 pretext traces: (x_corrupt, v_t, y_t) where v_t is a generated intermediate image
  Stage-2 instruction demonstrations: (q, v_1..v_T, y_1..y_T)
Model: MLLM policy pi_theta generating interleaved (v_t, y_t)

Phase A - standard interleaved SFT (retained from ThinkMorph-SSL):
  L_sft = -sum_t log pi_theta(y_t | prefix, v_t)

Phase B - interventional trace identifiability (ITI) loss, applied to every trace step t:
  1. Build the factual context: c_t = (prefix, v_t)
  2. Build the content-matched null context by drawing a trace from the model's own
     trace distribution for a distributionally matched but different prefix:
       v_t^null = pi_theta(. | prefix_null)   # prefix_null is a different question
                                              # from the same task family, matched on
                                              # answer length and image resolution;
                                              # sampled once per step with temperature 1.0
     (This replaces the earlier fixed blur+noise null. The null is in-distribution by
      construction because it is a sample from the same generator, and it removes the
      specific visual content of v_t while preserving the conditioning-slot format.)
     c_t^null = (prefix, v_t^null)
  3. Compute step log-likelihood gap, normalized per step length:
     Delta_t = [log pi_theta(y_t | c_t) - log pi_theta(y_t | c_t^null)] / |y_t|
     where |y_t| is the number of tokens in y_t.
  4. Hinge identifiability loss with margin m=0.5 (on the normalized gap):
     L_iti = sum_t max(0, m - Delta_t)
  5. Anti-spuriousness penalty (retained but inverted in sign relative to the earlier
     version, so the model is NOT trained to make the null plausible):
     L_spur = lambda * max(0, log pi_theta(y_t | c_t^null) - log pi_theta(y_t | c_t) + m)
     with lambda=0.1. This term penalizes the null context only when it is MORE
     likely than the factual context by more than the margin, i.e., it prevents the
     null from becoming a better predictor than the factual trace, without rewarding
     high null likelihood per se.

Phase C - joint objective:
  L = L_sft + alpha * L_iti + beta * L_spur
  alpha = 0.3, beta = 0.1, both linearly warmed up over first 2k steps.
  Hyperparameters alpha, beta, m are selected by a held-out sweep on a 512-example
  calibration split from the Stage-2 demonstrations (grid: alpha in {0.1,0.3,0.5},
  beta in {0.05,0.1,0.2}, m in {0.25,0.5,1.0}); the values above are the defaults
  before sweeping.

Phase D - null calibration check (run once before Phase B, on a held-out set of 512
  Stage-2 demonstrations):
  - Compute Delta_t for all steps under the content-matched null.
  - Fit a linear regression of Delta_t on measured low-level statistics of v_t^null
    (mean blur proxy = Laplacian variance, noise std, mean brightness) and report R^2.
    If R^2 > 0.3, the null is judged to be artifact-driven and the null generator is
    re-drawn with a different prefix_null sampling seed until R^2 <= 0.3.
  - Report the fraction of steps with Delta_t > m; this fraction is the calibration
    baseline used to set the inference filter threshold in Phase D (see below).

Phase E - trace filtering at inference:
  For each generated v_t, compute Delta_t on the fly (normalized as in Phase B);
  if Delta_t < m/2, regenerate v_t once with a crop-zoom transform (reuse VLM-R3-style
  region selection), else keep v_t. No extra training required. The threshold m/2 is
  fixed from the calibration check in Phase D (the 25th percentile of Delta_t on the
  calibration split), not asserted.
```

**Why this design.** We chose a content-matched null drawn from the model's own trace distribution over a different but distributionally matched prefix, rather than a fixed blur+noise transform, because a fixed corruption is an out-of-distribution input whose likelihood effect is dominated by distribution shift rather than by removal of recoverable content; the cost is that the null is stochastic and requires a calibration check (Phase D) to confirm it is not artifact-driven. We chose a hinge on the length-normalized log-likelihood gap over a learned causal-effect head because a learned head would re-introduce the coupling between visual-trace quality and answer correctness that the motivation identifies as the root problem, accepting the cost that the margin m=0.5 is a fixed hyperparameter (swept on a 512-example calibration split) rather than adaptive. We chose to apply the loss at every step rather than only at the final answer because the motivation is about sub-goal-level identifiability, accepting the cost that early traces with weak textual sub-goals receive noisy gradients. We chose the anti-spuriousness penalty, inverted so it penalizes only a null that beats the factual trace by more than the margin, over a mutual-information estimator because MI estimators over high-dimensional images are unstable at this scale, accepting the cost that lambda must be tuned. Unlike Process Reward Models (Lightman 2023) or VinePPO-style per-step Monte-Carlo value rollouts, which score a step by its contribution to final-answer correctness, our loss scores a visual trace by its contribution to the *immediately following textual sub-goal* under a fixed prefix, so a domain expert would not describe this as a variant of PRM: the outcome variable is a sub-goal likelihood, not a terminal reward, and the intervention is on the visual modality, not on the textual prefix.

**Why it measures what we claim.** The quantity Delta_t measures causal necessity of the visual trace v_t for the textual sub-goal y_t under the assumption A that the content-matched null (a trace sampled from the same generator for a different prefix) is a valid intervention that removes v_t's specific content while preserving its role as a conditioning slot; this assumption fails when the model conditions on low-level statistics (e.g., blur level, noise statistics) that correlate with the pretext task, in which case Delta_t reflects the model's sensitivity to corruption artifacts rather than to visual content. Phase D tests A directly by regressing Delta_t on measured blur/noise statistics and requiring R^2 <= 0.3. The hinge loss L_iti measures identifiability of the trace because it is minimized exactly when the factual log-likelihood exceeds the null by at least m (after length normalization), under the assumption that the null context is a plausible prefix that the model would otherwise accept; this assumption fails when the model assigns near-zero likelihood to y_t under the null for reasons unrelated to v_t (e.g., distribution shift from the different prefix), in which case the hinge is satisfied trivially and L_spur is what actually enforces the constraint. The anti-spuriousness term L_spur measures the plausibility of the null context because it penalizes the model only when the null likelihood exceeds the factual likelihood by more than the margin, under the assumption that a null that beats the factual trace indicates the model is exploiting the null rather than the trace; this assumption fails when the model memorizes the null distribution during training, in which case L_spur reflects memorization rather than counterfactual fairness. The inference-time filter using Delta_t < m/2 measures trace dispensability because it compares the factual and null step likelihoods on the same prefix, under the assumption that the margin m/2 (set from the 25th percentile of the calibration split) is a stable threshold across steps; this assumption fails when step-level likelihood scales vary widely (e.g., long vs. short sub-goals), in which case the filter reflects step length rather than trace necessity — which is why Delta_t is length-normalized.

## Contribution

["An interventional trace identifiability (ITI) loss that converts ThinkMorph-SSL's pretext-derived interleaved traces into causally grounded supervision by requiring each generated visual thought to raise the likelihood of its immediately following textual sub-goal above a content-matched null context, with an anti-spuriousness penalty that keeps the null a fair counterfactual.", 'A training-time and inference-time criterion for visual-thought dispensability that operates at sub-goal granularity rather than at final-answer granularity, together with a one-shot crop-zoom regeneration filter for traces whose interventional gap falls below half the margin.', 'A design principle for interleaved MCoT supervision: visual traces should be optimized against a step-level counterfactual outcome, not against terminal answer accuracy, because terminal-accuracy objectives cannot distinguish necessary from decorative visual thoughts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ThinkMorph-SSL twelve-benchmark suite | Same suite as source; isolates our loss's effect |
| Primary metric | Accuracy on spatial / fine-grained-cue subsets | Where trace necessity should matter most |
| Baseline | Instruction tuning alone (same budget) | Isolates pretext-stage contribution from ITI |
| Baseline | ThinkMorph-SSL full two-stage recipe | Tests whether ITI adds beyond source training |
| Baseline | Interleaved-Modal CoT (ADS) prompting | Tests whether training-time ITI beats inference-time selection |
| Baseline | Corruption-contrastive loss (blur+noise null, no calibration) | Tests whether content-matched null beats naive null |
| Baseline | Token-deletion null (drop v_t, keep prefix) | Tests whether content-matched null beats deletion null |
| Ablation-of-ours | Ours without L_iti (L_sft + L_spur only) | Isolates identifiability hinge from anti-spuriousness |
| Ablation-of-ours | Ours with L_spur removed (L_sft + L_iti only) | Tests whether anti-spuriousness term is load-bearing |
| Diagnostic | Regression of Delta_t on blur/noise statistics (R^2) | Tests assumption A directly (Phase D) |

### Why this setup validates the claim

The central claim is that a visual trace earns its place in an interleaved chain only if the following textual sub-goal is causally dependent on it, and that a step-level likelihood gap under a content-matched null intervention can enforce this. The design is falsifiable because the baselines each remove a different competing explanation. Instruction tuning alone removes the pretext-stage visual skill, so any gain of ours over it could be attributed to ThinkMorph-SSL rather than to ITI; the full ThinkMorph-SSL recipe removes that confound, so the marginal gain of ours over it is the ITI effect proper. ADS prompting is inference-only region selection, so if our training-time loss beats it, the claim that identifiability must be trained rather than prompted is supported. The corruption-contrastive baseline uses the original blur+noise null without the Phase D calibration check, so if our content-matched null beats it, the claim that the null design matters is supported; the token-deletion baseline tests whether the conditioning-slot preservation is necessary. The ablation without L_iti separates the hinge from the anti-spuriousness term, and the ablation without L_spur tests whether the inverted penalty is doing any work. The Phase D regression of Delta_t on blur/noise statistics directly tests assumption A: if R^2 > 0.3, the null is artifact-driven and the mechanism story fails. Accuracy on spatial and fine-grained-cue subsets is the right primary metric because Delta_t is defined to be large precisely when a visual trace carries recoverable spatial content; if the loss works, gains must concentrate there, and parity on perception-heavy or OCR subsets is the predicted null.

### Expected outcome and causal chain

**vs. Instruction tuning alone** — On a case where a question requires locating a small object before a spatial sub-goal, the instruction-tuned model emits a plausible-sounding visual trace that is not actually consulted, because it never learned that the trace must be recoverable. Our method instead is pushed by the hinge to raise the factual step likelihood above the content-matched-null likelihood, so the trace becomes load-bearing. We expect a noticeable accuracy gap on the spatial and fine-grained-cue subsets but near-parity on OCR and general-scene subsets.

**vs. ThinkMorph-SSL full recipe** — On a case where the pretext stage taught the model to generate a visually coherent but semantically redundant intermediate image, the source recipe accepts it because its SFT loss only rewards matching the demonstration trace, and a redundant trace still matches. Our method instead penalizes that step through L_iti, since the content-matched null yields nearly the same sub-goal likelihood. We expect a modest but concentrated gain over the source recipe on the hardest spatial subset, with the gap shrinking on easier subsets where redundant traces are rare.

**vs. Interleaved-Modal CoT (ADS) prompting** — On a case where the attention map points at a salient but irrelevant region, ADS inserts that region into the chain because it selects by attention rather than by sub-goal necessity, and the model then reasons over the wrong visual evidence. Our method instead filters at inference via Delta_t < m/2 and regenerates with a crop-zoom, because the loss trained the model to make the trace's contribution measurable. We expect our method to beat ADS specifically on questions with distractor regions, with parity on single-salient-object questions.

**vs. Corruption-contrastive loss (blur+noise null, no calibration)** — On a case where the model conditions on blur level rather than content, the corruption-contrastive baseline's Delta_t is large for artifact reasons, so its hinge is satisfied trivially and its inference filter triggers on the wrong steps. Our method instead uses a content-matched null and the Phase D calibration check, so Delta_t reflects content dependence. We expect our method to beat the corruption-contrastive baseline on the spatial subset, with the gap shrinking on subsets where blur artifacts are rare.

**vs. Token-deletion null (drop v_t, keep prefix)** — On a case where deleting the visual trace changes the sequence length and lets the model exploit positional artifacts, the token-deletion baseline's Delta_t reflects positional sensitivity rather than visual content. Our method instead preserves the conditioning slot with a content-matched null, so Delta_t reflects content. We expect our method to beat the token-deletion baseline on the spatial subset, with parity on subsets where positional artifacts are weak.

**vs. Ours without L_iti** — On a case where the model has learned to keep the null context plausible by memorizing the null distribution, the ablation satisfies L_spur without ever enforcing a factual-null gap, so Delta_t stays small at inference and the filter never triggers. Our method instead keeps the hinge active, so the gap is maintained. We expect the ablation to match ours on easy subsets but to lose the filter-driven gains on the hardest spatial subset, confirming the hinge rather than the penalty drives the effect.

**vs. Ours without L_spur** — On a case where the null context becomes a better predictor than the factual trace, the ablation without L_spur has no term to penalize that, so Delta_t can go negative and the hinge saturates. Our method instead keeps the inverted penalty active, so the null cannot beat the factual trace by more than the margin. We expect the ablation to match ours on easy subsets but to lose on the hardest spatial subset where null-beats-factual cases are most frequent.

### What would falsify this idea

If our gain over the full ThinkMorph-SSL recipe is uniform across all twelve benchmarks rather than concentrated on spatial and fine-grained-cue subsets, the claim that step-level visual-trace identifiability is the operative mechanism is wrong. Likewise, if the ablation without L_iti matches ours everywhere, the hinge is not doing the causal work; if the Phase D regression yields R^2 > 0.3, the content-matched null is artifact-driven and assumption A fails.

## References

1. ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks
