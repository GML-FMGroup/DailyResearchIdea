# Causal Perturbation Verification of Latent Grounding in Frozen-Backbone GUI Agents

## Motivation

Frozen-backbone GUI agents such as FocusGUI inject a compressed relevance latent into the decoder, but the latent's output is assumed to be a valid decision reference without any identifiability property: if the backbone's visual grounding is wrong (e.g., it attends to a visually salient but action-irrelevant region), the latent silently injects the wrong evidence and action supervision alone cannot detect this because no grounding labels exist. This is a meta-level failure that recurs across the whole family of frozen-backbone focusing methods (FocusGUI and the pixel-level grounding approaches of CogAgent and Pix2Struct): the decision reference is never tested for causal necessity, so grounding errors propagate undetected. The root cause is structural — relevance estimators are trained to predict actions, and an action can be predicted from a spurious region, so action loss alone cannot separate causally necessary grounding from correlational grounding.

## Key Insight

A relevance latent is a valid decision reference exactly when its contribution is invariant under action-irrelevant visual perturbations and necessary under action-relevant ones, which is a property of the estimator's input-output map that can be tested by intervening on the frozen backbone's visual tokens rather than by observing labels.

## Method

**What it is.** Causal Perturbation Verifier (CPV) is a training-time verifier attached to a frozen-backbone GUI agent (FocusGUI-style adapter). Given a step's screenshot, instruction, and candidate action label, CPV produces a scalar *grounding-consistency score* by running the frozen backbone and relevance estimator on a set of token-level interventions and measuring whether the latent injection's effect on the action logit is causally tied to the action-relevant region.

**Load-bearing assumption (stated explicitly).** CPV's delta measure assumes **A**: the only pathway from region r to the action logit is through the intervened visual tokens and the frozen backbone's fixed computation, i.e., the backbone has no residual pathway (global attention, context reconstruction, positional leakage) that lets the action logit recover masked/swapped content. When A holds, delta reflects region r's causal necessity. When A fails — the backbone reconstructs masked content from context, or shuffling destroys spatial order that the backbone relies on — delta reflects the backbone's redundancy rather than the region's necessity. CPV therefore includes a redundancy-control step (step 3b below) that measures and subtracts this violation before forming the contrast.

**How it works.**

```pseudocode
Input: screenshot S, instruction I, action label a*, frozen backbone B, relevance estimator R_theta, candidate region set C (from R_theta's top-k attention, k=5)
Output: verifier loss L_cpv; at test time a scalar consistency score

1. Compute base latent:  z0 = R_theta(S, I);  logits0 = B(S, I, inject=z0);  p0 = softmax(logits0)[a*]
2. Build perturbation set P (three families):
   a. MASK(r): zero the visual tokens of candidate region r  (action-relevant if r in C)
   b. SWAP(r, r'): replace tokens of r with tokens of a distractor region r' of same area
   c. SHUFFLE: permute visual tokens within a random non-candidate region (action-irrelevant control)
3. For each p in P:
      z_p = R_theta(S_p, I)                 # re-estimate relevance on perturbed input
      logits_p = B(S_p, I, inject=z_p)
      delta_p = logit_p[a*] - logit_0[a*]   # causal effect on the action label
   3b. Redundancy control (applied to every delta_p):
       - Sufficient intervention: for MASK(r) and SWAP(r,r'), also compute delta_p^attn by attention-masking region r across all L layers (zero its key/value entries in every self-attention block) instead of only zeroing input tokens. Use delta_p^attn as the primary causal estimate; delta_p (input-token only) is logged as a diagnostic of backbone redundancy.
       - Leakage check: compute delta_p^leak by giving the backbone region r's content through a separate additive channel (concatenate r's token embeddings to the instruction tokens) while masking r in the visual stream; require |delta_p^leak| < 0.1 * |delta_p^attn| for the estimate to be trusted; otherwise down-weight that perturbation's contribution by factor 0.1.
       - Null calibration: replace SHUFFLE with a permutation-invariant null that permutes only content embeddings within a region while preserving positional encodings, and a matched random-region mask of equal area to the candidate region; delta_irr is the average over these two nulls.
4. Contrastive objective over (relevant, irrelevant) pairs:
      L_cpv = -log( exp(-delta_rel/tau) / (exp(-delta_rel/tau) + sum_{irr} exp(-delta_irr/tau)) )
      with tau=0.1; delta_rel = mean of delta_p^attn over MASK/SWAP on candidate regions (after leakage down-weighting), delta_irr = mean over the two nulls in 3b.
5. Train R_theta with L = L_action + lambda * L_cpv,  lambda=0.5, R_theta only (backbone frozen)
```

At inference, the consistency score is the normalized gap between the largest action-relevant delta (delta_rel) and the largest action-irrelevant delta (delta_irr), computed with the same redundancy control (3b). Steps with score below a fixed threshold (calibrated on a 512-example action-only validation split) are flagged for a re-grounding retry.

**Why this design.** We chose to perturb the frozen backbone's *visual tokens* rather than the input image pixels because token-level masking gives an exact, reversible intervention whose effect on the latent injection is attributable to the masked region, accepting the cost that token boundaries may not align with GUI element boundaries (mitigated by using R_theta's own attention-derived candidate regions rather than a fixed grid). We chose a *contrastive* objective over a regression on delta magnitudes because the verifier only needs to rank action-relevant above action-irrelevant perturbations, not calibrate absolute effects, accepting that the score is ordinal and requires a threshold at test time. We chose to train only R_theta and keep the backbone frozen because unfreezing would let the backbone absorb the perturbation signal and destroy the causal interpretation of delta, accepting that CPV cannot correct backbone grounding errors, only detect and route around them. This is not a variant of FocusGUI: FocusGUI learns *what to inject* from action supervision alone, whereas CPV learns *whether the injection is causally necessary* by intervening on the injection's input — a signal that no action-only loss on the unperturbed input can produce. The redundancy-control step (3b) is included because the no-residual-pathway assumption A is structurally at risk for transformer backbones; by using attention-masking as the sufficient intervention and calibrating against a permutation-invariant null, delta_rel and delta_irr measure necessity net of backbone redundancy rather than raw perturbation sensitivity.

**Why it measures what we claim.** The quantity `delta_p^attn = logit_p[a*] - logit_0[a*]` under attention-masking measures *causal necessity of region r for the action label* because we assume A: the only pathway from region r to the action logit is through the visual tokens we intervene on and the frozen backbone's fixed computation; this assumption fails when the backbone has residual pathways (e.g., global attention that reconstructs masked content from context), in which case delta reflects the backbone's redundancy rather than the region's necessity — which is why step 3b attention-masks across all layers and applies the leakage check. The contrastive ranking of delta_rel over delta_irr measures *grounding consistency* because we assume the null perturbations (content-only shuffle and matched random-region mask) are action-irrelevant by construction; this assumption fails when a shuffled region happens to contain task-critical text, in which case the verifier penalizes a genuinely relevant region — mitigated by the matched random-region mask. The invariance of z_p to the null perturbations measures *identifiability of the decision reference* because we assume the relevance estimator's output should not depend on order-within-region of non-candidate tokens; this fails when the estimator uses positional encodings that leak shuffle information, in which case the score reflects encoding artifacts rather than grounding — mitigated by permuting only content embeddings while preserving positional encodings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | GUI grounding benchmark with region-annotated screenshots (e.g., ScreenSpot-style) | Provides ground-truth action-relevant regions for causal validation |
| Primary metric | Step success rate on flagged-error subset | Directly measures whether CPV routes around grounding failures |
| Secondary metric | Pearson correlation between CPV consistency score and ground-truth grounding correctness (region IoU > 0.5) | Validates score reflects grounding quality, not just flagging |
| Secondary metric | False positive rate and false negative rate of the flagging mechanism (threshold calibrated on 512-example action-only validation split) | Quantifies flagging reliability |
| Baseline | Frozen backbone without FocusGUI | Tests whether CPV's gain exceeds adapter-only benefit |
| Baseline | FocusGUI adapter (action-only training) | Tests whether causal signal beats action supervision alone |
| Baseline | Random-perturbation verifier | Tests whether causal contrast matters vs any perturbation |
| Ablation-of-ours | CPV with SHUFFLE-only (no MASK/SWAP) | Isolates contribution of relevant-region interventions |
| Ablation-of-ours | CPV without redundancy control (input-token masking only, no attention-mask, no leakage check) | Quantifies violation of assumption A |
| Diagnostic | Backbone redundancy measure: compare delta under input-token masking vs. attention-masking across all layers | Directly measures residual pathway strength |

### Why this setup validates the claim

The central claim is that causal necessity of the injected region, measured by token-level interventions on a frozen backbone, yields a grounding-consistency signal that action-only training cannot produce. The dataset must supply region annotations so that MASK and SWAP perturbations target genuinely action-relevant versus distractor regions — without ground-truth regions, the contrastive pairs are unverifiable. The primary metric, step success rate on the subset of steps CPV flags as low-consistency, is the right metric because the claim predicts that flagged steps are precisely the ones where the baseline's grounding is causally wrong; if CPV's flag correlates with baseline failure, retry should recover those steps. The secondary calibration metrics directly test the load-bearing assumption A: if CPV's consistency score correlates with ground-truth grounding correctness (region IoU > 0.5) and the flagging FPR/FNR are acceptable, then delta measures necessity net of backbone redundancy; if not, assumption A is violated. The diagnostic comparing input-token masking vs. attention-masking quantifies how much of delta is due to residual pathways. The Frozen-backbone baseline tests whether CPV's gain is merely from the adapter; the FocusGUI baseline tests whether causal signal adds over action supervision; the random-perturbation baseline tests whether the contrast structure matters; the SHUFFLE-only ablation tests whether MASK/SWAP interventions carry the signal; and the no-redundancy-control ablation tests whether the redundancy control is necessary. Together these form a falsifiable test: if CPV's flags do not concentrate on baseline failure cases, or if the no-redundancy-control ablation matches full CPV, the causal-necessity claim is wrong.

### Expected outcome and causal chain

**vs. Frozen backbone without FocusGUI** — On a case where the instruction says "click the submit button" but the backbone attends to a visually salient header, the baseline produces a click on the header because it has no mechanism to tie the action logit to the instruction-relevant region. Our method instead computes a low consistency score because MASK on the header region barely changes the action logit while MASK on the actual submit button does, so CPV flags the step for re-grounding. We expect a noticeable gap on the flagged subset (success rate recovery from near-chance to substantially above chance) but parity on unflagged steps, where the backbone was already grounded.

**vs. FocusGUI adapter (action-only training)** — On a case where two regions are visually similar (e.g., two "OK" buttons in a dialog), FocusGUI produces a click on the wrong one because its relevance estimator was trained only to maximize action likelihood on the unperturbed input, which does not penalize injecting a distractor region that happens to correlate with the label. Our method instead penalizes this via the contrastive objective: SWAP between the two regions yields a small delta, so the consistency score is low and the step is flagged. We expect CPV to reduce errors specifically on visually ambiguous, multi-candidate screens, with the largest gains on the subset where FocusGUI's attention entropy is high.

**vs. Random-perturbation verifier** — On a case where the action-relevant region is large and the irrelevant region is small, a random-perturbation verifier produces a noisy score because random masks sometimes hit relevant tokens by chance, inflating delta_irr and erasing the contrast. Our method instead uses R_theta's own candidate regions to define relevant versus irrelevant interventions, so the contrast is structurally aligned with the grounding hypothesis. We expect CPV's score distribution to be more bimodal (clear separation between correct and incorrect groundings) than the random verifier's, yielding higher precision on the flagged-error subset.

**vs. CPV with SHUFFLE-only (no MASK/SWAP)** — On a case where the relevant region is text-dense, SHUFFLE-only produces a near-zero delta for both relevant and irrelevant regions because shuffling preserves the token multiset and the backbone reconstructs content from context, so the contrast collapses. Our method with MASK/SWAP produces a large delta on the relevant region because masking removes the content entirely, restoring the causal contrast. We expect the SHUFFLE-only ablation to show greatly reduced separation between relevant and irrelevant deltas, confirming that MASK/SWAP are the load-bearing interventions.

**vs. CPV without redundancy control (input-token masking only)** — On a case where the backbone reconstructs masked content from context (e.g., repeated button labels), input-token masking yields a small delta for the relevant region because the backbone re-derives its content, while attention-masking across all layers yields a large delta because it severs the pathway. Our method with redundancy control (attention-masking + leakage check) restores the contrast, whereas the no-control ablation shows inverted or collapsed separation. We expect the diagnostic to show that input-token masking underestimates delta_rel by a factor of 2–5× on text-dense regions, and that the no-control ablation's flagging FPR is substantially higher.

### What would falsify this idea

If CPV's consistency score is uniformly low across both baseline-failure and baseline-success steps — i.e., the flag does not concentrate on cases where the baseline's grounding is causally wrong — then the causal-necessity signal is not measuring grounding consistency, and the central claim fails. Similarly, if the SHUFFLE-only ablation matches full CPV's separation, or if the no-redundancy-control ablation matches full CPV's separation, the MASK/SWAP intervention design or the redundancy control is not doing the claimed work.

## References

1. Focused Omni-Modal GUI Agents
2. Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents
3. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
4. CogAgent: A Visual Language Model for GUI Agents
5. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
6. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
