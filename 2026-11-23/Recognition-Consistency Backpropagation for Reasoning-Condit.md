# Recognition-Consistency Backpropagation for Reasoning-Conditioned Text-Aware Super-Resolution

## Motivation

Text-aware super-resolution (TAISR) methods treat text restoration as a purely visual inpainting problem, and the diagnostic work of Reading the Unreadable (E4) shows this fails structurally on text that cannot be recovered from local pixels: all twelve evaluated SR models exhibit a consistent, substantial drop on reasoning-required text, and GTTCA/ReasonText are measurement instruments with no restoration mechanism. The root cause is that recognition confidence and pixel synthesis are optimized on disjoint surfaces — the SR generator never receives a gradient that tells it which pixel configurations would make a downstream recognizer more certain — so the joint identifiability of textual content and visual detail under severe low-resolution degradation is never established, only measured post hoc.

## Key Insight

A differentiable recognition-consistency term that back-propagates recognizer uncertainty through a text-region renderer into the SR generator's pixel output creates a shared gradient surface on which textual hypotheses and visual detail are forced to agree, turning recognition confidence from a post-hoc metric into a training signal that is monotone in restoration correctness by construction.

## Method

**What it is.** We propose RC-SR (Recognition-Consistency Super-Resolution), a TAISR estimator that takes a low-resolution image `I_LR` and outputs a high-resolution image `I_HR` together with a text-region mask `M` and a per-region recognition-confidence vector `c`. RC-SR couples an MLLM hypothesis generator, a differentiable text renderer, and an SR generator through a shared recognition-consistency loss, and refines the textual prior across `T=3` sampling steps.

**Load-bearing assumption (stated explicitly).** The recognition-consistency gradient is useful only if the recognizer's confidence is monotone in restoration correctness rather than in agreement with whatever string the hypothesis generator supplied. Because the loss target is the MLLM hypothesis `h_{t-1}` (not ground truth), minimizing `L_rec` alone would train the generator to make the recognizer read back `h_{t-1}` — a consistency/agreement objective, not a correctness objective. We therefore make the assumption explicit and add two concrete guards inside the existing loop: (i) a margin/ranking form of `L_rec` that requires the rendered output to be more recognizable as `h_{t-1}` than as hard negative strings sampled to be visually confusable with `h_{t-1}`, so the gradient rewards discriminative glyph detail rather than mere agreement; and (ii) a correctness gate that only back-propagates `L_rec` when an independent cross-recognizer agreement check confirms `h_{t-1}`. The assumption fails when the recognizer's training distribution does not cover the degraded font/style or when the MLLM hallucinates a plausible but wrong string; in that case `L_rec` reflects recognizer bias or hypothesis plausibility rather than legibility.

**How it works.**

```pseudocode
Input: I_LR, MLLM H, differentiable renderer R, recognizer C, cross-recognizer C2, SR generator G_theta, steps T=3
Output: I_HR, text mask M, confidences c

# Phase 1: hypothesis generation
h_0 = H(I_LR)                      # list of (text_string, bbox, confidence) hypotheses
M   = union of bboxes in h_0       # text-region mask

# Phase 2: iterative refinement loop
for t in 1..T:
    # (a) render current hypotheses into a differentiable text layer
    L_txt = R(h_{t-1}, M)          # RGB text layer, gradients flow to h_{t-1} params
    # (b) SR generator fuses LR image with rendered text layer
    I_HR^t = G_theta(I_LR, L_txt, M)
    # (c) recognition-consistency loss: margin/ranking form with hard negatives
    #     hard negatives N(h_{t-1}) = 5 strings sampled from a visually confusable set
    #     (same length, edit distance 1-2, same font/style statistics as h_{t-1})
    L_rec_pos = CE( C(I_HR^t * M), h_{t-1} )
    L_rec_neg = mean_{n in N(h_{t-1})} CE( C(I_HR^t * M), n )
    L_rec     = max(0, L_rec_pos - L_rec_neg + margin)   # margin = 0.2
    # (d) correctness gate: only back-propagate L_rec when cross-recognizer agreement confirms h_{t-1}
    #     agreement = (C(I_HR^t * M) == C2(I_HR^t * M) == h_{t-1})
    gate = 1 if agreement else 0
    # (e) perceptual + pixel fidelity
    L_fid  = L1(I_HR^t, I_HR_gt) + LPIPS(I_HR^t, I_HR_gt)
    # (f) joint objective
    L      = L_fid + lambda_rec * gate * L_rec         # lambda_rec = 0.5
    # (g) update generator and re-query MLLM with current HR estimate
    theta  = theta - eta * grad_theta L
    h_t    = H(I_HR^t)                          # refreshed hypotheses
    c_t    = C(I_HR^t * M).confidence           # per-region confidence

return I_HR^T, M, c_T
```

**Why this design.** We chose a differentiable recognizer-in-the-loop over a post-hoc GTTCA-style evaluation (as in Reading the Unreadable, E4) because post-hoc metrics cannot supply gradients to the generator, accepting the cost that the recognizer must be differentiable and may itself be imperfect. We chose a text-layer renderer `R` that composites hypotheses into the SR input over a hard mask `M` rather than free-form text conditioning, because it localizes the consistency gradient to text regions and avoids perturbing non-text pixels, accepting the cost that mask errors propagate. We chose `T=3` refinement steps with MLLM re-querying rather than a single forward pass, because recognition confidence on degraded text is only reliable after the first SR pass has denoised the region, accepting the cost of 3x inference. We chose a fixed `lambda_rec=0.5` over a learned weighting, because a learned weighting reintroduces the calibration problem that Anti-Pattern 3 warns about; the fixed weight keeps the loss surface interpretable. We added the margin/ranking form and the cross-recognizer gate because the original self-referential target `h_{t-1}` alone would reward agreement rather than correctness; the margin forces the generator to produce discriminative glyph detail, and the gate prevents back-propagating a gradient when the hypothesis is likely wrong. A domain expert would not describe RC-SR as a variant of PBR-SR (E2) or diffusion-inversion SR (E1): those methods use pretrained image priors as a frozen regularizer and never back-propagate a recognition signal into the generator, whereas RC-SR's load-bearing mechanism is the recognition-consistency gradient itself.

**Why it measures what we claim.** The recognition-consistency loss `L_rec` measures textual restoration correctness because the margin/ranking form with hard negatives is minimized exactly when the rendered hypothesis is more legible than visually confusable alternatives, and the cross-recognizer gate ensures the gradient is applied only when `h_{t-1}` is independently confirmed; this assumption fails when the recognizer's training distribution does not cover the degraded font/style or when both recognizers agree on a wrong string, in which case `L_rec` reflects recognizer bias rather than legibility. The per-region confidence `c_t` measures reasoning-required-text difficulty because confidence is computed on the SR output after the MLLM has been re-queried with the refined estimate; this assumption fails when the MLLM hallucinates a plausible but wrong string, in which case `c_t` reflects hypothesis plausibility rather than ground-truth correctness. The mask `M` measures text-region localization because it is the union of MLLM bounding boxes; this assumption fails when the MLLM misses small text, in which case `M` reflects salience rather than text presence.

## Contribution

['RC-SR, a TAISR framework that introduces a differentiable recognition-consistency loss back-propagated through a text-layer renderer into the SR generator, making recognition confidence a training signal rather than a post-hoc metric.', 'An iterative hypothesis-refinement loop (T=3) in which the MLLM is re-queried on the current HR estimate, so textual priors and pixel detail are jointly optimized across sampling steps.', 'A text-region-masked joint objective that localizes the consistency gradient to text pixels, preserving non-text fidelity while enforcing legibility.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ReasonText (reasoning-required subset) | Directly isolates reasoning-required text from locally recoverable text. |
| Primary metric | GTTCA on reasoning-required subset | Recognition-based, avoids detection errors confounding legibility. |
| Baseline | Twelve-model SR average (E4) | Establishes the blind-spot gap we claim to close. |
| Baseline | Text-conditioned SR with ground-truth text (E4) | Upper bound on what a text prior can buy. |
| Baseline | Diffusion-inversion SR (E1) | Tests whether a frozen generative prior alone suffices. |
| Baseline | RC-SR with lambda_rec=0 (no recognition gradient) | Isolates the recognition-consistency gradient as the load-bearing part. |
| Ablation-of-ours | RC-SR with hypothesis-oracle (feed ground-truth strings to renderer) | Isolates gradient quality from hypothesis quality. |
| Ablation-of-ours | RC-SR trained with recognizer C1, evaluated with unseen recognizer C2 | Tests whether learned legibility transfers rather than overfitting one recognizer. |
| Ablation-of-ours | RC-SR with margin=0 and gate=1 (original self-referential L_rec) | Tests whether the margin/ranking and gate guards are necessary. |

### Why this setup validates the claim

The central claim is that a recognition-consistency gradient back-propagated into the SR generator recovers reasoning-required text that prior-only SR cannot. ReasonText is the right dataset because it is the only one that separates reasoning-required from locally recoverable text, so a gain concentrated on the former and absent on the latter is the signature of our mechanism rather than generic sharpening. GTTCA is the right metric because it measures recognition of restored text without detection errors, so it cannot be inflated by drawing crisper but wrong glyphs. The twelve-model average tests the broad claim that existing SR has a systematic blind spot; the ground-truth-text-conditioned baseline tests whether our gain is merely what any text prior buys, since if we match it we have captured the prior's value without GT text; the diffusion-inversion baseline tests whether a frozen generative prior is sufficient. The lambda_rec=0 ablation is decisive: if removing the recognition gradient does not hurt on reasoning-required text specifically, the mechanism is not doing the work we claim. The hypothesis-oracle ablation separates gradient quality from hypothesis quality: if RC-SR with ground-truth strings does not outperform RC-SR with MLLM hypotheses, the bottleneck is the gradient rather than the hypothesis generator. The cross-recognizer ablation tests whether the learned legibility transfers to an unseen recognizer, which is what would make the result community-relevant rather than overfitting one recognizer. The margin=0/gate=1 ablation tests whether the guards against self-referential agreement are necessary for the claimed gain.

### Expected outcome and causal chain

**vs. Twelve-model SR average (E4)** — On a case where a degraded sign reads as a plausible but wrong word under local visual evidence, the baseline produces a sharp but misread string because it optimizes pixel and perceptual fidelity with no signal that the glyphs must be recognizable. Our method instead renders the MLLM hypothesis into the text region and back-propagates a margin/ranking cross-entropy from the recognizer, forcing the generator toward legible glyphs that are discriminable from confusable alternatives. We expect a clear GTTCA gap on the reasoning-required subset with near-parity on the locally recoverable subset.

**vs. Text-conditioned SR with ground-truth text (E4)** — On a case where the hypothesis is correct but the region is heavily degraded, the baseline with GT text restores well because it is handed the answer, but this is an oracle unavailable at test time. Our method instead must reach comparable legibility using only the MLLM-refreshed hypothesis, so we expect RC-SR to close most of the gap to this oracle on reasoning-required text, and if it does not, the hypothesis generator rather than the consistency gradient is the bottleneck.

**vs. Diffusion-inversion SR (E1)** — On a case where the prior strongly favors a common word over the true rare string, the baseline produces a fluent but incorrect restoration because its frozen prior is never corrected by a recognition signal. Our method instead overrides the prior via the recognition-consistency gradient localized to the mask, so we expect RC-SR to win specifically on cases where the true string is low-prior, with the diffusion baseline remaining competitive on high-prior text.

**vs. RC-SR with lambda_rec=0** — On a reasoning-required case, the ablation produces output statistically indistinguishable from a plain SR generator because the recognition gradient is absent and only fidelity terms act. Our method instead gains precisely where the recognizer can supply a corrective gradient, so we expect the full model to beat this ablation only on the reasoning-required subset, confirming the gradient is load-bearing rather than the renderer or MLLM alone.

**vs. RC-SR with hypothesis-oracle** — On a reasoning-required case where the MLLM hypothesis is wrong, the hypothesis-oracle ablation uses the ground-truth string and thus provides a clean gradient signal, while the full model must rely on the MLLM hypothesis. We expect the hypothesis-oracle to outperform the full model on such cases, with the gap quantifying the cost of hypothesis errors; if the gap is near zero, the gradient quality rather than hypothesis quality is the bottleneck.

**vs. RC-SR trained with C1, evaluated with C2** — On a reasoning-required case, if the learned legibility is specific to the training recognizer C1, the cross-recognizer evaluation will show a drop relative to C1 evaluation. We expect the cross-recognizer GTTCA to remain close to the C1 GTTCA, indicating that the margin/ranking form encourages generalizable glyph detail rather than recognizer-specific artifacts.

**vs. RC-SR with margin=0 and gate=1** — On a case where the MLLM hallucinates a plausible but wrong string, the unguarded ablation will render that wrong string more legibly, increasing agreement with the hypothesis but decreasing GTTCA. We expect the full model with margin and gate to outperform this ablation on reasoning-required text, confirming that the guards against self-referential agreement are necessary.

### What would falsify this idea

If RC-SR's gain over the twelve-model average is uniform across reasoning-required and locally recoverable text, or if the lambda_rec=0 ablation matches the full model on the reasoning-required subset, or if the margin=0/gate=1 ablation matches the full model, then the recognition-consistency gradient is not the mechanism and the central claim is wrong.

## References

1. Reading the Unreadable: Text-Aware Image Super-Resolution Needs Reasoning
