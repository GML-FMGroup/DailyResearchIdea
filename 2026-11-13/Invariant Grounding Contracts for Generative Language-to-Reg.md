# Invariant Grounding Contracts for Generative Language-to-Region Decoders

## Motivation

Generative visual perception models such as OVB (Open Vision Banana) recast grounding as conditional image editing, but they supervise the phrase-to-region map as a pointwise association: each referring expression is paired with one target mask, and nothing in the loss forbids the decoded region from changing when the phrase is paraphrased or when a distractor object is inserted. This is a structural failure of the supervision contract, not a capacity issue — replacing the readout head (as in Talking to DINO) or scaling the base generator cannot impose an invariance property that was never expressed in the objective, so paraphrase sensitivity and distractor confusion persist across model families.

## Key Insight

Grounding should be an invariant relation on the generative decoder — the decoded region is a function of the phrase's referential content rather than its surface form — and this relation can be enforced as a contrastive consistency contract over paraphrase and distractor interventions on the same decoder.

## Method

**What it is.** We introduce IGC (Invariant Grounding Contract), a training-time contract added to a generative visual-perception decoder that maps a referring expression plus an image to a decoded region. Inputs: an image, a referring expression, an automatically generated paraphrase set, and a distractor-perturbed image. Outputs: a region decoded by the same conditional-editing decoder, trained under an invariance/robustness loss in addition to the standard reconstruction loss.

**How it works.**

```pseudocode
# Per training step, given (image I, expression p, target mask M)
1. Generate K=4 paraphrases {p_k} of p with a frozen T5-large paraphraser,
   filtered by a MiniLM sentence-embedding cosine >= 0.80 to p.
   # Referent-preserving filter (applied after the MiniLM filter):
   # For each candidate p_k, decode r_k^0 = G_frozen(I, Phi(p_k)) using the
   # frozen base decoder (no IGC weights), and keep p_k only if
   # IoU(r_k^0, M) >= 0.70. Drop candidates below threshold; if fewer than
   # K=4 survive, use the surviving set (minimum 1).
2. Generate D=2 distractor images {I_d}: paste a random COCO object
   whose class differs from p's referent into a non-overlapping region.
   # Non-occluding paste gate: reject any paste whose bounding box has
   # IoU > 0.0 with the target mask M; resample until a non-overlapping
   # location is found (max 10 attempts; if all fail, skip this step).
3. Encode phrase embeddings e = Phi(p), e_k = Phi(p_k) with the frozen
   phrase encoder used by the base decoder.
4. Decode regions: r = G(I, e);  r_k = G(I, e_k);  r_d = G(I_d, e).
5. Reconstruction loss: L_rec = BCE(r, M) + Dice(r, M).
6. Paraphrase invariance loss (one-sided pull over decoded regions):
   L_para = mean_k [ 1 - sim(r, r_k) ]
   with sim = soft IoU on decoded masks, averaged over K.
   # One-sided: penalize divergence of r_k from r, but do NOT push
   # paraphrase regions apart from each other (no contrastive negatives).
7. Distractor robustness loss: L_dist = max(0, m - IoU(r, r_d) + IoU(r, r_d^shift)),
   where r_d^shift is r_d translated by the distractor's bounding box;
   margin m = 0.3. Applied only when the non-occluding paste gate in step 2
   succeeds.
8. Re-verification loss: decode the region, re-encode it with a frozen
   region encoder Psi, and require cos(Psi(r), e) >= cos(Psi(r), e_k)
   via L_ver = max(0, cos(Psi(r), e_k) - cos(Psi(r), e) + delta), delta = 0.05.
   # Psi and Phi are taken from the same pretrained grounding backbone
   # (frozen); compatibility is asserted by construction, not verified.
9. Total: L = L_rec + lambda_1 L_para + lambda_2 L_dist + lambda_3 L_ver,
   with lambda_1 = 0.5, lambda_2 = 0.3, lambda_3 = 0.2.
```

**Why this design.** We chose to enforce invariance on the *decoded region* rather than on the phrase embedding because the grounding map is what must be stable, and embedding-space smoothness does not imply output-space invariance; the cost is that mask-level losses are noisier than embedding losses, which we absorb with soft-IoU similarity. We chose a *one-sided pull* over a contrastive loss because contrastive negatives (the other paraphrases' regions) can push apart regions that are legitimately different when paraphrases shift the referent; the cost is that the one-sided pull does not prevent representational collapse, which we mitigate with the referent-preserving filter that removes referent-shifting paraphrases before they enter L_para. We chose *synthetic distractor paste-in* over real multi-object scenes because it gives a controlled counterfactual with a known distractor location, letting us define a margin-based loss that specifically penalizes region drift toward the distractor; the cost is a domain gap between pasted and natural distractors, which we bound by gating on non-occluding pastes so the margin reflects distractor rejection rather than occlusion handling. We chose a *re-verification* term that decodes and re-encodes rather than a direct phrase-region alignment because it closes the loop through the decoder, so the invariance is checked on the same surface used at inference; the cost is an extra forward pass through a frozen region encoder, and the assumption that Psi and Phi share a compatible embedding space is asserted by construction (same pretrained backbone) rather than verified. Unlike Talking to DINO, which replaces the readout head but keeps pointwise supervision, IGC changes the supervision contract itself — a domain expert would not describe it as a head swap.

**Why it measures what we claim.** The paraphrase invariance loss L_para measures *referential invariance* because we assume the referent-preserving filter (IoU(r_k^0, M) >= 0.70 under the frozen base decoder) removes paraphrases that shift the referent, so the surviving paraphrases preserve referential content while varying surface form; this assumption fails when the frozen base decoder itself is wrong on a paraphrase (decodes to a mask with IoU >= 0.70 against M despite a referent shift), in which case L_para reflects surface-form stability rather than referential stability. The distractor margin L_dist measures *distractor robustness* because we assume the non-occluding paste gate ensures the pasted object is a true non-referent and does not occlude the referent, so region drift toward it is drift caused by a non-referent; this assumption fails when the paste gate admits a bounding box that overlaps the referent's true extent despite zero IoU with M (e.g., partial occlusion outside the mask), in which case L_dist reflects occlusion handling rather than distractor rejection. The re-verification term L_ver measures *phrase-region faithfulness* because we assume the frozen region encoder Psi and phrase encoder Phi share a compatible embedding space (they are from the same pretrained grounding backbone), so higher cosine means the decoded region is closer to the phrase's referential content; this assumption fails when Psi is domain-shifted from the decoder's output distribution, in which case L_ver reflects encoder-space alignment rather than true grounding.

## Contribution

['A training-time invariance contract (IGC) for generative language-to-region decoders that operationalizes paraphrase invariance and distractor robustness directly on decoded regions via contrastive and margin losses, rather than on phrase embeddings or readout heads.', 'A decode-then-re-encode verification loss that closes the loop through the generative decoder, making the invariance property checkable on the same surface used at inference.', 'An automatic paraphrase-and-distractor intervention pipeline that turns any pointwise phrase-region dataset into an invariance-supervised one without new human annotations.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RefCOCOg + distractor-augmented val split | Paraphrase-rich expressions with controlled non-referents |
| Primary metric | cIoU on target mask | Directly measures decoded grounding, not embedding proximity |
| Baseline | Vision Banana (reported) | Strong generative-perception reference point |
| Baseline | Talking to DINO | Pointwise-supervised head-swap grounding competitor |
| Baseline | L_rec-only decoder | Isolates whether invariance contract itself adds value |
| Ablation-of-ours | IGC minus L_para & L_dist | Tests that gains come from invariance, not re-verification |
| Ablation-of-ours | IGC with embedding-space L_para & L_dist | Tests output-space vs. embedding-space contract |
| Diagnostic | Intervention-space IoU logging | Correlates loss terms with final cIoU gap |

### Why this setup validates the claim

The central claim is that moving supervision from pointwise phrase-region alignment to an invariance/robustness contract on the *decoded region* yields grounding that is stable to surface form and to non-referent distractors. RefCOCOg is the right testbed because its expressions naturally admit paraphrase variation, and the distractor-augmented split gives us a subset where the predicted failure mode should bite. Vision Banana tests the sub-claim that a generative-perception recipe without an explicit contract leaves surface-form and distractor sensitivity on the table; Talking to DINO tests the sub-claim that pointwise supervision of a readout head is insufficient even when the backbone is strong. The L_rec-only decoder is the sharpest control: it shares our architecture and data but lacks the contract, so any gap isolates the contract itself. The minus-invariance ablation checks that L_ver alone is not doing the work. The embedding-space ablation isolates whether the mask-space contract is a genuinely distinct mechanism from embedding-space regularization. cIoU is the correct metric because the claim is about the decoded mask, and embedding-space smoothness is exactly what we argue does *not* imply mask stability; a metric on the mask is the only one that can detect the predicted effect. Intervention-space diagnostics (per-step IoU(r, r_k) and IoU(r, r_d) vs. IoU(r, r_d^shift)) directly measure the intermediate quantities the loss terms claim to control, allowing us to check that they correlate with the final cIoU gap.

### Expected outcome and causal chain

**vs. Vision Banana** — On an expression like "the man on the left holding a cup" with a paraphrased query and a pasted non-referent cup on the right, Vision Banana's generative decoder produces a region that drifts toward the salient cup because nothing in its recipe penalizes non-referent attraction. Our method instead decodes the same expression under L_dist, which explicitly pushes the mask away from the pasted object's location (gated to non-occluding pastes), so we expect a clear cIoU gap on the distractor-augmented subset and near-parity on the clean subset.

**vs. Talking to DINO** — On a case where the expression is paraphrased from "the leftmost dog" to "the dog on the left", Talking to DINO's pointwise-supervised readout can shift its region because supervision never required two surface forms to map to the same mask. Our method instead enforces L_para as a one-sided pull across decoded regions, with paraphrases filtered to preserve the referent (IoU(r_k^0, M) >= 0.70), so paraphrases are pulled to the same mask without pushing apart legitimately different regions, and we expect a noticeable gap concentrated on the paraphrase-heavy subset with parity elsewhere.

**vs. L_rec-only decoder** — On a case where a distractor object of a different class is pasted adjacent to the referent, the L_rec-only decoder can bleed into the distractor because reconstruction loss alone does not distinguish referent from non-referent context. Our method instead adds the margin-based L_dist (gated on non-occluding pastes) and the closed-loop L_ver, so the decoded region stays anchored to the phrase-referent, and we expect a moderate but consistent cIoU advantage on the distractor subset, with the ablation showing that removing L_para and L_dist collapses most of the gain.

**vs. IGC with embedding-space L_para & L_dist** — On the paraphrase-heavy subset, the embedding-space variant regularizes Phi(p) and Phi(p_k) to be close, but does not directly constrain the decoded masks; we expect the mask-space contract to yield strictly higher cIoU on this subset, confirming that output-space invariance is a distinct mechanism.

### What would falsify this idea

If the IGC gain over the L_rec-only decoder is uniform across the clean, paraphrase-heavy, and distractor subsets rather than concentrated where the predicted failure modes occur, the contract is not doing what we claim. Equally falsifying: if the minus-invariance ablation matches full IGC, the re-verification term alone would explain the result and the invariance story is wrong; or if the embedding-space ablation matches full IGC, the mask-space contract is not a distinct mechanism.

## References

1. Open Vision Banana: A Fully Open Recipe for Generative Visual Perception
