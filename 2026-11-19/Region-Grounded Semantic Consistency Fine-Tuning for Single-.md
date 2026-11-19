# Region-Grounded Semantic Consistency Fine-Tuning for Single-Pass Parallel Image Generation

## Motivation

Single-pass parallel generators such as Jeveration repurpose a frozen vision-language model to emit all image positions in one forward pass, but they only enforce global coherence through a shared random state; nothing in the training objective ties any specific image region to the conditioning text semantics, so compositional elements (rendered words, charts, graphical symbols) can drift from the prompt while the whole image stays locally plausible. DnD-Transformer shows that autoregressive image models can spontaneously produce text-graphic content when trained on images alone, yet it likewise offers no verifiable invariant that a generated glyph or object actually corresponds to the requested concept. The structural root cause is that both methods treat the frozen VLM as an implicit prior whose semantics will 'leak' into pixels, with no differentiable quantity that measures region-to-text alignment, so misalignment is never penalized during training.

## Key Insight

A frozen VLM's own cross-modal attention provides a region-conditioned alignment score whose gradient, when routed back through the parallel decoder's soft code assignments, turns an implicit hope about backbone semantics into an explicit, trainable region-text consistency invariant that the single-pass architecture otherwise cannot express.

## Method

**What it is.** We propose SCIP (Semantic Consistency Invariant Penalty), a fine-tuning procedure for Jeveration-style single-pass parallel decoders. Inputs: a prompt $c$, the frozen VLM $\mathcal{V}$, and the parallel decoder $\mathcal{D}_\theta$ that maps a shared random state $z$ and prompt embedding to per-position discrete-code logits $\ell\in\mathbb{R}^{N\times K}$. Outputs: an updated decoder $\theta'$ that produces images whose VLM-attended regions align with prompt-referenced concepts, while retaining one-forward-pass generation.

**How it works.**

```pseudocode
# Phase 1: region proposal from the frozen VLM (no decoder gradients)
for each prompt c in batch:
    spans = noun_phrases_and_glyph_queries(c)        # e.g. "red cube", "the word CAT"
    A = VLM_cross_attention(c, spans)                 # [L_layers, N_pos, |spans|], frozen
    R_s = top_k_positions(mean_over_layers(A[:,:,s]), k=0.05*N)   # region mask per span s
    # attention-concentration gate: compute entropy H_s of mean_over_layers(A[:,:,s])
    # apply L_cons for span s only if H_s < H_thresh (H_thresh = 0.7 * log(N))

# Phase 2: single-pass generation with soft codes
z ~ N(0, I)                                           # shared random state (Jeveration)
logits = D_theta(z, VLM_text_embed(c))                # [N, K], one forward pass
p = softmax(logits / tau),  tau = 1.0
I_soft = decode_soft(p)                               # differentiable image, [H, W, 3]
I_hard = decode_hard(argmax(logits))                  # committed discrete image, [H, W, 3]
# straight-through: forward uses I_hard, backward uses I_soft
I_st = I_hard + (I_soft - I_soft.detach())            # straight-through estimator

# Phase 3: region-text consistency score via frozen VLM
for each span s with region mask R_s and gate H_s < H_thresh:
    f_img_s = VLM_region_pool(I_st, R_s)              # pooled visual feature from straight-through image
    f_txt_s = VLM_text_pool(span_s)                   # pooled text feature
    a_s     = cosine(f_img_s, f_txt_s)                # alignment in [-1, 1]
L_cons = mean_s( 1 - a_s )                            # consistency invariant loss
L_task = cross_entropy(logits, target_codes)          # original Jeveration objective
L = L_task + lambda * L_cons,  lambda = 0.5

# Phase 4: update only decoder, VLM stays frozen
update theta via dL/dtheta through I_st and VLM_region_pool
```

Hyperparameters are inline: region top-k fraction 0.05, temperature $\tau=1.0$, consistency weight $\lambda=0.5$, cosine alignment, frozen VLM throughout, attention-entropy gate $H_{\text{thresh}} = 0.7 \log N$.

**Why this design.** We chose a *frozen* VLM as the verifier rather than training a separate alignment network because the decoder already inherits the VLM's visual-semantic space, so alignment scores live in the same representation the decoder was built on, accepting the cost that verifier errors are inherited rather than corrected. We chose region masks derived from the VLM's own cross-attention rather than an external segmentation model because attention gives span-conditioned, prompt-specific regions without a second training pipeline, accepting the cost that attention maps can be diffuse for abstract prompts. We chose a straight-through estimator that computes the alignment on the hard-sampled image while routing gradients through the soft codes, rather than a pure soft-code decode, because the invariant must be enforced on the actual committed output and soft-code gradients do not reliably transfer to the sampled discrete output; we accept the cost that the straight-through gradient is a biased proxy for the true discrete gradient. We chose a single consistency term over per-element classifiers because it penalizes any misalignment uniformly without enumerating element types, accepting the cost that the loss is coarser than a typed detector. A domain expert would not describe SCIP as a variant of a retrieval-based verifier (e.g., CLIP-score reranking) because no candidate images are retrieved or reranked: the invariant is enforced *inside* the single forward pass by back-propagating region-text alignment through the decoder's own soft code assignments, which retrieval-based scoring cannot do.

**Why it measures what we claim.** The quantity $a_s = \cos(f_{img,s}, f_{txt,s})$ measures region-text semantic alignment because the frozen VLM's joint embedding is assumed to place a correctly rendered region near its span's text embedding; this assumption fails when the span is abstract or polysemous (e.g., "style"), in which case $a_s$ reflects embedding-space proximity of the region's dominant visual statistics rather than true concept correspondence. The region mask $R_s$ from cross-attention measures span-specific spatial grounding because attention is assumed to concentrate on positions the VLM would attend to when reading the span; this fails when attention is diffuse, in which case $R_s$ reflects generic saliency rather than span-specific location, and we therefore gate $L_\text{cons}$ on attention entropy $H_s < 0.7 \log N$ so that only verifiably span-specific masks contribute. The straight-through image $I_\text{st}$ measures the decoder's committed output because its forward value is the hard-sampled image $I_\text{hard}$ while its backward pass uses the soft codes; this fails when the straight-through gradient is a poor proxy for the true discrete gradient, in which case $I_\text{st}$ reflects a biased gradient estimate rather than the exact discrete output.

## Contribution

(1) SCIP, a fine-tuning framework that installs an explicit region-text consistency invariant into single-pass parallel image generation by back-propagating a frozen-VLM alignment score through soft discrete-code assignments, requiring no architectural change to the decoder. (2) The finding that region masks derived from the frozen VLM's own cross-attention are sufficient to localize compositional elements (rendered text, graphical symbols) for consistency supervision, avoiding an external segmentation pipeline. (3) An analysis of the trade-off between soft-code proxy alignment and final discrete-sample alignment, yielding a practical schedule for the consistency weight.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ISG-Bench interleaved text-image prompts | Complex language-vision dependencies stress region-text alignment |
| Primary metric | Region-text alignment score on span-masked subsets | Directly measures the invariant SCIP claims to enforce |
| Baseline | Base Jeveration decoder (no fine-tune) | Isolates the contribution of the consistency penalty |
| Baseline | CLIP-score reranking over sampled candidates | Tests whether retrieval-style verification suffices |
| Baseline | MindOmni-style CoT fine-tuning | Tests whether reasoning supervision substitutes for in-pass invariant |
| Baseline | SCIP with gradient-weighted saliency masks from decoder logits (no VLM attention) | Tests whether VLM-attention region proposal is uniquely necessary |
| Baseline | SCIP with learned lightweight grounding head (jointly trained) | Tests whether a learned mask source substitutes for VLM attention |
| Ablation-of-ours | SCIP with random region masks instead of VLM attention | Tests whether span-conditioned masks carry the effect |
| Ablation-of-ours | SCIP without attention-entropy gate | Tests whether gating on verifiably span-specific masks is necessary |
| Ablation-of-ours | SCIP with pure soft decode (no straight-through) | Tests whether enforcing the invariant on committed output matters |
| Calibration study | Attention-mask IoU vs. human-annotated span regions on 512 held-out ISG-Bench examples | Pass/fail gate: mean IoU > 0.3 justifies VLM-attention masks |
| Calibration study | Correlation between $a_s$ and human span-alignment ratings on 512 examples | Pass/fail gate: Spearman $\rho > 0.4$ justifies cosine-as-correspondence |
| Calibration study | Agreement rate between argmax(soft decode) and sampled hard image | Pass/fail gate: agreement > 0.8 justifies soft-decode-as-commitment |
| Calibration study | Distractor-span control: $a_s$ against mismatched spans | Pass/fail gate: matched $a_s$ exceeds distractor $a_s$ by margin > 0.1 |

### Why this setup validates the claim

The central claim is that enforcing region-text alignment *inside* the single forward pass, via a frozen VLM verifier back-propagated through straight-through code assignments, yields images whose attended regions match prompt-referenced concepts better than either an unmodified single-pass decoder or a retrieval-style reranker. ISG-Bench supplies prompts with dense, compositional language-vision dependencies (style transfer, multi-object scenes), so span-level alignment failures are frequent and observable rather than rare edge cases. The base Jeveration decoder tests the sub-claim that the penalty adds alignment beyond what the inherited VLM prior already gives; the CLIP-reranking baseline tests the sub-claim that in-pass gradient enforcement beats post-hoc candidate selection, which is the specific distinction SCIP rests on; the CoT fine-tuning baseline tests whether the gain is merely from extra supervision rather than from the in-pass invariant. The gradient-weighted-saliency and learned-grounding-head baselines test whether the VLM-attention region proposal is uniquely necessary or whether any mask source suffices, directly addressing the load-bearing assumption. The calibration study provides pass/fail gates for each linking assumption before the main sweep: attention-mask IoU against human annotations, correlation of $a_s$ with human ratings, argmax-vs-sampled agreement, and a distractor-span control showing $a_s$ is discriminative rather than uniformly high. The primary metric — span-masked region-text alignment — is the right detector because it is computed exactly where the predicted effect should concentrate (attended regions for named spans), so a true effect should appear as a gap on span-masked subsets while leaving non-span regions at parity. The random-mask ablation isolates whether the VLM cross-attention region proposal, rather than the consistency term alone, is doing the causal work; the no-gate and pure-soft-decode ablations isolate the contributions of the entropy gate and straight-through estimator respectively.

### Expected outcome and causal chain

**vs. Base Jeveration decoder** — On a case where the prompt names a small glyph region (e.g. "a red cube beside the word CAT"), the baseline produces an image whose glyph region is blurry or miscolored because its task loss never penalizes region-level semantic mismatch. Our method instead back-propagates cosine alignment between the VLM-pooled region feature and the span text embedding through the straight-through code logits, so we expect a visible alignment gain concentrated on glyph- and attribute-heavy prompts, with parity on simple single-object prompts where the base decoder already succeeds.

**vs. CLIP-score reranking** — On a case where many sampled candidates are individually plausible but none places "red" on the cube, reranking selects the least-bad candidate because it can only choose among committed samples and cannot revise code assignments. Our method instead shifts the decoder's logits toward codes whose straight-through decode aligns with the span, so we expect SCIP to beat reranking specifically on prompts with rare attribute-object bindings while both remain comparable on easy prompts where a good candidate already exists in the pool.

**vs. MindOmni-style CoT fine-tuning** — On a case where the prompt requires spatial binding across several spans, the CoT baseline produces a coherent scene whose individual regions drift from their spans because its supervision is sequence-level, not region-level. Our method instead applies per-span alignment gradients at attended positions, so we expect SCIP to show a larger gap on multi-span compositional prompts than on single-span prompts, where CoT supervision is already sufficient.

**vs. SCIP with gradient-weighted saliency masks from decoder logits** — On a case where a span refers to a small object, the decoder-saliency mask may highlight high-logit-confidence regions that are not span-specific. If SCIP with VLM-attention masks outperforms this baseline on span-masked alignment, the VLM-attention region proposal is uniquely necessary; if not, the contribution is the straight-through gradient path rather than the mask source.

**vs. SCIP with learned lightweight grounding head** — On a case where the prompt contains multiple overlapping spans, a jointly trained grounding head may overfit to the training distribution of spans. We expect comparable performance on seen span types but a gap favoring VLM-attention masks on held-out span categories, indicating that the frozen VLM's cross-modal attention provides better generalization than a learned mask source.

**vs. SCIP with random region masks** — On a case where a span refers to a small object, the random-mask ablation applies alignment gradients to positions unrelated to the span, diluting the signal. Our method instead concentrates gradients on VLM-attended positions, so we expect the ablation to retain only a fraction of SCIP's alignment gain, confirming that span-conditioned masks carry the effect.

**vs. SCIP without attention-entropy gate** — On a case where the span is abstract (e.g., "style"), the ungated variant applies alignment gradients to diffuse, generic-saliency masks, potentially degrading non-span regions. We expect the gated SCIP to match or exceed the ungated variant on span-masked alignment while showing less degradation on non-span regions, confirming that the entropy gate successfully filters non-span-specific masks.

**vs. SCIP with pure soft decode** — On a case where the decoder's logits are near-uniform early in training, the pure-soft-decode variant enforces alignment on a blurred average image that does not match the committed output. We expect the straight-through SCIP to achieve higher alignment on the final hard-sampled images, confirming that enforcing the invariant on the committed output matters.

### What would falsify this idea

If SCIP's alignment gain is uniform across span-masked and non-span regions, or if CLIP-reranking matches SCIP on attribute-object binding prompts, then the claim that in-pass region-gradient enforcement is the causal mechanism would be wrong. Likewise, if the random-mask ablation recovers SCIP's full gain, or if the calibration gates fail (attention-mask IoU ≤ 0.3, $a_s$-human correlation ≤ 0.4, argmax-vs-sampled agreement ≤ 0.8, or distractor-span margin ≤ 0.1), then the VLM cross-attention region proposal and cosine alignment are not doing the work we attribute to them.

## References

1. Jeveration: System One Models Now Generate
