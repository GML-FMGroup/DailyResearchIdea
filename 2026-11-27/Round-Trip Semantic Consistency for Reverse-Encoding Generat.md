# Round-Trip Semantic Consistency for Reverse-Encoding Generative Models

## Motivation

Across visual representation learning, methods that learn a representation-to-pixel or representation-to-view map (RevFlow's reverse encoding, dense contrastive learning in CR2PQ, occlusion-based OCL) validate only the forward or the reverse direction and never close the loop by re-encoding their own outputs. The structural gap is the absence of a round-trip invariant: RevFlow (E6) generates pixels from frozen-encoder targets but never checks that those pixels re-encode to the same target under the frozen encoder, so any drift in the generated distribution is invisible to the training objective. This gap recurs because the frozen-encoder assumption plus single-domain ImageNet validation never tests closure under re-encoding.

## Key Insight

Binding the generative reverse map to the forward semantic space through a frozen-encoder re-encoding loss creates a round-trip invariant that makes the reverse-encoding path a contraction toward the encoder's own semantic equivalence classes rather than an unconstrained pixel decoder.

## Method

**What it is.** Round-Trip Semantic Consistency (RTSC) augments RevFlow's stage-wise reverse-encoding training with a cyclic loss: generated pixels are re-encoded by the same frozen encoder, and the re-encoded deep semantic representation is pulled toward the original deep semantic target that seeded generation. Inputs are the frozen encoder E, the HyC-DiT generator G, an image x, and its encoder targets {z_k}; outputs are the trained generator with a closed encoder→generation→re-encode loop.

**How it works.**

```pseudocode
# Frozen encoder E with intermediate layers L_1..L_K (shallow -> deep)
# HyC-DiT generator G shared across stages, condition injection per stage
# lambda_rt: round-trip loss weight (default 0.5)
# lambda_stage: per-stage native-matching weight (default 1.0)
# tau: InfoNCE temperature (default 0.07)
# B: minibatch size (default 256)

for each minibatch of images x (size B):
    targets = E.forward_intermediates(x)          # {z_k}, deep z_K is semantic target
    # --- reverse-encoding generation (RevFlow) ---
    zhat_K = sample_semantic(z_K)                 # native matching in semantic space
    for k = K-1 down to 1:
        zhat_k = G(zhat_{k+1}, stage=k)           # shared DiT, stage-specific condition
    xhat = G(zhat_1, stage=0)                     # decode to pixels
    # --- round-trip re-encoding ---
    z_re = E.forward_intermediates(xhat)          # frozen encoder, no grad to E
    # contrastive round-trip loss (InfoNCE) on deep semantic layer:
    #   positives: (z_re.K[i], z_K[i]) for each i in batch
    #   negatives: all other z_K[j], j != i, within the same minibatch
    loss_rt = InfoNCE(z_re.K, z_K, tau=0.07)      # symmetric over batch
    loss_stage = sum_k || zhat_k - z_k ||^2       # RevFlow native matching
    loss = loss_stage + lambda_rt * loss_rt
    update G only; E stays frozen
```

A stop-gradient is applied to z_K inside loss_rt so the target is a fixed anchor; the gradient flows only through xhat into G. The round-trip term is computed every step (not intermittently) and back-propagated through the pixel decoder and every stage of G, so the shared backbone receives a signal that ties all stage transitions to a single closed-loop objective. The InfoNCE form penalizes collapse to the encoder's invariant subspace: it requires z_re.K to be close to its own z_K and far from other batch targets, so generating any image in the encoder's preimage of z_K is not sufficient to reduce the loss.

**Why this design.** We chose a frozen-encoder re-encoding loss over training a separate inverse model because the frozen encoder is the exact semantic space RevFlow already commits to (E6), so any inverse model would introduce a second, unvalidated semantic map and re-open the question of which space is canonical; the cost is that RTSC inherits the frozen encoder's blind spots and cannot correct errors the encoder itself makes. We chose a contrastive InfoNCE round-trip loss over a pure deep-semantic MSE because the deep target z_K is the quantity RevFlow actually generates first, making the invariant directly checkable against the generative target, while the contrastive form prevents the loss from being trivially minimized by collapsing to the encoder's invariant subspace (which a pure MSE would reward); the cost is that shallow appearance-level consistency is only indirectly constrained through the stage-wise native-matching term. We chose to back-propagate through the full stage hierarchy rather than only through the final decoder because the round-trip error must be attributed to the stage transitions that produced the drift, not just to the last pixel layer; the cost is higher memory and a longer backward pass. We chose a fixed weight lambda_rt=0.5 rather than a learned schedule because the round-trip term is an invariant, not a regularizer whose strength should adapt to data, accepting that a single weight may be suboptimal across encoder choices. A domain expert would not describe RTSC as 'RevFlow plus a reconstruction loss': reconstruction losses in autoencoders operate in pixel or feature space without a frozen semantic anchor, whereas RTSC's target is the frozen encoder's own deep representation, making the loss a closure condition on the generative map rather than a fidelity term.

**Why it measures what we claim.** The contrastive InfoNCE round-trip loss measures round-trip semantic closure because the assumption is that the frozen encoder's deep layer is a sufficient statistic for the semantic content RevFlow is asked to generate, so equality of z_re.K and z_K (relative to other batch targets) implies the generated pixels lie in the same semantic equivalence class as the original; this assumption fails when the encoder is not injective on the generation distribution (two visually different images map to the same z_K), in which case the metric reflects semantic-class membership rather than pixel-level fidelity. The stop-gradient on z_K measures a fixed anchor because the assumption is that the encoder target is not co-adapted during generator training, so the loss cannot be trivially reduced by drifting the target; this fails if the encoder is later fine-tuned, in which case the invariant becomes relative to a moving reference. The full-hierarchy back-propagation measures stage-attributed drift because the assumption is that each stage's output zhat_k is a faithful intermediate of the reverse map, so gradient through stage k attributes round-trip error to that transition; this fails when stage errors cancel across the hierarchy, in which case the loss under-attributes drift to any single stage.

**Assumption A (explicit).** The load-bearing assumption is that the frozen encoder's deep feature z_K is a sufficient statistic whose equality (z_re.K ≈ z_K) implies the generated pixels lie in the same semantic equivalence class as the source, so minimizing the round-trip loss is a valid closure condition on the generative map rather than a degenerate feature-matching objective. This assumption fails when the encoder is not injective on the generation distribution (two visually distinct images collapse to the same z_K), making the loss reward class membership not fidelity. The contrastive InfoNCE form is chosen to penalize such collapse by requiring separation from other batch targets, but the assumption remains load-bearing and is tested empirically via the collision-rate diagnostic described in the experiment.

## Contribution

['A round-trip semantic consistency loss (RTSC) that closes the encoder→generation→re-encode loop for reverse-encoding generative models by re-encoding generated pixels with the frozen encoder and matching the deep semantic target, with stop-gradient anchoring and full-hierarchy back-propagation.', "A design principle: the reverse-encoding map should be trained against a closure invariant in the encoder's own semantic space rather than against pixel or perceptual fidelity alone, making the frozen encoder the arbiter of generation correctness.", 'An evaluation protocol that reports round-trip semantic error alongside FID to expose distribution-shift drift that FID alone cannot reveal.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ImageNet 256x256 | RevFlow's native benchmark; enables direct comparison. |
| Primary metric | FID at matched epochs | Measures generation fidelity where closure should help most. |
| Secondary metric | Round-trip collision rate: fraction of distinct images in a held-out set of 10,000 that map to the same z_K bucket (quantized to 1e-3 per dimension) | Directly measures the injectivity assumption A on the generation distribution. |
| Secondary metric | Per-stage round-trip error decomposition: ||z_re.k − z_k||^2 / ||z_k||^2 for each k | Tests the stage-attribution claim. |
| Baseline | RevFlow (no round-trip loss) | Isolates the contribution of the cyclic term. |
| Baseline | RevFlow + larger decoder (2x width) | Tests whether closure substitutes for decoder capacity. |
| Baseline | RevFlow + pixel-space reconstruction loss (MSE on xhat vs x, weight 0.5) | Tests whether the semantic anchor matters. |
| Ablation-of-ours | RTSC with stop-gradient removed | Tests whether the fixed-anchor design is load-bearing. |
| Ablation-of-ours | RTSC with pure deep-semantic MSE instead of InfoNCE | Tests whether the contrastive form is necessary to avoid collapse. |

### Why this setup validates the claim

The central claim is that closing the encoder→generation→re-encode loop with a frozen deep-semantic anchor improves the fidelity and convergence of RevFlow's reverse-encoding path. ImageNet 256x256 is the right testbed because RevFlow already reports FID there at 5 epochs, giving us a tightly matched comparison rather than a cross-paper one. FID is the right primary metric because the predicted effect is a reduction in semantic drift between generated and source content, which manifests as better distributional fidelity at fixed training budget; a metric like pixel MSE would conflate fidelity with appearance matching and would not isolate the closure effect. The baselines form a nested test: plain RevFlow asks whether the round-trip term adds anything at all; RevFlow with a larger decoder asks whether the gain is just capacity (if closure only matches a bigger decoder, the mechanism claim is weakened); and RevFlow with pixel-space reconstruction asks whether the semantic anchor specifically is what matters, or whether any auxiliary reconstruction signal helps. The stop-gradient ablation is the sharpest test of the design argument: if removing the fixed anchor still yields the same gain, then the loss is behaving as a generic regularizer rather than a closure condition, and the stated mechanism is wrong. The collision-rate diagnostic directly tests assumption A: if the encoder is highly non-injective on the generation distribution, the round-trip loss may reward class membership rather than fidelity, and the contrastive form is designed to mitigate this; the per-stage error decomposition tests whether the full-hierarchy back-propagation attributes drift to the correct stages.

### Expected outcome and causal chain

**vs. RevFlow (no round-trip loss)** — On a case where the generator produces a plausible-looking image whose deep semantic content drifts from the source (e.g., a texture-consistent but semantically shifted object), plain RevFlow has no signal tying the generated pixels back to the original semantic target, because its native matching only constrains each stage's output toward the encoder target during the forward pass and never checks the composed result. Our method instead re-encodes the generated pixels and pulls the re-encoded deep representation toward the fixed anchor while pushing it away from other batch targets, so the composed generation is penalized exactly when it drifts. We expect a visible FID gap concentrated on semantically diverse images, with near-parity on visually simple ones, and the collision-rate diagnostic to show that the contrastive form reduces the fraction of generated images that collapse to the same z_K as other batch items.

**vs. RevFlow + larger decoder** — On a case where the decoder has enough capacity to fit appearance but the stage transitions still lose semantic content, the larger-decoder baseline produces sharper pixels without recovering the semantic target, because extra capacity addresses the pixel-fidelity bottleneck rather than the representation-to-pixel drift. Our method instead attributes the round-trip error through every stage transition, so the shared backbone is corrected where the drift originates. We expect RTSC at the base model size to approach or match the larger-decoder baseline's FID while using fewer parameters, with the gap widening at shorter training budgets, and the per-stage error decomposition to show that the round-trip loss reduces error at intermediate stages rather than only at the final decoder.

**vs. RevFlow + pixel-space reconstruction loss** — On a case where two images share a semantic class but differ in fine appearance, a pixel-space reconstruction term pushes toward exact appearance matching and can fight the semantic objective, because its target is the raw image rather than the encoder's deep representation. Our method instead uses the frozen encoder's deep layer as the anchor, so the loss is invariant to appearance differences within the same semantic class. We expect the pixel-reconstruction baseline to show a smaller or absent gain on FID and to degrade more on semantically diverse subsets, while RTSC shows a consistent gain concentrated where semantic drift is the dominant error.

**vs. RTSC with stop-gradient removed** — If the stop-gradient is removed, the encoder target z_K can co-adapt with the generator, so the round-trip loss can be trivially reduced by drifting the target rather than by improving generation. We expect the no-stop-gradient variant to show a smaller or absent FID gain and higher collision rate, confirming that the fixed anchor is load-bearing.

**vs. RTSC with pure deep-semantic MSE instead of InfoNCE** — A pure MSE round-trip loss can be driven to near-zero by generating any image in the encoder's preimage of z_K, including blurry or mode-collapsed outputs, because it does not penalize collapse to the encoder's invariant subspace. We expect the MSE variant to show worse FID and higher collision rate than the InfoNCE variant, especially on semantically diverse subsets, confirming that the contrastive form is necessary to avoid the non-injectivity degeneracy.

### What would falsify this idea

If the round-trip gain is uniform across all image subsets rather than concentrated on semantically diverse or drift-prone cases, and if removing the stop-gradient produces the same improvement, then the loss is acting as a generic regularizer rather than a semantic closure condition, and the central claim is wrong. A second falsifier: if RTSC only matches, but never exceeds, the larger-decoder baseline at equal parameter count, then closure is not substituting for capacity and the mechanism argument fails. A third falsifier: if the collision-rate diagnostic shows that the frozen encoder is highly non-injective on the generation distribution and the InfoNCE variant does not reduce collision rate relative to the MSE variant, then assumption A is broken and the round-trip loss is measuring class membership rather than fidelity.

## References

1. Generation as Reverse Encoding
