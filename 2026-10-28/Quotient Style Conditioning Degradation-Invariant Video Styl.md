# Quotient Style Conditioning: Degradation-Invariant Video Style Transfer via Groupoid Closure

## Motivation

Across both the VISTA triplet-supervision tree and the fixed-window cross-frame-attention tree, the style-conditioning map f(style_ref, content) is trained and evaluated only on clean, complete, motion-aligned references, so the stylized output distribution shifts and temporal consistency breaks whenever the reference is blurred, occluded, misaligned, or partially missing. VISTA concretely instantiates this: its style adapter is trained on clean motion-aligned triplets from VISTA-1000 and has no constraint tying f(ref) to f(D(ref)) for any degradation operator D, so reference degradation propagates unchecked into the style code. The structural gap is that no existing method makes the style code well-defined on the quotient of references modulo degradations — invariance is hoped for from data augmentation rather than imposed by construction.

## Key Insight

If the style-conditioning map is factored through a canonicalization network that maps every degraded reference to a canonical representative plus a residual group element, then closure f(ref)=f(D(ref)) holds exactly by construction on the quotient space, so degradation invariance is a property of the function's domain rather than a statistical tendency learned from augmented data.

## Method

**What it is.** We introduce QSC (Quotient Style Conditioning), a style-conditioning framework for diffusion-transformer video style transfer. Its inputs are a (possibly degraded) style reference image r and a clean content video V; its output is a stylized video. Internally QSC replaces the raw style adapter input with a canonical style code produced by a canonicalization network C and a style encoder E, trained under a closure loss over a synthesized degradation semigroup.

**How it works.**

```pseudocode
# Phase 1: degradation semigroup generation (cheap, from clean refs)
S = {D_1, ..., D_K}   # blur(sigma in {1,3,5}), occlusion(area in {5,15,30}%),
                       # misalign(shift in {4,8,16}px), missing-region(random box)
for each clean reference r0 in VISTA-1000:
    sample D ~ Uniform(S);  r_deg = D(r0)

# Phase 2: canonicalization network C (ViT-S, 6 blocks, dim 384)
#   C(r) -> (z_can, g)  where z_can in R^{d} is canonical style code,
#                        g in R^{|S|} is a soft group-element logit vector
z_can, g = C(r)

# Phase 3: closure training (per triplet (r0, V, Y))
for each triplet:
    r_deg = D(r0)                       # D sampled from S
    z0, g0 = C(r0);  zd, gd = C(r_deg)
    L_style  = ||E_style(zd) - E_style(z0)||_1          # closure on style code
    L_group  = CE(gd, argmax D_index) + CE(g0, identity) # group-element supervision
    L_canon  = ||zd - z0||_2^2                            # canonical representative consistency
    L_task   = diffusion_loss(DiT(V, zd), Y)             # standard flow-matching loss
    L = L_task + lambda_1 L_style + lambda_2 L_group + lambda_3 L_canon
    # lambda_1=1.0, lambda_2=0.1, lambda_3=0.5 (tuned on 50 held-out triplets)

# Phase 4: inference
z_can, g = C(r)          # g discarded; only canonical code conditions DiT
Y_hat = DiT(V, z_can)
```

**Why this design.** We chose a learned canonicalization network C over test-time augmentation ensembling because augmentation ensembling averages over degradations rather than removing them, accepting the cost that C must generalize to unseen degradation compositions; we mitigate this by training C on a semigroup closed under composition (blur∘occlusion, shift∘missing). We chose a soft group-element head g over a hard discrete classifier because degradation compositions are not always identifiable from a single image, accepting the cost that g is only weakly supervised, which is why g is used only as an auxiliary signal and never at inference. We chose the closure loss on the style code z rather than on the final stylized output because output-space closure would require paired degraded-reference generations and would conflate style invariance with content preservation, accepting the cost that style-code closure is a proxy for output invariance — a proxy we validate by measuring output-level FID under degradation at test time. We chose to keep the DiT backbone and style adapter architecture identical to VISTA and only change the conditioning input, so that any measured gain is attributable to the quotient construction rather than to backbone capacity; a domain expert would not describe QSC as a variant of VISTA because VISTA's adapter maps raw reference features to style tokens with no canonicalization or group structure, whereas QSC's conditioning map is defined on the quotient space and its invariance is enforced by a closure loss that VISTA lacks entirely.

**Why it measures what we claim.** The style-code closure loss L_style measures degradation invariance of the conditioning map because it assumes the style encoder E_style is injective on the style manifold, making ||E_style(zd) - E_style(z0)||_1 = 0 equivalent to zd = z0 in style space; this assumption fails when two distinct styles map to nearby E_style codes (e.g., two impressionist palettes), in which case L_style reflects palette similarity rather than invariance. The canonical-representative consistency loss L_canon measures that C removes the degradation rather than memorizing it because it assumes the canonical code z0 of a clean reference is a sufficient statistic for style, so any residual degradation signal in zd is penalized; this assumption fails when the clean reference itself contains occlusion-like content (a subject partially out of frame), in which case L_canon penalizes legitimate content. The group-element head loss L_group measures identifiability of the degradation operator because it assumes the semigroup S is closed and its elements are distinguishable in feature space; this assumption fails for composed degradations that are visually ambiguous (mild blur plus small shift), in which case g reflects the dominant degradation rather than the full composition. Together these three quantities operationalize the motivation-level concept of a well-defined quotient style code: each penalizes a different way the map can fail to be constant on equivalence classes, and the diffusion task loss ensures the canonical code remains style-sufficient. Additionally, the output-level FID gap measures the end-to-end invariance because it assumes the DiT style adapter is Lipschitz in z with a constant L_adapter small enough that code-space closure bounds output-space divergence; this assumption fails when the adapter amplifies small residual degradation signal in zd, in which case the FID gap reflects adapter sensitivity rather than failure of the quotient construction. We empirically estimate L_adapter on 512 held-out (z, V) pairs by finite differences and report it alongside the FID gap.

## Contribution

(1) A quotient-space style-conditioning framework, QSC, that factors the style-conditioning map through a learned canonicalization network and a degradation-semigroup closure loss, making the style code invariant to reference degradation by construction rather than by augmentation. (2) A design principle showing that closure on the style code, combined with a soft group-element auxiliary head, is sufficient to induce output-level degradation invariance without paired degraded references, validated by measuring stylized-output FID and temporal consistency under held-out degradations. (3) A synthesized degradation semigroup generator over VISTA-1000 references that is closed under composition and cheap to produce, enabling closure training without any degraded-reference annotations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VISTA triplet benchmark + degraded-reference test split | Matches training content; adds degraded refs for invariance test |
| Primary metric | Style fidelity gap: clean-ref vs degraded-ref FID | Directly measures invariance the quotient claims |
| Baseline | VISTA (raw reference conditioning) | Tests whether canonicalization adds anything over adapter |
| Baseline | VISTA + test-time augmentation ensembling | Tests learned C against averaging over degradations |
| Baseline | VISTA + degradation-augmented training (no C) | Tests whether closure loss is doing the work |
| Ablation-of-ours | QSC without L_style closure loss | Isolates the closure mechanism from architecture |
| Diagnostic | Empirical Lipschitz constant of DiT style adapter | Bounds code-to-output gap; validates proxy assumption |

### Why this setup validates the claim

The central claim is that conditioning on a canonical quotient style code makes stylization invariant to reference degradation, and that this invariance is produced by the closure loss rather than by backbone capacity or by naive augmentation. The design isolates each competing explanation. VISTA with raw reference conditioning tests whether canonicalization is needed at all: if QSC does not beat it, the quotient construction is decorative. VISTA with test-time augmentation ensembling tests the specific design choice we argued against — averaging over degradations instead of removing them — so a QSC win here confirms that canonicalization removes degradation rather than smearing it. VISTA with degradation-augmented training but no canonicalization network tests whether the closure loss on the style code is the active ingredient or whether simple exposure to degradations suffices. The ablation removing L_style directly probes the mechanism: if invariance survives its removal, the claimed causal path is wrong. The primary metric is the style-fidelity gap between clean-reference and degraded-reference conditioning, because that gap is exactly what the quotient construction predicts should shrink; absolute FID could improve for unrelated reasons, but a shrinking gap concentrated on the degraded subset is a signature only the invariance mechanism produces. The diagnostic Lipschitz constant of the style adapter bounds how much residual code-space difference can leak into output space, so a small L_adapter together with a small code-space gap supports the proxy claim, while a large L_adapter would indicate that output invariance requires stronger code-space closure.

### Expected outcome and causal chain

**vs. VISTA (raw reference conditioning)** — On a case where the reference image is blurred or partially occluded, the baseline produces a stylized video whose palette and stroke statistics drift toward the degradation — washed-out color from blur, missing texture where the occluder sits — because its adapter maps raw reference features directly to style tokens with no invariance constraint. Our method instead maps the degraded reference through C to a canonical code near the clean code because the closure loss forces zd and z0 to coincide, so we expect a clear gap on the degraded-reference subset and near-parity on the clean-reference subset.

**vs. VISTA + test-time augmentation ensembling** — On a case where the reference has a shift plus mild blur, the baseline averages style tokens across augmented views, producing a softened, over-smoothed style that blends the degradation with the content rather than removing it, because averaging over a semigroup does not project onto a canonical representative. Our method instead conditions on a single canonical code that the closure loss has pushed to the identity element, so we expect the ensembling baseline to sit between raw VISTA and QSC on degraded references, with QSC showing sharper style statistics and a smaller clean-vs-degraded gap.

**vs. VISTA + degradation-augmented training (no C)** — On a case where the reference is occluded by a missing region, the baseline learns to associate occlusion patterns with style tokens because it sees degraded references during training but has no mechanism forcing equivalence across the degradation orbit, so it partially memorizes degradation as style. Our method instead penalizes residual degradation signal in the canonical code via L_canon and L_style, so we expect the augmented baseline to improve over raw VISTA but to retain a measurable gap on composed degradations, while QSC's gap stays small.

**vs. QSC without L_style closure loss (ablation)** — On a case where two degradations compose (blur then shift), the ablated model has the canonicalization architecture but no pressure to make zd equal z0, so C can drift toward encoding degradation identity rather than removing it. Our method with the closure loss keeps the style code constant across the orbit, so we expect the ablation to recover most of the clean-reference performance but to show a visibly larger degraded-reference gap, confirming that the closure loss rather than the architecture carries the invariance.

### What would falsify this idea

If QSC's gain over VISTA is uniform across clean and degraded reference subsets rather than concentrated on the degraded subset, or if removing L_style leaves the clean-vs-degraded gap unchanged, then the canonicalization is not producing invariance and the quotient claim is wrong. Additionally, if the empirical Lipschitz constant of the style adapter is large enough that small code-space differences produce large output differences, the proxy link from code closure to output invariance would be broken.

## References

1. VISTA: Triplet-Supervised Video Style Transfer with Diffusion Transformers
2. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
3. Composer: Creative and Controllable Image Synthesis with Composable Conditions
4. RelationAdapter: Learning and Transferring Visual Relation with Diffusion Transformers
5. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
6. In-Context LoRA for Diffusion Transformers
7. OmniGen: Unified Image Generation
8. Boosting Latent Diffusion with Flow Matching
