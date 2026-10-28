# Relational Rate-Distortion Quantization: Enforcing Spatial Layout in Factorized Visual Codebooks via Index-Only Relational Decoding

## Motivation

Factorized parallel codebooks (e.g., the Factorized Vector Quantization of 'Can Discrete Tokens Serve as Semantically Lossless Representations of Visual Signals?') optimize only per-view appearance reconstruction, so the discrete token stream that feeds a decoder-only next-token-prediction transformer (as in BAGEL, ILLUME, MetaMorph) carries relational/layout structure only as an emergent byproduct of bit-budget scaling. Across these paper-trees, no quantization objective ever penalizes loss of pairwise spatial relations, so downstream generation inherits whatever survives and has no invariant enforcing layout fidelity — the meta-gap is that relational fidelity is never a constrained property of the codebook assignment itself. The root cause is structural: the rate-distortion Lagrangian is defined purely over per-view appearance error, and the relational information is only recoverable from the continuous features that quantization is allowed to discard.

## Key Insight

Pairwise spatial relations between feature patches are a sufficient statistic for layout, and by adding an index-only relational decoder's distortion as a first-class term in the quantization Lagrangian, the codebook assignment becomes constrained to preserve that statistic — making layout an invariant of the discrete tokens rather than a byproduct of bit budget.

## Method

**What it is.** R-RDQ (Relational Rate-Distortion Quantization) augments factorized vector quantization with a relational distortion term. Input: visual feature map split into V complementary fixed-dimensional views. Output: V parallel codebook index streams whose assignment minimizes a joint Lagrangian combining per-view appearance distortion, a relational distortion computed by a lightweight decoder that sees only the discrete indices, and the rate. The relational decoder is trained jointly but discarded at inference; only the quantizer changes.

**Load-bearing assumption (stated explicitly).** The method assumes that a decoder restricted to discrete codebook indices can recover pairwise relative offsets and adjacency between patches, so that minimizing this index-only relational distortion forces the codebook assignment itself to encode spatial layout rather than the decoder exploiting positional priors or index-identity memorization. This assumption is load-bearing: if false, L_rel is non-identifying for layout content. To make it testable, the relational decoder g_phi is made permutation- and position-invariant to the index-tensor geometry: it receives only the pair of index embeddings (I_p, I_q) with no positional encoding, and is trained on patch pairs sampled across a randomized, position-shuffled grid so that grid coordinates carry no information. Additionally, L_rel is evaluated on held-out codebook assignments where the same index pair appears at multiple distinct offsets; only then does minimizing L_rel force the codebook to disambiguate layout rather than the decoder to memorize coordinates.

**How it works.**

```pseudocode
Input: feature map F in R^{HxWxD}, V parallel codebooks C_1..C_V (K=1024 entries each, dim 8),
       relational decoder g_phi: 2-layer MLP, hidden=256, GeLU, input = concat(emb(I_p), emb(I_q)),
       emb = learned embedding table of size K x 16, NO positional encoding
1. Split F into V views F_1..F_V (fixed-dim factorization, as in FVQ)
2. For each view v, patch p: z_v,p = quantize(F_v,p; C_v) -> index i_v,p
   (nearest-neighbor assignment, straight-through gradient)
3. Appearance distortion: L_app = sum_v sum_p ||F_v,p - C_v[i_v,p]||^2
4. Build index tensor I in {1..K}^{HxWxV} from all i_v,p
5. Relational targets: for each patch pair (p,q) in a local window W (radius r=2),
   t_pq = (delta_x, delta_y, adjacency in {0,1}) from ground-truth patch grid
6. Relational prediction: r_hat_pq = g_phi(I_p, I_q)  # sees ONLY discrete indices, no position
7. Relational distortion: L_rel = mean_{p,q in W} ||r_hat_pq - t_pq||^2
   - Patch pairs are sampled from a randomized, position-shuffled grid during g_phi training
     so that grid coordinates carry no information.
   - L_rel is evaluated on held-out codebook assignments where the same index pair appears
     at multiple distinct offsets.
8. Total: L = L_app + lambda_rel * L_rel + beta * L_commit + gamma * L_rate
   with lambda_rel warm-started at 0 and ramped to 0.5 over 10k steps;
   L_rate = -log p(i) under an empirical codebook histogram (entropy proxy)
9. Update C_v by straight-through gradient of L; update g_phi by gradient of L_rel
10. At inference: drop g_phi; emit index streams I for downstream NTP transformer
```

**Why this design.** We chose an *index-only* relational decoder over a decoder that also sees continuous features because only the index-only path makes the relational distortion a function of the codebook assignment itself, accepting the cost that g_phi must be trained jointly and can be noisy early. We chose *pairwise relative offsets and adjacency within a local window* over global layout coordinates because local relations are translation-invariant and thus a well-posed target for a permutation-sensitive index stream, accepting that long-range layout is only indirectly constrained. We warm-start lambda_rel at 0 and ramp it, rather than fixing it high from step 0, because a high relational weight before the codebooks have learned appearance would collapse assignments toward degenerate layout-preserving codes, accepting a longer training schedule. We use an empirical-histogram entropy proxy for the rate term rather than a learned prior because it is differentiable through the straight-through estimator without an extra network, accepting a coarser rate estimate. A domain expert would not describe this as a variant of FVQ because FVQ's objective has no term that depends on the *joint* index configuration across patches; here the relational decoder's input is precisely that joint configuration, so the assignment is optimized against a statistic that FVQ never computes.

**Why it measures what we claim.** The relational distortion `L_rel` measures preservation of pairwise spatial structure because the named assumption is that relative offset and adjacency between patches are a sufficient statistic for layout, making reconstruction of these quantities from indices equivalent to layout preservation; this assumption fails when the feature map has repeated textures with ambiguous patch correspondence, in which case `L_rel` reflects texture-periodicity consistency rather than true object layout. The index-only input to `g_phi` measures that the *discrete tokens* carry relational information because the assumption is that if a decoder restricted to indices can recover offsets, that information is present in the index stream and not smuggled through continuous residuals; this fails when `g_phi` memorizes the training grid geometry (since offsets are partly determined by patch position), in which case `L_rel` reflects positional-prior leakage rather than learned relational content — mitigated by holding out patch-pair positions during g_phi training. The warm-start ramp of `lambda_rel` measures that relational constraint is applied after appearance codes are meaningful because the assumption is that early appearance learning is a prerequisite for non-degenerate relational codes; this fails when the ramp is too fast, in which case `L_rel` reflects collapse toward layout-only codes.

## Contribution

(1) A relational rate-distortion quantization objective that adds an index-only relational decoder's distortion as a first-class Lagrangian term, changing the codebook assignment problem so layout is a constrained invariant rather than an emergent byproduct. (2) A design principle — local pairwise relative-offset/adjacency targets within a translation-invariant window — that makes relational distortion well-posed for a permutation-sensitive discrete index stream, with a warm-start schedule that avoids layout-appearance collapse. (3) An analysis protocol that separates genuine relational content from positional-prior leakage by holding out patch-pair positions during relational-decoder training.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet-1k reconstruction + downstream NTP understanding | Tests layout-sensitive token quality at scale. |
| Primary metric | Downstream multimodal understanding accuracy at matched bit budget | Isolates token informativeness, not raw reconstruction. |
| Baseline | Factorized VQ (FVQ) | Same factorization, no joint-index relational term. |
| Baseline | VQ-VAE | Standard appearance-only quantizer, no relational structure. |
| Baseline | Continuous features (no quantization) | Upper bound on information, no discrete-token constraint. |
| Ablation-of-ours | R-RDQ with lambda_rel fixed at 0 | Removes relational term, keeps all else identical. |
| Ablation-of-ours | R-RDQ with position-shuffled g_phi training | Tests whether positional leakage drives L_rel. |
| Ablation-of-ours | R-RDQ with inference-time relational-signal ablation | Tests whether downstream NTP consumes relational signal. |

### Why this setup validates the claim

The central claim is that optimizing the *joint* index configuration against an index-only relational distortion yields discrete tokens that preserve layout better than appearance-only quantization, and that this advantage shows up as downstream understanding gains at matched bit budget. FVQ is the decisive baseline because it shares our factorization and rate term but lacks any objective depending on joint index configuration — so any gap between R-RDQ and FVQ is attributable to the relational term, not to the factorized architecture. VQ-VAE tests whether the gain is merely a factorization effect. Continuous features bound how much of the remaining gap is quantization loss versus representation loss. The lambda_rel=0 ablation is the sharpest control: identical code, identical schedule, relational term disabled, so it isolates the causal contribution of L_rel itself. The position-shuffled g_phi ablation directly tests the load-bearing assumption: if L_rel is driven by positional leakage, then shuffling patch-pair positions during g_phi training should degrade downstream layout accuracy, whereas if L_rel captures genuine relational content, performance should be preserved. The inference-time relational-signal ablation tests whether the downstream NTP transformer actually consumes the relational signal rather than re-deriving layout from token order: we ablate the relational signal by replacing the index stream with a version where relational information is scrambled (e.g., randomly permuting indices within each local window) and measure the drop in layout-sensitive accuracy. Downstream accuracy at matched bit budget is the right metric because it detects whether the relational information is actually usable by a downstream transformer, rather than merely reconstructible by a decoder that we discard at inference.

### Expected outcome and causal chain

**vs. Factorized VQ (FVQ)** — On images containing repeated textures with locally ambiguous patch correspondence (e.g. tiled facades, fabric patterns), FVQ assigns each view's patch independently by nearest-neighbor appearance, so patches with near-identical appearance but different grid position can receive interchangeable indices; its objective never sees the joint configuration, so layout is unconstrained. R-RDQ's relational decoder g_phi is trained to recover relative offsets and adjacency from the index tensor, forcing the assignment to disambiguate positionally distinct patches that look alike. We expect a noticeable downstream accuracy gap concentrated on texture-heavy and structured-layout subsets, with near-parity on homogeneous single-object subsets where FVQ's appearance codes already suffice. The position-shuffled g_phi ablation should show a significant drop in this gap if positional leakage is the driver, confirming the load-bearing assumption.

**vs. VQ-VAE** — On scenes with multiple small objects at distinct spatial relations (e.g. cluttered tabletop images), VQ-VAE's single non-factorized codebook and appearance-only loss let it spend codebook capacity on low-level texture, producing tokens that reconstruct well but lose relational layout; a downstream NTP model then cannot recover object arrangement from the token stream. R-RDQ's factorized views plus relational distortion allocate index structure to pairwise offsets within the local window, so the token stream carries layout explicitly. We expect a clear gap on relational-reasoning and counting-style understanding queries, smaller on single-object classification, and a reconstruction-quality gap that is modest or even slightly negative for R-RDQ.

**vs. Continuous features (no quantization)** — On fine-grained spatial reasoning, continuous features retain full positional information, so this baseline should be the strongest; the question is how much of that advantage R-RDQ closes relative to FVQ. Because our relational term is index-only, it cannot smuggle continuous residuals, so we expect R-RDQ to remain below continuous features but to close a substantial fraction of the FVQ-to-continuous gap specifically on layout-sensitive subsets, while FVQ stays near the appearance-only floor there. A flat gap across all subsets would indicate the relational term is not doing layout-specific work. The inference-time relational-signal ablation should show a larger drop for R-RDQ than for FVQ on layout-sensitive subsets, confirming that the downstream NTP transformer consumes the relational signal.

### What would falsify this idea

If R-RDQ's gain over FVQ is uniform across all subsets rather than concentrated where repeated-texture and multi-object layout ambiguity is predicted to hurt FVQ, or if the lambda_rel=0 ablation matches R-RDQ on the layout-sensitive subsets, then L_rel is not preserving the relational structure we claim and the central mechanism is wrong. A second falsifier: if g_phi's relational distortion is low on held-out patch-pair positions yet downstream layout accuracy does not improve, the index-only relational signal is not the causal carrier we assert. A third falsifier: if the position-shuffled g_phi ablation shows no degradation in downstream layout accuracy, then the load-bearing assumption that g_phi recovers relational content without positional leakage is false.

## References

1. Can Discrete Tokens Serve as Semantically Lossless Representations of Visual Signals?
2. Emerging Properties in Unified Multimodal Pretraining
3. ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance
4. MetaMorph: Multimodal Understanding and Generation via Instruction Tuning
5. LMFusion: Adapting Pretrained Language Models for Multimodal Generation
