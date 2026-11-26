# Compositional Token Factorization for Faithful Multimodal Image Generation

## Motivation

Unified multimodal models such as Integral (E6) build a shared visual token space via Abstraction Fusion over a frozen vision encoder, but they provide no structural rule ensuring that a compositional prompt decomposes into shared tokens whose recombination reconstructs each sub-part faithfully. This gap is meta-level: the same frozen-backbone assumption recurs across text-to-image synthesis with LLM–DiT fusion (E1), pretrained-LLM adaptation (E7), and discrete-token multimodal pretraining (E9), where spatial tokens lack recursive structure. Because compositionality is left to empirical hope rather than enforced as a representational invariant, all these branches inherit a shared residual failure on multi-object, multi-attribute, and spatial-relation prompts.

## Key Insight

Compositional faithfulness becomes a representational invariant when the shared token space is factorized by prompt sub-parts and constrained by a round-trip reconstruction cycle through the frozen encoder, so that recombination of sub-token groups is faithful by construction rather than by training luck.

## Method

**What it is — CTF (Compositional Token Factorization).** CTF takes a compositional prompt `p` and a frozen vision encoder `E`, and produces (i) a factorized shared token set `{T_k}` where each `T_k` is aligned to a prompt sub-part (object, attribute, or spatial relation) and (ii) a generative stream trained so that any recombination of sub-token groups re-encodes to a faithful image under `E`. It plugs into the Integral-style Abstraction Fusion + Block-wise MoT architecture without changing the frozen encoder.

**Load-bearing assumption (stated explicitly).** CTF assumes that the frozen encoder `E` (and the Abstraction Fusion projector `A`) is injective and compositional on the image manifold relevant to prompt sub-parts: that is, for any two visually distinct sub-parts, their `E`-codes are separated by a margin, so that minimizing `||E(D(T_k)) - A(E(p_k))||^2` cannot be achieved by encoder-invariance alone. This assumption is not taken on faith; it is calibrated and, where it fails, replaced by a contrastive target (see step 7 and the collision-rate diagnostic below).

**How it works.**

```pseudocode
# Inputs: prompt p, frozen encoder E, prompt parser P (spaCy dependency parser, frozen),
#         shared-token projector A (Abstraction Fusion), generative stream G (Block-wise MoT),
#         reconstruction decoder D, sub-part discriminator M (2-layer MLP, hidden=256, GeLU)
# Hyperparameters: K = max sub-parts (K=8), lambda_comp = 0.5, lambda_rt = 1.0, tau = 0.1,
#                 margin m = 0.2, collision threshold delta = 0.05, calibration set size N_cal = 512

1. Parse p with P into sub-part spans s_1..s_K  # objects, attributes, relations; spaCy en_core_web_trf, frozen
2. Compute shared tokens Z = A(E(p))            # Integral-style Abstraction Fusion
3. Factorize: for each k, T_k = soft_assign(Z, s_k; tau)   # attention-pooled sub-token group, tau=0.1
4. Generate image x_hat = G(T_1, ..., T_K)       # Block-wise MoT generative stream
5. Round-trip: for each k, x_hat_k = D(T_k)      # decode each sub-token group alone (batched single pass)
6. Re-encode: z_k = E(x_hat_k)                  # frozen encoder re-encoding
7. Compositional loss with collision-aware target:
     # Precompute collision rate on calibration set of N_cal=512 sub-part pairs:
     #   for each pair (i,j) of visually distinct sub-parts, compute d_ij = ||A(E(p_i)) - A(E(p_j))||.
     #   If d_ij < delta (collision), mark the pair as colliding.
     # For each k, let C_k = {j : pair (k,j) collides}.
     # If C_k is empty: L_comp_k = || z_k - A(E(p_k)) ||^2
     # Else: L_comp_k = max(0, || z_k - A(E(p_k)) ||^2 - min_{j in C_k} || z_k - A(E(p_j)) ||^2 + m)
     #       (contrastive margin loss using the colliding sub-part codes as negatives)
     L_comp = sum_k L_comp_k
     L_rt   = || E(x_hat) - A(E(p)) ||^2        # full-image round-trip
     L_gen  = diffusion loss of G on x_hat
8. Total: L = L_gen + lambda_comp * L_comp + lambda_rt * L_rt

# Diagnostic (reported, not optimized): collision rate = fraction of calibration pairs with d_ij < delta.
# If collision rate > 0.1, the contrastive branch in step 7 is active for those pairs.
```

**Why this design.** We chose prompt-span factorization with attention-pooled soft assignment over a learned disentanglement head because the sub-part alignment is supplied by the parser, not inferred statistically, accepting the cost that parser errors propagate; this avoids the anti-pattern of asserting independence over tightly coupled variables. We chose a round-trip through the frozen encoder `E` over a separate perceptual loss because `E` is the same encoder that defines the shared space, so the reconstruction target is in the space the model actually uses, accepting the cost that `E`'s invariances limit sensitivity to fine texture. We chose per-sub-part decoding `D(T_k)` over decoding the full token set only, because it forces each `T_k` to carry its sub-part's content rather than distributing content arbitrarily across tokens, accepting the cost of `K` extra decoder passes (mitigated by batching all `K` decodes in a single pass). We chose a fixed parser `P` over a jointly trained segmenter to keep the alignment signal stable, accepting the cost that out-of-distribution prompts with unusual syntax may be mis-parsed. We chose a collision-aware contrastive target over a plain L2 target because the plain target can be minimized by encoder-invariance when distinct sub-parts collide in `E`-space; the contrastive branch requires `z_k` to be closer to its own sub-part code than to any colliding sub-part's code by margin `m`, accepting the cost that the calibration set must be representative of the sub-part manifold. A domain expert would not describe CTF as a variant of Integral (E6) because Integral only builds a shared space; CTF adds a factorized invariant with a round-trip constraint that Integral does not contain. Nor is it a variant of LMFusion (E7) or DDT-LLaMA (E9), which adapt pretrained LLMs or learn discrete tokens without a per-sub-part reconstruction constraint.

**Why it measures what we claim.** The quantity `L_comp = sum_k L_comp_k` measures compositional faithfulness of the shared token space because the assumption is that the frozen encoder `E` is injective on the image manifold relevant to prompt sub-parts, so equality in `E`-space implies equality of sub-part content; this assumption fails when two visually distinct sub-parts map to nearby `E` codes (e.g., fine textures or small objects), in which case `L_comp` reflects encoder-invariance rather than true sub-part faithfulness. To guard against this failure, we compute the collision rate on a calibration set of 512 sub-part pairs and, for colliding pairs, replace the L2 target with a contrastive margin loss that requires `z_k` to be closer to its own sub-part code than to any colliding sub-part's code by margin `m=0.2`; this makes `L_comp` discriminative even when `E` collapses distinct sub-parts. The quantity `L_rt = || E(G(T_1..T_K)) - A(E(p)) ||^2` measures full-prompt reconstruction fidelity because the assumption is that `A∘E` is a sufficient statistic of the prompt's visual content for generation; this assumption fails when the generative stream `G` introduces content not present in `Z`, in which case `L_rt` reflects generation drift rather than token-space faithfulness. The soft-assignment temperature `tau` measures the sharpness of sub-part-to-token alignment because the assumption is that each token belongs predominantly to one sub-part; this assumption fails for relational sub-parts spanning multiple objects, in which case `tau` reflects relation ambiguity rather than alignment quality. The parser `P` operationalizes the motivation-level notion of 'prompt sub-part' because the assumption is that syntactic spans coincide with semantic compositional units; this assumption fails for nested or elliptical prompts, in which case the factorization reflects syntactic rather than semantic structure.

## Contribution

['A compositional token factorization (CTF) mechanism that splits Abstraction-Fusion shared tokens into prompt-sub-part-aligned groups and enforces a frozen-encoder round-trip reconstruction constraint, turning compositionality into a representational invariant of the shared visual token space.', 'A design principle: per-sub-part decoding plus re-encoding through the same frozen encoder that defines the shared space yields a training signal that is in-distribution for the generative stream, unlike auxiliary perceptual or CLIP losses that live in a different space.', 'A reusable evaluation protocol that measures compositional faithfulness by re-encoding generated images and their sub-part crops through the frozen encoder and reporting per-sub-part reconstruction error, complementing human and FID-style metrics.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ScImage scientific compositional prompts | Stresses spatial/numeric/attribute sub-parts, not just overall fidelity |
| Dataset | T2I-CompBench (attribute-binding and spatial-relation subsets) | Tests generality beyond scientific prompts on standard compositional splits |
| Primary metric | Sub-part recombination faithfulness (re-encode distance) | Directly measures the claimed factorized invariant |
| Secondary metric | Collision rate on calibration set (N=512 pairs, delta=0.05) | Validates the injectivity assumption behind L_comp |
| Baseline | Integral (E6) | Shared-space unification without per-sub-part constraint |
| Baseline | LMFusion (E7) | Parallel-stream generation without factorized round-trip |
| Baseline | DDT-LLaMA (E9) | Discrete recursive tokens without per-sub-part decoding |
| Ablation-of-ours | CTF w/o per-sub-part decode (full-token round-trip only) | Isolates the factorization contribution from round-trip alone |
| Ablation-of-ours | CTF w/o collision-aware contrastive target (plain L2 only) | Isolates the injectivity-repair contribution |

### Why this setup validates the claim

The central claim is that forcing each shared sub-token group to re-encode its own sub-part makes the token space compositionally faithful, beyond what shared-space unification alone achieves. ScImage is the right dataset because it decomposes prompts into spatial, numeric, and attribute dimensions that map cleanly onto the parser's sub-part spans, giving us instances where a baseline can plausibly get the whole image roughly right while misplacing or omitting a specific sub-part. T2I-CompBench adds standard attribute-binding and spatial-relation subsets to test whether the effect generalizes beyond scientific prompts. The primary metric — re-encoding each decoded sub-token group through the frozen encoder and measuring distance to the sub-prompt's target code — is the only metric that directly tests the invariant CTF asserts; holistic image-quality scores would confound generation drift with factorization failure. The collision-rate diagnostic (512 sub-part pairs, threshold delta=0.05) directly tests the load-bearing injectivity assumption: if the collision rate is high, the plain L2 target is unreliable and the contrastive branch is necessary. Integral tests the sub-claim that a shared space alone is insufficient; LMFusion and DDT-LLaMA test whether alternative unification mechanisms (parallel streams, discrete recursive tokens) already deliver the invariant without an explicit per-sub-part constraint. The ablation — dropping per-sub-part decoding while keeping the full round-trip — is the decisive control: if the gain survives without factorization, the mechanism we credit is wrong. The second ablation — dropping the collision-aware contrastive target — isolates whether the injectivity repair matters. Together these form a falsifiable test because the predicted effect is localized: CTF should win specifically on recombination-heavy and multi-sub-part subsets, and the ablations should recover the baseline pattern there.

### Expected outcome and causal chain

**vs. Integral (E6)** — On a ScImage prompt combining a spatial relation with a numeric attribute (e.g., three squares left of a circle), Integral produces a globally plausible image but often swaps or drops the numeric count because its shared space is optimized for whole-prompt reconstruction, letting content distribute arbitrarily across tokens. Our method instead forces the numeric sub-token group to decode and re-encode to the count-restricted sub-prompt, because the per-sub-part loss pins that content to that group. We expect a noticeable gap on the multi-sub-part and relational subsets, with parity on single-dimension prompts where Integral's shared space already suffices.

**vs. LMFusion (E7)** — On a prompt where an attribute belongs to one object among several (e.g., a red square next to a blue circle), LMFusion's parallel image stream generates a coherent scene but frequently bleeds the attribute onto the wrong object, because its diffusion modules are trained on the full image and have no signal tying a token region to a specific sub-part. Our method instead constrains each object-attribute group through its own round-trip, so attribute bleed shows up as a large sub-part re-encode distance and is penalized. We expect a concentrated gain on attribute-binding cases and near-parity on simple single-object prompts.

**vs. DDT-LLaMA (E9)** — On a prompt with a spatial relation spanning two objects, DDT-LLaMA's discrete recursive tokens reconstruct the image well but exhibit relation ambiguity, because its timestep tokens encode progressive reconstruction rather than an explicit per-sub-part binding, and relational content is spread across the token stream. Our method instead assigns the relation span to its own token group with a sharpened soft assignment and a dedicated decode, so relation errors surface as that group's re-encode distance. We expect a modest but consistent gain on relational subsets and parity on purely object-listing prompts, where the mechanism has little to add.

**vs. CTF w/o per-sub-part decode** — This ablation keeps the full-image round-trip but drops the per-sub-part decode and loss. If the gain over baselines survives without factorization, then the round-trip alone is responsible and the factorization claim is wrong. We expect this ablation to match the baseline pattern on multi-sub-part and relational subsets, confirming that the per-sub-part constraint is the active ingredient.

**vs. CTF w/o collision-aware contrastive target** — This ablation uses plain L2 for all sub-parts, ignoring the collision diagnostic. If the collision rate is non-negligible (>0.1), we expect this ablation to underperform full CTF specifically on fine-texture and small-object sub-parts where E-space collisions occur, confirming that the injectivity repair is necessary.

### What would falsify this idea

If CTF's gain over Integral, LMFusion, and DDT-LLaMA is uniform across single-dimension, attribute-binding, and relational subsets rather than concentrated where the predicted factorization failure occurs — or if the no-per-sub-part-decode ablation matches full CTF — then the claimed factorized invariant is not the cause of any improvement and the central claim is wrong. Additionally, if the collision rate is high (>0.1) and the collision-aware contrastive ablation does not recover the gap on colliding sub-parts, then the injectivity assumption is fatally violated and the L_comp metric does not measure sub-part faithfulness.

## References

1. Let Vision-Language Models Generate Images through Their Representations
