# Collision-Aware Codebook Learning for Identifiable Fine-Grained Image Tokens in Single-Pass Vision-Language Generation

## Motivation

Discrete image-code tokenization is the shared substrate of recent single-pass and autoregressive generators — Jeveration learns a distribution over discrete image codes to guide its parallel one-pass decoder, and DnD-Transformer predicts discrete codes along a model-depth direction — yet both rest on an unverified codebook injectivity assumption: that distinct fine-grained visual primitives (text strokes, symbols, graphical marks) receive distinct codes. When two primitives quantize to the same code, the decoder's parallel prediction is under-determined for exactly the regions where fine detail lives, so the shared random state can only coordinate a coarse whole and the collision is invisible to the training loss because reconstruction of the merged code is locally plausible. Jeveration's abstract reports only 'competitive image quality' with no analysis of primitive identifiability, which is consistent with this structural blind spot being unaddressed.

## Key Insight

Injectivity of the codebook on the set of fine-grained primitives is a property that can be certified by a reconstruction-identifiability test on the codebook itself, so collisions can be converted from an invisible downstream failure into a directly penalized, codebook-level training signal.

## Method

**What it is.** Collision-Aware Quantization (CAQ) is a codebook-learning procedure for the discrete image-code stage that Jeveration (and any VQ-style autoregressive image generator) uses before single-pass generation. Its inputs are the encoder features of training images, a set of primitive-labeled crops (text strokes, symbols, graphical marks), and an initial codebook; its outputs are a revised codebook in which distinct primitives provably map to distinct codes, plus a collision report that flags any primitive pair the codebook cannot separate.

**Load-bearing assumption (stated explicitly).** CAQ assumes that a code shared by two distinct fine-grained primitives necessarily forces the decoder to emit an averaged, unreadable output, so that codebook injectivity on primitive labels is both necessary and sufficient for primitive-level generation fidelity. This assumption can fail: the decoder is conditioned on spatial position and surrounding context, so a single code can be disambiguated by context. To make the assumption testable rather than assumed, CAQ replaces the codebook-level injectivity audit with a decoder-conditioned identifiability test: a code is split or rejected only when the frozen decoder, given the shared code plus its spatial/contextual conditioning, produces outputs that are actually ambiguous for the two primitive classes. The collision report therefore distinguishes codebook collisions that are decoder-ambiguous (actionable) from those that are decoder-resolvable (benign).

**How it works.**

```pseudocode
INPUT: encoder E, decoder Dec, codebook C = {c_1..c_K}, primitive-crop set P = {(x_i, p_i)}  # p_i = primitive instance label
HYPER: tau_inj = 0.02 (collision margin), lambda_col = 0.5, lambda_rec = 1.0, K_max = 8192, rho = 0.1,
       tau_amb = 0.05 (decoder-ambiguity threshold on output distance), audit_budget = 3 iterations

PHASE 1  primitive feature extraction
  for each (x_i, p_i) in P:
      z_i = E(x_i)                         # continuous feature
      k_i = argmin_k || z_i - c_k ||       # current assignment
  build collision table T: pairs (i,j) with p_i != p_j and k_i == k_j

PHASE 2  collision-aware objective (per training step)
  L_rec  = || Dec(z_i) - x_i ||^2                      # standard reconstruction
  L_col  = sum over colliding pairs (i,j) in T of
             max(0, tau_inj - || z_i - z_j ||)          # push distinct primitives apart in feature space
  L_assign = sum_i || z_i - c_{k_i} ||^2                # keep assignments tight
  L = L_rec + lambda_rec * L_assign + lambda_col * L_col
  update E and C by gradient descent on L

PHASE 3  decoder-conditioned identifiability check (codebook audit)
  for each code c_k:
      S_k = { i : k_i == k }
      for each pair (i,j) in S_k with p_i != p_j:
          # decoder-conditioned ambiguity: does the frozen decoder, given c_k and the spatial/contextual conditioning of crop i vs crop j, produce distinguishable outputs?
          y_i = Dec(c_k, context_i)   # context_i = spatial position + surrounding tokens of crop i
          y_j = Dec(c_k, context_j)
          if || y_i - y_j || < tau_amb:
              # decoder cannot disambiguate -> actionable collision
              if |C| < K_max:
                  split c_k into c_k1, c_k2 initialized at z_i, z_j; add to C
              else:
                  mark c_k as REJECTED; route its members to nearest non-rejected code
          else:
              # decoder resolves via context -> benign collision, record in report only
              mark (i,j) as BENIGN in collision_report
  repeat PHASE 2-3 until T contains no actionable collisions or audit_budget exhausted

PHASE 4  export
  return C_clean, collision_report (pairs still colliding, codes rejected, benign pairs flagged)
```

The revised codebook C_clean then replaces Jeveration's learned discrete code distribution's support, and the frozen vision-language model's single-pass decoder is fine-tuned on the cleaned code distribution.

**Why this design.** We chose a *feature-space margin* penalty (L_col) over a *post-hoc codebook pruning* because pruning removes capacity that the generator needs for coarse content, accepting the cost that the margin must be tuned per primitive granularity; we chose *code splitting* over *codebook reinitialization* in Phase 3 because splitting preserves the assignments already learned for non-colliding regions, accepting the cost that the codebook grows and the code distribution must be re-estimated; we chose a *primitive-labeled audit set* over a fully unsupervised collision detector because identifiability is only meaningful relative to a notion of 'distinct primitive', accepting the cost that the audit set must be curated and may not cover all primitive types; and we chose to *reject* codes at the capacity ceiling rather than silently keep them, accepting the cost that some fine detail is explicitly declared unrecoverable rather than hallucinated. A domain expert would not describe this as a variant of standard VQ codebook learning (e.g., the RQ-Transformer / DnD-Transformer family) because those methods optimize reconstruction and commitment losses only and never test whether the codebook is injective on primitive classes; nor is it a variant of retrieval-based codebook lookup, since the mechanism intervenes on the codebook itself (splitting and rejecting codes) rather than selecting among fixed codes.

**Why it measures what we claim.** The collision count |T| measures *codebook non-injectivity on fine-grained primitives* because we assume that two crops with different primitive labels that land on the same code are genuinely distinguishable visual primitives rather than label noise; this assumption fails when the primitive labels are coarse (e.g., two different fonts of the same letter labeled as different primitives), in which case |T| reflects label granularity rather than quantization failure. The margin term L_col measures *feature-space separability of distinct primitives* because we assume Euclidean distance in the encoder feature space is monotone in primitive distinctness; this assumption fails when the encoder entangles primitive identity with style or position, in which case L_col pushes apart style variants rather than primitive variants. The decoder-conditioned identifiability check measures *whether a code can support primitive-level decoding* because we assume that a code shared by two primitive classes forces the decoder to output an average that matches neither; this assumption fails when the decoder has enough capacity to disambiguate via spatial context, in which case the audit flags the pair as BENIGN rather than over-rejecting codes that were in fact usable.

## Contribution

(1) Collision-Aware Quantization (CAQ), a codebook-learning procedure that converts primitive-level code collisions into a directly optimized margin penalty plus a reconstruction-identifiability audit that splits or rejects non-injective codes. (2) A design principle: codebook injectivity on fine-grained primitives is a trainable, auditable property rather than an assumption, and capacity ceilings should be reported as explicit rejection sets rather than absorbed into reconstruction loss. (3) A collision report artifact that names which primitive pairs a given codebook cannot separate, enabling targeted comparison across tokenizers used by single-pass and autoregressive generators.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Primitive-rich text/graphic image set (crops labeled) | Supplies ground-truth primitive classes for collision audit |
| Primary metric | Primitive-collision rate | Directly measures codebook non-injectivity claim |
| Baseline | Jeveration (unmodified codebook) | Tests whether CAQ improves over the base generator |
| Baseline | Standard VQ codebook learning | Tests whether margin/splitting beats reconstruction-only |
| Baseline | Post-hoc codebook pruning | Tests whether feature-margin beats capacity removal |
| Ablation-of-ours | CAQ without Phase 3 splitting | Isolates contribution of code splitting |
| Ablation-of-ours | CAQ with decoder-conditioned audit disabled (codebook-level audit only) | Tests whether decoder-conditioned identifiability is necessary or whether codebook-level collision detection suffices |

### Why this setup validates the claim

The central claim is that CAQ makes the discrete codebook injective on fine-grained primitives, and that this injectivity translates into better generation of text strokes, symbols, and graphical marks. The primitive-collision rate is the right primary metric because it is exactly the quantity the method intervenes on: two distinct primitive labels sharing a code is the failure CAQ is built to remove, so a drop in collision rate is a direct signature that the mechanism did what it claims. The unmodified Jeveration baseline tests whether the intervention helps at all relative to the frozen starting point; the standard VQ baseline tests whether the margin and splitting mechanism beats reconstruction-only learning, isolating the value of collision-awareness; the post-hoc pruning baseline tests whether feature-space separation beats simply deleting colliding codes, which is the core design choice. The no-splitting ablation isolates Phase 3, so if gains vanish without splitting we know the margin term alone is insufficient. The decoder-conditioned-audit-disabled ablation isolates the load-bearing assumption: if codebook-level collision detection alone matches full CAQ, then the decoder-conditioned identifiability test adds nothing and the simpler proxy is sufficient. Together these form a falsifiable test: if CAQ's collision rate is not lower than all three baselines, or if the gain is uniform rather than concentrated on primitive-rich regions, the claim fails.

### Expected outcome and causal chain

**vs. Jeveration (unmodified codebook)** — On a case where a training image contains a digit and a similarly-shaped symbol that co-occur, the baseline assigns both to one code because its reconstruction loss treats them as interchangeable, so its decoder emits a blurry average that reads as neither. Our method instead splits that code in Phase 3 because the decoder-conditioned audit detects that the frozen decoder produces indistinguishable outputs for the two primitives under their respective spatial contexts, giving each primitive its own code, so we expect a clear collision-rate drop concentrated on primitive-rich crops and near-parity on plain natural-image regions.

**vs. Standard VQ codebook learning** — On a case where two primitive classes are close in encoder feature space, the baseline keeps them colliding because reconstruction and commitment losses do not penalize shared codes, so it silently loses the distinction. Our method instead applies L_col to push those features apart before assignment, so the margin prevents the collision from forming rather than repairing it afterward; we expect a moderate but consistent gap favoring CAQ on the closest primitive pairs, with little difference on well-separated primitives.

**vs. Post-hoc codebook pruning** — On a case where a colliding code also carries useful coarse content, the baseline deletes it and loses that content, producing degraded coarse structure. Our method instead splits the code, retaining capacity for the coarse content while separating the primitives, so we expect CAQ to match pruning on collision rate but outperform it on overall image quality and coarse-structure fidelity.

**vs. CAQ with decoder-conditioned audit disabled (codebook-level audit only)** — On a case where two primitives share a code but the decoder can disambiguate them via spatial context (e.g., a symbol that only appears in a fixed corner region), the codebook-level audit would split or reject that code unnecessarily, consuming capacity and potentially degrading coarse content. Our method instead flags the pair as BENIGN and leaves the code intact, so we expect full CAQ to match the codebook-level variant on collision rate for genuinely ambiguous pairs while outperforming it on overall image quality and codebook utilization, confirming that the decoder-conditioned test is necessary to avoid over-rejection.

### What would falsify this idea

If the collision-rate gain is uniform across all subsets rather than concentrated on primitive-rich crops, or if the no-splitting ablation matches full CAQ, the claim that collision-aware codebook intervention is the causal driver would be wrong. If post-hoc pruning matches CAQ on both collision rate and image quality, the feature-margin design adds nothing. If the decoder-conditioned-audit-disabled ablation matches full CAQ on both collision rate and image quality, the load-bearing assumption that codebook injectivity is necessary is false and the decoder-conditioned test adds nothing.

## References

1. Jeveration: System One Models Now Generate
