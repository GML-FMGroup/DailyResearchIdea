# Decomposition-Closure Memory: Enforcing Compositional Sufficiency in Chunk-Level Parametric Memory via Fixed-Point Fusion

## Motivation

Across two independent frontier trees, a learned localized representation is composed into a larger computation under an unchecked sufficiency assumption: Deep Fusion Memory fuses a single chunk-level memory vector shared across all tokens of a chunk and assumes it captures the token-level variation needed downstream, while MCircKE edits parameters inside a mapped multi-hop circuit and assumes the edited circuit remains sufficient for every alternative reasoning path that decomposes the same query. Neither work imposes a closure condition on the composition operator, so sufficiency is assumed rather than enforced — there is no operational test that the fused representation is a fixed point across the decomposition lattice of the chunk, and no training signal that would make it one. This is the same structural gap in both trees: a localized representation's sufficiency under composition is never made checkable.

## Key Insight

Compositional sufficiency is exactly the fixed-point condition of the memory-fusion operator over the chunk's decomposition lattice, so a fused chunk memory is admissible iff re-deriving it from any admissible decomposition of the chunk reproduces it — a closure property that can be enforced as a training-time residual and read out by causal intervention at inference.

## Method

**What it is.** Decomposition-Closure Memory (DCM) augments a chunk-level parametric memory pipeline with a closure operator C that maps a fused chunk memory m to the memory re-derived from each decomposition of the chunk, and trains the fusion operator F so that F is a fixed point of C. Inputs: preceding context x, retrieved chunk embedding target e, and the chunk's token set T. Outputs: an admissible fused memory m* and a closure residual r that certifies (or flags) admissibility at inference.

**Load-bearing assumption (stated explicitly).** The closure residual ||m - m_d||_2^2 is meaningful only if COMPOSE_d — the gated cross-attention readout restricted to decomposition d's token positions — faithfully reproduces the frozen backbone's actual downstream multi-hop readout over the chunk given the fused memory. This is an assumption, not a guarantee: fixed-depth transformers cannot in general compose arbitrary multi-hop paths (Merrill & Sabharwal, TACL 2023; Merrill, Sabharwal & Smith, TACL 2022), and multi-hop composition failures are not always captured by representation-consistency signals (Dziri et al., NeurIPS 2023). Consequently, a small residual can reflect readout self-consistency rather than genuine compositional sufficiency, and L_closure can be driven to zero by collapsing m toward a decomposition-invariant point (a trivial fixed point, e.g., m constant across decompositions), which satisfies closure while destroying the token-level variation the motivation claims to preserve. DCM addresses this by (i) anchoring COMPOSE_d to the frozen backbone's own next-token/logit computation over the chunk given the fused memory, not to F_theta's gated cross-attention alone, (ii) adding a non-degeneracy constraint that excludes the trivial constant fixed point, and (iii) validating on held-out decompositions not in LATTICE(T).

**How it works.**

```pseudocode
# --- Training (predictor + fusion only; backbone frozen) ---
for each chunk (x, e, tokens T):
    m = F_theta(x, e)                       # gated cross-attn fusion, as in Deep Fusion Memory
    # Build the decomposition lattice D(T) = {token singletons, contiguous spans, hop-paths}
    D = LATTICE(T)                          # includes |T| singletons + span partitions + K hop paths
    residuals = []
    for d in D:                             # each decomposition d is a set of sub-units
        # COMPOSE_d is the frozen backbone's actual downstream readout over the chunk
        # given the fused memory, restricted to decomposition d's token positions:
        # logits_d = BACKBONE(x, m, positions(d)); m_d = READOUT(logits_d)
        m_d = COMPOSE_d(x, m)               # frozen backbone readout, not F_theta's gated cross-attn
        residuals.append( || m - m_d ||_2^2 )
    L_closure = mean(residuals)             # decomposition-consistency loss
    # Non-degeneracy: exclude trivial constant fixed point via variance/injectivity penalty
    L_var = -Var_{d in D}( m_d )            # penalize m_d constant across decompositions
    # Information preservation against chunk embedding target e
    L_info = || m - e ||_2^2                # chunk-embedding supervision
    L = L_info + lambda * L_closure + mu * L_var   # lambda = 1.0, annealed 0.1 -> 1.0; mu = 0.1
    update theta by gradient on L

# --- Inference ---
for each chunk:
    m = F_theta(x, e)
    r = max_d || m - COMPOSE_d(x, m) ||_2   # closure residual, no gradient
    if r < tau:  fuse m via gated cross-attention into all tokens of the chunk
    else:        split chunk into the decomposition d* attaining min residual and fuse per sub-unit
```

`COMPOSE_d` is the frozen backbone's actual downstream readout over the chunk given the fused memory, restricted to the token positions of decomposition d: it computes the backbone's next-token/logit distribution over the chunk conditioned on (x, m) and extracts the logits at positions(d), then maps them to a memory vector via a fixed linear readout (the backbone's final-layer projection). `LATTICE(T)` enumerates token singletons, contiguous span partitions up to length 4, and the K=8 highest-probability hop paths returned by the backbone's own next-token distribution over the chunk. The non-degeneracy term `L_var` is the negative variance of `m_d` across decompositions d in D, with weight mu=0.1, which penalizes collapse of m_d to a constant. The closure residual `r` is a scalar per chunk; `tau` is set to the 90th percentile of training residuals. Held-out decompositions (not in LATTICE(T)) are used during validation to check that the residual is not gamed by the enumerated lattice.

**Why this design.** We chose a fixed-point residual over a separately-trained discriminator because the discriminator would need labels of 'sufficient vs. insufficient' memories that do not exist, whereas the residual is defined by the fusion operator itself and needs no new supervision — accepting the cost that the residual is only as meaningful as the lattice enumeration. We chose to enumerate decompositions from the backbone's own next-token distribution over hop paths rather than a fixed syntactic parser because the downstream reasoning paths MCircKE-style editing actually traverses are distributional, not syntactic — accepting the cost that rare long-hop paths are undersampled and the lattice is approximate. We chose to keep the backbone frozen and train only the predictor and fusion modules, matching Deep Fusion Memory's construction-cost motivation, because unfreezing would let the backbone absorb closure violations into its own weights and destroy the diagnostic value of `r` — accepting the cost that closure must be achieved by the fusion operator alone. A domain expert would not describe DCM as a variant of Deep Fusion Memory: Deep Fusion Memory trains fusion against a chunk-embedding target with no consistency term, whereas DCM's load-bearing signal is the residual between the fused memory and its re-derivation across decompositions, which Deep Fusion Memory never computes.

**Why it measures what we claim.** The closure residual `||m - m_d||_2^2` measures compositional sufficiency because it is zero exactly when the fused memory is a fixed point of the composition map over decomposition d; this equivalence assumes that `COMPOSE_d` faithfully implements the downstream readout that decomposition d induces (including nonlinearity match), and it fails when a decomposition's readout is nonlinear in a way the frozen backbone's readout does not reproduce, in which case the residual reflects readout mismatch rather than genuine insufficiency. The lattice enumeration `LATTICE(T)` measures the space of admissible downstream decompositions because it is generated from the backbone's own hop-path distribution; this assumes the backbone's next-token distribution over the chunk is a faithful proxy for the reasoning paths MCircKE-style editing would traverse, and it fails on tasks whose correct path has low next-token probability, in which case the lattice omits the decomposition that would expose insufficiency. The inference-time flag `r < tau` measures admissibility because `tau` is calibrated to the training residual distribution where closure was enforced; this assumes the test chunk distribution matches the training chunk distribution, and it fails under distribution shift, in which case `r` reflects novelty rather than closure violation.

## Contribution

(1) A decomposition-closure operator and training objective for chunk-level parametric memory that turns the compositional-sufficiency assumption into a checkable fixed-point residual, together with a lattice enumeration over token singletons, span partitions, and backbone-derived hop paths. (2) An inference-time admissibility flag that uses the closure residual to decide between whole-chunk fusion and per-sub-unit fusion, yielding a concrete design principle: chunk-level sharing is safe exactly on the sub-lattice where the residual is below the calibrated threshold. (3) A diagnostic protocol that reuses causal-intervention tooling (token-level probing, circuit mapping) to attribute closure violations to specific decompositions, enabling comparison of sufficiency across Deep Fusion Memory-style and MCircKE-style compositions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MQuAKE multi-hop editing suite | Multi-hop paths expose compositional insufficiency directly |
| Dataset | Long-context compression task (e.g., chunked language modeling on PG-19) | Tests closure residual in non-editing parametric memory setting |
| Primary metric | Multi-hop reasoning accuracy | Directly reflects whether fused memory composes correctly |
| Secondary metric | Correlation between closure residual r and downstream error; calibration curve (residual decile vs. error rate) | Tests whether r tracks insufficiency vs. novelty |
| Baseline | Deep Fusion Memory | Same fusion pipeline, no closure consistency term |
| Baseline | MEMIT | Layer-localized edit, no chunk-level compositional memory |
| Baseline | CaKE | Circuit-aware edit, no closure residual certification |
| Baseline | Average of sub-unit memories (ensembling) | Tests whether closure operator reduces to known ensembling |
| Ablation-of-ours | DCM without closure loss (λ=0) | Isolates contribution of the closure residual signal |
| Ablation-of-ours | DCM without non-degeneracy term (μ=0) | Tests whether trivial constant fixed point is excluded |
| Ablation-of-ours | DCM with COMPOSE_d = F_theta's gated cross-attention (self-referential) | Tests whether external anchoring is necessary |

### Why this setup validates the claim

The central claim is that a fused chunk memory is admissible exactly when it is a fixed point of the decomposition-closure operator, and that this residual — computable without new supervision — predicts and repairs compositional insufficiency on multi-hop reasoning. MQuAKE is the right testbed because its correct answers require traversing multi-hop paths whose intermediate decompositions are exactly the objects DCM's lattice enumerates; a chunk memory that looks fine against its embedding target but fails to re-derive across those decompositions should produce multi-hop errors. Deep Fusion Memory is the load-bearing baseline: it shares DCM's fusion operator and backbone but lacks the closure term, so any gap between them isolates the closure signal rather than confounding from architecture. MEMIT and CaKE test whether chunk-level closure certification adds anything over layer-localized editing that never constructs an explicit per-chunk memory. The average-of-sub-unit-memories baseline tests whether the closure operator is reducible to known ensembling. The λ=0 ablation is decisive: it keeps the lattice and residual machinery at inference but removes the training pressure that makes F a fixed point, so if DCM's gain comes from closure enforcement rather than from the lattice enumeration alone, the ablation should lose most of the gain. The μ=0 ablation tests whether the non-degeneracy term is necessary to exclude the trivial constant fixed point. The self-referential COMPOSE_d ablation tests whether external anchoring to the frozen backbone's readout is necessary for the residual to measure sufficiency rather than self-consistency. The long-context compression task tests whether the closure residual predicts downstream errors in a non-editing parametric-memory setting, broadening significance beyond a single benchmark family. The calibration curve (residual decile vs. error rate) directly tests whether r tracks insufficiency rather than novelty or readout mismatch.

### Expected outcome and causal chain

**vs. Deep Fusion Memory** — On a case where a chunk's fused memory encodes the first hop of a two-hop chain but not the composition with the second, Deep Fusion Memory produces a plausible first-hop answer because its fusion is trained only against the chunk-embedding target and has no pressure to be re-derivable from sub-units. Our method instead detects a high closure residual r for that chunk and, at inference, splits into the decomposition attaining the minimum residual, fusing per sub-unit, because training drove F toward the fixed point of C. We expect a noticeable multi-hop accuracy gap concentrated on chunks whose training residual sits near the τ threshold, with near-parity on single-hop chunks where closure is trivially satisfied.

**vs. MEMIT** — On a case where an edit must propagate through a reasoning circuit spanning multiple layers, MEMIT produces the edited fact in isolation but fails the multi-hop query because its layer-localized update cannot represent a chunk-level memory whose decomposition-closure is certified. Our method instead constructs an explicit per-chunk fused memory and fuses it into all chunk tokens only when r < τ, so the edit is carried by a memory whose compositional sufficiency was checked. We expect a large gap on multi-hop MQuAKE items and parity on single-fact recall, since MEMIT's mechanism is adequate exactly where composition is not required.

**vs. CaKE** — On a case where the correct reasoning path has low next-token probability under the frozen backbone, CaKE produces a generalizable edit along the circuit it diagnosed but misses the low-probability hop, because its circuit analysis does not enumerate distributional hop paths. Our method instead flags the chunk via a high residual and falls back to the minimum-residual decomposition, which is chosen from the lattice that includes the K highest-probability hop paths. We expect DCM to recover a fraction of the low-probability-hop subset that CaKE misses, with parity on high-probability-hop items, and this concentration is the signature that the lattice enumeration is doing the claimed work.

**vs. Average of sub-unit memories (ensembling)** — On a case where sub-unit memories are individually plausible but their average is not a fixed point of the composition operator, the ensembling baseline produces a memory that satisfies the embedding target but fails to re-derive across decompositions. Our method instead enforces closure, so the fused memory is a fixed point and the residual is small. We expect DCM to outperform the ensembling baseline on multi-hop items where composition is non-linear, demonstrating that the closure operator is not reducible to known ensembling.

**vs. DCM without non-degeneracy term (μ=0)** — On a case where the closure loss can be driven to zero by collapsing m toward a decomposition-invariant constant, the μ=0 ablation achieves low closure residual but destroys token-level variation, leading to poor multi-hop accuracy. Our method with μ=0.1 excludes this trivial fixed point via the variance penalty, maintaining both low residual and high accuracy. We expect the μ=0 ablation to show a large gap on multi-hop items, confirming the necessity of the non-degeneracy constraint.

**vs. DCM with self-referential COMPOSE_d** — On a case where the gated cross-attention readout does not faithfully reproduce the frozen backbone's downstream readout, the self-referential variant achieves low residual through readout self-consistency but fails to certify genuine sufficiency, leading to multi-hop errors. Our method anchors COMPOSE_d to the frozen backbone's actual readout, so the residual reflects true compositional sufficiency. We expect the self-referential variant to show a significant accuracy drop on multi-hop items, validating the external anchoring.

### What would falsify this idea

If DCM's multi-hop gain over Deep Fusion Memory is uniform across chunks rather than concentrated on chunks with high training closure residuals, the residual is not measuring compositional sufficiency and the central claim is wrong. Equally, if the λ=0 ablation matches full DCM on multi-hop accuracy, the closure term is not the load-bearing signal. Additionally, if the calibration curve shows no monotonic relationship between residual decile and error rate, or if the μ=0 ablation matches full DCM, the residual is not tracking insufficiency and the non-degeneracy constraint is unnecessary.

## References

1. Deep Fusion Memory: A Chunk-Level Parametric Memory for Large Language Models
2. Addressing the Reasoning Gap: Mechanistic Circuit-Based Knowledge Editing in Large Language Models
3. CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners
4. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models
5. Uncovering Overfitting in Large Language Model Editing
6. PMET: Precise Model Editing in a Transformer
