# Invertible Query-to-Coordinate Decoding for Identifiable Spatial Grounding in Embodied World Models

## Motivation

Foundation models coupled to spatial representations (as surveyed in 'Foundation Models Meet Spatial Representations: Advancing Zero-Shot Navigation for Embodied AI') still rely on a fixed-cardinality Q-Former whose query tokens are trained only by aligning to a target representation, with no constraint that distinct queries map to distinct spatial locations. This is a structural defect that recurs across the whole family of query-bottleneck embodied world models: because injectivity is never imposed, the query-to-coordinate mapping stays non-identifiable, so a query token can silently collapse onto another token's location or onto a location irrelevant to the action, and perceptual uncertainty then propagates into uninterpretable planning decisions. Replacing the alignment target (as prior spatial-representation work does) cannot fix this, because the missing ingredient is a spatial identifiability constraint, not a better target.

## Key Insight

Enforcing an invertible query-to-coordinate decoder makes the query set a bijective code for action-relevant spatial coordinates, so each query's location is recoverable from the token itself rather than inferred up to an unconstrained permutation and collapse.

## Method

**What it is.** We introduce SPIN-Q (SPatially INvertible Query decoding), which augments a Q-Former-style interface with an invertible spatial decoder. Inputs: a frozen vision-language backbone's patch features and a set of N learnable query tokens. Outputs: (i) N query embeddings used by the downstream policy, and (ii) for each query, a decoded 3D coordinate in the agent's egocentric frame, produced by an invertible map so that distinct queries are forced onto distinct coordinates.

**How it works.**

```pseudocode
# N queries, D-dim; coordinate space R^3; invertible decoder is a normalizing flow
init: queries Q = {q_1..q_N} ~ N(0, I)          # N=32, D=256
for each training step:
    F = frozen_VLM_patch_features(observation)   # partial-observability: random 40-70% mask
    Q' = cross_attention(Q, F)                    # standard Q-Former update
    # invertible decoder: q_i -> coordinate c_i, bijective by construction
    c_i = flow_inverse(q_i)  for i in 1..N        # RealNVP-style coupling, 6 layers
    # contrastive spatial grounding loss under partial observability
    P = sample_visible_anchors(F, mask)           # action-relevant coords visible this step
    L_ground = InfoNCE({c_i}, P, tau=0.07)        # each query matched to its anchor
    # injectivity penalty: decoded coords must be pairwise separated
    L_sep = mean_{i!=j} max(0, eps - ||c_i - c_j||)   # eps=0.15 m
    # reconstruction keeps the map truly invertible in practice
    L_rec = || flow_forward(c_i) - q_i ||^2
    L = L_ground + lambda1*L_sep + lambda2*L_rec       # lambda1=0.5, lambda2=1.0
    update(Q, flow_params) by Adam
```

At inference the decoder yields an explicit coordinate for every query, so the policy consumes both the embedding and its grounded location; the invertibility means a coordinate can be mapped back to its query token for inspection.

**Why this design.** We chose an invertible normalizing-flow decoder over a plain MLP coordinate head because only a bijection guarantees that distinct query tokens cannot collapse onto the same coordinate, accepting the cost of a heavier decoder and a reconstruction term that must be balanced against grounding. We chose a pairwise separation penalty over a soft cardinality regularizer because separation acts directly on decoded coordinates (the quantity we claim is identifiable) rather than on embedding norms, accepting that it introduces one distance threshold that must be tuned per metric scale. We chose an InfoNCE grounding loss over an L2 regression to anchors because contrastive matching tolerates the many-to-one ambiguity induced by partial observability, accepting noisier gradients when few anchors are visible. A domain expert would not describe SPIN-Q as a variant of the standard Q-Former alignment pipeline (e.g., BLIP-2's query-target alignment) because that pipeline only aligns query outputs to a target representation; SPIN-Q adds an invertibility constraint and a coordinate-space separation term that the alignment pipeline structurally lacks, making the query set a code for spatial positions rather than a free-floating bottleneck.

**Why it measures what we claim.** The InfoNCE grounding loss measures action-relevant spatial grounding because it is equivalent to a maximum-likelihood assignment under the assumption that each visible anchor has exactly one query responsible for it; this assumption fails when partial observability hides the anchor that a query was matched to in a previous step, in which case the loss reflects matching to whichever visible anchor is nearest rather than the true responsible location. The pairwise separation penalty measures query-to-coordinate injectivity because, under the flow's bijectivity, coordinate separation is equivalent to query-token distinctness; this assumption fails when the flow's coupling layers become near-singular, in which case separation reflects decoder conditioning rather than token distinctness. The reconstruction term measures whether the decoder is genuinely invertible because a bijective map must round-trip its input; this assumption fails when the flow overfits the training queries, in which case reconstruction reflects memorization rather than invertibility on novel queries.

## Contribution

(1) SPIN-Q, an invertible query-to-coordinate decoder that imposes a bijective mapping between Q-Former query tokens and action-relevant spatial coordinates, turning the query set into an identifiable spatial code. (2) A contrastive spatial grounding objective combined with a coordinate-space separation penalty that, unlike target-replacement approaches, directly constrains injectivity of the query-to-location map under simulated partial observability. (3) A design principle showing that spatial identifiability in query-bottleneck embodied world models is governed by decoder invertibility rather than by the choice of alignment target.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HM3D-based zero-shot navigation episodes with 40–70% observation masking | Partial observability is the claimed failure regime. |
| Primary metric | Success rate on masked-anchor navigation episodes | Directly measures grounding under hidden anchors. |
| Baseline | BLIP-2 Q-Former alignment pipeline | Tests whether invertibility adds anything beyond alignment. |
| Baseline | Q-Former + MLP coordinate head | Tests whether bijection beats plain regression. |
| Baseline | SLAM-based zero-shot navigation | Tests against explicit geometric mapping. |
| Ablation-of-ours | SPIN-Q without separation penalty | Isolates injectivity contribution. |

### Why this setup validates the claim

The central claim is that forcing query tokens onto distinct 3D coordinates via an invertible decoder yields grounded, identifiable spatial codes that survive partial observability. The BLIP-2 Q-Former baseline tests the sub-claim that alignment alone is insufficient: it has the same frozen features and query budget but no coordinate-space constraint, so any gap isolates the invertibility contribution. The MLP-head baseline tests whether bijectivity specifically matters versus merely predicting coordinates — it can regress to anchors but permits query collapse. The SLAM baseline tests whether learned spatial codes beat classical geometric mapping when anchors are masked. The ablation removes only the separation penalty, so if performance drops only there, injectivity is doing causal work. Success rate on masked-anchor episodes is the right metric because it is the only measure that directly reflects whether a query's decoded coordinate still points at the responsible location when that anchor is hidden — exactly the assumption the method claims to preserve.

### Expected outcome and causal chain

**vs. BLIP-2 Q-Former alignment pipeline** — On a case where the agent must reach a target whose anchor was visible two steps ago but is now masked, the baseline produces queries that drift toward whichever currently-visible anchor is nearest because its alignment loss has no coordinate-space identity constraint. Our method instead keeps each query pinned to its decoded coordinate through the separation penalty and reconstruction term because the bijection forbids two queries sharing a location, so we expect a noticeable gap on high-masking episodes but parity when masking is low.

**vs. Q-Former + MLP coordinate head** — On a case where two action-relevant anchors are spatially close, the baseline produces collapsed coordinates for both queries because an unconstrained MLP head has no injectivity pressure and can map distinct tokens to the same point. Our method instead maintains separation because the flow's bijectivity plus the eps-margin penalty forces distinct coordinates, so we expect a concentrated gain on close-anchor subsets and near-parity on well-separated subsets.

**vs. SLAM-based zero-shot navigation** — On a case where a semantic target is described in language but its anchor is briefly occluded, the baseline produces a stale or missing map entry because SLAM has no learned query-to-coordinate code to fall back on. Our method instead retrieves the grounded coordinate from the query embedding because the invertible decoder preserves the token-coordinate association across steps, so we expect a moderate gap on occlusion-heavy episodes and parity on fully-visible ones.

**vs. SPIN-Q without separation penalty** — On a case where the query budget exceeds the number of visible anchors, the ablation produces duplicate decoded coordinates because nothing prevents collapse, degrading downstream policy input. Our method instead spreads queries across distinct coordinates because the separation penalty acts directly on the claimed identifiable quantity, so we expect the ablation to lose accuracy specifically when visible anchors are fewer than queries.

### What would falsify this idea

If SPIN-Q's gain over the MLP-head baseline is uniform across close-anchor and well-separated subsets rather than concentrated where coordinate collapse is predicted, then injectivity is not the causal mechanism and the central claim is wrong. Likewise, if removing the separation penalty leaves masked-anchor success unchanged, the invertibility constraint is doing no identifiable work.

## References

1. Foundation Models Meet Spatial Representations: Advancing Zero-Shot Navigation for Embodied AI
