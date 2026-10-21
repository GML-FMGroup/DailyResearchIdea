# Reference-Anchored Positional Re-encoding with a Reference Sink for Drift-Free Streaming Video Generation

## Motivation

The convergent frontier exemplified by SolarRenderer inherits the streaming substrate (block-causal MiniMax-H3) and few-step LoRA objectives from prior work, but silently drops the two mechanisms StreamAvatar identified as necessary for long-horizon stability: a retained reference anchor and positional re-encoding of the conditioning signal. Because the control-LoRA pathway places structural reference tokens at the positions of the frames they describe while the streaming latent state advances under its own block-causal positional frame, the two positional frames are allowed to drift relative to each other over long horizons, so structural conditioning silently decoheres even though each individual four-step render looks locally correct. This is a meta-level gap: both branches of the frontier assume the streaming substrate plus few-step training is sufficient, so no branch establishes long-horizon stability under structural reference conditioning.

## Key Insight

Stability across a long horizon is a property of a shared positional frame, not of the denoising step count: if the reference sink and the streaming latent state are re-encoded into one drift-free frame at every block, their relative offset is an invariant of the rollout rather than a quantity that accumulates error.

## Method

**What it is.** We introduce RAPR-Sink (Reference-Anchored Positional Re-encoding with a Reference Sink), a drop-in modification to the control-LoRA pathway of a block-causal streaming renderer. Inputs: a structural reference video R (frames r_1..r_M), a plain-text description, and a streaming latent state z_t. Output: a structure-conditioned latent update in which the reference tokens and z_t occupy a single shared positional frame, with a persistent sink slot holding the anchor. It adds no new backbone and no new trainable large module; it adds one positional re-encoding operator and one sink slot per attention block.

**Load-bearing assumption (stated explicitly).** The method assumes that the backbone's native RoPE is a faithful relative-position encoding *within its trained rotation range* [0, θ_max], where θ_max is the maximum rotation angle seen during pre-training (read from the backbone config; for MiniMax-H3, θ_max = 4096 positions). Under this assumption, the relative rotation angle pos_ref − pos_str exactly encodes the reference-to-stream offset, and a shared origin that is *advanced monotonically and clamped to [0, θ_max]* keeps that offset an invariant of the rollout. The assumption fails when reference positions exceed θ_max, in which case the relative angle reflects out-of-window extrapolation rather than true drift; the method therefore clamps all positions to the trained range and reports the clamp fraction as a diagnostic.

**How it works.**

```pseudocode
# Per streaming block b, at every denoising step, inside the control-LoRA pathway
# Fixed hyperparameters: sink length S=4 tokens; anchor refresh period P=8 blocks;
# block stride Δ = 1 position per block; trained rotation range θ_max = 4096 (from backbone config);
# re-encode temperature tau=1.0 (no softmax sharpening); LoRA rank unchanged.

initialize:
    sink = Encoder(R[0:S])                  # S anchor tokens, frozen after init
    frame_offset = 0                        # shared positional origin, monotone

for each streaming block b = 1..B:
    # 1. Reference-Anchored Positional Re-encoding (RAPR)
    #    Project reference tokens and streaming latents into ONE frame.
    #    The shared origin is advanced by a fixed stride and clamped to the trained range.
    frame_offset = min(frame_offset + Δ, θ_max - max(M, L_z))   # monotone, bounded
    pos_ref  = clamp(phi(r_i) - frame_offset, 0, θ_max)        # reference positions, shifted and clamped
    pos_str  = clamp(phi(z_t) - frame_offset, 0, θ_max)        # streaming positions, same origin and clamp
    # phi = the backbone's native RoPE; subtraction is applied to the rotation
    # angle, i.e. we rotate reference keys/queries by (pos_ref) and streaming
    # keys/queries by (pos_str) using the SAME origin, so relative offset is exact
    # within the trained window. The clamp fraction (fraction of tokens whose
    # unclamped position exceeded θ_max) is logged per block as a diagnostic.

    # 2. Sink re-injection
    #    The sink tokens are re-inserted at the head of the reference segment
    #    with positions pinned to the current origin (pos = frame_offset .. frame_offset+S-1),
    #    so they move with the shared frame rather than staying at absolute zero.
    ref_tokens = concat(sink, R_tokens_at(pos_ref))

    # 3. Control-LoRA attention with shared frame
    attn_out = ControlLoRA_Attn(
        query = z_t rotated by pos_str,
        key   = ref_tokens rotated by their pinned/shifted positions,
        value = ref_tokens)

    # 4. Anchor refresh (bounded drift, not accumulation)
    if b % P == 0:
        sink = EMA(sink, Encoder(z_t[anchor_window]), alpha=0.9)
        # frame_offset is advanced monotonically and clamped; it is NOT reset to zero.
        # The shared origin is re-established each block by the clamp, not by a reset.

    z_{t+1} = z_t + attn_out
```

The key structural difference from the naive control-LoRA pathway is that `frame_offset` is advanced by a fixed stride and clamped to the backbone's trained rotation range at every block (a bounded, monotone re-anchoring, not an unbounded accumulation), and the sink tokens carry positions pinned to the current origin so they act as a fixed landmark against which both the reference segment and the streaming state are measured.

**Why this design.** We chose a bounded, monotone shared origin advanced by a fixed stride and clamped to the trained rotation range rather than a per-block reset to zero, because a reset destroys the reference-to-stream offset (it converts a smooth accumulation into a per-block sawtooth) and because RoPE is only faithful within its trained range; the clamp keeps every rotation angle in-window, accepting the cost that the origin saturates at θ_max − max(M, L_z) and the method must report the clamp fraction. We chose a persistent sink of S=4 pinned tokens over re-encoding the full reference each block because the full reference is expensive and the sink is the minimal sufficient statistic for the anchor, accepting the cost that a very long reference may not be fully captured by four tokens. We chose an EMA anchor refresh with period P=8 over a hard replace because a hard replace reintroduces the very discontinuity the sink is meant to remove, accepting the cost that the sink lags the true anchor by at most P blocks. Unlike SolarRenderer, which places structural tokens into the native reference pathway at described frame positions with no shared origin and no sink, and unlike StreamAvatar, which retains an anchor but does not tie it to the streaming positional frame, RAPR-Sink is not a variant of either: it is the composition of anchor retention and positional re-encoding into a single operator whose invariant is the relative offset within the trained window, not the step count.

**Why it measures what we claim.** The computational quantity `pos_ref - pos_str` (the relative rotation angle between reference and streaming tokens) measures *positional drift* because the assumption that the backbone's RoPE is a faithful relative-position encoding within the trained rotation range [0, θ_max] makes equal rotation angles equivalent to zero relative offset; this assumption fails when the structural reference is placed at described frame positions that exceed θ_max, in which case the relative angle reflects an out-of-window extrapolation rather than true drift, and the clamp fraction diagnostic reports exactly when this occurs. The quantity `sink` (the EMA of the anchor window) measures *anchor retention* because the assumption that the anchor window is representative of the structural reference makes the EMA a sufficient statistic for the anchor; this assumption fails when the reference is multi-modal (e.g., a reference containing a cut), in which case the EMA reflects a blend of two anchors rather than either one. The quantity `frame_offset advance + clamp` measures *drift-free re-anchoring* because the assumption that block boundaries are semantically compatible re-anchoring points makes a bounded monotone advance equivalent to zero accumulated drift within the trained window; this assumption fails when a structural reference spans a block boundary, in which case the advance reflects a local re-alignment rather than a global one.

## Contribution

(1) RAPR-Sink, a positional re-encoding operator plus persistent reference sink that ties structural reference tokens and the streaming latent state to a single reset-per-block positional frame inside the control-LoRA pathway, requiring no new backbone and no new large trainable module. (2) A design principle for long-horizon streaming stability under structural conditioning: stability is governed by the invariance of the reference-to-streaming relative offset, not by the number of denoising steps, so few-step distillation and long-horizon stability are separable concerns. (3) A bounded anchor-refresh schedule (EMA with period P) that converts unbounded drift accumulation into a bounded lag, with the lag/continuity trade-off characterized empirically.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SolarRenderer frame-aligned clips, long-reference subset (20–30 clips, pre-registered block-length sweep: 1k, 5k, 20k blocks) | Long references stress positional drift and anchor retention |
| Primary metric | Structural fidelity vs. reference (relative-offset error), logged per block | Directly reads the quantity the method claims to control |
| Diagnostic metric | Relative rotation angle pos_ref − pos_str and clamp fraction, logged per block | Grounds the claim that relative angle predicts decoherence |
| Baseline | SolarRenderer block-causal renderer | Tests whether shared-origin re-encoding beats native reference placement |
| Baseline | StreamAvatar-style anchor retention | Tests whether tying anchor to positional frame matters |
| Baseline | Naive control-LoRA with running frame_offset (unbounded, no clamp) | Tests whether bounded monotone origin beats unbounded accumulation |
| Ablation-of-ours | RAPR-Sink with sink disabled (S=0) | Isolates the anchor sink from the positional re-encoding |
| Ablation-of-ours | RAPR-Sink with clamp disabled (θ_max = ∞) | Isolates the in-window clamp from the monotone advance |

### Why this setup validates the claim

The central claim is that drift under block-causal attention is a positional-frame problem, and that a bounded, monotone shared origin plus a pinned sink removes it within the trained rotation window. Each baseline isolates one half of that claim. SolarRenderer places structural tokens at described frame positions with no shared origin, so if our gain comes from re-anchoring, the gap should appear exactly where references exceed the native context window. StreamAvatar retains an anchor but does not tie it to the streaming positional frame, so it tests whether anchor retention alone is sufficient — if it were, our method would show parity, falsifying the composition claim. The naive running-offset control-LoRA is the direct counterfactual for the bounded-advance decision: it accumulates without clamp, so its error should grow monotonically while ours saturates at the clamp. The S=0 ablation separates the sink's contribution from the re-encoding's contribution, and the clamp-disabled ablation separates the clamp's contribution from the monotone advance, so a gain cannot be attributed to the wrong operator. Relative-offset error is the right metric because it is the quantity the method manipulates; fidelity metrics that average over all frames would dilute a drift signal that concentrates at long-reference, late-block positions. The per-block logging of pos_ref − pos_str and clamp fraction directly tests the load-bearing assumption: if relative angle predicts structural fidelity, the two curves should co-vary across blocks and baselines.

### Expected outcome and causal chain

**vs. SolarRenderer block-causal renderer** — On a case where the structural reference spans more frames than the native context window, SolarRenderer places reference tokens at described frame positions that extrapolate beyond the trained rotation range, so the relative angle between reference and streaming tokens no longer encodes true offset and structure drifts late in the stream. Our method re-rotates reference keys against a shared origin advanced monotonically and clamped to θ_max, so the relative offset stays in-window; the clamp fraction diagnostic will be near zero for our method and high for SolarRenderer on long references. We expect a visible gap concentrated on long-reference clips and late blocks, with near-parity on short references.

**vs. StreamAvatar-style anchor retention** — On a case where the anchor is retained but the streaming state's positional frame advances independently, the anchor and the stream slowly desynchronize, so the retained anchor points at a structural location the stream has already passed. StreamAvatar produces a correct-looking early stream that degrades once the accumulated offset exceeds the anchor's tolerance. Our method pins the sink to the current origin and measures both segments against it, so the anchor stays a fixed landmark. We expect parity early and a growing gap as block count increases.

**vs. Naive control-LoRA with running frame_offset** — On a case where block boundaries are crossed repeatedly, the running offset accumulates rounding and re-encoding error, so the reference segment is progressively rotated away from the stream. This baseline produces drift that is monotone in block index. Our bounded, monotone advance with clamp saturates the origin at θ_max − max(M, L_z), bounding error rather than accumulating it. We expect our error curve to flatten once the clamp engages while the baseline's rises, with the gap widening monotonically up to the saturation point.

### What would falsify this idea

If our gain over SolarRenderer is uniform across short and long references rather than concentrated on long-reference, late-block clips, the drift explanation is wrong. If the S=0 ablation matches the full method, the sink is not doing the work we claim and the composition argument collapses. If the clamp-disabled ablation matches the full method, the in-window clamp is not doing the work we claim.

## References

1. SolarRenderer: Few-Step Generative Reference Rendering
2. Video-As-Prompt: Unified Semantic Control for Video Generation
3. Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield
