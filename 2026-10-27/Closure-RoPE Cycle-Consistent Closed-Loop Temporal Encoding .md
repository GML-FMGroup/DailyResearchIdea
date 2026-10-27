# Closure-RoPE: Cycle-Consistent Closed-Loop Temporal Encoding for Interleaved Control in Interactive Video World Models

## Motivation

Interactive video world models built on In-Context Conditioning, such as World-in-Context, encode temporal position with IT-RoPE by accumulating control transitions across execution intervals. Because the accumulation is open-loop, the positional phase drifts monotonically with the number and length of interleaved control segments, so the model's implied world state after returning to a previously visited control configuration is not the state it produced before — temporal closure fails under arbitrary variable-length interleaving. This is the shared residual gap across the interactive-generation branch: conditioning is treated as static token context rather than as a stateful transition operator with a well-defined return map, so no existing method can guarantee that a cycle of controls returns the latent world state to its origin.

## Key Insight

Temporal consistency under arbitrary interleaving is a closure property of the control-to-state transition operator, so enforcing it as a cycle-consistency invariant on latent transitions — combined with a positional encoding that is normalized to a bounded phase rather than accumulated — makes world-state return exact by construction instead of hoping drift stays small.

## Method

**What it is.** Closure-RoPE is a drop-in replacement for IT-RoPE's accumulated temporal parameterization plus an accompanying training objective. Its inputs are the interleaved control-token and video-token sequence produced by an In-Context Conditioning backbone (World-in-Context-5B); its outputs are per-token temporal phases that are closed under control cycles, and a latent transition operator T whose composition around any closed control loop is trained to be the identity. **Load-bearing assumption (stated explicitly):** the cycle residual computed by rolling out T around a closed control loop measures the full backbone's temporal closure only if T is the dominant carrier of the world-state update—i.e., backbone attention contributes negligibly to the per-token state change relative to T. This assumption fails when attention dominates the state update, in which case the residual reflects only T's local dynamics, not the full model's closure. We make this assumption testable by (i) routing the decoded latent through T (see Phase 2) and (ii) adding a decoding-consistency term to the cycle loss (see Phase 3), so that a trivially contractive T cannot satisfy the objective without also making the decoded frames match.

**How it works.**

```pseudocode
# Phase 1: closed-loop positional encoding (replaces IT-RoPE accumulation)
input: token stream x_1..x_N, control boundaries b_1..b_K (variable lengths)
for each segment k between boundaries b_k, b_{k+1}:
    s_k = segment length
    phi_k = 2*pi * ( (tau_k mod L) / L )        # L = 4096 phase period, tau_k = cumulative control time
    for token i in segment k:
        phase_i = phi_k + 2*pi * (i - b_k) / s_k   # normalize WITHIN segment, never accumulate
        attn_bias_ij = cos(phase_i - phase_j)       # relative, bounded in [-1,1]
# Phase 2: stateful transition operator (composed with backbone attention)
h_attn_i = backbone_attention(x_i, attn_bias)   # attention output per token
h_i = LayerNorm(h_attn_i + T(h_{i-1}, control_token_{i-1}))  # T is a GRU cell, hidden=1280, shared across segments; h_i is the tensor decoded to video
g = T(h_i, control_token_i)                     # predicted next latent state (used in h_{i+1})
# Phase 3: cycle-consistency loss on latent transitions + decoding consistency
sample a closed control cycle C = (c_1..c_m) with sum of segment lengths < L
roll out T around C starting from h_start -> h_return
L_cycle = || h_return - h_start ||^2 / ||h_start||^2
L_decode = || Decode(h_return) - Decode(h_start) ||_1   # Decode is the backbone's video decoder (frozen during cycle-loss training)
L_total = L_video + 0.1 * L_cycle + 0.1 * L_decode      # L_video = flow-matching / diffusion loss of backbone
```

Hyperparameters: phase period L=4096 (matches max training sequence length, so no wrap occurs inside a clip); cycle weight 0.1; decode-consistency weight 0.1; cycle length m sampled uniformly in [2,8] segments; GRU hidden width matched to backbone token width (1280 for the 5B model); LayerNorm applied after residual addition; decoder frozen during cycle-loss training to avoid co-adaptation.

**Why this design.** We chose a *bounded, segment-normalized* phase over IT-RoPE's accumulated phase because accumulation is the direct cause of drift — the relative bias between two tokens then depends on how many control segments happened to precede them, not on their true temporal separation — accepting the cost that absolute long-horizon timestamps are no longer representable and must be recovered from the GRU state instead. We chose an explicit recurrent transition operator T over leaving state implicit in attention because closure is a property of a *map* and can only be regularized if the map is named; the cost is one extra GRU pass per token at training time. We chose a cycle-consistency loss over per-step latent smoothness because smoothness is satisfied by a constant-state collapse while closure is not; the cost is that cycle sampling must cover the control distribution, which we address by sampling cycles from the same control prior used in training. A domain expert would not describe this as "IT-RoPE + a consistency loss": IT-RoPE's failure is structural (open-loop phase), and no auxiliary loss on top of a drifting parameterization can restore closure — the parameterization itself must be made closed. Nor is it the standard "learned disentanglement head" pattern: we do not assert that control and content are independent; we assert only that the transition operator composes to identity on closed loops, which is a property of the operator, not of the data's independence structure. The addition of the decoding-consistency term and the residual composition of T with attention is a specification of the original claim that T is the state-update path: it forces the cycle residual to be computed on the same tensor that is decoded to video, so a trivially contractive T cannot satisfy the objective without also making the decoded frames match.

**Why it measures what we claim.** The normalized cycle residual `||h_return - h_start||^2 / ||h_start||^2` measures *temporal closure of the world-state transition* because we assume the GRU transition T is a faithful summary of how controls move the latent state — i.e., that h_return is computed by the same operator applied during generation; this assumption fails when the backbone's attention, not T, carries most of the state update, in which case the residual reflects only the GRU's local dynamics and not the full model's closure. The segment-normalized phase `phi_k + 2*pi*(i-b_k)/s_k` measures *bounded temporal separation* because we assume relative phase within a segment is the only temporal signal the attention needs; this fails for tasks requiring absolute long-horizon ordering, where the metric reflects within-segment ordering only. The cycle-sampling distribution measures *coverage of the control manifold* because we assume cycles drawn from the training control prior are representative of test-time interleavings; this fails for out-of-distribution control sequences, in which case the loss reflects in-distribution closure only. The decoding-consistency term `||Decode(h_return) - Decode(h_start)||_1` measures *perceptual closure* because we assume the frozen decoder maps latent states to frames faithfully; this fails when the decoder is insensitive to the latent differences that matter for closure, in which case the term reflects only decoder-invariant features. Each quantity is thus tied to a named assumption with a stated failure mode, rather than being presented as closure by fiat.

## Contribution

(1) A closed-loop temporal positional encoding for interleaved control-video token streams that replaces accumulated phase with a bounded, segment-normalized phase and a shared recurrent transition operator, making world-state return well-defined under arbitrary interleaving lengths. (2) A cycle-consistency training objective on latent state transitions that operationalizes temporal closure as a measurable residual, together with a cycle-sampling procedure drawn from the control prior. (3) An empirical design principle: closure is a property of the transition operator's parameterization, not of an auxiliary loss, so the positional encoding and the operator must be changed together — demonstrated by ablating each independently.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WBench navigation subset | Interactive control cycles with ground-truth closure |
| Primary metric | Cycle-residual vs. control-loop return error | Directly measures closure, not perceptual quality |
| Baseline | 22B Ec model | Strongest open-loop interactive baseline to beat |
| Baseline | World-in-Context-5B + IT-RoPE | Isolates the parameterization change from the backbone |
| Baseline | World-in-Context-5B + smoothness loss | Tests cycle-consistency against the collapse-prone alternative |
| Ablation-of-ours | Closure-RoPE without L_cycle | Tests whether the loss, not just the phase, drives closure |
| Ablation-of-ours | Closure-RoPE with learned linear return map instead of GRU | Isolates recurrent operator vs. bounded phase |
| Ablation-of-ours | Closure-RoPE with learned per-segment embedding instead of segment-normalized phase | Isolates bounded phase vs. recurrent operator |
| Diagnostic | Ablate GRU at inference (set T to identity) and report cycle residual | Measures how much state update is carried by attention vs. GRU |

### Why this setup validates the claim

The central claim is that IT-RoPE's drift is structural and that closure requires a closed parameterization plus a named transition operator. WBench navigation gives control cycles with a well-defined return state, so the return error is a ground-truth-anchored closure probe rather than a proxy. The 22B Ec baseline tests the sub-claim that scale alone does not buy closure: if a much larger open-loop model still drifts on long loops, the failure is parameterization, not capacity. World-in-Context-5B + IT-RoPE isolates the phase change at fixed backbone, so any gap is attributable to Closure-RoPE rather than to the 5B model. The smoothness-loss baseline tests the sub-claim that closure is not implied by local latent regularity, since a constant-state collapse satisfies smoothness but not cycle return. The no-L_cycle ablation tests whether the GRU operator's closure needs the explicit cycle objective or emerges from the phase alone. The linear-return-map and per-segment-embedding ablations isolate whether closure arises from the bounded phase, the recurrent operator, or their interaction. The GRU-ablation diagnostic directly tests the load-bearing assumption that T carries the state update: if removing T at inference leaves the cycle residual unchanged, attention dominates and the metric reflects only local GRU dynamics. The primary metric is the right detector because it is computed on the same tensor that is decoded to video (h_i), so it cannot be satisfied by a model that merely looks temporally coherent.

### Expected outcome and causal chain

**vs. 22B Ec model** — On a case where a navigation loop revisits a room after six control segments, the baseline produces a drifted world state because accumulated IT-RoPE phase makes relative bias depend on segment count, and scale does not remove the accumulation. Our method instead normalizes phase within each segment and rolls the GRU transition around the loop, so we expect a noticeable gap on long-cycle subsets (m≥4) but parity on short single-segment clips, with the 5B model matching or beating the 22B on return error for long loops.

**vs. World-in-Context-5B + IT-RoPE** — On a case where two control segments of unequal length precede a return, the baseline produces a phase offset proportional to the accumulated segment lengths because IT-RoPE carries absolute accumulated time into the relative bias. Our method instead computes relative phase from bounded within-segment normalization, so we expect the cycle residual to stay roughly flat as segment count grows while the IT-RoPE baseline's residual grows with loop length, producing a clear separation concentrated on variable-length loops.

**vs. World-in-Context-5B + smoothness loss** — On a case where the control sequence returns to the start state, the baseline produces a latent trajectory that is locally smooth but ends far from the start, because smoothness is satisfied by a slow constant-state collapse and imposes no return constraint. Our method instead penalizes the composed transition around the closed loop, so we expect the smoothness baseline to show low per-step latent change but high cycle residual, while ours shows both low, with the gap largest on loops whose controls are drawn from the training prior.

**vs. Closure-RoPE without L_cycle** — On a case where the control sequence returns to the start state, the no-L_cycle ablation produces a latent trajectory that drifts because the GRU transition is not explicitly trained to compose to identity around closed loops. Our method instead adds the cycle-consistency loss, so we expect the no-L_cycle ablation to show higher cycle residual than full Closure-RoPE, with the gap largest on long, variable-length loops.

**vs. Closure-RoPE with learned linear return map** — On a case where the control sequence returns to the start state, the linear return map may fail to capture nonlinear state transitions, leading to higher cycle residual than the GRU-based operator. Our method uses a GRU, so we expect the linear-map ablation to show higher cycle residual, especially on loops with complex control sequences.

**vs. Closure-RoPE with learned per-segment embedding** — On a case where the control sequence returns to the start state, the learned per-segment embedding may not provide bounded temporal separation, leading to drift similar to IT-RoPE. Our method uses segment-normalized phase, so we expect the per-segment-embedding ablation to show higher cycle residual, with the gap increasing with segment count.

**vs. GRU-ablation diagnostic** — On a case where the control sequence returns to the start state, if attention dominates the state update, setting T to identity at inference will leave the cycle residual unchanged. Our method composes T with attention, so we expect the GRU-ablation diagnostic to show a significant increase in cycle residual when T is removed, confirming that T carries a substantial portion of the state update.

### What would falsify this idea

If the gain over IT-RoPE is uniform across short and long cycles rather than concentrated on long, variable-length loops, the drift mechanism we attribute to accumulation is not the operative cause. If the no-L_cycle ablation matches full Closure-RoPE on cycle residual, the closure is coming from the phase alone and the transition-operator claim is wrong. If the GRU-ablation diagnostic shows no change in cycle residual when T is removed, the load-bearing assumption that T carries the state update is false and the metric reflects only local GRU dynamics.

## References

1. World-in-Context: Interactive Video World Modeling with In-Context Conditioning
