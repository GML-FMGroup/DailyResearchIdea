# Order-Invariant Right-Context Aggregation for Mixture-of-Experts Diffusion Language Models

## Motivation

Clover-35B-A3B-Preview conditions block-diffusion denoising on a zero-initialized right-context convolution branch and a Gated Delta Network recurrent memory, but the abstract reports no property establishing that the generated output is invariant to the order in which right-context information becomes available during iterative denoising; zero-initialization is treated as sufficient to preserve the denoising fixed point without evidence. This is a structural gap that recurs across conditional diffusion language models that inject partial context through a convolutional or recurrent channel: the conditioning pathway is order-sensitive by construction, so the fixed point of the denoising process is not guaranteed to be unique or consistent across denoising trajectories. The same meta-level problem appears in OLMoE (venue:iclr:2025:8e938be1974a), where sparse MoE routing is analyzed for specialization but not for invariance of the conditioning signal to the order in which expert information is aggregated.

## Key Insight

A right-context integration mechanism whose output is a permutation-invariant function of the set of available right-context tokens (rather than of their arrival sequence) makes the denoising fixed point independent of the order in which context becomes available, so the model's conditional distribution at convergence is uniquely determined by the context set rather than by the denoising trajectory.

## Method

**What it is.** We introduce ORCA (Order-invariant Right-Context Aggregation), a drop-in replacement for Clover's zero-initialized right-context convolution branch. ORCA takes as input the set of right-context *chunks* available at denoising step t (a subset S_t of the block's right-context chunks, where each chunk is a contiguous span of tokens — e.g., a retrieved file, a tool-call result, or a document — and order *within* each chunk is preserved by a fixed positional encoding) and returns a fixed-dimensional right-context summary r_t that is unchanged by any permutation of the chunks in S_t, plus a scalar consistency residual delta_t measuring how far the current denoising state is from the order-invariant fixed point. **Load-bearing assumption (stated explicitly):** the invariance claim applies only to the set-valued axis — the order in which *chunks* are supplied — not to token order within a chunk; token order within each chunk carries syntax and execution semantics and is preserved by positional encoding. Permuting chunks is meaning-preserving because chunks are independent retrieved artifacts or tool outputs whose linear concatenation order is an artifact of the retrieval pipeline, not of the task semantics.

**How it works.**

```pseudocode
# Inputs per denoising step t:
#   h_t        : current block hidden states (B, L, D)
#   S_t        : set of available right-context chunks (B, K_t, C, D), K_t chunks of length C
#   m_t        : availability mask for S_t (B, K_t)
#   pos        : fixed sinusoidal positional encoding within each chunk (C, D)
# Outputs:
#   r_t        : order-invariant right-context summary (B, L, D)
#   delta_t    : fixed-point consistency residual (B,)

1. Encode each chunk independently with intra-chunk positional encoding:
     for each chunk k: v_k = mean_{c=1..C} ( W_chunk @ (S_t[:, k, c, :] + pos[c]) + b_chunk )   # (B, D), shared W_chunk
     u_k = v_k                                                                                    # (B, K_t, D)

2. Permutation-invariant pooling over available chunks:
     w_k = softmax_k( (W_score @ u_k) * m_t[:, k] )   # masked softmax over K_t, (B, K_t)
     p   = sum_k w_k * u_k                            # (B, D) set-pooled summary

3. Broadcast and fuse with block hidden states via a gated residual:
     g   = sigmoid(W_gate @ [h_t ; p_broadcast])      # (B, L, D)
     r_t = g * (W_out @ p_broadcast) + (1 - g) * h_t  # (B, L, D)

4. Fixed-point consistency check (exact group average over sampled permutations):
     # Draw P = 8 random permutations pi_1..pi_P of S_t (or all K_t! if K_t <= 4)
     r_t_pi^(j) = ORCA(h_t, permute_chunks(S_t, pi_j), m_t)   for j = 1..P
     r_bar = (1/P) sum_j r_t_pi^(j)
     delta_t = || r_t - r_bar ||_2 / (|| r_t ||_2 + eps)   # relative residual

5. If delta_t > tau (tau = 1e-3), apply one consistency-projection step:
     r_t <- 0.5 * (r_t + r_bar)   # symmetrize; repeat up to 2 times

6. Return r_t, delta_t
```

**Why this design.** We chose permutation-invariant masked softmax pooling over a convolutional branch because a convolution's output depends on the spatial ordering of its inputs, so any finite-width convolutional kernel applied to a variable-length right-context sequence is order-sensitive by construction — the structural reason Clover's zero-initialized branch cannot guarantee order invariance. We chose a per-chunk shared projection followed by a set-pooling operation over a recurrent aggregation because a Gated Delta Network's recurrent state is a function of the arrival sequence, so reusing it for right-context integration would reintroduce order dependence; the cost is that we lose the recurrent memory's ability to weight tokens by recency, which we accept because right-context chunks are unordered within a block. We chose an explicit consistency residual with a symmetrization projection over a learned invariance penalty because a penalty term only encourages invariance in expectation and cannot certify it at inference; the cost is an extra forward pass through ORCA at each denoising step (approximately 1.3x the branch's FLOPs), which we accept because the residual also serves as a diagnostic and a stopping signal. We chose a gated residual fusion over additive injection because the gate lets the model learn when right-context information is informative versus when the block hidden state should dominate, and the trade-off is that the gate adds parameters that must be trained jointly with the MoE backbone. A domain expert would not describe ORCA as a variant of the Clover right-context convolution because the convolution's output is a function of the ordered sequence, whereas ORCA's output is a function of the multiset of chunks; the consistency residual is not present in any convolution-based conditioning pathway we are aware of.

**Why it measures what we claim.** The relative residual delta_t measures order-invariance of the right-context summary because it compares the branch output under P=8 permutations of the same available chunk set; this equivalence holds under the assumption that the only source of order dependence in the branch is the ordering of S_t, which fails when the block hidden states h_t themselves carry order-dependent information from previous denoising steps, in which case delta_t reflects residual order dependence in h_t rather than in the right-context branch alone. The masked softmax weight w_k measures the salience of right-context chunk k because softmax normalization over the available set assigns higher weight to chunks with larger score projections; this holds under the assumption that the score projection is trained to correlate with downstream denoising loss, which fails when the score projection collapses to a constant (all weights equal), in which case w_k reflects uniform averaging rather than salience. The symmetrization step measures the distance from the order-invariant fixed point because averaging r_t and r_bar projects onto the permutation-invariant subspace; this holds under the assumption that the permutation group acting on S_t is finite and the averaging is exact, which fails when K_t is large and P=8 does not cover the group, in which case the projection is approximate and delta_t underestimates the true residual.

## Contribution

['ORCA, an order-invariant right-context aggregation mechanism for mixture-of-experts diffusion language models that replaces a zero-initialized convolutional branch with permutation-invariant set pooling plus a fixed-point consistency residual, making the denoising fixed point independent of the order in which right-context information becomes available.', "A design principle: order invariance of a conditioning pathway in iterative denoising is a property of the aggregation function's symmetry group, not of its initialization, so zero-initialization is neither necessary nor sufficient for fixed-point consistency.", 'A consistency-projection procedure that symmetrizes the right-context summary at inference when the residual exceeds a threshold, providing a test-time mechanism for enforcing order invariance without retraining the backbone.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-bench Verified + Terminal-Bench 2.1 agent tasks, with chunk-order-shuffled variants | Clover's own agent benchmarks; stresses chunk-set conditioning |
| Primary metric | Pass@1 under shuffled *chunk* order (within-chunk token order preserved) | Directly probes chunk-order-invariance claim |
| Baseline | Clover right-context convolution branch | The ordered mechanism ORCA replaces |
| Baseline | Additive-injection right-context fusion | Tests whether gating, not just invariance, matters |
| Baseline | Learned invariance-penalty pooling | Tests whether certified residual beats soft penalty |
| Ablation-of-ours | ORCA without consistency-residual projection | Isolates contribution of symmetrization step |

### Why this setup validates the claim

The central claim is that order-invariant set-pooling over right-context *chunks*, with a certified consistency residual, outperforms an ordered convolutional branch on agent tasks where right-context chunk availability is variable and unordered within a block. The dataset choice matters because SWE-bench Verified and Terminal-Bench 2.1 exercises exactly the regime Clover targets: multi-step agentic generation where right-context chunks (retrieved files, tool outputs) arrive as a set whose ordering carries no semantic meaning, while token order within each chunk is preserved. The primary metric — Pass@1 under shuffled chunk order — is the right detector because it isolates the predicted effect: if ORCA is genuinely chunk-order-invariant, shuffling S_t should leave performance essentially unchanged, whereas an ordered convolution should degrade. The Clover convolution baseline tests the core sub-claim (invariance beats ordering); the additive-injection baseline tests whether the gated fusion contributes beyond invariance; the invariance-penalty baseline tests whether the certified residual beats a soft penalty. The ablation isolates the symmetrization projection, so a gain that survives its removal is attributable to pooling, not to the projection step. Together these form a falsifiable contrast: the claim predicts a specific interaction (gap concentrated on chunk-order-shuffled subsets), not a uniform lift.

### Expected outcome and causal chain

**vs. Clover right-context convolution branch** — On a case where the block's right-context chunks arrive in a different order than training (e.g., a retrieved-file context whose chunk sequence is permuted by the retrieval pipeline), the baseline produces a shifted right-context summary because its convolutional kernel is a function of spatial position, so permutation changes the output. Our method instead pools over the multiset of chunks via masked softmax, so permutation leaves r_t unchanged, and we expect a noticeable Pass@1 gap on chunk-order-shuffled subsets but near-parity on canonical-order subsets — the signature of an invariance effect rather than a general capacity gain.

**vs. Additive-injection right-context fusion** — On a case where right-context is uninformative or partially missing (K_t small, mask sparse), the additive baseline produces a fixed-strength injection that dilutes the block hidden state because it has no mechanism to down-weight context. Our method instead lets the gate g suppress the pooled summary when context is weak, so we expect fewer regressions on low-K_t subsets and a larger relative gain there than on full-context subsets, where both methods should be close.

**vs. Learned invariance-penalty pooling** — On a case where the score projection is under-trained or collapses toward uniform weights, the penalty baseline produces a summary whose invariance is only encouraged in expectation, so residual order dependence persists at inference. Our method instead certifies invariance via the symmetrization projection and exposes delta_t, so we expect lower delta_t and a smaller shuffle-induced performance drop, with the gap widest on tasks where the penalty baseline's weights are most uniform.

### What would falsify this idea

If the gain over the Clover convolution baseline is uniform across shuffled and canonical-order subsets rather than concentrated on the chunk-order-shuffled subsets, the order-invariance mechanism is not what drives the improvement and the central claim is wrong. Likewise, if removing the symmetrization projection (the ablation) leaves shuffle-robustness unchanged, the certified-residual component is not doing the work we attribute to it.

## References

1. Clover-35B-A3B-Preview Technical Report
