# Carry-Over-Bounded Consistency Between Differentiable Rate Proxies and Parallel rANS Code Lengths in Fixed-Step Tree Codecs

## Motivation

Learned image compression systems optimize a differentiable rate proxy (per-symbol conditional cross-entropy) but deploy a non-differentiable entropy coder, and the gap between the two is treated as negligible even though it is the quantity that determines whether training-time rate improvements survive deployment. This surrogate-to-coder gap is structural across the field: balanced R-D optimization (E3/E8) tunes gradient balance on the proxy, MambaIC (E9) optimizes context-modeling cross-entropy, P-LLM (E10) optimizes next-pixel log-likelihood, and FMT (E4/E11) reports realized rANS sizes but does not characterize how far the proxy can drift from the realized bit count. In FMT specifically the proxy is computed per tree step while the realized code length is accumulated across parallel rANS lanes whose states carry over between steps, so the gap is not random noise but a lane-state carry-over term that no existing work isolates or measures.

## Key Insight

In a fixed-step coarse-to-fine tree codec with parallel rANS lanes, the realized code length differs from the sum of per-step conditional cross-entropies by a carry-over residual that is determined by lane state transitions at step boundaries, so the residual can be measured per step boundary and bounded as a function of lane count, alphabet size, and resolution rather than benchmarked end-to-end.

## Method

```pseudocode
# CARRY-BOUND: Carry-Over Residual Bounding for Fixed-Step Tree Codecs
# Inputs: FMT-style tree codec (coarse-to-fine, S steps, L parallel rANS lanes),
#         image x of resolution HxW, trained network f_theta (or no-network rule)
# Outputs: per-step proxy CE, realized rANS bits, per-boundary carry-over residual,
#          empirical residual envelope vs (S, L, H, W, GPU arch)
#
# Load-bearing assumption (stated explicitly): The realized rANS code length minus the sum of
# per-step conditional cross-entropies is dominated by a lane-state carry-over term that is
# separable at step boundaries and additive/separable in (S, L, H, W, arch), so that a
# boundary-decomposed residual isolates a real mechanism rather than renormalization and
# quantization noise. This assumption is verified in step 0 (synthetic ground-truth) and
# calibrated in step 2b (quantization-adjusted residual).

0. Synthetic ground-truth validation (before real images)
   # Generate a known symbol stream with analytically computable rANS code length.
   for trial in 1..100:
       symbols = sample from a known distribution (e.g., uniform over alphabet size A in {4,16,256})
       # Compute exact rANS code length analytically for given L lanes and S steps:
       #   exact_bits = sum over symbols of -log2 p(symbol) + renormalization overhead
       #   renormalization overhead = sum over lanes of (bytes emitted to keep state in [2^23, 2^31))
       exact_bits = analytic_rans_length(symbols, L, S, A)
       # Run the instrumented encoder on the same symbols:
       measured_bits, proxy_bits, carry_residual, delta_state = instrumented_encode(symbols, L, S)
       # Verify: measured_bits == exact_bits (bit-exact), and carry_residual == renormalization overhead
       assert measured_bits == exact_bits
       assert carry_residual == renormalization_overhead
   # Report max absolute error between measured and analytic; must be 0 for bit-exact codecs.

1. Encode with instrumentation
   for step s in 1..S:
       latents_s = f_theta(context_s)            # conditional distribution per node
       # Quantize the conditional distribution exactly as the coder does:
       p_quant_s = quantize_probabilities(latents_s, precision=16-bit fixed-point)  # matches rANS input
       proxy_bits[s] = sum_nodes -log2 p_quant_s(node | context_s)   # proxy on quantized distribution
       state_before[s] = snapshot(rANS_lane_states)   # L lane states, 32-bit each
       rans_encode_step(latents_s, lanes=state_before[s])
       state_after[s] = snapshot(rANS_lane_states)
       realized_bits[s] = bits_emitted_between(state_before[s], state_after[s])
       carry_residual[s] = realized_bits[s] - proxy_bits[s]

2. Per-boundary residual decomposition
   for each boundary b between step s and s+1:
       # Direct renormalization-emission count per lane (deterministic function of state and alphabet):
       renorm_bits[b] = sum over lanes of renormalization_emissions(state_after[s], state_before[s+1], alphabet_size)
       # Quantization-induced gap (proxy computed on exact quantized probabilities):
       quant_gap[b] = proxy_bits[s+1] - proxy_bits_on_continuous[s+1]   # if needed, else 0
       # Carry-over residual is the renormalization term (not XOR magnitude):
       boundary_residual[b] = renorm_bits[b] - quant_gap[b]
       # delta_state is retained as a diagnostic but not as the primary measure:
       delta_state[b] = state_after[s] XOR state_before[s+1]   # reported for reference only

2b. Quantization-adjusted residual (calibration)
   # Recompute proxy using the exact quantized symbol probabilities the coder uses:
   proxy_bits_quant[s] = sum_nodes -log2 p_quant_s(node | context_s)
   # Then carry_residual_adj[s] = realized_bits[s] - proxy_bits_quant[s]
   # This isolates renormalization from quantization error.

3. Residual envelope fit (empirical, not a proof)
   # Factorial subset (power analysis: 80% power to detect 10% change in residual, alpha=0.05):
   sweep S in {4,8}, L in {8,64}, resolution in {256,512,1024}, GPU arch in {A100, RTX4090}
   # 2*2*3*2 = 24 configurations, 5 images each = 120 runs.
   fit residual_envelope(S,L,H,W,arch) = a*S + b*L + c*log2(H*W) + d*arch_offset
   report R^2 and max |carry_residual_adj| / proxy_bits_quant per configuration

4. Consistency check
   total_proxy = sum_s proxy_bits_quant[s]
   total_realized = sum_s realized_bits[s]
   gap_ratio = (total_realized - total_proxy) / total_proxy
   report gap_ratio per (S,L,resolution,arch); flag configurations where
   gap_ratio exceeds the fitted envelope by > 2 sigma
```

**What it is.** CARRY-BOUND is an instrumented encoder/decoder harness for FMT-style fixed-step tree codecs that records, at every step boundary, the rANS lane-state transition and the difference between per-step conditional cross-entropy (the differentiable rate proxy) and the bits actually emitted by the parallel rANS coder. Its inputs are a trained tree codec and an image; its outputs are per-boundary carry-over residuals, a fitted residual envelope as a function of step count, lane count, resolution, and GPU architecture, and a flag for configurations where the realized code length departs from the proxy beyond the fitted envelope. The load-bearing assumption is stated explicitly in the pseudocode header: the residual is dominated by lane-state carry-over, separable at boundaries, and additive in (S, L, H, W, arch). This assumption is verified in step 0 (synthetic ground-truth) and calibrated in step 2b (quantization-adjusted residual).

**Why this design.** We chose to instrument the encoder at step boundaries rather than compare only end-to-end bit counts because the end-to-end gap conflates lane-state carry-over with arithmetic-coder renormalization noise and with network prediction error, and only the boundary decomposition isolates the term that the motivation names; the cost is that we must snapshot lane states, which adds memory overhead during measurement but not during deployment. We chose a fixed-step tree codec with parallel rANS lanes over a sequential autoregressive coder because FMT's design (E4/E11) already exposes step boundaries as first-class objects, so the residual is measurable without modifying the codec's decoding order; the cost is that the envelope is specific to tree-structured codecs and does not transfer to MambaIC-style (E9) or P-LLM-style (E10) coders without re-derivation. We chose an empirical envelope fit over a closed-form bound because a closed-form bound would require assumptions about lane-state reachability that are not verifiable on real GPU rANS implementations, and an unverifiable bound is worse than a measured envelope; the cost is that the envelope is hardware- and resolution-scoped rather than universal. A domain expert would not describe CARRY-BOUND as a variant of FMT because FMT's contribution is the codec itself, whereas CARRY-BOUND's contribution is the measurement and envelope of the surrogate-to-coder residual that FMT's design makes observable; the codec is the instrument, not the claim.

**Why it measures what we claim.** The computational quantity `carry_residual[s] = realized_bits[s] - proxy_bits[s]` measures the surrogate-to-coder consistency gap because the named assumption is that the rANS coder is bit-exact and the per-step conditional cross-entropy is computed from the same conditional distribution that the coder uses, so any difference between the two is attributable to lane-state carry-over and renormalization; this assumption fails when the network's conditional distribution is quantized or clipped before entropy coding, in which case the residual reflects quantization error rather than carry-over and must be reported separately (step 2b). The computational quantity `renorm_bits[b]` measures the carry-over contribution because the named assumption is that lane states evolve as a deterministic function of the symbols encoded between boundaries, so the exact number of bits the coder must emit to renormalize each lane given its state is a deterministic function of the rANS state and alphabet; this assumption fails when the rANS implementation uses a renormalization schedule that depends on lane index in a non-uniform way, in which case renorm_bits reflects lane-index scheduling rather than carry-over and the envelope must be conditioned on lane index. The fitted envelope `residual_envelope(S,L,H,W,arch)` measures the claim that the gap is bounded by a function of step count, lane count, resolution, and architecture because the named assumption is that the residual is additive across boundaries and separable in these factors; this assumption fails when resolution changes the number of tree nodes per step non-uniformly, in which case the envelope reflects node-count scaling rather than resolution per se and must be refit per tree depth.

## Contribution

['CARRY-BOUND, an instrumented encoder/decoder harness for fixed-step tree codecs that decomposes the surrogate-to-coder code-length gap into per-step-boundary carry-over residuals and records lane-state transitions as first-class measurements.', 'An empirical residual envelope that characterizes how the gap between per-step conditional cross-entropy and realized parallel rANS bits scales with step count, lane count, image resolution, and GPU architecture, together with a consistency flag for configurations that exceed the envelope.', 'A measurement protocol and released instrumentation for comparing differentiable rate proxies against deployed entropy-coder bit counts in learned image compression, applicable to any fixed-step tree codec.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | High-resolution image benchmarks (Kodak, DIV2K, CLIC) | Spans resolutions where node counts per step vary |
| Primary metric | Boundary-decomposed carry-over residual (renorm_bits - quant_gap) vs proxy CE on quantized distribution | Isolates surrogate-to-coder gap from network error |
| Baseline | End-to-end bit-count gap (total realized − total proxy) | Tests whether boundary decomposition adds diagnostic value |
| Baseline | FMT no-network rule codec | Tests envelope when conditional distribution is trivial |
| Baseline | FMT large-network codec | Tests envelope under realistic learned conditionals |
| Ablation-of-ours | Single-lane (L=1) sequential rANS variant | Tests whether carry-over residual scales with lane count |
| Validation | Synthetic ground-truth symbol streams (step 0) | Verifies analytic decomposition before real images |

### Why this setup validates the claim

The central claim is that the surrogate-to-coder gap in FMT-style fixed-step tree codecs is decomposable at step boundaries and bounded by a measurable envelope in (S, L, H, W, arch). The synthetic ground-truth validation (step 0) is the crucial control: if the measured carry_residual and renorm_bits do not match the analytic decomposition on known symbol streams, the harness is not measuring what it claims. The end-to-end bit-count baseline tests whether boundary decomposition adds diagnostic value beyond the aggregate gap. The no-network FMT rule and large-network FMT codecs bracket the conditional-distribution regime — the rule gives near-deterministic conditionals where quantization error is small, while the large network exercises realistic learned conditionals where the residual should be largest if the carry-over mechanism is real. The L=1 ablation is the direct test of the lane-count term: if the envelope's dependence on L is an artifact of lane scheduling rather than genuine state carry-over, collapsing to one lane should not change the residual structure in the predicted way. The boundary-decomposed residual is the right primary metric because it is the only quantity that separates lane-state carry-over from renormalization noise and prediction error, which the end-to-end gap conflates.

### Expected outcome and causal chain

**vs. End-to-end bit-count gap** — On a high-resolution image where per-step node counts vary sharply across tree depth, the baseline produces a single aggregate gap that mixes lane carry-over with renormalization and prediction error, because it has no boundary resolution. Our method instead attributes the gap to individual step boundaries via renorm_bits and per-boundary residual, because it snapshots lane states at each boundary and computes exact renormalization emissions. We expect the boundary-resolved residual to concentrate at a few boundaries rather than spread uniformly, revealing structure the aggregate gap hides.

**vs. FMT no-network rule codec** — On a smooth image where the rule's conditionals are near-deterministic, the baseline produces a very small realized-vs-proxy gap because there is little for lane carry-over to accumulate. Our method instead predicts a near-zero carry-over residual envelope for this regime, because the deterministic conditionals leave little state-dependent variation between boundaries. We expect the fitted envelope to be flat in S and L here, confirming that the residual tracks conditional uncertainty rather than codec mechanics.

**vs. FMT large-network codec** — On a textured image where the network's conditionals are broad and quantized before coding, the baseline produces a larger and more variable realized-vs-proxy gap, because quantization error and lane carry-over both inflate realized bits. Our method instead separates the quantization component (reported separately per the stated failure condition) from the carry-over component via renorm_bits, because the boundary decomposition isolates the state-transition term. We expect the carry-over residual to grow with L and S while the quantization term stays roughly fixed, matching the envelope's predicted scaling.

### What would falsify this idea

If the synthetic ground-truth validation (step 0) fails to match the analytic decomposition, or if the boundary-resolved carry-over residual fails to concentrate at specific boundaries and instead tracks the aggregate end-to-end gap uniformly across all configurations — or if the L=1 ablation shows the same envelope dependence on L as the multi-lane case — then the claimed lane carry-over mechanism is not what the residual measures, and the decomposition adds no diagnostic value over the end-to-end gap.

## References

1. FMT: Lossless Image Compression in a Fixed Number of Parallel Steps
