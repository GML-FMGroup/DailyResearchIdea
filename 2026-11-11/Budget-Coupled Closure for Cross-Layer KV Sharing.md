# Budget-Coupled Closure for Cross-Layer KV Sharing

## Motivation

Cross-layer KV-sharing methods such as DepthGraft, FusedKV, and KVSharer all commit to a single fixed sharing topology (stride-aligned shallow-middle source pairs, bottom/middle fusion, or dissimilar-pair matching) that is assumed valid regardless of how compute is allocated at inference. This assumption breaks under Qwen3-style thinking-budget switching, where the effective depth and per-layer compute allocation change after prefill, so the prefill-completion and variance-control invariants that a fixed stride-aligned topology was designed to preserve no longer hold. The meta-level structural problem across these trees is that the sharing graph and the normalization/compute regime are treated as independent variables, so no closure invariant can survive budget switching without retraining the backbone.

## Key Insight

If the KV-sharing topology and the LayerNorm affine parameters are both derived from one per-layer compute-allocation vector through a shared closure invariant, then structural closure (every layer's KV representation is reconstructible from its declared sources with bounded variance) holds by construction under any budget switch, rather than being a property of one fixed stride pattern.

## Method

**What it is.** We introduce BCC (Budget-Coupled Closure), an inference-time controller that takes as input a per-layer compute-allocation vector `b ∈ R^L` (derived from the active thinking budget, with `b_ℓ` the fraction of full compute granted to layer `ℓ`) and a deployment position range `P` (the min/max RoPE position indices in the current forward pass), and outputs (i) a cross-layer KV-sharing graph `G(b, P)` and (ii) per-layer LayerNorm affine parameters `(γ_ℓ(b), β_ℓ(b))`, both as closed-form functions of `b` so no backbone weights are retrained. Inputs: `b`, `P`, the pretrained per-layer LN statistics `(μ_ℓ, σ_ℓ)`, the pretrained KV projections, and a position-conditioned calibration table `R_ℓs(Δ)` (defined below). Outputs: a sharing edge set and a set of LN affine parameters used for the current forward pass.

**Load-bearing assumption (stated explicitly).** The closure certificate is valid at deployment only if the post-RoPE key reconstruction residual measured on the calibration pass is a sufficient statistic for the inference-time reconstruction error at the deployment positions. This holds under assumption A: the calibration key geometry is distributionally representative of the deployment positional ranges. A fails when long-context prompts contain unseen positional ranges (failure mode F), because RoPE makes the residual a function of relative positional phase. To make the certificate valid where it is checked, we replace the single calibration scalar with a position-conditioned, per-head residual bound and evaluate the closure test at the actual deployment positions of each forward pass.

**How it works.**

```pseudocode
# Precompute once from a calibration pass (no labels, forward-only)
# Calibration set: 512 examples from the model's pretraining-distribution
# validation split, truncated to 4096 tokens (within the pretrained range).
for each layer l:  E_l = mean_pooled_key_direction_norm(l)   # scalar: how 'self-sufficient' layer l's KV is
                   V_l = mean_residual_variance(l)            # scalar: variance layer l contributes
# Position-conditioned residual table: for each ordered pair (t, s) with s < t,
# and each relative-position bin Δ in {0, 64, 128, 256, 512, 1024, 2048, 4096},
# fit R_ts(Δ) = mean over calibration tokens and heads of the post-RoPE key
# reconstruction error between layer t and layer s at relative offset Δ.
# Fit a per-pair linear extrapolation R_ts(Δ) = a_ts + b_ts * Δ for Δ beyond 4096.
# Closure invariant: for every layer t and every deployment position p in P,
# the residual after reconstruction from its declared source must be
# <= tau * V_t, where tau is a fixed budget (0.1).

BCC(b, P):
  # Step 1: closure-consistent mass allocation
  m_l = b_l * E_l                       # compute mass actually available to layer l
  # Step 2: derive sharing graph as a function of b and P via closure feasibility
  for each target layer t in increasing depth:
      candidates = {s < t : m_s >= m_t}          # only layers with >= mass may source t
      # pick the shallowest candidate s* such that the position-conditioned
      # closure residual is <= tau*V_t at every deployment position in P
      s* = argmin_{s in candidates} ( |t - s| )
           s.t.  max_{p in P} R_ts(|p - p_source|) <= tau * V_t
      if no feasible s*: s* = t (no sharing; layer keeps own KV)
      G(b, P)[t] = s*
  # Step 3: LN affine parameters as a function of the same b and the graph
  for each layer l:
      gamma_l(b) = 1 + alpha * (b_l - b_bar) / (sigma_l + eps)
      beta_l(b)  = beta_l^0 + eta * sum_{(s,l) in G(b,P)} (mu_s - mu_l)
  return G(b, P), {(gamma_l(b), beta_l(b))}
```

Here `alpha` and `eta` are fixed scalar gains (set to 0.5 and 0.1 respectively in all experiments), `tau` is the closure tolerance (set to 0.1), `b_bar` is the mean of `b`, and `R_ts(Δ)` is the position-conditioned post-RoPE key reconstruction error measured on the calibration pass and linearly extrapolated beyond the calibration range. The deployment position range `P` is read from the attention mask of the current forward pass (min and max RoPE index). The calibration pass uses 512 examples, 4096 tokens each, on a single A100-80GB (≈0.5 GPU-hours).

**Why this design.** We chose to derive the sharing graph from the compute-allocation vector rather than from a learned router because a learned router would reintroduce the very independence assumption (topology vs. budget) that DepthGraft and KVSharer make, accepting the cost that our graph is restricted to a closure-feasible family and cannot express arbitrary topologies. We chose a closed-form LN coupling `gamma_l(b) = 1 + alpha*(b_l - b_bar)/(sigma_l+eps)` over a separately learned normalization module because the closure invariant requires the normalization regime to track the same budget variable as the graph, accepting the cost that `alpha` must be tuned per model family. We chose the shallowest-feasible-source rule over a nearest-neighbor or dissimilarity-matching rule (as in KVSharer) because closure feasibility is a variance condition, not a similarity condition, and shallow sources minimize the residual path length that the closure bound accumulates, accepting the cost that some layers fall back to no-sharing when the budget is tight. We chose a position-conditioned residual table over a single calibration scalar because RoPE makes the residual a function of relative positional phase, so a scalar cannot certify closure at unseen deployment positions; accepting the cost that the calibration pass must cover a position grid and that extrapolation beyond 4096 tokens is linear. A domain expert would not describe this as a variant of DepthGraft because DepthGraft fixes a stride-aligned topology at inference and never conditions it on a budget vector or deployment position range, whereas BCC's graph is a closed-form function of `b` and `P` and is coupled to LN parameters through one shared invariant.

**Why it measures what we claim.** The quantity `max_{p in P} R_ts(|p - p_source|) <= tau * V_t` measures structural closure of the KV-sharing graph because the named assumption is that the position-conditioned residual table, fit on the calibration pass and linearly extrapolated beyond 4096 tokens, is a sufficient statistic for the inference-time reconstruction error at deployment positions; this assumption fails when the deployment position range extends far beyond the calibration grid and the linear extrapolation underestimates the true residual (e.g., long-context prompts with positional ranges > 8192 tokens), in which case the residual reflects extrapolation error rather than deployment closure. The quantity `m_l = b_l * E_l` measures the effective compute mass available to layer `l` because the assumption is that `E_l` (mean-pooled key-direction norm) is monotone in the layer's KV self-sufficiency; this assumption fails for layers whose keys are near-degenerate (norm ≈ 0), in which case `m_l` reflects numerical noise rather than usable mass. The quantity `gamma_l(b) = 1 + alpha*(b_l - b_bar)/(sigma_l+eps)` measures the normalization regime consistent with the budget because the assumption is that LN output variance scales linearly with granted compute fraction; this assumption fails at extreme budget values (b_l near 0 or 1), in which case `gamma_l` reflects a saturated affine rather than a variance-matched normalization.

## Contribution

(1) BCC, a closed-form inference-time controller that derives both the cross-layer KV-sharing graph and the per-layer LayerNorm affine parameters from a single compute-allocation vector through one closure invariant, requiring no backbone retraining. (2) The design principle that closure feasibility (a variance condition on reconstruction residual) rather than similarity or stride alignment should determine sharing topology under budget switching, together with the shallowest-feasible-source rule that operationalizes it. (3) A calibration procedure that estimates per-layer key-direction norms and residual variances forward-only, enabling budget-conditional topology selection at inference time.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongBench long-context QA (16-task English subset, 200 examples per task) | Stresses unseen positional ranges where calibration geometry diverges |
| Primary metric | Perplexity on budget-swept forward passes (budget grid: b_l ∈ {0.25, 0.5, 0.75, 1.0} uniform and dispersed) | Directly reflects closure residual under varying compute budgets |
| Secondary metric | Calibration-vs-deployment residual correlation (Pearson r between R_ts(Δ) and measured deployment residual on held-out positional ranges 4096–8192) | Validates the load-bearing assumption A |
| Baseline | DepthGraft (fixed stride topology) | Tests budget-conditioned graph vs fixed topology |
| Baseline | KVSharer (dissimilarity-matched sharing) | Tests closure feasibility vs similarity matching |
| Baseline | FusedKV (cross-layer fusion) | Tests budget-coupled LN vs static fusion normalization |
| Ablation-of-ours | BCC with tau→∞ (no closure constraint) | Isolates whether closure feasibility drives gains |
| Ablation-of-ours | BCC with scalar (position-independent) residual | Isolates whether position conditioning drives gains |

Compute budget: calibration pass 0.5 GPU-hours on one A100-80GB; budget sweep 4 budget levels × 2 dispersion patterns × 3,200 LongBench examples ≈ 12 GPU-hours; total ≈ 15 GPU-hours. Model: Qwen3-8B checkpoint, thinking-budget API `enable_thinking=True` with `thinking_budget` token cap mapped to `b` by `b_l = min(1, budget / (L * avg_tokens_per_layer))`.

### Why this setup validates the claim

The central claim is that deriving the KV-sharing graph and LN affine parameters from the same budget vector `b` yields closure-consistent behavior that budget-independent topologies cannot match. LongBench long-context QA is the right dataset because it induces exactly the deployment distribution where the calibration-vs-deployment key geometry gap is largest, so the closure residual assumption is most stressed. Budget-swept perplexity is the right metric because it directly exposes whether the graph and normalization track the budget variable rather than a fixed regime. The secondary calibration-vs-deployment residual correlation metric directly tests assumption A: if Pearson r between the position-conditioned calibration residual and the measured deployment residual on held-out positional ranges 4096–8192 falls below 0.7, the closure certificate is not transferable and the method's causal chain is broken. DepthGraft tests the sub-claim that topology must be a function of `b`; KVSharer tests the sub-claim that closure feasibility, not similarity, governs source selection; FusedKV tests the sub-claim that LN coupling must share the budget variable. The tau→∞ ablation is the decisive control: if our gains survive when closure is removed, the mechanism is not closure and the idea is misattributed. The scalar-residual ablation isolates whether position conditioning is necessary or whether a single calibration scalar suffices.

### Expected outcome and causal chain

**vs. DepthGraft** — On a long-context prompt with positional ranges beyond calibration, DepthGraft applies its fixed stride topology regardless of budget, so layers whose mass `m_l = b_l * E_l` collapses under tight budget still receive stale KV from a stride-aligned source, producing elevated reconstruction residual and perplexity that degrades sharply as budget tightens. Our method instead recomputes `G(b, P)` per forward pass, falling back to `s* = t` (no sharing) when no candidate satisfies `max_{p in P} R_ts(|p - p_source|) <= tau*V_t`, so we expect a widening perplexity gap versus DepthGraft concentrated in the low-budget regime and near-parity at full budget.

**vs. KVSharer** — On a case where two layers are dissimilar in key direction but one is closure-feasible as a source, KVSharer rejects the feasible shallow source because its dissimilarity criterion is a similarity condition, not a variance condition, so it either shares from a dissimilar-but-similar-scoring layer or skips sharing entirely, leaving residual above the closure bound. Our method selects the shallowest closure-feasible candidate via the position-conditioned residual test, so we expect a visible advantage on layers where dissimilarity and closure feasibility disagree, with parity where they coincide.

**vs. FusedKV** — On a case where budget is uneven across depth (some `b_l` near 0, others near 1), FusedKV applies static fusion normalization that does not track `b`, so layers with low granted compute receive normalization calibrated for full compute, inflating activation variance mismatch and perplexity. Our method sets `gamma_l(b) = 1 + alpha*(b_l - b_bar)/(sigma_l+eps)`, tracking the same budget variable as the graph, so we expect a normalization-driven perplexity gap that grows with budget dispersion across layers and shrinks when `b` is uniform.

**vs. BCC with scalar residual** — On held-out positional ranges 4096–8192, the scalar-residual variant certifies closure using a calibration average that ignores positional phase, so it admits edges whose true deployment residual exceeds `tau*V_t`; we expect its perplexity to degrade relative to full BCC as the deployment position range grows, with parity within the calibration range.

### What would falsify this idea

If the tau→∞ ablation (no closure constraint) matches BCC's perplexity across the budget sweep, the gains are not attributable to closure feasibility and the central coupling claim is wrong. Likewise, if the calibration-vs-deployment residual correlation on held-out positional ranges 4096–8192 falls below 0.7, assumption A fails and the position-conditioned certificate is not transferable; and if BCC's advantage over DepthGraft is uniform across budget levels rather than concentrated in the low-budget, long-context subset where the predicted failure mode lives, the causal chain is falsified.

## References

1. DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction
2. Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers
3. KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing
