# Certified Routing-Set Containment for Quantized MoE Speculative Decoding

## Motivation

Specter and SpecMoE both accelerate MoE inference by trusting a quantized or self-drafted proxy to predict the full-precision model's expert usage, yet neither bounds how far the proxy's top-k expert set can drift from the target's. The root cause is structural: speculative acceptance is decided per token by an empirical likelihood ratio, which says nothing about whether the draft activated the same experts the target would need, so a token can be accepted while the prefetched expert set is wrong, forcing synchronous expert fetches that erase the speedup. This is the meta-gap of treating the draft proxy as a stationary approximation of routing rather than certifying the routing-set divergence per token.

## Key Insight

Because MoE routing is a top-k selection over a finite expert set, a calibrated per-layer envelope on quantizer-induced router-logit perturbation converts each draft token into a distribution-free containment certificate for the target's top-k expert set, turning speculative acceptance into a routing-set test that is auditable before the target runs.

## Method

**What it is — RSC (Routing-Set Certificate).** RSC takes as input a draft token's per-layer router logits from the quantized proxy, a per-layer perturbation radius calibrated offline from a held-out probe, and the top-k margin of the draft's routing scores; it outputs a binary certificate `C_t ∈ {0,1}` stating whether the full-precision target's top-k expert set is guaranteed to be contained in the draft's top-k set, together with the certified expert set used to gate speculative acceptance and prefetch scheduling.

**How it works.**

```pseudocode
# Offline: calibrate per-layer perturbation envelope
for each MoE layer l:
    probe D_l = 512 held-out tokens (disjoint from eval)
    for each token in D_l:
        delta_i = |logit_full(l,i) - logit_quant(l,i)| for all experts i
    rho_l = Quantile_{0.995}( max_i delta_i over D_l )   # empirical radius
    # finite-sample bound: with n=512, the 0.995 quantile is estimated from ~2.5 tail points;
    # we report the DKW 95% confidence interval for the true 0.995 quantile and use its upper
    # endpoint as rho_l (conservative).

# Online: per draft token t, per layer l
s = topk_scores(l, draft_logits, k)          # draft top-k scores
m = s[k-1] - max_{j not in topk} draft_logits[j]   # k-th margin
topk_draft = argtopk(draft_logits, k)

# Certified containment: every expert outside draft top-k has
# full-precision score < m + 2*rho_l, so cannot enter target top-k
C_t = 1  if  m > 2 * rho_l  for ALL layers l
certified_set_t = union_l topk_draft(l)   if C_t else union_l topk_draft(l) ∪ fallback(l)

# Gate speculative acceptance and prefetch
if C_t == 1:
    accept token t; prefetch certified_set_t only
else:
    reject token t; run target verification on token t; prefetch target top-k
```

Hyperparameters: `k` is the MoE top-k (model-defined, e.g., 2); probe size 512; quantile 0.995; `rho_l` is stored per layer (tens of floats). Calibration uses the same quantized proxy and full-precision target as inference; probe tokens are drawn from the held-in training distribution.

**Why this design.** We chose a per-layer calibrated quantile envelope `rho_l` over a global bound because quantization error is layer-heterogeneous, so a global radius would be either vacuous on sensitive layers or over-conservative on stable ones, accepting the cost that `rho_l` must be re-calibrated when the quantization scheme changes. We chose a margin test `m > 2*rho_l` over comparing full draft distributions because the containment question depends only on the k-th boundary, so the margin is the minimal sufficient statistic and avoids the cost of estimating full routing distributions. We chose to gate acceptance on the certificate rather than on the likelihood ratio used by Specter and SpecMoE because likelihood acceptance is blind to expert-set divergence, accepting the cost that some tokens with high likelihood but small margin are rejected, which we offset by multi-step lookahead prefetching the union of certified sets across the draft window. A domain expert would not describe RSC as a variant of Specter: Specter uses the draft's routing as a heuristic prefetch hint with no containment property, whereas RSC's acceptance rule is derived from a calibrated perturbation envelope and certifies the target's expert set before the target runs.

**Why it measures what we claim.** The certified margin `m - 2*rho_l` measures routing-set containment because the assumption that the quantizer-induced logit perturbation is bounded by `rho_l` with probability at least the calibration quantile makes the k-th margin a sufficient statistic for whether any outside expert can overtake the k-th inside expert; this assumption fails when the draft's quantization error is input-dependent beyond the probe distribution (e.g., out-of-domain prompts), in which case the certificate reflects probe-distribution calibration rather than true per-token containment. The per-layer radius `rho_l` measures the quantizer's routing perturbation because the assumption that the probe tokens are exchangeable with inference tokens (so the empirical quantile transfers) makes the empirical max-perturbation quantile an upper envelope; this assumption fails under distribution shift, in which case `rho_l` reflects probe coverage rather than a distribution-free bound. The binary certificate `C_t` measures auditable routing fidelity because the assumption that the target's top-k is determined solely by router logits (no auxiliary routing bias or capacity dropping) makes logit containment equivalent to expert-set containment; this assumption fails if the target applies routing normalization or capacity dropping, in which case `C_t` reflects logit-level rather than executed-set containment.

## Contribution

(1) A per-token routing-set certificate (RSC) that replaces empirical speculative acceptance with a calibrated top-k containment test derived from a per-layer quantizer perturbation envelope. (2) A prefetch-scheduling policy that uses the certified expert set to eliminate synchronous expert fetches on accepted tokens, with a fallback path for uncertified tokens. (3) An offline calibration procedure requiring only a small held-out probe and no retraining of the draft or target model.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Open-domain prompts from held-in distribution | Tests calibration validity under nominal conditions |
| Dataset | Out-of-domain prompts (e.g., code, math) | Probes distribution shift failure mode |
| Primary metric | Expert-set containment rate | Directly measures certificate correctness |
| Secondary metric | Certifiable token fraction vs. top-k margin | Measures applicability across margin regimes |
| Secondary metric | Out-of-domain violation rate for C_t=1 tokens | Quantifies probe-exchangeability failure |
| Baseline | Specter | Quantization-enabled speculative offloading without containment guarantee |
| Baseline | SpecMoE | Speculative decoding for MoE without routing-set certificate |
| Baseline | No-speculation MoE | Lower bound on acceptance and prefetch benefit |
| Ablation-of-ours | RSC with global radius (no per-layer ρ_l) | Tests necessity of layer-heterogeneous calibration |

### Why this setup validates the claim

The central claim is that RSC's margin-based certificate correctly identifies when the full-precision target's top-k expert set is contained in the draft's top-k set, enabling safe speculative acceptance and prefetching. The expert-set containment rate metric directly tests this: if the certificate is valid, tokens with C_t=1 should have target top-k ⊆ draft top-k with high probability, and tokens with C_t=0 should show violations. Specter tests the sub-claim that routing-aware prefetching without a containment guarantee is insufficient: its heuristic prefetch may miss experts that the target actually selects, causing stalls. SpecMoE tests whether likelihood-based acceptance, blind to expert-set divergence, accepts tokens whose routing differs, leading to verification overhead. The no-speculation baseline establishes the lower bound on throughput. The ablation with a global radius tests whether per-layer calibration is necessary: if a single global ρ suffices, the ablation should match RSC, falsifying the layer-heterogeneity motivation. The in-domain vs. out-of-domain split is critical: RSC's certificate relies on probe-calibrated radii, so we expect high containment on in-domain prompts and degradation on out-of-domain prompts, which would confirm the stated assumption failure mode. Additionally, we measure the fraction of tokens that are certifiable (C_t=1) as a function of top-k margin (binned into deciles) on both in-domain and out-of-domain prompts, and report the throughput ceiling implied by that fraction (i.e., the maximum speedup if all certifiable tokens were accepted). We also measure the out-of-domain violation rate for C_t=1 tokens to quantify the failure mode F. This design is falsifiable: if RSC's containment rate is not higher than Specter's prefetch hit rate, or if the global-radius ablation matches RSC, the central claims are wrong.

### Expected outcome and causal chain

**vs. Specter** — On a case where the draft's quantized router logits have a small top-k margin (e.g., near-tie between k-th and (k+1)-th experts), Specter's heuristic prefetch uses draft top-k without certifying containment, so the target often selects an expert outside the prefetched set, causing an offload stall. Our method instead rejects such tokens (C_t=0) and prefetches the target's top-k after verification, because the margin test m > 2ρ_l fails, so we expect a noticeable reduction in stall rate on low-margin tokens but parity on high-margin tokens, yielding a moderate overall TPOT improvement. The certifiable token fraction vs. margin plot will show that RSC certifies most tokens above a margin threshold, and the throughput ceiling will indicate the maximum possible gain.

**vs. SpecMoE** — On a case where the draft assigns high likelihood to a token but the routing margin is small, SpecMoE accepts the token based on likelihood ratio, yet the target's expert set diverges, forcing a costly verification that negates the speculative gain. Our method instead rejects the token because the certificate fails, avoiding wasted verification, so we expect a lower acceptance rate but higher effective speedup, especially on prompts with many near-tie routing decisions, manifesting as a larger TPOT reduction on out-of-domain prompts where margins are tighter. The out-of-domain violation rate for C_t=1 tokens will quantify the residual risk.

**vs. No-speculation MoE** — On a case with high routing margins, the baseline must run the target for every token, incurring full router and expert computation. Our method certifies containment and accepts draft tokens without target verification, prefetching only the certified expert set, so we expect a substantial TPOT speedup on high-margin prompts, with the gap shrinking on low-margin prompts where certification fails more often. The certifiable token fraction vs. margin curve will show the regime where RSC provides benefit.

### What would falsify this idea

If the expert-set containment rate for certified tokens (C_t=1) is not significantly higher than the prefetch hit rate of Specter, or if the global-radius ablation achieves similar containment and TPOT as RSC, then the central claim that per-layer calibrated margins provide a valid and necessary containment certificate is wrong. Additionally, if the out-of-domain violation rate for C_t=1 tokens is high (e.g., >5%), the probe-exchangeability assumption fails in practice, undermining the certificate's reliability.

## References

1. Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading
2. SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding
