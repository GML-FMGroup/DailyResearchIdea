# SpecLoad Closure: A Joint Budget Identity for Expert-Loading Traffic and Speculative Verification in Offloaded MoE Decoding

## Motivation

Existing offloaded-MoE speculative systems each optimize one side of a coupled cost under a fixed per-decode-step model: Specter minimizes PCIe bytes via quantized-draft prefetching while leaving verification compute unmodeled, and SpecMoE minimizes draft compute via self-assisted drafting while leaving expert-loading traffic a second-order concern. The structural failure is that multi-step speculative verification makes per-step cost depend jointly on draft length and on which experts the draft touches, so the two costs are bound through the same acceptance event and cannot be budgeted independently; no closure rule expresses this coupling as a single testable identity, which is why neither system can predict when increasing draft length will help versus hurt.

## Key Insight

For any draft length and expert-activation trace, the product of bytes-moved-per-verified-token and tokens-verified-per-unit-bandwidth is pinned by the draft's acceptance profile and the expert-reuse distance, so the two costs trade off along a surface fixed by the draft rather than by either cost alone.

## Method

**What it is.** SpecLoad Closure (SLC) is a runtime budget rule, not a new model: given a draft length k, an observed acceptance profile, and a measured expert-reuse distance, it emits a single scalar budget B(k) and a per-step feasible set of (bytes moved, tokens verified) pairs that the scheduler must stay inside. Inputs are the draft-acceptance trace and the per-layer expert-reuse distances; the output is the admissible (k, prefetch-depth) operating point for the next decode window.

**Load-bearing assumption (stated explicitly).** SLC assumes that the expected number of resident experts per layer during multi-step speculative verification is given by L = alpha*k*(1 - exp(-k/d)), i.e., that expert residency is a smooth Poisson-arrival/Little's-law function of acceptance rate alpha and a scalar reuse distance d, so that a single scalar budget B(k) = L*E_bytes/bw faithfully bounds the joint (bytes-moved, tokens-verified) product. This assumption is load-bearing: the feasibility test compares an observed product against B(k) derived from this model. We do not assert it holds universally; instead, we calibrate B(k) empirically as described below and report the gap between analytic L and measured resident experts as a function of routing skew, bounding the closure's validity domain.

**How it works.**

```pseudocode
# Per decode window, after each verification step
def slc_step(draft_len k, accept_trace A, reuse_dist d, bandwidth_bw, compute_cap,
             calib_window=256, quantile=0.95, tol=0.05):
    # 1. Acceptance profile: empirical mean acceptance rate over last W=32 steps
    alpha = mean(A[-W:])                       # in (0,1]
    # 2. Little's-law closure: expected resident experts per layer
    #    L = alpha * k * (1 - exp(-k / d))    (reuse distance d in tokens)
    L = alpha * k * (1.0 - exp(-k / d))
    # 3. Analytic budget identity: B_analytic(k) = (bytes/token) * (tokens/bandwidth)
    #    bytes/token = L * E_bytes; tokens/bandwidth = 1 / (bw_eff)
    E_bytes = expert_size_bytes(layer)        # from model config
    B_analytic = L * E_bytes / max(bandwidth_bw, eps)
    # 4. Empirical calibration: maintain a rolling window of the last calib_window
    #    observed (bytes_per_token * tokens_per_bw) products, conditioned on k and
    #    measured cache-hit rate h (fraction of expert fetches that hit resident cache).
    #    B_calib(k) = quantile_{quantile}(observed_product | k, h)
    B_calib = rolling_quantile(observed_products, k, h, window=calib_window, q=quantile)
    # 5. Calibrated budget: use B_calib when at least calib_window samples exist for k,
    #    else fall back to B_analytic.
    B = B_calib if count(observed_products | k) >= calib_window else B_analytic
    # 6. Feasible set: scheduler must keep observed product within B
    feasible = (observed_bytes_per_token * observed_tokens_per_bw) <= B * (1 + tol)
    # 7. If infeasible, shrink k to k' = argmin_k' |B(k') - observed|, tol=0.05
    return B, feasible
```

Diagram of the control loop:

```
accept_trace A ──► alpha ──┐
                           ├──► L (Little's law) ──► B_analytic ──┐
reuse_dist d ──────────────┘                                      ├──► B(k) ──► feasible? ──► scheduler
observed_products ──► rolling_quantile ──► B_calib ──────────────┘         │
                                    shrink k if infeasible ◄────────────────┘
```

**Why this design.** We chose an analytic closure over a learned predictor because the acceptance identity and Little's law give a closed-form product that a learned head would only approximate, accepting the cost that the rule is only as accurate as the reuse-distance estimate. We chose to shrink draft length k rather than increase prefetch depth when infeasible because prefetch depth is bounded by host-memory residency and shrinking k directly reduces verification compute, accepting the cost that throughput may drop below the unconstrained optimum. We chose a 32-step acceptance window over a full-history average because expert activation is non-stationary and a long window would lag regime changes, accepting the cost of higher variance in B(k). We chose to calibrate B(k) from a rolling window of 256 observed products at the 95th percentile rather than rely solely on the analytic formula because the Poisson and i.i.d. assumptions fail under bursty routing and correlated acceptance, accepting the cost of a warm-up period before calibration activates. A domain expert would not describe SLC as a variant of Specter or SpecMoE: those systems schedule prefetch or draft compute under a fixed per-step model, whereas SLC is a closure rule that constrains the joint product and can be layered on either system's scheduler.

**Why it measures what we claim.** The quantity alpha (mean acceptance rate) measures the draft's acceptance profile because the speculative acceptance identity makes the expected number of verified tokens per draft equal alpha*k under an i.i.d. acceptance assumption; this assumption fails when acceptance is correlated across positions (e.g., repetitive text), in which case alpha reflects a biased per-step rate rather than the true expected accepted length. The quantity (1 - exp(-k/d)) measures expert-reuse distance because under a Poisson expert-activation process the probability a draft touches an expert within distance d is exactly 1 - exp(-k/d); this assumption fails when routing is bursty or skewed, in which case the term reflects a smoothed reuse probability rather than the actual cache hit rate. The product L*E_bytes/bw measures the joint budget because Little's law equates expected resident experts to arrival rate times residence time; this assumption fails when prefetch and verification overlap imperfectly, in which case B reflects an idealized bandwidth demand rather than measured traffic. The calibrated budget B_calib measures the empirical feasible frontier because it is the 95th percentile of observed products conditioned on k and measured cache-hit rate h; this calibration fails when fewer than 256 samples exist for a given k, in which case B falls back to the analytic B_analytic and inherits its assumptions.

## Contribution

(1) A joint closure rule (SpecLoad Closure) that expresses expert-loading memory traffic and speculative verification compute as a single budget identity parameterized by draft length and expert-reuse distance, derivable from Little's law and the speculative acceptance identity. (2) A runtime control policy that uses the closure to select the admissible draft length and prefetch depth per decode window, with a feasibility test that shrinks k when the observed cost product exceeds the budget. (3) An empirical characterization of the acceptance-profile and reuse-distance regimes under which multi-step speculative offloading beats single-step offloading, reported as the closure surface rather than as a single speedup number.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-form generation traces from MoE LLMs (e.g. Mixtral, DeepSeek-MoE) | Repetitive and bursty routing stress the reuse-distance assumption |
| Primary metric | Time-per-output-token (TPOT) at fixed bandwidth budget | Directly reflects the joint bytes/tokens closure SLC constrains |
| Baseline | Specter (quantization + multi-step speculative offloading) | Tests whether SLC closure beats fixed per-step prefetch scheduling |
| Baseline | SpecMoE (self-assisted speculative decoding) | Tests whether SLC closure beats fixed draft-length scheduling |
| Baseline | Unconstrained SpecMoE/Specter with no budget rule | Tests whether the closure itself, not just the components, drives gains |
| Baseline | Naive LRU prefetcher with SLC constraint | Tests drop-in generality of SLC on a third scheduler |
| Ablation-of-ours | SLC with fixed k=1 (no shrink rule) | Isolates the contribution of the feasibility/shrink feedback |
| Ablation-of-ours | SLC with analytic B only (no calibration) | Isolates the contribution of empirical calibration |
| Diagnostic | Measured cache-hit rate vs. analytic 1-exp(-k/d) | Quantifies gap between analytic and measured residency |

### Why this setup validates the claim

The central claim is that a single scalar closure budget B(k), derived from the acceptance identity and Little's law and calibrated from observed traffic, can constrain the joint (bytes-moved, tokens-verified) product and thereby avoid the regime where speculative MoE offloading thrashes bandwidth. Specter tests the sub-claim that fixed per-step prefetch scheduling cannot adapt when reuse distance shifts, because it has no closure rule tying prefetch depth to observed acceptance. SpecMoE tests the sub-claim that fixed draft length k over-verifies when alpha drops, since its scheduler does not shrink k against a bandwidth budget. The unconstrained baseline tests whether SLC's gain is genuinely from the closure rather than from the underlying speculative machinery. The naive LRU prefetcher with SLC tests whether the closure is a drop-in constraint that improves a third, distinct scheduler, supporting generality. The fixed-k=1 ablation is the critical control: if SLC's advantage disappears when the shrink rule is removed, the closure is doing the work; if it survives, the gain is from something else. The analytic-only ablation tests whether empirical calibration of B(k) is necessary or whether the analytic formula suffices. The diagnostic directly measures the gap between analytic and measured residency, bounding the closure's validity domain. TPOT is the right metric because the closure constrains exactly the bandwidth-per-token that dominates TPOT in CPU-offloaded MoE decode, so a real closure effect must appear as a TPOT gap concentrated on traces where reuse distance departs from the nominal regime.

### Expected outcome and causal chain

**vs. Specter** — On a case where a long repetitive span drives reuse distance d down sharply while acceptance alpha stays high, Specter's fixed prefetch depth keeps resident experts pinned to the nominal regime, so it over-prefetches experts that would have been reused anyway and wastes bandwidth. Our method instead recomputes B(k) from the shrunken d and the calibrated 95th-percentile frontier, because the (1 - exp(-k/d)) term collapses and the closure demands fewer resident experts. We expect a noticeable TPOT advantage on repetitive long-form subsets and near-parity on diverse, high-entropy prompts where d stays near nominal.

**vs. SpecMoE** — On a case where acceptance alpha drops mid-window (e.g., a topic shift makes the draft model mispredict), SpecMoE keeps its draft length k fixed and continues verifying k tokens per step, so verification compute and expert fetches stay high while accepted tokens per step fall. Our method instead shrinks k toward the argmin of |B(k') - observed| because the closure sees alpha fall and the calibrated budget tightens, reducing both verification compute and expert traffic. We expect a visible TPOT gap concentrated on windows following regime changes, with parity on stationary stretches.

**vs. Unconstrained SpecMoE/Specter with no budget rule** — On a case where bandwidth is the binding constraint (host-memory-bound decode), the unconstrained system overshoots the observed bytes-per-token times tokens-per-bandwidth product and stalls on memory transfers, because nothing ties prefetch and verification to a shared budget. Our method instead keeps the observed product inside B*(1+tol) by construction, because the feasibility check fires and shrinks k before the stall. We expect the unconstrained baseline to show high TPOT variance and long tail latencies, while SLC shows a tighter TPOT distribution with a lower tail, especially at tight bandwidth settings.

**vs. Naive LRU prefetcher with SLC** — On a case where routing is bursty and the LRU prefetcher's hit rate diverges from the analytic 1-exp(-k/d), the naive prefetcher without SLC would either over- or under-prefetch. With SLC, the calibrated budget B_calib uses the measured cache-hit rate h to set the feasible set, so the prefetcher's depth is adjusted to keep the observed product within the 95th-percentile frontier. We expect a consistent TPOT improvement over the naive LRU prefetcher alone, demonstrating that SLC is a drop-in constraint across schedulers.

**vs. SLC with analytic B only (no calibration)** — On a case where routing is bursty and acceptance is correlated, the analytic B_analytic underestimates or overestimates the true feasible frontier, causing either unnecessary shrinking (throughput loss) or missed infeasibility (stalls). The calibrated B_calib corrects this by using the empirical 95th percentile. We expect the calibrated version to outperform the analytic-only version on bursty/skewed subsets, with parity on stationary subsets.

### What would falsify this idea

If SLC's TPOT gain over Specter and SpecMoE were uniform across all subsets rather than concentrated on traces where reuse distance or acceptance departs from the nominal regime, the closure would not be the causal mechanism and the idea would be wrong. Likewise, if removing the shrink rule (fixed k=1 ablation) left TPOT unchanged, the feasibility feedback would be inert and the central claim would fail. If the calibrated B(k) did not reduce the gap between analytic and measured residency on bursty traces, the calibration would be ineffective and the load-bearing assumption would remain unverified.

## References

1. Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading
2. SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding
