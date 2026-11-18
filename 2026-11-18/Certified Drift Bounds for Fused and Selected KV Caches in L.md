# Certified Drift Bounds for Fused and Selected KV Caches in Long-Context Transformers

## Motivation

Efficient long-context decoding increasingly replaces full attention with an approximate cache: Multi-Branch Transformer's Joint-KV fuses branch representations into a shared KV cache, and Radar-style decoders dynamically select a subset of context tokens, yet neither reports any link between the approximation rule and the deviation of the resulting output from full attention. The structural cause is that both methods define their efficiency gain through a heuristic aggregation (branch fusion) or a heuristic importance score (token selection) with no residual accounting, so semantic drift is asserted to be small rather than measured against a full-attention reference. This leaves practitioners unable to state, per decoding step, how far the accelerated model can have departed from the exact model.

## Key Insight

Because attention output is a convex combination of value vectors, the deviation induced by any cache approximation can be decomposed into a per-token weight-mass term times a value-dispersion term, giving a computable residual budget that is tight when the approximation preserves the aggregated attention mass — a property that neither branch fusion nor dynamic selection currently tracks.

## Method

**What it is.** We introduce **CABER** (Certified Approximation Bound for Efficient Retrieval-caches), a diagnostic-and-repair layer that wraps any KV-approximating decoder (Joint-KV fused cache or Radar-style token selection) and emits, per decoding step, a scalar residual budget `R_t` that bounds the L2 deviation between the approximate attention output and the full-attention output over the retained context.

**How it works.** CABER is a per-step wrapper; it does not change the architecture.

```pseudocode
Input: query q_t, full key/value set K, V (or branch set {K_b, V_b}),
       approximation rule A (fusion weights w_b, or selection mask S),
       reference mode: 'oracle' (full attention weights a_i computed exactly) or
                       'periodic' (a_i recomputed every P=32 steps and cached),
       repair threshold tau (default 0.05 * ||o_t||_2)

1. Compute full-attention weights:  a_i = softmax(q_t K^T / sqrt(d))_i
   (in 'periodic' mode, reuse the cached a_i from the last recomputation)
2. Compute approximate weights:     a~_i = weights induced by A on the same K
3. Mass residual:                   M_t = sum_i |a_i - a~_i|   (exact TV distance)
4. Dispersion term:                 Delta_t = max_i ||v_i - o_t||_2
   where o_t = sum_i a_i v_i is the full-attention output
   (deterministic maximum over all retained tokens; no sampling)
5. Certified budget:                R_t = sum_i |a_i - a~_i| * ||v_i - o_t||_2
   (exact triangle-inequality bound; note R_t <= M_t * Delta_t)
6. If R_t > tau (default 0.05 * ||o_t||_2):
       trigger repair: expand the cache by re-adding the top-k dropped
       tokens (k = ceil(M_t * n / 2)) or, for fusion, recompute the fused
       KV with per-branch weights renormalized to match a_i mass
7. Emit (R_t, repaired or not, o~_t) to the decoding log
```

**Why this design.** We chose a convex-combination residual decomposition (mass residual times value dispersion) over a learned error predictor because the decomposition follows directly from the definition of attention as a weighted average of values, accepting the cost that the dispersion term requires a deterministic maximum over all retained value norms rather than a sampled estimate. We chose to wrap the existing approximation rule rather than replace Joint-KV or Radar's selection with a new rule, because the research target is the efficiency mechanism itself and a replacement would confound architecture with certification, accepting the cost that CABER inherits any structural bias in the underlying rule. We chose a repair trigger with a fixed relative threshold `tau` over an adaptive schedule, because a fixed threshold makes the emitted budget comparable across steps and branches, accepting the cost that a single `tau` may be too conservative at some context lengths. Relative to a retrieval-style KV-eviction baseline (e.g., H2O / Radar's importance scoring), CABER does not score tokens for retention; it computes a post-hoc deviation budget that retrieval methods do not produce, so a domain expert would not describe CABER as a variant of those eviction schemes.

**Why it measures what we claim.** The mass residual `M_t` measures the semantic drift of the approximation because it equals the total-variation distance between the full and approximate attention distributions, under the assumption that the value vectors are bounded in norm by `Delta_t`; this assumption fails when a small number of high-norm outlier values dominate the output, in which case `M_t` understates the true output deviation and the deterministic dispersion term partially compensates. The dispersion term `Delta_t` measures the sensitivity of the output to weight perturbations because it is the maximum L2 distance from the full-attention output to any retained value vector, under the assumption that the retained set contains all tokens with non-negligible attention mass; this assumption fails when attention mass concentrates on a single token, in which case `Delta_t` collapses to near zero and the budget reflects mass error only. The repair trigger `R_t > tau` measures whether the approximation has exceeded an acceptable drift because `tau` is set relative to the output norm, under the assumption that relative output deviation is the quantity practitioners care about; this assumption fails when downstream tasks are sensitive to absolute rather than relative error, in which case the budget should be compared in absolute terms.

**Assumption A (exactness of the bound).** The budget `R_t = sum_i |a_i - a~_i| * ||v_i - o_t||_2` is a true upper bound on `||o~_t - o_t||_2` because it follows from the triangle inequality applied to the difference of two convex combinations: `||sum_i (a_i - a~_i) v_i||_2 <= sum_i |a_i - a~_i| * ||v_i - o_t||_2`. This requires no sampling and no representativeness assumption. The failure mode F is when attention mass concentrates on a single high-norm token: then `Delta_t` (the maximum distance from `o_t`) is small, but the mass residual `M_t` can still be large, and the budget correctly reflects mass error only; conversely, when a dropped token has large `||v_i - o_t||_2`, the budget inflates even if `M_t` is small, correctly flagging value-dispersion error.

## Contribution

(1) CABER, a per-step certified residual budget that decomposes cache-approximation drift into an attention-mass term and a value-dispersion term, applicable to both fused (Joint-KV) and selected (Radar-style) caches without changing the underlying architecture. (2) A repair trigger that uses the budget to selectively expand the cache or renormalize fusion weights, converting an unquantified efficiency trade-off into a bounded one. (3) An empirical characterization of where the budget is tight versus loose across branch counts and context lengths in Multi-Branch Transformer decoding.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-context language modeling (PG-19 / long-doc QA) | Stresses KV-cache approximation where drift accumulates |
| Primary metric | Task accuracy vs. full-attention gap | Directly measures whether repair recovers lost quality |
| Baseline | Joint-KV fused cache (no CABER) | Tests whether certification+repair improves a fusion approximator |
| Baseline | Radar token selection (no CABER) | Tests whether budget catches selection-induced drift |
| Baseline | H2O-style KV eviction | Tests against retrieval-style eviction competitor |
| Ablation-of-ours | CABER without repair trigger (budget only) | Isolates whether repair, not logging, drives gains |
| Ablation-of-ours | CABER with sampled dispersion (m=16 pairs) | Tests whether deterministic bound is necessary |

### Why this setup validates the claim

The central claim is that a post-hoc deviation budget `R_t` reliably flags steps where an approximate KV mechanism has drifted from full attention, and that triggering repair on those steps recovers quality that the approximator alone loses. Long-context language modeling is the right testbed because it is precisely where KV approximation is used and where drift compounds across many decoding steps. The two approximation baselines (Joint-KV fusion and Radar selection) each test a different sub-claim: Joint-KV tests whether CABER's mass-residual term detects fusion-weight mass error, while Radar tests whether the dispersion term detects selection-induced value-cloud distortion. The H2O eviction baseline tests the framing claim that CABER is not merely another retention scorer. The no-repair ablation is the linchpin: if CABER's gains come from certification-and-repair rather than from the budget log itself, then budget-only should show little accuracy recovery while full CABER shows a clear gap. The sampled-dispersion ablation tests whether the deterministic bound is necessary for the certified claim. Task accuracy relative to full attention is the correct metric because the claim is about bounding output deviation, and accuracy gap to full attention is the observable consequence of that deviation.

### Expected outcome and causal chain

**vs. Joint-KV fused cache (no CABER)** — On a case where branch representations diverge in mass (one branch's fused weights under-allocate probability to a decisive token), the baseline produces a fused output that silently skews away from full attention because its fixed fusion weights cannot detect per-step mass error. Our method instead computes `M_t` against the full-attention weights and triggers renormalized fusion when `R_t` exceeds `tau`, because the mass residual directly exposes the branch-weight mismatch. We expect a noticeable accuracy gap on long-context subsets with high branch divergence but near-parity on short, low-divergence contexts.

**vs. Radar token selection (no CABER)** — On a case where the importance search drops a token that carries modest attention mass but large value-norm, the baseline produces an output with bounded mass error yet inflated value deviation because selection scoring ignores value dispersion. Our method instead catches this via the dispersion term `Delta_t`, which inflates `R_t` even when `M_t` is small, triggering re-addition of dropped tokens. We expect recovery concentrated on examples with high-norm outlier values, with parity elsewhere.

**vs. H2O-style KV eviction** — On a case where a low-importance-scored token is repeatedly evicted across steps, the baseline produces cumulative drift it never flags because eviction heuristics emit no deviation estimate. Our method instead emits a per-step budget that accumulates visible drift and repairs before compounding, because it measures post-hoc output deviation rather than retention importance. We expect CABER's advantage to grow with decoding length, showing a widening gap late in generation.

### What would falsify this idea

If CABER's accuracy gain over the no-repair ablation is uniform across all subsets rather than concentrated on the high-mass-error and high-dispersion subsets the causal chain predicts, the claim that the budget identifies genuine drift is wrong. Likewise, if the budget-only ablation matches full CABER, repair is not doing the work and the certification claim collapses.

## References

1. Multi-Branch Transformer
