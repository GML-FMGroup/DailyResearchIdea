# Routing-Composition Certificates for Semantics-Preserving Sparse Attention

## Motivation

Sparse-attention systems in the MoBA lineage already decouple retention from GPU scheduling granularity (LightningMoBA thresholds per-query block scores and regroups selected rows into KV-centric column tasks), but no invariant guarantees that composing thresholded selection with column-grouped reuse—including bounded splits of high-fanout columns—yields the same weighted sum as the dense attention kernel restricted to the selected set. The root cause is structural: selection is defined per query while execution is reorganized per KV column, and the regrouping (plus split accumulation order) changes the summation tree without any certificate that the reduction is order-invariant, so correctness of any recomposition is assumed rather than verified. This compositional-closure gap recurs across the lineage—MoBA-style routing, Twilight's adaptive top-p pruning, and DuSA's two-stage sparsity each select a set but never certify that their execution recomposition preserves the selected-set semantics.

## Key Insight

If every KV block column's contribution is accumulated in a fixed-point scaled-integer representation whose reduction is exactly associative, then any recomposition of thresholded per-query selection and column-grouped task decomposition (including bounded splits) computes a bit-identical weighted sum to the per-query selected-set kernel, making routing decisions certifiable rather than assumed.

## Method

**What it is.** Routing-Composition Certificates (RCC) is a sparse-attention execution layer that takes per-query block scores, a threshold τ, and KV blocks, and emits attention outputs together with a per-(query-head, KV-column) certificate asserting that the column-grouped schedule reproduces the per-query selected-set dense reduction. It wraps dynamic thresholded routing and KV-centric ownership with a block-floating-point accumulation algebra that makes recomposition order-invariant.

**How it works.**

```pseudocode
INPUT: Q (heads H, queries N, dim d), K,V blocks (B blocks of size s), threshold tau, split cap C

# Phase 1: per-query selection (dynamic routing)
for each query i, head h:
    s[b] = block_score(Q[h,i], K[b])          # MoBA-style block score
    S[i,h] = { b : s[b] >= tau }               # variable-cardinality selected set

# Phase 2: block-floating-point quantization (order-invariance substrate)
for each query i, head h:
    p = softmax over S[i,h] of s[b]          # per-query weights on selected set
    e[i,h] = floor(log2(max_b p[b]))          # shared exponent per (i,h) row
    w_q[i,b] = round(p[b] / 2^{e[i,h]} * 2^F) # int32 mantissa, F=24 fractional bits
    a_q[i,b] = round((V[b] @ Q[h,i]) / 2^{e[i,h]} * 2^F) # int32 partial products

# Phase 3: KV-centric column grouping with bounded splits
for each (h, b):
    rows = { i : b in S[i,h] }                # fanout
    tasks = partition(rows, size <= C)         # C=64 bounded split
    for task t: acc[h,b,t] = sum_{i in t} w_q[i,b]*a_q[i,b]   # int64 exact sum

# Phase 4: certificate check + reduction
for each (h,i):
    total = sum over selected b of acc[h,b,·]   # int64 exact, order-invariant
    out[h,i] = total / (2^F * sum_b w_q[i,b])   # dequantize
    ref = float64 sum over selected b of p[b] * (V[b] @ Q[h,i])  # higher-precision reference
    cert[h,i] = (abs(out[h,i] - ref) <= eps)    # eps = 1e-5 * abs(ref) + 1e-8

GATE: if any cert fails, fall back to per-query dense-on-selected execution for that (h,i)
OUTPUT: out, cert
```

**Why this design.** We chose block-floating-point int32/int64 accumulation over floating-point because FP addition is non-associative, so column-grouped splits would produce order-dependent rounding and no certificate could be exact; the cost is a quantization error bounded by 2^-F per term, which we accept because F=24 places it below FP32 mantissa noise for the tested score ranges. We chose a recompute-and-compare certificate against a float64 reference over a purely analytic bound because the analytic route would require distributional assumptions on scores that the dynamic thresholding explicitly violates; the cost is duplicated reduction work on the certificate path, which we amortize by reusing the loaded KV tile from the execution path. We chose a per-query fallback gate over aborting the whole kernel because certificate failures are expected to be rare and localized (tie-breaking at the threshold boundary); the cost is a branch that must be handled by the runtime pipeline, which we hide by issuing the fallback on a separate warp. A domain expert would not describe this as a variant of LightningMoBA: LightningMoBA reorganizes execution for reuse but never certifies that the reorganization preserves the selected-set sum, whereas RCC makes the reduction algebra itself order-invariant and verifies closure per (query-head, column).

**Why it measures what we claim.** The certificate boolean `cert[h,i]` measures compositional closure of selection and execution because we assume that exact integer reduction is order-invariant and that block-floating-point quantization with per-row shared exponent preserves the relative ordering and weighting of all selected blocks (no aliasing-induced reweighting); this assumption fails when two distinct softmax weights round to the same integer or a large-magnitude term dominates the int64 sum, in which case the certificate reflects quantization aliasing rather than a genuine schedule discrepancy. The threshold τ measures the retention decision because we assume block scores are a sufficient statistic for block relevance, which fails when two blocks tie at τ, in which case the certificate detects the tie-induced ambiguity and the gate falls back. The split cap C measures bounded task granularity because we assume each split's rows are disjoint and exhaustive over the column fanout, which fails if partitioning drops a row, in which case the certificate mismatch exposes the dropped contribution rather than silently under-counting.

## Contribution

(1) A routing-composition certificate mechanism that verifies, per query head and KV block column, that thresholded selection composed with column-grouped reuse reproduces the per-query selected-set reduction, using a fixed-point accumulation algebra that makes recomposition order-invariant. (2) A gated dynamic-routing design principle: retention decisions are admitted only when the certificate holds, with a per-query fallback that localizes failures to threshold ties and quantization aliasing. (3) An execution layer that integrates certificate checking into the KV-centric pipeline without a separate pass, reusing loaded KV tiles.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Long-context RULER suite (16K–128K context lengths) | Matched-retention comparison across score distributions |
| Primary metric | Mean RULER score at fixed retention (Top-16 blocks per query) | Detects schedule-induced retention loss directly |
| Baseline | FlashMoBA (fixed Top-k, query-centric) | Static budget, no recomposition certificate |
| Baseline | LightningMoBA (dynamic routing, KV-centric) | Closest execution-layer rival, no closure check |
| Ablation-of-ours | RCC without certificate gate | Isolates certificate from routing algebra |
| Diagnostic | Drift measurement: max and mean absolute difference between grouped sum and per-query selected-set sum in FP32 across RULER subsets | Establishes that schedule-induced recomposition drift is real and non-trivial |

### Why this setup validates the claim

The central claim is that column-grouped KV-centric execution silently perturbs the per-query selected-set reduction, and that an order-invariant block-floating-point algebra plus a per-(query-head, column) certificate restores exact closure while retaining the reuse benefits. RULER at matched retention is the right metric because it makes the retention decision itself the controlled variable: if RCC and the baselines select the same blocks (verified by logging selected sets), any score gap must come from how the grouped schedule recomposes contributions, not from a different sparsity level. FlashMoBA tests the sub-claim that fixed budgets plus query-centric ownership lose accuracy at tie-heavy boundaries; LightningMoBA tests the harder sub-claim that even a well-engineered dynamic KV-centric schedule without closure verification drifts from the selected-set dense reduction. The no-certificate ablation is the crucial control: it keeps our dynamic routing and block-floating-point algebra but drops the recompute-and-compare gate, so any gain attributable to detection-and-fallback is separated from the gain attributable to order-invariance alone. The drift diagnostic directly measures the magnitude of schedule-induced recomposition error in the baselines, providing the empirical basis for expecting a non-trivial gap. Together these form a falsifiable test: the certificate must buy accuracy precisely where schedule recomposition diverges.

### Expected outcome and causal chain

**vs. FlashMoBA (fixed Top-k, query-centric)** — On a case where several blocks score near the threshold and the score distribution is heavy-tailed, FlashMoBA forces one Top-k budget across all queries, so queries whose relevant blocks fall just outside k are truncated and their selected-set sum is silently wrong. Our method instead routes each query by threshold, emits variable-cardinality sets, and certifies the grouped reduction against a float64 reference of the per-query sum. We expect near-parity on easy uniform-score subsets but a visible RULER gap concentrated on retrieval items with diffuse or tie-adjacent score mass.

**vs. LightningMoBA (dynamic routing, KV-centric)** — On a case where many query ranges select the same KV column, LightningMoBA groups rows by column for reuse but never verifies that the regrouped accumulation equals the per-query selected-set reduction, so split-order rounding drifts. Our method instead accumulates in block-floating-point int64 with per-row shared exponents and recomputes the per-query sum as a certificate against a float64 reference, falling back only on genuine mismatches. We expect a modest but consistent score advantage at matched retention, largest on long-context items where column fanout is high.

**vs. RCC without certificate gate** — On a case where quantization pushes two distinct selected-set weights to the same integer or a partition boundary drops a row, the gated-off variant proceeds with the grouped sum and never notices. Our method instead detects the mismatch and falls back to per-query dense-on-selected execution for that (query-head, column). We expect the ablation to match us on clean subsets but lose accuracy on boundary-heavy subsets, isolating the certificate's contribution.

### What would falsify this idea

If our gain over LightningMoBA is uniform across all subsets rather than concentrated where column fanout and threshold ties are high, the certificate is not doing the claimed work. Likewise, if the no-certificate ablation matches the full method everywhere, compositional closure is not the mechanism behind any improvement.

## References

1. Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining
