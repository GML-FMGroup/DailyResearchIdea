# Attention-Mass Certificates for KV Cache Drops via Sparse Linear Sketches

## Motivation

Across the surveyed paper-trees, every approach that shrinks an agent's working context — Cache Engineering for Agents' drop/repos operations, AgentFold's learned multi-scale folding, MEPIC's block-level recomputation — commits an irreversible exclusion of information at compression time while assuming the agent already knows what is dispensable. The structural defect is not the compression operator but the absence of any check-time invariant: once a KV entry is dropped, no downstream mechanism can falsify the decision, so a single over-aggressive drop silently removes task-critical attention mass for every future query. Cache Engineering for Agents explicitly formalizes drop as excluding entries from future attention but provides no recoverability or sufficiency test, inheriting the lossy-summarization assumption even after switching to exact KV edits.

## Key Insight

Attention output for any query is a convex combination of value vectors with weights equal to its softmax attention mass, so the contribution of dropped entries to a query is exactly a weighted sum whose total weight can be lower-bounded from a fixed-size linear sketch of the dropped keys — making the drop decision falsifiable at commit time without storing or re-prefilling the dropped entries.

## Method

**What it is.** We introduce AMC (Attention-Mass Certificate), a drop-time admission test for KV cache edits. Inputs: the candidate drop set $D$ of KV entries, the retained set $R$, a held-out probe query distribution $Q_{\text{probe}}$ (a fixed window of the 64 most recent queries plus a stratified sample of 256 past queries drawn uniformly across the trace), and a sketch budget $m=64$. Output: a per-probe-query lower bound $\ell_q$ on the attention mass that $R$ retains for $q$, and a binary commit/reject decision for the drop.

**Load-bearing assumption (stated explicitly).** The certificate is sound only if the per-entry score error bound $\text{err}_i$ used below is a valid upper bound on $|\hat{s}_i - s_i|$ for every dropped key $i$ and every probe query $q$ at check time. We do not assume a single global slack transfers across distributions; instead we compute $\text{err}_i$ per entry from a relative-error CountSketch guarantee plus a Cauchy–Schwarz term, and we store the top-$k$ dropped keys exactly so that the exponentially dominant terms in $Z_D$ are never approximated. The assumption that remains is that the CountSketch hash seed is fixed and that the per-entry norm bound $\|k_i\|_2$ stored at sketch time is exact (it is, since we store it as a float32 scalar per entry).

**How it works.**

```pseudocode
# Phase 1: sketch construction (once per candidate drop, O(|D| * m))
# m = 64, fixed hash seed (seed=0), top-k = 8 exact keys stored
for each key k_i in D:
    phi_i = CountSketch_hash(k_i)          # m-dim, fixed hash seed
    S += phi_i * (k_i^T)                    # m x d sketch matrix
    store ||k_i||_2 as float32              # exact per-entry norm bound
    store ||v_i||_2 as float32              # exact per-entry value norm bound
# additionally store the top-k=8 dropped keys with largest ||k_i||_2 exactly
# (full d-dim vectors) so their scores are computed without sketch error

# Phase 2: per-probe-query certificate (O(m * d + k * d) per query)
for each probe query q in Q_probe:
    # exact scores for retained set and for top-k dropped keys
    s_j = q k_j^T / sqrt(d)  for j in R
    s_i_exact = q k_i^T / sqrt(d)  for i in top-k dropped
    # sketch surrogate for remaining dropped entries
    s_i_hat = (S phi_i)^T q / sqrt(d)
    # per-entry error bound: relative-error CountSketch + Cauchy-Schwarz
    # CountSketch guarantee (Charikar et al. 2002): with prob >= 1-delta over hash,
    # |s_i_hat - s_i| <= (1/sqrt(m)) * ||k_i||_2 * ||q||_2 / sqrt(d) * C_delta
    # where C_delta = sqrt(2 log(1/delta)) for delta = 1e-3 (so C_delta ≈ 3.72)
    err_i = (C_delta / sqrt(m)) * ||k_i||_2 * ||q||_2 / sqrt(d)
    # lower bound on dropped mass: exact top-k + sketched tail with per-entry err
    Z_R   = sum_{j in R} exp(s_j)
    Z_D_lb = sum_{i in top-k} exp(s_i_exact) + sum_{i in D \ top-k} exp(s_i_hat - err_i)
    ell_q = Z_R / (Z_R + Z_D_lb)            # certified lower bound on retained mass
    if ell_q < 1 - eps:  reject drop      # eps = 0.02 default

# Phase 3: commit or fall back
if all_q ell_q >= 1 - eps: commit drop, discard sketch
else: reject drop, retain D, log failing query for probe refresh

# Phase 0 (offline, once): no calibration of a global scalar tau.
# Instead, we verify the per-entry error bound on a calibration split:
# run Phases 1-2 on a split where D is known and full attention is computable,
# and record the realized violation rate: fraction of entries i and queries q
# where |s_i_hat - s_i| > err_i. We report this rate; if it exceeds 1e-3 on the
# calibration split, we increase m (to 128 or 256) until it does not.
# The constant C_delta = sqrt(2 log(1/delta)) with delta = 1e-3 is used as above.
```

The certificate is a *lower* bound on retained attention mass because $Z_D^{\text{lb}} \le Z_D$ under the per-entry error bound (which holds with probability at least $1-\delta$ over the hash seed for each entry, by the CountSketch relative-error guarantee), so $Z_R/(Z_R + Z_D^{\text{lb}}) \ge Z_R/(Z_R+Z_D) = $ true retained mass. The agent commits the drop only if this bound exceeds $1-\epsilon$ for every probe query.

**Why this design.** We chose a *fixed-seed CountSketch with per-entry error bounds and exact top-$k$ storage* over a learned verifier because the certificate must be checkable in microseconds at every drop and must not itself be a lossy model of the dropped content; storing the top-$k$ keys exactly removes the exponential amplification of sketch error on the dominant terms, accepting the cost that the bound is loose when $D$ is large and keys are near-orthogonal to the sketch. We chose a *held-out probe distribution* over the agent's own most recent queries because recent queries are exactly the queries that will be issued next, accepting the cost that a genuinely novel future query is not covered (this is the honest scope of the certificate). We chose a *hard commit/reject rule* over a soft penalty because Cache Engineering for Agents already establishes that drop is irreversible under history dependence — a soft signal would let the agent commit anyway, accepting the cost that the agent cannot trade off a marginal drop against a marginal gain. A domain expert will recognize that this is *not* a variant of MEPIC's recomputation or AgentFold's learned folding: those methods decide *what* to compress; AMC decides *whether* a proposed compression is admissible, and its decision rule is closed-form in the sketch, not learned.

**Why it measures what we claim.** The quantity $\ell_q = Z_R/(Z_R + Z_D^{\text{lb}})$ measures *recoverability of downstream attention mass* because the softmax denominator decomposes exactly into retained and dropped contributions and the sketch lower-bounds the dropped term under the per-entry error bound; this assumption fails when dropped keys are adversarially correlated with the hash (collision inflating $s_i$ beyond the bound), in which case $\ell_q$ reflects hash quality rather than true retained mass. The quantity $\text{err}_i$ measures *per-entry sketch error* because it is computed from the CountSketch relative-error guarantee and the exact stored norms; this assumption fails when the hash seed is not fixed or when the norm bound is stale (e.g., after a key update), in which case $\text{err}_i$ reflects stale norms rather than current error. The quantity $\min_q \ell_q$ measures *worst-case downstream sufficiency over the probe distribution* because the probe is drawn from the agent's own recent query stream; this assumption fails when the next query is out-of-distribution relative to the probe, in which case the certificate reflects past-query sufficiency rather than future-query sufficiency.

## Contribution

(1) A drop-time recoverability certificate for KV cache edits that reduces the commit decision to a per-query attention-mass lower bound computed from a fixed-size sparse linear sketch of the dropped entries, requiring no storage of dropped KV and no re-prefill. (2) A calibrated slack procedure that converts the sketch into a one-sided bound with a stated failure mode (hash collision), together with a probe-refresh rule that ties the certificate's coverage to the agent's own query stream. (3) An analysis of when the certificate is vacuous (large drop sets with near-orthogonal keys) and a practical fallback that splits large drops into certified chunks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-horizon agent traces with reusable prefixes (BrowseComp-style) | History-dependent queries expose irreversible drop errors |
| Primary metric | Downstream task success at matched KV budget | Directly measures whether admitted drops preserved sufficiency |
| Baseline | Cache Engineering for Agents (recompute-on-trim) | Tests whether AMC matches safety without re-prefill cost |
| Baseline | MEPIC position-independent caching | Tests whether AMC's admission gate beats structural compression alone |
| Baseline | AgentFold learned folding | Tests whether closed-form admission beats learned what-to-compress |
| Ablation-of-ours | AMC with fixed τ (no calibration), no probe refresh | Isolates whether calibration and probe coverage carry the guarantee |
| Ablation-of-ours | AMC with global τ (99th percentile from calibration) instead of per-entry err_i | Tests whether per-entry error bounds are necessary for soundness |
| Ablation-of-ours | AMC without exact top-k storage (sketch all dropped keys) | Tests whether exact top-k storage prevents exponential error amplification |
| Diagnostic | Realized violation rate of per-entry error bound on live cache vs calibration split | Measures whether the bound holds under domain shift |
| Diagnostic | Fraction of queries outside probe distribution (via novelty detector on query embeddings) | Bounds the scope of the certificate |

### Why this setup validates the claim

The central claim is that a closed-form, sketch-based lower bound on retained attention mass can gate irreversible KV drops so that downstream task success is preserved at a smaller cache. The dataset must therefore contain queries whose answers depend on tokens that a compression heuristic would happily drop — long-horizon agent traces with repeated prefixes and history-dependent follow-ups do exactly this. The primary metric, downstream task success at a *matched* KV budget, is the right detector because it is the only quantity that reflects whether the retained set actually sufficed; a lower bound that is loose but sound should show up as preserved success at smaller budgets, whereas a loose-but-unsound bound would show up as success collapse concentrated on the drop-heavy subset. Cache Engineering for Agents tests the safety sub-claim (no re-prefill needed to stay safe), MEPIC tests whether admission gating adds value over position-independent structural compression, and AgentFold tests whether a learned what-to-compress policy can be replaced by a closed-form whether-to-admit rule. The ablation with global τ isolates whether the per-entry error bound is necessary for soundness, and the ablation without exact top-k storage tests whether the exponential amplification of sketch error is prevented by storing the dominant terms exactly. The diagnostic violation-rate experiment directly measures whether the per-entry bound holds on the live cache, and the novelty-detector experiment bounds the fraction of queries that fall outside the probe distribution, so the significance claim is tied to measured coverage rather than asserted.

### Expected outcome and causal chain

**vs. Cache Engineering for Agents (recompute-on-trim)** — On a case where a trimmed span is later referenced by a follow-up query (e.g., a tool output cited three turns later), the baseline either invalidates the cache and re-prefills or drops the span and loses the reference, because its mechanism has no per-drop sufficiency test. Our method instead computes $\ell_q$ over a probe window that includes the recent queries likely to reference that span, and rejects the drop when the bound falls below $1-\epsilon$, so we expect near-parity on task success with a materially lower re-prefill rate — the gap should appear as fewer cache invalidations at equal success, not as higher success.

**vs. MEPIC position-independent caching** — On a case where position-independent chunks are dropped because they look structurally redundant but carry a query-specific key that the next query attends to, the baseline commits the drop unconditionally and loses that attention mass, because its mechanism compresses structure without checking downstream recoverability. Our method's per-entry error bound and exact top-k storage flag the dropped key's contribution to $Z_D$ and reject the drop when the certified retained mass is too low, so we expect a visible success gap on the subset of traces with late-referenced unique keys and parity on traces with genuinely redundant prefixes.

**vs. AgentFold learned folding** — On a case where the folding policy summarizes away a detail that a later long-horizon query needs (a specific identifier or intermediate result), the baseline's learned compressor has no per-instance guarantee and fails silently, because its mechanism optimizes aggregate compression rather than worst-case retained mass. Our method instead refuses the fold whenever any probe query's $\ell_q$ drops below threshold, so we expect the gap to concentrate on the long-horizon tail where folding errors compound, with parity on short traces where folding rarely loses information.

**vs. AMC with global τ (no per-entry err_i)** — On the domain-shift subset where the live key distribution differs from the calibration split, the global τ is stale and the bound $Z_D^{\text{lb}}$ overestimates the dropped mass, so $\ell_q$ overstates retained mass and the drop is wrongly admitted. Our method's per-entry error bound adapts to each key's norm and query norm, so we expect the global-τ ablation to show a higher violation rate and lower task success on the domain-shift subset, with parity on in-distribution traces.

**vs. AMC without exact top-k storage** — On traces where a few dropped keys carry high attention mass, sketching those keys introduces positive error that is amplified exponentially in $Z_D^{\text{lb}}$, making the bound loose or unsound. Our method stores the top-k=8 dropped keys exactly, so we expect the no-top-k ablation to show a larger gap between certified and true retained mass, and lower task success on the late-referenced-key subset, with parity on traces where attention is diffuse.

### What would falsify this idea

If the ablation with global τ (no per-entry err_i) achieves the same task success at the same KV budget as full AMC, then the per-entry error bound is not necessary for soundness and the central mechanism claim is wrong. Likewise, if the measured violation rate of the per-entry bound on the live cache exceeds 1e-3 on the domain-shift subset, the bound does not hold under distribution shift and the certificate is unsound in that regime.

## References

1. Cache Engineering for Agents
2. MEPIC: Memory Efficient Position Independent Caching for LLM Serving
3. AgentFold: Long-Horizon Web Agents with Proactive Context Management
