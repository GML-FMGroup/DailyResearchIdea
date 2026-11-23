# Ranking-Consistent Sparse Memory Attention: Aligning MemOnce Retrieval with Softmax Token Importance

## Motivation

MemOnce replaces full token-level KV caches with a compact final-layer memory and cross-layer sparse indices, but it never verifies that the tokens its sparse memory attention retrieves are the same tokens standard softmax attention would rank as most important; the surrogate attention is optimized only for reconstruction and language-model loss, so retrieval can systematically drift from the softmax importance ordering under bounded KV reconstruction error. This is a shared gap across efficient-attention branches: ToST (E8) replaces pairwise softmax with a linear-time token-statistics operator and also provides no ordering-preservation check, while MemOnce (E5) provides no ranking-consistency condition on its cross-layer sparse indices. The root cause is structural — both branches optimize a surrogate attention objective that is never coupled to the ordering induced by the attention it approximates, so no training signal forces the top-k retrieved indices to coincide with the softmax top-k.

## Key Insight

Under a bounded per-token KV reconstruction error, the softmax attention score gap between any two tokens is preserved up to a monotone calibration factor, so the top-k memory retrieval indices can be made to coincide with the softmax top-k by enforcing a ranking-consistency regularizer on the memory encoder rather than by increasing memory capacity.

## Method

**What it is.** We introduce RC-MemOnce (Ranking-Consistent MemOnce), a training-time regularizer plus a retrieval-time calibration for MemOnce's memory sparse attention. Inputs: a continual input sequence, MemOnce's compact final-layer memory states, and the decoded KV caches for each attention layer. Outputs: per-layer sparse retrieval indices whose ordering matches the softmax-attention token-importance ordering on the same input, together with the usual next-token predictions.

**Load-bearing assumption (stated explicitly).** The method assumes that the reconstruction-induced score distortion between MemOnce's surrogate attention and full-cache softmax attention is, to first order, a query-independent, layer-level monotone scale factor gamma_l — i.e., that the per-token KV reconstruction error term q_q · (k_t - k_hat_t) / sqrt(d) is dominated by a shared component across queries attending to the same decoded cache. This assumption is what licenses a single per-layer scalar gamma_l and a set-intersection ranking loss optimized through s_cal. It is tested directly (see the diagnostic in the experiment section) and is expected to fail on queries whose reconstruction error is query-specific; the method therefore includes a per-layer rank-correlation gate (Phase 6) that falls back to per-layer indices when the assumption is violated.

**How it works.**

```pseudocode
# Phase 1: reference softmax importance (teacher, no grad)
for each layer l and query position q:
    s_softmax[l, q, t] = softmax_t( q_q · k_t / sqrt(d) )   for all cached tokens t
    rank_ref[l, q] = argsort_desc( s_softmax[l, q, :] )[:K]

# Phase 2: MemOnce surrogate importance (student, grad)
for each layer l and query position q:
    k_hat_t = MemoryDecoder(memory_states)                  # reconstructed KV
    s_mem[l, q, t] = q_q · k_hat_t / sqrt(d)
    rank_mem[l, q] = argsort_desc( s_mem[l, q, :] )[:K]

# Phase 3: ranking-consistency loss (per layer, per query)
L_rank = mean_{l,q} [ 1 - |rank_ref[l,q] ∩ rank_mem[l,q]| / K ]

# Phase 4: monotone calibration head (one scalar per layer, shared across queries)
gamma_l = softplus(w_l)                                     # w_l learned, init 0
s_cal[l, q, t] = gamma_l * s_mem[l, q, t]

# Phase 5: joint training
L_total = L_LM + lambda_rank * L_rank + lambda_cal * ||gamma_l - 1||^2
           # lambda_rank = 0.1, lambda_cal = 1e-3, K = 64

# Phase 6: retrieval at inference
# Per-layer rank-correlation gate (tests the load-bearing assumption)
rho_l = Spearman( s_mem[l, :, :], s_softmax[l, :, :] )      # per-layer, over all queries/tokens
if rho_l >= rho_min:                                        # rho_min = 0.7
    indices[l, q] = argsort_desc( s_cal[l, q, :] )[:K]      # shared across layers via MemOnce's cross-layer index reuse
else:
    indices[l, q] = argsort_desc( s_mem[l, q, :] )[:K]      # per-layer fallback, no cross-layer reuse
```

Hyperparameters: K = 64 retrieved tokens per query; lambda_rank = 0.1, lambda_cal = 1e-3; gamma_l initialized to 1 via w_l = 0; ranking loss computed only on the top-2K candidate pool to keep cost linear; rank-correlation gate threshold rho_min = 0.7 (Spearman rank correlation between s_mem and s_softmax per layer, computed on a held-out calibration set of 512 sequences of length 4096).

**Why this design.** We chose a set-intersection ranking loss over a differentiable soft-rank (e.g., ListNet-style) because the intersection is exactly the quantity MemOnce's cross-layer index reuse consumes at inference, accepting the cost that the loss is non-differentiable through the argsort and must be optimized via the calibrated scores s_cal rather than the raw s_mem. We chose a single per-layer monotone calibration scalar gamma_l over a per-query temperature because the bounded-KV-error argument predicts a layer-level, query-independent scale factor (the reconstruction error is shared by all queries attending to the same decoded KV cache), accepting the cost that query-specific miscalibration is not corrected; the Phase 6 rank-correlation gate makes this assumption testable at inference and falls back to per-layer indices when it fails. We chose to anchor the reference ordering to the full-cache softmax attention of the same model rather than to an external teacher, because the target invariant is self-consistency between MemOnce's own two attention paths, accepting the cost that we cannot claim alignment to an oracle attention. A domain expert would not describe this as a variant of GPTQ (E2) or Radar (E3): GPTQ quantizes weights with asymmetric calibration and Radar evicts tokens by dynamic search, whereas RC-MemOnce leaves weights and token set untouched and instead enforces an ordering invariant between a reconstructed-KV attention and the full-cache softmax attention it replaces.

**Why it measures what we claim.** The set-intersection overlap |rank_ref ∩ rank_mem|/K is the computational quantity that operationalizes the motivation-level concept of ranking consistency: it measures ranking consistency because the assumption that the monotone calibration gamma_l is query-independent makes the calibrated top-k equal to the raw top-k after a shared monotone transform, so set overlap is invariant to gamma_l; this assumption fails when reconstruction error is query-dependent (e.g., a query whose dot-product with a badly reconstructed key flips ordering), in which case the overlap reflects reconstruction noise rather than surrogate-attention ordering error. The calibration scalar gamma_l measures the layer-level monotone distortion between surrogate and softmax scores because the assumption that dot-product scores are the sufficient statistic for token importance makes any monotone rescaling rank-preserving; this assumption fails when attention logits are dominated by a shared bias term across tokens, in which case gamma_l absorbs the bias instead of the reconstruction-induced scale. The bounded-KV-error condition (per-token ||k_t - k_hat_t|| <= epsilon) measures the regime in which ranking consistency is attainable because the assumption that the softmax score gap between the K-th and (K+1)-th token exceeds 2*epsilon*||q||/sqrt(d) makes the top-k stable under reconstruction; this assumption fails when the score spectrum is flat near rank K, in which case the overlap metric reflects ties rather than genuine alignment. The per-layer Spearman rank correlation rho_l measures the query-independence assumption behind gamma_l because the assumption that reconstruction error is shared across queries implies rho_l is high and stable across queries; this assumption fails when rho_l falls below rho_min = 0.7, in which case the Phase 6 gate falls back to per-layer indices and the overlap gain is attributed to the ranking loss rather than to calibration.

## Contribution

["RC-MemOnce: a ranking-consistency regularizer and per-layer monotone calibration for MemOnce's memory sparse attention that enforces top-k retrieval indices to coincide with the softmax-attention top-k under bounded KV reconstruction error.", "A design principle for efficient transformer architectures: surrogate-attention branches (MemOnce's memory sparse attention, ToST's token-statistics attention) should be trained with an explicit ordering-preservation signal against the full-cache softmax attention they replace, rather than only with reconstruction or language-model loss.", 'An evaluation protocol that measures top-k index overlap between surrogate and softmax attention across layers and query positions, isolating ranking drift from reconstruction error.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Long-context continual language modeling on a fixed subset of PG-19 (512 sequences of length 4096 tokens for training, 128 for validation, 128 for test) | Stresses KV-cache growth and memory retrieval; fixed context windows make the compute budget pre-registerable |
| Primary metric | Top-K retrieval overlap with full-cache softmax (K=64), reported overall and stratified by reference score-spectrum density near rank K (dense: gap between rank-K and rank-(K+1) softmax scores < 0.01; separated: gap >= 0.01) | Directly measures the invariant we enforce and tests the predicted concentration of gains |
| Secondary metric | Per-layer Spearman rank correlation rho_l between s_mem and s_softmax, and per-query reconstruction error ||q_q·(k_t - k_hat_t)||/sqrt(d) | Tests the load-bearing query-independence assumption directly |
| Baseline | Unmodified MemOnce | Tests whether ranking consistency is already implicit |
| Baseline | MemOnce + per-query temperature | Tests layer-level vs query-level calibration |
| Baseline | MemOnce + ListNet soft-rank loss | Tests set-intersection vs differentiable ranking |
| Ablation-of-ours | RC-MemOnce without gamma calibration (lambda_cal=0) | Isolates calibration from ranking loss |
| Ablation-of-ours | RC-MemOnce without the Phase 6 rank-correlation gate (always use cross-layer reuse) | Tests whether the gate is necessary to avoid silent dependence on layer invariance |

### Why this setup validates the claim

The central claim is that MemOnce's reconstructed-KV sparse retrieval orders tokens differently from the full-cache softmax attention it replaces, and that a ranking-consistency loss plus a layer-level monotone calibration scalar closes that ordering gap. The primary metric — top-K overlap between the calibrated surrogate ranking and the full-cache softmax ranking — is exactly the invariant the method targets, so a gain on it is not a proxy but the quantity itself. Unmodified MemOnce tests whether the ordering gap exists at all: if overlap is already near ceiling, the premise is false. The per-query-temperature baseline tests the sub-claim that calibration should be layer-level and query-independent; if a query-specific temperature matches or beats gamma_l, our design rationale is wrong. The ListNet baseline tests the sub-claim that set-intersection is the right loss despite non-differentiability; if soft-rank wins, our design tradeoff is misjudged. The no-calibration ablation isolates how much of any gain comes from the ranking loss alone versus the calibration scalar, so a uniform gain cannot be misattributed. The per-layer Spearman correlation rho_l and the per-query reconstruction error directly test the load-bearing assumption: we regress overlap on per-query reconstruction error and on rho_l; if overlap is uncorrelated with reconstruction error, the assumption that error drives ordering drift is unsupported; if rho_l is low on layers where the gate fires, the gate is doing real work and the cross-layer reuse is not silently relying on an untested layer-invariance property.

### Expected outcome and causal chain

**vs. Unmodified MemOnce** — On a case where a decoded KV cache has reconstruction error that shifts dot-product scores near the K-th rank boundary, the baseline retrieves a token set whose top-K diverges from the full-cache softmax ordering because its sparse indices are computed from raw reconstructed scores with no ordering constraint. Our method instead penalizes exactly this set divergence during training and rescales scores monotonically, so we expect a clear overlap gain concentrated on queries whose reference score spectrum is dense near rank K, with near-parity on queries whose top-K is well separated. We additionally expect the overlap gain to correlate positively with per-query reconstruction error, which is the diagnostic that tests the load-bearing assumption.

**vs. MemOnce + per-query temperature** — On a case where reconstruction error is shared across all queries attending to the same decoded cache, the per-query baseline fits a separate temperature per query and overfits query-specific noise, leaving the layer-level ordering distortion uncorrected. Our method instead applies one monotone scalar per layer matching the shared error structure, so we expect our overlap to exceed the per-query baseline on layers with high shared reconstruction error (high rho_l) while tying on layers where error is genuinely query-specific (low rho_l, where the Phase 6 gate falls back to per-layer indices).

**vs. MemOnce + ListNet soft-rank loss** — On a case where the calibrated scores are already close to the reference ordering, the soft-rank baseline optimizes a smooth surrogate that can drift the top-K set even as the surrogate loss decreases, because it does not directly optimize the set the retrieval step consumes. Our method instead optimizes the exact set-intersection quantity, so we expect a modest but consistent overlap advantage, largest on queries near the K boundary where surrogate-to-set mismatch bites hardest.

**vs. RC-MemOnce without gamma calibration (lambda_cal=0)** — On a case where the ranking loss alone is optimized, the calibrated scores s_cal are not rescaled and the layer-level monotone distortion remains, so we expect the no-calibration ablation to recover most of the ranking-loss gain but to lose the calibration-specific component on layers with high rho_l; the gap between full RC-MemOnce and this ablation isolates the calibration contribution.

**vs. RC-MemOnce without the Phase 6 rank-correlation gate** — On a case where rho_l is low (query-specific reconstruction error), always using cross-layer reuse forces the shared index set onto layers where the assumption fails, so we expect the no-gate ablation to underperform full RC-MemOnce on low-rho_l layers and to match it on high-rho_l layers; this tests whether the gate is necessary to avoid silent dependence on layer invariance.

### What would falsify this idea

If the overlap gain over unmodified MemOnce is uniform across queries rather than concentrated where the reference score spectrum is dense near rank K, or if per-query temperature matches gamma_l, or if overlap is uncorrelated with per-query reconstruction error and rho_l, then the ranking-consistency and layer-level-calibration claims are wrong. A no-calibration ablation matching the full method would likewise falsify the calibration component.

## References

1. MemOnce: A Scalable Memory Architecture for Large Language Models
