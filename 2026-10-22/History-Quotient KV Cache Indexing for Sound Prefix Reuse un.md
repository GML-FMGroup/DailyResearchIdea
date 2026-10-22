# History-Quotient KV Cache Indexing for Sound Prefix Reuse under Drop/Repos Edits

## Motivation

Cache Engineering for Agents has just made drop/repos first-class KV operations, but it explicitly leaves open how a serving system should match caches when the same token prefix can correspond to different KV states depending on the edit history. This is the recurring meta-gap across the paper-trees: MEPIC assumes chunk KV is reusable at arbitrary positions via PE adjustment plus first-block recomputation, AgentFold assumes folded summaries remain valid prefixes, and both implicitly rely on prefix matching — yet neither defines when two reachable cache states are actually interchangeable. The root cause is structural: once drop removes rows from attention and repos rewrites their position encodings, the cache is a function of the operation history, not of the token sequence, so token-prefix equality is neither necessary nor sufficient for state equality.

## Key Insight

Drop/repos operations form a rewriting system whose reachable cache states are exactly characterized by a canonical normal form of (retained original token IDs, surviving attention mask, and monotone-in-order repositioned coordinates), so two caches are interchangeable iff their normal forms coincide — an invariant that token-prefix matching cannot see and that positional-encoding adjustment alone cannot restore.

## Method

**What it is.** We introduce **HQ-Index (History-Quotient Index)**, a serving-layer cache index. Its input is any reachable cache state produced by a sequence of drop/repos operations over a base token stream; its output is a canonical representative `canon(state)` plus a hash key, such that two states are declared reusable iff their canonical keys agree. HQ-Index sits between the agent's edit API and the KV page table, and never re-prefills retained rows.

**Load-bearing assumption (stated explicitly).** HQ-Index's soundness rests on the following assumption, which we name **A**: *attention output over a set of live KV slots depends only on (token id, surviving mask, relative position order) **and** the per-slot absolute rotary phase offset `delta_i` recorded at edit time, so that two edited cache states with identical canonical keys are interchangeable after the load path applies the exact per-row re-rotation `R(P_new - P_old)` to both the stored K rows and the corresponding Q rows.* Under RoPE (Su et al., 2021), attention depends on absolute per-token rotation phases; the relative-offset invariance of dot products holds only for the unmodified relative offset between a query and key. Therefore, once `repos` rewrites `P[i]`, the stored K rows carry stale rotations that must be re-rotated by `(P_new - P_old)` before any reuse, and the corresponding Q rows must be re-rotated consistently because attention scores are query-dependent. The canonical key therefore hashes `(chunks.token_ids, normP, delta)` — not `normP` alone — so that two states with equal keys carry identical absolute phases up to the global shift that `normP` normalizes away. The load path performs the per-row re-rotation of both K and Q as part of the reuse operation, not as an afterthought. For models with relative or NoPE attention, `delta` is identically zero and the re-rotation is the identity; the soundness claim is then unconditional. For RoPE models, the soundness claim is conditional on the re-rotation being applied, and we verify this empirically (see experiment, equivariance test).

**How it works.**

```pseudocode
# State representation after any edit history H:
#   state = (T, A, P, delta) where
#     T[i] = original token id at cache slot i
#     A[i] in {0,1}  = attention-surviving flag (drop sets A[i]=0)
#     P[i] in Z     = repositioned coordinate (repos writes P[i])
#     delta[i] in Z = P[i] - original_position[i], recorded per slot (I3)
# Invariant enforced by HQ-Index on every op:
#   (I1) A is monotone-nondecreasing over live slots in insertion order
#   (I2) P restricted to live slots is strictly increasing
#   (I3) P[i] = original_position[i] + delta_i, delta_i recorded per slot

function CANON(state):
    live = [i for i in slots if A[i] == 1]          # drop survivors
    # (a) collapse runs of identical (T, A) into a chunk descriptor
    chunks = RUN_LENGTH_ENCODE(live, key=lambda i:(T[i],A[i]))
    # (b) normalize coordinates: subtract min live P so canon is
    #     translation-invariant to a global repos shift
    pmin = min(P[i] for i in live)
    normP = tuple(P[i]-pmin for i in live)
    # (c) absolute phase offset per live slot (folds delta_i into key)
    delta = tuple(P[i]-original_position[i] for i in live)
    # (d) canonical key = hash over (chunk token ids, normP shape, delta)
    return key = HASH(chunks.token_ids, normP, delta)

# Serving lookup:
function LOOKUP(query_state):
    k = CANON(query_state)
    if k in index: return index[k].pages       # sound reuse
    else:          return MISS                    # must prefill

# Load path (applied on every HIT, before attention):
function LOAD(pages, delta):
    for each live slot i:
        K[i] = ROTATE(K[i], delta[i])   # re-apply absolute phase
        Q[i] = ROTATE(Q[i], delta[i])   # consistent query rotation
    return pages

# Index maintenance (O(#live) per op, amortized):
function ON_DROP(i):  A[i]=0; INVALIDATE_IF_KEY_CHANGES()
function ON_REPOS(i,d): P[i]+=d; delta[i]+=d; CHECK(I2); INVALIDATE_IF_KEY_CHANGES()
```

Hyperparameters: run-length chunk granularity = page size (16 tokens, matching MEPIC's paged layout); hash = 128-bit; invalidation is lazy (recompute key only on lookup, cached per state version counter).

**Why this design.** We chose a *canonical-form hash* over a *learned cache-state embedding* because the equivalence relation must be exact (a false positive causes silent attention corruption, not graceful degradation), accepting the cost that HQ-Index cannot generalize across genuinely different token streams that a learned metric might cluster. We chose *translation-normalized coordinates* (subtract min live P) over *raw positions* because Cache Engineering's repos is a relative shift, so two histories differing only by a global offset produce identical attention geometry, accepting the cost that absolute-position-dependent rotary phases must be re-applied at load time (a cheap per-row rotation, not a prefill). We chose *lazy invalidation keyed on a version counter* over *eager reindexing per op* because agents issue many drop/repos between lookups, accepting the cost that a burst of lookups after a long edit chain pays one O(#live) canonicalization. A domain expert would not describe this as a variant of MEPIC: MEPIC *assumes* chunk KV is position-independent and recomputes the first block to make it so; HQ-Index *proves which states are equivalent* and refuses reuse otherwise, so it is a soundness layer MEPIC lacks, not a recomputation variant.

**Why it measures what we claim.** The canonical key `HASH(chunks.token_ids, normP, delta)` measures *state interchangeability* because the named assumption A is that attention output depends on live slots only through (token id, surviving mask, relative position order, and absolute phase offset `delta`), i.e., the model's attention is equivariant to a global coordinate shift on live slots once the per-row re-rotation `R(delta)` is applied at load; this assumption fails when a model uses absolute positional embeddings with no re-rotation applied at load, in which case the key reflects *geometric* equivalence but the loaded KV carries stale phases — the failure mode F. The `normP` strictly-increasing check measures *reachability under repos* because the assumption is that repos only ever applies monotone shifts that preserve order; this fails when an agent repositions a slot before an earlier survivor, in which case the key reflects an *invalid* state that HQ-Index flags as a miss rather than silently reusing. The run-length chunk descriptor measures *page-level shareability* because the assumption is that page-aligned chunks are the unit of KV storage (as in MEPIC); this fails under sub-page drops, in which case the descriptor reflects a *finer* granularity than the page table can share, and HQ-Index conservatively misses.

## Contribution

(1) A canonical-form equivalence relation and normal-form construction over drop/repos operation histories, with an explicit statement of the invariants (surviving mask monotonicity, strictly-increasing repositioned coordinates, translation-normalized coordinates) that make the quotient well-defined. (2) HQ-Index, a serving-layer index that maps any reachable cache state to a canonical key and answers reuse queries soundly, with lazy version-counter invalidation and page-aligned chunk descriptors. (3) A design principle: sound KV reuse under agent edits requires indexing the *operation-history quotient*, not the token prefix, and positional-encoding adjustment is a load-time rotation that must be paired with a canonical key to avoid false-positive reuse.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Agent edit traces with drop/repos histories | Exercises the exact operation space HQ-Index reasons about |
| Primary metric | Sound-reuse rate at matched attention fidelity | Directly measures correct interchangeability, not speed alone |
| Baseline | Prior context-engineering trim/summarize | Tests re-prefill cost we claim to avoid |
| Baseline | MEPIC position-independent caching | Tests whether soundness layer adds reuse beyond recomputation |
| Baseline | Raw-position prefix cache | Tests translation-invariance claim under repos |
| Ablation-of-ours | HQ-Index without normP (raw positions) | Isolates translation normalization's contribution |
| Ablation-of-ours | HQ-Index without delta in key (normP only) | Isolates absolute-phase folding's contribution |
| Equivariance test | Global repos shift + re-rotation vs unshifted | Directly tests assumption A across layers/models |

**Matched attention fidelity** is defined as: max per-token output L2 divergence between the reused state (after load-path re-rotation) and a full re-prefill of the same live-slot state is ≤ 1e-3 relative to the re-prefill output norm, measured at every layer's attention output. A reuse event counts as *sound* only if this threshold is met; otherwise it is counted as a false positive and reported separately.

**Equivariance test protocol.** Take a fixed live-slot state from the trace set, apply a global repos shift of `+d` for `d ∈ {1, 4, 16, 64}`, re-rotate K and Q by `d` at load, and measure max/mean L2 divergence of attention output against the unshifted state across all layers for two models (one RoPE-based, one NoPE or relative-position). Report the divergence distribution (mean, 95th percentile, max) per layer and per model. This directly bounds assumption A.

**Trace-analysis measurement.** On the agent edit traces, report the fraction of post-edit state pairs that are equivalent-up-to-global-shift (i.e., share a canonical key after normP normalization) versus genuinely distinct. This fraction is reported up front as a motivating measurement; if it is small, the significance claim is bounded accordingly.

### Why this setup validates the claim

The central claim is that HQ-Index declares reuse exactly when two edited cache states are attention-equivalent, and never otherwise. The dataset must therefore contain edit histories whose states are equivalent up to a global repos shift but differ in raw coordinates — that is the only regime where the claim has teeth. MEPIC tests the sub-claim that soundness adds reuse beyond position-independent recomputation: if MEPIC already recovers the same states, our index is redundant. The trim/summarize baseline tests the sub-claim that avoiding re-prefill on retained rows matters at all. The raw-position prefix cache tests whether translation-invariance is a real source of misses. The ablation without normP isolates whether the gain comes specifically from coordinate normalization rather than from hashing or chunking. The ablation without delta isolates whether folding absolute phase offsets into the key is necessary for soundness under RoPE. The equivariance test directly measures assumption A: if re-rotated global shifts produce attention divergence above the fidelity threshold, the soundness claim is bounded to NoPE/relative models. Sound-reuse rate at matched attention fidelity is the right metric because a faster but corrupted reuse would inflate any throughput measure; matching fidelity forces the metric to count only genuinely interchangeable states.

### Expected outcome and causal chain

**vs. Prior context-engineering trim/summarize** — On a case where an agent drops a mid-history span and repositions a later block, the baseline invalidates the whole cached prefix and re-prefills every retained token because its mechanism has no notion of a surviving-slot state. Our method instead canonicalizes the post-edit state and hits the index on the retained rows because the drop/repos history maps to an already-keyed canonical form. We expect a large reuse-rate gap concentrated on edit-heavy traces and near-parity on append-only traces.

**vs. MEPIC position-independent caching** — On a case where two histories reach the same live-slot geometry via different repos sequences, MEPIC recomputes the first block to force position independence and may still miss when the chunk boundary does not align with the edit. Our method instead proves the two states share a canonical key and reuses without recomputation because normP collapses the global offset exactly and delta folds the absolute phase. We expect our advantage to appear specifically on repos-heavy, page-aligned subsets and to vanish where edits are absent.

**vs. Raw-position prefix cache** — On a case where a global repos shift moves all live slots by a constant, the raw-position baseline misses because its keys embed absolute coordinates that changed. Our method instead subtracts min live P and hits because the shift is normalized away. We expect a reuse gap that scales with the frequency of global shifts and approaches zero when no repos occurs.

**vs. HQ-Index without normP (ablation)** — On a case with a pure global repos shift, the ablation misses because raw coordinates enter the hash. Our method hits because normP is translation-invariant. We expect the ablation to recover most of the gain on drop-only traces but lose it on repos traces, isolating normalization as the causal contributor.

**vs. HQ-Index without delta in key (ablation)** — On a RoPE model with a repos shift, the ablation declares a hit on states whose absolute phases differ, causing attention divergence above the fidelity threshold (false positive). Our method includes delta in the key and applies re-rotation at load, so it either hits soundly or misses. We expect the ablation to show false-positive reuse on RoPE models and near-parity on NoPE models, isolating delta folding as necessary for soundness.

**vs. Equivariance test** — For the NoPE/relative model, we expect max divergence below 1e-3 across all shifts and layers, confirming assumption A unconditionally. For the RoPE model with re-rotation applied, we expect max divergence below 1e-3 if the re-rotation is exact; any divergence above threshold bounds the soundness claim to models where re-rotation is exact or to NoPE/relative models.

### What would falsify this idea

If the reuse-rate gain over MEPIC and the raw-position baseline is uniform across edit types rather than concentrated on repos-heavy and drop-heavy subsets, the claimed causal mechanism (translation normalization and surviving-slot canonicalization) is wrong. Likewise, if HQ-Index's canonical key ever declares reuse on a state pair that a fidelity check shows to be attention-inequivalent, or if the equivariance test shows re-rotated global shifts exceed the fidelity threshold on RoPE models, the soundness claim is falsified or bounded.

## References

1. Cache Engineering for Agents
2. MEPIC: Memory Efficient Position Independent Caching for LLM Serving
3. AgentFold: Long-Horizon Web Agents with Proactive Context Management
