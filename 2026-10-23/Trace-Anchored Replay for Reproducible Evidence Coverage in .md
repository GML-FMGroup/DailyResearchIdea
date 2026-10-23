# Trace-Anchored Replay for Reproducible Evidence Coverage in Live-Web Deep Search

## Motivation

Live-web deep-search benchmarks such as OpenDS-World and BrowseComp-style evaluations cannot isolate the retriever's contribution from LLM reasoning because every run re-queries a dynamic web whose index, ranking, and document contents drift between runs; BrowseComp-Plus fixes this by replacing the live web with a static corpus, but in doing so removes browser interaction and interactive-access tasks that OpenDS-World was designed to test. The structural root cause is that evidence coverage — the set of documents an agent actually surfaces — is a function of both retriever behavior and an uncontrolled environment, so no amount of repeated evaluation can attribute variance to the retriever versus the reasoner. What is missing is a record of the environment's response surface at the moment of each action, so that the same evidence coverage can be re-instantiated on demand without freezing the web.

## Key Insight

If every retriever and browser response is captured as a content-addressed snapshot keyed to the exact action that produced it, then evidence coverage becomes a deterministic function of the trace rather than of the live web, letting one replay branch the retriever while holding the environment fixed and thereby making retriever-versus-reasoner attribution an intervention rather than a correlation.

## Method

**What it is.** We introduce **TARSE** (Trace-Anchored Replay of Search Evidence), a wrapper around OpenDS-World that records each agent step — search query, browser action, and the full retrieved document snapshot — into a content-addressed, replayable trace, and provides a replay engine that re-executes a recorded trajectory with a *substituted* retriever while serving all environment responses from the snapshot store. Inputs: an OpenDS-World task and an agent policy. Outputs: a deterministic trace plus, for each replay branch, a per-step evidence-coverage vector.

**Load-bearing assumption (stated explicitly).** TARSE's replay substitutes a retriever by reranking a *fixed, recorded candidate pool*; the estimand `delta_R` is therefore the **reranker's contribution to evidence coverage within that pool**, not the retriever's contribution as if it had issued the search against the live web. The pool is produced by the original retriever (plus, at record time, a widened union pool described below), so any substituted retriever is scored only on documents already surfaced; its ability to retrieve documents outside the pool is structurally invisible. We make this explicit and bound it with a measured **pool-recall ceiling** rather than claiming retriever-invariant attribution.

**How it works.**

```pseudocode
# Phase 1: RECORD (live web, unchanged agent behavior)
trace = []
for step t in 1..T:
    a_t = policy(query_history, observations)          # search query OR browser action
    if a_t is SEARCH:
        # widened union pool: original retriever top-20 + dense top-50 + sparse top-50
        raw_orig  = live_search(a_t.query, top_k=20)          # original retriever
        raw_dense = live_search_dense(a_t.query, top_k=50)    # heterogeneous retriever A
        raw_sparse= live_search_sparse(a_t.query, top_k=50)   # heterogeneous retriever B
        raw = union_dedup(raw_orig, raw_dense, raw_sparse)    # union pool, dedup by URL
        pool_recall_ceiling_t = |gold_spans_in(raw)| / |gold_spans|  # measured per step
    else:
        raw = live_browser(a_t.url, a_t.element_selector)  # live page fetch
    snap_t = SnapshotStore.put(hash(raw), raw, fetch_time=now())
    trace.append(Step(t, a_t, snap_t.hash, coverage_meta(raw), pool_recall_ceiling_t))

# Phase 2: REPLAY (no live web; retriever is the intervention variable)
def replay(trace, retriever_R):
    obs = []
    for t, step in enumerate(trace):
        if step.a_t is SEARCH:
            # substitute retriever; environment responses come from store
            cand = SnapshotStore.get(step.snap_t.hash)   # union pool from record time
            docs = retriever_R.rerank(step.a_t.query, cand)   # rerank cached union pool
            obs.append(docs)
        else:
            obs.append(SnapshotStore.get(step.snap_t.hash))  # exact replay
    return obs

# Phase 3: ATTRIBUTE (evidence coverage per branch)
for R in {original, R_ablate_1, R_ablate_2}:
    cov_R = coverage(replay(trace, R))   # fraction of gold evidence spans surfaced
    delta_R = cov_original - cov_R      # reranker contribution within fixed pool
    # report pool-recall ceiling as explicit bound on delta_R
    ceiling_R = mean_t(pool_recall_ceiling_t)   # upper bound on achievable coverage
```

Hyperparameters: `top_k=20` for the original retriever, `top_k=50` each for the dense and sparse union retrievers (chosen so the union pool has headroom beyond the original retriever's recall while keeping store size bounded; union is deduplicated by URL, yielding typically 60–90 unique candidates per search step). Snapshot retention = full document body plus a DOM hash for browser steps; replay branches reuse the identical query sequence so only the retriever varies. Storage budget: ~1.5 MB per search step (union pool of ~80 docs × ~20 KB average body) and ~2 MB per browser step (DOM + assets), so a 30-step trace is ~50 MB; GC policy: retain all snapshots for the 200-task evaluation (~10 GB total), evict browser-step assets older than 30 days. Replay cost per branch: ~0.2 s per search step for reranking 80 candidates, ~0.05 s per browser step for store lookup, so a 30-step branch replays in ~5 s on CPU.

**Why this design.** We chose content-addressed snapshots over a static curated corpus (the BrowseComp-Plus approach) because the corpus removes browser interaction and interactive-access tasks that OpenDS-World is built to test, accepting the cost that the snapshot store grows with trace length and must be garbage-collected. We chose to rerank the *cached candidate pool* rather than re-issue a live search on replay because re-issuing reintroduces the very web drift we are eliminating, accepting the cost that replay can only measure retriever behavior within the recorded candidate set — a retriever that would have surfaced a document never cached cannot be credited. We widened the recorded pool to a union of three heterogeneous retrievers (original top-20, dense top-50, sparse top-50) precisely to reduce this recall ceiling, and we report the pool-recall ceiling as an explicit bound on `delta_R`. We chose to keep the query sequence fixed across branches rather than let the policy re-plan, because re-planning conflates retriever substitution with policy adaptation, accepting the cost that we measure retriever contribution *conditional on the recorded trajectory* rather than in a fully re-planned rollout. Unlike a generic caching layer (e.g., a standard HTTP cache or a RAG document store), TARSE's snapshots are keyed to the *action that produced them* and carry coverage metadata, so replay is an intervention on the retriever, not merely a speedup; a domain expert would not describe this as a cache because the load-bearing object is the attribution delta, not the stored bytes.

**Why it measures what we claim.** The per-branch evidence-coverage delta `delta_R = cov_original - cov_R` measures the reranker's contribution to evidence coverage *within the recorded union pool* because we hold the query sequence, the cached candidate pool, and the reasoner's observations fixed and vary only the reranker; this assumption fails when the substituted retriever would have changed the *query* the policy issues (e.g., a retriever whose results steer the next query), in which case `delta_R` reflects retriever-induced policy drift rather than pure retrieval quality — we report the fraction of steps where the next query is unchanged as a diagnostic. The pool-recall ceiling `ceiling_R` measures the maximum coverage any reranker could achieve on the recorded pool because it is the fraction of gold evidence spans present anywhere in the union pool; this assumption fails when the substituted retriever would have surfaced a gold document outside the union pool, in which case `delta_R` is biased toward zero and we report the gap between `ceiling_R` and 1.0 as the explicit bound on attribution. The snapshot hash `snap_t.hash` measures environment stability because it is a content hash of the exact bytes served at step `t`; this assumption fails when a page's *rendered* content differs from its fetched bytes (client-side JS), in which case the hash reflects the pre-render payload, and we flag such steps. The `coverage_meta(raw)` field measures evidence coverage because it records which gold evidence spans from the task's human-verified evidence set appear in the retrieved documents; this assumption fails when a gold span is paraphrased rather than verbatim in the snapshot, in which case coverage under-counts, and we report both exact and fuzzy match rates.

## Contribution

(1) TARSE, a trace-anchored replay framework for live-web deep-search agents that records action-keyed, content-addressed snapshots and re-executes trajectories with a substituted retriever while serving all environment responses from the store, making retriever-versus-reasoner attribution an intervention rather than a correlation. (2) A design principle — action-keyed content addressing with coverage metadata — that preserves interactive-access and browser tasks which static-corpus benchmarks necessarily drop, while restoring determinism of evidence coverage. (3) An open replay engine and snapshot format compatible with OpenDS-World tasks, enabling controlled ablation of retriever contribution across frontier models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OpenDS-World (200 tasks) | Multi-step live-web tasks with verified evidence |
| Primary metric | Evidence-coverage delta per replay branch, reported alongside pool-recall ceiling | Isolates reranker contribution within fixed pool and bounds it explicitly |
| Baseline | Live re-query replay (no snapshot) | Tests whether drift, not retriever, drives deltas |
| Baseline | Static-corpus replay (BrowseComp-Plus style) | Tests whether interaction tasks are lost |
| Baseline | Full re-planned rollout with substituted retriever | Tests whether fixed-query assumption is load-bearing |
| Baseline | Standard HTTP-cache-plus-reranker (generic cache, no action-keyed coverage metadata) | Tests whether action-keyed coverage metadata yields attribution a generic cache cannot |
| Ablation-of-ours | TARSE without coverage_meta (hash-only replay) | Tests whether evidence attribution needs metadata |
| Ablation-of-ours | TARSE with original top-20 pool only (no union widening) | Tests whether pool widening reduces the recall ceiling and changes delta_R |
| Ablation-of-ours | Live-search-augmented replay (re-issue live search at replay, union with cached pool) | Bounds the bias from pool-recall ceiling by measuring delta_R when the pool is widened at replay time |

### Why this setup validates the claim

The central claim is that TARSE's `delta_R` is a *causal* measure of the reranker's contribution to evidence coverage within a fixed recorded pool, not a proxy contaminated by web drift or policy adaptation, and that the pool-recall ceiling explicitly bounds this attribution. The live re-query baseline tests the drift sub-claim: if replaying with fresh live searches produces deltas of similar magnitude to TARSE's snapshot-anchored deltas, then the snapshot store is doing no causal work and the whole design collapses to "just re-run the search." The static-corpus baseline tests the interaction sub-claim: if a fixed corpus reproduces TARSE's deltas on interactive-access tasks, then the snapshot mechanism is unnecessary overhead. The full re-planned rollout tests the fixed-query assumption: if allowing the policy to re-plan under a substituted retriever yields deltas indistinguishable from TARSE's fixed-query deltas, then the diagnostic fraction-of-unchanged-queries is not load-bearing and the simpler design suffices. The HTTP-cache-plus-reranker baseline tests whether action-keyed coverage metadata is what enables attribution: if a generic cache with the same reranker reproduces TARSE's deltas, then the action-keying and coverage metadata are doing no work. The hash-only ablation tests whether coverage metadata is what enables attribution or whether content hashes alone suffice. The original-top-20-only ablation and the live-search-augmented replay condition test the pool-invariance assumption directly: if widening the pool changes `delta_R` substantially, then the original top-20 pool was not a sufficient statistic and the ceiling bound is load-bearing. Evidence-coverage delta is the right metric because it is defined relative to a human-verified gold evidence set, so a null delta is interpretable as "this reranker change did not move evidence into the reasoner's view" rather than as noise on a downstream accuracy number, and the pool-recall ceiling makes explicit the fraction of gold evidence that no reranker could have surfaced from the recorded pool.

### Expected outcome and causal chain

**vs. Live re-query replay** — On a case where the gold evidence sits on a page whose search rank drifted between record and replay (a common occurrence for time-sensitive or frequently-updated pages), the live re-query baseline produces a different candidate pool at that step, so its coverage delta conflates reranker quality with web drift and becomes noisy or sign-flipped. Our method instead serves the exact recorded union pool from the snapshot store and reranks only that pool, so the delta reflects the reranker alone. We expect TARSE's deltas to be stable across replay runs (near-identical on re-execution) while the live baseline's deltas vary run-to-run, with the gap concentrated on drift-prone task subsets and parity on static-page subsets.

**vs. Static-corpus replay** — On an interactive-access task where the agent must click through a login wall, paginate, or follow a dynamically rendered link, the static-corpus baseline produces no valid observation because the corpus was built by crawling and does not contain the post-interaction DOM state, so the agent's trajectory breaks and coverage collapses to near zero on those steps. Our method instead replays the exact browser-step snapshot including the post-interaction DOM hash, so the trajectory remains intact and only search steps are intervened on. We expect a large coverage gap on interactive-access tasks and near-parity on pure-search tasks, which is the signature that snapshots (not the corpus) are carrying the interaction burden.

**vs. Full re-planned rollout** — On a case where the substituted retriever surfaces a plausible but wrong document early, the full re-planned rollout lets the policy issue a follow-up query steered by that wrong document, so the trajectory diverges and the measured delta reflects retriever-induced policy drift rather than retrieval quality. Our method instead holds the query sequence fixed, so the same wrong document is visible but the next query is unchanged, isolating the retrieval effect. We expect TARSE's deltas to be systematically smaller in magnitude than the re-planned rollout's on tasks where the fraction-of-unchanged-queries diagnostic is low, and the two to converge where that fraction is near one — confirming the fixed-query design is measuring what it claims.

**vs. Standard HTTP-cache-plus-reranker** — On a case where the substituted retriever needs to know which gold evidence spans were surfaced at each step, the generic cache stores bytes but not coverage metadata, so its delta cannot be computed per evidence span and collapses to a document-level overlap score that is insensitive to whether the gold span is inside the cached document. Our method carries `coverage_meta(raw)` keyed to the action, so the delta is computed over human-verified evidence spans. We expect TARSE to produce non-zero, interpretable deltas on tasks where the generic cache produces zero or uninterpretable deltas, with the gap concentrated on tasks whose gold evidence is a span within a larger document rather than the whole document.

**vs. TARSE with original top-20 pool only** — On a case where the substituted retriever would have surfaced a gold document that the original retriever ranked below 20, the top-20-only pool has a lower pool-recall ceiling, so `delta_R` is biased toward zero for that retriever. Our method widens the pool to a union of three heterogeneous retrievers, raising the ceiling. We expect the union pool to have a higher measured pool-recall ceiling than top-20-only on tasks with diverse retrieval characteristics, and `delta_R` to be larger in magnitude (less biased toward zero) on those tasks, with the gap between `ceiling_R` and 1.0 shrinking.

**vs. Live-search-augmented replay** — On a case where the substituted retriever's true top-k includes documents outside the recorded union pool, the live-search-augmented condition re-issues the live search at replay and unions it with the cached pool, so its pool-recall ceiling is higher but it reintroduces web drift. Our method holds the pool fixed. We expect the live-search-augmented condition to produce a higher ceiling and a larger `delta_R` on tasks where the recorded pool missed gold documents, and the two to converge where the recorded pool already contains all gold evidence — bounding the bias from the pool-recall ceiling.

### What would falsify this idea

If TARSE's coverage deltas were statistically indistinguishable from the live re-query baseline's deltas on drift-prone subsets, the snapshot store would be doing no causal work and the central claim that environment-fixing enables attribution would be false. Likewise, if the hash-only ablation matched full TARSE's deltas, then coverage metadata would be unnecessary and the attribution mechanism would reduce to plain caching. If the original-top-20-only ablation and the live-search-augmented condition produced deltas indistinguishable from full TARSE's, then the pool-recall ceiling would not be load-bearing and the union widening would be unnecessary.

## References

1. Deep Search in Real-World Web Environments
2. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents
3. BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent
4. HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models
5. LAB-Bench: Measuring Capabilities of Language Models for Biology Research
6. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
7. BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval
8. GPQA: A Graduate-Level Google-Proof Q&A Benchmark
