# Query-Conditioned Page-Memory Routing with Verifiable Evidence Coverage for Multi-Page Document VQA

## Motivation

Multi-page document vision-language models still fail at transitive, unbounded cross-page reasoning because the dominant cross-page fusion designs are structurally bounded-window: DocSEAM fuses each page only with its immediate neighbors via center-anchored masked self-attention, so the receptive field is fixed at radius one and information from page k cannot reach page k+5 without repeated, lossy re-encoding. The field's trajectory — toward thousand-page haystacks (Document Haystacks) and native document-level models (Docopilot) — makes this a structural bottleneck rather than a tuning issue: bounded-window fusion cannot represent long-range evidence chains, and no existing branch exposes a checkable criterion for whether the evidence needed by a query was ever assembled.

## Key Insight

If page-level memory slots are iteratively re-routed by a query-conditioned operator while the emitted center-page token stream is held to a fixed count, then the effective cross-page receptive field grows with the number of propagation steps rather than with page distance, and the same routing weights that move evidence also define a per-query evidence-coverage mask whose support is auditable against retrieved query terms.

## Method

**What it is.** Q-PMR (Query-conditioned Page-Memory Routing) is a drop-in replacement for DocSEAM's fusion stage. It takes as input a stack of per-page visual token streams `X_p ∈ R^{N×d}` (N tokens per page, unchanged from DocSEAM), a query token embedding `q`, and a sparse page-relevance graph `G`; it outputs an enriched center-page token stream `Y_p ∈ R^{N×d}` with the *same* token count N, plus a per-query evidence-coverage mask `M_q` over page regions. **Load-bearing assumption (stated explicitly):** a single d-dimensional mean-pooled slot per page, propagated by query-conditioned softmax routing over K=3 steps and written back via residual cross-attention, can carry the fine-grained, spatially-localized evidence (a specific table cell, a figure caption, a signature field) needed for multi-hop document VQA without loss, such that routing mass on that slot is a faithful proxy for whether the query's evidence was actually assembled. This assumption is load-bearing and is tested directly in the experiment via the coverage-mask precision/recall calibration against human-annotated evidence regions and the LM-attention attribution on Δ_p; if it fails, the coverage-mask sub-claim is dropped and the contribution reduces to query-conditioned slot propagation.

**How it works.**

```pseudocode
# Phase 1: initialize page-level memory slots
for each page p:
    S_p^0 = MeanPool(X_p)                      # one slot per page, dim d
    # DocSEAM local triplet fusion is retained as the local prior
    L_p   = CenterAnchoredMaskedAttn(X_{p-1}, X_p, X_{p+1})   # DocSEAM op, unchanged

# Phase 2: query-conditioned iterative propagation (K = 3 steps)
for k in 1..K:
    for each page p:
        # routing weights over pages within G-neighborhood and beyond
        a_{p,r} = softmax_r( (W_q q)^T (W_k S_r^{k-1}) / sqrt(d) )   # r ranges over ALL pages in G
        # retain DocSEAM's directional salience gate, now query-conditioned
        g_p     = sigmoid(W_g [S_p^{k-1}; q])
        S_p^k   = g_p * S_p^{k-1} + (1-g_p) * Σ_r a_{p,r} S_r^{k-1}

# Phase 3: write back into token stream WITHOUT changing token count
for each page p:
    # cross-attention: N center tokens attend to K+1 page slots
    Δ_p = CrossAttn(Q=X_p, K=[S_p^0..S_p^K], V=[S_p^0..S_p^K])
    Y_p = L_p + Δ_p                            # residual; shape N×d preserved

# Phase 4: evidence-coverage mask from the SAME routing weights
for each page p, token n:
    c_{p,n} = Σ_k Σ_r a_{p,r}^{(k)} · 1[token n of page p was attended by r at step k]
    M_q[p,n] = 1[ c_{p,n} ≥ τ ]                # τ = 0.5, fixed
```

**Why this design.** We chose iterative slot-level propagation over DocSEAM's single-shot triplet fusion because DocSEAM's center-anchored masked self-attention makes each page representation a bounded-window function of its two neighbors — a structural property of the attention mask, not a capacity limit — so no amount of width or depth inside that operator can carry evidence across a hop distance greater than one; accepting the cost that we add K=3 sequential passes whose latency grows linearly in K rather than being a single fused op. We chose to keep the emitted token count exactly N (the DocSEAM contract) and inject propagated evidence through a residual cross-attention over K+1 slots rather than by expanding the token stream, because downstream LM positional budgets and the frozen-LM fine-tuning recipe in DocSEAM depend on a fixed per-page token count; accepting the cost that slot-level propagation compresses page content into d-dimensional summaries and may lose fine-grained layout detail that a token-expanding design would retain. We chose query-conditioned routing weights `a_{p,r}` over a query-agnostic page-graph attention because the coverage mask must be *per-query* to be checkable against query terms — a query-agnostic propagation would produce a single mask that cannot falsify 'this query's evidence was never assembled'; accepting the cost that the router must be re-run per query, which rules out caching one propagation pass across a multi-turn session. We chose a fixed threshold τ=0.5 on accumulated routing mass over a learned coverage classifier because a learned classifier reintroduces exactly the calibration problem the mask is meant to avoid; accepting the cost that τ is a hyperparameter tuned on a held-out split rather than adapted per document. A domain expert would not describe this as 'DocSEAM + RAG': Document Haystacks' V-RAG retrieves pages from an external index and feeds them to the LM, whereas Q-PMR never selects a page subset — every page's slot participates in every propagation step, and the routing weights are an internal attention quantity, not a retrieval score; the coverage mask is a read-out of that internal quantity, not a retrieval result.

**Why it measures what we claim.** The routing mass `a_{p,r}^{(k)}` measures *evidence reachability from page r to page p at hop k* because the softmax over all pages in G is normalized over the full page set, so a page r whose content is irrelevant to q receives vanishing mass and contributes negligibly to `S_p^k`; this assumption fails when two pages share surface-level lexical overlap with q but only one carries the answer (e.g., a table of contents and the body section both mention the query terms), in which case `a_{p,r}` reflects lexical salience rather than evidentiary relevance. The coverage mask `M_q[p,n] = 1[Σ_k Σ_r a_{p,r}^{(k)} · 1[token n attended]] ≥ τ` measures *whether the evidence needed by q was assembled into page p's representation* because accumulated routing mass above τ is assumed to imply that some page-region contributing to the answer was actually propagated; this assumption fails when the answer requires a *negation* or *absence* (e.g., 'which invoice is missing a signature'), in which case high coverage on present regions does not certify that the absence was checked, and `M_q` reflects presence-of-mention rather than sufficiency-of-evidence. The residual write-back `Y_p = L_p + Δ_p` measures *that propagated evidence reaches the LM's input surface* because the cross-attention output Δ_p is added to the same N tokens the LM consumes, so any change in the LM's answer attributable to cross-page evidence must flow through Δ_p; this assumption fails when the LM's attention over Y_p is dominated by L_p (the local triplet term), in which case Δ_p's contribution is masked and `Y_p` reflects local context despite nonzero propagation.

## Contribution

['A query-conditioned page-memory routing operator that replaces bounded-window cross-page fusion with K-step iterative propagation over page-level slots while provably preserving the per-page visual token count, enabling cross-page evidence transport at hop distances independent of K.', 'An evidence-coverage mask derived from the same routing weights used for propagation, providing a per-query, per-region audit signal that can be checked against retrieved query terms without a separate classifier or calibration step.', 'A design principle — that cross-page receptive field should scale with propagation steps rather than with page adjacency — instantiated as a drop-in module for DocSEAM-style architectures and evaluated on multi-page document VQA.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DocHaystack-1000 / InfoHaystack-1000 | Multi-page evidence spans beyond one-hop neighbors |
| Primary metric | Answer accuracy stratified by hop distance | Isolates the predicted long-hop failure mode |
| Baseline | DocSEAM (9B, frozen fusion) | Tests single-shot triplet fusion limits |
| Baseline | V-RAG + LMM | Tests external retrieval vs internal propagation |
| Baseline | Docopilot (native document-level MLLM) | Tests dataset-scale native understanding |
| Baseline | Full-attention-over-all-page-tokens model (same token budget) | Tests whether Q-PMR beats a strong non-DocSEAM long-range baseline |
| Baseline | Learned page-graph transformer with K=3 steps but no query conditioning | Tests whether query-conditioning drives the gain |
| Ablation-of-ours | Q-PMR with query-agnostic routing | Tests whether query-conditioning drives the gain |
| Ablation-of-ours | Q-PMR with K=1,2,3,4,5 | Measures hop-distance at which coverage mass saturates |
| Calibration metric | AUROC of coverage-vs-correctness on held-out queries | Tests standalone utility of M_q beyond accuracy |
| Calibration metric | Precision/recall of M_q against human-annotated evidence regions | Tests whether M_q certifies token-level evidence coverage |
| Attribution metric | Fraction of answer-token attention mass on Δ_p vs L_p per hop stratum | Tests whether write-back reaches the LM |

### Why this setup validates the claim

The central claim is causal and narrow: DocSEAM's center-anchored masked attention makes each page representation a bounded-window function of its two neighbors, so evidence requiring more than one hop cannot be assembled regardless of operator width. The design must therefore isolate *hop distance* as the independent variable and hold everything else fixed. Stratifying accuracy by hop distance on DocHaystack-1000/InfoHaystack-1000 does exactly this: it converts a vague "better document understanding" claim into a prediction about where the gap should appear. DocSEAM is the load-bearing baseline because it shares our encoder, token contract, and frozen-LM recipe — any gain over it on multi-hop subsets cannot be attributed to capacity or training data. V-RAG tests the competing hypothesis that external retrieval, not internal propagation, is the right mechanism; if V-RAG matches us on multi-hop items, our architectural argument is unnecessary. Docopilot tests whether dataset-scale native training already absorbs the long-hop problem. The full-attention-over-all-page-tokens baseline tests whether Q-PMR's propagation is necessary at all: if full attention at equal token budget matches Q-PMR on hop≥2 items, the novelty collapses and the contribution must be repositioned around the coverage mask. The learned page-graph transformer with K=3 steps but no query conditioning tests whether query-conditioning is the active ingredient. The query-agnostic ablation is the falsification lever for our second sub-claim: if removing query-conditioning leaves accuracy unchanged, then the routing weights are not doing evidentiary work and the coverage mask is decorative. The K-sweep measures the hop distance at which coverage mass saturates, bounding the "unbounded receptive field" claim honestly. The calibration metrics test the load-bearing assumption directly: AUROC of coverage-vs-correctness measures whether M_q predicts answer correctness better than a simple entropy baseline; precision/recall against human-annotated evidence regions measures whether M_q certifies token-level evidence coverage rather than slot-level routing mass. The attribution metric measures whether Δ_p reaches the LM: if the fraction of answer-token attention mass on Δ_p is near zero on multi-hop items, the write-back mechanism is not grounded and the design must change. Accuracy stratified by hop distance is the right primary metric because it is the only observable that distinguishes "our propagation reaches further" from "our model is generally stronger" — a uniform gain would be consistent with a dozen alternative explanations, while a gain concentrated at hop≥2 is not.

### Expected outcome and causal chain

**vs. DocSEAM (9B, frozen fusion)** — On a case where the answer requires joining a figure caption on page 3 with a table on page 7, DocSEAM produces a plausible but wrong answer grounded in the page-3 caption alone, because its center-anchored masked attention never lets page-7 tokens enter page-3's representation — the mask is a structural bound, not a capacity bound. Q-PMR instead propagates page-level slots across the G-neighborhood for K=3 steps, so page-7's slot reaches page-3 before write-back, and we expect a clear accuracy gap concentrated on hop≥2 items with near-parity on hop-0/1 items.

**vs. V-RAG + LMM** — On a case where the query's lexical surface matches a table-of-contents entry on page 1 but the answering passage is on page 12, V-RAG retrieves the TOC page because its relevance module scores surface overlap, then feeds the LM a fragmented context. Q-PMR instead lets every page's slot participate in propagation with softmax-normalized routing, so the body passage accumulates mass even without being retrieved, and we expect V-RAG to trail on items where the top-1 retrieved page is a distractor, with the gap shrinking when retrieval is clean.

**vs. Docopilot (native document-level MLLM)** — On a case requiring cross-page dependency resolution across a long invoice bundle, Docopilot produces locally coherent but globally inconsistent answers because its document-level training does not guarantee that evidence from distant pages survives into the answer context. Q-PMR instead injects propagated slot evidence through residual cross-attention onto the same N tokens the LM consumes, so the evidence reaches the LM's input surface directly, and we expect Docopilot to be competitive on single-page items but to fall behind on multi-hop items, with our advantage tracking hop distance.

**vs. Full-attention-over-all-page-tokens model (same token budget)** — On a case where the answer requires joining a figure caption on page 3 with a table on page 7, full attention over all page tokens can in principle attend to page 7 from page 3, but at equal token budget the attention is diluted across thousands of tokens and the model lacks the query-conditioned routing that concentrates mass on relevant pages. We expect Q-PMR to match or exceed full attention on hop≥2 items; if it does not, the novelty claim collapses and the contribution must be repositioned around the coverage mask.

**vs. Learned page-graph transformer with K=3 steps but no query conditioning** — On a case where the query's lexical surface matches a table-of-contents entry on page 1 but the answering passage is on page 12, a query-agnostic page-graph transformer propagates all pages uniformly and cannot concentrate mass on the body passage. We expect Q-PMR to outperform it on multi-hop items, with the gap shrinking on single-hop items where query-conditioning is less critical.

**vs. Q-PMR with query-agnostic routing (ablation)** — On a case where the query's lexical surface matches a table-of-contents entry on page 1 but the answering passage is on page 12, query-agnostic routing cannot distinguish the TOC from the body passage and propagates both equally. We expect the full model to outperform the ablation on multi-hop items; if the ablation matches the full model, query-conditioned routing is not what assembles evidence and the coverage mask does not measure what we claim.

**Coverage-mask calibration** — On a held-out set of queries with human-annotated evidence regions, we expect M_q to achieve AUROC ≥ 0.7 for predicting answer correctness, outperforming a simple entropy baseline, and precision/recall ≥ 0.6 against annotated evidence regions. If M_q cannot flag low-evidence queries better than the entropy baseline, the significance claim should be dropped and the paper reframed as a pure fusion improvement.

**LM-attention attribution** — On multi-hop items, we expect the fraction of answer-token attention mass on Δ_p to be significantly above zero (e.g., ≥ 0.2) and to increase with hop distance. If Δ_p mass is near zero on multi-hop items, the write-back mechanism is not grounded and the design must change (e.g., token-expanding write-back or gating Δ_p into the LM input).

### What would falsify this idea

If Q-PMR's gain over DocSEAM is uniform across hop-distance strata rather than concentrated on hop≥2 items, the bounded-window mechanism claim is wrong and the improvement must come from something other than cross-page propagation. Likewise, if the query-agnostic ablation matches the full model on multi-hop items, then query-conditioned routing is not what assembles evidence and the coverage mask does not measure what we claim. Additionally, if M_q's precision/recall against annotated evidence regions is near chance or if Δ_p attention mass is near zero on multi-hop items, the load-bearing assumption fails and the coverage-mask sub-claim must be dropped.

## References

1. DocSEAM: Bidirectional Cross-Page Encoding for Multi-Page Document Understanding
