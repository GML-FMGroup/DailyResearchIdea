# Belief-State Consistency Enforcement for Persistent Personal Assistant Memory

## Motivation

Persistent personal AI assistants such as the Avatar Intelligence paradigm (related work) treat memory as an external key-value/vector store retrieved by similarity, so nothing structurally prevents a retrieved memory from contradicting the agent's currently active beliefs — the store and the belief state are never required to satisfy a joint constraint. This is a meta-level problem that recurs across memory architectures (Memory OS of AI Agents, 2025, and any RAG-style personalization layer): because memory is an *external* store with its own update schedule, the agent can act on mutually inconsistent propositions (e.g., an old preference that was later revoked) and no component owns the invariant that retrieved memory must be jointly satisfiable with active beliefs. The root cause is the absence of a formal consistency relation between the two surfaces, not a lack of retrieval capacity.

## Key Insight

If memory items are represented as typed beliefs carrying provenance and a revision order, then consistency between memory and the active belief state becomes a checkable constraint-satisfaction problem over a partial order, so retrieval can be restricted to a maximal consistent subset and belief updates can be propagated as minimal repairs rather than left to similarity ranking.

## Method

**What it is.** BACON (Belief-Aware Consistency-Enforced memory) takes a stream of candidate memory writes and belief-update events and returns (i) a repaired belief state and (ii) a retrieval operator that only emits memory items jointly consistent with that state. Inputs: candidate belief `b` with provenance `(source, timestamp, revision_id)`; outputs: accepted/repaired belief set `B` and, at query time, a maximal consistent subset `M_q ⊆ M` of memory.

**Scope of the consistency invariant (load-bearing assumption, stated explicitly).** The joint-satisfiability check is defined over a *declared, closed predicate ontology* with mandatory qualifier slots: `predicate ∈ P` (a fixed list of 40 personal-assistant predicates, e.g. `prefers_food`, `prefers_drink`, `available_at`, `located_in`, `works_as`), plus mandatory slots `time_window` (start/end or `ALL`), `condition` (a literal from a fixed 12-value list or `NONE`), and `polarity_scope` (which argument the negation applies to). A memory item that cannot be parsed into this ontology is stored as an **opaque item**: it is retrievable but is *never* passed to `CHECK` and is flagged `opaque=true` in the retrieval output. The invariant BACON guarantees is therefore: *every non-opaque item in `M_q` is jointly satisfiable with `B` and with the other non-opaque items in `M_q` under the declared ontology*. It is **not** a claim of full semantic consistency over free text; qualifier clashes that the parser drops are outside the guarantee by construction, and the parse-fidelity study below measures how large that outside is.

**How it works.**

```pseudocode
# Belief representation: b = (predicate ∈ P, args, polarity, polarity_scope,
#                            time_window, condition, revision_id, source)
# Constraint types: (1) logical: b and not-b cannot both hold when their
#                              time_window and condition slots overlap
#                   (2) probabilistic: P(head) over mutually exclusive values of
#                              the same predicate sums to 1 within an overlapping
#                              time_window
#                   (3) temporal: revision_id ordering forbids a newer belief being
#                              overridden by an older one
# Opaque items: parse failure -> stored with opaque=true, excluded from CHECK

function PARSE(text) -> b or OPAQUE:
    # rule-based extractor over the 40-predicate ontology; returns OPAQUE if
    # predicate not in P or any mandatory qualifier slot cannot be filled

function CHECK(b_new, B):
    if b_new.opaque: return empty          # opaque items never conflict-checked
    conflicts = {}
    for c in B where not c.opaque:
        if c.predicate == b_new.predicate
           and OVERLAP(c.time_window, b_new.time_window)
           and COMPATIBLE(c.condition, b_new.condition)
           and VIOLATES(c, b_new):          # polarity clash (1) or
                                            # exclusive-value clash (2) or
                                            # revision inversion (3)
            conflicts.add(c)
    return conflicts

function WRITE(b_new, B):
    conflicts = CHECK(b_new, B)
    if conflicts is empty:
        return B ∪ {b_new}
    # Minimal repair: prefer revising the belief with the OLDEST revision_id
    # (provenance-ranked, deterministic tie-break by source reliability tier)
    victim = argmin_{c in conflicts} (c.revision_id, source_tier(c))
    B' = (B \ {victim}) ∪ {b_new}
    LOG_REPAIR(victim, b_new, reason=conflict_type)
    return B'

function RETRIEVE(q, B, M, k=8):
    C = top_k_by_similarity(q, M, k=3k)   # over-fetch 3x candidate pool
    # Greedy maximal-consistent-subset selection, ordered by similarity
    M_q = []
    for m in C ordered by sim(q,m) desc:
        if m.opaque:
            M_q.append(m); FLAG(m, opaque=true)   # retrieved, not CHECKed
        elif CHECK(m, B ∪ M_q) is empty:          # consistency w.r.t. BOTH active
                                                  # beliefs and already-selected items
            M_q.append(m)
        if len(M_q) == k: break
    return M_q

function BELIEF_UPDATE(b_old -> b_new):
    B = WRITE(b_new, B \ {b_old})
    # Propagate: any memory item whose provenance chain cites b_old is re-checked
    for m in DEPENDENTS(b_old, M):
        if not m.opaque and CHECK(m, B) is nonempty: MARK_STALE(m)
    return B
```

Hyperparameters: over-fetch factor 3, retrieval budget k=8, source reliability tiers (user-explicit > user-inferred > assistant-inferred) fixed a priori; predicate ontology size |P| = 40, condition vocabulary size 12, qualifier slots mandatory (parse fails to OPAQUE if any slot unfilled).

**Why this design.** We chose a typed belief representation with explicit revision order over free-text memory chunks because consistency is only *checkable* when propositions are comparable, accepting the cost that writes must be parsed into predicates and some nuance is lost — and we make that cost explicit by scoping the invariant to a closed ontology and routing unparseable items to an opaque, flagged channel rather than silently CHECKing them. We chose deterministic provenance-ranked minimal repair over LLM-judged conflict resolution because an LLM judge reintroduces the same unverifiable surface the invariant is meant to eliminate, accepting the cost that repair may occasionally discard a correct older belief. We chose greedy maximal-consistent-subset retrieval over re-ranking by a learned consistency score because greedy selection has a stated, inspectable guarantee (every emitted non-opaque item is jointly checkable against `B` and prior selections), accepting the cost that it is not similarity-optimal. A domain expert would not describe BACON as a variant of standard RAG (e.g., Memory OS retrieval) because the retrieval operator's output set is constrained by a joint satisfiability check against the active belief state, not by a similarity score alone.

**Why this measures what we claim.** The `CHECK` conflict set measures *belief-memory inconsistency within the declared ontology* because it enumerates propositional/temporal violations under the assumption that parsed predicates faithfully encode the memory item's content; this assumption fails when parsing drops a qualifier (e.g., "coffee after 5pm"), in which case `CHECK` reflects only surface-level predicate clashes — which is why opaque items are excluded from the guarantee and why parse fidelity is reported as a covariate. The provenance-ranked `victim` selection measures *revision recency priority* because revision_id is assumed monotone with actual user intent change; this fails when a user re-asserts an old preference without a new revision event, in which case it reflects write order rather than intent. The greedy `M_q` selection measures *retrieval consistency* because the greedy order is assumed to make the maximal-consistent-subset approximation tight enough for k=8; this fails when many high-similarity items conflict with each other, in which case `M_q` reflects a similarity-biased consistent subset rather than the true maximum — which is why we report the greedy approximation ratio against an exact ILP solver on a sample.

## Contribution

(1) BACON, a belief-memory architecture that represents memory as typed, provenance-stamped beliefs and enforces a checkable consistency invariant between the memory store and the agent's active belief state at both write and retrieval time. (2) A deterministic provenance-ranked minimal-repair procedure and a greedy maximal-consistent-subset retrieval operator, yielding an inspectable repair log that attributes each memory revision to a named conflict type. (3) A design principle — that personal-assistant memory consistency should be enforced as constraint satisfaction over a revision partial order rather than delegated to similarity ranking or LLM judging — supported by the invariant's checkability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-horizon personal-assistant belief-revision logs, constructed by seeding 200 queries from an existing preference dataset (e.g., LaMP-style user profiles) and injecting controlled revision and source-tier conflict events with ground-truth labels | Contains provenance and revision order needed for consistency |
| Primary metric | Joint-consistency rate of retrieved memory sets, computed against an independent human-annotated contradiction set (not the method's own parser output) | Directly measures the operator's stated guarantee |
| Covariate metric | Predicate-extraction fidelity: human-annotated parse accuracy and qualifier-retention rate on a 200-item sample | Bounds the scope of the invariant |
| Covariate metric | Greedy approximation ratio vs. exact ILP maximal-consistent-subset solver on a 200-query sample | Grounds the greedy guarantee empirically |
| Baseline | Standard similarity RAG (Memory OS-style) | Tests whether consistency constraint adds value over similarity alone |
| Baseline | LLM-judge conflict resolution | Tests whether deterministic repair beats unverifiable judging |
| Baseline | Recency-only overwrite memory | Tests whether provenance-ranked repair beats naive recency |
| Ablation-of-ours | BACON minus joint CHECK at retrieval | Isolates contribution of consistency-constrained selection |
| Cross-architecture check | Same joint-CHECK retrieval operator applied on top of Memory OS and a vector-RAG baseline | Tests architecture-agnosticism of the invariant |

### Why this setup validates the claim

The central claim is that constraining retrieval to a jointly satisfiable subset of memory, repaired under typed belief constraints, produces fewer belief-memory inconsistencies than similarity-ranked or judge-based alternatives. The primary metric — joint-consistency rate of emitted memory sets, scored against an independent human-annotated contradiction set rather than the method's own parser — is the right detector because it is exactly the property the method asserts it guarantees, and scoring it independently breaks the circularity that would arise from scoring on the same lossy parse. The parse-fidelity and greedy-approximation covariates bound the scope of that guarantee: parse fidelity tells us what fraction of memory items the invariant actually covers (non-opaque), and the approximation ratio tells us how far greedy `M_q` is from the true maximum on a sample. A similarity-only baseline tests whether the constraint is load-bearing, an LLM-judge baseline tests whether deterministic provenance-ranked repair beats unverifiable resolution, and a recency-only baseline tests whether the revision-order ranking matters. The ablation (dropping the joint CHECK at retrieval) isolates the retrieval-consistency sub-claim from the write-repair sub-claim: if the full method beats the ablation, the gain is attributable to the joint satisfiability check rather than to belief repair alone. The cross-architecture check tests whether the operator's benefit survives when dropped onto two existing memory systems, which is the claim that would shift practice. Together these form a falsifiable test: the claim predicts a specific ordering of outcomes rather than a uniform improvement.

### Expected outcome and causal chain

**vs. Standard similarity RAG (Memory OS-style)** — On a case where a user's stated preference was revised months ago but an older memory chunk remains lexically closer to the query, the baseline emits both the stale and current items because its retrieval ranks purely by similarity. Our method instead filters candidates greedily against the active belief state, rejecting the stale chunk because it conflicts with the repaired belief. We expect a noticeable gap on the "revised-preference" subset of queries but near parity on queries with no revision history.

**vs. LLM-judge conflict resolution** — On a case where a conflict requires resolving which of two beliefs is authoritative, the baseline produces an inconsistent or non-reproducible judgment because the judge has no monotone revision signal and varies across runs. Our method instead deterministically selects the victim by oldest revision_id with a fixed source-tier tie-break, yielding a stable repaired state. We expect the gap to concentrate on conflict-heavy queries and to show near-zero run-to-run variance for our method versus non-trivial variance for the judge.

**vs. Recency-only overwrite memory** — On a case where a newer but lower-reliability inference conflicts with an older user-explicit statement, the baseline discards the correct explicit belief simply because it is older. Our method instead ranks victims by revision_id then source tier, preserving high-reliability older beliefs when revision order does not genuinely supersede them. We expect a modest but consistent gain on the "source-tier conflict" subset, with parity on unambiguous recency cases.

**vs. Ablation (BACON minus joint CHECK at retrieval)** — On a case where two high-similarity memory items conflict with each other but each is individually consistent with `B`, the ablation emits both because it only checks against `B`, while the full method rejects the second because it conflicts with the already-selected first. We expect the full method's gain over the ablation to be concentrated on the "intra-`M_q` conflict" subset and near zero elsewhere; a uniform gain would falsify the mechanism attribution.

**Cross-architecture check (joint-CHECK on Memory OS and vector-RAG)** — On the same revised-preference and source-tier-conflict subsets, we expect the joint-CHECK operator to recover most of BACON's gain when layered on either baseline, with the residual gap attributable to write-time repair. If the operator's gain vanishes on either architecture, the architecture-agnosticism claim fails.

**Covariate reporting** — We report parse fidelity (human-annotated parse accuracy and qualifier-retention rate) and the greedy-vs-ILP approximation ratio alongside the primary metric, and we report the primary metric separately on the non-opaque subset so the guarantee is stated over the representation the method actually controls.

### What would falsify this idea

If the full method's gain over the ablation is uniform across query subsets rather than concentrated on revision-conflict and source-tier-conflict cases, then the joint-consistency mechanism is not the cause of any improvement and the central claim is wrong. Likewise, if the LLM-judge baseline matches our joint-consistency rate while being equally stable across runs, the deterministic-repair argument fails; and if parse fidelity is so low that the non-opaque subset is unrepresentative, the scoped invariant does not cover the memory the assistant actually uses.

## References

1. Avatar Intelligence: Amplifying Human Agency Through Persistent Personal Agents
