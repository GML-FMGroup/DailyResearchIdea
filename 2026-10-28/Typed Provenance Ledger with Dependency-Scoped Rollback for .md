# Typed Provenance Ledger with Dependency-Scoped Rollback for Committed Retrieval Grounding

## Motivation

The field's trajectory toward single-forward-pass structured grounding — InfoDeepSeek-style agentic query construction and CausalRAG-style causal tracing — has made commitment cheap but error containment structurally impossible: when a later retrieval contradicts an early query-path or causal-trace choice, neither CausalRAG's causal graph nor InfoDeepSeek's query pipeline carries any representation of which downstream decisions were contingent on that choice, so correction degenerates to full restart or a generic self-consistency re-check. This is the recurring bottleneck across paper-trees: commitment mechanisms advance while the provenance structure that would make a commitment revocable locally does not exist. The root cause is that decisions are stored as an unordered set of outputs rather than as a typed dependency DAG whose edges record the kind of support each decision gave to its successors.

## Key Insight

If every committed decision records the typed support relation (query-refinement, causal-edge-warrant, evidence-binding) it supplies to each successor, then a contradicted commitment induces a unique minimal downstream cut — the set of decisions whose validity predicate is falsified — so rollback is exactly the transitive closure of edges carrying that support type, not a heuristic re-verification.

## Method

**What it is.** TRACE-LEDGER is a pipeline wrapper around a committed RAG grounding pass (query construction + causal tracing + evidence binding). Its input is a committed decision trace; its output is either a finalized grounded answer or a minimal typed rollback set that is re-executed. The ledger is a DAG `L = (D, E)` where each node `d` is a decision record and each edge `(d -> d')` carries a support type `tau` from `{REFINE, WARRANT, BIND}` and a recorded dependency witness `w` (the retrieval key or evidence span the child consumed from the parent).

**Load-bearing assumption (stated explicitly).** The method assumes that a decision's validity predicate is a conjunction over its incoming typed supports, so that falsifying a commitment of type `tau` invalidates exactly the transitive closure of edges whose recorded witness is entailed by the contradicted commitment, and nothing else. This assumption fails when a decision depends on a joint configuration of two ancestors (an interaction the DAG cannot express) or when a causal warrant is retrieved using a refined query as its retrieval key (so the warrant is conditioned on the REFINE it was retrieved under). In those cases the closure under-approximates and the rollback set is minimal but not sound. The method therefore records per-edge witnesses and conditions invalidation on witness entailment, not on type alone.

**How it works.**

```pseudocode
def run_pipeline(query):
    L = Ledger()
    frontier = [L.commit(Decision(kind=QUERY_SEED, payload=query, deps=[]))]
    while frontier:
        d = frontier.pop()
        # 1. Execute decision, record typed support edges and witnesses
        out = execute(d)                       # retrieval / causal-trace / bind
        for child in out.children:
            tau = classify_support(d, child)   # REFINE | WARRANT | BIND
            w = record_witness(d, child)       # retrieval key or evidence span consumed
            L.add_edge(d, child, tau, w)
            frontier.append(child)
        # 2. Check downstream evidence against committed ancestors
        for anc, tau in L.ancestors(d):
            if contradicts(d.evidence, anc.commitment):
                # 3. Witness-conditioned rollback: include edges whose witness is entailed
                cut = L.transitive_closure(
                    anc, edge_filter=lambda e: entails(anc.commitment, e.witness))
                L.invalidate(cut)
                frontier = L.resume_frontier(cut)   # re-execute ONLY cut
                break
    return L.final_answer()

def classify_support(parent, child):
    if child.kind == REFINED_QUERY and parent.kind == QUERY_SEED: return REFINE
    if child.kind == CAUSAL_EDGE and parent.kind == CAUSAL_TRACE: return WARRANT
    if child.kind == EVIDENCE_BIND and parent.kind in (REFINED_QUERY, CAUSAL_EDGE): return BIND
    return BIND  # default conservative

def record_witness(parent, child):
    # For REFINE: the refined query string.
    # For WARRANT: the retrieval key used to fetch the causal evidence span.
    # For BIND: the evidence span ID and the commitment payload it binds.
    return child.consumed_key_or_span

def entails(commitment, witness):
    # True if the witness was derived from or is conditioned on the commitment payload.
    # Implemented as exact string match for query keys and span-ID containment for evidence spans.
    return commitment.payload in witness or witness in commitment.payload
```

The FALSIFIED map is retained as a fallback for edges whose witness is missing (e.g., legacy traces), but the primary invalidation rule is witness entailment: a WARRANT edge is exempt from a REFINE contradiction only if its recorded retrieval key is disjoint from the refined query (i.e., `entails` returns False). The ledger stores per-decision the commitment payload, the evidence that produced it, the typed outgoing edges, and the per-edge witness. The transitive closure under the witness-entailment filter is the minimal sound rollback set under the conjunctive-validity assumption.

**Why this design.** We chose a typed dependency DAG with recorded witnesses over an untyped provenance log because the support type plus witness is what determines whether a contradiction propagates: a query-refinement contradiction does not falsify a causal warrant that was retrieved independently, but it does falsify a warrant whose retrieval key was the refined query. An untyped log forces the conservative full-closure rollback that CausalRAG implicitly performs (restarting the trace). We chose a fixed support-type lattice `{REFINE, WARRANT, BIND}` over a learned edge classifier because the lattice is checkable against the decision kind at commit time, accepting the cost that some legitimate cross-type dependencies are conservatively defaulted to `BIND` (over-rollback, never under-rollback). We chose local re-execution of the cut over a global self-consistency verifier because the cut is derived from the dependency structure rather than from a re-sampling of the model, accepting the cost that the ledger's soundness is bounded by the correctness of `classify_support` and `record_witness`. A domain expert would not describe this as a variant of CausalRAG: CausalRAG builds a causal graph over the *corpus* to select context; TRACE-LEDGER builds a typed provenance graph over the *pipeline's own decisions* to scope rollback, and the two graphs have different node sets and different edge semantics.

**Why it measures what we claim.** The transitive closure under the witness-entailment filter measures *minimal sound rollback set* because the assumption that a decision's validity predicate is a conjunction over its incoming typed supports makes the falsified-witness closure exactly the set of decisions whose predicate is false; this assumption fails when a decision's validity depends on a *joint* configuration of two ancestors (an interaction the DAG cannot express), in which case the closure under-approximates and the metric reflects only pairwise-dependency rollback. The support type `tau` measures *the kind of evidential dependence* between two decisions because the assumption that decision kind determines support kind (a refined query is supported only by its parent query seed) makes `classify_support` a total function on kinds; this assumption fails when a causal-trace decision reuses a refined query as its retrieval key, in which case `tau` reflects the commit-time kind pair rather than the actual evidential role. The contradiction test `contradicts(d.evidence, anc.commitment)` measures *commitment falsification* because the assumption that a retrieved evidence span either entails or contradicts the commitment payload (no partial support) makes the binary test sound; this assumption fails on graded evidence, in which case the test reflects a thresholded contradiction rather than strict falsification.

## Contribution

(1) A typed provenance ledger (TRACE-LEDGER) that records REFINE/WARRANT/BIND support edges over committed query-construction and causal-trace decisions, with a closed-form typed-rollback rule that maps a contradicted commitment to a minimal downstream cut. (2) A design principle — support-type-conditioned propagation — showing that rollback scope must be determined by the *kind* of evidential dependence rather than by graph distance or a generic verifier, and a demonstration that untyped provenance forces strictly larger rollback sets. (3) An instrumented evaluation harness that reports rollback-set size, re-execution cost, and answer preservation under injected contradictions, enabling comparison against restart and generic self-consistency baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-hop QA with contradiction-injected traces (2–4 hops; 1–2 injected contradictions per trace) | Forces downstream evidence to conflict with committed ancestors |
| Dataset (natural) | A long-horizon agentic RAG benchmark (e.g., a 3–5 hop subset of HotpotQA or MuSiQue with naturally occurring contradictions) | Tests whether scoped rollback helps when contradictions are not synthetic |
| Primary metric | Minimal rollback set size at matched answer accuracy (accuracy matched to within ±1% by adjusting the contradiction threshold) | Directly measures scoped vs. full rollback |
| Secondary metric | Wall-clock and token cost per query | Establishes practical significance of scoped re-execution |
| Baseline | CausalRAG (corpus causal graph) with identical retrieval budget (same number of retrieval calls per hop) | Tests whether pipeline-provenance typing beats corpus-graph restart |
| Baseline | Untyped provenance log with full-closure rollback (same retrieval budget) | Tests whether typed edge filtering is the source of gain |
| Baseline | Regular RAG (chunk-similarity) with same retrieval budget | Tests whether grounding structure matters at all |
| Ablation-of-ours | TRACE-LEDGER with all edges defaulted to BIND (type-conditioned only, no witnesses) | Isolates the support-type lattice's contribution |
| Ablation-of-ours | TRACE-LEDGER with witnesses but type-conditioned FALSIFIED map (no witness entailment) | Isolates the witness-conditioned pruning contribution |

**Contradiction-injection protocol.** For each multi-hop trace, we select a span from a later hop that is topically similar (cosine similarity ≥ 0.8 to the committed evidence) but asserts a conflicting fact (e.g., a different entity or date). We inject 1–2 such spans per trace, replacing the original retrieved span. The accuracy-matching procedure fixes the contradiction threshold (the score below which `contradicts` returns True) per method so that final answer accuracy on the full test set is within ±1% of the best baseline; rollback-set size is then compared at that matched accuracy.

### Why this setup validates the claim

The central claim is that a typed provenance DAG over pipeline decisions with recorded witnesses yields a smaller sound rollback set than untyped provenance or type-only pruning, without sacrificing answer accuracy. The dataset must therefore contain traces where a contradiction arises mid-pipeline, so that rollback scope is observable; multi-hop QA with injected contradictions provides exactly this, and the natural-contradiction benchmark tests whether the effect survives without injection. The primary metric—rollback set size at matched accuracy—is the right detector because it directly operationalizes "minimal sound rollback": if witness-conditioned filtering is doing real work, rollback sets shrink while accuracy holds. CausalRAG tests whether corpus-level causal graphs (a different node set) can substitute for decision-level provenance; the untyped-log baseline tests whether typing is the active ingredient rather than provenance per se; regular RAG tests whether the whole structure matters. The BIND-default ablation is the decisive control for the lattice: it preserves the ledger machinery but collapses the lattice, so any gap between it and full TRACE-LEDGER is attributable specifically to typed edge semantics. The witness-conditioned vs. type-conditioned ablation is the decisive control for the load-bearing assumption: it preserves typing but removes witness entailment, so any gap isolates the contribution of recording and checking witnesses. All baselines use identical retrieval budgets (same number of retrieval calls per hop) to ensure fair comparison.

### Expected outcome and causal chain

**vs. CausalRAG** — On a case where a refined query retrieves a span that later contradicts an independently warranted causal edge, CausalRAG restarts the entire trace because its corpus graph cannot distinguish the query-refinement branch from the causal-warrant branch. Our method instead invalidates only edges whose recorded witness is entailed by the contradicted refinement, leaving the WARRANT subtree intact only when its retrieval key was disjoint from the refined query; when the warrant was retrieved under the refined query, it is correctly included in the cut. We expect a markedly smaller rollback set on contradiction-heavy subsets with parity on clean subsets, and we expect the natural-contradiction benchmark to show smaller but still positive gains.

**vs. Untyped provenance log with full-closure rollback** — On the same case, the untyped log treats all dependencies as equivalent, so any contradiction forces transitive closure over the whole ancestor set, re-executing decisions whose support was never falsified. Our method cuts only edges whose witness is entailed by the contradicted commitment, so we expect rollback-set size to drop substantially on instances with multiple independent branches while accuracy stays matched, isolating witness-conditioned typing as the cause.

**vs. Regular RAG (chunk-similarity)** — On a case where a retrieved chunk is topically similar but contradicts an earlier committed binding, regular RAG has no commitment record to contradict and silently produces an inconsistent answer. Our method detects the contradiction against the ancestor commitment and rolls back the BIND edge, so we expect a visible accuracy gap on contradiction-injected instances and near-parity on clean single-hop instances.

**vs. TRACE-LEDGER with all edges defaulted to BIND (ablation)** — On a case where a query-refinement contradiction should not propagate to an independently retrieved causal warrant, the BIND-default variant over-rolls-back because every edge is falsified together, mimicking the untyped log. Our method's typed lattice with witnesses scopes the cut, so we expect the ablation to land near the untyped baseline on rollback size, confirming that the lattice—not the ledger—produces the gain.

**vs. TRACE-LEDGER with witnesses but type-conditioned FALSIFIED map (ablation)** — On a case where a causal warrant was retrieved using the refined query as its retrieval key, the type-conditioned variant exempts the WARRANT edge from the REFINE contradiction (because the FALSIFIED map says WARRANT is independent of REFINE), leaving a stale warrant whose evidence was retrieved under a now-falsified query. Our witness-conditioned method includes that edge in the cut because the recorded retrieval key is entailed by the contradicted refinement, so we expect higher answer accuracy on these cases with a slightly larger rollback set, confirming that witness entailment is necessary for soundness.

### What would falsify this idea

If the typed ledger's rollback-set reduction were uniform across all subsets rather than concentrated on contradiction-heavy multi-branch instances, or if the BIND-default ablation matched full TRACE-LEDGER, the claim that typed edge semantics scope rollback would be wrong—the gain would be attributable to ledger bookkeeping rather than support typing. If the witness-conditioned variant did not improve accuracy over the type-conditioned variant on cases where warrants reuse refined queries, the load-bearing assumption would be false and the method would need redesign.

## References

1. CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
2. InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation
3. Benchmarking Large Language Models in Retrieval-Augmented Generation
4. Large Language Models with Controllable Working Memory
5. Few-shot Learning with Retrieval Augmented Language Models
6. Retrieval-Generation Synergy Augmented Large Language Models
