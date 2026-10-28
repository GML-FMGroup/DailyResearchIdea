# Bijective Code-Token Grounding for Scene-Graph Consolidation

## Motivation

Scene-graph consolidation pipelines (e.g., the one in Fine-Grained Captioning of Long Videos through Scene Graph Consolidation, 2025) produce consolidated graphs whose nodes have no verifiable, executable mapping to code tokens, so CodeCap-Bench cannot certify faithful reconstruction; the root cause is that consolidation treats captions as terminal natural-language output rather than as a typed, executable surface. This is the meta-level bottleneck: across both the consolidation tree and the code-caption tree, token-level grounding and executable semantics are never jointly enforced, so any claim of faithfulness is unverifiable.

## Key Insight

If every consolidated scene-graph node is emitted as a typed code token whose runtime trace deterministically re-derives that node's identity, then token-level faithfulness becomes a bijection check between graph nodes and trace events rather than a similarity score, making faithful reconstruction a structural invariant of the emission.

## Method

**What it is.** BCT-Ground (Bidirectional Code-Token Grounding) is a layer inserted between scene-graph consolidation and caption emission: it takes a consolidated scene graph G = (V, E, τ) with typed nodes and edges, and emits an executable program P plus a node↔token bijection table M. Each node v ∈ V is emitted as exactly one typed code token (a function call or data-structure field assignment), and each token is annotated with the node id it realizes.

**Load-bearing assumption (stated explicitly).** The method assumes that each consolidated node v can be emitted as exactly one registered primitive whose deterministic runtime trace output is a canonical, injective encoding of v's type and attributes, so that exact string equality between trace output and canonical_form(v) certifies node identity. This assumption fails under aliasing (two nodes with identical type and attributes) and under aggregation (a consolidated node merging multiple source mentions with an open-ended attribute set). To make the assumption testable and to decouple the verifier from the emitter, canonical_form(v) is defined independently of the primitive library L: it is the deterministic serialization of (a) the node's type τ(v), (b) its attribute set sorted lexicographically, and (c) a disambiguating node-id field id(v) drawn from the upstream consolidation graph's provenance record. The node-id field guarantees injectivity of canonical_form over V by construction, so aliased nodes receive distinct canonical forms. The trace assertion is therefore not circular: it checks that the emitted token's runtime output re-derives the node's provenance-bearing identity, not merely that the emitter's own serialization convention is self-consistent. For aggregate nodes whose provenance set contains multiple source mentions, the bijection is defined over provenance mentions rather than over consolidated nodes: each mention m ∈ prov(v) is emitted as one token, and M maps mentions to token ids; the consolidated node v is grounded iff all its mentions are grounded and their tokens share the same primitive type. This many-to-one emission makes aggregation representable rather than punted to the UNGROUNDABLE set.

**How it works.**

```pseudocode
INPUT: consolidated graph G=(V,E,τ) with provenance records prov(v) for each v,
       primitive library L (graphical primitives, layout rules, time functions),
       node-identity oracle O (held-out human-annotated node-identity labels for a
       calibration subset C ⊂ V, |C| = 512 nodes)
OUTPUT: program P, bijection table M: (v, m) <-> Tokens(P) for m ∈ prov(v),
        ungroundable set U

1. TYPED EMISSION (forward direction)
   for each v in V in topological order of E:
       for each mention m in prov(v):
           tok(m) = L.emit(type=τ(v), payload=attributes(m),
                           refs=[tok(m') for m' in pred_mentions(m)],
                           node_id=id(v))
           P.append(tok(m));  M[(v,m)] = id(tok(m))
   # each emit() call is a registered primitive with fixed arity = |pred_mentions(m)|+1
   # node_id is passed as a reserved argument so canonical_form(v) is injective over V

2. TRACE INSTRUMENTATION
   P' = instrument(P)   # wrap each primitive call with a trace record (token_id, args, output)

3. BIDIRECTIONAL VERIFICATION (backward direction)
   run P' in sandbox -> trace T
   for each v in V:
       for each m in prov(v):
           assert T[M[(v,m)]].output == canonical_form(v, m)
           # canonical_form(v,m) = serialize(τ(v), sorted(attributes(m)), id(v))
           assert reconstruct_node(T[M[(v,m)]]) == (v, m)
       assert all tokens for v share the same primitive type
   edge_check: for each (u,v) in E, for each mention pair (m_u, m_v) in prov(u)×prov(v):
       assert M[(u,m_u)] in T[M[(v,m_v)]].args
   # edge realized as argument reference between provenance mentions

4. REPAIR LOOP (max 3 iterations, only on assertion failure)
   for failing (v,m): re-emit with expanded payload = attributes(m) ∪ {trace-hint from T}
   re-run steps 2-3; if still failing, mark (v,m) as UNGROUNDABLE and report

5. CALIBRATION CHECK (verifier-emitter decoupling)
   on calibration subset C (512 nodes with oracle labels O):
       compute agreement between M's node-identity assignments and O
       if agreement < 0.95, flag the primitive library L as under-specified and
       report the disagreement rate as a library-coverage gap (not a grounding success)

6. OUTPUT: P (executable caption), M (bijection over provenance mentions),
           U (ungroundable mention set), calibration agreement report
```

The bijection M is the load-bearing artifact: it is a total, injective map from provenance mentions (v,m) to token ids, and the trace T is the runtime witness. `canonical_form(v,m)` is a deterministic serialization of node type + mention attributes + node id, defined independently of L; equality is exact string match, not embedding similarity. The calibration check on C (512 nodes) with held-out oracle O provides the independent verification that the emitter's canonical form is not merely self-consistent.

**Why this design.** We chose typed one-token-per-mention emission over free-form code generation because a bijection requires that token count and mention count be equal by construction, accepting the cost that the primitive library L must be fixed and expressive enough to cover every node type (an open-ended graph would need library extension). We chose exact canonical-form equality over embedding-based faithfulness scoring because the motivation demands certification, not approximation, accepting the cost that a semantically correct but syntactically different emission is flagged as failure and must be repaired. We chose a bounded repair loop (3 iterations) over end-to-end retraining because the bijection is a hard constraint that can be enforced by local re-emission, accepting the cost that some mentions remain UNGROUNDABLE and are reported rather than silently passed. We chose to define canonical_form independently of L and to include a node-id disambiguator because otherwise the assertion is circular over the emitter's own serialization; the calibration check on 512 oracle-labeled nodes makes this independence testable. Crucially, this is not "code captioning + graph consolidation" composed: CodeCap-Bench (the matched prior work) generates code captions freely and scores faithfulness with a multi-level framework, whereas BCT-Ground makes faithfulness a runtime-checkable bijection by construction — a domain expert would not describe it as a variant of CodeCap-Bench because CodeCap-Bench has no node↔token map and no trace-based assertion.

**Why it measures what we claim.** The bijection table M measures token-level faithfulness because the named assumption is that each registered primitive's trace output is a canonical, injective encoding of its mention's type, attributes, and node id; this assumption fails when two distinct mentions share identical type and attributes and the node-id disambiguator is omitted, in which case M reflects token identity rather than node identity. The trace assertion `T[M[(v,m)]].output == canonical_form(v,m)` measures executable semantics because the named assumption is that the sandbox executes primitives deterministically (no floating-point nondeterminism across runs); this fails when time-dependent functions produce non-reproducible outputs, in which case the assertion reflects run-to-run variance rather than semantic correctness. The edge check `M[(u,m_u)] ∈ args(T[M[(v,m_v)]])` measures graph-structure preservation because the named assumption is that edges are realized as argument references between provenance mentions in the emission order; this fails when an edge is expressed implicitly through a shared global variable, in which case the check reflects argument-passing style rather than edge existence. The UNGROUNDABLE set U measures coverage honestly because the assumption is that repair either succeeds or the mention is genuinely inexpressible in L; this fails when the repair budget (3) is exhausted on an expressible mention, in which case U reflects budget limits rather than library expressiveness. The calibration agreement on C (512 oracle-labeled nodes) measures verifier-emitter independence because the assumption is that the oracle O is held out from the emitter; this fails when O is derived from the same pipeline, in which case agreement reflects self-consistency rather than faithfulness.

## Contribution

['A bidirectional code-token grounding layer (BCT-Ground) that emits every consolidated scene-graph node as exactly one typed code token and maintains a runtime-verified bijection table between nodes and tokens.', 'A trace-based faithfulness certification procedure that replaces similarity scoring with exact canonical-form assertions on instrumented program execution, yielding an explicit UNGROUNDABLE set rather than a scalar score.', "A design principle: token-level faithfulness in code captions is achievable as a structural invariant (bijection + trace) rather than a learned or heuristic property, with the primitive library's expressiveness as the binding constraint."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CodeCap-Bench video split | Matched prior work; code-caption ground truth available |
| Dataset (calibration) | 512-node subset of CodeCap-Bench with held-out human node-identity labels | Tests verifier-emitter independence |
| Primary metric | Grounding rate (fraction of provenance mentions in M, not U) | Directly measures bijection coverage claim |
| Secondary metric | Calibration agreement (fraction of 512 oracle-labeled nodes where M matches O) | Detects circular canonical_form |
| Baseline | CodeCap-Bench free-form code captions | Tests whether bijection beats unconstrained code generation |
| Baseline | Natural language captions | Tests whether executable structure beats prose faithfulness |
| Baseline | Graph-only emission (no trace verification) | Tests whether runtime witness adds grounding beyond structure |
| Baseline | Grammar-constrained decoder emitting one primitive per node, no M table | Isolates what the explicit bijection table adds |
| Ablation-of-ours | BCT-Ground with repair loop disabled | Isolates repair contribution to coverage |
| Ablation-of-ours | BCT-Ground with node-id disambiguator removed from canonical_form | Tests aliasing failure mode |

### Why this setup validates the claim

The central claim is that faithfulness becomes runtime-checkable by construction when each provenance mention maps to exactly one traced code token whose canonical form is defined independently of the emitter. The grounding-rate metric is the right detector because it counts only mentions whose trace output exactly matches the canonical form and whose edges appear as argument references — precisely the property the bijection asserts. The calibration agreement on 512 oracle-labeled nodes tests the decoupling: if canonical_form were circular, agreement would be trivially high regardless of faithfulness; a held-out oracle makes the check falsifiable. Free-form CodeCap-Bench captions test the sub-claim that unconstrained generation cannot guarantee token-node correspondence; natural language captions test whether executability itself matters; graph-only emission (structure without trace assertions) tests whether the runtime witness, not the graph, is doing the grounding work; the grammar-constrained decoder baseline tests whether the explicit M table adds anything beyond constrained decoding. The repair-disabled and disambiguator-removed ablations isolate the repair loop and the injectivity mechanism respectively. Together these form a falsifiable test: if grounding rate does not separate our method from graph-only emission, or if calibration agreement is near 1.0 for the disambiguator-removed ablation, the trace-assertion and injectivity mechanisms are not load-bearing.

### Expected outcome and causal chain

**vs. CodeCap-Bench free-form code captions** — On a case where a scene graph has two same-typed objects with distinct spatial relations (e.g., two cups at different offsets), the baseline emits plausible code whose token count and ordering bear no guaranteed relation to nodes, so faithfulness scoring must approximate and cannot certify per-node correspondence. Our method emits one registered primitive per provenance mention in topological order, so each token is pinned to a mention by M and verified by trace equality against a canonical form that includes the node-id disambiguator. We expect a large grounding-rate gap concentrated on multi-object graphs, with near-parity on single-object clips.

**vs. Natural language captions** — On a case where a relation is implicit in prose ("the cup is being pushed"), the baseline cannot expose edge structure, so reconstruction and QA must infer the agent-patient link from surface text. Our method realizes each edge as an argument reference in the successor's emission, making the edge check `M[(u,m_u)] ∈ args(T[M[(v,m_v)]])` pass by construction. We expect our method to show a clear advantage on relation-heavy subsets and parity on attribute-only descriptions.

**vs. Graph-only emission (no trace verification)** — On a case where two distinct mentions share identical type and attributes (aliasing), graph-only emission produces tokens but cannot distinguish which token realizes which mention, so M is unverifiable. Our trace assertion with the node-id disambiguator flags this as UNGROUNDABLE rather than silently passing. We expect graph-only to report inflated grounding on aliased subsets, while ours reports honest U membership — a signal difference, not just a score difference.

**vs. Grammar-constrained decoder emitting one primitive per node, no M table** — On a case where the grammar permits multiple valid emission orders for the same graph, the constrained decoder can produce a syntactically valid program whose token-to-node correspondence is ambiguous without an explicit M table. Our method's M table pins each token to a mention id, so the trace assertion is checkable per mention. We expect the constrained decoder to match our grounding rate on simple graphs but to fall behind on graphs with aliasing or aggregation, where the absence of M prevents per-mention verification.

**vs. BCT-Ground with repair loop disabled** — On a case where a mention is expressible in L but initially mis-emitted (e.g., wrong argument order), the repair-disabled variant marks it UNGROUNDABLE, while the full method re-emits with a trace hint and grounds it. We expect the repair loop to convert a measurable fraction of expressible-but-misemitted mentions into grounded ones, concentrated on graphs with high edge density.

**vs. BCT-Ground with node-id disambiguator removed** — On a case with aliased mentions, the disambiguator-removed variant produces identical canonical forms for distinct mentions, so the trace assertion passes for both tokens regardless of which mention they realize, inflating grounding rate. The full method with the disambiguator reports honest U membership for aliased mentions. We expect a significant grounding-rate gap on aliased subsets, confirming the injectivity mechanism is load-bearing.

### What would falsify this idea

If our grounding-rate gain over graph-only emission is uniform across subsets rather than concentrated where aliasing, implicit edges, and non-canonical emission are predicted to break the baselines, or if calibration agreement on the 512 oracle-labeled nodes is near 1.0 even for the disambiguator-removed ablation, the claim that the bijection plus trace witness is the causal mechanism would be wrong. Equally, if repair-disabled BCT-Ground matches the full method, the bounded repair loop is not load-bearing and the coverage argument collapses.

## References

1. CodeCap-Bench: Can Code Serve as a Good Video Caption?
