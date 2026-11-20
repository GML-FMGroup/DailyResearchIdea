# Composable Edit Algebra for Persistent Panorama World States

## Motivation

WM^MLLM treats each MLLM-mediated local edit as an independent region operation: the abstract specifies that a local edit leaves every pixel outside its region unchanged, but says nothing about how two edits whose regions overlap spatially or whose effects interact semantically (e.g., 'move the red chair' followed by 'recolor the chair') compose into a well-defined next state. This is the same closure gap that recurs at the meta level: any persistent external state written by an MLLM controller must define an operation algebra that is closed under composition, yet WM^MLLM's state is closed only under single edits, so the revision history cannot faithfully represent interacting edits and revisits after multi-edit sequences are undefined.

## Key Insight

If each edit is represented as a typed partial function on the panorama state carrying an explicit footprint and a semantic read/write dependency set, then composition of edits is closed whenever overlapping footprints are reconciled by a deterministic priority order derived from dependency direction, so the revision history becomes a confluent rewrite system whose normal form is the well-defined next state.

## Method

**What it is.** We introduce CEA (Composable Edit Algebra), a state-operation algebra layered on WM^MLLM's external panorama. CEA takes as input a panorama state S (pixels + revision history) and an ordered list of MLLM-proposed edit operations, and outputs a single committed next state S' together with a composition certificate that records, for every pair of interacting edits, which one dominated and why.

**Load-bearing assumption (stated explicitly).** CEA's interaction graph and priority rule are only as good as the read-dependency sets R_i and semantic tags t_i that the MLLM controller emits. We assume the MLLM can emit complete and correct R_i and t_i for each edit. This assumption is load-bearing: if R_i is incomplete (e.g., 'recolor the chair' silently assumes the chair still exists after a prior move), the graph misses edges, the topological order is wrong, and the normal form is an arbitrary serialization rather than a semantically justified one. We therefore do not assume completeness silently; we (i) restrict the first experiment to a fixed set of four edit types with hand-specified dependency templates, (ii) add a conservative edge rule that adds edges for all edits sharing a semantic tag or object class, and (iii) record missing dependencies as a first-class uncertainty flag in the certificate.

**How it works.**

```pseudocode
# Input: state S (panorama P, history H), edits e_1..e_n proposed by MLLM controller
# Each edit e_i = (footprint F_i ⊆ P, write_op w_i, read_deps R_i, semantic_tag t_i)
# Edit types restricted to {move, recolor, place, relight} in first experiment.

1. FOOTPRINT EXTRACTION
   For each e_i: F_i = MLLM.localize(request_i)      # box or mask on panorama
   R_i = MLLM.read_deps(request_i)                  # objects/attributes the edit assumes
   t_i = MLLM.semantic_tag(request_i)               # e.g., recolor, move, place, relight
   # Dependency templates: for each edit type, a hand-specified template lists the
   # precondition attributes that must be read (e.g., recolor reads object identity
   # and current color; move reads object identity and current position).
   # R_i is the union of MLLM-emitted deps and template deps; any template dep
   # not emitted by the MLLM is flagged as 'template_filled' in the certificate.

2. INTERACTION GRAPH
   Build directed graph G over edits:
     edge e_i -> e_j (i<j) if F_i ∩ F_j ≠ ∅
                          OR (t_j reads an attribute that w_i writes)
                          OR (R_j references a pixel region changed by w_i)
                          OR (t_i == t_j)  # conservative same-tag edge
                          OR (object_class(e_i) == object_class(e_j))  # conservative same-class edge
   Edge label = interaction type ∈ {spatial_overlap, semantic_write_read, referential, conservative_same_tag, conservative_same_class}
   # Conservative edges are added to guard against missing R_i; they are marked
   # 'conservative' in the certificate and do not override explicit edges.

3. PRIORITY RESOLUTION (deterministic, order-based)
   For each connected component of G, topologically order by request index (later request wins on
   its own footprint); within equal index, resolve by interaction type priority:
   referential > semantic_write_read > spatial_overlap > conservative_same_tag > conservative_same_class.
   This yields a total order π on each component.

4. COMPOSITION
   S' = S
   for e_i in π:  S' = apply(w_i, F_i, S')     # w_i is a partial function; undefined outside F_i
   For every overridden edit e_k, record (e_k, e_i, reason) in the composition certificate.
   If e_k was overridden due to a conservative edge, record reason='conservative_edge'.

5. CANDIDATE VERIFICATION
   MLLM verifies S' against the ORIGINAL user requests in π order; if any request is violated
   because a later edit erased its precondition, mark that edit as 'superseded' in the certificate
   rather than silently dropping it.
   # Verifier equivalence assumption: the MLLM verifier detects precondition erasure from
   # observable pixels. This fails for non-visual preconditions (e.g., object identity),
   # in which case the flag reflects verifier recall rather than true supersession.

6. COMMIT
   Append (S', π, certificate) to H; S' becomes the current state.
```

**Why this design.** We chose an explicit interaction graph with typed edges over WM^MLLM's independent-region assumption because independent-region composition is not closed once footprints overlap or one edit's write feeds another's read, accepting the cost that the MLLM must now emit read-dependency sets, which increases controller output length. We chose a deterministic order-based priority rule (later request wins; ties broken by interaction-type priority) over a learned arbitration head because a learned head would reproduce Anti-Pattern 1's independence assumption on coupled variables, whereas the order rule is a total order on a finite component and therefore always yields a unique normal form, accepting the cost that genuinely simultaneous conflicting edits cannot be represented and must be serialized. We chose to record superseded edits in a composition certificate rather than silently overwrite because the revision history must remain auditable for revisits, accepting a small storage overhead per commit. We added conservative same-tag and same-class edges because the load-bearing assumption that MLLM-emitted R_i is complete is unverified; these edges trade a small number of spurious interactions for robustness to missing dependencies, and they are marked as conservative in the certificate so that the audit record distinguishes explicit from conservative dominance. A domain expert would not describe CEA as a variant of WM^MLLM's edit pipeline, because WM^MLLM applies edits as isolated region writes with no interaction graph and no composition certificate; CEA changes the state's algebraic closure property, not just its edit granularity.

**Why it measures what we claim.** The interaction-graph edge set measures *interacting-edit coupling* because an edge is added only when footprints intersect or a write feeds a later read, under the assumption that MLLM-emitted read-dependency sets are complete for the request's preconditions; this assumption fails when the MLLM omits an implicit precondition (e.g., 'recolor the chair' silently assumes the chair still exists after a prior move), in which case the graph reflects only explicitly stated dependencies and under-counts interaction. To bound this failure, we add conservative same-tag and same-class edges and flag template-filled dependencies in the certificate. The composition certificate's override records measure *closure of the edit algebra* because every overridden edit is paired with the dominating edit and a named reason, under the assumption that the priority order π is total on each connected component; this assumption fails when two edits of equal index and equal interaction type conflict on the same pixel, in which case the certificate reflects an arbitrary but deterministic tie-break rather than a semantically justified dominance. The superseded-edit flag measures *faithfulness of the committed state to the user's request sequence* because a request is flagged only when the MLLM verifier detects that a later edit erased its precondition, under the assumption that the verifier can detect precondition erasure from the panorama alone; this assumption fails when the erased precondition is not visually observable (e.g., an object's identity rather than its pixels), in which case the flag reflects verifier recall rather than true supersession. We therefore pair the verifier metric with a human-annotated faithfulness subset to bound verifier recall.

## Contribution

['A compositional edit algebra (CEA) for persistent MLLM-controlled panorama states, with typed interaction edges, a deterministic order-based priority rule, and a composition certificate that makes multi-edit state transitions closed and auditable.', 'A design principle: interaction-type priority (referential > semantic write-read > spatial overlap) yields a total order on each interaction component, so multi-edit commits have a unique normal form without a learned arbitration module.', 'A composition-certificate representation that distinguishes overridden edits from superseded edits, enabling revisits to report why a prior request is no longer reflected in the current state.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MBench-A multi-edit revisit scenes, restricted to edit types {move, recolor, place, relight} | Tests composition under overlapping edits and revisits |
| Dataset | Synthetic oracle benchmark with ground-truth footprints and dependency sets | Quantifies dependency-set completeness assumption |
| Primary metric | Request-faithfulness rate after composition | Directly measures committed-state fidelity to request sequence |
| Secondary metric | Dependency-set recall (MLLM-emitted vs. oracle) | Measures completeness of R_i |
| Secondary metric | Certificate completeness (fraction of overrides with named reason) | Measures auditability |
| Baseline | WM^MLLM independent-region pipeline | Isolated writes; no interaction graph or certificate |
| Baseline | Sequential overwrite without interaction graph | Later edit blindly overwrites; no supersession record |
| Baseline | Learned arbitration head | Reproduces independence assumption on coupled variables |
| Ablation-of-ours | CEA without composition certificate | Isolates contribution of auditable override records |
| Ablation-of-ours | CEA without conservative edges | Isolates contribution of robustness to missing dependencies |

### Why this setup validates the claim

The central claim is that CEA changes the algebraic closure of WM^MLLM's edit state, so that composed edits remain faithful to the request sequence even when footprints overlap or one edit's write feeds another's read. MBench-A is the right dataset because it already contains revisits and multi-object edits where isolated-region composition breaks; the primary metric, request-faithfulness rate after composition, is the direct observable of closure, since a non-closed algebra silently drops or corrupts earlier requests. The synthetic oracle benchmark is necessary because the load-bearing assumption that MLLM-emitted read-dependency sets are complete is unverified; by measuring dependency-set recall against ground truth and plotting faithfulness as a function of recall, we quantify how much of CEA's gain depends on that assumption rather than hand-waving it. The three baselines each probe a different sub-claim: WM^MLLM's independent-region pipeline tests whether an interaction graph is needed at all; sequential overwrite tests whether ordering alone suffices without typed edges; the learned arbitration head tests whether our deterministic order rule beats a learned resolver on coupled variables. The ablations isolate whether the certificate contributes to faithfulness or is mere bookkeeping, and whether conservative edges recover performance when MLLM dependencies are incomplete. If CEA's gains are concentrated on interacting-edit subsets and vanish on independent-region subsets, the closure claim is supported; if gains are uniform, the mechanism is not doing the claimed work. We also report a human-annotated faithfulness subset (200 examples) to bound verifier recall and separate true supersession from verifier error.

### Expected outcome and causal chain

**vs. WM^MLLM independent-region pipeline** — On a case where a "move chair" edit shifts a chair into the footprint of a later "recolor chair" edit, the baseline produces a recolor applied to the old region because it treats edits as isolated region writes with no interaction graph. Our method instead detects the spatial-overlap edge, orders the recolor after the move, and records the move as dominated, because the interaction graph makes footprint intersection explicit. We expect a noticeable gap on the overlapping-edit subset of MBench-A but near-parity on edits with disjoint footprints.

**vs. Sequential overwrite without interaction graph** — On a case where a "relight" edit writes an attribute that a later "place subject" edit reads (subject shadow direction), the baseline produces an inconsistently lit subject because it applies edits in request order without knowing the write-read dependency. Our method instead adds a semantic_write_read edge, reorders within the component, and flags the superseded edit, because read-dependency sets make the coupling visible. We expect a moderate gap concentrated on write-read-coupled scenes, with parity on purely spatial edits.

**vs. Learned arbitration head** — On a case where two edits of equal index conflict on the same pixels, the baseline produces an arbitration that varies with training distribution and cannot be audited, because a learned head reproduces the independence assumption on coupled variables and lacks a total order. Our method instead yields a deterministic normal form via the tie-break rule and records the reason in the certificate, because order-based priority is total on each component. We expect a small but consistent faithfulness advantage on tied-conflict cases plus a large auditability advantage in certificate completeness.

**Ablation: CEA without conservative edges** — On the synthetic oracle benchmark, we expect faithfulness to degrade as dependency-set recall drops, and conservative edges to recover a portion of that loss; if conservative edges provide no recovery, the completeness assumption is not mitigated and the certificate's uncertainty flags are the only safeguard.

### What would falsify this idea

If CEA's faithfulness gain over WM^MLLM is uniform across independent-region and interacting-edit subsets rather than concentrated where footprints overlap or writes feed reads, the claimed closure mechanism is not what produces the effect. Likewise, if the no-certificate ablation matches full CEA on every subset, the certificate is not contributing to faithfulness and the algebraic-closure claim is unsupported. If faithfulness does not degrade with dependency-set recall on the synthetic oracle benchmark, the load-bearing assumption is not actually load-bearing and the conservative-edge mitigation is unnecessary.

## References

1. WM^MLLM: Turning MLLMs into World Models
