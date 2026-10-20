# Grammar Induction over Procedural Algorithm Primitives for Open-Ended Asset Synthesis

## Motivation

InfiniDream's Symbolically Constrained Asset Generation treats the existing procedural algorithm library as a fixed set of symbolic building blocks, so any object category outside that library cannot be generated at all — the asset space is bounded by library coverage, not by what a VLM could describe. This is not a per-category shortcoming but a structural property of the frontier: the symbolic constraint set is closed under instantiation (new seeds, new parameters) yet open only under no operation, so compositional novelty is impossible by construction. The root cause is that the grammar of algorithms itself is never induced; only the grammar of parameters is, and no prior work in the Idea Cards supplies the missing induction step.

## Key Insight

If the procedural library is re-expressed as a typed primitive algebra whose composition operators are themselves discovered from pairs of existing algorithms, then a novel category can be reduced to a composition specification that is guaranteed to type-check against the primitive signatures — making the asset grammar closed under composition rather than merely under parameterization.

## Method

**What it is.** META-PCG (Meta-Procedural Content Grammar) takes a text description of a novel object category and the existing procedural library, and outputs a new executable procedural algorithm expressed as a typed composition of library primitives. It has two coupled phases: a *composition-operator induction* phase that mines reusable higher-order operators from the library itself, and a *bridge_add* phase that synthesizes a new algorithm by applying an induced operator to a typed slot assignment derived from the text.

**Load-bearing assumption (stated explicitly).** The method assumes that composition operators induced by splicing pairs of existing library algorithms (top-64, ranked by reuse × non-triviality) are sufficient to express the compositions required by genuinely novel object categories — i.e., that the operator set mined from L×L pairs covers the higher-order structure needed for categories outside the library's coverage. This is a closure assumption on a finite, self-generated operator set: the operators are induced only from pairs of algorithms already in L, so the induced grammar's expressive reach is bounded by the combinatorics of L, not by the space of novel categories. Splicing pairs of library members is itself a closed operation over L×L, so the induced operator set is a function of L and cannot in general escape L's structural vocabulary. The top-K=64 truncation compounds this: dedupe_and_rank by reuse_count actively selects for operators that recur across L, i.e., for the most library-typical compositions, biasing the operator set toward the categories already covered and away from the rare compositions novel categories would need. We therefore treat the induced operator set as a *seed prior* rather than a hard ceiling: when no induced operator closes the invariant set, the VLM may propose a new composition operator (a higher-order lambda over primitive signatures), which is type-checked against the primitive signatures rather than rejected outright. This preserves the typed-composition contribution (the type-check and invariant-closure machinery) while removing the finite-operator-set closure that would otherwise make the expressivity claim circular; the top-K=64 set becomes a prior/retrieval aid rather than the hard ceiling on expressible categories.

**How it works.**

```pseudocode
# Phase 0: primitive typing (once, offline)
for each algorithm A in library L:
    sig[A] = (inputs, outputs, invariants, params)   # invariants = structural facts (manifold? closed? symmetric?)

# Phase 1: composition-operator induction (once, offline)
D = []                                  # discovered operators
for each ordered pair (A, B) in L x L with compatible sig:
    for each injection point p in A.outputs matching B.inputs:
        C = splice(A, B, p)             # concrete composed program
        if executes(C) and not is_trivial(C, A, B):
            op = abstract(C, A, B, p)   # lambda over the two slots + injection point
            D.append(op)
D = dedupe_and_rank(D, by=reuse_count * non_triviality)   # keep top-K, K=64

# Phase 2: bridge_add (per novel category, given text T)
spec = VLM_parse(T) -> CompositionSpec{ required_invariants, part_decomposition, symmetries }
slots = []
for part in spec.part_decomposition:               # e.g., trunk / canopy / roots
    candidates = [A for A in L if satisfies(sig[A].invariants, part.invariants)]
    slot = argmax_{A in candidates} invariant_match(sig[A].invariants, part.invariants)
    slots.append(slot)
op = argmax_{op in D} arity_match(op, len(slots)) * invariant_closure(op, spec.required_invariants)
if op is None or not type_checks(op(*slots), spec.required_invariants):
    # open operator vocabulary: VLM proposes a new higher-order lambda over primitive signatures
    op = VLM_propose_operator(spec, slots, sig)   # lambda over slots, type-checked below
program = op(*slots)                               # typed composition, must type-check
assert type_checks(program, spec.required_invariants)   # reject + repair loop, max 3 retries
return program
```

Hyperparameters: top-K=64 induced operators (seed prior); invariant_match is a Jaccard score over named structural invariants (not embeddings); up to 3 type-repair retries per category; VLM_propose_operator is invoked only when no induced operator type-checks, and its output is subject to the same type-check and repair loop.

**Why this design.** We chose to induce composition operators from *existing* algorithm pairs rather than to have the VLM write free-form code over primitives, because free-form code generation is exactly what InfiniDream already does and it inherits the library-coverage bound; inducing operators from pairs makes the operators themselves carry type signatures, accepting the cost that the discovered operator set is finite (top-64) and may miss categories requiring genuinely new control flow — which is why the VLM may propose a new operator when no induced one closes the invariant set, type-checked against primitive signatures rather than rejected outright. We chose a typed slot assignment over a nearest-neighbor descriptor match because descriptor retrieval is precisely the retrieval baseline (Toolformer/Gorilla-style tool selection) that cannot produce a *new* program — only select an existing one — accepting the cost that invariant names must be declared in the library metadata. We chose a hard type-check with a bounded repair loop over a soft score-and-rank because a soft score admits programs that execute but violate the category's required invariants, accepting the cost of occasional outright rejection when no induced operator closes the invariant set. A domain expert would not call this a variant of InfiniDream's symbolic constraint method: InfiniDream constrains *code generation* with primitives as blocks, whereas META-PCG induces the *composition grammar* over primitives and never asks the VLM to emit code at all.

**Why it measures what we claim.** The invariant_match score between a primitive's declared structural invariants and a part's required invariants measures *category faithfulness* because the assumption is that the named invariants (e.g., 'closed surface', 'bilateral symmetry', 'branching topology') are necessary conditions for the part to belong to the category; this assumption fails when a category is defined by an invariant not in the declaration vocabulary, in which case the score reflects lexical overlap of invariant names rather than true structural membership. The operator's invariant_closure score measures *compositional validity* because the assumption is that an induced operator preserves the intersection of its slots' invariants, which holds by construction of the splice only when the injection point preserves manifoldness; it fails when the splice introduces a non-manifold junction, in which case the score reflects slot compatibility rather than the composed program's actual validity. The type-check rejection rate measures *library expressivity* because the assumption is that a category is expressible iff some induced operator type-checks against it; this fails when the library contains the needed primitives but no induced operator of the right arity, in which case the rate reflects operator-arity coverage rather than true expressivity.

**(D) Linking assumptions and failure modes (operationalized).** For each computational quantity in the pseudocode that operationalizes a motivation-level concept, we state the assumption and the failure mode, and we log the empirical frequency of each failure mode per task:
- *invariant_match* measures category faithfulness under assumption A1: the declared invariant vocabulary is complete and necessary for each benchmark category. Failure mode F1: a category is defined by an invariant absent from the vocabulary; then the score reflects lexical overlap of invariant names, not structural membership. We log, per task, whether each required invariant appears in the declaration vocabulary (binary flag per invariant) and report the frequency of F1.
- *invariant_closure* measures compositional validity under assumption A2: the induced operator preserves the intersection of its slots' invariants, which holds by construction of the splice only when the injection point preserves manifoldness. Failure mode F2: the splice introduces a non-manifold junction; then the score reflects slot compatibility rather than the composed program's actual validity. We log, per task, whether the chosen splice junction preserves manifoldness (checked on the executed mesh) and report the frequency of F2.
- *type-check rejection rate* measures library expressivity under assumption A3: a category is expressible iff some induced operator type-checks against it. Failure mode F3: the library contains the needed primitives but no induced operator of the right arity; then the rate reflects operator-arity coverage rather than true expressivity. We log, per task, whether the library contains primitives satisfying the part invariants but no induced operator of matching arity, and report the frequency of F3.

## Contribution

(1) A meta-procedural grammar induction algorithm that mines typed higher-order composition operators from an existing procedural library and uses them to synthesize new algorithms for out-of-library categories via a bridge_add composition step. (2) The design principle that closing a symbolic asset grammar under compositional novelty requires inducing the composition operators rather than the parameter values, demonstrated by a measurable separation between categories reachable by parameter instantiation and categories reachable only by operator composition. (3) A typed invariant vocabulary and type-checking procedure that makes synthesized procedural algorithms verifiably faithful to category-declared structural constraints rather than merely executable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 100 novel-category object tasks from InfiniDream benchmark | Same task distribution as nearest prior work |
| Primary metric | Category-invariant satisfaction rate on rendered output | Directly tests faithfulness, not just executability |
| Baseline | InfiniDream symbolic-constraint codegen | Tests whether induced grammar beats constrained free-form code |
| Baseline | Retrieval / tool-selection over library primitives | Tests whether composition beats nearest-neighbor primitive pick |
| Baseline | Untyped free-form VLM codegen over primitives | Tests whether typing and invariant closure matter |
| Baseline | DreamCoder-style library learning (or LLM-driven operator abstraction) on the same 100 tasks | Tests whether the induced-operator set is reproducible by existing library-learning methods |
| Ablation-of-ours | META-PCG with top-K=0 (no induced operators; slots only) | Isolates contribution of composition-operator induction |
| Ablation-of-ours | META-PCG with top-K in {8,16,32,64,128} | Tests scaling behavior: whether the approach opens the asset space or merely shifts the coverage bound |
| Diagnostic logging | Per-task flags for F1 (invariant absent from vocabulary), F2 (non-manifold splice junction), F3 (no induced operator of matching arity) | Grounds the linking assumptions A1–A3 with empirical failure-mode frequencies |
| Feasibility scope | Curated subset of 20–30 primitives with pre-declared invariants; report wall-clock cost of Phase 1 induction | Establishes feasibility before scaling to full library |

### Why this setup validates the claim

The central claim is that inducing a typed composition grammar over existing primitives yields *new* executable programs that satisfy a novel category's structural invariants, where free-form codegen and primitive retrieval cannot. The dataset is the same 100 novel-category tasks used by InfiniDream, so any gain is not attributable to an easier distribution. The three original baselines each attack a distinct sub-claim: InfiniDream's symbolic-constraint codegen tests whether constraining code generation with primitives is sufficient (it should fail when the needed composition is not a single primitive call); retrieval/tool-selection tests whether selecting an existing algorithm suffices (it should fail whenever no single library algorithm matches the category's invariant set); untyped free-form codegen tests whether typing and invariant closure matter (it should fail on manifold/symmetry violations that a type-check would reject). The added DreamCoder-style library-learning baseline tests whether our induced-operator set is reproducible by an existing library-learning method, isolating the contribution of our specific induction procedure. The top-K=0 ablation isolates the operator-induction phase: if gains come from induced operators rather than from typed slot assignment alone, the ablation must lose most of the benefit. The top-K sweep {8,16,32,64,128} tests whether increasing the operator set monotonically improves category-invariant satisfaction or saturates, distinguishing genuine opening of the asset space from a mere shift of the coverage bound. The per-task diagnostic logging of F1–F3 grounds assumptions A1–A3: if F1 is frequent, invariant_match reflects lexical overlap; if F2 is frequent, invariant_closure reflects slot compatibility; if F3 is frequent, the type-check rejection rate reflects operator-arity coverage. The primary metric — category-invariant satisfaction on the *rendered* output, not on the program text — is the right detector because our whole mechanism is a claim about structural invariants surviving composition; a metric on executability would be passed trivially by all baselines.

### Expected outcome and causal chain

**vs. InfiniDream symbolic-constraint codegen** — On a case where the category requires a part decomposition (trunk + branching canopy) whose composition is not any single library primitive, the baseline produces code that calls one factory per part but violates the branching-topology invariant at the junction because its constraint set operates on primitive *selection*, not on a typed composition operator with an invariant-closure guarantee. Our method instead selects an induced operator whose signature closes over the required invariants and applies it to typed slots, so the junction is type-checked before execution. We expect a noticeable gap on multi-part categories with cross-part invariants but near-parity on single-part categories.

**vs. Retrieval / tool-selection over primitives** — On a case where the novel category is genuinely novel (no library algorithm's invariant set matches), the baseline produces the closest existing algorithm, which executes but fails the category's defining invariant (e.g., returns a closed surface where an open branching structure is required) because retrieval can only select, never compose. Our method instead composes two primitives through an induced operator whose invariant_closure matches the spec, producing a program that did not exist in the library. We expect a large gap concentrated on the subset of tasks where the nearest library algorithm's invariant Jaccard is below a threshold, and parity where a near-exact match exists.

**vs. Untyped free-form VLM codegen over primitives** — On a case where the required composition introduces a non-manifold junction (e.g., roots merging into a trunk with mismatched boundary), the baseline produces code that executes but yields a non-manifold mesh because nothing in free-form generation checks the junction's structural invariants. Our method instead rejects the candidate at type-check and repairs within the bounded loop, or falls back to an operator whose splice preserves manifoldness. We expect the gap to appear as a lower rate of invariant violations in rendered output, with the largest separation on categories whose invariants include manifoldness or symmetry.

**vs. DreamCoder-style library learning (or LLM-driven operator abstraction)** — On a case where the novel category requires a composition whose operator is not among the top-64 induced by our reuse × non-triviality ranking, the baseline's library-learning objective (compression-driven) may recover a different operator set that either misses the needed composition or overfits to frequent library patterns. Our method's open operator vocabulary allows the VLM to propose a new operator when no induced one closes the invariant set, type-checked against primitive signatures. We expect parity on categories whose needed composition is frequent in L, and a gap on categories whose needed composition is rare, where the baseline's compression objective biases away from the rare operator.

**Top-K sweep** — We expect category-invariant satisfaction to increase from K=8 to K=64 and then saturate or slightly degrade at K=128 (due to noisier operator ranking), with the saturation point indicating whether the induced operator set is the binding constraint or whether the open operator vocabulary is doing the work. If satisfaction is flat across K, the induced operator set is not the causal mechanism.

**Diagnostic logging** — We expect F1 (invariant absent from vocabulary) to occur on a minority of tasks; if it is frequent, invariant_match is not measuring structural membership. We expect F2 (non-manifold splice junction) to occur on a subset of multi-part tasks; if it is frequent, invariant_closure is not measuring composed-program validity. We expect F3 (no induced operator of matching arity) to occur on tasks where the library has the primitives but not the operator; if it is frequent, the type-check rejection rate is not measuring expressivity.

### What would falsify this idea

If META-PCG's gain over the untyped free-form baseline is uniform across all task subsets rather than concentrated on tasks where the predicted failure mode (cross-part invariants, no near-exact library match, non-manifold junctions) occurs, the central claim that induced typed composition is the causal mechanism is wrong. Equivalently, if the top-K=0 ablation matches the full method, operator induction contributes nothing and the idea reduces to typed slot assignment; and if the diagnostic logging shows F1, F2, or F3 is frequent, the corresponding linking assumption (A1, A2, or A3) is false and the metric reflects the stated failure mode rather than the claimed concept.

## References

1. InfiniDream: Open-Ended Photorealistic 3D Worlds Through Procedural Generation
