# Semantic Verification Oracle for Procedural Asset Generation via Scene-Graph Constraint Satisfaction

## Motivation

Procedural asset generation frameworks such as InfiniDream rely on VLM-driven code synthesis constrained by symbolic building blocks, but they lack any mechanism to verify that the generated procedural parameters faithfully encode the input text's semantics. This gap is structural: the VLM translates text directly into code without an intermediate formal representation of what the text requires, so there is no reference against which the output can be checked. Without such verification, photorealistic assets may be semantically incorrect—missing objects, wrong spatial relations, or incorrect attributes—undermining the core promise of text-to-world generation.

## Key Insight

A formal scene graph extracted from the text serves as a complete, checkable specification of semantic requirements, and any procedural asset can be verified against it by compiling graph constraints into a differentiable rendering loss whose minimum is zero exactly when the asset satisfies the graph.

## Method

**What it is.** SVO (Semantic Verification Oracle) takes a text prompt and a procedurally generated 3D asset (as produced by InfiniDream's Symbolically Constrained Asset Generation) and outputs a three-way verdict (pass / fail / underdetermined) plus, on failure, a set of violated constraints that guide repair. It operates by extracting a formal scene graph from the text, compiling that graph into a set of differentiable rendering objectives, and checking the asset against those objectives. The load-bearing assumption is stated explicitly: **the LLM-extracted scene graph G is treated as a complete and correct specification of the text's semantic requirements, so that minimizing the compiled constraint loss to zero is equivalent to semantic faithfulness of the asset.** This assumption is made explicit and is partially mitigated by the extraction-uncertainty handling and textual-entailment check described below; the oracle never certifies faithfulness for semantics it cannot check.

**How it works.**

```pseudocode
Input: text prompt T, procedural asset A (mesh + parameters), threshold tau, confidence threshold gamma=0.5, entailment threshold eta=0.7
Output: verdict v in {pass, fail, underdetermined}, violation set V, unverified set U

1. SCENE GRAPH EXTRACTION
   G, conf = LLM_extract(T)   # nodes: objects with attributes; edges: spatial/functional relations
   # G is validated against a fixed ontology (object types, relation types) via schema checking
   # conf: per-constraint confidence in [0,1] from the extractor (e.g., token log-probability of the relation/attribute span)
   # Textual entailment check: for each constraint c in G, compute entailment score s_c = entail(T, verbalize(c))
   #   using a held-out NLI model (e.g., DeBERTa-v3-large fine-tuned on MNLI, frozen).
   #   If s_c < eta, mark c as unverified (do not compile into loss).

2. CONSTRAINT COMPILATION
   For each node n in G.nodes:
     C_n = { existence(n), attribute_match(n, A) }   # attribute_match uses CLIP-style embedding distance < eps_attr (eps_attr=0.25)
   For each edge e = (n1, r, n2) in G.edges:
     C_e = spatial_constraint(r, n1, n2)   # e.g., 'above' -> y(n1) > y(n2) + delta (delta=0.1 in normalized coordinates)
   C = union of all C_n and C_e
   # Filter: keep only constraints with conf >= gamma AND entailment score >= eta; others go to U (unverified).

3. DIFFERENTIABLE VERIFICATION
   Render A from K=8 fixed viewpoints using a differentiable renderer R
   For each constraint c in C:
     L_c = constraint_loss(c, R(A))   # 0 if satisfied, >0 otherwise
   L_total = sum_c w_c * L_c   # weights w_c = 1 by default

4. VERDICT AND REPAIR SIGNAL
   if L_total < tau: v = pass
   else:
     v = fail
     V = {c : L_c > tau_c}   # top violated constraints (tau_c = 0.1 per constraint)
     # repair: re-prompt VLM with V to regenerate A
   if U is non-empty and L_total < tau: v = underdetermined  # cannot certify due to unverified constraints
```

**Why this design.** We chose LLM-based scene graph extraction over direct VLM scoring because a graph provides discrete, checkable constraints that can be individually verified and repaired, accepting the cost that extraction may miss implicit semantics. We chose differentiable rendering over discrete mesh inspection because it yields gradients that localize violations to specific objects and relations, accepting the cost of rendering overhead. We chose a fixed ontology for graph validation over open-vocabulary extraction to ensure constraints are compilable, accepting reduced coverage of unusual objects. This design is not a variant of retrieval-based verification (e.g., CLIP score thresholding) because it computes a structured constraint satisfaction signal that retrieval cannot: it identifies which specific relation is violated, not just overall similarity. The extraction-uncertainty and entailment checks are added to make the load-bearing assumption explicit and to avoid certifying semantics that the extractor is not confident about.

**Why this measures what we claim.** The constraint loss L_c for an edge (n1, r, n2) measures semantic faithfulness of relation r between objects n1 and n2 because we assume the differentiable renderer accurately reflects 3D geometry and the spatial constraint function correctly encodes the relation's semantics; this assumption fails when the renderer's viewpoint sampling misses occluded relations, in which case L_c reflects visibility rather than semantic violation. The attribute_match distance measures attribute faithfulness because we assume the CLIP embedding space aligns visual attributes with text descriptions; this assumption fails for fine-grained attributes (e.g., material texture), in which case it reflects coarse category similarity. The total loss L_total measures overall semantic faithfulness because we assume constraints are independent and equally weighted; this assumption fails when constraints interact (e.g., 'above' and 'near'), in which case L_total reflects a weighted combination that may mask individual violations. The three-way verdict with unverified set U measures the oracle's ability to avoid false passes: it returns 'underdetermined' when the extracted graph is incomplete or low-confidence, so it does not certify faithfulness it cannot check.

## Contribution

['A closed-loop semantic verification oracle (SVO) that parses text into a formal scene graph, compiles it into differentiable rendering constraints, and checks procedural assets against it, enabling rejection and repair of semantically unfaithful generations.', 'A constraint compilation scheme that translates scene graph edges into differentiable spatial and attribute losses, providing localized violation signals that guide VLM-based repair.', 'An empirical demonstration that SVO detects semantic errors in InfiniDream-generated assets that perceptual metrics miss, and that repair guided by SVO violations improves faithfulness.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | InfiniDream's 100 object + 20 scene tasks | Matches proposed method's asset generation pipeline |
| Primary metric | Constraint satisfaction rate (fraction of extracted constraints satisfied) | Directly measures structured verification signal, not similarity |
| Secondary metric | Extraction fidelity: F1 between SVO's extracted scene graph and human-annotated ground-truth scene graph | Breaks circularity by validating G against human reference |
| Secondary metric | False pass rate: fraction of assets with L_total < tau but human-judged semantic violation | Directly measures the load-bearing assumption failure |
| Baseline | VLM-guided procedural method (InfiniDream) | Tests whether structured constraints beat API-guided generation |
| Baseline | Direct VLM scoring (CLIP threshold) | Tests whether graph localization beats holistic similarity |
| Baseline | LLM scene graph extraction without differentiable rendering | Tests whether differentiable verification adds value over static parsing |
| Ablation-of-ours | SVO with fixed weights (w_c=1) vs. learned constraint weights | Tests whether uniform weighting masks interacting constraints |
| Ablation-of-ours | SVO without extraction-uncertainty/entailment filtering (i.e., compile all constraints regardless of conf/entailment) | Tests whether the uncertainty handling reduces false passes |

### Why this setup validates the claim

The central claim is that SVO's structured constraint satisfaction signal localizes semantic violations better than holistic similarity, enabling targeted repair. The dataset choice is critical: InfiniDream's tasks include complex spatial relations and attributes that expose the gap between discrete constraint checking and VLM scoring. The primary metric—constraint satisfaction rate—is the right detector because it directly quantifies the signal SVO computes, and baselines cannot produce it without a scene graph. The secondary metrics address the load-bearing assumption: extraction fidelity (F1 against human-annotated graphs) measures how well G captures the text's semantics, and false pass rate measures how often SVO certifies a semantically wrong asset. Each baseline isolates a sub-claim: VLM-guided procedural tests whether structured constraints beat API-guided generation; direct VLM scoring tests whether graph localization outperforms holistic similarity; LLM-only extraction tests whether differentiable rendering adds verification value. The ablation tests whether uniform constraint weighting masks interacting violations, a predicted failure mode. The uncertainty ablation tests whether the confidence/entailment filtering reduces false passes. Together, these form a falsifiable test: if SVO's advantage is uniform across task subsets rather than concentrated where baselines' mechanism gaps predict failures (e.g., spatial relations, fine-grained attributes), or if extraction fidelity is low and false pass rate is high, the claim is wrong.

### Expected outcome and causal chain

**vs. VLM-guided procedural method (InfiniDream)** — On a case where the prompt requires a precise spatial relation like "a lamp above a table with a book to its left," the baseline produces an asset with correct objects but wrong placement because its VLM guidance offers only holistic feedback, not per-relation constraints. Our method instead extracts the scene graph, compiles spatial constraints, and uses differentiable rendering to localize the violated relation, so we expect a noticeable gap on spatial-relation-heavy tasks but parity on simple single-object tasks. Additionally, we expect SVO's extraction fidelity to be high on these explicit spatial relations (F1 > 0.8) and false pass rate to be low (< 5%).

**vs. Direct VLM scoring (CLIP threshold)** — On a case where the prompt specifies a fine-grained attribute like "a matte ceramic vase," the baseline passes the asset because CLIP similarity is high for a glossy vase (coarse category match) and provides no violation signal. Our method instead uses attribute_match distance and, on failure, returns the specific violated attribute constraint, so we expect higher constraint satisfaction on attribute-sensitive tasks and a larger gap on fine-grained materials than on coarse categories. However, we expect extraction fidelity for fine-grained attributes to be lower (F1 ~ 0.6) and false pass rate higher (~10%) due to CLIP's coarse alignment, which the uncertainty filtering should partially mitigate.

**vs. LLM scene graph extraction without differentiable rendering** — On a case where the prompt implies occlusion or viewpoint-dependent relations (e.g., "a chair behind a desk"), the baseline's static graph may miss or misparse the relation because it lacks geometric grounding, producing a false pass. Our method instead checks the relation via differentiable rendering from multiple viewpoints, so we expect fewer false passes on occlusion-sensitive tasks and a gap concentrated on relations requiring 3D verification rather than purely lexical ones. We also expect the entailment check to flag some implicit relations as unverified, leading to 'underdetermined' verdicts rather than false passes.

### What would falsify this idea

If SVO's gain over baselines is uniform across all task subsets rather than concentrated on tasks where baselines' mechanism gaps predict failures (spatial relations, fine-grained attributes, occlusion), or if the LLM-only ablation matches SVO's constraint satisfaction, or if extraction fidelity is low (F1 < 0.5) and false pass rate is high (> 20%) even with uncertainty filtering, then the central claim that differentiable structured verification adds value is wrong.

## References

1. InfiniDream: Open-Ended Photorealistic 3D Worlds Through Procedural Generation
