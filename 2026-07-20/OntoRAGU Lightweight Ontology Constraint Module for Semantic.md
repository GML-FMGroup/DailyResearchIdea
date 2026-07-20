# OntoRAGU: Lightweight Ontology Constraint Module for Semantic Consistency in GraphRAG

## Motivation

RAGU (Multi-Step GraphRAG Engine) constructs a knowledge graph from documents using LLM extraction but discards explicit ontology grounding, leading to type and relation inconsistencies (e.g., entity assigned wrong type, relation with invalid domain/range). These errors propagate to downstream reasoning. Existing post-hoc correction methods are either heavy (require retraining) or use global constraints without domain specificity. We argue that a lightweight ontology constraint module applied after graph construction can resolve inconsistencies with minimal overhead.

## Key Insight

Ontology constraints provide a finite set of admissible type and relation patterns that the data-driven graph is likely to violate, and these violations can be corrected by a simple rule-based check with LLM fallback because the violations are sparse and localized.

## Method

**(A) What it is**: OntoRAGU is a lightweight, plug-in module that validates and corrects entity types and relation assignments in a GraphRAG-constructed KG using a domain ontology, ensuring semantic coherence without modifying the base extraction pipeline. Input: RAGU output KG (entities with types, relations with labels), domain ontology (types, relation domains/ranges). Output: corrected KG.

**(B) How it works**:
```pseudocode
Input: G = (E, R, T)  // entities, relations, types from RAGU
       O = (Ty, Rel, domain, range)  // ontology: types, relation types, domain/range maps
       confidence: Map<triple, float>  // extraction confidence from RAGU (LLM probability)
Output: G' = corrected KG

Step 1: Type validation for each entity e in E:
    if e.type not in O.types:
        similarity_function = cosine similarity via GloVe 300d embeddings
        candidates = [t for t in O.types if similarity(e.label, t) > threshold]  // threshold=0.7
        if candidates:
            e.type = t* = argmax similarity(e.label, t)
            assign confidence = max similarity(e.label, t)
        else:
            remove entity  // trade-off: remove vs assign generic

Step 2: Relation validation for each relation r=(s, p, o) in R:
    if p not in O.rel: remove r
    else:
        valid_subject = (type(s) in O.domain[p]) or (len(O.domain[p])==0)
        valid_object = (type(o) in O.range[p]) or (len(O.range[p])==0)
        if not (valid_subject and valid_object):
            if use_llm_fallback:  // True, LLM=GPT-3.5-turbo
                prompt = "Is the triple (" + s.label + ", " + p + ", " + o.label + ") plausible in the medical domain? Respond with yes or no."
                LLM response = query_LLM(prompt)
                if LLM_response == "yes": keep r (mark as LLM-validated)
                else: remove r
            else:
                remove r

Step 3: Consistency check: If any entity has conflicting types (e.g., assigned two disjoint types), resolve by removing the less specific or using LLM.

Step 4 (fallback for ontology incompleteness): For each triple r removed in Step 2, if confidence[r] > 0.9, then retain r in G' but flag for human review or incremental ontology extension.

Return G'
```

**(C) Why this design**: We chose rule-based type validation over retraining a joint model because the ontology is static and violations are sparse, making rule-based correction efficient and interpretable. The trade-off is that rule-based mapping may misclassify entities with ambiguous labels; using LLM fallback adds cost but resolves ambiguity. We chose to remove invalid relations rather than relabel them because relation relabeling would require generating plausible alternative relations, which is complex and may introduce hallucination. This design prioritizes precision over recall, accepting that some valid but untyped relations may be lost. We chose to run validation after graph construction (post-hoc) rather than during extraction to avoid modifying RAGU's pipeline, making the module plug-and-play. The cost is that errors already generated cannot be prevented, only corrected. Finally, we chose to use a similarity threshold for type mapping rather than always using LLM to keep the module lightweight for large graphs. **Load-bearing assumption**: This design assumes the ontology O is complete and correct for the target domain. Violations may cause valid triples to be removed. Step 4 mitigates this by retaining high-confidence triples and flagging them for review.

**(D) Why it measures what we claim**: The removal of relations whose subject/object types violate ontology domain/range (Step 2) measures *semantic coherence* because the ontology enforces that a relation can only connect entities of specific types; this assumption holds when the ontology is complete and accurate, but fails when the ontology is incomplete (e.g., missing a valid relation type) or when the entity type was incorrectly assigned earlier, in which case a valid relation may be erroneously removed. The reduction in type violations after Step 1 measures *type consistency* because the ontology defines admissible types; this measure relies on the assumption that the ontology's type hierarchy captures all relevant distinctions, which fails when the ontology does not cover domain-specific subtypes (e.g., medical sub-entities not in ontology), causing correct labels to be overwritten. The overall change in graph accuracy is not measured directly; instead we proxy it by the number of corrected inconsistencies, assuming that ontology-validated graphs have higher semantic accuracy. This assumption fails when the ontology itself contains errors or the validation process introduces spurious corrections. The addition of Step 4 provides a safety net: high-confidence triples that violate ontology are retained and flagged, so that the metric is not artificially inflated by removing valid triples.

## Contribution

(1) A lightweight, plug-in ontology constraint module for GraphRAG pipelines that post-hoc corrects type and relation inconsistencies using a rule-based approach with LLM fallback. (2) Empirical validation that ontology correction improves downstream QA answer consistency (e.g., reduces contradictions) on GraphRAG-Bench Medical dataset, without requiring retraining of the base extraction model. (3) Analysis of trade-offs between rule-based and LLM-based correction in terms of precision, recall, and computational cost.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|-------|--------|-----------------------|
| Dataset | GraphRAG-Bench (Medical) | Rich ontology coverage and entity types. |
| Primary metric | Ontology-validated precision | Directly measures semantic coherence. |
| Secondary metric | Downstream QA accuracy on 100 MedQA questions | Tests practical impact on question answering. |
| Baseline 1 | RAGU (original) | Baseline without any correction. |
| Baseline 2 | RAGU + LLM correction (no ontology) | Tests LLM correction without ontology. |
| Ablation-of-ours | Ours without LLM fallback | Isolates effect of rule-based correction. |
| Robustness test | Perturbed ontology (randomly remove 10% of types and 20% of relation constraints) | Tests sensitivity to ontology completeness. |

### Why this setup validates the claim

This combination forms a falsifiable test because it decomposes the contribution of each component. The dataset provides a known ontology, allowing ground-truth evaluation of semantic coherence. Comparing our method against RAGU (no correction) directly tests whether any correction improves metric. Against RAGU+LLM (no ontology) tests whether ontology guidance is necessary beyond LLM correction. The ablation (rule-only) isolates the effect of rule-based type mapping. The primary metric, ontology-validated precision, captures exactly what the method aims to improve: the proportion of relations conforming to domain constraints. The secondary metric, downstream QA accuracy, measures whether corrections translate into better performance on a practical task. The robustness test with perturbed ontology tests the sensitivity to the load-bearing assumption of ontology completeness; if our method degrades significantly on perturbed ontology, it confirms that the method relies on accurate ontology.

### Expected outcome and causal chain

**vs. RAGU** — On a case where RAGU extracts a relation like (patient, prescribed, aspirin) but the type of "aspirin" is mislabeled as "disease", the baseline keeps the triple unchanged, producing a semantic violation. Our method detects that "disease" is not in the ontology's types, maps "aspirin" to "drug" via similarity, and validates the relation domain/range (prescribed domain: patient, range: drug), so the triple remains correct. We expect our method to show a noticeable gain in ontology-validated precision on examples with type misclassification, while parity on clean examples. On downstream QA, our method should yield higher accuracy because the corrected graph provides more accurate facts.

**vs. RAGU+LLM correction** — On a case where RAGU outputs a relation (doctor, treats, patient) but the relation "treats" is not in the ontology, LLM-only correction may hallucinate a plausible but nonexistent relation (e.g., "cures") or fail to remove it, still violating ontology. Our method removes the relation because it is not in the ontology's relation set, avoiding hallucination. We expect our method to achieve higher precision at the cost of slightly lower recall, especially on rare relations not covered by ontology, where LLM-only may keep incorrect ones. Downstream QA accuracy may be slightly lower for our method due to recall loss, but overall precision benefits should yield net improvement.

**Robustness test with perturbed ontology** — On a case where the ontology is missing a valid type, say "antibiotic" is removed, our method may map "amoxicillin" to a more general type "drug" (which is still valid for prescriptions) or, if similarity threshold is not met, remove the entity. This could degrade precision and recall. We expect our method to show a smaller drop in performance compared to a baseline that does not have the Step 4 fallback, because high-confidence triples are retained.

### What would falsify this idea

If our method's gain in ontology-validated precision is uniform across all relation types rather than concentrated on those that violate type constraints, or if the rule-only ablation matches our full method even on ambiguous cases, then the central claim that ontology-guided correction and LLM fallback are both necessary would be falsified. Also, if downstream QA accuracy does not improve despite improved ontology-validated precision, the practical significance claim is weakened.

## References

1. RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM
2. From Local to Global: A Graph RAG Approach to Query-Focused Summarization
3. Knowledge Graph Prompting for Multi-Document Question Answering
4. Grape: Knowledge Graph Enhanced Passage Reader for Open-domain Question Answering
5. Visconde: Multi-document QA with GPT-3 and Neural Reranking
