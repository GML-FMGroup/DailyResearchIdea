# Entity Consistency Verification for Reliable Graph Construction in GraphRAG

## Motivation

RAGU's two-stage extraction pipeline does not validate the correctness of extracted entities, causing false negatives and incorrect entities to propagate into the knowledge graph. This limitation arises because the extraction LLM is not explicitly constrained by domain-specific consistency rules, leading to cascade errors in downstream summarization and retrieval. A lightweight verification step after extraction that enforces domain heuristics can catch and correct these inconsistencies, improving graph reliability.

## Key Insight

Domain heuristics define a closed-form constraint satisfaction problem that LLM-extracted entities typically violate due to lack of procedural consistency knowledge; verifying extraction outputs against these constraints corrects errors without requiring additional LLM calls or training data.

## Method

(A) **What it is**: Entity Consistency Verifier (ECV) is a post-extraction module that takes raw (entity, relation) triples from an LLM extractor (e.g., RAGU's two-stage output) and domain-specific heuristics (type constraints, relation arity, attribute ranges) and outputs corrected triples by detecting and fixing inconsistencies via rule-based matching and a lightweight logistic regression classifier for ambiguous cases.

(B) **How it works**:
```pseudocode
Input: raw_extractions = list of (subject, relation, object) triples
       domain_ontology = {entity_types with allowed attributes, relation_schemas with subject/object types}
       heuristics = list of rules (each a pattern: if condition then correction_action)

1. For each triple (s, r, o) in raw_extractions:
   a. Type check: verify that type of s and o match relation schema of r.
      - If mismatch, flag as 'type_error'.
   b. Attribute check: if entity has attributes (e.g., age > 0), ensure they satisfy domain bounds.
   c. Uniqueness check: if entity already exists in graph, compare attributes; if conflicting, flag as 'duplicate_conflict'.

2. For each flagged error:
   - If rule exists in heuristics (e.g., "if type_error and entity has a single candidate type from context, correct type"): apply rule.
   - Else: compute entity embedding from LLM's last hidden layer (stored from extraction), feed into binary logistic regression classifier (pre-trained on domain data) to predict most likely correct type.

3. Apply corrections: update triples accordingly.
4. Output: corrected_triples.

Hyperparameters: logistic regression regularization C=1.0; rule priority order (higher specificity first).
```
(C) **Why this design**: We chose rule-based verification as the primary mechanism over a fully learned approach because heuristics are interpretable, require no training data, and align with domain expert knowledge; this sacrifices coverage of rare edge cases where no rule applies. We placed verification after extraction rather than embedding it into the LLM prompt because it decouples the consistency logic from generation, avoiding prompt engineering complexity and allowing independent updates to heuristics. We use a logistic regression classifier only for ambiguous cases falling through the rules, accepting the small overhead of feature extraction, because it provides a lightweight fallback without requiring large models. The trade-off is that rule maintenance may be costly for domains with rapidly changing ontologies, but for stable domains it is efficient.

(D) **Why it measures what we claim**: The type-check step (1a) operationalizes *entity type consistency* because it assumes the domain ontology exactly specifies valid types per relation; this assumption fails when the ontology is incomplete, in which case the system may incorrectly flag valid entities as errors. The attribute check (1b) measures *attribute validity* because it assumes domain bounds are known and static; this fails when attributes have dynamic ranges (e.g., age in a historical dataset), causing false positives. The duplicate conflict check (1c) measures *entity identity consistency* by assuming that two entities with the same name cannot have conflicting attributes; this fails when entities are truly distinct but accidentally share a name, leading to erroneous merges. The logistic regression fallback measures *type classification confidence* under the assumption that the embedding space separates entity types linearly; this fails when entity embeddings are not linearly separable, in which case the classifier's probability is uncalibrated. Together, these components ensure that each inconsistency detected is tied to a specific domain constraint, with the correction mechanism explicitly designed to restore consistency according to those constraints.

## Contribution

(1) Introduction of Entity Consistency Verifier (ECV), a post-extraction module that applies domain heuristics to correct entity-relation inconsistencies, reducing cascading errors in GraphRAG pipelines. (2) Demonstration that a lightweight rule-based system with a logistic regression fallback can effectively rectify extraction errors without retraining the LLM or requiring additional LLM calls. (3) Analysis of error types (type mismatch, attribute violation, duplicate conflict) and their prevalence in medical graph construction, providing guidance for heuristic design.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GraphRAG-Bench (Medical) | Multi-hop medical QA with gold KG |
| Primary metric | Triple F1 | Directly measures extraction consistency |
| Baseline 1 | RAGU (no verification) | Baseline LLM extraction |
| Baseline 2 | Hearst-pattern IE | Traditional rule-based extraction |
| Ablation | ECV (rules only) | Isolates contribution of logistic regression |

### Why this setup validates the claim
This combination tests the central claim that ECV improves triple consistency. Comparing to RAGU shows improvement over state-of-the-art LLM extraction; comparing to Hearst-pattern tests against simple rules; ablation tests the classifier's value. Triple F1 directly measures consistency against gold KG. The dataset provides gold triples for multi-hop QA, enabling falsifiable test: if ECV outperforms all baselines on Triple F1, claim is supported; if not, claim is rejected.

### Expected outcome and causal chain

**vs. RAGU** — On a case where the LLM extracts "Aspirin" as relation "treats" with object "disease" but domain ontology restricts "treats" to medications, RAGU outputs a wrong triple because no post-hoc check. Our method corrects via type check, so we expect higher precision on entity-type consistency, leading to a noticeable gap in Triple F1 on queries involving relations with strict type constraints.

**vs. Hearst-pattern IE** — On a sentence like "Treatment with X reduced symptoms of Y" with no explicit pattern, Hearst fails to extract any triple. Our method uses LLM extraction then applies attribute and uniqueness checks, recovering the correct (X, reduces, Y) triple. Thus we expect recall improvement on complex sentences, observable as a gap in Triple F1 on clauses with implicit relations.

**vs. ECV (rules only)** — On an ambiguous case like "lung cancer" where it could be disease or symptom, rule-only may mislabel it as disease if no rule exists. Full ECV uses logistic regression on LLM embeddings to correctly classify as symptom, improving accuracy. We expect a small but measurable gain in Triple F1 on cases where rule coverage is incomplete.

### What would falsify this idea
If ECV shows no significant improvement over RAGU on Triple F1, or if the improvement is uniform across all error types rather than concentrated on type/attribute constraints (e.g., also improves on cases with no domain violation), then the specific mechanisms are not causal.

## References

1. RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM
2. From Local to Global: A Graph RAG Approach to Query-Focused Summarization
3. Knowledge Graph Prompting for Multi-Document Question Answering
4. Grape: Knowledge Graph Enhanced Passage Reader for Open-domain Question Answering
5. Visconde: Multi-document QA with GPT-3 and Neural Reranking
