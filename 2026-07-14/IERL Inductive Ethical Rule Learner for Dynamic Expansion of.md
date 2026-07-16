# IERL: Inductive Ethical Rule Learner for Dynamic Expansion of Culturally-Grounded Moral Reasoning

## Motivation

Theory-grounded methods like MET rely on expert-curated theory grounds that do not scale to novel cultural contexts, creating a coverage bottleneck. Inductive logic programming can derive ethical rules from few examples, but prior work lacks a mechanism to ensure new rules are consistent with universal ethical axioms, risking contradictions that undermine trustworthiness. MET's static grounds cannot dynamically adapt, and no existing ILP approach for moral reasoning addresses this consistency requirement.

## Key Insight

By constraining rule induction to be logically entailed by a fixed set of universal ethical axioms, new rules inherit validity guarantees and can be automatically added to the theory ground set without human oversight.

## Method

### (A) What it is
IERL takes a handful of culturally annotated moral dilemmas (few-shot) and a background knowledge base (KB) of universal ethical axioms (e.g., "do no harm", "respect autonomy") as inputs, and outputs a set of interpretable logical rules per culture that are consistent with the axioms.

### (B) How it works
```pseudocode
Input: 
  - Few-shot examples E = {(context_facts, moral_judgment)} for a target culture
  - Universal axiom KB = {horn_clauses} (e.g., permissible(A) :- not(harmful(A)))
  - Pretrained multilingual sentence encoder (e.g., LaBSE)
Output: Set of induced rules R (Horn clauses) for the culture

Procedure IERL(E, KB):
    // Phase 1: Induce candidate rules via ILP
    Background = fact predicates from E (e.g., agent_action, victim_condition)
    Target = moral_judgment predicate
    CandidateRules = FOIL(Target, Background, E)  // standard FOIL; max_clause_length=5
    
    // Phase 2: Filter by logical consistency
    ConsistentRules = empty set
    for each rule in CandidateRules:
        if not(entails(rule, contradiction)) and KB ∪ {rule} is consistent:
            // Use Prolog-style theorem prover; consistency = no false from KB+rule
            ConsistentRules = ConsistentRules ∪ {rule}
    
    // Phase 3: For novel cultures, optionally transfer rules
    if culture is novel (no prior rules):
        TransferRules = empty
        SimilarCultures = find_top_k_similar(culture, embedding_space)  // k=3
        for each similar_culture in SimilarCultures:
            for each rule in existing_rules[similar_culture]:
                rule' = generalize(rule, aligned_concepts via LaBSE)  // e.g., replace culture-specific predicates with generic ones
                if KB ∪ {rule'} is consistent:
                    TransferRules = TransferRules ∪ {rule'}
        ConsistentRules = ConsistentRules ∪ TransferRules
    
    return ConsistentRules
```

### (C) Why this design
We chose FOIL over neural rule induction (e.g., Neural LP) because FOIL produces human-readable Horn clauses that can be formally verified against axioms, enabling interpretability and auditability—at the cost of lower robustness to noise and inability to handle continuous features. We selected logical consistency verification rather than statistical filtering (e.g., cross-validation) because axioms are normative invariants that must be preserved; the trade-off is that verification is computationally expensive (NP-hard in worst case) and may reject plausible rules due to incomplete or overly strict axioms. For novel cultures, we use analogical rule transfer from similar cultures (measured by LaBSE embedding cosine similarity) instead of inducing rules from scratch, to reduce sample complexity; this sacrifices coverage for cultures with no close linguistic or cultural analogues, potentially missing idiosyncratic rules. The embedding similarity approach assumes cultural proximity in language space correlates with ethical similarity, which may fail for cultures that share language but differ in moral values (e.g., due to religious differences). We accept this cost because the transfer can be overridden by Phase 2 consistency check, and the system can fall back to few-shot induction if no transfer rules pass.

### (D) Why it measures what we claim
The logical consistency check (entails(rule, contradiction)) operationalizes "preserving logical consistency with universal ethical axioms" because we assume the axiom KB is a correct and exhaustive set of moral invariants; this assumption fails when axioms are underspecified for edge cases (e.g., lying to save a life), in which case consistency becomes a vacuous filter (any rule passes). The FOIL accuracy on few-shot examples measures "cultural alignment" under the assumption that the provided dilemmas are representative of the culture's moral reasoning; this assumption fails when examples are outliers or biased, causing induced rules to overfit and misalign. The analogical transfer uses embedding cosine similarity as a proxy for "cultural similarity in ethics"; this assumes that linguistic proximity in LaBSE space implies ethical proximity, which fails for culturally distinct groups that speak the same language (e.g., US vs. UK on privacy norms), in which case transferred rules may be culturally inappropriate. Each component is causally linked to the motivation-level concept it claims to measure through these explicit assumptions, and the failure modes show where the measurement breaks.

## Contribution

(1) A framework (IERL) that combines inductive logic programming with logical consistency verification against universal ethical axioms, enabling automatic generation and expansion of culturally-grounded ethical rule sets without expert curation. (2) A demonstration that axiom-based consistency checking can replace manual validation for new rules, providing a principled bridge between theory grounds and novel cultural contexts. (3) A methodology for analogical rule transfer across cultures using multilingual embeddings, reducing the number of required examples for induction.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MCLASH (multilingual moral dilemmas) | Diverse cultures & languages |
| Primary metric | Moral judgment accuracy | Directly tests rule correctness |
| Baseline 1 | Zero-shot CoT prompting | No cultural adaptation |
| Baseline 2 | Direct translation + English reasoning | Ignores target moral nuances |
| Baseline 3 | English-centric value scaffolding | Assumes universal moral reasoning |
| Ablation | IERL without rule transfer (no analogical transfer) | Isolates transfer benefit |

### Why this setup validates the claim
This experimental design tests the central claim that IERL produces culturally aligned, axiom-consistent rules better than heuristic baselines. MCLASH provides cross-cultural dilemmas with ground truth judgments. Accuracy on held-out examples directly measures rule alignment. Baselines isolate the failure modes that IERL targets: zero-shot CoT lacks cultural context; translation+English loses nuanced ethical terms; English-centric scaffolding imposes fixed value hierarchies. The ablation tests whether analogical transfer improves few-shot induction. The metric captures the predicted effect: IERL should outperform baselines specifically on culturally loaded dilemmas where these failures occur.

### Expected outcome and causal chain

**vs. Zero-shot CoT prompting** — On a dilemma involving a culture-specific norm like honor killing, zero-shot CoT applies Western deontological principles and incorrectly predicts impermissibility. Our method uses few-shot examples from that culture to induce a rule like permissible(kill) :- family_honor_threatened, which aligns with local norms. We expect a noticeable gap (e.g., 15-20% accuracy difference) on cultures with strong contextual norms but parity on universal dilemmas.

**vs. Direct translation + English reasoning** — On a dilemma where translation loses the ethical concept of "face" in East Asian cultures, the baseline fails to capture the moral weight of saving face, leading to wrong judgment. IERL's logical rules include culture-specific predicates like face_loss directly from examples, so it correctly predicts permissibility. We expect a gap concentrated on dilemmas with culturally embedded concepts (e.g., 10-15% higher accuracy).

**vs. English-centric value scaffolding** — On a dilemma where autonomy and community values conflict differently, e.g., in collectivist cultures, the baseline imposes a fixed harm-minimization hierarchy, mismatching local judgments. IERL learns a rule like permissible(A) :- not(harmful(A)), but weighted by local examples, so it aligns with community-centric reasoning. Expect a gap on dilemmas with value trade-offs, with IERL showing 10-20% improvement.

### What would falsify this idea
If IERL fails to show a noticeable accuracy advantage over baselines on culturally specific subsets (especially where translation loss or value conflicts occur), and if the ablation without transfer performs comparably to the full method, then our central claim that rule induction with consistency and analogical transfer improves cross-cultural moral alignment would be falsified.

## References

1. MET: Theory-Grounded and Culture-Aware Multilingual Moral Reasoning
2. Structured Moral Reasoning in Language Models: A Value-Grounded Evaluation Framework
3. One Model, Many Morals: Uncovering Cross-Linguistic Misalignments in Computational Moral Reasoning
4. MoralBench: Moral Evaluation of LLMs
5. Understanding and Mitigating Language Confusion in LLMs
6. HELP ME THINK: A Simple Prompting Strategy for Non-experts to Create Customized Content with Models
7. Large Language Models are Zero-Shot Reasoners
8. Decomposed Prompting: A Modular Approach for Solving Complex Tasks
