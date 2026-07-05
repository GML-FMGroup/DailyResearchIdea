# INVASE: Invariant Evidence Synthesis with Probabilistic Entity Resolution for Incomplete Biomedical Databases

## Motivation

Existing multi-agent systems for biomedical knowledge discovery (e.g., BioInsight) assume curated databases are complete, leading to overconfident conclusions when entities or relations are missing. This structural failure arises because they lack a mechanism to model missingness and to test robustness against incomplete sources. We address this by integrating probabilistic entity resolution and null-hypothesis generation with a database-removal invariance constraint.

## Key Insight

Because missing entries in biomedical databases create systematic bias that can be modeled via null hypotheses in a probabilistic relational model, enforcing invariance to removal of any single database ensures conclusions depend on relational evidence structure rather than data completeness.

## Method

### INVASE: Invariant Evidence Synthesis with Probabilistic Entity Resolution

**(A) What it is:** INVASE is a multi-agent evidence aggregation framework that takes a set of biomedical databases (potentially incomplete) and a query, and returns a probability distribution over answers that is invariant to removal of any single database. It uses a probabilistic relational model (PRM) to jointly resolve entity mentions across databases and generate null hypotheses for missing data, then enforces database-removal invariance in the aggregation step.

**(B) How it works:**
```pseudocode
// Phase 1: Probabilistic Entity Resolution and Null-Hypothesis Generation
Input: Databases D = {D_1, ..., D_n} with schema S (relations, attributes)
Output: Probabilistic knowledge base K with uncertainty measures

For each pair of entity mentions (e_i, e_j) across databases:
    - Compute coreference probability P(coref | context) using a PRM (factor graph with pairwise potentials, hyperparameter: similarity threshold α=0.7).
    - Resolve entities into canonical identifiers with confidence scores.
For each missing attribute a for entity e in D_i:
    - Generate null hypothesis H0: v_a = NULL (no evidence supporting relationship).
    - Assign prior probability P(H0) based on database coverage statistics (e.g., fraction of missing entries for that entity type, with Beta prior β=2,2).

K = set of resolved entity facts with probabilities, including null hypotheses.

// Phase 2: Multi-Agent Evidence Aggregation with Invariance Constraint
Input: Query Q, knowledge base K, set of agents A = {A_1, ..., A_m}
Output: Answer distribution P_A over candidate answers C

For each agent A_j:
    - Retrieve relevant facts from K via a (potentially learned) retrieval function (e.g., TF-IDF with top-k=10).
    - Compute a probability distribution P_j(c | Q, K) for c in C.

Aggregate: P_agg(c) = weighted geometric mean of {P_j(c)} with weights w_j learned via cross-validation on held-out complete queries. 
    [Specification: w_j are learned using a 2-layer MLP (hidden=32, ReLU) that takes database meta-features (coverage fraction, overlap fraction with other databases, entity count) as input, with a softmax output to ensure they sum to 1. Temperature τ=0.5 controls sharpness.]

// Database-Removal Invariance (DRI) enforcement
Load-bearing assumption: databases overlap sufficiently that large KL divergence after removal indicates over-reliance rather than unique evidence. To handle cases where a database has unique evidence, we learn a per-database weight w_i ∈ [0,1] that scales the KL divergence for that database's removal, learned from a held-out set of queries where that database is known to be critical (e.g., the only source for a fact).

Let DRI_loss = Σ over i of w_i * KL( P_agg || P_agg^{(-i)} ), where P_agg^{(-i)} is the aggregated distribution when database D_i is removed (recompute Phase 1 and Phase 2 without D_i).

Optimize (e.g., fine-tune agent weights and PRM parameters) to minimize DRI_loss + λ * negative log-likelihood of correct answers (λ=0.5).

During inference, use learned weights w_i; no thresholding needed.

Calibration: We verify the load-bearing assumption by computing the correlation between w_i and the fraction of facts unique to D_i across queries. A high correlation indicates that the weights correctly downweight removal penalties for databases with unique evidence.
```

**(C) Why this design:** We chose probabilistic relational models (PRM) over pre-trained entity linking as the core inference engine because PRM captures joint uncertainty across database relations rather than treating each mention independently. The cost is higher inference latency due to iterative message passing, but this is acceptable for offline evidence curation. We enforce the database-removal invariance (DRI) as a training objective rather than a post-hoc test because invariance must be built into the aggregation weights to prevent cherry-picking of sources. This adds computational cost (n+1 runs per query) but ensures that invariance is not an artifact of chance agreement. We use geometric mean aggregation over arithmetic mean because it applies a softer consensus penalty: if one agent assigns near-zero probability to a candidate, the geometric mean heavily penalizes it, preventing a single rogue agent from driving the answer. The tradeoff is that geometric mean can be too conservative when agents have complementary strengths, but we mitigate this by learning per-agent weights. Finally, we generate null hypotheses for missing attributes rather than ignoring them because ignoring introduces systematic bias toward negative conclusions; by including a prior probability of absence, the model can express genuine uncertainty.

**(D) Why it measures what we claim:** Computational quantity: Entity coreference probability P(coref|context) measures entity resolution reliability because we assume that coreference likelihood is monotone with actual identity given observed evidence; this assumption fails when context is insufficient (e.g., ambiguous gene symbols), in which case probability reflects lexical similarity. Null-hypothesis prior P(H0) measures database incompleteness belief because we assume missingness at random conditional on coverage statistics; this fails when missingness is correlated with entity type (e.g., rare genes), understating true absence likelihood. DRI loss KL(P_agg || P_agg^{(-i)}) measures evidence robustness to single-source removal because we assume invariance violation indicates over-reliance; this fails when removal also removes unique evidence (e.g., a specialized database), where large KL signals loss of unique information. We use learned weights w_i to downweight such cases. Geometric mean P_agg measures consensus among agents because it penalizes disagreement exponentially; this assumes agents conditionally independent given query and K, which fails when agents share biases (e.g., same LLM backbone), leading to overconfident consensus.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|-------|--------|-----------------------|
| Dataset | BioASQ dataset | Multi-hop biomedical QA requiring evidence synthesis |
| Primary metric | Macro F1 score | Balances precision/recall across answer types |
| Baseline 1 | BioRAGent | Standard multi-agent RAG without invariance |
| Baseline 2 | BioMedSearch | Multi-source retrieval without probabilistic resolution |
| Baseline 3 | BioInsight | Interactive multi-agent without explicit invariance |
| Ablation | INVASE w/o DRI | Isolates effect of database-removal invariance |

### Why this setup validates the claim

This setup tests the central claim that database-removal invariance (DRI) and probabilistic entity resolution improve evidence aggregation robustness. BioASQ queries require integrating facts across multiple databases, where missing or conflicting evidence is common. Comparing against BioRAGent (no invariance) tests whether DRI prevents over-reliance on a single source. BioMedSearch (deterministic linking) tests whether probabilistic resolution reduces error propagation from incorrect coreference. BioInsight (interactive but no invariance) tests whether passive aggregation with uncertainty suffices. The ablation isolates DRI's contribution. Macro F1 is chosen because it penalizes false positives and negatives evenly, crucial for balanced evidence across answer candidates. The combination of dataset, baselines, and metric creates a falsifiable test: if invariance is effective, the gain should concentrate on queries where one database is unreliable or ambiguous. Additionally, we compute Pearson correlation between DRI loss (weighted) and answer accuracy on a held-out set of queries with artificially removed facts from one database (controlled missingness). A negative correlation would indicate that lower DRI loss corresponds to higher accuracy, supporting the claim.

### Expected outcome and causal chain

**vs. BioRAGent** — On a query about a drug-disease association where one database is incomplete, BioRAGent may over-rely on that database and produce a wrong answer because its aggregation weights are learned without considering invariance. INVASE instead reweights agents to maintain the same answer distribution after removing that database, so it will discount the faulty source. We expect a noticeable macro F1 gap on queries where one database is an outlier (e.g., low coverage), but parity on balanced queries.

**vs. BioMedSearch** — On a query involving ambiguous gene symbols that appear in multiple databases, BioMedSearch's deterministic entity linking may corefer incorrectly, introducing contradictory facts. INVASE's probabilistic resolution assigns a coreference probability, expressing uncertainty and allowing the aggregation to weigh evidence accordingly. Thus we expect a higher macro F1 on queries with high entity ambiguity (e.g., synonymous gene names), while performance is similar on unambiguous cases.

**vs. BioInsight** — On a query requiring integration of rare disease data from a specialized database, BioInsight's interactive agent may miss that evidence if it conflicts with more common sources, because it lacks invariance. INVASE's DRI ensures each database influences the final answer, so we expect INVASE to outperform on subset of queries that rely on a single specialized source, with comparable performance otherwise.

### What would falsify this idea

If INVASE's performance gain is uniform across all query types rather than concentrated on subsets where one database is missing, erroneous, or ambiguous, then DRI is not addressing the predicted failure mode and the central claim is wrong.

## References

1. BioInsight: Multi-Agent Orchestration for Interactive Biomedical Knowledge Discovery
2. BioRAGent: natural language biomedical querying with retrieval-augmented multiagent systems
3. BioMedSearch: A Multi-Source Biomedical Retrieval Framework Based on LLMs
4. An Evidence-Grounded Research Assistant for Functional Genomics and Drug Target Assessment
5. Database resources of the National Center for Biotechnology Information in 2025.
6. Phenomics Assistant: An Interface for LLM-based Biomedical Knowledge Graph Exploration
7. BioRAG: A RAG-LLM Framework for Biological Question Reasoning
8. CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis
