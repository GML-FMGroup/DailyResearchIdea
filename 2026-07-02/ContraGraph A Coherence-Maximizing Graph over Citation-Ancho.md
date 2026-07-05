# ContraGraph: A Coherence-Maximizing Graph over Citation-Anchored Claims for Contradiction Resolution in Biomedical Knowledge Discovery

## Motivation

Existing multi-agent systems for biomedical knowledge discovery, such as BioInsight, process multiple sources independently without resolving contradictions, leading to inconsistent outputs. This structural limitation arises from the absence of a mechanism to enforce logical consistency across evidence, a problem that the rich citation network can be exploited to address by leveraging mutual constraints between claims and their cited sources.

## Key Insight

The citation network provides a structural constraint where claims citing the same source must be consistent, enabling a coherence graph that propagates credibility to detect and resolve contradictions.

## Method

(A) **What it is**: ContraGraph is a coherence-maximizing graph algorithm that takes as input a set of claims, each anchored to one or more citations, and outputs updated credibility scores for both claims and citations. The graph is bipartite between claims and citations, with edges representing citation relationships. (B) **How it works**:

```pseudocode
Input: Claims C, each with a set of citations R(c).
       Initial claim credibility s(c) = 1/|C| for all c.
       Initial citation reliability t(r) = 1/|R| for all r.
       Hyperparameters: number of iterations K = 10, consistency threshold θ = 0.7,
                        calibration threshold θ_cal = 0.8,
                        calibration set size N_cal = 200.

Construct bipartite graph G = (C ∪ R, E) where (c, r) ∈ E if r ∈ R(c).
For iteration = 1 to K:
    For each claim c ∈ C:
        For each citation r ∈ R(c):
            For each other claim c' ∈ C(r) \ {c}:
                Compute pairwise consistency score f(c, c', r) using LLM (e.g., GPT-4, temperature=0) with prompt: "Do the following two claims about citation {r} support or contradict each other? Output a score between 0 (contradiction) and 1 (full support)."
                // Calibration: use LLM confidence (softmax probability for predicted label)
                Let p = LLM probability for predicted label (support or contradict);
                If p < θ_cal, then set f = 0.5 (neutral);
                Else set f = p.
        Aggregate: s_new(c) = (1/|R(c)|) * sum_{r ∈ R(c)} [ t(r) * (1/(|C(r)|-1)) * sum_{c' ∈ C(r) \ {c}} f(c, c', r) ]
    For each citation r ∈ R:
        t_new(r) = (1/|C(r)|) * sum_{c ∈ C(r)} s(c)
    Normalize s and t to sum to 1.
    If max change < 0.01, break.
Return s, t.
```

(C) **Why this design**: We chose a bipartite claim-citation graph over a general claim graph because it reduces computational complexity from O(|C|^2) to O(|E| * avg_degree), accepting that indirect contradictions between claims without shared citations are not directly modeled. We chose iterative propagation over a one-shot solver (e.g., spectral clustering) to allow gradual credibility adjustment, trading off potential slow convergence for interpretability of intermediate scores. We chose an LLM-based pairwise consistency evaluator over simple lexical metrics (e.g., cosine similarity) because it captures nuanced biomedical contradictions (e.g., dosage vs. effect), accepting higher cost and sensitivity to LLM biases. To mitigate LLM biases, we incorporate a calibration step: for each pair, we record the LLM's confidence (probability of predicted label), and if confidence < 0.8, we default to a neutral score of 0.5, flagging the pair for optional human review. We use a held-out calibration set of 200 pairs to tune θ_cal. Finally, we normalize scores to sum to 1 each iteration to prevent drift, at the cost of losing absolute magnitude. (D) **Why it measures what we claim**: The computational quantity `s_new(c)` measures evidential coherence because it aggregates consistency with peer claims sharing a citation, under the assumption that two claims citing the same source should logically agree; this assumption fails when the source is ambiguous or misinterpreted, in which case the score reflects the majority interpretation rather than ground truth. The quantity `t_new(r)` measures citation reliability because it averages credibility of claims citing it, assuming reliable sources attract consistent claims; this assumption fails when a citation is used to support opposing views (e.g., controversial paper), in which case `t(r)` becomes a measure of controversy rather than reliability. The pairwise consistency function `f(c,c',r)` measures logical conflict under the assumption that the LLM can accurately judge support/contradiction; this assumption fails for subtle or domain-specific statements, in which case `f` reflects surface similarity. Our calibration step mitigates this by defaulting to neutral when confidence is low.

## Contribution

['(1) ContraGraph, a novel coherence-maximizing graph algorithm that iteratively adjusts claim and citation credibility based on mutual consistency across citation-anchored claims.', '(2) The insight that the citation network provides a natural structural constraint for contradiction detection and resolution in multi-source biomedical evidence synthesis.', '(3) A practical design for integrating LLM-based pairwise consistency evaluation into an iterative propagation framework, with explicit trade-offs between accuracy, cost, and interpretability.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PubMed Claim Corpus (full) + Clinical Trial subset (RCTs from PubMed, ~2000 claims) | Full corpus tests general coherence; clinical trial subset tests real-world inconsistency detection |
| Primary metric | F1 for claim credibility | Balances precision and recall |
| Secondary metric | Precision/Recall of contradictory claim pairs (among claims sharing a citation) | Directly evaluates contradiction resolution |
| Baseline 1 | CitationCount | Ignores content, only counts citations |
| Baseline 2 | LLM-Direct | No graph structure, individual scoring |
| Ablation | Ours-NoLLM | Replaces LLM with cosine similarity |
| Ablation | Ours-NoCalibration | Removes calibration step (uses raw LLM score) |

### Why this setup validates the claim

The dataset provides ground-truth credibility labels for claims, enabling direct evaluation of our method's core claim: that coherence among claims sharing citations improves credibility scoring. By comparing against CitationCount, we test whether coherence adds value beyond mere citation quantity. By comparing against LLM-Direct, we test whether propagating scores through the bipartite graph outperforms isolated LLM judgments. The ablation Ours-NoLLM isolates the contribution of the LLM-based consistency function, while Ours-NoCalibration tests the importance of calibration. The secondary metric of contradiction pair precision/recall directly validates whether our method correctly identifies inconsistent claims. Additionally, before running the full evaluation, we conduct a preliminary analysis on a random subset of 500 claims to measure the actual consistency of claims sharing citations (e.g., percentage of pairs where claims actually agree or contradict according to expert annotators) and compute the correlation between our coherence score and ground-truth credibility; this validates the assumption that shared citations enforce consistency. The F1 metric captures both false positives and false negatives, which is essential for a credibility task where both over-confidence and under-confidence matter. This combination creates a falsifiable test: if our method outperforms both baselines on claims with shared citations but not on isolated claims, the coherence mechanism is validated.

### Expected outcome and causal chain

**vs. CitationCount** — On a case where a claim is controversial (e.g., "Drug X increases risk of Y") and cited by both supporting and contradicting claims, CitationCount gives a high score because it counts many citations, ignoring contradictions. Our method instead detects pairwise inconsistencies via LLM and reduces the claim's score due to low coherence. Thus we expect a noticeable F1 gap on controversial claims (subset where citations have high entropy of support/contradict) but parity on non-controversial claims.

**vs. LLM-Direct** — On a case where a claim is supported by three citations, each individually plausible but collectively contradictory (e.g., one cites a study showing efficacy, another cites a retraction of that study), LLM-Direct independently scores each citation and may assign high credibility to the claim. Our method propagates credibility across the graph, causing the retraction citation to lower the reliability of the shared source, reducing the claim's score. We expect higher recall on claims with multiple conflicting citations, especially where the conflict is not obvious from individual citation content.

**vs. Ours-NoCalibration** — On a case where the LLM overconfidently assigns a high consistency score to a pair that is actually contradictory (due to hallucination), the calibration step will detect low confidence and default to neutral, preventing overestimation of credibility. We expect higher precision on contradiction detection and more robust F1 scores.

### What would falsify this idea

If our method's F1 gain over CitationCount is uniform across all claims (i.e., not concentrated on high-citation controversial cases), then the coherence mechanism is not the source of improvement and the method likely relies on spurious correlations. Additionally, if the contradiction detection precision/recall is not significantly better than random on pairs with shared citations, the assumption that shared citations enforce consistency is falsified.

### Feasibility estimate

For a dataset of 10,000 claims with average 3 citations per claim and average 3 claims per citation, the total number of LLM calls is approximately sum_{r} |C(r)|*(|C(r)|-1)/2 ≈ 10,000 * (3-1)/2 = 10,000 calls per iteration if we reuse scores across iterations. With K=10, that's 100,000 calls. Using GPT-4-1106 at $0.01/1K input tokens and average prompt length of 200 tokens, total cost ~$200. This is feasible for a single experiment. A caching strategy can reduce calls by storing f scores for each unique (c,c',r) triple across iterations.

## References

1. BioInsight: Multi-Agent Orchestration for Interactive Biomedical Knowledge Discovery
2. BioRAGent: natural language biomedical querying with retrieval-augmented multiagent systems
3. BioMedSearch: A Multi-Source Biomedical Retrieval Framework Based on LLMs
4. An Evidence-Grounded Research Assistant for Functional Genomics and Drug Target Assessment
5. Database resources of the National Center for Biotechnology Information in 2025.
6. Phenomics Assistant: An Interface for LLM-based Biomedical Knowledge Graph Exploration
7. BioRAG: A RAG-LLM Framework for Biological Question Reasoning
8. CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis
