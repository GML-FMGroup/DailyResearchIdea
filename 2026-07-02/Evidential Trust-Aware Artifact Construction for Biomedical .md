# Evidential Trust-Aware Artifact Construction for Biomedical Knowledge Discovery under Data Uncertainty

## Motivation

Existing multi-agent systems for biomedical knowledge discovery, such as BioInsight, generate structured artifacts assuming complete and coherent input data. However, real-world biomedical data is often incomplete, contradictory, or from sources with varying reliability, leading to brittle artifact generation that cannot adjust confidence or content as new evidence emerges. This structural limitation arises because artifact construction and trust verification are typically treated as separate, sequential steps, preventing the system from dynamically reconciling contradictions during generation.

## Key Insight

By interleaving artifact construction with an evidential trust model that updates source reliability in real-time, each piece of evidence contributes to artifact content proportionally to its trustworthiness, resolving contradictions dynamically without a separate verification stage.

## Method

### EviConstruct: Concrete Specifications

(A) **What it is**: EviConstruct is a multi-agent framework that interleaves artifact generation with evidential trust updates. It takes as input a query and a set of biomedical databases/evidence sources, and outputs typed artifacts (e.g., ranked pathways, evidence packets) along with a confidence score per artifact. Each source maintains a subjective opinion tuple (belief, disbelief, uncertainty) that evolves as evidence is ingested.

(B) **How it works**:
1. **Initialize Source Opinions**: Assign each source an initial opinion (e.g., authoritative sources: (0.8, 0.0, 0.2); less known: (0.5, 0.1, 0.4)).
2. **Query Decomposition**: The Planner agent decomposes the user query into subqueries.
3. **Evidence Retrieval**: The Retriever agent fetches evidence from each source for each subquery.
4. **Provisional Artifact Assignment**: Each piece of evidence is assigned to a provisional artifact structure (e.g., a pathway or evidence packet) based on cosine similarity between TF-IDF vectors of evidence text and artifact description. Threshold for assignment: cosine similarity > 0.5.
5. **Weighted Evidence Fusion**: For each artifact, merge evidence from contributing sources using a weighted average, where weights are the source's belief mass normalized: weight = b / (b + d + u).
6. **Trust Update**: After artifact update, the Trust Updater revises source opinions using a discounting rule: if evidence from source S conflicts with the majority (agreement threshold > 0.6) of other trusted sources, update S's opinion as: new_belief = old_belief * (1 - discount_factor), new_disbelief = old_disbelief + discount_factor * (1 - old_disbelief), new_uncertainty unchanged. discount_factor = 0.9. **Load-bearing assumption**: The majority of initially trusted sources are correct, so disagreement with the majority reliably indicates source unreliability. To calibrate hyperparameters (discount_factor, agreement_threshold), we run a sensitivity analysis on a synthetic dataset with known source reliability (20 sources, 500 queries, ground truth reliability known). The optimal range is chosen where correlation between belief mass and actual reliability is maximized (Pearson r > 0.8).
7. **Iterate**: Repeat steps 4-6 as new evidence arrives (e.g., from iterative retrieval or user feedback).
8. **Output**: Final artifact with a confidence score = fused belief from contributing sources.

Hyperparameters: initial trust values per source type (authoritative: (0.8,0.0,0.2); known: (0.5,0.1,0.4); unknown: (0.2,0.1,0.7)), discount_factor=0.9, agreement_threshold=0.6, similarity threshold=0.5.

(C) **Why this design**: We chose subjective logic over Dempster-Shafer because subjective logic explicitly models uncertainty and provides a closed-form update rule (opinion discounting and fusion) that is computationally cheap and interpretable. The trade-off is that subjective logic assumes source opinions are independent, which may not hold if sources share underlying data; we accept this cost because the update rule still provides a reasonable approximation. We chose online trust updating instead of batch verification because it allows the system to adapt to new evidence without reprocessing all previous data, at the cost of not having a global optimum guarantee. We chose to integrate trust into artifact construction rather than as a separate post-hoc filter because this allows the artifact content itself to be influenced by source reliability (e.g., less trusted sources contribute less to the final artifact), rather than just excluding them; this trade-off increases complexity but yields finer-grained confidence.

(D) **Why it measures what we claim**: The computed quantity `source belief mass` measures `source reliability` because we define reliability as the probability that a source provides correct information, and the belief mass is updated based on agreement with other sources under the assumption that the majority of initially trusted sources are correct; this assumption fails when the majority is biased, in which case belief mass reflects conformity rather than accuracy. The quantity `artifact confidence` (fused opinion) measures `artifact trustworthiness` because it aggregates source opinions under the assumption that sources are independent and their opinions are based on evidence; this assumption fails when sources share systematic errors, leading to overconfidence. The quantity `disagreement ratio` (implicit in trust update step) measures `evidence contradiction` because it quantifies the proportion of sources that conflict; this assumes conflict signals unreliability, but it could also signal genuine rare findings. To verify the trust update, we perform a synthetic experiment where ground truth reliability is known and measure the Pearson correlation between belief mass and actual reliability; we also calibrate hyperparameters on synthetic data before applying to BioASQ.

## Contribution

(1) A novel multi-agent architecture that interleaves evidential trust modeling with artifact construction for real-time biomedical knowledge synthesis. (2) A design principle showing that integrating trust updates into the construction process enables handling of contradictory data without separate verification stages. (3) A subjective logic-based trust update mechanism tailored for biomedical evidence sources with varying reliability, including a discounting rule that adapts source opinions based on consensus.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset 1 | BioASQ (biomedical QA) | Rich multi-source evidence required. |
| Dataset 2 | Synthetic dataset (20 sources, 500 queries, known source reliability and ground truth) | Enables calibration and verification of trust update rule. |
| Primary metric | Answer accuracy (F1) | Directly measures correctness of constructed artifacts. |
| Secondary metric | Expert agreement (3 domain experts rate 30 artifact confidence scores on 1-5 Likert scale) | Assesses practical alignment with human trustworthiness judgment. |
| Baseline 1 | BioInsight | Multi-agent without trust updates. |
| Baseline 2 | BioRAGent | RAG with multi-agent but no trust. |
| Baseline 3 | Traditional Trust (PageRank on sources, static weights) | Isolates novelty of interleaving vs. static trust weighting. |
| Ablation of ours | EviConstruct w/o trust (static source opinions, no updates) | Pinpoints effect of online trust updating. |
| Sensitivity analysis | Grid search over discount_factor ∈ {0.1, 0.5, 0.9} and agreement_threshold ∈ {0.5, 0.6, 0.7} on synthetic dataset | Reports optimal range and robustness. |

### Why this setup validates the claim

This experimental design tests the central claim that interleaving artifact generation with evidential trust updates improves reliability in multi-source biomedical discovery. BioASQ requires synthesizing conflicting evidence, directly challenging the trust mechanism. The synthetic dataset provides ground truth reliability, enabling direct measurement of whether belief mass correlates with actual reliability, thereby addressing the load-bearing assumption. Baseline BioInsight uses multi-agents but no trust, isolating the benefit of subjective logic updates. BioRAGent adds retrieval-augmented generation but still lacks dynamic trust. Traditional Trust (PageRank) provides static source weights to separate the effect of dynamic updates from static trust weighting. The ablation (static opinions) pinpoints the effect of online updating. Answer accuracy (F1) captures both precision and recall of constructed knowledge. The user study with experts provides evidence of practical significance by checking if confidence scores align with human judgment. Sensitivity analysis on synthetic data ensures hyperparameters are robust.

### Expected outcome and causal chain

**vs. BioInsight** — On a case where two authoritative sources contradict (e.g., one states a gene drives cancer, another says it suppresses), BioInsight fuses evidence equally or by simple majority, producing an ambiguous or incorrect artifact. Our method instead discounts the source that disagrees with the majority of trusted sources (if it deviates beyond the agreement threshold) after iterative trust updates, so the final artifact reflects the more consistent evidence. We expect a noticeable gain on queries with high inter-source conflict (e.g., >0.3 disagreement ratio), but parity on queries with unanimous agreement.

**vs. BioRAGent** — On a case where a low-credibility source provides a plausible but false fact not present in reliable sources, BioRAGent retrieves and incorporates it without credibility weighting, reducing accuracy. Our method assigns low initial belief to such sources and further decreases it if they conflict with established evidence, minimizing their contribution. Thus we expect an accuracy gap on queries involving rare or misleading sources, but similar performance when all sources are uniformly reliable.

**vs. Traditional Trust (PageRank)** — On a case where source reliability changes over time (e.g., a source initially reliable but later becomes corrupted), PageRank's static weights cannot adapt, while our dynamic updates decrease its influence as contradictions emerge. We expect our method to outperform on queries where source reliability evolves, but perform similarly on static settings.

### What would falsify this idea

If EviConstruct's accuracy is comparable to the ablation (static trust) across all conflict levels, or if the improvement is uniform across high- and low-conflict subsets, then the online trust update is not providing the claimed adaptive benefit. Additionally, if the correlation between belief mass and ground truth reliability on synthetic data is low (Pearson r < 0.5), the trust update rule does not measure what we claim.

## References

1. BioInsight: Multi-Agent Orchestration for Interactive Biomedical Knowledge Discovery
2. BioRAGent: natural language biomedical querying with retrieval-augmented multiagent systems
3. BioMedSearch: A Multi-Source Biomedical Retrieval Framework Based on LLMs
4. An Evidence-Grounded Research Assistant for Functional Genomics and Drug Target Assessment
5. Database resources of the National Center for Biotechnology Information in 2025.
6. Phenomics Assistant: An Interface for LLM-based Biomedical Knowledge Graph Exploration
7. BioRAG: A RAG-LLM Framework for Biological Question Reasoning
8. CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis
